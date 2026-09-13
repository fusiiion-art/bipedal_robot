"""固定足立位ロボット「旋風丸」— MJXベースPPO学習エントリーポイント。

このモジュールは以下を実装する:
  - envs/mjx_env.py の SenpuuMaruMJXEnv (固定足立位環境) をBrax PPOで学習
  - Brax内蔵のAdaptive KL学習率制御によるKLダイバージェンス監視
    (--target_klはepoch内early stoppingではなく、Adaptive LR制御に接続される)
  - checkpoint保存 (best/worst/final/last) と学習曲線ログ (log.json)
  - NaN/Inf検出による即時停止 (改良規約 §18)
  - 報酬ハッキング・学習破綻の監査 (非致命的、警告のみ)
    実機投入前に「シミュレーション学習が正当な報酬最大化をしているか」
    「報酬関数の設計ミスによる異常学習が起きていないか」を検出する。
    NaN/Infと違い致命的ではないため学習は止めず、
    log/<exp_name>/version_*/REWARD_AUDIT_ALERTS.txt に警告を蓄積する。

主要な関数:
  - parse_args(): CLI引数パース (--seed, --target_kl, --exp_name 等)
  - _audit_reward_metrics(): 報酬ハッキング・学習破綻の検出 (5項目)
  - progress_callback(): 学習中の進捗表示・ログ保存・NaN検出・報酬監査 (main()内部で定義)
  - main() 相当のスクリプト本体: 環境構築 → PPO学習 → checkpoint保存

使用例:
  python train/train_mjx.py --seed=42 --target_kl=0.02
  python train/train_mjx.py --exp_name phase0_debug_seed42 --seed=42 --target_kl=0.02

環境仕様 (robot/config.py が正本):
  - 観測: 625次元 (base 84 + history 420 + action_history 100 + temp 20 + volt 1)
  - 行動: 20次元 (関節角の残差 Δq、トルク直接指令ではない)
  - エピソード長: 500 step (100Hz制御、5秒)
  - Phase 0: 外乱無効 (DISTURBANCE_CURRICULUM=False)

改良規約上の制約 (docs/current.md, docs/master_plan.md 参照):
  - 1 iteration = 1変更カテゴリ (報酬とPPO設定を同時に変えない)
  - 合格済みcheckpointを上書きしない (--exp_name で世代管理する)
  - NaN/Inf検出時は即座に停止し、log/<exp_name>/NAN_DETECTED.txt に記録する

ハードウェア:
  - CPU: 単体テスト・形状確認用 (num_envs=32等、小規模)
  - GPU: 本番学習用 (RTX 4060+推奨、num_envs=256、10M step で約30-60分)
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

# sysモジュールのパッチ (Windows上のbrax/orbax依存対策)
if not hasattr(sys.modules.get("uvloop", None), "__name__"):
    sys.modules["uvloop"] = type(sys)("uvloop")

import jax
import jax.numpy as jnp

# Reuse XLA executables across repeated WSL validation/training runs.
jax.config.update("jax_compilation_cache_dir", "/mnt/c/bipedal_robot/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        return jax.tree_util.tree_map(lambda leaf: jax.device_put(jnp.expand_dims(leaf, 0)), x)
    jax.device_put_replicated = _device_put_replicated

from brax import envs
from brax.envs import training as brax_training
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import distribution as brax_distribution

POLICY_MEAN_CLIP_SCALE = 3.0
# [KL-1 PROPOSED 2026-09-11] POLICY_MIN_STD を 0.05 → 0.15 に引き上げる提案。
#
# 根拠（Brax実ソース brax/training/distribution.py の _NormalDistribution.kl_divergence
# を直接確認して導出。詳細は docs/status.md の該当セクション参照）:
#   Braxのkl_mean計算は、20関節分のKLを sum(axis=-1) してからbatch平均を取る実装。
#   scale(std)がほぼ変化しない場合、1関節あたりの寄与は近似的に
#     kl_per_joint ≈ Δμ² / (2σ²)
#   となり、20関節合計は
#     kl_total ≈ 20 × Δμ² / (2σ²)
#   σ=0.05（現状）のとき、1関節あたり平均 Δμ≈0.24rad のシフトだけで
#   kl_total≈230 となり、報告されていたKL=232とほぼ一致することを確認した
#   （docs/status.md 2026-09-01 記載の値）。
#   Δμ=0.24rad は、学習初期（コールドスタート、観測正規化とAdaptive-KLの
#   フィードバックがまだ効いていない最初の数ミニバッチ）では十分あり得る
#   規模である。
#
#   σを0.05→0.15（3倍）に引き上げると、kl_totalは同じΔμに対して
#   1/9に減少する見込み（232 → 約26）。既にstatus.md 2026-09-01時点で
#   min_std=0.00283→0.05019への引き上げが KL=18418→232 (98.7%減) を
#   達成した実績があり、同じ方向の追加調整として位置付けられる。
#
#   【重要】この変更は改良規約の「1 iteration = 1変更カテゴリ」に基づき、
#   PPO最適化系（policy分布パラメータ）の単独変更として提案するもの。
#   報酬系(mjx_rewards.py)とは同時変更しないこと。
#   GPU Debug run (D-6) で実測KLトレンドを確認してから正式採用を判断すること。
#   探索性能(policy_dist_mean_std等)への悪影響がないかも合わせて確認する。
POLICY_MIN_STD = 0.15
POLICY_MAX_STD = 3.0


def _install_policy_std_cap():
    """Cap tanh-normal scale while preserving Brax's existing distribution API."""
    original_create_dist = brax_distribution.NormalTanhDistribution.create_dist

    def clipped_create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        loc = POLICY_MEAN_CLIP_SCALE * (loc / (1.0 + jnp.abs(loc)))
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        scale = jnp.clip(scale, POLICY_MIN_STD, POLICY_MAX_STD)
        return brax_distribution._NormalDistribution(loc=loc, scale=scale)

    if not hasattr(brax_distribution, '_NormalDistribution'):
        raise RuntimeError('Brax distribution API changed: _NormalDistribution is unavailable')
    brax_distribution.NormalTanhDistribution.create_dist = clipped_create_dist
    return original_create_dist


_install_policy_std_cap()

# Patch brax _unpmap for JAX 0.4+ Multi-GPU safety
def _safe_unpmap(v):
    def _unpmap_leaf(x):
        if hasattr(x, "addressable_shards"):
            d = x.addressable_shards[0].data
            if d.ndim > 0 and d.shape[0] == 1:
                return d.squeeze(0)
            return d
        if hasattr(x, "device_buffers"):
            return x[0]
        return x
    return jax.tree_util.tree_map(_unpmap_leaf, v)

ppo._unpmap = _safe_unpmap

# Safe wrapper for make_inference_fn to handle leading pmap dimension in params
_orig_make_inference_fn = ppo_networks.make_inference_fn
def _safe_make_inference_fn(ppo_networks_tuple, **make_kwargs):
    orig_fn = _orig_make_inference_fn(ppo_networks_tuple, **make_kwargs)
    def safe_inference_fn(params, *args, **kwargs):
        def _strip_leading_dim(leaf):
            if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 1 and leaf.shape[0] == 1:
                return leaf.squeeze(0)
            return leaf
        params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
        return orig_fn(params_stripped, *args, **kwargs)
    return safe_inference_fn

ppo_networks.make_inference_fn = _safe_make_inference_fn






sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from envs.mjx_env import SenpuuMaruMJXEnv  # noqa: F401 (Brax環境登録のため)
from envs.training_wrapper import TrainingProgressWrapper

# ============================================================================
# [監査追加 2026-09-13] 報酬ハッキング・学習破綻の検出
# ============================================================================
# 実機計測にはまだ入っていない段階で、シミュレーション学習が
# 「正当な報酬最大化」をしているか、報酬関数の設計ミス(reward hacking)や
# 実装バグによる異常な学習が起きていないかを検証するための追加監査。
#
# 既存のNaN/Inf検出(改良規約 §18、上のprogress_callback内)とは異なり、
# ここでの検出は致命的エラーではなく「疑わしい兆候」の警告であるため、
# raiseはせず学習を継続する。検出結果は
# log/<exp_name>/version_*/REWARD_AUDIT_ALERTS.txt に蓄積され、
# 学習終了後にサマリーが表示される。
#
# 検出項目:
#   1. Reward Exploitation  - 単一の報酬成分(r_cp, r_recovery, r_upright,
#                              r_com_stab, disturbance_recovery_bonus,
#                              pbrs_reward, alive)が不合理に大きくないか
#   2. Shaping Mismatch     - 高報酬なのに姿勢系の正報酬(r_upright,
#                              r_com_stab, both_feet_contact)が乏しい、
#                              または total_penalty が total_reward を
#                              圧倒していないか
#   3. Metric Corruption    - stability_index が [0,1] の範囲外、
#                              zmp_margin が直近N回連続でほぼ一定値
#                              (envs/stability_metrics.py の計算が
#                              死んでいる可能性)、または同ファイルが
#                              自己申告する stability_metrics_finite
#                              フラグ(NaN/Inf自己診断)がFalse
#   4. Potential Decay      - potential が増加しているのに pbrs_reward が
#                              大きく負(compute_potential()の符号ミス等)
#   5. Action Distortion    - CBFによるaction_saturationが高い
#                              (envs/mjx_env.py 及び safety/cbf.py の
#                              compute_saturation_ratio() 参照。方策が
#                              実行不能な指令を多発させている、または
#                              CBF/可動域制限が過剰に効いている可能性)
#   6. Sensor/Kinematics Fallback - envs/mjx_rewards.py の com_accel が
#                              qacc取得失敗によるフォールバック値
#                              [0,0,-9.81]を使用中(com_accel_is_fallback)。
#                              stability_metrics.py のv2 [CRITICAL FIX]
#                              で説明されている「zmp_marginが死んだ指標に
#                              なる」バグの片割れの原因だったため、
#                              本番での再発を監視する。
#
# 依存するmetricsキー (envs/mjx_rewards.py, envs/mjx_env.py,
# envs/stability_metrics.py, safety/cbf.py で提供):
#   total_reward, total_penalty, stability_index, zmp_margin, r_upright,
#   r_com_stab, both_feet_contact, r_cp, r_recovery,
#   disturbance_recovery_bonus, pbrs_reward, alive, potential,
#   action_saturation, stability_metrics_finite, com_accel_is_fallback
# ============================================================================

REWARD_AUDIT_THRESHOLDS = {
    'exploitation_abs_max': 50.0,   # 各報酬成分の絶対値の上限目安
    'shaping_high_reward': 10.0,    # これ以上の報酬でpositive componentが乏しいと疑う
    'shaping_low_positive': 1.0,
    'shaping_severe_negative': -100.0,
    'stability_index_max': 1.05,    # [0,1]からの逸脱許容
    'metric_frozen_window': 10,     # 直近何件で「固定値」と判定するか
    'metric_frozen_std': 1e-6,
    'action_saturation_max': 0.5,   # CBF補正の平均飽和率(50%)
}


def _find_metric_key(metrics: dict, suffix: str):
    """Brax集計後のキー(例 'eval/episode_metrics/xxx')から末尾一致で探す。
    既存の reward_is_finite 探索(このファイル内、progress_callback参照)と
    同じ方式に合わせている。"""
    for key in metrics.keys():
        if key == suffix or key.endswith(suffix):
            return key
    return None


def _audit_reward_metrics(metrics_dict: dict, metrics_history: list) -> list:
    """1ステップ分のmetrics_dict(既にfloat化済み)を検査し、報酬ハッキングや
    学習破綻の兆候をチェックする。致命的ではないため raise はしない。

    Args:
        metrics_dict: progress_callback内で構築される、その時点のfloat化
            済みmetrics辞書 (まだmetrics_historyには追加する前のもの)。
        metrics_history: これまでの metrics_dict のリスト(現在のステップは
            含まない)。Metric CorruptionやPotential Decayのトレンド検出に使う。

    Returns:
        alerts: 検出されたアラートメッセージのリスト(空なら異常なし)。
    """
    alerts = []
    th = REWARD_AUDIT_THRESHOLDS

    def _get(suffix, default=0.0):
        key = _find_metric_key(metrics_dict, suffix)
        return metrics_dict[key] if key is not None else default

    total_reward = _get('total_reward', metrics_dict.get('reward', 0.0))
    total_penalty = _get('total_penalty', 0.0)
    stability_index = _get('stability_index', 0.5)
    r_upright = None
    r_upright_key = _find_metric_key(metrics_dict, 'r_upright')
    if r_upright_key is not None:
        r_upright = metrics_dict[r_upright_key]
    r_com_stab = _get('r_com_stab', 0.0)
    both_feet_contact = _get('both_feet_contact', 0.0)

    # --- 1. Reward Exploitation ---
    component_suffixes = [
        'r_cp', 'r_recovery', 'r_upright', 'r_com_stab',
        'disturbance_recovery_bonus', 'pbrs_reward', 'alive',
    ]
    for suffix in component_suffixes:
        key = _find_metric_key(metrics_dict, suffix)
        if key is None:
            continue
        val = metrics_dict[key]
        if abs(val) > th['exploitation_abs_max']:
            alerts.append(
                f"[Exploitation] 報酬成分 '{key}' が異常に大きい: {val:.2f} "
                f"(閾値 ±{th['exploitation_abs_max']:.0f})"
            )

    # --- 2. Shaping Mismatch ---
    if r_upright is not None:
        positive_sum = max(r_upright, 0.0) + max(r_com_stab, 0.0) + both_feet_contact
        if total_reward > th['shaping_high_reward'] and positive_sum < th['shaping_low_positive']:
            alerts.append(
                f"[Shaping Mismatch] 高報酬(total_reward={total_reward:.2f})だが"
                f"姿勢系の正報酬が乏しい(r_upright+r_com_stab+both_feet_contact="
                f"{positive_sum:.2f})。ペナルティ符号反転や他成分の異常な"
                f"寄与を疑う。"
            )
    if total_reward < th['shaping_severe_negative'] and total_penalty > 0:
        alerts.append(
            f"[Shaping Mismatch] 報酬が著しく負(total_reward={total_reward:.2f})、"
            f"total_penalty={total_penalty:.2f} が報酬設計を圧倒している"
            f"可能性。mjx_rewards.py の重み(REWARD_WEIGHTS)を確認。"
        )

    # --- 3. Metric Corruption ---
    if stability_index < 0.0 or stability_index > th['stability_index_max']:
        alerts.append(
            f"[Metric Corruption] stability_index が範囲外: "
            f"{stability_index:.3f} (期待範囲 [0, 1])"
        )
    finite_key = _find_metric_key(metrics_dict, 'stability_metrics_finite')
    if finite_key is not None and metrics_dict[finite_key] < 0.5:
        alerts.append(
            "[Metric Corruption] envs/stability_metrics.py の "
            "compute_unified_stability_index() がNaN/Infを検出 "
            "(stability_metrics_finite=0)。CP/ZMP/バランス/姿勢マージンの"
            "いずれかの幾何計算が破綻している。"
        )
    window = th['metric_frozen_window']
    zmp_key = _find_metric_key(metrics_dict, 'zmp_margin')
    if zmp_key is not None and len(metrics_history) >= window:
        recent = [m[zmp_key] for m in metrics_history[-window:] if zmp_key in m]
        if len(recent) >= window and np.std(recent) < th['metric_frozen_std']:
            alerts.append(
                f"[Metric Corruption] zmp_margin が直近{window}回連続で"
                f"ほぼ一定値({metrics_dict[zmp_key]:.6f}) — "
                f"envs/stability_metrics.py の計算が死んでいる可能性"
                f"(過去のcompute_zmp_marginバグ再発等)。"
            )

    # --- 4. Potential Decay ---
    pot_key = _find_metric_key(metrics_dict, 'potential')
    pbrs_key = _find_metric_key(metrics_dict, 'pbrs_reward')
    if pot_key is not None and pbrs_key is not None and len(metrics_history) >= 1:
        prev = metrics_history[-1]
        if pot_key in prev:
            delta_potential = metrics_dict[pot_key] - prev[pot_key]
            pbrs_val = metrics_dict[pbrs_key]
            if delta_potential > 0.1 and pbrs_val < -2.0:
                alerts.append(
                    f"[Potential Decay] potentialは増加({delta_potential:+.3f})"
                    f"だが pbrs_reward が大きく負({pbrs_val:.2f})。"
                    f"envs/mjx_rewards.py の compute_potential() や "
                    f"discounting(gamma)を確認。"
                )

    # --- 5. Action Distortion ---
    sat_key = _find_metric_key(metrics_dict, 'action_saturation')
    if sat_key is not None and metrics_dict[sat_key] > th['action_saturation_max']:
        alerts.append(
            f"[Action Distortion] CBFによるアクション補正の飽和率が高い: "
            f"{metrics_dict[sat_key]*100:.1f}% — 方策が実行不能な指令を"
            f"多発させているか、safety/cbf.py の制限が過剰に効いている"
            f"可能性。"
        )

    # --- 6. Sensor/Kinematics Fallback ---
    fallback_key = _find_metric_key(metrics_dict, 'com_accel_is_fallback')
    if fallback_key is not None and metrics_dict[fallback_key] > 0.5:
        alerts.append(
            "[Sensor Fallback] com_accel が qacc 取得失敗によりフォール"
            "バック値[0,0,-9.81]を使用中。envs/mjx_rewards.py の compute() "
            "内、nq/qacc の条件分岐を確認。ZMPが重心追従に退化し、外乱下の"
            "不安定性を過小評価している可能性がある(stability_metrics.py "
            "のv2 changelog参照)。"
        )

    # [予防追加 2026-09-13] envs/mjx_env.py の physics_step ロールバック
    # 機構が実際に発動した頻度を記録する。ロールバックにより学習自体は
    # 汚染されないが、頻発する場合はカリキュラム(外乱強度)や物理タイム
    # ステップ・ソルバー設定が実際の限界に近いことを示すシグナルになる。
    diverged_key = _find_metric_key(metrics_dict, 'physics_diverged')
    if diverged_key is not None and metrics_dict[diverged_key] > 0.5:
        alerts.append(
            "[Physics Divergence] 物理サブステップがNaN/Infに発散し、"
            "envs/mjx_env.py のロールバック機構が作動した(エピソードは"
            "安全に終了済み)。頻発する場合は外乱の強さ・timestep・"
            "solver設定の見直しを検討。"
        )

    return alerts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment subfolder under log. Leave empty to write directly to log/version_x.")
    parser.add_argument("--num_envs", type=int, default=None, help="並列環境数 (GPUなら2048〜4096推奨, CPU自動設定)")
    parser.add_argument("--steps", type=int, default=None, help="総学習ステップ数")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--unroll_length", type=int, default=10, help="PPOのアクションアンロール長")
    parser.add_argument("--episode_length", type=int, default=None, help="1エピソードのステップ数")
    parser.add_argument("--num_evals", type=int, default=None, help="評価回数")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--num_updates_per_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_kl", type=float, default=0.02)
    return parser.parse_args()

def make_policy_network_factory(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn=lambda x, _=None: x,
):
    """
    Standard PPO network factory for the fixed-foot standing policy.
    
    観測空間の構成 (OBS_DIM):
      - Base Obs (現在の状態)
      - Obs/Action History (遅延補償用の履歴バッファ)
      - Servo Temperature (各関節の温度)
      - Supply Voltage (電源電圧)
      
    The observation already contains measured-sensor equivalents and their
    short history; no privileged teacher or adaptation network is used.
    """
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        mean_clip_scale=POLICY_MEAN_CLIP_SCALE,
    )

def main():
    args = parse_args()
    
    print("=== MJX GPU Training Pipeline (RMA Enabled) ===")
    devices = jax.devices()
    print(f"JAX Devices: {devices}")
    is_cpu = devices[0].platform == 'cpu'
    
    if is_cpu:
        print("[Warning] JAX is running on CPU. GPU未使用 - パラメータをCPU向けに自動縮小します。")
        print("[Info] GPU使用にはWSL2 + JAX CUDA版が必要です。")
        # CPUモード: コンパイル時間を最小化する小さなパラメータ
        num_envs       = args.num_envs       or 32
        steps          = args.steps          or 100_000
        episode_length = args.episode_length or 50
        num_evals      = args.num_evals      or 3
        batch_size     = args.batch_size     or 32
        num_minibatches = args.num_minibatches or 1
    else:
        print(f"[GPU] {devices[0]} で学習開始！")
        # GPUモード: RTX 4060 (8GB) でのコンパイルハングを避けるため、パラメータを軽量化
        num_envs       = args.num_envs       or 256
        steps          = args.steps          or 10_000_000
        episode_length = args.episode_length or RobotConfig.MAX_EPISODE_STEPS
        num_evals      = args.num_evals      or 20
        batch_size     = args.batch_size     or 256
        num_minibatches = args.num_minibatches or 16

    # --- 複数GPU環境向けの自動最適化 (Divisibilityの担保) ---
    num_devices = len(devices)
    if num_envs % num_devices != 0:
        old_num_envs = num_envs
        num_envs = (num_envs // num_devices) * num_devices
        print(f"[Auto-Tune] num_envs をGPU数({num_devices})で割り切れる {num_envs} に自動調整しました (元: {old_num_envs})")
    
    if batch_size % num_devices != 0:
        old_batch_size = batch_size
        batch_size = max(1, batch_size // num_devices) * num_devices
        print(f"[Auto-Tune] batch_size をGPU数({num_devices})で割り切れる {batch_size} に自動調整しました (元: {old_batch_size})")

    # batch_size * num_minibatches は num_envs で割り切れる必要がある
    if (batch_size * num_minibatches) % num_envs != 0:
        # 割り切れるように num_minibatches を自動調整
        import math
        old_minibatches = num_minibatches
        # 必要な最小の倍数を探す
        target_total_batch = math.ceil((batch_size * num_minibatches) / num_envs) * num_envs
        num_minibatches = target_total_batch // batch_size
        print(f"[Auto-Tune] (batch_size * num_minibatches) % num_envs == 0 を満たすため、num_minibatches を {num_minibatches} に自動調整しました (元: {old_minibatches})")

    # 1. 環境生成
    env = envs.get_environment('senpuu_maru_mjx')
    
    # 2. ログディレクトリとバージョン管理
    from pathlib import Path
    import json
    
    root_path = Path(__file__).resolve().parent.parent
    base_log_dir = root_path / "log"
    if args.exp_name:
        base_log_dir = base_log_dir / args.exp_name
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    version = 0
    while (base_log_dir / f"version_{version}").exists():
        version += 1
    run_dir = base_log_dir / f"version_{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Logging to {run_dir}", flush=True)
    
    # 状態トラッキング用変数
    best_reward = -float('inf')
    worst_reward = float('inf')
    current_params = None
    metrics_history = []
    # [監査追加 2026-09-13] 報酬ハッキング監査(_audit_reward_metrics)の
    # 検出件数を種類別に集計する。学習終了後にサマリー表示する。
    reward_audit_alert_counts = {}
    reward_audit_total_alerts = 0
    
    def policy_params_callback(current_step, make_policy, params):
        import pickle
        nonlocal current_params
        current_params = params
        
        # 毎回ラストのモデルを保存
        with open(run_dir / "last_params.pkl", "wb") as f:
            pickle.dump(params, f)
            
    # コールバック関数（プログレス表示・ログ保存用）
    def progress_callback(num_steps, metrics):
        import pickle
        nonlocal best_reward, worst_reward
        nonlocal reward_audit_alert_counts, reward_audit_total_alerts
        reward = metrics.get('eval/episode_reward', metrics.get('training/total_reward', float('nan')))

        # --- NaN/Inf 即時停止チェック（改良規約 §18: 即時停止条件） ---
        # KLスパイクや勾配爆発が発生すると reward や他の主要metricsが
        # NaN/Infになりうる。これを検出しないまま学習を続けると、
        # 壊れたcheckpointをbest_paramsとして保存してしまう危険がある。
        if not np.isfinite(reward):
            print(f"\n{'='*70}", flush=True)
            print(f"❌ FATAL: NaN/Inf detected in reward at step {num_steps}!", flush=True)
            print(f"   reward={reward}", flush=True)
            print(f"   metrics keys={list(metrics.keys())}", flush=True)
            print(f"{'='*70}\n", flush=True)
            # 直近のmetrics_historyを保存してから停止（原因調査用）
            with open(run_dir / "log.json", "w") as f:
                json.dump(metrics_history, f, indent=2)
            with open(run_dir / "NAN_DETECTED.txt", "w") as f:
                f.write(f"step={num_steps}\nreward={reward}\nmetrics={metrics}\n")
            raise RuntimeError(
                f"NaN/Inf detected in reward at step {num_steps}. "
                f"Training stopped per改良規約 §18 (即時停止条件). "
                f"Details written to {run_dir / 'NAN_DETECTED.txt'}"
            )

        # 主要metrics全体もチェック（reward以外にKL, value_loss等も対象）
        for key, value in metrics.items():
            try:
                val_float = float(value.item() if hasattr(value, 'item') else value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val_float):
                print(f"\n{'='*70}", flush=True)
                print(f"❌ FATAL: NaN/Inf detected in metric '{key}' at step {num_steps}!", flush=True)
                print(f"   value={val_float}", flush=True)
                print(f"{'='*70}\n", flush=True)
                with open(run_dir / "log.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)
                with open(run_dir / "NAN_DETECTED.txt", "w") as f:
                    f.write(f"step={num_steps}\nmetric={key}\nvalue={val_float}\nmetrics={metrics}\n")
                raise RuntimeError(
                    f"NaN/Inf detected in metric '{key}' at step {num_steps}. "
                    f"Training stopped per改良規約 §18 (即時停止条件)."
                )

        # --- 専用フラグ 'reward_is_finite' のチェック ---
        # envs/mjx_rewards.py の compute() が jnp.isfinite で算出したフラグ。
        # 0.0/1.0 という値自体は有限なので、上の汎用isfiniteチェックでは
        # 検出できない(0.0は有限)。このフラグが0.0の場合は
        # 「reward算出の途中経路でNaN/Infが発生した」ことを意味するため、
        # 専用に検査する。値はBraxのepisode集約で平均化されるため、
        # 1エピソードでも非有限値を含めば1.0未満になる。
        reward_is_finite_key = None
        for key in metrics.keys():
            if key.endswith("reward_is_finite"):
                reward_is_finite_key = key
                break
        if reward_is_finite_key is not None:
            finite_ratio = metrics[reward_is_finite_key]
            finite_ratio = float(finite_ratio.item() if hasattr(finite_ratio, 'item') else finite_ratio)
            if finite_ratio < 1.0:
                print(f"\n{'='*70}", flush=True)
                print(f"❌ FATAL: reward computation produced NaN/Inf at step {num_steps}!", flush=True)
                print(f"   {reward_is_finite_key}={finite_ratio} (< 1.0 means some envs saw non-finite reward)", flush=True)
                print(f"   → envs/mjx_rewards.py の compute() 内の各報酬成分を確認してください", flush=True)
                print(f"{'='*70}\n", flush=True)
                with open(run_dir / "log.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)
                with open(run_dir / "NAN_DETECTED.txt", "w") as f:
                    f.write(
                        f"step={num_steps}\n{reward_is_finite_key}={finite_ratio}\n"
                        f"source=envs/mjx_rewards.py compute()\nmetrics={metrics}\n"
                    )
                raise RuntimeError(
                    f"reward_is_finite={finite_ratio} at step {num_steps}: "
                    f"non-finite value detected inside mjx_rewards.compute(). "
                    f"Training stopped per改良規約 §18 (即時停止条件)."
                )

        # 学習進捗率の計算と表示。ここでは総乱数ステップではなく、
        # 各環境の累積ステップを単調に増やす構造を優先し、
        # 1.0 を超えないようにする。
        training_progress = min(num_steps / max(steps, 1), 1.0) if steps > 0 else 0.0
        print(f"Step: {num_steps:10d} | Reward: {reward:.4f} | Progress: {training_progress:.2%}", flush=True)

        # JSONログ用の辞書作成
        metrics_dict = {
            "step": int(num_steps),
            "reward": float(reward),
            "training_progress": float(training_progress),
            "num_envs": int(num_envs),
            "episode_length": int(episode_length),
        }
        for k, v in metrics.items():
            if k == "training_progress":
                continue
            metrics_dict[k] = float(v.item() if hasattr(v, 'item') else v)
        # --- 報酬ハッキング・学習破綻の監査 (非致命的、警告のみ) ---
        # metrics_history にはまだ現在のステップを追加していないため、
        # ここでは「これまでの履歴 vs 現在のステップ」の比較として機能する。
        reward_audit_alerts = _audit_reward_metrics(metrics_dict, metrics_history)
        if reward_audit_alerts:
            reward_audit_total_alerts += len(reward_audit_alerts)
            print(f"\n⚠️  [Reward Audit] Step {num_steps}: "
                  f"{len(reward_audit_alerts)}件の異常兆候を検出", flush=True)
            with open(run_dir / "REWARD_AUDIT_ALERTS.txt", "a") as f:
                f.write(f"\n[Step {num_steps}]\n")
                for alert in reward_audit_alerts:
                    print(f"   - {alert}", flush=True)
                    f.write(f"  - {alert}\n")
                    # カテゴリ別カウント (例: "[Exploitation] ..." → "Exploitation")
                    category = alert.split(']', 1)[0].lstrip('[')
                    reward_audit_alert_counts[category] = reward_audit_alert_counts.get(category, 0) + 1

        metrics_history.append(metrics_dict)
        
        with open(run_dir / "log.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
            
        # 最高のモデルと最低のモデルを保存
        if current_params is not None:
            if reward > best_reward:
                best_reward = reward
                with open(run_dir / "best_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)
                print(f"  >>> Best Model Saved! (Reward: {reward:.4f})", flush=True)
                
            if reward < worst_reward:
                worst_reward = reward
                with open(run_dir / "worst_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)
                print(f"  >>> Worst Model Saved! (Reward: {reward:.4f})", flush=True)

    print(f"Starting training: num_envs={num_envs}, steps={steps}, episode_length={episode_length}")
    start_time = time.time()
    
    # Learning Rate: Brax内蔵のAdaptive KL LRスケジュールを使用。
    # KL爆発時に自動的に学習率を下げ、KLが低すぎる場合は上げる。
    # 以前のoptax.warmup_cosine_decay_scheduleはBrax PPOの内部optimizerには
    # 渡されておらず機能していなかったため削除。
    
    # --- TrainingProgressWrapper の注入 ---
    # brax.envs.training.wrap を一時的に差し替え、AutoResetWrapper の
    # 外側に TrainingProgressWrapper を配置する。
    # これにより training_progress がエピソード境界を跨いで単調増加する。
    steps_per_env = max(steps // num_envs, 1)
    _original_wrap = brax_training.wrap

    def _wrap_with_progress(env, **kwargs):
        wrapped = _original_wrap(env, **kwargs)
        return TrainingProgressWrapper(wrapped, total_steps_per_env=steps_per_env)

    brax_training.wrap = _wrap_with_progress

    # 3. PPO学習実行 (RMA Network Architecture)
    try:
        make_inference_fn, params, metrics = ppo.train(
            environment=env,
            network_factory=make_policy_network_factory,
            num_timesteps=steps,
            num_evals=num_evals,
            reward_scaling=0.01,  # 報酬クリップ後の値をPPOの更新量に合わせる
            episode_length=episode_length,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=args.unroll_length,
            num_minibatches=num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            discounting=0.99,
            bootstrap_on_timeout=True,
            learning_rate=args.learning_rate,
            entropy_cost=1e-3,
            # --- KLダイバージェンス制御 ---
            clipping_epsilon=0.2,           # 0.3(Braxデフォルト)→0.2に縮小
            max_grad_norm=1.0,              # 勾配クリッピングで勾配爆発を防止
            learning_rate_schedule='ADAPTIVE_KL',  # Brax内蔵Adaptive KL LR
            desired_kl=args.target_kl,
            learning_rate_schedule_min_lr=1e-5,   # KL爆発時のフロア（1e-6では低すぎてLRがstuckする）
            learning_rate_schedule_max_lr=5e-4,   # KL安定時の天井

            num_envs=num_envs,
            batch_size=batch_size,
            seed=args.seed,
            progress_fn=progress_callback,
            policy_params_fn=policy_params_callback
        )
    finally:
        # 他のモジュールに影響しないよう必ず復元
        brax_training.wrap = _original_wrap

    elapsed_time = time.time() - start_time
    print(f"Training finished in {elapsed_time/60:.1f} minutes!")

    # --- 報酬ハッキング監査サマリー ---
    # 実機投入前に「この学習は信頼してよいか」を判断するための最終報告。
    # 詳細な各アラートは REWARD_AUDIT_ALERTS.txt を参照。
    summary_lines = []
    if reward_audit_total_alerts > 0:
        summary_lines.append(
            f"⚠️  Reward Audit: 学習中に {reward_audit_total_alerts} 件の"
            f"異常兆候を検出しました。"
        )
        for category, count in sorted(
            reward_audit_alert_counts.items(), key=lambda x: -x[1]
        ):
            summary_lines.append(f"   - {category}: {count}件")
        summary_lines.append(
            f"   詳細: {run_dir / 'REWARD_AUDIT_ALERTS.txt'}"
        )
        summary_lines.append(
            "   実機投入前に、上記カテゴリに対応する報酬関数・安定性"
            "指標・CBF実装を確認することを推奨します。"
        )
    else:
        summary_lines.append(
            "✅ Reward Audit: 学習全体を通して異常兆候は検出されませんでした。"
        )
    print("\n" + "\n".join(summary_lines))
    with open(run_dir / "REWARD_AUDIT_SUMMARY.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # 4. パラメータ保存
    import pickle
    model_path = os.path.join(run_dir, "final_params.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()