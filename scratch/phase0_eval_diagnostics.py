#!/usr/bin/env python3
"""Phase 0 / Gate A diagnosis: deterministic vs stochastic evaluation +
termination-reason histogram (docs/master_plan.md 付録A §3.5, Task0).

status.md (2026-09-01) の「次のTask」= 「deterministic/stochastic評価の実装と
終了理由ヒストグラム化」に対応する。エスカレーション項目の「次の切り分け」の
3項目のうち、deterministic評価・終了stepヒストグラム化・報酬成分分解ログの
3つをまとめてこのスクリプトで実施する。

設計方針（master_plan.md §3.5 に基づく）:
  - 2x2評価: {deterministic, stochastic} x {fixed_dr, randomized_dr}
    注意: 本リポジトリの reset() は物理初期姿勢(qpos/qvel)を常に同一の
    nominal poseに固定しており、初期姿勢そのもののrandomizationは
    master_plan.md §1.6で「Task1(Gate A是正)の時点で必ず導入する」と
    定義された未実装機能である。したがって本スクリプトの
    「初期状態randomize」軸は、既存の実装済みrandomization経路である
    domain randomization (質量/摩擦/重心オフセット/サーボ温度/電圧) の
    on/offとして操作する。これは近似であり、真の初期姿勢randomizationの
    代替ではない。この制約は出力レポートに明記する。
  - deterministic x fixed_dr のセルは、同一checkpoint・同一初期状態・
    同一policyであれば理論上ビット単位で再現するはずのセルであり、
    複数episodeを回す意味は「決定論性テスト」（master_plan.md §4.2）を
    兼ねる以外にない。デフォルトのepisode数を他セルより少なくしている。
  - 各episodeについて、終了理由 (fallen_roll / fallen_pitch /
    fallen_height / time_limit / unknown の組み合わせ) を分類する。
    非足裏接触・トルク上限による終了は、現行の envs/mjx_rewards.py の
    done判定 (is_fallen_roll or is_fallen_pitch or is_low のみ) に
    実装されていないため分類対象にできない。これは
    master_plan.md Task4 (C-08, 複合成功条件) が未着手であることの
    追加の裏付けとしてレポートに記録する。
  - Kaplan-Meier型の生存曲線を打ち切り(truncated=time_limit)を
    考慮して計算する。
  - 失敗episodeについて、終了直前 collapse_window step分の
    roll/pitch/base角速度/base位置の時系列を記録する。
  - reward metrics (envs/mjx_rewards.py が返す metrics dict) の
    episode平均をあわせて記録し、reward成分分解ログを兼ねる。
  - master_plan.md §3.6 の決定木を単純な閾値ヒューリスティックとして
    実装し、失敗タイミングの偏り(序盤/後半/ランダム)を自動判定する。
    これは補助的な一次判定であり、最終診断は人間 / 記録を見た
    Copilotが行うことを想定している。

Done条件 (pytest, tests/test_phase0_eval_diagnostics.py 側):
  - classify_termination_reason の分類ロジック
  - kaplan_meier_survival の生存曲線計算
  - diagnose_failure_timing の決定木ヒューリスティック
  これらは純Python/NumPyのみで完結し、JAX/MJX/GPU無しでCPU上で検証できる。

実行には学習済みcheckpoint (log/<exp_name>/version_x/*.pkl) と
JAX/MJX/Brax環境 (WSLのvenv_wsl等) が必要。このリポジトリのsandboxには
GPUも実際の学習済みcheckpointも存在しないため、本スクリプト作成時には
以下2段階で検証した:
  1. 純Python/NumPyの解析ロジック(classify_termination_reason /
     kaplan_meier_survival / diagnose_failure_timing /
     summarize_episode_alive)はtests/test_phase0_eval_diagnostics.pyで
     単体テスト済み(CPU、JAX不要)。
  2. ロールアウト部分(run_episode/run_condition/main)は、CPU上に
     JAX/MuJoCo/MJX/Braxをインストールし、ランダム初期化した
     (未学習の)policy checkpointを使って実際にreset/step/評価の
     全経路を通しで実行確認した。この過程で以下の実装上の罠を
     発見・修正済み:
       - env.reset/env.stepは必ずjax.jit()経由で呼ぶ必要がある。
         eager実行では reset() 内の `info['step'] = 0` がPython int の
         まま伝播し、`truncated.astype(...)` (envs/mjx_env.py) で
         AttributeErrorになる。
       - jax.jit(env.reset) はbound methodの等価性でコンパイル結果を
         キャッシュするため、RobotConfig.RANDOM_* を条件間で書き換えても
         同一envインスタンスに対する再jitでは古いコンパイル結果が
         再利用されてしまう(2つ目以降のDR条件が1つ目の設定のまま
         実行される、気付きにくい誤結果)。DRスコープ確定後に毎回
         新しいenvインスタンスを作ることで回避した。
       - スクリプト自身の--max-stepsが環境本来のMAX_EPISODE_STEPSより
         小さい場合、terminated/truncatedのどちらも立たないままループが
         尽きることがある。これを終了理由に混ぜず
         "eval_budget_cutoff"として区別し、Kaplan-Meier計算上も
         event(実イベント)ではなくcensoredとして扱うようにした。
     未学習ランダムpolicyでの動作確認であり、実際に学習済み
     checkpointとGPU/WSL環境で実行した結果ではない。次の残作業は、
     WSL/GPU環境で実checkpointに対して
     `python scratch/phase0_eval_diagnostics.py --exp_name <name>` を
     実行し、結果を docs/status.md ・ docs/gate_a_diagnosis.md に
     記録すること。
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CPU固定はデフォルトのみ。GPU評価したい場合は呼び出し前に環境変数を上書きすること。
os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ============================================================================
# 純Python/NumPyの解析ロジック（JAX/MJX非依存、単体テスト対象）
# ============================================================================

def classify_termination_reason(
    is_fallen_roll: bool,
    is_fallen_pitch: bool,
    is_low: bool,
    truncated: bool,
) -> str:
    """終了理由を分類する。

    master_plan.md 付録A §1.5 の終了条件定義のうち、現行コード
    (envs/mjx_rewards.py) が実装しているのは roll/pitch/height の3つと
    time-limitのみ。non_illegal_contact / slip_ok / torque_ok による
    terminationは未実装のため、このスクリプトでも分類できない
    （Task4 C-08 未着手であることの根拠として記録する）。
    """
    if truncated:
        return "time_limit"
    reasons = []
    if is_fallen_roll:
        reasons.append("fallen_roll")
    if is_fallen_pitch:
        reasons.append("fallen_pitch")
    if is_low:
        reasons.append("fallen_height")
    if not reasons:
        # terminated=Trueだが既知のフラグがどれも立っていない場合。
        # 実装上は起こらないはずだが、バグ検知のため明示的に区別する。
        return "unknown_terminated"
    return "+".join(reasons)


def kaplan_meier_survival(
    episode_lengths: Sequence[int],
    event_observed: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier生存曲線を計算する（500stepで打ち切られる右側打ち切り分布）。

    Args:
        episode_lengths: 各episodeが終了した(打ち切られた)step数。
        event_observed: Trueなら真のterminationイベント、Falseなら
            time-limitによる打ち切り(censoring)。

    Returns:
        (times, survival): times[0]=0, survival[0]=1.0 から始まる
        ステップ関数のノード列。
    """
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    events = np.asarray(event_observed, dtype=bool)
    if len(lengths) == 0:
        return np.array([0]), np.array([1.0])
    if len(lengths) != len(events):
        raise ValueError("episode_lengths and event_observed must be same length")

    event_times = np.unique(lengths[events])
    times = [0]
    survival = [1.0]
    s = 1.0
    for t in sorted(event_times.tolist()):
        n_t = int(np.sum(lengths >= t))  # tの直前時点でまだ生存(risk set)にいる数
        d_t = int(np.sum((lengths == t) & events))  # t時点での真のイベント数
        if n_t > 0:
            s *= (1.0 - d_t / n_t)
        times.append(int(t))
        survival.append(s)
    return np.array(times), np.array(survival)


def diagnose_failure_timing(
    termination_steps: Sequence[int],
    max_step: int,
    early_frac: float = 1.0 / 3.0,
    late_frac: float = 2.0 / 3.0,
    concentration_threshold: float = 0.6,
) -> Dict[str, object]:
    """master_plan.md 付録A §3.6 の決定木を単純な閾値ヒューリスティックで実装する。

    real terminationのみ(truncatedは除く)を入力に使うこと。
    """
    steps = np.asarray(termination_steps, dtype=np.float64)
    if len(steps) == 0:
        return {
            "classification": "no_failures",
            "suggested_action": (
                "terminatedによる失敗episodeが観測されなかった。"
                "time-limit到達のみであれば§3.3(truncation/termination処理)の"
                "疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。"
            ),
            "normalized_mean": None,
            "early_rate": None,
            "late_rate": None,
        }

    normalized = steps / float(max(max_step, 1))
    early_rate = float(np.mean(normalized < early_frac))
    late_rate = float(np.mean(normalized > late_frac))
    normalized_mean = float(np.mean(normalized))

    if early_rate >= concentration_threshold:
        classification = "序盤集中"
        suggested_action = (
            "失敗がepisode序盤に集中 → 初期状態・初期transientの問題の疑い。"
            "初期状態分布の縮小・初期姿勢安定化を検討する（master_plan.md §3.6）。"
        )
    elif late_rate >= concentration_threshold:
        classification = "後半集中"
        suggested_action = (
            "失敗がepisode後半に集中 → 長期ドリフト or time-limitバグの疑い。"
            "truncation/termination処理(§3.3)を再疑う。"
        )
    else:
        classification = "ランダム分布"
        suggested_action = (
            "失敗時刻がランダムに分布 → 状態空間の局所不安定領域の疑い。"
            "失敗直前の状態を特定し、該当領域の報酬/観測を強化する。"
        )

    return {
        "classification": classification,
        "suggested_action": suggested_action,
        "normalized_mean": normalized_mean,
        "early_rate": early_rate,
        "late_rate": late_rate,
    }


def summarize_episode_alive(episode_lengths: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(episode_lengths, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


# ============================================================================
# ロールアウト（JAX/MJX依存、GPU/WSL環境での実行を想定）
# ============================================================================

@dataclass
class EpisodeResult:
    length: int
    terminated: bool
    truncated: bool
    reason: str
    collapse_window: List[dict] = field(default_factory=list)
    reward_component_means: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    both_feet_contact: bool = False
    max_foot_displacement: float = 0.0
    max_roll_rad: float = 0.0
    max_pitch_rad: float = 0.0
    recovery_time_steps: Optional[int] = None
    torque_saturation_rate: float = 0.0


def _lazy_imports():
    """JAX/MJX関連のimportを遅延させ、--help等をGPU無し環境でも高速に扱えるようにする。"""
    import jax  # noqa: F401
    import jax.numpy as jp  # noqa: F401
    from robot.config import RobotConfig
    from envs.mjx_env import SenpuuMaruMJXEnv
    from robot.math_utils import quat_to_euler
    from train.visualize_rl import (
        get_model_path,
        load_checkpoint,
        make_rma_network_factory,
    )
    from brax.training.agents.ppo import networks as ppo_networks

    return {
        "jax": jax,
        "jp": jp,
        "RobotConfig": RobotConfig,
        "SenpuuMaruMJXEnv": SenpuuMaruMJXEnv,
        "quat_to_euler": quat_to_euler,
        "get_model_path": get_model_path,
        "load_checkpoint": load_checkpoint,
        "make_rma_network_factory": make_rma_network_factory,
        "ppo_networks": ppo_networks,
    }


class _DomainRandomizationScope:
    """RobotConfigのDR幅を一時的に固定値へ差し替え、終了時に復元するコンテキストマネージャ。

    物理初期姿勢(qpos/qvel)はreset()で常に固定のため、これは
    「初期状態randomize」軸の近似実装であることに注意
    (モジュールdocstring参照)。
    """

    FIELDS = (
        "RANDOM_MASS_SCALE",
        "RANDOM_FRICTION",
        "RANDOM_COM_OFFSET",
        "RANDOM_TEMP",
        "RANDOM_VOLT",
    )

    def __init__(self, RobotConfig, fixed: bool):
        self._cfg = RobotConfig
        self._fixed = fixed
        self._saved = {}

    def __enter__(self):
        for name in self.FIELDS:
            self._saved[name] = getattr(self._cfg, name)
        if self._fixed:
            # 全フィールドは [lo, hi] のスカラー対 (envs/mjx_env.py の reset() が
            # minval=X[0], maxval=X[1] として読む前提と一致させる)。
            for name in self.FIELDS:
                lo, hi = self._saved[name]
                mid = (lo + hi) / 2.0
                setattr(self._cfg, name, [mid, mid])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, value in self._saved.items():
            setattr(self._cfg, name, value)
        return False


def run_episode(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    rng,
    max_steps: int,
    collapse_window: int,
) -> EpisodeResult:
    """1エピソードをロールアウトする。

    重要: reset_fn/step_fnは呼び出し側で必ず jax.jit(env.reset) /
    jax.jit(env.step) として渡すこと。env.reset/env.stepを素の(非jit)
    状態で呼ぶと、reset()内で `info['step'] = 0` のようにPython int
    リテラルとして初期化されたフィールドがPython int のまま
    stepに渡り、`truncated.astype(...)` (envs/mjx_env.py) で
    `AttributeError: 'bool' object has no attribute 'astype'` になる
    (jitされた関数の戻り値はJAXが自動的に配列型へ変換するため、
    jit経由なら発生しない。本スクリプト作成時にeager実行で実際に
    再現・確認済み)。
    """
    RobotConfig = ctx["RobotConfig"]
    quat_to_euler = ctx["quat_to_euler"]

    rng, rng_reset = ctx["jax"].random.split(rng)
    state = reset_fn(rng_reset)

    history = []
    metric_sums: Dict[str, float] = {}
    metric_count = 0
    initial_foot_positions = None
    max_foot_displacement = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    both_feet_contact = True
    recovery_start = None
    recovery_time_steps = None
    saturated_steps = 0
    measured_torque_steps = 0

    terminated = False
    truncated = False
    step_index = 0
    for step_index in range(1, max_steps + 1):
        rng, rng_step = ctx["jax"].random.split(rng)
        action, _ = policy_fn(state.obs, rng_step)
        state = step_fn(state, action)

        qpos = np.asarray(state.pipeline_state.qpos)
        qvel = np.asarray(state.pipeline_state.qvel)
        rpy = np.asarray(quat_to_euler(state.pipeline_state.qpos[3:7]))
        base_pos = qpos[0:3]
        base_ang_vel = qvel[3:6] if len(qvel) >= 6 else np.zeros(3)
        xpos = np.asarray(state.pipeline_state.xpos)
        foot_ids = ctx.get("foot_ids")
        if foot_ids is not None and xpos.ndim == 2:
            foot_positions = xpos[list(foot_ids)]
            if initial_foot_positions is None:
                initial_foot_positions = foot_positions.copy()
            max_foot_displacement = max(
                max_foot_displacement,
                float(np.max(np.linalg.norm(foot_positions[:, :2] - initial_foot_positions[:, :2], axis=1))),
            )
        max_roll = max(max_roll, abs(float(rpy[0])))
        max_pitch = max(max_pitch, abs(float(rpy[1])))
        contact_metric = float(np.asarray(getattr(state, "metrics", {}).get("both_feet_contact", 0.0)))
        both_feet_now = contact_metric >= 0.5
        both_feet_contact = both_feet_contact and both_feet_now
        if bool(state.info.get("was_disturbed", False)) and recovery_start is None:
            recovery_start = step_index
        if recovery_start is not None and recovery_time_steps is None:
            if (both_feet_now and abs(rpy[0]) < np.deg2rad(10.0)
                    and abs(rpy[1]) < np.deg2rad(10.0)
                    and np.linalg.norm(base_ang_vel[:2]) < 0.5):
                recovery_time_steps = step_index - recovery_start
        torque = np.asarray(getattr(state.pipeline_state, "actuator_force", []))
        if torque.size:
            measured_torque_steps += 1
            limit = np.asarray(ctx["torque_limit"])
            saturated_steps += int(np.any(np.abs(torque) >= 0.98 * limit))

        is_fallen_roll = bool(abs(rpy[0]) > RobotConfig.TERMINATION_ROLL)
        is_fallen_pitch = bool(abs(rpy[1]) > RobotConfig.TERMINATION_PITCH)
        is_low = bool(base_pos[2] < RobotConfig.TERMINATION_HEIGHT)

        history.append({
            "step": step_index,
            "roll_rad": float(rpy[0]),
            "pitch_rad": float(rpy[1]),
            "base_pos": [float(v) for v in base_pos],
            "base_ang_vel": [float(v) for v in base_ang_vel],
            "is_fallen_roll": is_fallen_roll,
            "is_fallen_pitch": is_fallen_pitch,
            "is_low": is_low,
        })
        if len(history) > collapse_window:
            history.pop(0)

        metrics = getattr(state, "metrics", {}) or {}
        for key, value in metrics.items():
            try:
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
        metric_count += 1

        info = state.info
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if terminated or truncated:
            break

    if terminated:
        reason = classify_termination_reason(
            is_fallen_roll=history[-1]["is_fallen_roll"] if history else False,
            is_fallen_pitch=history[-1]["is_fallen_pitch"] if history else False,
            is_low=history[-1]["is_low"] if history else False,
            truncated=False,
        )
    elif truncated:
        reason = "time_limit"
    else:
        # env自身のterminated/truncatedがどちらも立たないまま、この関数の
        # max_stepsループを使い切った状態。これは真のepisode終了ではなく、
        # 呼び出し側のmax_stepsがRobotConfig.MAX_EPISODE_STEPSより小さい
        # 場合にのみ起こる「評価予算による打ち切り」であり、
        # is_fallen_*フラグの状態に関わらずtermination reasonとしては
        # 扱わない(=真のterminationイベントとして誤集計しない)。
        reason = "eval_budget_cutoff"

    reward_component_means = {
        key: value / metric_count for key, value in metric_sums.items()
    } if metric_count else {}

    has_required_contact = both_feet_contact
    success = (
        not terminated and truncated and has_required_contact
        and max_roll <= RobotConfig.TERMINATION_ROLL
        and max_pitch <= RobotConfig.TERMINATION_PITCH
        and max_foot_displacement <= RobotConfig.MAX_FOOT_TRANSLATION
    )

    return EpisodeResult(
        length=step_index,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        collapse_window=history if terminated else [],
        reward_component_means=reward_component_means,
        success=success,
        both_feet_contact=has_required_contact,
        max_foot_displacement=max_foot_displacement,
        max_roll_rad=max_roll,
        max_pitch_rad=max_pitch,
        recovery_time_steps=recovery_time_steps,
        torque_saturation_rate=(saturated_steps / measured_torque_steps
                    if measured_torque_steps else 0.0),
    )


def run_condition(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    n_episodes: int,
    base_seed: int,
    max_steps: int,
    collapse_window: int,
) -> dict:
    lengths, terminated_flags, truncated_flags, reasons = [], [], [], []
    reward_component_accum: Dict[str, List[float]] = {}
    collapse_examples = []
    successes = 0
    foot_displacements = []
    recovery_times = []
    torque_saturation_rates = []
    max_rolls = []
    max_pitches = []
    contact_successes = 0

    rng = ctx["jax"].random.PRNGKey(base_seed)
    for ep in range(n_episodes):
        rng, rng_ep = ctx["jax"].random.split(rng)
        result = run_episode(ctx, reset_fn, step_fn, policy_fn, rng_ep, max_steps, collapse_window)
        lengths.append(result.length)
        terminated_flags.append(result.terminated)
        truncated_flags.append(result.truncated)
        reasons.append(result.reason)
        successes += int(result.success)
        contact_successes += int(result.both_feet_contact)
        foot_displacements.append(result.max_foot_displacement)
        max_rolls.append(result.max_roll_rad)
        max_pitches.append(result.max_pitch_rad)
        torque_saturation_rates.append(result.torque_saturation_rate)
        if result.recovery_time_steps is not None:
            recovery_times.append(result.recovery_time_steps)
        for key, value in result.reward_component_means.items():
            reward_component_accum.setdefault(key, []).append(value)
        if result.terminated and len(collapse_examples) < 5:
            collapse_examples.append({
                "episode": ep,
                "length": result.length,
                "reason": result.reason,
                "window": result.collapse_window,
            })

    event_observed = terminated_flags  # True=event(termination), False=censored(time_limit)
    km_times, km_survival = kaplan_meier_survival(lengths, event_observed)

    real_failure_steps = [l for l, t in zip(lengths, terminated_flags) if t]
    timing_diag = diagnose_failure_timing(real_failure_steps, max_steps)

    return {
        "n_episodes": n_episodes,
        "episode_alive": summarize_episode_alive(lengths),
        "termination_reason_counts": dict(Counter(reasons)),
        "termination_reason_rate": {
            k: v / n_episodes for k, v in Counter(reasons).items()
        },
        "kaplan_meier": {"times": km_times.tolist(), "survival": km_survival.tolist()},
        "failure_timing_diagnosis": timing_diag,
        "success_rate": successes / n_episodes if n_episodes else 0.0,
        "both_feet_contact_rate": contact_successes / n_episodes if n_episodes else 0.0,
        "max_foot_displacement_m": float(max(foot_displacements, default=0.0)),
        "max_roll_deg": float(np.rad2deg(max(max_rolls, default=0.0))),
        "max_pitch_deg": float(np.rad2deg(max(max_pitches, default=0.0))),
        "recovery_time_steps": recovery_times,
        "recovery_time_mean_steps": float(np.mean(recovery_times)) if recovery_times else None,
        "torque_saturation_rate_mean": float(np.mean(torque_saturation_rates)) if torque_saturation_rates else 0.0,
        "reward_component_means": {
            key: float(np.mean(vals)) for key, vals in reward_component_accum.items()
        },
        "collapse_examples": collapse_examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="", help="log/<exp_name> 配下のcheckpointを使う")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--model", default="best_params.pkl")
    parser.add_argument("--episodes", type=int, default=20, help="stochastic/randomizedセルのepisode数")
    parser.add_argument(
        "--fixed-episodes", type=int, default=3,
        help="deterministic x fixed_dr セルのepisode数(再現性確認用、通常は少数でよい)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="未指定ならRobotConfig.MAX_EPISODE_STEPS")
    parser.add_argument("--collapse-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-levels", default=None,
        help="評価する外乱力[N]をカンマ区切りで指定。未指定はRobotConfig.PUSH_FORCE_LEVELS",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "log" / "phase0_eval_diagnostics.json")
    parser.add_argument(
        "--diagnosis-md", type=Path, default=ROOT / "docs" / "gate_a_diagnosis.md",
        help="§3.6決定木の一次判定ドラフトを書き出す先(人間/Copilotによるレビュー前提)",
    )
    args = parser.parse_args()

    ctx = _lazy_imports()
    RobotConfig = ctx["RobotConfig"]
    SenpuuMaruMJXEnv = ctx["SenpuuMaruMJXEnv"]
    ppo_networks = ctx["ppo_networks"]

    # Gate AはPhase 0 (無外乱)の診断であるため、外乱は明示的に無効化する。
    RobotConfig.DISTURBANCE_CURRICULUM = False
    RobotConfig.RANDOM_PUSH_MAX_FORCE = 0.0

    max_steps = args.max_steps or RobotConfig.MAX_EPISODE_STEPS
    if max_steps < RobotConfig.MAX_EPISODE_STEPS:
        print(
            f"[Phase0 Eval][WARN] --max-steps={max_steps} < "
            f"RobotConfig.MAX_EPISODE_STEPS={RobotConfig.MAX_EPISODE_STEPS}. "
            "env自身のtime-limit(truncated)に到達する前にロールアウトを打ち切るため、"
            "'eval_budget_cutoff'エピソードが混入しうる(これはtime_limitでも"
            "termination失敗でもない)。開発中の高速確認用途以外では"
            "--max-stepsを指定しないことを推奨する。"
        )

    model_path = ctx["get_model_path"](args.exp_name, args.version, args.model)
    if model_path is None:
        raise SystemExit(
            f"checkpoint not found for exp_name={args.exp_name!r}, version={args.version}, "
            f"model={args.model!r}. --exp_name / --version / --model を確認してください。"
        )
    params = ctx["load_checkpoint"](model_path)

    # obs/action次元はDR設定に依存しないstructuralな値なので、使い捨てのenv
    # インスタンスから一度だけ取得すれば十分(policy networkの構築もここでよい)。
    _probe_env = SenpuuMaruMJXEnv()
    network = ctx["make_rma_network_factory"](_probe_env.observation_size, _probe_env.action_size)
    make_policy = ppo_networks.make_inference_fn(network)

    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf

    params_stripped = ctx["jax"].tree_util.tree_map(strip_leading_dim, params)

    # deterministic/stochasticはpolicyのみに依存するため一度だけjitする。
    policy_fns = {
        det: ctx["jax"].jit(make_policy(params_stripped, deterministic=det))
        for det in (True, False)
    }

    force_levels = (
        [float(value) for value in args.force_levels.split(",")]
        if args.force_levels else list(RobotConfig.PUSH_FORCE_LEVELS)
    )
    conditions = [
        ("deterministic", "fixed_dr", True, True, args.fixed_episodes, 0.0),
        ("deterministic", "randomized_dr", True, False, args.episodes, 0.0),
        ("stochastic", "fixed_dr", False, True, args.episodes, 0.0),
        ("stochastic", "randomized_dr", False, False, args.episodes, 0.0),
    ]
    conditions.extend(
        ("deterministic", f"push_{force:g}N", True, False, args.episodes, force)
        for force in force_levels if force > 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(model_path),
        "max_steps": max_steps,
        "note_initial_state_randomization": (
            "物理初期姿勢(qpos/qvel)は常にnominal poseに固定されている(未実装機能、"
            "master_plan.md付録A §1.6参照)。ここでの'randomized_dr'は質量/摩擦/"
            "重心オフセット/サーボ温度/電圧のdomain randomizationのon/offを指す近似軸。"
        ),
        "note_termination_reasons": (
            "現行実装(envs/mjx_rewards.py)はfallen_roll/fallen_pitch/fallen_height/"
            "time_limitのみをterminationとして判定する。non_illegal_contact/slip_ok/"
            "torque_okによるterminationは未実装(master_plan.md Task4 C-08未着手)。"
        ),
        "conditions": {},
        "disturbance_model": {
            "force_levels_N": force_levels,
            "directions": int(RobotConfig.PUSH_DIRECTIONS),
            "duration_steps": int(RobotConfig.PUSH_DURATION_STEPS),
            "duration_s": float(RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT),
            "impulse_levels_Ns": [
                float(force * RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT)
                for force in force_levels
            ],
            "implementation": "MJX random horizontal push; direction is sampled continuously",
        },
    }

    for label, dr_label, deterministic, fixed_dr, n_episodes, push_force in conditions:
        policy_fn = policy_fns[deterministic]
        RobotConfig.RANDOM_PUSH_MAX_FORCE = push_force
        RobotConfig.DISTURBANCE_CURRICULUM = push_force > 0.0
        with _DomainRandomizationScope(RobotConfig, fixed=fixed_dr):
            # env.reset/step本体は `minval=RobotConfig.RANDOM_MASS_SCALE[0]` の
            # ようにRobotConfigのクラス属性をトレース時にPython定数として
            # 直接埋め込む。jax.jitのコンパイルキャッシュはbound method
            # (env.reset)の等価性で引かれるため、同じenvインスタンスに対して
            # 単に`jax.jit(env.reset)`を呼び直すだけでは、RobotConfigを
            # 変更後でも古いコンパイル結果が再利用されてしまい、
            # 2つ目以降の条件が1つ目のDR設定のまま実行される
            # ——という気付きにくい誤結果を生む。これはこのスクリプト作成時に
            # 実機で再現・確認した(jax.jit(env.reset)を使い回すとDR変更が
            # 反映されず、envインスタンスを条件ごとに新規作成するか
            # jax.clear_caches()を呼べば正しく反映されることを確認済み)。
            # 最も単純で既存コード(scratch/gate0_formal_eval.pyの
            # configure→インスタンス化の順序)とも整合する対策として、
            # DR設定確定後に毎回新しいenvインスタンスを作る。
            env = SenpuuMaruMJXEnv()
            ctx["foot_ids"] = (env._reward_system._left_foot_id, env._reward_system._right_foot_id)
            ctx["torque_limit"] = np.asarray(env._mjx_model.actuator_ctrlrange[:, 1])
            reset_fn = ctx["jax"].jit(env.reset)
            step_fn = ctx["jax"].jit(env.step)
            result = run_condition(
                ctx, reset_fn, step_fn, policy_fn,
                n_episodes=n_episodes,
                base_seed=args.seed,
                max_steps=max_steps,
                collapse_window=args.collapse_window,
            )
        key = f"{label}__{dr_label}"
        report["conditions"][key] = result
        print(f"[{key}] episode_alive mean={result['episode_alive']['mean']:.1f} "
              f"reasons={result['termination_reason_counts']} "
              f"timing={result['failure_timing_diagnosis']['classification']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Phase0 Eval] detailed report: {args.out}")

    _write_diagnosis_draft(args.diagnosis_md, report)
    print(f"[Phase0 Eval] diagnosis draft: {args.diagnosis_md}")


def _write_diagnosis_draft(path: Path, report: dict) -> None:
    lines = [
        "# Gate A 診断ドラフト（自動生成 — 人間 / Copilotによるレビュー必須）",
        "",
        f"生成日時: {report['generated_at']}",
        f"checkpoint: {report['checkpoint']}",
        "",
        "この文書は scratch/phase0_eval_diagnostics.py により自動生成された一次判定です。",
        "master_plan.md §3.7 (Task0完了基準) の「§3.6の決定木に基づく主因の暫定結論」",
        "に相当しますが、機械的な閾値ヒューリスティックによる分類であり、",
        "最終結論には人間またはCopilotによるログ・collapse_examplesの目視確認を要します。",
        "",
        f"- {report['note_initial_state_randomization']}",
        f"- {report['note_termination_reasons']}",
        "",
        "## 条件別サマリー",
        "",
    ]
    for key, result in report["conditions"].items():
        ea = result["episode_alive"]
        diag = result["failure_timing_diagnosis"]
        lines.append(f"### {key}")
        lines.append(
            f"- episode_alive: mean={ea['mean']:.1f}, std={ea['std']:.1f}, "
            f"n={ea['n']}"
        )
        lines.append(f"- termination reasons: {result['termination_reason_counts']}")
        lines.append(f"- failure timing: {diag['classification']}")
        lines.append(f"- suggested action: {diag['suggested_action']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
