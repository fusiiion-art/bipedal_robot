#!/usr/bin/env python3
"""Phase 0 / Gate A diagnosis: deterministic vs stochastic evaluation +
termination-reason histogram (docs/master_plan.md 付録A §3.5, Task0).

status.md (2026-09-01) の「次のTask」= 「deterministic/stochastic評価の実装と
終了理由ヒストグラム化」に対応する。エスカレーション項目の「次の切り分け」の
3項目のうち、deterministic評価・終了stepヒストグラム化・報酬成分分解ログの
3つをまとめてこのスクリプトで実施する。

設計方針（master_plan.md §3.5 に基づく）:
  - 2x2評価: {deterministic, stochastic} x {fixed_dr, randomized_dr}
    初期状態分布(master_plan.md §1.6, envs/mjx_env.py の reset())は全セルで有効。
    fixed_dr/randomized_dr は domain randomization (質量/摩擦/重心オフセット/
    サーボ温度/電圧) の幅を中央値に固定するか否か。
  - deterministic x fixed_dr のセルは DR の影響を除いた参考セルで、
    デフォルトのepisode数を他セルより少なくしている。
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
import hashlib
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
    physics_diverged: bool = False,
    reward_is_finite: bool = True,
) -> str:
    """終了理由を分類する。

    master_plan.md 付録A §1.5 の終了条件定義のうち、現行コード
    (envs/mjx_rewards.py) が実装しているのは roll/pitch/height の3つと
    time-limitのみ。non_illegal_contact / slip_ok / torque_ok による
    terminationは未実装のため、このスクリプトでも分類できない
    （Task4 C-08 未着手であることの根拠として記録する）。

    [改造 2026-09-13] envs/mjx_env.py の物理発散ロールバック機構、
    envs/mjx_rewards.py の報酬NaN無害化機構の追加に伴い、doneが
    roll/pitch/height以外の理由(物理シミュレーションの数値発散、
    報酬計算のNaN)でもTrueになるようになった。これらは「本物の
    転倒」ではなく「数値的な安全装置の作動」であり、is_fallen_*では
    検出できないため、従来はunknown_terminatedに埋もれ、実際の
    発生頻度が見えなくなっていた。physics_diverged/reward_is_finite
    (envs/mjx_env.py, envs/mjx_rewards.py が state.metrics に記録する
    フラグ) を渡すことで、これらを明示的に分類できるようにする。
    デフォルト値は既存の呼び出し・テストとの後方互換性のため
    「発生していない」側に設定してある。
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
    if physics_diverged:
        reasons.append("physics_diverged")
    if not reward_is_finite:
        reasons.append("reward_nan")
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


def track_sustained_contact_loss(
    both_feet_now: bool,
    consecutive_loss_steps: int,
    real_loss: bool,
    grace_steps: int,
) -> Tuple[int, bool]:
    """1step分の両足接地判定を受け取り、連続`grace_steps`以上の非接触が
    続いた場合にのみ「本物の接地喪失」として確定させる。

    [2026-09-26追加] 以前は`both_feet_contact and both_feet_now`という
    全step ANDのラッチで、1step分の瞬間的なFSR推定値のブレだけで
    episode全体が恒久的に「両足接地失敗」として扱われ、success判定を
    実態以上に厳しく倒す既知バグがあった(grace_steps=1相当)。
    master_plan.md §6.1がGate Bの外乱復帰猶予として定義している
    「両足の鉛直接触力が同時に閾値未満、かつ連続20ms以上」という
    考え方をGate Aのsuccess判定にも適用する。real_loss=Trueに一度
    なった後は接触が戻ってもTrueのまま(=episodeを通じた恒久的な
    「本物の接地喪失」フラグ)。
    """
    if both_feet_now:
        return 0, real_loss
    consecutive_loss_steps += 1
    if consecutive_loss_steps >= grace_steps:
        real_loss = True
    return consecutive_loss_steps, real_loss


def tilt_from_quat(quat) -> float:
    """胴体z軸とworld鉛直のなす角 [rad] (= arccos(R[2,2]))。quat は [w, x, y, z]。"""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def track_illegal_contact(
    illegal_now: bool,
    consecutive_steps: int,
    violations: int,
    required_steps: int,
) -> Tuple[int, int]:
    """[T2] 床と足裏以外の接触が `required_steps` 以上連続したら違反1回と数える。

    1回の連続区間は長さに関わらず違反1回 (区間が途切れて再発したら別の1回)。"""
    if not illegal_now:
        return 0, violations
    consecutive_steps += 1
    if consecutive_steps == required_steps:
        violations += 1
    return consecutive_steps, violations


def illegal_floor_contact(geom1, geom2, dist, geom_bodyid, sole_bodies) -> bool:
    """接触候補のうち実際に接触している(dist<0)ものに、床(body 0)と足裏以外のbodyの組があるか。"""
    b1 = np.asarray(geom_bodyid)[np.asarray(geom1)]
    b2 = np.asarray(geom_bodyid)[np.asarray(geom2)]
    active = np.asarray(dist) < 0.0
    other = np.where(b1 == 0, b2, b1)
    floor_pair = (b1 == 0) | (b2 == 0)
    return bool(np.any(active & floor_pair & ~np.isin(other, list(sole_bodies))))


def gate_a_criteria(
    *,
    terminated: bool,
    truncated: bool,
    both_feet_contact: bool,
    max_tilt_deg: float,
    max_foot_displacement: float,
    torque_saturation_rate: float,
    rel_height_drop: float,
    illegal_contact_events: int,
    cfg,
) -> Dict[str, bool]:
    """[T2] master_plan §0.3 の成功条件の各項と、その論理積 ('success')。閾値は robot/config.py。"""
    checks = {
        "alive": (not terminated) and truncated,
        "both_feet_contact": both_feet_contact,
        "upright": max_tilt_deg <= cfg.GATE_A_MAX_TILT_DEG,
        "slip_ok": max_foot_displacement <= cfg.MAX_FOOT_TRANSLATION,
        "torque_ok": torque_saturation_rate <= cfg.GATE_A_MAX_TORQUE_SAT_RATE,
        "height_ok": rel_height_drop <= cfg.GATE_A_MAX_REL_HEIGHT_DROP,
        "no_illegal_contact": illegal_contact_events == 0,
    }
    checks["success"] = all(checks.values())
    return checks


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
    max_foot_displacement_since_settling: float = 0.0
    max_roll_rad: float = 0.0
    max_pitch_rad: float = 0.0
    recovery_time_steps: Optional[int] = None
    torque_saturation_rate: float = 0.0
    final_state_digest: str = ""
    criteria: Dict[str, bool] = field(default_factory=dict)
    max_tilt_deg: float = 0.0
    rel_height_drop: float = 0.0
    illegal_contact_events: int = 0
    action_change_rms: float = 0.0
    joint_vel_rms: float = 0.0


def state_digest(qpos, qvel, decimals: int = 6) -> str:
    """終端状態(qpos/qvel)の指紋。同一軌道の重複(擬似反復)検出に使う。"""
    arr = np.concatenate([np.asarray(qpos, dtype=np.float64).ravel(),
                          np.asarray(qvel, dtype=np.float64).ravel()])
    rounded = np.round(arr, decimals) + 0.0  # -0.0 と 0.0 を同一視する
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def interpolate_threshold_crossing(
    x_values: Sequence[float],
    y_values: Sequence[float],
    target: float = 0.5,
) -> Optional[float]:
    """成功率(y_values)が`target`(既定0.5、master_plan.md §6.4のJ_50/θ_50定義)を
    跨ぐ点を、外乱強度(x_values)に対する線形補間で求める。

    x_valuesは外乱強度の昇順、y_valuesは外乱強度が強くなるほど単調非増加で
    あることを想定するが、評価ノイズによる多少の前後は許容し、targetを跨ぐ
    最初の隣接区間で補間する。全区間でtargetを跨がない場合は外挿せずNoneを
    返す(グリッド範囲が不足しているサイン。範囲を広げて再評価すること)。
    """
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    pairs = sorted(zip(x_values, y_values), key=lambda p: p[0])
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if (y0 - target) * (y1 - target) <= 0:
            if y1 == y0:
                return x0
            frac = (target - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0)
    return None


def _lazy_imports():
    """JAX/MJX関連のimportを遅延させ、--help等をGPU無し環境でも高速に扱えるようにする。"""
    import jax  # noqa: F401
    import jax.numpy as jp  # noqa: F401
    from robot.config import RobotConfig
    from envs.mjx_env import SenpuuMaruMJXEnv
    from robot.math_utils import quat_to_euler
    from robot.policy_network import find_checkpoint, load_checkpoint, make_inference_fn_from_params

    return {
        "jax": jax,
        "jp": jp,
        "RobotConfig": RobotConfig,
        "SenpuuMaruMJXEnv": SenpuuMaruMJXEnv,
        "quat_to_euler": quat_to_euler,
        "find_checkpoint": find_checkpoint,
        "load_checkpoint": load_checkpoint,
        "make_inference_fn_from_params": make_inference_fn_from_params,
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
    settling_steps: int = 50,
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
    # [2026-09-26追加] max_foot_displacement(既存、successの正式判定に使用)は
    # reset直後(step 1)を基準点にした累積最大値。reset姿勢が自然な静止姿勢と
    # 一致しない場合、最初の沈み込み・姿勢確立(settling)による1回限りの移動が
    # 「滑り」として累算され、以降ずっと定常状態でも良好であるかのような
    # ケースと見分けがつかない。settling_steps経過後を基準点にした
    # 補助指標を別途記録し、判定(success)には使わず診断専用とする
    # (§0.3: 閾値そのものの変更はrobot/config.py側で行うべきで、
    # このスクリプト側で勝手に緩めない)。
    settled_foot_positions = None
    max_foot_displacement_since_settling = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    both_feet_contact = True
    # [2026-09-26修正] track_sustained_contact_loss()参照。20ms未満の
    # 瞬間的な非接触はGate Aのsuccess判定において失敗として数えない。
    consecutive_contact_loss_steps = 0
    real_contact_loss = False
    contact_loss_grace_steps = max(1, round(0.020 / RobotConfig.CONTROL_DT))
    recovery_start = None
    recovery_time_steps = None
    saturated_steps = 0
    measured_torque_steps = 0
    # [T2] master_plan §0.3 の追加判定項目と、判定に使わないジッター指標
    foot_ids = ctx.get("foot_ids")
    max_tilt = 0.0
    initial_rel_height = _rel_height(state.pipeline_state, foot_ids)
    min_rel_height = initial_rel_height
    illegal_consecutive = 0
    illegal_events = 0
    prev_action = None
    action_change_sq_sum = 0.0
    action_change_count = 0
    joint_vel_sq_sum = 0.0
    joint_vel_count = 0

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

        action_np = np.asarray(action, dtype=np.float64)
        if prev_action is not None:
            action_change_sq_sum += float(np.sum(np.square(action_np - prev_action)))
            action_change_count += 1
        prev_action = action_np
        joint_qvel_idx = ctx.get("joint_qvel_idx")
        if joint_qvel_idx is not None:
            joint_vel = qvel[np.asarray(joint_qvel_idx)]
            joint_vel_sq_sum += float(np.sum(np.square(joint_vel)))
            joint_vel_count += joint_vel.size
        max_tilt = max(max_tilt, tilt_from_quat(qpos[3:7]))
        min_rel_height = min(min_rel_height, _rel_height(state.pipeline_state, foot_ids))
        if ctx.get("geom_bodyid") is not None:
            contact = state.pipeline_state.contact
            illegal_now = illegal_floor_contact(
                contact.geom1, contact.geom2, contact.dist, ctx["geom_bodyid"], ctx["sole_bodies"])
            illegal_consecutive, illegal_events = track_illegal_contact(
                illegal_now, illegal_consecutive, illegal_events, RobotConfig.GATE_A_ILLEGAL_CONTACT_STEPS)

        if foot_ids is not None and xpos.ndim == 2:
            foot_positions = xpos[list(foot_ids)]
            if initial_foot_positions is None:
                initial_foot_positions = foot_positions.copy()
            max_foot_displacement = max(
                max_foot_displacement,
                float(np.max(np.linalg.norm(foot_positions[:, :2] - initial_foot_positions[:, :2], axis=1))),
            )
            if step_index >= settling_steps:
                if settled_foot_positions is None:
                    settled_foot_positions = foot_positions.copy()
                max_foot_displacement_since_settling = max(
                    max_foot_displacement_since_settling,
                    float(np.max(np.linalg.norm(foot_positions[:, :2] - settled_foot_positions[:, :2], axis=1))),
                )
        max_roll = max(max_roll, abs(float(rpy[0])))
        max_pitch = max(max_pitch, abs(float(rpy[1])))
        contact_metric = float(np.asarray(getattr(state, "metrics", {}).get("both_feet_contact", 0.0)))
        both_feet_now = contact_metric >= 0.5
        consecutive_contact_loss_steps, real_contact_loss = track_sustained_contact_loss(
            both_feet_now, consecutive_contact_loss_steps, real_contact_loss, contact_loss_grace_steps,
        )
        both_feet_contact = not real_contact_loss
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
        # [2026-09-29修正] env(envs/mjx_rewards.py)と同じく足裏基準の相対高さで判定する。
        # 従来はワールド絶対Zと比較しており、終了理由の分類がenvと食い違い得た。
        if foot_ids is not None and xpos.ndim == 2:
            lowest_foot_z = float(np.min(xpos[list(foot_ids), 2]))
        else:
            lowest_foot_z = 0.0
        is_low = bool(base_pos[2] - lowest_foot_z < RobotConfig.TERMINATION_HEIGHT)

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
    torque_saturation_rate = saturated_steps / measured_torque_steps if measured_torque_steps else 0.0
    rel_height_drop = max(0.0, initial_rel_height - min_rel_height)
    criteria = gate_a_criteria(
        terminated=terminated,
        truncated=truncated,
        both_feet_contact=has_required_contact,
        max_tilt_deg=float(np.rad2deg(max_tilt)),
        max_foot_displacement=max_foot_displacement,
        torque_saturation_rate=torque_saturation_rate,
        rel_height_drop=rel_height_drop,
        illegal_contact_events=illegal_events,
        cfg=RobotConfig,
    )

    return EpisodeResult(
        criteria=criteria,
        max_tilt_deg=float(np.rad2deg(max_tilt)),
        rel_height_drop=rel_height_drop,
        illegal_contact_events=illegal_events,
        action_change_rms=float(np.sqrt(action_change_sq_sum / action_change_count)) if action_change_count else 0.0,
        joint_vel_rms=float(np.sqrt(joint_vel_sq_sum / joint_vel_count)) if joint_vel_count else 0.0,
        final_state_digest=state_digest(state.pipeline_state.qpos, state.pipeline_state.qvel),
        length=step_index,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        collapse_window=history if terminated else [],
        reward_component_means=reward_component_means,
        success=criteria["success"],
        both_feet_contact=has_required_contact,
        max_foot_displacement=max_foot_displacement,
        max_foot_displacement_since_settling=max_foot_displacement_since_settling,
        max_roll_rad=max_roll,
        max_pitch_rad=max_pitch,
        recovery_time_steps=recovery_time_steps,
        torque_saturation_rate=torque_saturation_rate,
    )


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


def _rel_height(pipeline_state, foot_ids) -> float:
    """胴体高さの、低い方の足裏bodyからの相対値 [m] (envs/mjx_rewards.py の終了判定と同じ定義)。"""
    qpos = np.asarray(pipeline_state.qpos)
    xpos = np.asarray(pipeline_state.xpos)
    if foot_ids is None or xpos.ndim != 2:
        return float(qpos[2])
    return float(qpos[2] - np.min(xpos[list(foot_ids), 2]))


def run_condition(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    n_episodes: int,
    base_seed: int,
    max_steps: int,
    collapse_window: int,
    settling_steps: int = 50,
) -> dict:
    lengths, terminated_flags, truncated_flags, reasons = [], [], [], []
    final_state_digests = []
    reward_component_accum: Dict[str, List[float]] = {}
    collapse_examples = []
    successes = 0
    foot_displacements = []
    foot_displacements_since_settling = []
    recovery_times = []
    torque_saturation_rates = []
    max_rolls = []
    max_pitches = []
    contact_successes = 0
    criteria_counts: Dict[str, int] = {}
    max_tilts, height_drops, illegal_events = [], [], []
    action_change_rms, joint_vel_rms = [], []

    rng = ctx["jax"].random.PRNGKey(base_seed)
    for ep in range(n_episodes):
        rng, rng_ep = ctx["jax"].random.split(rng)
        result = run_episode(ctx, reset_fn, step_fn, policy_fn, rng_ep, max_steps, collapse_window, settling_steps)
        lengths.append(result.length)
        terminated_flags.append(result.terminated)
        truncated_flags.append(result.truncated)
        reasons.append(result.reason)
        final_state_digests.append(result.final_state_digest)
        successes += int(result.success)
        contact_successes += int(result.both_feet_contact)
        foot_displacements.append(result.max_foot_displacement)
        foot_displacements_since_settling.append(result.max_foot_displacement_since_settling)
        max_rolls.append(result.max_roll_rad)
        max_pitches.append(result.max_pitch_rad)
        torque_saturation_rates.append(result.torque_saturation_rate)
        for name, ok in result.criteria.items():
            criteria_counts[name] = criteria_counts.get(name, 0) + int(ok)
        max_tilts.append(result.max_tilt_deg)
        height_drops.append(result.rel_height_drop)
        illegal_events.append(result.illegal_contact_events)
        action_change_rms.append(result.action_change_rms)
        joint_vel_rms.append(result.joint_vel_rms)
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

    n_unique_final_states = len(set(final_state_digests))
    return {
        "n_episodes": n_episodes,
        # [2026-10-02追加] Wilson区間は独立試行を前提とする。同一の終端状態に至った
        # episodeが複数あれば、それらは同一軌道の繰り返し(擬似反復)であり実効サンプル数は
        # n_episodesより小さい。scratch/gate_a_qualification.py がこの値を検査する。
        "n_unique_final_states": n_unique_final_states,
        "episode_alive": summarize_episode_alive(lengths),
        "termination_reason_counts": dict(Counter(reasons)),
        "termination_reason_rate": {
            k: v / n_episodes for k, v in Counter(reasons).items()
        },
        "kaplan_meier": {"times": km_times.tolist(), "survival": km_survival.tolist()},
        "failure_timing_diagnosis": timing_diag,
        "success_rate": successes / n_episodes if n_episodes else 0.0,
        "n_successes": successes,
        # [T2] master_plan §0.3 の各成功条件を満たしたepisodeの割合 (どの条件で落ちたかの切り分け用)
        "criteria_pass_rate": {
            name: count / n_episodes for name, count in criteria_counts.items()
        } if n_episodes else {},
        "max_tilt_deg_max": float(max(max_tilts, default=0.0)),
        "rel_height_drop_max_m": float(max(height_drops, default=0.0)),
        "illegal_contact_episode_count": int(sum(1 for e in illegal_events if e > 0)),
        "foot_displacement_m": _percentiles(foot_displacements),
        # 判定には使わない (レポートのみ): ジッター指標
        "action_change_rms": _percentiles(action_change_rms),
        "joint_vel_rms_rad_s": _percentiles(joint_vel_rms),
        "both_feet_contact_rate": contact_successes / n_episodes if n_episodes else 0.0,
        "max_foot_displacement_m": float(max(foot_displacements, default=0.0)),
        "max_foot_displacement_since_settling_m": float(max(foot_displacements_since_settling, default=0.0)),
        "note_foot_displacement_since_settling": (
            f"settling_steps={settling_steps}制御step経過後を基準点にした補助指標。"
            "successの正式判定には使わない(診断専用、§0.3準拠でrobot/config.pyの"
            "MAX_FOOT_TRANSLATIONは変更していない)。max_foot_displacement_mとの差が"
            "大きい場合、reset直後の沈み込み/姿勢確立が主因である可能性が高い。"
        ),
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
    parser.add_argument(
        "--zero-policy", action="store_true",
        help="[T2] checkpointを使わず常に行動0(デフォルト姿勢のPD保持)を返すベースライン方策で評価する",
    )
    # [T2] n=20 では全成功でも Wilson 95% 下限が 0.839 にしかならないため既定を200にする
    parser.add_argument("--episodes", type=int, default=200, help="stochastic/randomizedセルのepisode数")
    parser.add_argument(
        "--fixed-episodes", type=int, default=3,
        help="deterministic x fixed_dr セルのepisode数(再現性確認用、通常は少数でよい)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="未指定ならRobotConfig.MAX_EPISODE_STEPS")
    parser.add_argument("--collapse-window", type=int, default=20)
    parser.add_argument(
        "--settling-steps", type=int, default=50,
        help="max_foot_displacement_since_settling_m診断指標の基準点をreset後何control step目"
             "にするか(既定50step=CONTROL_DT基準で約0.5秒)。judgeには使わない。",
    )
    # [T2] 学習seed(0,1,2,...)と評価の乱数系列を分ける
    parser.add_argument("--seed", type=int, default=1000, help="評価seed (学習seedとは別系列)")
    parser.add_argument(
        "--force-levels", default=None,
        help="評価する外乱力[N]をカンマ区切りで指定。未指定はRobotConfig.PUSH_FORCE_LEVELS",
    )
    # [T2] 旧既定は全runで共有のパスで、seedごとに上書きされていた。既定はcheckpointと同じrun_dir。
    parser.add_argument(
        "--out", type=Path, default=None,
        help="詳細レポート(JSON)。既定は <checkpointのrun_dir>/gate_a.json (--zero-policy時は必須)",
    )
    parser.add_argument(
        "--diagnosis-md", type=Path, default=None,
        help="§3.6決定木の一次判定ドラフト。既定は --out と同じディレクトリの gate_a_diagnosis.md",
    )
    args = parser.parse_args()

    ctx = _lazy_imports()
    RobotConfig = ctx["RobotConfig"]
    SenpuuMaruMJXEnv = ctx["SenpuuMaruMJXEnv"]

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

    if args.zero_policy:
        if args.out is None:
            raise SystemExit("--zero-policy では --out を指定してください (例: log/baseline_<commit>/gate_a_zero.json)")
        model_path = None
        policy_label = "zero-policy"
        zero_action = ctx["jp"].zeros(RobotConfig.NUM_JOINTS)

        def zero_policy(obs, rng):
            return zero_action, {}

        policy_fns = {True: zero_policy, False: zero_policy}
    else:
        model_path = ctx["find_checkpoint"](args.exp_name, args.version, args.model)
        if model_path is None:
            raise SystemExit(
                f"checkpoint not found for exp_name={args.exp_name!r}, version={args.version}, "
                f"model={args.model!r}. --exp_name / --version / --model を確認してください。"
            )
        policy_label = str(model_path)
        params = ctx["load_checkpoint"](model_path)

        # deterministic/stochasticはpolicyのみに依存するため一度だけjitする。
        policy_fns = {
            det: ctx["jax"].jit(ctx["make_inference_fn_from_params"](params, deterministic=det))
            for det in (True, False)
        }
    out_path = args.out or model_path.parent / "gate_a.json"
    diagnosis_md = args.diagnosis_md or out_path.parent / "gate_a_diagnosis.md"

    force_levels = (
        [float(value) for value in args.force_levels.split(",")]
        if args.force_levels else list(getattr(RobotConfig, 'PUSH_FORCE_LEVELS', []))
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
        "checkpoint": policy_label,
        "zero_policy": bool(args.zero_policy),
        "eval_seed": args.seed,
        "max_steps": max_steps,
        "gate_a_thresholds": {
            "max_tilt_deg": RobotConfig.GATE_A_MAX_TILT_DEG,
            "max_foot_translation_m": RobotConfig.MAX_FOOT_TRANSLATION,
            "max_torque_saturation_rate": RobotConfig.GATE_A_MAX_TORQUE_SAT_RATE,
            "max_rel_height_drop_m": RobotConfig.GATE_A_MAX_REL_HEIGHT_DROP,
            "illegal_contact_consecutive_steps": RobotConfig.GATE_A_ILLEGAL_CONTACT_STEPS,
        },
        "note_initial_state_randomization": (
            f"全セルで初期状態分布(master_plan.md §1.6)を適用: 関節角 ±{RobotConfig.INIT_JOINT_POS_NOISE} rad, "
            f"関節角速度 ±{RobotConfig.INIT_JOINT_VEL_NOISE} rad/s。'fixed_dr'/'randomized_dr' は"
            "質量/摩擦/重心オフセット/サーボ温度/電圧のdomain randomizationのon/off。"
        ),
        "note_termination_reasons": (
            "現行実装(envs/mjx_rewards.py)はfallen_roll/fallen_pitch/fallen_height/"
            "time_limitのみをterminationとして判定する。no_illegal_contact/slip_ok/"
            "torque_ok/height_ok/uprightはterminationではなく、本スクリプトのsuccess判定"
            "(master_plan.md §0.3の論理積、criteria_pass_rate)で評価する。"
        ),
        "conditions": {},
        "disturbance_model": {
            "force_levels_N": force_levels,
            # [2026-09-26修正] 従来はRobotConfig.PUSH_DIRECTIONS /
            # RobotConfig.PUSH_DURATION_STEPSという存在しない属性を参照しており、
            # --force-levels未指定時にAttributeErrorで即クラッシュしていた
            # (force_levels=list(RobotConfig.PUSH_FORCE_LEVELS)がmain()冒頭で
            # 無条件に評価されるため)。実際の外乱印加(envs/mjx_env.py)は
            # 「各制御stepごとに3%の確率でtrigger、方向は3軸連続一様分布から
            # 正規化、1 control step分のみ印加」という確率的な単step impulseで
            # あり、固定のduration_steps・離散方向数という概念は実装に存在しない。
            # ここではその実際の挙動をそのまま記録する。
            "mechanism": (
                "各control stepで独立に3%の確率でpush発火。方向はx:[-0.5,0.5], "
                "y:[-1.0,1.0], z:[-0.2,0.2]から一様サンプルして正規化した3軸連続方向"
                "(離散方位ではない)。1回のpushはCONTROL_DT(1 control step)のみ"
                "qfrc_appliedへ印加される(envs/mjx_env.py 参照)。"
            ),
            "push_trigger_probability_per_step": 0.03,
            "control_dt_s": float(RobotConfig.CONTROL_DT),
            "impulse_per_trigger_Ns": [
                float(force * RobotConfig.CONTROL_DT) for force in force_levels
            ],
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
            # [2026-09-29修正] 従来は actuator_ctrlrange[:, 1] (目標関節角の上限[rad])を
            # トルク上限[N.m]として使っており、上限0.0の関節(hip_yaw/shoulder_pitch/elbow)で
            # |τ|>=0 が常に真となり torque_saturation_rate が必ず1.0になっていた。
            ctx["torque_limit"] = np.full(env._mjx_model.nu, RobotConfig.MOTOR_MAX_TORQUE)
            ctx["joint_qvel_idx"] = np.asarray(env._actuator_to_qvel_idx)
            ctx["geom_bodyid"] = np.asarray(env._mj_model.geom_bodyid)
            ctx["sole_bodies"] = {int(b) for b in np.asarray(env._foot_ids)}
            reset_fn = ctx["jax"].jit(env.reset)
            step_fn = ctx["jax"].jit(env.step)
            result = run_condition(
                ctx, reset_fn, step_fn, policy_fn,
                n_episodes=n_episodes,
                base_seed=args.seed,
                max_steps=max_steps,
                collapse_window=args.collapse_window,
                settling_steps=args.settling_steps,
            )
        key = f"{label}__{dr_label}"
        report["conditions"][key] = result
        print(f"[{key}] success={result['n_successes']}/{n_episodes} "
              f"criteria={ {k: round(v, 3) for k, v in result['criteria_pass_rate'].items()} } "
              f"episode_alive mean={result['episode_alive']['mean']:.1f} "
              f"reasons={result['termination_reason_counts']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Phase0 Eval] detailed report: {out_path}")

    _write_diagnosis_draft(diagnosis_md, report)
    print(f"[Phase0 Eval] diagnosis draft: {diagnosis_md}")


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
        lines.append(f"- success: {result['n_successes']}/{result['n_episodes']}")
        lines.append(f"- criteria pass rate: {result['criteria_pass_rate']}")
        fd = result["foot_displacement_m"]
        lines.append(
            f"- foot displacement [mm]: p50={fd['p50'] * 1000:.1f}, p95={fd['p95'] * 1000:.1f}, "
            f"max={fd['max'] * 1000:.1f}"
        )
        lines.append(
            f"- jitter (判定外): action_change_rms p50={result['action_change_rms']['p50']:.4f}, "
            f"joint_vel_rms p50={result['joint_vel_rms_rad_s']['p50']:.3f} rad/s"
        )
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
