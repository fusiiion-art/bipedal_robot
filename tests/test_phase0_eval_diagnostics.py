"""Phase 0 診断ロジックの単体テスト。

[項目9] 実処理は scratch/phase0_eval_diagnostics.py に一本化。
このファイルは import + pytest テスト関数のみを含む薄いファイル。
"""
import sys
import os
from pathlib import Path

# scratch/ をimportパスに追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scratch.phase0_eval_diagnostics import (
    classify_termination_reason,
    kaplan_meier_survival,
    diagnose_failure_timing,
    summarize_episode_alive,
    track_sustained_contact_loss,
    interpolate_threshold_crossing,
    gate_a_criteria,
    illegal_floor_contact,
    tilt_from_quat,
    track_illegal_contact,
    track_contact_after_settle,
)
from robot.config import RobotConfig
import numpy as np


# ============================================================================
# [T2] Gate A 成功条件 (master_plan §0.3) の単体テスト
# ============================================================================

def _passing_episode(**overrides):
    kwargs = dict(
        terminated=False, truncated=True, both_feet_contact=True,
        max_tilt_deg=0.5, max_foot_displacement=0.004, torque_saturation_rate=0.0,
        rel_height_drop=0.001, illegal_contact_events=0, cfg=RobotConfig,
    )
    kwargs.update(overrides)
    return gate_a_criteria(**kwargs)


def test_gate_a_success_requires_every_criterion():
    assert _passing_episode()["success"]
    failing = {
        "alive": dict(terminated=True),
        "both_feet_contact": dict(both_feet_contact=False),
        "upright": dict(max_tilt_deg=RobotConfig.GATE_A_MAX_TILT_DEG + 0.1),
        "slip_ok": dict(max_foot_displacement=RobotConfig.MAX_FOOT_TRANSLATION + 1e-4),
        "torque_ok": dict(torque_saturation_rate=RobotConfig.GATE_A_MAX_TORQUE_SAT_RATE + 1e-3),
        "height_ok": dict(rel_height_drop=RobotConfig.GATE_A_MAX_REL_HEIGHT_DROP + 1e-3),
        "no_illegal_contact": dict(illegal_contact_events=1),
    }
    for name, override in failing.items():
        checks = _passing_episode(**override)
        assert not checks[name], name
        assert not checks["success"], name
        assert all(v for k, v in checks.items() if k not in (name, "success")), name


def test_gate_a_not_alive_without_time_limit():
    # env の time limit に届かず評価予算で打ち切られた episode は成功に数えない
    assert not _passing_episode(truncated=False)["success"]


def test_contact_loss_during_settle_window_is_ignored():
    consecutive, lost = 0, False
    # reset 直後 (step 1〜30) は何 step 接地が切れても失敗にしない
    for t in range(1, 31):
        consecutive, lost = track_contact_after_settle(t, 30, False, consecutive, lost, grace_steps=2)
    assert not lost and consecutive == 0
    # 31 step 目以降は従来どおり、2step 連続で失敗
    for t in (31, 32):
        consecutive, lost = track_contact_after_settle(t, 30, False, consecutive, lost, grace_steps=2)
    assert lost


def test_tilt_from_quat():
    assert tilt_from_quat([1.0, 0.0, 0.0, 0.0]) == 0.0
    half = np.deg2rad(10.0) / 2.0
    assert np.isclose(np.rad2deg(tilt_from_quat([np.cos(half), np.sin(half), 0.0, 0.0])), 10.0)
    # ヨーだけの回転は傾きに含めない
    assert np.isclose(tilt_from_quat([np.cos(0.5), 0.0, 0.0, np.sin(0.5)]), 0.0, atol=1e-9)


def test_illegal_contact_needs_consecutive_steps():
    consecutive, events = 0, 0
    # 1step だけの接触は違反にしない
    for now in (True, False, True, False):
        consecutive, events = track_illegal_contact(now, consecutive, events, required_steps=2)
    assert events == 0
    # 連続2step以上で1回。長く続いても1回、途切れて再発したら別の1回
    for now in (True, True, True, True, False, True, True):
        consecutive, events = track_illegal_contact(now, consecutive, events, required_steps=2)
    assert events == 2


def test_illegal_floor_contact_detects_only_active_non_sole_floor_pairs():
    geom_bodyid = np.array([0, 1, 5, 6, 7])   # geom0=床, geom2/3=足裏body 5/6, geom4=足首body 7
    soles = {5, 6}
    # 足裏と床の接触、足首と床の非接触候補(dist>0)、胴体同士の接触は違反ではない
    assert not illegal_floor_contact([0, 0, 1], [2, 4, 4], [-0.001, 0.003, -0.002], geom_bodyid, soles)
    # 足首と床が実際に接触していたら違反
    assert illegal_floor_contact([0, 4], [2, 0], [-0.001, -0.0005], geom_bodyid, soles)


# ============================================================================
# [項目10] pytest テスト
# ============================================================================

def test_classify_termination_reason_priority():
    """終了理由分類の優先度テスト。"""
    assert classify_termination_reason(True, False, False, truncated=False) == "fallen_roll"
    assert classify_termination_reason(False, False, False, truncated=True) == "time_limit"
    assert classify_termination_reason(
        False, False, False, truncated=False,
        physics_diverged=True
    ) == "physics_diverged"
    assert classify_termination_reason(
        False, False, False, truncated=False,
        reward_is_finite=False
    ) == "reward_nan"
    assert classify_termination_reason(False, False, False, truncated=False) == "unknown_terminated"


def test_classify_combined_reasons():
    """複合的な終了理由のテスト。"""
    result = classify_termination_reason(True, True, False, truncated=False)
    assert "fallen_roll" in result
    assert "fallen_pitch" in result


def test_kaplan_meier_survival_basic():
    """KM生存曲線の基本テスト。"""
    times, survival = kaplan_meier_survival([100, 200, 500], [True, True, False])
    assert survival[0] == 1.0
    assert survival[-1] < 1.0


def test_kaplan_meier_survival_empty():
    """空の入力に対するKM生存曲線。"""
    times, survival = kaplan_meier_survival([], [])
    assert len(times) == 1
    assert survival[0] == 1.0


def test_diagnose_failure_timing_early_concentration():
    """序盤集中パターンの検出テスト。"""
    result = diagnose_failure_timing([10, 15, 20], max_step=500)
    assert result["classification"] == "序盤集中"


def test_diagnose_failure_timing_empty():
    """空の入力に対する失敗タイミング診断。"""
    result = diagnose_failure_timing([], max_step=500)
    assert result["classification"] == "no_failures"


def test_summarize_episode_alive_empty():
    """空の入力に対するエピソード生存サマリー。"""
    result = summarize_episode_alive([])
    assert result["n"] == 0


def test_summarize_episode_alive_basic():
    """基本的なエピソード生存サマリー。"""
    result = summarize_episode_alive([100, 200, 300])
    assert result["n"] == 3
    assert result["mean"] == 200.0

def test_track_sustained_contact_loss_momentary_blip_not_a_failure():
    """grace_steps未満の瞬間的な非接触は本物の接地喪失として扱わない
    (2026-09-26修正: 全step ANDラッチだった旧バグの回帰テスト)。"""
    consecutive, real_loss = 0, False
    grace_steps = 2
    # 1step分だけ非接触 → まだreal_lossにはならない
    consecutive, real_loss = track_sustained_contact_loss(False, consecutive, real_loss, grace_steps)
    assert real_loss is False
    # 接触が戻ればconsecutiveは0にリセットされる
    consecutive, real_loss = track_sustained_contact_loss(True, consecutive, real_loss, grace_steps)
    assert consecutive == 0
    assert real_loss is False


def test_track_sustained_contact_loss_sustained_is_a_failure():
    """grace_steps以上連続した非接触は本物の接地喪失として確定する。"""
    consecutive, real_loss = 0, False
    grace_steps = 2
    for _ in range(grace_steps):
        consecutive, real_loss = track_sustained_contact_loss(False, consecutive, real_loss, grace_steps)
    assert real_loss is True
    # 一度real_loss=Trueになったら、その後接触が戻ってもTrueのまま
    consecutive, real_loss = track_sustained_contact_loss(True, consecutive, real_loss, grace_steps)
    assert real_loss is True


def test_track_sustained_contact_loss_grace_steps_one_matches_old_behavior():
    """grace_steps=1なら旧実装(全step AND)と同じ、1stepの非接触で即失敗になる。"""
    consecutive, real_loss = 0, False
    consecutive, real_loss = track_sustained_contact_loss(False, consecutive, real_loss, grace_steps=1)
    assert real_loss is True


def test_interpolate_threshold_crossing_basic():
    """成功率が単調に下がる典型例で、線形補間によるJ_50計算が妥当な範囲に入る。"""
    forces = [0.0, 10.0, 20.0, 30.0]
    success = [1.0, 0.9, 0.4, 0.1]
    j50 = interpolate_threshold_crossing(forces, success, target=0.5)
    assert j50 is not None
    # 0.9→0.4の区間(10〜20N)でtarget=0.5を跨ぐので、その範囲内であるべき
    assert 10.0 <= j50 <= 20.0


def test_interpolate_threshold_crossing_no_crossing_returns_none():
    """全区間でtargetを跨がない場合は外挿せずNoneを返す。"""
    forces = [0.0, 10.0, 20.0]
    success = [1.0, 0.95, 0.9]  # 常に0.5を上回る
    assert interpolate_threshold_crossing(forces, success, target=0.5) is None

