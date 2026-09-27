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
)


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

