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