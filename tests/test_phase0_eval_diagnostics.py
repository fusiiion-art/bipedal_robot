"""scratch/phase0_eval_diagnostics.py の純Python/NumPyロジックの単体テスト。

JAX/MJX/GPUに依存しないため、CPU環境でいつでも実行できる
(master_plan.md 運用ルール: 「Done条件を可能な限りpytest関数に落とす」)。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.phase0_eval_diagnostics import (  # noqa: E402
    classify_termination_reason,
    diagnose_failure_timing,
    kaplan_meier_survival,
    summarize_episode_alive,
)


# ---------------------------------------------------------------------------
# classify_termination_reason
# ---------------------------------------------------------------------------

def test_classify_time_limit_takes_priority():
    # truncatedがTrueなら、他のフラグが立っていてもtime_limit扱いにする
    # (info['truncated']はenvs/mjx_env.py側でterminated確定後は立たない設計だが、
    # 分類関数自体は防御的にtruncated優先とする)。
    reason = classify_termination_reason(
        is_fallen_roll=True, is_fallen_pitch=False, is_low=False, truncated=True,
    )
    assert reason == "time_limit"


def test_classify_single_reason():
    assert classify_termination_reason(True, False, False, False) == "fallen_roll"
    assert classify_termination_reason(False, True, False, False) == "fallen_pitch"
    assert classify_termination_reason(False, False, True, False) == "fallen_height"


def test_classify_combined_reasons_are_joined():
    reason = classify_termination_reason(True, True, False, False)
    assert reason == "fallen_roll+fallen_pitch"


def test_classify_unknown_when_no_flag_but_terminated():
    reason = classify_termination_reason(False, False, False, False)
    assert reason == "unknown_terminated"


# ---------------------------------------------------------------------------
# kaplan_meier_survival
# ---------------------------------------------------------------------------

def test_km_all_survive_to_time_limit_gives_flat_curve():
    # 全episodeがtime-limitまで生存(=すべて打ち切り、真のeventなし)
    lengths = [500, 500, 500, 500]
    events = [False, False, False, False]
    times, survival = kaplan_meier_survival(lengths, events)
    # イベントが一件も無いので生存曲線は1.0のまま(times=[0]のみ)
    assert times.tolist() == [0]
    assert survival.tolist() == [1.0]


def test_km_all_die_immediately():
    lengths = [1, 1, 1, 1]
    events = [True, True, True, True]
    times, survival = kaplan_meier_survival(lengths, events)
    assert times.tolist() == [0, 1]
    assert survival[-1] == pytest.approx(0.0)


def test_km_matches_hand_computation():
    # 手計算: n=4, t=10でd=1 (残りrisk set=4) -> S(10)=1*(1-1/4)=0.75
    #          t=20でd=1 (risk set=3、time10終了1名脱落) -> S(20)=0.75*(1-1/3)=0.5
    #          t=30は打ち切りのみ(event無し) -> Sは変化しない
    lengths = [10, 20, 30, 30]
    events = [True, True, False, False]
    times, survival = kaplan_meier_survival(lengths, events)
    assert times.tolist() == [0, 10, 20]
    assert survival[1] == pytest.approx(0.75)
    assert survival[2] == pytest.approx(0.5)


def test_km_empty_input():
    times, survival = kaplan_meier_survival([], [])
    assert times.tolist() == [0]
    assert survival.tolist() == [1.0]


def test_km_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        kaplan_meier_survival([1, 2], [True])


# ---------------------------------------------------------------------------
# diagnose_failure_timing
# ---------------------------------------------------------------------------

def test_diagnose_no_failures():
    result = diagnose_failure_timing([], max_step=500)
    assert result["classification"] == "no_failures"


def test_diagnose_early_concentration():
    # 500stepのepisodeで、失敗が序盤(<1/3 = step<167)に集中
    steps = [10, 20, 30, 40, 50, 60]
    result = diagnose_failure_timing(steps, max_step=500)
    assert result["classification"] == "序盤集中"


def test_diagnose_late_concentration():
    steps = [450, 460, 470, 480, 490, 495]
    result = diagnose_failure_timing(steps, max_step=500)
    assert result["classification"] == "後半集中"


def test_diagnose_random_distribution():
    steps = [50, 150, 250, 350, 450, 100, 480, 20]
    result = diagnose_failure_timing(steps, max_step=500)
    assert result["classification"] == "ランダム分布"


# ---------------------------------------------------------------------------
# summarize_episode_alive
# ---------------------------------------------------------------------------

def test_summarize_episode_alive_basic():
    summary = summarize_episode_alive([100, 200, 300])
    assert summary["mean"] == pytest.approx(200.0)
    assert summary["min"] == 100.0
    assert summary["max"] == 300.0
    assert summary["n"] == 3


def test_summarize_episode_alive_empty():
    summary = summarize_episode_alive([])
    assert summary["n"] == 0
    assert summary["mean"] == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
