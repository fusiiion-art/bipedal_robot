"""固定足立位ミッションの設定・評価契約を検証する。"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig
from scratch.phase0_eval_diagnostics import summarize_episode_alive


def test_standing_mission_forbids_walking_and_stepping():
    assert RobotConfig.ALLOW_WALKING is False
    assert RobotConfig.ALLOW_STEPPING is False
    assert RobotConfig.TARGET_VEL_X == 0.0
    assert RobotConfig.TARGET_VEL_Y == 0.0
    assert RobotConfig.TARGET_YAW_RATE == 0.0
    assert RobotConfig.MAX_SINGLE_FOOT_LIFT == 0.0


def test_external_push_levels_have_explicit_impulses():
    duration_s = RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT
    impulses = np.asarray(RobotConfig.PUSH_FORCE_LEVELS) * duration_s
    assert np.all(impulses >= 0.0)
    assert len(RobotConfig.PUSH_FORCE_LEVELS) >= 2
    assert RobotConfig.PUSH_DIRECTIONS == 8


def test_success_summary_is_not_episode_alive_only():
    summary = summarize_episode_alive([500, 500, 100])
    assert summary["mean"] < RobotConfig.MAX_EPISODE_STEPS
