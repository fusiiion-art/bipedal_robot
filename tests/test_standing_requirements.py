"""固定足立位ミッションの設定・評価契約を検証する。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig
from scratch.phase0_eval_diagnostics import summarize_episode_alive


def test_success_summary_is_not_episode_alive_only():
    summary = summarize_episode_alive([500, 500, 100])
    assert summary["mean"] < RobotConfig.MAX_EPISODE_STEPS
