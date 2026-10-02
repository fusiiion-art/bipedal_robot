"""[T3] scratch/gate_a_qualification.py の単体テスト。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.gate_a_qualification import required_successes, wilson_lower_bound


def test_required_successes_matches_wilson_bound():
    assert required_successes(200, 0.95) == 197
    assert required_successes(200, 0.90) == 189
    # n=20 では全成功でも Wilson 下限 0.839 で 0.95 に届かない
    assert round(wilson_lower_bound(20, 20), 3) == 0.839
    assert required_successes(20, 0.95) is None
    k = required_successes(200, 0.95)
    assert wilson_lower_bound(k, 200) >= 0.95 > wilson_lower_bound(k - 1, 200)


def _report(path: Path, successes: int, n: int = 200) -> Path:
    cond = {
        "n_episodes": n,
        "n_successes": successes,
        "success_rate": successes / n,
        "n_unique_final_states": n,
        "criteria_pass_rate": {"alive": 1.0, "slip_ok": successes / n, "success": successes / n},
        "foot_displacement_m": {"p50": 0.004, "p95": 0.009, "max": 0.012},
    }
    path.write_text(json.dumps({"checkpoint": path.stem, "conditions": {"deterministic__randomized_dr": cond}}))
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "gate_a_qualification.py"), *map(str, args)],
        capture_output=True, text=True, check=True,
    ).stdout


def test_compare_only_without_threshold_lists_baseline(tmp_path):
    out = _run("--reports", _report(tmp_path / "s0.json", 190),
               "--baseline", _report(tmp_path / "zero.json", 200))
    assert "baseline(zero)" in out
    assert "ゼロ行動ベースラインより成功率が低い" in out
    assert "判定:" not in out


def test_compare_seeds_accepts_any_number_of_reports(tmp_path):
    reports = [_report(tmp_path / f"s{i}.json", k) for i, k in enumerate((200, 150, 100))]
    out = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "compare_seeds.py"), "--reports", *map(str, reports),
         "--baseline", str(_report(tmp_path / "zero.json", 200))],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "150/200" in out and "100/200" in out
    assert "pass:slip_ok" in out


def test_threshold_verdict_uses_min_seed(tmp_path):
    reports = [_report(tmp_path / f"s{i}.json", k) for i, k in enumerate((200, 198, 196))]
    out = _run("--reports", *reports, "--threshold", 0.95)
    assert "197/200 成功以上" in out
    assert "判定: FAIL" in out
