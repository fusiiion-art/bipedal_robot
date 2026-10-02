#!/usr/bin/env python3
"""phase0_eval_diagnostics.py のレポートを並べて比較する (判定はしない)。

[T3 2026-10-02] 旧版は2seed分のパスが固定されていた。任意個のレポートと、
ゼロ行動方策のベースライン (phase0_eval_diagnostics.py --zero-policy) を受け取る。
評価条件ごとに、成功数・成功条件ごとの通過率・足ずれ・報酬成分の平均を並べ、
先頭レポートとの差が大きい報酬成分に *** を付ける。合否判定は scratch/gate_a_qualification.py。

使い方:
  python scratch/compare_seeds.py --reports log/gateA_dbg_<commit>_s0/version_0/gate_a.json \\
      --baseline log/baseline_<commit>/gate_a_zero.json
"""

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, width=12, digits=4):
    return f"{v:>{width}.{digits}f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--conditions", nargs="*", default=None, help="比較する条件 (既定: 全条件)")
    args = parser.parse_args()

    columns = ([("baseline", _load(args.baseline))] if args.baseline else []) + [
        (path.parent.parent.name if path.parent.name.startswith("version_") else path.stem, _load(path))
        for path in args.reports
    ]
    conditions = args.conditions or list(columns[-1][1]["conditions"])
    header = f"{'':<32}" + "".join(f"{label[:12]:>13}" for label, _ in columns)

    for cond in conditions:
        cells = [rep["conditions"].get(cond, {}) for _, rep in columns]
        print(f"=== {cond} ===")
        print(header)
        print("-" * len(header))
        print(f"{'success':<32}" + "".join(
            f"{str(c.get('n_successes', '-')) + '/' + str(c.get('n_episodes', '-')):>13}" for c in cells))
        names = sorted({k for c in cells for k in c.get("criteria_pass_rate", {})})
        for name in names:
            print(f"{'pass:' + name:<32}" + "".join(
                " " + _fmt(c.get("criteria_pass_rate", {}).get(name), 12, 3) for c in cells))
        for stat in ("p50", "p95", "max"):
            print(f"{'foot_disp_' + stat + ' [mm]':<32}" + "".join(
                " " + _fmt((c.get("foot_displacement_m") or {}).get(stat, float("nan")) * 1000, 12, 2)
                for c in cells))
        ref_index = 1 if args.baseline else 0
        ref = cells[ref_index].get("reward_component_means", {})
        keys = sorted({k for c in cells for k in c.get("reward_component_means", {})})
        for k in keys:
            values = [c.get("reward_component_means", {}).get(k) for c in cells]
            r0 = ref.get(k)
            mark = ""
            if isinstance(r0, (int, float)):
                for v in values[ref_index + 1:]:
                    if isinstance(v, (int, float)):
                        diff = v - r0
                        ratio = v / r0 if abs(r0) > 1e-9 else float("nan")
                        if abs(diff) > 0.05 or ratio < 0.8 or ratio > 1.25:
                            mark = " ***"
            print(f"{k:<32}" + "".join(" " + _fmt(v, 12, 4) for v in values) + mark)
        print()


if __name__ == "__main__":
    main()
