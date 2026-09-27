#!/usr/bin/env python3
"""Gate B 正式合否判定（master_plan.md §6.5）。

`scratch/gate_b_eval.py --out ...`を学習seedごとに実行して得たJSONレポートを
複数受け取り、§6.5の合格基準のうち**現時点で評価可能なもの**を判定する。

§6.5の4項目のうち、このスクリプトが扱うもの:
  1. 各外乱系統(push)単独、目標値までの範囲での成功率 >= 90%
     → 正確には「目標範囲内でランダム化した外力」での評価を要求しているが、
       gate_b_eval.pyは固定外力の掃引(グリッド)なので、ここでは
       --target-forceで指定した掃引点(通常は目標範囲の上限)における
       成功率を代理指標として使う(単調性を仮定すれば上限点の成功率は
       範囲内評価より厳しめの下限に近い近似になる)。本来のランダム化
       評価とは異なる近似であることに注意。
  3. Gate A条件(外乱なし)の性能を維持していること
     → gate_b_eval.pyのpush_0N条件と、Gate A本judgment(gate_a_qualification.py
       に渡したレポート)の該当seed分を比較する。
  4. J_50が§0.1の目標値を上回っていること(--target-j50を指定した場合のみ)

**扱わないもの(正直に明記):**
  2. 複合条件(push+傾斜同時)の成功率 >= 80% は、床傾斜(§6.2)が
     未実装のため評価不可。このスクリプトは判定に含めない。

master_plan.md §0.1(目標外乱スペック)は未記入のため、--target-force /
--regression-margin / --target-j50 に既定値を与えていない(根拠のない
数値を勝手に補わないため)。

使い方の例:
  python3 scratch/gate_b_qualification.py \\
      --reports log/version_0/gate_b_seed0.json log/version_1/gate_b_seed1.json \\
      --target-force 20.0 --threshold 0.90 \\
      --gate-a-reports log/version_0/gate_a_seed0.json log/version_1/gate_a_seed1.json \\
      --regression-margin 0.05
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "robot").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.gate_a_qualification import wilson_lower_bound


def _lookup_or_interpolate_success_rate(report: dict, target_force: float) -> float:
    """reportのforce_levels_Nに厳密一致があればその成功率を、無ければ
    隣接2点の線形補間で近似する(J_50計算と同じ考え方)。"""
    force_levels = report["force_levels_N"]
    rates = {f: report["conditions"][f"push_{f:g}N"]["success_rate"] for f in force_levels}
    if target_force in rates:
        return rates[target_force]
    pairs = sorted(rates.items())
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x0 <= target_force <= x1:
            if x1 == x0:
                return y0
            frac = (target_force - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    raise SystemExit(
        f"--target-force={target_force} はreportの掃引レンジ"
        f"({pairs[0][0]}〜{pairs[-1][0]}N)の外にあり、補間できない。"
        "--force-levelsのレンジを広げてgate_b_eval.pyを再実行すること。"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", nargs="+", required=True, type=Path, help="gate_b_eval.pyの出力JSON。学習seedごとに1ファイル。")
    parser.add_argument(
        "--target-force", type=float, required=True,
        help="§0.1の目標外乱レンジ上限に相当する外力[N]。この点の成功率(厳密一致が"
             "無ければ線形補間)をWilson CI判定に使う。",
    )
    parser.add_argument("--threshold", type=float, default=0.90, help="Wilson下限の合格閾値(master_plan.md §6.5既定: 0.90)")
    parser.add_argument(
        "--gate-a-reports", nargs="*", type=Path, default=None,
        help="対応するGate A評価レポート(phase0_eval_diagnostics.py --out の出力)。"
             "指定するとGate A回帰チェックを行う。--reportsと同じ順序・同じseedを渡すこと。",
    )
    parser.add_argument(
        "--gate-a-condition", default="deterministic__randomized_dr",
        help="Gate Aレポート側でGate A回帰チェックに使うcondition key",
    )
    parser.add_argument(
        "--regression-margin", type=float, default=None,
        help="Gate A回帰チェックの許容低下幅(例0.05=5ポイント)。--gate-a-reports指定時は必須。"
             "master_plan.mdはこの数値を確定していないため既定値を与えていない。",
    )
    parser.add_argument("--target-j50", type=float, default=None, help="§0.1のJ_50目標値[N]。指定時のみJ_50比較を行う。")
    parser.add_argument("--z", type=float, default=1.959963985)
    args = parser.parse_args()

    if args.gate_a_reports and args.regression_margin is None:
        raise SystemExit("--gate-a-reportsを指定する場合は--regression-marginも必須です。")
    if args.gate_a_reports and len(args.gate_a_reports) != len(args.reports):
        raise SystemExit("--gate-a-reportsと--reportsは同数(同じseed順)で指定してください。")

    per_seed = []
    for i, path in enumerate(args.reports):
        data = json.loads(path.read_text(encoding="utf-8"))
        rate = _lookup_or_interpolate_success_rate(data, args.target_force)
        # n_episodesはtarget_forceに最も近い掃引点のものを使う(補間点の厳密なnは無いため)。
        nearest_force = min(data["force_levels_N"], key=lambda f: abs(f - args.target_force))
        n = data["conditions"][f"push_{nearest_force:g}N"]["n_episodes"]
        successes = round(rate * n)
        lower = wilson_lower_bound(successes, n, z=args.z)
        row = {
            "report": str(path), "checkpoint": data.get("checkpoint"),
            "n_episodes": n, "success_rate_at_target": rate, "wilson_lower_95": lower,
            "n_sufficient": n >= 200, "j50_N": data.get("J_50_N"),
        }
        if args.gate_a_reports:
            ga_data = json.loads(args.gate_a_reports[i].read_text(encoding="utf-8"))
            ga_cond = ga_data["conditions"][args.gate_a_condition]
            row["gate_a_success_rate"] = ga_cond["success_rate"]
            row["gate_b_no_disturbance_success_rate"] = data.get("gate_a_baseline_success_rate")
            if row["gate_b_no_disturbance_success_rate"] is None:
                raise SystemExit(
                    f"{path}: 'gate_a_baseline_success_rate'が無い。"
                    "gate_b_eval.pyの--force-levelsに0を含めて再実行すること。"
                )
            row["regression"] = row["gate_a_success_rate"] - row["gate_b_no_disturbance_success_rate"]
        per_seed.append(row)

    print(f"target_force={args.target_force}N　閾値(Wilson下限)={args.threshold}\n")
    for row in per_seed:
        flag = "" if row["n_sufficient"] else "  [WARN] n<200"
        print(
            f"  {row['checkpoint']}: n={row['n_episodes']:>4d}  "
            f"success_rate={row['success_rate_at_target']:.4f}  "
            f"wilson_lower_95={row['wilson_lower_95']:.4f}{flag}"
        )
        if args.gate_a_reports:
            reg = row["regression"]
            reg_flag = "OK" if reg <= args.regression_margin else "REGRESSION"
            print(
                f"      Gate A回帰チェック: Gate A success_rate={row['gate_a_success_rate']:.4f} "
                f"→ Gate B push_0N success_rate={row['gate_b_no_disturbance_success_rate']:.4f} "
                f"(低下幅={reg:.4f}, 許容={args.regression_margin}) [{reg_flag}]"
            )
        if row["j50_N"] is not None:
            j50_flag = ""
            if args.target_j50 is not None:
                j50_flag = "OK" if row["j50_N"] >= args.target_j50 else "UNDER TARGET"
                print(f"      J_50={row['j50_N']:.2f}N (目標={args.target_j50}N) [{j50_flag}]")
            else:
                print(f"      J_50={row['j50_N']:.2f}N (--target-j50未指定のため目標比較は省略)")

    worst = min(per_seed, key=lambda r: r["wilson_lower_95"])
    success_pass = worst["wilson_lower_95"] >= args.threshold
    regression_pass = True
    if args.gate_a_reports:
        regression_pass = all(r["regression"] <= args.regression_margin for r in per_seed)
    j50_pass = True
    if args.target_j50 is not None:
        j50_pass = all(r["j50_N"] is not None and r["j50_N"] >= args.target_j50 for r in per_seed)

    overall = success_pass and regression_pass and j50_pass
    print(f"\nmin(Wilson下限) = {worst['wilson_lower_95']:.4f}  (最悪seed: {worst['checkpoint']})")
    print(f"項目1(単独外乱>=閾値): {'PASS' if success_pass else 'FAIL'}")
    if args.gate_a_reports:
        print(f"項目3(Gate A回帰なし): {'PASS' if regression_pass else 'FAIL'}")
    else:
        print("項目3(Gate A回帰なし): 未評価(--gate-a-reports未指定)")
    if args.target_j50 is not None:
        print(f"項目4(J_50>=目標): {'PASS' if j50_pass else 'FAIL'}")
    else:
        print("項目4(J_50>=目標): 未評価(--target-j50未指定、§0.1目標値が未記入のため)")
    print("項目2(複合条件>=80%): 未評価(床傾斜§6.2が未実装のため評価不可)")
    print(f"\n総合判定: {'PASS' if overall else 'FAIL'}（項目2を除く。項目2は別途床傾斜実装後に判定すること）")


if __name__ == "__main__":
    main()
