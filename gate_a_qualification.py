#!/usr/bin/env python3
"""Gate A 正式合否判定（master_plan.md §4.5 / §8）。

`scratch/phase0_eval_diagnostics.py --out log/.../seedN_report.json` を
学習seedごとに実行して得られたJSONレポートを複数受け取り、以下を計算する:

  - 各学習seedについて、指定した評価条件(既定: deterministic__randomized_dr,
    held-outのdomain randomization条件)の成功率のWilson score 95%信頼区間
    下限を計算する
  - 判定: 全学習seedのWilson下限が閾値以上であること(平均ではなくmin基準。
    1つの良いseedが悪いseedを隠すことを防ぐ)

master_plan.md はこの閾値の具体的な数値を確定していない(v3の「95%」は
サンプル設計が未定義として明示的に無効化されており、後継の具体的な数値は
記載されていない)。したがってこのスクリプトは閾値を--thresholdで
明示的に受け取る必須引数とし、既定値を持たない(根拠のない数値を
勝手に補わないため)。

使い方の例:
  python3 scratch/gate_a_qualification.py \\
      --reports log/version_0/gate_a_seed0.json log/version_1/gate_a_seed1.json log/version_2/gate_a_seed2.json \\
      --threshold 0.95

n_episodes < 200 の場合はmaster_plan.md §4.5/§8の要求(n>=200)を満たして
いない旨の警告を出す(判定は行うが、正式なGate Aクローズの根拠としては
不十分であることを明示する)。
"""

import argparse
import json
import math
from pathlib import Path


def wilson_lower_bound(successes: int, n: int, z: float = 1.959963985) -> float:
    """Wilson score区間の下限を返す(2値比率の信頼区間、正規近似より小標本で頑健)。"""
    if n <= 0:
        return 0.0
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = p_hat + z * z / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (center - margin) / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reports", nargs="+", required=True, type=Path,
        help="phase0_eval_diagnostics.py --out で書き出したJSONレポート。学習seedごとに1ファイル。",
    )
    parser.add_argument(
        "--condition", default="deterministic__randomized_dr",
        help="reportの'conditions'内でどのセルを主判定に使うか(既定: held-out DR下のdeterministic評価)。",
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="Wilson下限に対する合格閾値(例: 0.95)。master_plan.mdはこの数値を確定していないため必須引数とする。",
    )
    parser.add_argument("--z", type=float, default=1.959963985, help="信頼区間のz値(既定: 95%%両側)")
    args = parser.parse_args()

    per_seed = []
    for path in args.reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        cond = data.get("conditions", {}).get(args.condition)
        if cond is None:
            raise SystemExit(
                f"{path}: condition '{args.condition}' が見つからない。"
                f"利用可能なcondition: {list(data.get('conditions', {}).keys())}"
            )
        n = int(cond["n_episodes"])
        success_rate = float(cond["success_rate"])
        successes = round(success_rate * n)
        lower = wilson_lower_bound(successes, n, z=args.z)
        per_seed.append({
            "report": str(path),
            "checkpoint": data.get("checkpoint"),
            "n_episodes": n,
            "success_rate": success_rate,
            "wilson_lower_95": lower,
            "n_sufficient": n >= 200,
        })

    print(f"評価条件: {args.condition}　閾値(Wilson下限): {args.threshold}\n")
    for row in per_seed:
        flag = "" if row["n_sufficient"] else "  [WARN] n<200 (master_plan.md §4.5/§8 の要求未達)"
        print(
            f"  {row['checkpoint']}: n={row['n_episodes']:>4d}  "
            f"success_rate={row['success_rate']:.4f}  "
            f"wilson_lower_95={row['wilson_lower_95']:.4f}{flag}"
        )

    worst = min(per_seed, key=lambda r: r["wilson_lower_95"])
    passed = worst["wilson_lower_95"] >= args.threshold
    all_n_sufficient = all(r["n_sufficient"] for r in per_seed)

    print(f"\nmin(Wilson下限) = {worst['wilson_lower_95']:.4f}  (最悪seed: {worst['checkpoint']})")
    print(f"判定: {'PASS' if passed else 'FAIL'}" + ("" if all_n_sufficient else "　※n<200のseedを含むため参考判定にとどめること"))


if __name__ == "__main__":
    main()
