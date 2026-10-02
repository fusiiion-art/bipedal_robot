#!/usr/bin/env python3
"""Gate A 正式合否判定（master_plan.md §4.5 / §8）。

`scratch/phase0_eval_diagnostics.py --out log/.../seedN_report.json` を
学習seedごとに実行して得られたJSONレポートを複数受け取り、以下を計算する:

  - 各学習seedについて、指定した評価条件(既定: deterministic__randomized_dr。
    評価の乱数系列は学習と別だが、DRの分布は学習と同一)の成功率のWilson score
    95%信頼区間下限を計算する
  - 判定: 全学習seedのWilson下限が閾値以上であること(平均ではなくmin基準。
    1つの良いseedが悪いseedを隠すことを防ぐ)
  - [T3] --baseline にゼロ行動方策(phase0_eval_diagnostics.py --zero-policy)の
    レポートを渡すと、成功率と成功条件ごとの通過率を各seedと並べて表示する

閾値の数値は人間が決める(master_plan.md)。このスクリプトは既定値を持たない。
--threshold を省略すると判定は行わず、seed とベースラインの比較表だけを出す
(1seed の Debug 評価用。旧 scratch/compare_seeds.py の役割)。

使い方の例:
  python3 scratch/gate_a_qualification.py \\
      --reports log/gateA_q_<commit>_s{0,1,2}/version_0/gate_a.json \\
      --baseline log/baseline_<commit>/gate_a_zero.json --threshold 0.95

n_episodes < 200 の場合はmaster_plan.md §4.5/§8の要求(n>=200)を満たして
いない旨の警告を出す(判定は行うが、正式なGate Aクローズの根拠としては
不十分であることを明示する)。

[2026-10-02追加] Wilson区間は独立試行を前提とする。レポートの
n_unique_final_states(終端状態の異なるepisode数)が n_episodes より小さい場合、
同一軌道の繰り返し(擬似反復)が含まれており実効サンプル数が n より小さいため、
警告を出して参考判定扱いにする。
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


def required_successes(n: int, threshold: float, z: float = 1.959963985):
    """Wilson下限が threshold 以上になる最小の成功数 (n回中)。全成功でも届かなければ None。"""
    for k in range(n + 1):
        if wilson_lower_bound(k, n, z) >= threshold:
            return k
    return None


def _condition_row(path: Path, condition: str, z: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cond = data.get("conditions", {}).get(condition)
    if cond is None:
        raise SystemExit(
            f"{path}: condition '{condition}' が見つからない。"
            f"利用可能なcondition: {list(data.get('conditions', {}).keys())}"
        )
    n = int(cond["n_episodes"])
    success_rate = float(cond["success_rate"])
    successes = int(cond.get("n_successes", round(success_rate * n)))
    n_unique = cond.get("n_unique_final_states")
    return {
        "report": str(path),
        "checkpoint": data.get("checkpoint"),
        "n_episodes": n,
        "successes": successes,
        "success_rate": success_rate,
        "wilson_lower_95": wilson_lower_bound(successes, n, z=z),
        "n_sufficient": n >= 200,
        "n_unique_final_states": n_unique,
        "independent": n_unique is not None and int(n_unique) == n,
        "criteria_pass_rate": cond.get("criteria_pass_rate", {}),
        "foot_displacement_m": cond.get("foot_displacement_m"),
    }


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
        "--threshold", type=float, default=None,
        help="Wilson下限に対する合格閾値(例: 0.95)。省略時は判定せず比較表だけを出す。",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="ゼロ行動方策のレポート(phase0_eval_diagnostics.py --zero-policy)。各seedと並べて表示する。",
    )
    parser.add_argument("--z", type=float, default=1.959963985, help="信頼区間のz値(既定: 95%%両側)")
    args = parser.parse_args()

    per_seed = [_condition_row(path, args.condition, args.z) for path in args.reports]
    baseline = _condition_row(args.baseline, args.condition, args.z) if args.baseline else None

    print(f"評価条件: {args.condition}　閾値(Wilson下限): {args.threshold if args.threshold is not None else '未指定(比較のみ)'}")
    if args.threshold is not None:
        for n in sorted({r["n_episodes"] for r in per_seed}):
            k = required_successes(n, args.threshold, args.z)
            need = f"{k}/{n} 成功以上" if k is not None else f"n={n} では全成功でも届かない"
            print(f"  合格に必要な成功数 (n={n}): {need}")
    print()

    rows = ([("baseline(zero)", baseline)] if baseline else []) + [(f"seed{i}", r) for i, r in enumerate(per_seed)]
    criteria_names = sorted({name for _, r in rows for name in r["criteria_pass_rate"] if name != "success"})
    if criteria_names:
        print("成功条件ごとの通過率:")
        print("  " + f"{'':16s}" + "".join(f"{name:>20s}" for name in criteria_names) + f"{'足ずれp95[mm]':>16s}")
        for label, r in rows:
            fd = r["foot_displacement_m"] or {}
            print("  " + f"{label:16s}" + "".join(f"{r['criteria_pass_rate'].get(name, float('nan')):>20.3f}"
                                                  for name in criteria_names)
                  + f"{fd.get('p95', float('nan')) * 1000:>16.1f}")
        print()

    if baseline:
        print(f"  baseline(zero): {baseline['successes']}/{baseline['n_episodes']}  "
              f"success_rate={baseline['success_rate']:.4f}  wilson_lower_95={baseline['wilson_lower_95']:.4f}")
    for row in per_seed:
        flag = "" if row["n_sufficient"] else "  [WARN] n<200 (master_plan.md §4.5/§8 の要求未達)"
        if row["n_unique_final_states"] is None:
            flag += "  [WARN] n_unique_final_states未記録(旧レポート。擬似反復を検査できない)"
        elif not row["independent"]:
            flag += (f"  [WARN] 終端状態の重複あり(unique={row['n_unique_final_states']}/{row['n_episodes']})"
                     "。同一軌道の反復が含まれ実効nが小さい")
        print(
            f"  {row['checkpoint']}: {row['successes']}/{row['n_episodes']}  "
            f"success_rate={row['success_rate']:.4f}  "
            f"wilson_lower_95={row['wilson_lower_95']:.4f}{flag}"
        )
        if baseline and row["success_rate"] < baseline["success_rate"]:
            print("      [NOTE] ゼロ行動ベースラインより成功率が低い")

    if args.threshold is None:
        return

    worst = min(per_seed, key=lambda r: r["wilson_lower_95"])
    passed = worst["wilson_lower_95"] >= args.threshold
    all_n_sufficient = all(r["n_sufficient"] for r in per_seed)
    all_independent = all(r["independent"] for r in per_seed)

    caveats = []
    if not all_n_sufficient:
        caveats.append("n<200のseedを含む")
    if not all_independent:
        caveats.append("擬似反復(終端状態の重複)を含む、または未検査のseedがある")

    print(f"\nmin(Wilson下限) = {worst['wilson_lower_95']:.4f}  (最悪seed: {worst['checkpoint']})")
    print(f"判定: {'PASS' if passed else 'FAIL'}"
          + ("" if not caveats else f"　※{'、'.join(caveats)}ため参考判定にとどめること"))


if __name__ == "__main__":
    main()
