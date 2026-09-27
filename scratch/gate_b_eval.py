#!/usr/bin/env python3
"""Gate B ロバストネス包絡線評価（master_plan.md §6.4）。

`scratch/phase0_eval_diagnostics.py`のrun_condition/run_episodeハーネスを
再利用し、突っつき外力(push force)の強度を掃引して各水準の成功率と
Wilson信頼区間を求め、成功率50%となる外力J_50(線形補間)を計算する。

**現時点でのスコープと未対応事項(正直に明記する):**
  - 印加方向・タイミングは envs/mjx_env.py の実装通り「各control stepで
    3%の確率でtrigger、方向は3軸連続一様分布」という確率的なものであり、
    master_plan.md §6.4が理想として書く「8方位×3タイミングの離散グリッド」
    を直接制御することはできない(この評価は力の大きさのみを掃引し、
    方向/タイミングは env の既存の乱数機構に任せるモンテカルロ近似)。
    8方位×3タイミングを本当に個別制御したい場合は envs/mjx_env.py に
    決定論的な単発push注入機構を追加する必要があり、JAXトレース対象の
    step関数を変更するその変更は、GPU/JAX実行環境で実機テストできる人が
    レビュー・検証してから取り込むこと(このサンドボックスにはJAX/MuJoCoが
    無く、この変更をここで検証できない)。
  - 床傾斜(§6.2)は物理的に未実装(envに傾斜床/gravity回転機構が無い)。
    このスクリプトはpush外乱のみを扱う。
  - 複合条件(push+傾斜同時、§6.5)は傾斜が未実装のため評価不可。

使い方の例:
  python3 scratch/gate_b_eval.py \\
      --exp_name <学習run名> --version 0 \\
      --force-levels 0,5,10,15,20,25,30 \\
      --episodes 200 \\
      --out log/version_0/gate_b_seed0.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "robot").is_dir() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.phase0_eval_diagnostics import (
    _lazy_imports,
    _DomainRandomizationScope,
    run_condition,
    interpolate_threshold_crossing,
)
from scratch.gate_a_qualification import wilson_lower_bound


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp_name", default="", help="log/<exp_name> 配下のcheckpointを使う")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--model", default="best_params.pkl")
    parser.add_argument(
        "--force-levels", required=True,
        help=(
            "掃引する外力[N]をカンマ区切りで指定(必須。master_plan.md §0.1の"
            "目標外乱スペックが未記入のため既定値を用意していない)。"
            "0を含めるとGate A(外乱なし)条件を同じrun内で回帰確認できる。"
        ),
    )
    parser.add_argument("--episodes", type=int, default=20, help="各外力水準あたりのheld-out評価episode数(正式判定にはn>=200推奨)")
    parser.add_argument("--max-steps", type=int, default=None, help="未指定ならRobotConfig.MAX_EPISODE_STEPS")
    parser.add_argument("--collapse-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "log" / "gate_b_eval.json")
    args = parser.parse_args()

    force_levels = sorted(float(v) for v in args.force_levels.split(","))
    if args.episodes < 200:
        print(
            f"[Gate B Eval][WARN] --episodes={args.episodes} < 200. "
            "master_plan.md §4.5/§8はn>=200を要求している。この結果は速報用にとどめ、"
            "正式なGate B判定には--episodes 200以上で再実行すること。"
        )

    ctx = _lazy_imports()
    RobotConfig = ctx["RobotConfig"]
    SenpuuMaruMJXEnv = ctx["SenpuuMaruMJXEnv"]
    ppo_networks = ctx["ppo_networks"]

    max_steps = args.max_steps or RobotConfig.MAX_EPISODE_STEPS

    model_path = ctx["get_model_path"](args.exp_name, args.version, args.model)
    if model_path is None:
        raise SystemExit(
            f"checkpoint not found for exp_name={args.exp_name!r}, version={args.version}, "
            f"model={args.model!r}. --exp_name / --version / --model を確認してください。"
        )
    params = ctx["load_checkpoint"](model_path)

    _probe_env = SenpuuMaruMJXEnv()
    network = ctx["make_policy_network_factory"](
        _probe_env.observation_size,
        _probe_env.action_size,
        preprocess_observations_fn=ctx["running_statistics"].normalize,
    )
    make_policy = ppo_networks.make_inference_fn(network)

    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf

    params_stripped = ctx["jax"].tree_util.tree_map(strip_leading_dim, params)
    # 主判定はdeterministic評価(master_plan.md §8)。
    policy_fn = ctx["jax"].jit(make_policy(params_stripped, deterministic=True))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(model_path),
        "max_steps": max_steps,
        "gate": "B",
        "scope_note": (
            "push外乱の強度掃引のみ。方向/タイミングはenv既定の確率的機構に一任した"
            "モンテカルロ近似(離散8方位x3タイミンググリッドではない)。床傾斜(§6.2)は"
            "未実装のため含まない。複合条件(§6.5)は評価不可。"
        ),
        "mechanism": (
            "各control stepで独立に3%の確率でpush発火。方向は3軸連続一様分布から"
            "正規化。1回のpushはCONTROL_DT(1 control step)のみqfrc_appliedへ印加。"
        ),
        "force_levels_N": force_levels,
        "conditions": {},
    }

    for force in force_levels:
        RobotConfig.RANDOM_PUSH_MAX_FORCE = force
        RobotConfig.DISTURBANCE_CURRICULUM = force > 0.0
        # held-out(randomized_dr)条件で評価する(§8: 評価seedは学習と別系列、
        # deterministic評価を主判定とする方針に合わせる)。
        with _DomainRandomizationScope(RobotConfig, fixed=False):
            # [注意] jax.jit(env.reset)のコンパイルキャッシュはbound methodの
            # 等価性で引かれるため、RobotConfig変更後も同じenvインスタンスを
            # 使い回すと古いコンパイル結果(=変更前のRANDOM_PUSH_MAX_FORCE)が
            # 再利用されてしまう(phase0_eval_diagnostics.pyのmain()内コメント参照)。
            # 外力水準ごとに新しいenvインスタンスを作ることでこれを回避する。
            env = SenpuuMaruMJXEnv()
            ctx["foot_ids"] = (env._reward_system._left_foot_id, env._reward_system._right_foot_id)
            ctx["torque_limit"] = ctx["jp"].asarray(env._mjx_model.actuator_ctrlrange[:, 1])
            reset_fn = ctx["jax"].jit(env.reset)
            step_fn = ctx["jax"].jit(env.step)
            result = run_condition(
                ctx, reset_fn, step_fn, policy_fn,
                n_episodes=args.episodes,
                base_seed=args.seed,
                max_steps=max_steps,
                collapse_window=args.collapse_window,
            )
        n = result["n_episodes"]
        successes = round(result["success_rate"] * n)
        wilson_lower = wilson_lower_bound(successes, n)
        result["wilson_lower_95"] = wilson_lower
        key = f"push_{force:g}N"
        report["conditions"][key] = result
        print(
            f"[{key}] success_rate={result['success_rate']:.4f} "
            f"wilson_lower_95={wilson_lower:.4f} "
            f"episode_alive mean={result['episode_alive']['mean']:.1f}"
        )

    success_rates = [report["conditions"][f"push_{f:g}N"]["success_rate"] for f in force_levels]
    j50 = interpolate_threshold_crossing(force_levels, success_rates, target=0.5)
    report["J_50_N"] = j50
    if j50 is None:
        print(
            "[Gate B Eval][WARN] J_50を求められなかった(掃引した外力レンジ内で"
            "成功率が50%を跨いでいない)。--force-levelsのレンジを広げて再実行すること。"
        )
    else:
        print(f"J_50 = {j50:.2f} N")

    if 0.0 in force_levels:
        gate_a_baseline = report["conditions"]["push_0N"]["success_rate"]
        report["gate_a_baseline_success_rate"] = gate_a_baseline
        print(f"Gate A baseline (push_0N) success_rate = {gate_a_baseline:.4f}")
    else:
        print(
            "[Gate B Eval][INFO] --force-levelsに0を含めなかったため、"
            "同一run内でのGate A回帰確認はできていない。"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Gate B Eval] report: {args.out}")


if __name__ == "__main__":
    main()
