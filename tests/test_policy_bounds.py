import ast
from pathlib import Path

import jax
import jax.numpy as jnp

from robot.policy_network import (
    POLICY_MAX_STD,
    POLICY_MEAN_CLIP_SCALE,
    POLICY_MIN_STD,
    make_policy_network_factory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    PROJECT_ROOT / "train" / "train_mjx.py",
    PROJECT_ROOT / "train" / "visualize_rl.py",
    PROJECT_ROOT / "train" / "export_onnx.py",
    PROJECT_ROOT / "train" / "export_trajectory.py",
    PROJECT_ROOT / "scratch" / "phase0_eval_diagnostics.py",
    PROJECT_ROOT / "scratch" / "gate0_eval.py",
)


def test_policy_distribution_bounds():
    policy = make_policy_network_factory(observation_size=4, action_size=20)
    logits = jnp.concatenate(
        [
            jnp.full((2, 20), 100.0),
            jnp.array(
                [
                    jnp.full((20,), -100.0),
                    jnp.full((20,), 100.0),
                ]
            ),
        ],
        axis=-1,
    )
    distribution = policy.parametric_action_distribution.create_dist(logits)

    assert float(jnp.max(jnp.abs(distribution.loc))) <= POLICY_MEAN_CLIP_SCALE + 1e-6
    assert float(jnp.min(distribution.scale)) >= POLICY_MIN_STD - 1e-6
    assert float(jnp.max(distribution.scale)) <= POLICY_MAX_STD + 1e-6


def test_policy_network_forward_pass_bounds():
    """回帰テスト: ネットワークforward passを通した後のloc/stdが
    意図通りの範囲に収まることを確認する。
    二重クリップが発生していた場合、max(|loc|) ≈ 2.25 になってしまう。
    修正後は max(|loc|) ≈ 3.0 (= POLICY_MEAN_CLIP_SCALE) のはず。"""
    obs_size = 4
    act_size = 20
    network = make_policy_network_factory(obs_size, act_size)
    params = network.policy_network.init(jax.random.PRNGKey(0))
    # 極端な入力でネットワーク出力を飽和させる
    obs = jnp.ones((1, obs_size)) * 1000.0
    logits = network.policy_network.apply(None, params, obs)
    dist = network.parametric_action_distribution.create_dist(logits)
    max_abs_loc = float(jnp.max(jnp.abs(dist.loc)))
    # クリップ済みのlocは POLICY_MEAN_CLIP_SCALE 以下であるべき
    assert max_abs_loc <= POLICY_MEAN_CLIP_SCALE + 1e-3, (
        f"max(|loc|)={max_abs_loc:.4f} exceeds POLICY_MEAN_CLIP_SCALE={POLICY_MEAN_CLIP_SCALE}"
    )
    assert float(jnp.min(dist.scale)) >= POLICY_MIN_STD - 1e-6
    assert float(jnp.max(dist.scale)) <= POLICY_MAX_STD + 1e-6


SHARED_ENTRYPOINT_NAMES = {"make_policy_network_factory", "make_inference_fn_from_params"}


def test_entrypoints_import_shared_policy_factory():
    """学習・評価・出力のスクリプトはネットワーク構成を robot/policy_network.py から取得し、
    独自に make_ppo_networks を呼ばないこと(分布・観測キーの不一致を防ぐ)。"""
    for path in ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports_shared = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "robot.policy_network"
            and any(alias.name in SHARED_ENTRYPOINT_NAMES for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imports_shared, f"{path} does not use robot/policy_network.py"
        assert "make_ppo_networks(" not in path.read_text(encoding="utf-8")
