import ast
from pathlib import Path

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
    PROJECT_ROOT / "deploy" / "export_onnx.py",
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


def test_entrypoints_import_shared_policy_factory():
    for path in ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports_shared_factory = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "robot.policy_network"
            and any(alias.name == "make_policy_network_factory" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imports_shared_factory, f"{path} does not import the shared factory"
        assert "make_ppo_networks(" not in path.read_text(encoding="utf-8")
