#!/usr/bin/env python3
"""Validate PPO policy loc/std bounds without starting a training run."""

import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.train_mjx import POLICY_MAX_STD, POLICY_MEAN_CLIP_SCALE, POLICY_MIN_STD
from brax.training.agents.ppo import networks


policy = networks.make_ppo_networks(
    observation_size=4,
    action_size=20,
    mean_clip_scale=POLICY_MEAN_CLIP_SCALE,
)
distribution = policy.parametric_action_distribution
logits = jnp.concatenate([
    jnp.full((2, 20), 100.0),
    jnp.array([
        jnp.full((20,), -100.0),
        jnp.full((20,), 100.0),
    ]),
], axis=-1)
created = distribution.create_dist(logits)

assert float(jnp.max(jnp.abs(created.loc))) <= POLICY_MEAN_CLIP_SCALE + 1e-6
assert float(jnp.min(created.scale)) >= POLICY_MIN_STD - 1e-6
assert float(jnp.max(created.scale)) <= POLICY_MAX_STD + 1e-6
print("Policy bounds PASS:")
print(f"  max_abs_loc={float(jnp.max(jnp.abs(created.loc))):.6f}")
print(f"  min_std={float(jnp.min(created.scale)):.6f}")
print(f"  max_std={float(jnp.max(created.scale)):.6f}")
