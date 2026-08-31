#!/usr/bin/env python3
"""Small non-MJX check for monotonic TrainingProgressWrapper counters."""

import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.envs import Wrapper
from brax.envs.base import State

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.training_wrapper import TrainingProgressWrapper


class DummyEnv:
    observation_size = 1
    action_size = 1
    backend = "generalized"

    def reset(self, rng):
        return State(jp.array(0.0), jp.array([0.0]), jp.array(0.0), jp.array(0.0), {}, {})

    def step(self, state, action):
        info = dict(state.info)
        info["inner_step"] = info.get("inner_step", jp.array(0)) + 1
        return state.replace(info=info)

    @property
    def unwrapped(self):
        return self


env = TrainingProgressWrapper(Wrapper(DummyEnv()), total_steps_per_env=10)
state = env.reset(jax.random.PRNGKey(0))
assert float(state.info["training_progress"]) == 0.0
for expected in range(1, 4):
    state = env.step(state, jp.array([0.0]))
    assert int(state.info["_env_steps"]) == expected
    assert int(state.info["global_step"]) == expected
    assert np.isclose(float(state.info["training_progress"]), expected / 10.0)
print("TrainingProgressWrapper PASS: counters are monotonic and progress uses total steps.")
