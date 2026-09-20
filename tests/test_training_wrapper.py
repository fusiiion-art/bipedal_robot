"""TrainingProgressWrapper の単体テスト。

[項目14] scratch/validate_progress_wrapper.py の内容を pytest テストとして移植。
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.envs import Wrapper
from brax.envs.base import State

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.training_wrapper import TrainingProgressWrapper


class _DummyEnv:
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


def test_training_progress_wrapper_counters():
    """カウンタが単調増加し、progressがtotal stepsに基づくことを確認。"""
    env = TrainingProgressWrapper(Wrapper(_DummyEnv()), total_steps_per_env=10)
    state = env.reset(jax.random.PRNGKey(0))
    assert float(state.info["training_progress"]) == 0.0

    for expected in range(1, 4):
        state = env.step(state, jp.array([0.0]))
        assert int(state.info["_env_steps"]) == expected
        assert int(state.info["global_step"]) == expected
        assert np.isclose(float(state.info["training_progress"]), expected / 10.0)


def test_training_progress_wrapper_progress_saturates():
    """total_steps_per_env を超えた場合に progress が 1.0 にクランプされることを確認。"""
    env = TrainingProgressWrapper(Wrapper(_DummyEnv()), total_steps_per_env=2)
    state = env.reset(jax.random.PRNGKey(0))
    for _ in range(5):
        state = env.step(state, jp.array([0.0]))
    assert float(state.info["training_progress"]) >= 1.0
