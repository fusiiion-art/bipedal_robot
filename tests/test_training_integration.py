"""訓練パイプラインの統合テスト（項目2, 3の回帰防止）。

Braxの訓練ループが通る経路（jax.vmap / jax.jit / AutoResetWrapper / EpisodeInfoResetWrapper）
を実際に通過させて動作を検証する。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# JAX CPU実行を設定
os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
mujoco = pytest.importorskip("mujoco")
brax = pytest.importorskip("brax")
from brax.envs.wrappers.training import AutoResetWrapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mjx_env import SenpuuMaruMJXEnv
from envs.training_wrapper import EpisodeInfoResetWrapper


def test_vmap_reset_then_step_no_crash():
    """項目2の回帰防止: reset()とstep()を別々にjit/vmapしても動くことを確認する。
    
    以前のDR実装では reset() で self._mjx_model にトレース値を代入していたため、
    後続の step() で UnexpectedTracerError が発生していた。
    """
    env = SenpuuMaruMJXEnv()
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    state = reset_fn(keys)
    action = jp.zeros((4, env.action_size))
    state = step_fn(state, action)  # UnexpectedTracerErrorが出ないこと
    assert state is not None
    assert state.obs.shape[0] == 4


def test_domain_randomization_differs_per_env_under_vmap():
    """項目2の回帰防止: vmap下で各並列環境が異なるDRサンプルを持つことを確認する。"""
    env = SenpuuMaruMJXEnv()
    reset_fn = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    state = reset_fn(keys)
    mass_scales = state.info["mass_scale"]
    unique_scales = set(np.asarray(mass_scales).tolist())
    assert len(unique_scales) > 1, f"Expected varied mass scales across envs, got {unique_scales}"


def test_auto_reset_resets_episode_scoped_info():
    """項目3の回帰防止: AutoResetWrapper配下でepisodeが終わったスロットの
    info['step']等がリセットされることを確認する。"""
    env = SenpuuMaruMJXEnv()
    wrapped = AutoResetWrapper(env)
    wrapped = EpisodeInfoResetWrapper(wrapped)

    reset_fn = jax.jit(jax.vmap(wrapped.reset))
    step_fn = jax.jit(jax.vmap(wrapped.step))

    num_envs = 2
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    state = reset_fn(keys)

    # 初期状態
    initial_steps = np.asarray(state.info["step"])
    assert np.all(initial_steps == 0), f"Expected 0 steps initially, got {initial_steps}"

    # 1ステップ実行
    action = jp.zeros((num_envs, env.action_size))
    state = step_fn(state, action)
    steps_after_step1 = np.asarray(state.info["step"])
    assert np.all(steps_after_step1 == 1), f"Expected 1 step after first transition, got {steps_after_step1}"

    # スロット0のみ強制的に done=1.0 にして次ステップを実行（AutoResetWrapper/EpisodeInfoResetWrapperを発動）
    state = state.replace(done=jp.array([1.0, 0.0]))
    state = step_fn(state, action)

    # スロット0はリセット後に1ステップ進んで step=1、スロット1はリセットされずに step=2 になるはず
    steps_after_reset = np.asarray(state.info["step"])
    assert steps_after_reset[0] == 1, f"Expected slot 0 to reset to 0 then advance to 1, got {steps_after_reset[0]}"
    assert steps_after_reset[1] == 2, f"Expected slot 1 to advance to 2, got {steps_after_reset[1]}"
