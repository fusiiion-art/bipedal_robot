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
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper

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
    assert state.obs["state"].shape[0] == 4


def test_domain_randomization_differs_per_env_under_vmap():
    """項目2の回帰防止: vmap下で各並列環境が異なるDRサンプルを持つことを確認する。"""
    env = SenpuuMaruMJXEnv()
    reset_fn = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    state = reset_fn(keys)
    mass_scales = state.info["mass_scale"]
    unique_scales = set(np.asarray(mass_scales).tolist())
    assert len(unique_scales) > 1, f"Expected varied mass scales across envs, got {unique_scales}"


def _assert_episode_info_reset_on_done_step(state, env, reset_slot, other_slots):
    steps = np.asarray(state.info["step"])
    done = np.asarray(state.done)
    assert done[reset_slot] == 1.0 and np.all(done[other_slots] == 0.0), done
    # doneになったstepの戻り値の時点で、pipeline_state(AutoResetWrapperが差し替え済み)と
    # 同時にinfoも初期化されていること(1step遅れると次episodeの初回stepが前episodeの
    # 指令履歴のまま実行される)。
    assert steps[reset_slot] == 0, f"Expected slot {reset_slot} to reset to 0, got {steps[reset_slot]}"
    assert np.all(steps[other_slots] == 2), f"Expected other slots to advance to 2, got {steps[other_slots]}"
    # 新episodeの指令系は、そのepisodeのspawn関節角(初期状態分布からの新しいサンプル)で初期化される
    filtered = np.asarray(state.info["filtered_action"])
    spawn_joints = np.asarray(state.pipeline_state.qpos)[:, np.asarray(env._actuator_to_qpos_idx)]
    assert np.allclose(filtered[reset_slot], spawn_joints[reset_slot], atol=1e-6)


def test_auto_reset_resets_episode_scoped_info():
    """項目3の回帰防止: AutoResetWrapper配下でepisodeが終わったスロットの
    info['step']等が、doneになったstepでリセットされることを確認する。"""
    env = SenpuuMaruMJXEnv()
    wrapped = EpisodeWrapper(env, episode_length=2, action_repeat=1)
    wrapped = AutoResetWrapper(wrapped)
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

    # スロット1のBraxエピソードカウンタだけ巻き戻し、次stepでスロット0のみ
    # time limit(episode_length=2)によるdoneを発生させる
    state.info["steps"] = jp.array([1.0, 0.0])
    state = step_fn(state, action)
    _assert_episode_info_reset_on_done_step(state, env, reset_slot=0, other_slots=[1])


def test_auto_reset_resets_episode_scoped_info_batched():
    """train_mjx.py(brax.envs.training.wrap)と同じ Vmap→Episode→AutoReset の順で検証する。"""
    env = SenpuuMaruMJXEnv()
    num_envs = 4
    wrapped = VmapWrapper(env)
    wrapped = EpisodeWrapper(wrapped, episode_length=2, action_repeat=1)
    wrapped = AutoResetWrapper(wrapped)
    wrapped = EpisodeInfoResetWrapper(wrapped)

    step_fn = jax.jit(wrapped.step)
    keys = jax.random.split(jax.random.PRNGKey(42), num_envs)
    state = jax.jit(wrapped.reset)(keys)

    action = jp.zeros((num_envs, env.action_size))
    state = step_fn(state, action)
    assert np.all(np.asarray(state.info["step"]) == 1)

    # スロット0のみtime limitでdoneにする
    state.info["steps"] = jp.array([1.0, 0.0, 0.0, 0.0])
    state = step_fn(state, action)
    _assert_episode_info_reset_on_done_step(state, env, reset_slot=0, other_slots=[1, 2, 3])

