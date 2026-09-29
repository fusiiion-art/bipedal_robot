"""Gate A 診断(2026-09-29)で見つかった reset/CBF/トルク上限まわりの契約テスト。"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from envs.cbf import CBFSafetyFilter
from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig


@pytest.fixture(scope="module")
def env():
    return SenpuuMaruMJXEnv()


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))


def _limits(mj_model):
    return (
        jp.asarray(mj_model.actuator_ctrlrange[:, 0], dtype=jp.float32),
        jp.asarray(mj_model.actuator_ctrlrange[:, 1], dtype=jp.float32),
    )


def _default_pose(mj_model):
    return jp.asarray(RobotConfig.DEFAULT_JOINT_ANGLES[:mj_model.nu], dtype=jp.float32)


def test_reset_command_state_matches_spawn_joint_pose(env):
    """指令系(LPF状態・遅延バッファ・前回指令)がspawn時の関節角で初期化されること。

    ゼロ初期化だと中腰spawn直後に「膝0rad(伸展)」を指令し、reset毎に跳ね上がりと
    足裏の滑りが起きる(Gate Aのmax_foot_displacementを構造的に超過させていた)。
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    joint_pos = np.asarray(state.pipeline_state.qpos)[np.asarray(env._actuator_to_qpos_idx)]

    for key in ("filtered_action", "last_action", "double_last_action", "triple_last_action"):
        assert np.allclose(np.asarray(state.info[key]), joint_pos), key
    history = np.asarray(state.info["action_history"])
    assert np.allclose(history, np.broadcast_to(joint_pos, history.shape))


def test_spawn_position_is_not_shifted_by_com_offset(env):
    """重心オフセットDRはspawn位置に加算されない(床へのめり込み/落下を防ぐ)。"""
    reset = jax.jit(env.reset)
    for seed in range(3):
        state = reset(jax.random.PRNGKey(seed))
        assert np.any(np.abs(np.asarray(state.info["com_offset"])) > 0.0)
        assert np.allclose(
            np.asarray(state.pipeline_state.qpos[0:3]),
            [0.0, 0.0, RobotConfig.INITIAL_HEIGHT],
            atol=1e-6,
        )


def test_cbf_safe_set_contains_default_pose(mj_model):
    lower, upper = _limits(mj_model)
    default = _default_pose(mj_model)
    cbf = CBFSafetyFilter(nominal_pose=default)

    safe = cbf.filter_action(default, lower, upper)
    assert np.allclose(np.asarray(safe), np.asarray(default))
    # softplus(0)=ln2 の定数オフセットが残っていないこと
    assert float(cbf.compute_cbf_penalty(default, safe, lower, upper)) == pytest.approx(0.0, abs=1e-5)


def test_cbf_margin_is_unchanged_where_default_is_away_from_limits(mj_model):
    lower, upper = _limits(mj_model)
    default = _default_pose(mj_model)
    new_lower, new_upper = CBFSafetyFilter(nominal_pose=default).compute_safe_margins(lower, upper)
    old_lower, old_upper = CBFSafetyFilter().compute_safe_margins(lower, upper)

    margin = np.asarray(upper - lower) * 0.05
    far = (np.asarray(default - lower) >= margin) & (np.asarray(upper - default) >= margin)
    assert far.any() and not far.all()
    assert np.allclose(np.asarray(new_lower)[far], np.asarray(old_lower)[far])
    assert np.allclose(np.asarray(new_upper)[far], np.asarray(old_upper)[far])


def test_cbf_still_clamps_and_penalizes_limit_violation(mj_model):
    lower, upper = _limits(mj_model)
    default = _default_pose(mj_model)
    cbf = CBFSafetyFilter(nominal_pose=default)

    nominal = upper + 0.1
    safe = cbf.filter_action(nominal, lower, upper)
    assert np.all(np.asarray(safe) <= np.asarray(upper))
    assert float(cbf.compute_cbf_penalty(nominal, safe, lower, upper)) > 0.0


def test_actuator_forcerange_matches_motor_max_torque(env, mj_model):
    expected = np.tile([-RobotConfig.MOTOR_MAX_TORQUE, RobotConfig.MOTOR_MAX_TORQUE], (mj_model.nu, 1))
    assert np.all(mj_model.actuator_forcelimited == 1)
    assert np.allclose(mj_model.actuator_forcerange, expected)
    assert np.allclose(np.asarray(env._mjx_model.actuator_forcerange), expected)
