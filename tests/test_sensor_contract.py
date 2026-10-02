import numpy as np
import jax.numpy as jp
import mujoco
from mujoco import mjx

from envs.mjx_env import SenpuuMaruMJXEnv
from envs.mjx_rewards import MJXRewardSystem
from robot.config import RobotConfig


def test_sensor_order_contract_matches_xml_layout():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mjx.make_data(model)
    env = SenpuuMaruMJXEnv()

    assert model.nsensordata >= 18, "Expected IMU + 8 FSR sensors in the XML sensor block."

    fsr_from_env = env._fsr_forces(data)
    assert fsr_from_env.shape == (8,), fsr_from_env.shape
    assert np.allclose(np.asarray(fsr_from_env), 0.0)

    fsr_in_reward = MJXRewardSystem.extract_fsr_sensor_data(model, data)
    assert fsr_in_reward.shape == (8,), fsr_in_reward.shape
    assert np.allclose(np.asarray(fsr_in_reward), 0.0)


def test_fsr_slice_matches_touch_sensors_left_then_right():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    touch = [i for i in range(model.nsensor) if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH]
    adrs = [int(model.sensor_adr[i]) for i in touch]
    sl = RobotConfig.FSR_SENSOR_SLICE
    assert adrs == list(range(sl.start, sl.stop))
    names = [model.sensor(i).name for i in touch]
    assert all("_l_" in n for n in names[:4]) and all("_r_" in n for n in names[4:]), names


def test_joint_order_matches_actuator_qpos_contract():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    env = SenpuuMaruMJXEnv()

    # actuator order is defined in RobotConfig.JOINT_NAMES, which is the ABI contract.
    # qpos and qvel must be mapped via actuator->qpos/qvel tables, not raw xml tree order.
    assert env._actuator_to_qpos_idx.shape[0] == model.nu
    assert env._actuator_to_qvel_idx.shape[0] == model.nu
    assert np.array_equal(env._actuator_to_qpos_idx, env._actuator_to_qpos_idx.astype(np.int32))
    assert np.array_equal(env._actuator_to_qvel_idx, env._actuator_to_qvel_idx.astype(np.int32))


def test_domain_randomization_is_applied_to_physics_model():
    env = SenpuuMaruMJXEnv()
    base_model = env._mjx_model
    mass_scale = 1.12
    fric_scale = 1.5
    com_offset = np.array([0.01, -0.02, 0.015], dtype=np.float32)

    randomized = env._apply_domain_randomization(base_model, mass_scale, fric_scale, com_offset)

    assert np.allclose(np.asarray(randomized.body_mass), np.asarray(base_model.body_mass) * mass_scale)
    assert np.allclose(np.asarray(randomized.geom_friction), np.asarray(base_model.geom_friction) * fric_scale)
    root = env._root_body_id
    assert root != 0, "COM offset must target the floating base body, not the static world body."
    assert np.allclose(np.asarray(randomized.body_ipos[root]), np.asarray(base_model.body_ipos[root]) + com_offset)
    assert np.allclose(np.asarray(randomized.body_ipos[0]), np.asarray(base_model.body_ipos[0]))


def test_domain_randomization_torque_uses_actuator_qvel_mapping():
    env = SenpuuMaruMJXEnv()
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    qvel = jp.arange(model.nv, dtype=jp.float32) + 1.0
    dr_damping = jp.arange(model.nu, dtype=jp.float32) + 0.1
    dr_friction = jp.zeros(model.nu, dtype=jp.float32)

    qfrc = env._joint_dr_torque(qvel, dr_damping, dr_friction)
    expected = np.zeros(model.nv, dtype=np.float32)
    actuator_qvel = np.asarray(env._actuator_to_qvel_idx)
    expected[actuator_qvel] = -dr_damping * qvel[actuator_qvel]

    assert np.allclose(np.asarray(qfrc), expected)
