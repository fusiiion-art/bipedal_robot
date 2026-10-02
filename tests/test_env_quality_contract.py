"""観測契約・初期状態分布・指令パイプラインの回帰テスト (2026-10-02 第2回監査)。

- actor観測 ('state') は実機 (real/real_env.py) が組み立てる観測と同じ内容になること
  (base_pos=0, 姿勢=重力射影, FSR=0/1フラグ, ZMP=0, 位相=[0,1], 参照角=0, 履歴は初回観測で充填)
- critic観測 ('privileged_state') は actor観測 + シミュレータ真値であること
- 初期状態分布: 関節角がnominal±ノイズ、低い方の足裏がnominalと同じ高さで接地すること
- CBFペナルティは指令の追従遅れではなく可動域違反だけを測ること
- 熱モデルの時定数がエピソード長より十分長いこと
- 実機側: IMUの取り付け補正・指令系の初期化
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("brax")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig
from robot.math_utils import projected_gravity_jax, projected_gravity_numpy, rotate_vector_by_quaternion

NU = RobotConfig.NUM_JOINTS
BASE = RobotConfig.BASE_OBS_DIM


@pytest.fixture(scope="module")
def env():
    return SenpuuMaruMJXEnv()


@pytest.fixture(scope="module")
def reset_state(env):
    return jax.jit(env.reset)(jax.random.PRNGKey(3))


def _base_obs_slices():
    i = 0
    out = {}
    for name, n in [("base_pos", 3), ("gravity", 3), ("lin_vel", 3), ("ang_vel", 3),
                    ("joint_pos", NU), ("joint_vel", NU), ("contact", 8), ("zmp", 2),
                    ("phase", 2), ("ref", NU)]:
        out[name] = slice(i, i + n)
        i += n
    assert i == BASE
    return out


def test_observation_dims(env, reset_state):
    assert reset_state.obs["state"].shape == (RobotConfig.OBS_DIM,)
    assert reset_state.obs["privileged_state"].shape == (RobotConfig.PRIVILEGED_OBS_DIM,)
    assert env.observation_size == {
        "state": (RobotConfig.OBS_DIM,), "privileged_state": (RobotConfig.PRIVILEGED_OBS_DIM,)}


def test_actor_observation_matches_real_robot_contract(env, reset_state):
    # spawn 直後は高い方の足が床から数mm浮いていることがあるため、接地が落ち着くまで進める
    step = jax.jit(env.step)
    state = reset_state
    for _ in range(30):
        state = step(state, jp.zeros(NU))
    base = np.asarray(state.obs["state"][:BASE])
    sl = _base_obs_slices()
    assert np.all(base[sl["base_pos"]] == 0.0)
    np.testing.assert_allclose(base[sl["gravity"]], [0.0, 0.0, -1.0], atol=0.1)
    contact = base[sl["contact"]]
    assert set(np.unique(contact)) <= {0.0, 1.0}
    assert contact[:4].sum() > 0 and contact[4:].sum() > 0, "両足が接地していること"
    assert np.all(base[sl["zmp"]] == 0.0)
    assert np.all(base[sl["phase"]] == [0.0, 1.0])
    assert np.all(base[sl["ref"]] == 0.0)


def test_observation_history_is_filled_with_first_observation(reset_state):
    obs = np.asarray(reset_state.obs["state"])
    base = obs[:BASE]
    history = obs[BASE:BASE + BASE * RobotConfig.HISTORY_LEN].reshape(RobotConfig.HISTORY_LEN, BASE)
    assert np.allclose(history, base[None, :])


def test_privileged_observation_contains_true_state(env, reset_state):
    priv = np.asarray(reset_state.obs["privileged_state"])
    actor = np.asarray(reset_state.obs["state"])
    assert np.allclose(priv[:RobotConfig.OBS_DIM], actor)
    extra = priv[RobotConfig.OBS_DIM:]
    assert extra.shape == (RobotConfig.PRIVILEGED_EXTRA_DIM,)
    np.testing.assert_allclose(extra[0:3], np.asarray(reset_state.pipeline_state.qpos[0:3]), atol=1e-6)
    np.testing.assert_allclose(
        extra[3:6], np.asarray(projected_gravity_jax(reset_state.pipeline_state.qpos[3:7])), atol=1e-6)


def test_projected_gravity_matches_mujoco_rotation():
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        expected = np.zeros(3)
        mujoco.mju_rotVecQuat(expected, np.array([0.0, 0.0, -1.0]), q * np.array([1, -1, -1, -1]))
        np.testing.assert_allclose(projected_gravity_numpy(q), expected, atol=1e-9)
        np.testing.assert_allclose(np.asarray(projected_gravity_jax(jp.asarray(q))), expected, atol=1e-5)


def test_initial_state_distribution(env):
    m = env._mj_model
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), 8))
    qpos = np.asarray(states.pipeline_state.qpos)
    joint_pos = qpos[:, np.asarray(env._actuator_to_qpos_idx)]
    deviation = joint_pos - RobotConfig.DEFAULT_JOINT_ANGLES
    assert np.all(np.abs(deviation) <= RobotConfig.INIT_JOINT_POS_NOISE + 1e-6)
    assert np.std(deviation) > RobotConfig.INIT_JOINT_POS_NOISE / 4, "関節角がランダム化されていること"
    lo, hi = m.jnt_range[m.actuator_trnid[:, 0]].T
    assert np.all(joint_pos >= lo - 1e-6) and np.all(joint_pos <= hi + 1e-6)

    # 低い方の足裏が nominal と同じ高さ (床面すれすれ) にあること
    data = mujoco.MjData(m)
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(m, data)
        z = float(env._lowest_sole_z(jp.asarray(data.geom_xpos), jp.asarray(data.geom_xmat)))
        assert abs(z - env._nominal_sole_z) < 1e-4
    assert 0.0 <= env._nominal_sole_z < 1e-3


def test_cbf_penalty_ignores_command_lag(env, reset_state):
    """可動域内で目標角を大きく動かしても(LPF/速度制限で指令は遅れる)、CBF診断は0のまま。"""
    action = jp.zeros(NU).at[3].set(0.5)  # 右膝を 15° 動かす (可動域内)
    state = jax.jit(env.step)(reset_state, action)
    assert float(state.metrics["cbf_correction_norm"]) == pytest.approx(0.0, abs=1e-6)
    assert float(state.metrics["action_saturation"]) == pytest.approx(0.0, abs=1e-6)

    # 可動域外を要求したときだけ反応する
    state = jax.jit(env.step)(reset_state, jp.ones(NU))
    assert float(state.metrics["cbf_correction_norm"]) > 0.0


def test_command_deadband_is_one_servo_count(env, reset_state):
    step = jax.jit(env.step)
    small = 2.0 * RobotConfig.SERVO_POSITION_RESOLUTION / RobotConfig.ACTION_SCALE
    cmd0 = np.asarray(reset_state.info["filtered_action"])
    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES)
    # 現在の指令から 2カウント離れた目標 → 動く / 0.5カウント → 動かない
    for counts, should_move in [(2.0, True), (0.5, False)]:
        target = cmd0[0] + counts * RobotConfig.SERVO_POSITION_RESOLUTION
        action = jp.zeros(NU).at[0].set((target - default[0]) / RobotConfig.ACTION_SCALE)
        state = step(reset_state, action)
        moved = abs(float(state.info["filtered_action"][0]) - cmd0[0]) > 1e-7
        assert moved == should_move, (counts, small)


def test_thermal_time_constant_exceeds_episode():
    from envs.actuator_model import ActuatorState, HX30HMModel
    tau = HX30HMModel.THERMAL_RESISTANCE * HX30HMModel.THERMAL_MASS
    episode_s = RobotConfig.MAX_EPISODE_STEPS * RobotConfig.CONTROL_DT
    assert tau >= 10 * episode_s
    state = ActuatorState(temperature=jp.full(NU, 80.0), supply_voltage=11.1)
    for _ in range(RobotConfig.MAX_EPISODE_STEPS):
        state = HX30HMModel.update_temperature(state, jp.zeros(NU), RobotConfig.CONTROL_DT)
    assert float(state.temperature[0]) > 70.0


def test_imu_mount_quat_matches_xml_site():
    m = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_bno055_site")
    site_quat = m.site_quat[sid]
    mount = np.asarray(RobotConfig.IMU_MOUNT_QUAT)
    assert np.allclose(site_quat, mount, atol=1e-6) or np.allclose(site_quat, -mount, atol=1e-6)


def _bare_real_env():
    from real.real_env import RealRobotEnv

    class _FakeSpine:
        servo_positions = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES) + 0.01

    env = object.__new__(RealRobotEnv)
    env.dt = RobotConfig.CONTROL_DT
    env.spine = _FakeSpine()
    env.last_action = np.zeros(NU)
    env.smoothed_action = np.zeros(NU)
    env._vel_estimate = np.zeros(3)
    env._prev_joint_pos = np.zeros(NU)
    env._episode_step = 0
    env._imu_mount_quat = np.asarray(RobotConfig.IMU_MOUNT_QUAT, dtype=np.float64)
    from collections import deque
    env.obs_history = deque(maxlen=RobotConfig.HISTORY_LEN)
    env.act_history = deque(maxlen=RobotConfig.HISTORY_LEN)
    env._history_needs_fill = True
    env.fsr_contacts = np.ones(8)
    env.servo_temps = np.full(NU, 30.0)
    env.servo_voltages = np.full(NU, 11.1)
    return env


def test_real_env_reset_and_observation_match_sim_contract():
    env = _bare_real_env()
    env.reset_episode()
    measured = env.spine.servo_positions
    # 指令系は現在の実関節角で初期化される (0rad からの急激な指令を出さない)
    assert np.allclose(env.smoothed_action, measured)
    assert np.allclose(env.last_action, measured)

    # 胴体が直立しているときの BNO055 の姿勢 = 取り付け姿勢そのもの
    gyro_torso = np.array([0.1, -0.2, 0.3])
    mount = np.asarray(RobotConfig.IMU_MOUNT_QUAT)
    mount_inv = mount * np.array([1, -1, -1, -1])
    env.imu_data = {
        "quat": mount,
        "gyro": rotate_vector_by_quaternion(gyro_torso, mount_inv),
        "lin_accel": np.zeros(3),
    }
    obs = env.build_observation()
    assert obs.shape == (RobotConfig.OBS_DIM,)
    sl = _base_obs_slices()
    base = obs[:BASE]
    np.testing.assert_allclose(base[sl["gravity"]], [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(base[sl["ang_vel"]], gyro_torso, atol=1e-9)
    assert np.all(base[sl["base_pos"]] == 0.0) and np.all(base[sl["zmp"]] == 0.0)
    assert np.all(base[sl["phase"]] == [0.0, 1.0]) and np.all(base[sl["ref"]] == 0.0)
    history = obs[BASE:BASE + BASE * RobotConfig.HISTORY_LEN].reshape(RobotConfig.HISTORY_LEN, BASE)
    assert np.allclose(history, base[None, :])
    act_hist = obs[BASE * (1 + RobotConfig.HISTORY_LEN):BASE * (1 + RobotConfig.HISTORY_LEN) + NU * RobotConfig.HISTORY_LEN]
    assert np.allclose(act_hist.reshape(RobotConfig.HISTORY_LEN, NU), measured[None, :])


def test_actor_needs_only_state_and_critic_reads_privileged_state():
    """ONNX出力・実機推論は 'state' だけで動き、critic は 'privileged_state' を読むこと。"""
    from brax.training.acme import running_statistics, specs
    from robot.policy_network import (
        default_observation_size, make_inference_fn_from_params, make_policy_network_factory)

    obs_size = default_observation_size()
    net = make_policy_network_factory(obs_size, NU, preprocess_observations_fn=running_statistics.normalize)
    normalizer = running_statistics.init_state(
        {k: specs.Array(v, jp.dtype('float32')) for k, v in obs_size.items()})
    policy = net.policy_network.init(jax.random.PRNGKey(0))
    value = net.value_network.init(jax.random.PRNGKey(1))

    infer = make_inference_fn_from_params((normalizer, policy, value), deterministic=True)
    action, _ = infer({"state": jp.zeros(RobotConfig.OBS_DIM)}, jax.random.PRNGKey(0))
    assert action.shape == (NU,)

    obs = {"state": jp.zeros(RobotConfig.OBS_DIM), "privileged_state": jp.zeros(RobotConfig.PRIVILEGED_OBS_DIM)}
    assert net.value_network.apply(normalizer, value, obs).shape == ()
    value_input_dims = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(value) if leaf.ndim == 2}
    policy_input_dims = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(policy) if leaf.ndim == 2}
    assert RobotConfig.PRIVILEGED_OBS_DIM in value_input_dims
    assert RobotConfig.OBS_DIM in policy_input_dims and RobotConfig.PRIVILEGED_OBS_DIM not in policy_input_dims
