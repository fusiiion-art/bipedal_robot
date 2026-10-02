"""[T7] 接触モデルの契約テスト (純MuJoCo + MJX)。

デフォルト姿勢を1秒PD保持した後:
  - 床と足裏以外のgeomの接触が無い (足首カプセルの接地で体重がFSRを迂回しない)
  - 自己接触が無い (股関節ヨー球と太ももカプセルの常時食い込みが無い)
  - FSR合計が総重量の ±2% (体重が全て足裏センサーに乗る)
  - 左足の荷重割合が 45〜55%
MJX 側は sensordata の FSR で同じ荷重条件を検査する (Gate 0.5 の一部)。
"""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

mujoco = pytest.importorskip("mujoco")

from robot.config import RobotConfig

_HOLD_STEPS = int(round(1.0 / RobotConfig.SIM_DT))


def _model():
    m = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    m.actuator_gainprm[:, 0] = RobotConfig.KP
    m.actuator_biasprm[:, 1] = -RobotConfig.KP
    m.actuator_biasprm[:, 2] = -RobotConfig.KD
    m.opt.timestep = RobotConfig.SIM_DT
    return m


def _default_qpos(m):
    qpos = np.zeros(m.nq)
    qpos[2] = RobotConfig.INITIAL_HEIGHT
    qpos[3] = 1.0
    qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]] = RobotConfig.DEFAULT_JOINT_ANGLES
    return qpos


def _sole_bodies(m):
    return {b for b in range(m.nbody) if "ashiura" in m.body(b).name}


def _total_weight(m):
    return float(m.body_mass.sum() * -m.opt.gravity[2])


def _assert_load(fsr, weight):
    total = float(np.sum(fsr))
    assert abs(total - weight) <= 0.02 * weight, f"FSR sum {total:.2f}N vs weight {weight:.2f}N"
    left_share = float(np.sum(fsr[:4])) / total
    assert 0.45 <= left_share <= 0.55, f"left load share {left_share:.3f}"


@pytest.fixture(scope="module")
def held_mujoco():
    m = _model()
    d = mujoco.MjData(m)
    d.qpos[:] = _default_qpos(m)
    d.ctrl[:] = RobotConfig.DEFAULT_JOINT_ANGLES
    for _ in range(_HOLD_STEPS):
        mujoco.mj_step(m, d)
    return m, d


def test_no_floor_contact_except_soles(held_mujoco):
    m, d = held_mujoco
    soles = _sole_bodies(m)
    illegal = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if 0 in (b1, b2):
            other = b2 if b1 == 0 else b1
            if other not in soles:
                illegal.append(m.body(other).name)
    assert not illegal, illegal


def test_no_self_contact(held_mujoco):
    m, d = held_mujoco
    pairs = [
        (m.body(m.geom_bodyid[d.contact[i].geom1]).name, m.body(m.geom_bodyid[d.contact[i].geom2]).name)
        for i in range(d.ncon)
        if 0 not in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])
    ]
    assert not pairs, pairs


def test_whole_weight_on_fsr_and_balanced(held_mujoco):
    m, d = held_mujoco
    _assert_load(d.sensordata[RobotConfig.FSR_SENSOR_SLICE], _total_weight(m))


def test_mjx_fsr_load_matches_weight():
    jax = pytest.importorskip("jax")
    jp = pytest.importorskip("jax.numpy")
    from mujoco import mjx

    m = _model()
    mx = mjx.put_model(m)
    d = mjx.make_data(mx).replace(
        qpos=jp.asarray(_default_qpos(m)),
        ctrl=jp.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=jp.float32),
    )

    @jax.jit
    def hold(d):
        return jax.lax.scan(lambda d, _: (mjx.step(mx, d), None), d, (), length=_HOLD_STEPS)[0]

    d = hold(d)
    _assert_load(np.asarray(d.sensordata[RobotConfig.FSR_SENSOR_SLICE]), _total_weight(m))
