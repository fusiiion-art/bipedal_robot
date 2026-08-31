#!/usr/bin/env python3
"""Extract physical-limit related constants from MuJoCo model at nominal standing pose."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig


def find_foot_body_ids(model):
    left_name = "doutai-v5_hidaridairou_hidarikokansetu_hidarimomo_hidarihizabu_hidariaikabu_hidariashiura_hidariashiura-1"
    right_name = "doutai-v5_migidaitou_migikokansetu_migimomo_migihizabu_migigaikabu_migiashiura_migiashiura-1"

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)

    if left_id == -1 or right_id == -1:
        left_id = -1
        right_id = -1
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if left_id == -1 and "hidariashiura" in name:
                left_id = body_id
            if right_id == -1 and "migiashiura" in name:
                right_id = body_id
        if left_id == -1 or right_id == -1:
            raise RuntimeError("Failed to locate foot body ids")

    return left_id, right_id


def main():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)

    qpos = np.zeros(model.nq, dtype=np.float64)
    if model.nq >= 7:
        qpos[0:3] = np.array([0.0, 0.0, 0.1773], dtype=np.float64)
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    for act_i in range(min(model.nu, len(default))):
        jnt_id = int(model.actuator_trnid[act_i, 0])
        qpos_idx = int(model.jnt_qposadr[jnt_id])
        qpos[qpos_idx] = default[act_i]

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_foot_id, right_foot_id = find_foot_body_ids(model)

    left_x = float(data.xpos[left_foot_id, 0])
    right_x = float(data.xpos[right_foot_id, 0])
    com = np.asarray(data.subtree_com[0], dtype=np.float64)

    # Estimate support half width from foot centers in x direction.
    support_half = abs(left_x - right_x) * 0.5

    result = {
        "total_mass_kg": float(np.sum(model.body_mass)),
        "gravity_m_s2": float(-model.opt.gravity[2]),
        "com_xyz_m": [float(com[0]), float(com[1]), float(com[2])],
        "left_foot_center_xyz_m": [float(v) for v in data.xpos[left_foot_id]],
        "right_foot_center_xyz_m": [float(v) for v in data.xpos[right_foot_id]],
        "support_half_width_x_m": float(support_half),
        "floor_friction": [float(v) for v in model.geom_friction[0]],
        "motor_max_torque_nm": float(RobotConfig.MOTOR_MAX_TORQUE),
        "control_dt_s": float(RobotConfig.CONTROL_DT),
        "sim_dt_s": float(RobotConfig.SIM_DT),
    }

    out_dir = ROOT / "log"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_physical_probe.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"[Phase-1] saved: {out_path}")


if __name__ == "__main__":
    main()
