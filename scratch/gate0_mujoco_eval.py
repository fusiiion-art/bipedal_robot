#!/usr/bin/env python3
"""Pure MuJoCo Gate 0 evaluation (no MJX/JAX compile)."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler


def as_float(value):
    return float(np.asarray(value))


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


def build_initial_qpos(model, torso_z):
    qpos = np.zeros(model.nq, dtype=np.float64)

    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    for act_i in range(min(model.nu, len(default))):
        jnt_id = int(model.actuator_trnid[act_i, 0])
        qpos_idx = int(model.jnt_qposadr[jnt_id])
        qpos[qpos_idx] = default[act_i]

    if model.nq >= 7:
        qpos[0:3] = np.array([0.0, 0.0, torso_z], dtype=np.float64)
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return qpos


def foot_geom_ids(model, left_foot_id, right_foot_id):
    ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == left_foot_id or body_id == right_foot_id:
            ids.append(geom_id)
    return ids


def split_foot_geom_ids(model, left_foot_id, right_foot_id):
    left_ids = []
    right_ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == left_foot_id:
            left_ids.append(geom_id)
        elif body_id == right_foot_id:
            right_ids.append(geom_id)
    return left_ids, right_ids


def foot_lower_z(model, data, geom_ids):
    values = []
    for geom_id in geom_ids:
        geom_type = int(model.geom_type[geom_id])
        center_z = float(data.geom_xpos[geom_id, 2])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            values.append(center_z - float(model.geom_size[geom_id, 2]))
        else:
            values.append(center_z)
    return values


def touch_values(model, data):
    if model.nsensordata < 8:
        return np.array([], dtype=np.float64)
    return np.asarray(data.sensordata[-8:], dtype=np.float64)


def contact_flags(data, left_geom_ids, right_geom_ids):
    left_set = set(left_geom_ids)
    right_set = set(right_geom_ids)
    left_contact = False
    right_contact = False
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if g1 in left_set or g2 in left_set:
            left_contact = True
        if g1 in right_set or g2 in right_set:
            right_contact = True
        if left_contact and right_contact:
            break
    return left_contact, right_contact


def measure(model, data, left_foot_id, right_foot_id, geom_ids, left_geom_ids, right_geom_ids):
    qpos = np.asarray(data.qpos)
    rpy = np.asarray(quat_to_euler(qpos[3:7]))
    lower = foot_lower_z(model, data, geom_ids)
    touch = touch_values(model, data)
    left_contact, right_contact = contact_flags(data, left_geom_ids, right_geom_ids)

    return {
        "torso_z": as_float(qpos[2]),
        "roll_rad": as_float(rpy[0]),
        "pitch_rad": as_float(rpy[1]),
        "yaw_rad": as_float(rpy[2]),
        "xy_m": as_float(np.linalg.norm(qpos[:2])),
        "foot_body_z": [
            as_float(data.xpos[left_foot_id, 2]),
            as_float(data.xpos[right_foot_id, 2]),
        ],
        "foot_geom_lower_z": [as_float(v) for v in lower],
        "touch_sum": as_float(np.sum(touch)) if touch.size else 0.0,
        "touch_values": [as_float(v) for v in touch],
        "left_contact": bool(left_contact),
        "right_contact": bool(right_contact),
        "max_abs_actuator_force": as_float(np.max(np.abs(data.actuator_force))) if model.nu > 0 else 0.0,
    }


def evaluate(seed, seconds, torso_z, output_dir):
    _ = seed  # deterministic evaluation for now
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    model.opt.timestep = RobotConfig.SIM_DT
    data = mujoco.MjData(model)

    left_foot_id, right_foot_id = find_foot_body_ids(model)
    geom_ids = foot_geom_ids(model, left_foot_id, right_foot_id)
    left_geom_ids, right_geom_ids = split_foot_geom_ids(model, left_foot_id, right_foot_id)

    qpos0 = build_initial_qpos(model, torso_z)
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0

    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    if model.nu > 0:
        data.ctrl[:] = 0.0
        data.ctrl[:min(model.nu, len(default))] = default[:min(model.nu, len(default))]

    mujoco.mj_forward(model, data)

    ctrl_steps = int(round(seconds / RobotConfig.CONTROL_DT))
    sim_steps_per_ctrl = max(1, int(round(RobotConfig.CONTROL_DT / RobotConfig.SIM_DT)))

    initial = measure(model, data, left_foot_id, right_foot_id, geom_ids, left_geom_ids, right_geom_ids)
    records = []

    for i in range(ctrl_steps):
        if model.nu > 0:
            data.ctrl[:min(model.nu, len(default))] = default[:min(model.nu, len(default))]
        for _ in range(sim_steps_per_ctrl):
            mujoco.mj_step(model, data)
        sample = measure(model, data, left_foot_id, right_foot_id, geom_ids, left_geom_ids, right_geom_ids)
        sample["step"] = i + 1
        records.append(sample)

    if not records:
        raise RuntimeError("No records generated")

    roll = np.array([r["roll_rad"] for r in records])
    pitch = np.array([r["pitch_rad"] for r in records])
    xy = np.array([r["xy_m"] for r in records])
    lower_z = np.array([r["foot_geom_lower_z"] for r in records])
    touch = np.array([r["touch_sum"] for r in records])
    both_contact = np.array([r["left_contact"] and r["right_contact"] for r in records])
    max_force = np.array([r["max_abs_actuator_force"] for r in records])

    time_s = np.arange(1, len(records) + 1) * RobotConfig.CONTROL_DT
    xy_slope = float(np.polyfit(time_s, xy, 1)[0]) if len(records) > 1 else 0.0

    result = {
        "seed": seed,
        "requested_seconds": seconds,
        "simulated_steps": len(records),
        "simulated_seconds": len(records) * RobotConfig.CONTROL_DT,
        "initial": initial,
        "final": records[-1],
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(roll)))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(pitch)))),
        "rms_roll_deg": float(np.rad2deg(np.sqrt(np.mean(roll ** 2)))),
        "rms_pitch_deg": float(np.rad2deg(np.sqrt(np.mean(pitch ** 2)))),
        "final_xy_m": float(xy[-1]),
        "max_xy_m": float(np.max(xy)),
        "xy_drift_speed_mm_s": xy_slope * 1000.0,
        "min_foot_geom_z_m": float(np.min(lower_z)),
        "max_foot_geom_z_m": float(np.max(lower_z)),
        "foot_touch_rate": float(np.mean(touch > 1e-6)),
        "both_foot_contact_rate": float(np.mean(both_contact)),
        "max_touch_signal": float(np.max(touch)),
        "max_abs_actuator_force": float(np.max(max_force)),
        "torque_saturation_rate": float(np.mean(max_force >= 0.98 * RobotConfig.MOTOR_MAX_TORQUE)),
        "records": records,
    }

    result["pass"] = bool(
        result["simulated_seconds"] >= seconds
        and result["max_abs_roll_deg"] < 10.0
        and result["max_abs_pitch_deg"] < 10.0
        and result["min_foot_geom_z_m"] >= -0.002
        and result["both_foot_contact_rate"] >= 0.99
        and result["torque_saturation_rate"] <= 0.01
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate0_seed_{seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "records"}
    print(json.dumps(summary, indent=2))
    print(f"[Gate0-MuJoCo] detailed log: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torso-z", type=float, default=0.1773)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "log" / "gate0_formal")
    args = parser.parse_args()

    result = evaluate(args.seed, args.seconds, args.torso_z, args.output_dir)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
