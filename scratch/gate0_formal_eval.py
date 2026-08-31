#!/usr/bin/env python3
"""Formal Gate 0 and initial foot-contact validation for the standing task."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this script usable on WSL and avoid inheriting a stale platform choice.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig
from robot.math_utils import quat_to_euler


def configure_deterministic_gate0():
    """Disable disturbances and reset randomization for the nominal baseline."""
    RobotConfig.DISTURBANCE_CURRICULUM = False
    RobotConfig.RANDOM_PUSH_MAX_FORCE = 0.0
    RobotConfig.RANDOM_MASS_SCALE = [1.0, 1.0]
    RobotConfig.RANDOM_FRICTION = [1.0, 1.0]
    RobotConfig.RANDOM_COM_OFFSET = [0.0, 0.0]
    RobotConfig.RANDOM_TEMP = [40.0, 40.0]
    RobotConfig.RANDOM_VOLT = [11.1, 11.1]


def as_float(value):
    return float(np.asarray(value))


def foot_geom_ids(env):
    model = env._mjx_model
    body_ids = [env._reward_system._left_foot_id, env._reward_system._right_foot_id]
    geom_bodyid = np.asarray(model.geom_bodyid)
    return [
        index for index, body_id in enumerate(geom_bodyid)
        if int(body_id) in body_ids
    ]


def measure_state(env, state, geom_ids, sensor_start, sensor_end):
    data = state.pipeline_state
    qpos = np.asarray(data.qpos)
    rpy = np.asarray(quat_to_euler(data.qpos[3:7]))
    geom_xpos = np.asarray(data.geom_xpos)
    geom_z = geom_xpos[geom_ids, 2] if geom_ids else np.array([np.nan])
    geom_size = np.asarray(env._mjx_model.geom_size)
    geom_type = np.asarray(env._mjx_model.geom_type)
    # The current XML foot collision geoms are boxes. For other geom types,
    # retain the center z and mark the result as an approximate lower bound.
    lower_z = []
    for geom_id in geom_ids:
        if int(geom_type[geom_id]) == 6:  # mjGEOM_BOX
            lower_z.append(geom_xpos[geom_id, 2] - geom_size[geom_id, 2])
        else:
            lower_z.append(geom_xpos[geom_id, 2])
    touch = np.asarray(data.sensordata)[sensor_start:sensor_end]
    return {
        "torso_z": as_float(qpos[2]),
        "roll_rad": as_float(rpy[0]),
        "pitch_rad": as_float(rpy[1]),
        "yaw_rad": as_float(rpy[2]),
        "xy_m": as_float(np.linalg.norm(qpos[:2])),
        "foot_body_z": [as_float(value) for value in np.asarray(data.xpos)[[
            env._reward_system._left_foot_id,
            env._reward_system._right_foot_id,
        ], 2]],
        "foot_geom_lower_z": [as_float(value) for value in lower_z],
        "touch_sum": as_float(np.sum(touch)),
        "touch_values": [as_float(value) for value in touch],
    }


def evaluate(seed, seconds, output_dir):
    configure_deterministic_gate0()
    env = SenpuuMaruMJXEnv()
    steps = int(round(seconds / RobotConfig.CONTROL_DT))
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    geom_ids = foot_geom_ids(env)
    sensor_count = int(env._mjx_model.nsensordata)
    if sensor_count < 8:
        raise RuntimeError(f"Expected 8 foot touch sensor values, found {sensor_count}")
    sensor_start = sensor_count - 8
    sensor_end = sensor_count

    initial = measure_state(env, state, geom_ids, sensor_start, sensor_end)
    records = []
    terminated_step = None
    for step_index in range(steps):
        action = jp.zeros(env.action_size, dtype=jp.float32)
        state = env.step(state, action)
        sample = measure_state(env, state, geom_ids, sensor_start, sensor_end)
        sample["step"] = step_index + 1
        records.append(sample)
        if bool(state.done):
            terminated_step = step_index + 1
            break

    if not records:
        raise RuntimeError("No simulation samples were collected")

    roll = np.array([item["roll_rad"] for item in records])
    pitch = np.array([item["pitch_rad"] for item in records])
    xy = np.array([item["xy_m"] for item in records])
    lower_z = np.array([item["foot_geom_lower_z"] for item in records])
    touch = np.array([item["touch_sum"] for item in records])
    time_s = np.arange(1, len(records) + 1) * RobotConfig.CONTROL_DT
    xy_slope = float(np.polyfit(time_s, xy, 1)[0]) if len(records) > 1 else 0.0
    settling_start = max(0, len(records) // 5)
    result = {
        "seed": seed,
        "requested_seconds": seconds,
        "simulated_steps": len(records),
        "simulated_seconds": len(records) * RobotConfig.CONTROL_DT,
        "terminated_step": terminated_step,
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
        "max_touch_signal": float(np.max(touch)),
        "records": records,
    }

    # Provisional development thresholds. Formal acceptance still requires the
    # v2 multi-seed and 10-30 second evaluation record.
    result["pass"] = bool(
        terminated_step is None
        and result["simulated_seconds"] >= seconds
        and result["max_abs_roll_deg"] < 10.0
        and result["max_abs_pitch_deg"] < 10.0
        and result["min_foot_geom_z_m"] >= -0.002
        and result["foot_touch_rate"] >= 0.99
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate0_seed_{seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    print(f"[Gate0] detailed log: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "log" / "gate0_formal")
    args = parser.parse_args()
    result = evaluate(args.seed, args.seconds, args.output_dir)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
