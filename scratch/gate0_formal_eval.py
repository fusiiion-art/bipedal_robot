#!/usr/bin/env python3
"""Formal Gate 0 evaluation for learned RL policy (no-disturbance baseline).

学習済みRL方策を読み込んで、無外乱条件でGate 0判定を実施する。
物理・初期姿勢の確認用ゼロ行動評価ではなく、実checkpointの方策を使用する。

Usage:
  python scratch/gate0_formal_eval.py \
    --exp_name phase0_qual_seed0 \
    --version 0 \
    --model best_params.pkl \
    --seconds 30 \
    --seed 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

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


def find_checkpoint(
    exp_name: str,
    version: Optional[int] = None,
    model: str = "best_params.pkl",
) -> Optional[Path]:
    """学習済みcheckpointのパスを検索する。
    
    Args:
        exp_name: log/<exp_name> の形式でexperiment name
        version: version番号。Noneなら最新を選ぶ
        model: checkpoint filename (best_params.pkl, final_params.pkl等)
    
    Returns:
        Path to checkpoint, or None if not found.
    """
    if not exp_name:
        return None
    
    log_base = ROOT / "log" / exp_name
    if not log_base.exists():
        print(f"[Gate0] WARNING: {log_base} not found")
        return None
    
    version_dirs = sorted([d for d in log_base.iterdir() if d.is_dir() and d.name.startswith("version_")])
    if not version_dirs:
        print(f"[Gate0] WARNING: no version_* directories in {log_base}")
        return None
    
    if version is None:
        # 最新版を選ぶ
        version_dir = version_dirs[-1]
    else:
        version_dir = log_base / f"version_{version}"
    
    checkpoint_path = version_dir / model
    if not checkpoint_path.exists():
        print(f"[Gate0] WARNING: {checkpoint_path} not found")
        return None
    
    return checkpoint_path


def load_checkpoint_and_make_policy(checkpoint_path: Path):
    """Checkpointを読み込んで、policyを生成する。
    
    Returns:
        policy_fn: (obs) -> action の関数
        params: 学習済みパラメータ
    """
    try:
        from train.visualize_rl import load_checkpoint, make_policy_network_factory
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as e:
        raise ImportError(f"Cannot import training utilities: {e}")
    
    # checkpointを読み込む
    params = load_checkpoint(str(checkpoint_path))
    if params is None:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
    
    # 観測・行動次元を把握するため、probe envを作る
    probe_env = SenpuuMaruMJXEnv()
    network = make_policy_network_factory(
        probe_env.observation_size,
        probe_env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    
    # params の leading dimension を strip（ある場合）
    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf
    
    params_stripped = jax.tree_util.tree_map(strip_leading_dim, params)
    
    # deterministic=False で policy を作る（stochastic sampling）
    policy_fn = jax.jit(make_policy(params_stripped, deterministic=False))
    
    return policy_fn, params_stripped


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


def evaluate(
    seed: int,
    seconds: float,
    output_dir: Path,
    exp_name: str = "",
    version: Optional[int] = None,
    model: str = "best_params.pkl",
    policy_fn = None,
):
    """Gate 0 evaluation with learned RL policy.
    
    Args:
        seed: random seed
        seconds: simulation duration in seconds
        output_dir: where to save results
        exp_name: experiment name (log/<exp_name>/<version_*>/)
        version: version number within exp_name
        model: checkpoint filename
        policy_fn: pre-loaded policy function. If None, load from checkpoint.
    """
    configure_deterministic_gate0()
    env = SenpuuMaruMJXEnv()
    steps = int(round(seconds / RobotConfig.CONTROL_DT))
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    # Policy を用意する
    if policy_fn is None:
        if not exp_name:
            raise ValueError("Either --exp_name or pre-loaded policy_fn is required")
        checkpoint_path = find_checkpoint(exp_name, version, model)
        if checkpoint_path is None:
            raise RuntimeError(f"Checkpoint not found for exp_name={exp_name}, version={version}")
        print(f"[Gate0] Loading checkpoint: {checkpoint_path}")
        policy_fn, params = load_checkpoint_and_make_policy(checkpoint_path)

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
        rng, rng_policy = jax.random.split(rng)
        action, _ = policy_fn(state.obs, rng_policy)
        state = env.step(state, action)
        sample = measure_state(env, state, geom_ids, sensor_start, sensor_end)
        sample["step"] = step_index + 1
        sample["action_norm"] = float(jp.linalg.norm(action))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="", help="log/<exp_name> の experiment name")
    parser.add_argument("--version", type=int, default=None, help="version number within exp_name")
    parser.add_argument("--model", default="best_params.pkl", help="checkpoint filename")
    parser.add_argument("--seconds", type=float, default=30.0, help="simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "log" / "gate0_formal", help="output directory")
    args = parser.parse_args()
    
    try:
        result = evaluate(
            seed=args.seed,
            seconds=args.seconds,
            output_dir=args.output_dir,
            exp_name=args.exp_name,
            version=args.version,
            model=args.model,
        )
        if not result["pass"]:
            print("[Gate0] FAIL: did not meet acceptance criteria")
            raise SystemExit(1)
        else:
            print("[Gate0] PASS: accepted baseline")
            raise SystemExit(0)
    except Exception as e:
        print(f"[Gate0] ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
