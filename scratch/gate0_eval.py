#!/usr/bin/env python3
"""Unified Gate 0 evaluation script.

[項目13] gate0_formal_eval.py / gate0_mujoco_eval.py / gate0_standing_eval.py の
3本を統合し、--mode で挙動を切り替える。

Usage:
  # ゼロ行動MJX評価（旧 gate0_standing_eval.py）
  python scratch/gate0_eval.py --mode zero_action_mjx --seconds 5

  # PDホールド MuJoCo評価（旧 gate0_mujoco_eval.py）
  python scratch/gate0_eval.py --mode pd_hold_mujoco --seconds 10

  # 学習済みポリシー評価（旧 gate0_formal_eval.py）
  python scratch/gate0_eval.py --mode policy --exp_name phase0_qual_seed0 --seconds 30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler


# ============================================================================
# 統一された合格基準
# ============================================================================

PASS_CRITERIA = {
    "max_abs_roll_deg": 10.0,
    "max_abs_pitch_deg": 10.0,
    "min_foot_geom_z_m": -0.002,
    "foot_touch_rate": 0.99,
    "torque_saturation_rate": 0.01,  # pd_hold_mujoco モードのみ使用
}


# ============================================================================
# DomainRandomization スコープ（phase0_eval_diagnostics.py から流用）
# ============================================================================

class DomainRandomizationScope:
    """RobotConfigのDR幅を一時的に決定論的な値に固定し、終了時に復元する。"""
    FIELDS = (
        "RANDOM_MASS_SCALE", "RANDOM_FRICTION", "RANDOM_COM_OFFSET",
        "RANDOM_TEMP", "RANDOM_VOLT",
    )

    def __init__(self, deterministic: bool = True):
        self._deterministic = deterministic
        self._saved = {}

    def __enter__(self):
        for name in self.FIELDS:
            self._saved[name] = getattr(RobotConfig, name)
        if self._deterministic:
            RobotConfig.DISTURBANCE_CURRICULUM = False
            RobotConfig.RANDOM_PUSH_MAX_FORCE = 0.0
            for name in self.FIELDS:
                lo, hi = self._saved[name]
                mid = (lo + hi) / 2.0
                setattr(RobotConfig, name, [mid, mid])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, value in self._saved.items():
            setattr(RobotConfig, name, value)
        return False


# ============================================================================
# ユーティリティ
# ============================================================================

def as_float(value):
    return float(np.asarray(value))


# ============================================================================
# MJX系モード (zero_action_mjx / policy)
# ============================================================================

def evaluate_mjx(
    mode: str,
    seed: int,
    seconds: float,
    output_dir: Path,
    exp_name: str = "",
    version: Optional[int] = None,
    model_name: str = "best_params.pkl",
):
    """MJX (JAX) ベースの Gate 0 評価。"""
    import jax
    import jax.numpy as jp
    from envs.mjx_env import SenpuuMaruMJXEnv

    with DomainRandomizationScope(deterministic=True):
        env = SenpuuMaruMJXEnv()
        steps = int(round(seconds / RobotConfig.CONTROL_DT))
        rng = jax.random.PRNGKey(seed)

        # jit化（ループ外で1回だけ作成）
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        state = reset_fn(rng)

        # ポリシーの準備
        policy_fn = None
        if mode == "policy":
            if not exp_name:
                raise ValueError("--exp_name is required for policy mode")
            policy_fn = _load_policy(exp_name, version, model_name, env)

        records = []
        terminated_step = None
        for step_idx in range(steps):
            if mode == "policy":
                rng, rng_policy = jax.random.split(rng)
                action, _ = policy_fn(state.obs, rng_policy)
            else:  # zero_action_mjx
                action = jp.zeros(env.action_size)

            state = step_fn(state, action)

            qpos = np.asarray(state.pipeline_state.qpos)
            rpy = np.asarray(quat_to_euler(state.pipeline_state.qpos[3:7]))
            sample = {
                "step": step_idx + 1,
                "roll_rad": as_float(rpy[0]),
                "pitch_rad": as_float(rpy[1]),
                "torso_z": as_float(qpos[2]),
                "xy_m": as_float(np.linalg.norm(qpos[:2])),
            }
            records.append(sample)

            if bool(state.done):
                terminated_step = step_idx + 1
                break

    return _finalize_results(records, seed, seconds, terminated_step, output_dir, mode)


def _load_policy(exp_name, version, model_name, env):
    """学習済みcheckpointからpolicyを読み込む。"""
    import jax
    from train.visualize_rl import load_checkpoint, get_model_path
    from robot.policy_network import make_policy_network_factory
    from brax.training.agents.ppo import networks as ppo_networks

    model_path = get_model_path(exp_name, version, model_name)
    if model_path is None:
        raise RuntimeError(f"Checkpoint not found for exp_name={exp_name}, version={version}")

    print(f"[Gate0] Loading checkpoint: {model_path}")
    params = load_checkpoint(model_path)

    network = make_policy_network_factory(env.observation_size, env.action_size)
    make_policy = ppo_networks.make_inference_fn(network)

    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf

    params_stripped = jax.tree_util.tree_map(strip_leading_dim, params)
    return jax.jit(make_policy(params_stripped, deterministic=True))


# ============================================================================
# Pure MuJoCo モード (pd_hold_mujoco)
# ============================================================================

def evaluate_pd_hold_mujoco(
    seed: int,
    seconds: float,
    output_dir: Path,
    torso_z: float = 0.1773,
):
    """Pure MuJoCo (CPU, no JAX) ベースの Gate 0 評価。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    model.opt.timestep = RobotConfig.SIM_DT
    data = mujoco.MjData(model)

    # 初期姿勢
    qpos = np.zeros(model.nq, dtype=np.float64)
    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    for act_i in range(min(model.nu, len(default))):
        jnt_id = int(model.actuator_trnid[act_i, 0])
        qpos_idx = int(model.jnt_qposadr[jnt_id])
        qpos[qpos_idx] = default[act_i]
    if model.nq >= 7:
        qpos[0:3] = np.array([0.0, 0.0, torso_z])
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:min(model.nu, len(default))] = default[:min(model.nu, len(default))]
    mujoco.mj_forward(model, data)

    ctrl_steps = int(round(seconds / RobotConfig.CONTROL_DT))
    sim_steps_per_ctrl = max(1, int(round(RobotConfig.CONTROL_DT / RobotConfig.SIM_DT)))

    records = []
    for i in range(ctrl_steps):
        if model.nu > 0:
            data.ctrl[:min(model.nu, len(default))] = default[:min(model.nu, len(default))]
        for _ in range(sim_steps_per_ctrl):
            mujoco.mj_step(model, data)

        qpos_cur = np.asarray(data.qpos)
        rpy = np.asarray(quat_to_euler(qpos_cur[3:7]))
        sample = {
            "step": i + 1,
            "roll_rad": as_float(rpy[0]),
            "pitch_rad": as_float(rpy[1]),
            "torso_z": as_float(qpos_cur[2]),
            "xy_m": as_float(np.linalg.norm(qpos_cur[:2])),
            "max_abs_actuator_force": as_float(np.max(np.abs(data.actuator_force))) if model.nu > 0 else 0.0,
        }
        records.append(sample)

    return _finalize_results(records, seed, seconds, None, output_dir, "pd_hold_mujoco")


# ============================================================================
# 結果集約と判定
# ============================================================================

def _finalize_results(records, seed, seconds, terminated_step, output_dir, mode):
    if not records:
        raise RuntimeError("No simulation samples were collected")

    roll = np.array([r["roll_rad"] for r in records])
    pitch = np.array([r["pitch_rad"] for r in records])

    result = {
        "mode": mode,
        "seed": seed,
        "requested_seconds": seconds,
        "simulated_steps": len(records),
        "simulated_seconds": len(records) * RobotConfig.CONTROL_DT,
        "terminated_step": terminated_step,
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(roll)))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(pitch)))),
        "rms_roll_deg": float(np.rad2deg(np.sqrt(np.mean(roll ** 2)))),
        "rms_pitch_deg": float(np.rad2deg(np.sqrt(np.mean(pitch ** 2)))),
    }

    # 合否判定
    passed = (
        terminated_step is None
        and result["simulated_seconds"] >= seconds
        and result["max_abs_roll_deg"] < PASS_CRITERIA["max_abs_roll_deg"]
        and result["max_abs_pitch_deg"] < PASS_CRITERIA["max_abs_pitch_deg"]
    )

    # pd_hold_mujoco 固有の判定
    if mode == "pd_hold_mujoco" and "max_abs_actuator_force" in records[0]:
        max_force = np.array([r["max_abs_actuator_force"] for r in records])
        result["torque_saturation_rate"] = float(
            np.mean(max_force >= 0.98 * RobotConfig.MOTOR_MAX_TORQUE)
        )
        passed = passed and result["torque_saturation_rate"] <= PASS_CRITERIA["torque_saturation_rate"]

    result["pass"] = bool(passed)

    # mode ごとにサブディレクトリを分けて衝突を回避
    out = output_dir / mode
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / f"gate0_seed_{seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "records"}
    print(json.dumps(summary, indent=2))
    print(f"[Gate0] detailed log: {output_path}")
    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["zero_action_mjx", "pd_hold_mujoco", "policy"],
                        default="zero_action_mjx", help="評価モード")
    parser.add_argument("--exp_name", default="", help="log/<exp_name> の experiment name (policyモード用)")
    parser.add_argument("--version", type=int, default=None, help="version number within exp_name")
    parser.add_argument("--model", default="best_params.pkl", help="checkpoint filename")
    parser.add_argument("--seconds", type=float, default=30.0, help="simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--torso-z", type=float, default=0.1773, help="initial torso height (pd_hold_mujoco)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "log" / "gate0_eval",
                        help="output directory")
    args = parser.parse_args()

    if args.mode == "pd_hold_mujoco":
        result = evaluate_pd_hold_mujoco(args.seed, args.seconds, args.output_dir, args.torso_z)
    else:
        result = evaluate_mjx(
            mode=args.mode,
            seed=args.seed,
            seconds=args.seconds,
            output_dir=args.output_dir,
            exp_name=args.exp_name,
            version=args.version,
            model_name=args.model,
        )

    if not result["pass"]:
        print(f"[Gate0] FAIL ({args.mode})")
        raise SystemExit(1)
    else:
        print(f"[Gate0] PASS ({args.mode})")


if __name__ == "__main__":
    main()
