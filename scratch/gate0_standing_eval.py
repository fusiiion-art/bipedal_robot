import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.math_utils import quat_to_euler


def main():
    env = SenpuuMaruMJXEnv()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_xy = 0.0

    for step_idx in range(5):
        action = jnp.zeros(env.action_size, dtype=jnp.float32)
        state = env.step(state, action)

        qpos = state.pipeline_state.qpos
        rpy = quat_to_euler(qpos[3:7])
        max_abs_roll = max(max_abs_roll, float(abs(rpy[0])))
        max_abs_pitch = max(max_abs_pitch, float(abs(rpy[1])))
        max_xy = max(max_xy, float(jnp.linalg.norm(qpos[0:2])))

        if step_idx % 1 == 0:
            print(f"[Gate0] step={step_idx + 1:02d} roll={abs(float(rpy[0])):.4f} pitch={abs(float(rpy[1])):.4f} xy={float(jnp.linalg.norm(qpos[0:2])):.4f}")

        if bool(state.done):
            print(f"Gate 0 terminated at step={step_idx + 1}, done={state.done}")
            break

    final_roll = float(abs(quat_to_euler(state.pipeline_state.qpos[3:7])[0]))
    final_pitch = float(abs(quat_to_euler(state.pipeline_state.qpos[3:7])[1]))
    final_xy = float(jnp.linalg.norm(state.pipeline_state.qpos[0:2]))
    final_z = float(state.pipeline_state.qpos[2])

    print("=== Gate 0 summary ===")
    print(f"max_abs_roll   = {max_abs_roll:.4f} rad ({max_abs_roll * 180.0 / 3.14159:.2f} deg)")
    print(f"max_abs_pitch  = {max_abs_pitch:.4f} rad ({max_abs_pitch * 180.0 / 3.14159:.2f} deg)")
    print(f"max_xy         = {max_xy:.4f} m")
    print(f"final_roll     = {final_roll:.4f} rad")
    print(f"final_pitch    = {final_pitch:.4f} rad")
    print(f"final_xy       = {final_xy:.4f} m")
    print(f"final_z        = {final_z:.4f} m")
    print(f"done           = {bool(state.done)}")

    if bool(state.done):
        raise SystemExit(1)

    if max_abs_roll > 0.25 or max_abs_pitch > 0.25:
        print("Gate 0 FAIL: roll/pitch exceeds 15 deg limit during static standing.")
        raise SystemExit(1)

    print("Gate 0 PASS: static standing remained within the provisional tilt limit for the short baseline window.")


if __name__ == "__main__":
    main()
