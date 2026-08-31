import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import pickle
import time
import enum
import argparse
import numpy as np
from pathlib import Path

# NumPy compatibility helpers
_old_np_asarray = np.asarray

def _compat_numpy_asarray(a, dtype=None, order=None, copy=True, subok=False, **kwargs):
    try:
        return _old_np_asarray(a, dtype=dtype, order=order, copy=copy, subok=subok)
    except TypeError:
        return _old_np_asarray(a, dtype=dtype, order=order)

np.asarray = _compat_numpy_asarray


# JAXヘッドレス化防止・CPU/EGL選択
# Viewer画面表示時はNative MuJoCo Viewerを起動
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from robot.config import RobotConfig
from envs.mjx_env import SenpuuMaruMJXEnv
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

if not hasattr(running_statistics, "NormalizationMode"):
    class NormalizationMode(enum.IntEnum):
        NONE = 0
    running_statistics.NormalizationMode = NormalizationMode


class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified MJX replay entrypoint")
    parser.add_argument("--exp_name", default="", help="Optional experiment subfolder under log. Leave empty to use log/version_x.")
    parser.add_argument("--version", type=int, default=None, help="Version number to load (e.g. 17 for version_17). If omitted, use latest.")
    parser.add_argument("--model", default="best_params.pkl", help="Checkpoint name")
    parser.add_argument("--mode", choices=["interactive", "video"], default="interactive", help="Replay mode")
    parser.add_argument("--steps", type=int, default=300, help="Number of frames for video mode")
    parser.add_argument("--output", default="simulation_output.gif", help="Output GIF path for video mode")
    return parser.parse_args()


def get_model_path(exp_name: str, version: int | None, model_name: str) -> Path | None:
    root = REPO_ROOT / "log"
    if exp_name:
        root = root / exp_name
    if not root.exists():
        return None
    if version is not None:
        candidate = root / f"version_{version}" / model_name
        return candidate if candidate.exists() else None
    versions = sorted([d for d in root.glob("version_*") if d.is_dir()], key=lambda x: int(x.name.split("_")[-1]))
    for v in reversed(versions):
        candidate = v / model_name
        if candidate.exists():
            return candidate
    return None


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        return CompatibilityUnpickler(f).load()


def make_rma_network_factory(observation_size: int, action_size: int, preprocess_observations_fn=lambda x, _=None: x):
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )


def build_inference_fn(params, env):
    ppo_network = make_rma_network_factory(env.observation_size, env.action_size)
    inference_fn = ppo_networks.make_inference_fn(ppo_network)
    
    # Strip leading pmap dimension from params (tuple: running_stats, policy_params, value_params)
    def _strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf
    
    params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
    return jax.jit(inference_fn(params_stripped))


def run_interactive(params):
    env = SenpuuMaruMJXEnv()
    inference_fn = build_inference_fn(params, env)

    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)

    print("Launching MuJoCo Passive Viewer... Close the window to stop.")
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0

        while viewer.is_running():
            step_start = time.time()
            rng, rng_step = jax.random.split(rng)
            action, _ = inference_fn(state.obs, rng_step)
            state = jax.jit(env.step)(state, action)

            data.qpos[:] = state.pipeline_state.qpos
            data.qvel[:] = state.pipeline_state.qvel
            mujoco.mj_forward(model, data)

            viewer.cam.lookat[:] = 0.92 * np.array(viewer.cam.lookat[:]) + 0.08 * np.array(data.qpos[0:3])
            viewer.sync()

            if getattr(state, "done", False):
                rng, reset_key = jax.random.split(rng)
                state = jax.jit(env.reset)(reset_key)

            elapsed = time.time() - step_start
            sleep_time = RobotConfig.CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def render_video(params, steps: int, output: str):
    env = SenpuuMaruMJXEnv()
    inference_fn = build_inference_fn(params, env)

    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.6
    camera.elevation = -15.0

    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)

    frames = []
    print(f"Rendering {steps} frames to {output}...")
    for step in range(steps):
        rng, rng_step = jax.random.split(rng)
        action, _ = inference_fn(state.obs, rng_step)
        state = jax.jit(env.step)(state, action)

        data.qpos[:] = state.pipeline_state.qpos
        data.qvel[:] = state.pipeline_state.qvel
        mujoco.mj_forward(model, data)

        camera.lookat = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]) + 0.1]
        camera.azimuth = 135.0 + (step * 0.2)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        if getattr(state, "done", False):
            rng, reset_key = jax.random.split(rng)
            state = jax.jit(env.reset)(reset_key)

    output_path = REPO_ROOT / "scratch" / "simulation_output" / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import PIL.Image
    imgs = [PIL.Image.fromarray(frame) for frame in frames]
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    print(f"Saved simulation GIF to: {output_path}")


def main():
    args = parse_args()
    model_path = get_model_path(args.exp_name, args.version, args.model)
    if model_path is None:
        print(f"Error: model file not found for exp_name={args.exp_name}, version={args.version}, model={args.model}")
        return

    print(f"Loading checkpoint from: {model_path}")
    params = load_checkpoint(model_path)

    if args.mode == "interactive":
        run_interactive(params)
    else:
        render_video(params, args.steps, args.output)


if __name__ == "__main__":
    main()
