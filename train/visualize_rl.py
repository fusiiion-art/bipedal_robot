import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys

# [2026-09-25 修正] --mode video はディスプレイを持たないヘッドレス実行
# (WSL2の裏側、CI、SSHのみ等)でも動く必要がある。mujoco.Renderer は
# デフォルトでGLFW(ウィンドウ系)を要求するため、DISPLAYが無い環境では
# 「OpenGL platform library has not been loaded」で必ず落ちていた。
# --mode interactive は逆にGLFWの実ウィンドウが必要なので、
# mujoco import 前に argv だけ見て video の時だけ EGL(ヘッドレスGPU)を
# 強制する。手元でEGLが使えない場合は MUJOCO_GL=osmesa を明示的に
# 環境変数で渡せば上書きできる(CPUソフトウェアレンダリングにフォールバック)。
if "--mode" in sys.argv:
    _mode_i = sys.argv.index("--mode")
    if len(sys.argv) > _mode_i + 1 and sys.argv[_mode_i + 1] == "video":
        os.environ.setdefault("MUJOCO_GL", "egl")

import pickle
import time
import enum
import argparse
import numpy as np
from pathlib import Path

# NumPy compatibility helpers (古いnumpy pickleとの互換用。現行環境では
# 基本的に素通りするが、他環境で保存された古いcheckpointの読み込みに備えて残す)
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
from robot.policy_network import make_policy_network_factory
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
    parser.add_argument("--mode", choices=["interactive", "video", "plot"], default="interactive", help="Replay mode")
    parser.add_argument("--steps", type=int, default=300, help="Number of frames for video/plot mode")
    parser.add_argument("--output", default="simulation_output.gif", help="Output path for video/plot mode (拡張子は自動調整)")
    # [2026-09-25 追加] 手動外乱注入(学習時のDISTURBANCE_CURRICULUMとは独立)。
    # 現状 robot/config.py は DISTURBANCE_CURRICULUM=False / RANDOM_PUSH_MAX_FORCE=0.0
    # (Phase 0につき意図的に無効化)なので、学習・評価のどのログにも実際の外乱は
    # 一度も加わっていない。「押しても耐えるか」を目視確認したい場合は、
    # ここで指定した外力を可視化スクリプト側から直接 qfrc_applied に注入する。
    parser.add_argument("--disturb_step", type=int, default=None, help="このフレーム番号で水平方向に速度インパルスを加える(例: 150)")
    parser.add_argument("--disturb_velocity", type=float, default=1.5, help="--disturb_step指定時に加える水平速度インパルスの大きさ[m/s]")
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


def build_inference_fn(params, env):
    # [2026-09-25 修正] train_mjx.py は ppo.train(normalize_observations=True, ...)
    # で学習しており、Brax内部で
    #   ppo_network = network_factory(obs_shape, action_size,
    #                                  preprocess_observations_fn=running_statistics.normalize)
    # として観測正規化込みでネットワークを構築している(brax/training/agents/ppo/train.py
    # で実際に確認済み)。ここで preprocess_observations_fn を指定せず(デフォルト=恒等関数)
    # ネットワークを組むと、学習時に正規化された分布を前提に訓練された方策へ
    # 生の(正規化されていない)観測をそのまま入力することになり、
    # 出力する行動が実質デタラメになる
    # (=可視化すると倒れる/暴れる、という「学習は成功しているのに再生だけ失敗する」
    #  典型パターン)。running_statistics.normalize を明示的に渡して整合させる。
    ppo_network = make_policy_network_factory(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    inference_fn = ppo_networks.make_inference_fn(ppo_network)
    
    # Strip leading pmap dimension from params (tuple: running_stats, policy_params, value_params)
    def _strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf
    
    params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
    return jax.jit(inference_fn(params_stripped, deterministic=True))


def apply_visual_push(state, args, step: int):
    """--disturb_step で指定したフレームでのみ、トルソの水平速度に瞬間的な
    インパルスを加えて「押す」動作を再現する。

    RobotConfig.DISTURBANCE_CURRICULUM=False / RANDOM_PUSH_MAX_FORCE=0.0
    (Phase 0につき意図的に無効化、robot/config.py参照)の影響を受けないよう、
    mjx_env.pyのstep()内部ロジックには一切触れず、step呼び出しの合間に
    pipeline_state.qvel を直接書き換えるだけの独立した仕組みにしてある。
    そのため学習・評価側の外乱設定を変更せずに「押しても耐えるか」を
    目視確認できる。"""
    if args.disturb_step is None or step != args.disturb_step:
        return state
    dvel = jp.array([args.disturb_velocity, 0.0, 0.0])
    new_qvel = state.pipeline_state.qvel.at[0:3].add(dvel)
    new_pipeline_state = state.pipeline_state.replace(qvel=new_qvel)
    print(f"  -> frame {step}: applying push (+{args.disturb_velocity} m/s, +X)")
    return state.replace(pipeline_state=new_pipeline_state)


TORSO_BODY_ID = 1  # doutai-v5_doutai (free joint root, qvel[0:3]=並進, robot/config.py非依存)


def apply_mouse_perturbation(state, model, data, viewer, dt: float):
    """MuJoCoビューア上でCtrl+ドラッグ(右クリック=力、左クリック=トルク)して
    ロボットを実際に押した分を、MJX側のpipeline_state.qvelへ反映する。

    mujoco.viewer.launch_passive自体は「passive」という名前の通り物理を
    自前で進めない(=このスクリプトのように外部でstep_fnを呼ぶ構成)ため、
    GUI上のドラッグ操作は素のままだとdata.xfrc_appliedに値が入るだけで、
    実際にMJXでシミュレートされている軌道には一切影響しない
    (=マウスでロボットを掴んでも何も起きないように見える)。
    ここでmujoco.mjv_applyPerturbForce()でGUI操作を実際の力へ変換し、
    F=ma の近似で1フレーム分の速度変化として qvel[0:3] に加算することで、
    「マウスで実際に押して耐えるか試す」を成立させている。
    質量はモデルの全質量を概算値として使用(厳密な運動方程式ではなく、
    目視確認用の近似)。"""
    if viewer.perturb.select <= 0:
        return state
    mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)
    force = np.asarray(data.xfrc_applied[TORSO_BODY_ID, 0:3])
    if not np.any(force):
        return state
    total_mass = float(model.body_mass.sum())
    dvel = jp.asarray(force / max(total_mass, 1e-3) * dt)
    new_qvel = state.pipeline_state.qvel.at[0:3].add(dvel)
    new_pipeline_state = state.pipeline_state.replace(qvel=new_qvel)
    return state.replace(pipeline_state=new_pipeline_state)


def run_interactive(params, args):
    env = SenpuuMaruMJXEnv()
    inference_fn = build_inference_fn(params, env)

    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)

    print("Launching MuJoCo Passive Viewer... Close the window to stop.")
    rng = jax.random.PRNGKey(0)
    # [項目12] jax.jit()をループ外で1回だけ作成して使い回す
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(rng)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0

        step = 0
        while viewer.is_running():
            step_start = time.time()
            rng, rng_step = jax.random.split(rng)
            action, _ = inference_fn(state.obs, rng_step)
            state = step_fn(state, action)
            state = apply_visual_push(state, args, step)
            state = apply_mouse_perturbation(state, model, data, viewer, RobotConfig.CONTROL_DT)
            step += 1

            data.qpos[:] = state.pipeline_state.qpos
            data.qvel[:] = state.pipeline_state.qvel
            mujoco.mj_forward(model, data)

            viewer.cam.lookat[:] = 0.92 * np.array(viewer.cam.lookat[:]) + 0.08 * np.array(data.qpos[0:3])
            viewer.sync()

            if getattr(state, "done", False):
                rng, reset_key = jax.random.split(rng)
                state = reset_fn(reset_key)
                step = 0

            elapsed = time.time() - step_start
            sleep_time = RobotConfig.CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def render_video(params, args):
    steps, output = args.steps, args.output
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
    # [項目12] jax.jit()をループ外で1回だけ作成して使い回す
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(rng)

    frames = []
    print(f"Rendering {steps} frames to {output}...")
    for step in range(steps):
        rng, rng_step = jax.random.split(rng)
        action, _ = inference_fn(state.obs, rng_step)
        state = step_fn(state, action)
        state = apply_visual_push(state, args, step)

        data.qpos[:] = state.pipeline_state.qpos
        data.qvel[:] = state.pipeline_state.qvel
        mujoco.mj_forward(model, data)

        camera.lookat = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]) + 0.1]
        camera.azimuth = 135.0 + (step * 0.2)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        if getattr(state, "done", False):
            rng, reset_key = jax.random.split(rng)
            state = reset_fn(reset_key)

    output_path = REPO_ROOT / "scratch" / "simulation_output" / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import PIL.Image
    imgs = [PIL.Image.fromarray(frame) for frame in frames]
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    print(f"Saved simulation GIF to: {output_path}")


def render_plot(params, args):
    """3DレンダリングやGPU/ディスプレイを一切使わず、ロールアウトの主要な
    物理量を時系列グラフとしてPNG保存する。GIF/インタラクティブでは
    目で追いにくい微小な振動・ドリフト・傾きの癖を数値で確認するための、
    GIFとは独立した検証手段。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, output = args.steps, args.output
    env = SenpuuMaruMJXEnv()
    inference_fn = build_inference_fn(params, env)

    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(rng)

    torso_z, tilt_deg, drift_xy, both_feet, disturb_marks = [], [], [], [], []
    print(f"Rolling out {steps} steps (no rendering)...")
    for step in range(steps):
        rng, rng_step = jax.random.split(rng)
        action, _ = inference_fn(state.obs, rng_step)
        state = step_fn(state, action)
        state = apply_visual_push(state, args, step)

        qpos = np.asarray(state.pipeline_state.qpos)
        torso_z.append(float(qpos[2]))
        drift_xy.append((float(qpos[0]), float(qpos[1])))
        # トルソZ軸(qpos[3:7]のクォータニオン)と世界Z軸との傾き角
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        body_z_world = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
        tilt_rad = float(np.arccos(np.clip(body_z_world[2], -1.0, 1.0)))
        tilt_deg.append(np.degrees(tilt_rad))
        both_feet.append(float(state.metrics.get("both_feet_contact", 0.0)))
        disturb_marks.append(args.disturb_step is not None and step == args.disturb_step)

        if getattr(state, "done", False):
            rng, reset_key = jax.random.split(rng)
            state = reset_fn(reset_key)

    t = np.arange(steps) * RobotConfig.CONTROL_DT
    dx = [p[0] - drift_xy[0][0] for p in drift_xy]
    dy = [p[1] - drift_xy[0][1] for p in drift_xy]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    labels_data = [
        ("Torso height z [m]", torso_z, None),
        ("Tilt from upright [deg]", tilt_deg, None),
        ("Horizontal drift from start [m]", dx, dy),
        ("both_feet_contact (per-step)", both_feet, None),
    ]
    for ax, (label, y1, y2) in zip(axes, labels_data):
        ax.plot(t, y1, label="x" if y2 is not None else None, linewidth=1.2)
        if y2 is not None:
            ax.plot(t, y2, label="y", linewidth=1.2)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        for i, marked in enumerate(disturb_marks):
            if marked:
                ax.axvline(t[i], color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[-1].set_xlabel("time [s]")
    if args.disturb_step is not None:
        axes[0].set_title(f"Rollout diagnostics (red dashed line = push injected at step {args.disturb_step})")
    else:
        axes[0].set_title("Rollout diagnostics")
    fig.tight_layout()

    out_name = Path(output).stem + ".png" if Path(output).suffix else output
    output_path = REPO_ROOT / "scratch" / "simulation_output" / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    print(f"Saved rollout plot to: {output_path}")


def main():
    args = parse_args()
    model_path = get_model_path(args.exp_name, args.version, args.model)
    if model_path is None:
        print(f"Error: model file not found for exp_name={args.exp_name}, version={args.version}, model={args.model}")
        return

    print(f"Loading checkpoint from: {model_path}")
    params = load_checkpoint(model_path)

    if args.mode == "interactive":
        run_interactive(params, args)
    elif args.mode == "video":
        render_video(params, args)
    else:
        render_plot(params, args)


if __name__ == "__main__":
    main()
