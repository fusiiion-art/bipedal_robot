import os
import sys
import pickle
import argparse

import jax
import jax.numpy as jnp
import numpy as np

# sysモジュールのパッチ (Windows/WSL上のbrax/orbax依存対策)
if not hasattr(sys.modules.get("uvloop", None), "__name__"):
    sys.modules["uvloop"] = type(sys)("uvloop")

from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from envs.mjx_env import SenpuuMaruMJXEnv  # noqa: F401
from train.train_mjx import make_policy_network_factory
from train.visualize_rl import load_checkpoint

def render_trajectory_to_gif(traj: np.ndarray, gif_path: str, height: int = 480, width: int = 640, fps: int = 30):
    """保存済み軌跡データからMuJoCoオフスクリーンレンダラーでGIFを生成する。"""
    try:
        import mujoco
        from PIL import Image
    except ImportError as e:
        print(f"GIFレンダリングに必要なライブラリがありません (mujoco, Pillow): {e}")
        return

    print("モデルと軌跡データを読み込み中...")
    m = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    d = mujoco.MjData(m)

    renderer = mujoco.Renderer(m, height=height, width=width)
    frames = []

    print(f"オフスクリーンレンダリング中 ({len(traj)} フレーム)...")
    # 3フレームごとに間引きして容量と描画負荷を抑制
    for i, qpos in enumerate(traj):
        if i % 3 != 0:
            continue
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        renderer.update_scene(d)
        pixels = renderer.render()
        frames.append(pixels)

    if not frames:
        print("レンダリング対象フレームがありません。")
        return

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
    print(f"GIFアニメーションを保存中: {gif_path}")
    try:
        img_list = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps)
        img_list[0].save(
            gif_path,
            save_all=True,
            append_images=img_list[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"✓ {gif_path} の保存に成功しました！")
    except Exception as e:
        print(f"GIF保存エラー: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export trajectory (.npy) and optional simulation GIF from policy.")
    parser.add_argument("--version", type=int, default=8, help="Model version directory under log/")
    parser.add_argument("--model", type=str, default="best_params.pkl", help="Model checkpoint filename")
    parser.add_argument("--output", type=str, default=None, help="Output path for trajectory .npy file")
    parser.add_argument("--gif", type=str, default=None, help="Output path for simulation GIF (e.g. simulation.gif)")
    parser.add_argument("--from-npy", type=str, default=None, help="Directly render existing trajectory .npy to GIF without running policy")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 既存の .npy から GIF のみ生成する場合
    if args.from_npy:
        if not os.path.exists(args.from_npy):
            print(f"エラー: 指定された軌跡ファイルが見つかりません: {args.from_npy}")
            return
        traj = np.load(args.from_npy)
        gif_out = args.gif or os.path.join(root_dir, "simulation.gif")
        render_trajectory_to_gif(traj, gif_out)
        return

    model_path = os.path.join(root_dir, "log", f"version_{args.version}", args.model)
    if not os.path.exists(model_path):
        model_path = os.path.join(root_dir, "log", "mjx_ppo_rma_100hz", f"version_{args.version}", args.model)
    
    print(f"モデルをロード中: {model_path}")
    params = load_checkpoint(model_path)
        
    jax.config.update('jax_platform_name', 'cpu')
    env = envs.get_environment('senpuu_maru_mjx')
    
    ppo_network = make_policy_network_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    def _strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf

    params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
    policy = make_policy(params_stripped, deterministic=True)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)
    state = jit_reset(key_reset)
    
    qpos_list = []
    
    print("AIが生成する軌跡を計算中 (WSLで実行中)...")
    for _ in range(800):  # 8秒間分シミュレーション
        rng, key_act = jax.random.split(rng)
        act_params = policy(state.obs, key_act)
        state = jit_step(state, act_params[0])
        qpos_list.append(np.array(state.pipeline_state.qpos))
        
        if state.done:
            break
            
    traj = np.array(qpos_list)
    out_path = args.output or os.path.join(root_dir, "trajectory.npy")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, traj)
    print(f"軌跡データを保存しました: {out_path}")

    if args.gif:
        render_trajectory_to_gif(traj, args.gif)


if __name__ == "__main__":
    main()
