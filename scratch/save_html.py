import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traj_path = os.path.join(root_dir, "trajectory.npy")
    
    if not os.path.exists(traj_path):
        print("エラー: trajectory.npy が見つかりません。")
        return

    print("モデルと軌跡データを読み込み中...")
    m = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    d = mujoco.MjData(m)
    traj = np.load(traj_path)

    # レンダーの初期化 (オフスクリーンレンダラー)
    renderer = mujoco.Renderer(m, height=480, width=640)
    frames = []

    print(f"オフスクリーンレンダリング中 ({len(traj)} フレーム)...")
    # 5ステップごとに1フレーム（間引きしてGIFの容量とレンダリング時間を軽量化）
    for i, qpos in enumerate(traj):
        if i % 3 != 0:
            continue
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        renderer.update_scene(d)
        pixels = renderer.render()
        frames.append(pixels)

    gif_out = os.path.join(root_dir, "log", "version_2", "simulation.gif")
    print(f"GIFアニメーションを保存中: {gif_out}")
    
    try:
        from PIL import Image
        img_list = [Image.fromarray(f) for f in frames]
        img_list[0].save(
            gif_out,
            save_all=True,
            append_images=img_list[1:],
            duration=33, # ~30fps
            loop=0
        )
        print("✓ simulation.gif の保存に成功しました！")
    except Exception as e:
        print(f"保存エラー: {e}")

if __name__ == "__main__":
    main()
