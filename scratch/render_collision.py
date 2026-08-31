import os
import sys
import numpy as np
import mujoco

# ヘッドレス環境(Linux/WSL)用EGL/OSMesaフラグ設定
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

# プロジェクトルートの設定
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from robot.config import RobotConfig

def render_collision_view():
    xml_path = RobotConfig.MUJOCO_MODEL_PATH
    if not os.path.exists(xml_path):
        xml_path = BASE_DIR / "envs" / "mjx_env.py" # fallback XML if any
        # または fallback XMLを直接文字列で作成
    
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # ロボットを初期姿勢にセット
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # レンダラーの初期化 (640x480)
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)
    
    # 視覚化オプションの設定
    vopt = mujoco.MjvOption()
    
    # 当たり判定(geomgroup[0])と視覚モデル(geomgroup[1], geomgroup[2])の両方を可視化
    vopt.geomgroup[0] = 1  # 当たり判定(Collision) geom
    vopt.geomgroup[1] = 1  # 視覚(Visual) geom
    vopt.geomgroup[2] = 1
    
    # 半透明表示で骨格・判定形状の内部構造を見やすくする
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
    vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

    
    # カメラ設定
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.0, 0.0, 0.4]
    camera.distance = 1.8
    camera.elevation = -15.0
    camera.azimuth = 135.0
    
    # 角度を変えて複数パースペクティブ画像をレンダリング
    views = [
        {"name": "collision_front_angle", "azimuth": 135.0, "elevation": -15.0, "distance": 1.6},
        {"name": "collision_side", "azimuth": 90.0, "elevation": -5.0, "distance": 1.5},
        {"name": "collision_top_down", "azimuth": 180.0, "elevation": -60.0, "distance": 1.8},
        {"name": "collision_close_foot", "azimuth": 140.0, "elevation": -20.0, "distance": 0.8, "lookat": [0.0, 0.0, 0.15]},
    ]
    
    output_dir = BASE_DIR / "scratch" / "collision_renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import PIL.Image
    
    rendered_files = []
    for view in views:
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]
        camera.distance = view["distance"]
        if "lookat" in view:
            camera.lookat = view["lookat"]
        else:
            camera.lookat = [0.0, 0.0, 0.4]
            
        renderer.update_scene(data, camera=camera, scene_option=vopt)
        pixels = renderer.render()
        
        img = PIL.Image.fromarray(pixels)
        filepath = output_dir / f"{view['name']}.png"
        img.save(filepath)
        print(f"Saved: {filepath}")
        rendered_files.append(str(filepath))

if __name__ == "__main__":
    render_collision_view()
