import os
import sys
import numpy as np
import mujoco

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from robot.config import RobotConfig

def render_pure_geoms_only():
    xml_path = RobotConfig.MUJOCO_MODEL_PATH
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)
    
    # 完全に純粋な GeomGroup 0 (Collision Geoms) のみ
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 1
    vopt.geomgroup[1] = 0
    vopt.geomgroup[2] = 0
    vopt.geomgroup[3] = 0
    vopt.geomgroup[4] = 0
    
    # 関節軸やその他のオーバーレイ表示をオフにして純粋なgeomのみ見せる
    vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    views = [
        {"name": "pure_collision_front", "azimuth": 135.0, "elevation": -15.0, "distance": 1.2, "lookat": [0.0, 0.0, 0.35]},
        {"name": "pure_collision_side", "azimuth": 90.0, "elevation": -5.0, "distance": 1.2, "lookat": [0.0, 0.0, 0.35]},
        {"name": "pure_collision_feet", "azimuth": 140.0, "elevation": -20.0, "distance": 0.6, "lookat": [0.0, 0.0, 0.15]},
    ]
    
    output_dir = BASE_DIR / "scratch" / "collision_renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import PIL.Image
    
    for view in views:
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]
        camera.distance = view["distance"]
        camera.lookat = view["lookat"]
            
        renderer.update_scene(data, camera=camera, scene_option=vopt)
        pixels = renderer.render()
        
        img = PIL.Image.fromarray(pixels)
        filepath = output_dir / f"{view['name']}.png"
        img.save(filepath)
        print(f"Saved: {filepath}")

if __name__ == "__main__":
    render_pure_geoms_only()
