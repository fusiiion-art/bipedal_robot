"""Collision geometry visualization.

[項目14] render_collision.py と render_pure_collision.py を統合。
--mode で表示切り替え。

Usage:
  python scratch/render_collision.py --mode full   # 全geomグループ表示
  python scratch/render_collision.py --mode pure   # Collision geom のみ
"""
import argparse
import os
import sys
import numpy as np
import mujoco

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from robot.config import RobotConfig


def render(mode: str = "full"):
    xml_path = RobotConfig.MUJOCO_MODEL_PATH
    if not os.path.exists(xml_path):
        print(f"Error: model file not found: {xml_path}")
        return

    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)

    vopt = mujoco.MjvOption()

    if mode == "pure":
        # Collision geom のみ表示
        vopt.geomgroup[0] = 1
        for g in range(1, 6):
            vopt.geomgroup[g] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        views = [
            {"name": "pure_collision_front", "azimuth": 135.0, "elevation": -15.0, "distance": 1.2, "lookat": [0.0, 0.0, 0.35]},
            {"name": "pure_collision_side", "azimuth": 90.0, "elevation": -5.0, "distance": 1.2, "lookat": [0.0, 0.0, 0.35]},
            {"name": "pure_collision_feet", "azimuth": 140.0, "elevation": -20.0, "distance": 0.6, "lookat": [0.0, 0.0, 0.15]},
        ]
    else:
        # 全geomグループ + 半透明 + 関節軸表示
        vopt.geomgroup[0] = 1
        vopt.geomgroup[1] = 1
        vopt.geomgroup[2] = 1
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
        vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        views = [
            {"name": "collision_front_angle", "azimuth": 135.0, "elevation": -15.0, "distance": 1.6},
            {"name": "collision_side", "azimuth": 90.0, "elevation": -5.0, "distance": 1.5},
            {"name": "collision_top_down", "azimuth": 180.0, "elevation": -60.0, "distance": 1.8},
            {"name": "collision_close_foot", "azimuth": 140.0, "elevation": -20.0, "distance": 0.8, "lookat": [0.0, 0.0, 0.15]},
        ]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    output_dir = BASE_DIR / "scratch" / "collision_renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    import PIL.Image

    for view in views:
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]
        camera.distance = view["distance"]
        camera.lookat = view.get("lookat", [0.0, 0.0, 0.4])

        renderer.update_scene(data, camera=camera, scene_option=vopt)
        pixels = renderer.render()

        img = PIL.Image.fromarray(pixels)
        filepath = output_dir / f"{view['name']}.png"
        img.save(filepath)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render collision geometry views")
    parser.add_argument("--mode", choices=["full", "pure"], default="full",
                        help="full: all geom groups + overlays; pure: collision geoms only")
    args = parser.parse_args()
    render(args.mode)
