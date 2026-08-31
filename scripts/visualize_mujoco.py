#!/usr/bin/env python3
"""MuJoCoビューアーで衝突ジオメトリを可視化"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer

def main():
    model_path = 'assets/humanoid/humanoid_visualize.xml'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run: python scripts/add_collision_colors.py")
        sys.exit(1)
    
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Geom情報を表示
    print("\n=== Collision Geometry Info ===")
    collision_count = 0
    visual_count = 0
    
    for i in range(model.ngeom):
        name_str = model.geom(i).name
        contype = model.geom_contype[i]
        conaffinity = model.geom_conaffinity[i]
        
        if 'collision' in name_str:
            collision_count += 1
            rgba = model.geom_rgba[i]
            print(f"  [{i}] {name_str}")
            print(f"      contype={contype}, conaffinity={conaffinity}")
            print(f"      rgba=[{rgba[0]:.2f}, {rgba[1]:.2f}, {rgba[2]:.2f}, {rgba[3]:.2f}]")
        elif 'geom' in name_str and contype == 0:
            visual_count += 1
    
    print(f"\nTotal: {collision_count} collision geoms, {visual_count} visual geoms")
    print("\n=== Opening MuJoCo Viewer ===")
    print("Tips:")
    print("  - Space: play/pause")
    print("  - Right-drag: rotate view")
    print("  - Middle-drag: pan view")
    print("  - Scroll: zoom")
    print("  - Press 'Escape' or close window to exit\n")
    
    # ビューアーで表示
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 視点を少し回転させて見やすく
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.5
        
        import time
        print("Starting physics simulation loop...")
        
        try:
            while viewer.is_running():
                step_start = time.time()
                
                # 物理シミュレーションを1ステップ進める
                mujoco.mj_step(model, data)
                
                # ビューアーの同期
                viewer.sync()
                
                # 物理演算のタイムステップ（デフォルトは0.002秒等）に同期
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
        except KeyboardInterrupt:
            print("\nViewer closed.")

if __name__ == '__main__':
    main()
