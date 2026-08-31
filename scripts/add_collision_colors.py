#!/usr/bin/env python3
"""
衝突ジオメトリ可視化スクリプト

使用方法:
  python scripts/add_collision_colors.py

このスクリプトは以下を実行します:
1. humanoid.xmlの衝突geomにrgbaを追加（修正版を生成）
2. MuJoCoビューアーで可視化
3. 衝突形状を部位別に色分け表示
"""

import os
import sys
import re
from pathlib import Path

def colorize_collision_geometry(xml_content):
    """
    XMLの衝突geom定義にrgbaを追加
    
    部位別の色設定:
    - 胴体: 赤 (1.0, 0.2, 0.2, 0.3)
    - 脚: 緑 (0.2, 1.0, 0.2, 0.3)
    - 腕: 青 (0.2, 0.2, 1.0, 0.3)
    - 足裏: 黄 (1.0, 1.0, 0.2, 0.3)
    """
    
    # パターンマッピング: 衝突geom名パターン → 色
    color_map = {
        r'doutai-v5_doutai_collision': ('1.0', '0.2', '0.2', '0.3'),    # 胴体: 赤
        r'_hidaridairou.*collision|_migidaitou.*collision': ('0.2', '1.0', '0.2', '0.3'),  # 脚: 緑
        r'_hidarimomo.*collision|_migimomo.*collision': ('0.2', '1.0', '0.2', '0.3'),     # 太もも: 緑
        r'_hidarihizabu.*collision|_migihizabu.*collision': ('0.2', '1.0', '0.2', '0.3'),  # ふくらはぎ: 緑
        r'ashiura.*collision': ('1.0', '1.0', '0.2', '0.3'),    # 足裏: 黄
        r'_hidarikata.*collision|_migikata.*collision': ('0.2', '0.2', '1.0', '0.3'),  # 腕: 青
    }
    
    lines = xml_content.split('\n')
    modified_lines = []
    
    for line in lines:
        modified = False
        
        # 衝突geom行を検出
        if 'type="' in line and '_collision' in line:
            # 既にrgbaがあれば行をスキップ
            if 'rgba=' in line:
                modified_lines.append(line)
                continue
            
            # 色を決定
            r, g, b, a = '0.5', '0.5', '0.5', '0.3'  # デフォルト
            
            for pattern, color in color_map.items():
                if re.search(pattern, line):
                    r, g, b, a = color
                    break
            
            # rgbaを追加（geom要素の終了前）
            if '/>' in line:
                # 単一タグの場合
                line = line.replace('/>', f' rgba="{r} {g} {b} {a}"/>')
                modified = True
            else:
                # 複数行にまたがる場合は末尾に追加
                line = line.rstrip() + f'\n      rgba="{r} {g} {b} {a}"'
                modified = True
        
        modified_lines.append(line)
    
    return '\n'.join(modified_lines)

def generate_visualize_script():
    """MuJoCoビューアーで可視化するスクリプトを生成"""
    
    script = '''#!/usr/bin/env python3
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
    print("\\n=== Collision Geometry Info ===")
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
    
    print(f"\\nTotal: {collision_count} collision geoms, {visual_count} visual geoms")
    print("\\n=== Opening MuJoCo Viewer ===")
    print("Tips:")
    print("  - Space: play/pause")
    print("  - Right-drag: rotate view")
    print("  - Middle-drag: pan view")
    print("  - Scroll: zoom")
    print("  - Press 'Escape' or close window to exit\\n")
    
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
            print("\\nViewer closed.")

if __name__ == '__main__':
    main()
'''
    
    return script

def main():
    print("=" * 60)
    print("衝突ジオメトリ可視化スクリプト")
    print("=" * 60)
    
    # XMLパス
    xml_path = Path('assets/humanoid/humanoid.xml')
    
    if not xml_path.exists():
        print(f"Error: {xml_path} not found")
        sys.exit(1)
    
    # XMLを読み込み
    print(f"\n[1] Reading: {xml_path}")
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # 衝突geomを色付け
    print("[2] Adding rgba to collision geometries...")
    modified_xml = colorize_collision_geometry(xml_content)
    
    # 修正版を出力
    output_path = Path('assets/humanoid/humanoid_visualize.xml')
    print(f"[3] Writing modified XML: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modified_xml)
    
    # ビューアースクリプトを生成
    script_path = Path('scripts/visualize_mujoco.py')
    script_path.parent.mkdir(exist_ok=True)
    
    print(f"[4] Creating viewer script: {script_path}")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(generate_visualize_script())
    
    print("\n" + "=" * 60)
    print("✅ 完了！")
    print("=" * 60)
    print("\n使用方法:")
    print(f"1. このスクリプトを実行:")
    print(f"   python scripts/add_collision_colors.py")
    print(f"\n2. ビューアースクリプトで可視化:")
    print(f"   python scripts/visualize_mujoco.py")
    print(f"\n📌 注意:")
    print(f"   - humanoid_visualize.xml は Visualization用です")
    print(f"   - 学習には元の humanoid.xml を使用してください")
    print(f"   - contype/conaffinity は変わっていません（学習への影響なし）")

if __name__ == '__main__':
    main()
