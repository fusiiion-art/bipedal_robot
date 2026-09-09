#!/usr/bin/env python3
"""
衝突ジオメトリ可視化スクリプト

使用方法:
  python scripts/add_collision_colors.py

このスクリプトは以下を実行します:
1. humanoid.xml の衝突 geom に rgba を追加して humanoid_visualize.xml を生成
2. その可視化用 XML を使って MuJoCo Viewer を開く
3. 衝突形状を部位別に色分け表示
"""

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_source_xml() -> Path:
    """
    元のモデル XML を解決する。
    優先順:
      1) assets/humanoid/humanoid.xml
      2) assets/all/all.xml
      3) どちらもなければ明示的なエラー
    """
    candidates = [
        PROJECT_ROOT / "assets" / "humanoid" / "humanoid.xml",
        PROJECT_ROOT / "assets" / "all" / "all.xml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No valid MuJoCo XML model found. Expected one of:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}"
    )


def colorize_collision_geometry(xml_content: str) -> str:
    """
    XML の衝突 geom 定義へ rgba を追加する。

    部位別カラー:
      - 胴体: 赤
      - 脚: 緑
      - 腕: 青
      - 足裏: 黄
    """
    color_map = {
        r"doutai-v5_doutai_collision": ("1.0", "0.2", "0.2", "0.3"),
        r"_hidaridairou.*collision|_migidaitou.*collision": ("0.2", "1.0", "0.2", "0.3"),
        r"_hidarimomo.*collision|_migimomo.*collision": ("0.2", "1.0", "0.2", "0.3"),
        r"_hidarihizabu.*collision|_migihizabu.*collision": ("0.2", "1.0", "0.2", "0.3"),
        r"ashiura.*collision": ("1.0", "1.0", "0.2", "0.3"),
        r"_hidarikata.*collision|_migikata.*collision": ("0.2", "0.2", "1.0", "0.3"),
    }

    modified_lines = []
    for line in xml_content.splitlines():
        if 'type="' in line and '_collision' in line:
            if 'rgba=' in line:
                modified_lines.append(line)
                continue

            r, g, b, a = "0.5", "0.5", "0.5", "0.3"
            for pattern, color in color_map.items():
                if re.search(pattern, line):
                    r, g, b, a = color
                    break

            if "/>" in line:
                line = line.replace("/>", f' rgba="{r} {g} {b} {a}"/>')
            else:
                line = line.rstrip() + f'\n      rgba="{r} {g} {b} {a}"'

        modified_lines.append(line)

    return "\n".join(modified_lines)


def generate_visualize_script() -> str:
    """MuJoCo Viewer を安全に起動するスクリプトを生成する。"""

    script = '''#!/usr/bin/env python3
"""MuJoCoビューアーで衝突ジオメトリを可視化"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mujoco
import mujoco.viewer


def resolve_model_path() -> Path:
    visual_model = project_root / "assets" / "humanoid" / "humanoid_visualize.xml"
    fallback_model = project_root / "assets" / "humanoid" / "humanoid.xml"

    if visual_model.exists():
        return visual_model
    if fallback_model.exists():
        return fallback_model

    raise FileNotFoundError(
        "No MuJoCo model found. Expected either "
        f"{visual_model} or {fallback_model}. "
        "Run: python scripts/add_collision_colors.py"
    )


def main():
    try:
        model_path = resolve_model_path()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print("\\n=== Collision Geometry Info ===")
    collision_count = 0
    visual_count = 0

    for i in range(model.ngeom):
        name_str = model.geom(i).name
        contype = model.geom_contype[i]
        conaffinity = model.geom_conaffinity[i]

        if "collision" in name_str:
            collision_count += 1
            rgba = model.geom_rgba[i]
            print(f"  [{i}] {name_str}")
            print(f"      contype={contype}, conaffinity={conaffinity}")
            print(f"      rgba=[{rgba[0]:.2f}, {rgba[1]:.2f}, {rgba[2]:.2f}, {rgba[3]:.2f}]")
        elif "geom" in name_str and contype == 0:
            visual_count += 1

    print(f"\\nTotal: {collision_count} collision geoms, {visual_count} visual geoms")
    print("\\n=== Opening MuJoCo Viewer ===")
    print("Tips:")
    print("  - Space: play/pause")
    print("  - Right-drag: rotate view")
    print("  - Middle-drag: pan view")
    print("  - Scroll: zoom")
    print("  - Press 'Escape' or close window to exit\\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.5

        print("Starting physics simulation loop...")

        try:
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
        except KeyboardInterrupt:
            print("\\nViewer closed.")


if __name__ == "__main__":
    main()
'''

    return script


def main():
    print("=" * 70)
    print("衝突ジオメトリ可視化スクリプト")
    print("=" * 70)

    try:
        source_xml = resolve_source_xml()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n[1] Reading source XML: {source_xml}")
    source_xml_text = source_xml.read_text(encoding="utf-8")

    print("[2] Adding rgba to collision geometries...")
    modified_xml = colorize_collision_geometry(source_xml_text)

    output_path = PROJECT_ROOT / "assets" / "humanoid" / "humanoid_visualize.xml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3] Writing modified XML: {output_path}")
    output_path.write_text(modified_xml, encoding="utf-8")

    viewer_script_path = PROJECT_ROOT / "scripts" / "visualize_mujoco.py"
    print(f"[4] Writing viewer script: {viewer_script_path}")
    viewer_script_path.write_text(generate_visualize_script(), encoding="utf-8")

    print("\n" + "=" * 70)
    print("✅ 完了")
    print("=" * 70)
    print("\n使用方法:")
    print("  python scripts/add_collision_colors.py")
    print("  python scripts/visualize_mujoco.py")
    print("\n注意:")
    print("  - humanoid_visualize.xml は visualization 用です")
    print("  - 学習は元の humanoid.xml を使用します")
    print("  - contype / conaffinity は変更しません")


if __name__ == "__main__":
    main()