#!/usr/bin/env python3
"""Set visual-only geom density to zero for inertiafromgeom models."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    PROJECT_ROOT / "assets" / "humanoid" / "humanoid.xml",
    PROJECT_ROOT / "assets" / "humanoid" / "humanoid_visualize.xml",
)


def fix_visual_geom_mass(xml_path: Path, out_path: Path) -> int:
    """Set density=0 on visual geoms that do not already define mass properties."""
    lines = xml_path.read_text(encoding="utf-8").splitlines()
    fixed = []
    changed = 0

    for line in lines:
        is_visual = (
            "<geom" in line
            and 'contype="0"' in line
            and 'conaffinity="0"' in line
            and 'group="1"' in line
        )
        has_mass_property = "density=" in line or "mass=" in line
        if is_visual and not has_mass_property:
            if "/>" not in line:
                raise ValueError(f"Expected self-closing geom definition: {xml_path}")
            line = line.replace("/>", ' density="0"/>', 1)
            changed += 1
        fixed.append(line)

    out_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    for xml_path in MODEL_PATHS:
        if not xml_path.exists():
            continue
        changed = fix_visual_geom_mass(xml_path, xml_path)
        print(f"fixed: {xml_path} ({changed} geoms)")


if __name__ == "__main__":
    main()
