import mujoco
import xml.etree.ElementTree as ET

from robot.config import RobotConfig


def test_visual_geoms_have_no_inertia_mass():
    root = ET.parse(RobotConfig.MUJOCO_MODEL_PATH).getroot()
    visual_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.get("contype") == "0"
        and geom.get("conaffinity") == "0"
        and geom.get("group") == "1"
    ]

    assert visual_geoms
    assert all(float(geom.get("density", "0")) == 0.0 for geom in visual_geoms)


def test_model_total_mass_is_within_design_bound():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))

    total_mass = float(model.body_mass.sum())
    assert total_mass < 6.0, (
        f"total mass suspiciously high: {total_mass} kg "
        "(visual mesh mass may be double-counted)"
    )
