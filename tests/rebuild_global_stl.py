"""
Rebuild robot.xml — preserve original nesting, only reparent top-level bodies.

The original all_clean.xml has bodies that are:
  - Top-level under worldbody (global pos/euler)
  - Already nested (jointed components with relative pos/euler)

Strategy:
  1. Keep ALL original nesting intact (don't flatten).
  2. Only move TOP-LEVEL worldbody children under the torso.
  3. For those moved bodies, convert pos from global to torso-relative.
  4. Joint positions inside nested bodies are already correct (relative).
"""
import xml.etree.ElementTree as ET
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation


INPUT_XML  = r"c:\bipedal_robot\assets\all\all_clean.xml"
OUTPUT_DIR = r"c:\bipedal_robot\assets\all\mujoco_ready"
OUTPUT_XML = os.path.join(OUTPUT_DIR, "robot.xml")


def read_xml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def sanitise_meshes(root, src_mesh_dir, dst_mesh_dir):
    os.makedirs(dst_mesh_dir, exist_ok=True)
    mesh_map = {}
    counter = 1
    asset = root.find('asset')
    to_remove = []

    for mesh in asset.findall('mesh'):
        orig_name = mesh.get('name')
        orig_file = mesh.get('file')
        src_path  = os.path.join(src_mesh_dir, os.path.basename(orig_file))

        if not os.path.exists(src_path):
            to_remove.append(mesh)
            continue
        if orig_name in mesh_map:
            to_remove.append(mesh)
            continue

        new_name = f"mesh_{counter:03d}"
        mesh_map[orig_name] = new_name
        shutil.copy2(src_path, os.path.join(dst_mesh_dir, f"{new_name}.stl"))
        mesh.set('name', new_name)
        mesh.set('file', f"meshes/{new_name}.stl")
        counter += 1

    for m in to_remove:
        asset.remove(m)
    return mesh_map


def remap_geom_meshes(root, mesh_map):
    for parent in list(root.iter()):
        for geom in list(parent.findall('geom')):
            ref = geom.get('mesh')
            if ref is None:
                continue
            if ref in mesh_map:
                geom.set('mesh', mesh_map[ref])
            else:
                parent.remove(geom)


def make_names_unique(root):
    counts = {}
    for el in root.iter():
        if el.tag == 'mesh':
            continue
        name = el.get('name')
        if name is None:
            continue
        key = (el.tag, name)
        if key in counts:
            counts[key] += 1
            el.set('name', f"{name}_{counts[key]}")
        else:
            counts[key] = 0


def get_pos(el):
    s = el.get('pos', '0 0 0')
    return np.array([float(x) for x in s.split()])


def get_euler(el):
    s = el.get('euler', '0 0 0')
    return np.array([float(x) for x in s.split()])


def fmt(v):
    return f"{v[0]} {v[1]} {v[2]}"


def reparent_toplevel_under_torso(worldbody):
    """
    Move top-level bodies under the torso, converting global coords to
    torso-relative coords. Preserve all internal nesting.
    """
    # Find torso
    torso = None
    top_bodies = list(worldbody.findall('body'))
    for b in top_bodies:
        if '胸' in (b.get('name') or ''):
            torso = b
            break
    if torso is None:
        raise RuntimeError("Torso not found")

    # Torso global transform
    t_pos = get_pos(torso)
    t_euler = get_euler(torso)
    t_rot = Rotation.from_euler('xyz', t_euler)
    t_rot_inv = t_rot.inv()

    # Add freejoint
    if torso.find('freejoint') is None:
        fj = ET.Element('freejoint')
        torso.insert(0, fj)

    # Move other top-level bodies under torso
    for b in top_bodies:
        if b is torso:
            continue
        worldbody.remove(b)

        # Convert global pos/euler to torso-relative
        b_pos = get_pos(b)
        b_euler = get_euler(b)
        b_rot = Rotation.from_euler('xyz', b_euler)

        rel_pos = t_rot_inv.apply(b_pos - t_pos)
        rel_rot = t_rot_inv * b_rot
        rel_euler = rel_rot.as_euler('xyz')

        b.set('pos', fmt(rel_pos))
        b.set('euler', fmt(rel_euler))
        torso.append(b)

    # Set torso to a height that places the robot above ground
    torso.set('pos', '0 0 0.20')
    torso.set('euler', '0 0 0')


def add_actuators(root):
    for act in root.findall('actuator'):
        root.remove(act)
    actuator = ET.SubElement(root, 'actuator')
    for joint in root.iter('joint'):
        jname = joint.get('name', '')
        if 'Main-Horn' in jname:
            mot = ET.SubElement(actuator, 'position')
            mot.set('name', f"motor_{jname}")
            mot.set('joint', jname)
            mot.set('kp', '10')
            mot.set('ctrlrange', '-3.14 3.14')


def add_sensors(root):
    for s in root.findall('sensor'):
        root.remove(s)
    for body in root.iter('body'):
        bname = body.get('name', '')
        if 'bno055' in bname or 'FSR' in bname:
            if not any(True for _ in body.findall('site')):
                site = ET.SubElement(body, 'site')
                site.set('name', f"site_{bname}")
                site.set('pos', '0 0 0')
                site.set('size', '0.005')
                site.set('type', 'sphere')
                site.set('rgba', '1 0 0 1')
    sensor = ET.SubElement(root, 'sensor')
    for body in root.iter('body'):
        bname = body.get('name', '')
        if 'bno055' in bname:
            acc = ET.SubElement(sensor, 'accelerometer')
            acc.set('name', 'accel')
            acc.set('site', f"site_{bname}")
            gyro = ET.SubElement(sensor, 'gyro')
            gyro.set('name', 'gyro')
            gyro.set('site', f"site_{bname}")
            break
    for body in root.iter('body'):
        bname = body.get('name', '')
        if 'FSR' in bname:
            t = ET.SubElement(sensor, 'touch')
            t.set('name', f"touch_{bname}")
            t.set('site', f"site_{bname}")


def add_ground_and_light(worldbody):
    if not any(el.tag == 'light' for el in worldbody):
        light = ET.SubElement(worldbody, 'light')
        light.set('directional', 'true')
        light.set('pos', '-0.5 0.5 3')
        light.set('dir', '0 0 -1')
    if not any(g.get('type') == 'plane' for g in worldbody.findall('geom')):
        plane = ET.SubElement(worldbody, 'geom')
        plane.set('pos', '0 0 0')
        plane.set('size', '1 1 1')
        plane.set('type', 'plane')
        plane.set('rgba', '1 0.83 0.61 0.5')


def remove_equality(root):
    for eq in root.findall('equality'):
        root.remove(eq)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xml_str = read_xml(INPUT_XML)
    root = ET.fromstring(xml_str)

    src_mesh_dir = os.path.join(os.path.dirname(INPUT_XML), 'meshes')
    dst_mesh_dir = os.path.join(OUTPUT_DIR, 'meshes')

    mesh_map = sanitise_meshes(root, src_mesh_dir, dst_mesh_dir)
    remap_geom_meshes(root, mesh_map)
    make_names_unique(root)

    worldbody = root.find('worldbody')
    add_ground_and_light(worldbody)
    reparent_toplevel_under_torso(worldbody)
    remove_equality(root)
    add_actuators(root)
    add_sensors(root)

    tree = ET.ElementTree(root)
    ET.indent(tree, space='    ')
    tree.write(OUTPUT_XML, encoding='utf-8', xml_declaration=True)

    n_bodies = len(list(root.iter('body')))
    n_joints = len(list(root.iter('joint')))
    n_act = len(list(root.iter('position')))
    print(f"Written to {OUTPUT_XML}")
    print(f"  Bodies: {n_bodies}, Joints: {n_joints}, Actuators: {n_act}")


if __name__ == '__main__':
    main()
