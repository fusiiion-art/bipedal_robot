import xml.etree.ElementTree as ET
import numpy as np

def parse_xml():
    xml_path = "assets/humanoid/humanoid.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    worldbody = root.find('worldbody')
    
    print("=" * 80)
    print("MuJoCo Model Body Tree & Geom Analysis")
    print("=" * 80)
    
    def print_body(body, indent=""):
        name = body.get('name', 'unnamed')
        pos = body.get('pos', '0 0 0')
        euler = body.get('euler', '0 0 0')
        
        print(f"{indent}[Body] Body: {name}")
        print(f"{indent}   pos: {pos} | euler: {euler}")
        
        # Joints
        for joint in body.findall('joint'):
            jname = joint.get('name', 'unnamed')
            jpos = joint.get('pos', '0 0 0')
            jaxis = joint.get('axis', '0 0 1')
            jrange = joint.get('range', 'N/A')
            print(f"{indent}   [Joint]: {jname} | pos: {jpos} | axis: {jaxis} | range: {jrange}")
            
        # Geoms (Visual & Collision)
        for geom in body.findall('geom'):
            gname = geom.get('name', 'unnamed')
            gtype = geom.get('type', 'box')
            gsize = geom.get('size', 'N/A')
            gpos = geom.get('pos', '0 0 0')
            gfromto = geom.get('fromto', '')
            rgba = geom.get('rgba', '')
            
            # visual check (group=1 or contype=0 usually means visual only)
            group = geom.get('group', '0')
            contype = geom.get('contype', '1')
            is_collision = (contype != '0' and group != '1')
            role = "[Collision]" if is_collision else "[Visual]"
            
            geom_info = f"{indent}   {role}: {gname} ({gtype})"
            if gfromto:
                geom_info += f" | fromto: {gfromto}"
            else:
                geom_info += f" | pos: {gpos} | size: {gsize}"
            
            if rgba:
                geom_info += f" | rgba: {rgba}"
            print(geom_info)
            
        # Recurse children
        for child in body.findall('body'):
            print_body(child, indent + "    ")
            
    # Find root body under worldbody
    for body in worldbody.findall('body'):
        print_body(body)

if __name__ == '__main__':
    parse_xml()
