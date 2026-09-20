"""
当たり判定(_collision ジオム)の自動整合スクリプト
====================================================
「見本を見せて真似させる」のではなく、MJCFファイル自身が持っている
関節ツリーの座標(body pos/euler, joint pos)から機械的に正しい値を
導出する。CADから再エクスポートするたびに実行すれば、腕や脚が
何本増えても・関節配置が変わっても同じロジックで追従できる。

ルール:
  1. sphere型 _collision (関節ハウジング想定):
     pos = そのボディ自身の <joint pos="..."> をそのままコピー
     (同じボディのローカル座標系なので変換不要)

  2. capsule型 _collision (ボーン/リンク想定):
     fromto の始点 = そのボディ自身の <joint pos="...">
     fromto の終点 = 子ボディの <joint pos="..."> を
                     子ボディの pos/euler で親のローカル座標系へ変換した値
     (子が複数ある/子に関節が無い場合は自動導出できないため要手動確認)

  3. box型 _collision (足裏など、関節位置とは無関係な実形状):
     自動修正の対象外。既存値を尊重し、レビュー対象として報告のみ行う。

制限事項:
  - MuJoCoの eulerseq='xyz' (intrinsic X->Y->Z) を前提にしている。
    コンパイラオプションでこれを変更している場合は要調整。
  - 「子が1つだけ」の単純なシリアルチェーンのみ自動計算する。
    分岐(子が2つ以上)や、子に独自の関節を持たないボディは
    "要確認"として報告するだけで自動修正しない。
"""

import os
import re
import struct
import sys
import numpy as np
import xml.etree.ElementTree as ET


def read_stl_vertices(path):
    """バイナリ/ASCII STLを自動判別して全頂点を読み込む(外部依存なし)。"""
    with open(path, 'rb') as f:
        raw = f.read()

    if len(raw) >= 84:
        ntri = struct.unpack_from('<I', raw, 80)[0]
        if 84 + ntri * 50 == len(raw):
            verts = np.empty((ntri * 3, 3), dtype=np.float64)
            offset = 84
            for i in range(ntri):
                v = struct.unpack_from('<9f', raw, offset + 12)
                verts[i*3:i*3+3] = np.array(v).reshape(3, 3)
                offset += 50
            return verts

    text = raw.decode('utf-8', errors='ignore')
    verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('vertex'):
            verts.append([float(x) for x in line.split()[1:4]])
    if not verts:
        raise ValueError(f'STLとして頂点を読み取れませんでした: {path}')
    return np.array(verts, dtype=np.float64)


def compute_mesh_aabb(path, scale=(1.0, 1.0, 1.0)):
    verts = read_stl_vertices(path) * np.array(scale)
    return verts.min(axis=0), verts.max(axis=0)


def parse_mesh_assets(xml_path):
    """<compiler meshdir> と <asset><mesh name file scale> を読み、
    mesh名 -> STLの絶対パス(存在すれば) を引けるようにする。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '.') if compiler is not None else '.'
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    base_dir = os.path.normpath(os.path.join(xml_dir, meshdir))

    meshes = {}
    asset = root.find('asset')
    if asset is not None:
        for m in asset.findall('mesh'):
            name = m.get('name')
            file = m.get('file')
            scale = parse_vec(m.get('scale'), 3) if m.get('scale') else np.array([1.0, 1.0, 1.0])
            path = os.path.join(base_dir, file) if file else None
            meshes[name] = {'path': path, 'scale': scale, 'exists': path is not None and os.path.isfile(path)}
    return meshes


def euler_to_R(e):
    ex, ey, ez = e
    Rx = np.array([[1, 0, 0], [0, np.cos(ex), -np.sin(ex)], [0, np.sin(ex), np.cos(ex)]])
    Ry = np.array([[np.cos(ey), 0, np.sin(ey)], [0, 1, 0], [-np.sin(ey), 0, np.cos(ey)]])
    Rz = np.array([[np.cos(ez), -np.sin(ez), 0], [np.sin(ez), np.cos(ez), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def parse_vec(s, n=3):
    if s is None:
        return np.zeros(n)
    return np.array([float(x) for x in s.split()])


def load_tree(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bodies = {}  # name -> dict

    def walk(elem, parent_name):
        name = elem.get('name')
        if name is None:
            for child in elem.findall('body'):
                walk(child, parent_name)
            return
        joint = elem.find('joint')
        joint_info = None
        if joint is not None and joint.get('type') != 'free':
            joint_info = {'name': joint.get('name'), 'pos': parse_vec(joint.get('pos'))}
        collisions = []
        mesh_name = None
        for geom in elem.findall('geom'):
            gname = geom.get('name', '')
            if geom.get('type') == 'mesh' and geom.get('mesh'):
                mesh_name = geom.get('mesh')
            if gname.endswith('_collision'):
                collisions.append({
                    'name': gname,
                    'type': geom.get('type'),
                    'pos': geom.get('pos'),
                    'fromto': geom.get('fromto'),
                    'size': geom.get('size'),
                })
        bodies[name] = {
            'parent': parent_name,
            'pos': parse_vec(elem.get('pos')),
            'euler': parse_vec(elem.get('euler')),
            'joint': joint_info,
            'mesh_name': mesh_name,
            'collisions': collisions,
            'children': [],
        }
        for child in elem.findall('body'):
            cname = child.get('name')
            bodies[name]['children'].append(cname)
            walk(child, name)

    for body in root.iter('body'):
        # start walk only from top-level bodies (direct children of worldbody)
        pass
    worldbody = root.find('worldbody')
    for top_body in worldbody.findall('body'):
        walk(top_body, None)

    return bodies


def compute_fixes(bodies, meshes=None):
    """戻り値: list of (geom_name, kind, old, new_str, note)"""
    meshes = meshes or {}
    fixes = []
    review = []

    for name, b in bodies.items():
        joint = b['joint']
        for g in b['collisions']:
            gname, gtype = g['name'], g['type']

            if gtype == 'sphere':
                if joint is None:
                    review.append((gname, 'sphere', '自分の関節が無い(自由関節/固定)ため自動導出不可'))
                    continue
                new_pos = joint['pos']
                new_str = ' '.join(f'{v:.4f}' for v in new_pos)
                old = g['pos']
                if old is None or tuple(round(float(x), 4) for x in old.split()) != tuple(round(v, 4) for v in new_pos):
                    fixes.append((gname, 'sphere_pos', old, new_str, ''))

            elif gtype == 'capsule':
                if joint is None:
                    review.append((gname, 'capsule', '自分の関節が無いため自動導出不可'))
                    continue
                children = b['children']
                if len(children) != 1:
                    review.append((gname, 'capsule', f'子ボディが{len(children)}個のため自動導出不可(分岐 or 末端)'))
                    continue
                child = bodies[children[0]]
                if child['joint'] is None:
                    review.append((gname, 'capsule', '子ボディに関節が無いため終点を導出不可'))
                    continue
                proximal = joint['pos']
                R = euler_to_R(child['euler'])
                distal = child['pos'] + R @ child['joint']['pos']
                new_str = ' '.join(f'{v:.4f}' for v in list(proximal) + list(distal))
                old = g['fromto']
                old_vals = tuple(round(float(x), 4) for x in old.split()) if old else None
                new_vals = tuple(round(v, 4) for v in list(proximal) + list(distal))
                if old_vals != new_vals:
                    fixes.append((gname, 'capsule_fromto', old, new_str, ''))

            elif gtype == 'box':
                mesh_name = b['mesh_name']
                mesh_info = meshes.get(mesh_name) if mesh_name else None
                if mesh_info is None:
                    review.append((gname, 'box', f'対応するmeshジオムが見つからない(body={name})'))
                    continue
                if not mesh_info['exists']:
                    review.append((gname, 'box',
                        f'STLが見つからない: {mesh_info["path"]}  '
                        f'-> Fusionのエクスポート先(meshdir配下にSTLがある場所)でこのスクリプトを'
                        f'実行してください。このマシンにはXMLしか無いため自動計算できません。'))
                    continue
                try:
                    mn, mx = compute_mesh_aabb(mesh_info['path'], mesh_info['scale'])
                except Exception as e:
                    review.append((gname, 'box', f'STL読み込み失敗: {e}'))
                    continue
                center = (mn + mx) / 2.0
                half = (mx - mn) / 2.0
                new_pos_str = ' '.join(f'{v:.4f}' for v in center)
                new_size_str = ' '.join(f'{v:.4f}' for v in half)

                old_pos = g['pos']
                old_pos_vals = tuple(round(float(x), 4) for x in old_pos.split()) if old_pos else None
                if old_pos_vals != tuple(round(v, 4) for v in center):
                    fixes.append((gname, 'box_pos', old_pos, new_pos_str, 'STLのAABB中心から算出'))

                old_size = g['size']
                old_size_vals = tuple(round(float(x), 4) for x in old_size.split()) if old_size else None
                if old_size_vals != tuple(round(v, 4) for v in half):
                    fixes.append((gname, 'box_size', old_size, new_size_str, 'STLのAABB半幅から算出'))

    return fixes, review


KIND_TO_ATTR = {
    'sphere_pos': 'pos',
    'capsule_fromto': 'fromto',
    'box_pos': 'pos',
    'box_size': 'size',
}


def apply_fixes(xml_path, fixes, out_path):
    with open(xml_path, encoding='utf-8') as f:
        text = f.read()

    for gname, kind, old, new_str, note in fixes:
        attr = KIND_TO_ATTR[kind]
        # その geom 行を name="..." で一意に特定し、対象属性だけを差し替える
        pattern = re.compile(
            r'(<geom\s+name="' + re.escape(gname) + r'"[^>]*?\s' + attr + r'=")([^"]*)(")'
        )
        m = pattern.search(text)
        if not m:
            # 属性が元々存在しない(省略されていた)ケース: type属性の直後に挿入
            pattern2 = re.compile(r'(<geom\s+name="' + re.escape(gname) + r'"\s+type="[a-z]+")')
            text, n = pattern2.subn(lambda mm: f'{mm.group(1)} {attr}="{new_str}"', text, count=1)
            if n == 0:
                print(f'  [WARN] パターン不一致でスキップ: {gname}')
            continue
        text = text[:m.start(2)] + new_str + text[m.end(2):]

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'humanoid.xml'
    out = sys.argv[2] if len(sys.argv) > 2 else src.replace('.xml', '_autofixed.xml')

    bodies = load_tree(src)
    meshes = parse_mesh_assets(src)
    fixes, review = compute_fixes(bodies, meshes)

    print(f'=== 自動修正が必要な箇所: {len(fixes)}件 ===')
    for gname, kind, old, new_str, note in fixes:
        print(f'  [{kind:15s}] {gname}')
        print(f'      旧: {old}')
        print(f'      新: {new_str}')

    print(f'\n=== 自動導出できず要確認: {len(review)}件 ===')
    for gname, gtype, reason in review:
        print(f'  [{gtype:8s}] {gname:70s} -> {reason}')

    if fixes:
        apply_fixes(src, fixes, out)
        print(f'\n修正版を書き出しました: {out}')
    else:
        print('\n修正の必要な箇所はありませんでした(既に整合済み)。')
