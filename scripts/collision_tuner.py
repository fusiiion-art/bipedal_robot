"""
collision_tuner.py — 当たり判定リアルタイム調整ツール (ブラウザベース)

MuJoCo Python バインディングを使わず、ブラウザ上で
STLメッシュ + 当たり判定プリミティブを 3D 表示する。

使い方:
  1. python scripts/collision_tuner.py を実行
  2. ブラウザが自動で開く (http://localhost:8742)
  3. VS Code で humanoid_visualize.xml を編集して保存
  4. ブラウザが自動リロードし、変更が即反映される
  5. Ctrl+C で終了 → humanoid.xml に自動同期
"""

import os
import sys
import json
import re
import time
import threading
import webbrowser
import http.server
import socketserver
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
VISUALIZE_XML = ASSETS_DIR / "humanoid" / "humanoid_visualize.xml"
TRAINING_XML  = ASSETS_DIR / "humanoid" / "humanoid.xml"
MESHES_DIR    = ASSETS_DIR / "all" / "meshes"

PORT = 8742

# --- XML パーサー ---

def parse_collision_geoms(xml_path: str) -> list:
    """XMLから当たり判定用 geom を抽出し、JSON用 dict のリストを返す"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    geoms = []

    # body の階層を再帰的に走査し、各 geom の world-frame 情報を収集
    def walk_body(body_elem, parent_name=""):
        for child in body_elem:
            if child.tag == "body":
                walk_body(child, child.get("name", ""))
            elif child.tag == "geom":
                name = child.get("name", "")
                geom_type = child.get("type", "sphere")

                # メッシュ geom も視覚用として収集
                is_collision = "collision" in name
                is_mesh = geom_type == "mesh"

                if not is_collision and not is_mesh:
                    continue

                geom_data = {
                    "name": name,
                    "type": geom_type,
                    "is_collision": is_collision,
                    "parent_body": parent_name,
                    "size": child.get("size", "0.01"),
                    "pos": child.get("pos", "0 0 0"),
                    "fromto": child.get("fromto", ""),
                    "rgba": child.get("rgba", "0.5 0.5 0.5 0.5"),
                    "mesh_name": child.get("mesh", ""),
                    "mass": child.get("mass", ""),
                    "friction": child.get("friction", ""),
                }
                geoms.append(geom_data)

    # worldbody 配下を走査
    worldbody = root.find("worldbody")
    if worldbody is not None:
        walk_body(worldbody)

    return geoms


def parse_bodies_recursive(xml_path: str) -> list:
    """XMLからbody階層を再帰的にパースし、各bodyの位置・回転と子geomを返す"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def parse_body(body_elem):
        joint_elem = body_elem.find("joint")
        joint_data = None
        if joint_elem is not None and joint_elem.get("type") != "free":
            joint_data = {
                "name": joint_elem.get("name", ""),
                "axis": joint_elem.get("axis", "1 0 0"),
                "pos": joint_elem.get("pos", "0 0 0"),
                "range": joint_elem.get("range", "-3.14 3.14"),
            }

        body_data = {
            "name": body_elem.get("name", "root"),
            "pos": body_elem.get("pos", "0 0 0"),
            "euler": body_elem.get("euler", "0 0 0"),
            "quat": body_elem.get("quat", ""),
            "joint": joint_data,
            "geoms": [],
            "children": [],
        }

        for child in body_elem:
            if child.tag == "geom":
                name = child.get("name", "")
                geom_type = child.get("type", "sphere")
                geom_data = {
                    "name": name,
                    "type": geom_type,
                    "is_collision": "collision" in name,
                    "size": child.get("size", "0.01"),
                    "pos": child.get("pos", "0 0 0"),
                    "fromto": child.get("fromto", ""),
                    "rgba": child.get("rgba", "0.5 0.5 0.5 1.0"),
                    "mesh_name": child.get("mesh", ""),
                    "contype": child.get("contype", "1"),
                    "conaffinity": child.get("conaffinity", "1"),
                    "group": child.get("group", "0"),
                }
                body_data["geoms"].append(geom_data)
            elif child.tag == "body":
                body_data["children"].append(parse_body(child))

        return body_data

    worldbody = root.find("worldbody")
    bodies = []
    if worldbody is not None:
        for child in worldbody:
            if child.tag == "body":
                bodies.append(parse_body(child))
    return bodies


def get_stl_files() -> list:
    """meshes/ ディレクトリ内のSTLファイル一覧を返す"""
    stl_files = []
    if MESHES_DIR.exists():
        for f in sorted(MESHES_DIR.glob("*.stl")):
            stl_files.append(f.name)
    return stl_files


def sync_to_training_xml():
    """humanoid_visualize.xml → humanoid.xml に当たり判定を同期"""
    with open(VISUALIZE_XML, "r", encoding="utf-8") as f:
        vis_content = f.read()
    with open(TRAINING_XML, "r", encoding="utf-8") as f:
        train_content = f.read()

    collision_pattern = re.compile(
        r'<geom\s+name="([^"]*_collision)"([^/]*)/>',
        re.DOTALL
    )

    sync_keys = ["type", "size", "pos", "fromto", "mass", "friction"]
    synced = 0

    for match in collision_pattern.finditer(vis_content):
        geom_name = match.group(1)
        vis_geom_str = match.group(0)

        train_pattern = re.compile(
            rf'(<geom\s+name="{re.escape(geom_name)}"[^/]*/>)'
        )
        train_match = train_pattern.search(train_content)
        if not train_match:
            continue

        old_geom = train_match.group(1)
        new_geom = old_geom

        for key in sync_keys:
            vis_attr = re.search(rf'{key}="([^"]*)"', vis_geom_str)
            if vis_attr:
                attr_pattern = re.compile(rf'{key}="[^"]*"')
                if attr_pattern.search(new_geom):
                    new_geom = attr_pattern.sub(f'{key}="{vis_attr.group(1)}"', new_geom)

        if new_geom != old_geom:
            train_content = train_content.replace(old_geom, new_geom)
            synced += 1

    if synced > 0:
        with open(TRAINING_XML, "w", encoding="utf-8") as f:
            f.write(train_content)
    return synced


# --- HTTP サーバー ---

# ファイル変更追跡
_file_version = {"v": 0}

def get_file_mtime():
    try:
        return os.path.getmtime(VISUALIZE_XML)
    except:
        return 0

_last_mtime = get_file_mtime()


def file_watcher():
    """ファイル変更を監視し、バージョンカウンタを更新"""
    global _last_mtime
    while True:
        time.sleep(0.5)
        try:
            current = os.path.getmtime(VISUALIZE_XML)
            if current != _last_mtime:
                _last_mtime = current
                _file_version["v"] += 1
                time.sleep(0.1)
                count = sync_to_training_xml()
                print(f"  [Hot Reload] XML変更検知 (v{_file_version['v']}) — {count}個の当たり判定を同期")
        except Exception as e:
            pass


HTML_PAGE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>旋風丸 — 当たり判定チューナー</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { 
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1a1a2e; color: #e0e0e0; overflow: hidden;
}
#container { width: 100vw; height: 100vh; cursor: grab; }
#container:active { cursor: grabbing; }

#tab-header {
    position: fixed; top: 12px; left: 12px; z-index: 10;
    display: flex; gap: 6px;
}
.tab-btn {
    background: rgba(20, 20, 40, 0.9); border: 1px solid #444; color: #aaa;
    padding: 8px 14px; border-radius: 6px; font-size: 12px; cursor: pointer;
    backdrop-filter: blur(8px); transition: all 0.2s;
}
.tab-btn.active {
    background: rgba(40, 60, 100, 0.95); border-color: #7ecbff; color: #fff;
    font-weight: bold; box-shadow: 0 0 10px rgba(126, 203, 255, 0.2);
}

.side-panel {
    position: fixed; top: 52px; left: 12px;
    background: rgba(20, 20, 40, 0.92); border: 1px solid #333;
    border-radius: 8px; padding: 14px 18px; width: 340px;
    max-height: calc(100vh - 120px); overflow-y: auto;
    font-size: 13px; line-height: 1.6; z-index: 10;
    backdrop-filter: blur(8px); display: none;
}
.side-panel::-webkit-scrollbar { width: 6px; }
.side-panel::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
.side-panel.active { display: block; }

.side-panel h2 { 
    font-size: 15px; color: #7ecbff; margin-bottom: 8px;
    border-bottom: 1px solid #333; padding-bottom: 6px;
}
.side-panel .key { 
    display: inline-block; background: #333; border-radius: 3px;
    padding: 1px 6px; font-family: monospace; font-size: 12px;
    color: #fff; margin: 0 2px;
}

.mode-btn {
    display: block; width: 100%; padding: 8px; margin: 6px 0;
    background: #2a3a5e; border: 1px solid #4a6a9e; color: #fff;
    border-radius: 6px; font-size: 12px; cursor: pointer; text-align: center;
    transition: background 0.2s;
}
.mode-btn:hover { background: #3a4a7e; }
.mode-btn.active { background: #1a5a3e; border-color: #4afe9e; }

.preset-group {
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 8px 0;
}
.preset-btn {
    padding: 6px; background: #252538; border: 1px solid #444; color: #ccc;
    border-radius: 4px; font-size: 11px; cursor: pointer; text-align: center;
    transition: all 0.15s;
}
.preset-btn:hover { background: #353550; color: #fff; border-color: #7ecbff; }

.joint-group-title {
    font-size: 12px; font-weight: bold; color: #ffcb7e;
    margin: 10px 0 4px 0; padding-bottom: 2px; border-bottom: 1px solid #333;
}
.joint-row {
    display: flex; align-items: center; justify-content: space-between;
    margin: 4px 0; font-size: 11px;
}
.joint-label { width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.joint-slider { flex: 1; margin: 0 6px; }
.joint-val { width: 40px; text-align: right; font-family: monospace; color: #7ecbff; }

.legend { margin-top: 10px; }
.legend-item { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
.legend-color { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #555; }

#selected-info {
    position: fixed; bottom: 12px; left: 12px;
    background: rgba(20, 20, 40, 0.95); border: 1px solid #ffcb7e;
    border-radius: 8px; padding: 12px 16px; font-size: 13px;
    z-index: 10; backdrop-filter: blur(8px); min-width: 300px;
    display: none;
}
#selected-info h3 { color: #ffcb7e; font-size: 14px; margin-bottom: 6px; }
#selected-info .prop { color: #aaa; font-family: monospace; font-size: 12px; margin: 2px 0; }
#selected-info .prop span { color: #7ecbff; }

#status {
    position: fixed; bottom: 12px; right: 12px;
    background: rgba(20, 20, 40, 0.92); border: 1px solid #333;
    border-radius: 8px; padding: 8px 14px; font-size: 11px;
    z-index: 10; backdrop-filter: blur(8px);
}

#geom-list {
    position: fixed; top: 12px; right: 12px;
    background: rgba(20, 20, 40, 0.92); border: 1px solid #333;
    border-radius: 8px; padding: 14px 18px; max-width: 380px;
    max-height: calc(100vh - 80px); overflow-y: auto;
    font-size: 12px; z-index: 10; backdrop-filter: blur(8px);
}
#geom-list::-webkit-scrollbar { width: 6px; }
#geom-list::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
#geom-list h2 { font-size: 14px; color: #ffcb7e; margin-bottom: 8px; }
.geom-entry { 
    padding: 6px 8px; border-bottom: 1px solid #2a2a3e;
    cursor: pointer; transition: all 0.2s; border-radius: 4px; margin: 2px 0;
}
.geom-entry:hover { background: rgba(126, 203, 255, 0.1); }
.geom-entry.selected { 
    background: rgba(255, 203, 126, 0.18) !important;
    border: 1px solid rgba(255, 203, 126, 0.5);
}
.geom-entry .name { color: #7ecbff; font-weight: bold; }
.geom-entry.selected .name { color: #ffcb7e; }
.geom-entry .detail { color: #888; font-family: monospace; font-size: 11px; }
.geom-entry .part-label { font-size: 10px; color: #666; margin-top: 1px; font-style: italic; }
</style>
</head>
<body>
<div id="container"></div>

<div id="tab-header">
    <button class="tab-btn active" onclick="switchTab('info')">ℹ️ 情報 & 操作方法</button>
    <button class="tab-btn" onclick="switchTab('joint')">🎮 関節操作・ポーズ確認</button>
</div>

<div id="panel-info" class="side-panel active">
    <h2>🌪️ 旋風丸 — 当たり判定チューナー</h2>
    <div>
        <span class="key">ドラッグ</span> 回転 &nbsp;
        <span class="key">右ドラッグ</span> パン<br>
        <span class="key">スクロール</span> ズーム &nbsp;
        <span class="key">M</span> メッシュ表示切替<br>
        <span class="key">C</span> 当たり判定表示切替 &nbsp;
        <span class="key">W</span> ワイヤフレーム<br>
        <span class="key">クリック</span> 3Dまたはリストで選択 &nbsp;
        <span class="key">Esc</span> 選択解除
    </div>
    
    <button id="toggle-mesh-only" class="mode-btn" onclick="toggleMeshOnlyMode()">
        ✨ 当たり判定プリミティブのみ表示モード (切替)
    </button>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:rgba(255,80,80,0.6)"></div> 胴体 (box)</div>
        <div class="legend-item"><div class="legend-color" style="background:rgba(80,80,255,0.6)"></div> 腕 (capsule/sphere)</div>
        <div class="legend-item"><div class="legend-color" style="background:rgba(80,255,80,0.6)"></div> 脚 (capsule/sphere/box)</div>
        <div class="legend-item"><div class="legend-color" style="background:rgba(200,200,200,0.4)"></div> メッシュ (視覚のみ)</div>
        <div class="legend-item"><div class="legend-color" style="background:rgba(255,220,50,0.8)"></div> 選択中のプリミティブ</div>
    </div>
</div>

<div id="panel-joint" class="side-panel">
    <h2>🎮 関節インタラクティブ操作 (20 DOF)</h2>
    <p style="font-size:11px;color:#aaa;margin-bottom:8px;">
        全20関節を操作し、メッシュなし（当たり判定のみ）での動きと自己干渉を確認できます。
    </p>

    <div style="font-size:11px;font-weight:bold;color:#7ecbff;margin-bottom:4px;">ポーズプリセット</div>
    <div class="preset-group">
        <button class="preset-btn" onclick="applyPreset('default')">🧍 標準 (直立)</button>
        <button class="preset-btn" onclick="applyPreset('squat')">🦵 屈伸 (Squat)</button>
        <button class="preset-btn" onclick="applyPreset('walk')">🚶 一歩踏み出し</button>
        <button class="preset-btn" onclick="applyPreset('kick')">⚽ ハイキック</button>
    </div>

    <div id="joint-sliders-container">スライダー読み込み中...</div>
</div>

<div id="selected-info">
    <h3 id="sel-title">—</h3>
    <div class="prop">type: <span id="sel-type">—</span></div>
    <div class="prop">size: <span id="sel-size">—</span></div>
    <div class="prop">pos:  <span id="sel-pos">—</span></div>
    <div class="prop">fromto: <span id="sel-fromto">—</span></div>
    <div class="prop">mass: <span id="sel-mass">—</span></div>
</div>

<div id="status">
    <span id="status-text">📡 ファイル監視中...</span>
</div>

<div id="geom-list">
    <h2>📐 当たり判定一覧 <small style="color:#888;font-weight:normal">(クリックで選択)</small></h2>
    <div id="geom-entries">読み込み中...</div>
</div>

<script type="importmap">
{
    "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"
    }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// --- Scene Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.001, 100);
camera.position.set(0.5, 0.35, 0.5);
camera.lookAt(0, 0.2, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
document.getElementById('container').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.2, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Lighting
const ambientLight = new THREE.AmbientLight(0x606080, 1.5);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
dirLight.position.set(1, 3, 2);
dirLight.castShadow = true;
scene.add(dirLight);
const fillLight = new THREE.DirectionalLight(0x8888ff, 0.5);
fillLight.position.set(-1, 1, -1);
scene.add(fillLight);

// Ground
const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshStandardMaterial({ color: 0x333344, roughness: 0.8, metalness: 0.1 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
scene.add(new THREE.GridHelper(2, 40, 0x444466, 0x2a2a3e));

// --- Root Transform: MuJoCo Z-up → Three.js Y-up ---
// MuJoCo: X-right, Y-forward, Z-up
// Three.js: X-right, Y-up, Z-toward-viewer
// 全body/geomはMuJoCoネイティブ座標のままにし、ルートで一括回転する
const rootTransform = new THREE.Group();
rootTransform.rotation.x = -Math.PI / 2;  // Z-up → Y-up
scene.add(rootTransform);

// --- Groups (rootTransformの子として配置) ---
const meshGroup = new THREE.Group();
const collisionGroup = new THREE.Group();
rootTransform.add(meshGroup);
rootTransform.add(collisionGroup);

let showMeshes = true;
let showCollisions = true;
let wireframeMode = false;

// --- Joint FK State ---
const jointMeshContainers = {};  // jointName -> THREE.Group
const jointColContainers = {};   // jointName -> THREE.Group
const jointDataMap = {};         // jointName -> { axis, pos, baseEuler, range, minDeg, maxDeg }
const jointAngles = {};          // jointName -> current angle in degrees

const JOINT_GROUPS = [
    { title: '🦵 右脚 (Right Leg)', joints: ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll'] },
    { title: '🦵 左脚 (Left Leg)', joints: ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll'] },
    { title: '💪 右腕 (Right Arm)', joints: ['right_shoulder_roll', 'right_shoulder_pitch', 'right_elbow', 'right_wrist_pitch'] },
    { title: '💪 左腕 (Left Arm)', joints: ['left_shoulder_roll', 'left_shoulder_pitch', 'left_elbow', 'left_wrist_pitch'] }
];

const PRESETS = {
    'default': {},
    'squat': {
        'right_hip_pitch': 45, 'right_knee': -90, 'right_ankle_pitch': 45,
        'left_hip_pitch': 45, 'left_knee': -90, 'left_ankle_pitch': 45,
        'right_shoulder_pitch': -20, 'left_shoulder_pitch': -20
    },
    'walk': {
        'right_hip_pitch': 30, 'right_knee': -40, 'right_ankle_pitch': 15,
        'left_hip_pitch': -20, 'left_knee': -10, 'left_ankle_pitch': -10,
        'right_shoulder_pitch': -30, 'left_shoulder_pitch': 30
    },
    'kick': {
        'right_hip_pitch': 70, 'right_knee': -20, 'right_ankle_pitch': 20,
        'left_hip_pitch': -10, 'left_knee': -30, 'left_ankle_pitch': 20,
        'right_shoulder_roll': 40, 'left_shoulder_roll': -40
    }
};

window.switchTab = (tabName) => {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.side-panel').forEach(p => p.classList.remove('active'));
    
    if (tabName === 'info') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('panel-info').classList.add('active');
    } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('panel-joint').classList.add('active');
    }
};

window.toggleMeshOnlyMode = () => {
    showMeshes = !showMeshes;
    meshGroup.visible = showMeshes;
    const btn = document.getElementById('toggle-mesh-only');
    if (btn) {
        if (!showMeshes) {
            btn.classList.add('active');
            btn.textContent = '✨ 当たり判定プリミティブのみ表示中 (メッシュ非表示)';
        } else {
            btn.classList.remove('active');
            btn.textContent = '✨ 当たり判定プリミティブのみ表示モード (切替)';
        }
    }
};

window.updateJointAngle = (jName, valDeg) => {
    const deg = parseFloat(valDeg);
    jointAngles[jName] = deg;
    const valEl = document.getElementById('jval-' + jName);
    if (valEl) valEl.textContent = deg.toFixed(0) + '°';
    
    applyJointRotation(jName);
};

window.applyPreset = (presetName) => {
    const pose = PRESETS[presetName] || {};
    for (const jName in jointDataMap) {
        const targetDeg = pose[jName] || 0;
        jointAngles[jName] = targetDeg;
        const sliderEl = document.getElementById('jslider-' + jName);
        const valEl = document.getElementById('jval-' + jName);
        if (sliderEl) sliderEl.value = targetDeg;
        if (valEl) valEl.textContent = targetDeg.toFixed(0) + '°';
        applyJointRotation(jName);
    }
};

function applyJointRotation(jName) {
    const info = jointDataMap[jName];
    const mContainer = jointMeshContainers[jName];
    const cContainer = jointColContainers[jName];
    if (!info || !mContainer || !cContainer) return;
    
    const deg = jointAngles[jName] || 0;
    const rad = deg * (Math.PI / 180.0);
    
    const baseQuat = new THREE.Quaternion().setFromEuler(info.baseEuler);
    const basePos = info.basePos;
    const jointPos = info.jointPos;
    
    const hingeQuat = new THREE.Quaternion().setFromAxisAngle(info.axis, rad);
    const totalQuat = baseQuat.clone().multiply(hingeQuat);
    
    // Joint pivot rotation offset: P_total = basePos + baseQuat * (jointPos - hingeQuat * jointPos)
    const rotatedJointPos = jointPos.clone().applyQuaternion(hingeQuat);
    const offset = jointPos.clone().sub(rotatedJointPos).applyQuaternion(baseQuat);
    const totalPos = basePos.clone().add(offset);
    
    mContainer.quaternion.copy(totalQuat);
    cContainer.quaternion.copy(totalQuat);
    mContainer.position.copy(totalPos);
    cContainer.position.copy(totalPos);
}

function buildJointSlidersUI() {
    const container = document.getElementById('joint-sliders-container');
    if (!container) return;
    
    let html = '';
    for (const group of JOINT_GROUPS) {
        html += `<div class="joint-group-title">${group.title}</div>`;
        for (const jName of group.joints) {
            const jData = jointDataMap[jName];
            if (!jData) continue;
            
            const minD = Math.round(jData.minDeg);
            const maxD = Math.round(jData.maxDeg);
            const currD = Math.round(jointAngles[jName] || 0);
            
            html += `<div class="joint-row">
                <span class="joint-label" title="${jName}">${jName}</span>
                <input type="range" class="joint-slider" id="jslider-${jName}" 
                    min="${minD}" max="${maxD}" value="${currD}" step="1" 
                    oninput="window.updateJointAngle('${jName}', this.value)">
                <span class="joint-val" id="jval-${jName}">${currD}°</span>
            </div>`;
        }
    }
    container.innerHTML = html || '<div style="color:#888">関節データがありません</div>';
}
const collisionMeshMap = {};  // geom_name -> THREE.Mesh
const collisionDataMap = {};  // geom_name -> geom data object
let selectedName = null;
let selectedOriginalColor = null;
let selectedOriginalOpacity = null;
const HIGHLIGHT_COLOR = new THREE.Color(1.0, 0.86, 0.2);  // gold
const HIGHLIGHT_OPACITY = 0.85;

// --- Part name mapping ---
function getPartLabel(name) {
    if (name.includes('ashiura')) return '足裏';
    if (name.includes('momo')) return '太腿';
    if (name.includes('hizabu') && !name.includes('aikabu') && !name.includes('gaikabu')) return '膝';
    if (name.includes('aikabu') || name.includes('gaikabu')) return '脛';
    if (name.includes('kokansetu')) return '股関節';
    if (name.includes('dairou') || name.includes('daitou')) return '大腿根';
    if (name.includes('doutai') && !name.includes('kata')) return '胴体';
    if (name.includes('kata') && !name.includes('jouwan')) return '肩';
    if (name.includes('jouwan')) return '上腕';
    if (name.includes('hiji') && !name.includes('te_')) return '肘〜前腕';
    if (name.includes('te_') || name.endsWith('te')) return '手';
    return '';
}

function getLR(name) {
    if (name.includes('hidari')) return '左';
    if (name.includes('migi')) return '右';
    return '';
}

// --- Highlight / Unhighlight ---
function highlightGeom(geomName) {
    // Unhighlight previous
    unhighlightGeom();

    const mesh = collisionMeshMap[geomName];
    if (!mesh) return;

    selectedName = geomName;
    selectedOriginalColor = mesh.material.color.clone();
    selectedOriginalOpacity = mesh.material.opacity;

    mesh.material.color.copy(HIGHLIGHT_COLOR);
    mesh.material.opacity = HIGHLIGHT_OPACITY;
    mesh.material.emissive = HIGHLIGHT_COLOR.clone().multiplyScalar(0.3);
    mesh.material.emissiveIntensity = 1.0;
    mesh.material.needsUpdate = true;

    // Scale pulse start
    mesh.userData._pulsePhase = 0;
    mesh.userData._pulsing = true;
    mesh.userData._baseScale = mesh.scale.clone();

    // Highlight list entry
    document.querySelectorAll('.geom-entry').forEach(el => el.classList.remove('selected'));
    const listEl = document.getElementById('ge-' + geomName);
    if (listEl) {
        listEl.classList.add('selected');
        listEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Show detail panel
    const data = collisionDataMap[geomName];
    if (data) {
        const panel = document.getElementById('selected-info');
        const lr = getLR(data.name);
        const part = getPartLabel(data.name);
        document.getElementById('sel-title').textContent = 
            `${lr ? '[' + lr + '] ' : ''}${part || geomName}`;
        document.getElementById('sel-type').textContent = data.type;
        document.getElementById('sel-size').textContent = data.size;
        document.getElementById('sel-pos').textContent = data.pos;
        document.getElementById('sel-fromto').textContent = data.fromto || '—';
        document.getElementById('sel-mass').textContent = data.mass || '—';
        panel.style.display = 'block';
    }
}

function unhighlightGeom() {
    if (selectedName && collisionMeshMap[selectedName]) {
        const mesh = collisionMeshMap[selectedName];
        mesh.material.color.copy(selectedOriginalColor);
        mesh.material.opacity = selectedOriginalOpacity;
        mesh.material.emissive = new THREE.Color(0, 0, 0);
        mesh.material.emissiveIntensity = 0;
        mesh.material.needsUpdate = true;
        mesh.userData._pulsing = false;
        if (mesh.userData._baseScale) {
            mesh.scale.copy(mesh.userData._baseScale);
        }
    }
    selectedName = null;
    document.querySelectorAll('.geom-entry').forEach(el => el.classList.remove('selected'));
    document.getElementById('selected-info').style.display = 'none';
}

// --- Keyboard ---
document.addEventListener('keydown', (e) => {
    if (e.key === 'm' || e.key === 'M') {
        showMeshes = !showMeshes;
        meshGroup.visible = showMeshes;
    } else if (e.key === 'c' || e.key === 'C') {
        showCollisions = !showCollisions;
        collisionGroup.visible = showCollisions;
    } else if (e.key === 'w' || e.key === 'W') {
        wireframeMode = !wireframeMode;
        collisionGroup.traverse(child => {
            if (child.isMesh && child.material) child.material.wireframe = wireframeMode;
        });
    } else if (e.key === 'Escape') {
        unhighlightGeom();
    }
});

// --- Raycaster for 3D click ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let isDragging = false;
let mouseDownPos = { x: 0, y: 0 };

renderer.domElement.addEventListener('mousedown', (e) => {
    mouseDownPos = { x: e.clientX, y: e.clientY };
    isDragging = false;
});

renderer.domElement.addEventListener('mousemove', (e) => {
    const dx = e.clientX - mouseDownPos.x;
    const dy = e.clientY - mouseDownPos.y;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) isDragging = true;
});

renderer.domElement.addEventListener('mouseup', (e) => {
    if (isDragging) return;  // Ignore drag
    
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    // Collect all collision meshes
    const collisionMeshes = [];
    collisionGroup.traverse(child => {
        if (child.isMesh) collisionMeshes.push(child);
    });
    
    const intersects = raycaster.intersectObjects(collisionMeshes, false);
    
    if (intersects.length > 0) {
        const hitMesh = intersects[0].object;
        const hitName = hitMesh.userData.geomName;
        if (hitName) {
            highlightGeom(hitName);
        }
    } else {
        unhighlightGeom();
    }
});

// --- Build Scene from Body Hierarchy ---
function buildScene(bodies) {
    // Clear existing
    while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0]);
    while (collisionGroup.children.length) collisionGroup.remove(collisionGroup.children[0]);
    
    // Clear maps
    for (const k in collisionMeshMap) delete collisionMeshMap[k];
    for (const k in collisionDataMap) delete collisionDataMap[k];

    const stlLoader = new STLLoader();
    const geomListEl = document.getElementById('geom-entries');
    let geomListHTML = '';
    let geomIndex = 0;

    function processBody(bodyData, parentGroup_mesh, parentGroup_col) {
        const pos = bodyData.pos.split(' ').map(Number);
        const euler = bodyData.euler.split(' ').map(Number);

        const meshContainer = new THREE.Group();
        const colContainer = new THREE.Group();
        // MuJoCoネイティブ座標をそのまま使用（ルートで一括回転済み）
        meshContainer.position.set(pos[0], pos[1], pos[2]);
        colContainer.position.set(pos[0], pos[1], pos[2]);

        // MuJoCoのeuler属性はXYZ intrinsic回転
        const eulerObj = new THREE.Euler(euler[0], euler[1], euler[2], 'XYZ');
        meshContainer.setRotationFromEuler(eulerObj);
        colContainer.setRotationFromEuler(eulerObj);

        if (bodyData.joint) {
            const j = bodyData.joint;
            const axisVec = j.axis.split(' ').map(Number);
            const rangeVals = j.range.split(' ').map(Number);
            const jPosVec = j.pos.split(' ').map(Number);
            jointMeshContainers[j.name] = meshContainer;
            jointColContainers[j.name] = colContainer;
            jointDataMap[j.name] = {
                axis: new THREE.Vector3(axisVec[0], axisVec[1], axisVec[2]).normalize(),
                basePos: new THREE.Vector3(pos[0], pos[1], pos[2]),
                jointPos: new THREE.Vector3(jPosVec[0], jPosVec[1], jPosVec[2]),
                baseEuler: eulerObj.clone(),
                minDeg: rangeVals[0] * (180.0 / Math.PI),
                maxDeg: rangeVals[1] * (180.0 / Math.PI)
            };
        }

        parentGroup_mesh.add(meshContainer);
        parentGroup_col.add(colContainer);

        for (const geom of bodyData.geoms) {
            if (geom.type === 'mesh' && geom.mesh_name) {
                const meshUrl = '/mesh/' + geom.mesh_name + '.stl';
                const gPos = geom.pos.split(' ').map(Number);

                stlLoader.load(meshUrl, (geometry) => {
                    geometry.computeVertexNormals();
                    geometry.scale(0.001, 0.001, 0.001);

                    const rgba = geom.rgba.split(' ').map(Number);
                    const mat = new THREE.MeshStandardMaterial({
                        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
                        transparent: true,
                        opacity: rgba[3] * 0.7,
                        roughness: 0.6,
                        metalness: 0.2,
                        side: THREE.DoubleSide,
                    });
                    const mesh = new THREE.Mesh(geometry, mat);
                    mesh.position.set(gPos[0], gPos[1], gPos[2]);
                    meshContainer.add(mesh);
                }, undefined, () => {});

            } else if (geom.is_collision) {
                const rgba = geom.rgba.split(' ').map(Number);
                const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
                const opacity = Math.max(rgba[3], 0.25);

                const mat = new THREE.MeshStandardMaterial({
                    color: color,
                    transparent: true,
                    opacity: opacity,
                    roughness: 0.5,
                    wireframe: wireframeMode,
                    side: THREE.DoubleSide,
                });

                let mesh;
                const sizes = geom.size.split(' ').map(Number);
                const gPos = geom.pos.split(' ').map(Number);

                if (geom.type === 'box') {
                    // MuJoCo box size = half-extents (x, y, z)
                    const geo = new THREE.BoxGeometry(sizes[0]*2, sizes[1]*2, sizes[2]*2);
                    mesh = new THREE.Mesh(geo, mat);
                    mesh.position.set(gPos[0], gPos[1], gPos[2]);

                } else if (geom.type === 'sphere') {
                    const geo = new THREE.SphereGeometry(sizes[0], 16, 12);
                    mesh = new THREE.Mesh(geo, mat);
                    mesh.position.set(gPos[0], gPos[1], gPos[2]);

                } else if (geom.type === 'capsule') {
                    if (geom.fromto) {
                        const ft = geom.fromto.split(' ').map(Number);
                        // MuJoCoネイティブ座標のまま
                        const p1 = new THREE.Vector3(ft[0], ft[1], ft[2]);
                        const p2 = new THREE.Vector3(ft[3], ft[4], ft[5]);
                        const length = p1.distanceTo(p2);
                        const radius = sizes[0];

                        const geo = new THREE.CapsuleGeometry(radius, length, 8, 16);
                        mesh = new THREE.Mesh(geo, mat);

                        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
                        mesh.position.copy(mid);

                        // CapsuleGeometry はY軸方向がデフォルト長軸
                        const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
                        const up = new THREE.Vector3(0, 1, 0);
                        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
                        mesh.quaternion.copy(quat);
                    } else {
                        const geo = new THREE.CapsuleGeometry(sizes[0], sizes[1]*2, 8, 16);
                        mesh = new THREE.Mesh(geo, mat);
                        mesh.position.set(gPos[0], gPos[1], gPos[2]);
                    }
                }

                if (mesh) {
                    // Tag mesh with geom name for raycaster identification
                    mesh.userData.geomName = geom.name;
                    
                    // Edge outline
                    const edgeMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 });
                    const edges = new THREE.EdgesGeometry(mesh.geometry);
                    const edgeLine = new THREE.LineSegments(edges, edgeMat);
                    mesh.add(edgeLine);

                    colContainer.add(mesh);
                    
                    // Register in map
                    collisionMeshMap[geom.name] = mesh;
                    collisionDataMap[geom.name] = geom;
                }

                // Build geom list HTML
                const typeEmoji = geom.type === 'box' ? '📦' : geom.type === 'sphere' ? '🔵' : '💊';
                const partName = geom.name.replace('_collision', '').split('_').slice(-2).join('_');
                const lr = getLR(geom.name);
                const part = getPartLabel(geom.name);
                const idx = geomIndex++;
                
                geomListHTML += `<div class="geom-entry" id="ge-${geom.name}" 
                    data-geom="${geom.name}" onclick="window._selectGeom('${geom.name}')">
                    <div class="name">${typeEmoji} ${lr ? '[' + lr + '] ' : ''}${partName}</div>
                    <div class="detail">type=${geom.type} size="${geom.size}"</div>
                    <div class="part-label">${part}</div>
                </div>`;
            }
        }

        for (const child of bodyData.children) {
            processBody(child, meshContainer, colContainer);
        }
    }

    for (const body of bodies) {
        processBody(body, meshGroup, collisionGroup);
    }

    geomListEl.innerHTML = geomListHTML || '<div style="color:#888">当たり判定なし</div>';
    
    // Build 20-joint interactive sliders UI
    buildJointSlidersUI();
    
    // Re-highlight if was selected
    if (selectedName && collisionMeshMap[selectedName]) {
        highlightGeom(selectedName);
    }
}

// Expose selection function to onclick handlers
window._selectGeom = (name) => {
    if (selectedName === name) {
        unhighlightGeom();
    } else {
        highlightGeom(name);
    }
};

// --- Load & Poll ---
let currentVersion = -1;

async function checkForUpdates() {
    try {
        const res = await fetch('/api/version');
        const data = await res.json();
        if (data.version !== currentVersion) {
            currentVersion = data.version;
            const bodiesRes = await fetch('/api/bodies');
            const bodies = await bodiesRes.json();
            buildScene(bodies);
            document.getElementById('status-text').textContent = 
                `✅ v${currentVersion} — ${new Date().toLocaleTimeString()}`;
        }
    } catch (e) {
        document.getElementById('status-text').textContent = '❌ 接続エラー';
    }
}

setInterval(checkForUpdates, 800);
checkForUpdates();

// --- Render Loop with Pulse Animation ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    // Pulse animation for selected mesh
    if (selectedName && collisionMeshMap[selectedName]) {
        const mesh = collisionMeshMap[selectedName];
        if (mesh.userData._pulsing && mesh.userData._baseScale) {
            mesh.userData._pulsePhase = (mesh.userData._pulsePhase || 0) + 0.05;
            const pulse = 1.0 + 0.15 * Math.sin(mesh.userData._pulsePhase);
            mesh.scale.copy(mesh.userData._baseScale).multiplyScalar(pulse);
            
            // Emissive pulse
            const emIntensity = 0.3 + 0.2 * Math.sin(mesh.userData._pulsePhase * 0.7);
            mesh.material.emissiveIntensity = emIntensity;
        }
    }
    
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""


class TunerHandler(http.server.BaseHTTPRequestHandler):
    """APIエンドポイントとHTMLを提供するハンドラ"""

    def log_message(self, format, *args):
        pass  # ログ出力を抑制

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/api/version":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"version": _file_version["v"]}).encode())

        elif self.path == "/api/bodies":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                bodies = parse_bodies_recursive(str(VISUALIZE_XML))
                self.wfile.write(json.dumps(bodies).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path.startswith("/mesh/"):
            # STLファイル配信
            filename = self.path[6:]  # /mesh/ を除去
            filepath = MESHES_DIR / filename
            if filepath.exists() and filepath.suffix.lower() == ".stl":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      旋風丸 — 当たり判定リアルタイムチューニングツール       ║")
    print("║              (ブラウザベース 3Dビューア)                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                        ║")
    print("║  ブラウザで当たり判定を3D表示し、XMLを編集 → 保存       ║")
    print("║  するだけで即座に反映されます。                         ║")
    print("║                                                        ║")
    print("║  操作:                                                  ║")
    print("║    ドラッグ   : 回転    右ドラッグ : パン               ║")
    print("║    スクロール : ズーム                                  ║")
    print("║    M         : メッシュ表示切替                        ║")
    print("║    C         : 当たり判定表示切替                      ║")
    print("║    W         : ワイヤフレーム                          ║")
    print("║                                                        ║")
    print("║  終了: Ctrl+C                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if not VISUALIZE_XML.exists():
        print(f"[Error] ファイルが見つかりません: {VISUALIZE_XML}")
        sys.exit(1)

    # 初回パース確認
    try:
        bodies = parse_bodies_recursive(str(VISUALIZE_XML))
        total_collision = sum(
            1 for b in json.loads(json.dumps(bodies))
            for g in (b.get("geoms", []))
            if g.get("is_collision")
        )
        # 再帰的にカウント
        def count_collisions(body_list):
            n = 0
            for b in body_list:
                n += sum(1 for g in b.get("geoms", []) if g.get("is_collision"))
                n += count_collisions(b.get("children", []))
            return n
        total_collision = count_collisions(bodies)
        print(f"  [OK] XML読み込み完了 — 当たり判定プリミティブ: {total_collision}個")
    except Exception as e:
        print(f"  [Error] XML解析失敗: {e}")
        sys.exit(1)

    # ファイル監視スレッド
    watcher = threading.Thread(target=file_watcher, daemon=True)
    watcher.start()

    # HTTPサーバー起動
    url = f"http://localhost:{PORT}"
    print(f"  [Server] {url} でサーバー起動中...")
    print(f"  [Tip] VS Code で humanoid_visualize.xml を編集 → Ctrl+S で保存")
    print()

    with socketserver.TCPServer(("", PORT), TunerHandler) as httpd:
        httpd.allow_reuse_address = True
        # ブラウザを自動で開く
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  [Exit] 終了中...")

    # 最終同期
    print("  [Final Sync] humanoid.xml に最終同期中...")
    count = sync_to_training_xml()
    print(f"  [Done] {count}個の当たり判定を同期しました ✓")


if __name__ == "__main__":
    main()
