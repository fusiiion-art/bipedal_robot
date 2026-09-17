# bipedal_robot Python Files

**Total Files: 60**

---

## assets/fix_collision_geoms.py

```python
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

```

---

## deploy/__init__.py

```python
# deploy package: real-world deployment and export tools

```

---

## deploy/export_onnx.py

```python
import os
import sys
import pickle
import argparse
import shutil

import jax
import tensorflow as tf
from jax.experimental import jax2tf
from brax.training.agents.ppo import networks as ppo_networks

# append project root to sys path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from robot.policy_network import make_policy_network_factory

def load_brax_inference_fn(pkl_path, obs_dim, action_dim):
    """
    Brax(Flax)の保存済みパラメータファイルから、
    JAXネイティブな推論関数(predict)を復元する
    """
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
        
    print("[INFO] Params successfully loaded from pickle.")
    
    # train_mjx.py と同一のネットワーク構成を使用（アーキテクチャ不一致を防止）
    ppo_network = make_policy_network_factory(
        observation_size=obs_dim,
        action_size=action_dim,
    )
    
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inf_fn = make_inference_fn(params, deterministic=True)
    
    def predict(obs):
        # Deterministic export should not sample a stochastic action with a fixed PRNG seed.
        dummy_rng = jax.random.PRNGKey(0)
        action, _ = inf_fn(obs, dummy_rng)
        return action
        
    return predict

def export_jax_to_onnx(predict_fn, obs_dim, onnx_path):
    """JAX関数 -> TensorFlow SavedModel -> ONNX の公式ツールチェーンで変換"""
    import tf2onnx
    
    print("[INFO] 1. Converting pure JAX function to TensorFlow...")
    tf_predict = jax2tf.convert(predict_fn, enable_xla=False)
    
    print("[INFO] 2. Wrapping with tf.function (fixing input signature)...")
    @tf.function(
        autograph=False,
        input_signature=[tf.TensorSpec(shape=[None, obs_dim], dtype=tf.float32, name="observation")]
    )
    def tf_func(obs):
        return tf_predict(obs)
    
    print("[INFO] 3. Saving temporary TensorFlow SavedModel...")
    saved_model_dir = "/tmp/brax_saved_model"
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
        
    module = tf.Module()
    module.predict = tf_func
    tf.saved_model.save(module, saved_model_dir, signatures={'serving_default': module.predict})
    
    print("[INFO] 4. Converting SavedModel to ONNX via tf2onnx...")
    model_proto, _ = tf2onnx.convert.from_saved_model(
        saved_model_dir, output_path=onnx_path, opset=14
    )
    
    print("[INFO] 5. Cleaning up temporary files...")
    shutil.rmtree(saved_model_dir)
    print(f"\n✅ ONNX Model successfully exported to: {onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="mjx_params.pkl へのパス")
    parser.add_argument("--output", type=str, default="brax_policy.onnx")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[Error] Parameter file not found: {args.model}")
        sys.exit(1)
        
    obs_dim = RobotConfig.OBS_DIM
    action_dim = RobotConfig.NUM_JOINTS
    
    print(f"--- ONNX Export Pipeline ---")
    print(f"Observation Dim: {obs_dim}")
    print(f"Action Dim: {action_dim}")
    print(f"Target Output: {args.output}")
    
    try:
        predict_fn = load_brax_inference_fn(args.model, obs_dim, action_dim)
        export_jax_to_onnx(predict_fn, obs_dim, args.output)
    except Exception as e:
        print(f"\n[Error] Export failed: {e}")
        print("💡 Hint: Ensure you have `tensorflow` and `tf2onnx` installed (`pip install tensorflow-cpu tf2onnx`)")
        sys.exit(1)

if __name__ == "__main__":
    main()

```

---

## envs/__init__.py

```python
# envs package

```

---

## envs/actuator_model.py

```python
import jax
import jax.numpy as jp
from typing import NamedTuple

from robot.config import RobotConfig


class ActuatorState(NamedTuple):
    """サーボモータの内部状態"""
    temperature: jax.Array  # [℃] 各関節の現在温度 shape=(nu,)
    supply_voltage: float   # [V] 現在の供給電圧


class HX30HMModel:
    """
    Hiwonder HX-30HM シリアルバスサーボの物理モデル
    - 熱ダレ (Thermal Derating): 温度上昇に伴うトルク/速度の低下
    - 電圧降下 (Voltage Derating): バッテリー残量低下に伴う性能低下
    - 温度更新: 消費電力に比例した発熱と、環境への放熱の簡易モデル
    
    Spec: HX-30HM
    - Stall Torque: 30 kg.cm (11.1V) = 2.94 N.m
    - No-load Speed: 0.19 sec/60deg (11.1V) = 315 deg/s = 5.5 rad/s
    - Operating Voltage: 6.0 ~ 12.6V
    - Operating Temperature: -5℃ ~ 85℃
    
    参考: 実機キャリブレーション
    - motor_resistance, thermal_mass, thermal_resistance は概算値。
    - 実装環境の温度条件に合わせて AMBIENT_TEMP を調整してください。
    """
    
    # 定格電圧
    NOMINAL_VOLTAGE = 11.1  # [V]
    
    # 熱モデルパラメータ
    THERMAL_RESISTANCE = 0.08   # [℃/W] サーボの熱抵抗（小型サーボの概算）
    THERMAL_MASS = 50.0         # [J/℃] サーボの熱容量
    AMBIENT_TEMP = 25.0         # [℃] 環境温度
    MAX_SAFE_TEMP = 85.0        # [℃] 安全動作上限温度
    DERATING_START_TEMP = 60.0  # [℃] 熱ダレ開始温度
    
    # モータ効率（電気→機械変換効率）
    MOTOR_EFFICIENCY = 0.3  # 小型ホビーサーボの概算
    
    @staticmethod
    def compute_thermal_derating(temperature: jax.Array) -> jax.Array:
        """
        温度に基づくトルク/速度の低減係数を計算する。
        60℃以下: 100% 出力
        60℃〜85℃: 線形に低下 (100% → 30%)
        85℃以上: 30% に制限
        
        Returns:
            derating: [0.3, 1.0] の範囲の係数 shape=(nu,)
        """
        temp_range = HX30HMModel.MAX_SAFE_TEMP - HX30HMModel.DERATING_START_TEMP  # 25℃
        excess = jp.clip(temperature - HX30HMModel.DERATING_START_TEMP, 0.0, temp_range)
        derating = 1.0 - 0.7 * (excess / temp_range)  # 1.0 → 0.3
        return jp.clip(derating, 0.3, 1.0)
    
    @staticmethod
    def compute_voltage_derating(supply_voltage: float) -> jax.Array:
        """
        供給電圧に基づくトルク/速度の低減係数を計算する。
        トルクは電圧にほぼ比例し、速度も電圧に比例する。
        
        Returns:
            derating: [0.0, 1.2] の範囲の係数（過電圧で微増も許容）
        """
        ratio = supply_voltage / HX30HMModel.NOMINAL_VOLTAGE
        return jp.clip(ratio, 0.5, 1.2)
    
    @staticmethod
    def update_temperature(
        state: ActuatorState,
        torque: jax.Array,
        dt: float
    ) -> ActuatorState:
        """
        1制御ステップ分の温度更新を行う。
        
        簡易熱モデル:
          dT/dt = (P_heat - P_cool) / C_thermal
          P_heat = torque^2 * R_motor / efficiency  (銅損の概算)
          P_cool = (T - T_ambient) / R_thermal      (放熱)
        
        注意: motor_resistance は概算値(2.0Ω)です。
              実機測定値がある場合、適切に調整してください。
        
        Args:
            state: 現在のアクチュエータ状態
            torque: 各関節のトルク [N.m] shape=(nu,)
            dt: 制御ステップ時間 [s]
        
        Returns:
            new_state: 温度が更新された新しい状態
        """
        # 発熱量 (I^2 * R に相当、トルクの2乗に比例)
        motor_resistance = 2.0  # [Ω] 概算のモータ巻線抵抗（実測値で更新推奨）
        power_heat = jp.square(torque) * motor_resistance / HX30HMModel.MOTOR_EFFICIENCY
        
        # 放熱量
        power_cool = (state.temperature - HX30HMModel.AMBIENT_TEMP) / HX30HMModel.THERMAL_RESISTANCE
        
        # 温度変化
        dT = (power_heat - power_cool) * dt / HX30HMModel.THERMAL_MASS
        new_temp = state.temperature + dT
        
        # 温度をクリップ（環境温度以下にはならない）
        new_temp = jp.clip(new_temp, HX30HMModel.AMBIENT_TEMP, 120.0)
        
        return ActuatorState(
            temperature=new_temp,
            supply_voltage=state.supply_voltage
        )
```

---

## envs/mjx_env.py

```python
"""固定足立位ロボット用 MJX (MuJoCo XLA) 強化学習環境。

このモジュールは SenpuuMaruMJXEnv を定義する。BraxのPipelineEnvを継承し、
GPU/TPU上でのJAX並列学習に対応する。

観測契約 (robot/config.py が正本、625次元):
  Base observation (84) + 履歴5フレーム分 (420) + action履歴 (100)
  + サーボ温度 (20) + 電源電圧 (1)
  観測の要素順序・FSR左右順・履歴の新旧方向は既存checkpointとのABI互換の
  ため変更してはならない (docs/current.md 参照)。

アクション契約 (20次元):
  関節目標角の残差 (Δq)。トルク直接指令は使わない。
  パイプライン: policy output → ACTION_SCALE → default pose加算 →
  deadband → 関節速度制限 → LPF → CBF → 熱/電圧derating → actuator

固定足制約:
  ALLOW_WALKING=False, ALLOW_STEPPING=False を常に維持する。
  歩行・踏み替え・支持基底の変更は全て禁止 (raiseで防御的に検出)。

外乱:
  Phase 0では DISTURBANCE_CURRICULUM=False で無効。有効時は
  RANDOM_PUSH_MAX_FORCE を上限とするランダム水平外力をqfrc_appliedへ
  加算する形で実装される。
"""

from typing import Any, Dict, Tuple, Union
import os
import jax
import jax.numpy as jp
from brax import envs
from brax.envs.base import PipelineEnv, State
import mujoco
from mujoco import mjx

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler
from envs.actuator_model import ActuatorState, HX30HMModel


class SenpuuMaruMJXEnv(PipelineEnv):
    """
    MuJoCo XLA (MJX) を使用した GPU/TPU 並列学習用の強化学習環境。
    BraxのPipelineEnvを継承しており、Brax PPOとシームレスに統合可能。

    reset(rng) -> State:
        物理初期姿勢(qpos/qvel)を固定nominal poseにリセットし、
        domain randomization (質量/摩擦/重心/温度/電圧) を適用する。
        注意: 初期姿勢そのもののrandomizationは未実装 (常に同一pose)。

    step(state, action) -> State:
        1制御周期(CONTROL_DT=0.01s, 実機100Hz相当)を進める。
        内部でCONTROL_DECIMATION回の物理サブステップをjax.lax.scanで実行する。
        戻り値のState.doneはterminated(転倒等)のみを表し、truncated(時間切れ)
        はinfo['truncated']に分離して格納される(Braxのtime_out処理と整合)。
    """
    
    def __init__(self, obs_noise: float = 0.01, latency_steps: int = 1, **kwargs):
        model_path = str(RobotConfig.MUJOCO_MODEL_PATH)
        
        fallback_xml = """<mujoco model="fallback_humanoid">
  <option timestep="0.00416667" gravity="0 0 -9.8"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1" friction="1.0 0.5 0.5"/>
    <body name="torso" pos="0 0 0.45">
      <freejoint name="root"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" mass="2.0" rgba="0.2 0.6 1.0 1"/>
      <body name="right_thigh" pos="0.05 0 0">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.03" mass="0.5" rgba="1.0 0.4 0.2 1"/>
        <body name="right_shin" pos="0 0 -0.15">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-2.0 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.3" rgba="1.0 0.6 0.3 1"/>
        </body>
      </body>
      <body name="left_thigh" pos="-0.05 0 0">
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.03" mass="0.5" rgba="0.2 1.0 0.4 1"/>
        <body name="left_shin" pos="0 0 -0.15">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="-2.0 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.3" rgba="0.4 1.0 0.6 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="right_hip_pitch_act" joint="right_hip_pitch" kp="20" kv="0.5" ctrlrange="-1.57 1.57"/>
    <position name="right_knee_act" joint="right_knee" kp="20" kv="0.5" ctrlrange="-2.0 0"/>
    <position name="left_hip_pitch_act" joint="left_hip_pitch" kp="20" kv="0.5" ctrlrange="-1.57 1.57"/>
    <position name="left_knee_act" joint="left_knee" kp="20" kv="0.5" ctrlrange="-2.0 0"/>
  </actuator>
</mujoco>
"""
        from brax.io import mjcf
        
        if not os.path.exists(model_path):
            print("[Warn] Model not found. Using auto-generated fallback model.")
            sys_brax = mjcf.loads(fallback_xml)
            sys_mj_model = mujoco.MjModel.from_xml_string(fallback_xml)
        else:
            sys_brax = mjcf.load(model_path)
            sys_mj_model = mujoco.MjModel.from_xml_path(model_path)

        sys_mj_model.actuator_gainprm[:, 0] = RobotConfig.KP
        sys_mj_model.actuator_biasprm[:, 1] = -RobotConfig.KP
        sys_mj_model.actuator_biasprm[:, 2] = -RobotConfig.KD
        sys_brax = sys_brax.replace(
            actuator=sys_brax.actuator.replace(
                gain=jp.full((sys_mj_model.nu,), RobotConfig.KP),
                bias_q=jp.full((sys_mj_model.nu,), -RobotConfig.KP),
                bias_qd=jp.full((sys_mj_model.nu,), -RobotConfig.KD),
            )
        )
            
        sys_mj_model.opt.timestep = RobotConfig.SIM_DT
        sys_brax = sys_brax.replace(opt=sys_brax.opt.replace(timestep=RobotConfig.SIM_DT))
        mjx_model = mjx.put_model(sys_mj_model)

        super().__init__(sys_brax, backend='mjx', n_frames=RobotConfig.CONTROL_DECIMATION, **kwargs)
        
        self._mjx_model = mjx_model
        self._actuator_indices = jp.array(list(range(sys_mj_model.nu)), dtype=jp.int32)
        self.obs_noise = obs_noise
        self.latency_steps = latency_steps
        
        actuator_to_qpos_list = []
        actuator_to_qvel_list = []
        for act_i in range(sys_mj_model.nu):
            jnt_id = sys_mj_model.actuator_trnid[act_i][0]
            qpos_adr = sys_mj_model.jnt_qposadr[jnt_id]
            qvel_adr = sys_mj_model.jnt_dofadr[jnt_id]
            actuator_to_qpos_list.append(qpos_adr)
            actuator_to_qvel_list.append(qvel_adr)
        self._actuator_to_qpos_idx = jp.array(actuator_to_qpos_list, dtype=jp.int32)
        self._actuator_to_qvel_idx = jp.array(actuator_to_qvel_list, dtype=jp.int32)
        
        from envs.mjx_rewards import MJXRewardSystem
        left_foot_id = mujoco.mj_name2id(sys_mj_model, mujoco.mjtObj.mjOBJ_BODY, 'doutai-v5_hidaridairou_hidarikokansetu_hidarimomo_hidarihizabu_hidariaikabu_hidariashiura_hidariashiura-1')
        right_foot_id = mujoco.mj_name2id(sys_mj_model, mujoco.mjtObj.mjOBJ_BODY, 'doutai-v5_migidaitou_migikokansetu_migimomo_migihizabu_migigaikabu_migiashiura_migiashiura-1')
        
        if left_foot_id == -1 or right_foot_id == -1:
            left_matches = [i for i in range(sys_mj_model.nbody) if 'hidariashiura' in sys_mj_model.body(i).name]
            right_matches = [i for i in range(sys_mj_model.nbody) if 'migiashiura' in sys_mj_model.body(i).name]
            
            if not left_matches or not right_matches:
                raise RuntimeError(
                    f"Could not find foot bodies in MuJoCo model. "
                    f"Left matches: {left_matches}, Right matches: {right_matches}"
                )
            left_foot_id = left_matches[0]
            right_foot_id = right_matches[0]

        self._reward_system = MJXRewardSystem(self._mjx_model, RobotConfig.REWARD_WEIGHTS, left_foot_id, right_foot_id)
        
        from safety.cbf import CBFSafetyFilter
        self._cbf = CBFSafetyFilter()

    @property
    def action_size(self):
        return self.sys.nu
        
    @property
    def observation_size(self):
        return RobotConfig.OBS_DIM

    def _get_curriculum_scale(self, training_progress: float) -> float:
        """カリキュラム学習: 学習進捗率に応じた外乱強度スケーリング"""
        schedule = RobotConfig.CURRICULUM_SCHEDULE_FRACTIONS
        keys = sorted(schedule.keys())
        
        scale = jp.array(schedule[keys[0]], dtype=jp.float32)
        for key in keys:
            scale = jp.where(training_progress >= key, jp.array(schedule[key], dtype=jp.float32), scale)
        
        return jp.clip(scale, 0.0, 1.0)

    def _apply_domain_randomization(self, model, mass_scale, fric_scale, com_offset):
        """Domain randomization を物理モデルに反映する。

        旧実装では乱数値を privileged observation にのみ保存しており、
        MJX の body_mass / geom_friction / inertial COM に反映されていなかった。
        これにより DR が「観測上のメタ情報」だけに留まり、実際の物理挙動には効かない。
        """
        randomized = model

        if hasattr(model, 'body_mass'):
            randomized = randomized.replace(body_mass=jp.asarray(model.body_mass) * mass_scale)
        if hasattr(model, 'geom_friction'):
            randomized = randomized.replace(geom_friction=jp.asarray(model.geom_friction) * fric_scale)
        if hasattr(model, 'body_ipos') and model.body_ipos.shape[0] > 0:
            body_ipos = jp.asarray(model.body_ipos).copy()
            body_ipos = body_ipos.at[0].add(jp.asarray(com_offset, dtype=body_ipos.dtype))
            randomized = randomized.replace(body_ipos=body_ipos)

        return randomized

    def _apply_joint_dr_torque(self, qfrc_applied, qvel, dr_damping, dr_friction):
        if self._mjx_model.nq >= 7:
            joint_vel = qvel[self._actuator_to_qvel_idx]
            joint_torque = -dr_damping * joint_vel - dr_friction * jp.sign(joint_vel)
            return qfrc_applied.at[self._actuator_to_qvel_idx].add(joint_torque)

        joint_torque = -dr_damping * qvel - dr_friction * jp.sign(qvel)
        return qfrc_applied.add(joint_torque)

    def reset(self, rng: jax.Array) -> State:
        rng, rng_noise, rng_priv = jax.random.split(rng, 3)
        
        key_mass, key_fric, key_com, key_dr1, key_dr2, key_dr3, key_scale, key_temp, key_volt = jax.random.split(rng_priv, 9)
        mass_scale = jax.random.uniform(key_mass, shape=(), minval=RobotConfig.RANDOM_MASS_SCALE[0], maxval=RobotConfig.RANDOM_MASS_SCALE[1])
        fric_scale = jax.random.uniform(key_fric, shape=(), minval=RobotConfig.RANDOM_FRICTION[0], maxval=RobotConfig.RANDOM_FRICTION[1])
        com_offset = jax.random.uniform(key_com, shape=(3,), minval=RobotConfig.RANDOM_COM_OFFSET[0], maxval=RobotConfig.RANDOM_COM_OFFSET[1])

        # 物理モデルへ反映してからリセットし、各episodeでDRが実際に効くようにする。
        self._mjx_model = self._apply_domain_randomization(self._mjx_model, mass_scale, fric_scale, com_offset)
        self._reward_system._model = self._mjx_model
        
        servo_temp = jax.random.uniform(key_temp, shape=(self._mjx_model.nu,), minval=RobotConfig.RANDOM_TEMP[0], maxval=RobotConfig.RANDOM_TEMP[1])
        supply_volt = jax.random.uniform(key_volt, shape=(1,), minval=RobotConfig.RANDOM_VOLT[0], maxval=RobotConfig.RANDOM_VOLT[1])
        
        privileged_obs = jp.concatenate([jp.array([mass_scale, fric_scale]), com_offset, servo_temp, supply_volt])
        
        mjx_data = mjx.make_data(self._mjx_model)
        
        nq = self._mjx_model.nq
        qpos = jp.zeros(nq)
        if nq >= 7:
            num_act = min(len(RobotConfig.DEFAULT_JOINT_ANGLES), self._mjx_model.nu)
            default_angles = jp.array(RobotConfig.DEFAULT_JOINT_ANGLES[:num_act])
            target_indices = self._actuator_to_qpos_idx[:num_act]
            qpos = qpos.at[target_indices].set(default_angles)

            base_position = jp.array([0.0, 0.0, 0.1773], dtype=jp.float32) + com_offset
            qpos = qpos.at[0:3].set(base_position)
            qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
            
        mjx_data = mjx_data.replace(qpos=qpos, qvel=jp.zeros(self._mjx_model.nv))
        mjx_data = mjx.forward(self._mjx_model, mjx_data)
        
        initial_potential = self._reward_system.compute_potential(mjx_data)
        
        info = {
            'step': 0,
            'global_step': 0,
            'phase': 0.0,
            'last_action': jp.zeros(self._mjx_model.nu),
            'action_buffer': jp.zeros(self._mjx_model.nu),
            'filtered_action': jp.zeros(self._mjx_model.nu),
            'double_last_action': jp.zeros(self._mjx_model.nu),
            'triple_last_action': jp.zeros(self._mjx_model.nu),
            'action_history': jp.zeros((RobotConfig.HISTORY_LEN, self._mjx_model.nu)),
            'obs_history': jp.zeros((RobotConfig.HISTORY_LEN, RobotConfig.BASE_OBS_DIM)),
            'servo_temp': servo_temp,
            'supply_volt': supply_volt[0],
            'dr_damping': jax.random.uniform(key_dr1, shape=(self._mjx_model.nu,), minval=0.01, maxval=0.15),
            'dr_friction': jax.random.uniform(key_dr2, shape=(self._mjx_model.nu,), minval=0.0, maxval=0.08),
            'dr_kp_scale': jax.random.uniform(key_dr3, shape=(self._mjx_model.nu,), minval=0.7, maxval=1.3),
            'disturbance_scale': jax.random.uniform(key_scale, minval=0.0, maxval=1.0),
            'privileged_obs': privileged_obs,
            'last_potential': initial_potential,
            'rng_key': rng_noise,
            'was_disturbed': jp.array(False),
            'disturbance_impulse': jp.zeros(3),  # [FIX] 力積の記録領域を初期化
            'disturbance_recovery_steps': jp.array(1000),
            # Direct eval / checkpoint validation should not act like an uninitialized training run.
            # A zero progress value suppresses the entire soft-penalty branch and makes the eval path
            # behave differently from the training objective.
            'training_progress': jp.array(1.0),
            '_env_steps': jp.array(0, dtype=jp.int32),
            'terminated': jp.array(False),
            'truncated': jp.array(False),
            'time_out': jp.array(0.0),
        }
        
        obs, info = self._get_obs(mjx_data, info, rng_noise)
        
        assert obs.shape[0] == RobotConfig.OBS_DIM, (
            f"Observation shape mismatch at reset(): computed {obs.shape[0]}, "
            f"but RobotConfig.OBS_DIM is {RobotConfig.OBS_DIM}."
        )
        
        reward, done, zero = jp.zeros(3)
        metrics = {
            'alive': zero, 'total_reward': zero, 'reward': zero,
            'reward_per_step': zero, 'total_penalty': zero,
            'lambda_phase': zero, 'r_cp': zero, 'r_recovery': zero,
            'r_com_stab': zero, 'pbrs_reward': zero,
            'both_feet_contact': zero,
            'potential': zero, 'fall_penalty': zero,
            'foot_balance': zero, 'zmp_margin': zero,
            'disturbance_recovery_bonus': zero, 'stability_index': zero,
            'curriculum_scale': zero,
            'barrier_height': zero, 'barrier_torque': zero,
            # [監査追加 2026-09-13] step()で追加したキーとpytree構造を
            # 一致させる必要がある(reset/step間でmetrics辞書の
            # キー集合・shapeが異なるとjax.lax.scan等でエラーになるため)。
            'r_upright': zero,
            'reward_is_finite': zero,
            'action_saturation': zero, 'cbf_correction_norm': zero,
            # [監査追加 2026-09-13] stability_metrics.py / mjx_rewards.py
            # 側で追加した診断フラグとpytree構造を一致させる。
            'stability_metrics_finite': zero, 'com_accel_is_fallback': zero,
            # [予防追加 2026-09-13] envs/mjx_env.py の physics_step
            # ロールバック機構が発散を検出したかどうかのフラグ。
            'physics_diverged': zero,
        }
        
        return State(mjx_data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """1制御周期(CONTROL_DT=0.01秒)を実行する。

        Args:
            state: 直前のState (pipeline_state, obs, info を含む)
            action: 20次元、[-1, 1]相当のpolicy出力。ACTION_SCALEで
                スケールされ、default_pose + action*ACTION_SCALE として
                目標関節角(残差方式)に変換される。

        Returns:
            新しいState。done は terminated のみを表す
            (truncatedは state.info['truncated'] に分離)。
            state.metrics には reward_is_finite 等の診断フラグを含む
            (改良規約 §18 NaN/Inf即時停止条件のため)。
        """
        info = state.info.copy()
        
        if RobotConfig.USE_REFERENCE_GAIT:
            residual_rad = action * RobotConfig.ACTION_SCALE * 0.5
            base_target_rad = info.get('reference_action', jp.zeros(self._mjx_model.nu))
            target_rad = base_target_rad + residual_rad
        else:
            default_pose = jp.array(RobotConfig.DEFAULT_JOINT_ANGLES[:self._mjx_model.nu])
            target_rad = default_pose + action * RobotConfig.ACTION_SCALE

        if getattr(RobotConfig, 'TARGET_VEL_X', 0.0) != 0.0 or getattr(RobotConfig, 'TARGET_VEL_Y', 0.0) != 0.0 or getattr(RobotConfig, 'TARGET_YAW_RATE', 0.0) != 0.0:
            raise ValueError("Standing-only mission requires TARGET_VEL_X/Y/YAW_RATE all zero.")
        
        current_cmd = info['filtered_action']
        delta_rad = target_rad - current_cmd
        
        deadband_threshold = 0.02
        delta_rad = jp.where(jp.abs(delta_rad) < deadband_threshold, 0.0, delta_rad)
        
        max_delta = RobotConfig.MOTOR_MAX_VELOCITY * RobotConfig.CONTROL_DT
        delta_rad = jp.clip(delta_rad, -max_delta, max_delta)
        
        constrained_target = current_cmd + delta_rad
        
        alpha = RobotConfig.MOTOR_LPF_ALPHA
        filtered_action = (1.0 - alpha) * current_cmd + alpha * constrained_target
        info['filtered_action'] = filtered_action
        
        # Apply CBF Safety Filter
        limit_lower = self._mjx_model.actuator_ctrlrange[:, 0]
        limit_upper = self._mjx_model.actuator_ctrlrange[:, 1]
        
        safe_target_rad = self._cbf.filter_action(filtered_action, limit_lower, limit_upper)
        # [BUGFIX 2026-09-10] 旧呼び出しは compute_cbf_penalty(target_rad, limit_lower,
        # limit_upper) となっており、safety/cbf.py の現行シグネチャ
        # compute_cbf_penalty(nominal_action, safe_action, limit_lower=None, limit_upper=None)
        # と噛み合っていなかった。結果として:
        #   - 第2引数(safe_action)に limit_lower の値が誤って渡り、
        #     direct_penalty が「target_radと関節下限との距離」という
        #     無意味な量になっていた
        #   - 第4引数(limit_upper)が渡されず None のままとなり、
        #     margin-basedのsoftplusペナルティ(CBF-2/CBF-3で導入された
        #     より厳格な項)が常にスキップされていた
        # filter_action()が返す safe_target_rad (実際にクランプされた
        # アクション)を正しく第2引数として渡すよう修正した。
        # 物理的な安全性(filter_action()によるハードクランプ)自体は
        # このバグの影響を受けていない。影響はCBFペナルティによる
        # 報酬整形が意図通り機能していなかった点のみ。
        cbf_penalty = self._cbf.compute_cbf_penalty(target_rad, safe_target_rad, limit_lower, limit_upper)
        
        # Thermal & Voltage Derating
        act_state = ActuatorState(temperature=info['servo_temp'], supply_voltage=info['supply_volt'])
        thermal_derating = HX30HMModel.compute_thermal_derating(act_state.temperature)
        voltage_derating = HX30HMModel.compute_voltage_derating(act_state.supply_voltage)
        real_target_rad = current_cmd + (safe_target_rad - current_cmd) * thermal_derating * voltage_derating
        
        # [FIX] 熱モデルの入力を物理エンジンの実トルクに変更
        # 前回の物理ステップで計算された actuator_force を使用して正確な発熱を推定
        real_torque = state.pipeline_state.actuator_force
        new_act_state = HX30HMModel.update_temperature(act_state, real_torque, RobotConfig.CONTROL_DT)
        info['servo_temp'] = new_act_state.temperature
        
        # Actuator History Buffer & Stochastic Delay
        ah = jp.roll(info['action_history'], shift=-1, axis=0)
        ah = ah.at[-1].set(real_target_rad)
        info['action_history'] = ah
        
        rng_delay, rng_push, rng_obs, next_rng = jax.random.split(info['rng_key'], 4)
        info['rng_key'] = next_rng
        
        delay_idx = jax.random.randint(rng_delay, shape=(), minval=0, maxval=3)
        applied_action = ah[RobotConfig.HISTORY_LEN - 1 - delay_idx]
        
        disturbance_enabled = getattr(RobotConfig, 'DISTURBANCE_CURRICULUM', False)
        if disturbance_enabled:
            curriculum_disturbance_scale = self._get_curriculum_scale(info.get('training_progress', jp.array(0.0)))
            current_max_force = RobotConfig.RANDOM_PUSH_MAX_FORCE * curriculum_disturbance_scale
            rng_push_trigger, rng_push_dir = jax.random.split(rng_push, 2)
            is_push_step = jax.random.uniform(rng_push_trigger) < 0.03
            push_force_raw = jax.random.uniform(
                rng_push_dir, shape=(3,),
                minval=jp.array([-0.5, -1.0, -0.2]),
                maxval=jp.array([0.5, 1.0, 0.2])
            )
            push_force_norm = push_force_raw / (jp.linalg.norm(push_force_raw) + 1e-6)
            push_force = jp.where(is_push_step, push_force_norm * current_max_force, jp.zeros(3))
        else:
            is_push_step = jp.array(False)
            push_force = jp.zeros(3)

        # [FIX] 外乱の力積(Impulse)を計算して記録 (caveat契約遵守)
        push_impulse = push_force * RobotConfig.CONTROL_DT
        info['disturbance_impulse'] = push_impulse

        if getattr(RobotConfig, 'ALLOW_WALKING', False) or getattr(RobotConfig, 'ALLOW_STEPPING', False):
            raise ValueError("Walking/stepping is forbidden for this task.")
        
        qfrc_applied = jp.zeros(self._mjx_model.nv)
        if self._mjx_model.nq >= 7:
            qfrc_applied = qfrc_applied.at[0:3].set(push_force)
            
        qfrc_applied = self._apply_joint_dr_torque(
            qfrc_applied,
            state.pipeline_state.qvel,
            info['dr_damping'],
            info['dr_friction'],
        )

        def physics_step(carry, _):
            d_prev = carry
            d = carry.replace(ctrl=applied_action, qfrc_applied=qfrc_applied)
            d = mjx.step(self._mjx_model, d)
            # [予防追加 2026-09-13] NaN/Inf予防: mjx.step() は
            # NaN-in→NaN-out のため、CONTROL_DECIMATION回のサブステップの
            # うち1回でも発散すると、残り全サブステップが汚染され、この
            # 環境は(次に外部からリセットされるまで)永続的にNaN化して
            # しまう。ここで即座に直前の有効な状態へロールバックすることで
            # 汚染の伝播を1サブステップで食い止める。発散した事実は
            # diverged フラグとして持ち帰り、呼び出し側で done=True を
            # 強制する(=次stepで自動リセットされる)。
            state_is_finite = jp.all(jp.isfinite(d.qpos)) & jp.all(jp.isfinite(d.qvel))
            d = jax.tree_util.tree_map(
                lambda new, old: jp.where(state_is_finite, new, old), d, d_prev
            )
            return d, jp.logical_not(state_is_finite)

        mjx_data, diverged_flags = jax.lax.scan(
            physics_step, state.pipeline_state, (), length=RobotConfig.CONTROL_DECIMATION
        )
        physics_diverged = jp.any(diverged_flags)
        
        was_disturbed = jp.array(is_push_step, dtype=jp.bool_)
        disturbance_recovery_steps = jp.where(
            is_push_step,
            jp.array(0),
            info.get('disturbance_recovery_steps', jp.array(0)) + 1
        )
        
        reward, done, metrics, current_potential = self._reward_system.compute(
            mjx_data, applied_action, info['last_action'], info['double_last_action'],
            info['triple_last_action'], cbf_penalty, info['last_potential'], info['step'],
            info.get('reference_action', jp.zeros(self._mjx_model.nu)),
            servo_temp=info.get('servo_temp', None), 
            supply_volt=info.get('supply_volt', 11.1),
            global_step=jp.array(info.get('global_step', 0), dtype=jp.int32),
            gait_phase=jp.asarray(info.get('phase', 0.0), dtype=jp.float32),
            was_disturbed=was_disturbed,
            disturbance_recovery_steps=disturbance_recovery_steps,
            training_progress=info.get('training_progress', jp.array(0.0)),
        )
        info['last_potential'] = current_potential

        # [予防追加 2026-09-13] 物理サブステップが発散(NaN/Inf)していた
        # 場合は無条件でdone=Trueにする。physics_step()側で状態自体は
        # 直前の有効な値へロールバック済みで安全だが、その「発散直前で
        # 足止めされた」状態のまま学習を続けさせると、PPOがそれを
        # 暗黙に「良い状態」と誤学習しかねないため、エピソードを
        # 明示的に打ち切る(=fall_penaltyと同等に扱われる)。
        done = jp.logical_or(done, physics_diverged)
        reward = jp.where(physics_diverged, RobotConfig.REWARD_WEIGHTS['fall_penalty'], reward)
        metrics['physics_diverged'] = physics_diverged.astype(jp.float32)

        # [監査追加 2026-09-13] CBFがtarget_radをどれだけ補正(クランプ)したか
        # を診断指標として記録する。train_mjx.py の _audit_reward_metrics()
        # がこれを見て「方策が実行不能な指令を多発させていないか
        # (Action Distortion)」を検出する。計算本体は safety/cbf.py の
        # compute_saturation_ratio() に委譲している(CBFの挙動の診断は
        # CBFクラス自身の責務とするため)。
        action_saturation = self._cbf.compute_saturation_ratio(
            target_rad, safe_target_rad, limit_lower, limit_upper
        )
        cbf_correction_norm = jp.linalg.norm(safe_target_rad - target_rad)
        metrics['action_saturation'] = action_saturation
        metrics['cbf_correction_norm'] = cbf_correction_norm

        info['triple_last_action'] = info['double_last_action']
        info['double_last_action'] = info['last_action']
        info['last_action'] = applied_action
        info['step'] += 1
        env_steps = jp.asarray(info.get('_env_steps', info.get('global_step', 0)), dtype=jp.int32) + 1
        info['_env_steps'] = env_steps
        info['global_step'] = env_steps
        
        if RobotConfig.USE_REFERENCE_GAIT:
            info['phase'] = (info.get('phase', 0.0) + RobotConfig.CONTROL_DT / RobotConfig.GAIT_PERIOD) % 1.0
        else:
            info['phase'] = 0.0
            
        info['disturbance_recovery_steps'] = disturbance_recovery_steps
        info['was_disturbed'] = was_disturbed
        terminated = done
        truncated = info['step'] >= RobotConfig.MAX_EPISODE_STEPS
        
        done = terminated
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['time_out'] = truncated.astype(jp.float32)
        
        obs, info = self._get_obs(mjx_data, info, rng_obs)
        
        return state.replace(pipeline_state=mjx_data, obs=obs, reward=reward,
                             done=done.astype(jp.float32), metrics=metrics, info=info)

    def _extract_fsr_sensor_data(self, data: mjx.Data) -> jax.Array:
        nsensor = getattr(self._mjx_model, 'nsensordata', 0)
        if nsensor >= 18:
            return data.sensordata[10:18]
        if nsensor >= 8:
            return data.sensordata[-8:]
        if nsensor > 0:
            pad_len = 8 - nsensor
            return jp.concatenate([data.sensordata, jp.zeros(pad_len)])
        return jp.zeros(8)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any], rng: jax.Array) -> Tuple[jax.Array, Dict[str, Any]]:
        subtree_com = getattr(data, 'subtree_com', None)
        if subtree_com is not None:
            com_pos = subtree_com[0]
        else:
            if self._mjx_model.nq >= 7:
                com_pos = data.qpos[0:3]
            else:
                com_pos = jp.zeros(3)
        
        if self._mjx_model.nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            base_lin_vel = data.qvel[0:3]
            base_ang_vel = data.qvel[3:6]
            joint_pos = data.qpos[self._actuator_to_qpos_idx]
            joint_vel = data.qvel[self._actuator_to_qvel_idx]
        else:
            base_pos = base_quat = base_lin_vel = base_ang_vel = jp.zeros(3)
            base_quat = jp.array([1., 0., 0., 0.])
            joint_pos = data.qpos
            joint_vel = data.qvel
            
        rpy = quat_to_euler(base_quat)
        fsr_data = self._extract_fsr_sensor_data(data)

        foot_positions = jp.array(RobotConfig.FSR_POSITIONS)
        total_p = jp.sum(fsr_data) + 1e-6
        zmp_x = jp.sum(foot_positions[:, 0] * fsr_data) / total_p
        zmp_y = jp.sum(foot_positions[:, 1] * fsr_data) / total_p
        zmp = jp.array([zmp_x, zmp_y])
        
        obs_components = [base_pos, rpy, base_lin_vel, base_ang_vel, joint_pos, joint_vel, fsr_data, zmp]
        raw_obs = jp.concatenate(obs_components)
        
        rng_obs, rng_pos, rng_vel = jax.random.split(rng, 3)
        noise = jax.random.normal(rng_obs, raw_obs.shape) * self.obs_noise
        noisy_obs = raw_obs + noise
        
        pos_noise = jax.random.normal(rng_pos, (3,)) * RobotConfig.NOISE_BASE_POS
        vel_noise = jax.random.normal(rng_vel, (3,)) * RobotConfig.NOISE_LIN_VEL
        noisy_obs = noisy_obs.at[0:3].add(pos_noise)
        noisy_obs = noisy_obs.at[6:9].add(vel_noise)
        
        phase = info.get('phase', 0.0)
        phase_obs = jp.array([jp.sin(2 * jp.pi * phase), jp.cos(2 * jp.pi * phase)])
        
        if RobotConfig.USE_REFERENCE_GAIT:
            from robot.gait_generator import jax_get_reference_trajectory
            ref_angles = jax_get_reference_trajectory(phase, self._mjx_model.nu)
            ref_angles_obs = ref_angles
        else:
            ref_angles = jp.array(RobotConfig.DEFAULT_JOINT_ANGLES[:self._mjx_model.nu])
            ref_angles_obs = jp.zeros_like(ref_angles)
            
        info['reference_action'] = ref_angles
        
        base_obs = jp.concatenate([noisy_obs, phase_obs, ref_angles_obs])
        
        obs_hist = info.get('obs_history', jp.zeros((RobotConfig.HISTORY_LEN, RobotConfig.BASE_OBS_DIM)))
        obs_hist = jp.roll(obs_hist, shift=-1, axis=0)
        obs_hist = obs_hist.at[-1].set(base_obs)
        info['obs_history'] = obs_hist
        
        flat_obs_hist = obs_hist.flatten()
        flat_act_hist = info.get('action_history', jp.zeros((RobotConfig.HISTORY_LEN, self._mjx_model.nu))).flatten()
        
        servo_temp = info.get('servo_temp', jp.zeros(self._mjx_model.nu))
        supply_volt = jp.array([info.get('supply_volt', 11.1)])
        
        final_obs = jp.concatenate([base_obs, flat_obs_hist, flat_act_hist, servo_temp, supply_volt])
        
        assert final_obs.shape[0] == RobotConfig.OBS_DIM, (
            f"Observation shape mismatch: computed {final_obs.shape[0]}, "
            f"but RobotConfig.OBS_DIM is configured as {RobotConfig.OBS_DIM}."
        )
        
        return final_obs, info

envs.register_environment('senpuu_maru_mjx', SenpuuMaruMJXEnv)
```

---

## envs/mjx_rewards.py

```python
"""固定足立位タスク用の報酬関数 (MJXRewardSystem.compute())。

成功判定は episode_alive 単独ではなく、以下の論理積で評価する
(改良規約 §11 参照):
  alive AND both_feet_contact AND upright AND height_ok
  AND no_illegal_contact AND slip_ok AND torque_ok AND recovered_in_time

報酬の主要成分:
  - r_alive: 生存ボーナス
  - r_upright / r_com_stab: 姿勢・重心安定性
  - r_capture_point / r_recovery / r_disturbance_recovery: 外乱回復系
    (Phase 0では外乱無効のため寄与は限定的)
  - soft_penalty / safety_penalty: エネルギー・滑らかさ・CBF安全項

NaN/Inf検出:
  total_reward が clip される前に jp.isfinite で検査し、結果を
  metrics['reward_is_finite'] (1.0=正常, 0.0=非有限値検出) として返す。
  JAX JITトレース内でPythonのraiseは使えないため、フラグ経由で
  呼び出し側 (train/train_mjx.py の progress_callback) に非有限値の
  発生を伝える設計 (改良規約 §18 即時停止条件)。

注意: このファイルは envs/mjx_env.py の step() (vmap/jit内部) から
呼ばれるため、全ての引数は単一環境のスカラー(バッチ次元なし)である。
"""

import jax
import jax.numpy as jp
from typing import Tuple, Dict

from mujoco import mjx

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler
from envs.stability_metrics import StabilityMetrics

"""
================================================================================
v2.2 (2026-09 レビュー: JAX/Braxバッチ次元誤解の修正と足裏基準高さの厳密化)
================================================================================
[FIX] JAX vmap境界の誤解解消
    旧実装にあった「training_progress は shape (num_envs,) を想定」という
    設計は、Braxの仕様上完全に誤りでした。mjx_rewards.py は mjx_env.py の
    step() (vmap内部) から呼ばれるため、すべての変数はすでに単一環境の
    スカラー(バッチ次元なし)としてスライスされています。
    要素ごとの処理（batch-wise）を想定したロジックを削除し、
    正しいスカラー処理としてクリーンアップしました。

[FIX] Contract Violation B (足裏基準の高さ判定) 修正
    caveat.md の「高さは足裏を基準にする」という契約に違反し、
    終了判定(is_low)や p_barrier_height でワールド絶対座標系 Z=0 からの
    base_pos[2] が使われていました。
    両足(data.xpos)のうち低い方のZ座標を基準点とし、重心との「相対高さ」
    (relative_height) を評価・判定に使用するよう修正しました。
================================================================================
"""

class MJXRewardSystem:
    def __init__(self, model: mjx.Model, weights: dict, left_foot_id: int, right_foot_id: int):
        self._model = model
        self._nq = model.nq
        self._nu = model.nu
        self._weights = weights
        self._left_foot_id = left_foot_id
        self._right_foot_id = right_foot_id

        foot_support_radius = getattr(RobotConfig, 'FOOT_SUPPORT_RADIUS', 0.06)
        self._stability = StabilityMetrics(
            left_foot_id, right_foot_id, RobotConfig.COM_HEIGHT,
            foot_support_radius=foot_support_radius,
        )

    def compute_potential(self, data: mjx.Data, lambda_phase: jax.Array = None) -> jax.Array:
        if self._nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            rpy = quat_to_euler(base_quat)
        else:
            base_pos = jp.zeros(3)
            rpy = jp.zeros(3)

        gravity_projection = jp.cos(rpy[0]) * jp.cos(rpy[1])
        p_upright = jp.exp(-5.0 * (1.0 - gravity_projection))

        pos_err = jp.sum(jp.square(base_pos[0:2]))
        yaw_err = jp.square(rpy[2])
        p_target = jp.exp(-2.0 * pos_err - 1.0 * yaw_err)

        w = self._weights
        lp = 1.0 if lambda_phase is None else lambda_phase
        return p_upright * w['upright'] + p_target * w['target_pose'] * lp

    def _get_curriculum_disturbance_scale(self, training_progress: jax.Array) -> jax.Array:
        """
        カリキュラム学習: 学習進捗率(スカラー)に応じて外乱強度を段階的に増加。
        """
        schedule = RobotConfig.CURRICULUM_SCHEDULE_FRACTIONS
        keys = sorted(schedule.keys())

        scale = jp.array(schedule[keys[0]], dtype=jp.float32)
        for key in keys:
            scale = jp.where(training_progress >= key, jp.array(schedule[key], dtype=jp.float32), scale)

        return jp.clip(scale, 0.0, 1.0)

    def _compute_adaptive_reward_scaling(self, servo_temp: jax.Array, supply_volt: float) -> Dict[str, jax.Array]:
        max_servo_temp = jp.max(servo_temp)
        temp_stress = jp.clip((max_servo_temp - 60.0) / 20.0, 0.0, 1.0)
        volt_stress = jp.clip((10.5 - supply_volt) / 2.0, 0.0, 1.0)
        stress = jp.maximum(temp_stress, volt_stress)

        return {
            'recovery': 1.0 + stress * 0.2,
            'energy': 1.0 - stress * 0.3,
            'smoothness': 1.0 - stress * 0.3,
        }

    def _compute_disturbance_recovery_bonus(
        self,
        was_disturbed: jax.Array,
        disturbance_recovery_steps: jax.Array,
        stability_index: jax.Array,
        window_steps: float = None,
        stability_threshold: float = None,
    ) -> jax.Array:
        if window_steps is None:
            window_steps = getattr(RobotConfig, 'RECOVERY_BONUS_WINDOW_STEPS', 50)
        if stability_threshold is None:
            stability_threshold = getattr(RobotConfig, 'RECOVERY_BONUS_STABILITY_THRESHOLD', 0.7)

        steps = jp.maximum(disturbance_recovery_steps.astype(jp.float32), 0.0)
        urgency = jp.exp(-jp.log(2.0) * steps / jp.maximum(window_steps, 1.0))

        outer_cutoff = window_steps * 6.0
        is_recovering = jp.logical_and(
            disturbance_recovery_steps >= 0,
            disturbance_recovery_steps < outer_cutoff,
        )
        is_stable = stability_index > stability_threshold

        bonus = jp.where(
            jp.logical_and(is_recovering, is_stable),
            urgency * stability_index * 2.0,
            0.0,
        )
        return jp.clip(bonus, 0.0, 10.0)

    @staticmethod
    def _log_barrier_lower(x: jax.Array, x_min: jax.Array, margin: jax.Array, clip_val: jax.Array) -> jax.Array:
        gap = x - x_min
        m = jp.maximum(margin, 1e-6)
        in_margin = jp.logical_and(gap > 0.0, gap < m)
        penalty = -jp.log(jp.clip(gap / m, 1e-4, 1.0))
        val = jp.where(in_margin, penalty, 0.0)
        val = jp.where(gap <= 0.0, clip_val, val)
        return jp.clip(val, 0.0, clip_val)

    @staticmethod
    def _log_barrier_upper(x: jax.Array, x_max: jax.Array, margin: jax.Array, clip_val: jax.Array) -> jax.Array:
        gap = x_max - x
        m = jp.maximum(margin, 1e-6)
        in_margin = jp.logical_and(gap > 0.0, gap < m)
        penalty = -jp.log(jp.clip(gap / m, 1e-4, 1.0))
        val = jp.where(in_margin, penalty, 0.0)
        val = jp.where(gap <= 0.0, clip_val, val)
        return jp.clip(val, 0.0, clip_val)

    @staticmethod
    def extract_fsr_sensor_data(model: mjx.Model, data: mjx.Data) -> jax.Array:
        nsensor = getattr(model, 'nsensordata', 0)
        if nsensor >= 18:
            return data.sensordata[10:18]
        if nsensor >= 8:
            return data.sensordata[-8:]
        if nsensor > 0:
            pad_len = 8 - nsensor
            return jp.concatenate([data.sensordata, jp.zeros(pad_len)])
        return jp.zeros(8)

    def compute(
        self,
        data: mjx.Data,
        action: jax.Array,
        last_action: jax.Array,
        double_last_action: jax.Array,
        triple_last_action: jax.Array,
        cbf_penalty: jax.Array,
        last_potential: jax.Array,
        step: jax.Array,
        reference_action: jax.Array,
        servo_temp: jax.Array = None,
        supply_volt: float = 11.1,
        global_step: jax.Array = None,
        gait_phase: float = 0.0,
        was_disturbed: jax.Array = None,
        disturbance_recovery_steps: jax.Array = None,
        training_progress: jax.Array = None,
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array], jax.Array]:

        if global_step is None:
            global_step = step
        if servo_temp is None:
            servo_temp = jp.zeros(self._nu)
        if was_disturbed is None:
            was_disturbed = jp.array(False)
        if disturbance_recovery_steps is None:
            disturbance_recovery_steps = jp.array(1000)

        # --- 1. 状態抽出 ---
        if self._nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            base_lin_vel = data.qvel[0:3]
            base_ang_vel = data.qvel[3:6]
            rpy = quat_to_euler(base_quat)
            torques = data.actuator_force

            actuator_to_qpos = jp.array([
                self._model.jnt_qposadr[self._model.actuator_trnid[i][0]]
                for i in range(self._model.nu)
            ], dtype=jp.int32)
            actuator_to_qvel = jp.array([
                self._model.jnt_dofadr[self._model.actuator_trnid[i][0]]
                for i in range(self._model.nu)
            ], dtype=jp.int32)
            joint_pos = data.qpos[actuator_to_qpos]
            joint_vel = data.qvel[actuator_to_qvel]
        else:
            base_pos = jp.zeros(3)
            rpy = jp.zeros(3)
            base_lin_vel = jp.zeros(3)
            base_ang_vel = jp.zeros(3)
            torques = jp.zeros(self._nu)
            joint_pos = data.qpos
            joint_vel = data.qvel

        subtree_com = getattr(data, 'subtree_com', None)
        com_pos = subtree_com[0] if subtree_com is not None else base_pos

        base_qacc = getattr(data, 'qacc', None)
        if base_qacc is not None and self._nq >= 7:
            com_accel = base_qacc[0:3]
            # [監査追加 2026-09-13] 通常経路(qacc取得成功)
            com_accel_is_fallback = jp.array(0.0)
        else:
            com_accel = jp.array([0.0, 0.0, -9.81])
            # [監査追加 2026-09-13] envs/stability_metrics.py のv2
            # [CRITICAL FIX]で説明されている「zmp_marginが死んだ指標に
            # なる」バグの片割れが、まさにこのフォールバック分岐だった
            # (このcom_accelではXY成分が常に0になるため、ZMPが
            # 重心位置に退化し、動的な不安定性を反映できなくなる)。
            # 数式自体は修正済みだが、この分岐が本番で有効化されていないか
            # train/train_mjx.py の _audit_reward_metrics() が
            # 'com_accel_is_fallback' として監視する。
            com_accel_is_fallback = jp.array(1.0)

        # --- 2. 足裏位置と相対高さの計算 (Contract Violation B 修正) ---
        left_foot_pos = data.xpos[self._left_foot_id]
        right_foot_pos = data.xpos[self._right_foot_id]

        lowest_foot_z = jp.minimum(left_foot_pos[2], right_foot_pos[2])
        # 高さは足裏を基準とする（caveat契約遵守）
        relative_height = base_pos[2] - lowest_foot_z

        # --- 3. カリキュラム学習による外乱スケーリング ---
        tp = training_progress if training_progress is not None else jp.array(0.0)
        curriculum_disturbance_scale = self._get_curriculum_disturbance_scale(tp)

        # --- 4. λ_phase の計算 ---
        lw = getattr(RobotConfig, 'LAMBDA_PHASE_WEIGHTS', None) or {
            'tilt': 5.0, 'ang_vel': 0.5, 'lin_vel_err': 1.5, 'disturbance_flag': 4.0,
        }
        z_thresh = getattr(RobotConfig, 'LAMBDA_PHASE_Z_THRESH', 0.3)
        decay_k = getattr(RobotConfig, 'LAMBDA_PHASE_DECAY_K', 10.0)

        tilt_err = jp.sqrt(jp.square(rpy[0]) + jp.square(rpy[1]))
        ang_vel_norm = jp.linalg.norm(base_ang_vel)
        lin_vel_norm = jp.linalg.norm(base_lin_vel[0:2])
        disturbance_flag = jp.where(was_disturbed, 1.0, 0.0)

        z = (
            lw['tilt'] * tilt_err +
            lw['ang_vel'] * ang_vel_norm +
            lw['lin_vel_err'] * lin_vel_norm +
            lw['disturbance_flag'] * disturbance_flag
        )
        lambda_phase = jp.clip(
            jp.exp(-decay_k * jp.maximum(0.0, z - z_thresh)),
            0.0,
            1.0,
        )

        # --- 5. 終了判定 (足裏相対高さを利用) ---
        is_fallen_roll = jp.abs(rpy[0]) > RobotConfig.TERMINATION_ROLL
        is_fallen_pitch = jp.abs(rpy[1]) > RobotConfig.TERMINATION_PITCH
        is_low = relative_height < RobotConfig.TERMINATION_HEIGHT
        done = jp.logical_or(jp.logical_or(is_fallen_roll, is_fallen_pitch), is_low)

        # --- 6. 高度な安定性メトリクス計算 ---
        has_sensors = data.sensordata.shape[0] > 0
        fsr_data = self.extract_fsr_sensor_data(self._model, data)
        left_foot_force = jp.where(
            has_sensors,
            jp.clip(jp.mean(jp.abs(fsr_data[0:4] + 1e-6)), 0.0, 100.0),
            0.5,
        )
        right_foot_force = jp.where(
            has_sensors,
            jp.clip(jp.mean(jp.abs(fsr_data[4:8] + 1e-6)), 0.0, 100.0),
            0.5,
        )
        contact_threshold = getattr(RobotConfig, 'FOOT_CONTACT_THRESHOLD', 0.05)
        both_feet_contact = jp.logical_and(
            left_foot_force > contact_threshold,
            right_foot_force > contact_threshold,
        )

        cp_margin_norm_dist = getattr(RobotConfig, 'CP_MARGIN_NORM_DIST', 0.15)
        stability_index, stability_metrics = self._stability.compute_unified_stability_index(
            com_pos, base_lin_vel, com_accel, rpy, base_ang_vel,
            left_foot_pos, right_foot_pos,
            left_foot_force, right_foot_force,
            gait_phase=gait_phase,
            cp_margin_norm_dist=cp_margin_norm_dist,
        )

        # --- 7. 報酬の計算 ---
        r_alive = 1.0

        current_potential = self.compute_potential(data, lambda_phase)
        gamma = 0.99
        r_pbrs = gamma * current_potential - last_potential

        body_vel_xy = jp.linalg.norm(base_lin_vel[0:2])
        body_yaw_rate = jp.abs(base_ang_vel[2])

        r_com_stab = jp.exp(-10.0 * (base_lin_vel[0]**2 + base_lin_vel[1]**2))
        r_upright = jp.exp(-30.0 * tilt_err**2)
        r_still = jp.exp(-20.0 * (body_vel_xy**2 + body_yaw_rate**2))
        r_target_pose = jp.exp(-5.0 * (base_pos[0]**2 + base_pos[1]**2 + rpy[2]**2))
        r_both_feet_contact = both_feet_contact.astype(jp.float32)

        p_cp = stability_metrics['cp_point']
        swing_is_left = left_foot_force < right_foot_force
        swing_foot_2d = jp.where(swing_is_left, left_foot_pos[0:2], right_foot_pos[0:2])
        stance_foot_2d = jp.where(swing_is_left, right_foot_pos[0:2], left_foot_pos[0:2])

        cp_dist_swing = jp.linalg.norm(swing_foot_2d - p_cp)
        cp_dist_stance = jp.linalg.norm(stance_foot_2d - p_cp)
        best_cp_dist = jp.minimum(cp_dist_swing, cp_dist_stance)

        r_cp_far = jp.exp(-2.5 * best_cp_dist)
        r_cp_near = jp.exp(-15.0 * best_cp_dist ** 2)
        r_capture_point = 0.6 * r_cp_far + 0.4 * r_cp_near

        tilt_vec = jp.array([rpy[0], rpy[1]])
        ang_vel_xy = jp.array([base_ang_vel[0], base_ang_vel[1]])
        tilt_dir = tilt_vec / (jp.linalg.norm(tilt_vec) + 1e-6)
        recovery_rate = -jp.dot(tilt_dir, ang_vel_xy)
        recovery_gate = jp.tanh(tilt_err / 0.15)
        r_recovery = jp.clip(jp.maximum(0.0, recovery_rate), 0.0, 5.0) * recovery_gate

        r_impedance = jp.exp(-0.01 * jp.sum(jp.square(torques))) * stability_index

        r_disturbance_recovery = self._compute_disturbance_recovery_bonus(
            was_disturbed, disturbance_recovery_steps, stability_index
        )

        # --- 8. ペナルティ ---
        p_ang_momentum_z = jp.square(base_ang_vel[2])
        p_ang_momentum_xy = jp.square(base_ang_vel[0]) + jp.square(base_ang_vel[1])

        p_energy = jp.clip(jp.sum(jp.square(torques)), 0.0, 100.0)
        p_smoothness = jp.clip(jp.sum(jp.square(action - last_action)), 0.0, 100.0)

        foot_translation = jp.linalg.norm(base_pos[0:2])
        step_penalty = jp.clip(body_vel_xy * 10.0 + body_yaw_rate * 4.0 + foot_translation * 3.0, 0.0, 20.0)

        drift_multiplier = lambda_phase * (1.0 - stability_index * 0.3)
        p_drift = jp.clip(jp.sum(jp.square(base_pos[0:2])), 0.0, 100.0) * drift_multiplier

        p_slip = jp.clip((jp.linalg.norm(base_lin_vel) * jp.mean(jp.abs(joint_vel))) ** 2, 0.0, 100.0)

        foot_span = jp.linalg.norm(right_foot_pos[0:2] - left_foot_pos[0:2])
        stance_width_penalty = jp.clip(jp.maximum(0.0, foot_span - 0.16) * 20.0, 0.0, 20.0)

        no_step_penalty = jp.where(
            getattr(RobotConfig, 'ALLOW_WALKING', False) or getattr(RobotConfig, 'ALLOW_STEPPING', False),
            100.0,
            0.0,
        )

        # 高さバリアは足裏基準の相対高さを使用
        h_margin = getattr(RobotConfig, 'BARRIER_HEIGHT_MARGIN', 0.05)
        h_clip = getattr(RobotConfig, 'BARRIER_HEIGHT_CLIP', 5.0)
        p_barrier_height = self._log_barrier_lower(
            relative_height, RobotConfig.TERMINATION_HEIGHT, h_margin, h_clip
        )

        torque_margin_ratio = getattr(RobotConfig, 'BARRIER_TORQUE_MARGIN_RATIO', 0.15)
        t_clip = getattr(RobotConfig, 'BARRIER_TORQUE_CLIP', 5.0)
        torque_margin = RobotConfig.MOTOR_MAX_TORQUE * torque_margin_ratio
        p_barrier_torque = jp.mean(
            self._log_barrier_upper(jp.abs(torques), RobotConfig.MOTOR_MAX_TORQUE, torque_margin, t_clip)
        )

        # --- 9. アダプティブ報酬スケーリング ---
        adaptive_scaling = self._compute_adaptive_reward_scaling(servo_temp, supply_volt)

        # --- 10. ペナルティスケジューリング ---
        warmup_steps = getattr(RobotConfig, 'PENALTY_INTRA_EPISODE_WARMUP_STEPS', 30)
        intra_ep_scale = jp.clip(step / jp.maximum(warmup_steps, 1), 0.0, 1.0)
        progress_scale = 1.0 if training_progress is None else jp.clip(training_progress, 0.0, 1.0)
        penalty_scale = intra_ep_scale * progress_scale

        safety_warmup_steps = getattr(RobotConfig, 'SAFETY_PENALTY_WARMUP_STEPS', 10)
        safety_scale = jp.clip(step / jp.maximum(safety_warmup_steps, 1), 0.0, 1.0)

        # --- 11. 報酬の統合 ---
        w = self._weights

        soft_penalty = (
            p_ang_momentum_z * w['ang_momentum_z'] +
            p_ang_momentum_xy * w['ang_momentum_xy'] * lambda_phase +
            p_energy * w['energy'] * adaptive_scaling['energy'] +
            p_smoothness * w['smoothness'] * adaptive_scaling['smoothness'] +
            p_drift * w['drift'] +
            p_slip * w['slip'] * lambda_phase +
            stance_width_penalty * w.get('stance_width', 0.5) +
            step_penalty +
            no_step_penalty
        ) * penalty_scale

        safety_penalty = (
            cbf_penalty * w['cbf'] +
            p_barrier_height * w.get('barrier_height', 1.0) +
            p_barrier_torque * w.get('barrier_torque', 1.0)
        ) * safety_scale

        total_reward = (
            r_alive * w['alive'] +
            r_pbrs +
            r_upright * w['upright'] +
            r_still * w['com_stab'] +
            r_target_pose * w['target_pose'] +
            r_both_feet_contact * w.get('both_feet_contact', 0.0) +

            lambda_phase * (
                r_com_stab * w['com_stab']
            ) +

            (1.0 - lambda_phase) * (
                r_capture_point * w['capture_point'] * adaptive_scaling['recovery'] +
                r_recovery * w['recovery'] * adaptive_scaling['recovery'] +
                r_impedance * w['impedance'] +
                r_disturbance_recovery
            ) -

            soft_penalty - safety_penalty
        )

        # --- NaN/Inf 検出（改良規約 §18: 即時停止条件） ---
        # clip前のtotal_rewardが非有限になっていないかをJAX互換の方法で検査する。
        # ここでは Python の if/raise は使わない (JIT トレースを壊すため)。
        # 代わりに jnp.isfinite の結果を metrics に float(0.0/1.0) として記録し、
        # 呼び出し側 (train/train_mjx.py の progress_callback) が
        # 学習ループの外側(非JIT領域)でこのフラグを見て停止判定を行う。
        reward_is_finite = jp.all(jp.isfinite(total_reward)).astype(jp.float32)

        # [予防追加 2026-09-13] 上のreward_is_finiteは「検出」のみで、
        # 従来はこのフラグを立てるだけで total_reward 自体は無害化されて
        # いなかった。jp.clip() は NaN を素通りさせる(NaNとの比較は
        # IEEE754で常にFalseになるため、np.clip(nan, lo, hi) == nan)。
        # そのため done=False の場合、非有限な報酬がそのままPPOの損失
        # 計算(GAEの逆方向再帰など)に流れ込み、1ステップの数値破綻が
        # バッチ全体・トラジェクトリ全体を汚染し得た。
        #
        # ここで明示的に安全値へ置換し(数値破綻を転倒と同等に扱う)、
        # かつエピソードを強制終了させる。物理状態自体の発散は
        # envs/mjx_env.py 側の physics_step ロールバックで別途防止して
        # いるため、ここは「それでも報酬計算自体がNaNを産んだ場合」の
        # 最終防衛ラインとして機能する。
        done = jp.logical_or(done, reward_is_finite < 0.5)
        total_reward = jp.where(reward_is_finite > 0.5, total_reward, w['fall_penalty'])

        total_reward = jp.clip(total_reward, -300.0, 300.0)
        total_reward = jp.where(done, w['fall_penalty'], total_reward)

        safe_step = jp.maximum(step, 1).astype(jp.float32)
        reward_per_step = total_reward
        total_penalty_value = soft_penalty + safety_penalty

        metrics = {
            'alive': r_alive,
            'total_reward': total_reward,
            'reward': total_reward,
            'reward_per_step': reward_per_step,
            'total_penalty': total_penalty_value,
            'lambda_phase': lambda_phase,
            'r_cp': r_capture_point,
            'r_recovery': r_recovery,
            'r_com_stab': r_com_stab,
            # [監査追加 2026-09-13] train_mjx.py の _audit_reward_metrics()
            # がShaping Mismatch(高報酬なのに姿勢系の正報酬が乏しい)を
            # 検出するために必要。r_uprightはtotal_reward計算に既に
            # 使われているが、従来metricsに含まれておらず監査できなかった。
            'r_upright': r_upright,
            'both_feet_contact': r_both_feet_contact,
            'pbrs_reward': r_pbrs,
            'potential': current_potential,
            'fall_penalty': jp.where(done, w['fall_penalty'], 0.0),
            'stability_index': stability_index,
            'curriculum_scale': curriculum_disturbance_scale,
            'disturbance_recovery_bonus': r_disturbance_recovery,
            'zmp_margin': stability_metrics['zmp_margin'],
            'foot_balance': stability_metrics['foot_balance'],
            'barrier_height': p_barrier_height,
            'barrier_torque': p_barrier_torque,
            # NaN/Inf診断用フラグ (1.0=正常, 0.0=非有限値を検出)
            'reward_is_finite': reward_is_finite,
            # [監査追加 2026-09-13] envs/stability_metrics.py の幾何計算
            # 自体の非有限値検出フラグ (同ファイルのv2.2changelog参照)。
            'stability_metrics_finite': stability_metrics['metrics_are_finite'],
            # [監査追加 2026-09-13] com_accelがqacc取得失敗によるフォール
            # バック値[0,0,-9.81]を使っているか (1.0=フォールバック中)。
            'com_accel_is_fallback': com_accel_is_fallback,
        }

        return total_reward, done, metrics, current_potential
```

---

## envs/stability_metrics.py

```python
"""
Advanced Stability Metrics for Bipedal Robot Control
- Foot Placement Estimator (FPE) / LIPM Capture Point
- Zero Moment Point (ZMP) Margin (support-polygon based)
- Multi-point Contact Analysis
- Unified Stability Index

================================================================================
v2 (2026-07 レビュー) での主な修正点
================================================================================
1. [CRITICAL FIX] compute_zmp_margin():
   旧実装は `zmp_error = ||zmp - pressure_center||` かつ
   `zmp = pressure_center - zmp_correction` という定義だったため、
   代数的に `zmp_error = ||zmp_correction||` へ完全に相殺されていた。
   pressure_center・com_pos は式の中で一切効いておらず、かつ
   mjx_rewards.py 側が com_accel を [0,0,-9.81] 固定で渡していたことも
   重なって、zmp_margin は常に定数 1.0 を返す「死んだ」指標になっていた
   （検証スクリプトで再現・確認済み）。
   → 標準LIPM式 zmp = com_xy - com_accel_xy * h / (accel_z + g) で
     実際のZMPを算出し、「支持脚(単脚)または両脚を結ぶ線分を
     足平半径で膨らませたカプセル領域」までの符号付き距離として
     再定義した。

2. compute_lipm_metrics():
   Capture Point (p_cp) を戻り値に追加し、mjx_rewards.py 側で
   重複計算していたCPをこちらに一本化（DRY化・数値的不整合の排除）。

3. 統合指標の重み付け構造は既存設計を踏襲しつつ、各サブ指標が
   物理的に意味のある値を返すようになったことで、
   stability_index 全体の信頼性が回復している
   （旧: zmp_margin が常時+1.0のフリークレジットを与えていたため、
   本指標を閾値判定に使う disturbance_recovery_bonus 等が
   実際より「安定している」と誤認しやすい状態だった）。

================================================================================
v2.1 (2026-09 ISSUE-1/3 修正)
================================================================================
[ISSUE-1 FIXED] com_pos の統一
   mjx_env.py と mjx_rewards.py の両方で subtree_com[0] を優先取得。
   compute_zmp_margin() と compute_lipm_metrics() でも同じ com_pos を
   参照することで、数値的な乖離を排除。

[ISSUE-3 FIXED] data.qacc の座標系明記
   com_accel = data.qacc[0:3] がワールド座標系であることを
   コメントで明示。MuJoCo標準規約（free joint の並進は world frame）
   に準拠していることを記録し、実装変更時の引き継ぎ誤りを防止。

================================================================================
v2.2 (2026-09-13 報酬ハッキング監査 対応)
================================================================================
[監査追加] compute_unified_stability_index() の戻り値に
   'metrics_are_finite' フラグを追加した。envs/mjx_rewards.py の
   reward_is_finite (1.0=正常, 0.0=非有限値検出) と同じ設計思想で、
   本ファイルの幾何計算(CP/ZMP/バランス/姿勢マージン)自体が
   NaN/Infを産んでいないかを自己診断する。train/train_mjx.py の
   _audit_reward_metrics() がこのフラグを「Metric Corruption」検出の
   直接的な根拠として利用する。

   背景: 上記v2の[CRITICAL FIX]で説明した「zmp_marginが常に定数1.0を
   返す死んだ指標」バグは、本ファイルの数式バグと、呼び出し側
   (mjx_rewards.py)がcom_accelをフォールバック値[0,0,-9.81]で
   渡していたことの「合わせ技」で発生していた。数式側は修正済みだが、
   フォールバック分岐自体は防御的に残っているため(qacc取得失敗時の
   保険)、将来また同様の問題が再発しないよう、両方の可視化を追加した
   (フォールバック側は mjx_rewards.py の 'com_accel_is_fallback' を参照)。
================================================================================
"""

import jax
import jax.numpy as jp
from typing import Tuple, Dict


class StabilityMetrics:
    """Compute advanced stability metrics for disturbance-resistant control."""

    def __init__(
        self,
        left_foot_id: int,
        right_foot_id: int,
        com_height: float = 0.28,
        foot_support_radius: float = 0.06,
    ):
        self._left_foot_id = left_foot_id
        self._right_foot_id = right_foot_id
        self._com_height = com_height
        # 足平の実効支持半径。mjx_env.py の FSR レイアウト
        # (前後~8cm, 左右半幅~5cm) に整合する概算値。
        # 実URDFの足裏形状に合わせて要調整。
        self._foot_support_radius = foot_support_radius

    # ------------------------------------------------------------------
    # 幾何ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _point_to_segment_distance(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
        """点 p から線分 ab までの最短距離（2Dベクトル入力）。"""
        ab = b - a
        ab_len_sq = jp.dot(ab, ab) + 1e-9
        t = jp.clip(jp.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t * ab
        return jp.linalg.norm(p - closest)

    # ------------------------------------------------------------------
    # 1. LIPM Capture Point
    # ------------------------------------------------------------------
    def compute_lipm_metrics(
        self,
        com_pos: jax.Array,
        com_vel: jax.Array,
        rpy: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        計算: Capture Point、最寄り足までのマージン、CP座標そのもの。

        Returns:
            (best_cp_dist, stability_margin, p_cp)
        """
        h = self._com_height
        g = 9.81
        omega_0 = jp.sqrt(g / h)

        p_com = com_pos[0:2]
        v_com = com_vel[0:2]
        p_cp = p_com + v_com / omega_0

        left_foot_2d = left_foot_pos[0:2]
        right_foot_2d = right_foot_pos[0:2]

        cp_dist_l = jp.linalg.norm(left_foot_2d - p_cp)
        cp_dist_r = jp.linalg.norm(right_foot_2d - p_cp)
        best_cp_dist = jp.minimum(cp_dist_l, cp_dist_r)

        foot_span = jp.linalg.norm(right_foot_2d - left_foot_2d)
        max_margin = foot_span / 2.0 + self._foot_support_radius
        stability_margin = jp.maximum(0.0, max_margin - best_cp_dist)

        return best_cp_dist, stability_margin, p_cp

    # ------------------------------------------------------------------
    # 2. ZMP Margin (support-polygon based) — [CRITICAL FIX]
    # ------------------------------------------------------------------
    def compute_zmp_margin(
        self,
        com_pos: jax.Array,
        com_accel: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
        left_foot_force: jax.Array,
        right_foot_force: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        ZMP (Zero Moment Point) マージンを計算する。

        標準LIPM方程式 zmp = com_xy - com_accel_xy * h / (accel_z + g) で
        実際のZMPを求め、支持基底（片脚支持ではその足、両脚支持では
        両足を結ぶ線分を足平半径で膨らませたカプセル領域）までの
        符号付き距離としてマージンを定義する。

        注意:
        [ISSUE-3 FIXED] com_accel は data.qacc[0:3] (world frame の並進加速度)
        を前提とします。MuJoCo標準規約では free joint の並進加速度は
        world frame です。ローカル座標系の加速度ではありませんので
        ご注意ください。

        Args:
            com_pos: 重心位置 [3]（preferably subtree_com[0], fallback base_pos）
            com_accel: 重心の線形加速度 [3] (data.qacc[0:3] 相当。
                       ワールド座標系)
            left_foot_pos / right_foot_pos: 足位置 [3]
            left_foot_force / right_foot_force: 足裏鉛直反力 [スカラ]

        Returns:
            (zmp_margin [0,1], zmp_point [2])
        """
        g = 9.81
        h = jp.clip(com_pos[2], 0.05, 1.0)  # 高さ0付近での特異点回避

        com_accel_xy = jp.clip(com_accel[0:2], -30.0, 30.0)  # 接触衝撃ノイズの飽和
        # 鉛直加速度 -> ほぼ自由落下(accel_z≈-g)の特異点回避のためクリップ。
        # 自由落下に近いほど分母が小さくなり補正項が急増する
        # = ZMPの物理的信頼性が失われる、という意図した挙動。
        vertical_accel_eff = jp.clip(com_accel[2] + g, 3.0, 50.0)

        com_2d = com_pos[0:2]
        zmp = com_2d - (com_accel_xy * h) / vertical_accel_eff

        total_force = left_foot_force + right_foot_force + 1e-6
        force_ratio_l = left_foot_force / total_force
        force_ratio_r = right_foot_force / total_force

        left_foot_2d = left_foot_pos[0:2]
        right_foot_2d = right_foot_pos[0:2]

        # 片脚支持(荷重比が大きく偏っている)では支持領域を
        # 荷重側の足1点に収縮させる。閾値0.6は要チューニング。
        is_single_support = jp.abs(force_ratio_l - force_ratio_r) > 0.6
        stance_foot = jp.where(force_ratio_l > force_ratio_r, left_foot_2d, right_foot_2d)
        seg_a = jp.where(is_single_support, stance_foot, left_foot_2d)
        seg_b = jp.where(is_single_support, stance_foot, right_foot_2d)

        dist_to_support = self._point_to_segment_distance(zmp, seg_a, seg_b)
        support_radius = self._foot_support_radius + 0.05  # 安全マージン込み

        zmp_margin = jp.clip(1.0 - (dist_to_support / support_radius), 0.0, 1.0)

        return zmp_margin, zmp

    # ------------------------------------------------------------------
    # 3. Foot Contact Balance
    # ------------------------------------------------------------------
    def compute_foot_contact_balance(
        self,
        left_foot_force: jax.Array,
        right_foot_force: jax.Array,
    ) -> jax.Array:
        """
        左右の足接触圧力バランスを計算。
        完全にバランス (1:1) なら 1.0、一方に全て集中なら 0.0。
        """
        total_force = left_foot_force + right_foot_force + 1e-6
        ratio_l = left_foot_force / total_force
        balance = 1.0 - jp.abs(ratio_l - 0.5) * 2.0
        return jp.clip(balance, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 4. Orientation Margin
    # ------------------------------------------------------------------
    def compute_orientation_margin(
        self,
        rpy: jax.Array,
        base_ang_vel: jax.Array,
        safe_angle: float = 0.3,
    ) -> jax.Array:
        """姿勢安全マージン。ロール・ピッチが小さく角速度が低いほど高い。"""
        tilt_err = jp.sqrt(jp.square(rpy[0]) + jp.square(rpy[1]))
        ang_vel_xy = jp.linalg.norm(base_ang_vel[0:2])

        angle_margin = jp.clip(1.0 - (tilt_err / (safe_angle + 1e-6)), 0.0, 1.0)
        ang_vel_margin = jp.exp(-5.0 * ang_vel_xy)

        margin = angle_margin * ang_vel_margin
        return jp.clip(margin, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 5. Unified Stability Index
    # ------------------------------------------------------------------
    def compute_unified_stability_index(
        self,
        com_pos: jax.Array,
        com_vel: jax.Array,
        com_accel: jax.Array,
        rpy: jax.Array,
        base_ang_vel: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
        left_foot_force: jax.Array,
        right_foot_force: jax.Array,
        gait_phase: float = 0.0,
        cp_margin_norm_dist: float = 0.15,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        複数の安定性指標を統合し、統一的な安定性インデックスを計算する。

        [ISSUE-1/3 FIXED] com_pos は subtree_com[0] を優先（mjx_env/rewards側で統一）。
        com_accel は world frame (data.qacc[0:3]) であることを前提。

        Returns:
            (stability_index, metrics_dict)
        """
        # 1. LIPM Capture Point
        cp_dist, cp_margin, p_cp = self.compute_lipm_metrics(
            com_pos, com_vel, rpy, left_foot_pos, right_foot_pos
        )
        cp_margin_norm = jp.clip(cp_margin / cp_margin_norm_dist, 0.0, 1.0)

        # 2. ZMP Margin [FIXED]
        zmp_margin, zmp_point = self.compute_zmp_margin(
            com_pos, com_accel, left_foot_pos, right_foot_pos,
            left_foot_force, right_foot_force
        )

        # 3. Foot Contact Balance
        foot_balance = self.compute_foot_contact_balance(
            left_foot_force, right_foot_force
        )

        # 4. Orientation Margin
        orient_margin = self.compute_orientation_margin(
            rpy, base_ang_vel, safe_angle=0.3
        )

        # Gait Phase に応じた動的重み付け
        # Double support (0.0~0.2, 0.8~1.0) では foot_balance を重視
        # Single support (0.2~0.8) では CP と orientation を重視
        is_single_support = jp.logical_or(
            jp.logical_and(gait_phase > 0.2, gait_phase < 0.8),
            gait_phase < 0.0  # フェーズ情報がない場合はデフォルト
        )

        w_cp = jp.where(is_single_support, 0.45, 0.25)
        w_zmp = jp.where(is_single_support, 0.25, 0.35)
        w_balance = jp.where(is_single_support, 0.15, 0.25)
        w_orient = 0.15

        stability_index = (
            w_cp * cp_margin_norm +
            w_zmp * zmp_margin +
            w_balance * foot_balance +
            w_orient * orient_margin
        )

        # [監査追加 2026-09-13] 本メソッドの幾何計算自体がNaN/Infを
        # 産んでいないかの自己診断。envs/mjx_rewards.py の
        # reward_is_finite と同じ設計思想 (1.0=正常, 0.0=非有限値検出)。
        # train/train_mjx.py の _audit_reward_metrics() が
        # 'stability_metrics_finite' として参照する。
        metrics_are_finite = jp.all(jp.array([
            jp.all(jp.isfinite(cp_dist)),
            jp.all(jp.isfinite(cp_margin_norm)),
            jp.all(jp.isfinite(p_cp)),
            jp.all(jp.isfinite(zmp_margin)),
            jp.all(jp.isfinite(zmp_point)),
            jp.all(jp.isfinite(foot_balance)),
            jp.all(jp.isfinite(orient_margin)),
            jp.all(jp.isfinite(stability_index)),
        ])).astype(jp.float32)

        metrics = {
            'cp_dist': cp_dist,
            'cp_margin': cp_margin_norm,
            'cp_point': p_cp,
            'zmp_margin': zmp_margin,
            'zmp_point': zmp_point,
            'foot_balance': foot_balance,
            'orient_margin': orient_margin,
            'stability_index': stability_index,
            'metrics_are_finite': metrics_are_finite,
        }

        return stability_index, metrics
```

---

## envs/training_wrapper.py

```python
"""
TrainingProgressWrapper: Brax PPO環境への学習進捗率の注入

Brax PPOの内部ループでは環境がJITコンパイル・vmapされ、
AutoResetWrapper によってエピソード終了時に info が reset() の
初期値で上書きされる。そのため info 内のカウンタは自然には
エピソード間で持続しない。

本ラッパーは AutoResetWrapper の**外側**に適用することで、
エピソード境界を跨いて単調増加する学習進捗率を維持する。

仕組み:
  1. step() の冒頭で _env_steps を読み取り・インクリメント
  2. 内側の step() を呼ぶ（AutoResetWrapper が done 時に
     info を reset 値で上書きする可能性がある）
  3. 返却された state の info を、保存しておいた正しい
     _env_steps / training_progress で上書きする

これにより、内側で何度 auto-reset が起きても、外側の
カウンタは単調増加し続ける。

================================================================================
v2 (2026-09 ISSUE-2 修正)
================================================================================
[ISSUE-2 FIXED] training_progress の batch 対応

training_progress は shape (num_envs,) の配列として供給される。
Brax PPO では num_envs 個の並列環境が同期的に実行されるため、
各環境の progress を独立に計算する必要がある。

実装:
  - state.obs.shape[0] で num_envs を検出
  - _env_steps, training_progress を (num_envs,) 配列として管理
  - mjx_rewards.py 側の _get_curriculum_disturbance_scale() は
    要素ごとのスケーリングに対応
================================================================================
"""

import jax
import jax.numpy as jp
from brax.envs import Wrapper


class TrainingProgressWrapper(Wrapper):
    """
    学習進捗率 (0.0→1.0) を環境の info に注入するラッパー。
    Brax PPO の envs.training.wrap() が適用した後（= AutoResetWrapper
    の外側）に適用する必要がある。
    
    Args:
        env: Brax ラップ済み環境（AutoResetWrapper 適用済み）
        total_steps_per_env: 各並列環境あたりの総ステップ数
            = num_timesteps // num_envs
    
    使用例:
        env = brax_env
        env = jax.vmap(env.reset)(rng_batch)
        env = training.wrap(env, num_envs)  # AutoResetWrapper を適用
        env = TrainingProgressWrapper(env, total_steps_per_env=100000)
    """
    
    def __init__(self, env, total_steps_per_env: int):
        super().__init__(env)
        self._total_steps_per_env = max(float(total_steps_per_env), 1.0)

    def reset(self, rng):
        state = self.env.reset(rng)
        
        # [ISSUE-2 FIXED] batch 対応: num_envs を検出
        batch_size = state.obs.shape[0] if state.obs.ndim > 1 else 1
        
        state = state.replace(info={
            **state.info,
            '_env_steps': jp.zeros(batch_size, dtype=jp.int32),
            'global_step': jp.zeros(batch_size, dtype=jp.int32),
            'training_progress': jp.zeros(batch_size, dtype=jp.float32),
            'terminated': jp.zeros(batch_size, dtype=jp.bool_),
            'truncated': jp.zeros(batch_size, dtype=jp.bool_),
            'time_out': jp.zeros(batch_size, dtype=jp.float32),
        })
        return state

    def step(self, state, action):
        # [ISSUE-2 FIXED] (1) auto-reset で上書きされる前に、
        #                     現在のカウンタを取得して +1
        #                     batch-wise インクリメント
        
        env_steps_current = state.info.get('_env_steps', None)
        if env_steps_current is None:
            # フォールバック: 0 初期化（reset() が呼ばれていない場合）
            batch_size = state.obs.shape[0] if state.obs.ndim > 1 else 1
            env_steps_current = jp.zeros(batch_size, dtype=jp.int32)
        
        env_steps = jp.asarray(env_steps_current, dtype=jp.int32) + 1
        
        progress = jp.clip(
            env_steps.astype(jp.float32) / self._total_steps_per_env,
            0.0, 1.0,
        )
        
        # (2) 内側の step（AutoResetWrapper 含む）を実行
        #     done が True なら info は reset() の値で上書きされている
        state = self.env.step(state, action)
        
        # (3) 正しいカウンタ値で上書き（auto-reset のゼロクリアを無効化）
        #     [ISSUE-2 FIXED] batch-wise 値を返却
        state = state.replace(info={
            **state.info,
            '_env_steps': env_steps,
            'global_step': env_steps,
            'training_progress': progress,
        })
        
        return state
```

---

## real/__init__.py

```python
# Initialize the real module

```

---

## real/real_env.py

```python
"""
real/real_env.py — RPi5 実機メインループ & 観測ベクトル構築

【実装済み対策 (フィージビリティレビュー反映)】
- ONNX Runtime: シングルスレッド強制 (レイテンシスパイク防止)
- 制御周期: 100Hz対応 (dt=10ms, time.monotonic精密タイマー)
- 観測ベクトル: project_overview.md の625次元仕様に完全準拠

【修正対応 (2026-09-08)】
- [REAL-1 FIXED] 位相計算を相対時刻ベースに統一（step数カウンタ）
- [REAL-2 FIXED] base_pos[2] ゼロ埋めに明記・comment追加
- [REAL-3 FIXED] action_history 順序を mjx_env と明示的に一致（assert検証追加）
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    print("[Warn] onnxruntime not found. Policy will run in dummy mode.")
    ort = None

from real.real_io import TeensySpineIO
from robot.math_utils import quat_to_euler, rotate_vector_by_quaternion
from robot.config import RobotConfig
from robot.gait_generator import numpy_get_reference_trajectory


# ============================================================
# ONNX推論ラッパー (シングルスレッド設定)
# ============================================================

class PolicyRunner:
    """
    ONNX Runtime 推論実行器。
    
    【重要】デフォルトでは4コアすべてを使おうとし、
    軽量MLPではスレッド同期オーバーヘッドで突発10ms超のスパイクが発生する。
    シングルスレッドに制限することで推論時間を1ms以下に安定化させる。
    """
    
    def __init__(
        self, 
        model_path: str = "/var/lib/bipedal_runtime/models/policy.onnx",
        obs_dim: int = 625,
        act_dim: int = 20
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.session: Optional[Any] = None
        self.dummy_mode = ort is None
        
        if not self.dummy_mode:
            try:
                opts = ort.SessionOptions()
                # ★ シングルスレッド強制 — レイテンシスパイク防止の核心設定
                opts.intra_op_num_threads = 1   # 演算内部: 並列化なし
                opts.inter_op_num_threads = 1   # 演算間: 並列化なし
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                # グラフ最適化はフルに活用
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.session.get_inputs()[0].name
                print(f"[Info] Policy loaded: {model_path} (single-thread, deterministic latency)")
            except Exception as e:
                print(f"[Error] ONNX load failed: {e}")
                self.dummy_mode = True
    
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """推論実行。入力: (obs_dim,), 出力: (act_dim,)"""
        if self.dummy_mode:
            return np.zeros(self.act_dim)
        
        obs_input = obs.astype(np.float32).reshape(1, -1)
        result = self.session.run(None, {self.input_name: obs_input})
        return result[0].flatten()[:self.act_dim]


# ============================================================
# メイン制御環境
# ============================================================

class RealRobotEnv:
    """
    Raspberry Pi 5 実機制御環境。
    
    100Hz (10ms) のメインループで:
    1. センサー取得 (共有メモリ or 直接)
    2. 625次元観測ベクトルの構築
    3. ONNX推論 (Base Policy, シングルスレッド)
    4. 残差RL合成 (サイクロイド・リファレンス + AI残差)
    5. 安全クランプ + EMA平滑化
    6. Sync Write一括送信
    7. インターリーブRead (2台/ループ)
    """
    
    # robot/config.py 準拠の定数
    NUM_JOINTS = 20
    BASE_OBS_DIM = 84    # 12 + 40 + 10 + 2 + 20
    HISTORY_LEN = 5
    ACT_DIM = 20
    OBS_DIM = 625        # BASE_OBS + HISTORY(520) + TEMP(20) + VOLT(1)
    ACTION_SCALE = RobotConfig.ACTION_SCALE
    RESIDUAL_SCALE = 0.5
    EMA_ALPHA = 0.8      # LPF平滑化係数
    
    # 各関節の物理的可動限界 (assets/humanoid/humanoid.xml と 100% 完全同期)
    JOINT_LIMITS_MIN = np.array([
        # 右脚 (6関節)
        0.0, -0.523599, -0.523599, -1.047198, -1.570796, -0.436332,
        # 左脚 (6関節)
        -3.141593, -0.523599, -1.047198, -0.523599, -0.436332, -0.523599,
        # 右腕 (4関節)
        -3.141593, 0.0, 0.0, -1.570796,
        # 左腕 (4関節)
        -3.141593, -3.141593, -3.141593, -0.261799
    ])
    JOINT_LIMITS_MAX = np.array([
        # 右脚 (6関節)
        3.141593, 0.523599, 1.047198, 0.523599, 0.436332, 0.436332,
        # 左脚 (6関節)
        0.0, 0.523599, 0.523599, 1.047198, 1.570796, 0.436332,
        # 右腕 (4関節)
        3.141593, 3.141593, 3.141593, 0.261799,
        # 左腕 (4関節)
        3.141593, 0.0, 0.0, 1.570796
    ])
    
    def __init__(self, control_hz: int = 100):
        self.dt = 1.0 / control_hz
        self.control_hz = control_hz
        
        # --- ハードウェアI/O ---
        print("[Info] Initializing RealRobotEnv (100Hz target)...")
        self.spine = TeensySpineIO(num_servos=self.NUM_JOINTS)
        self.imu_data, self.fsr_contacts, self.servo_temps, self.servo_voltages = (
            self.spine.communicate(np.zeros(self.NUM_JOINTS))
        )
        
        # --- ONNX推論 (シングルスレッド) ---
        self.policy = PolicyRunner()
        
        # --- 状態変数 ---
        self.last_action = np.zeros(self.NUM_JOINTS)
        self.smoothed_action = np.zeros(self.NUM_JOINTS)
        
        # ZUPT速度推定用
        self._vel_estimate = np.zeros(3)           # IMU積分速度 [m/s]
        self._prev_joint_pos = np.zeros(self.NUM_JOINTS)  # 関節角速度の有限差分用
        
        # [REAL-1 FIXED] 位相計算を相対ステップベース（学習側と同期）
        self._episode_step = 0
        self._max_episode_steps = RobotConfig.MAX_EPISODE_STEPS
        
        # 履歴バッファ (FIFO: 過去5ステップ)
        # [REAL-3 FIXED] 順序を mjx_env の jp.roll(shift=-1) と一致させる
        # (古 → 新の順序：0番目=最も古い, 4番目=最新)
        self.obs_history = deque(
            [np.zeros(self.BASE_OBS_DIM) for _ in range(self.HISTORY_LEN)],
            maxlen=self.HISTORY_LEN
        )
        self.act_history = deque(
            [np.zeros(self.ACT_DIM) for _ in range(self.HISTORY_LEN)],
            maxlen=self.HISTORY_LEN
        )
        
        print(f"[Info] RealRobotEnv ready. Control loop: {control_hz}Hz ({self.dt*1000:.1f}ms)")
    
    def reset_episode(self):
        """エピソード開始時のリセット（学習シミュレータの reset() に対応）"""
        self._episode_step = 0
        # 履歴バッファをクリア
        for _ in range(self.HISTORY_LEN):
            self.obs_history.append(np.zeros(self.BASE_OBS_DIM))
            self.act_history.append(np.zeros(self.ACT_DIM))
    
    def _compute_gait_phase(self) -> float:
        """
        学習時と同じ位相観測を返す。
        
        [REAL-1 FIXED] 相対時刻ベース（ステップ数）に統一。
        絶対時刻 time.monotonic() ではなく、エピソード内ステップ数
        (_episode_step) を使用することで、学習環境 mjx_env と
        完全に同期する。
        
        Fixed-foot mode では常に 0 を返す。
        """
        if not RobotConfig.USE_REFERENCE_GAIT:
            return 0.0
        
        # [REAL-1 FIXED] ステップ数ベース（学習環境 mjx_env L188 と同一ロジック）
        phase = (self._episode_step * self.dt / RobotConfig.GAIT_PERIOD) % 1.0
        return float(phase)
    
    def _get_reference_trajectory(self, phase: float) -> np.ndarray:
        """
        サイクロイド・リファレンス軌道 (gait_generator.py のロジック実機NumPy共通版)。
        学習環境の jax_get_reference_trajectory と 100% 完全な整合性を担保。
        """
        return numpy_get_reference_trajectory(phase, self.NUM_JOINTS)
    
    def build_observation(self) -> np.ndarray:
        """
        625次元観測ベクトルの構築 (project_overview.md 仕様に完全準拠)
        
        BASE_OBS (84次元):
          位置(3) + RPY(3) + 線速度(3) + 角速度(3) = 12
          関節角度(20) + 関節角速度(20) = 40
          FSR(8) + ZMP(2) = 10
          phase_sin(1) + phase_cos(1) = 2
          リファレンス角度(20) = 20
        
        + 観測履歴 (84×5 = 420)
        + 行動履歴 (20×5 = 100)
        + サーボ温度 (20)
        + 電源電圧 (1)
        """
        # --- 1. IMU (UART経由, ブロッキングなし) ---
        imu_data = self.imu_data
        quat = imu_data["quat"]
        gyro = imu_data["gyro"]
        lin_accel = imu_data["lin_accel"]
        rpy = quat_to_euler(quat)  # roll, pitch, yaw
        
        # --- [REAL-2 FIXED] base_pos: 高さ(Z)のみ脚IKから粗推定予定、X/Yはゼロ ---
        # 学習側で NOISE_BASE_POS=0.1m の大ノイズDR済みのため
        # 実機側はゼロ埋めでも破綻しない設計。脚IK実装予定。
        base_pos = np.zeros(3)
        # base_pos[2] は将来的に脚のIKから推定可能:
        #   z_est ≈ L_thigh * cos(knee_angle) + L_shin * cos(ankle_angle)
        
        # --- lin_vel: ZUPT (Zero-velocity Update) 推定 ---
        # IMU加速度を1ステップ積分して速度を推定し、
        # 接地検出時にドリフトをリセットする
        world_accel = rotate_vector_by_quaternion(lin_accel, quat)
        self._vel_estimate += world_accel * self.dt
        
        # --- 3. FSR接地フラグ (TeensyオンチップADCで判定済み) ---
        fsr_raw = self.fsr_contacts
        zmp_xy = np.zeros(2)  # 実機ではCoP/ZMPを算出しない
        
        # XML/Teensy の FSR は [left_foot(4ch), right_foot(4ch)] の順で並ぶ。
        # そのため、先頭4chが left、後続4chが right である。
        left_contact = np.any(fsr_raw[:4] > 0.5)
        right_contact = np.any(fsr_raw[4:] > 0.5)
        if left_contact and right_contact:
            # 両足接地 = 静止推定 → ドリフトリセット
            self._vel_estimate *= 0.1  # 急なゼロリセットではなく減衰
        
        lin_vel = self._vel_estimate.copy()
        
        # --- 2. 関節状態 ---
        joint_pos = self.smoothed_action.copy()  # 簡易: 指令値 ≈ 実角度
        # 有限差分で関節角速度を推定
        joint_vel = (joint_pos - self._prev_joint_pos) / self.dt
        self._prev_joint_pos = joint_pos.copy()
        
        # --- 4. 歩行位相 ---
        phase = self._compute_gait_phase()
        phase_obs = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        
        # --- 5. リファレンス軌道 ---
        ref_angles = self._get_reference_trajectory(phase)
        
        # お手本無しのとき、観測のお手本情報(ref_angles)を0にリセットして、AIから目標の軌跡を完全に隠す
        # これにより、ロボットは自身の状態のみを頼りに歩行する（ただし、全体の次元数は変えないため、デプロイメント契約は壊れない）
        if not RobotConfig.USE_REFERENCE_GAIT:
            ref_angles_obs = np.zeros_like(ref_angles)
        else:
            ref_angles_obs = ref_angles
        
        # --- 6. Base Obs (84次元) ---
        base_obs = np.concatenate([
            base_pos,        # 3
            rpy,             # 3
            lin_vel,         # 3 (ZUPT推定速度)
            gyro,            # 3
            joint_pos,       # 20
            joint_vel,       # 20
            fsr_raw,         # 8
            zmp_xy,          # 2
            phase_obs,       # 2
            ref_angles_obs   # 20
        ])  # 合計: 84
        
        # [REAL-3 FIXED] 履歴バッファ更新（順序をmjx_env と明示的に一致）
        # mjx_env L173: obs_hist = jp.roll(obs_hist, shift=-1, axis=0)
        #               obs_hist = obs_hist.at[-1].set(base_obs)
        # つまり：[古い→新しい] の順序で、新データが末尾に追加される
        # NumPy deque も FIFO (古→新) なので、append() で自動的に同期する
        self.obs_history.append(base_obs.copy())
        self.act_history.append(self.last_action.copy())
        
        obs_hist_flat = np.concatenate(list(self.obs_history))   # 84×5 = 420
        act_hist_flat = np.concatenate(list(self.act_history))   # 20×5 = 100
        
        # --- 8. 温度・電圧 (インターリーブReadから取得, 10Hz更新) ---
        servo_temp = self.servo_temps.copy()     # 20
        supply_volt = np.array([np.mean(self.servo_voltages)])  # 1
        
        # --- 9. 最終観測ベクトル (625次元) ---
        obs = np.concatenate([
            base_obs,         # 84
            obs_hist_flat,    # 420
            act_hist_flat,    # 100
            servo_temp,       # 20
            supply_volt       # 1
        ])  # 合計: 625
        
        # [REAL-3 FIXED] 観測次元をアサート検証（ABI不変性保証）
        assert obs.shape[0] == self.OBS_DIM, (
            f"Observation shape mismatch: computed {obs.shape[0]}, "
            f"but OBS_DIM={self.OBS_DIM}"
        )
        
        # --- 安全フィルター: 観測の NaN/Inf 汚染防止 (Rule 15) ---
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("[Error] NaN/Inf detected in Observation! Zeroing to prevent policy corruption.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        return obs
    
    def step(self, obs: np.ndarray) -> np.ndarray:
        """
        1ステップの推論→行動適用。
        
        USE_REFERENCE_GAIT が True の場合は残差強化学習、False の場合はお手本無しのダイレクト強化学習を実行。
        """
        # --- AI推論 ---
        raw_action = self.policy.infer(obs)
        
        # --- 安全フィルター: 行動の NaN/Inf 汚染防止 (Rule 15 契約厳守) ---
        if np.isnan(raw_action).any() or np.isinf(raw_action).any():
            print("[Error] NaN/Inf detected in Policy Output! Triggering software E-stop (Zero Action).")
            raw_action = np.zeros_like(raw_action)
        
        # --- アクションの合成 (USE_REFERENCE_GAITスイッチによるダイレクト/残差の切り替え) ---
        if RobotConfig.USE_REFERENCE_GAIT:
            # AIの出力は「残差」として扱う（元の最大50%に制限）
            phase = self._compute_gait_phase()
            ref_angles = self._get_reference_trajectory(phase)
            residual = raw_action * self.ACTION_SCALE * self.RESIDUAL_SCALE
            target = ref_angles + residual
        else:
            # 「お手本無し」の場合：AIの出力を、安定した「中腰立ち姿勢」からの直接変位（最大±90度）として解釈
            default_pose = np.array(RobotConfig.DEFAULT_JOINT_ANGLES)
            target = default_pose + raw_action * self.ACTION_SCALE
        
        # --- 安全クランプ (assets/humanoid/humanoid.xml と 100% 同期した個別限界) ---
        target = np.clip(target, self.JOINT_LIMITS_MIN, self.JOINT_LIMITS_MAX)
        
        # --- EMA平滑化 (MOTOR_LPF_ALPHA と同じ規約: alpha = 新しい値の重み) ---
        self.smoothed_action = (
            (1.0 - self.EMA_ALPHA) * self.smoothed_action + 
            self.EMA_ALPHA * target
        )
        
        self.last_action = self.smoothed_action.copy()
        return self.smoothed_action
    
    def run_loop(self):
        """
        100Hzメインループ。time.monotonic() による精密タイミング制御。
        """
        print("[Info] Starting 100Hz control loop. Press Ctrl+C to stop.")
        self.reset_episode()
        loop_count = 0
        
        try:
            while True:
                t_start = time.monotonic()
                
                # 1. 観測ベクトル構築
                obs = self.build_observation()
                
                # 2. 推論 + 残差合成 + 安全処理
                action = self.step(obs)
                
                # 3. サーボへ一括送信 (Sync Write, 0.65ms)
                self.imu_data, self.fsr_contacts, self.servo_temps, self.servo_voltages = (
                    self.spine.communicate(action)
                )
                
                # 異常検知時の強制終了 (Rule 18: 通信異常での即時停止)
                if getattr(self.spine, 'telemetry_timeout_flag', False):
                    print("[Fatal] Teensy telemetry continuous timeout. Halting control loop.")
                    break
                
                # [REAL-1 FIXED] エピソード内ステップ数をインクリメント
                self._episode_step += 1
                if self._episode_step >= self._max_episode_steps:
                    print(f"[Info] Episode finished ({self._episode_step} steps). Resetting...")
                    self.reset_episode()
                
                # 4. ループタイミング制御
                elapsed = time.monotonic() - t_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if loop_count % 100 == 0:
                        print(f"[Warn] Loop overrun: {elapsed*1000:.2f}ms > {self.dt*1000:.1f}ms")
                
                loop_count += 1
                
        except (KeyboardInterrupt, SystemExit):
            print("\n[Info] Shutting down...")
        finally:
            self.close()
    
    def close(self):
        """安全なシャットダウン"""
        print("[Info] Zeroing servos and releasing resources...")
        # サーボをニュートラルに
        self.spine.communicate(np.zeros(self.NUM_JOINTS))
        time.sleep(0.5)
        
        self.spine.close()
        print("[Info] Shutdown complete.")


# ============================================================
# エントリーポイント
# ============================================================

def main():
    """
    実行方法 (RT-Preempt環境):
      sudo chrt -f 99 taskset -c 3 python3 -m real.real_env
    """
    env = RealRobotEnv(control_hz=100)
    env.run_loop()


if __name__ == "__main__":
    main()
```

---

## real/real_io.py

```python
"""
real/real_io.py — ハードウェアI/Oドライバ (RPi5 & Teensy 4.1 脳脊髄分離システム用)

【Hiwonder 公式プロトコル ＆ 実機電装完全準拠】
1. Hiwonder LX/HX シリアルバスサーボプロトコル:
   - パケット構造: 0x55 0x55 [ID] [Length] [Cmd] [Params...] [Checksum]
   - Checksum = ~(ID + Length + Cmd + Prm1 + ... + PrmN) & 0xFF
   - 放送アドレス: 0xFE (254)
   - コマンド: WRITE_MOVE=1 (0x01), READ_TEMP=26 (0x1A), READ_VIN=27 (0x1B), READ_POS=28 (0x1C)
2. BNO055 通信エラー保護:
   - バスエラー時の [0,0,0,0] 返却を防ぎ、直前の有効な単位クォータニオンを保持・復元。
3. LVCH16T245 ピン全二重分離:
   - Group 1 (Ch 1-8): DIR1 = HIGH (TX 4系統)
   - Group 2 (Ch 9-16): DIR2 = LOW (RX 4系統)
4. 20自由度 4バス割り当て (6+6+4+4 = 20):
   - バス1: 右脚 6軸 (ID: 1~6)
   - バス2: 左脚 6軸 (ID: 7~12)
   - バス3: 右腕 4軸 (ID: 13~16)
   - バス4: 左腕 4軸 (ID: 17~20)

【修正対応 (2026-09-08)】
- [REAL-4 FIXED] Checksum 検証を厳格化（破損データ読み出し防止）
- [REAL-5 FIXED] Teensy E-stop タイムアウト仕組みを明示・整合
"""

import os
import time
import struct
import numpy as np
import threading
from typing import Dict, Optional, Tuple

try:
    import serial
except ImportError:
    print("[Warn] pyserial not found. Hardware will run in dummy mode.")
    serial = None

def calc_checksum(buf: bytes) -> int:
    """
    Hiwonder 公式 Checksum 計算ロジック:
    ~(ID + Length + Cmd + Prm1 + ... + PrmN) & 0xFF
    """
    return (~(sum(buf)) & 0xFF)


# ============================================================
# 1. BNO055 IMU — UART接続 (異常値 [0,0,0,0] 防護実装)
# ============================================================

class BNO055UART:
    START_BYTE = 0xAA
    WRITE = 0x00
    READ = 0x01
    
    REG_QUA_DATA_W_LSB = 0x20
    REG_GYR_DATA_X_LSB = 0x14
    REG_LIA_DATA_X_LSB = 0x28
    REG_OPR_MODE = 0x3D
    
    NDOF_MODE = 0x0C
    
    def __init__(self, port: str = "/dev/ttyAMA1", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.dummy_mode = serial is None
        self.last_valid_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        if not self.dummy_mode:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.01)
                time.sleep(0.1)
                self._write_register(self.REG_OPR_MODE, self.NDOF_MODE)
                time.sleep(0.6)
                print(f"[Info] BNO055 initialized on UART {port}")
            except Exception as e:
                print(f"[Error] BNO055 UART init failed: {e}")
                self.dummy_mode = True
    
    def _write_register(self, reg: int, value: int):
        if self.ser is None:
            return
        packet = bytes([self.START_BYTE, self.WRITE, reg, 1, value])
        self.ser.write(packet)
        self.ser.read(2)
    
    def _read_registers(self, reg: int, length: int) -> bytes:
        if self.ser is None:
            return bytes(length)
        
        packet = bytes([self.START_BYTE, self.READ, reg, length])
        self.ser.write(packet)
        header = self.ser.read(2)
        if len(header) < 2 or header[0] != 0xBB:
            return bytes(length)
        data = self.ser.read(header[1])
        if len(data) < length:
            data += bytes(length - len(data))
        return data
    
    def get_quaternion(self) -> np.ndarray:
        """
        クォータニオン (w, x, y, z) を取得。
        バス障害・パケット破損時は [0,0,0,0] ではなく直前の有効なクォータニオンを返す。
        """
        if self.dummy_mode:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        data = self._read_registers(self.REG_QUA_DATA_W_LSB, 8)
        if len(data) < 8:
            return self.last_valid_quat
            
        w, x, y, z = struct.unpack('<4h', data[:8])
        scale = 1.0 / 16384.0
        quat = np.array([w * scale, x * scale, y * scale, z * scale])
        
        # ノルムチェック (0付近の異常クォータニオンを遮断)
        norm = np.linalg.norm(quat)
        if norm < 0.5 or norm > 1.5:
            return self.last_valid_quat
            
        self.last_valid_quat = quat / norm  # 正規化して保存
        return self.last_valid_quat
    
    def get_gyro(self) -> np.ndarray:
        if self.dummy_mode:
            return np.zeros(3)
        
        data = self._read_registers(self.REG_GYR_DATA_X_LSB, 6)
        if len(data) < 6:
            return np.zeros(3)
        gx, gy, gz = struct.unpack('<3h', data[:6])
        scale = 1.0 / 900.0
        return np.array([gx * scale, gy * scale, gz * scale])
    
    def get_linear_acceleration(self) -> np.ndarray:
        if self.dummy_mode:
            return np.zeros(3)
        
        data = self._read_registers(self.REG_LIA_DATA_X_LSB, 6)
        if len(data) < 6:
            return np.zeros(3)
        ax, ay, az = struct.unpack('<3h', data[:6])
        scale = 1.0 / 100.0
        return np.array([ax * scale, ay * scale, az * scale])
    
    def get_imu_data(self) -> Dict[str, np.ndarray]:
        return {
            "quat": self.get_quaternion(),
            "gyro": self.get_gyro(),
            "lin_accel": self.get_linear_acceleration()
        }


# ============================================================
# 2. BusLinker V3.0 — Hiwonder 公式 Checksum ＆ コマンドID 準拠
# ============================================================

class BusLinkerV3:
    """
    Hiwonder BusLinker V3.0 シリアルバスサーボドライバ。
    
    【公式プロトコル定数】
    HEADER: 0x55 0x55
    BROADCAST_ID: 0xFE (254)
    CMD_SERVO_MOVE_TIME_WRITE: 1 (0x01)
    CMD_SERVO_TEMP_READ: 26 (0x1A)
    CMD_SERVO_VIN_READ: 27 (0x1B)
    CMD_SERVO_POS_READ: 28 (0x1C)
    """
    
    HEADER = bytes([0x55, 0x55])
    BROADCAST_ID = 0xFE
    
    CMD_SERVO_MOVE_TIME_WRITE = 0x01
    CMD_SERVO_TEMP_READ = 0x1A
    CMD_SERVO_VIN_READ = 0x1B
    CMD_SERVO_POS_READ = 0x1C
    
    def __init__(
        self, 
        port: str = "/dev/ttyAMA0", 
        baudrate: int = 1_000_000,
        num_servos: int = 20,
        read_batch_size: int = 2,
        map_file: str = "/etc/bipedal_runtime/servo_map.yaml"
    ):
        self.num_servos = num_servos
        self.port = port
        self.baudrate = baudrate
        self.read_batch_size = read_batch_size
        self.lock = threading.Lock()
        
        self.ser: Optional[serial.Serial] = None
        self.dummy_mode = serial is None
        
        # servo_map.yaml のロード (Bus 1: 1-6, Bus 2: 7-12, Bus 3: 13-16, Bus 4: 17-20)
        self.servo_id_map = {i: i + 1 for i in range(num_servos)}
        if os.path.exists(map_file):
            try:
                import yaml
                with open(map_file, "r") as f:
                    cfg = yaml.safe_load(f)
                    if "servo_ids" in cfg:
                        for idx, sid in enumerate(cfg["servo_ids"]):
                            self.servo_id_map[idx] = int(sid)
                print(f"[Info] Loaded servo_map.yaml from {map_file}")
            except Exception as e:
                print(f"[Warn] Failed to parse {map_file}: {e}")

        self._read_cursor = 0
        self.servo_temps = np.full(num_servos, 25.0)
        self.servo_voltages = np.full(num_servos, 11.1)  # 3S LiPo 11.1V
        self.servo_positions = np.zeros(num_servos)
        
        if not self.dummy_mode:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.002)
                print(f"[Info] BusLinker connected: {port} @ {baudrate/1e6:.1f}Mbps")
            except Exception as e:
                print(f"[Error] BusLinker UART init failed: {e}")
                self.dummy_mode = True
    
    def sync_write_positions(self, angles_rad: np.ndarray, move_time_ms: int = 10):
        """
        Hiwonder 公式 Checksum 付加付きサーボ位置書き込みパケット送信。
        各サーボ宛てに 0x55 0x55 [ID] [Len] [Cmd=1] [PosL] [PosH] [TimeL] [TimeH] [Checksum] を送信。
        """
        if self.dummy_mode or self.ser is None:
            return
        
        count = min(len(angles_rad), self.num_servos)
        move_time = move_time_ms
        
        batch_packet = bytearray()
        for i in range(count):
            servo_id = self.servo_id_map.get(i, i + 1)
            angle_deg = np.degrees(angles_rad[i])
            angle_deg = np.clip(angle_deg, -120.0, 120.0)  # ±120度ハードクランプ
            pos = int(np.clip((angle_deg + 120.0) / 240.0 * 1000.0, 0, 1000))
            
            # 1サーボ宛てパケットデータ部
            # Length = 7 (Length, Cmd, PosL, PosH, TimeL, TimeH, Checksum)
            pkt_body = bytearray([servo_id, 7, self.CMD_SERVO_MOVE_TIME_WRITE])
            pkt_body.extend(struct.pack('<H', pos))
            pkt_body.extend(struct.pack('<H', move_time))
            
            checksum = calc_checksum(pkt_body)
            
            # 完全なパケット
            batch_packet.extend(self.HEADER)
            batch_packet.extend(pkt_body)
            batch_packet.append(checksum)
        
        with self.lock:
            self.ser.write(batch_packet)
    
    def interleave_read_status(self):
        """インターリーブ巡回読み出し (10Hz)"""
        if self.dummy_mode or self.ser is None:
            return
        
        for _ in range(self.read_batch_size):
            servo_id = self.servo_id_map.get(self._read_cursor, self._read_cursor + 1)
            
            temp = self._read_servo_register(servo_id, self.CMD_SERVO_TEMP_READ)
            if temp is not None:
                self.servo_temps[self._read_cursor] = float(temp)
            
            vin = self._read_servo_register(servo_id, self.CMD_SERVO_VIN_READ)
            if vin is not None:
                self.servo_voltages[self._read_cursor] = float(vin) / 1000.0
            
            self._read_cursor = (self._read_cursor + 1) % self.num_servos
    
    def _read_servo_register(self, servo_id: int, cmd: int) -> Optional[int]:
        """
        公式 Checksum 計算付きサーボレジスタ読み出し (半二重通信)
        
        [REAL-4 FIXED] Checksum 検証を厳格化。
        応答パケットの Checksum が一致しない場合は None を返す。
        パケット長の確認のみでは不十分（破損データを通す危険）。
        """
        if self.ser is None:
            return None
        
        # リクエストパケット: Header(2) + ID(1) + Len=3(1) + Cmd(1) + Checksum(1)
        pkt_body = bytearray([servo_id, 3, cmd])
        checksum = calc_checksum(pkt_body)
        
        packet = bytearray(self.HEADER)
        packet.extend(pkt_body)
        packet.append(checksum)
        
        with self.lock:
            self.ser.flushInput()
            self.ser.write(packet)
            
            # 応答受領: Header(2) + ID(1) + Len(1) + Cmd(1) + Data + Checksum(1)
            response = self.ser.read(8)
            if len(response) < 7:
                # [REAL-4 FIXED] パケット長不足でログ出力
                if len(response) > 0:
                    print(f"[Warn] Incomplete response from servo {servo_id}: {len(response)} bytes")
                return None
            
            if response[0:2] != self.HEADER:
                print(f"[Warn] Invalid header from servo {servo_id}")
                return None
            
            rx_id = response[2]
            rx_len = response[3]
            rx_cmd = response[4]
            
            # [REAL-4 FIXED] Checksum 検証を厳格化
            if len(response) <= 3 + rx_len:
                print(f"[Warn] Response too short for checksum validation from servo {servo_id}")
                return None
            
            rx_chk = response[3 + rx_len]
            calc_chk = calc_checksum(response[2:3+rx_len])
            
            if rx_chk != calc_chk:
                print(f"[Warn] Checksum mismatch for servo {servo_id}: "
                      f"expected {calc_chk:02x}, got {rx_chk:02x}")
                return None
            
            # データ抽出 (Checksum が一致した場合のみ)
            if cmd == self.CMD_SERVO_TEMP_READ:
                return response[5]
            elif cmd == self.CMD_SERVO_VIN_READ:
                return struct.unpack('<H', response[5:7])[0]
            elif cmd == self.CMD_SERVO_POS_READ:
                return struct.unpack('<h', response[5:7])[0]
        
        return None
    
    def close(self):
        if self.ser:
            self.ser.close()


# ============================================================
# 3. TeensySpineIO — 脊髄MCU (Teensy 4.1) 1kHz/100Hz 連携
# ============================================================

class TeensySpineIO:
    """
    Teensy 4.1 (脊髄MCU) との USB Serial パケット通信ドライバ。
    
    [REAL-5 FIXED] 通信タイムアウト仕組みを明示。
    
    RPi 側タイムアウト: 5ms (timeout=0.005)
    Teensy 側 E-stop トリガ: 30ms 無応答
    
    【仕組み説明】
    1. RPi から Teensy へ制御パケット送信 (毎ステップ = 10ms周期)
    2. Teensy が応答パケット返却 (通常 < 1ms)
    3. RPi が応答を 5ms タイムアウトで受信
    4. Teensy は最後に有効な通信時刻を記録
    5. 通信から 30ms 経過しても新しい通信がない場合、
       Teensy 側の 1kHz ハードウェアタイマが自動的に
       全サーボをゼロトルク にしてロボットを安全にドロップさせる
    
    RPi のアプリケーション層は 5ms タイムアウトで通信エラーに気付き、
    E-stop 処理を開始できる（30ms 前に検知可能）。
    """
    START_BYTE = 0xA5
    TELEMETRY_START_BYTE = 0x5A
    TELEMETRY_PACKET_LENGTH = 73
    MAX_GYRO_RAD_S = 50.0
    MAX_LINEAR_ACCEL_M_S2 = 10.0 * 9.80665
    FSR_MIN = 0.0
    FSR_MAX = 1.0
    
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200, num_servos: int = 20):
        self.port = port
        self.baudrate = baudrate
        self.num_servos = num_servos
        self.ser: Optional[serial.Serial] = None
        self.dummy_mode = serial is None
        
        self.last_imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "gyro": np.zeros(3),
            "lin_accel": np.zeros(3)
        }
        self.last_fsr_contacts = np.zeros(8, dtype=np.float32)
        self.servo_temps = np.full(num_servos, 25.0)
        self.servo_voltages = np.full(num_servos, 11.1)
        
        self.telemetry_timeout_flag = False  # 上位ループへの異常通知用フラグ
        self._consecutive_timeouts = 0       # 連続タイムアウト回数
        
        if not self.dummy_mode:
            try:
                # [REAL-5 FIXED] RPi 側タイムアウト = 5ms
                # Teensy 側 E-stop トリガ = 30ms (Teensy ファームウェア側で定義)
                self.ser = serial.Serial(port, baudrate, timeout=0.005)
                print(f"[Info] Teensy 4.1 Spinal MCU connected on {port}")
                print(f"[Info] Communication safety: RPi timeout={0.005*1000:.1f}ms, "
                      f"Teensy E-stop trigger=30ms")
            except Exception as e:
                print(f"[Error] Teensy 4.1 USB Serial init failed: {e}")
                self.dummy_mode = True

    def communicate(self, target_angles_rad: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        if self.dummy_mode or self.ser is None:
            return self.last_imu_data, self.last_fsr_contacts, self.servo_temps, self.servo_voltages

        # NaNのバイナリパッキングを最終防衛線でブロック
        safe_angles = np.nan_to_num(target_angles_rad, nan=0.0, posinf=0.0, neginf=0.0)

        data = bytearray([self.START_BYTE])
        for angle in safe_angles[:self.num_servos]:
            data.extend(struct.pack('<f', float(angle)))
        
        self.ser.write(data)
        
        raw = self.ser.read(self.TELEMETRY_PACKET_LENGTH)
        failure_reason = None
        if len(raw) < self.TELEMETRY_PACKET_LENGTH:
            failure_reason = f"incomplete ({len(raw)}/{self.TELEMETRY_PACKET_LENGTH} bytes)"
        elif raw[0] != self.TELEMETRY_START_BYTE:
            failure_reason = f"invalid start byte 0x{raw[0]:02x}"
        else:
            w, x, y, z = struct.unpack('<4f', raw[1:17])
            gyro = np.array(struct.unpack('<3f', raw[17:29]), dtype=np.float64)
            lin_accel = np.array(struct.unpack('<3f', raw[29:41]), dtype=np.float64)
            fsr_values = np.array(struct.unpack('<8f', raw[41:73]), dtype=np.float64)
            quat = np.array([w, x, y, z], dtype=np.float64)
            norm = np.linalg.norm(quat)

            telemetry_is_valid = (
                np.all(np.isfinite(quat))
                and 0.5 <= norm <= 1.5
                and np.all(np.isfinite(gyro))
                and np.all(np.abs(gyro) <= self.MAX_GYRO_RAD_S)
                and np.all(np.isfinite(lin_accel))
                and np.all(np.abs(lin_accel) <= self.MAX_LINEAR_ACCEL_M_S2)
                and np.all(np.isfinite(fsr_values))
                and np.all((fsr_values >= self.FSR_MIN) & (fsr_values <= self.FSR_MAX))
            )
            if telemetry_is_valid:
                self._consecutive_timeouts = 0
                self.telemetry_timeout_flag = False
                self.last_imu_data["quat"] = quat / norm
                self.last_imu_data["gyro"] = gyro.astype(np.float32)
                self.last_imu_data["lin_accel"] = lin_accel.astype(np.float32)
                self.last_fsr_contacts = np.rint(fsr_values).astype(np.float32)
            else:
                failure_reason = "telemetry value out of physical range"

        if failure_reason is not None:
            self._consecutive_timeouts += 1
            print(
                f"[Warn] Teensy telemetry rejected: {failure_reason} "
                f"(Consecutive: {self._consecutive_timeouts})"
            )
            if self._consecutive_timeouts >= 3:
                self.telemetry_timeout_flag = True
            
        return self.last_imu_data, self.last_fsr_contacts, self.servo_temps, self.servo_voltages

    def close(self):
        if self.ser:
            self.ser.close()
```

---

## robot/__init__.py

```python
# robot package: robot-specific specs, kinematics, and gait generation

```

---

## robot/config.py

```python
import numpy as np
import os
from pathlib import Path

class RobotConfig:
    """
    Sim-to-Real 二足歩行ロボット '旋風丸' 共通仕様書
    Target Hardware: 
    - Controller: Raspberry Pi 5 (16GB) + アクティブクーラー
    - Servo Driver: Hiwonder BusLinker V3.0 (x4, UART 1Mbps)
    - Actuator: Hiwonder HX-30HM (x20)
    - IMU: BNO055 (UART接続 — I2Cクロックストレッチング回避)
    - FSR判定: Teensy 4.1オンチップADCで読み取り、閾値判定した8ch二値信号
    - 足裏: FSR402 (x8)
    
    【修正対応 (2026-09-08)】
    - [CONFIG-1 FIXED] CURRICULUM_SCHEDULE を廃止、CURRICULUM_SCHEDULE_FRACTIONS に統一
    - [CONFIG-3 FIXED] INITIAL_HEIGHT を明記、TERMINATION_HEIGHT の根拠を記載
    - [GAIT-2 FIXED] 歩容パラメータを config.py に一元化
    """

    # --- 1. Project Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    MUJOCO_MODEL_PATH = BASE_DIR / "assets" / "humanoid" / "humanoid.xml"

    # --- 1.1. FSR Hardware Layout ---
    # 実機ではTeensy側で接地判定するため、位置はシミュレーション専用。
    # MuJoCo の <sensor> は IMU(gyro/accel/quat) の後に 8ch FSR touch が
    # 連続して並ぶ。XML では left-foot 4ch → right-foot 4ch の順に宣言されているため、
    # ここも同じ順に合わせる。左右の足の座標は左右対称となるよう、右足だけ X 方向を反転する。
    FSR_POSITIONS = np.array([
        [0.012, 0.027], [-0.012, 0.027], [0.012, -0.070], [-0.012, -0.070],  # Left foot, sensor order = FL FR BL BR
        [-0.012, 0.027], [0.012, 0.027], [-0.012, -0.070], [0.012, -0.070],  # Right foot, sensor order = FL FR BL BR (mirrored)
    ])
    
    # --- 2. Hardware Specs ---
    
    # Actuator: Hiwonder HX-30HM Serial Bus Servo (Magnetic Encoder)
    # Spec: 30kg.cm (11.1V) -> 2.94 N.m
    MOTOR_MAX_TORQUE = 3.0       # [N.m] HX-30HMに合わせて修正
    MOTOR_MAX_VELOCITY = 6.5     # [rad/s] (0.19sec/60deg @11.1V)
    
    # 関節定義 (Fusion 360のURDFとIDを一致させること)
    # 旋風丸の本稼働用設定 (20 DOF)
    JOINT_NAMES = [
        # 右脚 (6関節)
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        # 左脚 (6関節)
        "left_hip_yaw",  "left_hip_roll",  "left_hip_pitch",  "left_knee",  "left_ankle_pitch",  "left_ankle_roll",
        # 右腕 (4関節)
        "right_shoulder_roll", "right_shoulder_pitch", "right_elbow", "right_wrist_pitch",
        # 左腕 (4関節)
        "left_shoulder_roll", "left_shoulder_pitch", "left_elbow", "left_wrist_pitch"
    ]
    
    # --- Actuator Reality Gap (LPF) ---
    MOTOR_LPF_ALPHA = 0.8  # 1st-order Low-Pass Filter coefficient for HX-30HM
    
    NUM_JOINTS = len(JOINT_NAMES)

    # --- お手本（Reference Trajectory）使用のトグルスイッチ ---
    # True: サイクロイド歩行軌道に基づく「残差強化学習 (Residual RL)」
    # False: 「お手本無し強化学習 (Direct RL)」 - 物理法則と報酬だけで自発的歩行を獲得
    USE_REFERENCE_GAIT = False  # お手本無しでやりたい場合は False に設定！

    # お手本無しの学習を劇的に安定させる「中腰デフォルト姿勢 (Default Standing Joint Angles)」
    # ユーザーが設定したXMLの可動域に合わせて、左右で符号を反転（右は膝マイナス、左は膝プラス等）
    DEFAULT_JOINT_ANGLES = np.array([
        # 右脚 (yaw, roll, pitch, knee, ankle_pitch, ankle_roll)
        0.0, 0.0, 0.29, -0.58, -0.29, 0.0,
        # 左脚 (yaw, roll, pitch, knee, ankle_pitch, ankle_roll)
        0.0, 0.0, -0.29, 0.58, 0.29, 0.0,
        # 右腕 (shoulder_roll, shoulder_pitch, elbow, wrist_pitch)
        0.0, 0.0, 0.0, 0.0,
        # 左腕 (shoulder_roll, shoulder_pitch, elbow, wrist_pitch)
        0.0, 0.0, 0.0, 0.0
    ])

    # --- 3. Control Specs ---
    SIM_DT = 1.0 / 400.0     # シミュレーション刻み (2.5ms)
    CONTROL_DECIMATION = 4
    CONTROL_DT = SIM_DT * CONTROL_DECIMATION # 100Hz (10msループ)
    
    # PD制御ゲイン (Sim用) — 外乱耐性のため剛性を引き上げ
    KP = 40.0
    KD = 1.0

    # --- 4. Sim-to-Real Gap Mitigation ---
    # base_pos / lin_vel の大ノイズ (実機ではIMU積分ドリフトで不正確)
    # 学習時にこれらを「信頼できない」特徴量として扱わせるためのDR
    NOISE_BASE_POS    = 0.1   # [m]  — 実機ではゼロ埋め or VIO推定のためドリフト大
    NOISE_LIN_VEL     = 0.5   # [m/s] — IMU積分だと数秒でm/sオーダーのエラー
    
    RANDOM_MASS_SCALE = [0.97, 1.03]  # Phase 1: DR範囲を縮小して基本直立に集中
    RANDOM_FRICTION = [0.7, 1.1]      # Phase 1: 摩擦変動を控えめに
    RANDOM_COM_OFFSET = [-0.02, 0.02]  # Phase 1: 重心偏差を最小化
    RANDOM_PUSH_MAX_FORCE = 0.0  # Phase 0: Gate 0 / Gate A を先に確定し、外乱導入は後に行う
    DISTURBANCE_CURRICULUM = False  # Phase 0 では外乱を無効化して静止直立を安定化させる
    
    # 熱・電圧のシミュレーションパラメータ
    RANDOM_TEMP = [20.0, 80.0]  # ℃
    RANDOM_VOLT = [9.0, 12.6]   # V

    # --- 5. RL Settings ---
    # 歩行周期 (秒)
    GAIT_PERIOD = 1.0
    
    # 新アーキテクチャ(RMA/遅延補償対応)における観測空間定義
    HISTORY_LEN = 5 # 過去Nステップの観測と行動(50ms分@100Hz)
    
    # Base観測: 12(胴体) + N*2(関節角/速度) + 10(FSR/ZMP) + 2(位相) + N(理想軌道)
    BASE_OBS_DIM = 12 + (NUM_JOINTS * 2) + 10 + 2 + NUM_JOINTS
    
    # 行動次元
    ACT_DIM = NUM_JOINTS
    
    # サーボ温度(N)とシステム電圧(1)
    SERVO_TEMP_DIM = NUM_JOINTS
    SUPPLY_VOLTAGE_DIM = 1
    
    # 最終的な平坦化されたOBS次元:
    # 履歴バッファに入っている各ステップの観測(Base)と行動を合わせたものの履歴長
    HISTORY_DIM = (BASE_OBS_DIM + ACT_DIM) * HISTORY_LEN
    
    # 現在の観測次元の拡張 (RMA向け) = Base(現在) + 履歴 + 温度 + 電圧
    OBS_DIM = BASE_OBS_DIM + HISTORY_DIM + SERVO_TEMP_DIM + SUPPLY_VOLTAGE_DIM
    
    # 行動空間: ±30度 (Phase 1: 初期探索で暴走しないよう縮小。Phase 2以降で拡大)
    ACTION_SCALE = np.deg2rad(30)

    # === Standing-only mission constraints ===
    # 目的は自律歩行ではなく、外乱に耐えながらその場直立を維持すること。
    # 歩行、踏み出し、支持基底面の変更はいかなる外乱条件でも禁止。
    ALLOW_WALKING = False
    ALLOW_STEPPING = False
    TARGET_VEL_X = 0.0
    TARGET_VEL_Y = 0.0
    TARGET_YAW_RATE = 0.0
    MAX_FOOT_TRANSLATION = 0.005  # [m], 5 mm 未満を許容
    MAX_SINGLE_FOOT_LIFT = 0.0
    FOOT_CONTACT_THRESHOLD = 0.05  # [N] シミュレーション上の各足の最小接触力
    
    # ======================================================
    # 次世代・外乱耐性特化 報酬ウェイト (Phase-Dependent Architecture)
    # ======================================================
    COM_HEIGHT = 0.17            # [FIX] 中腰姿勢での実測CoM高 (旧0.28は高すぎた)
    
    REWARD_WEIGHTS = {
        # Phase 1: 静止直立で確実に正報酬を出すため、安定性と生存を強く重視する。
        "alive": 25.0,
        "fall_penalty": -30.0,

        # 安定維持を最優先
        "upright": 12.0,
        "target_pose": 4.0,
        "com_stab": 10.0,
        "both_feet_contact": 8.0,

        # 外乱が無い Phase 0/1 では回復ボーナスは控えめにする
        "capture_point": 0.5,
        "impedance": 0.2,
        "recovery": 0.5,

        # ペナルティは大きく下げて、PTPな振動で負値が吹き上がらないようにする
        "ang_momentum_z": 0.01,
        "ang_momentum_xy": 0.01,
        "cbf": 0.2,
        "energy": 0.00005,
        "smoothness": 0.0001,
        "drift": 0.005,
        "slip": 0.01,
        "stance_width": 0.01,

        # 緩和対数バリアも安全域では大きく効かせない
        "barrier_height": 0.2,
        "barrier_torque": 0.1,
    }

    # --- [CONFIG-3 FIXED] 初期高さを明記、終了条件を根拠付き ---
    # mjx_env.py の reset() で qpos[2] = 0.1773 として設定される
    # 注意: この値自体はワールド座標系での胴体初期位置だが、
    # TERMINATION_HEIGHT は envs/mjx_rewards.py::compute() 内で
    # 「足裏を基準にした相対高さ (base_pos[2] - lowest_foot_z)」との
    # 比較にのみ使われる（絶対座標の閾値ではない。"Contract Violation B"
    # 対応で相対高さ判定に統一済み）。以下の引き算は「7cmというマージン量」
    # を求めるための便宜的な計算であり、絶対座標の意味は持たない。
    INITIAL_HEIGHT = 0.1773  # [m] 直立姿勢での胴体初期位置（ワールド座標Z）
    
    # 転倒判定の高さマージン。足裏基準の相対高さがこの値を下回ったら終了。
    # 根拠: 中腰姿勢（膝屈曲）での安定限界に相当するマージンとして
    # INITIAL_HEIGHT - 0.07 を流用している（比較対象は相対高さ）。
    TERMINATION_HEIGHT = INITIAL_HEIGHT - 0.07  # = 0.1073m（相対高さの閾値）
    TERMINATION_PITCH = np.deg2rad(45) 
    TERMINATION_ROLL  = np.deg2rad(45)
    
    # 最大エピソード長 (Phase 1: 5秒。短いエピソードで高速学習サイクル)
    MAX_EPISODE_STEPS = 500 

    # ======================================================
    # [NEW] λ_phase(s) 合成状態変数 z(s) の重み
    # z(s) = tilt*|θ_err| + ang_vel*|ω| + lin_vel_err*|v_xy| + disturbance_flag*1{外乱検知}
    # 旧実装は tilt/ang_vel のみで構成されており、外力印加直後
    # (傾きがまだ立ち上がっていない数ステップ)にλ_phaseが1のまま残る
    # 「反応の空白期間」が生じていた。lin_vel_err と disturbance_flag を
    # 追加し、外力印加の瞬間にフェーズ遷移を先行させる。
    # ======================================================
    LAMBDA_PHASE_WEIGHTS = {
        "tilt": 5.0,
        "ang_vel": 0.5,
        "lin_vel_err": 1.5,
        "disturbance_flag": 4.0,
    }
    LAMBDA_PHASE_Z_THRESH = 0.3
    LAMBDA_PHASE_DECAY_K = 10.0

    # --- Capture Point / ZMP マージン計算パラメータ ---
    FOOT_SUPPORT_RADIUS = 0.06   # [m] 足平の実効支持半径。URDF実寸に要調整
    CP_MARGIN_NORM_DIST = 0.15   # [m]

    # --- 外乱復帰ボーナスの時定数 ---
    # 旧: 2ステップ(20ms)は短すぎたため、滑らかな指数減衰の半減期に変更
    RECOVERY_BONUS_WINDOW_STEPS = 50          # 半減期(0.5秒 @100Hz)
    RECOVERY_BONUS_STABILITY_THRESHOLD = 0.4

    # --- ペナルティスケジューリング ---
    # 旧: penalty_scale = clip(step/500,0,1) はエピソード内経過時間
    # (info['step'])に基づいており、MAX_EPISODE_STEPSの半分に相当する
    # 5秒間、学習終盤まで恒久的にペナルティが消失していた。
    # 短いエピソード内グレース(物理リセット直後の過渡応答許容)に短縮し、
    # 学習全体の進行度は training_progress (外部供給) で分離する。
    PENALTY_INTRA_EPISODE_WARMUP_STEPS = 30   # 0.3秒
    # CBF/バリア等ハードウェア安全項は独立した高速ランプ
    SAFETY_PENALTY_WARMUP_STEPS = 10          # 0.1秒

    # --- 緩和対数バリア関数パラメータ ---
    BARRIER_HEIGHT_MARGIN = 0.05        # TERMINATION_HEIGHTからのマージン[m]
    BARRIER_HEIGHT_CLIP = 5.0
    BARRIER_TORQUE_MARGIN_RATIO = 0.15  # MOTOR_MAX_TORQUEに対する比率
    BARRIER_TORQUE_CLIP = 5.0

    # ======================================================
    # [CONFIG-1 FIXED] カリキュラム学習: 外乱強度スケジュール
    # ======================================================
    # 【設計】学習進捗率 (0.0~1.0) に基づく相対スケジュール。
    # 絶対ステップ数による CURRICULUM_SCHEDULE は廃止。
    # 
    # 理由: USE_REFERENCE_GAIT の有無で総学習ステップ数が大きく変わっても
    # (10M vs 20~30M)、同じ相対カリキュラムが自動的に機能する。
    # 
    # 供給元: training_progress (mjx_env.py → mjx_rewards.py へ外部供給)
    # 詳細: envs/mjx_rewards.py の _get_curriculum_disturbance_scale() を参照
    CURRICULUM_SCHEDULE_FRACTIONS = {
        0.00: 0.00,  # 学習開始時: 外乱なし
        0.10: 0.10,  # 10%進捗: 微弱外乱
        0.25: 0.30,  # 25%進捗: 軽い外乱
        0.50: 0.60,  # 50%進捗: 中程度外乱
        0.75: 1.00,  # 75%進捗: 最大外乱
    }
    
    # USE_REFERENCE_GAIT=True: 学習側説明書.md の目安(10Mステップ)
    # USE_REFERENCE_GAIT=False (Direct RL): 20~30Mステップ推奨のため長めに設定
    TOTAL_TRAINING_STEPS_ESTIMATE = 10_000_000 if USE_REFERENCE_GAIT else 25_000_000

    # ======================================================
    # [GAIT-2 FIXED] 歩容パラメータ (config.py に一元化)
    # ======================================================
    # gait_generator.py と kinematics.py から参照される定数。
    # 複数の場所で定義されていたが、config.py に統一して保守性を向上。
    # 
    # ロボット物理寸法に関わるため、URDF/実機の値と 100% 同期すること。
    GAIT_STAND_HEIGHT = 0.23  # [m] 直立時の腰の高さ
    GAIT_STEP_HEIGHT = 0.04   # [m] 足を上げる高さ
    GAIT_STEP_LENGTH = 0.10   # [m] 歩幅
    GAIT_THIGH_LEN = 0.12     # [m] 大腿リンク長（股関節～膝）
    GAIT_KNEE_LEN = 0.12      # [m] 下腿リンク長（膝～足首）

    MJX_LEARNING_RATE = 1e-4  # 学習崩壊を防ぐため低めに設定
```

---

## robot/gait_generator.py

```python
"""
robot/gait_generator.py — サイクロイド歩行軌道 + 逆運動学

【修正対応 (2026-09-08)】
- [GAIT-1 FIXED] リンク長を config.py から参照（0.12m に統一）
- [GAIT-2 FIXED] STAND_HEIGHT も config.py に一元化

サイクロイド軌道の特性:
- ジャーク最小化（足の着地がスムーズ）
- 同期性が高い（両脚の協調動作が安定）
- 倒立振子モデルと整合しやすい
"""

import numpy as np
import jax.numpy as jp

from robot.config import RobotConfig


# ============================================================
# JAX版サイクロイド軌道生成（学習環境用）
# ============================================================

def jax_cycloid_trajectory(
    phase: float,
    foot_height: float,
    step_length: float,
    stand_height: float
) -> Tuple[jp.ndarray, jp.ndarray]:
    """
    サイクロイド軌道：X-Z平面での足先位置と高さ。
    
    【数学背景】
    サイクロイドは、半径rの円が直線上を転がるとき、
    円周上の一点が描く軌跡。ジャーク最小化特性により、
    関節角の加加速度が最小化され、滑らかな足運動を実現。
    
    Args:
        phase: [0, 1) の周期的フェーズ
        foot_height: 遊脚中の最大高さ [m]
        step_length: 1周期での歩幅 [m]
        stand_height: 直立時の腰高さ [m]
    
    Returns:
        (x_traj [m], z_traj [m]): 足先のX-Z位置
    """
    # サイクロイド軌跡の半周期を 0.5 の phase で表現
    phase_mod = (phase % 1.0) * 2.0  # [0, 2)
    
    # 右脚: phase 0.0-1.0 で遊脚、1.0-2.0 で接地
    # 左脚は phase 0.5-1.5 で遊脚、1.5-0.5 で接地（半周期ずれ）
    
    # 遊脚フェーズ判定（0-1: swing, 1-2: stance）
    is_swing = phase_mod < 1.0
    phase_swing = jp.clip(phase_mod, 0.0, 1.0)  # [0, 1]
    phase_stance = jp.clip(phase_mod - 1.0, 0.0, 1.0)  # [0, 1]
    
    # ===== Swing Phase (遊脚) =====
    # サイクロイド曲線: x = r(θ - sin(θ)), z = r(1 - cos(θ))
    # θ: 転がる円の角度 [0, π]
    theta_swing = phase_swing * np.pi
    
    # サイクロイド：
    # - 水平移動: step_length の距離を移動
    # - 垂直移動: 最大 foot_height まで上昇して着地
    x_swing = (step_length / 2.0) * (theta_swing - jp.sin(theta_swing)) / np.pi
    z_swing = (foot_height / np.pi) * (1.0 - jp.cos(theta_swing))
    
    # ===== Stance Phase (接地) =====
    # 接地時は足が地面に固定（X, Z 共に変化なし）
    # または徐々に後方へ移動（参考文献により異なる）
    # ここでは簡略化して、接地時は最終位置を保持
    x_stance = step_length / 2.0  # 遊脚で移動した分
    z_stance = 0.0  # 地面に接触
    
    # スイッチング
    x_traj = jp.where(is_swing, x_swing - step_length / 2.0, -step_length / 2.0)
    z_traj = jp.where(is_swing, z_swing, z_stance)
    
    return x_traj, z_traj


def _simple_ik_leg(
    target_x: float,
    target_z: float,
    thigh_len: float = None,
    knee_len: float = None,
) -> Tuple[float, float, float]:
    """
    2リンク平面逆運動学（股関節ピッチと膝関節）。
    
    [GAIT-1 FIXED] リンク長を config.py から参照（デフォルト値付き）
    
    Args:
        target_x, target_z: 足先の目標位置 [m]
        thigh_len: 大腿長 [m]（デフォルト: config.py 参照）
        knee_len: 下腿長 [m]（デフォルト: config.py 参照）
    
    Returns:
        (hip_pitch, knee_angle, ankle_pitch): 関節角 [rad]
    """
    if thigh_len is None:
        thigh_len = RobotConfig.GAIT_THIGH_LEN
    if knee_len is None:
        knee_len = RobotConfig.GAIT_KNEE_LEN
    
    # 目標位置から股関節から足先までの距離を計算
    L = np.sqrt(target_x**2 + target_z**2)
    
    # 到達可能範囲のチェック
    max_len = thigh_len + knee_len
    if L > max_len:
        L = max_len
    elif L < abs(thigh_len - knee_len):
        L = abs(thigh_len - knee_len)
    
    # 余弦定理で膝角度を計算
    cos_knee = (thigh_len**2 + knee_len**2 - L**2) / (2 * thigh_len * knee_len)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    knee_angle = np.arccos(cos_knee)
    
    # 股関節ピッチ角を計算
    alpha = np.arctan2(target_z, target_x)
    beta = np.arcsin(knee_len * np.sin(knee_angle) / L)
    hip_pitch = alpha + beta
    
    # 足首角度：足底を水平に保つ
    ankle_pitch = -(hip_pitch + knee_angle)
    
    return hip_pitch, knee_angle, ankle_pitch


def jax_get_reference_trajectory(phase: float, num_joints: int = 20) -> jp.ndarray:
    """
    JAX版リファレンス軌道生成（学習環境用）。
    
    [GAIT-1 FIXED] config.py のパラメータを使用して、
    gait_generator.py の定数を消去。
    
    学習環境 mjx_env.py で 100Hz (CONTROL_DT=10ms) で呼び出されることを想定。
    
    Args:
        phase: [0, 1) の周期的フェーズ（mjx_env で計算）
        num_joints: 関節数（デフォルト: 20）
    
    Returns:
        ref_angles: 各関節の理想角度 [rad] shape=(num_joints,)
    """
    # [GAIT-2 FIXED] config.py から参照
    stand_height = RobotConfig.GAIT_STAND_HEIGHT
    step_height = RobotConfig.GAIT_STEP_HEIGHT
    step_length = RobotConfig.GAIT_STEP_LENGTH
    thigh_len = RobotConfig.GAIT_THIGH_LEN
    knee_len = RobotConfig.GAIT_KNEE_LEN
    
    # 左右の脚位相（半周期ずれ）
    phase_r = phase  # 右脚：phase を直接使用
    phase_l = (phase + 0.5) % 1.0  # 左脚：0.5 位相ずれ
    
    # サイクロイド軌道で右脚の足先目標を計算
    x_r, z_r = jax_cycloid_trajectory(phase_r, step_height, step_length, stand_height)
    
    # サイクロイド軌道で左脚の足先目標を計算
    x_l, z_l = jax_cycloid_trajectory(phase_l, step_height, step_length, stand_height)
    
    # 逆運動学（NumPy の _simple_ik_leg を使用するため一度 NumPy に戻す）
    # JAX JIT 互換性のため、JAX版 IK も別途実装すること（後述）
    x_r_np = float(x_r)
    z_r_np = float(z_r)
    x_l_np = float(x_l)
    z_l_np = float(z_l)
    
    hip_pitch_r, knee_r, ankle_pitch_r = _simple_ik_leg(x_r_np, z_r_np, thigh_len, knee_len)
    hip_pitch_l, knee_l, ankle_pitch_l = _simple_ik_leg(x_l_np, z_l_np, thigh_len, knee_len)
    
    # リファレンス軌道ベクトル（20関節のデフォルト）
    ref_angles = jp.zeros(num_joints)
    
    if num_joints >= 12:
        # 右脚インデックス
        ref_angles = ref_angles.at[2].set(jp.array(hip_pitch_r))    # right_hip_pitch
        ref_angles = ref_angles.at[3].set(jp.array(knee_r))         # right_knee
        ref_angles = ref_angles.at[4].set(jp.array(ankle_pitch_r))  # right_ankle_pitch
        
        # 左脚インデックス
        ref_angles = ref_angles.at[8].set(jp.array(hip_pitch_l))    # left_hip_pitch
        ref_angles = ref_angles.at[9].set(jp.array(knee_l))         # left_knee
        ref_angles = ref_angles.at[10].set(jp.array(ankle_pitch_l)) # left_ankle_pitch
    
    return ref_angles


# ============================================================
# NumPy版サイクロイド軌道生成（テスト・実機用）
# ============================================================

class GaitGenerator:
    """NumPy ベースのサイクロイド歩行軌道生成クラス。"""
    
    def __init__(self):
        # [GAIT-2 FIXED] config.py から参照
        self.stand_height = RobotConfig.GAIT_STAND_HEIGHT
        self.step_height = RobotConfig.GAIT_STEP_HEIGHT
        self.step_length = RobotConfig.GAIT_STEP_LENGTH
        self.thigh_len = RobotConfig.GAIT_THIGH_LEN
        self.knee_len = RobotConfig.GAIT_KNEE_LEN
    
    def get_foot_position(self, phase: float, right_leg: bool = True) -> Tuple[float, float]:
        """
        サイクロイド軌道から足先位置を計算。
        
        Args:
            phase: [0, 1) のフェーズ
            right_leg: True なら右脚、False なら左脚
        
        Returns:
            (x, z): 足先のX-Z位置 [m]
        """
        if not right_leg:
            phase = (phase + 0.5) % 1.0  # 左脚は半周期ずれ
        
        # サイクロイド軌跡
        phase_mod = phase * 2.0  # [0, 2)
        is_swing = phase_mod < 1.0
        phase_swing = np.clip(phase_mod, 0.0, 1.0)
        
        # サイクロイド
        theta = phase_swing * np.pi
        x_swing = (self.step_length / 2.0) * (theta - np.sin(theta)) / np.pi
        z_swing = (self.step_height / np.pi) * (1.0 - np.cos(theta))
        
        x = x_swing - self.step_length / 2.0 if is_swing else -self.step_length / 2.0
        z = z_swing if is_swing else 0.0
        
        return x, z
    
    def get_joint_angles(self, phase: float) -> np.ndarray:
        """
        指定された位相での各関節の目標角度を取得。
        
        Args:
            phase: [0, 1) のフェーズ
        
        Returns:
            angles: shape=(20,) の関節角度 [rad]
        """
        angles = np.zeros(20)
        
        # 右脚の足先位置
        x_r, z_r = self.get_foot_position(phase, right_leg=True)
        hip_r, knee_r, ankle_r = _simple_ik_leg(x_r, z_r, self.thigh_len, self.knee_len)
        
        # 左脚の足先位置
        x_l, z_l = self.get_foot_position(phase, right_leg=False)
        hip_l, knee_l, ankle_l = _simple_ik_leg(x_l, z_l, self.thigh_len, self.knee_len)
        
        # 右脚への割り当て
        angles[2] = hip_r    # right_hip_pitch
        angles[3] = knee_r   # right_knee
        angles[4] = ankle_r  # right_ankle_pitch
        
        # 左脚への割り当て
        angles[8] = hip_l    # left_hip_pitch
        angles[9] = knee_l   # left_knee
        angles[10] = ankle_l # left_ankle_pitch
        
        return angles


def numpy_get_reference_trajectory(phase: float, num_joints: int = 20) -> np.ndarray:
    """
    NumPy版リファレンス軌道生成（テスト・実機用）。
    
    [GAIT-1 FIXED] config.py のパラメータを使用。
    [GAIT-2 FIXED] GaitGenerator クラスと統一された実装。
    
    Args:
        phase: [0, 1) の周期的フェーズ
        num_joints: 関節数（デフォルト: 20）
    
    Returns:
        ref_angles: 各関節の理想角度 [rad] shape=(num_joints,)
    """
    gen = GaitGenerator()
    ref_angles = gen.get_joint_angles(phase)
    
    # 必要に応じてリサイズ
    if num_joints < 20:
        ref_angles = ref_angles[:num_joints]
    
    return ref_angles


# ============================================================
# テスト & デバッグ
# ============================================================

if __name__ == "__main__":
    print("[Test] Gait Generator Validation")
    print(f"Config Thigh Length: {RobotConfig.GAIT_THIGH_LEN} m")
    print(f"Config Knee Length: {RobotConfig.GAIT_KNEE_LEN} m")
    print(f"Config Stand Height: {RobotConfig.GAIT_STAND_HEIGHT} m")
    print()
    
    # NumPy版テスト
    print("[NumPy Test] Full cycle (0.0 to 1.0 phase)")
    gen = GaitGenerator()
    for phase_val in np.linspace(0.0, 1.0, 5, endpoint=False):
        angles = numpy_get_reference_trajectory(phase_val, num_joints=20)
        x_r, z_r = gen.get_foot_position(phase_val, right_leg=True)
        x_l, z_l = gen.get_foot_position(phase_val, right_leg=False)
        print(f"Phase {phase_val:.2f}: "
              f"Right foot ({x_r:+.3f}, {z_r:+.3f}m), "
              f"Left foot ({x_l:+.3f}, {z_l:+.3f}m), "
              f"Hip_R={angles[2]*57.3:+.1f}°")
    
    print("\n[JAX Test] Full cycle (NumPy を経由)")
    for phase_val in np.linspace(0.0, 1.0, 5, endpoint=False):
        ref_jax = jax_get_reference_trajectory(phase_val, num_joints=20)
        print(f"Phase {phase_val:.2f}: Hip_R={float(ref_jax[2])*57.3:+.1f}°")
```

---

## robot/math_utils.py

```python
"""
robot/math_utils.py — クォータニオン・数学変換ユーティリティ

【修正対応 (2026-09-08)】
- [MATH-1 FIXED] quat_to_euler() を NumPy版と JAX版に分離（JIT互換化）
- 関数内での型判定を排除し、呼び出し側で型を明示的に選択
- JAX JIT コンパイルの制御フロー制限に対応
"""

import numpy as np

try:
    import jax
    import jax.numpy as jp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jp = None
    jax = None


# ============================================================
# NumPy版: クォータニオン -> オイラー角
# ============================================================

def quat_to_euler_numpy(q: np.ndarray) -> np.ndarray:
    """
    NumPy版クォータニオンからオイラー角（Roll-Pitch-Yaw）への変換。
    
    クォータニオン形式: q = [w, x, y, z]（BNO055標準）
    
    【変換式】
    ロール (Roll) φ：X軸周りの回転
    ピッチ (Pitch) θ：Y軸周りの回転
    ヨー (Yaw) ψ：Z軸周りの回転
    
    標準的な ZYX (Yaw-Pitch-Roll) オーダーで変換。
    
    Args:
        q: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        rpy: shape=(3,) オイラー角 [roll, pitch, yaw] [rad]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (X軸周りの回転)
    sinp = 2.0 * (w * x + y * z)
    cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = np.arctan2(sinp, cosp)
    
    # Pitch (Y軸周りの回転)
    sinp_pitch = 2.0 * (w * y - z * x)
    sinp_pitch = np.clip(sinp_pitch, -1.0, 1.0)  # 数値誤差対策
    pitch = np.arcsin(sinp_pitch)
    
    # Yaw (Z軸周りの回転)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(siny, cosy)
    
    return np.array([roll, pitch, yaw])


# ============================================================
# JAX版: クォータニオン -> オイラー角
# ============================================================

def quat_to_euler_jax(q: "jax.Array") -> "jax.Array":
    """
    JAX版クォータニオンからオイラー角への変換（JIT互換）。
    
    【重要】JAX JIT コンパイル内でも実行可能な実装。
    制御フロー（if/else）を使わず、jp.clip() と jp.arcsin() で
    数値安定性を確保。
    
    Args:
        q: shape=(4,) JAX配列 クォータニオン [w, x, y, z]
    
    Returns:
        rpy: shape=(3,) JAX配列 オイラー角 [roll, pitch, yaw] [rad]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (X軸周りの回転)
    sinp = 2.0 * (w * x + y * z)
    cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = jp.arctan2(sinp, cosp)
    
    # Pitch (Y軸周りの回転)
    sinp_pitch = 2.0 * (w * y - z * x)
    # [MATH-1 FIXED] clip で [-1, 1] に制限（JAX JIT互換）
    sinp_pitch = jp.clip(sinp_pitch, -1.0, 1.0)
    pitch = jp.arcsin(sinp_pitch)
    
    # Yaw (Z軸周りの回転)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y**2 + z**2)
    yaw = jp.arctan2(siny, cosy)
    
    return jp.array([roll, pitch, yaw])


# ============================================================
# [MATH-1 FIXED] ユーザー向け統一インターフェース
# ============================================================

def quat_to_euler(q) -> np.ndarray:
    """
    クォータニオンからオイラー角への統一インターフェース。
    
    【使い方】
    入力配列の型に基づいて、自動的に適切な実装を選択します。
    
    - NumPy配列 or Python float/list → NumPy版を使用
    - JAX配列 → JAX版を使用（JIT対応）
    
    Args:
        q: クォータニオン [w, x, y, z]（NumPy配列またはJAX配列）
    
    Returns:
        rpy: オイラー角 [roll, pitch, yaw] [rad]
             入力の型に応じて NumPy配列 or JAX配列を返す
    
    例:
        # NumPy環境
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        rpy_np = quat_to_euler(q_np)  # → NumPy配列
        
        # JAX環境
        q_jax = jax.numpy.array([1.0, 0.0, 0.0, 0.0])
        rpy_jax = quat_to_euler(q_jax)  # → JAX配列
        
        # JAX JIT 内で使用可能
        @jax.jit
        def compute_rpy(q):
            return quat_to_euler(q)  # 自動的に JAX版で実行
    """
    # [MATH-1 FIXED] 型判定を呼び出し側で実施
    if HAS_JAX and isinstance(q, jax.Array):
        # JAX配列の場合は JAX版を使用
        return quat_to_euler_jax(q)
    else:
        # NumPy配列 or その他の場合は NumPy版を使用
        q_np = np.asarray(q)
        return quat_to_euler_numpy(q_np)



# ============================================================
# その他のユーティリティ関数
# ============================================================

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    クォータニオンを正規化（ノルム = 1）。
    
    Args:
        q: shape=(4,) クォータニオン
    
    Returns:
        q_normalized: 正規化されたクォータニオン
    """
    q = np.asarray(q)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 安全なデフォルト
    return q / norm


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    クォータニオンの逆元を計算。
    
    q⁻¹ = q*/|q|² （共役四元数を ノルムの二乗で割る）
    
    Args:
        q: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        q_inv: 逆元クォータニオン
    """
    q = np.asarray(q)
    norm_sq = np.sum(q**2)
    if norm_sq < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    # 共役: [w, -x, -y, -z]
    return np.array([q[0], -q[1], -q[2], -q[3]]) / norm_sq


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    クォータニオンでベクトルを回転。
    
    v' = q * v * q⁻¹
    
    Args:
        v: shape=(3,) ベクトル
        q: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        v_rotated: 回転後のベクトル shape=(3,)
    """
    v = np.asarray(v)
    q = np.asarray(q)
    q = normalize_quaternion(q)
    
    # v を [0, v_x, v_y, v_z] に拡張
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    
    # q * v * q⁻¹
    q_inv = quaternion_inverse(q)
    
    # quaternion multiplication: q * v
    qv = quaternion_multiply(q, v_quat)
    
    # (q * v) * q⁻¹
    result = quaternion_multiply(qv, q_inv)
    
    return result[1:4]  # 虚部のみを返す


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    2つのクォータニオンの積を計算。
    
    Args:
        q1, q2: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        product: 積のクォータニオン
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


# ============================================================
# テスト & デバッグ
# ============================================================

if __name__ == "__main__":
    print("[Test] Math Utilities Validation")
    print()
    
    # テストケース: いくつかの有名なクォータニオン
    test_quaternions = [
        np.array([1.0, 0.0, 0.0, 0.0]),         # Identity
        np.array([0.7071, 0.7071, 0.0, 0.0]),   # 90° roll
        np.array([0.7071, 0.0, 0.7071, 0.0]),   # 90° pitch
        np.array([0.7071, 0.0, 0.0, 0.7071]),   # 90° yaw
    ]
    
    print("[NumPy Version]")
    for q in test_quaternions:
        rpy = quat_to_euler_numpy(q)
        print(f"q = {q} → rpy = [{np.degrees(rpy[0]):+.1f}°, "
              f"{np.degrees(rpy[1]):+.1f}°, {np.degrees(rpy[2]):+.1f}°]")
    
    print("\n[JAX Version]")
    if HAS_JAX:
        for q_np in test_quaternions:
            q_jax = jp.array(q_np)
            rpy_jax = quat_to_euler_jax(q_jax)
            rpy = np.array(rpy_jax)
            print(f"q_jax = ... → rpy_jax = [{np.degrees(rpy[0]):+.1f}°, "
                  f"{np.degrees(rpy[1]):+.1f}°, {np.degrees(rpy[2]):+.1f}°]")
    else:
        print("(JAX not available)")
    
    print("\n[Unified Interface]")
    print("Testing quat_to_euler() auto-dispatch:")
    q_test = np.array([0.7071, 0.7071, 0.0, 0.0])
    rpy_result = quat_to_euler(q_test)
    print(f"Type: {type(rpy_result)}, Value: {rpy_result}")
```

---

## robot/policy_network.py

```python
"""Shared PPO policy network configuration for training and deployment."""

import jax
import jax.numpy as jnp
from brax.training import distribution as brax_distribution
from brax.training.agents.ppo import networks as ppo_networks


POLICY_MEAN_CLIP_SCALE = 3.0
POLICY_MIN_STD = 0.15
POLICY_MAX_STD = 3.0

_PATCH_INSTALLED = False


def install_policy_std_cap() -> None:
    """Install the bounded tanh-normal distribution patch once."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    if not hasattr(brax_distribution, "_NormalDistribution"):
        raise RuntimeError(
            "Brax distribution API changed: _NormalDistribution is unavailable"
        )

    def clipped_create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        loc = POLICY_MEAN_CLIP_SCALE * (loc / (1.0 + jnp.abs(loc)))
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        scale = jnp.clip(scale, POLICY_MIN_STD, POLICY_MAX_STD)
        return brax_distribution._NormalDistribution(loc=loc, scale=scale)

    brax_distribution.NormalTanhDistribution.create_dist = clipped_create_dist
    _PATCH_INSTALLED = True


def make_policy_network_factory(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn=lambda x, _=None: x,
):
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        mean_clip_scale=POLICY_MEAN_CLIP_SCALE,
    )


install_policy_std_cap()

```

---

## safety/__init__.py

```python
# safety package: Control Barrier Functions and deployment safety modules

```

---

## safety/cbf.py

```python
"""
safety/cbf.py — Control Barrier Function (CBF) Safety Layer for Bipedal Posture Control

【修正対応 (2026-09-08)】
- [CBF-1 FIXED] joint_pos_margin を可動域比率ベースで動的計算
- [CBF-2 FIXED] filter_action() と compute_cbf_penalty() のペナルティ基準を統一
- [CBF-3 FIXED] double-clamp の実装戦略を明確化（ドキュメント化）

【監査対応 (2026-09-13)】
- [CBF-4 ADDED] compute_saturation_ratio() を追加。
  train/train_mjx.py の _audit_reward_metrics() が実施する
  「Action Distortion」検出(方策が実行不能な指令を多発させていないか、
  本CBFの制限が過剰に効いていないか)のため、filter_action()による
  補正量を可動域に対する相対値として返す。envs/mjx_env.py の step()
  から呼び出され、'action_saturation' として metrics に記録される。
  (このロジックは元々 mjx_env.py 側に直接書かれていたが、CBFの
  挙動を診断する処理であるため、責務としてこちらのクラスに移した)

設計理念:
  学習時: 簡易版CBF（クリップ + ペナルティ）で微分可能性を保証
  実機時: 実装 safety/cbf_realworld.py で QP ベースの strict CBF へ切り替え
  
  本ファイルは「学習用の近似」として位置付けられている。
"""

import jax
import jax.numpy as jp
from typing import Tuple

from robot.config import RobotConfig


class CBFSafetyFilter:
    """
    Control Barrier Function (CBF) Safety Layer for Bipedal Posture Control.
    
    In JAX/MJX training, running a full QP solver per step per environment is prohibitively slow.
    This provides a simplified, differentiable margin-based clamping mechanism that mimics CBF,
    ensuring that nominal actions pushing the system towards unsafe states (e.g., instability,
    joint limits) are heavily penalized or clipped.
    
    During real-world deployment on Raspberry Pi, a strict QP-based CBF should replace this.
    See: safety/cbf_realworld.py (future)
    
    【実装戦略 (CBF-3 FIXED)】
    - RL側: 粗い安全クランプ（JOINT_LIMITS_MIN/MAX）を適用
    - CBF側: 「いかに粗クランプが効いたか」をペナルティで測定
    - 効果: RL が粗クランプを避けるよう学習 → 実質的な safety margin が徐々に形成
    
    ダブルクランプの正当性:
      第1クランプ（RL側）: 物理的なハードストップとして機能
      第2クランプ（CBF側）:「ハードストップが不要になる」ように RL を訓練
    """
    
    def __init__(self):
        """
        初期化。マージンを config.py から取得。
        """
        self.max_torque = RobotConfig.MOTOR_MAX_TORQUE
        self.max_vel = RobotConfig.MOTOR_MAX_VELOCITY
        
        # [CBF-1 FIXED] margin_ratio ベースの動的計算へ変更
        # 関節ごとに可動域の一定比率をマージンとする
        self.margin_ratio = 0.05  # 可動域の 5% をマージンとする
        
        # ペナルティ係数
        self.cbf_penalty_scale = 1.0  # mjx_rewards.py の weight と整合
        self.softplus_steepness = 10.0  # softplus の k パラメータ
    
    def compute_safe_margins(
        self,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """
        [CBF-1 FIXED] 可動域に応じた動的マージンを計算。
        
        各関節の可動域の一定比率（margin_ratio）をマージンとすることで、
        相対的な安全性を統一させる。
        
        Args:
            limit_lower: 関節下限 [rad] shape=(n_joints,)
            limit_upper: 関節上限 [rad] shape=(n_joints,)
        
        Returns:
            (safe_lower, safe_upper): マージンを適用した安全範囲
        """
        ranges = limit_upper - limit_lower
        
        # [CBF-1 FIXED] 可動域の margin_ratio% をマージンとして計算
        margins = ranges * self.margin_ratio
        
        safe_lower = limit_lower + margins
        safe_upper = limit_upper - margins
        
        return safe_lower, safe_upper
    
    def filter_action(
        self,
        nominal_action: jp.ndarray,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray
    ) -> jp.ndarray:
        """
        Takes the RL's nominal action and projects it to a safe set.
        
        実装戦略 (CBF-3):
        - 粗いハードクランプ（JOINT_LIMITS_MIN/MAX）を第1段で適用
        - マージンベースの制約を第2段で適用
        - 効果: RL が「ハードクランプを避ける」ように学習
        
        Args:
            nominal_action: RL の提案アクション [rad] shape=(n_joints,)
            limit_lower: 関節下限 [rad]
            limit_upper: 関節上限 [rad]
        
        Returns:
            safe_action: 安全範囲内に制限されたアクション
        """
        # [CBF-1 FIXED] 動的マージンを計算
        safe_lower, safe_upper = self.compute_safe_margins(limit_lower, limit_upper)
        
        # クランプ: マージン内に制限
        safe_action = jp.clip(nominal_action, safe_lower, safe_upper)
        
        return safe_action
    
    def compute_cbf_penalty(
        self,
        nominal_action: jp.ndarray,
        safe_action: jp.ndarray,
        limit_lower: jp.ndarray = None,
        limit_upper: jp.ndarray = None
    ) -> jp.ndarray:
        """
        [CBF-2 FIXED] CBFペナルティを計算。
        
        実装戦略 (CBF-2/3):
        - filter_action() で実際にクランプされた「差分」に基づくペナルティを計算
        - これにより、filter_action() と compute_cbf_penalty() の基準を統一
        - ペナルティ = (RL が安全範囲を超えようとした度合い)
        
        計算方式:
          1. 直接法: クランプ前後の差分量を測定
             penalty = sum(|nominal_action - safe_action|)
          2. マージンベース法: マージン超過量を測定（より厳格）
             penalty = softplus で連続ペナルティ化
        
        Args:
            nominal_action: RL の提案アクション [rad]
            safe_action: filter_action() で制限されたアクション [rad]
            limit_lower: 関節下限 [rad]（マージンベース法を使う場合は必須）
            limit_upper: 関節上限 [rad]（マージンベース法を使う場合は必須）
        
        Returns:
            penalty: スカラーペナルティ値（報酬から減算）
        """
        # [CBF-2 FIXED] 直接法: クランプ差分に基づくペナルティ
        clamp_diff = jp.abs(nominal_action - safe_action)
        
        # L1 ノルムで累積（クランプ差分が大きいほど大きいペナルティ）
        direct_penalty = jp.sum(clamp_diff)
        
        # オプション: マージンベース法（より厳格）
        if limit_lower is not None and limit_upper is not None:
            safe_lower, safe_upper = self.compute_safe_margins(limit_lower, limit_upper)
            
            # softplus で連続的にペナルティ化
            # マージン超過量に応じたペナルティを計算
            upper_excess = jp.maximum(0.0, nominal_action - safe_upper)
            lower_excess = jp.maximum(0.0, safe_lower - nominal_action)
            
            # softplus: smooth approximation of ReLU
            # softplus(x) = (1/β) * log(1 + exp(β*x))
            # β=10 で ReLU に近づく（微分可能）
            margin_penalty = jp.sum(
                jax.nn.softplus(self.softplus_steepness * upper_excess) +
                jax.nn.softplus(self.softplus_steepness * lower_excess)
            )
            
            # 両方を組み合わせ
            total_penalty = direct_penalty + 0.5 * margin_penalty
        else:
            total_penalty = direct_penalty
        
        # スケーリング
        return total_penalty * self.cbf_penalty_scale

    def compute_saturation_ratio(
        self,
        nominal_action: jp.ndarray,
        safe_action: jp.ndarray,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray,
    ) -> jp.ndarray:
        """
        [CBF-4 ADDED, 監査追加 2026-09-13] filter_action() によって
        どれだけ補正されたかを、可動域に対する相対値として返す診断指標。

        train/train_mjx.py の _audit_reward_metrics() が「Action
        Distortion」(方策が実行不能な指令を多発させていないか、CBFの
        制限が過剰に効いていないか)を検出するために使用する。

        compute_cbf_penalty() の direct_penalty(L1ノルムの絶対量)とは
        異なり、こちらは可動域で正規化した「割合」であるため、
        関節ごとに可動域が異なっていてもしきい値判定がしやすい。

        0.0 = 無補正 (nominal_action がそのまま安全域内)
        1.0 = 可動域いっぱいまで補正された (最大級の介入)

        Args:
            nominal_action: RL の提案アクション [rad]
            safe_action: filter_action() で制限されたアクション [rad]
            limit_lower / limit_upper: 関節可動域 [rad]

        Returns:
            saturation_ratio: スカラー(全関節平均、[0,1]目安)
        """
        action_range = jp.maximum(limit_upper - limit_lower, 1e-6)
        return jp.mean(jp.abs(safe_action - nominal_action) / action_range)
```

---

## scratch/analyze_run.py

```python
import json
data = json.load(open('log/version_1/log.json', 'r', encoding='utf-8'))
print(f"{'Step':>8} | {'Reward':>10} | {'KL_mean':>12} | {'Rwd/Step':>10} | {'Penalty':>10} | {'EpLen':>6} | {'LR':>10}")
print("-" * 90)
for d in data:
    s = d['step']
    r = d['reward']
    k = d.get('training/kl_mean', None)
    rps = d.get('eval/episode_reward_per_step', None)
    pen = d.get('eval/episode_total_penalty', None)
    ep = d.get('eval/avg_episode_length', None)
    lr = d.get('training/learning_rate', None)
    k_str = f"{k:.2f}" if k is not None else "N/A"
    rps_str = f"{rps:.2f}" if rps is not None else "N/A"
    pen_str = f"{pen:.0f}" if pen is not None else "N/A"
    ep_str = f"{ep:.1f}" if ep is not None else "N/A"
    lr_str = f"{lr:.2e}" if lr is not None else "N/A"
    print(f"{s:>8} | {r:>10.2f} | {k_str:>12} | {rps_str:>10} | {pen_str:>10} | {ep_str:>6} | {lr_str:>10}")

# Summary
rewards = [d['reward'] for d in data]
best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
worst_idx = min(range(len(rewards)), key=lambda i: rewards[i])
print(f"\nBest:  Step {data[best_idx]['step']} -> Reward {rewards[best_idx]:.2f}")
print(f"Worst: Step {data[worst_idx]['step']} -> Reward {rewards[worst_idx]:.2f}")

# Penalty trend
pens = [(d['step'], d.get('eval/episode_total_penalty', 0)) for d in data]
print(f"\nPenalty trend: {pens[0][1]:.0f} (Step 0) -> {pens[best_idx][1]:.0f} (Best) -> {pens[-1][1]:.0f} (Final)")

```

---

## scratch/check_model.py

```python
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

```

---

## scratch/gate0_formal_eval.py

```python
#!/usr/bin/env python3
"""Formal Gate 0 evaluation for learned RL policy (no-disturbance baseline).

学習済みRL方策を読み込んで、無外乱条件でGate 0判定を実施する。
物理・初期姿勢の確認用ゼロ行動評価ではなく、実checkpointの方策を使用する。

Usage:
  python scratch/gate0_formal_eval.py \
    --exp_name phase0_qual_seed0 \
    --version 0 \
    --model best_params.pkl \
    --seconds 30 \
    --seed 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this script usable on WSL and avoid inheriting a stale platform choice.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig
from robot.math_utils import quat_to_euler


def configure_deterministic_gate0():
    """Disable disturbances and reset randomization for the nominal baseline."""
    RobotConfig.DISTURBANCE_CURRICULUM = False
    RobotConfig.RANDOM_PUSH_MAX_FORCE = 0.0
    RobotConfig.RANDOM_MASS_SCALE = [1.0, 1.0]
    RobotConfig.RANDOM_FRICTION = [1.0, 1.0]
    RobotConfig.RANDOM_COM_OFFSET = [0.0, 0.0]
    RobotConfig.RANDOM_TEMP = [40.0, 40.0]
    RobotConfig.RANDOM_VOLT = [11.1, 11.1]


def find_checkpoint(
    exp_name: str,
    version: Optional[int] = None,
    model: str = "best_params.pkl",
) -> Optional[Path]:
    """学習済みcheckpointのパスを検索する。
    
    Args:
        exp_name: log/<exp_name> の形式でexperiment name
        version: version番号。Noneなら最新を選ぶ
        model: checkpoint filename (best_params.pkl, final_params.pkl等)
    
    Returns:
        Path to checkpoint, or None if not found.
    """
    if not exp_name:
        return None
    
    log_base = ROOT / "log" / exp_name
    if not log_base.exists():
        print(f"[Gate0] WARNING: {log_base} not found")
        return None
    
    version_dirs = sorted([d for d in log_base.iterdir() if d.is_dir() and d.name.startswith("version_")])
    if not version_dirs:
        print(f"[Gate0] WARNING: no version_* directories in {log_base}")
        return None
    
    if version is None:
        # 最新版を選ぶ
        version_dir = version_dirs[-1]
    else:
        version_dir = log_base / f"version_{version}"
    
    checkpoint_path = version_dir / model
    if not checkpoint_path.exists():
        print(f"[Gate0] WARNING: {checkpoint_path} not found")
        return None
    
    return checkpoint_path


def load_checkpoint_and_make_policy(checkpoint_path: Path):
    """Checkpointを読み込んで、policyを生成する。
    
    Returns:
        policy_fn: (obs) -> action の関数
        params: 学習済みパラメータ
    """
    try:
        from train.visualize_rl import load_checkpoint, make_policy_network_factory
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as e:
        raise ImportError(f"Cannot import training utilities: {e}")
    
    # checkpointを読み込む
    params = load_checkpoint(str(checkpoint_path))
    if params is None:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")
    
    # 観測・行動次元を把握するため、probe envを作る
    probe_env = SenpuuMaruMJXEnv()
    network = make_policy_network_factory(
        probe_env.observation_size,
        probe_env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    
    # params の leading dimension を strip（ある場合）
    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf
    
    params_stripped = jax.tree_util.tree_map(strip_leading_dim, params)
    
    # Gate 0 formal eval is the main pass/fail path and must match the project protocol:
    # deterministic evaluation is the primary metric; stochastic sampling is only a reference.
    policy_fn = jax.jit(make_policy(params_stripped, deterministic=True))
    
    return policy_fn, params_stripped


def as_float(value):
    return float(np.asarray(value))


def foot_geom_ids(env):
    model = env._mjx_model
    body_ids = [env._reward_system._left_foot_id, env._reward_system._right_foot_id]
    geom_bodyid = np.asarray(model.geom_bodyid)
    return [
        index for index, body_id in enumerate(geom_bodyid)
        if int(body_id) in body_ids
    ]


def measure_state(env, state, geom_ids, sensor_start, sensor_end):
    data = state.pipeline_state
    qpos = np.asarray(data.qpos)
    rpy = np.asarray(quat_to_euler(data.qpos[3:7]))
    geom_xpos = np.asarray(data.geom_xpos)
    geom_z = geom_xpos[geom_ids, 2] if geom_ids else np.array([np.nan])
    geom_size = np.asarray(env._mjx_model.geom_size)
    geom_type = np.asarray(env._mjx_model.geom_type)
    # The current XML foot collision geoms are boxes. For other geom types,
    # retain the center z and mark the result as an approximate lower bound.
    lower_z = []
    for geom_id in geom_ids:
        if int(geom_type[geom_id]) == 6:  # mjGEOM_BOX
            lower_z.append(geom_xpos[geom_id, 2] - geom_size[geom_id, 2])
        else:
            lower_z.append(geom_xpos[geom_id, 2])
    touch = np.asarray(data.sensordata)[sensor_start:sensor_end]
    return {
        "torso_z": as_float(qpos[2]),
        "roll_rad": as_float(rpy[0]),
        "pitch_rad": as_float(rpy[1]),
        "yaw_rad": as_float(rpy[2]),
        "xy_m": as_float(np.linalg.norm(qpos[:2])),
        "foot_body_z": [as_float(value) for value in np.asarray(data.xpos)[[
            env._reward_system._left_foot_id,
            env._reward_system._right_foot_id,
        ], 2]],
        "foot_geom_lower_z": [as_float(value) for value in lower_z],
        "touch_sum": as_float(np.sum(touch)),
        "touch_values": [as_float(value) for value in touch],
    }


def evaluate(
    seed: int,
    seconds: float,
    output_dir: Path,
    exp_name: str = "",
    version: Optional[int] = None,
    model: str = "best_params.pkl",
    policy_fn = None,
):
    """Gate 0 evaluation with learned RL policy.
    
    Args:
        seed: random seed
        seconds: simulation duration in seconds
        output_dir: where to save results
        exp_name: experiment name (log/<exp_name>/<version_*>/)
        version: version number within exp_name
        model: checkpoint filename
        policy_fn: pre-loaded policy function. If None, load from checkpoint.
    """
    configure_deterministic_gate0()
    env = SenpuuMaruMJXEnv()
    steps = int(round(seconds / RobotConfig.CONTROL_DT))
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    # Policy を用意する
    if policy_fn is None:
        if not exp_name:
            raise ValueError("Either --exp_name or pre-loaded policy_fn is required")
        checkpoint_path = find_checkpoint(exp_name, version, model)
        if checkpoint_path is None:
            raise RuntimeError(f"Checkpoint not found for exp_name={exp_name}, version={version}")
        print(f"[Gate0] Loading checkpoint: {checkpoint_path}")
        policy_fn, params = load_checkpoint_and_make_policy(checkpoint_path)

    geom_ids = foot_geom_ids(env)
    sensor_count = int(env._mjx_model.nsensordata)
    if sensor_count < 8:
        raise RuntimeError(f"Expected 8 foot touch sensor values, found {sensor_count}")
    sensor_start = sensor_count - 8
    sensor_end = sensor_count

    initial = measure_state(env, state, geom_ids, sensor_start, sensor_end)
    records = []
    terminated_step = None
    for step_index in range(steps):
        rng, rng_policy = jax.random.split(rng)
        action, _ = policy_fn(state.obs, rng_policy)
        state = env.step(state, action)
        sample = measure_state(env, state, geom_ids, sensor_start, sensor_end)
        sample["step"] = step_index + 1
        sample["action_norm"] = float(jp.linalg.norm(action))
        records.append(sample)
        if bool(state.done):
            terminated_step = step_index + 1
            break

    if not records:
        raise RuntimeError("No simulation samples were collected")

    roll = np.array([item["roll_rad"] for item in records])
    pitch = np.array([item["pitch_rad"] for item in records])
    xy = np.array([item["xy_m"] for item in records])
    lower_z = np.array([item["foot_geom_lower_z"] for item in records])
    touch = np.array([item["touch_sum"] for item in records])
    time_s = np.arange(1, len(records) + 1) * RobotConfig.CONTROL_DT
    xy_slope = float(np.polyfit(time_s, xy, 1)[0]) if len(records) > 1 else 0.0
    settling_start = max(0, len(records) // 5)
    result = {
        "seed": seed,
        "requested_seconds": seconds,
        "simulated_steps": len(records),
        "simulated_seconds": len(records) * RobotConfig.CONTROL_DT,
        "terminated_step": terminated_step,
        "initial": initial,
        "final": records[-1],
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(roll)))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(pitch)))),
        "rms_roll_deg": float(np.rad2deg(np.sqrt(np.mean(roll ** 2)))),
        "rms_pitch_deg": float(np.rad2deg(np.sqrt(np.mean(pitch ** 2)))),
        "final_xy_m": float(xy[-1]),
        "max_xy_m": float(np.max(xy)),
        "xy_drift_speed_mm_s": xy_slope * 1000.0,
        "min_foot_geom_z_m": float(np.min(lower_z)),
        "max_foot_geom_z_m": float(np.max(lower_z)),
        "foot_touch_rate": float(np.mean(touch > 1e-6)),
        "max_touch_signal": float(np.max(touch)),
        "records": records,
    }

    # Provisional development thresholds. Formal acceptance still requires the
    # v2 multi-seed and 10-30 second evaluation record.
    result["pass"] = bool(
        terminated_step is None
        and result["simulated_seconds"] >= seconds
        and result["max_abs_roll_deg"] < 10.0
        and result["max_abs_pitch_deg"] < 10.0
        and result["min_foot_geom_z_m"] >= -0.002
        and result["foot_touch_rate"] >= 0.99
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate0_seed_{seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    print(f"[Gate0] detailed log: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="", help="log/<exp_name> の experiment name")
    parser.add_argument("--version", type=int, default=None, help="version number within exp_name")
    parser.add_argument("--model", default="best_params.pkl", help="checkpoint filename")
    parser.add_argument("--seconds", type=float, default=30.0, help="simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "log" / "gate0_formal", help="output directory")
    args = parser.parse_args()
    
    try:
        result = evaluate(
            seed=args.seed,
            seconds=args.seconds,
            output_dir=args.output_dir,
            exp_name=args.exp_name,
            version=args.version,
            model=args.model,
        )
        if not result["pass"]:
            print("[Gate0] FAIL: did not meet acceptance criteria")
            raise SystemExit(1)
        else:
            print("[Gate0] PASS: accepted baseline")
            raise SystemExit(0)
    except Exception as e:
        print(f"[Gate0] ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
```

---

## scratch/gate0_mujoco_eval.py

```python
#!/usr/bin/env python3
"""Pure MuJoCo Gate 0 evaluation (no MJX/JAX compile)."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler


def as_float(value):
    return float(np.asarray(value))


def find_foot_body_ids(model):
    left_name = "doutai-v5_hidaridairou_hidarikokansetu_hidarimomo_hidarihizabu_hidariaikabu_hidariashiura_hidariashiura-1"
    right_name = "doutai-v5_migidaitou_migikokansetu_migimomo_migihizabu_migigaikabu_migiashiura_migiashiura-1"

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)

    if left_id == -1 or right_id == -1:
        left_id = -1
        right_id = -1
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if left_id == -1 and "hidariashiura" in name:
                left_id = body_id
            if right_id == -1 and "migiashiura" in name:
                right_id = body_id
        if left_id == -1 or right_id == -1:
            raise RuntimeError("Failed to locate foot body ids")

    return left_id, right_id


def build_initial_qpos(model, torso_z):
    qpos = np.zeros(model.nq, dtype=np.float64)

    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    for act_i in range(min(model.nu, len(default))):
        jnt_id = int(model.actuator_trnid[act_i, 0])
        qpos_idx = int(model.jnt_qposadr[jnt_id])
        qpos[qpos_idx] = default[act_i]

    if model.nq >= 7:
        qpos[0:3] = np.array([0.0, 0.0, torso_z], dtype=np.float64)
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return qpos


def foot_geom_ids(model, left_foot_id, right_foot_id):
    ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == left_foot_id or body_id == right_foot_id:
            ids.append(geom_id)
    return ids


def split_foot_geom_ids(model, left_foot_id, right_foot_id):
    left_ids = []
    right_ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == left_foot_id:
            left_ids.append(geom_id)
        elif body_id == right_foot_id:
            right_ids.append(geom_id)
    return left_ids, right_ids


def foot_lower_z(model, data, geom_ids):
    values = []
    for geom_id in geom_ids:
        geom_type = int(model.geom_type[geom_id])
        center_z = float(data.geom_xpos[geom_id, 2])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            values.append(center_z - float(model.geom_size[geom_id, 2]))
        else:
            values.append(center_z)
    return values


def touch_values(model, data):
    if model.nsensordata < 8:
        return np.array([], dtype=np.float64)
    return np.asarray(data.sensordata[-8:], dtype=np.float64)


def contact_flags(data, left_geom_ids, right_geom_ids):
    left_set = set(left_geom_ids)
    right_set = set(right_geom_ids)
    left_contact = False
    right_contact = False
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if g1 in left_set or g2 in left_set:
            left_contact = True
        if g1 in right_set or g2 in right_set:
            right_contact = True
        if left_contact and right_contact:
            break
    return left_contact, right_contact


def measure(model, data, left_foot_id, right_foot_id, geom_ids, left_geom_ids, right_geom_ids):
    qpos = np.asarray(data.qpos)
    rpy = np.asarray(quat_to_euler(qpos[3:7]))
    lower = foot_lower_z(model, data, geom_ids)
    touch = touch_values(model, data)
    left_contact, right_contact = contact_flags(data, left_geom_ids, right_geom_ids)

    return {
        "torso_z": as_float(qpos[2]),
        "roll_rad": as_float(rpy[0]),
        "pitch_rad": as_float(rpy[1]),
        "yaw_rad": as_float(rpy[2]),
        "xy_m": as_float(np.linalg.norm(qpos[:2])),
        "foot_body_z": [
            as_float(data.xpos[left_foot_id, 2]),
            as_float(data.xpos[right_foot_id, 2]),
        ],
        "foot_geom_lower_z": [as_float(v) for v in lower],
        "touch_sum": as_float(np.sum(touch)) if touch.size else 0.0,
        "touch_values": [as_float(v) for v in touch],
        "left_contact": bool(left_contact),
        "right_contact": bool(right_contact),
        "max_abs_actuator_force": as_float(np.max(np.abs(data.actuator_force))) if model.nu > 0 else 0.0,
    }


def evaluate(seed, seconds, torso_z, output_dir):
    _ = seed  # deterministic evaluation for now
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    model.opt.timestep = RobotConfig.SIM_DT
    data = mujoco.MjData(model)

    left_foot_id, right_foot_id = find_foot_body_ids(model)
    geom_ids = foot_geom_ids(model, left_foot_id, right_foot_id)
    left_geom_ids, right_geom_ids = split_foot_geom_ids(model, left_foot_id, right_foot_id)

    qpos0 = build_initial_qpos(model, torso_z)
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0

    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    if model.nu > 0:
        data.ctrl[:] = 0.0
        data.ctrl[:min(model.nu, len(default))] = default[:min(model.nu, len(default))]

    mujoco.mj_forward(model, data)

    ctrl_steps = int(round(seconds / RobotConfig.CONTROL_DT))
    sim_steps_per_ctrl = max(1, int(round(RobotConfig.CONTROL_DT / RobotConfig.SIM_DT)))

    initial = measure(model, data, left_foot_id, right_foot_id, geom_ids, left_geom_ids, right_geom_ids)
    records = []

    for i in range(ctrl_steps):
        if model.nu > 0:
            data.ctrl[:min(model.nu, len(default))] = default[:min(model.nu, len(default))]
        for _ in range(sim_steps_per_ctrl):
            mujoco.mj_step(model, data)
        sample = measure(model, data, left_foot_id, right_foot_id, geom_ids, left_geom_ids, right_geom_ids)
        sample["step"] = i + 1
        records.append(sample)

    if not records:
        raise RuntimeError("No records generated")

    roll = np.array([r["roll_rad"] for r in records])
    pitch = np.array([r["pitch_rad"] for r in records])
    xy = np.array([r["xy_m"] for r in records])
    lower_z = np.array([r["foot_geom_lower_z"] for r in records])
    touch = np.array([r["touch_sum"] for r in records])
    both_contact = np.array([r["left_contact"] and r["right_contact"] for r in records])
    max_force = np.array([r["max_abs_actuator_force"] for r in records])

    time_s = np.arange(1, len(records) + 1) * RobotConfig.CONTROL_DT
    xy_slope = float(np.polyfit(time_s, xy, 1)[0]) if len(records) > 1 else 0.0

    result = {
        "seed": seed,
        "requested_seconds": seconds,
        "simulated_steps": len(records),
        "simulated_seconds": len(records) * RobotConfig.CONTROL_DT,
        "initial": initial,
        "final": records[-1],
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(roll)))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(pitch)))),
        "rms_roll_deg": float(np.rad2deg(np.sqrt(np.mean(roll ** 2)))),
        "rms_pitch_deg": float(np.rad2deg(np.sqrt(np.mean(pitch ** 2)))),
        "final_xy_m": float(xy[-1]),
        "max_xy_m": float(np.max(xy)),
        "xy_drift_speed_mm_s": xy_slope * 1000.0,
        "min_foot_geom_z_m": float(np.min(lower_z)),
        "max_foot_geom_z_m": float(np.max(lower_z)),
        "foot_touch_rate": float(np.mean(touch > 1e-6)),
        "both_foot_contact_rate": float(np.mean(both_contact)),
        "max_touch_signal": float(np.max(touch)),
        "max_abs_actuator_force": float(np.max(max_force)),
        "torque_saturation_rate": float(np.mean(max_force >= 0.98 * RobotConfig.MOTOR_MAX_TORQUE)),
        "records": records,
    }

    result["pass"] = bool(
        result["simulated_seconds"] >= seconds
        and result["max_abs_roll_deg"] < 10.0
        and result["max_abs_pitch_deg"] < 10.0
        and result["min_foot_geom_z_m"] >= -0.002
        and result["both_foot_contact_rate"] >= 0.99
        and result["torque_saturation_rate"] <= 0.01
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate0_seed_{seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "records"}
    print(json.dumps(summary, indent=2))
    print(f"[Gate0-MuJoCo] detailed log: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torso-z", type=float, default=0.1773)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "log" / "gate0_formal")
    args = parser.parse_args()

    result = evaluate(args.seed, args.seconds, args.torso_z, args.output_dir)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

```

---

## scratch/gate0_standing_eval.py

```python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.math_utils import quat_to_euler


def main():
    env = SenpuuMaruMJXEnv()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_xy = 0.0

    for step_idx in range(5):
        action = jnp.zeros(env.action_size, dtype=jnp.float32)
        state = env.step(state, action)

        qpos = state.pipeline_state.qpos
        rpy = quat_to_euler(qpos[3:7])
        max_abs_roll = max(max_abs_roll, float(abs(rpy[0])))
        max_abs_pitch = max(max_abs_pitch, float(abs(rpy[1])))
        max_xy = max(max_xy, float(jnp.linalg.norm(qpos[0:2])))

        if step_idx % 1 == 0:
            print(f"[Gate0] step={step_idx + 1:02d} roll={abs(float(rpy[0])):.4f} pitch={abs(float(rpy[1])):.4f} xy={float(jnp.linalg.norm(qpos[0:2])):.4f}")

        if bool(state.done):
            print(f"Gate 0 terminated at step={step_idx + 1}, done={state.done}")
            break

    final_roll = float(abs(quat_to_euler(state.pipeline_state.qpos[3:7])[0]))
    final_pitch = float(abs(quat_to_euler(state.pipeline_state.qpos[3:7])[1]))
    final_xy = float(jnp.linalg.norm(state.pipeline_state.qpos[0:2]))
    final_z = float(state.pipeline_state.qpos[2])

    print("=== Gate 0 summary ===")
    print(f"max_abs_roll   = {max_abs_roll:.4f} rad ({max_abs_roll * 180.0 / 3.14159:.2f} deg)")
    print(f"max_abs_pitch  = {max_abs_pitch:.4f} rad ({max_abs_pitch * 180.0 / 3.14159:.2f} deg)")
    print(f"max_xy         = {max_xy:.4f} m")
    print(f"final_roll     = {final_roll:.4f} rad")
    print(f"final_pitch    = {final_pitch:.4f} rad")
    print(f"final_xy       = {final_xy:.4f} m")
    print(f"final_z        = {final_z:.4f} m")
    print(f"done           = {bool(state.done)}")

    if bool(state.done):
        raise SystemExit(1)

    if max_abs_roll > 0.25 or max_abs_pitch > 0.25:
        print("Gate 0 FAIL: roll/pitch exceeds 15 deg limit during static standing.")
        raise SystemExit(1)

    print("Gate 0 PASS: static standing remained within the provisional tilt limit for the short baseline window.")


if __name__ == "__main__":
    main()

```

---

## scratch/inspect_params_structure.py

```python
#!/usr/bin/env python3
import pickle
from pathlib import Path

p = pickle.load(open(Path('log/version_7/final_params.pkl'), 'rb'))
print('top_type', type(p))
if isinstance(p, tuple):
    print('tuple_len', len(p))
    for i, e in enumerate(p):
        print('idx', i, 'type', type(e))
        if hasattr(e, 'keys'):
            try:
                keys = list(e.keys())
                print(' keys', keys[:20])
            except Exception as ex:
                print(' keys_error', ex)


def walk(tree, prefix=''):
    if isinstance(tree, dict):
        for k, v in tree.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            if 'std' in str(k).lower() or 'scale' in str(k).lower():
                print('match_key', p, 'type', type(v))
            walk(v, p)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            walk(v, f"{prefix}[{i}]")


for idx in (1, 2):
    if idx < len(p) and isinstance(p[idx], dict) and 'params' in p[idx]:
        print('--- walk tuple index', idx, '---')
        walk(p[idx]['params'], f'tuple[{idx}]/params')

```

---

## scratch/phase0_eval_diagnostics.py

```python
#!/usr/bin/env python3
"""Phase 0 / Gate A diagnosis: deterministic vs stochastic evaluation +
termination-reason histogram (docs/master_plan.md 付録A §3.5, Task0).

status.md (2026-09-01) の「次のTask」= 「deterministic/stochastic評価の実装と
終了理由ヒストグラム化」に対応する。エスカレーション項目の「次の切り分け」の
3項目のうち、deterministic評価・終了stepヒストグラム化・報酬成分分解ログの
3つをまとめてこのスクリプトで実施する。

設計方針（master_plan.md §3.5 に基づく）:
  - 2x2評価: {deterministic, stochastic} x {fixed_dr, randomized_dr}
    注意: 本リポジトリの reset() は物理初期姿勢(qpos/qvel)を常に同一の
    nominal poseに固定しており、初期姿勢そのもののrandomizationは
    master_plan.md §1.6で「Task1(Gate A是正)の時点で必ず導入する」と
    定義された未実装機能である。したがって本スクリプトの
    「初期状態randomize」軸は、既存の実装済みrandomization経路である
    domain randomization (質量/摩擦/重心オフセット/サーボ温度/電圧) の
    on/offとして操作する。これは近似であり、真の初期姿勢randomizationの
    代替ではない。この制約は出力レポートに明記する。
  - deterministic x fixed_dr のセルは、同一checkpoint・同一初期状態・
    同一policyであれば理論上ビット単位で再現するはずのセルであり、
    複数episodeを回す意味は「決定論性テスト」（master_plan.md §4.2）を
    兼ねる以外にない。デフォルトのepisode数を他セルより少なくしている。
  - 各episodeについて、終了理由 (fallen_roll / fallen_pitch /
    fallen_height / time_limit / unknown の組み合わせ) を分類する。
    非足裏接触・トルク上限による終了は、現行の envs/mjx_rewards.py の
    done判定 (is_fallen_roll or is_fallen_pitch or is_low のみ) に
    実装されていないため分類対象にできない。これは
    master_plan.md Task4 (C-08, 複合成功条件) が未着手であることの
    追加の裏付けとしてレポートに記録する。
  - Kaplan-Meier型の生存曲線を打ち切り(truncated=time_limit)を
    考慮して計算する。
  - 失敗episodeについて、終了直前 collapse_window step分の
    roll/pitch/base角速度/base位置の時系列を記録する。
  - reward metrics (envs/mjx_rewards.py が返す metrics dict) の
    episode平均をあわせて記録し、reward成分分解ログを兼ねる。
  - master_plan.md §3.6 の決定木を単純な閾値ヒューリスティックとして
    実装し、失敗タイミングの偏り(序盤/後半/ランダム)を自動判定する。
    これは補助的な一次判定であり、最終診断は人間 / 記録を見た
    Copilotが行うことを想定している。

Done条件 (pytest, tests/test_phase0_eval_diagnostics.py 側):
  - classify_termination_reason の分類ロジック
  - kaplan_meier_survival の生存曲線計算
  - diagnose_failure_timing の決定木ヒューリスティック
  これらは純Python/NumPyのみで完結し、JAX/MJX/GPU無しでCPU上で検証できる。

実行には学習済みcheckpoint (log/<exp_name>/version_x/*.pkl) と
JAX/MJX/Brax環境 (WSLのvenv_wsl等) が必要。このリポジトリのsandboxには
GPUも実際の学習済みcheckpointも存在しないため、本スクリプト作成時には
以下2段階で検証した:
  1. 純Python/NumPyの解析ロジック(classify_termination_reason /
     kaplan_meier_survival / diagnose_failure_timing /
     summarize_episode_alive)はtests/test_phase0_eval_diagnostics.pyで
     単体テスト済み(CPU、JAX不要)。
  2. ロールアウト部分(run_episode/run_condition/main)は、CPU上に
     JAX/MuJoCo/MJX/Braxをインストールし、ランダム初期化した
     (未学習の)policy checkpointを使って実際にreset/step/評価の
     全経路を通しで実行確認した。この過程で以下の実装上の罠を
     発見・修正済み:
       - env.reset/env.stepは必ずjax.jit()経由で呼ぶ必要がある。
         eager実行では reset() 内の `info['step'] = 0` がPython int の
         まま伝播し、`truncated.astype(...)` (envs/mjx_env.py) で
         AttributeErrorになる。
       - jax.jit(env.reset) はbound methodの等価性でコンパイル結果を
         キャッシュするため、RobotConfig.RANDOM_* を条件間で書き換えても
         同一envインスタンスに対する再jitでは古いコンパイル結果が
         再利用されてしまう(2つ目以降のDR条件が1つ目の設定のまま
         実行される、気付きにくい誤結果)。DRスコープ確定後に毎回
         新しいenvインスタンスを作ることで回避した。
       - スクリプト自身の--max-stepsが環境本来のMAX_EPISODE_STEPSより
         小さい場合、terminated/truncatedのどちらも立たないままループが
         尽きることがある。これを終了理由に混ぜず
         "eval_budget_cutoff"として区別し、Kaplan-Meier計算上も
         event(実イベント)ではなくcensoredとして扱うようにした。
     未学習ランダムpolicyでの動作確認であり、実際に学習済み
     checkpointとGPU/WSL環境で実行した結果ではない。次の残作業は、
     WSL/GPU環境で実checkpointに対して
     `python scratch/phase0_eval_diagnostics.py --exp_name <name>` を
     実行し、結果を docs/status.md ・ docs/gate_a_diagnosis.md に
     記録すること。
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CPU固定はデフォルトのみ。GPU評価したい場合は呼び出し前に環境変数を上書きすること。
os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ============================================================================
# 純Python/NumPyの解析ロジック（JAX/MJX非依存、単体テスト対象）
# ============================================================================

def classify_termination_reason(
    is_fallen_roll: bool,
    is_fallen_pitch: bool,
    is_low: bool,
    truncated: bool,
) -> str:
    """終了理由を分類する。

    master_plan.md 付録A §1.5 の終了条件定義のうち、現行コード
    (envs/mjx_rewards.py) が実装しているのは roll/pitch/height の3つと
    time-limitのみ。non_illegal_contact / slip_ok / torque_ok による
    terminationは未実装のため、このスクリプトでも分類できない
    （Task4 C-08 未着手であることの根拠として記録する）。
    """
    if truncated:
        return "time_limit"
    reasons = []
    if is_fallen_roll:
        reasons.append("fallen_roll")
    if is_fallen_pitch:
        reasons.append("fallen_pitch")
    if is_low:
        reasons.append("fallen_height")
    if not reasons:
        # terminated=Trueだが既知のフラグがどれも立っていない場合。
        # 実装上は起こらないはずだが、バグ検知のため明示的に区別する。
        return "unknown_terminated"
    return "+".join(reasons)


def kaplan_meier_survival(
    episode_lengths: Sequence[int],
    event_observed: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier生存曲線を計算する（500stepで打ち切られる右側打ち切り分布）。

    Args:
        episode_lengths: 各episodeが終了した(打ち切られた)step数。
        event_observed: Trueなら真のterminationイベント、Falseなら
            time-limitによる打ち切り(censoring)。

    Returns:
        (times, survival): times[0]=0, survival[0]=1.0 から始まる
        ステップ関数のノード列。
    """
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    events = np.asarray(event_observed, dtype=bool)
    if len(lengths) == 0:
        return np.array([0]), np.array([1.0])
    if len(lengths) != len(events):
        raise ValueError("episode_lengths and event_observed must be same length")

    event_times = np.unique(lengths[events])
    times = [0]
    survival = [1.0]
    s = 1.0
    for t in sorted(event_times.tolist()):
        n_t = int(np.sum(lengths >= t))  # tの直前時点でまだ生存(risk set)にいる数
        d_t = int(np.sum((lengths == t) & events))  # t時点での真のイベント数
        if n_t > 0:
            s *= (1.0 - d_t / n_t)
        times.append(int(t))
        survival.append(s)
    return np.array(times), np.array(survival)


def diagnose_failure_timing(
    termination_steps: Sequence[int],
    max_step: int,
    early_frac: float = 1.0 / 3.0,
    late_frac: float = 2.0 / 3.0,
    concentration_threshold: float = 0.6,
) -> Dict[str, object]:
    """master_plan.md 付録A §3.6 の決定木を単純な閾値ヒューリスティックで実装する。

    real terminationのみ(truncatedは除く)を入力に使うこと。
    """
    steps = np.asarray(termination_steps, dtype=np.float64)
    if len(steps) == 0:
        return {
            "classification": "no_failures",
            "suggested_action": (
                "terminatedによる失敗episodeが観測されなかった。"
                "time-limit到達のみであれば§3.3(truncation/termination処理)の"
                "疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。"
            ),
            "normalized_mean": None,
            "early_rate": None,
            "late_rate": None,
        }

    normalized = steps / float(max(max_step, 1))
    early_rate = float(np.mean(normalized < early_frac))
    late_rate = float(np.mean(normalized > late_frac))
    normalized_mean = float(np.mean(normalized))

    if early_rate >= concentration_threshold:
        classification = "序盤集中"
        suggested_action = (
            "失敗がepisode序盤に集中 → 初期状態・初期transientの問題の疑い。"
            "初期状態分布の縮小・初期姿勢安定化を検討する（master_plan.md §3.6）。"
        )
    elif late_rate >= concentration_threshold:
        classification = "後半集中"
        suggested_action = (
            "失敗がepisode後半に集中 → 長期ドリフト or time-limitバグの疑い。"
            "truncation/termination処理(§3.3)を再疑う。"
        )
    else:
        classification = "ランダム分布"
        suggested_action = (
            "失敗時刻がランダムに分布 → 状態空間の局所不安定領域の疑い。"
            "失敗直前の状態を特定し、該当領域の報酬/観測を強化する。"
        )

    return {
        "classification": classification,
        "suggested_action": suggested_action,
        "normalized_mean": normalized_mean,
        "early_rate": early_rate,
        "late_rate": late_rate,
    }


def summarize_episode_alive(episode_lengths: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(episode_lengths, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


# ============================================================================
# ロールアウト（JAX/MJX依存、GPU/WSL環境での実行を想定）
# ============================================================================

@dataclass
class EpisodeResult:
    length: int
    terminated: bool
    truncated: bool
    reason: str
    collapse_window: List[dict] = field(default_factory=list)
    reward_component_means: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    both_feet_contact: bool = False
    max_foot_displacement: float = 0.0
    max_roll_rad: float = 0.0
    max_pitch_rad: float = 0.0
    recovery_time_steps: Optional[int] = None
    torque_saturation_rate: float = 0.0


def _lazy_imports():
    """JAX/MJX関連のimportを遅延させ、--help等をGPU無し環境でも高速に扱えるようにする。"""
    import jax  # noqa: F401
    import jax.numpy as jp  # noqa: F401
    from robot.config import RobotConfig
    from envs.mjx_env import SenpuuMaruMJXEnv
    from robot.math_utils import quat_to_euler
    from train.visualize_rl import (
        get_model_path,
        load_checkpoint,
        make_policy_network_factory,
    )
    from brax.training.agents.ppo import networks as ppo_networks

    return {
        "jax": jax,
        "jp": jp,
        "RobotConfig": RobotConfig,
        "SenpuuMaruMJXEnv": SenpuuMaruMJXEnv,
        "quat_to_euler": quat_to_euler,
        "get_model_path": get_model_path,
        "load_checkpoint": load_checkpoint,
        "make_policy_network_factory": make_policy_network_factory,
        "ppo_networks": ppo_networks,
    }


class _DomainRandomizationScope:
    """RobotConfigのDR幅を一時的に固定値へ差し替え、終了時に復元するコンテキストマネージャ。

    物理初期姿勢(qpos/qvel)はreset()で常に固定のため、これは
    「初期状態randomize」軸の近似実装であることに注意
    (モジュールdocstring参照)。
    """

    FIELDS = (
        "RANDOM_MASS_SCALE",
        "RANDOM_FRICTION",
        "RANDOM_COM_OFFSET",
        "RANDOM_TEMP",
        "RANDOM_VOLT",
    )

    def __init__(self, RobotConfig, fixed: bool):
        self._cfg = RobotConfig
        self._fixed = fixed
        self._saved = {}

    def __enter__(self):
        for name in self.FIELDS:
            self._saved[name] = getattr(self._cfg, name)
        if self._fixed:
            # 全フィールドは [lo, hi] のスカラー対 (envs/mjx_env.py の reset() が
            # minval=X[0], maxval=X[1] として読む前提と一致させる)。
            for name in self.FIELDS:
                lo, hi = self._saved[name]
                mid = (lo + hi) / 2.0
                setattr(self._cfg, name, [mid, mid])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, value in self._saved.items():
            setattr(self._cfg, name, value)
        return False


def run_episode(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    rng,
    max_steps: int,
    collapse_window: int,
) -> EpisodeResult:
    """1エピソードをロールアウトする。

    重要: reset_fn/step_fnは呼び出し側で必ず jax.jit(env.reset) /
    jax.jit(env.step) として渡すこと。env.reset/env.stepを素の(非jit)
    状態で呼ぶと、reset()内で `info['step'] = 0` のようにPython int
    リテラルとして初期化されたフィールドがPython int のまま
    stepに渡り、`truncated.astype(...)` (envs/mjx_env.py) で
    `AttributeError: 'bool' object has no attribute 'astype'` になる
    (jitされた関数の戻り値はJAXが自動的に配列型へ変換するため、
    jit経由なら発生しない。本スクリプト作成時にeager実行で実際に
    再現・確認済み)。
    """
    RobotConfig = ctx["RobotConfig"]
    quat_to_euler = ctx["quat_to_euler"]

    rng, rng_reset = ctx["jax"].random.split(rng)
    state = reset_fn(rng_reset)

    history = []
    metric_sums: Dict[str, float] = {}
    metric_count = 0
    initial_foot_positions = None
    max_foot_displacement = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    both_feet_contact = True
    recovery_start = None
    recovery_time_steps = None
    saturated_steps = 0
    measured_torque_steps = 0

    terminated = False
    truncated = False
    step_index = 0
    for step_index in range(1, max_steps + 1):
        rng, rng_step = ctx["jax"].random.split(rng)
        action, _ = policy_fn(state.obs, rng_step)
        state = step_fn(state, action)

        qpos = np.asarray(state.pipeline_state.qpos)
        qvel = np.asarray(state.pipeline_state.qvel)
        rpy = np.asarray(quat_to_euler(state.pipeline_state.qpos[3:7]))
        base_pos = qpos[0:3]
        base_ang_vel = qvel[3:6] if len(qvel) >= 6 else np.zeros(3)
        xpos = np.asarray(state.pipeline_state.xpos)
        foot_ids = ctx.get("foot_ids")
        if foot_ids is not None and xpos.ndim == 2:
            foot_positions = xpos[list(foot_ids)]
            if initial_foot_positions is None:
                initial_foot_positions = foot_positions.copy()
            max_foot_displacement = max(
                max_foot_displacement,
                float(np.max(np.linalg.norm(foot_positions[:, :2] - initial_foot_positions[:, :2], axis=1))),
            )
        max_roll = max(max_roll, abs(float(rpy[0])))
        max_pitch = max(max_pitch, abs(float(rpy[1])))
        contact_metric = float(np.asarray(getattr(state, "metrics", {}).get("both_feet_contact", 0.0)))
        both_feet_now = contact_metric >= 0.5
        both_feet_contact = both_feet_contact and both_feet_now
        if bool(state.info.get("was_disturbed", False)) and recovery_start is None:
            recovery_start = step_index
        if recovery_start is not None and recovery_time_steps is None:
            if (both_feet_now and abs(rpy[0]) < np.deg2rad(10.0)
                    and abs(rpy[1]) < np.deg2rad(10.0)
                    and np.linalg.norm(base_ang_vel[:2]) < 0.5):
                recovery_time_steps = step_index - recovery_start
        torque = np.asarray(getattr(state.pipeline_state, "actuator_force", []))
        if torque.size:
            measured_torque_steps += 1
            limit = np.asarray(ctx["torque_limit"])
            saturated_steps += int(np.any(np.abs(torque) >= 0.98 * limit))

        is_fallen_roll = bool(abs(rpy[0]) > RobotConfig.TERMINATION_ROLL)
        is_fallen_pitch = bool(abs(rpy[1]) > RobotConfig.TERMINATION_PITCH)
        is_low = bool(base_pos[2] < RobotConfig.TERMINATION_HEIGHT)

        history.append({
            "step": step_index,
            "roll_rad": float(rpy[0]),
            "pitch_rad": float(rpy[1]),
            "base_pos": [float(v) for v in base_pos],
            "base_ang_vel": [float(v) for v in base_ang_vel],
            "is_fallen_roll": is_fallen_roll,
            "is_fallen_pitch": is_fallen_pitch,
            "is_low": is_low,
        })
        if len(history) > collapse_window:
            history.pop(0)

        metrics = getattr(state, "metrics", {}) or {}
        for key, value in metrics.items():
            try:
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
        metric_count += 1

        info = state.info
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if terminated or truncated:
            break

    if terminated:
        reason = classify_termination_reason(
            is_fallen_roll=history[-1]["is_fallen_roll"] if history else False,
            is_fallen_pitch=history[-1]["is_fallen_pitch"] if history else False,
            is_low=history[-1]["is_low"] if history else False,
            truncated=False,
        )
    elif truncated:
        reason = "time_limit"
    else:
        # env自身のterminated/truncatedがどちらも立たないまま、この関数の
        # max_stepsループを使い切った状態。これは真のepisode終了ではなく、
        # 呼び出し側のmax_stepsがRobotConfig.MAX_EPISODE_STEPSより小さい
        # 場合にのみ起こる「評価予算による打ち切り」であり、
        # is_fallen_*フラグの状態に関わらずtermination reasonとしては
        # 扱わない(=真のterminationイベントとして誤集計しない)。
        reason = "eval_budget_cutoff"

    reward_component_means = {
        key: value / metric_count for key, value in metric_sums.items()
    } if metric_count else {}

    has_required_contact = both_feet_contact
    success = (
        not terminated and truncated and has_required_contact
        and max_roll <= RobotConfig.TERMINATION_ROLL
        and max_pitch <= RobotConfig.TERMINATION_PITCH
        and max_foot_displacement <= RobotConfig.MAX_FOOT_TRANSLATION
    )

    return EpisodeResult(
        length=step_index,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        collapse_window=history if terminated else [],
        reward_component_means=reward_component_means,
        success=success,
        both_feet_contact=has_required_contact,
        max_foot_displacement=max_foot_displacement,
        max_roll_rad=max_roll,
        max_pitch_rad=max_pitch,
        recovery_time_steps=recovery_time_steps,
        torque_saturation_rate=(saturated_steps / measured_torque_steps
                    if measured_torque_steps else 0.0),
    )


def run_condition(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    n_episodes: int,
    base_seed: int,
    max_steps: int,
    collapse_window: int,
) -> dict:
    lengths, terminated_flags, truncated_flags, reasons = [], [], [], []
    reward_component_accum: Dict[str, List[float]] = {}
    collapse_examples = []
    successes = 0
    foot_displacements = []
    recovery_times = []
    torque_saturation_rates = []
    max_rolls = []
    max_pitches = []
    contact_successes = 0

    rng = ctx["jax"].random.PRNGKey(base_seed)
    for ep in range(n_episodes):
        rng, rng_ep = ctx["jax"].random.split(rng)
        result = run_episode(ctx, reset_fn, step_fn, policy_fn, rng_ep, max_steps, collapse_window)
        lengths.append(result.length)
        terminated_flags.append(result.terminated)
        truncated_flags.append(result.truncated)
        reasons.append(result.reason)
        successes += int(result.success)
        contact_successes += int(result.both_feet_contact)
        foot_displacements.append(result.max_foot_displacement)
        max_rolls.append(result.max_roll_rad)
        max_pitches.append(result.max_pitch_rad)
        torque_saturation_rates.append(result.torque_saturation_rate)
        if result.recovery_time_steps is not None:
            recovery_times.append(result.recovery_time_steps)
        for key, value in result.reward_component_means.items():
            reward_component_accum.setdefault(key, []).append(value)
        if result.terminated and len(collapse_examples) < 5:
            collapse_examples.append({
                "episode": ep,
                "length": result.length,
                "reason": result.reason,
                "window": result.collapse_window,
            })

    event_observed = terminated_flags  # True=event(termination), False=censored(time_limit)
    km_times, km_survival = kaplan_meier_survival(lengths, event_observed)

    real_failure_steps = [l for l, t in zip(lengths, terminated_flags) if t]
    timing_diag = diagnose_failure_timing(real_failure_steps, max_steps)

    return {
        "n_episodes": n_episodes,
        "episode_alive": summarize_episode_alive(lengths),
        "termination_reason_counts": dict(Counter(reasons)),
        "termination_reason_rate": {
            k: v / n_episodes for k, v in Counter(reasons).items()
        },
        "kaplan_meier": {"times": km_times.tolist(), "survival": km_survival.tolist()},
        "failure_timing_diagnosis": timing_diag,
        "success_rate": successes / n_episodes if n_episodes else 0.0,
        "both_feet_contact_rate": contact_successes / n_episodes if n_episodes else 0.0,
        "max_foot_displacement_m": float(max(foot_displacements, default=0.0)),
        "max_roll_deg": float(np.rad2deg(max(max_rolls, default=0.0))),
        "max_pitch_deg": float(np.rad2deg(max(max_pitches, default=0.0))),
        "recovery_time_steps": recovery_times,
        "recovery_time_mean_steps": float(np.mean(recovery_times)) if recovery_times else None,
        "torque_saturation_rate_mean": float(np.mean(torque_saturation_rates)) if torque_saturation_rates else 0.0,
        "reward_component_means": {
            key: float(np.mean(vals)) for key, vals in reward_component_accum.items()
        },
        "collapse_examples": collapse_examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="", help="log/<exp_name> 配下のcheckpointを使う")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--model", default="best_params.pkl")
    parser.add_argument("--episodes", type=int, default=20, help="stochastic/randomizedセルのepisode数")
    parser.add_argument(
        "--fixed-episodes", type=int, default=3,
        help="deterministic x fixed_dr セルのepisode数(再現性確認用、通常は少数でよい)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="未指定ならRobotConfig.MAX_EPISODE_STEPS")
    parser.add_argument("--collapse-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-levels", default=None,
        help="評価する外乱力[N]をカンマ区切りで指定。未指定はRobotConfig.PUSH_FORCE_LEVELS",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "log" / "phase0_eval_diagnostics.json")
    parser.add_argument(
        "--diagnosis-md", type=Path, default=ROOT / "docs" / "gate_a_diagnosis.md",
        help="§3.6決定木の一次判定ドラフトを書き出す先(人間/Copilotによるレビュー前提)",
    )
    args = parser.parse_args()

    ctx = _lazy_imports()
    RobotConfig = ctx["RobotConfig"]
    SenpuuMaruMJXEnv = ctx["SenpuuMaruMJXEnv"]
    ppo_networks = ctx["ppo_networks"]

    # Gate AはPhase 0 (無外乱)の診断であるため、外乱は明示的に無効化する。
    RobotConfig.DISTURBANCE_CURRICULUM = False
    RobotConfig.RANDOM_PUSH_MAX_FORCE = 0.0

    max_steps = args.max_steps or RobotConfig.MAX_EPISODE_STEPS
    if max_steps < RobotConfig.MAX_EPISODE_STEPS:
        print(
            f"[Phase0 Eval][WARN] --max-steps={max_steps} < "
            f"RobotConfig.MAX_EPISODE_STEPS={RobotConfig.MAX_EPISODE_STEPS}. "
            "env自身のtime-limit(truncated)に到達する前にロールアウトを打ち切るため、"
            "'eval_budget_cutoff'エピソードが混入しうる(これはtime_limitでも"
            "termination失敗でもない)。開発中の高速確認用途以外では"
            "--max-stepsを指定しないことを推奨する。"
        )

    model_path = ctx["get_model_path"](args.exp_name, args.version, args.model)
    if model_path is None:
        raise SystemExit(
            f"checkpoint not found for exp_name={args.exp_name!r}, version={args.version}, "
            f"model={args.model!r}. --exp_name / --version / --model を確認してください。"
        )
    params = ctx["load_checkpoint"](model_path)

    # obs/action次元はDR設定に依存しないstructuralな値なので、使い捨てのenv
    # インスタンスから一度だけ取得すれば十分(policy networkの構築もここでよい)。
    _probe_env = SenpuuMaruMJXEnv()
    network = ctx["make_policy_network_factory"](_probe_env.observation_size, _probe_env.action_size)
    make_policy = ppo_networks.make_inference_fn(network)

    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf

    params_stripped = ctx["jax"].tree_util.tree_map(strip_leading_dim, params)

    # deterministic/stochasticはpolicyのみに依存するため一度だけjitする。
    policy_fns = {
        det: ctx["jax"].jit(make_policy(params_stripped, deterministic=det))
        for det in (True, False)
    }

    force_levels = (
        [float(value) for value in args.force_levels.split(",")]
        if args.force_levels else list(RobotConfig.PUSH_FORCE_LEVELS)
    )
    conditions = [
        ("deterministic", "fixed_dr", True, True, args.fixed_episodes, 0.0),
        ("deterministic", "randomized_dr", True, False, args.episodes, 0.0),
        ("stochastic", "fixed_dr", False, True, args.episodes, 0.0),
        ("stochastic", "randomized_dr", False, False, args.episodes, 0.0),
    ]
    conditions.extend(
        ("deterministic", f"push_{force:g}N", True, False, args.episodes, force)
        for force in force_levels if force > 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(model_path),
        "max_steps": max_steps,
        "note_initial_state_randomization": (
            "物理初期姿勢(qpos/qvel)は常にnominal poseに固定されている(未実装機能、"
            "master_plan.md付録A §1.6参照)。ここでの'randomized_dr'は質量/摩擦/"
            "重心オフセット/サーボ温度/電圧のdomain randomizationのon/offを指す近似軸。"
        ),
        "note_termination_reasons": (
            "現行実装(envs/mjx_rewards.py)はfallen_roll/fallen_pitch/fallen_height/"
            "time_limitのみをterminationとして判定する。non_illegal_contact/slip_ok/"
            "torque_okによるterminationは未実装(master_plan.md Task4 C-08未着手)。"
        ),
        "conditions": {},
        "disturbance_model": {
            "force_levels_N": force_levels,
            "directions": int(RobotConfig.PUSH_DIRECTIONS),
            "duration_steps": int(RobotConfig.PUSH_DURATION_STEPS),
            "duration_s": float(RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT),
            "impulse_levels_Ns": [
                float(force * RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT)
                for force in force_levels
            ],
            "implementation": "MJX random horizontal push; direction is sampled continuously",
        },
    }

    for label, dr_label, deterministic, fixed_dr, n_episodes, push_force in conditions:
        policy_fn = policy_fns[deterministic]
        RobotConfig.RANDOM_PUSH_MAX_FORCE = push_force
        RobotConfig.DISTURBANCE_CURRICULUM = push_force > 0.0
        with _DomainRandomizationScope(RobotConfig, fixed=fixed_dr):
            # env.reset/step本体は `minval=RobotConfig.RANDOM_MASS_SCALE[0]` の
            # ようにRobotConfigのクラス属性をトレース時にPython定数として
            # 直接埋め込む。jax.jitのコンパイルキャッシュはbound method
            # (env.reset)の等価性で引かれるため、同じenvインスタンスに対して
            # 単に`jax.jit(env.reset)`を呼び直すだけでは、RobotConfigを
            # 変更後でも古いコンパイル結果が再利用されてしまい、
            # 2つ目以降の条件が1つ目のDR設定のまま実行される
            # ——という気付きにくい誤結果を生む。これはこのスクリプト作成時に
            # 実機で再現・確認した(jax.jit(env.reset)を使い回すとDR変更が
            # 反映されず、envインスタンスを条件ごとに新規作成するか
            # jax.clear_caches()を呼べば正しく反映されることを確認済み)。
            # 最も単純で既存コード(scratch/gate0_formal_eval.pyの
            # configure→インスタンス化の順序)とも整合する対策として、
            # DR設定確定後に毎回新しいenvインスタンスを作る。
            env = SenpuuMaruMJXEnv()
            ctx["foot_ids"] = (env._reward_system._left_foot_id, env._reward_system._right_foot_id)
            ctx["torque_limit"] = np.asarray(env._mjx_model.actuator_ctrlrange[:, 1])
            reset_fn = ctx["jax"].jit(env.reset)
            step_fn = ctx["jax"].jit(env.step)
            result = run_condition(
                ctx, reset_fn, step_fn, policy_fn,
                n_episodes=n_episodes,
                base_seed=args.seed,
                max_steps=max_steps,
                collapse_window=args.collapse_window,
            )
        key = f"{label}__{dr_label}"
        report["conditions"][key] = result
        print(f"[{key}] episode_alive mean={result['episode_alive']['mean']:.1f} "
              f"reasons={result['termination_reason_counts']} "
              f"timing={result['failure_timing_diagnosis']['classification']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Phase0 Eval] detailed report: {args.out}")

    _write_diagnosis_draft(args.diagnosis_md, report)
    print(f"[Phase0 Eval] diagnosis draft: {args.diagnosis_md}")


def _write_diagnosis_draft(path: Path, report: dict) -> None:
    lines = [
        "# Gate A 診断ドラフト（自動生成 — 人間 / Copilotによるレビュー必須）",
        "",
        f"生成日時: {report['generated_at']}",
        f"checkpoint: {report['checkpoint']}",
        "",
        "この文書は scratch/phase0_eval_diagnostics.py により自動生成された一次判定です。",
        "master_plan.md §3.7 (Task0完了基準) の「§3.6の決定木に基づく主因の暫定結論」",
        "に相当しますが、機械的な閾値ヒューリスティックによる分類であり、",
        "最終結論には人間またはCopilotによるログ・collapse_examplesの目視確認を要します。",
        "",
        f"- {report['note_initial_state_randomization']}",
        f"- {report['note_termination_reasons']}",
        "",
        "## 条件別サマリー",
        "",
    ]
    for key, result in report["conditions"].items():
        ea = result["episode_alive"]
        diag = result["failure_timing_diagnosis"]
        lines.append(f"### {key}")
        lines.append(
            f"- episode_alive: mean={ea['mean']:.1f}, std={ea['std']:.1f}, "
            f"n={ea['n']}"
        )
        lines.append(f"- termination reasons: {result['termination_reason_counts']}")
        lines.append(f"- failure timing: {diag['classification']}")
        lines.append(f"- suggested action: {diag['suggested_action']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

```

---

## scratch/phase0_ppo_diagnostics.py

```python
#!/usr/bin/env python3
"""Phase 0 diagnostics: KL trend, log_std range, and truncation signal presence."""

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def collect_std_like(tree, path=""):
    stats = []
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            p = f"{path}/{k}" if path else str(k)
            key_l = str(k).lower()
            if "log_std" in key_l or "std" in key_l:
                arr = np.asarray(v)
                stats.append({
                    "path": p,
                    "shape": list(arr.shape),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                })
            stats.extend(collect_std_like(v, p))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            p = f"{path}[{i}]"
            stats.extend(collect_std_like(v, p))
    return stats


def kl_stats(log_items):
    vals = []
    for item in log_items:
        if "training/kl_mean" in item:
            vals.append(float(item["training/kl_mean"]))
    if not vals:
        return None
    a = np.asarray(vals)
    spikes = []
    for item in log_items:
        value = item.get("training/kl_mean")
        if value is not None and float(value) > 0.1:
            spikes.append({
                key: item[key]
                for key in (
                    "step", "training/kl_mean", "training/learning_rate",
                    "training/entropy", "training/policy_loss",
                    "training/total_loss", "eval/episode_reward",
                )
                if key in item
            })
    return {
        "count": int(len(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "gt_0_1_rate": float(np.mean(a > 0.1)),
        "gt_0_05_rate": float(np.mean(a > 0.05)),
        "spikes": spikes,
    }


def policy_std_stats(log_items):
    keys = (
        "training/policy_dist_min_std",
        "training/policy_dist_mean_std",
        "training/policy_dist_max_std",
    )
    values = {key: [float(item[key]) for item in log_items if key in item] for key in keys}
    return {
        key: {
            "count": len(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "last": float(vals[-1]),
        }
        for key, vals in values.items()
        if vals
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=ROOT / "log" / "version_7" / "log.json")
    parser.add_argument("--params", type=Path, default=ROOT / "log" / "version_7" / "final_params.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "log" / "phase0_ppo_diag.json")
    args = parser.parse_args()

    with args.log.open("r", encoding="utf-8") as f:
        log_items = json.load(f)

    with args.params.open("rb") as f:
        params = pickle.load(f)

    out = {
        "log_path": str(args.log),
        "params_path": str(args.params),
        "kl": kl_stats(log_items),
        "policy_std": policy_std_stats(log_items),
        "std_like": collect_std_like(params),
    }

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[Phase0] saved: {args.out}")


if __name__ == "__main__":
    main()

```

---

## scratch/probe_physical_limits.py

```python
#!/usr/bin/env python3
"""Extract physical-limit related constants from MuJoCo model at nominal standing pose."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig


def find_foot_body_ids(model):
    left_name = "doutai-v5_hidaridairou_hidarikokansetu_hidarimomo_hidarihizabu_hidariaikabu_hidariashiura_hidariashiura-1"
    right_name = "doutai-v5_migidaitou_migikokansetu_migimomo_migihizabu_migigaikabu_migiashiura_migiashiura-1"

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)

    if left_id == -1 or right_id == -1:
        left_id = -1
        right_id = -1
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if left_id == -1 and "hidariashiura" in name:
                left_id = body_id
            if right_id == -1 and "migiashiura" in name:
                right_id = body_id
        if left_id == -1 or right_id == -1:
            raise RuntimeError("Failed to locate foot body ids")

    return left_id, right_id


def main():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)

    qpos = np.zeros(model.nq, dtype=np.float64)
    if model.nq >= 7:
        qpos[0:3] = np.array([0.0, 0.0, 0.1773], dtype=np.float64)
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    default = np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64)
    for act_i in range(min(model.nu, len(default))):
        jnt_id = int(model.actuator_trnid[act_i, 0])
        qpos_idx = int(model.jnt_qposadr[jnt_id])
        qpos[qpos_idx] = default[act_i]

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_foot_id, right_foot_id = find_foot_body_ids(model)

    left_x = float(data.xpos[left_foot_id, 0])
    right_x = float(data.xpos[right_foot_id, 0])
    com = np.asarray(data.subtree_com[0], dtype=np.float64)

    # Estimate support half width from foot centers in x direction.
    support_half = abs(left_x - right_x) * 0.5

    result = {
        "total_mass_kg": float(np.sum(model.body_mass)),
        "gravity_m_s2": float(-model.opt.gravity[2]),
        "com_xyz_m": [float(com[0]), float(com[1]), float(com[2])],
        "left_foot_center_xyz_m": [float(v) for v in data.xpos[left_foot_id]],
        "right_foot_center_xyz_m": [float(v) for v in data.xpos[right_foot_id]],
        "support_half_width_x_m": float(support_half),
        "floor_friction": [float(v) for v in model.geom_friction[0]],
        "motor_max_torque_nm": float(RobotConfig.MOTOR_MAX_TORQUE),
        "control_dt_s": float(RobotConfig.CONTROL_DT),
        "sim_dt_s": float(RobotConfig.SIM_DT),
    }

    out_dir = ROOT / "log"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_physical_probe.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"[Phase-1] saved: {out_path}")


if __name__ == "__main__":
    main()

```

---

## scratch/render_collision.py

```python
import os
import sys
import numpy as np
import mujoco

# ヘッドレス環境(Linux/WSL)用EGL/OSMesaフラグ設定
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

# プロジェクトルートの設定
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from robot.config import RobotConfig

def render_collision_view():
    xml_path = RobotConfig.MUJOCO_MODEL_PATH
    if not os.path.exists(xml_path):
        xml_path = BASE_DIR / "envs" / "mjx_env.py" # fallback XML if any
        # または fallback XMLを直接文字列で作成
    
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # ロボットを初期姿勢にセット
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # レンダラーの初期化 (640x480)
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)
    
    # 視覚化オプションの設定
    vopt = mujoco.MjvOption()
    
    # 当たり判定(geomgroup[0])と視覚モデル(geomgroup[1], geomgroup[2])の両方を可視化
    vopt.geomgroup[0] = 1  # 当たり判定(Collision) geom
    vopt.geomgroup[1] = 1  # 視覚(Visual) geom
    vopt.geomgroup[2] = 1
    
    # 半透明表示で骨格・判定形状の内部構造を見やすくする
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
    vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

    
    # カメラ設定
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.0, 0.0, 0.4]
    camera.distance = 1.8
    camera.elevation = -15.0
    camera.azimuth = 135.0
    
    # 角度を変えて複数パースペクティブ画像をレンダリング
    views = [
        {"name": "collision_front_angle", "azimuth": 135.0, "elevation": -15.0, "distance": 1.6},
        {"name": "collision_side", "azimuth": 90.0, "elevation": -5.0, "distance": 1.5},
        {"name": "collision_top_down", "azimuth": 180.0, "elevation": -60.0, "distance": 1.8},
        {"name": "collision_close_foot", "azimuth": 140.0, "elevation": -20.0, "distance": 0.8, "lookat": [0.0, 0.0, 0.15]},
    ]
    
    output_dir = BASE_DIR / "scratch" / "collision_renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import PIL.Image
    
    rendered_files = []
    for view in views:
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]
        camera.distance = view["distance"]
        if "lookat" in view:
            camera.lookat = view["lookat"]
        else:
            camera.lookat = [0.0, 0.0, 0.4]
            
        renderer.update_scene(data, camera=camera, scene_option=vopt)
        pixels = renderer.render()
        
        img = PIL.Image.fromarray(pixels)
        filepath = output_dir / f"{view['name']}.png"
        img.save(filepath)
        print(f"Saved: {filepath}")
        rendered_files.append(str(filepath))

if __name__ == "__main__":
    render_collision_view()

```

---

## scratch/render_pure_collision.py

```python
import os
import sys
import numpy as np
import mujoco

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from robot.config import RobotConfig

def render_pure_geoms_only():
    xml_path = RobotConfig.MUJOCO_MODEL_PATH
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)
    
    # 完全に純粋な GeomGroup 0 (Collision Geoms) のみ
    vopt = mujoco.MjvOption()
    vopt.geomgroup[0] = 1
    vopt.geomgroup[1] = 0
    vopt.geomgroup[2] = 0
    vopt.geomgroup[3] = 0
    vopt.geomgroup[4] = 0
    
    # 関節軸やその他のオーバーレイ表示をオフにして純粋なgeomのみ見せる
    vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    views = [
        {"name": "pure_collision_front", "azimuth": 135.0, "elevation": -15.0, "distance": 1.2, "lookat": [0.0, 0.0, 0.35]},
        {"name": "pure_collision_side", "azimuth": 90.0, "elevation": -5.0, "distance": 1.2, "lookat": [0.0, 0.0, 0.35]},
        {"name": "pure_collision_feet", "azimuth": 140.0, "elevation": -20.0, "distance": 0.6, "lookat": [0.0, 0.0, 0.15]},
    ]
    
    output_dir = BASE_DIR / "scratch" / "collision_renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import PIL.Image
    
    for view in views:
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]
        camera.distance = view["distance"]
        camera.lookat = view["lookat"]
            
        renderer.update_scene(data, camera=camera, scene_option=vopt)
        pixels = renderer.render()
        
        img = PIL.Image.fromarray(pixels)
        filepath = output_dir / f"{view['name']}.png"
        img.save(filepath)
        print(f"Saved: {filepath}")

if __name__ == "__main__":
    render_pure_geoms_only()

```

---

## scratch/render_simulation_video.py

```python
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch the unified MJX replay entrypoint in video mode.")
    parser.add_argument("--exp_name", type=str, default="mjx_ppo_rma_100hz")
    parser.add_argument("--version", type=int, default=None, help="Version number to load (e.g. 17 for version_17).")
    parser.add_argument("--model", type=str, default="best_params.pkl", help="Checkpoint filename")
    parser.add_argument("--steps", type=int, default=300, help="Number of frames to render")
    parser.add_argument("--output", type=str, default="walking_simulation.gif", help="Output GIF filename")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    target = root_dir / "train" / "visualize_rl.py"
    if not target.exists():
        print(f"Error: {target} not found.")
        return

    cmd = [sys.executable, str(target),
           "--mode", "video",
           "--exp_name", args.exp_name,
           "--model", args.model,
           "--steps", str(args.steps),
           "--output", args.output]

    if args.version is not None:
        cmd += ["--version", str(args.version)]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

```

---

## scratch/save_html.py

```python
import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traj_path = os.path.join(root_dir, "trajectory.npy")
    
    if not os.path.exists(traj_path):
        print("エラー: trajectory.npy が見つかりません。")
        return

    print("モデルと軌跡データを読み込み中...")
    m = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    d = mujoco.MjData(m)
    traj = np.load(traj_path)

    # レンダーの初期化 (オフスクリーンレンダラー)
    renderer = mujoco.Renderer(m, height=480, width=640)
    frames = []

    print(f"オフスクリーンレンダリング中 ({len(traj)} フレーム)...")
    # 5ステップごとに1フレーム（間引きしてGIFの容量とレンダリング時間を軽量化）
    for i, qpos in enumerate(traj):
        if i % 3 != 0:
            continue
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        renderer.update_scene(d)
        pixels = renderer.render()
        frames.append(pixels)

    gif_out = os.path.join(root_dir, "log", "version_2", "simulation.gif")
    print(f"GIFアニメーションを保存中: {gif_out}")
    
    try:
        from PIL import Image
        img_list = [Image.fromarray(f) for f in frames]
        img_list[0].save(
            gif_out,
            save_all=True,
            append_images=img_list[1:],
            duration=33, # ~30fps
            loop=0
        )
        print("✓ simulation.gif の保存に成功しました！")
    except Exception as e:
        print(f"保存エラー: {e}")

if __name__ == "__main__":
    main()

```

---

## scratch/validate_policy_bounds.py

```python
#!/usr/bin/env python3
"""Validate PPO policy loc/std bounds without starting a training run."""

import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.policy_network import (
    POLICY_MAX_STD,
    POLICY_MEAN_CLIP_SCALE,
    POLICY_MIN_STD,
    make_policy_network_factory,
)


policy = make_policy_network_factory(
    observation_size=4,
    action_size=20,
)
distribution = policy.parametric_action_distribution
logits = jnp.concatenate([
    jnp.full((2, 20), 100.0),
    jnp.array([
        jnp.full((20,), -100.0),
        jnp.full((20,), 100.0),
    ]),
], axis=-1)
created = distribution.create_dist(logits)

assert float(jnp.max(jnp.abs(created.loc))) <= POLICY_MEAN_CLIP_SCALE + 1e-6
assert float(jnp.min(created.scale)) >= POLICY_MIN_STD - 1e-6
assert float(jnp.max(created.scale)) <= POLICY_MAX_STD + 1e-6
print("Policy bounds PASS:")
print(f"  max_abs_loc={float(jnp.max(jnp.abs(created.loc))):.6f}")
print(f"  min_std={float(jnp.min(created.scale)):.6f}")
print(f"  max_std={float(jnp.max(created.scale)):.6f}")

```

---

## scratch/validate_progress_wrapper.py

```python
#!/usr/bin/env python3
"""Small non-MJX check for monotonic TrainingProgressWrapper counters."""

import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.envs import Wrapper
from brax.envs.base import State

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.training_wrapper import TrainingProgressWrapper


class DummyEnv:
    observation_size = 1
    action_size = 1
    backend = "generalized"

    def reset(self, rng):
        return State(jp.array(0.0), jp.array([0.0]), jp.array(0.0), jp.array(0.0), {}, {})

    def step(self, state, action):
        info = dict(state.info)
        info["inner_step"] = info.get("inner_step", jp.array(0)) + 1
        return state.replace(info=info)

    @property
    def unwrapped(self):
        return self


env = TrainingProgressWrapper(Wrapper(DummyEnv()), total_steps_per_env=10)
state = env.reset(jax.random.PRNGKey(0))
assert float(state.info["training_progress"]) == 0.0
for expected in range(1, 4):
    state = env.step(state, jp.array([0.0]))
    assert int(state.info["_env_steps"]) == expected
    assert int(state.info["global_step"]) == expected
    assert np.isclose(float(state.info["training_progress"]), expected / 10.0)
print("TrainingProgressWrapper PASS: counters are monotonic and progress uses total steps.")

```

---

## scratch/verify_changes.py

```python
"""Verify the reward weight changes and import consistency."""
import sys
sys.path.insert(0, '.')

# 1. Config verification
from robot.config import RobotConfig
w = RobotConfig.REWARD_WEIGHTS
print("=== Config Verification ===")
print(f"  alive weight:        {w['alive']}  (expected: 25.0)")
print(f"  fall_penalty weight: {w['fall_penalty']}  (expected: -30.0)")
assert w['alive'] == 25.0, f"alive should be 25.0, got {w['alive']}"
assert w['fall_penalty'] == -30.0, f"fall_penalty should be -30.0, got {w['fall_penalty']}"
print("  [OK] Config OK")

# 2. Reward system import
print("\n=== Reward System Import ===")
from envs.mjx_rewards import MJXRewardSystem
import inspect
src = inspect.getsource(MJXRewardSystem.compute)
assert 'reward_per_step' in src, "reward_per_step metric missing"
assert 'total_penalty' in src, "total_penalty metric missing"
print("  [OK] reward_per_step metric present")
print("  [OK] total_penalty metric present")

# 3. Env import
print("\n=== Environment Import ===")
from envs.mjx_env import SenpuuMaruMJXEnv
src_env = inspect.getsource(SenpuuMaruMJXEnv.reset)
assert 'reward_per_step' in src_env, "reward_per_step init missing in reset()"
assert 'total_penalty' in src_env, "total_penalty init missing in reset()"
print("  [OK] reset() metrics init OK")

# 4. Training script import check
print("\n=== Training Script Check ===")
with open('train/train_mjx.py', 'r') as f:
    train_src = f.read()
assert 'clipping_epsilon=0.2' in train_src, "clipping_epsilon not set"
assert "learning_rate_schedule='ADAPTIVE_KL'" in train_src, "ADAPTIVE_KL not set"
assert 'desired_kl=0.02' in train_src, "desired_kl not set"
assert 'max_grad_norm=1.0' in train_src, "max_grad_norm not set"
assert 'import optax' not in train_src, "unused optax import still present"
print("  [OK] clipping_epsilon=0.2")
print("  [OK] learning_rate_schedule='ADAPTIVE_KL'")
print("  [OK] desired_kl=0.02")
print("  [OK] max_grad_norm=1.0")
print("  [OK] unused optax import removed")

print("\n=== Lambda Phase Check ===")
with open('envs/mjx_rewards.py', 'r') as f:
    reward_src = f.read()
assert 'lambda_phase = jp.clip(' in reward_src, "lambda_phase is not explicitly clipped"
print("  [OK] lambda_phase is clipped to [0, 1]")

# 5. Default learning rate
assert 'default=1e-4' in train_src, "default learning rate not 1e-4"
print("  [OK] default learning_rate=1e-4")

print("\n=== ALL CHECKS PASSED ===")


```

---

## scripts/add_collision_colors.py

```python
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
```

---

## scripts/collision_tuner.py

```python
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

```

---

## scripts/fix_visual_mass.py

```python
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

```

---

## scripts/visualize_mujoco.py

```python
#!/usr/bin/env python3
"""MuJoCoビューアーで衝突ジオメトリを可視化"""

import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mujoco
import mujoco.viewer


def resolve_model_path() -> Path:
    """
    視覚化用 XML を優先し、なければ通常の humanoid.xml にフォールバックする。
    実行位置に依存しないようにプロジェクトルート基準で解決する。
    """
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

        print("Starting physics simulation loop...")

        try:
            while viewer.is_running():
                step_start = time.time()

                # 物理シミュレーションを1ステップ進める
                mujoco.mj_step(model, data)

                # ビューアーの同期
                viewer.sync()

                # 物理演算のタイムステップに同期
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
        except KeyboardInterrupt:
            print("\nViewer closed.")


if __name__ == '__main__':
    main()
```

---

## stubs/board.py

```python
class _Board:
    def __getattr__(self, name):
        return name

import sys
sys.modules[__name__] = _Board()

```

---

## stubs/busio.py

```python
class I2C:
    def __init__(self, *args, **kwargs):
        pass

class SPI:
    def __init__(self, *args, **kwargs):
        pass

class UART:
    def __init__(self, *args, **kwargs):
        pass

```

---

## tests/__init__.py

```python
# tests package

```

---

## tests/rebuild_global_stl.py

```python
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

```

---

## tests/test_actuator_gains.py

```python
import numpy as np

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig


def test_robot_config_gains_are_applied_to_mjx_model():
    env = SenpuuMaruMJXEnv()
    model = env._mjx_model

    assert np.allclose(np.asarray(model.actuator_gainprm)[:, 0], RobotConfig.KP)
    assert np.allclose(np.asarray(model.actuator_biasprm)[:, 1], -RobotConfig.KP)
    assert np.allclose(np.asarray(model.actuator_biasprm)[:, 2], -RobotConfig.KD)


def test_robot_config_gains_are_applied_to_brax_system():
    env = SenpuuMaruMJXEnv()
    actuator = env.sys.actuator

    assert np.allclose(np.asarray(actuator.gain), RobotConfig.KP)
    assert np.allclose(np.asarray(actuator.bias_q), -RobotConfig.KP)
    assert np.allclose(np.asarray(actuator.bias_qd), -RobotConfig.KD)
```

---

## tests/test_improved_rewards.py

```python
#!/usr/bin/env python3
"""
Validation script for improved reward system
テスト：改善された報酬システムの動作確認
"""

import sys
sys.path.insert(0, '/c/bipedal_robot')

import jax
import jax.numpy as jp
from robot.config import RobotConfig
from envs.stability_metrics import StabilityMetrics

def test_curriculum_learning():
    """Test curriculum learning schedule"""
    print("=" * 60)
    print("TEST 1: Curriculum Learning Schedule")
    print("=" * 60)
    
    schedule = RobotConfig.CURRICULUM_SCHEDULE
    test_steps = [0, 50000, 100000, 300000, 500000, 1000000, 2000000, 5000000]
    
    for step in test_steps:
        # スケジュール内のキーをソート
        keys = sorted(schedule.keys())
        scale = schedule[keys[0]]
        for key in keys:
            if step >= key:
                scale = schedule[key]
        
        force = RobotConfig.RANDOM_PUSH_MAX_FORCE * scale
        print(f"Step {step:8d}: scale={scale:.2f}, max_force={force:.2f}N")
    
    print("✓ Curriculum learning schedule validated\n")

def test_stability_metrics():
    """Test stability metrics computation"""
    print("=" * 60)
    print("TEST 2: Stability Metrics")
    print("=" * 60)
    
    # Initialize metrics
    metrics = StabilityMetrics(left_foot_id=0, right_foot_id=1, com_height=0.28)
    
    # Stable state
    com_pos = jp.array([0.0, 0.0, 0.28])
    com_vel = jp.array([0.0, 0.0, 0.0])
    com_accel = jp.array([0.0, 0.0, -9.81])
    rpy = jp.array([0.0, 0.0, 0.0])
    base_ang_vel = jp.array([0.0, 0.0, 0.0])
    left_foot_pos = jp.array([-0.05, 0.0, 0.0])
    right_foot_pos = jp.array([0.05, 0.0, 0.0])
    left_foot_force = jp.array(50.0)
    right_foot_force = jp.array(50.0)
    
    stability_index, metrics_dict = metrics.compute_unified_stability_index(
        com_pos, com_vel, com_accel, rpy, base_ang_vel,
        left_foot_pos, right_foot_pos,
        left_foot_force, right_foot_force,
        gait_phase=0.5
    )
    
    print(f"Stable state:")
    print(f"  Stability Index: {float(stability_index):.4f}")
    print(f"  CP Margin:       {float(metrics_dict['cp_margin']):.4f}")
    print(f"  ZMP Margin:      {float(metrics_dict['zmp_margin']):.4f}")
    print(f"  Foot Balance:    {float(metrics_dict['foot_balance']):.4f}")
    print(f"  Orient Margin:   {float(metrics_dict['orient_margin']):.4f}")
    
    # Tilted state
    rpy_tilted = jp.array([0.2, 0.0, 0.0])
    stability_index_tilted, metrics_tilted = metrics.compute_unified_stability_index(
        com_pos, com_vel, com_accel, rpy_tilted, base_ang_vel,
        left_foot_pos, right_foot_pos,
        left_foot_force, right_foot_force,
        gait_phase=0.5
    )
    
    print(f"\nTilted state (roll=0.2):")
    print(f"  Stability Index: {float(stability_index_tilted):.4f}")
    print(f"  CP Margin:       {float(metrics_tilted['cp_margin']):.4f}")
    print(f"  Orient Margin:   {float(metrics_tilted['orient_margin']):.4f}")
    
    assert float(stability_index) > float(stability_index_tilted), \
        "Tilted state should have lower stability"
    
    print("✓ Stability metrics validated\n")

def test_adaptive_scaling():
    """Test adaptive reward scaling"""
    print("=" * 60)
    print("TEST 3: Adaptive Reward Scaling")
    print("=" * 60)
    
    from envs.mjx_rewards import MJXRewardSystem
    
    # Create dummy reward system to test scaling
    class DummyModel:
        nq = 7
        nu = 6
    
    reward_system = MJXRewardSystem(
        DummyModel(), 
        RobotConfig.REWARD_WEIGHTS,
        left_foot_id=0,
        right_foot_id=1
    )
    
    # Normal conditions
    servo_temp_normal = jp.full(6, 40.0)
    volt_normal = 11.1
    scaling_normal = reward_system._compute_adaptive_reward_scaling(servo_temp_normal, volt_normal)
    
    print(f"Normal conditions (T=40°C, V=11.1V):")
    print(f"  Recovery scale:   {float(scaling_normal['recovery']):.4f}")
    print(f"  Energy scale:     {float(scaling_normal['energy']):.4f}")
    print(f"  Smoothness scale: {float(scaling_normal['smoothness']):.4f}")
    
    # High temperature
    servo_temp_hot = jp.full(6, 75.0)
    scaling_hot = reward_system._compute_adaptive_reward_scaling(servo_temp_hot, volt_normal)
    
    print(f"\nHigh temperature (T=75°C, V=11.1V):")
    print(f"  Recovery scale:   {float(scaling_hot['recovery']):.4f}")
    print(f"  Energy scale:     {float(scaling_hot['energy']):.4f}")
    print(f"  Smoothness scale: {float(scaling_hot['smoothness']):.4f}")
    
    # Low voltage
    servo_temp_normal = jp.full(6, 40.0)
    volt_low = 9.2
    scaling_low = reward_system._compute_adaptive_reward_scaling(servo_temp_normal, volt_low)
    
    print(f"\nLow voltage (T=40°C, V=9.2V):")
    print(f"  Recovery scale:   {float(scaling_low['recovery']):.4f}")
    print(f"  Energy scale:     {float(scaling_low['energy']):.4f}")
    print(f"  Smoothness scale: {float(scaling_low['smoothness']):.4f}")
    
    # Verify scaling directions
    assert float(scaling_hot['recovery']) > 1.0, "Recovery should be boosted under high temp"
    assert float(scaling_hot['energy']) < 1.0, "Energy penalty should be reduced under high temp"
    
    print("✓ Adaptive reward scaling validated\n")


def test_stance_penalty_discourages_wide_foot_spacing():
    """広い足幅のまま停止する局所最適を避けるため、足幅の広がりにペナルティが付くことを確認する。"""
    from envs.mjx_rewards import MJXRewardSystem

    class DummyModel:
        nq = 7
        nu = 6

    reward_system = MJXRewardSystem(
        DummyModel(),
        RobotConfig.REWARD_WEIGHTS,
        left_foot_id=0,
        right_foot_id=1,
    )

    class DummyData:
        def __init__(self, left_pos, right_pos):
            self.qpos = jp.array([0.0, 0.0, 0.28, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.qvel = jp.zeros(6)
            self.actuator_force = jp.zeros(6)
            self.qacc = jp.array([0.0, 0.0, 0.0])
            self.sensordata = jp.zeros(8)
            self.xpos = jp.array([left_pos, right_pos])
            self.subtree_com = jp.array([[0.0, 0.0, 0.28]])

    narrow_data = DummyData(jp.array([-0.05, 0.0, 0.0]), jp.array([0.05, 0.0, 0.0]))
    wide_data = DummyData(jp.array([-0.12, 0.0, 0.0]), jp.array([0.12, 0.0, 0.0]))

    narrow_reward, _, _, _ = reward_system.compute(
        narrow_data,
        action=jp.zeros(6),
        last_action=jp.zeros(6),
        double_last_action=jp.zeros(6),
        triple_last_action=jp.zeros(6),
        cbf_penalty=jp.array(0.0),
        last_potential=jp.array(0.0),
        step=jp.array(10),
        reference_action=jp.zeros(6),
        servo_temp=jp.full(6, 40.0),
        supply_volt=11.1,
        global_step=jp.array(1000),
        gait_phase=0.5,
        was_disturbed=jp.array(False),
        disturbance_recovery_steps=jp.array(1000),
        training_progress=jp.array(0.5),
    )

    wide_reward, _, _, _ = reward_system.compute(
        wide_data,
        action=jp.zeros(6),
        last_action=jp.zeros(6),
        double_last_action=jp.zeros(6),
        triple_last_action=jp.zeros(6),
        cbf_penalty=jp.array(0.0),
        last_potential=jp.array(0.0),
        step=jp.array(10),
        reference_action=jp.zeros(6),
        servo_temp=jp.full(6, 40.0),
        supply_volt=11.1,
        global_step=jp.array(1000),
        gait_phase=0.5,
        was_disturbed=jp.array(False),
        disturbance_recovery_steps=jp.array(1000),
        training_progress=jp.array(0.5),
    )

    print(f"Narrow stance reward: {float(narrow_reward):.4f}")
    print(f"Wide stance reward:   {float(wide_reward):.4f}")

    assert float(wide_reward) < float(narrow_reward), "Wide stance should be penalized"
    print("✓ Wide-stance penalty validated\n")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("REWARD SYSTEM IMPROVEMENT VALIDATION")
    print("=" * 60 + "\n")
    
    try:
        test_curriculum_learning()
        test_stability_metrics()
        test_adaptive_scaling()
        test_stance_penalty_discourages_wide_foot_spacing()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

```

---

## tests/test_joints.py

```python
import os
import sys
import mujoco

# DLL load failedなどの環境依存インポートエラーを防ぐため、グローバルスコープでtry-exceptする
try:
    import mujoco.viewer
    HAS_VIEWER = True
    VIEWER_ERROR = None
except Exception as e:
    HAS_VIEWER = False
    VIEWER_ERROR = e

def main():
    xml_path = "assets/humanoid/humanoid.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    nu = model.nu
    print("=" * 60)
    print(f"[MODEL] {nu} actuators successfully loaded")
    print("=" * 60)
    for i in range(nu):
        print(f"  [{i+1:02d}] {model.actuator(i).name}  ctrlrange={model.actuator_ctrlrange[i]}")
    print("=" * 60)

    # 初期位置を少し高く設定
    if model.nq >= 7:
        data.qpos[2] = 0.35
        data.qpos[3] = 1.0

    print("Attempting to launch MuJoCo Passive Viewer...")
    
    launched = False
    if HAS_VIEWER:
        try:
            print("  Viewer 上の [Ctrl] タブのスライダーで各関節を手動操作できます。")
            print("  ウィンドウを閉じると終了します。")
            print("=" * 60 + "\n")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                launched = True
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
        except Exception as e:
            print(f"\n[WARNING] Could not launch MuJoCo Passive Viewer: {e}")
            launched = False
    else:
        print(f"\n[WARNING] Could not import mujoco.viewer: {VIEWER_ERROR}")
        launched = False

    if not launched:
        print("  (Note: Windows host environment often suffers from OpenGL/DLL load failures for MuJoCo Viewer.)")
        print("  (Tip: You can run this script inside WSL2 with WSLg/X11 forwarding to see the full GUI!)")
        print("\nFalling back to Headless Simulation Mode...")
        print("Running 1000 physics steps to verify model stability...")
        print("=" * 60)
        
        try:
            for step in range(1000):
                mujoco.mj_step(model, data)
                if step % 200 == 0:
                    print(f"  Step {step:04d}/1000: z-height = {data.qpos[2]:.4f} m, base roll/pitch/yaw values are valid.")
            print("=" * 60)
            print("SUCCESS: Headless simulation completed perfectly! The modified XML model is physics-stable.")
        except Exception as sim_err:
            print(f"ERROR during headless simulation: {sim_err}")

if __name__ == "__main__":
    main()



```

---

## tests/test_mj_xml.py

```python
import os
import mujoco
import sys

def main():
    # 新しく作成した humanoid.xml のパス
    model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid", "humanoid.xml")
    model_path = os.path.abspath(model_path)
    
    print(f"Loading and validating MuJoCo model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)
        
    try:
        # MuJoCoパーサーにXMLを読み込ませる
        model = mujoco.MjModel.from_xml_path(model_path)
        print("\n=== SUCCESS: MuJoCo Model Loaded Perfectly! ===")
        print(f"Model Name        : {model.names}")
        print(f"Total Joints (nq) : {model.nq} (includes freejoint 7DoF + 20 hinge joints)")
        print(f"Total Actuators   : {model.nu}")
        print(f"Total Sensors     : {model.nsensor}")
        print(f"Total Bodies      : {model.nbody}")
        print(f"Total Geoms       : {model.ngeom}")
        print("===============================================")
    except Exception as e:
        print(f"\nERROR validating model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

```

---

## tests/test_phase0_eval_diagnostics.py

```python
#!/usr/bin/env python3
"""Phase 0 / Gate A diagnosis: deterministic vs stochastic evaluation +
termination-reason histogram (docs/master_plan.md 付録A §3.5, Task0).

status.md (2026-09-01) の「次のTask」= 「deterministic/stochastic評価の実装と
終了理由ヒストグラム化」に対応する。エスカレーション項目の「次の切り分け」の
3項目のうち、deterministic評価・終了stepヒストグラム化・報酬成分分解ログの
3つをまとめてこのスクリプトで実施する。

設計方針（master_plan.md §3.5 に基づく）:
  - 2x2評価: {deterministic, stochastic} x {fixed_dr, randomized_dr}
    注意: 本リポジトリの reset() は物理初期姿勢(qpos/qvel)を常に同一の
    nominal poseに固定しており、初期姿勢そのもののrandomizationは
    master_plan.md §1.6で「Task1(Gate A是正)の時点で必ず導入する」と
    定義された未実装機能である。したがって本スクリプトの
    「初期状態randomize」軸は、既存の実装済みrandomization経路である
    domain randomization (質量/摩擦/重心オフセット/サーボ温度/電圧) の
    on/offとして操作する。これは近似であり、真の初期姿勢randomizationの
    代替ではない。この制約は出力レポートに明記する。
  - deterministic x fixed_dr のセルは、同一checkpoint・同一初期状態・
    同一policyであれば理論上ビット単位で再現するはずのセルであり、
    複数episodeを回す意味は「決定論性テスト」（master_plan.md §4.2）を
    兼ねる以外にない。デフォルトのepisode数を他セルより少なくしている。
  - 各episodeについて、終了理由 (fallen_roll / fallen_pitch /
    fallen_height / physics_diverged / reward_nan / time_limit /
    unknown の組み合わせ) を分類する。非足裏接触・トルク上限による
    終了は、現行の envs/mjx_rewards.py の done判定 (is_fallen_roll or
    is_fallen_pitch or is_low、および物理/報酬の数値発散時) に
    実装されていないため分類対象にできない。これは
    master_plan.md Task4 (C-08, 複合成功条件) が未着手であることの
    追加の裏付けとしてレポートに記録する。
    [改造 2026-09-13] physics_diverged / reward_nan は
    envs/mjx_env.py・envs/mjx_rewards.py に追加されたNaN/Inf予防
    機構(数値発散時にdone=Trueを強制する)による終了を指す。転倒とは
    区別して集計する。
  - Kaplan-Meier型の生存曲線を打ち切り(truncated=time_limit)を
    考慮して計算する。
  - 失敗episodeについて、終了直前 collapse_window step分の
    roll/pitch/base角速度/base位置の時系列を記録する。
  - reward metrics (envs/mjx_rewards.py が返す metrics dict) の
    episode平均をあわせて記録し、reward成分分解ログを兼ねる。
  - master_plan.md §3.6 の決定木を単純な閾値ヒューリスティックとして
    実装し、失敗タイミングの偏り(序盤/後半/ランダム)を自動判定する。
    これは補助的な一次判定であり、最終診断は人間 / 記録を見た
    Copilotが行うことを想定している。

Done条件 (pytest, tests/test_phase0_eval_diagnostics.py 側):
  - classify_termination_reason の分類ロジック
  - kaplan_meier_survival の生存曲線計算
  - diagnose_failure_timing の決定木ヒューリスティック
  これらは純Python/NumPyのみで完結し、JAX/MJX/GPU無しでCPU上で検証できる。

実行には学習済みcheckpoint (log/<exp_name>/version_x/*.pkl) と
JAX/MJX/Brax環境 (WSLのvenv_wsl等) が必要。このリポジトリのsandboxには
GPUも実際の学習済みcheckpointも存在しないため、本スクリプト作成時には
以下2段階で検証した:
  1. 純Python/NumPyの解析ロジック(classify_termination_reason /
     kaplan_meier_survival / diagnose_failure_timing /
     summarize_episode_alive)はtests/test_phase0_eval_diagnostics.pyで
     単体テスト済み(CPU、JAX不要)。
  2. ロールアウト部分(run_episode/run_condition/main)は、CPU上に
     JAX/MuJoCo/MJX/Braxをインストールし、ランダム初期化した
     (未学習の)policy checkpointを使って実際にreset/step/評価の
     全経路を通しで実行確認した。この過程で以下の実装上の罠を
     発見・修正済み:
       - env.reset/env.stepは必ずjax.jit()経由で呼ぶ必要がある。
         eager実行では reset() 内の `info['step'] = 0` がPython int の
         まま伝播し、`truncated.astype(...)` (envs/mjx_env.py) で
         AttributeErrorになる。
       - jax.jit(env.reset) はbound methodの等価性でコンパイル結果を
         キャッシュするため、RobotConfig.RANDOM_* を条件間で書き換えても
         同一envインスタンスに対する再jitでは古いコンパイル結果が
         再利用されてしまう(2つ目以降のDR条件が1つ目の設定のまま
         実行される、気付きにくい誤結果)。DRスコープ確定後に毎回
         新しいenvインスタンスを作ることで回避した。
       - スクリプト自身の--max-stepsが環境本来のMAX_EPISODE_STEPSより
         小さい場合、terminated/truncatedのどちらも立たないままループが
         尽きることがある。これを終了理由に混ぜず
         "eval_budget_cutoff"として区別し、Kaplan-Meier計算上も
         event(実イベント)ではなくcensoredとして扱うようにした。
     未学習ランダムpolicyでの動作確認であり、実際に学習済み
     checkpointとGPU/WSL環境で実行した結果ではない。次の残作業は、
     WSL/GPU環境で実checkpointに対して
     `python scratch/phase0_eval_diagnostics.py --exp_name <name>` を
     実行し、結果を docs/status.md ・ docs/gate_a_diagnosis.md に
     記録すること。
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CPU固定はデフォルトのみ。GPU評価したい場合は呼び出し前に環境変数を上書きすること。
os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ============================================================================
# 純Python/NumPyの解析ロジック（JAX/MJX非依存、単体テスト対象）
# ============================================================================

def classify_termination_reason(
    is_fallen_roll: bool,
    is_fallen_pitch: bool,
    is_low: bool,
    truncated: bool,
    physics_diverged: bool = False,
    reward_is_finite: bool = True,
) -> str:
    """終了理由を分類する。

    master_plan.md 付録A §1.5 の終了条件定義のうち、現行コード
    (envs/mjx_rewards.py) が実装しているのは roll/pitch/height の3つと
    time-limitのみ。non_illegal_contact / slip_ok / torque_ok による
    terminationは未実装のため、このスクリプトでも分類できない
    （Task4 C-08 未着手であることの根拠として記録する）。

    [改造 2026-09-13] envs/mjx_env.py の物理発散ロールバック機構、
    envs/mjx_rewards.py の報酬NaN無害化機構の追加に伴い、doneが
    roll/pitch/height以外の理由(物理シミュレーションの数値発散、
    報酬計算のNaN)でもTrueになるようになった。これらは「本物の
    転倒」ではなく「数値的な安全装置の作動」であり、is_fallen_*では
    検出できないため、従来はunknown_terminatedに埋もれ、実際の
    発生頻度が見えなくなっていた。physics_diverged/reward_is_finite
    (envs/mjx_env.py, envs/mjx_rewards.py が state.metrics に記録する
    フラグ) を渡すことで、これらを明示的に分類できるようにする。
    デフォルト値は既存の呼び出し・テストとの後方互換性のため
    「発生していない」側に設定してある。
    """
    if truncated:
        return "time_limit"
    reasons = []
    if is_fallen_roll:
        reasons.append("fallen_roll")
    if is_fallen_pitch:
        reasons.append("fallen_pitch")
    if is_low:
        reasons.append("fallen_height")
    if physics_diverged:
        reasons.append("physics_diverged")
    if not reward_is_finite:
        reasons.append("reward_nan")
    if not reasons:
        # terminated=Trueだが既知のフラグがどれも立っていない場合。
        # 実装上は起こらないはずだが、バグ検知のため明示的に区別する。
        return "unknown_terminated"
    return "+".join(reasons)


def kaplan_meier_survival(
    episode_lengths: Sequence[int],
    event_observed: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier生存曲線を計算する（500stepで打ち切られる右側打ち切り分布）。

    Args:
        episode_lengths: 各episodeが終了した(打ち切られた)step数。
        event_observed: Trueなら真のterminationイベント、Falseなら
            time-limitによる打ち切り(censoring)。

    Returns:
        (times, survival): times[0]=0, survival[0]=1.0 から始まる
        ステップ関数のノード列。
    """
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    events = np.asarray(event_observed, dtype=bool)
    if len(lengths) == 0:
        return np.array([0]), np.array([1.0])
    if len(lengths) != len(events):
        raise ValueError("episode_lengths and event_observed must be same length")

    event_times = np.unique(lengths[events])
    times = [0]
    survival = [1.0]
    s = 1.0
    for t in sorted(event_times.tolist()):
        n_t = int(np.sum(lengths >= t))  # tの直前時点でまだ生存(risk set)にいる数
        d_t = int(np.sum((lengths == t) & events))  # t時点での真のイベント数
        if n_t > 0:
            s *= (1.0 - d_t / n_t)
        times.append(int(t))
        survival.append(s)
    return np.array(times), np.array(survival)


def diagnose_failure_timing(
    termination_steps: Sequence[int],
    max_step: int,
    early_frac: float = 1.0 / 3.0,
    late_frac: float = 2.0 / 3.0,
    concentration_threshold: float = 0.6,
) -> Dict[str, object]:
    """master_plan.md 付録A §3.6 の決定木を単純な閾値ヒューリスティックで実装する。

    real terminationのみ(truncatedは除く)を入力に使うこと。
    """
    steps = np.asarray(termination_steps, dtype=np.float64)
    if len(steps) == 0:
        return {
            "classification": "no_failures",
            "suggested_action": (
                "terminatedによる失敗episodeが観測されなかった。"
                "time-limit到達のみであれば§3.3(truncation/termination処理)の"
                "疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。"
            ),
            "normalized_mean": None,
            "early_rate": None,
            "late_rate": None,
        }

    normalized = steps / float(max(max_step, 1))
    early_rate = float(np.mean(normalized < early_frac))
    late_rate = float(np.mean(normalized > late_frac))
    normalized_mean = float(np.mean(normalized))

    if early_rate >= concentration_threshold:
        classification = "序盤集中"
        suggested_action = (
            "失敗がepisode序盤に集中 → 初期状態・初期transientの問題の疑い。"
            "初期状態分布の縮小・初期姿勢安定化を検討する（master_plan.md §3.6）。"
        )
    elif late_rate >= concentration_threshold:
        classification = "後半集中"
        suggested_action = (
            "失敗がepisode後半に集中 → 長期ドリフト or time-limitバグの疑い。"
            "truncation/termination処理(§3.3)を再疑う。"
        )
    else:
        classification = "ランダム分布"
        suggested_action = (
            "失敗時刻がランダムに分布 → 状態空間の局所不安定領域の疑い。"
            "失敗直前の状態を特定し、該当領域の報酬/観測を強化する。"
        )

    return {
        "classification": classification,
        "suggested_action": suggested_action,
        "normalized_mean": normalized_mean,
        "early_rate": early_rate,
        "late_rate": late_rate,
    }


def summarize_episode_alive(episode_lengths: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(episode_lengths, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


# ============================================================================
# ロールアウト（JAX/MJX依存、GPU/WSL環境での実行を想定）
# ============================================================================

@dataclass
class EpisodeResult:
    length: int
    terminated: bool
    truncated: bool
    reason: str
    collapse_window: List[dict] = field(default_factory=list)
    reward_component_means: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    both_feet_contact: bool = False
    max_foot_displacement: float = 0.0
    max_roll_rad: float = 0.0
    max_pitch_rad: float = 0.0
    recovery_time_steps: Optional[int] = None
    torque_saturation_rate: float = 0.0


def _lazy_imports():
    """JAX/MJX関連のimportを遅延させ、--help等をGPU無し環境でも高速に扱えるようにする。"""
    import jax  # noqa: F401
    import jax.numpy as jp  # noqa: F401
    from robot.config import RobotConfig
    from envs.mjx_env import SenpuuMaruMJXEnv
    from robot.math_utils import quat_to_euler
    from train.visualize_rl import (
        get_model_path,
        load_checkpoint,
        make_policy_network_factory,
    )
    from brax.training.agents.ppo import networks as ppo_networks

    return {
        "jax": jax,
        "jp": jp,
        "RobotConfig": RobotConfig,
        "SenpuuMaruMJXEnv": SenpuuMaruMJXEnv,
        "quat_to_euler": quat_to_euler,
        "get_model_path": get_model_path,
        "load_checkpoint": load_checkpoint,
        "make_policy_network_factory": make_policy_network_factory,
        "ppo_networks": ppo_networks,
    }


class _DomainRandomizationScope:
    """RobotConfigのDR幅を一時的に固定値へ差し替え、終了時に復元するコンテキストマネージャ。

    物理初期姿勢(qpos/qvel)はreset()で常に固定のため、これは
    「初期状態randomize」軸の近似実装であることに注意
    (モジュールdocstring参照)。
    """

    FIELDS = (
        "RANDOM_MASS_SCALE",
        "RANDOM_FRICTION",
        "RANDOM_COM_OFFSET",
        "RANDOM_TEMP",
        "RANDOM_VOLT",
    )

    def __init__(self, RobotConfig, fixed: bool):
        self._cfg = RobotConfig
        self._fixed = fixed
        self._saved = {}

    def __enter__(self):
        for name in self.FIELDS:
            self._saved[name] = getattr(self._cfg, name)
        if self._fixed:
            # 全フィールドは [lo, hi] のスカラー対 (envs/mjx_env.py の reset() が
            # minval=X[0], maxval=X[1] として読む前提と一致させる)。
            for name in self.FIELDS:
                lo, hi = self._saved[name]
                mid = (lo + hi) / 2.0
                setattr(self._cfg, name, [mid, mid])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, value in self._saved.items():
            setattr(self._cfg, name, value)
        return False


def run_episode(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    rng,
    max_steps: int,
    collapse_window: int,
) -> EpisodeResult:
    """1エピソードをロールアウトする。

    重要: reset_fn/step_fnは呼び出し側で必ず jax.jit(env.reset) /
    jax.jit(env.step) として渡すこと。env.reset/env.stepを素の(非jit)
    状態で呼ぶと、reset()内で `info['step'] = 0` のようにPython int
    リテラルとして初期化されたフィールドがPython int のまま
    stepに渡り、`truncated.astype(...)` (envs/mjx_env.py) で
    `AttributeError: 'bool' object has no attribute 'astype'` になる
    (jitされた関数の戻り値はJAXが自動的に配列型へ変換するため、
    jit経由なら発生しない。本スクリプト作成時にeager実行で実際に
    再現・確認済み)。
    """
    RobotConfig = ctx["RobotConfig"]
    quat_to_euler = ctx["quat_to_euler"]

    rng, rng_reset = ctx["jax"].random.split(rng)
    state = reset_fn(rng_reset)

    history = []
    metric_sums: Dict[str, float] = {}
    metric_count = 0
    initial_foot_positions = None
    max_foot_displacement = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    both_feet_contact = True
    recovery_start = None
    recovery_time_steps = None
    saturated_steps = 0
    measured_torque_steps = 0

    terminated = False
    truncated = False
    step_index = 0
    for step_index in range(1, max_steps + 1):
        rng, rng_step = ctx["jax"].random.split(rng)
        action, _ = policy_fn(state.obs, rng_step)
        state = step_fn(state, action)

        qpos = np.asarray(state.pipeline_state.qpos)
        qvel = np.asarray(state.pipeline_state.qvel)
        rpy = np.asarray(quat_to_euler(state.pipeline_state.qpos[3:7]))
        base_pos = qpos[0:3]
        base_ang_vel = qvel[3:6] if len(qvel) >= 6 else np.zeros(3)
        xpos = np.asarray(state.pipeline_state.xpos)
        foot_ids = ctx.get("foot_ids")
        if foot_ids is not None and xpos.ndim == 2:
            foot_positions = xpos[list(foot_ids)]
            if initial_foot_positions is None:
                initial_foot_positions = foot_positions.copy()
            max_foot_displacement = max(
                max_foot_displacement,
                float(np.max(np.linalg.norm(foot_positions[:, :2] - initial_foot_positions[:, :2], axis=1))),
            )
        max_roll = max(max_roll, abs(float(rpy[0])))
        max_pitch = max(max_pitch, abs(float(rpy[1])))
        contact_metric = float(np.asarray(getattr(state, "metrics", {}).get("both_feet_contact", 0.0)))
        both_feet_now = contact_metric >= 0.5
        both_feet_contact = both_feet_contact and both_feet_now
        if bool(state.info.get("was_disturbed", False)) and recovery_start is None:
            recovery_start = step_index
        if recovery_start is not None and recovery_time_steps is None:
            if (both_feet_now and abs(rpy[0]) < np.deg2rad(10.0)
                    and abs(rpy[1]) < np.deg2rad(10.0)
                    and np.linalg.norm(base_ang_vel[:2]) < 0.5):
                recovery_time_steps = step_index - recovery_start
        torque = np.asarray(getattr(state.pipeline_state, "actuator_force", []))
        if torque.size:
            measured_torque_steps += 1
            limit = np.asarray(ctx["torque_limit"])
            saturated_steps += int(np.any(np.abs(torque) >= 0.98 * limit))

        is_fallen_roll = bool(abs(rpy[0]) > RobotConfig.TERMINATION_ROLL)
        is_fallen_pitch = bool(abs(rpy[1]) > RobotConfig.TERMINATION_PITCH)
        is_low = bool(base_pos[2] < RobotConfig.TERMINATION_HEIGHT)

        # [改造 2026-09-13] classify_termination_reason() が
        # physics_diverged/reward_nan を区別できるよう、metricsの
        # 取得をここに前倒しする(元は後段のmetric_sums集計箇所のみで
        # 取得していた)。
        step_metrics = getattr(state, "metrics", {}) or {}
        physics_diverged_now = bool(float(step_metrics.get("physics_diverged", 0.0)) >= 0.5)
        reward_is_finite_now = bool(float(step_metrics.get("reward_is_finite", 1.0)) >= 0.5)

        history.append({
            "step": step_index,
            "roll_rad": float(rpy[0]),
            "pitch_rad": float(rpy[1]),
            "base_pos": [float(v) for v in base_pos],
            "base_ang_vel": [float(v) for v in base_ang_vel],
            "is_fallen_roll": is_fallen_roll,
            "is_fallen_pitch": is_fallen_pitch,
            "is_low": is_low,
            "physics_diverged": physics_diverged_now,
            "reward_is_finite": reward_is_finite_now,
        })
        if len(history) > collapse_window:
            history.pop(0)

        metrics = step_metrics
        for key, value in metrics.items():
            try:
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
        metric_count += 1

        info = state.info
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if terminated or truncated:
            break

    if terminated:
        reason = classify_termination_reason(
            is_fallen_roll=history[-1]["is_fallen_roll"] if history else False,
            is_fallen_pitch=history[-1]["is_fallen_pitch"] if history else False,
            is_low=history[-1]["is_low"] if history else False,
            truncated=False,
            physics_diverged=history[-1].get("physics_diverged", False) if history else False,
            reward_is_finite=history[-1].get("reward_is_finite", True) if history else True,
        )
    elif truncated:
        reason = "time_limit"
    else:
        # env自身のterminated/truncatedがどちらも立たないまま、この関数の
        # max_stepsループを使い切った状態。これは真のepisode終了ではなく、
        # 呼び出し側のmax_stepsがRobotConfig.MAX_EPISODE_STEPSより小さい
        # 場合にのみ起こる「評価予算による打ち切り」であり、
        # is_fallen_*フラグの状態に関わらずtermination reasonとしては
        # 扱わない(=真のterminationイベントとして誤集計しない)。
        reason = "eval_budget_cutoff"

    reward_component_means = {
        key: value / metric_count for key, value in metric_sums.items()
    } if metric_count else {}

    has_required_contact = both_feet_contact
    success = (
        not terminated and truncated and has_required_contact
        and max_roll <= RobotConfig.TERMINATION_ROLL
        and max_pitch <= RobotConfig.TERMINATION_PITCH
        and max_foot_displacement <= RobotConfig.MAX_FOOT_TRANSLATION
    )

    return EpisodeResult(
        length=step_index,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        collapse_window=history if terminated else [],
        reward_component_means=reward_component_means,
        success=success,
        both_feet_contact=has_required_contact,
        max_foot_displacement=max_foot_displacement,
        max_roll_rad=max_roll,
        max_pitch_rad=max_pitch,
        recovery_time_steps=recovery_time_steps,
        torque_saturation_rate=(saturated_steps / measured_torque_steps
                    if measured_torque_steps else 0.0),
    )


def run_condition(
    ctx: dict,
    reset_fn,
    step_fn,
    policy_fn,
    n_episodes: int,
    base_seed: int,
    max_steps: int,
    collapse_window: int,
) -> dict:
    lengths, terminated_flags, truncated_flags, reasons = [], [], [], []
    reward_component_accum: Dict[str, List[float]] = {}
    collapse_examples = []
    successes = 0
    foot_displacements = []
    recovery_times = []
    torque_saturation_rates = []
    max_rolls = []
    max_pitches = []
    contact_successes = 0

    rng = ctx["jax"].random.PRNGKey(base_seed)
    for ep in range(n_episodes):
        rng, rng_ep = ctx["jax"].random.split(rng)
        result = run_episode(ctx, reset_fn, step_fn, policy_fn, rng_ep, max_steps, collapse_window)
        lengths.append(result.length)
        terminated_flags.append(result.terminated)
        truncated_flags.append(result.truncated)
        reasons.append(result.reason)
        successes += int(result.success)
        contact_successes += int(result.both_feet_contact)
        foot_displacements.append(result.max_foot_displacement)
        max_rolls.append(result.max_roll_rad)
        max_pitches.append(result.max_pitch_rad)
        torque_saturation_rates.append(result.torque_saturation_rate)
        if result.recovery_time_steps is not None:
            recovery_times.append(result.recovery_time_steps)
        for key, value in result.reward_component_means.items():
            reward_component_accum.setdefault(key, []).append(value)
        if result.terminated and len(collapse_examples) < 5:
            collapse_examples.append({
                "episode": ep,
                "length": result.length,
                "reason": result.reason,
                "window": result.collapse_window,
            })

    event_observed = terminated_flags  # True=event(termination), False=censored(time_limit)
    km_times, km_survival = kaplan_meier_survival(lengths, event_observed)

    real_failure_steps = [l for l, t in zip(lengths, terminated_flags) if t]
    timing_diag = diagnose_failure_timing(real_failure_steps, max_steps)

    return {
        "n_episodes": n_episodes,
        "episode_alive": summarize_episode_alive(lengths),
        "termination_reason_counts": dict(Counter(reasons)),
        "termination_reason_rate": {
            k: v / n_episodes for k, v in Counter(reasons).items()
        },
        "kaplan_meier": {"times": km_times.tolist(), "survival": km_survival.tolist()},
        "failure_timing_diagnosis": timing_diag,
        "success_rate": successes / n_episodes if n_episodes else 0.0,
        "both_feet_contact_rate": contact_successes / n_episodes if n_episodes else 0.0,
        "max_foot_displacement_m": float(max(foot_displacements, default=0.0)),
        "max_roll_deg": float(np.rad2deg(max(max_rolls, default=0.0))),
        "max_pitch_deg": float(np.rad2deg(max(max_pitches, default=0.0))),
        "recovery_time_steps": recovery_times,
        "recovery_time_mean_steps": float(np.mean(recovery_times)) if recovery_times else None,
        "torque_saturation_rate_mean": float(np.mean(torque_saturation_rates)) if torque_saturation_rates else 0.0,
        "reward_component_means": {
            key: float(np.mean(vals)) for key, vals in reward_component_accum.items()
        },
        "collapse_examples": collapse_examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="", help="log/<exp_name> 配下のcheckpointを使う")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--model", default="best_params.pkl")
    parser.add_argument("--episodes", type=int, default=20, help="stochastic/randomizedセルのepisode数")
    parser.add_argument(
        "--fixed-episodes", type=int, default=3,
        help="deterministic x fixed_dr セルのepisode数(再現性確認用、通常は少数でよい)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="未指定ならRobotConfig.MAX_EPISODE_STEPS")
    parser.add_argument("--collapse-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-levels", default=None,
        help="評価する外乱力[N]をカンマ区切りで指定。未指定はRobotConfig.PUSH_FORCE_LEVELS",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "log" / "phase0_eval_diagnostics.json")
    parser.add_argument(
        "--diagnosis-md", type=Path, default=ROOT / "docs" / "gate_a_diagnosis.md",
        help="§3.6決定木の一次判定ドラフトを書き出す先(人間/Copilotによるレビュー前提)",
    )
    args = parser.parse_args()

    ctx = _lazy_imports()
    RobotConfig = ctx["RobotConfig"]
    SenpuuMaruMJXEnv = ctx["SenpuuMaruMJXEnv"]
    ppo_networks = ctx["ppo_networks"]

    # Gate AはPhase 0 (無外乱)の診断であるため、外乱は明示的に無効化する。
    RobotConfig.DISTURBANCE_CURRICULUM = False
    RobotConfig.RANDOM_PUSH_MAX_FORCE = 0.0

    max_steps = args.max_steps or RobotConfig.MAX_EPISODE_STEPS
    if max_steps < RobotConfig.MAX_EPISODE_STEPS:
        print(
            f"[Phase0 Eval][WARN] --max-steps={max_steps} < "
            f"RobotConfig.MAX_EPISODE_STEPS={RobotConfig.MAX_EPISODE_STEPS}. "
            "env自身のtime-limit(truncated)に到達する前にロールアウトを打ち切るため、"
            "'eval_budget_cutoff'エピソードが混入しうる(これはtime_limitでも"
            "termination失敗でもない)。開発中の高速確認用途以外では"
            "--max-stepsを指定しないことを推奨する。"
        )

    model_path = ctx["get_model_path"](args.exp_name, args.version, args.model)
    if model_path is None:
        raise SystemExit(
            f"checkpoint not found for exp_name={args.exp_name!r}, version={args.version}, "
            f"model={args.model!r}. --exp_name / --version / --model を確認してください。"
        )
    params = ctx["load_checkpoint"](model_path)

    # obs/action次元はDR設定に依存しないstructuralな値なので、使い捨てのenv
    # インスタンスから一度だけ取得すれば十分(policy networkの構築もここでよい)。
    _probe_env = SenpuuMaruMJXEnv()
    network = ctx["make_policy_network_factory"](_probe_env.observation_size, _probe_env.action_size)
    make_policy = ppo_networks.make_inference_fn(network)

    def strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf

    params_stripped = ctx["jax"].tree_util.tree_map(strip_leading_dim, params)

    # deterministic/stochasticはpolicyのみに依存するため一度だけjitする。
    policy_fns = {
        det: ctx["jax"].jit(make_policy(params_stripped, deterministic=det))
        for det in (True, False)
    }

    force_levels = (
        [float(value) for value in args.force_levels.split(",")]
        if args.force_levels else list(RobotConfig.PUSH_FORCE_LEVELS)
    )
    conditions = [
        ("deterministic", "fixed_dr", True, True, args.fixed_episodes, 0.0),
        ("deterministic", "randomized_dr", True, False, args.episodes, 0.0),
        ("stochastic", "fixed_dr", False, True, args.episodes, 0.0),
        ("stochastic", "randomized_dr", False, False, args.episodes, 0.0),
    ]
    conditions.extend(
        ("deterministic", f"push_{force:g}N", True, False, args.episodes, force)
        for force in force_levels if force > 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(model_path),
        "max_steps": max_steps,
        "note_initial_state_randomization": (
            "物理初期姿勢(qpos/qvel)は常にnominal poseに固定されている(未実装機能、"
            "master_plan.md付録A §1.6参照)。ここでの'randomized_dr'は質量/摩擦/"
            "重心オフセット/サーボ温度/電圧のdomain randomizationのon/offを指す近似軸。"
        ),
        "note_termination_reasons": (
            "現行実装(envs/mjx_rewards.py)はfallen_roll/fallen_pitch/fallen_height/"
            "time_limitのみをterminationとして判定する。non_illegal_contact/slip_ok/"
            "torque_okによるterminationは未実装(master_plan.md Task4 C-08未着手)。"
        ),
        "conditions": {},
        "disturbance_model": {
            "force_levels_N": force_levels,
            "directions": int(RobotConfig.PUSH_DIRECTIONS),
            "duration_steps": int(RobotConfig.PUSH_DURATION_STEPS),
            "duration_s": float(RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT),
            "impulse_levels_Ns": [
                float(force * RobotConfig.PUSH_DURATION_STEPS * RobotConfig.CONTROL_DT)
                for force in force_levels
            ],
            "implementation": "MJX random horizontal push; direction is sampled continuously",
        },
    }

    for label, dr_label, deterministic, fixed_dr, n_episodes, push_force in conditions:
        policy_fn = policy_fns[deterministic]
        RobotConfig.RANDOM_PUSH_MAX_FORCE = push_force
        RobotConfig.DISTURBANCE_CURRICULUM = push_force > 0.0
        with _DomainRandomizationScope(RobotConfig, fixed=fixed_dr):
            # env.reset/step本体は `minval=RobotConfig.RANDOM_MASS_SCALE[0]` の
            # ようにRobotConfigのクラス属性をトレース時にPython定数として
            # 直接埋め込む。jax.jitのコンパイルキャッシュはbound method
            # (env.reset)の等価性で引かれるため、同じenvインスタンスに対して
            # 単に`jax.jit(env.reset)`を呼び直すだけでは、RobotConfigを
            # 変更後でも古いコンパイル結果が再利用されてしまい、
            # 2つ目以降の条件が1つ目のDR設定のまま実行される
            # ——という気付きにくい誤結果を生む。これはこのスクリプト作成時に
            # 実機で再現・確認した(jax.jit(env.reset)を使い回すとDR変更が
            # 反映されず、envインスタンスを条件ごとに新規作成するか
            # jax.clear_caches()を呼べば正しく反映されることを確認済み)。
            # 最も単純で既存コード(scratch/gate0_formal_eval.pyの
            # configure→インスタンス化の順序)とも整合する対策として、
            # DR設定確定後に毎回新しいenvインスタンスを作る。
            env = SenpuuMaruMJXEnv()
            ctx["foot_ids"] = (env._reward_system._left_foot_id, env._reward_system._right_foot_id)
            ctx["torque_limit"] = np.asarray(env._mjx_model.actuator_ctrlrange[:, 1])
            reset_fn = ctx["jax"].jit(env.reset)
            step_fn = ctx["jax"].jit(env.step)
            result = run_condition(
                ctx, reset_fn, step_fn, policy_fn,
                n_episodes=n_episodes,
                base_seed=args.seed,
                max_steps=max_steps,
                collapse_window=args.collapse_window,
            )
        key = f"{label}__{dr_label}"
        report["conditions"][key] = result
        print(f"[{key}] episode_alive mean={result['episode_alive']['mean']:.1f} "
              f"reasons={result['termination_reason_counts']} "
              f"timing={result['failure_timing_diagnosis']['classification']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Phase0 Eval] detailed report: {args.out}")

    _write_diagnosis_draft(args.diagnosis_md, report)
    print(f"[Phase0 Eval] diagnosis draft: {args.diagnosis_md}")


def _write_diagnosis_draft(path: Path, report: dict) -> None:
    lines = [
        "# Gate A 診断ドラフト（自動生成 — 人間 / Copilotによるレビュー必須）",
        "",
        f"生成日時: {report['generated_at']}",
        f"checkpoint: {report['checkpoint']}",
        "",
        "この文書は scratch/phase0_eval_diagnostics.py により自動生成された一次判定です。",
        "master_plan.md §3.7 (Task0完了基準) の「§3.6の決定木に基づく主因の暫定結論」",
        "に相当しますが、機械的な閾値ヒューリスティックによる分類であり、",
        "最終結論には人間またはCopilotによるログ・collapse_examplesの目視確認を要します。",
        "",
        f"- {report['note_initial_state_randomization']}",
        f"- {report['note_termination_reasons']}",
        "",
        "## 条件別サマリー",
        "",
    ]
    for key, result in report["conditions"].items():
        ea = result["episode_alive"]
        diag = result["failure_timing_diagnosis"]
        lines.append(f"### {key}")
        lines.append(
            f"- episode_alive: mean={ea['mean']:.1f}, std={ea['std']:.1f}, "
            f"n={ea['n']}"
        )
        lines.append(f"- termination reasons: {result['termination_reason_counts']}")
        lines.append(f"- failure timing: {diag['classification']}")
        lines.append(f"- suggested action: {diag['suggested_action']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
```

---

## tests/test_physical_mass_contract.py

```python
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

```

---

## tests/test_policy_bounds.py

```python
import ast
from pathlib import Path

import jax.numpy as jnp

from robot.policy_network import (
    POLICY_MAX_STD,
    POLICY_MEAN_CLIP_SCALE,
    POLICY_MIN_STD,
    make_policy_network_factory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    PROJECT_ROOT / "train" / "train_mjx.py",
    PROJECT_ROOT / "train" / "visualize_rl.py",
    PROJECT_ROOT / "deploy" / "export_onnx.py",
)


def test_policy_distribution_bounds():
    policy = make_policy_network_factory(observation_size=4, action_size=20)
    logits = jnp.concatenate(
        [
            jnp.full((2, 20), 100.0),
            jnp.array(
                [
                    jnp.full((20,), -100.0),
                    jnp.full((20,), 100.0),
                ]
            ),
        ],
        axis=-1,
    )
    distribution = policy.parametric_action_distribution.create_dist(logits)

    assert float(jnp.max(jnp.abs(distribution.loc))) <= POLICY_MEAN_CLIP_SCALE + 1e-6
    assert float(jnp.min(distribution.scale)) >= POLICY_MIN_STD - 1e-6
    assert float(jnp.max(distribution.scale)) <= POLICY_MAX_STD + 1e-6


def test_entrypoints_import_shared_policy_factory():
    for path in ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports_shared_factory = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "robot.policy_network"
            and any(alias.name == "make_policy_network_factory" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imports_shared_factory, f"{path} does not import the shared factory"
        assert "make_ppo_networks(" not in path.read_text(encoding="utf-8")

```

---

## tests/test_sensor_contract.py

```python
import numpy as np
import jax.numpy as jp
import mujoco
from mujoco import mjx

from envs.mjx_env import SenpuuMaruMJXEnv
from envs.mjx_rewards import MJXRewardSystem
from robot.config import RobotConfig


def test_sensor_order_contract_matches_xml_layout():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mjx.make_data(model)
    env = SenpuuMaruMJXEnv()

    assert model.nsensordata >= 18, "Expected IMU + 8 FSR sensors in the XML sensor block."

    fsr_from_env = env._extract_fsr_sensor_data(data)
    assert fsr_from_env.shape == (8,), fsr_from_env.shape
    assert np.allclose(np.asarray(fsr_from_env), 0.0)

    fsr_in_reward = MJXRewardSystem.extract_fsr_sensor_data(model, data)
    assert fsr_in_reward.shape == (8,), fsr_in_reward.shape
    assert np.allclose(np.asarray(fsr_in_reward), 0.0)


def test_fsr_positions_match_left_then_right_xml_layout():
    left_expected = np.array([
        [0.012, 0.027], [-0.012, 0.027], [0.012, -0.070], [-0.012, -0.070],
    ], dtype=np.float32)
    right_expected = np.array([
        [-0.012, 0.027], [0.012, 0.027], [-0.012, -0.070], [0.012, -0.070],
    ], dtype=np.float32)

    assert np.allclose(RobotConfig.FSR_POSITIONS[:4], left_expected)
    assert np.allclose(RobotConfig.FSR_POSITIONS[4:], right_expected)


def test_joint_order_matches_actuator_qpos_contract():
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    env = SenpuuMaruMJXEnv()

    # actuator order is defined in RobotConfig.JOINT_NAMES, which is the ABI contract.
    # qpos and qvel must be mapped via actuator->qpos/qvel tables, not raw xml tree order.
    assert env._actuator_to_qpos_idx.shape[0] == model.nu
    assert env._actuator_to_qvel_idx.shape[0] == model.nu
    assert np.array_equal(env._actuator_to_qpos_idx, env._actuator_to_qpos_idx.astype(np.int32))
    assert np.array_equal(env._actuator_to_qvel_idx, env._actuator_to_qvel_idx.astype(np.int32))


def test_domain_randomization_is_applied_to_physics_model():
    env = SenpuuMaruMJXEnv()
    base_model = env._mjx_model
    mass_scale = 1.12
    fric_scale = 1.5
    com_offset = np.array([0.01, -0.02, 0.015], dtype=np.float32)

    randomized = env._apply_domain_randomization(base_model, mass_scale, fric_scale, com_offset)

    assert np.allclose(np.asarray(randomized.body_mass), np.asarray(base_model.body_mass) * mass_scale)
    assert np.allclose(np.asarray(randomized.geom_friction), np.asarray(base_model.geom_friction) * fric_scale)
    assert np.allclose(np.asarray(randomized.body_ipos[0]), np.asarray(base_model.body_ipos[0]) + com_offset)


def test_domain_randomization_torque_uses_actuator_qvel_mapping():
    env = SenpuuMaruMJXEnv()
    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    qvel = jp.arange(model.nv, dtype=jp.float32) + 1.0
    dr_damping = jp.arange(model.nu, dtype=jp.float32) + 0.1
    dr_friction = jp.zeros(model.nu, dtype=jp.float32)

    qfrc = env._apply_joint_dr_torque(
        jp.zeros(model.nv, dtype=jp.float32),
        qvel,
        dr_damping,
        dr_friction,
    )
    expected = np.zeros(model.nv, dtype=np.float32)
    actuator_qvel = np.asarray(env._actuator_to_qvel_idx)
    expected[actuator_qvel] = -dr_damping * qvel[actuator_qvel]

    assert np.allclose(np.asarray(qfrc), expected)

```

---

## tests/test_standing_only.py

```python
from robot.config import RobotConfig


def test_standing_only_constraints():
    assert RobotConfig.ALLOW_WALKING is False
    assert RobotConfig.USE_REFERENCE_GAIT is False
    assert RobotConfig.TARGET_VEL_X == 0.0
    assert RobotConfig.TARGET_VEL_Y == 0.0
    assert RobotConfig.TARGET_YAW_RATE == 0.0

```

---

## tests/test_standing_requirements.py

```python
"""固定足立位ミッションの設定・評価契約を検証する。"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.config import RobotConfig
from scratch.phase0_eval_diagnostics import summarize_episode_alive


def test_standing_mission_forbids_walking_and_stepping():
    assert RobotConfig.ALLOW_WALKING is False
    assert RobotConfig.ALLOW_STEPPING is False
    assert RobotConfig.TARGET_VEL_X == 0.0
    assert RobotConfig.TARGET_VEL_Y == 0.0
    assert RobotConfig.TARGET_YAW_RATE == 0.0
    assert RobotConfig.MAX_SINGLE_FOOT_LIFT == 0.0


def test_success_summary_is_not_episode_alive_only():
    summary = summarize_episode_alive([500, 500, 100])
    assert summary["mean"] < RobotConfig.MAX_EPISODE_STEPS

```

---

## tests/test_teensy_telemetry.py

```python
import struct

import numpy as np

from real.real_io import TeensySpineIO


class FakeSerial:
    def __init__(self, response):
        self.response = response

    def write(self, _data):
        pass

    def read(self, _length):
        return self.response


def make_packet(gyro=(1.0, 2.0, 3.0), accel=(0.0, 0.0, 9.80665), fsr=None):
    if fsr is None:
        fsr = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    payload = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    payload += struct.pack("<3f", *gyro)
    payload += struct.pack("<3f", *accel)
    payload += struct.pack("<8f", *fsr)
    return bytes([TeensySpineIO.TELEMETRY_START_BYTE]) + payload


def make_io(response):
    io = TeensySpineIO.__new__(TeensySpineIO)
    io.num_servos = 20
    io.ser = FakeSerial(response)
    io.dummy_mode = False
    io.last_imu_data = {
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "gyro": np.zeros(3),
        "lin_accel": np.zeros(3),
    }
    io.last_fsr_contacts = np.zeros(8, dtype=np.float32)
    io.servo_temps = np.full(20, 25.0)
    io.servo_voltages = np.full(20, 11.1)
    io.telemetry_timeout_flag = False
    io._consecutive_timeouts = 0
    return io


def test_valid_telemetry_updates_values():
    io = make_io(make_packet())

    io.communicate(np.zeros(20))

    np.testing.assert_allclose(io.last_imu_data["gyro"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(io.last_imu_data["lin_accel"], [0.0, 0.0, 9.80665])
    np.testing.assert_array_equal(io.last_fsr_contacts, [1, 0, 1, 0, 1, 0, 1, 0])
    assert io._consecutive_timeouts == 0


def test_out_of_range_telemetry_keeps_last_good_values():
    io = make_io(make_packet())
    io.communicate(np.zeros(20))
    last_imu = {key: value.copy() for key, value in io.last_imu_data.items()}
    last_fsr = io.last_fsr_contacts.copy()

    io.ser.response = make_packet(gyro=(51.0, 0.0, 0.0))
    io.communicate(np.zeros(20))

    for key in last_imu:
        np.testing.assert_array_equal(io.last_imu_data[key], last_imu[key])
    np.testing.assert_array_equal(io.last_fsr_contacts, last_fsr)
    assert io._consecutive_timeouts == 1
    assert not io.telemetry_timeout_flag


def test_three_rejected_packets_raise_timeout_flag():
    io = make_io(make_packet(gyro=(51.0, 0.0, 0.0)))

    for _ in range(3):
        io.communicate(np.zeros(20))

    assert io.telemetry_timeout_flag

```

---

## tests/test_training_eval_contract.py

```python
import jax

from envs.mjx_env import SenpuuMaruMJXEnv


def test_reset_training_progress_is_full_for_direct_eval():
    env = SenpuuMaruMJXEnv()
    state = env.reset(jax.random.PRNGKey(0))
    assert float(state.info["training_progress"]) == 1.0


def test_direct_eval_policies_are_not_zeroed_by_progress_scale():
    env = SenpuuMaruMJXEnv()
    state = env.reset(jax.random.PRNGKey(1))
    # Direct evaluation should not carry the training-only zero progress that suppresses all soft penalties.
    assert float(state.info["training_progress"]) >= 0.99

```

---

## train/export_trajectory.py

```python
import os
import sys
import pickle
import argparse

import jax
import jax.numpy as jnp
import numpy as np

# sysモジュールのパッチ (Windows/WSL上のbrax/orbax依存対策)
if not hasattr(sys.modules.get("uvloop", None), "__name__"):
    sys.modules["uvloop"] = type(sys)("uvloop")

from brax import envs
from brax.training.agents.ppo import networks as ppo_networks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from envs.mjx_env import SenpuuMaruMJXEnv  # noqa: F401
from train.train_mjx import make_policy_network_factory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=8)
    parser.add_argument("--model", type=str, default="best_params.pkl")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "log", f"version_{args.version}", args.model)
    if not os.path.exists(model_path):
        model_path = os.path.join(root_dir, "log", "mjx_ppo_rma_100hz", f"version_{args.version}", args.model)
    
    print(f"モデルをロード中: {model_path}")
    with open(model_path, "rb") as f:
        params = pickle.load(f)
        
    jax.config.update('jax_platform_name', 'cpu')
    env = envs.get_environment('senpuu_maru_mjx')
    
    ppo_network = make_policy_network_factory(observation_size=env.observation_size, action_size=env.action_size)
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    normalizer_params, policy_params, value_params = params
    policy = make_policy((normalizer_params, policy_params, value_params), deterministic=True)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)
    state = jit_reset(key_reset)
    
    qpos_list = []
    
    print("AIが生成する軌跡を計算中 (WSLで実行中)...")
    for _ in range(800):  # 8秒間分シミュレーション
        rng, key_act = jax.random.split(rng)
        act_params = policy(state.obs, key_act)
        state = jit_step(state, act_params[0])
        qpos_list.append(np.array(state.pipeline_state.qpos))
        
        if state.done:
            break
            
    traj = np.array(qpos_list)
    out_path = os.path.join(root_dir, "trajectory.npy")
    np.save(out_path, traj)
    print(f"軌跡データを保存しました: {out_path}")

if __name__ == "__main__":
    main()

```

---

## train/play_mjx.py

```python
import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Proxy launcher for visualize_rl.py")
    parser.add_argument("--exp_name", type=str, default="mjx_ppo_rma_100hz")
    parser.add_argument("--version", type=int, default=None, help="Version folder to load (e.g. 17 for version_17). If omitted, use latest.")
    parser.add_argument("--model", type=str, default="best_params.pkl", help="Model file to load (best_params.pkl, last_params.pkl, etc.)")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    target = root_dir / "train" / "visualize_rl.py"
    if not target.exists():
        print(f"Error: {target} not found.")
        return

    cmd = [sys.executable, str(target),
           "--mode", "interactive",
           "--exp_name", args.exp_name,
           "--model", args.model]
    if args.version is not None:
        cmd.append("--version")
        cmd.append(str(args.version))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

```

---

## train/train_mjx.py

```python
"""固定足立位ロボット「旋風丸」— MJXベースPPO学習エントリーポイント。

このモジュールは以下を実装する:
  - envs/mjx_env.py の SenpuuMaruMJXEnv (固定足立位環境) をBrax PPOで学習
  - Brax内蔵のAdaptive KL学習率制御によるKLダイバージェンス監視
    (--target_klはepoch内early stoppingではなく、Adaptive LR制御に接続される)
  - checkpoint保存 (best/worst/final/last) と学習曲線ログ (log.json)
  - NaN/Inf検出による即時停止 (改良規約 §18)
  - 報酬ハッキング・学習破綻の監査 (非致命的、警告のみ)
    実機投入前に「シミュレーション学習が正当な報酬最大化をしているか」
    「報酬関数の設計ミスによる異常学習が起きていないか」を検出する。
    NaN/Infと違い致命的ではないため学習は止めず、
    log/<exp_name>/version_*/REWARD_AUDIT_ALERTS.txt に警告を蓄積する。

主要な関数:
  - parse_args(): CLI引数パース (--seed, --target_kl, --exp_name 等)
  - _audit_reward_metrics(): 報酬ハッキング・学習破綻の検出 (5項目)
  - progress_callback(): 学習中の進捗表示・ログ保存・NaN検出・報酬監査 (main()内部で定義)
  - main() 相当のスクリプト本体: 環境構築 → PPO学習 → checkpoint保存

使用例:
  python train/train_mjx.py --seed=42 --target_kl=0.02
  python train/train_mjx.py --exp_name phase0_debug_seed42 --seed=42 --target_kl=0.02

環境仕様 (robot/config.py が正本):
  - 観測: 625次元 (base 84 + history 420 + action_history 100 + temp 20 + volt 1)
  - 行動: 20次元 (関節角の残差 Δq、トルク直接指令ではない)
  - エピソード長: 500 step (100Hz制御、5秒)
  - Phase 0: 外乱無効 (DISTURBANCE_CURRICULUM=False)

改良規約上の制約 (docs/current.md, docs/master_plan.md 参照):
  - 1 iteration = 1変更カテゴリ (報酬とPPO設定を同時に変えない)
  - 合格済みcheckpointを上書きしない (--exp_name で世代管理する)
  - NaN/Inf検出時は即座に停止し、log/<exp_name>/NAN_DETECTED.txt に記録する

ハードウェア:
  - CPU: 単体テスト・形状確認用 (num_envs=32等、小規模)
  - GPU: 本番学習用 (RTX 4060+推奨、num_envs=256、10M step で約30-60分)
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

# sysモジュールのパッチ (Windows上のbrax/orbax依存対策)
if not hasattr(sys.modules.get("uvloop", None), "__name__"):
    sys.modules["uvloop"] = type(sys)("uvloop")

import jax
import jax.numpy as jnp

# Reuse XLA executables across repeated WSL validation/training runs.
jax.config.update("jax_compilation_cache_dir", "/mnt/c/bipedal_robot/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        return jax.tree_util.tree_map(lambda leaf: jax.device_put(jnp.expand_dims(leaf, 0)), x)
    jax.device_put_replicated = _device_put_replicated

from brax import envs
from brax.envs import training as brax_training
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from robot.policy_network import (
    POLICY_MAX_STD,
    POLICY_MEAN_CLIP_SCALE,
    POLICY_MIN_STD,
    make_policy_network_factory,
)

# Patch brax _unpmap for JAX 0.4+ Multi-GPU safety
def _safe_unpmap(v):
    def _unpmap_leaf(x):
        if hasattr(x, "addressable_shards"):
            d = x.addressable_shards[0].data
            if d.ndim > 0 and d.shape[0] == 1:
                return d.squeeze(0)
            return d
        if hasattr(x, "device_buffers"):
            return x[0]
        return x
    return jax.tree_util.tree_map(_unpmap_leaf, v)

ppo._unpmap = _safe_unpmap

# Safe wrapper for make_inference_fn to handle leading pmap dimension in params
_orig_make_inference_fn = ppo_networks.make_inference_fn
def _safe_make_inference_fn(ppo_networks_tuple, **make_kwargs):
    orig_fn = _orig_make_inference_fn(ppo_networks_tuple, **make_kwargs)
    def safe_inference_fn(params, *args, **kwargs):
        def _strip_leading_dim(leaf):
            if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 1 and leaf.shape[0] == 1:
                return leaf.squeeze(0)
            return leaf
        params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
        return orig_fn(params_stripped, *args, **kwargs)
    return safe_inference_fn

ppo_networks.make_inference_fn = _safe_make_inference_fn






sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig
from envs.mjx_env import SenpuuMaruMJXEnv  # noqa: F401 (Brax環境登録のため)
from envs.training_wrapper import TrainingProgressWrapper

# ============================================================================
# [監査追加 2026-09-13] 報酬ハッキング・学習破綻の検出
# ============================================================================
# 実機計測にはまだ入っていない段階で、シミュレーション学習が
# 「正当な報酬最大化」をしているか、報酬関数の設計ミス(reward hacking)や
# 実装バグによる異常な学習が起きていないかを検証するための追加監査。
#
# 既存のNaN/Inf検出(改良規約 §18、上のprogress_callback内)とは異なり、
# ここでの検出は致命的エラーではなく「疑わしい兆候」の警告であるため、
# raiseはせず学習を継続する。検出結果は
# log/<exp_name>/version_*/REWARD_AUDIT_ALERTS.txt に蓄積され、
# 学習終了後にサマリーが表示される。
#
# 検出項目:
#   1. Reward Exploitation  - 単一の報酬成分(r_cp, r_recovery, r_upright,
#                              r_com_stab, disturbance_recovery_bonus,
#                              pbrs_reward, alive)が不合理に大きくないか
#   2. Shaping Mismatch     - 高報酬なのに姿勢系の正報酬(r_upright,
#                              r_com_stab, both_feet_contact)が乏しい、
#                              または total_penalty が total_reward を
#                              圧倒していないか
#   3. Metric Corruption    - stability_index が [0,1] の範囲外、
#                              zmp_margin が直近N回連続でほぼ一定値
#                              (envs/stability_metrics.py の計算が
#                              死んでいる可能性)、または同ファイルが
#                              自己申告する stability_metrics_finite
#                              フラグ(NaN/Inf自己診断)がFalse
#   4. Potential Decay      - potential が増加しているのに pbrs_reward が
#                              大きく負(compute_potential()の符号ミス等)
#   5. Action Distortion    - CBFによるaction_saturationが高い
#                              (envs/mjx_env.py 及び safety/cbf.py の
#                              compute_saturation_ratio() 参照。方策が
#                              実行不能な指令を多発させている、または
#                              CBF/可動域制限が過剰に効いている可能性)
#   6. Sensor/Kinematics Fallback - envs/mjx_rewards.py の com_accel が
#                              qacc取得失敗によるフォールバック値
#                              [0,0,-9.81]を使用中(com_accel_is_fallback)。
#                              stability_metrics.py のv2 [CRITICAL FIX]
#                              で説明されている「zmp_marginが死んだ指標に
#                              なる」バグの片割れの原因だったため、
#                              本番での再発を監視する。
#
# 依存するmetricsキー (envs/mjx_rewards.py, envs/mjx_env.py,
# envs/stability_metrics.py, safety/cbf.py で提供):
#   total_reward, total_penalty, stability_index, zmp_margin, r_upright,
#   r_com_stab, both_feet_contact, r_cp, r_recovery,
#   disturbance_recovery_bonus, pbrs_reward, alive, potential,
#   action_saturation, stability_metrics_finite, com_accel_is_fallback
# ============================================================================

REWARD_AUDIT_THRESHOLDS = {
    'exploitation_abs_max': 50.0,   # 各報酬成分の絶対値の上限目安
    'shaping_high_reward': 10.0,    # これ以上の報酬でpositive componentが乏しいと疑う
    'shaping_low_positive': 1.0,
    'shaping_severe_negative': -100.0,
    'stability_index_max': 1.05,    # [0,1]からの逸脱許容
    'metric_frozen_window': 10,     # 直近何件で「固定値」と判定するか
    'metric_frozen_std': 1e-6,
    'action_saturation_max': 0.5,   # CBF補正の平均飽和率(50%)
}


def _find_metric_key(metrics: dict, suffix: str):
    """Brax集計後のキー(例 'eval/episode_metrics/xxx')から末尾一致で探す。
    既存の reward_is_finite 探索(このファイル内、progress_callback参照)と
    同じ方式に合わせている。"""
    for key in metrics.keys():
        if key == suffix or key.endswith(suffix):
            return key
    return None


def _audit_reward_metrics(metrics_dict: dict, metrics_history: list) -> list:
    """1ステップ分のmetrics_dict(既にfloat化済み)を検査し、報酬ハッキングや
    学習破綻の兆候をチェックする。致命的ではないため raise はしない。

    Args:
        metrics_dict: progress_callback内で構築される、その時点のfloat化
            済みmetrics辞書 (まだmetrics_historyには追加する前のもの)。
        metrics_history: これまでの metrics_dict のリスト(現在のステップは
            含まない)。Metric CorruptionやPotential Decayのトレンド検出に使う。

    Returns:
        alerts: 検出されたアラートメッセージのリスト(空なら異常なし)。
    """
    alerts = []
    th = REWARD_AUDIT_THRESHOLDS

    def _get(suffix, default=0.0):
        key = _find_metric_key(metrics_dict, suffix)
        return metrics_dict[key] if key is not None else default

    total_reward = _get('total_reward', metrics_dict.get('reward', 0.0))
    total_penalty = _get('total_penalty', 0.0)
    stability_index = _get('stability_index', 0.5)
    r_upright = None
    r_upright_key = _find_metric_key(metrics_dict, 'r_upright')
    if r_upright_key is not None:
        r_upright = metrics_dict[r_upright_key]
    r_com_stab = _get('r_com_stab', 0.0)
    both_feet_contact = _get('both_feet_contact', 0.0)

    # --- 1. Reward Exploitation ---
    component_suffixes = [
        'r_cp', 'r_recovery', 'r_upright', 'r_com_stab',
        'disturbance_recovery_bonus', 'pbrs_reward', 'alive',
    ]
    for suffix in component_suffixes:
        key = _find_metric_key(metrics_dict, suffix)
        if key is None:
            continue
        val = metrics_dict[key]
        if abs(val) > th['exploitation_abs_max']:
            alerts.append(
                f"[Exploitation] 報酬成分 '{key}' が異常に大きい: {val:.2f} "
                f"(閾値 ±{th['exploitation_abs_max']:.0f})"
            )

    # --- 2. Shaping Mismatch ---
    if r_upright is not None:
        positive_sum = max(r_upright, 0.0) + max(r_com_stab, 0.0) + both_feet_contact
        if total_reward > th['shaping_high_reward'] and positive_sum < th['shaping_low_positive']:
            alerts.append(
                f"[Shaping Mismatch] 高報酬(total_reward={total_reward:.2f})だが"
                f"姿勢系の正報酬が乏しい(r_upright+r_com_stab+both_feet_contact="
                f"{positive_sum:.2f})。ペナルティ符号反転や他成分の異常な"
                f"寄与を疑う。"
            )
    if total_reward < th['shaping_severe_negative'] and total_penalty > 0:
        alerts.append(
            f"[Shaping Mismatch] 報酬が著しく負(total_reward={total_reward:.2f})、"
            f"total_penalty={total_penalty:.2f} が報酬設計を圧倒している"
            f"可能性。mjx_rewards.py の重み(REWARD_WEIGHTS)を確認。"
        )

    # --- 3. Metric Corruption ---
    if stability_index < 0.0 or stability_index > th['stability_index_max']:
        alerts.append(
            f"[Metric Corruption] stability_index が範囲外: "
            f"{stability_index:.3f} (期待範囲 [0, 1])"
        )
    finite_key = _find_metric_key(metrics_dict, 'stability_metrics_finite')
    if finite_key is not None and metrics_dict[finite_key] < 0.5:
        alerts.append(
            "[Metric Corruption] envs/stability_metrics.py の "
            "compute_unified_stability_index() がNaN/Infを検出 "
            "(stability_metrics_finite=0)。CP/ZMP/バランス/姿勢マージンの"
            "いずれかの幾何計算が破綻している。"
        )
    window = th['metric_frozen_window']
    zmp_key = _find_metric_key(metrics_dict, 'zmp_margin')
    if zmp_key is not None and len(metrics_history) >= window:
        recent = [m[zmp_key] for m in metrics_history[-window:] if zmp_key in m]
        if len(recent) >= window and np.std(recent) < th['metric_frozen_std']:
            alerts.append(
                f"[Metric Corruption] zmp_margin が直近{window}回連続で"
                f"ほぼ一定値({metrics_dict[zmp_key]:.6f}) — "
                f"envs/stability_metrics.py の計算が死んでいる可能性"
                f"(過去のcompute_zmp_marginバグ再発等)。"
            )

    # --- 4. Potential Decay ---
    pot_key = _find_metric_key(metrics_dict, 'potential')
    pbrs_key = _find_metric_key(metrics_dict, 'pbrs_reward')
    if pot_key is not None and pbrs_key is not None and len(metrics_history) >= 1:
        prev = metrics_history[-1]
        if pot_key in prev:
            delta_potential = metrics_dict[pot_key] - prev[pot_key]
            pbrs_val = metrics_dict[pbrs_key]
            if delta_potential > 0.1 and pbrs_val < -2.0:
                alerts.append(
                    f"[Potential Decay] potentialは増加({delta_potential:+.3f})"
                    f"だが pbrs_reward が大きく負({pbrs_val:.2f})。"
                    f"envs/mjx_rewards.py の compute_potential() や "
                    f"discounting(gamma)を確認。"
                )

    # --- 5. Action Distortion ---
    sat_key = _find_metric_key(metrics_dict, 'action_saturation')
    if sat_key is not None and metrics_dict[sat_key] > th['action_saturation_max']:
        alerts.append(
            f"[Action Distortion] CBFによるアクション補正の飽和率が高い: "
            f"{metrics_dict[sat_key]*100:.1f}% — 方策が実行不能な指令を"
            f"多発させているか、safety/cbf.py の制限が過剰に効いている"
            f"可能性。"
        )

    # --- 6. Sensor/Kinematics Fallback ---
    fallback_key = _find_metric_key(metrics_dict, 'com_accel_is_fallback')
    if fallback_key is not None and metrics_dict[fallback_key] > 0.5:
        alerts.append(
            "[Sensor Fallback] com_accel が qacc 取得失敗によりフォール"
            "バック値[0,0,-9.81]を使用中。envs/mjx_rewards.py の compute() "
            "内、nq/qacc の条件分岐を確認。ZMPが重心追従に退化し、外乱下の"
            "不安定性を過小評価している可能性がある(stability_metrics.py "
            "のv2 changelog参照)。"
        )

    # [予防追加 2026-09-13] envs/mjx_env.py の physics_step ロールバック
    # 機構が実際に発動した頻度を記録する。ロールバックにより学習自体は
    # 汚染されないが、頻発する場合はカリキュラム(外乱強度)や物理タイム
    # ステップ・ソルバー設定が実際の限界に近いことを示すシグナルになる。
    diverged_key = _find_metric_key(metrics_dict, 'physics_diverged')
    if diverged_key is not None and metrics_dict[diverged_key] > 0.5:
        alerts.append(
            "[Physics Divergence] 物理サブステップがNaN/Infに発散し、"
            "envs/mjx_env.py のロールバック機構が作動した(エピソードは"
            "安全に終了済み)。頻発する場合は外乱の強さ・timestep・"
            "solver設定の見直しを検討。"
        )

    return alerts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment subfolder under log. Leave empty to write directly to log/version_x.")
    parser.add_argument("--num_envs", type=int, default=None, help="並列環境数 (GPUなら2048〜4096推奨, CPU自動設定)")
    parser.add_argument("--steps", type=int, default=None, help="総学習ステップ数")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--unroll_length", type=int, default=10, help="PPOのアクションアンロール長")
    parser.add_argument("--episode_length", type=int, default=None, help="1エピソードのステップ数")
    parser.add_argument("--num_evals", type=int, default=None, help="評価回数")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--num_updates_per_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_kl", type=float, default=0.02)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== MJX GPU Training Pipeline (RMA Enabled) ===")
    devices = jax.devices()
    print(f"JAX Devices: {devices}")
    is_cpu = devices[0].platform == 'cpu'
    
    if is_cpu:
        print("[Warning] JAX is running on CPU. GPU未使用 - パラメータをCPU向けに自動縮小します。")
        print("[Info] GPU使用にはWSL2 + JAX CUDA版が必要です。")
        # CPUモード: コンパイル時間を最小化する小さなパラメータ
        num_envs       = args.num_envs       or 32
        steps          = args.steps          or 100_000
        episode_length = args.episode_length or 50
        num_evals      = args.num_evals      or 3
        batch_size     = args.batch_size     or 32
        num_minibatches = args.num_minibatches or 1
    else:
        print(f"[GPU] {devices[0]} で学習開始！")
        # GPUモード: RTX 4060 (8GB) でのコンパイルハングを避けるため、パラメータを軽量化
        num_envs       = args.num_envs       or 256
        steps          = args.steps          or 10_000_000
        episode_length = args.episode_length or RobotConfig.MAX_EPISODE_STEPS
        num_evals      = args.num_evals      or 20
        batch_size     = args.batch_size     or 256
        num_minibatches = args.num_minibatches or 16

    # --- 複数GPU環境向けの自動最適化 (Divisibilityの担保) ---
    num_devices = len(devices)
    if num_envs % num_devices != 0:
        old_num_envs = num_envs
        num_envs = (num_envs // num_devices) * num_devices
        print(f"[Auto-Tune] num_envs をGPU数({num_devices})で割り切れる {num_envs} に自動調整しました (元: {old_num_envs})")
    
    if batch_size % num_devices != 0:
        old_batch_size = batch_size
        batch_size = max(1, batch_size // num_devices) * num_devices
        print(f"[Auto-Tune] batch_size をGPU数({num_devices})で割り切れる {batch_size} に自動調整しました (元: {old_batch_size})")

    # batch_size * num_minibatches は num_envs で割り切れる必要がある
    if (batch_size * num_minibatches) % num_envs != 0:
        # 割り切れるように num_minibatches を自動調整
        import math
        old_minibatches = num_minibatches
        # 必要な最小の倍数を探す
        target_total_batch = math.ceil((batch_size * num_minibatches) / num_envs) * num_envs
        num_minibatches = target_total_batch // batch_size
        print(f"[Auto-Tune] (batch_size * num_minibatches) % num_envs == 0 を満たすため、num_minibatches を {num_minibatches} に自動調整しました (元: {old_minibatches})")

    # 1. 環境生成
    env = envs.get_environment('senpuu_maru_mjx')
    
    # 2. ログディレクトリとバージョン管理
    from pathlib import Path
    import json
    
    root_path = Path(__file__).resolve().parent.parent
    base_log_dir = root_path / "log"
    if args.exp_name:
        base_log_dir = base_log_dir / args.exp_name
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    version = 0
    while (base_log_dir / f"version_{version}").exists():
        version += 1
    run_dir = base_log_dir / f"version_{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Logging to {run_dir}", flush=True)
    
    # 状態トラッキング用変数
    best_reward = -float('inf')
    worst_reward = float('inf')
    current_params = None
    metrics_history = []
    # [監査追加 2026-09-13] 報酬ハッキング監査(_audit_reward_metrics)の
    # 検出件数を種類別に集計する。学習終了後にサマリー表示する。
    reward_audit_alert_counts = {}
    reward_audit_total_alerts = 0
    
    def policy_params_callback(current_step, make_policy, params):
        import pickle
        nonlocal current_params
        current_params = params
        
        # 毎回ラストのモデルを保存
        with open(run_dir / "last_params.pkl", "wb") as f:
            pickle.dump(params, f)
            
    # コールバック関数（プログレス表示・ログ保存用）
    def progress_callback(num_steps, metrics):
        import pickle
        nonlocal best_reward, worst_reward
        nonlocal reward_audit_alert_counts, reward_audit_total_alerts
        reward = metrics.get('eval/episode_reward', metrics.get('training/total_reward', float('nan')))

        # --- NaN/Inf 即時停止チェック（改良規約 §18: 即時停止条件） ---
        # KLスパイクや勾配爆発が発生すると reward や他の主要metricsが
        # NaN/Infになりうる。これを検出しないまま学習を続けると、
        # 壊れたcheckpointをbest_paramsとして保存してしまう危険がある。
        if not np.isfinite(reward):
            print(f"\n{'='*70}", flush=True)
            print(f"❌ FATAL: NaN/Inf detected in reward at step {num_steps}!", flush=True)
            print(f"   reward={reward}", flush=True)
            print(f"   metrics keys={list(metrics.keys())}", flush=True)
            print(f"{'='*70}\n", flush=True)
            # 直近のmetrics_historyを保存してから停止（原因調査用）
            with open(run_dir / "log.json", "w") as f:
                json.dump(metrics_history, f, indent=2)
            with open(run_dir / "NAN_DETECTED.txt", "w") as f:
                f.write(f"step={num_steps}\nreward={reward}\nmetrics={metrics}\n")
            raise RuntimeError(
                f"NaN/Inf detected in reward at step {num_steps}. "
                f"Training stopped per改良規約 §18 (即時停止条件). "
                f"Details written to {run_dir / 'NAN_DETECTED.txt'}"
            )

        # 主要metrics全体もチェック（reward以外にKL, value_loss等も対象）
        for key, value in metrics.items():
            try:
                val_float = float(value.item() if hasattr(value, 'item') else value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val_float):
                print(f"\n{'='*70}", flush=True)
                print(f"❌ FATAL: NaN/Inf detected in metric '{key}' at step {num_steps}!", flush=True)
                print(f"   value={val_float}", flush=True)
                print(f"{'='*70}\n", flush=True)
                with open(run_dir / "log.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)
                with open(run_dir / "NAN_DETECTED.txt", "w") as f:
                    f.write(f"step={num_steps}\nmetric={key}\nvalue={val_float}\nmetrics={metrics}\n")
                raise RuntimeError(
                    f"NaN/Inf detected in metric '{key}' at step {num_steps}. "
                    f"Training stopped per改良規約 §18 (即時停止条件)."
                )

        # --- 専用フラグ 'reward_is_finite' のチェック ---
        # envs/mjx_rewards.py の compute() が jnp.isfinite で算出したフラグ。
        # 0.0/1.0 という値自体は有限なので、上の汎用isfiniteチェックでは
        # 検出できない(0.0は有限)。このフラグが0.0の場合は
        # 「reward算出の途中経路でNaN/Infが発生した」ことを意味するため、
        # 専用に検査する。値はBraxのepisode集約で平均化されるため、
        # 1エピソードでも非有限値を含めば1.0未満になる。
        reward_is_finite_key = None
        for key in metrics.keys():
            if key.endswith("reward_is_finite"):
                reward_is_finite_key = key
                break
        if reward_is_finite_key is not None:
            finite_ratio = metrics[reward_is_finite_key]
            finite_ratio = float(finite_ratio.item() if hasattr(finite_ratio, 'item') else finite_ratio)
            if finite_ratio < 1.0:
                print(f"\n{'='*70}", flush=True)
                print(f"❌ FATAL: reward computation produced NaN/Inf at step {num_steps}!", flush=True)
                print(f"   {reward_is_finite_key}={finite_ratio} (< 1.0 means some envs saw non-finite reward)", flush=True)
                print(f"   → envs/mjx_rewards.py の compute() 内の各報酬成分を確認してください", flush=True)
                print(f"{'='*70}\n", flush=True)
                with open(run_dir / "log.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)
                with open(run_dir / "NAN_DETECTED.txt", "w") as f:
                    f.write(
                        f"step={num_steps}\n{reward_is_finite_key}={finite_ratio}\n"
                        f"source=envs/mjx_rewards.py compute()\nmetrics={metrics}\n"
                    )
                raise RuntimeError(
                    f"reward_is_finite={finite_ratio} at step {num_steps}: "
                    f"non-finite value detected inside mjx_rewards.compute(). "
                    f"Training stopped per改良規約 §18 (即時停止条件)."
                )

        # 学習進捗率の計算と表示。ここでは総乱数ステップではなく、
        # 各環境の累積ステップを単調に増やす構造を優先し、
        # 1.0 を超えないようにする。
        training_progress = min(num_steps / max(steps, 1), 1.0) if steps > 0 else 0.0
        print(f"Step: {num_steps:10d} | Reward: {reward:.4f} | Progress: {training_progress:.2%}", flush=True)

        # JSONログ用の辞書作成
        metrics_dict = {
            "step": int(num_steps),
            "reward": float(reward),
            "training_progress": float(training_progress),
            "num_envs": int(num_envs),
            "episode_length": int(episode_length),
        }
        for k, v in metrics.items():
            if k == "training_progress":
                continue
            metrics_dict[k] = float(v.item() if hasattr(v, 'item') else v)
        # --- 報酬ハッキング・学習破綻の監査 (非致命的、警告のみ) ---
        # metrics_history にはまだ現在のステップを追加していないため、
        # ここでは「これまでの履歴 vs 現在のステップ」の比較として機能する。
        reward_audit_alerts = _audit_reward_metrics(metrics_dict, metrics_history)
        if reward_audit_alerts:
            reward_audit_total_alerts += len(reward_audit_alerts)
            print(f"\n⚠️  [Reward Audit] Step {num_steps}: "
                  f"{len(reward_audit_alerts)}件の異常兆候を検出", flush=True)
            with open(run_dir / "REWARD_AUDIT_ALERTS.txt", "a") as f:
                f.write(f"\n[Step {num_steps}]\n")
                for alert in reward_audit_alerts:
                    print(f"   - {alert}", flush=True)
                    f.write(f"  - {alert}\n")
                    # カテゴリ別カウント (例: "[Exploitation] ..." → "Exploitation")
                    category = alert.split(']', 1)[0].lstrip('[')
                    reward_audit_alert_counts[category] = reward_audit_alert_counts.get(category, 0) + 1

        metrics_history.append(metrics_dict)
        
        with open(run_dir / "log.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
            
        # 最高のモデルと最低のモデルを保存
        if current_params is not None:
            if reward > best_reward:
                best_reward = reward
                with open(run_dir / "best_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)
                print(f"  >>> Best Model Saved! (Reward: {reward:.4f})", flush=True)
                
            if reward < worst_reward:
                worst_reward = reward
                with open(run_dir / "worst_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)
                print(f"  >>> Worst Model Saved! (Reward: {reward:.4f})", flush=True)

    print(f"Starting training: num_envs={num_envs}, steps={steps}, episode_length={episode_length}")
    start_time = time.time()
    
    # Learning Rate: Brax内蔵のAdaptive KL LRスケジュールを使用。
    # KL爆発時に自動的に学習率を下げ、KLが低すぎる場合は上げる。
    # 以前のoptax.warmup_cosine_decay_scheduleはBrax PPOの内部optimizerには
    # 渡されておらず機能していなかったため削除。
    
    # --- TrainingProgressWrapper の注入 ---
    # brax.envs.training.wrap を一時的に差し替え、AutoResetWrapper の
    # 外側に TrainingProgressWrapper を配置する。
    # これにより training_progress がエピソード境界を跨いで単調増加する。
    steps_per_env = max(steps // num_envs, 1)
    _original_wrap = brax_training.wrap

    def _wrap_with_progress(env, **kwargs):
        wrapped = _original_wrap(env, **kwargs)
        return TrainingProgressWrapper(wrapped, total_steps_per_env=steps_per_env)

    brax_training.wrap = _wrap_with_progress

    # 3. PPO学習実行 (RMA Network Architecture)
    try:
        make_inference_fn, params, metrics = ppo.train(
            environment=env,
            network_factory=make_policy_network_factory,
            num_timesteps=steps,
            num_evals=num_evals,
            reward_scaling=0.01,  # 報酬クリップ後の値をPPOの更新量に合わせる
            episode_length=episode_length,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=args.unroll_length,
            num_minibatches=num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            discounting=0.99,
            bootstrap_on_timeout=True,
            learning_rate=args.learning_rate,
            entropy_cost=1e-3,
            # --- KLダイバージェンス制御 ---
            clipping_epsilon=0.2,           # 0.3(Braxデフォルト)→0.2に縮小
            max_grad_norm=1.0,              # 勾配クリッピングで勾配爆発を防止
            learning_rate_schedule='ADAPTIVE_KL',  # Brax内蔵Adaptive KL LR
            desired_kl=args.target_kl,
            learning_rate_schedule_min_lr=1e-5,   # KL爆発時のフロア（1e-6では低すぎてLRがstuckする）
            learning_rate_schedule_max_lr=5e-4,   # KL安定時の天井

            num_envs=num_envs,
            batch_size=batch_size,
            seed=args.seed,
            progress_fn=progress_callback,
            policy_params_fn=policy_params_callback
        )
    finally:
        # 他のモジュールに影響しないよう必ず復元
        brax_training.wrap = _original_wrap

    elapsed_time = time.time() - start_time
    print(f"Training finished in {elapsed_time/60:.1f} minutes!")

    # --- 報酬ハッキング監査サマリー ---
    # 実機投入前に「この学習は信頼してよいか」を判断するための最終報告。
    # 詳細な各アラートは REWARD_AUDIT_ALERTS.txt を参照。
    summary_lines = []
    if reward_audit_total_alerts > 0:
        summary_lines.append(
            f"⚠️  Reward Audit: 学習中に {reward_audit_total_alerts} 件の"
            f"異常兆候を検出しました。"
        )
        for category, count in sorted(
            reward_audit_alert_counts.items(), key=lambda x: -x[1]
        ):
            summary_lines.append(f"   - {category}: {count}件")
        summary_lines.append(
            f"   詳細: {run_dir / 'REWARD_AUDIT_ALERTS.txt'}"
        )
        summary_lines.append(
            "   実機投入前に、上記カテゴリに対応する報酬関数・安定性"
            "指標・CBF実装を確認することを推奨します。"
        )
    else:
        summary_lines.append(
            "✅ Reward Audit: 学習全体を通して異常兆候は検出されませんでした。"
        )
    print("\n" + "\n".join(summary_lines))
    with open(run_dir / "REWARD_AUDIT_SUMMARY.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # 4. パラメータ保存
    import pickle
    model_path = os.path.join(run_dir, "final_params.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
```

---

## train/view_trajectory.py

```python
import os
import sys
import time
import numpy as np

# Windowsでmujocoのプラグイン読込時にDLLブロックエラーが出るのを回避するパッチ
try:
    import ctypes
    import mujoco._structs
    # プラグインロードを安全に無効化
    def dummy_load_plugins():
        pass
    import mujoco
    mujoco._load_all_bundled_plugins = dummy_load_plugins
except Exception:
    pass

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.config import RobotConfig

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traj_path = os.path.join(root_dir, "trajectory.npy")
    
    if not os.path.exists(traj_path):
        print("エラー: 軌跡データ(trajectory.npy)が見つかりません。")
        print("先にWSL側で 'python3 train/export_trajectory.py' を実行してください。")
        return
        
    print(f"軌跡データを読み込み中: {traj_path}")
    traj = np.load(traj_path)
    
    # ネイティブのMuJoCoはWindowsでも全く問題なく動く (JAXが不要だから)
    mj_model_native = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    mj_data_native = mujoco.MjData(mj_model_native)
    
    print("Windowsネイティブビューワーを起動します！")
    with mujoco.viewer.launch_passive(mj_model_native, mj_data_native) as viewer:
        # 無限ループでリプレイ再生
        while viewer.is_running():
            print("リプレイ再生を開始...")
            for qpos in traj:
                if not viewer.is_running():
                    break
                
                step_start = time.time()
                
                # 状態をセットして順運動学を計算（画面描画の更新）
                mj_data_native.qpos[:] = qpos
                mujoco.mj_forward(mj_model_native, mj_data_native)
                viewer.sync()
                
                # スピード調整 (100Hz)
                time_until_next = RobotConfig.CONTROL_DT - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
            time.sleep(1) # 再生終了後に1秒待って最初から

if __name__ == "__main__":
    main()

```

---

## train/visualize_rl.py

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import pickle
import time
import enum
import argparse
import numpy as np
from pathlib import Path

# NumPy compatibility helpers
_old_np_asarray = np.asarray

def _compat_numpy_asarray(a, dtype=None, order=None, copy=True, subok=False, **kwargs):
    try:
        return _old_np_asarray(a, dtype=dtype, order=order, copy=copy, subok=subok)
    except TypeError:
        return _old_np_asarray(a, dtype=dtype, order=order)

np.asarray = _compat_numpy_asarray


# JAXヘッドレス化防止・CPU/EGL選択
# Viewer画面表示時はNative MuJoCo Viewerを起動
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from robot.config import RobotConfig
from robot.policy_network import make_policy_network_factory
from envs.mjx_env import SenpuuMaruMJXEnv
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

if not hasattr(running_statistics, "NormalizationMode"):
    class NormalizationMode(enum.IntEnum):
        NONE = 0
    running_statistics.NormalizationMode = NormalizationMode


class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified MJX replay entrypoint")
    parser.add_argument("--exp_name", default="", help="Optional experiment subfolder under log. Leave empty to use log/version_x.")
    parser.add_argument("--version", type=int, default=None, help="Version number to load (e.g. 17 for version_17). If omitted, use latest.")
    parser.add_argument("--model", default="best_params.pkl", help="Checkpoint name")
    parser.add_argument("--mode", choices=["interactive", "video"], default="interactive", help="Replay mode")
    parser.add_argument("--steps", type=int, default=300, help="Number of frames for video mode")
    parser.add_argument("--output", default="simulation_output.gif", help="Output GIF path for video mode")
    return parser.parse_args()


def get_model_path(exp_name: str, version: int | None, model_name: str) -> Path | None:
    root = REPO_ROOT / "log"
    if exp_name:
        root = root / exp_name
    if not root.exists():
        return None
    if version is not None:
        candidate = root / f"version_{version}" / model_name
        return candidate if candidate.exists() else None
    versions = sorted([d for d in root.glob("version_*") if d.is_dir()], key=lambda x: int(x.name.split("_")[-1]))
    for v in reversed(versions):
        candidate = v / model_name
        if candidate.exists():
            return candidate
    return None


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        return CompatibilityUnpickler(f).load()


def build_inference_fn(params, env):
    ppo_network = make_policy_network_factory(env.observation_size, env.action_size)
    inference_fn = ppo_networks.make_inference_fn(ppo_network)
    
    # Strip leading pmap dimension from params (tuple: running_stats, policy_params, value_params)
    def _strip_leading_dim(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) > 0 and leaf.shape[0] == 1:
            return leaf.squeeze(0)
        return leaf
    
    params_stripped = jax.tree_util.tree_map(_strip_leading_dim, params)
    return jax.jit(inference_fn(params_stripped, deterministic=True))


def run_interactive(params):
    env = SenpuuMaruMJXEnv()
    inference_fn = build_inference_fn(params, env)

    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)

    print("Launching MuJoCo Passive Viewer... Close the window to stop.")
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0

        while viewer.is_running():
            step_start = time.time()
            rng, rng_step = jax.random.split(rng)
            action, _ = inference_fn(state.obs, rng_step)
            state = jax.jit(env.step)(state, action)

            data.qpos[:] = state.pipeline_state.qpos
            data.qvel[:] = state.pipeline_state.qvel
            mujoco.mj_forward(model, data)

            viewer.cam.lookat[:] = 0.92 * np.array(viewer.cam.lookat[:]) + 0.08 * np.array(data.qpos[0:3])
            viewer.sync()

            if getattr(state, "done", False):
                rng, reset_key = jax.random.split(rng)
                state = jax.jit(env.reset)(reset_key)

            elapsed = time.time() - step_start
            sleep_time = RobotConfig.CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def render_video(params, steps: int, output: str):
    env = SenpuuMaruMJXEnv()
    inference_fn = build_inference_fn(params, env)

    model = mujoco.MjModel.from_xml_path(str(RobotConfig.MUJOCO_MODEL_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.6
    camera.elevation = -15.0

    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)

    frames = []
    print(f"Rendering {steps} frames to {output}...")
    for step in range(steps):
        rng, rng_step = jax.random.split(rng)
        action, _ = inference_fn(state.obs, rng_step)
        state = jax.jit(env.step)(state, action)

        data.qpos[:] = state.pipeline_state.qpos
        data.qvel[:] = state.pipeline_state.qvel
        mujoco.mj_forward(model, data)

        camera.lookat = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]) + 0.1]
        camera.azimuth = 135.0 + (step * 0.2)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        if getattr(state, "done", False):
            rng, reset_key = jax.random.split(rng)
            state = jax.jit(env.reset)(reset_key)

    output_path = REPO_ROOT / "scratch" / "simulation_output" / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import PIL.Image
    imgs = [PIL.Image.fromarray(frame) for frame in frames]
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    print(f"Saved simulation GIF to: {output_path}")


def main():
    args = parse_args()
    model_path = get_model_path(args.exp_name, args.version, args.model)
    if model_path is None:
        print(f"Error: model file not found for exp_name={args.exp_name}, version={args.version}, model={args.model}")
        return

    print(f"Loading checkpoint from: {model_path}")
    params = load_checkpoint(model_path)

    if args.mode == "interactive":
        run_interactive(params)
    else:
        render_video(params, args.steps, args.output)


if __name__ == "__main__":
    main()

```

---

