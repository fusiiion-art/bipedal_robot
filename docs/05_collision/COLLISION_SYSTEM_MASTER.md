# 当たり判定（衝突検出）・足圧・可視化 統合仕様書 (COLLISION_SYSTEM_MASTER)

本ドキュメントは、二足歩行ロボット『旋風丸』のシミュレーション環境における **剛体衝突検出 (Geom Collision)**、**足圧・ZMP測定 (FSR/Contact Force)**、**衝突統計 (Statistics)**、および **衝突幾何体の可視化・デバッグ手法 (Visualization)** を完全統合したマスター仕様書です。

---

## 1. システム概要 ＆ クイックガイド

### 1.1 全体アーキテクチャ
```
┌────────────────────────────────────────────────────────┐
│               MuJoCo シミュレーション環境 (500Hz / 2ms)  │
├────────────────────────────────────────────────────────┤
│ 1. 剛体衝突検出 (Geom Collision) : ~31個の幾何体        │
│ 2. 足圧・接触検出 (8ch FSR & Contact Force)            │
│ 3. 複合安定性指標計算 (ZMP + Capture Point + 姿勢)     │
│ 4. 報酬関数への統合 (Reward Shaping & Penalties)       │
└────────────────────────────────────────────────────────┘
```

### 1.2 3行要約
1. **衝突検出**: 約30個のPrimitive Geom（box, sphere, capsule）を用いて、高速かつ正確なペアワイズ衝突判定を実行。
2. **足圧検出**: 左右脚に配置された計8チャンネルのFSRセンサーで接地圧力を測定し、ZMP（Zero Moment Point）およびCoP（圧力中心）をリアルタイム算定。
3. **可視化・検証**: 表示専用ビジュアルメッシュ (`contype=0`) と学習用衝突Geom (`contype=1`) を完全分離し、デバッグ時は衝突Geomを半透明表示して幾何形状を直感的に検証可能。

---

## 2. MuJoCo 衝突検出・幾何体設定

### 2.1 XMLパラメータ仕様 (humanoid.xml)
```xml
<default>
    <!-- 全関節の粘性抵抗・摩擦・慣性を定義 -->
    <joint limited="true" damping="0.5" frictionloss="0.05" armature="0.01" />
    
    <!-- 衝突用geomのデフォルトパラメータ -->
    <geom contype="1" conaffinity="1" condim="3" friction="1.5 0.005 0.0001" />
</default>
```

| パラメータ | 設定値 | 意味・役割 |
|---|---|---|
| `contype` | `1` | このgeomは衝突ペアの第1グループに所属 |
| `conaffinity` | `1` | contype=1 との衝突判定を有効化 |
| `condim` | `3` | 3D接触（法線力 ＋ 2軸の接地摩擦力） |
| `friction` | `[1.5, 0.005, 0.0001]` | `[すべり摩擦係数, 転がり摩擦係数, スピン摩擦係数]` |

### 2.2 ビジュアルメッシュ と 衝突Geom の分離構造
メッシュ複雑度によるシミュレーションの低速化を防ぐため、**視覚用（表示のみ）**と**物理用（計算用）**の2種類のgeomを定義しています。

```xml
<!-- 視覚表示用メッシュ (contype=0, conaffinity=0) -->
<geom name="doutai-v5_doutai_geom" type="mesh" mesh="doutai-v5_doutai"
      rgba="0.2 0.6 1.0 1" contype="0" conaffinity="0" group="1"/>

<!-- 物理計算用衝突Geom (Primitive Box, contype=1) -->
<geom name="doutai-v5_doutai_collision" type="box" size="0.06 0.08 0.1" mass="0.6915"/>
```

---

## 3. 衝突幾何体 ＆ 衝突ペア統計

### 3.1 Geom（幾何体）構成数サマリー
```
胴体・頭部:
  ├─ 胴体 (Torso)   : Box × 1
  └─ 頭部 (Head)    : Sphere × 1

腕部 (左右計8関節):
  ├─ 肩 (Shoulder)  : Sphere × 2
  ├─ 上腕 (Upper)   : Capsule × 2
  ├─ 肘 (Elbow)     : Capsule × 2
  └─ 手 (Hand)      : Sphere × 2

脚部 (左右計12関節):
  ├─ 股関節 (Hip)   : Sphere × 4
  ├─ 太もも (Thigh) : Capsule × 2
  ├─ 膝 (Knee)      : Capsule × 2
  ├─ 足首 (Ankle)   : Sphere × 2
  └─ 足裏 (Foot)    : Box × 2 + FSR 8ch

計: 30 個のPrimitive Geom + 床面 (Plane Geom) = 31 衝突幾何体
```

### 3.2 アクティブ衝突ペア
- **胴体 ↔ 脚**: 4 ペア (自他接触ガード)
- **胴体 ↔ 腕**: 4 ペア
- **脚（足裏Box） ↔ 床面**: 2 ペア (接地判定の最重要ペア)
- **腕・その他 ↔ 床面**: 8 ペア (転倒・手つき判定用)
- **脚 ↔ 脚 / 腕 ↔ 腕**: 0 ペア (干渉回避設定)

---

## 4. 足圧・接触力測定 ＆ ZMP算定メカニズム

### 4.1 FSR 8チャンネル配分
- **左足底**: 四隅 (Front-Left, Front-Right, Rear-Left, Rear-Right) 計 4ch
- **右足底**: 四隅 (Front-Left, Front-Right, Rear-Left, Rear-Right) 計 4ch

### 4.2 安定性指標の算定 (Foot Placement & ZMP)
1. **CoP (圧力中心) 近似**: 4点のFSR荷重 $\sum w_i p_i / \sum w_i$ により足裏ローカル受圧中心を算出。
2. **ZMP (Zero Moment Point)**: 接触力ベクトル $F_{contact}$ と床面モーメントからダイナミックバランス点を同定。
3. **Capture Point (CP)**: LIPMモデル $p_{cp} = p_{com} + v_{com} / \omega_0$ により踏み出し最適着地点を計算。

---

## 5. 衝突ジオメトリの可視化 ＆ デバッグ手法

### 5.1 XMLでの半透明表示設定 (`rgba`)
衝突Geomに半透明カラー（`rgba="1 0 0 0.5"` など）を指定することで、MuJoCo Viewer上でモデル内部の物理当たり判定領域を表示できます。

```xml
<geom name="doutai-v5_doutai_collision" type="box" size="0.06 0.08 0.1" 
      rgba="1 0 0 0.5" mass="0.6915"/>
```

### 5.2 MuJoCo Viewer でのトグル操作
- `V` キー: 幾何体（Geom）の表示・非表示切替
- `C` キー: 接触点 (Contact Points) 及び接触力ベクトルの可視化
- `F` キー: 接触力 (Contact Forces) の矢印描画

### 5.3 Pythonスクリプト経由での動的可視化
```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/all/all.xml")
data = mujoco.MjData(model)

# ビジュアルメッシュを非表示にし、衝突Geomのみを表示設定
opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```
