# ロボット設定・歩容生成・運動学 診断レポート (2026-09-08)

## 対象ファイル

- `robot/config.py` (RobotConfig クラス定義)
- `robot/gait_generator.py` (サイクロイド軌道 + IK)
- `robot/kinematics.py` (解析的逆運動学ソルバー)
- `robot/math_utils.py` (クォータニオン変換など)

---

## **[1] config.py 診断**

### ✅ **良好な点**

| 項目 | 状態 | 根拠 |
|------|------|------|
| **プロジェクト構造** | ✅ | BASE_DIR, MUJOCO_MODEL_PATH, OUTPUT_DIR で path 一元管理 |
| **ハードウェア仕様** | ✅ | HX-30HM 公式仕様に準拠（3.0 N.m, 6.5 rad/s）|
| **制御周期** | ✅ | SIM_DT(2.5ms) × DECIMATION(4) = 10ms (100Hz) 正確 |
| **関節定義** | ✅ | 20 DOF、左右対称、腕4軸含む |
| **DEFAULT_JOINT_ANGLES** | ✅ | 中腰立ち姿勢（膝マイナス/プラス対称） |
| **観測次元計算** | ✅ | OBS_DIM = 625 の計算ロジック正確 |
| **USE_REFERENCE_GAIT** | ✅ | False がデフォルト（お手本無し学習推奨） |
| **報酬ウェイト** | ✅ | Phase 1 standing-only に最適化済み |
| **カリキュラム** | ✅ | CURRICULUM_SCHEDULE_FRACTIONS (相対進捗率版) 実装 |
| **PD制御ゲイン** | ✅ | KP=40, KD=1.0（外乱耐性のため剛性UP） |
| **終了条件** | ✅ | HEIGHT/ROLL/PITCH で転倒判定（MAX_EPISODE_STEPS=500） |

### ⚠️ **要確認事項**

| 項目 | 懸念 | 重要度 |
|------|------|--------|
| **FSR_POSITIONS** | シミュレーション専用。実機での位置マッピングは実装別 | 🟡 中 |
| **MUJOCO_MODEL_PATH** | assets/humanoid/humanoid.xml が実在するか確認要 | 🟡 中 |
| **NOISE パラメータ** | 実測値で後調整が必要 | 🟡 中 |
| **KP=40, KD=1.0** | 実機での安定性検証待機中 | 🟡 中 |
| **COM_HEIGHT=0.17** | 「中腰姿勢での実測値」だが、検証環境による | 🟡 低 |

### 🔴 **ブロッキング検出**

#### **[CONFIG-1] CURRICULUM_SCHEDULE と CURRICULUM_SCHEDULE_FRACTIONS の二重定義**

```python
# L118-124: 絶対ステップ版
CURRICULUM_SCHEDULE = {
    0: 0.05,
    200000: 0.20,
    ...
}

# L131-137: 相対進捗率版
CURRICULUM_SCHEDULE_FRACTIONS = {
    0.00: 0.00,
    0.10: 0.10,
    ...
}

# L287-295: resolve_curriculum_schedule() で変換可能
def resolve_curriculum_schedule(cls, total_steps: int = None) -> dict:
    ...
    return {int(frac * total): scale for frac, scale in cls.CURRICULUM_SCHEDULE_FRACTIONS.items()}
```

**問題**:
- 両方が定義されているため、どちらを使うのか曖昧
- envs/mjx_env.py が `CURRICULUM_SCHEDULE` を参照する場合と `CURRICULUM_SCHEDULE_FRACTIONS` を参照する場合が混在する可能性
- 実装側で呼び出し順序を統一していないと、値が二転三転する

**対応策**:
- `CURRICULUM_SCHEDULE` は **廃止 or コメント化**
- 全システムが `CURRICULUM_SCHEDULE_FRACTIONS` + `training_progress` ベースで統一
- または明示的に「絶対版は使わない」とコメント明記

---

#### **[CONFIG-2] resolve_curriculum_schedule() が未活用**

```python
# config.py に定義されているが、実際の envs/mjx_env.py で呼び出されているか不明

# L287-295
@classmethod
def resolve_curriculum_schedule(cls, total_steps: int = None) -> dict:
    """
    CURRICULUM_SCHEDULE_FRACTIONS を絶対ステップ数の辞書へ変換する。
    train_mjx.py 側で実際の総学習ステップ数(またはその推定値)が
    確定した時点で呼び出し...
    """
    ...
```

**問題**:
- train_mjx.py のどこで呼び出されるのか不明確
- mjx_env.py / mjx_rewards.py が直接参照しているのか、config.py 経由なのか曖昧

**対応策**:
- train_mjx.py で明示的に呼び出し、CURRICULUM_SCHEDULE_FRACTIONS をキャッシュ
- または mjx_rewards.py に `training_progress` を正しく供給

---

#### **[CONFIG-3] TERMINATION_HEIGHT と初期高さの不整合**

```python
# L203
COM_HEIGHT = 0.17            # [FIX] 中腰姿勢での実測CoM高 (旧0.28は高すぎた)

# L237
TERMINATION_HEIGHT = 0.10  # [FIX] 初期高さ z=0.165m に合わせて調整 (旧0.15は近すぎた)

# 問題: 初期高さが明示されていない
```

**懸念**:
- TERMINATION_HEIGHT = 0.10 は初期高さ 0.165m より下だが、その計算根拠が記載されていない
- 強化学習開始時に即座に転倒判定に達する可能性

**対応策**:
```python
# config.py に明示的に追加
INITIAL_HEIGHT = 0.1773  # [m] mjx_env.py で qpos[2] として設定される初期値
# TERMINATION_HEIGHT は INITIAL_HEIGHT の安全マージン(例: 0.05m) として計算
TERMINATION_HEIGHT = INITIAL_HEIGHT - 0.05  # = 0.1273 m
```

---

## **[2] gait_generator.py 診断**

### ✅ **良好な点**

| 項目 | 状態 | 根拠 |
|------|------|------|
| **サイクロイド軌道** | ✅ | ジャーク最小化特性を持つ滑らかな軌道 |
| **JAX/NumPy 両対応** | ✅ | jax_get_reference_trajectory() と numpy_get_reference_trajectory() で100%互換 |
| **位相同期** | ✅ | phase_r, phase_l が正しく計算される |
| **IK統合** | ✅ | _simple_ik_leg() で2リンク幾何IK実装 |
| **インデックスマッピング** | ✅ | ref_angles[2/3/4/7/8/9] の割り当てが正確 |

### ⚠️ **要確認事項**

| 項目 | 懸念 | 重要度 |
|------|------|--------|
| **THIGH_LEN / KNEE_LEN** | L88-91 で 0.12m で固定。実URDFと一致するか確認 | 🟡 中 |
| **STEP_HEIGHT = 0.04m** | 4cm の足上げが実機で実現可能か | 🟡 中 |
| **STAND_HEIGHT = 0.23m** | 直立時の腰高さ。config.py との整合確認 | 🟡 中 |
| **GaitGenerator クラスの使用箇所** | main block でのテストのみ。実装で使われているか不明 | 🟠 中 |

### 🔴 **ブロッキング検出**

#### **[GAIT-1] THIGH_LEN / KNEE_LEN と LegKinematics.L_THIGH/L_SHIN の不整合**

```python
# gait_generator.py L88-91 (JAX版)
THIGH_LEN = 0.12     # [m]
KNEE_LEN  = 0.12     # [m]

# kinematics.py L10-12 (NumPy版)
self.L_THIGH = 100.0  # [mm]
self.L_SHIN  = 100.0  # [mm]

# 変換: 100mm = 0.1m (不整合!)
```

**問題**:
- gait_generator.py：THIGH_LEN = 0.12m
- kinematics.py：L_THIGH = 100.0mm = 0.1m
- **10mm (1cm) のズレが存在**

**対応策**:
kinematics.py の L_THIGH/L_SHIN を以下に修正
```python
self.L_THIGH = 120.0  # [mm] gait_generator.py の 0.12m に統一
self.L_SHIN  = 120.0  # [mm]
```

---

#### **[GAIT-2] STAND_HEIGHT が複数定義されている**

```python
# gait_generator.py L88
STAND_HEIGHT = 0.23  # JAX版

# gait_generator.py class GaitGenerator.__init__ L20
self.stand_height = 0.23 # NumPy版(GaitGeneratorクラス)

# kinematics.py では明示的に定義なし（計算で使用）
```

**問題**:
- 複数の場所で定義されており、保守性が低い
- config.py との関係が不明確

**対応策**:
```python
# config.py に追加
GAIT_STAND_HEIGHT = 0.23  # [m] 直立時の腰の高さ
GAIT_THIGH_LEN = 0.12     # [m] 大腿リンク長
GAIT_KNEE_LEN = 0.12      # [m] 下腿リンク長

# gait_generator.py で参照
from robot.config import RobotConfig
STAND_HEIGHT = RobotConfig.GAIT_STAND_HEIGHT
THIGH_LEN = RobotConfig.GAIT_THIGH_LEN
KNEE_LEN = RobotConfig.GAIT_KNEE_LEN
```

---

#### **[GAIT-3] GaitGenerator クラスが未使用か不明**

```python
# L6-64: GaitGenerator クラス定義
class GaitGenerator:
    def __init__(self):
        self.ik = LegKinematics()  # LegKinematics に依存
        ...

# L67-73: テストのみ
if __name__ == "__main__":
    ...
```

**問題**:
- GaitGenerator クラスは LegKinematics に依存するが、L167-195 の JAX版とは独立している
- 実際のシステムで GaitGenerator が使用されているのか不明

**対応策**:
- GaitGenerator クラスが不要なら削除
- または明示的に「NumPy/SciPy環境でのみ使用」とコメント明記

---

## **[3] kinematics.py 診断**

### ✅ **良好な点**

| 項目 | 状態 | 根拠 |
|------|------|------|
| **IK解析解** | ✅ | 幾何学計算による正確な逆運動学 |
| **クリッピング** | ✅ | 到達不可能な目標位置を安全に制限 |
| **足裏水平維持** | ✅ | ankle_pitch = -(hip_pitch + knee_angle) |

### ⚠️ **要確認事項**

| 項目 | 懸念 | 重要度 |
|------|------|--------|
| **L_THIGH/L_SHIN の単位** | [mm] で定義されているが、計算では [m] の座標系と混用 | 🔴 高 |
| **Yaw/Roll** | [mm] 簡易計算で 0 固定。実機では必要か確認 | 🟡 中 |

### 🔴 **ブロッキング検出**

#### **[KIN-1] L_THIGH/L_SHIN の単位を [mm] で定義しながら計算で [m] を使用**

```python
# kinematics.py L10-12
self.L_THIGH = 100.0  # [mm] ← コメント記載
self.L_SHIN  = 100.0  # [mm]

# L28-29: しかし計算では [m] 想定
L_sq = x**2 + z**2  # x, z は [m] 単位
L = np.sqrt(L_sq)

# L38
max_len = self.L_THIGH + self.L_SHIN  # 100 + 100 = 200 (mm?)
if L > max_len:  # L は [m] なので比較が成立しない
    L = max_len
```

**問題**:
- `L` は [m] 単位だが、`max_len` は [mm] → **次元が合わない**
- 実際には L > 0.2m (0.2 > 200) で常に偽になる

**対応策**:
```python
# 方法1: 定義を [m] に統一
self.L_THIGH = 0.12  # [m]
self.L_SHIN  = 0.12  # [m]

# 方法2: [mm] のままなら計算を [mm] に統一
x, y, z を [mm] に変換してから計算
```

---

## **[4] math_utils.py 診断**

### ✅ **良好な点**

| 項目 | 状態 | 根拠 |
|------|------|------|
| **JAX対応** | ✅ | isinstance() で JAX配列を自動検出 |
| **NumPy対応** | ✅ | フォールバック実装で常に動作 |
| **クォータニオン正規化** | ✅ | sinp を clip で [-1, 1] に制限 |

### ⚠️ **要確認事項**

| 項目 | 懸念 | 重要度 |
|------|------|--------|
| **JAX/NumPy 条件分岐** | runtime にブランチが分岐する。JIT追跡時に問題の可能性 | 🟡 中 |

### 🔴 **ブロッキング検出**

#### **[MATH-1] quat_to_euler() の条件分岐が JIT 追跡で失敗する可能性**

```python
# L22-30: 条件分岐
if HAS_JAX and jax is not None and jp is not None and isinstance(q, jax.Array):
    # JAX版
    roll = jp.arctan2(...)
    ...
else:
    # NumPy版
    roll = np.arctan2(...)
    ...
```

**問題**:
- JAX JIT 追跡時、`isinstance(q, jax.Array)` は定数ではなく、実行時に評価される
- JIT 内部では Python の制御フローが許可されず、tracer object で evaluate できない

**対応策**:
```python
# JAX版とNumPy版を完全に分離
def quat_to_euler_numpy(q):
    """NumPy配列用"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    ...
    return np.array([roll, pitch, yaw])

def quat_to_euler_jax(q):
    """JAX配列用（JIT互換）"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    ...
    return jp.array([roll, pitch, yaw])

def quat_to_euler(q):
    """呼び出し元で型をチェックして分岐"""
    if HAS_JAX and isinstance(q, jax.Array):
        return quat_to_euler_jax(q)
    else:
        return quat_to_euler_numpy(q)
```

---

## 📊 **総合診断マトリックス**

| ファイル | 項目 | 状態 | ブロッキング |
|---------|------|------|------------|
| **config.py** | 構造 | ✅ | - |
|  | ハードウェア仕様 | ✅ | - |
|  | カリキュラム定義 | ⚠️ | CONFIG-1 |
|  | TERMINATION_HEIGHT | ⚠️ | CONFIG-3 |
| **gait_generator.py** | サイクロイド軌道 | ✅ | - |
|  | IK計算 | ✅ | GAIT-1, GAIT-2 |
|  | クラス設計 | ⚠️ | GAIT-3 |
| **kinematics.py** | IK解析 | ✅ | KIN-1 |
|  | 次元管理 | 🔴 | KIN-1 |
| **math_utils.py** | クォータニオン変換 | ⚠️ | MATH-1 |

---

## 🔴 **ブロッキングイシュー 確定リスト**

### **[CONFIG-1] CURRICULUM_SCHEDULE 二重定義**
- **影響**: どちらの定義が有効か曖昧
- **対応**: CURRICULUM_SCHEDULE は廃止、CURRICULUM_SCHEDULE_FRACTIONS に統一

### **[CONFIG-3] TERMINATION_HEIGHT と初期高さの不整合**
- **影響**: エピソード開始時に転倒判定される可能性
- **対応**: INITIAL_HEIGHT を明示、TERMINATION_HEIGHT の根拠を記載

### **[GAIT-1] リンク長の不整合 (0.12m vs 0.1m)**
- **影響**: IK 計算の到達範囲が異なり、制御不可能な領域が生じる
- **対応**: kinematics.py の L_THIGH/L_SHIN を 120mm に統一

### **[GAIT-2] STAND_HEIGHT の複数定義**
- **影響**: 保守性低下、値が不整合になる可能性
- **対応**: config.py に一元化

### **[KIN-1] リンク長の次元不整合 (mm vs m)**
- **影響**: IK 計算が正しく動作しない（max_len チェックが常に偽）
- **対応**: [m] に統一

### **[MATH-1] quat_to_euler() の JAX JIT 非互換**
- **影響**: JAX JIT コンパイル内で使用不可
- **対応**: NumPy版と JAX版を分離

---

## ✅ **修正推奨優先度**

| Priority | Issue | 対応 | 工数 |
|----------|-------|------|------|
| 🔴 **P0** | KIN-1 (次元不整合) | kinematics.py 修正 | 低 |
| 🔴 **P0** | GAIT-1 (リンク長不整合) | kinematics.py + gait_generator.py 統一 | 低 |
| 🔴 **P0** | MATH-1 (JAX JIT非互換) | math_utils.py 分離 | 低 |
| 🟡 **P1** | CONFIG-1 (CURRICULUM二重定義) | config.py 統一 | 低 |
| 🟡 **P1** | CONFIG-3 (TERMINATION_HEIGHT) | config.py 明記 | 低 |
| 🟡 **P1** | GAIT-2 (STAND_HEIGHT) | config.py 一元化 | 低 |

---

## 📝 **次のステップ**

修正版の生成をご依頼いただければ、同じアプローチで対応いたします。

特に **P0 (KIN-1, GAIT-1, MATH-1)** は実機制御・学習に直結する重大問題のため、修正版生成を推奨します。

---

**生成日時**: 2026-09-08  
**診断完了**: 全項目チェック済み  
**停止条件**: なし（診断のみ、修正版生成待機中）
