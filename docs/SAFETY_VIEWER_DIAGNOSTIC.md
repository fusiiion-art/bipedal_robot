# 安全層・ビューアー診断レポート (2026-09-08)

## 対象ファイル

- `safety/cbf.py` (Control Barrier Function 安全層)
- `scripts/viewer.py` (MuJoCoビューアースクリプト)

---

## **[1] safety/cbf.py 診断**

### ✅ **良好な点**

| 項目 | 状態 | 根拠 |
|------|------|------|
| **JAX互換性** | ✅ | jp.clip(), jax.nn.softplus() で vmap/jit 安全 |
| **設計意図** | ✅ | 学習時の簡易版 + 実機向けの注釈明確 |
| **ペナルティ計算** | ✅ | softplus で微分可能な罰則項 |
| **アーキテクチャ** | ✅ | QP求解を避けて計算効率重視 |

### ⚠️ **要確認事項**

| 項目 | 懸念 | 重要度 |
|------|------|--------|
| **joint_pos_margin = 0.1 rad** | ハードコード値。実URDFの可動域に合わせて要調整 | 🟡 中 |
| **limit_lower/upper の入力形式** | actuator_ctrlrange の形式と一致するか確認 | 🟡 中 |
| **ペナルティ係数 k=10.0** | 報酬スケールとの関係が不明確 | 🟡 中 |

### 🔴 **ブロッキング検出**

#### **[CBF-1] joint_pos_margin が静的で、可動域の比率に基づいていない**

```python
# 現実装
self.joint_pos_margin = 0.1  # [rad] 全関節共通

# 問題
# - 股関節: 可動域 ±π (約±3.14rad) → 0.1rad は 3% のみ
# - 膝関節: 可動域 ±0.52rad → 0.1rad は 19% で大きすぎる
# - 足首: 可動域 ±0.44rad → 0.1rad は 23%
```

**理由**:
- 各関節の可動域が異なるのに、マージンが固定
- 狭い可動域の関節ほど、安全マージンが相対的に大きくなり、不必要に制限される
- 逆に広い可動域の関節は、実質的な安全マージンが小さい

**対応策**:
```python
# 可動域の比率ベースで計算
def compute_safe_margins(self, limit_lower, limit_upper):
    ranges = limit_upper - limit_lower
    # 各可動域の 5% をマージンとする（例）
    margin_ratio = 0.05
    return margin_ratio * ranges
```

---

#### **[CBF-2] filter_action() と compute_cbf_penalty() が異なる判定基準を持つ**

```python
# filter_action(): limit ± margin でクランプ
safe_lower = limit_lower + self.joint_pos_margin
safe_upper = limit_upper - self.joint_pos_margin
safe_action = jp.clip(nominal_action, safe_lower, safe_upper)

# compute_cbf_penalty(): softplus で「距離」に基づくペナルティ
upper_violation = softplus(k * (nominal_action - safe_upper))
lower_violation = softplus(k * (safe_lower - nominal_action))
```

**問題**:
- filter_action() で安全に制限されたアクションが、compute_cbf_penalty() で再度「違反」と判定される
- 報酬函数で大きなペナルティが加えられ、学習シグナルが矛盾する

**対応策**:
```python
# 統一: クランプ後のペナルティは 0
def compute_cbf_penalty(self, safe_action, nominal_action):
    # 実際にクランプされた量のペナルティ
    clamp_diff = jp.sum(jp.abs(nominal_action - safe_action))
    return clamp_diff * penalty_scale
```

---

#### **[CBF-3] mjx_env.py との連携で double-clamp の可能性**

```python
# mjx_env.py L161 (real_env.py もほぼ同じ)
target = default_pose + action * ACTION_SCALE
target = np.clip(target, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)  # ← Clamp 1

# mjx_env.py L165 (CBF内)
safe_target_rad = self.cbf.filter_action(target, ...)  # ← Clamp 2
```

**問題**:
- 既に JOINT_LIMITS でクランプされたアクションを、CBF が再度クランプする
- 効果は重複し、計算コスト無駄、ペナルティ計算が複雑化

**対応策**:
```python
# 実装戦略を明確に:
# 戦略A: RL側で粗いクランプ + CBF で微調整
# 戦略B: CBF 単独でクランプ（RL側では action を直接使用）

# 推奨: 戦略A（学習安定性のため）
# ただし、CBF は「オーバーシュート検出」に特化
```

---

### ⚠️ **その他の懸念**

| 項目 | 懸念 | 対応 |
|------|------|------|
| **実機への備考** | cbf_realworld.py への言及のみ。実装は？ | 別ファイル計画でOK |
| **トルク制約** | MOTOR_MAX_TORQUE は使用されていない | 拡張の余地あり |
| **ジョイント速度制約** | max_vel は保持されているが使用されていない | 拡張対象 |

---

## **[2] scripts/viewer.py 診断**

### ✅ **良好な点**

| 項目 | 状態 | 根拠 |
|------|------|------|
| **シンプル設計** | ✅ | 可視化用途に特化、不要な複雑性なし |
| **エラーハンドリング** | ✅ | モデルファイル欠損時の親切なメッセージ |
| **ユーザーガイド** | ✅ | キー操作などの説明が親切 |
| **物理ループ** | ✅ | タイムステップに同期したシミュレーション |

### ⚠️ **要確認事項**

| 項目 | 懸念 | 重要度 |
|------|------|--------|
| **model_path ハードコード** | 'assets/humanoid/humanoid_visualize.xml' 固定 | 🟡 中 |
| **衝突ジオメトリ表示** | プリント出力のみ。色分けは見えない | 🟡 中 |
| **ビューアーコンフィグ** | cam.azimuth/elevation/distance が固定 | 🟡 低 |

### 🔴 **ブロッキング検出**

#### **[VIEWER-1] humanoid_visualize.xml の生成スクリプトが明記されていない**

```python
model_path = 'assets/humanoid/humanoid_visualize.xml'

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please run: python scripts/add_collision_colors.py")  # ← この script はどこに？
    sys.exit(1)
```

**問題**:
- `add_collision_colors.py` の有無が不明
- スクリプトが存在しない場合、ユーザーが困ってしまう

**対応策**:
```python
# viewer.py 内で自動生成 or 別スクリプトのパスを明示
# または humanoid.xml をそのまま使用

model_path = os.environ.get('MUJOCO_MODEL_PATH', 'assets/humanoid/humanoid.xml')
```

---

#### **[VIEWER-2] 衝突ジオメトリの色分けが機能していない**

```python
# プリント出力はしているが、MuJoCo Viewer の表示に反映されない
for i in range(model.ngeom):
    name_str = model.geom(i).name
    if 'collision' in name_str:
        rgba = model.geom_rgba[i]
        print(f"      rgba=[{rgba[0]:.2f}, ...]")  # ← 表示されるだけ
```

**問題**:
- ユーザーが視覚的に衝突ジオメトリを確認できない
- 「ジオメトリが見える」という名目だが、実質的に見えていない

**対応策**:
```python
# XML生成時に衝突ジオメトリに色を設定
# または viewer.py で rgba を動的に更新
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_COLLISION] = 1
```

---

#### **[VIEWER-3] パス管理が相対パス依存で、実行位置に依存**

```python
model_path = 'assets/humanoid/humanoid_visualize.xml'
# 実行位置がプロジェクト直下でない場合、ファイルが見つからない
```

**問題**:
- `python scripts/viewer.py` で実行した場合、`assets/` が見つからない可能性
- `python -m scripts.viewer` や別の実行方法では失敗

**対応策**:
```python
import os
from pathlib import Path

# スクリプトの位置からの相対パス
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
model_path = project_root / 'assets' / 'humanoid' / 'humanoid.xml'
```

---

## 📊 **総合診断マトリックス**

| ファイル | 項目 | 状態 | ブロッキング |
|---------|------|------|------------|
| **safety/cbf.py** | JAX互換性 | ✅ | - |
|  | 設計意図 | ✅ | - |
|  | margin計算 | ⚠️ | CBF-1 |
|  | double-clamp | ⚠️ | CBF-3 |
|  | ペナルティ一貫性 | ⚠️ | CBF-2 |
| **scripts/viewer.py** | シンプル性 | ✅ | - |
|  | エラーハンドリング | ✅ | - |
|  | 衝突表示 | ❌ | VIEWER-2 |
|  | パス管理 | ⚠️ | VIEWER-3 |
|  | 生成スクリプト | ❌ | VIEWER-1 |

---

## 🔴 **ブロッキングイシュー 確定リスト**

### **[CBF-1] joint_pos_margin が静的、可動域比率に基づかない**
- **影響**: 狭い可動域の関節が過度に制限される
- **対応**: 可動域に応じた動的マージン計算

### **[CBF-2] filter_action() と compute_cbf_penalty() の基準が異なる**
- **影響**: 学習シグナルが矛盾（同じアクションで違反と判定）
- **対応**: ペナルティ計算を統一

### **[CBF-3] mjx_env.py と double-clamp**
- **影響**: 計算コスト無駄、ペナルティ計算の複雑化
- **対応**: 実装戦略を明確化（粗クランプ + 微調整 or CBF単独）

### **[VIEWER-1] add_collision_colors.py の有無が不明**
- **影響**: ユーザーがスクリプト実行手順がわからない
- **対応**: パス管理を改善、生成スクリプトを明示

### **[VIEWER-2] 衝突ジオメトリの色分けが機能していない**
- **影響**: 視覚化の目的が達成されていない
- **対応**: XML生成 or Viewer設定で色を有効化

### **[VIEWER-3] 相対パス依存で実行位置に依存**
- **影響**: 異なる実行位置では失敗する
- **対応**: 絶対パス or プロジェクト相対パスに修正

---

## ✅ **修正推奨優先度**

| Priority | Issue | 対応 | 工数 |
|----------|-------|------|------|
| 🔴 **P0** | CBF-1 (margin 計算) | config 連携 | 低 |
| 🔴 **P0** | CBF-2 (ペナルティ一貫性) | 論理統一 | 低 |
| 🟡 **P1** | CBF-3 (double-clamp) | 戦略明確化 | 低 |
| 🟡 **P1** | VIEWER-1 (スクリプト生成) | パス管理 | 低 |
| 🟡 **P1** | VIEWER-2 (色分け表示) | XML/Viewer設定 | 中 |
| 🟡 **P1** | VIEWER-3 (相対パス) | パス修正 | 低 |

---

## 📝 **次のステップ**

修正版の生成をご依頼いただければ、同じアプローチで対応いたします。

特に **P0 (CBF-1, CBF-2)** は学習のシグナル品質に影響するため、修正版生成を推奨します。

---

**生成日時**: 2026-09-08  
**診断完了**: 全項目チェック済み  
**停止条件**: なし（診断のみ、修正版生成待機中）
