# 強化学習・報酬関数 ＆ 安定性指標バグ修正・改良最終仕様書

本ドキュメントは、二足歩行ロボット『旋風丸 (Senpumaru)』の強化学習環境における重要バグの特定・修正結果、および最新の報酬関数・安定性指標の統合仕様をまとめたマスタードキュメントです。

---

## 🛠 特定・解消された5大🔴致命的バグ

| バグ項目 | 修正前の不具合・症状 | 修正後の実装・挙動 |
|---|---|---|
| **1. `global_step` の到達不能** | `info['global_step']` が `reset()` ごとに `0` に初期化され、`CURRICULUM_SCHEDULE` (10万ステップ〜) に到達せず外乱が常に最弱 (0.5N) で固定されていた。 | `train_progress` (0.0〜1.0) および累積ステップの伝搬を正しく接続し、カリキュラムが 200万ステップにかけて正常スケーリングされるよう修正。 |
| **2. ZMP margin の自己相殺** | `zmp_error = ‖zmp - pressure_center‖` において `zmp = pressure_center - correction` と定義されていたため代数的に中和され、`zmp_margin` が常に `1.0` 固定となっていた。 | LIPM公式 $ZMP_{xy} = p_{com,xy} - \frac{a_{com,xy}\cdot h}{a_{com,z}+g}$ を用い、足裏支持多角形からの符号付き正当距離として計算。 |
| **3. Capture Point 勾配消失** | `exp(-10d²) * margin` の乗算により、CPが足裏支持基底(7〜10cm)を出た瞬間に勾配が `0` へ完全消失していた。 | 2段階評価 ($0.6e^{-2.5d} + 0.4e^{-15d^2}$) へ変更し、大外乱時 ($d=0.5\text{m}$) でも $r_{cp} \approx 0.18$ の学習勾配を保持。 |
| **4. 復帰ボーナス発火失敗** | `max_bonus_steps=2` (=20ms@100Hz) と短すぎ、数百msかかる踏み出し復帰動作でボーナスが発火しなかった。 | `max_bonus_steps=50` (=0.5秒) へ拡大し、実用的なステップ踏み出し動作で正しくボーナスが付与されるよう変更。 |
| **5. エピソード冒頭での全ペナルティ消失** | `penalty_scale = step/500` の `step` がエピソード内カウンタとなっていたため、毎エピソード冒頭5秒間ペナルティ（CBF含む）が消絶していた。 | 全体学習進行度 `training_progress` に同期させ、エピソード内での安全項失効を防止。 |

---

## 📐 最新の数理・理論仕様

### 1. $\lambda_{phase}(s)$ （反応空白期間の解消版）
外力印加の瞬間に即座にフェーズ遷移（$1.0 \to 1.9 \times 10^{-22}$）させるため、胴体傾き・角速度・水平速度ノルム・外乱フラグを統合：

$$z(s) = 5.0\cdot|\theta_{err}| + 0.5\cdot\|\omega_{xy}\| + 1.5\cdot\|v_{xy}\| + 2.0\cdot\mathbb{1}[\text{was\_disturbed}]$$
$$\lambda_{phase} = \exp\left(-10.0\cdot\max(0,\, z - 0.3)\right)$$

### 2. 位相ゲート適用 PBRS (Ng et al. 1999 適合)
外乱リカバリー時に目標位置追従圧力と踏み出し動作が衝突（Stiffening）するのを防ぐため、$p_{target}(s)$ に $\lambda_{phase}(s)$ を結合：

$$\Phi(s) = w_{up}\cdot p_{upright}(s) + w_{tgt}\cdot p_{target}(s)\cdot\lambda_{phase}(s)$$

※ $\lambda_{phase}(s)$ は状態 $s$ のみの関数のため、Ng et al. (1999) のポリシー不変性は厳密に維持されます。

### 3. Log-Barrier（対数バリア）ソフト制約
- **胴体最低高度バリア** ($h < 0.35\text{m}$ で対数罰則)
- **関節トルクバリア** (定格 85% 超過で対数罰則)

$$B_{torque}(\tau) = -\log\left(1 - \text{clip}\left(\frac{\max|\tau|/\tau_{max} - 0.85}{0.14}, 0, 0.99\right)\right)$$
