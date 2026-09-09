# 次ステップ アクションプラン

**作成日**: 2026-09-09  
**診断状況**: Phase 0 のコード確認は一部完了。Phase 0 は未合格  
**次フェーズ**: D-1〜D-6 の計測・原因切り分け後、GPU Debug PASS

---

## 📋 実施順序

`docs/master_plan.md` の Task D/E と実装済みCLIを正本とする。既存の `KL=232` と `episode_alive` 低下が未解決のため、checkpoint生成だけでGate 0合格とは判定しない。報酬とPPO設定を同時に変更しない。

### ステップ 0: GPU学習前の計測準備

1. D-1: `--target_kl` がBraxのAdaptive KL学習率制御に接続されていることを確認する。epoch内early stoppingではない。
2. D-2〜D-5: epoch/minibatch KL、deterministic/stochastic評価、終了理由、報酬内訳の計測を準備する。
3. D-6: 現行設定（`min_std=0.05`、`max_std=3.0`）を変更せず、1回計測する。

計測結果に応じて、次の変更カテゴリをPPO最適化系または報酬系のどちらか一つだけ選ぶ。

### ステップ 1: 計測用GPU Debug run（GPU/WSL 必須）

#### 1-1. WSL2 + JAX CUDA 環境確認

```bash
# WSL2 内で実行
python -c "import jax; print(jax.devices())"

# 出力例:
# [cuda(id=0)]  (GPU が正しく認識されている場合)
```

**もし CPU と表示される場合**:
- WSL2 + CUDA の再インストールが必要
- 一時的に CPU で小規模学習テストは可能（時間がかかる）

#### 1-2. 学習実行コマンド

```bash
cd /mnt/c/bipedal_robot

# 実験名を付け、既存checkpointを上書きしない
python train/train_mjx.py \
  --exp_name phase0_debug_20260909_seed42 \
  --seed=42 \
  --target_kl=0.02

# オプション: 異なるシード（複数シードで安定性確認）
python train/train_mjx.py --exp_name phase0_qual_seed0 --seed=0 --target_kl=0.02
python train/train_mjx.py --exp_name phase0_qual_seed1 --seed=1 --target_kl=0.02
python train/train_mjx.py --exp_name phase0_qual_seed2 --seed=2 --target_kl=0.02
```

#### 1-3. 学習パラメータ（自動設定）

| パラメータ | CPU 設定 | GPU 設定 | 用途 |
|---|---|---|---|
| num_envs | 32 | 256 | 並列環境数 |
| steps | 100k | 10M | 総学習ステップ |
| episode_length | 50 | 500 | エピソード長 |
| batch_size | 32 | 256 | バッチサイズ |
| learning_rate | 1e-4 | 1e-4 | 学習率 |
| unroll_length | 10 | 10 | PPO アンロール長 |

#### 1-4. 学習時間の目安

| 環境 | 学習時間 | 備考 |
|---|---|---|
| **CPU** | フル学習には使用しない | 単体テスト・形状確認のみ |
| **GPU (RTX 4060 8GB)** | 30～60 分 | 推奨（コンパイルキャッシュで高速化） |
| **GPU (RTX 4090 24GB)** | 10～15 分 | 最速（複数シード並列可能） |

#### 1-5. 出力ファイルの生成場所

```
log/
  <exp_name>/
    version_0/
    ├─ final_params.pkl      (最終モデル)
    ├─ best_params.pkl       (最高報酬モデル ← 評価用)
    ├─ worst_params.pkl      (最低報酬モデル)
    ├─ last_params.pkl       (直前のモデル)
    └─ log.json              (学習曲線ログ)
```

**checkpoint は `--exp_name`、`--version`、`--model` で指定します。**

---

### ステップ 2: Checkpoint 評価（GPU/WSL）

#### 2-1. 診断スクリプトで詳細分析

```bash
cd /mnt/c/bipedal_robot

python scratch/phase0_eval_diagnostics.py \
  --exp_name phase0_debug_20260909_seed42 \
  --version 0 \
  --model best_params.pkl \
  --episodes 100 \
  --fixed-episodes 20 \
  --force-levels 0 \
    --seed 42
```

#### 2-2. 出力ファイル

```
log/version_0/
  └─ eval_diagnostics_<timestamp>.json
      ├─ episode_alive 分布 (mean, std, min, max)
      ├─ kl_divergence (初期値と比較)
      ├─ termination_reasons (転倒, 時間切れ, recovery の割合)
      ├─ reward_breakdown (各成分の寄与度)
      └─ stability_metrics (トルク飽和率, 両足接地率 等)
```

#### 2-3. 期待される出力例

```json
{
  "episode_alive": {
    "mean": 480,
    "std": 15,
    "min": 420,
    "max": 500
  },
  "kl_divergence": 0.035,
  "termination_reasons": {
    "timeout": 0.85,
    "fallen": 0.12,
    "recovered": 0.03
  },
  "reward_breakdown": {
    "alive_reward": 45000,
    "upright_reward": 8500,
    "contact_reward": 2000
  }
}
```

---

### ステップ 3: 結果分析と判定

#### 3-1. Debug/Qualification判定基準

| 指標 | 目標値 | OK 判定 |
|---|---|---|
| **kl_mean** | 全区間 < 0.1、単発スパイク < 1.0 | ✅ PPO健全性 |
| **episode_alive** | 末尾20%平均 / 最高20%平均 >= 0.7 | ✅ 崩壊なし |
| **policy std** | min >= 0.05、max <= 3.0 | ✅ 分布範囲内 |
| **value_loss** | 発散スパイクなし | ✅ 安定 |
| **NaN/Inf** | なし | ✅ 学習継続可能 |

#### 3-2. 問題検出時の判定ツリー

```
KL = 232 (前回の値)?
  ├─ YES → 初期 learning rate が高い可能性
  │   └─ 対策: --target_kl=0.01 で re-training
  └─ NO → episode_alive の低下を確認

episode_alive = 77 (前回の値)?
  ├─ YES → Reward hacking の可能性
  │   └─ 対策: 報酬成分を分解して確認
  └─ NO → 改善 or 悪化を確認

転倒率 (fallen) が 20% 以上?
  ├─ YES → 外乱かバランス能力不足
  │   └─ 対策: 報酬重み調整
  └─ NO → 安定している
```

#### 3-3. 分析スクリプト（手動実行）

```python
# log/version_0/log.json を読み込んで分析
import json

with open("log/version_0/log.json") as f:
    logs = json.load(f)

# KL トレンド抽出
kls = [log.get("kl_divergence", 0) for log in logs]
print(f"KL trajectory: {kls[:10]} ... {kls[-10:]}")
print(f"KL max: {max(kls)}, mean: {sum(kls)/len(kls):.4f}")

# episode_alive トレンド抽出
alives = [log.get("episode_alive_mean", 0) for log in logs]
print(f"Alive trajectory: {alives[:10]} ... {alives[-10:]}")
```

---

### ステップ 4: KL スパイク・episode_alive 低下の原因特定

#### 4-1. 確認項目

```
□ KL スパイク（232）
  ├─ □ 初期 learning rate が高すぎるか？
  ├─ □ policy std の下限が低すぎるか？（現在 0.05）
  ├─ □ clipping_epsilon が小さすぎるか？（現在 0.2）
  └─ □ desired_kl の初期値は適切か？

□ episode_alive 低下（110 → 77）
  ├─ □ 報酬成分分解: 各component の推移
  ├─ □ 終了理由分析: 転倒増加 vs 時間切れ
  ├─ □ Reward hacking: 短時間で高報酬？
  └─ □ Deterministic evaluation で再評価
```

#### 4-2. Deterministic 評価（重要）

```python
# stochastic policy を mean-action で評価
python scratch/phase0_eval_diagnostics.py \
  --exp_name phase0_debug_20260909_seed42 \
  --version 0 \
  --model best_params.pkl \
  --episodes 50 \
  --fixed-episodes 50 \
  --force-levels 0 \
  --seed 42
```

**理由**: stochastic action による分散を除外して、pure policy quality を測定

---

### ステップ 5: Qualification評価（Phase 0判定）

#### 5-1. 複数シード評価

```bash
# 3 シード評価（再現性確認）
for seed in 0 1 2; do
  python train/train_mjx.py --exp_name=phase0_qual_seed${seed} --seed=$seed --target_kl=0.02
done

# 各seedのbest checkpointをphase0診断で評価する
python scratch/phase0_eval_diagnostics.py \
    --exp_name phase0_qual_seed0 --version 0 --model best_params.pkl \
    --episodes 200 --fixed-episodes 50 --force-levels 0 --seed 0
```

#### 5-2. Phase 0 Qualification判定

```
✅ Phase 0 Qualification PASS 条件（全seedで満たす）:
  1. kl_mean が全区間 0.1 未満、単発スパイクも 1.0 未満
  2. episode_alive の末尾20%/最高20%比率が 0.7 以上
  3. policy_dist_min_std >= 0.05、policy_dist_max_std <= 3.0
  4. value_loss に発散スパイクがない
  5. NaN/Inf がない

❌ Qualification FAIL → 原因を記録し、PPO最適化系または報酬系の
   どちらか一つだけを次のiterationで変更する。Gate 0/Gate Aには進まない。
```

---

## 🛠️ トラブルシューティング

### シナリオ A: KL が依然として高い（>0.1）

**原因候補**:
1. Learning rate が高い
2. Policy std の範囲が広すぎる
3. clipping_epsilon が小さすぎる

**対策**:
```bash
# target_kl をより厳しく
python train/train_mjx.py --seed=42 --target_kl=0.01

# または train_mjx.py を修正
# - clipping_epsilon: 0.2 → 0.3
# - max_grad_norm: 1.0 → 0.5
```

### シナリオ B: episode_alive が短すぎる（<300）

**原因候補**:
1. バランス能力不足
2. 報酬が小さすぎる
3. ペナルティが大きすぎる

**対策**:
```bash
# 報酬スケール確認
grep -n "reward_scaling\|alive_reward\|upright_reward" \
  train/train_mjx.py envs/mjx_rewards.py

# 修正例: reward_scaling を 0.01 から 0.02 に
# または mjx_rewards.py の alive_reward coefficient を増加
```

### シナリオ C: 転倒率が高い（>30%）

**原因候補**:
1. 制御が不安定
2. 関節速度制限が足りない
3. LPF カットオフが適切でない

**対策**:
```python
# envs/mjx_env.py で確認
# - Joint velocity limit
# - LPF time constant
# - CBF damping gain
```

---

## 📊 進捗追跡用チェックリスト

```
Phase 0 学習・評価チェックリスト
=====================================

□ ステップ 1: 初回学習
  □ 1-1 JAX CUDA 環境確認
  □ 1-2 seed=42 で学習実行
  □ 1-3 seed=123, 456 で re-training
  □ 1-4 log/version_x/ で checkpoint 確認

□ ステップ 2: Checkpoint 評価
  □ 2-1 phase0_eval_diagnostics.py 実行
  □ 2-2 KL, episode_alive, 終了理由 の出力確認
  □ 2-3 結果を JSON に保存

□ ステップ 3: 結果分析
  □ 3-1 合格基準に対してチェック
  □ 3-2 KL スパイク、episode_alive 低下の原因特定
  □ 3-3 報酬トレンド確認

□ ステップ 4: 原因分析
  □ 4-1 deterministic evaluation 実行
  □ 4-2 問題検出時は対策を実施

□ ステップ 5: 正式評価
  □ 5-1 3 シード で re-training
  □ 5-2 gate0_formal_eval.py で Gate 0 判定
  □ 5-3 PASS or FAIL を docs/status.md に記録

□ その他
  □ 各 step の ログを timestamps で整理
  □ 診断レポート を outputs/ に出力
  □ 異常検出時は即停止（docs/status.md に記録）
```

---

## 📌 重要な注意事項

### ❌ やってはいけないこと

1. **合格済み checkpoint の上書き**
   - 必ず version_x で世代管理

2. **報酬・PPO・観測・物理モデルの同時変更**
   - 1 iteration = 1 変更カテゴリ

3. **NaN/Inf を無視して続行**
   - 即座に停止して docs/status.md に記録

4. **実機テスト無しで外乱を有効化**
   - Gate C 合格まで外乱無効（DISTURBANCE_CURRICULUM=False）

### ✅ 必ずやること

1. **各 iteration で docs/status.md を更新**
   - 仮説、実施内容、結果、判定

2. **checkpoint 生成時に log.json を保存**
   - 学習曲線の証拠

3. **異常検出で即停止**
   - KL 爆発、torque 制限違反、性能低下

---

## 🎯 成功の定義

**Phase 0 クリア条件**:
```
✅ 3 seed の Qualification PASS
✅ kl_mean、episode_alive、policy std、value loss が基準内
✅ NaN/Infなし
✅ checkpointを再現可能な形で評価
✅ Checkpoint 生成・評価可能
✅ 全ステップで docs 更新・記録完全
```

上記を達成すれば、学習済み方策の無外乱Gate 0評価へ進み、その後にGate Aを判定する。

---

**報告者**: Claude  
**作成日**: 2026-09-09  
**ステータス**: ✅ 診断完了、GPU 学習実行待ち
