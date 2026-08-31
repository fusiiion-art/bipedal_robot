# 外乱耐性直立姿勢維持の達成に向けた学習改善

## 解析結果サマリ

v0/v1 ともに学習は**完全に失敗**しています。根本原因は以下の通りです。

| 問題 | 深刻度 | 証拠 |
|------|--------|------|
| エピソード生存 ≈ 40/1000 ステップ (0.4秒で転倒) | 致命的 | `episode_alive ≈ 40`, `fall_penalty = -500` が全エピソード |
| 報酬の後半崩壊 | 致命的 | v0: 349→-1120, v1: 392→-1099 (中盤ピーク→終盤崩壊) |
| カリキュラム外乱が機能しない | 致命的 | `curriculum_scale ≈ 2.0` (閾値200000に到達不可能) |
| 復帰ボーナスがほぼゼロ | 致命的 | `disturbance_recovery_bonus ≈ 0` |
| λ_phase ≈ 0.21 で固定 | 重要 | 分散 0.0007 → 外乱フェーズ遷移が不活性 |

> [!CAUTION]
> **「まず直立を安定させる」ことが最優先課題です。** 外乱耐性を議論する前に、ロボットが外乱なしで1000ステップ(10秒)立ち続けられるようにしなければなりません。現在は0.4秒で転倒しており、外乱云々以前の問題です。

## 根本原因の分析

### なぜ0.4秒で転倒するのか

1. **alive報酬 100.0 vs 転倒ペナルティ -500.0 のバランス不良**: 40ステップ × 100.0 = 4000 の生存報酬を得ても、total_penalty ≈ 4900 + fall_penalty = -500 で相殺される。ペナルティ項の合計が大きすぎて「立っているだけ」で報酬が負になる。
2. **初期姿勢の不安定性**: `DEFAULT_JOINT_ANGLES` で中腰姿勢を設定しているが、高さ z=0.235m からの開始で、制御開始直後の過渡応答で転倒。
3. **行動空間が大きすぎる**: `ACTION_SCALE = ±90°` は HX-30HM の可動域全域を使う設定。初期ポリシーのランダム探索で関節が大きく動き、即座にバランスを崩す。
4. **学習ステップ数の不足**: GPUモードで `steps=10M` だが、実際にログに記録されている最終ステップは 389,120。これは `num_envs=512` で `steps=200,000` 相当。10Mステップには到達していない。

> [!IMPORTANT]
> **確認事項**: 学習実行時に `--steps` 引数を明示的に指定していましたか？ デフォルトの `10_000_000` が使用された場合、ログの 389,120 ステップは全体の 3.89% しか進んでいません。つまり学習が途中で中断された可能性があります。もし意図的に短縮して実行した場合、ステップ数不足が最大の原因です。

## Open Questions

> [!IMPORTANT]
> 1. **学習は途中で中断されましたか？** ログの最終ステップが 389,120 (全体の3.89%) しかありません。10Mステップを完走した場合と途中中断では対策が異なります。
> 2. **使用GPU**: RTX 4060 (8GB) ですか？ VRAM制約で `num_envs` を下げる必要がある場合、学習効率に大きく影響します。
> 3. **目標**: 「外乱耐性で直立維持」は歩行なし(その場立ち)でよいですか？ 現在 `TARGET_VEL_X = 0.0`, `USE_REFERENCE_GAIT = False` になっています。

## Proposed Changes

学習を成功させるために、**3段階のアプローチ** を提案します。

### Phase 1: 基本直立の安定化 (まず立たせる)

---

#### [MODIFY] [config.py](file:///c:/bipedal_robot/robot/config.py)

1. **ACTION_SCALE を縮小**: `±90° → ±30°` — 初期探索での暴走を防止
2. **REWARD_WEIGHTS の再調整**:
   - `alive`: 100.0 → 200.0 (生存報酬を増加、立っているだけで正の報酬が蓄積)
   - `fall_penalty`: -500.0 → -300.0 (転倒ペナルティを緩和し、探索を促進)
   - 各種ペナルティウェイトを大幅に縮小 (energy, smoothness, drift, slip, ang_momentum)
3. **MAX_EPISODE_STEPS を短縮**: 1000 → 500 (最初は短いエピソードで高速回転)

---

#### [MODIFY] [mjx_env.py](file:///c:/bipedal_robot/envs/mjx_env.py)

1. **外乱を完全無効化 (Phase 1)**: `RANDOM_PUSH_MAX_FORCE = 0.0` or プッシュ確率を 0 にする
2. **初期姿勢のランダマイゼーションを最小化**: 質量・摩擦のDRレンジを縮小
3. **global_step のエピソード跨ぎ修正**: `TrainingProgressWrapper` が注入する `_env_steps` を `global_step` として使用 (既存の `[CRITICAL-1]` バグ修正)

---

#### [MODIFY] [mjx_rewards.py](file:///c:/bipedal_robot/envs/mjx_rewards.py)

1. **ペナルティ全体のスケールダウン**: 初期学習フェーズではペナルティを 10-20% に抑制
2. **alive + upright に集中した報酬構造**: 回復報酬系 (capture_point, recovery, impedance) のウェイトを一時的に下げ、まず「直立が最も得」であることを学習

---

### Phase 2: 外乱耐性の段階的導入

Phase 1 で直立が安定したら:

#### [MODIFY] [config.py](file:///c:/bipedal_robot/robot/config.py)

1. **CURRICULUM_SCHEDULE の修正**: `training_progress` ベースに切り替え (既に `CURRICULUM_SCHEDULE_FRACTIONS` が定義済み)
2. **外乱を段階的に導入**: 0% → 5% → 20% → 45% → 75% → 100%
3. **ACTION_SCALE を段階的に拡大**: 30° → 45° → 60°

---

#### [MODIFY] [mjx_env.py](file:///c:/bipedal_robot/envs/mjx_env.py)

1. **`_get_curriculum_scale` を `training_progress` ベースに変更**: `global_step` (バグあり) → `training_progress` (0.0~1.0) を直接使用
2. **DR範囲のカリキュラム化**: `DR_CURRICULUM_RANGES` を実際に接続

---

### Phase 3: 回復行動の強化

Phase 2 で軽い外乱に耐えられるようになったら:

#### [MODIFY] [mjx_rewards.py](file:///c:/bipedal_robot/envs/mjx_rewards.py)

1. **回復報酬系のウェイトを段階的に引き上げ**
2. **RECOVERY_BONUS_STABILITY_THRESHOLD を緩和**: 0.7 → 0.4 (閾値が高すぎて発火しない)

---

### [MODIFY] [train_mjx.py](file:///c:/bipedal_robot/train/train_mjx.py)

1. **GPU向けパラメータの確認**: `num_envs=512, steps=10M` が意図通り完走しているか確認
2. **`reward_scaling` の調整**: 0.1 → 0.01 (報酬スケールが大きすぎるとvalue lossが不安定に)
3. **Early stopping / チェックポイントからの再開機能**: best_params からの再学習

## Verification Plan

### Automated Tests
```bash
# Phase 1 完了条件の確認
wsl bash -lc "cd /mnt/c/bipedal_robot && /mnt/c/bipedal_robot/venv_wsl/bin/python train/train_mjx.py --steps 200000 --num_envs 512"
# → episode_alive > 200 (2秒以上生存) かつ fall_penalty != -500 のエピソードが出現すること
```

### Manual Verification
1. **学習カーブ**: 報酬が単調増加し、後半崩壊しないこと
2. **GIF確認**: simulation.gif でロボットが数秒間立ち続けていること
3. **段階的外乱テスト**: Phase 2以降で外乱印加時に復帰動作が観察されること
