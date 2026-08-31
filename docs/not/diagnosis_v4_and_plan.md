# v4 修正計画 — KL爆発・学習率崩壊の解消（`implementation_plan.md` Phase 1 の修正版）

## 目標とロードマップ

- **最終ゴール**: 外乱（プッシュ）を受けても2足直立姿勢を維持できること
- **現在の目標**: まず外乱なしで安定して直立できること（ここが土台にならないと外乱耐性の評価自体ができない）

この2つは対立しないが、順番と「次に進んでよい基準」を明確にしておく:

| ステージ | 内容 | 次へ進む基準（Gate） | 現状 |
|---------|------|----------------------|------|
| **Stage 0(今回)** | ①②(KL爆発・学習率崩壊)の修正 | 序盤`kl_mean`が桁違いに爆発しない／`learning_rate`が床に張り付いたまま回復しないという現象が消える | 未着手 |
| **Stage 1** | 外乱なし(or 現状の最弱カリキュラム)での直立安定化 | `eval/episode_alive`が安定して高水準（目安: 一旦500以上、最終的には900〜1000近くまで）に達する。`stability_index`実測平均が0.5以上 | 現状ピーク148.7、最終86.5。未達 |
| **Stage 2** | ④（`training_progress`/`curriculum_scale`の不一致）を解消し、カリキュラムが実際に0→1へ進むようにする | `curriculum_scale`実測平均が学習経過とともに単調に増加することをログで確認 | 現状停滞中(0.05〜0.14)。未着手 |
| **Stage 3** | 外乱を受けた際の復帰行動の強化（ゴール本体） | `disturbance_recovery_bonus`が非ゼロで頻繁に発火、外乱後も転倒せず`stability_index`が閾値以上に回復する | v3で`r_recovery`が出現し始めた段階 |

> [!NOTE]
> Stage 1を「外乱ゼロに手動で固定する」(Phase 1提案)のではなく「今のカリキュラムのまま進める」のは、④が未解決なため実質的にカリキュラムがほぼ最弱で止まっており、**手を加えなくても現状がStage 1相当の条件になっている**ため。Stage 1のGateを満たしたら、Phase 1のように外乱を明示的にいじる必要はなく、そのままStage 2(④の修正)に進めばよい。

---

## 背景

`implementation_plan.md`（既存のPhase 1提案）は「v0/v1が完全に学習失敗している」という前提で書かれていますが、実際には**v3の時点で復帰行動が既に出現しており**（`r_recovery: 0.008→4.83`、転倒回避 `0/20→4/20`）、その後 `log.json`（`--steps 200000`, 約32分のテスト実行）を数値解析した結果、Phase 1が想定していたものとは**別の、より特定しやすい原因**が見つかりました。

> [!CAUTION]
> **Phase 1（外乱完全無効化・`ACTION_SCALE`を±30°に圧縮・`MAX_EPISODE_STEPS`を500に短縮）は、現時点ではまだ実行しないことを推奨します。** 根拠は下記「やらないこと」を参照。v3の進捗を無駄に巻き戻すリスクがあります。

---

## 診断結果（優先度順、いずれもログの実測値から確認済み）

### 🔴 ① `alive` / `fall_penalty` のスケールが他の報酬項と2桁以上ズレている（根本原因）

- `train_mjx.py` に `reward_scaling=0.01,  # Phase 1: alive=200.0 に対応し報酬スケールを縮小` というコメントがあり、`alive≈200 / fall_penalty≈-300` になっていると推測される。他の項（`capture_point=5.0`, `recovery=4.0` 等）は1桁のまま。
- この不均衡が、学習序盤の破局的なポリシー更新（`kl_mean`: 42,101 → 56,709、正常値は0.01〜0.05程度）の直接原因と考えられる。

### 🔴 ② KL爆発が `learning_rate` を床(floor)に叩き落とし、二度と回復していない（①の直接的な帰結・後半崩壊の真因）

```
step= 40,960  lr=1.58e-04  kl=56,709(爆発)  reward=27,360 ← ピーク
step=102,400  lr=1.58e-05  kl=0.033(正常)   reward=23,098 ← lrが床に張り付き始める
step=389,120  lr=2.25e-05  kl=0.026(正常)   reward=14,817 ← lrは床のまま、報酬だけ低下し続ける
```

`learning_rate_schedule='ADAPTIVE_KL'` が序盤のKL爆発を検知して学習率を `learning_rate_schedule_min_lr=1e-5` 付近まで下げ、以後ずっとそこに張り付いたまま（設定上限 `5e-4` の5%未満しか使えていない）。KL自体はその後ずっと健全なので、「①のKL爆発 → 学習率が過剰に萎縮 → 良い方策に戻れないまま緩やかに悪化」という一本の連鎖として説明できる。**分析対象の"後半崩壊"は、カリキュラムの暴走が原因ではないことをログで確認済み**（下記④）。

> [!IMPORTANT]
> ①を直せば、学習序盤でこのKL爆発自体が起きなくなり、②の学習率崩壊も連鎖的に解消する可能性が高い。**まずはこの1点に絞って再テストすることを推奨。**

### 🟡 ③ `RECOVERY_BONUS_STABILITY_THRESHOLD=0.7` が実態に対して厳しすぎる

実際の `stability_index` 平均は学習を通して概ね `0.34〜0.35`（`episode_stability_index` 合計値を `episode_alive` で割った実測値）。閾値0.7には遠く及ばないため、`disturbance_recovery_bonus` が全ログで完全に0.0（分散も0）。`task.md` にある「0.7→0.4への緩和」提案はこのデータに照らして妥当。

### 🟢 ④ 【要調査・保留】ログの `training_progress` と、実際に報酬計算へ渡っている `training_progress` が食い違っている可能性

`progress_callback` がログに書く `training_progress = num_steps / steps`（CLI引数の`--steps`が分母）は0→1.95まで単調増加するが、`curriculum_scale` の実測平均（`episode_curriculum_scale ÷ episode_alive`）は学習を通じて `0.05〜0.14` の間で停滞しており、1.0に近づく気配がない。つまり **`TrainingProgressWrapper`（`envs/training_wrapper.py`）が計算している実際の`training_progress`は、ログに出ている値とは別物** の可能性が高い。優先度は①②より低いが、外乱カリキュラムが意図通り機能しているか最終確認するために、後述の検証で `envs/training_wrapper.py` の中身を確認したい。

---

## 実装タスク

### [MODIFY] `robot/config.py`

- [ ] `REWARD_WEIGHTS["alive"]` を他の回復報酬項（`capture_point=5.0`, `recovery=4.0`）と同じ桁数へ戻す。目安: `2.0`（以前納品したベースライン値）
- [ ] `REWARD_WEIGHTS["fall_penalty"]` も同様に戻す。目安: `-20.0`
- [ ] `RECOVERY_BONUS_STABILITY_THRESHOLD` を `0.7 → 0.4` に変更

### [MODIFY] `train/train_mjx.py`

- [ ] `reward_scaling` を `alive=200` 前提の `0.01` から、`alive=2.0` 前提の値へ戻す。目安: `0.1〜1.0`（①②とセットで変更しないと、今度は報酬が小さすぎて別の学習不安定を招く可能性があるため、**この2つは必ず同時に変更・同時に再テストする**）

### [INVESTIGATE] `envs/training_wrapper.py`

- [ ] `TrainingProgressWrapper` が `training_progress` をどう計算しているか確認し、`progress_callback` がログに書く値との食い違いの原因を特定する（①②の効果を確認した後で着手）

---

## 検証計画

### 再実行コマンド（前回と同一条件で比較する）

```bash
wsl bash -lc 'cd /mnt/c/bipedal_robot && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 /mnt/c/bipedal_robot/venv_wsl/bin/python /mnt/c/bipedal_robot/train/train_mjx.py --num_envs 128 --batch_size 128 --num_minibatches 8 --learning_rate 5e-5 --steps 200000'
```

### 成功基準

| 指標 | v3(修正前) | 期待される修正後の変化 |
|------|-----------|----------------------|
| 序盤(step 20,480〜40,960)の `kl_mean` | 42,101 / 56,709 | 1桁台〜0.1程度に収まる（少なくとも桁違いの爆発が消える） |
| `learning_rate` の推移 | 序盤で床(1e-5付近)に張り付き回復せず | 床に張り付かない、または張り付いても後半で回復する |
| `eval/episode_reward` のカーブ | step 40,960でピーク→単調減少 | ピーク後の急減が緩和される、または単調増加に近づく |
| `disturbance_recovery_bonus` | 全期間0.0 | 非ゼロの値が時折出現する |

> [!TIP]
> ログを見る際は `eval/episode_*` が**エピソード内合計値**であることに注意。平均を見たい場合は必ず対応する `eval/episode_alive`（生存ステップ数）で割ること（例: `episode_lambda_phase ÷ episode_alive`）。合計値のまま「1.0を超えているからバグ」と判断しないこと。

---

## 明示的に「今はやらないこと」（`implementation_plan.md` Phase 1 からの変更点）

| Phase 1の提案 | 今回見送る理由 |
|---------------|----------------|
| `RANDOM_PUSH_MAX_FORCE = 0.0`（外乱完全無効化） | 実測では外乱カリキュラム自体が既にほぼ最弱で停滞している（④）ため、無効化しても現状とほぼ変わらない可能性が高く、根本原因(①②)の検証を遅らせるだけ |
| `ACTION_SCALE` を ±90° → ±30° | v3は既に復帰行動を獲得しつつあり、行動空間を縮小する必然性がログから読み取れない。①②を直した上でまだ暴れるようなら再検討 |
| `MAX_EPISODE_STEPS` を 1000 → 500 | 同上。生存ステップは既に86.5まで伸びており、エピソードを短縮する動機がログにない |

これらは「v0/v1が完全に学習失敗している」という当初の前提に基づく緊急リセット策であり、①②という具体的でより確度の高い原因が見つかった今は温存し、①②だけを直した上で同条件のテストを再実行して効果を確認するのが安全です。それでも改善しない場合の"保険"として残しておいてください。
