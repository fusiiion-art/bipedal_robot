# 進捗ステータス

最終更新: 2026-09-09（Copilot）

## 現在地

- 完了: FSR実機経路をTeensyオンチップADCの二値接地判定へ統一。実機のMCP3208/MCP6004/SPI依存を削除。
- 目的確定: 歩行・踏み替えなしで、外乱後も両足接地の直立姿勢を維持する固定足立位。
- 学習前検証: 固定足設定、両足接地報酬、外乱力レベル／力積／方向数／印加時間の定義を追加。
- 評価基盤: 成功率、両足接地率、最大足移動量、最大roll/pitch、回復時間、トルク飽和率の集計を追加。
- PPO基盤: `--seed`／`--target_kl`をCLI化し、`State.done`をterminated限定へ修正。
- 整理完了: 未使用RMAネットワーク／共有メモリ、旧センサーフュージョン、未使用抽象環境、仕様外地形テスト、旧SPI資料を廃止。学習・実機・評価経路を標準PPO MLPへ統一。
- 着手中: Phase 0 PPO安定性診断（D-1〜D-5の計測準備）
- 次: 現行設定を変えずにD-6のGPU Debug runを実行し、KL・終端理由・報酬内訳・deterministic評価を収集

## 直近の判定根拠

関連Pythonの構文検査、VS Codeエラー検査、立位設定・外乱設定のWSL上のassert検証、学習CLIの`--seed`／`--target_kl`確認に合格。`--target_kl`はBraxのAdaptive KL学習率制御に接続されているが、epoch内early stoppingではない。pytestはWSL環境にも未インストール。実checkpointによる評価は未実行。

## エスカレーション中の項目

Phase 0未合格。KLスパイクと学習後半の`episode_alive`低下が未解決のため、Gate A以降は保留。`log`配下に評価用checkpointがないため、実checkpointによる診断は未実行。
# 進捗ステータス - Phase 0 PPO安定性検証（2026-08-26～09-01）

最終更新: 2026-09-01 最終更新者: Copilot

## 現在地

- 完了Task: C-03 KL計算（reduce軸確認、方策クリップ実装・テスト）、C-04 std範囲修正（min 0.05、max 3.0）
- 着手中Task: Phase 0 PPO安定性診断（KLスパイク・episode_alive低下の原因分析）
- 次のTask: deterministic/stochastic評価の実装と終了理由ヒストグラム化

## 直近の判定根拠

min_std=0.05版GPU学習（phase0_policy_bounds_gpu_min005）完走。KL最大232（前回18418から98.7%低下）、min_std=0.05019確認。
ただしKL=232は依然健全域外（目標0.02～0.05）。episode_alive=77（初期110から30%低下）。
Brax GAE/bootstrap実装確認：termination正しく分離（=(1-discount)*(1-truncation)）、time_out有効。
Horizon 500step統一確認、報酬clip各step±300（epoch累積ではない）。初期KLスパイク原因未解決。

## エスカレーション中の項目

**Phase 0未合格、Gate A進行保留**
- KLスパイク=232：初期更新で方策が大きく跳ぶ（健全域0.02-0.05未達）
- episode_alive低下：報酬上昇と生存時間が乖離（reward hacking兆候）
- 次の切り分け：deterministic評価、終了stepヒストグラム、報酬成分分解ログ

このセッションの主な成果：

✅ 実装完了

PPO方策 loc soft clip + std下限/上限 クリップ（0.05-3.0）
min/max両側の境界テスト実装・PASS
JAX永続コンパイルキャッシュ設定
✅ 診断完了

Horizon 500step統一確認（不一致なし）
Brax GAE/bootstrap正しく実装済み
std下限がKL爆発の主要因（min_std=0.00283→0.05019）
⏸️ 要分析

初期KLスパイク（232）が残存
episode_alive後半低下の詳細原因
報酬構成とreward hackingの関係

# 進捗ステータス

最終更新: 2026-09-09

## 現在地

- 完了:
  - `robot/config.py` の curriculum / initial height / gait parameter 整理
  - `robot/kinematics.py` のリンク長単位整合と到達範囲判定修正
  - `robot/math_utils.py` の JAX/NumPy 分離と JIT 互換化
  - `real/real_env.py` の位相同期と FIFO 履歴整合
  - `real/real_io.py` の checksum 検証厳格化
  - `safety/cbf.py` の margin 計算とペナルティ基準統一
- 着手中:
  - viewer 系の実行パス/生成手順の最終整備
  - status 文書の更新反映
- 目標:
  - 固定足直立制御における外乱耐性の検証を継続
  - 実測値反映による sim-to-real 整合性向上

## 直近の判定根拠

- `robot/config.py` で `CURRICULUM_SCHEDULE_FRACTIONS` へ統一済み
- `GAIT_THIGH_LEN` / `GAIT_KNEE_LEN` が 0.12 m に統一済み
- `kinematics.py` の長さ比較が [m] 単位で正しく動作
- `math_utils.py` の `quat_to_euler` が NumPy/JAX で分離され、JIT 互換の設計になっている
- `real_io.py` の checksum 検証は破損データを `None` で落とすように修正済み
- viewer 側は相対パス依存の修正と生成スクリプトの明確化が未反映

## エスカレーション中の項目

- viewer 系のパス解決と自動生成手順の整備
- status 文書の最新状態への反映
- 実測値を取り込んだ後の sim-to-real 再検証