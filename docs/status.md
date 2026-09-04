# 進捗ステータス

最終更新: 2026-09-03（Copilot）

## 現在地

- 完了: FSR実機経路をTeensyオンチップADCの二値接地判定へ統一。実機のMCP3208/MCP6004/SPI依存を削除。
- 目的確定: 歩行・踏み替えなしで、外乱後も両足接地の直立姿勢を維持する固定足立位。
- 学習前検証: 固定足設定、両足接地報酬、外乱力レベル／力積／方向数／印加時間の定義を追加。
- 評価基盤: 成功率、両足接地率、最大足移動量、最大roll/pitch、回復時間、トルク飽和率の集計を追加。
- PPO基盤: `--seed`／`--target_kl`をCLI化し、`State.done`をterminated限定へ修正。
- 着手中: Phase 0 PPO安定性診断
- 次: GPU/WSLで固定足立位モデルをseed指定で学習し、checkpoint生成後に外乱強度別評価を実行

## 直近の判定根拠

関連Pythonの構文検査、VS Codeエラー検査、立位設定・外乱設定のWSL上のassert検証、学習CLIの`--seed`／`--target_kl`確認に合格。pytestはWSL環境にも未インストール。実checkpointによる評価は未実行。

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