# Copilot Instructions — 二足直立ロボット 外乱耐性RL

## 最初にやること

1. `.docs/status.md` を読み、現在地（どのTaskまで完了・着手中か）を確認する  
2. `.docs/master_plan.md`（実行計画。Task 0〜9、付録Aに詳細仕様）を読み、現在地に対応するTaskの内容を確認する  
3. `.docs/status.md`が空・存在しない場合は、コードを変更する前に現状棚卸しタスクを実行する（会話で別途指示する）

## 絶対厳守（プロジェクト全体で不変）

- 姿勢誤差はworld鉛直基準、高さは足裏相対（`.docs/master_plan.md` 付録A §1.2）  
- 傾斜床対応は不採用。傾斜床関連のコード・reward項は追加しない  
- 行動空間は関節目標角residual（`Δq`）。トルク直接指令は使わない  
- 1 iteration \= 1変更カテゴリ。報酬変更とPPOハイパーパラメータ変更を同時に行わない  
- 合格checkpointを上書きしない。性能が低下したらrollbackする  
- 成功基準・外乱上限を自動変更しない。実機コマンドを自動実行しない  
- NaN/Inf、torque limit違反、既存合格モデルからの性能低下を検知したら即座に作業を止め、`.docs/status.md`に記録して人間の判断を待つ（`.docs/master_plan.md` §9.3のエスカレーション基準）

## 各Task完了時に必ずやること

- `.docs/status.md`を更新する（完了Task／次のTask／判定根拠を簡潔に）  
- 変更をgit commitする（1コミット1変更カテゴリ、commit messageにTask番号を含める）  
- 実験ログ・設定・seedを`.docs/master_plan.md` §9.1の形式で記録する

## 参照ファイル

- `.docs/master_plan.md` — 実行計画本体（Task定義、Gate合格基準、エスカレーション基準、付録Aに詳細仕様）  
- `.docs/status.md` — 現在地・進捗ステータス（最も頻繁に更新するファイル）  
- `.docs/current_config_baseline.md` — 現在のハイパーパラメータ・報酬重み等の棚卸し（Task 1で参照・更新）

