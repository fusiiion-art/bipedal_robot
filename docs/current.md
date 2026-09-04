# 二足直立ロボット 現行仕様

最終更新: 2026-09-03

この文書は、リポジトリ内のコードに対する短い運用メモです。実装が正本であり、古い設計案・レビュー・詳細マニュアルは保持しません。

## 現行アーキテクチャ

- 学習: `train/train_mjx.py` + `envs/mjx_env.py` の JAX/MJX + Brax PPO
- 実機: Raspberry Pi 5の `real/real_env.py` とTeensy 4.1の `real/real_io.py`
- 制御周期: 実機100Hz、Teensy側のセンサー・サーボ安全処理は1kHz想定
- アクション: 関節目標角の差分（residual）。トルク直接指令は使わない
- 姿勢: world鉛直基準。高さは足裏相対。傾斜床対応は実装しない

## 実機FSR経路

`FSR402 x8 -> 分圧 + RC -> TeensyオンチップADC -> Teensy側閾値判定 -> USB -> Raspberry Pi`

- 外付けMCP3208、MCP6004、Pi側SPIは使用しない
- Teensyから受けるFSR値は8個の二値接地フラグ（`0.0`または`1.0`）
- 実機ではCoP/ZMPをFSRから算出しない
- 観測契約は既存モデル互換のため、FSR 8要素とZMP 2要素を含む625次元を維持する
- シミュレーション内のFSR連続値・ZMP指標は学習／評価専用

## コード上の主要契約

設定の正本は `robot/config.py`:

- `NUM_JOINTS = 20`
- `BASE_OBS_DIM = 84`
- `HISTORY_LEN = 5`
- `OBS_DIM = 625`
- `CONTROL_DT = 0.01` 秒
- `MAX_EPISODE_STEPS = 500`
- `DISTURBANCE_CURRICULUM = False`（Phase 0）

実機I/Oの正本は `real/real_io.py`:

- `TeensySpineIO.communicate()` がIMU、FSR接地フラグ、サーボ状態を受信する
- USBテレメトリは既存の73バイト形式を維持し、末尾8スロットをFSRフラグとして扱う
- Teensy側ファームウェアは各FSRを閾値判定してから送信する

観測生成の正本は `real/real_env.py`:

- FSR 8要素は接地フラグ
- 実機ZMP 2要素はゼロ
- 観測順序・次元を学習側と変更しない

## 開発ルール

- 1 iterationにつき変更カテゴリは1つだけにする
- 合格済みcheckpointを上書きしない
- NaN/Inf、トルク制限違反、性能低下を検出したら停止して `status.md` に記録する
- 成功基準や外乱上限を自動変更しない
- 実機コマンドを自動実行しない
- コード変更後は、可能な範囲で対象テストまたは `py_compile` を実行する

## 現在の検証課題

Phase 0 PPO安定性検証は未合格です。KLスパイクと学習後半の `episode_alive` 低下について、決定論的評価・終了理由・報酬内訳の確認が必要です。詳細な判定と次の作業は `status.md` に記録します。

## 廃止した文書

旧設計案、ハードウェアBOM、詳細マニュアル、報酬・衝突レビューは、現行コードとの重複や旧仕様の混在を避けるため廃止しました。新しい仕様はコードとこの文書だけを更新します。
## Copilotが自律実行できる範囲

実機計測を除き、コード変更、テスト、GPU学習、ログ監視、
checkpoint評価、外乱強度別評価、失敗分析、rollback管理、
Teensyファームウェア作成、ONNX・通信プロトコル検証、
進捗ドキュメント更新を実行できる。

## 人間の実機作業

実機計測、FSRキャリブレーション、Teensy書き込み、
実機E-stop試験、実機外乱試験、最終合否承認は人間が行う。

## 自律反復の停止条件

NaN/Inf、トルク制限違反、既存合格モデルからの性能低下、
通信・安全系の異常を検出した場合は反復を停止し、
docs/status.mdへ記録して人間の判断を待つ。