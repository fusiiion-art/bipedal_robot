# 二足直立ロボット 現行仕様

最終更新: 2026-09-21（Copilot）

この文書は、リポジトリ内のコードに対する短い運用メモです。実装が正本であり、古い設計案・レビュー・詳細マニュアルは保持しません。

---

## 現行アーキテクチャ

- **学習**: `train/train_mjx.py` + `envs/mjx_env.py` の JAX/MJX + Brax PPO
  - ラッパー構成: `raw env` → `AutoResetWrapper` → `EpisodeInfoResetWrapper` → `TrainingProgressWrapper`
  - DR (Domain Randomization): ベースモデルは不変とし、`step()` 冒頭でスケーリング適用モデルを動的生成して `mjx.step()` および `reward_system.compute()` に渡す（JIT トレーサーリーク防止）
  - 安全フィルタ: `envs/cbf.py` による Control Barrier Function + 熱・電圧 derating クランプ (`jp.clip(..., 0.0, 1.0)`)
- **実機**: Raspberry Pi 5 の `real/real_env.py` と Teensy 4.1 の `real/real_io.py`
  - 制御周期: 実機 100Hz、Teensy 側のセンサー・サーボ安全処理は 1kHz 想定
  - フィードバック: `CMD_SERVO_POS_READ` による実サーボ角度のポーリング取得（通信途絶時は `smoothed_action` へフォールバック）
- **アクション**: 関節目標角の差分（residual）。トルク直接指令は使わない
- **姿勢**: world 鉛直基準。高さは足裏相対。傾斜床対応は実装しない

---

## ディレクトリ・構成の現行配置

- `envs/`: MJX 環境 (`mjx_env.py`), 報酬計算 (`mjx_rewards.py`), 安全フィルタ (`cbf.py`), 訓練ラッパー (`training_wrapper.py`)
- `train/`: PPO 学習スクリプト (`train_mjx.py`), 可視化 (`visualize_rl.py`), ONNX エクスポート (`export_onnx.py`), 軌跡 & GIF エクスポート (`export_trajectory.py`)
- `robot/`: ロボット定数 (`config.py`), 運動学 (`kinematics.py`), 歩容生成 (`gait_generator.py`), 方策ネットワーク (`policy_network.py`)
- `real/`: 実機環境 (`real_env.py`), Teensy 通信 (`real_io.py`)
- `scratch/`: 統合 Gate 0 評価スクリプト (`gate0_eval.py`), 診断スクリプト (`phase0_eval_diagnostics.py`), 衝突可視化 (`render_collision.py`)
- `scripts/`: メッシュ変換・ジオメトリ修正スクリプト (`fix_collision_geoms.py` 等)
- `tests/`: 単体・契約・統合テストスイート（全37件、pytest 完全対応）
- `stubs/`: CircuitPython 互換スタブ（`board.py`, `busio.py`）

---

## 実機 FSR 経路

`FSR402 x8 -> 分圧 + RC -> TeensyオンチップADC -> Teensy側閾値判定 -> USB -> Raspberry Pi`

- 外付け MCP3208、MCP6004、Pi 側 SPI は使用しない
- Teensy から受ける FSR 値は 8 個の二値接地フラグ（`0.0` または `1.0`）
- 実機では CoP/ZMP を FSR から算出しない
- 観測契約は既存モデル互換のため、FSR 8 要素と ZMP 2 要素を含む 625 次元を維持する
- シミュレーション内の FSR 連続値・ZMP 指標は学習／評価専用

---

## コード上の主要契約

設定の正本は `robot/config.py`:

- `NUM_JOINTS = 20`
- `BASE_OBS_DIM = 84`
- `HISTORY_LEN = 5`
- `OBS_DIM = 625`
- `CONTROL_DT = 0.01` 秒 (100Hz)
- `MAX_EPISODE_STEPS = 500`
- `DISTURBANCE_CURRICULUM = False`（Phase 0）
- `CURRICULUM_SCHEDULE_FRACTIONS`（進捗率ベースの外乱スケジュール）

実機 I/O の正本は `real/real_io.py`:

- `TeensySpineIO.communicate()` が IMU、FSR 接地フラグ、サーボ状態（温度・電圧・位置）を受信する
- USB テレメトリは 73 バイト形式を維持し、末尾 8 スロットを FSR フラグとして扱う

方策ネットワークの正本は `robot/policy_network.py`:

- `POLICY_MEAN_CLIP_SCALE = 3.0`
- `POLICY_MIN_STD = 0.15`
- `POLICY_MAX_STD = 3.0`
- softsign 式による単一平均値クリップ + std bounds クリップ

---

## 開発ルール

- 1 iteration につき変更カテゴリは 1 つだけにする
- 合格済み checkpoint を上書きしない
- NaN/Inf、トルク制限違反、性能低下を検出したら停止して `status.md` に記録する
- 成功基準や外乱上限を自動変更しない
- 実機コマンドを自動実行しない
- コード変更後は必ず `./venv_wsl/bin/python -m pytest tests/ -v` を実行して回帰テストを行う

---

## 検証ステータス

2026-09-21 時点で **全37件の pytest 単体・統合テストが PASS** し、以前のコードレビューで指摘された重大バグ（DR トレースリーク、AutoReset 境界リーク、mean 二重クリップ等）は解消済みです。現在は本番学習を実行して学習収束・報酬推移を確認するフェーズにあります。