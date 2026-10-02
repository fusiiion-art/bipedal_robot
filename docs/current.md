# 二足直立ロボット 現行仕様

最終更新: 2026-10-02（Claude）

この文書は、リポジトリ内のコードに対する短い運用メモです。実装が正本です。

---

## 現行アーキテクチャ

- **学習**: `train/train_mjx.py` + `envs/mjx_env.py` の JAX/MJX + Brax PPO
  - ラッパー構成: `raw env` → `EpisodeWrapper` → `AutoResetWrapper` → `EpisodeInfoResetWrapper` → `TrainingProgressWrapper`
  - `EpisodeInfoResetWrapper` は done になった env の `info` 全キー（`rng_key` と学習進捗・終了種別を除く）と `pipeline_state`/`obs` を同じ fresh reset の値に揃える
  - env の `max_episode_steps`（time_out 判定）は PPO の `episode_length` と同じ値を渡す
  - asymmetric actor-critic: 観測は `{'state': actor用625次元, 'privileged_state': critic用}` の dict
  - 学習開始時に `<run_dir>/run_manifest.json`（git コミット・解決済み PPO 設定・実際の総 step 数）を書き出す。未コミットの変更があると起動しない（`--allow_dirty` で明示的に許可）
  - DR: ベースモデルは不変とし、`step()` 冒頭で DR 値（質量/摩擦/胴体重心）を適用したモデルを生成して `mjx.step()` に渡す。関節粘性・クーロン摩擦は `qfrc_applied`、サーボ温度・電圧は derating として反映
  - 初期状態分布: nominal 姿勢の関節角・関節角速度に一様ノイズ（`INIT_JOINT_POS_NOISE` / `INIT_JOINT_VEL_NOISE`）、低い方の足裏が nominal と同じ高さで接地するよう胴体高さを補正
  - 安全フィルタ: `envs/cbf.py` の CBF 近似（可動域5%マージン + ペナルティ）+ 熱・電圧 derating
- **実機**: Raspberry Pi 5 の `real/real_env.py` と Teensy 4.1 の `real/real_io.py`
  - 制御周期: 実機 100Hz、Teensy 側のセンサー・サーボ安全処理は 1kHz 想定
  - フィードバック: `CMD_SERVO_POS_READ` による実サーボ角度のポーリング取得（未取得時は default 姿勢）
- **アクション**: 中腰デフォルト姿勢からの関節目標角の残差 Δq（±30°）。トルク直接指令は使わない
- **姿勢**: world 鉛直基準。高さは足裏相対。傾斜床対応は実装しない。歩行・踏み替えはスコープ外

---

## ディレクトリ

- `envs/`: MJX 環境 (`mjx_env.py`), 報酬 (`mjx_rewards.py`), 安定性指標 (`stability_metrics.py`), 安全フィルタ (`cbf.py`), サーボモデル (`actuator_model.py`), 学習用ラッパー (`training_wrapper.py`)
- `robot/`: 設定 (`config.py`), 方策ネットワーク・checkpoint 読み込み・推論関数 (`policy_network.py`), 数学関数 (`math_utils.py`)
- `train/`: PPO 学習 (`train_mjx.py`), 可視化 (`visualize_rl.py`), ONNX 出力 (`export_onnx.py`), 軌跡・GIF 出力 (`export_trajectory.py`), 軌跡ビューア (`view_trajectory.py`)
- `scratch/`: Gate 0 評価 (`gate0_eval.py`), Gate A 診断・判定 (`phase0_eval_diagnostics.py`, `gate_a_qualification.py`), Gate B 評価・判定, PPO ログ診断, 物理限界の抽出, 当たり判定の描画, WSL 用シェルスクリプト
- `scripts/`: MuJoCo モデルの当たり判定・質量の調整ツール
- `real/`: 実機環境 (`real_env.py`), Teensy 通信 (`real_io.py`)
- `tests/`: 単体・契約・統合テスト

---

## 観測契約

actor 観測 `'state'`（`RobotConfig.OBS_DIM` = 625、実機 `real/real_env.py` と同じ内容・順序）:

| 範囲 | 内容 |
|---|---|
| 0–83 | Base 観測: base_pos(3, 常に0) / 重力射影ベクトル(3, 胴体座標系) / 線速度(3, σ0.5のノイズ) / 角速度(3, 胴体座標系) / 関節角(20) / 関節角速度(20) / FSR 接地フラグ(8, 0/1) / ZMP(2, 常に0) / 位相(2, 常に[0,1]) / 参照角(20, 常に0) |
| 84–503 | 直近5stepの Base 観測（古→新、エピソード開始時は最初の観測で充填） |
| 504–603 | 直近5stepの指令関節角（古→新、開始時は現在の関節角で充填） |
| 604–623 | サーボ温度 |
| 624 | 電源電圧 |

常に定数のチャネルは歩行用の旧設計の名残だが、実機 ONNX との625次元契約を保つため残している。

critic 観測 `'privileged_state'`（`RobotConfig.PRIVILEGED_OBS_DIM`）= actor 観測 + 真の胴体位置・重力射影・線速度・角速度、
関節角・角速度、FSR 連続値、足裏の水平変位、DR 値、外力。

---

## 実機 FSR 経路

`FSR402 x8 -> 分圧 + RC -> TeensyオンチップADC -> Teensy側閾値判定 -> USB -> Raspberry Pi`

- Teensy から受ける FSR 値は 8 個の二値接地フラグ（`0.0` または `1.0`）。sim も同じく `FSR_CONTACT_THRESHOLD` で二値化して観測させる
- 実機では CoP/ZMP を FSR から算出しない（観測の ZMP は常に0）
- シミュレーション内の FSR 連続値は報酬・評価と critic 観測専用

---

## コード上の主要契約

設定の正本は `robot/config.py`:

- `NUM_JOINTS = 20`, `BASE_OBS_DIM = 84`, `HISTORY_LEN = 5`, `OBS_DIM = 625`
- `CONTROL_DT = 0.01` 秒 (100Hz), `MAX_EPISODE_STEPS = 500`
- `DISTURBANCE_CURRICULUM = False`（Phase 0）

方策ネットワークの正本は `robot/policy_network.py`:

- `POLICY_MEAN_CLIP_SCALE = 3.0`, `POLICY_MIN_STD = 0.15`, `POLICY_MAX_STD = 3.0`
- softsign 式による平均値クリップ + std クリップを `BoundedNormalTanhDistribution` として factory が注入する（brax 本体は書き換えない）
- checkpoint の読み込み・推論関数の構築は `load_checkpoint()` / `make_inference_fn_from_params()` を使う
- 実機用 ONNX は `train/export_onnx.py` が重みから直接組み立て、JAX の推論と一致することを検証して書き出す（入力 `observation` [N, 625] → 出力 `action` [N, 20]）

---

## 開発ルール

- 1 iteration につき変更カテゴリは 1 つだけにする
- 合格済み checkpoint を上書きしない
- NaN/Inf、トルク制限違反、性能低下を検出したら停止して `status.md` に記録する
- 成功基準や外乱上限を自動変更しない
- 実機コマンドを自動実行しない
- コード変更後は必ず `./venv_wsl/bin/python -m pytest tests/ -v` を実行して回帰テストを行う
