# 進捗ステータス

最終更新: 2026-09-21（Copilot）

---

## 現在地

- **完了 (Phase 1 〜 Phase 3 全項目実装・統合完了)**:
  - Claude レビュー結果に基づく修正指示書（`copilot_fix_instructions.md`）の全項目（項目1〜15、Phase 2 統合テスト、Phase 3 引き継ぎ文書）を実装完了。
- **完了 (WSL2/GPU テストスイート全件 PASS)**:
  - WSL2 環境（Python 3.12.3 / JAX 0.4.35 / Brax 0.14.2 / MuJoCo 3.2.4）にて `./venv_wsl/bin/python -m pytest tests/ -v` を実行し、**37件全件合格（100% PASS）**。
- **現在状態**:
  - JIT トレーサーリーク、AutoReset 境界リーク、mean 二重クリップ、JIT 再コンパイル遅延などの構造的障害はすべて排除済み。
  - **本番学習（`python3 train/train_mjx.py`）を実行可能な状態**。

---

## 主な実装・改修成果（2026-09-20〜09-21）

### 1. 訓練パイプラインの健全化（Phase 1）
- **[項目1] `Tuple` import 漏れ修正**: `robot/gait_generator.py` に追加。`real/real_env.py` のトップレベル import エラーを解消。
- **[項目2] Domain Randomization (DR) の動的物理モデル再構成（案A）**:
  - `reset()` 内での `self._mjx_model` 破壊的代入を廃止し、DR パラメータ（`mass_scale`, `fric_scale`, `com_offset`）を `info` に保持。
  - `step()` 冒頭でスケーリング適用モデルをその場で構築して `mjx.step()` および `reward_system.compute()` に渡す方式に変更。
  - `jax.jit(env.reset)` と `jax.jit(env.step)` のトレース境界を分離し、`UnexpectedTracerError` を根絶。
- **[項目3] `EpisodeInfoResetWrapper` の追加**:
  - `envs/training_wrapper.py` に `EpisodeInfoResetWrapper` を実装し、`train/train_mjx.py` の `AutoResetWrapper` 外側に配置。
  - `AutoResetWrapper` がリセットしないエピソードスコープ変数（`step`, DR パラメータ, アクション履歴, サーボ温度等）を、エピソード終了時（`state.done=True`）のスロットについてのみ確実にリセット値（0 等）へ復元。
- **[項目4] CBF 後 derating クリップの修正**:
  - `envs/mjx_env.py` で `jp.clip(thermal_derating * voltage_derating, 0.0, 1.0)` により derating の安全限界オーバーを抑止。
- **[項目5] 方策ネットワーク mean 二重クリップの解消**:
  - `robot/policy_network.py` の `make_ppo_networks` 呼び出しから冗長な `mean_clip_scale` を削除。
  - `clipped_create_dist`（モンキーパッチ側）のみでクリップを行い、意図通りの bounds（≈3.0）を確保。forward pass 回帰テストも PASS。
- **[項目6 & 7] 実機環境の配線と定数一元化**:
  - `real/real_io.py` で `CMD_SERVO_POS_READ` をポーリングし、実サーボ角度を `real/real_env.py` の観測に配線。
  - `real/real_env.py` のハードコード定数を `RobotConfig.*` 参照に統一。
- **[項目8] サイクロイド軌道の `stand_height` 反映**:
  - `robot/gait_generator.py` の Z 軌道計算に `-stand_height` オフセットを適用し、将来の歩容拡張時の IK 破綻を予防。
- **[項目9 & 10] 診断スクリプトの一本化と pytest テスト追加**:
  - `scratch/phase0_eval_diagnostics.py` にロジックを集約し、`tests/test_phase0_eval_diagnostics.py` に pytest 単体テスト群（8件）を追加。
- **[項目11 & 12] JIT コンパイルの最適化**:
  - `scratch/gate0_standing_eval.py` および `train/visualize_rl.py` で `jax.jit` をループ外で 1 回のみ作成・使い回す設計に改修。

### 2. リポジトリ・フォルダ構成の整理（Phase 1.5）
- **[項目13] Gate 0 評価スクリプトの統合**:
  - `gate0_formal_eval.py` / `gate0_mujoco_eval.py` / `gate0_standing_eval.py` の 3 ファイルを [`scratch/gate0_eval.py`](file:///c:/bipedal_robot/scratch/gate0_eval.py)（`--mode` 引数切り替え）に統合。
- **[項目14] 重複・不要スクリプトの整理**:
  - `render_collision.py` に `--mode` を追加して `render_pure_collision.py` を統合。
  - `validate_progress_wrapper.py` を [`tests/test_training_wrapper.py`](file:///c:/bipedal_robot/tests/test_training_wrapper.py) に正式 pytest 化。
  - `save_html.py` の GIF レンダリング処理を [`train/export_trajectory.py`](file:///c:/bipedal_robot/train/export_trajectory.py) に `--output`/`--gif` 引数付きで移植。
  - `render_simulation_video.py`, `validate_policy_bounds.py` 等の不要スクリプトを削除。
- **[項目15] フォルダ構造の正規化**:
  - `assets/fix_collision_geoms.py` → `scripts/fix_collision_geoms.py` へ移動。
  - `safety/cbf.py` → `envs/cbf.py` へ移動し `safety/` フォルダを削除。
  - `deploy/export_onnx.py` → `train/export_onnx.py` へ移動し `deploy/` フォルダを削除。
  - `stubs/` はインポート破壊防止のため配置を維持。

### 3. 統合テスト追加と回帰防止（Phase 2）
- [`tests/test_training_integration.py`](file:///c:/bipedal_robot/tests/test_training_integration.py) を新規作成し、以下の 3 本の統合テストを追加（すべて PASS）:
  1. `test_vmap_reset_then_step_no_crash`: vmap + jit 下でのクラッシュフリー検証
  2. `test_domain_randomization_differs_per_env_under_vmap`: 並列環境ごとの DR 多様性検証
  3. `test_auto_reset_resets_episode_scoped_info`: AutoResetWrapper 配下でのエピソードスコープ変数初期化検証

---

## 直近の判定根拠

- **WSL2 上でのテスト実行結果**:
  - `./venv_wsl/bin/python -m pytest tests/ -v`
  - **37 passed, 17 warnings (0:06:11)**
  - 詳細は [`docs/pytest結果.md`](file:///c:/bipedal_robot/docs/pytest%E7%B5%90%E6%9E%9C.md) を参照。

---

## 次のステップ（Next Steps）

1. **WSL2 / GPU 環境での本番学習の実行**:
   ```bash
   ./venv_wsl/bin/python train/train_mjx.py
   ```
2. **学習挙動の監視**:
   - DR 有効下でのステップ実行速度（JIT 再コンパイルが発生しないこと）
   - 報酬の推移および KL ダイバージェンスの安定性（以前の KL スパイクや episode_alive 急落が解消しているか）
3. **学習完了後の評価**:
   - `scratch/gate0_eval.py --mode policy --checkpoint <path>` による Gate 0 判定