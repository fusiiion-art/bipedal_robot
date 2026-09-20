# Phase 3 引き継ぎ事項（要実機・WSL/GPU検証項目）

本書は `copilot_fix_instructions.md` の Phase 3 および実装プロセスで確認された未確定事項・申し送り事項を整理したドキュメントです。
静的コードレビューと単体修正では確定できない事項について、WSL/GPU環境および実機環境での検証項目をまとめています。

---

## 1. WSL/GPU環境での学習・シミュレーション検証

### 1.1 DR修正（項目2）の学習収束検証
- **背景**: `envs/mjx_env.py` のDomain Randomization実装を「ステップごとの物理モデル再構成」方式（案A）へリファクタリングしました。
- **検証項目**:
  - `python3 train/train_mjx.py` を実行し、`UnexpectedTracerError` や JIT 再コンパイルループが発生しないこと。
  - 学習中の報酬（Reward）および方策エントロピー/KLダイバージェンスが以前のベースラインと同等以上に推移し、正常に収束すること。
  - `tests/test_training_integration.py` がすべて PASS すること。

### 1.2 エピソード境界リセット（項目3）の検証
- **背景**: `AutoResetWrapper` がリセットしない独自 `info` フィールド（`step`, DRパラメータ等）を `EpisodeInfoResetWrapper` でリセットするようにしました。
- **検証項目**:
  - 複数エピソードを連続実行した際に、各環境スロットの `step` カウンタがエピソード終了時に 0 に戻り、次エピソードに前エピソードの履歴やDR値がリークしないこと。

### 1.3 方策ネットワーク bounds 回帰テスト（項目5）
- **背景**: `make_ppo_networks()` に二重で渡されていた `mean_clip_scale` を削除しました。
- **検証項目**:
  - `pytest tests/test_policy_bounds.py` が PASS し、出力平均値が意図通りのクリッピング幅（約 ±3.0）に収まっていること。

---

## 2. モデルデータ・ジオメトリ整合性

### 2.1 `scripts/fix_collision_geoms.py` の複合関節対応
- **背景**: `assets/fix_collision_geoms.py` を `scripts/fix_collision_geoms.py` へ移動しました。
- **注意点**: スクリプト内の `joint = elem.find('joint')` は1ボディにつき最初の1関節のみを探索しています。
- **確認事項**:
  - ロボットのモデルファイル（`assets/humanoid/humanoid.xml` や `assets/all/all.xml`）に複合関節（1ボディに複数 `<joint>`）や、名前なし中間リンクが存在するか確認してください。
  - 該当する構造が存在する場合、衝突ジオメトリの自動導出処理の拡張が必要です。

### 2.2 `robot/config.py` の定数群の相互整合性
- 物理定数・センサ配置・報酬の重み（`REWARD_WEIGHTS` 等）のバランスは、物理モデルの変更や学習の収束挙動を見ながらチューニングしてください。

---

## 3. 実機環境（Hardware In the Loop）検証

### 3.1 実機関節角度フィードバック（項目6）
- **背景**: `real/real_io.py` の `interleave_read_status()` に `CMD_SERVO_POS_READ` を追加し、`real/real_env.py` でコマンド値ではなく実サーボ角度を観測に反映するようにしました。
- **検証項目**:
  - 実機の通信バスにおいて、サーボ位置のポーリング読み取りが 100Hz 制御ループの許容レイテンシ内に収まること。
  - 受信エラーやタイムアウト時のフォールバック（`smoothed_action` への退避）が正常に機能すること。

### 3.2 実機とシミュレータの観測空間整合性
- `real/real_env.py` の観測構築順序と `envs/mjx_env.py` の観測順序は一致していますが、実機センサの生値レンジ・ノイズレベルがシム側の想定（正規化・スケール）と乖離していないかを実測値で確認してください。

---

## 4. プロジェクト構成・設定ファイルの残課題

### 4.1 `stubs/` ディレクトリの扱い（項目15申し送り事項）
- **背景**: `stubs/board.py` および `stubs/busio.py` は CircuitPython 互換スタブです。
- **現状**: `stubs/` を `PYTHONPATH` に通している設定ファイル（`conftest.py`, `pytest.ini`, `pyproject.toml`, CIスクリプト等）がリポジトリ内に見当たらなかったため、インポート破壊を防ぐ目的で `stubs/` をそのままルート直下に残しています。
- **今後の対応**:
  - 開発環境やCIでの CircuitPython スタブ読み込み方式を確認した上で、必要に応じて `real/stubs/` への移動とパス設定の追加を行ってください。

### 4.2 フロントエンド・可視化ツールのレビュー
- `scripts/collision_tuner.py` のフロントエンド（HTML/JS）部分は静的レビューの対象外でした。GUIの動作確認を行ってください。
