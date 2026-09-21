# Phase 3 引き継ぎ事項（要実機・WSL/GPU検証項目）

最終更新: 2026-09-21（Copilot）

本書は `copilot_fix_instructions.md` の Phase 3 および実装・テストプロセスで確認された未確定事項・申し送り事項を整理したドキュメントです。
静的コード修正および単体・統合テスト（全37件 PASS）完了後の、WSL/GPU環境での学習検証および実機環境での検証項目をまとめています。

---

## 1. テスト検証状況（2026-09-21 完了）

以下の項目は、WSL2 環境での pytest テストスイート（`tests/test_training_integration.py`, `tests/test_policy_bounds.py` 等）により**正常動作を確認済み（PASS）**です：

- ✅ **JIT / vmap 下での DR クラッシュ防止**: `reset()` と `step()` を別々に vmap/jit しても `UnexpectedTracerError` が発生しないことを確認。
- ✅ **並列環境ごとの DR 多様性**: vmap 下で各環境スロットが異なる質量スケール等を持つことを確認。
- ✅ **エピソード境界リセット**: `AutoResetWrapper` + `EpisodeInfoResetWrapper` 配下で、エピソード終了スロットの `step` カウンタ等の独自 info が確実に 0（初期値）へリセットされることを確認。
- ✅ **方策ネットワーク bounds**: forward pass を通した出力平均値が意図通りの bounds（≈3.0）に収まり、二重クリップが発生していないことを確認。

---

## 2. WSL/GPU環境での学習・シミュレーション検証項目

### 2.1 DR修正（項目2）の学習収束検証
- **検証項目**:
  - `python3 train/train_mjx.py` を実行し、毎ステップのモデル生成による過剰な JIT 再コンパイル遅延が発生せず、高速にステップが進むこと。
  - 学習中の報酬（Reward）および方策エントロピー/KLダイバージェンスが健全域（KL: 0.02〜0.05 目安）に収まり、正常に収束すること。

### 2.2 報酬・カリキュラム挙動の確認
- **検証項目**:
  - `DISTURBANCE_CURRICULUM=False`（Phase 0）下で外力が確実に 0 のまま静止直立が安定すること。
  - 報酬ハッキング（姿勢が崩れているのに正の報酬だけを貪る状態）が発生していないこと。

---

## 3. モデルデータ・ジオメトリ整合性

### 3.1 `scripts/fix_collision_geoms.py` の複合関節対応
- **背景**: `assets/fix_collision_geoms.py` を `scripts/fix_collision_geoms.py` へ移動しました。
- **注意点**: スクリプト内の `joint = elem.find('joint')` は1ボディにつき最初の1関節のみを探索しています。
- **確認事項**:
  - ロボットのモデルファイル（`assets/humanoid/humanoid.xml` や `assets/all/all.xml`）に複合関節（1ボディに複数 `<joint>`）や、名前なし中間リンクが存在するか確認してください。
  - 該当する構造が存在する場合、衝突ジオメトリの自動導出処理の拡張が必要です。

### 3.2 `robot/config.py` の定数群の相互整合性
- 物理定数・センサ配置・報酬の重み（`REWARD_WEIGHTS` 等）のバランスは、物理モデルの変更や学習の収束挙動を見ながらチューニングしてください。

---

## 4. 実機環境（Hardware In the Loop）検証

### 4.1 実機関節角度フィードバック（項目6）
- **背景**: `real/real_io.py` の `interleave_read_status()` に `CMD_SERVO_POS_READ` を追加し、`real/real_env.py` で実サーボ角度を観測に反映するようにしました。
- **検証項目**:
  - 実機の通信バスにおいて、サーボ位置のポーリング読み取りが 100Hz 制御ループの許容レイテンシ内に収まること。
  - 受信エラーやタイムアウト時のフォールバック（`smoothed_action` への退避）が正常に機能すること。

### 4.2 実機とシミュレータの観測空間整合性
- `real/real_env.py` の観測構築順序と `envs/mjx_env.py` の観測順序は一致していますが、実機センサの生値レンジ・ノイズレベルがシム側の想定（正規化・スケール）と乖離していないかを実測値で確認してください。

---

## 5. プロジェクト構成・設定ファイルの残課題

### 5.1 `stubs/` ディレクトリの扱い（項目15申し送り事項）
- **背景**: `stubs/board.py` および `stubs/busio.py` は CircuitPython 互換スタブです。
- **現状**: `stubs/` を `PYTHONPATH` に通している設定ファイル（`conftest.py`, `pytest.ini`, `pyproject.toml`, CIスクリプト等）がリポジトリ内に見当たらなかったため、インポート破壊を防ぐ目的で `stubs/` をそのままルート直下に残しています。
- **今後の対応**:
  - 開発環境やCIでの CircuitPython スタブ読み込み方式を確認した上で、必要に応じて `real/stubs/` への移動とパス設定の追加を行ってください。

### 5.2 フロントエンド・可視化ツールのレビュー
- `scripts/collision_tuner.py` のフロントエンド（HTML/JS）部分は静的レビューの対象外でした。GUIの動作確認を行ってください。
