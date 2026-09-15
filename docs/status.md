# 進捗ステータス

最終更新: 2026-09-09（Copilot）

## 現在地

- 完了: 改良案の優先度高項目として、FSR/IMUのセンサ順序誤りと joint order ABI のミスマッチを修正。
- 完了: `reset()` / `step()` 間の metrics pytree 構造を揃え、`reward_is_finite` を追加して JAX トレース時の構造不整合を回避。
- 完了: `RobotConfig.FSR_POSITIONS` を XML の FSR サイト配置に合わせて更新。
- 完了: DR の本体が観測パラメータだけに留まっていた問題を修正し、`body_mass` / `geom_friction` / `body_ipos` を実際の MJX model に反映させるようにした。
- 着手中: 学習の本番検証（GPU / WSL 実行）と、残りの設計レビュー項目（reward gating / evaluation freeze / runtime regression validation）の確認。
- 次: 非 packaged shell での最小回帰実行が可能な環境を確保し、DR と eval contract を実行時に確認してから本番検証へ進む。

## 直近の判定根拠

- `envs/mjx_env.py` の `qpos/qvel` 取得を `actuator_trnid` と `jnt_qposadr` / `jnt_dofadr` ベースに修正し、シミュレーションと実機の関節順序 ABI が揃うようにした。
- `envs/mjx_rewards.py` の FSR 取得を `sensordata[10:18]`（IMU後の 8ch FSR）へ修正し、左右足の接触判定が実際の XML 順序に合わせるようにした。
- `reset()` に `reward_is_finite` を追加し、JAX の metrics pytree 構造不一致を抑止した。
- VS Code の静的エラー検査では 3 ファイルとも `No errors found` となった。
- 実行確認は、ターミナルが Windows の packaged PowerShell / bash を直接起動できず、`pytest` と Python 実行が sandbox 内で失敗したため未完了。詳細は下記のエスカレーション項目を参照。

## エスカレーション中の項目

- 実行環境の制約: `run_in_terminal` は packaged MSIX の `powershell.exe` / `bash.exe` を起動できず、Python の実行テストが実施できない。
- このため、`pytest` の成否確認は未実施。現時点では静的検査のみを根拠としている。
- 次回は、非 packaged な shell / Python 実行環境が使える形に切り替えた上で、最小回帰テストを実行してから本番検証へ進む。
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