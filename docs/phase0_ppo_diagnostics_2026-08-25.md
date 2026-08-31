# Phase 0 PPO Diagnostics (2026-08-25)

## 1. 実施内容

- version_7 ログから KL 統計を抽出
- `final_params.pkl` から `log_std/std` 系パラメータの存在確認
- Brax PPO 学習コードに `truncation` / `bootstrap` の実装痕跡があるか確認
- `training_progress` 計算ロジックを修正

成果物:
- `log/phase0_ppo_diag.json`
- `envs/mjx_env.py` 修正
## 2. 結果

### 2.1 KL 統計（version_7）
- count: 19
- min: 0.01585
解釈:
- 保存パラメータでは分散キーを直接特定できないが、Braxの学習ログには実効分布統計がある。
- KL爆発の直接原因はstd崩壊ではなく、tanh前のlocとscaleの同時暴走と判断する。
- p95: 5795.01
- `KL > 0.1` 比率: 10.5%


- `final_params.pkl` は tuple(len=3)
  - [1] dict(params)
  - [2] dict(params)
- 再帰探索で `log_std` / `std` / `scale` に該当する明示キーは未検出
- C-04 は「関節別log_std」ではなく「方策分布の実効分散（出力分布側）」を監視対象に切り替える必要がある。

- `policy_dist_max_std`: 0.7783〜15.6330
- KL最大値のステップで `policy_dist_max_std=15.6330`

判定:
- KLスパイクと方策分散の異常拡大が同じ学習系列に存在する。C-04 は異常確認済み、原因未解決。
- 使用中の Brax `losses.py` では `jnp.mean(new_dist.kl_divergence(old_dist))` によりイベント次元を平均している。
- 同じ箇所で `policy_dist_mean_std`、`policy_dist_min_std`、`policy_dist_max_std` を計算している。
- 環境に `terminated`、`truncated`、Brax互換の `time_out` を追加。
- 学習側で `bootstrap_on_timeout=True` を有効化。
- 残作業は実ロールアウトで timeout 時の値ブートストラップを確認すること。

## 3. 実装修正（C-06）

- `_env_steps` を環境内でも単調増加
- `global_step` を `_env_steps` と同期
- `training_progress` を `MAX_EPISODE_STEPS` 基準で再計算しない
- 代わりに `TOTAL_TRAINING_STEPS_ESTIMATE` 基準の進捗を使用

判定:
- C-06 は「実装修正・単体検証済み」。
- `scratch/validate_progress_wrapper.py` を実行し、単調増加と総学習ステップ基準を確認済み。
- C-06 は「実装修正・単体検証済み」。

## 3.1 C-05 向け観測点追加

`envs/mjx_env.py` に以下を追加:

- `info['terminated']`: 姿勢崩れ・低高度などの終了判定

## 4. 次アクション（最短）
3. C-04 は `log_std` 直読を捨て、行動分布サンプルの分散統計（関節別）に切り替え

## 5. Gate A 判定

version_7 の KL 最大値は 52866.55 であり、Phase 0 の健全性条件を満たさない。
したがって、Gate A 本番評価および追加学習の開始は保留する。

## 6. 追加修正

- `envs/mjx_env.py`: `terminated` / `truncated` / `time_out` を `info` に出力
- `train/train_mjx.py`: `bootstrap_on_timeout=True` を指定
- touched files の `py_compile` は成功

## 7. 修正版GPU比較（2026-08-26）

- 成果物: `log/phase0_policy_bounds_gpu/version_0/log.json`、`ppo_diagnostics.json`
- KL最大値: 18418.11（version_7: 52866.55）
- 最大loc: 2.694（version_7: 16.74）
- 最大std: 2.995（上限3.0）
- episode_alive: 最大136.05、その後49.77まで低下、最終57.21
- 判定: クリップは作用したが、KLスパイクと生存時間低下が残る。Gate Aは保留。

## 8. 下限std追加修正（2026-08-26）

- 修正版GPU runのKL最大step=20480で `policy_dist_min_std=0.00283277` を確認。
- KLスパイクの残存原因として、極小scaleが有力と判断。
- `train/train_mjx.py` のscale clipを `[0.001, 3.0]` から `[0.05, 3.0]` に変更。
- `scratch/validate_policy_bounds.py` をmin/max両側テストへ拡張。
- テスト結果: `max_abs_loc=2.970297`、`min_std=0.050000`、`max_std=3.000000`。
- 次回は同一GPU条件で再学習し、KLスパイクとepisode_alive低下を再評価する。

## 9. 下限std=0.05 GPU比較結果（2026-08-26）

- 成果物: `log/phase0_policy_bounds_gpu_min005/version_0/log.json`、`ppo_diagnostics.json`
- KL最大値: 232.03（前回の下限0.001相当: 18418.11）
- KLスパイク: step=20480に1回残存
- KLスパイク時の `policy_dist_min_std`: 0.05019
- `policy_dist_max_loc`: 2.653
- `policy_dist_max_std`: 2.990
- 後半KL: 約0.019〜0.029
- `episode_alive`: step=20480の110.11から後半77.19へ低下

判定:

- std下限0.05は有効で、KLスパイクを約98.7%縮小し、極小stdも除去できた。
- ただし初回KL=232.03は依然として健全域外であり、生存時間低下も残る。
- Phase 0未合格、Gate A保留。次は初回更新だけを抑える学習設定の検証が必要。

## 10. Horizon / evaluation mode check（2026-08-26）

- version_7 と修正版GPU runの学習ログで `episode_length=500` を確認。
- `train/train_mjx.py` のGPU分岐も `episode_length = RobotConfig.MAX_EPISODE_STEPS`（現値500）。
- よって「学習時horizonが評価の500未満」という仮説は今回のrunでは該当しない。
- Brax PPOの `deterministic_eval` は既定値Falseであり、現行呼び出しでは明示していない。
- したがって `episode_alive` 低下は、確率的評価ノイズと方策の長期劣化を分離して再評価する必要がある。
- 報酬側では `r_alive * 25.0` が毎step加算され、最終報酬はclipされるため、報酬値を長期安定性の代替にしない。
