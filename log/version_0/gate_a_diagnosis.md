# Gate A 診断ドラフト（自動生成 — 人間 / Copilotによるレビュー必須）

生成日時: 2026-09-26T12:48:18.509533+00:00
checkpoint: /mnt/c/bipedal_robot/log/version_0/best_params.pkl

この文書は scratch/phase0_eval_diagnostics.py により自動生成された一次判定です。
master_plan.md §3.7 (Task0完了基準) の「§3.6の決定木に基づく主因の暫定結論」
に相当しますが、機械的な閾値ヒューリスティックによる分類であり、
最終結論には人間またはCopilotによるログ・collapse_examplesの目視確認を要します。

- 物理初期姿勢(qpos/qvel)は常にnominal poseに固定されている(未実装機能、master_plan.md付録A §1.6参照)。ここでの'randomized_dr'は質量/摩擦/重心オフセット/サーボ温度/電圧のdomain randomizationのon/offを指す近似軸。
- 現行実装(envs/mjx_rewards.py)はfallen_roll/fallen_pitch/fallen_height/time_limitのみをterminationとして判定する。non_illegal_contact/slip_ok/torque_okによるterminationは未実装(master_plan.md Task4 C-08未着手)。

## 条件別サマリー

### deterministic__fixed_dr
- episode_alive: mean=500.0, std=0.0, n=3
- termination reasons: {'time_limit': 3}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。

### deterministic__randomized_dr
- episode_alive: mean=500.0, std=0.0, n=200
- termination reasons: {'time_limit': 200}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。

### stochastic__fixed_dr
- episode_alive: mean=500.0, std=0.0, n=200
- termination reasons: {'time_limit': 200}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。

### stochastic__randomized_dr
- episode_alive: mean=500.0, std=0.0, n=200
- termination reasons: {'time_limit': 200}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。
