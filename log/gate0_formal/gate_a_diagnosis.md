# Gate A 診断ドラフト（自動生成 — 人間 / Copilotによるレビュー必須）

生成日時: 2026-10-03T02:10:21.549725+00:00
checkpoint: zero-policy

この文書は scratch/phase0_eval_diagnostics.py により自動生成された一次判定です。
master_plan.md §3.7 (Task0完了基準) の「§3.6の決定木に基づく主因の暫定結論」
に相当しますが、機械的な閾値ヒューリスティックによる分類であり、
最終結論には人間またはCopilotによるログ・collapse_examplesの目視確認を要します。

- 全セルで初期状態分布(master_plan.md §1.6)を適用: 関節角 ±0.05 rad, 関節角速度 ±0.2 rad/s。'fixed_dr'/'randomized_dr' は質量/摩擦/重心オフセット/サーボ温度/電圧のdomain randomizationのon/off。
- 現行実装(envs/mjx_rewards.py)はfallen_roll/fallen_pitch/fallen_height/time_limitのみをterminationとして判定する。no_illegal_contact/slip_ok/torque_ok/height_ok/uprightはterminationではなく、本スクリプトのsuccess判定(master_plan.md §0.3の論理積、criteria_pass_rate)で評価する。

## 条件別サマリー

### deterministic__fixed_dr
- success: 1/3
- criteria pass rate: {'alive': 1.0, 'both_feet_contact': 0.3333333333333333, 'upright': 1.0, 'slip_ok': 1.0, 'torque_ok': 1.0, 'height_ok': 1.0, 'no_illegal_contact': 1.0, 'success': 0.3333333333333333}
- foot displacement [mm]: p50=4.1, p95=6.0, max=6.3
- jitter (判定外): action_change_rms p50=0.0000, joint_vel_rms p50=0.041 rad/s
- episode_alive: mean=500.0, std=0.0, n=3
- termination reasons: {'time_limit': 3}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。

### deterministic__randomized_dr
- success: 48/200
- criteria pass rate: {'alive': 1.0, 'both_feet_contact': 0.24, 'upright': 1.0, 'slip_ok': 1.0, 'torque_ok': 1.0, 'height_ok': 1.0, 'no_illegal_contact': 1.0, 'success': 0.24}
- foot displacement [mm]: p50=4.4, p95=9.4, max=11.9
- jitter (判定外): action_change_rms p50=0.0000, joint_vel_rms p50=0.038 rad/s
- episode_alive: mean=500.0, std=0.0, n=200
- termination reasons: {'time_limit': 200}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。

### stochastic__fixed_dr
- success: 49/200
- criteria pass rate: {'alive': 1.0, 'both_feet_contact': 0.245, 'upright': 1.0, 'slip_ok': 1.0, 'torque_ok': 1.0, 'height_ok': 1.0, 'no_illegal_contact': 1.0, 'success': 0.245}
- foot displacement [mm]: p50=4.4, p95=9.1, max=13.4
- jitter (判定外): action_change_rms p50=0.0000, joint_vel_rms p50=0.038 rad/s
- episode_alive: mean=500.0, std=0.0, n=200
- termination reasons: {'time_limit': 200}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。

### stochastic__randomized_dr
- success: 48/200
- criteria pass rate: {'alive': 1.0, 'both_feet_contact': 0.24, 'upright': 1.0, 'slip_ok': 1.0, 'torque_ok': 1.0, 'height_ok': 1.0, 'no_illegal_contact': 1.0, 'success': 0.24}
- foot displacement [mm]: p50=4.4, p95=9.4, max=11.9
- jitter (判定外): action_change_rms p50=0.0000, joint_vel_rms p50=0.038 rad/s
- episode_alive: mean=500.0, std=0.0, n=200
- termination reasons: {'time_limit': 200}
- failure timing: no_failures
- suggested action: terminatedによる失敗episodeが観測されなかった。time-limit到達のみであれば§3.3(truncation/termination処理)の疑いは後退し、他の症状(KLスパイク等)の切り分けを優先する。
