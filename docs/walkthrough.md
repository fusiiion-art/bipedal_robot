# FSRセンサー修正 & Reward Audit誤検知修正 ウォークスルー

## 1. 修正の概要

[fix-fsr-sensor-and-reward-audit.patch](file:///c:/bipedal_robot/docs/fix-fsr-sensor-and-reward-audit.patch) および [FSR修正とreward-audit修正_調査報告.md](file:///c:/bipedal_robot/docs/FSR修正とreward-audit修正_調査報告.md) に基づき、以下の修正を適用・検証しました。

### ① FSRタッチセンサーサイトのZ座標ミスアライメント修正
- **対象ファイル**: [assets/humanoid/humanoid.xml](file:///c:/bipedal_robot/assets/humanoid/humanoid.xml)
- **変更内容**:
  - 足裏の FSR サイト（左足4点 `l_foot_fsr_*`、右足4点 `r_foot_fsr_*`、計8点）がソール底面（Z=-0.0200）より 3.4cm 下（Z=-0.0541）の床下空間にあったため、常に無接触（0.0）を返していた不具合を修正。
  - Z座標をソール底面に合わせて `-0.0200` に修正。
  - MuJoCo の box-vs-plane 接触判定で生成される4隅の接触点を確実に捉えられるよう、サイトの size を `0.005 0.005 0.001` から `0.013 0.014 0.003` に拡大。
  - X, Y座標は [RobotConfig.FSR_POSITIONS](file:///c:/bipedal_robot/robot/config.py) の ABI contract に合致したまま保持。

### ② Reward Audit「Potential Decay」誤検知の修正
- **対象ファイル**: [train/train_mjx.py](file:///c:/bipedal_robot/train/train_mjx.py)
- **変更内容**:
  - `_audit_reward_metrics()` の Potential Decay 判定において、エピソード累積値ではなく 1 ステップ平均値（`_per_step()`）に正規化。
  - PBRS（割引率 $\gamma=0.99$）における理論的減衰期待値 `(gamma - 1.0) * max(potential_per_step, 0.0)` を算出し、その理論値からの乖離 `pbrs_deviation < -0.5` かつ `delta_potential_per_step > 0.01` の場合のみ異常アラートを発報するよう修正。

---

## 2. 検証結果

### (1) シミュレーションでの FSR 接触検出およびメトリクス判定
シミュレーション接地状態での検証スクリプトを実行：
```
=== Testing FSR Contact in Simulation ===
FSR readings (8ch): [16.275799  13.711314   8.826716   6.2622533 11.875049   9.733765  5.5599117  3.418635 ]
Left foot mean force: 11.269 N (threshold: 0.05)
Right foot mean force: 7.647 N (threshold: 0.05)
both_feet_contact: True
```
- 全8チャンネルで体重を支持する正常な床反力（各 3.4N 〜 16.3N）が検出されました。
- `both_feet_contact`（報酬重み 8.0）が正しく `True`（1.0）として判定されることを確認。

### (2) Reward Audit 判定テスト
- **通常ケース（学習に伴いエピソード長・累積値が増加）**: アラート 0件（誤検知解消）
- **通常成長ケース（potential/step が健全に成長）**: アラート 0件
- **異常ケース（potential と乖離して pbrs_reward が異常低下）**: アラート 1件正しく発報（検出能力を維持）

### (3) 全体 pytest スイート
```
38 passed, 19 warnings in 462.10s (0:07:42)
```
- 既存の全38テスト（センサー ABI contract、アクチュエータゲイン、学習統合テスト含む）が全て PASS。リグレッションはありません。
