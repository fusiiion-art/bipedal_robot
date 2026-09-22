# FSRタッチセンサー不具合 / reward audit誤検知 調査・修正報告

- **対象リポジトリ**: `fusiiion-art/bipedal_robot`
- **調査のきっかけ**: `docs/status.md`, `docs/pytestと学習結果.md` に記載された、`log/version_9` 学習実行時の reward audit 警告151件
- **修正コミット**: `dbff443` (2026-09-22)
- **検証環境**: `requirements-lock.txt` 準拠 (JAX 0.11.0 / MuJoCo・MJX 3.11.0 / Brax 0.14.2), CPU

---

## 1. 発見した実バグ: FSRタッチセンサーサイトのZ座標ミスアライメント

### 症状
`envs/mjx_rewards.py` の `both_feet_contact` 判定（`REWARD_WEIGHTS["both_feet_contact"] = 8.0`）が、学習全体を通じて常に `False`。観測空間625次元中のFSR8要素・ZMP2要素も常にゼロ相当。

### 根拠
`assets/humanoid/humanoid.xml` のFSRタッチサイト（`l/r_foot_fsr_{fl,fr,bl,br}`）が、足裏の衝突ジオメトリ（ソールbox、ローカルZ底面 `-0.0200`）より **3.41cm下（`-0.0541`）** に置かれていた。この位置は接触が発生し得ない床下の空間にあり、8ch全てのタッチセンサーが常時0を返していた。

`git log` で確認したところ、このサイト定義は **2026-08-31にFSRセンサーが実装されて以来一貫してこの値**であり、`version_9` に限らずこれまでの全学習run（少なくともFSR実装以降）に影響していたと見られる。

追加調査として、MuJoCoのbox-vs-plane接触判定はソールboxの面全体ではなく **4隅にのみ接触点を生成する**ことを直接シミュレーションで確認した。単純にZ座標だけを底面へ合わせても、サイトの検出ボックスが小さいままだと隅の接触点を捉えられず、依然として0のままだった。

### 修正内容
X, Y座標は `RobotConfig.FSR_POSITIONS`（`tests/test_sensor_contract.py` が検証する既存contract）と一致させたまま変更せず、Z座標をソール底面 `-0.0200` に合わせ、サイズを対応する隅を確実に捉えられる大きさ（`0.005 0.005 0.001` → `0.013 0.014 0.003`）に拡大した。

```diff
-<site name="l_foot_fsr_fl" pos="0.012 0.027 -0.0541" type="box" size="0.005 0.005 0.001" .../>
+<site name="l_foot_fsr_fl" pos="0.012 0.027 -0.0200" type="box" size="0.013 0.014 0.003" .../>
```
(8サイト全て同様に修正)

### 検証結果

実際の学習パイプライン (`SenpuuMaruMJXEnv.reset`/`step`) を通して比較:

| | 修正前 | 修正後 |
|---|---|---|
| FSR (8ch) | `[0,0,0,0,0,0,0,0]` | `[1.69,1.29,4.13,1.46,1.54,0.68,1.51,2.95]` |
| `both_feet_contact` メトリクス | `0.0` | `1.0` |

純MuJoCo単体でも同様の結果（総反力が体重の約65%相当を検出、4秒間安定）を確認。

---

## 2. reward audit「Potential Decay」誤検知の修正

### 症状
`docs/pytestと学習結果.md` に記載の「Potential Decay」警告17件（`compute_potential()` の符号ミスを疑う内容）。

### 根拠
`train_mjx.py` の `_audit_reward_metrics()` 内、Potential Decay判定が `eval/episode_potential` / `eval/episode_pbrs_reward` の**エピソード累積値**をそのままstep間比較していた。episode_alive（平均生存長）が学習進行とともに伸びる（131→500）だけで累積値は見かけ上増大し続け、必ず誤検知する構造だった。

さらに、`gamma=0.99` のPBRSでは potential がエピソード内でおおむね一定でも、割引の性質上 `r_pbrs` の総和が `(gamma-1)×potential` 程度の負値に収束するのが**設計通りの正常な挙動**であり、「pbrs_rewardが負」であること自体は異常の兆候ではない。

なお、他の3カテゴリ（Exploitation / Metric Corruption / Action Distortion）は既に `avg_ep_len` による1step正規化が実装済みで、これは今回の調査以前から対応済みだったことを確認した（`pytestと学習結果.md` は正規化修正前の古い実行結果を記録したまま残っていたため、文書上は未解決に見えていた）。

### 修正内容
potential / pbrs_reward を1step平均値に正規化し、割引による理論的な減衰量 `(gamma-1)×potential_per_step` からの乖離のみを異常判定の対象にするよう変更（詳細は `dbff443` のコミットメッセージ参照）。

### 検証結果
`log/version_9/log.json` に対し、修正後の監査関数で再実行:

| カテゴリ | 修正前(記録上) | 現行コードで再監査 |
|---|---|---|
| Exploitation | 94 | 0 |
| Metric Corruption | 20 | 0 |
| Action Distortion | 20 | 0 |
| Potential Decay | 17 | 0 |
| **合計** | **151** | **0** |

異常注入テスト（`pbrs_reward` を potential と無関係に大きく負へ改変）でも正しく検知されることを確認し、検出力そのものは維持されていることを確認済み。

---

## 3. テスト結果

既存pytest全38件、両修正を適用した最終状態で再実行し全てPASS（回帰なし）。`tests/test_sensor_contract.py` のFSR座標contractテストも含む。

---

## 4. 重要な注意点（未検証・要判断事項）

- **`both_feet_contact`（重み8.0、`alive`・`upright`・`com_stab`に次ぐ4番目の大きさ）は、FSR実装以来一度も学習で機能していなかった。** 今回の修正でこれが初めて有効になるため、単なるバグ修正というより報酬関数の形が実質的に変わるレベルの変更である。
- `version_9` の学習済みモデル（`final_params.pkl`）は本修正を反映せずに学習されたものであり、修正を適用してもモデル自体が遡って改善されるわけではない。恩恵を受けるには再学習が必要。
- このチェックポイントからウォームスタートする場合、序盤で学習が不安定化したりcurriculum進行が変わったりする可能性がある。
- `status.md` の次ステップ「Gate 0評価（`scratch/gate0_eval.py --mode policy`）」は未実施。
- 本調査は `docs/` 記載の症状を起点にした調査であり、コードベース全体の網羅的監査ではない。他の潜在バグの有無は保証しない。
- 学習が実際にGate A（外乱下で両足接地のまま直立維持）に到達するかは、再学習を実施して初めて判断できる。

## 5. 次のアクション（提案）

1. パッチ (`fix-fsr-sensor-and-reward-audit.patch`, コミット `dbff443`) を適用してゼロから再学習
2. 学習序盤から `eval/episode_both_feet_contact` が0でなくなっているか確認
3. reward auditが今回のような誤報を再燃させないか確認
4. `version_9` と比較して学習曲線の荒れ具合を確認（荒れても新しい報酬項への適応過程の可能性があり、即座に異常とは限らない）
5. 収束後、`scratch/gate0_eval.py` でGate 0評価を実施し人の目で合否判断
