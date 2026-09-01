# 外乱耐性直立制御 — 統合実行計画 (Master Plan v7)

作成日: 2026-08-27 改訂履歴:

- 2026-08-28: standing\_robustness\_plan\_v4.mdの全文を付録Aとして本文書に統合。外部ファイル参照ではなく本文書単体で完結する構成に変更（standing\_robustness\_plan\_v2.mdは本文書作成者が原文を保持していないため未統合。統合が必要な場合は原文を提供してください）  
- 2026-08-28: 参照セクションが一部欠落していたため、master\_plan.md（v3）を土台文書として明記し復元  
- 2026-08-28: Task 1にtarget\_kl early stopping（保険）を追加。std下限クリップ修正が根本原因への対処である一方、KL健全化の二段目の防波堤として低コストで追加できるため。  
- 2026-08-28: タスク8に実機質量計測の前提条件を追加。各STLパーツ（特にモータ・センサー・バッテリー搭載部）の実測重量が未確定なため、Phase \-1算出値は実測完了まで暫定扱いとする。 他の内容はv3から変更なし。 最終目標: **外乱（突き・押し）を受けても転倒せず、両足接地の直立姿勢を維持する** 参照: `master_plan.md`（v3、本文書の土台。2026-08-27作成、Claudeとの検討を経てv4→v7へ改訂）、`standing_robustness_plan_v4.md`（詳細仕様。付録Aとして全文統合済み）、`standing_robustness_plan_v2.md`（Gate定義の一次ソース。原文未保持のため未統合、本文中の言及のみ）、`課題管理.md`（課題ID台帳）

## 0\. この文書の位置づけ

本文書は`v2`の決定事項を継承し、その**上位の実行ルール**として機能します。Copilotはこの文書に従って自走し、Claudeへの相談は9章の基準に該当する場合のみ行ってください。`standing_robustness_plan_v4.md`の詳細仕様（報酬式の正規化制約、統計設計、実機安全Gate等）は**付録A（本文書末尾）として全文統合済み**であり、外部ファイルを別途参照する必要はありません。`v2`は原文を統合できていないため、本文中の言及（Isaac Lab不採用等の決定事項）のみを引き継いでいます。

**v4の§1.2/§6.2にあった「傾斜床対応」は、実際のロボワンのルールと一致しないため不採用と確定しました（2026-08-27）。** 傾斜床関連の実装（動的プラットフォーム、床法線基準の座標系拡張）は行いません。姿勢判定はworld鉛直基準のみで統一します。もし`v4`由来のコードやテンプレートに床傾斜関連の記述が残っていれば、着手前に削除して構いません。

各作業は完了時に必ず: 実行日時・git commit hash・コマンド・設定・seed・ログへのリンク・合否・次工程への根拠、を記録すること。

---

## 1\. 現状サマリー（2026-08-27時点で確定している事実）

### 確定して良い（再検証不要）

- 物理接地: `torso z=0.1773`で足裏誤差0.00004m、CoM(x,y)は支持基底内。8秒間のPD単体シミュレーションでroll/pitchが発散せず収束することを確認済み（純MuJoCo検証）  
- Gate 0-P（物理ベースライン、非公式）: 純MuJoCoフォールバックで10秒・seed=0、max\_roll=2.155°、max\_pitch=0.340°、両足接地率1.0、トルク飽和率0。**これは学習済みRLポリシーの正式なGate 0合格ではない**（4章参照）  
- `time_out`フィールドの配線: Brax本体の要求仕様(`state.info['time_out']`\+`done=True`)と一致していることをソースコードで確認済み  
- `mean_clip_scale`と`std`上限クリップ(3.0)は実装・検証済みで、実際のGPU学習でも`max_loc=2.694`, `max_std=2.995`と設計通りに機能している  
- `train/networks.py`の`AdaptationModule`/`BasePolicy`/`TeacherPolicy`（RMAアーキテクチャ）は現在の学習に一切使われていない（importされているだけの死んだコード）。Gate A合格まで着手しない  
- `bootstrap_on_timeout`が使う値は`V(s_T)`（打ち切り時点の状態）ではなく、Brax実装上**1ステップ前の`V(s_{T-1})`**（`policy_extras['value']`、遷移前の観測から計算）である。auto-resetによる観測汚染の心配はこの経路には無い（ソース確認済み、2026-08-27）。ただし標準の`truncation`ベースのGAE経路は別途タスク2の修正が必要

### 未解決・最優先（2章で詳細）

- KLが依然として壊滅的に大きい（直近run: 最大18418.11、健全域は0.02〜0.05）  
- `episode_alive`が学習後半で低下し続ける（直近run: 136.05 → 49.77 → 57.21）  
- `done = terminated OR truncated`が、Braxの`EpisodeWrapper`のtruncation自動計算を構造的に無効化している（詳細はタスク2）  
- 評価時にobservation normalizerの統計が学習時と正しく凍結・共有されているか未確認（新規、タスク5）  
- 純MuJoCo(Gate 0-P確認済み)とMJX(実際の学習環境)の数値的整合性が未確認（新規、タスク6）  
- 各STLパーツ（特にモータ・センサー・バッテリー搭載部）の実機重量が未計測。Phase \-1（タスク8）の物理限界値算出に影響（新規、タスク8）

---

## 2\. 直近タスク（実行順。安価なCPU検証を先に、GPU学習は最後にまとめて実施する）

### タスク1: 分布std下限クリップの修正 \+ KL安定化の保険 【CPU検証のみ】

**根拠**: `scale = jnp.clip(scale, self._min_std, 3.0)`は上限のみ修正済みで下限`0.001`のまま。KL式`(scale_old² + Δloc²)/(2*scale_new²)`は分母が小さいほど爆発するため、`max_std`が正常でも別次元の`min_std`が0.001付近まで潰れていればKLは爆発したままになる。直近runのKL最大値18418.11は、方策分布のどこかの次元がほぼ決定的（std≈0）まで潰れて更新が数値的に破綻していたことを示唆する。

**作業**:

1. 直近ログのKL最大step付近で`training/policy_dist_min_std`を確認（仮説の裏付け）  
2. `min_std`を`0.001`から`0.05`へ引き上げる  
3. `scratch/validate_policy_bounds.py`に**min側のアサーション**を追加（現状max側のみ）  
4. `training/entropy`と`training/value_loss`がログに出ているか確認し、無ければ追加  
5. **（保険・新規）** `target_kl`によるepoch内early stopping（PPO実装が対応していれば有効化、無ければ追加実装）を導入する。目安値は0.01〜0.02。std下限修正がKL爆発の根本原因への対処である一方、この修正だけで健全域(0.02〜0.05)に収まらなかった場合の二段目の防波堤として、低コストなため合わせて入れておく。学習スクリプトに`--target_kl`引数が無ければ追加する（`--seed`と同様の扱い）  
6. 上記5を追加した場合、`training/kl_early_stop_triggered_ratio`（1 iterationのうち何割のミニバッチ更新がtarget\_kl超過でスキップされたか）をログに追加する。std修正が効いていればこの比率はほぼ0になるはずで、高止まりする場合はstd修正だけでは不十分という追加のシグナルになる

### タスク2: `done`/`truncation`/`time_out`の配線修正 (C-05、Phase 0必須) 【CPU検証のみ】

**根拠**: `mjx_env.py`は`done = terminated OR truncated`をStateとして返している。Brax `EpisodeWrapper`は`state.info['truncation'] = where(steps>=episode_length, 1-state.done, 0)`を**無条件に上書き**するため、環境が先に`info['truncation']`へ何を書き込んでも意味がない。`state.done`が既にTrueだと常に0になる。`discount = 1-state.done`（`acting.py`）も同様に常に0。結果、時間切れも転倒も同じ「真の終端」として扱われ、GAEが時間切れ時に`V(s_T)`ではなく0でbootstrapする。

**正しい修正**:

1. `mjx_env.py`の`step()`で、Brax Stateへ返す`done`を\*\*`terminated`のみ\*\*にする  
     
2. `info['time_out']`は引き続き`truncated`から計算（`bootstrap_on_timeout`が読む別経路）  
     
3. `info['terminated']`/`info['truncated']`はログ用にそのまま残す  
     
4. 3ケースの単体テストでGAEに渡る値を直接検証する（フィールドの存在確認だけで合格としない）:  
   

| ケース | 期待`done`(State) | 期待`truncation`(EpisodeWrapper後) |
| :---- | :---- | :---- |
| 通常遷移 | False | 0 |
| 転倒終了 | True | 0 |
| 時間切れ | False→`EpisodeWrapper`が後段でTrue化 | 1 |

   

5. 決定論性テストも追加する: 同一checkpoint・同一seed・同一入力で評価結果が再現することを確認（v4§4.2）

### タスク3: C-07（外乱無効設定の完全性） 【CPU検証のみ】

`DISTURBANCE_CURRICULUM=False`のとき、物理へ渡る外力配列（`xfrc_applied`相当）が全step・全env・全seedで厳密に0であることを単体テストで保証する。ログ値だけでなく物理更新に渡る配列そのものを見ること。`eval/episode_curriculum_scale`はエピソード内合計値なので、解釈時は必ず`episode_alive`で割る。

### タスク4: C-08（報酬・生存の複合成功条件チェック） 【CPU検証のみ】

3状態（安定直立／軽度傾き／転倒直後）で1ステップ報酬を単体計算し`reward_standing > reward_tilted > reward_fallen`を確認する。あわせて以下の**不変条件**をランダムサンプル`10^4`個の(state, action)対で自動検証するテスト（`tests/test_reward.py`）を用意する:

- 任意の(state, action)で1ステップ報酬の下限が正であること（`r_min > 0`。alive bonus撤廃時に生じる「早く転んだ方が得」という逆方向ハックを防ぐ）  
- 転倒による終端ペナルティは`terminated`のみに適用され、`truncated`には適用されないこと（タスク2との整合）

また、成功条件を`episode_alive`単体ではなく以下の論理積で再定義する（`configs/standing/success_criteria.yaml`に数値定義。本書ではなく設定ファイルを正とする）:

success \= alive(T) AND upright AND height\_ok AND no\_illegal\_contact

          AND slip\_ok AND torque\_ok AND (外乱時) recovered\_in\_time

### タスク5: 評価時のobservation normalizer凍結確認 【新規・CPU検証のみ・優先度高】

**根拠**: `normalize_observations=True`を使用中。評価（deterministic eval）時にrunning mean/std統計が学習時のものと正しく凍結・共有されているか、evaluation中に更新され続けていないかは未確認。もしeval時に統計がまだ収束していない、または学習用と異なる統計を使っていると、「学習後半で評価成績が落ちる」という現在の症状（`episode_alive`の低下）と区別がつかない見かけ上のバグになりうる。タスク1・2よりコストが低く、原因の切り分けに直結するため優先度を上げる。

**作業**: 評価呼び出し前後でnormalizer統計のパラメータを比較するテストを書き、evaluation中に変化しないことを確認する。

### タスク6: Gate 0.5（純MuJoCo–MJX整合性確認） 【新規】

**根拠**: Gate 0-Pは純MuJoCo（CPU、フル精度）での確認であり、実際の学習はMJX（GPU、異なるsolver/接触処理）で行っている。接触の多い二足姿勢制御ではバックエンド差が結果に影響しうる。Gate 0-Pの物理健全性がMJX側でも成立している保証がまだ無い。

**作業**: 同一初期状態・同一PD指令列（外乱ゼロ、タスク1〜4の修正は無関係）で純MuJoCoとMJX双方をrolloutし、`qpos`/`qvel`/base姿勢/joint torque/contact state/termination時刻を比較する。完全一致は要求せず、短時間誤差とイベント一致率に許容値を設定して記録する。

### タスク7: GPU学習（タスク1〜6を全て反映した状態で1本にまとめて実行）

**Debug PASS（1 seed、方向性確認）**:

- 3章の標準コマンドをseed=0で実行  
- 合格基準: `kl_mean`が全区間で1.0未満、`policy_dist_min_std >= 0.05`、NaN/Infなし、`episode_alive`が明確に崩壊しない

Debug PASSを満たしたら\*\*Qualification PASS（3 seeds: 0,1,2）\*\*を実行し、4章のPhase 0完了基準で判定する。

### タスク8: Phase \-1（物理限界）とスコープ確定 【並行実施可、ただしGate A合格前に必須】

**前提条件（実機質量計測、2026-08-28追加）**: 各STLパーツ、特にモータ・センサー・バッテリーなど質量が集中する部品は、STLの体積×材質密度だけでは実際の重量と一致しない。実測（可能な限りアッセンブリ単位で計量。個別部品をバラで積み上げるより速く誤差も少ない）が完了するまで、以下1〜3で算出する数値はすべて**暫定値**として扱う。`docs/physical_limits.md`には実測完了まで「暫定（実機質量計測前）」と明記すること。安価な検算として、完成機体の総重量を1回はかりで測りsimの合計質量（各bodyのmass合計）と突き合わせる、および紐で吊るす等の簡易法で重心高さ`h`を概算しsimの計算値と比較する、の2点は今すぐ実施できる。実測完了・モデルの各bodyの`<inertial>`更新後に1〜3を再計算し、値を確定させる。**この確定なしにGate Bの外乱目標（項目2〜3）を最終決定してはならない。**

1. トルク限界・支持余裕・摩擦から`F_max`、`v_cap`、`J_max`(N·s)、`θ_max ≈ min(arctan(μ), arctan(d/h))`を算出し`docs/physical_limits.md`に記録する  
2. 目標とする外乱スペック（突っつきインパルス\[N·s\]、静的傾斜角\[deg\]があれば）を数値で確定し、Phase-1限界値との\*\*余裕度（限界/目標）\*\*を計算する  
3. **余裕度が1.0を下回る項目がある場合**、現在の`standing_fixed_feet`（足を動かさず腰・足首戦略のみで復帰）のままでは目標に届かない可能性が高い。この場合はGate Aへは進んで良いが、**Gate Bには進まず、エスカレーション基準7章-4に従って人間の判断を仰ぐ**（目標値の見直し、または`standing_with_recovery_step`への制約緩和）

---

## 3\. 標準実行コマンド・環境ルール

### GPU学習（実績のあるコマンド、これを標準とする）

実行前に`git rev-parse HEAD`と`git diff`を保存し、`--exp_name`には変更内容・seed・日付・commit短縮hashを含める。`--seed`引数が未実装ならまず追加する。

wsl bash \-lc '

cd /mnt/c/bipedal\_robot

git rev-parse HEAD \> log/\<exp\_name\>\_commit.txt

git diff \> log/\<exp\_name\>\_diff.patch

export PYTHONPATH=/mnt/c/bipedal\_robot

export XLA\_PYTHON\_CLIENT\_PREALLOCATE=false

export XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.7

ps \-eo pid,etime,cmd | grep "\[t\]rain\_mjx" || true

/mnt/c/bipedal\_robot/venv\_wsl/bin/python /mnt/c/bipedal\_robot/train/train\_mjx.py \\

  \--num\_envs 128 \--batch\_size 128 \--num\_minibatches 8 \--learning\_rate 5e-5 \\

  \--steps 200000 \--seed \<seed\> \--exp\_name \<exp\_name\>'

※ タスク1-step5で`target_kl`を追加した場合、上記コマンドに`--target_kl 0.02`（目安値、既存ログのKLスパイク幅を見て調整）を追加すること。`--target_kl`引数が未実装ならまず追加する（`--seed`と同様の扱い）。

### 即時中断してよい条件（GPU runを最後まで待たずに止めてよい）

- NaN/Infが1件でも出た  
- `policy_dist_min_std`が0.05を明確に下回り続ける、または`max_std`が3.0を超え続ける  
- `kl_mean`が100を超える状態が複数回連続する  
- `DISTURBANCE_CURRICULUM=False`のはずが外力が非ゼロ

これら以外は最後まで走らせてログを記録してから次の仮説へ進むこと。

### 禁止事項

- **CPUでのフルスケール学習検証は行わない**（単体テスト・形状チェックはCPUで良い）  
- **120秒応答が無いだけでプロセスを強制終了しない**。バックグラウンド実行＋`log.json`のタイムスタンプ更新で生存確認する  
- 新規プロセス起動前に`ps -eo pid,etime,cmd | grep train_mjx`で孤立プロセスが無いか確認する  
- 合格checkpointを上書きしない。性能が低下したらrollbackする  
- 成功基準そのものを自動変更しない。外乱上限を解析限界(`J_max`)以上へ自動拡大しない  
- 実機コマンドを自動実行しない

### 推奨: JAXコンパイルキャッシュの有効化（バックエンド初期化前に設定すること）

import jax

jax.config.update("jax\_compilation\_cache\_dir", "/mnt/c/bipedal\_robot/.jax\_cache")

jax.config.update("jax\_persistent\_cache\_min\_compile\_time\_secs", 0\)

WSL環境では`/mnt/c/`（Windows側マウント）はI/Oが遅いため、キャッシュ肥大化で速度低下を感じたらWSL側ネイティブファイルシステム（`~/.jax_cache`等）への変更も検討する。

### Copilot運用ルール

- 1 iteration \= 「1つの仮説を選ぶ→変更を1カテゴリに限定→検証1本→記録」。報酬とPPOハイパーパラメータの同時変更は禁止  
- 座標系・命名規則・本文書の禁止事項など不変ルールは`.github/copilot-instructions.md`にも複製しておく（長大な単一文書はセッション途中でコンテキストから落ちやすいため、Copilotが自動参照するこのファイルに要点を残す）  
- 可能な限りDone条件をpytest関数に落とす。Markdownのチェックボックスだけで合格としない

---

## 4\. フェーズとGate

Phase \-1 (物理限界・スコープ確定) → 2章タスク8

Phase 0  (PPO健全性)             → 2章タスク1〜7。Debug PASS→Qualification PASSの2段階

Gate 0-P (物理ベースライン)       → 純MuJoCo/PD、無外乱10-30秒。確認済み(1章)

Gate 0.5 (MuJoCo-MJX整合性)       → 2章タスク6

Gate 0   (学習済み方策の無外乱10-30秒) → Phase 0 Qualification PASS後、実際のRLポリシーで実施

Gate A   (無外乱500step)          → 統計設計は下記参照

Gate B-0 (外乱インフラ検証)       → Gate A合格後。xfrc\_applied反映・impulse実測値・座標系(world/body frame)を確認

Gate B   (軽外乱curriculum)      → Gate B-0合格後。力積J(N·s)で定義、J\_maxから設定

Gate C   (試合想定外乱・sim2real較正) → Gate B合格後

Gate D   (実機移行・安全Gate)     → Gate C合格後

**重要な区別**: 「Gate 0-P」（1章）は純MuJoCo/PD制御による物理モデルのsanity checkであり、**学習済みRLポリシーの正式なGate 0合格ではない**。学習済み方策でのGate 0は、Phase 0 Qualification PASS後に別途実施する。

**Phase 0完了（Gate 0/Gate Aへ進んで良い）条件**: 2章タスク7のQualification PASS（3 seeds）で、以下を全seed同時に満たすこと:

1. `kl_mean`が全区間で0.1未満（単発スパイクも1.0未満）  
2. `episode_alive`の移動窓比率 \= （末尾20%区間平均）÷（最高20%区間平均） ≥ 0.7  
3. `policy_dist_min_std >= 0.05` かつ `policy_dist_max_std <= 3.0`（全区間）  
4. `value_loss`に発散スパイクがない  
5. NaN/Infなし

**Gate A以降の統計設計（v4を採用、私の以前の「60/60」より精緻）**:

- 学習seed ≥ 3（理想5）、各学習seedにつき評価エピソード n ≥ 200、評価条件は学習用randomizationと別系列で凍結（held-out）  
- 判定: 各学習seedの成功率のWilson score 95%信頼区間の下限が閾値以上、**かつ全学習seedで達成**（平均ではなくmin基準）  
- 主判定はdeterministic評価、stochasticは参考指標  
- 成功条件は2章タスク4の複合定義を使う（`episode_alive`単体では合否判定しない）  
- 開発中の探索的な最低ラインとしては20/20完走で先へ進んで良いが、Gate B着手前には上記の統計設計で正式再評価する

**Gate B設計時に反映すること（v4から、実装は今しない。方針だけ記録）**:

- 外乱は力積`J = F·Δt`で定義し、印加方向8方位・タイミング複数種のグリッド評価で「成功率50%となる外力`J_50`」を主要指標にする  
- 突っつき後の一時的な片足浮き・荷重低下に猶予を持たせる終了条件にする（「両足の鉛直接触力が同時に閾値未満、かつ連続20ms以上」等）。即terminationにすると正常な腰戦略での復帰まで失敗扱いにしてしまう。現行の`MAX_SINGLE_FOOT_LIFT=0.0`は「意図的な踏み出し禁止」であって「瞬間的な荷重変動の禁止」ではないことを明確にする  
- **catastrophic forgetting対策**: 常に外乱なし／既習の弱い外乱／現在段階／過去段階を一定割合（例30%）混ぜる。各curriculum段階終了時にGate A条件を再評価し、成功率が規定値以上低下したら1段階戻す  
- actor観測とcritic観測を非対称にする（asymmetric actor-critic）かどうかは、Phase \-1の余裕度確認（タスク8）の結果を見てから判断する。理由: 突っつきは方策側から観測不能な外力であり、criticのみに真の外力ベクトル・真の重心速度等の特権情報を与えるとvalue推定の分散を下げられる可能性がある。ただしこれはネットワーク入力の構造を変える変更であり、Gate B開始後に追加すると学習済み方策が無価値になるため、**やるならGate A是正の時点で構造だけ確定させる**（現時点では方針の記録のみ、実装はしない）

**後回しにしてよい項目（Gate A合格まで着手しない）**:

- RMAアーキテクチャの実装への接続

---

## 5\. 完了記録のフォーマット

各タスク完了時、`課題管理.md`へ追記:

\- 実行日時 / git commit hash

\- 実行コマンド / 設定・seed

\- 生ログ・モデル・算出表へのリンク

\- 合否と判定根拠

\- 次工程への根拠、または次の反証可能な仮説（「改善しなかったので別設定を試す」は不可。観測→診断→仮説→次の単一変更→棄却条件、の形式で書く）

---

## 6\. 付録: 判明しているBrax内部仕様（再調査不要、2026-08-27時点でソース確認済み）

- **`NormalTanhDistribution`**（`brax/training/distribution.py`）: `scale = (softplus(raw) + min_std) * var_scale`。`min_std`のデフォルトは`0.001`で上限は元々存在しない（本プロジェクトでモンキーパッチにより追加）  
- **KL計算**（`brax/training/agents/ppo/losses.py`）: `KL(old||new) = Σ[log(scale_new/scale_old) + (scale_old² + (loc_old-loc_new)²)/(2*scale_new²) - 0.5]`。`scale_new`が小さいほど爆発する  
- **`mean_clip_scale`**（`brax/training/networks.py`）: `mean = mean_clip_scale * (mean / (1+|mean|))`というソフトクリップ。`distribution_type='tanh_normal'`使用時に通らない経路があるため、`create_dist`パッチ側で一元的にlocもクリップする方式を採用済み（正しい）  
- **`state_dependent_std`**: デフォルト`False`（stdはグローバル学習パラメータ）。本プロジェクトはこの既定値のまま使用  
- **`bootstrap_on_timeout=True`**: 環境側が`state.info['time_out']`（この名前で厳密一致）と`done=True`を同時に設定する必要がある。実際にbootstrapへ使われる値は`data.extras['policy_extras']['value']`＝\*\*遷移前の観測から計算された`V(s_{T-1})`\*\*であり、`V(s_T)`（打ち切り時点の観測の価値）ではない（`brax/training/acting.py`の`actor_step`で確認）。auto-resetによる観測汚染はこの経路には無い  
- **`EpisodeWrapper`のtruncation自動計算**（`brax/envs/wrappers/training.py`）: `truncation = where(steps>=episode_length, 1-state.done, 0)`という**無条件の代入**。環境側が`info['truncation']`を先に設定してもこの行で必ず上書きされる。正しい修正は環境がStateとして返す`done`自体を`terminated`のみにすること（2章タスク2）  
- **`discount`**（`brax/training/acting.py`）: `discount = 1 - state.done`として導出される。環境側が明示的に設定するものではない  
- **標準GAEのtruncationマスク**（`losses.py`）: `deltas *= (1-truncation)`。truncation境界のtransitionは標準GAEのdelta計算そのものから除外される（0倍される）。これにより`next_observation`がauto-reset由来で汚染されていても、truncation境界の標準GAE経路には実害が伝播しない（ただし`bootstrap_on_timeout=False`の場合の扱いは別途要確認）

---

## 7\. Claudeへのエスカレーション基準

以下に該当する場合のみチャットで質問する。それ以外は本文書・v2・v4・課題管理.mdの記録ルールに従って自走してよい。

1. **Qualification PASS（3 seeds）を2回連続で実施しても、Phase 0完了基準（4章）を満たせない場合** — 「観測→診断→仮説→次の単一変更→棄却条件」の形式で記録してから相談する  
2. **6章に無いBrax/JAX/MJXの内部仕様の確認が必要な場合**  
3. **物理モデルに関わる、シミュレーションで直接検証すべき新しい疑義が出た場合**  
4. **タスク8で外乱スペックへの余裕度が1.0を下回った場合**（`standing_fixed_feet`の妥当性判断、目標値の見直しが必要なため）  
5. **記録された結果が2つの診断手法で矛盾する場合**  
6. **タスク2の3ケース単体テストが期待結果と一致しない場合**  
7. 同一Taskで異なる仮説に基づく修正が3セット失敗した場合、または評価結果が再実行で再現しない場合

上記に該当しない日常的な実行・記録・小さなパラメータ調整は、この文書の範囲内でCopilotが判断して進めてください。

---

# 付録A: standing\_robustness\_plan\_v4（詳細仕様、全文統合）

以下は`standing_robustness_plan_v4.md`の全文です。本文書（Master Planタスク群）から参照される詳細仕様として、外部ファイルを介さず本文書内で完結させるために統合しています。見出し番号（\#\# 0\. 〜 \#\# 10.）は元ファイルのものをそのまま維持しています。床傾斜対応（旧§1.2/§6.2に含まれていた記述）はMaster Plan §0の決定により不採用ですが、原文保存の観点からこの付録では削除せず残しています。実装時はMaster Plan本文の決定（傾斜床不採用）を優先してください。

- 作成日: 2026-08-27  
- 位置づけ: v3を6モデル（GPT系、Gemini、Perplexity、Kimi、Claude Opus、GPT Pro）によるレビューに基づき改訂。Copilotが本書単体で自律実装できる仕様まで具体化することを目標とする。  
- 運用方針:  
  - 各TaskのDoneは可能な限りスクリプトのexit code（pytest等）で機械判定する。Markdownのチェックボックスは自己申告であり、単独では合格根拠にしない。  
  - 1 iteration \= 「1つの仮説を選ぶ → 変更を1カテゴリに限定 → 学習/評価1本 → 記録」。報酬とPPOハイパーパラメータなど複数カテゴリの同時変更は禁止。  
  - 同一Taskで3 iteration失敗、またはNaN/Inf発生・torque limit違反・既存合格モデルからの性能低下など重大異常が起きた場合は即エスカレーション（§9.3）。  
  - 本書に未記載の判断が必要な場合、既存リポジトリ規約を優先しつつ変更点を`docs/`に記録する。

---

## 0\. 目標仕様の確定（Gate 0.5より前に必須）

v3では「外乱に耐えて転倒しない」という目標に対し、耐えるべき外乱の定量値が定義されていなかった。これを最初に固定する。

### 0.1 目標外乱スペックとPhase-1物理限界の関係

| 項目 | 目標値 | Phase-1限界値 | 余裕度(限界/目標) |
| :---- | :---- | :---- | :---- |
| 突っつきインパルス \[N·s\] | `<記入>` | `<Phase-1より転記>` | `<計算>` |
| 静的傾斜角 \[deg\] | `<記入>` | `<Phase-1より転記>` | `<計算>` |
| 動的傾斜速度 \[deg/s\] | `<記入>` | `<記入、未計算なら追加計算>` | — |

**余裕度が1.0を下回る項目がある場合、`standing_fixed_feet`のままでは目標に届かない可能性が高い。** その場合は人間が「目標値を見直す」か「§0.2のstanding制約を緩める」かを判断する。この判断がつくまでTask 1（Gate A是正）には進めるが、Gate Bには進まない。

### 0.2 standing制約のレベル定義

- **standing\_fixed\_feet**（本計画のスコープ）: 足裏接地点を大きく移動させない直立維持。ankle/hip strategyのみで回復する。  
- **standing\_with\_recovery\_step**: 一歩程度の踏み出しを許す。0.1の余裕度が不足する場合の代替スコープ。  
- **walking\_robustness**: 歩行中の外乱耐性。本計画の対象外、将来の別計画とする。

### 0.3 「転倒しない」の成功条件（多層定義）

`episode_alive == episode_length` だけを成功条件にしない。以下の論理積で定義する。

success \=

    alive(T)

    AND upright              (torso tiltが閾値以内)

    AND height\_ok             (base高さがnominal比で閾値以上)

    AND no\_illegal\_contact    (膝・腰・胴体・腕など足裏以外の床接触がない)

    AND slip\_ok                (足裏の水平滑り量が閾値以下)

    AND torque\_ok              (actuator saturation率が閾値以下)

    AND (外乱時) recovered\_in\_time  (外乱後、規定秒数内に姿勢誤差・角速度が回復範囲内に戻る)

各閾値は`configs/standing/success_criteria.yaml`（新規）に数値で定義し、本書ではなく設定ファイルを正とする。目安値は§0.4で物理限界から逆算して記載する。

### 0.4 物理限界値の転記（Phase \-1結果）

Phase \-1の計算結果を以下に転記する（**空欄のままGate Bに進むことを禁止**）。

| 項目 | 値 |
| :---- | :---- |
| 総質量 `m` | `<値>` |
| 重心高さ `h` | `<値>` |
| 足裏寸法 | `<値>` |
| 支持多角形実効半径 `d` | `<値>` |
| 各関節最大トルク | `<値>` |
| サーボ角速度上限 | `<値>` |
| 静止摩擦係数レンジ `μ` | `<値>` |
| インパルス限界 `J_max` \[N·s\] | `<値>`（対応する重心速度変化 `Δv = J/m`） |
| 傾斜限界 `θ_max` \[deg\] | `<値>` |

補足: v2でCapture Point系の**報酬設計**は不採用と決定済みだが、外乱限界の見積もりと**評価指標**としてのCapture Point `x_cp = x_com + ẋ_com * sqrt(h/g)` は別物であり、転倒余裕を定量化する低コストな指標として評価には採用する。

**注記（実機制約）**: Capture Point (CoP) はシミュレーション環境でのみ算出・評価する（実機ハードウェアからは算出しない）。足裏ADC構成（MCP3208等）を撤去し、FSRは閾値判定によるバイナリ接触のみを扱う構成としたため、4点からの連続的な荷重分布を実機から取得できず、CoP算出はハードウェア的に不可能である。Copilotは実機（`real/`配下）コードにCoP算出ロジックを実装しないこと。

---

## 1\. 環境仕様（MDP定義・Single Source of Truth）

**本節の値と既存コードが矛盾する場合、本節を正としてコードを修正する。変更する場合は本節を先に更新し、diffを`docs/`に記録する。**

### 1.1 制御・時間

| 項目 | 値 |
| :---- | :---- |
| 物理timestep `dt` | `<値>` s |
| policy周波数 `f_c` | `<値>` Hz |
| decimation `n = 1/(f_c * dt)` | `<値>` |
| episode長 | 500 step \= `<dt*decimation*500>` s |
| soak test時間（学習horizonより長い長時間評価） | `<値>` s（目安60s） |
| 外乱印加時間窓 | `<値>` s |
| 回復判定時間 | `<値>` s |

`500 step`は単独では意味を持たない。以降すべての基準はstepと秒を併記する。

### 1.2 座標系（傾斜床対応・最重要）

- **姿勢誤差**: world鉛直（重力方向）基準。傾斜面に胴体を垂直に立てると転倒するため、床法線基準は使わない。  
- **高さ**: 足裏接地点を原点とする相対高さ。world zで判定すると傾斜の下り側で偽の転倒判定が出るため使わない。  
- **重心速度**: world frameを基本とし、評価時は床接線平面への射影も記録する。  
- **支持多角形の判定**: 床面座標系で行う。

静的傾斜面上で転倒しない物理条件はおおむね次で見積もれる。

θ\_max ≈ min( arctan(μ), arctan(d / h) )

（`μ`: 静止摩擦係数、`d`: 重心から支持多角形境界までの水平距離、`h`: 重心高さ）。この式と実測値は§0.4に転記済み。

### 1.3 観測空間

**actor観測**（実機で取得可能な情報のみ）:

| \# | 要素 | 次元 | 備考 |
| :---- | :---- | :---- | :---- |
| 1 | Base重力射影ベクトル（IMU相当） | 3 | base座標系での下向き単位ベクトル |
| 2 | Base角速度 | 3 | IMU gyro相当 |
| 3 | Base線形加速度 | 3 | IMU accel相当 |
| 4 | 関節角度誤差 `(q - q_default)` | N |  |
| 5 | 関節角速度 | N |  |
| 6 | 直前action | N |  |
| 7 | 足裏接触（バイナリ: 0 or 1） | 左右各1〜4点 |  |

上記を直近`H`ステップ（目安`H=3〜5`、0.06〜0.1s相当）分frame stackする。RNNより先にframe stackで評価すること（実装・デバッグコストが低いため）。

**criticのみが使う特権観測**（asymmetric actor-critic）: 印加中の外力ベクトル、床法線ベクトル、真の重心位置・速度、シミュレータから得られる真の足裏接触力（ニュートンの連続値）。actorはこれらに一切アクセスしない。

理由: 突っつき外乱（インパルス外力）は観測できない。単一フレームの固有感覚情報のみだと、方策は外力直後の状態変化からしか推定できず、value関数は「いつどの向きに外力が来るか」を予測できないためreturnの分散が構造的に大きくなる。これはvalue lossの発散・KLスパイクという、現在Gate Aで観測されている症状に直結しうる。**これはGate Bの成否を左右する設計判断であり、Task 2（Gate A是正）の時点でactor/critic入力を別テンソルとして扱う構造にしておく。Gate B開始後に追加すると学習済み方策が無価値になる。**

### 1.4 行動空間

- 出力: 関節目標角に対するオフセット `Δq`（nominal PD制御へのresidual）  
- レンジ: `Δq ∈ [<値>, <値>] rad`  
- レートリミット: `<値> rad/step`、指令ローパス時定数`<値>`（方策に高周波を学習させないための物理的な実装。action rate penaltyだけに頼らない）  
- トルク直接指令は採用しない。ROBO-OneクラスサーボはPD位置指令が主体であり、トルク直接出力の方策は実機に載らない。**この決定はGate A是正より前に固定し、Gate Aの再学習はこのaction空間で行う**（Gate Cで作り直すと、それまでの学習成果が無価値になる）。

### 1.5 終了条件

| 判定量 | 種別 | 座標系 |
| :---- | :---- | :---- |
| torso tilt超過 | terminated | world鉛直基準 |
| base高さ低下 | terminated | 足裏相対 |
| 非足裏接触 | terminated | — |
| 関節角/速度/トルク限界 | terminated | — |
| time-limit到達（500step） | truncated | — |

`terminated`と`truncated`は明確に分離する（§4.2で詳述）。

### 1.6 初期状態分布

- 関節角: `±<値>`  
- base roll/pitch: `±<値>`  
- base角速度: `±<値>`

外乱からの回復姿勢へ汎化させるため、この摂動はGate A是正の時点で必ず導入する。

### 1.7 常時ONのdomain randomization（Gate A是正時点から）

- アクチュエータ遅延: 1〜3制御周期  
- 観測ノイズ: IMU角度・角速度に実センサ実測値ベースのσ  
- 観測遅延

質量・慣性・摩擦のランダム化はGate Bと同時開始。Gate C（§7）は「DR導入」ではなく「DRレンジの実機実測に基づく較正」フェーズと位置づける。

---

## 2\. Gate 0.5: MuJoCo–MJX整合性確認（Gate Aより前に追加）

Gate 0は純MuJoCo、学習はMJXで行っている。接触の多い二足ロボットではバックエンド差が学習結果に影響しうる。

### 実施内容

同一初期状態・同一PD指令・同一外乱列で純MuJoCoとMJX双方をrolloutし、`qpos`、`qvel`、base姿勢、joint torque、contact state、foot slip、termination時刻を比較する。完全一致は要求せず、短時間誤差とイベント一致率に許容値を設定する。

### Done

- [ ] timestep・actuator・friction・solver設定が`docs/`に記録されている  
- [ ] 比較結果が可視化され、差分が許容範囲内であることが確認されている（または原因が特定されている）  
- [ ] termination判定の一致率が基準以上  
- [ ] 最終評価（Gate A以降）を純MuJoCoでも実行できる状態になっている

---

## 3\. Task 0: Gate A不安定の切り分け（診断・拡張版）

新規学習は行わず、既存の保存済みモデル・ログを使う。

### 3.0 現行設定の完全な棚卸し（手順0）

診断に着手する前に、全ハイパーパラメータ（`γ`、GAE `λ`、学習率、`clip_range`、entropy係数、batch/minibatchサイズ、ネットワーク構成、報酬の全重み、終了条件の全閾値）を`docs/current_config_baseline.md`に一覧化して固定する。これがないと診断結果の解釈も再現もできない。

### 3.1 reward経路監査（修正版）

**旧v3の誤り**: alive bonus理論累積値`25.0 * 500 = 12500`とreward clip上限を単純比較していたが、多くの実装ではclipされるのは**episode累積報酬ではなく各stepのreward**である。その場合、比較すべきは`12500`ではなく、`alive_bonus_per_step = 25.0`と`per_step_reward`のclip上限である。

まず、clip対象が次のどれかを確認する。

- per-step rewardのclip  
- episode returnのclip  
- value targetのclip  
- reward normalization / return normalization  
- PPO value function clip

そのうえで、以下を実測しログする。

- clip前の各reward成分 / clip前の合計reward / clip後のreward  
- discount後return / value target / predicted value  
- reward normalization前後の値  
- 飽和したstepの割合

reward hackingの判定は`12500`との単純比較ではなく、以下の複合指標で行う。

- 方策が高報酬でも§0.3の成功条件を満たさない  
- reward成分の大部分がalive bonusで説明される  
- 姿勢品質とreturnの相関が低い  
- episode長とreturnだけがほぼ完全に相関する

### 3.2 value関数・正規化の診断（新規・最優先で実施）

KLスパイクの主因は方策側よりvalue推定の崩壊や観測正規化統計のドリフトであることが多い。「評価が後半で低下する」症状は、方策劣化ではなく**評価時に正規化統計が学習時と異なる**という実装バグでも同一症状を示す。最も安価に検証できる仮説のため、Task 0の最初に置く。

確認項目:

- explained varianceの推移（0.5を下回る区間の有無）  
- value lossの推移とスパイク位置、policy KLスパイクとの時間相関  
- 観測正規化（running mean/std）の統計値の推移  
- 評価時に学習時の正規化統計が正しく凍結・共有されているか、evaluation中にnormalizerが更新され続けていないか  
- 学習後半のentropy推移（早期collapseの有無）

### 3.3 終了処理監査

- `terminated`と`truncated`が別フラグとして環境から返っているか  
- GAE計算で`truncated`は次状態のvalueでbootstrap、`terminated`はゼロでbootstrapしているか  
- **auto-reset環境特有の罠**: vectorized env/MJX・Brax系実装では`done=True`後に自動resetされ、`next_obs`が終了時の観測ではなくreset後の観測になっている場合がある。`truncated`でbootstrapする際は、reset後観測ではなく\*\*time-limit到達時点の最終観測（final\_observation）\*\*を使う必要がある。この区別は「`docs`に記録するだけ」ではなく、コード上の単体テストまで必須とする（§4.2）。

### 3.4 checkpoint評価カーブ

「学習完走・モデル保存済み」の保存済みモデルがbestかlastか不明な場合、それ自体が問題の可能性がある。保存済み全checkpointについて`episode_alive`・deterministic評価成功率・KL・entropy・action std・value loss・torque saturation率の評価カーブを作成する。

### 3.5 評価

同一モデルについて、初期状態randomizeの有無を軸にした2×2設計（deterministic×stochastic × 初期状態固定×randomize）で評価する。deterministic方策で初期状態を固定すると毎回同一結果になるため、単純な「seed 1〜5」だけでは意味を持たない点に注意する。

記録項目:

- `episode_alive`の平均・分散に加え、Kaplan–Meier型の生存曲線（500stepで打ち切られる右側打ち切り分布のため、単峰・二峰の判定だけでは不十分）  
- 終了理由の内訳（姿勢角超過／高さ低下／非足裏接触／関節・トルク限界／time-limit）  
- 失敗episodeについて、崩れ始めた時点でのbase orientation・角速度・重心位置の時系列

### 3.6 診断→アクション決定木

| 観測結果 | 示唆 | 次アクション |
| :---- | :---- | :---- |
| 失敗がepisode序盤に集中 | 初期状態・初期transientの問題 | 初期状態分布の縮小・初期姿勢安定化 |
| 失敗時刻がランダムに分布 | 状態空間の局所不安定領域 | 失敗直前の状態を特定し、該当領域の報酬/観測を強化 |
| 失敗がepisode後半に集中 | 長期ドリフト or time-limit bug | truncation/termination処理を再疑う（§3.3） |
| explained variance \< 0.5 の区間がある | value学習不良 | 報酬設計より先にvalue側を修正（学習率・ネットワーク・正規化） |
| 評価時のみ性能が崩れる | 正規化統計の不一致（§3.2） | evaluation側の実装を修正、再評価 |

### 3.7 Task 0 完了基準（Done）

- [ ] `docs/current_config_baseline.md`に全ハイパラが記録されている  
- [ ] reward clipの対象（per-step / episode return等）と、それに基づく正しい理論値比較が判明している  
- [ ] explained variance・value loss・観測正規化の状況が判明している  
- [ ] truncation/terminationの区別実装状況とfinal\_observationの扱いが判明している  
- [ ] checkpoint評価カーブから、保存済みモデルがbestかどうかが判明している  
- [ ] `episode_alive`の生存曲線・終了理由内訳が判明している  
- [ ] §3.6の決定木に基づき、主因の暫定結論（優先順位付き）が`docs/gate_a_diagnosis.md`に記載されている

**上記いずれかが判定不能な場合、Task 1には進まず人間にエスカレーションする。**

---

## 4\. Task 1: Gate A是正・再学習

Task 0の結論に基づき実施する。**1 iteration \= 1変更カテゴリ**を厳守し、報酬変更とPPOハイパーパラメータ変更は同時に行わない。やむを得ず複数変更する場合は、事後に1つずつ戻して効果を確認する順序（ablation順序）を事前に定義する。

推奨実施順序: truncation修正 → reward logging追加 → reward scale修正 → PPO安定化 → learning rate調整、の順に1つずつ。

### 4.1 報酬関数の再設計

reward \= w\_orient \* exp(-k1 \* orientation\_error^2)

       \+ w\_height \* exp(-k2 \* height\_error^2)

       \+ w\_com\_vel \* exp(-k3 \* com\_vel\_norm^2)

       \- w\_torque \* (torque\_norm / torque\_max)^2

       \- w\_rate \* action\_rate\_norm^2

- `orientation_error`: base座標系での射影重力ベクトル`ĝ_b`を用いた`θ_tilt = arccos(-ĝ_b,z)`（world鉛直基準、§1.2参照）  
- 各項は無次元化する（角度は許容角、速度は許容速度、トルクは定格トルク、高さ誤差は許容誤差で割る）  
- スケール設計: 許容誤差で報酬が半減するよう`k = ln(2) / 許容値`とする、またはガウス型で基準値を1つに集約する  
- 傾斜床対応: 胴体はworld鉛直基準、足裏は床法線基準、高さは足裏相対（§1.2の座標系定義を厳守）

**設計制約（不変条件、テストで保証する）**:

- 全ての正の項は`[0, 1]`に正規化してから重み付けする  
- ペナルティ項の合計重みは正の項の合計重みの20%を超えない  
- 任意の`(state, action)`で1ステップ報酬の下限が正であること（`r_min > 0`）。これはalive bonusを撤廃・減衰させた場合に生じうる**逆方向のハック**（「早く転んだ方が累積報酬が高い」という、action\_rate\_penaltyが支配的になることで生じる現象）を防ぐための必須制約。  
- 転倒terminated時のみ明示的な終端ペナルティ`r_term = -R_remain`を与える場合、truncatedには適用しない（誤適用厳禁）。value bootstrap修正（§4.2）と二重計上がないことを単体テストで確認する。  
- 上記はランダムサンプルした`10^4`個の`(state, action)`対で自動検証するテスト（`tests/test_reward.py`）を用意する。

報酬成分ごとの寄与を常時ログできるようにする（デバッグ資産として必須）。目標寄与比率を先に決め（例: orientation 40% / height 25% / com\_vel 20% / penalties 15%）、収束時のログでその比率になるよう重みを調整する。

非足裏接触（膝・腰・胴体・腕などの接地）は原則terminationまたは強いペナルティとし、成功扱いにしない。

### 4.2 truncation / termination \+ auto-reset final\_observation の修正

環境のstep関数で`terminated`（真の失敗）と`truncated`（time-limit到達）を明確に分離し、GAE計算で以下を保証する。

\# terminated: bootstrapしない、traceも継続しない

\# truncated: reset前の最終観測（final\_observation）からbootstrapする、次episodeへtraceは継続しない

mask \= 1.0 \- terminated.astype(float)

gae\_target \= reward \+ gamma \* next\_value \* mask

`next_value`の計算に使う`next_obs`は、auto-reset後のリセット状態ではなく、time-limit到達時点の最終観測（`final_observation`）でなければならない。

**追加テスト（unit test合格をDone条件にする）**:

- 人工的に一定valueを返す環境でreturnを手計算と比較  
- `terminated` episodeのbootstrapが0であること  
- `truncated` episodeのbootstrapが最終観測valueであること  
- auto-reset後の観測を誤ってbootstrapに使用していないこと  
- batch末尾とepisode末尾が重なった場合のテスト  
- 決定論性テスト（同一checkpoint・同一seed・同一入力で評価結果が再現すること）

### 4.3 PPO安定化

- `target_kl`によるepoch内early stopping（目安0.01〜0.02、既存ログのKLスパイク幅を見て調整）  
- 学習率を下げる、または`clip_range`を狭める（例: 0.2 → 0.1）  
- ミニバッチ／バッチサイズを増やして勾配分散を下げる  
- advantage正規化、value clipping、gradient norm clipping、entropy係数のスケジュール、学習率decayの有無を確認  
- explained varianceを常時監視し、KLスパイクの原因がpolicy側かvalue側かを切り分ける

### 4.4 checkpoint選定基準の変更

モデル保存基準は「training reward最大」ではなく「**deterministic評価での成功率（§0.3の複合成功条件）最大**」に変更する。定期評価（例: 20 iterationごと、deterministic、n=50）を実施し、best checkpointと全checkpointを保持する。以降のGate判定は最終モデルではなく、best checkpointを別seedで再評価して行う。

### 4.5 Gate A合格基準（統計設計を修正）

v3の「500step生存率 ≥ 95%（5 seed平均）」は、seedあたりのエピソード数・平均か全達成かの判定基準が未定義で、検証として機能しない（1 seed=1 episodeなら5試行では80%/100%しか観測できない）。

**推奨する再定義**:

- 学習seed ≥ 3（理想は5）  
- 各学習seedにつき評価エピソード n ≥ 200、評価用の初期状態・条件は学習用randomizationと別系列で事前生成・凍結（held-out化）  
- 判定: 各学習seedの成功率のWilson score 95%信頼区間の下限が閾値以上、**かつ全学習seedで達成**（平均ではなくmin基準。1つの良いseedが悪いseedを平均で隠すことを防ぐ）  
- 主判定はdeterministic評価。stochastic評価は診断・ノイズ耐性の参考指標に格下げする  
- 成功条件は§0.3の複合定義（alive \+ upright \+ no\_illegal\_contact \+ slip\_ok \+ torque\_ok）を用いる。`episode_alive`単体では合否判定しない

**この基準を満たすまでTask 2（Gate B）には進まない。**

---

## 5\. Task 2: Gate B-0 外乱インフラの検証（Gate B本体より前に追加）

外乱生成そのものにバグがあると、RLの失敗なのか外乱実装のバグなのか切り分けられなくなる。Gate B本体に入る前に以下を検証する。

- 外力が指定bodyに正しく入っている（`xfrc_applied`等への反映確認）  
- impulse量が指定値と一致している（`J = ∫F dt`の実測値と設定値の比較）  
- 印加方向がworld frame / body frameのどちらかが明確になっている  
- 床傾斜角が指定値と一致している  
- 床傾斜時の接触が破綻していない  
- 外乱なし条件でGate Aの合格性能を維持している

### Done

- [ ] 上記すべてが検証済みで`docs/`に記録されている

---

## 6\. Task 3: Gate B 外乱耐性curriculum

外乱は物理的にもreward設計上も別系統として扱う。

### 6.1 突っつき（インパルス外力）の定義

- 外乱は力ではなくインパルス`J = F * Δt`中心で定義する（同じ力でも印加時間が異なれば影響は全く異なるため）  
- 印加body: torso（体幹）のCoM付近。Phase-1限界値の計算条件と同一の印加条件にする（例: 胴体CoM高さ±X%の範囲でランダム化）  
- パラメータ: インパルス大きさ、印加方向（水平全方位、必要なら鉛直成分も）、印加タイミング（episode内ランダム、1エピソードあたり1〜3回）、印加時間窓  
- 外乱量は絶対値だけでなく`λ_J = J / J_max`の無次元比で管理し、curriculumを明確にする  
- **接触喪失の終了条件に猶予を持たせる**: 突っつき後の回復では一時的な荷重低下・片足浮きが正常に起こりうる。即terminationにすると回復挙動自体を失敗扱いしてしまうため、「両足の鉛直接触力が同時に閾値未満、かつ連続20ms以上」のような猶予付き判定にする。

### 6.2 床傾斜の定義

- **静的傾斜**: episode開始前に床を傾ける。重力ベクトルの回転で数学的に等価に実装可能。  
- **動的傾斜**: 床geomの姿勢を瞬間的に書き換えるだけでは物理的な可動床にならない。可動platform body・回転軸・pivot位置・角度・角速度・角加速度・motion profileを定義し、実際に床を動かす実装が必要。  
- パラメータ: 傾斜角、傾斜速度、傾斜方向（前傾・後傾・左右・斜め）  
- 静的→動的へ段階拡張する場合、各段階でどちらの実装方式を使うか（重力回転／床body回転）を明記する

鉛直外力（接触喪失や床衝突を引き起こす）は水平push外乱とは別問題として後段に分ける。

### 6.3 curriculum進行ロジック

- **前提**: Gate BはGate A合格モデルからの継続学習とする（スクラッチ再学習ではない）  
- 昇格条件: 直近N評価窓での成功率が閾値（例0.85）以上で難易度パラメータを段階的に増加  
- 降格条件: 成功率が閾値（例0.6）未満で1段階戻す  
- 難易度上限は§0.4のPhase-1物理限界値（の70〜80%程度。限界値の前提条件が理想アクチュエータ計算なら特にマージンを取る）  
- 各環境インスタンスに難易度を分布として持たせ、全環境を同時に難易度アップさせない  
- **catastrophic forgetting対策（最重要）**: 常に「外乱なし／既習の弱い外乱／現在段階の外乱／過去段階の外乱」を一定割合（例30%）混ぜる。各curriculum段階終了時にGate A条件（外乱なし）を再評価し、成功率が前回比で規定値以上低下したら外乱強度を1段階戻す。

### 6.4 評価: robustness envelope

単一の成功率ではなく、境界として評価する。

- 外力グリッド評価: インパルス大きさ`{0.2, 0.4, ..., 1.2} × J_max` × 方向8方位 × 印加タイミング3種 → 成功率ヒートマップ  
- 傾斜グリッド評価: 傾斜角 × 傾斜方位 → 成功率ヒートマップ  
- 耐性境界の定義: 成功率50%となる外力大きさ`J_50`、傾斜角`θ_50`を主要指標とする  
- 学習分布外（限界値の1.2倍など）を含むheld-out条件を必ず含める

### 6.5 Gate B合格基準

- 各外乱系統単独で、目標値（§0.1）までの範囲でランダム化した条件下、成功率 ≥ 90%（§4.5と同じ統計設計: 学習seedごとWilson信頼区間下限、min基準）  
- 両外乱を同時に加えた複合条件でも成功率 ≥ 80%  
- Gate A条件（外乱なし）の性能を維持していること（低下が規定値未満）  
- `J_50`、`θ_50`が§0.1の目標値を上回っていること

---

## 7\. Task 4: Gate C sim2real較正・実機準備

Gate CはDR「導入」ではなく、§1.7で早期導入済みのDRレンジを実機実測に基づいて較正するフェーズと位置づける。

### 較正対象

- 質量・慣性モーメント: ±10%目安（実測で更新）  
- 関節摩擦・ダンピング: ±20%目安  
- コントローラ遅延: 1〜2制御周期  
- 観測ノイズ: IMU角速度・加速度への実センサ実測ベースのノイズ

### 実機前安全Gate（必須）

- [ ] harness（吊り下げ／台上拘束）を使用した段階的テスト計画  
- [ ] hardware E-stop  
- [ ] software watchdog / command timeout  
- [ ] joint soft limit、torque/current制限  
- [ ] 通信切断時の安全姿勢への遷移  
- [ ] policy出力のNaN検出、observation異常値検出  
- [ ] 「突っつき」は人手ではなく、calibrated pusherまたは振り子で外乱を定量化してから実機テストする（人手は再現性が低く危険）  
- [ ] human operator承認プロセス

実機での実テスト自体は本計画書のスコープ外とし、上記安全Gate確認後に人間が判断して開始する。

---

## 8\. 評価プロトコル（全Gate共通）

- 評価対象: best checkpoint（最終モデルではない、§4.4）  
- 学習seed ≥ 3、各学習seedにつき評価エピソード n ≥ 200、評価seedは学習と別系列（held-out）  
- deterministic評価を主判定、stochastic評価を副判定として併記  
- 判定: 各学習seedの成功率のWilson 95%信頼区間下限が閾値以上、かつ全学習seedで達成（min基準、平均ではない）  
- 記録必須項目: 成功率と信頼区間、`episode_alive`の生存曲線、終了理由内訳、報酬成分別の平均寄与比率、Capture Point余裕の時系列統計（**シミュレーション評価専用。実機ハードウェアからは連続的な荷重分布を取得できないため、実機評価では算出・記録しない**）  
- Gate B以降は外力グリッド・傾斜グリッドのヒートマップと`J_50`・`θ_50`を必ず出力する

---

## 9\. Copilot自律反復運用ルール

### 9.1 実験管理・再現性

- 全runにrun ID（日時＋git commit短縮ハッシュ＋config hash）を付与する  
- configは単一のYAML/dataclassに集約し、ランごとに完全なスナップショットを保存する  
- 指標はCSV/JSONLなど機械可読形式で保存し、Markdownは要約のみとする  
- 乱数seed（training seed / environment seed / disturbance seed / policy sampling seed）を分離して記録する  
- 依存ライブラリ（MuJoCo/MJX/JAX等）のバージョンを固定・記録する  
- raw logは`runs/`、要約は`docs/experiments/YYYY-MM-DD_gate_x.md`に分離する（リポジトリ肥大化を防ぐ）  
- `run_manifest.json`に git commit hash、config hash、seed、checkpoint path、model file hash、バージョン情報、domain randomization設定、外乱設定、評価episode数、成功率、termination reason内訳を含める

### 9.2 変更制約

- 1 iteration \= 1つの仮説を選ぶ→変更を1カテゴリに限定→学習/評価1本→記録、というサイクル  
- 1 iterationあたりの計算予算上限（環境ステップ数 or wall-clock時間）をTaskごとに設定する  
- 変更前に期待結果を文書化する  
- 合格checkpointを上書きしない。合格checkpointから性能が低下した場合はrollbackする  
- 評価コードと学習コードを同時に変更した場合、旧checkpointを再評価する  
- 成功基準そのものを自動変更してはならない  
- ロボットモデル・関節制限・トルク上限を人間承認なしで変更してはならない  
- 外乱上限を解析限界以上へ自動拡大してはならない  
- 実機コマンドを自動実行してはならない

### 9.3 エスカレーション条件

以下のいずれかで即時停止し、人間に報告して指示を待つ。

- 同一Taskで異なる仮説に基づく修正が3セット失敗した場合  
- NaNまたはInfの発生  
- torque limit違反  
- 成功率が直前の合格モデルより規定値以上低下  
- 純MuJoCoとMJXでtermination結果が大きく不一致  
- reward上昇と独立成功率低下が同時発生（reward hackingの再発兆候）  
- evaluation結果が再実行で再現しない  
- 設定・checkpoint・commitの対応関係が追跡不能

報告フォーマット:

\#\#\# ESCALATION REPORT

\- Current Task:

\- Failure Count / Trigger:

\- 現象の定量要約:

\- 実施済み修正と各結果（diff付き）:

\- 残仮説（優先度順）:

\- 人間に判断してほしい論点:

### 9.4 ドキュメント構成（Copilot実行時の注意）

- 本書をそのまま1つの巨大な文書としてCopilotに渡さない。1タスク1ファイル（`docs/tasks/task0_diagnosis.md`等）に分割する  
- 座標系・命名規則・テスト方針・記録形式などの不変ルールは`.github/copilot-instructions.md`に置く（長大な単一文書はコンテキストから落ちやすい）  
- 「実装せよ」ではなく「このテストを通せ」という形でタスクを与える（テストファースト）。§3〜§7の各Done定義は可能な限り`pytest`関数に落とす  
- 数値の合格基準は機械可読な判定スクリプト（例: `scripts/check_gate_a.py`がexit codeを返す）にする。Markdownのチェックボックスだけでは自己申告になる  
- 1コミット1変更カテゴリを規約化し、報酬変更とハイパーパラメータ変更を分離する

---

## 10\. 実施順序まとめ

| \# | フェーズ | 内容 |
| :---- | :---- | :---- |
| 0 | 目標仕様確定 | §0.1〜0.4の記入。standing\_fixed\_feetの妥当性確認 |
| 0.5 | MuJoCo–MJX整合性 | §2 |
| 1 | Gate A診断 | §3（value/正規化診断を含む拡張版Task 0） |
| 2 | Gate A是正・再学習 | §4（制御インターフェース＝§1.4のaction空間をこの時点で確定） |
| 3 | Gate B-0 外乱インフラ検証 | §5 |
| 4 | Gate B curriculum | §6 |
| 5 | Gate C sim2real較正 | §7 |

制御インターフェース（action空間）とasymmetric actor-critic構造は、Gate A是正の時点で確定させる。Gate B開始後に変更すると、それまでの学習成果が無価値になるため。  
