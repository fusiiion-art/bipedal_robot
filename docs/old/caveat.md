以下は、リポジトリ全体から抽出した内容を、Claudeへそのまま渡せる改良ルールとして整理したものです。

なお、現在の `caveat.md` には調査中の会話ログや検索結果も混在しています。Claude用資料としては、以下の整理版だけを使用する方が安全です。

# Claude向けプロジェクト改良規約

## 1. プロジェクトの目的

このプロジェクトの目的は、歩行ではありません。

```text
外乱を受けても、足を踏み替えず、
両足を接地したまま、その場で直立姿勢を維持する。
```

したがって、以下は禁止です。

- 歩行
- 踏み替え
- 支持基底の変更
- 外乱対策としての足の移動
- 傾斜床への対応
- トルク直接指令への変更

対象は固定足立位、すなわち `standing_fixed_feet` です。

## 2. 参照資料の優先順位

仕様を判断するときは、次の順序で確認します。

1. 実装コード
2. `current.md`
3. `config.py`
4. `status.md`
5. `master_plan.md`
6. その他の文書

`README.md` には古いディレクトリや存在しない文書への参照が残っているため、仕様の正本として使用しません。

`master_plan.md` の付録には、現在は不採用の傾斜床仕様が残っています。実装時は本文の「傾斜床不採用」を優先します。

## 3. 現在の状態

現在は Phase 0 PPO安定性検証中です。

未解決事項:

- KLスパイク
- 学習後半の `episode_alive` 低下
- observation normalizerの評価時凍結確認
- checkpointを使った正式評価
- 純MuJoCoとMJXの数値整合性確認

したがって、Gate A以降や外乱耐性学習へ進む前に、PPOの安定性を確認する必要があります。

## 4. 絶対に維持する契約

### 座標系

- 姿勢はworld鉛直基準
- 床法線基準の姿勢判定は使わない
- 高さは足裏を基準にする
- 傾斜床や可動床を追加しない
- quaternion順序は `(w, x, y, z)`
- RPY順序は roll、pitch、yaw

### 単位

- 長さ: m
- 角度: rad
- 力: N
- トルク: N·m
- 力積: N·s
- 実機制御周期: 100 Hz
- `CONTROL_DT`: 0.01秒
- MJX内部刻み: 0.0025秒

外乱は力だけでなく、必ず力積で記録します。

$$J = F \Delta t$$

### アクション

アクションはトルクではなく、関節目標角の残差です。

```text
policy output
 -> ACTION_SCALE
 -> default poseへの加算
 -> deadband
 -> 関節速度制限
 -> LPF
 -> CBF
 -> 温度・電圧derating
 -> actuator
```

この契約を位置制御からトルク制御へ変更する場合、単なる改善ではなく別設計・別検証になります。

## 5. 観測契約

現在の観測次元は625です。

```text
Base observation       84
Base observation履歴   84 × 5 = 420
Action履歴             20 × 5 = 100
サーボ温度             20
電源電圧               1
合計                   625
```

観測の以下の要素は、学習済みモデルと実機ONNXのABIに相当します。

- 要素の順番
- FSRの左右順
- phaseの表現
- 関節角・関節速度の順番
- 履歴の新旧方向
- action履歴の順番
- 温度と電圧の位置
- 観測次元

変更対象:

- `config.py`
- `mjx_env.py`
- `real_env.py`
- `export_onnx.py`

観測変更時は、既存checkpointとの互換性がなくなる前提で、モデル再学習と実機推論確認が必要です。

## 6. 関節・アクチュエータ契約

以下はすべて同じ順序でなければなりません。

- `RobotConfig.JOINT_NAMES`
- MuJoCoのjoint順
- XMLのactuator順
- `DEFAULT_JOINT_ANGLES`
- XMLの`ctrlrange`
- 実機のサーボID
- `real_env.py`の可動域
- ONNX出力順
- Teensy・BusLinkerのチャンネル順

主な確認対象:

- `config.py`
- `humanoid.xml`
- `real_env.py`
- `real_io.py`

関節順、左右符号、可動域を変更すると、シミュレーション上は動いても実機で逆方向に動く危険があります。

## 7. 固定足制約

以下は常に維持します。

```python
ALLOW_WALKING = False
ALLOW_STEPPING = False
TARGET_VEL_X = 0.0
TARGET_VEL_Y = 0.0
TARGET_YAW_RATE = 0.0
```

また、次の値を緩める場合は、固定足立位の目的そのものを変更することになります。

- `MAX_FOOT_TRANSLATION`
- `MAX_FOOT_YAW_ROT`
- `MAX_SINGLE_FOOT_LIFT`

注意点として、これらの値が設定ファイルに存在することと、すべてが環境内で実際に強制されていることは別です。設定値、報酬、評価スクリプトの3箇所を確認してください。

## 8. 終了条件

終了状態は区別します。

```text
terminated = 転倒・危険状態による真の終了
truncated  = 時間制限による終了
```

期待する動作:

| 状態 | `State.done` | `truncated` |
|---|---:|---:|
| 通常遷移 | False | 0 |
| 転倒 | True | 0 |
| 時間切れ | False | 1 |

Braxの`EpisodeWrapper`が時間切れを処理するため、環境側で時間切れを`State.done`に含めてはいけません。

`time_out`というキー名は変更しません。

関連箇所:

- `mjx_env.py`
- `training_wrapper.py`
- `train_mjx.py`

この部分を変更した場合は、GAE、discount、bootstrap、auto-resetを直接テストします。

## 9. 学習進捗カウンタ

以下を混同してはいけません。

- `step`: エピソード内step
- `_env_steps`: 環境ごとの累積step
- `global_step`: 現在の累積step
- `training_progress`: 0.0〜1.0の学習全体進捗

AutoResetによってエピソードごとの情報がリセットされるため、学習進捗は `training_wrapper.py` で維持しています。

報酬スケジュールや外乱カリキュラムに、エピソード内の`step`を誤って使わないでください。

## 10. 外乱

Phase 0では次を維持します。

```python
DISTURBANCE_CURRICULUM = False
RANDOM_PUSH_MAX_FORCE = 0.0
```

外乱無効時は、ログ上だけでなく、物理更新へ渡す外力配列そのものが全step・全環境・全seedでゼロでなければなりません。

外乱関連の値は別概念です。

- 最大力
- 力積
- 印加時間
- 印加方向数
- 発生確率
- curriculum scale
- 評価用の力レベル

これらを同じ変更でまとめて変更してはいけません。

## 11. 報酬

成功判定を `episode_alive` だけで定義してはいけません。

最低限、次の論理積で評価します。

```text
alive
AND both_feet_contact
AND upright
AND height_ok
AND no_illegal_contact
AND slip_ok
AND torque_ok
AND recovered_in_time
```

報酬変更時の注意:

- 報酬とPPOハイパーパラメータを同じiterationで変更しない
- 転倒ペナルティは`terminated`にのみ適用する
- `truncated`に転倒ペナルティを適用しない
- 生存報酬だけで高得点になる状態を作らない
- 早く転倒した方が得になる報酬を作らない
- 報酬内訳を必ずログに残す
- recovery報酬が振動動作を助長しないか確認する
- torque、heightの安全ペナルティを不用意に弱めない
- PBRSのポテンシャルは状態のみに依存させる

主な実装は `mjx_rewards.py` です。

## 12. PPO、JAX、Brax

現状の制約:

- `min_std >= 0.05`
- `max_std <= 3.0`
- `target_kl`は記録する
- NaN/Infが1件でも出たら停止
- 勾配爆発やvalue loss発散を見逃さない
- JAXの乱数キーを再利用しない
- JIT中にPythonのbool化をしない
- 動的shapeを避ける
- CPUは単体テストと形状確認に使う
- 本格学習はGPU/WSLで行う

関連ファイル:

- `train_mjx.py`
- `validate_policy_bounds.py`
- `validate_progress_wrapper.py`

現在は標準PPO MLPが正しい構成です。廃止済みのRMA Teacher/Adaptation/Base構成を再導入する場合は、別設計として扱います。

## 13. MuJoCo、MJX、XML

`humanoid.xml` の変更は、単なる形状変更ではありません。

影響する項目:

- 接地
- 摩擦
- 重心
- 慣性
- 関節可動域
- actuator出力
- トルク
- 転倒判定
- 接触判定
- sim-to-real差

変更後に確認する項目:

- timestep
- gravity
- solver
- integrator
- body mass
- inertial
- friction
- collision geometry
- foot body名
- actuator順
- actuator `ctrlrange`
- joint range
- `contype`
- `conaffinity`

`mjx_env.py`にはfallback XMLがあります。正式評価では、fallback XMLを使っていないことを検証してください。モデル欠損時に自動fallbackすると、本来のXML不備を見逃す可能性があります。

実機重量が未計測の場合、質量・重心・物理限界は暫定値です。外乱目標を最終確定してはいけません。

## 14. 実機FSR契約

実機経路は次の通りです。

```text
FSR402 × 8
 -> TeensyオンチップADC
 -> Teensy側で二値判定
 -> USB
 -> Raspberry Pi
```

維持すること:

- FSRは8要素
- 実機FSR値は0.0または1.0
- 実機ではZMP/CoPを計算しない
- MCP3208を復活させない
- MCP6004を復活させない
- Pi側SPIを復活させない
- 左右4点の分割順を変更しない

シミュレーションの連続FSR値やZMPは、学習・評価用であり、実機センサー仕様とは異なります。

## 15. 実機通信・安全

維持する契約:

- Teensy安全処理は1 kHz
- Pi制御ループは100 Hz
- USBテレメトリは73バイト
- 通信タイムアウトは30 ms
- IMU異常時は直前の有効quaternionを保持
- サーボ角度は送信前にクランプ
- NaN/Infを実機へ送信しない
- 実機E-stop、Teensy書き込み、FSR校正は人間が実施

特に `real_io.py` には、サーボ応答のchecksum検証を弱める条件があります。通信改良時は、応答長だけで成功扱いにならないことを確認してください。

## 16. 変更単位

1 iterationは次の単位に限定します。

```text
1つの仮説
1つの変更カテゴリ
1つの検証
1つの記録
```

変更カテゴリの例:

- 観測
- 報酬
- 終了条件
- PPO設定
- 物理モデル
- 外乱
- 実機通信
- 評価指標

報酬とPPO設定、XMLと観測、外乱と成功基準を同時に変更してはいけません。

## 17. 検証手順

改良前:

1. `status.md`を読む
2. `master_plan.md`を読む
3. 正本ファイルを特定する
4. 現在のテストとログを確認する
5. 反証可能な仮説を記録する

改良後:

1. 対象範囲の単体テスト
2. 観測shape確認
3. `py_compile`または型・構文確認
4. termination確認
5. NaN/Inf確認
6. 必要に応じてMJX・MuJoCo比較
7. seed固定の再現性確認
8. GPUで1 seedのDebug PASS
9. GPUで3 seedのQualification PASS
10. statusと実験ログを更新

## 18. 即時停止条件

以下のいずれかが発生したら、学習や改良を継続しません。

- NaN/Inf
- torque limit違反
- 実機通信異常
- E-stop経路の異常
- 外乱無効設定なのに非ゼロ外力
- 合格checkpointからの性能低下
- 観測shape不一致
- 関節順序不一致
- `terminated`と`truncated`の分類不一致
- KLの異常な連続スパイク
- value lossの発散
- 評価結果の再現性消失

発生時は `status.md` に記録し、人間の判断を待ちます。

## 19. Gate順序

Gateを飛ばしてはいけません。

```text
Phase -1  物理限界・目標外乱の確定
Phase 0   PPO安定性
Gate 0-P  純MuJoCo物理ベースライン
Gate 0.5  MuJoCo-MJX整合性
Gate 0    学習済み方策の無外乱評価
Gate A    無外乱500step
Gate B-0  外乱インフラ検証
Gate B    軽外乱耐性
Gate C    試合想定外乱・sim-to-real
Gate D    実機安全評価
```

純MuJoCoのPD評価であるGate 0-Pは、学習済みRL方策の正式なGate 0合格ではありません。

## 20. Claudeに渡す作業指示

以下をClaudeへの冒頭指示として使用できます。

```text
このリポジトリでは、コード改良前に以下を必ず守ること。

1. docs/status.mdとdocs/master_plan.mdを読む。
2. 現行仕様はdocs/current.md、設定正本はrobot/config.py、実装コードを優先する。
3. 目的は歩行ではなく、固定足での外乱耐性直立である。
4. world鉛直基準、足裏相対高さ、Δq残差アクション、625次元観測を維持する。
5. 傾斜床、踏み替え、トルク直接指令を追加しない。
6. 1 iterationにつき、仮説・変更カテゴリ・検証を1つずつに限定する。
7. 報酬、PPOハイパーパラメータ、物理モデル、観測を同時に変更しない。
8. 合格checkpointを上書きしない。
9. NaN/Inf、torque limit違反、通信異常、性能低下があれば停止し、docs/status.mdへ記録する。
10. done、terminated、truncated、time_outの関係を壊さない。
11. 観測順、関節順、FSR順、action scale、履歴順を変更する場合はABI変更として扱う。
12. CPUでは単体テストと形状検証を行い、フル学習はGPU/WSLで行う。
13. 変更前後で具体的な検証を実行し、seed、設定、commit、ログ、合否を記録する。
14. 不確かな仕様を推測で変更せず、コードと近隣テストで確認する。
```

現時点でClaudeに最初に依頼するなら、コード変更ではなく、`done/truncated`、observation normalizer、checkpoint評価、外乱ゼロ保証の4項目を検証する診断作業から始めるのが適切です。
完全には不要ではありません。Claudeが実際にコードを改良するなら、**重要な定義名・ファイル名・関数名は残すべき**です。

ただし、すべての定数名を列挙すると読みにくくなるため、次の分け方がよいです。

- 本文: 人間向けの仕様・禁止事項
- 末尾: Claude向けの実装アンカー
- 細かい定数一覧: `config.py`を参照させる

残すべき名前は以下です。

```text
robot/config.py
RobotConfig.OBS_DIM
RobotConfig.ACTION_SCALE
RobotConfig.ALLOW_WALKING
RobotConfig.ALLOW_STEPPING
RobotConfig.DISTURBANCE_CURRICULUM
RobotConfig.MAX_EPISODE_STEPS

envs/mjx_env.py
SenpuuMaruMJXEnv.reset()
SenpuuMaruMJXEnv.step()
State.done
info["terminated"]
info["truncated"]
info["time_out"]

envs/mjx_rewards.py
MJXRewardSystem.compute()

train/train_mjx.py
parse_args()
ppo.train()

real/real_env.py
RealRobotEnv.build_observation()
RealRobotEnv.step()

real/real_io.py
TeensySpineIO.communicate()
```

逆に、`RANDOM_TEMP`や細かい報酬重みなどは、本文で全列挙せず「設定正本は `config.py`」で十分です。

結論としては、**定義名をゼロにするのではなく、変更判断に必要な名前だけ残す**のが最適です。Claude用資料では、仕様本文と実装アンカーを分ける構成がおすすめです。