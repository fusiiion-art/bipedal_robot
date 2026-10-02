# 進捗ステータス

最終更新: 2026-10-02（Claude）

---

## 2026-10-02: 3seed学習(Gate A)前の追加レビュー対応

外部レビュー（N1〜N14、リポジトリ非参照で作成）を実コード・Brax 0.14.2 のソースと突き合わせ、
実際に該当したものだけを修正した。回帰テストは `tests/test_seed_readiness.py`。

| 指摘 | 判定 | 対応 |
|---|---|---|
| N1 公開リポジトリにコードが無い | 前提誤り（main に全コードあり） | `run_manifest.json` に git コミット・dirty 状態を記録。未コミット変更があると起動拒否（`--allow_dirty`） |
| N2 方策分布パッチの import 依存 | 現状の全経路は patch 済みだが構造的に脆い | グローバルパッチを廃止し、サブクラスを factory で注入 |
| N3 AutoReset の info 非リセット | 一部該当 | done 時に info 全キー（`rng_key`/学習進捗/終了種別を除く）と `pipeline_state`/`obs` を同じ fresh reset から取得。旧実装は obs 末尾の温度・電圧が最初の episode の DR 値のままだった |
| N4 timeout が −30 で上書き | 前提誤り（env の done は転倒のみ）。別の不整合あり | `time_out` を転倒と同時の時間切れで立てない。env の打ち切り step を PPO の `episode_length` に一致させる（旧実装は episode_length>500 で step≥500 の全 step に γV(s) が加算された） |
| N5 Wilson 下限の擬似反復 | 前提誤り（評価 episode ごとに観測ノイズ・遅延・関節 DR の乱数が異なる） | 終端状態の指紋 `n_unique_final_states` を記録し、重複があれば Gate A 判定を参考扱いにする |
| N6 観測正規化 std の下限 | 該当（Brax 既定 eps=0 → std 下限 1e-6） | `--obs_norm_std_eps` 既定 1e-4（std 下限≈0.01）。学習後に `normalizer_stats.json` を出力 |
| N7 `cvel[:,3:6]` は足の速度ではない | 該当（足ごと 0.5rad/s 回転で約8cm/s の偽速度） | `mj_objectVelocity(mjOBJ_XBODY)` と同値の変換を実装 |
| N8 CP/ZMP 入力がノイズ支配 | 前提誤り（報酬はノイズなしの真値を使用） | 変更なし（胴体速度/加速度で COM を近似している点は Phase 1 の外乱導入時に再検討） |
| N9 Euler+kv の陽的ダンピング | 該当せず（armature=0.01 により最小 I_eff=0.010、dt·kd/I=0.25 で安定限界2に対し8倍の余裕） | 変更なし |
| N10 PBRS 終端 | 終端のみ該当（reset 時 Φ(s0) 初期化と λ の状態依存性は問題なし） | 終端 step の報酬を `fall_penalty − Φ(s_prev)` に変更 |
| N11 PPO のステップ算術 | 記録の欠如のみ | `run_manifest.json` に env_step/iter・方策更新回数・実際の総 step 数を記録（既定設定で 247 回、10,117,120 step） |
| N12 学習中 eval が確率的方策 | 該当 | `deterministic_eval=True` |
| N13/N14 実機系 | Phase 1 前に対応 | 変更なし |

3seed は修正後のコミットで、`--seed` だけを変えて実行すること。

---

## 2026-10-02（第2回）: 全体監査・学習品質の改善・不要部分の削除

回帰テストは `tests/test_env_quality_contract.py`（新規）と `tests/test_seed_readiness.py`。
**観測の中身・報酬・初期状態分布が変わったため、これ以前の checkpoint とは比較できない**
（actor 観測は625次元のままだが、critic 用の観測が増えたため checkpoint の構造も変わった）。

### 学習の品質に関わる変更

| 変更 | 理由 |
|---|---|
| 初期状態分布を導入（関節角 ±0.05 rad、関節角速度 ±0.2 rad/s、足裏が接地するよう胴体高さを補正） | master_plan.md §1.6。導入前はゼロ行動でも報酬 73.6/75・足裏変位 1.1 mm で500step立ち続け、Gate A が学習済み方策と「何もしない方策」を区別できなかった。導入後のゼロ行動は 16/16 生存・足裏変位 中央値4.8 mm/最大12.7 mm（±0.1 rad/±0.5 rad/s では中央値11.7 mm、16本中1本が20 mm超）。値は `robot/config.py` の `INIT_JOINT_*_NOISE` |
| asymmetric actor-critic（観測を `{'state', 'privileged_state'}` の dict に） | master_plan.md §1.3。critic だけが真の胴体位置・速度、FSR連続値、足裏変位、DR値（質量/摩擦/重心/関節粘性・摩擦）、外力を見る。actor と実機ONNXの入力は625次元のまま |
| actor 観測を実機と一致させた | 実機は FSR を 0/1 フラグで送り ZMP は 0、base_pos は取得不可なのに、sim は連続の接触力・ZMP・ノイズ付き真値位置を観測させていた。姿勢はヨー（実機BNO055では磁北基準の絶対方位）を含むオイラー角から重力射影ベクトルへ（§1.3） |
| 観測履歴を最初の観測で埋める（sim/実機） | ゼロ埋めだと最初の4stepが「全関節0rad」の偽履歴になる。実機は指令履歴も0埋めでsimと不一致だった |
| CBFペナルティを可動域違反だけで測る | 旧実装は目標角とLPF・速度制限後の指令の差を測っており、可動域と無関係な追従遅れをペナルティ化していた |
| 指令デッドバンド 0.02 rad → サーボ1カウント（0.0042 rad） | 実機サーボの分解能は 240°/1000。旧値では1.15°未満の微修正ができなかった |
| 関節クーロン摩擦DRを `sign(v)` → `tanh(v/0.05)` | 静止付近で ±摩擦トルクが制御周期ごとに反転する人工的な振動源だった |
| 熱モデルの時定数 4 s → 80 s | 80℃で始まったサーボが1エピソード中に35℃前後まで冷え、温度DR・熱ダレがエピソード内で消えていた |

### 修正したバグ（学習以外）

- `train/visualize_rl.py` が `np.asarray` をプロセス全体で書き換えており、これを import する Gate A 評価スクリプトにも波及していた → 削除
- `scratch/gate_b_eval.py` がトルク飽和率の上限に関節角の可動域[rad]を使っていた（上限0の関節で飽和率が常に1.0）
- `assets/humanoid/humanoid.xml` の IMU site が `euler="-90 0 0"`（`angle="radian"` のため −90 rad ≒ −116.6°）→ −π/2。sim はこのセンサーを使わないため学習への影響はない
- 実機 `real/real_env.py` が LPF 状態を 0 rad で初期化しており、最初の指令が「全関節0radへ80%」になっていた → 実関節角で初期化。IMU の値を取り付け姿勢 `IMU_MOUNT_QUAT` で胴体座標系へ回転
- `train/train_mjx.py` の JAX キャッシュパスが `/mnt/c/bipedal_robot` 固定 → リポジトリ相対
- 未使用の `dr_kp_scale` をサンプルしていた（KP の DR は実際には適用されていなかった）→ 削除

### 削除したもの

- 歩行関連（master_plan.md で歩行はスコープ外）: `robot/gait_generator.py`、`USE_REFERENCE_GAIT`/`GAIT_*` と分岐、実機側の参照軌道
- 使われない設定・分岐: フォールバックXML、`nq<7` 分岐、`ALLOW_WALKING`/`TARGET_VEL_*` 等の常に定数のガード、常に0の `no_step_penalty`、`com_accel_is_fallback`、`FSR_POSITIONS`、`latency_steps` ほか未使用の info キー・報酬引数
- テストを含まないファイル: `tests/test_joints.py`、`tests/test_mj_xml.py`、`tests/rebuild_global_stl.py`
- 一回限り・壊れたスクリプト: `train/play_mjx.py`、`scratch/{analyze_run,compare_seeds,inspect_params_structure,verify_changes,check_model}.py`、`scratch/run_gate0_formal_wsl.sh`（存在しないスクリプトを呼んでいた）
- 参照されないファイル: `assets/humanoid.xml`（meshdir も解決できない旧コピー）、`stubs/`、`docs/old/`、`docs/pytestと学習結果.md`、`docs/phase3_handover.md`
- checkpoint 読み込み・推論関数の構築を `robot/policy_network.py` に一本化（6か所の重複を削除）

### 実機側の未対応事項（Phase 1 前に要対応）

- サーボ位置の読み出しが1ループ1台の巡回のため、各関節の値は最大200 ms 古く、有限差分の関節角速度はほぼ0とスパイクになる（sim の関節角速度ノイズはσ0.01）
- 実機の安全クランプは関節可動域そのもの、sim は CBF のマージン付き安全域
- サーボの指令範囲は ±120°、XML の可動域は一部 ±180°
- `IMU_MOUNT_QUAT` は XML の IMU site に合わせた値で、実機の取り付け向きは未確認

---

## 次のステップ

1. WSL2/GPU 環境で `./venv_wsl/bin/python -m pytest tests/ -v` を実行する
2. 変更をコミットしてから、`--seed` だけを変えて3本学習する（未コミットの変更があると起動しない）
3. 各 seed の `best_params.pkl` を `scratch/phase0_eval_diagnostics.py --episodes 200` で評価し、
   `scratch/gate_a_qualification.py --threshold <値>` で判定する
