# bipedal_robot 修正指示書（Claudeレビュー結果 → Copilot実装用）

## この文書について

Claude（設計・計画担当）がコードレビューで発見した問題点を、Copilot（実装担当）向けに
実行順の指示として整理したものです。**このリポジトリを実際に実行して検証したものではなく、
静的なコードリーディングと一部の最小再現実験（別環境でのJAX単体テスト）に基づく推論です。**
Phase 0の検証結果次第で、Phase 1以降の優先順位や内容を見直す必要があります。

各項目には確度ラベルを付けています。

- **[確認済]**: 実際にエラーを再現するコードを実行して確認した
- **[高確度]**: 該当箇所のコードと周辺の証拠（他ファイルでの既知の罠の記録、関連テストの欠落など）から、
  ほぼ確実にバグだと判断できる
- **[推定]**: コードを読んだ限りでは問題に見えるが、実機/実データを見ないと断定できない

**Copilotへの依頼**: 各Phaseは上から順に実施してください（Phase 0 → Phase 1 → Phase 1.5 → Phase 2）。
Phase内の項目は番号順に1つずつ実施し、1項目終わるごとにdiffを提示して次に進んでください。
Phase 0とPhase 1の項目2（DRリファクタリング）は影響範囲が大きいので、必ず実行結果／変更内容を
人間が確認してから次のPhaseに進んでください。Phase 1.5（フォルダ整理）は動作を変えないリファクタリング
なので、Phase 1の機能修正とは別コミットに分け、各移動のあとに必ず`pytest`一式を回してください。

---

## Phase 0: コード変更前の実行検証（最優先・必須）

以下はコードを直す前に、まず「本当に起きるのか」を実行して確定させるステップです。
GPU/WSL環境で実施してください。

### 0.1 DR実装のトレーサーリーク再現確認 [確認済（簡略モデルで再現）／実物は未確認]

```bash
python3 -c "
import jax
from envs.mjx_env import SenpuuMaruMJXEnv
env = SenpuuMaruMJXEnv()
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
state = reset_fn(jax.random.PRNGKey(0))
print('reset OK')
import jax.numpy as jp
action = jp.zeros(env.action_size)
state = step_fn(state, action)
print('step OK')
"
```

- `UnexpectedTracerError` が出たら → Phase 1の項目2（DRリファクタリング）を最優先で実施。
- 何もエラーが出なければ → 項目2の緊急度を下げてよい（ただしPhase 2のvmap統合テストは追加すること。
  単発では偶然踏まなかっただけの可能性が残るため）。

### 0.2 `robot/gait_generator.py` のimportエラー再現確認 [確認済（最小再現）]

```bash
python3 -c "from real.real_env import RealRobotEnv" 2>&1 | tail -20
```

- `NameError: name 'Tuple' is not defined` が出るはずです。Phase 1の項目1で直します。

### 0.3 `scratch/gate0_standing_eval.py` のクラッシュ確認 [高確度・未実行]

```bash
python3 scratch/gate0_standing_eval.py 2>&1 | tail -20
```

- `AttributeError: 'bool' object has no attribute 'astype'` が出るはずです。Phase 1の項目11で直します。

**この3つの実行結果をログに残してから、Phase 1に進んでください。**

---

## Phase 1: 確定/高確度バグの修正（優先順位順）

### 1. `robot/gait_generator.py` — `Tuple` の import漏れ [確認済]

ファイル先頭に1行追加するだけです。

```python
from typing import Tuple
```

これにより `real/real_env.py` のトップレベル import
(`from robot.gait_generator import numpy_get_reference_trajectory`) が成立するようになります。
**他の修正より先に、まずこれだけ直してください**（以降の検証作業で `real_env.py` を import する
機会が増えるため）。

---

### 2. `envs/mjx_env.py` — DR実装が `self._mjx_model` をreset()内で書き換えている問題 [Phase 0.1の結果次第]

現状のコード（`reset()` 内、L776-777相当）:

```python
self._mjx_model = self._apply_domain_randomization(self._mjx_model, mass_scale, fric_scale, com_offset)
self._reward_system._model = self._mjx_model
```

`jax.jit(env.reset)` と `jax.jit(env.step)` は別々にトレース・コンパイルされるため、
`reset()` のトレース中に生成された `self._mjx_model`（トレーサー）を、別トレースの `step()` が
読もうとして `UnexpectedTracerError` になる、またはDRが1回サンプルされたきり
二度と更新されない、という問題です。`step()` 側は `mjx.step(self._mjx_model, d)` で
このモデルを直接読んでいます。

**修正方針**: `self.*` へのPython属性ミューテーションをやめ、DR済みモデルを`step()`側でも
再現できる形に変える。具体的には以下のどちらかを実装してください。

**案A（推奨・実装コストが低い）**: `self._mjx_model` は不変のベースモデルのままにし、
DRのスケール値（`mass_scale` / `fric_scale` / `com_offset`）を `info` に保存。`step()` の
冒頭で毎回 `self._apply_domain_randomization(self._mjx_model, info['mass_scale'], ...)` を
呼んで、その場で乱数反映済みモデルを作ってから `mjx.step()` に渡す。

```python
# reset() 側: self._mjx_model は書き換えない
info['mass_scale'] = mass_scale
info['fric_scale'] = fric_scale
info['com_offset'] = com_offset

# step() 側
randomized_model = self._apply_domain_randomization(
    self._mjx_model, info['mass_scale'], info['fric_scale'], info['com_offset']
)
d = mjx.step(randomized_model, d)
```

毎ステップ `_apply_domain_randomization` を再計算するオーバーヘッドが発生しますが、
`body_mass`/`geom_friction`/`body_ipos` のスケーリングは軽い演算なので、物理ステップ本体の
コストに比べれば無視できるはずです。`self._reward_system._model` も同様に、`reward_system.compute()`
呼び出し時に `randomized_model` を明示的に渡す形に変更してください（`self._reward_system._model = ...`
という代入自体を削除する）。

**案B**: `num_envs` 分の `mjx.Model` を学習ループの外側で事前に構築し、`step()`に環境インデックス
付きで渡す。ただしBraxの`training.wrap()`前提の構造を大きく変える必要があり、実装コストが高いです。
まずは案Aで進めてください。

**実施後**: Phase 0.1のコマンドを再実行し、エラーが出ないこと、かつ複数回reset()した際に
`randomized_model.body_mass` が呼び出しごとに変わることを確認してください。

---

### 3. `envs/mjx_env.py` + `envs/training_wrapper.py` — AutoResetWrapperがinfoをリセットしない問題 [高確度]

`envs/training_wrapper.py` の `TrainingProgressWrapper` と同じパターンで、新しいラッパー
`EpisodeInfoResetWrapper` を追加してください。**Wrapperの合成順序は
`training.wrap(env, ...)` の外側（`TrainingProgressWrapper` と同じ階層）に置くこと。**

```python
class EpisodeInfoResetWrapper(Wrapper):
    """AutoResetWrapperはpipeline_state/obs/Brax自身のinfo['steps']しかリセットしない。
    raw envが独自にinfoへ積んでいるepisode-scopedなフィールドは、そのままでは
    エピソード境界をまたいで持ち越されてしまう。これを、直前episodeがdoneだった
    envスロットについてのみ、reset()相当の初期値に戻す。"""

    RESETTABLE_KEYS = (
        "step", "last_potential", "action_history", "obs_history",
        "filtered_action", "last_action", "double_last_action", "triple_last_action",
        "servo_temp", "supply_volt", "dr_damping", "dr_friction", "dr_kp_scale",
        "disturbance_scale", "privileged_obs", "disturbance_recovery_steps",
        # 項目2でinfoに追加した場合はこれも入れる:
        "mass_scale", "fric_scale", "com_offset",
    )

    def step(self, state, action):
        was_done = state.done  # 直前ステップでこのスロットのepisodeが終わっていたか
        rng_for_fresh, next_rng = jax.random.split(state.info["rng_key"])
        fresh_state = self.env.unwrapped.reset(rng_for_fresh)
        state = self.env.step(state, action)

        def pick(fresh, cur):
            d = was_done
            if d.ndim:
                d = jp.reshape(d, [d.shape[0]] + [1] * (cur.ndim - 1))
            return jp.where(d, fresh, cur)

        new_info = dict(state.info)
        for key in self.RESETTABLE_KEYS:
            if key in fresh_state.info and key in state.info:
                new_info[key] = pick(fresh_state.info[key], state.info[key])
        return state.replace(info=new_info)
```

`train/train_mjx.py` 側の `_wrap_with_progress` を修正し、`TrainingProgressWrapper` と
同じ場所で `EpisodeInfoResetWrapper` も適用してください（両方が `AutoResetWrapper` の外側に
来るように）。`self.env.unwrapped` でraw envの `reset()` に到達できることを確認してください
（Braxの `Wrapper` 基底クラスに `unwrapped` プロパティがあるはずです）。

**実施後**: Phase 2のテスト（後述）で、同一env slotで2エピソード連続実行し、2エピソード目の
`info['step']`が0から始まること、`servo_temp`が2エピソード目で異なる値にサンプルされることを
確認してください。

---

### 4. `envs/mjx_env.py` — CBF後のderatingで安全域を超える問題 [高確度]

該当箇所（L918-940相当）:

```python
real_target_rad = current_cmd + (safe_target_rad - current_cmd) * thermal_derating * voltage_derating
```

`voltage_derating` は最大1.2まで許容されており、CBFでクランプしたはずの `safe_target_rad` を
超えてしまいます。以下のいずれかに修正してください（案Aを推奨）。

**案A**:
```python
derating = jp.clip(thermal_derating * voltage_derating, 0.0, 1.0)
real_target_rad = current_cmd + (safe_target_rad - current_cmd) * derating
```

**案B**: derating適用後に `self._cbf.filter_action()` をもう一度通す。

---

### 5. `robot/policy_network.py` — mean(loc)のクリップが二重適用されている問題 [確認済（brax 0.14.2のソースで確認）]

`make_policy_network_factory` が `mean_clip_scale=POLICY_MEAN_CLIP_SCALE` を
`ppo_networks.make_ppo_networks(...)` に渡しており、ネットワーク出力層で1回、
モンキーパッチした `clipped_create_dist` でもう1回、同じsoftsign式でクリップされています。
**`make_policy_network_factory` から `mean_clip_scale=POLICY_MEAN_CLIP_SCALE` を削除**してください
（std側のクリップ機能はbraxにネイティブ実装が無いため、`clipped_create_dist` 側は残す）。

修正後、`tests/test_policy_bounds.py` と `scratch/validate_policy_bounds.py` の両方に、
**実際のポリシーネットワークのforward passを通した**回帰テストを追加してください
（既存テストは `create_dist()` に合成logitsを直接渡しており、ネットワーク層のクリップを
経由していないため、この不具合を検出できません）。

```python
# 追加すべきテストの骨子
network = make_policy_network_factory(obs_size, action_size)
params = network.policy_network.init(jax.random.PRNGKey(0))
obs = jp.ones((1, obs_size)) * 1000.0  # 極端な入力でネットワーク出力を飽和させる
logits = network.policy_network.apply(params, obs)
dist = network.parametric_action_distribution.create_dist(logits)
assert jp.max(jp.abs(dist.loc)) <= POLICY_MEAN_CLIP_SCALE + 1e-3  # ≈3.0のはずが約2.25になっていないか確認
```

---

### 6. `real/real_io.py` + `real/real_env.py` — 実機の関節角度フィードバック欠如 [高確度]

`real/real_io.py` の `interleave_read_status()` に `CMD_SERVO_POS_READ` のポーリングを追加し、
`self.servo_positions` を実際に更新してください（パース処理自体は既にあります）。

```python
# interleave_read_status() 内、既存のtemp/vin読み出しループに追加
pos = self._read_servo_register(servo_id, self.CMD_SERVO_POS_READ)
if pos is not None:
    self.servo_positions[servo_id] = pos  # 単位変換が必要ならここで
```

`real/real_env.py` の `build_observation()` を修正し、

```python
joint_pos = self.smoothed_action.copy()  # 簡易: 指令値 ≈ 実角度
```

を、上記で更新される `self.io.servo_positions`（実際の変数名は要確認）から取得する形に
差し替えてください。ポーリング周期が足りるか（温度・電圧・位置の3種を同じ巡回に入れて
制御周期100Hzを維持できるか）は実機で確認が必要です。

---

### 7. `real/real_env.py` — `RobotConfig` の値をハードコードで再定義している問題 [確認済（コード上の事実として）]

```python
class RealRobotEnv:
    NUM_JOINTS = 20
    BASE_OBS_DIM = 84
    HISTORY_LEN = 5
    ACT_DIM = 20
    OBS_DIM = 625
```

これらを `RobotConfig.NUM_JOINTS` / `RobotConfig.BASE_OBS_DIM` / `RobotConfig.HISTORY_LEN` /
`RobotConfig.ACT_DIM` / `RobotConfig.OBS_DIM` への直接参照に置き換えてください
（`RobotConfig` は既にimport済みです）。

---

### 8. `robot/gait_generator.py` — `stand_height` が未使用でIKターゲットが破綻する問題 [高確度]

`jax_cycloid_trajectory()` と `GaitGenerator.get_foot_position()` の両方で、Z座標の計算に
`stand_height` を反映させてください。

```python
# 修正前: z_traj = jp.where(is_swing, z_swing, z_stance)
# 修正後（イメージ）:
z_traj = jp.where(is_swing, -stand_height + z_swing, -stand_height)
```

具体的な符号・基準点の取り方は、`_simple_ik_leg()`/対応するIK関数が期待する座標系
（股関節を原点とした下向き正か負か）に合わせて確認しながら実装してください。
現状は `USE_REFERENCE_GAIT=False` のため学習には影響しませんが、直しておかないと
将来この機能を有効化した瞬間に破綻したポーズが使われます。

---

### 9. `scratch/phase0_eval_diagnostics.py` と `tests/test_phase0_eval_diagnostics.py` の重複解消 [確認済]

`tests/test_phase0_eval_diagnostics.py` は `scratch/phase0_eval_diagnostics.py` の
全文コピーで、かつ `physics_diverged`/`reward_is_finite` によるterminated分類の追加
（2026-09-13改修）が `tests/` 側にしか反映されていません。

**修正方針**: `classify_termination_reason` / `kaplan_meier_survival` /
`diagnose_failure_timing` / `summarize_episode_alive` / `EpisodeResult` /
`run_episode` / `run_condition` / `_DomainRandomizationScope` などの実処理は
`scratch/phase0_eval_diagnostics.py` 側に一本化し、`physics_diverged`/`reward_is_finite`
対応もそちらに反映してください。`tests/test_phase0_eval_diagnostics.py` は

```python
from scratch.phase0_eval_diagnostics import (
    classify_termination_reason, kaplan_meier_survival,
    diagnose_failure_timing, summarize_episode_alive,
)
```

のようにimportするだけの薄いファイルにし、実際の `def test_...()` 関数を書いてください
（次項目10と合わせて実施）。

---

### 10. `tests/test_phase0_eval_diagnostics.py` に実際のpytestテストが無い問題 [確認済]

項目9の整理と合わせて、最低限以下のテストを追加してください。

```python
def test_classify_termination_reason_priority():
    assert classify_termination_reason(True, False, False, truncated=False) == "fallen_roll"
    assert classify_termination_reason(False, False, False, truncated=True) == "time_limit"
    assert classify_termination_reason(False, False, False, truncated=False,
                                        physics_diverged=True) == "physics_diverged"
    assert classify_termination_reason(False, False, False, truncated=False,
                                        reward_is_finite=False) == "reward_nan"
    assert classify_termination_reason(False, False, False, truncated=False) == "unknown_terminated"

def test_kaplan_meier_survival_basic():
    times, survival = kaplan_meier_survival([100, 200, 500], [True, True, False])
    assert survival[0] == 1.0
    assert survival[-1] < 1.0

def test_diagnose_failure_timing_early_concentration():
    result = diagnose_failure_timing([10, 15, 20], max_step=500)
    assert result["classification"] == "序盤集中"

def test_summarize_episode_alive_empty():
    result = summarize_episode_alive([])
    assert result["n"] == 0
```

pytestで `tests/test_phase0_eval_diagnostics.py` を実行し、「collected 0 items」ではなく
実際に4件以上のテストが走ることを確認してください。

---

### 11. `scratch/gate0_standing_eval.py` — `jax.jit()` を経由していない問題 [確認済]

```python
state = env.reset(rng)
...
state = env.step(state, action)
```

を

```python
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
state = reset_fn(rng)
...
state = step_fn(state, action)
```

に変更してください（`reset_fn`/`step_fn` はループの外で1回だけ作ること。項目12参照）。

---

### 12. `train/visualize_rl.py` — ループ内で`jax.jit()`を毎回呼び直している問題 [確認済]

`run_interactive()` と `render_video()` の両方で、`jax.jit(env.reset)` / `jax.jit(env.step)` を
ループの**外**で1回だけ作成し、変数に保持して使い回すように変更してください。

```python
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
state = reset_fn(rng)

while viewer.is_running():
    ...
    state = step_fn(state, action)
    ...
    if getattr(state, "done", False):
        state = reset_fn(reset_key)
```

---

## Phase 1.5: フォルダ構成の整理（統廃合）

Phase 1の機能修正が終わってから実施してください。**ここは動作を変えないリファクタリングなので、
Phase 1のdiffとは別コミットに分けること**。各ステップの後に必ず`pytest`一式を回し、
importエラーが出ないことを確認してから次のステップに進んでください。

### 13. `scratch/gate0_*.py` 3本の統合 [高確度]

`gate0_formal_eval.py` / `gate0_mujoco_eval.py` / `gate0_standing_eval.py` は、
同じ「Gate 0判定」のはずなのに以下の点でバラバラでした。

- roll/pitch閾値が10.0度・10.0度・0.25rad(≈14.3度)と不一致
- 評価時間が30秒・10秒・5ステップ(≈0.1秒)と不一致
- `gate0_formal_eval.py`と`gate0_mujoco_eval.py`の出力先が両方とも
  `log/gate0_formal/gate0_seed_{seed}.json`で衝突する（同じseedで両方走らせると上書きされる）
- `gate0_formal_eval.py`も`gate0_standing_eval.py`と同じ理由（`env.reset`/`env.step`が
  `jax.jit()`を経由していない）でクラッシュする可能性が高い。docstringでは
  「Gate 0 formal eval is the main pass/fail path」と明言されている本命スクリプトなので、
  これは項目11と同格の緊急度で扱ってください
- `gate0_formal_eval.py`の`configure_deterministic_gate0()`が`RobotConfig`のクラス属性を
  スコープなしで恒久的に書き換える。`phase0_eval_diagnostics.py`にある
  `_DomainRandomizationScope`（復元付きcontext manager）と同じパターンに揃えるべき

**統合方針**: `scratch/gate0_eval.py`という1本に統合し、`--mode`で挙動を切り替える。

```python
# scratch/gate0_eval.py の骨子
PASS_CRITERIA = {
    "max_abs_roll_deg": 10.0,
    "max_abs_pitch_deg": 10.0,
    "min_foot_geom_z_m": -0.002,
    "foot_touch_rate": 0.99,
    "torque_saturation_rate": 0.01,  # pd_hold_mujocoモードのみ使用
}

def evaluate(mode: str, seconds: float, seed: int, output_dir: Path, ...):
    """mode: 'zero_action_mjx' | 'pd_hold_mujoco' | 'policy'"""
    with DomainRandomizationScope(deterministic=True):  # _DomainRandomizationScopeを流用
        if mode == "pd_hold_mujoco":
            # gate0_mujoco_eval.py の evaluate() をそのまま移植
            ...
        else:
            env = SenpuuMaruMJXEnv()
            reset_fn = jax.jit(env.reset)   # ← jit必須(項目11と同じ修正を反映)
            step_fn = jax.jit(env.step)
            ...
            if mode == "policy":
                # gate0_formal_eval.py のcheckpointロード処理を移植
                ...
            else:  # zero_action_mjx
                action = jp.zeros(env.action_size)

    output_path = output_dir / mode / f"gate0_seed_{seed}.json"  # ← modeごとにサブディレクトリを分けて衝突を回避
    ...
```

移行後、`gate0_formal_eval.py` / `gate0_mujoco_eval.py` / `gate0_standing_eval.py` の3ファイルは削除してください。
**Phase 1の項目11（`gate0_standing_eval.py`のjit化）を先に実施済みなら、その修正内容をそのままこの統合スクリプトの
`zero_action_mjx`モードに引き継いでください**（`gate0_standing_eval.py`自体は削除されるため、修正が消えないように）。

---

### 14. `scratch/`のその他の重複整理 [確認済〜高確度]

- **`render_collision.py` + `render_pure_collision.py`** → ほぼ同一コードなので
  `render_collision.py`に`--mode {full, pure}`引数を追加して統合し、`render_pure_collision.py`は削除。
- **`validate_policy_bounds.py`** → `tests/test_policy_bounds.py::test_policy_distribution_bounds`と
  実質同一のテストです。削除し、`pytest tests/test_policy_bounds.py -v`に一本化してください。
- **`validate_progress_wrapper.py`** → production code（`TrainingProgressWrapper`）の検証なので、
  scratchではなく`tests/`が本来の置き場所です。新規`tests/test_training_wrapper.py`を作り、
  このファイルの内容と、Phase 2で追加する`test_auto_reset_resets_episode_scoped_info`等の
  統合テストをまとめてください。移行後、`scratch/validate_progress_wrapper.py`は削除。
- **`render_simulation_video.py`** → 中身は`subprocess.run(["python", "train/visualize_rl.py", "--mode", "video", ...])`
  だけの薄いラッパーです。削除し、`train/visualize_rl.py --mode video`を直接使う運用に変更してください。
- **`save_html.py`** → 出力先が`log/version_2/simulation.gif`にハードコードされたまま古くなっています
  （現在は`version_7`超え）。「保存済み軌跡(.npy)からGIFを作る」という役割は`train/export_trajectory.py`の
  延長にあるため、`train/`側に`--output`引数付きで移植し、`scratch/save_html.py`は削除してください。

---

### 15. `assets/` / `safety/` / `deploy/` / `stubs/` の整理 [確認済（依存関係をgrepで確認）]

まず重要な前提として、**`assets/`フォルダ自体は削除・統合しないでください**。`assets/`は
`fix_collision_geoms.py`という1ファイルだけでなく、`assets/humanoid/humanoid.xml`・
`assets/humanoid/humanoid_visualize.xml`・`assets/all/all.xml`というモデルデータ本体が
置かれており、`robot/config.py`の`MUJOCO_MODEL_PATH`をはじめ12箇所以上から参照されています。
移動してよいのは`fix_collision_geoms.py`という**Pythonファイル1つだけ**です。

以下、それぞれ実施してください（すべて`grep -rn "from safety\|from deploy\|from assets\|import stubs"`
で依存箇所を洗い出し済みですが、移動後に必ずリポジトリ全体で再度importエラーが無いか確認すること）。

- **`assets/fix_collision_geoms.py` → `scripts/fix_collision_geoms.py`**
  他ファイルからのimportは無く、standaloneスクリプトなので単純移動でOKです。
- **`safety/cbf.py` → `envs/cbf.py`**
  参照は`envs/mjx_env.py`の1箇所のみ（`from safety.cbf import CBFSafetyFilter`）。
  これを`from envs.cbf import CBFSafetyFilter`（または相対import）に書き換えてください。
  移行後`safety/`フォルダごと削除してよいです。
- **`deploy/export_onnx.py` → `train/export_onnx.py`**
  他ファイルからのimportは無いですが、**`tests/test_policy_bounds.py`の`ENTRYPOINTS`タプルが
  パスを文字列でハードコードしています**。以下を必ず修正してください。
  ```python
  ENTRYPOINTS = (
      PROJECT_ROOT / "train" / "train_mjx.py",
      PROJECT_ROOT / "train" / "visualize_rl.py",
      PROJECT_ROOT / "train" / "export_onnx.py",  # deploy/ から変更
  )
  ```
  移行後`deploy/`フォルダごと削除してよいです。
- **`stubs/` → `real/stubs/`**
  **これだけは移動前に確認が必要です。** `stubs/board.py`・`stubs/busio.py`はどこからも
  `import stubs...`されておらず、`stubs/`ディレクトリ自体を`PYTHONPATH`に追加する方式
  （CircuitPythonの`import board`をそのまま通す）で使われていると推測されますが、
  その設定を行っている`conftest.py`・`pytest.ini`・`pyproject.toml`・CI設定などが
  今回のレビュー対象（Pythonファイル一式）には含まれていませんでした。**該当の設定ファイルを
  探し、`stubs/`へのパスがどこで指定されているか確認してから移動してください。** 見つからない場合は
  無理に移動せず、`stubs/`はそのまま残してPhase 3の申し送り事項として記録してください。

---

## Phase 2: 回帰防止のための統合テスト追加

現状、リポジトリ全体で `jax.vmap`/`jax.jit` を実際に使うテストが1つも存在しません
（`scratch/validate_progress_wrapper.py` はAutoResetWrapperを使わないダミー環境、
`tests/test_sensor_contract.py::test_domain_randomization_is_applied_to_physics_model` は
vmap/jitを通さない直接呼び出し、`tests/test_policy_bounds.py` はネットワークforward passを
経由しない、という具合に、いずれも「実際にBraxの訓練ループが通る経路」を検証していません）。
これが①③⑤を長期間見逃してきた根本原因なので、Phase 1と同じくらい重要です。

以下を新規追加してください（`tests/test_training_integration.py` などとして）。

```python
def test_vmap_reset_then_step_no_crash():
    """項目2の回帰防止: reset()とstep()を別々にjit/vmapしても動くことを確認する。"""
    env = SenpuuMaruMJXEnv()
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    state = reset_fn(keys)
    action = jp.zeros((4, env.action_size))
    state = step_fn(state, action)  # UnexpectedTracerErrorが出ないこと

def test_domain_randomization_differs_per_env_under_vmap():
    """項目2の回帰防止: vmap下で各並列環境が異なるDRサンプルを持つことを確認する。"""
    env = SenpuuMaruMJXEnv()
    reset_fn = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    state = reset_fn(keys)
    mass_scales = state.info["mass_scale"]  # 項目2の実装に応じてキー名を調整
    assert len(set(np.asarray(mass_scales).tolist())) > 1

def test_auto_reset_resets_episode_scoped_info():
    """項目3の回帰防止: AutoResetWrapper配下でepisodeが終わったスロットの
    info['step']等がリセットされることを確認する。"""
    # RobotConfig.MAX_EPISODE_STEPSを小さい値に一時変更するか、
    # 強制的にterminatedになる初期状態を用意して2エピソード分stepし、
    # 2エピソード目のinfo['step']が0スタートになっていることを確認する。
    ...
```

---

## Phase 3: Claudeが確認できなかったこと（引き継ぎ事項）

以下はコードリーディングの範囲では判断がつかず、実行環境・実測データ・実物のXMLが
無いと確定できません。Copilot/開発者側で判断・確認してください。

1. **項目2・3・5の修正を実際にWSL/GPU環境で走らせて、Phase 0で確認したエラーが
   本当に解消するか**。特に項目2は設計変更を伴うため、修正後に学習が実際に
   進む（reward/KLが以前のバージョンと同程度の挙動を示す）ことまで確認が必要です。
2. **`assets/fix_collision_geoms.py`**: `joint = elem.find('joint')` は1ボディにつき
   最初の1関節しか見ていません。このロボットのXMLに複合関節（1ボディに複数`<joint>`）や、
   名前無しの中間ボディを挟んだ構造が実在するかどうかは、実物の `humanoid.xml` /
   `all_clean.xml` を見ないと判断できません。存在する場合、該当ジオムの自動導出結果が
   誤っている可能性があります。
3. **`robot/config.py` の定数群の相互整合性**: 全定数を1行ずつ突き合わせる検証はしていません。
   特に報酬の重み付け（`REWARD_WEIGHTS`等）のバランスが物理的に妥当かは、実際に学習を
   回して収束挙動を見る以外に検証方法がありません。
4. **項目1〜12の修正を同時に入れたときの相互作用**: 例えば項目2（DR実装のリファクタ）の
   実装方法次第で、項目3で追加する`EpisodeInfoResetWrapper`の`RESETTABLE_KEYS`に
   含めるべきキー名が変わります。実装順序は本書の通りで問題ありませんが、項目2の
   実装が確定してから項目3のキー一覧を最終調整してください。
5. **`scripts/collision_tuner.py`のJS/HTML部分（約790行）**: Python側の
   `sync_to_training_xml()` 関数のみレビューし、フロントエンド部分は未読です。
6. **`real/real_env.py`の観測構成順序としim側(`envs/mjx_env.py`)の突き合わせ**:
   フィールドの並び順は一致を確認しましたが、実機での実測値レンジ（ノイズ幅、
   センサのスケール等）がシム側の想定と一致しているかは実機無しでは検証不能です。
7. **`train/train_mjx.py`の全体的なハイパーパラメータ**（学習率スケジュール、
   バッチサイズ等）については、明らかな実装バグ以外の「値として妥当か」という
   観点ではレビューしていません。
8. **`stubs/`を`PYTHONPATH`に通している設定ファイルの所在**：項目15で触れた通り、
   `conftest.py`・`pytest.ini`・`pyproject.toml`・CI設定など、`stubs/`ディレクトリを
   importパスに追加している箇所が今回のレビュー対象に含まれていませんでした。これが
   見つからない限り`stubs/`は移動しないでください。
9. **フォルダ整理（項目13〜15）の依存関係チェックはgrepベース**です。動的import
   （`importlib.import_module`や文字列組み立てによるパス参照）が無いことは確認できていません。
   各移動の後に必ずリポジトリ全体で`pytest`を実行し、収集エラー（collection error）が
   出ないことを確認してください。

---

## 実施順まとめ（チェックリスト）

- [ ] Phase 0.1: DRクラッシュの実機確認
- [ ] Phase 0.2: Tupleエラーの実機確認
- [ ] Phase 0.3: gate0_standing_evalクラッシュの実機確認
- [ ] 項目1: Tuple import追加
- [ ] 項目2: DR実装リファクタリング（要人間レビュー）
- [ ] 項目3: EpisodeInfoResetWrapper追加
- [ ] 項目4: CBF後derating clip
- [ ] 項目5: mean二重クリップ解消
- [ ] 項目6: 実機関節角度フィードバック配線
- [ ] 項目7: real_env.pyのRobotConfig参照化
- [ ] 項目8: stand_height反映
- [ ] 項目9: phase0診断スクリプトの重複解消
- [ ] 項目10: pytestテスト追加
- [ ] 項目11: gate0_standing_evalのjit化
- [ ] 項目12: visualize_rl.pyのjit使い回し
- [ ] 項目13: gate0_*.py 3本を`gate0_eval.py`に統合
- [ ] 項目14: scratch/その他の重複整理（render/validate/save_html等）
- [ ] 項目15: assets/safety/deploy/stubsのフォルダ整理（stubsは要事前確認）
- [ ] Phase 2: 統合テスト3本追加
- [ ] Phase 3の引き継ぎ事項を`docs/`配下に転記
