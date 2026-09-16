# Copilot向け 差分パッチ集

`docs/implementation_roadmap.md`（前回渡したロードマップ）の各Phaseに対応する、
**そのままCopilot Chatに貼れる検索/置換ブロック**です。Copilotに「設計を再検討して」
と探索させるとクレジットを消費するため、各ブロックの冒頭に付けた前置き文をそのまま
使い、"言われた通りに適用するだけ"のタスクとして渡してください。

## 使い方（推奨）

1. 1つのPhase = 1メッセージでCopilot Chatに貼る（全部まとめて貼らない。文脈が
   大きいほど誤爆・再確認のクレジットが増えるため）。
2. 各ブロックの「検索」テキストは実際のファイル内容から一字一句そのまま抜き出して
   あります。Copilotに「このテキストを探して置換して」と明示すれば、設計判断は
   発生せず機械的な置換で終わります。
3. 適用後は必ず対応するテストを実行させてください（各Phaseの末尾に記載）。

共通の前置き文（毎回コピペ推奨）：

```
以下の指示だけに従ってください。実装方針の妥当性を再検討したり、代替案を提案したり、
関係ない箇所をリファクタリングしたりしないでください。指定した「検索」テキストを
ファイル内で探し、「置換後」のコードに完全一致で置き換えてください。
```

---

## Phase A — 依存関係・壊れたテスト

### A-1. requirements-lock.txt（診断のみ、diff不可）

このファイルの中身を渡されていないため、事前に正確なdiffは書けません。以下をそのまま
Copilotに指示してください（探索は必要だが範囲は狭いのでクレジット消費は小さいです）：

```
requirements-lock.txt を開き、このリポジトリ内の import / from 文を全ファイル横断で
grep して、requirements-lock.txt に記載が無いトップレベルパッケージが無いか確認して。
特に deploy/export_onnx.py が import している tensorflow と tf2onnx、
real/real_io.py と real/real_env.py が import している pyserial（import名は serial）と
onnxruntime が記載されているか確認し、無ければ既存のバージョン指定の書式に合わせて
追記して。requirements-lock.txt 以外のファイルは変更しないで。
```

### A-2. tests/test_gui.py の削除

差分不要、コマンド一発です：

```bash
git rm tests/test_gui.py
```

理由：`from envs.base_env import MuJoCoSim` を参照しているが `envs/base_env.py` は
存在せず、`pytest` 収集時に全体が落ちるため。

---

## Phase C — Domain Randomizationの関節順マッピング漏れ

**対象ファイル**: `envs/mjx_env.py`（`step()` メソッド内）

**検索**:
```python
        joint_vel = state.pipeline_state.qvel[6:] if self._mjx_model.nq >= 7 else state.pipeline_state.qvel
        damping_torque = -info['dr_damping'] * joint_vel
        friction_torque = -info['dr_friction'] * jp.sign(joint_vel)
        
        if self._mjx_model.nq >= 7:
            qfrc_applied = qfrc_applied.at[6:].add(damping_torque + friction_torque)
        else:
            qfrc_applied = qfrc_applied.add(damping_torque + friction_torque)
```

**置換後**:
```python
        # [PHASE-C FIX] dr_damping/dr_friction は actuator順(nu,)で生成されているため、
        # 適用先も qvel[6:]（XMLツリー順）ではなく _actuator_to_qvel_idx で
        # actuator順→qvel順に変換したインデックスを使う（_get_obs()と同じ変換方式）。
        if self._mjx_model.nq >= 7:
            joint_vel = state.pipeline_state.qvel[self._actuator_to_qvel_idx]
        else:
            joint_vel = state.pipeline_state.qvel
        damping_torque = -info['dr_damping'] * joint_vel
        friction_torque = -info['dr_friction'] * jp.sign(joint_vel)
        
        if self._mjx_model.nq >= 7:
            qfrc_applied = qfrc_applied.at[self._actuator_to_qvel_idx].add(damping_torque + friction_torque)
        else:
            qfrc_applied = qfrc_applied.add(damping_torque + friction_torque)
```

**適用後の検証**:
```bash
pytest tests/test_sensor_contract.py tests/test_training_eval_contract.py -q
```

---

## Phase E — KP/KDをXMLより優先して物理モデルへ反映

**対象ファイル**: `envs/mjx_env.py`（`__init__` メソッド内）

**検索**:
```python
        sys_mj_model.opt.timestep = RobotConfig.SIM_DT
        sys_brax = sys_brax.replace(opt=sys_brax.opt.replace(timestep=RobotConfig.SIM_DT))
        mjx_model = mjx.put_model(sys_mj_model)
```

**置換後**:
```python
        sys_mj_model.opt.timestep = RobotConfig.SIM_DT
        sys_brax = sys_brax.replace(opt=sys_brax.opt.replace(timestep=RobotConfig.SIM_DT))

        # [PHASE-E FIX] RobotConfig.KP/KD をXML静的値より優先して適用する。
        # MuJoCoの<position>アクチュエータは gainprm[:,0]=kp, biasprm[:,1]=-kp,
        # biasprm[:,2]=-kv として保持される。実際の物理ステップは
        # self._mjx_model (= mjx.put_model(sys_mj_model)) を経由するため、
        # sys_mj_model側だけを書き換えれば十分反映される。
        sys_mj_model.actuator_gainprm[:, 0] = RobotConfig.KP
        sys_mj_model.actuator_biasprm[:, 1] = -RobotConfig.KP
        sys_mj_model.actuator_biasprm[:, 2] = -RobotConfig.KD

        mjx_model = mjx.put_model(sys_mj_model)
```

**適用後の検証**（新規テストとして追加を指示）:
```
tests/test_kp_kd_contract.py という新規ファイルを作り、SenpuuMaruMJXEnv() を
インスタンス化した env について、
env._mjx_model.actuator_gainprm[:, 0] が全要素 RobotConfig.KP と一致し、
env._mjx_model.actuator_biasprm[:, 2] が全要素 -RobotConfig.KD と一致することを
アサートするテストを1つ書いて。
```

---

## Phase D — 評価スクリプトの方策分布clipを学習時と統一

**訂正**: `deploy/export_onnx.py` と `train/export_trajectory.py` は既に
`from train.train_mjx import make_policy_network_factory` で学習時と同じ分布clip
（`_install_policy_std_cap()`の副作用込み）を使っています。**修正が必要なのは
`train/visualize_rl.py`だけ**です（この関数はGate 0評価・phase0診断・play_mjxからも
再importされて使われるため、ここ1箇所を直せば連鎖的に全部直ります）。

**対象ファイル**: `train/visualize_rl.py`

**検索**:
```python
def make_policy_network_factory(observation_size: int, action_size: int, preprocess_observations_fn=lambda x, _=None: x):
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )
```

**置換後**（関数定義を丸ごと削除し、代わりにimportで学習時と同じ実装を再利用）:
```python
# make_policy_network_factory は train/train_mjx.py の定義を単一の情報源として使う。
# ここで独自定義すると、学習時に適用される方策分布のclip（mean soft-clip, std clamp）
# が評価経路にだけ反映されない、という不整合が起きるため。
from train.train_mjx import make_policy_network_factory
```

このimport文は、ファイル冒頭の既存importブロック（`from brax.training.agents.ppo import
networks as ppo_networks` の直後あたり）に追加してください。`ppo_networks` はこの
ファイルの他の箇所（`build_inference_fn`内の`ppo_networks.make_inference_fn`）でも
引き続き使うので、そちらのimportは削除しないでください。

**適用後の検証**:
```
scratch/validate_policy_bounds.py を tests/test_policy_bounds.py としてコピーし、
train.visualize_rl.make_policy_network_factory と train.train_mjx.make_policy_network_factory
が同一オブジェクト（is演算子でTrue）であることを確認するテストを1行追加して。
```

---

## Phase G — デッドコード削除（低リスク・レビューが軽いのでまとめて実施可）

### G-1. `safety/cbf.py` の非推奨メソッド削除

**検索**:
```python
    def compute_cbf_penalty_legacy(
        self,
        nominal_action: jp.ndarray,
        limit_lower: jp.ndarray,
        limit_upper: jp.ndarray
    ) -> jp.ndarray:
        """
        [DEPRECATED] 旧実装。後方互換性のために保持。
        
        【使用禁止】代わりに compute_cbf_penalty(nominal_action, safe_action) を使用。
        
        旧実装の問題点:
        - filter_action() と異なる基準でペナルティ計算
        - double-counting のリスク
        
        このメソッドは近い将来削除される予定です。
        """
        safe_lower, safe_upper = self.compute_safe_margins(limit_lower, limit_upper)
        
        k = self.softplus_steepness
        upper_violation = jax.nn.softplus(k * (nominal_action - safe_upper))
        lower_violation = jax.nn.softplus(k * (safe_lower - nominal_action))
        
        return jp.sum(upper_violation + lower_violation) * self.cbf_penalty_scale
```

**置換後**: （何も書かない＝この関数定義ごと削除）

適用前に念のため確認：`grep -rn "compute_cbf_penalty_legacy" --include=*.py .` で
呼び出し元が無いことを確認してから削除してください。

### G-2. `robot/config.py` の未使用定数を削除

以下4箇所、それぞれ独立した検索/置換です。

**(a) INIT_JOINT_ANGLES**

検索:
```python
    NUM_JOINTS = len(JOINT_NAMES)
    INIT_JOINT_ANGLES = np.zeros(NUM_JOINTS)
```
置換後:
```python
    NUM_JOINTS = len(JOINT_NAMES)
```

**(b) LATENCY_STEPS**

検索:
```python
    NOISE_LIN_VEL     = 0.5   # [m/s] — IMU積分だと数秒でm/sオーダーのエラー
    
    LATENCY_STEPS = 1 # 1Mbps通信なので遅延は少ないはず

    RANDOM_MASS_SCALE = [0.97, 1.03]  # Phase 1: DR範囲を縮小して基本直立に集中
```
置換後:
```python
    NOISE_LIN_VEL     = 0.5   # [m/s] — IMU積分だと数秒でm/sオーダーのエラー

    RANDOM_MASS_SCALE = [0.97, 1.03]  # Phase 1: DR範囲を縮小して基本直立に集中
```

**(c) ALLOW_ARM_SWING / ARM_SWING_LIMIT_DEG**

検索:
```python
    MAX_SINGLE_FOOT_LIFT = 0.0
    ALLOW_ARM_SWING = True
    ARM_SWING_LIMIT_DEG = 12.0
    FOOT_CONTACT_THRESHOLD = 0.05  # [N] シミュレーション上の各足の最小接触力
```
置換後:
```python
    MAX_SINGLE_FOOT_LIFT = 0.0
    FOOT_CONTACT_THRESHOLD = 0.05  # [N] シミュレーション上の各足の最小接触力
```

**(d) FSR_CONTACT_THRESHOLD（実際に使われているのはFOOT_CONTACT_THRESHOLDのみ）**

検索:
```python
    FSR_CONTACT_THRESHOLD = 0.5
    
    # --- 2. Hardware Specs ---
```
置換後:
```python
    # --- 2. Hardware Specs ---
```

**(e) REWARD_WEIGHTS["symmetry"]**

検索:
```python
        "ang_momentum_z": 0.01,
        "ang_momentum_xy": 0.01,
        "cbf": 0.2,
        "symmetry": 0.0,
        "energy": 0.00005,
```
置換後:
```python
        "ang_momentum_z": 0.01,
        "ang_momentum_xy": 0.01,
        "cbf": 0.2,
        "energy": 0.00005,
```

適用前に念のため確認：`grep -rn "REWARD_WEIGHTS\['symmetry'\]\|REWARD_WEIGHTS\[\"symmetry\"\]\|w\.get('symmetry'\|w\['symmetry'\]" --include=*.py .`
で参照が無いことを確認してから適用してください。

### G-3. `envs/mjx_env.py` の到達不能な分岐削除

Phase Sの修正（`reset()`が必ず`info['_env_steps']=0`を設定）により、
以下の`if`ブロックは構造的に到達不能な死コードになっています。

**検索**:
```python
        if '_env_steps' not in state.info:
            info['training_progress'] = jp.clip(
                jp.asarray(env_steps, dtype=jp.float32) / float(max(RobotConfig.TOTAL_TRAINING_STEPS_ESTIMATE, 1)),
                0.0,
                1.0,
            )
        
        terminated = done
```

**置換後**:
```python
        # [PHASE-G CLEANUP] reset()は必ず info['_env_steps']=0 を設定するため、
        # 「_env_stepsが無い場合」の分岐は到達不能だった。training_progressの
        # 実際の更新は envs/training_wrapper.py::TrainingProgressWrapper が担う。
        
        terminated = done
```

**適用後の検証（Phase G全体）**:
```bash
pytest tests/ -q
python scratch/gate0_mujoco_eval.py --seconds 10
```

---

## ボーナス修正（監査リストには無いが今回のコード精査で発見）

### 実機ZUPT速度推定が機体座標系のまま積分されている

**対象ファイル**: `real/real_env.py`

**検索（import部分）**:
```python
from real.real_io import TeensySpineIO
from robot.math_utils import quat_to_euler
from robot.config import RobotConfig
from robot.gait_generator import numpy_get_reference_trajectory
```
**置換後**:
```python
from real.real_io import TeensySpineIO
from robot.math_utils import quat_to_euler, rotate_vector_by_quaternion
from robot.config import RobotConfig
from robot.gait_generator import numpy_get_reference_trajectory
```

**検索（本体部分）**:
```python
        # --- lin_vel: ZUPT (Zero-velocity Update) 推定 ---
        # IMU加速度を1ステップ積分して速度を推定し、
        # 接地検出時にドリフトをリセットする
        self._vel_estimate += lin_accel * self.dt
```
**置換後**:
```python
        # --- lin_vel: ZUPT (Zero-velocity Update) 推定 ---
        # IMU加速度を1ステップ積分して速度を推定し、
        # 接地検出時にドリフトをリセットする。
        # [BONUS FIX] BNO055のlin_accelは機体座標系(body frame)の値のため、
        # ワールド座標系速度として積分する前に姿勢クォータニオンで回転する。
        # rotate_vector_by_quaternion は robot/math_utils.py に実装済みだったが
        # これまでどこからも呼ばれていなかった（傾いた状態での速度推定が
        # 常に不正確になっていた）。
        world_accel = rotate_vector_by_quaternion(lin_accel, quat)
        self._vel_estimate += world_accel * self.dt
```

理由：`quat`は同じ関数内で既に`imu_data["quat"]`から取得済み変数なので、追加の
引数取得は不要です。

---

## Phase Bだけは自動diffにできません（要XML確認）

質量二重計上（`inertiafromgeom="true"`）の修正は、実際の`assets/humanoid/humanoid.xml`
の中身を見ないと正確な検索テキストを書けません。以下をそのままCopilotに渡してください
（Copilotにはファイルを実際に読ませる必要があるので、これだけは多少クレジットを使います）：

```
scripts/fix_visual_mass.py という新規ファイルを、以下の内容そのままで作成して：

import re
from pathlib import Path

def fix_visual_geom_mass(xml_path: Path, out_path: Path):
    text = xml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    fixed = []
    for line in lines:
        is_visual = 'contype="0"' in line and 'conaffinity="0"' in line and '<geom' in line
        if is_visual and 'density=' not in line and 'mass=' not in line:
            line = line.replace("/>", ' density="0"/>')
        fixed.append(line)
    out_path.write_text("\n".join(fixed), encoding="utf-8")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "assets" / "humanoid"
    for name in ["humanoid.xml", "humanoid_visualize.xml"]:
        p = root / name
        if p.exists():
            fix_visual_geom_mass(p, p)
            print(f"fixed: {p}")

作成後、python scripts/fix_visual_mass.py を実行して、
assets/humanoid/humanoid.xml の git diff を見せて。
それ以外のファイルは変更しないで。
```

diffが出たら、視覚用ジオム（group="1"のmesh）にだけ`density="0"`が付いているか
必ず目視確認してから採用してください。当たり判定用ジオム（`_collision`、既に
`mass="..."`が明示されているもの）に誤って`density="0"`が付いていないかも確認してください。

適用後の検証：
```bash
python scratch/probe_physical_limits.py   # total_mass_kg が修正前より減っているはず
```

---

## Phase G追加分（④：拾い漏れていたデッドコード。すべて機械的削除で安全）

### G-4. `robot/config.py` — `print_config()` の削除

**検索**:
```python
    MJX_LEARNING_RATE = 1e-4  # 学習崩壊を防ぐため低めに設定

    @classmethod
    def print_config(cls):
        print(f"=== Robot Configuration: {cls.ROBOT_NAME} ===")
        print(f"Joints: {cls.NUM_JOINTS}")
        print(f"Max Torque: {cls.MOTOR_MAX_TORQUE} Nm (HX-30HM)")
        print(f"Initial Height: {cls.INITIAL_HEIGHT} m")
        print(f"Termination Height: {cls.TERMINATION_HEIGHT} m")
        print(f"Gait Parameters: THIGH={cls.GAIT_THIGH_LEN}m, KNEE={cls.GAIT_KNEE_LEN}m")
```

**置換後**:
```python
    MJX_LEARNING_RATE = 1e-4  # 学習崩壊を防ぐため低めに設定
```

適用前に`grep -rn "print_config" --include=*.py .`で呼び出し元が無いことを確認してください。

### G-5. `robot/config.py` — G-4適用後に完全未参照になる定数の削除

G-4を先に適用してから、以下4箇所を削除してください（`ROBOT_NAME`は`print_config()`からしか
参照されていなかったため、G-4適用後は完全に死にます）。

**(a) ROBOT_NAME**

検索:
```python
    # --- 2. Hardware Specs ---
    ROBOT_NAME = "SenpuuMaru_GIY_Type"
    
    # Actuator: Hiwonder HX-30HM Serial Bus Servo (Magnetic Encoder)
```
置換後:
```python
    # --- 2. Hardware Specs ---
    
    # Actuator: Hiwonder HX-30HM Serial Bus Servo (Magnetic Encoder)
```

**(b) OUTPUT_DIR**（`BASE_DIR`/`MUJOCO_MODEL_PATH`は他所で使用中のため残す）

検索:
```python
    BASE_DIR = Path(__file__).resolve().parent.parent
    MUJOCO_MODEL_PATH = BASE_DIR / "assets" / "humanoid" / "humanoid.xml"
    OUTPUT_DIR = BASE_DIR / "log"
```
置換後:
```python
    BASE_DIR = Path(__file__).resolve().parent.parent
    MUJOCO_MODEL_PATH = BASE_DIR / "assets" / "humanoid" / "humanoid.xml"
```

**(c) MOTOR_VOLTAGE**（`envs/actuator_model.py::HX30HMModel.NOMINAL_VOLTAGE`が独自に同値を保持している）

検索:
```python
    MOTOR_MAX_TORQUE = 3.0       # [N.m] HX-30HMに合わせて修正
    MOTOR_MAX_VELOCITY = 6.5     # [rad/s] (0.19sec/60deg @11.1V)
    MOTOR_VOLTAGE = 11.1         # [V]
    
    # 関節定義 (Fusion 360のURDFとIDを一致させること)
```
置換後:
```python
    MOTOR_MAX_TORQUE = 3.0       # [N.m] HX-30HMに合わせて修正
    MOTOR_MAX_VELOCITY = 6.5     # [rad/s] (0.19sec/60deg @11.1V)
    
    # 関節定義 (Fusion 360のURDFとIDを一致させること)
```

**(d) GAIT_SWAY_WIDTH**（サイクロイド軌道はX-Z平面のみでY方向を扱わないため未参照）

検索:
```python
    GAIT_STAND_HEIGHT = 0.23  # [m] 直立時の腰の高さ
    GAIT_STEP_HEIGHT = 0.04   # [m] 足を上げる高さ
    GAIT_STEP_LENGTH = 0.10   # [m] 歩幅
    GAIT_SWAY_WIDTH = 0.03    # [m] 重心移動の幅
    GAIT_THIGH_LEN = 0.12     # [m] 大腿リンク長（股関節～膝）
    GAIT_KNEE_LEN = 0.12      # [m] 下腿リンク長（膝～足首）
```
置換後:
```python
    GAIT_STAND_HEIGHT = 0.23  # [m] 直立時の腰の高さ
    GAIT_STEP_HEIGHT = 0.04   # [m] 足を上げる高さ
    GAIT_STEP_LENGTH = 0.10   # [m] 歩幅
    GAIT_THIGH_LEN = 0.12     # [m] 大腿リンク長（股関節～膝）
    GAIT_KNEE_LEN = 0.12      # [m] 下腿リンク長（膝～足首）
```

### G-6. `real/real_env.py` — 未使用 `import os` の削除

**検索**:
```python
import os
import time
import numpy as np
from typing import Dict, Any, Optional
from collections import deque
```
**置換後**:
```python
import time
import numpy as np
from typing import Dict, Any, Optional
from collections import deque
```

（`Dict`は別件で以前から未使用と分かっていますが、優先度低として据え置き済みのため
今回は触りません。）

### 訂正：`deploy/export_onnx.py` の `import jax.numpy as jp` について

`改良案.md`にはこの行が未使用importとして挙がっていますが、**現在のコードには
この import 文自体が存在しません**（`import jax`のみで`jax.numpy`は使っていません）。
該当なしのため、このパッチは作成していません。

**適用後の検証（G-4〜G-6共通）**:
```bash
pytest tests/ -q
```

---

## Phase F追加分 — TERMINATION_HEIGHTのコメント修正（③、ロジック変更なし・安全）

変数名は変更せず（他ファイルからの参照名を変えるリスクを避けるため）、
「絶対座標系の高さ」という誤解を招くコメントだけを、実装（足裏基準の相対高さ判定）
と一致する内容に修正します。

**対象ファイル**: `robot/config.py`

**検索**:
```python
    # --- [CONFIG-3 FIXED] 初期高さを明記、終了条件を根拠付き ---
    # mjx_env.py の reset() で qpos[2] = 0.1773 として設定される
    INITIAL_HEIGHT = 0.1773  # [m] 直立姿勢での重心高さ（胴体位置）
    
    # 転倒判定の高さ閾値。初期高さから 7cm 低下したら終了と判定。
    # 根拠: 中腰姿勢（膝屈曲）での安定限界が約 0.107m（0.1773 - 0.07）
    TERMINATION_HEIGHT = INITIAL_HEIGHT - 0.07  # = 0.1073m
```

**置換後**:
```python
    # --- [CONFIG-3 FIXED] 初期高さを明記、終了条件を根拠付き ---
    # mjx_env.py の reset() で qpos[2] = 0.1773 として設定される
    # 注意: この値自体はワールド座標系での胴体初期位置だが、
    # TERMINATION_HEIGHT は envs/mjx_rewards.py::compute() 内で
    # 「足裏を基準にした相対高さ (base_pos[2] - lowest_foot_z)」との
    # 比較にのみ使われる（絶対座標の閾値ではない。"Contract Violation B"
    # 対応で相対高さ判定に統一済み）。以下の引き算は「7cmというマージン量」
    # を求めるための便宜的な計算であり、絶対座標の意味は持たない。
    INITIAL_HEIGHT = 0.1773  # [m] 直立姿勢での胴体初期位置（ワールド座標Z）
    
    # 転倒判定の高さマージン。足裏基準の相対高さがこの値を下回ったら終了。
    # 根拠: 中腰姿勢（膝屈曲）での安定限界に相当するマージンとして
    # INITIAL_HEIGHT - 0.07 を流用している（比較対象は相対高さ）。
    TERMINATION_HEIGHT = INITIAL_HEIGHT - 0.07  # = 0.1073m（相対高さの閾値）
```

---

## 要確認のうえ適用（③の一部：自動適用せず、必ず一言確認してから）

### resolve_curriculum_schedule() の削除（AA）

`train_mjx.py`側で将来呼び出す計画が無いなら削除して問題ありません。ただし
「相対進捗率→絶対ステップ数変換」を今後どこかに配線する予定があるなら**削除しないで
ください**。方針が決まってから、以下を使ってください。

**対象ファイル**: `robot/config.py`

**検索**:
```python
    @classmethod
    def resolve_curriculum_schedule(cls, total_steps: int = None) -> dict:
        """
        CURRICULUM_SCHEDULE_FRACTIONS を絶対ステップ数の辞書へ変換する。
        train_mjx.py 側で実際の総学習ステップ数(またはその推定値)が
        確定した時点で呼び出し、正しく機能する global_step 相当の値と
        併せて envs/mjx_env.py へ供給することを推奨する。
        
        [CONFIG-1 FIXED] 絶対ステップ版 CURRICULUM_SCHEDULE は廃止。
        このメソッドは「相対進捗率版から絶対ステップ版への変換」用のみ。
        """
        total = total_steps if total_steps is not None else cls.TOTAL_TRAINING_STEPS_ESTIMATE
        return {int(frac * total): scale for frac, scale in cls.CURRICULUM_SCHEDULE_FRACTIONS.items()}
```
**置換後**: （削除する場合は何も書かない）

---

## ③のうち、今回もdiff化を見送ったもの（理由つき）

以下は「機械的な検索/置換」に落とし込むと、あなたの意図と違う値を決め打ちしてしまう
リスクがあるため、あえてCopilotへの指示化を見送っています。Phase Hのissueとして
方針だけ先に決めてください。

| ID | 見送り理由 |
|---|---|
| F/G（歩容パラメータ） | `USE_REFERENCE_GAIT=False`で潜伏中。直すには`GAIT_THIGH_LEN/KNEE_LEN`の実測値が要る |
| H（ACTION_SCALE） | CBF飽和閾値との兼ね合いで再設計が必要、値を決め打ちできない |
| N/P（BusLinkerV3/BNO055UART） | 削除するか将来の代替経路として残すかは実機構成の方針次第 |
| X/AH（報酬監査閾値） | 現在のREWARD_WEIGHTSのスケール感に合わせた再較正が必要、値を決め打ちできない |
| AG（action_history初期化） | 初期化方法（DEFAULT_JOINT_ANGLES由来／delay_idx序盤固定など）の設計選択が要る |
| AI/AJ/AK | 挙動の実証確認が先（Brax内部の呼び出し順序をログで確認するなど） |
| kinematics.py統合 | `gait_generator.py`との重複実装をどちらに寄せるかの設計判断 |
