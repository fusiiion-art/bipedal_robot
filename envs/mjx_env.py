"""固定足立位ロボット用 MJX (MuJoCo XLA) 強化学習環境。

このモジュールは SenpuuMaruMJXEnv を定義する。BraxのPipelineEnvを継承し、
GPU/TPU上でのJAX並列学習に対応する。

観測 (dict, docs/master_plan.md §1.3 asymmetric actor-critic):
  'state' (RobotConfig.OBS_DIM=625): actor 用。実機 (real/real_env.py) が組み立てる
      観測と同じ内容・順序。Base observation (84) + 履歴5フレーム分 (420)
      + 指令履歴 (100) + サーボ温度 (20) + 電源電圧 (1)。
      Base observation の内訳は robot/config.py の BASE_OBS_DIM のコメント参照。
  'privileged_state' (RobotConfig.PRIVILEGED_OBS_DIM): critic 用。'state' に
      シミュレータの真値・DR値・外力を連結したもの (_get_privileged_obs)。

アクション契約 (20次元):
  関節目標角の残差 (Δq)。トルク直接指令は使わない。
  パイプライン: policy output → ACTION_SCALE → default pose加算 →
  deadband(サーボ1カウント) → 関節速度制限 → LPF → CBF → 熱/電圧derating
  → 0〜2制御周期のランダム遅延 → actuator

初期状態 (docs/master_plan.md §1.6):
  nominal姿勢の関節角・関節角速度に一様ノイズ (INIT_JOINT_POS_NOISE /
  INIT_JOINT_VEL_NOISE) を加え、低い方の足裏が nominal と同じ高さで接地する
  よう胴体高さを補正する。

外乱:
  Phase 0では DISTURBANCE_CURRICULUM=False で無効。有効時は
  RANDOM_PUSH_MAX_FORCE を上限とするランダム水平外力をqfrc_appliedへ
  加算する形で実装される。
"""

from typing import Any, Dict

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco import mjx

from robot.config import RobotConfig
from robot.math_utils import projected_gravity_jax
from envs.actuator_model import ActuatorState, HX30HMModel
from envs.cbf import CBFSafetyFilter
from envs.mjx_rewards import MJXRewardSystem, curriculum_disturbance_scale

# 関節クーロン摩擦DRの平滑化速度 [rad/s]。sign(v) のままだと静止付近で
# ±dr_friction が制御周期ごとに反転し、人工的な振動源になる。
_JOINT_FRICTION_SMOOTHING_VEL = 0.05

_LEFT_FOOT_BODY = 'doutai-v5_hidaridairou_hidarikokansetu_hidarimomo_hidarihizabu_hidariaikabu_hidariashiura_hidariashiura-1'
_RIGHT_FOOT_BODY = 'doutai-v5_migidaitou_migikokansetu_migimomo_migihizabu_migigaikabu_migiashiura_migiashiura-1'


def _box_corners(half_size: np.ndarray) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32)
    return signs * np.asarray(half_size, dtype=np.float32)


class SenpuuMaruMJXEnv(PipelineEnv):
    """MuJoCo XLA (MJX) を使用した GPU/TPU 並列学習用の強化学習環境。

    reset(rng) -> State:
        初期状態分布からspawnし、domain randomization (質量/摩擦/重心/関節粘性・摩擦/
        温度/電圧) をサンプルする。DR値は info に保持し、step() で毎回物理モデルへ反映する。

    step(state, action) -> State:
        1制御周期(CONTROL_DT=0.01s)を進める。内部でCONTROL_DECIMATION回の
        物理サブステップをjax.lax.scanで実行する。
        State.done は terminated(転倒等)のみを表し、時間切れは info['truncated'] /
        info['time_out'] に分離する(Braxの EpisodeWrapper/bootstrap_on_timeout と整合)。
    """

    def __init__(self, obs_noise: float = 0.01, max_episode_steps: int = None, **kwargs):
        # info['truncated']/info['time_out'] を立てるstep数。Brax PPOの
        # EpisodeWrapper(episode_length)と必ず一致させること(train_mjx.pyが同じ値を渡す)。
        self._max_episode_steps = int(
            RobotConfig.MAX_EPISODE_STEPS if max_episode_steps is None else max_episode_steps
        )
        if self._max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self._max_episode_steps}")

        model_path = str(RobotConfig.MUJOCO_MODEL_PATH)
        sys_brax = mjcf.load(model_path)
        mj_model = mujoco.MjModel.from_xml_path(model_path)

        mj_model.actuator_gainprm[:, 0] = RobotConfig.KP
        mj_model.actuator_biasprm[:, 1] = -RobotConfig.KP
        mj_model.actuator_biasprm[:, 2] = -RobotConfig.KD
        sys_brax = sys_brax.replace(
            actuator=sys_brax.actuator.replace(
                gain=jp.full((mj_model.nu,), RobotConfig.KP),
                bias_q=jp.full((mj_model.nu,), -RobotConfig.KP),
                bias_qd=jp.full((mj_model.nu,), -RobotConfig.KD),
            )
        )
        mj_model.opt.timestep = RobotConfig.SIM_DT
        sys_brax = sys_brax.replace(opt=sys_brax.opt.replace(timestep=RobotConfig.SIM_DT))

        super().__init__(sys_brax, backend='mjx', n_frames=RobotConfig.CONTROL_DECIMATION, **kwargs)

        if mj_model.nu != RobotConfig.NUM_JOINTS:
            raise RuntimeError(f"model has {mj_model.nu} actuators, expected {RobotConfig.NUM_JOINTS}")
        if mj_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("the first joint of the model must be the torso freejoint")

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)
        self.obs_noise = obs_noise

        joint_ids = mj_model.actuator_trnid[:, 0]
        self._actuator_to_qpos_idx = jp.asarray(mj_model.jnt_qposadr[joint_ids], dtype=jp.int32)
        self._actuator_to_qvel_idx = jp.asarray(mj_model.jnt_dofadr[joint_ids], dtype=jp.int32)
        self._joint_range = jp.asarray(mj_model.jnt_range[joint_ids], dtype=jp.float32)
        self._default_pose = jp.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=jp.float32)
        self._ctrl_lower = jp.asarray(mj_model.actuator_ctrlrange[:, 0], dtype=jp.float32)
        self._ctrl_upper = jp.asarray(mj_model.actuator_ctrlrange[:, 1], dtype=jp.float32)

        # 重心オフセットDRの適用先 (freejointを持つ胴体)
        self._root_body_id = int(mj_model.jnt_bodyid[0])

        left_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, _LEFT_FOOT_BODY)
        right_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, _RIGHT_FOOT_BODY)
        if left_foot_id < 0 or right_foot_id < 0:
            raise RuntimeError(f"foot bodies not found in {model_path}")
        self._foot_ids = jp.array([left_foot_id, right_foot_id], dtype=jp.int32)

        # spawn高さ補正用: 足裏の衝突boxの8頂点のうち最も低い点の高さ
        foot_geoms = [
            g for g in range(mj_model.ngeom)
            if mj_model.geom_bodyid[g] in (left_foot_id, right_foot_id)
            and (mj_model.geom_contype[g] or mj_model.geom_conaffinity[g])
        ]
        if not foot_geoms or any(mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in foot_geoms):
            raise RuntimeError("foot collision geoms must be boxes")
        self._foot_geom_ids = jp.asarray(foot_geoms, dtype=jp.int32)
        self._foot_geom_corners = jp.asarray(
            np.stack([_box_corners(mj_model.geom_size[g]) for g in foot_geoms]))
        self._nominal_sole_z = self._nominal_lowest_sole_z(mj_model)

        self._reward_system = MJXRewardSystem(self._mjx_model, RobotConfig.REWARD_WEIGHTS, left_foot_id, right_foot_id)
        # DEFAULT姿勢が可動域端にある関節(hip_yaw/shoulder_pitch/elbow)でCBFマージンが
        # DEFAULT自体を安全域外にしないよう、nominal_poseを渡す。
        self._cbf = CBFSafetyFilter(nominal_pose=self._default_pose)

    def _nominal_lowest_sole_z(self, mj_model) -> float:
        data = mujoco.MjData(mj_model)
        data.qpos[:] = 0.0
        data.qpos[2] = RobotConfig.INITIAL_HEIGHT
        data.qpos[3] = 1.0
        data.qpos[np.asarray(self._actuator_to_qpos_idx)] = RobotConfig.DEFAULT_JOINT_ANGLES
        mujoco.mj_kinematics(mj_model, data)
        return float(self._lowest_sole_z(jp.asarray(data.geom_xpos), jp.asarray(data.geom_xmat)))

    def _lowest_sole_z(self, geom_xpos: jax.Array, geom_xmat: jax.Array) -> jax.Array:
        pos = geom_xpos[self._foot_geom_ids]                     # (G, 3)
        rot = geom_xmat[self._foot_geom_ids].reshape(-1, 3, 3)    # (G, 3, 3)
        corners = pos[:, None, :] + jp.einsum('gij,gkj->gki', rot, self._foot_geom_corners)
        return jp.min(corners[..., 2])

    @property
    def action_size(self):
        return self._mj_model.nu

    @property
    def observation_size(self):
        return {
            'state': (RobotConfig.OBS_DIM,),
            'privileged_state': (RobotConfig.PRIVILEGED_OBS_DIM,),
        }

    def _apply_domain_randomization(self, model, mass_scale, fric_scale, com_offset):
        """質量・摩擦・胴体重心オフセットのDRを物理モデルに反映する。"""
        body_ipos = model.body_ipos.at[self._root_body_id].add(com_offset)
        return model.replace(
            body_mass=model.body_mass * mass_scale,
            geom_friction=model.geom_friction * fric_scale,
            body_ipos=body_ipos,
        )

    def _joint_dr_torque(self, qvel, dr_damping, dr_friction):
        joint_vel = qvel[self._actuator_to_qvel_idx]
        friction = dr_friction * jp.tanh(joint_vel / _JOINT_FRICTION_SMOOTHING_VEL)
        torque = -dr_damping * joint_vel - friction
        return jp.zeros(self._mjx_model.nv).at[self._actuator_to_qvel_idx].add(torque)

    def reset(self, rng: jax.Array) -> State:
        rng_noise, rng_dr, rng_init, rng_obs = jax.random.split(rng, 4)
        k_mass, k_fric, k_com, k_damp, k_jfric, k_temp, k_volt = jax.random.split(rng_dr, 7)
        nu = self._mjx_model.nu
        mass_scale = jax.random.uniform(k_mass, minval=RobotConfig.RANDOM_MASS_SCALE[0], maxval=RobotConfig.RANDOM_MASS_SCALE[1])
        fric_scale = jax.random.uniform(k_fric, minval=RobotConfig.RANDOM_FRICTION[0], maxval=RobotConfig.RANDOM_FRICTION[1])
        com_offset = jax.random.uniform(k_com, shape=(3,), minval=RobotConfig.RANDOM_COM_OFFSET[0], maxval=RobotConfig.RANDOM_COM_OFFSET[1])
        dr_damping = jax.random.uniform(k_damp, shape=(nu,), minval=0.01, maxval=0.15)
        dr_friction = jax.random.uniform(k_jfric, shape=(nu,), minval=0.0, maxval=0.08)
        servo_temp = jax.random.uniform(k_temp, shape=(nu,), minval=RobotConfig.RANDOM_TEMP[0], maxval=RobotConfig.RANDOM_TEMP[1])
        supply_volt = jax.random.uniform(k_volt, minval=RobotConfig.RANDOM_VOLT[0], maxval=RobotConfig.RANDOM_VOLT[1])

        # self._mjx_model は不変のベースモデル。DR値は info に保存し step() で毎回再構築する
        # (jax.jit(reset) と jax.jit(step) を別々にコンパイルしても tracer がリークしない)。
        randomized_model = self._apply_domain_randomization(self._mjx_model, mass_scale, fric_scale, com_offset)

        # --- 初期状態分布 (docs/master_plan.md §1.6) ---
        k_qpos, k_qvel = jax.random.split(rng_init)
        joint_pos = self._default_pose + jax.random.uniform(
            k_qpos, shape=(nu,), minval=-RobotConfig.INIT_JOINT_POS_NOISE, maxval=RobotConfig.INIT_JOINT_POS_NOISE)
        joint_pos = jp.clip(joint_pos, self._joint_range[:, 0], self._joint_range[:, 1])
        joint_vel = jax.random.uniform(
            k_qvel, shape=(nu,), minval=-RobotConfig.INIT_JOINT_VEL_NOISE, maxval=RobotConfig.INIT_JOINT_VEL_NOISE)

        qpos = jp.zeros(self._mjx_model.nq)
        qpos = qpos.at[self._actuator_to_qpos_idx].set(joint_pos)
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, RobotConfig.INITIAL_HEIGHT]))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = jp.zeros(self._mjx_model.nv).at[self._actuator_to_qvel_idx].set(joint_vel)

        data = mjx.make_data(randomized_model).replace(qpos=qpos, qvel=qvel)
        # 関節角を変えると足裏の高さ・傾きが変わるため、低い方の足裏が nominal と
        # 同じ高さ (床面すれすれ) になるよう胴体の高さを補正してから順運動学を確定する。
        data = mjx.kinematics(randomized_model, data)
        dz = self._nominal_sole_z - self._lowest_sole_z(data.geom_xpos, data.geom_xmat)
        data = data.replace(qpos=data.qpos.at[2].add(dz))
        data = mjx.forward(randomized_model, data)

        # 指令系(LPF状態・遅延バッファ・前回指令)は spawn 時の実関節角で初期化する
        # (ゼロ初期化すると直後の数stepが「全関節0rad」を指令し、足裏が滑る)。
        initial_cmd = data.qpos[self._actuator_to_qpos_idx]

        info = {
            'step': jp.array(0, dtype=jp.int32),
            'last_action': initial_cmd,
            'filtered_action': initial_cmd,
            'action_history': jp.tile(initial_cmd, (RobotConfig.HISTORY_LEN, 1)),
            'obs_history': jp.zeros((RobotConfig.HISTORY_LEN, RobotConfig.BASE_OBS_DIM)),
            'servo_temp': servo_temp,
            'supply_volt': supply_volt,
            'dr_damping': dr_damping,
            'dr_friction': dr_friction,
            'mass_scale': mass_scale,
            'fric_scale': fric_scale,
            'com_offset': com_offset,
            'last_potential': self._reward_system.compute_potential(data),
            'foot_xy0': data.xpos[self._foot_ids, 0:2],
            'rng_key': rng_noise,
            'was_disturbed': jp.array(False),
            'disturbance_force': jp.zeros(3),
            'disturbance_recovery_steps': jp.array(1000, dtype=jp.int32),
            # 直接評価(TrainingProgressWrapper無し)では学習終了時と同じ報酬(ソフトペナルティ満額)にする
            'training_progress': jp.array(1.0),
            'terminated': jp.array(False),
            'truncated': jp.array(False),
            'time_out': jp.array(0.0),
        }

        obs, info = self._get_obs(data, info, rng_obs, is_reset=True)

        zero = jp.array(0.0)
        metrics = {key: zero for key in self._reward_system.METRIC_KEYS}
        metrics.update({
            'physics_diverged': zero,
            'action_saturation': zero,
            'cbf_correction_norm': zero,
            'foot_displacement': zero,
        })
        return State(data, obs, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """1制御周期(CONTROL_DT=0.01秒)を実行する。

        Args:
            action: 20次元、[-1, 1]のpolicy出力。default_pose + action*ACTION_SCALE を
                目標関節角とする(残差方式)。
        """
        info = dict(state.info)
        target_rad = self._default_pose + action * RobotConfig.ACTION_SCALE

        # --- 指令パイプライン: deadband → 速度制限 → LPF ---
        current_cmd = info['filtered_action']
        delta_rad = target_rad - current_cmd
        delta_rad = jp.where(jp.abs(delta_rad) < RobotConfig.SERVO_POSITION_RESOLUTION, 0.0, delta_rad)
        max_delta = RobotConfig.MOTOR_MAX_VELOCITY * RobotConfig.CONTROL_DT
        delta_rad = jp.clip(delta_rad, -max_delta, max_delta)
        alpha = RobotConfig.MOTOR_LPF_ALPHA
        filtered_action = current_cmd + alpha * delta_rad
        info['filtered_action'] = filtered_action

        # --- CBF安全フィルタ ---
        lo, hi = self._ctrl_lower, self._ctrl_upper
        safe_target_rad = self._cbf.filter_action(filtered_action, lo, hi)
        # [2026-10-02 FIX] CBFペナルティ・診断は「方策の目標角がどれだけ安全域を越えたか」で測る。
        # 旧実装は target_rad と LPF/速度制限後の指令の差を測っており、可動域とは無関係な
        # 指令の追従遅れ(LPFの過渡応答)までCBF違反としてペナルティ化していた。
        clamped_target = self._cbf.filter_action(target_rad, lo, hi)
        cbf_penalty = self._cbf.compute_cbf_penalty(target_rad, clamped_target, lo, hi)

        # --- 熱・電圧 derating ---
        act_state = ActuatorState(temperature=info['servo_temp'], supply_voltage=info['supply_volt'])
        derating = jp.clip(
            HX30HMModel.compute_thermal_derating(act_state.temperature)
            * HX30HMModel.compute_voltage_derating(act_state.supply_voltage),
            0.0, 1.0,
        )
        real_target_rad = current_cmd + (safe_target_rad - current_cmd) * derating
        # 熱モデルの入力は前回の物理ステップの実トルク
        info['servo_temp'] = HX30HMModel.update_temperature(
            act_state, state.pipeline_state.actuator_force, RobotConfig.CONTROL_DT).temperature

        # --- 指令履歴と 0〜2 制御周期のランダム遅延 ---
        action_history = jp.roll(info['action_history'], shift=-1, axis=0).at[-1].set(real_target_rad)
        info['action_history'] = action_history
        rng_delay, rng_push, rng_obs, next_rng = jax.random.split(info['rng_key'], 4)
        info['rng_key'] = next_rng
        delay_idx = jax.random.randint(rng_delay, shape=(), minval=0, maxval=3)
        applied_action = action_history[RobotConfig.HISTORY_LEN - 1 - delay_idx]

        # --- 外乱 (DISTURBANCE_CURRICULUM=False の間は常に0) ---
        if RobotConfig.DISTURBANCE_CURRICULUM:
            max_force = RobotConfig.RANDOM_PUSH_MAX_FORCE * curriculum_disturbance_scale(info['training_progress'])
            rng_push_trigger, rng_push_dir = jax.random.split(rng_push)
            is_push_step = jax.random.uniform(rng_push_trigger) < 0.03
            direction = jax.random.uniform(
                rng_push_dir, shape=(3,),
                minval=jp.array([-0.5, -1.0, -0.2]), maxval=jp.array([0.5, 1.0, 0.2]))
            direction = direction / (jp.linalg.norm(direction) + 1e-6)
            push_force = jp.where(is_push_step, direction * max_force, jp.zeros(3))
        else:
            is_push_step = jp.array(False)
            push_force = jp.zeros(3)

        qfrc_applied = self._joint_dr_torque(
            state.pipeline_state.qvel, info['dr_damping'], info['dr_friction']
        ).at[0:3].add(push_force)

        randomized_model = self._apply_domain_randomization(
            self._mjx_model, info['mass_scale'], info['fric_scale'], info['com_offset'])

        def physics_step(d_prev, _):
            d = mjx.step(randomized_model, d_prev.replace(ctrl=applied_action, qfrc_applied=qfrc_applied))
            # mjx.step は NaN-in→NaN-out のため、1サブステップでも発散すると以降が全て汚染される。
            # 直前の有効な状態へロールバックし、発散した事実は done=True で持ち帰る。
            ok = jp.all(jp.isfinite(d.qpos)) & jp.all(jp.isfinite(d.qvel))
            d = jax.tree_util.tree_map(lambda new, old: jp.where(ok, new, old), d, d_prev)
            return d, jp.logical_not(ok)

        data, diverged_flags = jax.lax.scan(
            physics_step, state.pipeline_state, (), length=RobotConfig.CONTROL_DECIMATION)
        physics_diverged = jp.any(diverged_flags)

        disturbance_recovery_steps = jp.where(is_push_step, 0, info['disturbance_recovery_steps'] + 1)
        reward, done, metrics, current_potential = self._reward_system.compute(
            data, applied_action, info['last_action'], cbf_penalty,
            info['last_potential'], info['step'],
            servo_temp=info['servo_temp'],
            supply_volt=info['supply_volt'],
            was_disturbed=is_push_step,
            disturbance_recovery_steps=disturbance_recovery_steps,
            training_progress=info['training_progress'],
        )
        # 物理発散は状態をロールバック済みだが、発散直前で足止めされた状態を
        # 「良い状態」と誤学習しないよう明示的に終了させる。
        done = jp.logical_or(done, physics_diverged)
        reward = jp.where(physics_diverged, RobotConfig.REWARD_WEIGHTS['fall_penalty'], reward)

        foot_disp = data.xpos[self._foot_ids, 0:2] - info['foot_xy0']
        metrics.update({
            'physics_diverged': physics_diverged.astype(jp.float32),
            'action_saturation': self._cbf.compute_saturation_ratio(target_rad, clamped_target, lo, hi),
            'cbf_correction_norm': jp.linalg.norm(clamped_target - target_rad),
            'foot_displacement': jp.max(jp.linalg.norm(foot_disp, axis=-1)),
        })

        info['last_potential'] = current_potential
        info['last_action'] = applied_action
        info['step'] = info['step'] + 1
        info['was_disturbed'] = is_push_step
        info['disturbance_force'] = push_force
        info['disturbance_recovery_steps'] = disturbance_recovery_steps
        terminated = done
        truncated = info['step'] >= self._max_episode_steps
        info['terminated'] = terminated
        info['truncated'] = truncated
        # 時間切れと同じstepで転倒した遷移は終端であり、bootstrap してはならない。
        info['time_out'] = jp.logical_and(truncated, jp.logical_not(terminated)).astype(jp.float32)

        obs, info = self._get_obs(data, info, rng_obs)
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=done.astype(jp.float32), metrics=metrics, info=info)

    def _fsr_forces(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[RobotConfig.FSR_SENSOR_SLICE]

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any], rng: jax.Array, is_reset: bool = False):
        nu = self._mjx_model.nu
        gravity = projected_gravity_jax(data.qpos[3:7])
        lin_vel = data.qvel[0:3]
        ang_vel = data.qvel[3:6]   # freejoint の角速度は胴体座標系 (IMU gyro 相当)
        joint_pos = data.qpos[self._actuator_to_qpos_idx]
        joint_vel = data.qvel[self._actuator_to_qvel_idx]
        # 実機は Teensy が閾値判定した 0/1 フラグを送る (real/real_io.py)。
        contact = (self._fsr_forces(data) > RobotConfig.FSR_CONTACT_THRESHOLD).astype(jp.float32)

        rng_noise, rng_vel = jax.random.split(rng)
        sensed = jp.concatenate([gravity, lin_vel, ang_vel, joint_pos, joint_vel])
        sensed = sensed + jax.random.normal(rng_noise, sensed.shape) * self.obs_noise
        sensed = sensed.at[3:6].add(jax.random.normal(rng_vel, (3,)) * RobotConfig.NOISE_LIN_VEL)

        base_obs = jp.concatenate([
            jp.zeros(3),               # base_pos: 実機で取得不可のため常に0
            sensed,                    # 重力射影(3) 線速度(3) 角速度(3) 関節角(N) 関節角速度(N)
            contact,                   # FSR 接地フラグ(8)
            jp.zeros(2),               # ZMP: 実機では算出しないため常に0
            jp.array([0.0, 1.0]),      # 位相 [sin, cos]: 立位タスクでは常に位相0
            jp.zeros(nu),              # 参照角: 立位タスクでは常に0
        ])

        if is_reset:
            # 履歴を最初の観測で埋める (ゼロ埋めだと最初の数stepが「全関節0rad」の偽履歴になる)
            obs_history = jp.tile(base_obs, (RobotConfig.HISTORY_LEN, 1))
        else:
            obs_history = jp.roll(info['obs_history'], shift=-1, axis=0).at[-1].set(base_obs)
        info['obs_history'] = obs_history

        actor_obs = jp.concatenate([
            base_obs,
            obs_history.reshape(-1),
            info['action_history'].reshape(-1),
            info['servo_temp'],
            jp.reshape(info['supply_volt'], (1,)),
        ])
        privileged_obs = jp.concatenate([actor_obs, self._get_privileged_obs(data, info)])
        return {'state': actor_obs, 'privileged_state': privileged_obs}, info

    def _get_privileged_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """critic専用の観測 (シミュレータの真値・DR値・外力)。内訳は RobotConfig.PRIVILEGED_EXTRA_DIM。"""
        return jp.concatenate([
            data.qpos[0:3],
            projected_gravity_jax(data.qpos[3:7]),
            data.qvel[0:3],
            data.qvel[3:6],
            data.qpos[self._actuator_to_qpos_idx],
            data.qvel[self._actuator_to_qvel_idx],
            self._fsr_forces(data),
            (data.xpos[self._foot_ids, 0:2] - info['foot_xy0']).reshape(-1),
            jp.stack([info['mass_scale'], info['fric_scale']]),
            info['com_offset'],
            info['dr_damping'],
            info['dr_friction'],
            info['disturbance_force'],
            jp.reshape(info['was_disturbed'].astype(jp.float32), (1,)),
        ])


envs.register_environment('senpuu_maru_mjx', SenpuuMaruMJXEnv)
