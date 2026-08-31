from typing import Any, Dict, Tuple, Union
import jax
import jax.numpy as jp
from brax import envs
from brax.envs.base import PipelineEnv, State
import mujoco
from mujoco import mjx

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler
from envs.actuator_model import ActuatorState, HX30HMModel

class SenpuuMaruMJXEnv(PipelineEnv):
    """
    MuJoCo XLA (MJX) を使用した GPU/TPU 並列学習用の強化学習環境
    BraxのPipelineEnvを継承しているため、Brax PPOとシームレスに統合可能。
    """
    
    def __init__(self, obs_noise: float = 0.01, latency_steps: int = 1, **kwargs):
        import os
        model_path = str(RobotConfig.MUJOCO_MODEL_PATH)
        if not os.path.exists(model_path):
            model_path = str(RobotConfig.URDF_PATH)
            
        if not os.path.exists(model_path):
            fallback_xml = """
<mujoco model="fallback_humanoid">
  <option timestep="0.00416667" gravity="0 0 -9.8"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1" friction="1.0 0.5 0.5"/>
    <body name="torso" pos="0 0 0.45">
      <freejoint name="root"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05" mass="2.0" rgba="0.2 0.6 1.0 1"/>
      <body name="right_thigh" pos="0.05 0 0">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.03" mass="0.5" rgba="1.0 0.4 0.2 1"/>
        <body name="right_shin" pos="0 0 -0.15">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-2.0 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.3" rgba="1.0 0.6 0.3 1"/>
        </body>
      </body>
      <body name="left_thigh" pos="-0.05 0 0">
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.03" mass="0.5" rgba="0.2 1.0 0.4 1"/>
        <body name="left_shin" pos="0 0 -0.15">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="-2.0 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" mass="0.3" rgba="0.4 1.0 0.6 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="right_hip_pitch_act" joint="right_hip_pitch" kp="20" kv="0.5" ctrlrange="-1.57 1.57"/>
    <position name="right_knee_act" joint="right_knee" kp="20" kv="0.5" ctrlrange="-2.0 0"/>
    <position name="left_hip_pitch_act" joint="left_hip_pitch" kp="20" kv="0.5" ctrlrange="-1.57 1.57"/>
    <position name="left_knee_act" joint="left_knee" kp="20" kv="0.5" ctrlrange="-2.0 0"/>
  </actuator>
</mujoco>
"""
        from brax.io import mjcf
        
        if not os.path.exists(model_path):
            print("[Warn] Model not found. Using auto-generated fallback model.")
            sys_brax = mjcf.loads(fallback_xml)
            sys_mj_model = mujoco.MjModel.from_xml_string(fallback_xml)
        else:
            sys_brax = mjcf.load(model_path)
            sys_mj_model = mujoco.MjModel.from_xml_path(model_path)
            
        sys_mj_model.opt.timestep = RobotConfig.SIM_DT
        sys_brax = sys_brax.replace(opt=sys_brax.opt.replace(timestep=RobotConfig.SIM_DT))
        mjx_model = mjx.put_model(sys_mj_model)

        super().__init__(sys_brax, backend='mjx', n_frames=RobotConfig.CONTROL_DECIMATION, **kwargs)
        
        self._mjx_model = mjx_model
        self._actuator_indices = jp.array(list(range(sys_mj_model.nu)), dtype=jp.int32)
        self.obs_noise = obs_noise
        self.latency_steps = latency_steps
        
        # [CRITICAL FIX] qpos関節順序（XML宣言順）とアクチュエータ/RobotConfigの関節順序の
        # マッピングテーブルを構築。
        # qpos: [left_shoulder_roll, ..., right_hip_yaw, ...] (XML body宣言順)
        # actuator/config: [right_hip_yaw, ..., left_shoulder_roll, ...] (アクチュエータ宣言順)
        # このマッピングがないと、DEFAULT_JOINT_ANGLESが間違った関節に適用される。
        self._actuator_to_qpos_idx = []
        for act_i in range(sys_mj_model.nu):
            jnt_id = sys_mj_model.actuator_trnid[act_i][0]
            # qpos内でのオフセットを算出 (freejointの後のインデックス)
            qpos_adr = sys_mj_model.jnt_qposadr[jnt_id]
            self._actuator_to_qpos_idx.append(qpos_adr)
        self._actuator_to_qpos_idx = jp.array(self._actuator_to_qpos_idx, dtype=jp.int32)
        
        from envs.mjx_rewards import MJXRewardSystem
        # mujoco.mj_name2id を使用してボディ名からIDを確実に取得
        left_foot_id = mujoco.mj_name2id(sys_mj_model, mujoco.mjtObj.mjOBJ_BODY, 'doutai-v5_hidaridairou_hidarikokansetu_hidarimomo_hidarihizabu_hidariaikabu_hidariashiura_hidariashiura-1')
        right_foot_id = mujoco.mj_name2id(sys_mj_model, mujoco.mjtObj.mjOBJ_BODY, 'doutai-v5_migidaitou_migikokansetu_migimomo_migihizabu_migigaikabu_migiashiura_migiashiura-1')
        
        # 万が一名前が変わっていた場合のフォールバック（部分一致検索）
        if left_foot_id == -1 or right_foot_id == -1:
             left_foot_id = [i for i in range(sys_mj_model.nbody) if 'hidariashiura' in sys_mj_model.body(i).name][0]
             right_foot_id = [i for i in range(sys_mj_model.nbody) if 'migiashiura' in sys_mj_model.body(i).name][0]

        self._reward_system = MJXRewardSystem(self._mjx_model, RobotConfig.REWARD_WEIGHTS, left_foot_id, right_foot_id)
        
        from safety.cbf import CBFSafetyFilter
        self._cbf = CBFSafetyFilter()

    @property
    def action_size(self):
        return self.sys.nu
        
    @property
    def observation_size(self):
        return RobotConfig.OBS_DIM

    def _get_curriculum_scale(self, training_progress: float) -> float:
        """
        カリキュラム学習: 学習進捗率に応じた外乱強度スケーリング
        
        [FIX] CRITICAL-1: 旧実装では global_step がエピソード毎に0にリセットされる
        ため、絶対ステップ数の閾値に到達不可能だった。
        training_progress (0.0～1.0) を直接使用することで修正。
        
        Args:
            training_progress: 学習進捗率 (0.0～1.0)、TrainingProgressWrapperから供給
            
        Returns:
            scale: [0, 1] のスケーリング係数
        """
        schedule = RobotConfig.CURRICULUM_SCHEDULE_FRACTIONS
        keys = sorted(schedule.keys())
        
        scale = jp.array(schedule[keys[0]], dtype=jp.float32)
        for key in keys:
            scale = jp.where(training_progress >= key, jp.array(schedule[key], dtype=jp.float32), scale)
        
        return jp.clip(scale, 0.0, 1.0)


    def reset(self, rng: jax.Array) -> State:
        rng, rng_noise, rng_priv = jax.random.split(rng, 3)
        
        key_mass, key_fric, key_com, key_dr1, key_dr2, key_dr3, key_scale, key_temp, key_volt = jax.random.split(rng_priv, 9)
        mass_scale = jax.random.uniform(key_mass, shape=(), minval=RobotConfig.RANDOM_MASS_SCALE[0], maxval=RobotConfig.RANDOM_MASS_SCALE[1])
        fric_scale = jax.random.uniform(key_fric, shape=(), minval=RobotConfig.RANDOM_FRICTION[0], maxval=RobotConfig.RANDOM_FRICTION[1])
        com_offset = jax.random.uniform(key_com, shape=(3,), minval=RobotConfig.RANDOM_COM_OFFSET[0], maxval=RobotConfig.RANDOM_COM_OFFSET[1])
        
        servo_temp = jax.random.uniform(key_temp, shape=(self._mjx_model.nu,), minval=RobotConfig.RANDOM_TEMP[0], maxval=RobotConfig.RANDOM_TEMP[1])
        supply_volt = jax.random.uniform(key_volt, shape=(1,), minval=RobotConfig.RANDOM_VOLT[0], maxval=RobotConfig.RANDOM_VOLT[1])
        
        privileged_obs = jp.concatenate([jp.array([mass_scale, fric_scale]), com_offset, servo_temp, supply_volt])
        
        mjx_data = mjx.make_data(self._mjx_model)
        
        nq = self._mjx_model.nq
        qpos = jp.zeros(nq)
        if nq >= 7:
            # [CRITICAL FIX] DEFAULT_JOINT_ANGLES はアクチュエータ順序（脚→腕）だが、
            # qpos は XML 宣言順序（腕→脚）。マッピングテーブルを使って
            # 各関節角度を正しい qpos インデックスに配置する。
            default_angles = jp.array(RobotConfig.DEFAULT_JOINT_ANGLES)
            for act_i in range(min(len(RobotConfig.DEFAULT_JOINT_ANGLES), self._mjx_model.nu)):
                qpos_idx = self._actuator_to_qpos_idx[act_i]
                qpos = qpos.at[qpos_idx].set(default_angles[act_i])
            
            # [FIX] 足裏が地面 (z=0) にピッタリ接地する高さ
            # MuJoCo単体検証で確認: z=0.1773m が適正な接地高さ
            qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.1773]))
            qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
            
        mjx_data = mjx_data.replace(qpos=qpos, qvel=jp.zeros(self._mjx_model.nv))
        mjx_data = mjx.forward(self._mjx_model, mjx_data)
        
        initial_potential = self._reward_system.compute_potential(mjx_data)
        
        info = {
            'step': 0,
            'global_step': 0,  # 累積学習ステップ（リセット跨ぎ）
            'phase': 0.0,
            'last_action': jp.zeros(self._mjx_model.nu),
            'action_buffer': jp.zeros(self._mjx_model.nu),
            'filtered_action': jp.zeros(self._mjx_model.nu),
            'double_last_action': jp.zeros(self._mjx_model.nu),
            'triple_last_action': jp.zeros(self._mjx_model.nu),
            'action_history': jp.zeros((RobotConfig.HISTORY_LEN, self._mjx_model.nu)),
            'obs_history': jp.zeros((RobotConfig.HISTORY_LEN, RobotConfig.BASE_OBS_DIM)),
            'servo_temp': servo_temp,
            'supply_volt': supply_volt[0],
            'dr_damping': jax.random.uniform(key_dr1, shape=(self._mjx_model.nu,), minval=0.01, maxval=0.15),
            'dr_friction': jax.random.uniform(key_dr2, shape=(self._mjx_model.nu,), minval=0.0, maxval=0.08),
            'dr_kp_scale': jax.random.uniform(key_dr3, shape=(self._mjx_model.nu,), minval=0.7, maxval=1.3),
            'disturbance_scale': jax.random.uniform(key_scale, minval=0.0, maxval=1.0),
            'privileged_obs': privileged_obs,
            'last_potential': initial_potential,
            'rng_key': rng_noise,
            'was_disturbed': jp.array(False),  # 前ステップで外乱があったか
            'disturbance_recovery_steps': jp.array(1000),  # 外乱から何ステップ経過したか
            'training_progress': jp.array(0.0),  # 学習進捗率 (0.0~1.0) — TrainingProgressWrapper から供給
            '_env_steps': jp.array(0, dtype=jp.int32),  # 累積環境ステップ — TrainingProgressWrapper が管理
            'terminated': jp.array(False),
            'truncated': jp.array(False),
            'time_out': jp.array(0.0),
        }
        
        obs, info = self._get_obs(mjx_data, info, rng_noise)
        
        reward, done, zero = jp.zeros(3)
        metrics = {
            'alive': zero, 'total_reward': zero, 'reward': zero,
            'reward_per_step': zero, 'total_penalty': zero,
            'lambda_phase': zero, 'r_cp': zero, 'r_recovery': zero,
            'r_com_stab': zero, 'pbrs_reward': zero,
            'potential': zero, 'fall_penalty': zero,
            'foot_balance': zero, 'zmp_margin': zero,
            'disturbance_recovery_bonus': zero, 'stability_index': zero,
            'curriculum_scale': zero,
            'barrier_height': zero, 'barrier_torque': zero
        }
        
        return State(mjx_data, obs, reward, done, metrics, info)


    def step(self, state: State, action: jax.Array) -> State:
        info = state.info.copy()
        
        if RobotConfig.USE_REFERENCE_GAIT:
            residual_rad = action * RobotConfig.ACTION_SCALE * 0.5
            base_target_rad = info.get('reference_action', jp.zeros(self._mjx_model.nu))
            target_rad = base_target_rad + residual_rad
        else:
            default_pose = jp.array(RobotConfig.DEFAULT_JOINT_ANGLES)
            target_rad = default_pose + action * RobotConfig.ACTION_SCALE

        # Standing-only mission constraint: target body velocity must remain zero.
        # This prevents walking objectives from creeping back into the task.
        if getattr(RobotConfig, 'TARGET_VEL_X', 0.0) != 0.0 or getattr(RobotConfig, 'TARGET_VEL_Y', 0.0) != 0.0 or getattr(RobotConfig, 'TARGET_YAW_RATE', 0.0) != 0.0:
            raise ValueError("Standing-only mission requires TARGET_VEL_X/Y/YAW_RATE all zero.")
        
        current_cmd = info['filtered_action']
        delta_rad = target_rad - current_cmd
        
        deadband_threshold = 0.02
        delta_rad = jp.where(jp.abs(delta_rad) < deadband_threshold, 0.0, delta_rad)
        
        max_delta = RobotConfig.MOTOR_MAX_VELOCITY * RobotConfig.CONTROL_DT
        delta_rad = jp.clip(delta_rad, -max_delta, max_delta)
        
        constrained_target = current_cmd + delta_rad
        
        alpha = RobotConfig.MOTOR_LPF_ALPHA
        filtered_action = (1.0 - alpha) * current_cmd + alpha * constrained_target
        info['filtered_action'] = filtered_action
        
        # Apply CBF Safety Filter (XMLの可動域を使用)
        limit_lower = self._mjx_model.actuator_ctrlrange[:, 0]
        limit_upper = self._mjx_model.actuator_ctrlrange[:, 1]
        
        safe_target_rad = self._cbf.filter_action(filtered_action, limit_lower, limit_upper)
        cbf_penalty = self._cbf.compute_cbf_penalty(target_rad, limit_lower, limit_upper)
        
        # Thermal & Voltage Derating
        act_state = ActuatorState(temperature=info['servo_temp'], supply_voltage=info['supply_volt'])
        thermal_derating = HX30HMModel.compute_thermal_derating(act_state.temperature)
        voltage_derating = HX30HMModel.compute_voltage_derating(act_state.supply_voltage)
        real_target_rad = current_cmd + (safe_target_rad - current_cmd) * thermal_derating * voltage_derating
        
        approx_torque = (real_target_rad - current_cmd) * getattr(RobotConfig, 'KP', 20.0)
        new_act_state = HX30HMModel.update_temperature(act_state, approx_torque, RobotConfig.CONTROL_DT)
        info['servo_temp'] = new_act_state.temperature
        
        # Actuator History Buffer & Stochastic Delay
        ah = jp.roll(info['action_history'], shift=-1, axis=0)
        ah = ah.at[-1].set(real_target_rad)
        info['action_history'] = ah
        
        rng_delay, rng_push, next_rng = jax.random.split(info['rng_key'], 3)
        info['rng_key'] = next_rng
        delay_idx = jax.random.randint(rng_delay, shape=(), minval=0, maxval=3)
        applied_action = ah[RobotConfig.HISTORY_LEN - 1 - delay_idx]
        
        # External disturbance
        # Phase 0 / Gate 0: external pushes are disabled until the standing-only
        # static baseline is validated. This prevents noisy disturbance signals from
        # destabilizing the PPO optimization before the base controller is stable.
        disturbance_enabled = bool(getattr(RobotConfig, 'DISTURBANCE_CURRICULUM', False))
        if disturbance_enabled:
            curriculum_disturbance_scale = self._get_curriculum_scale(info.get('training_progress', jp.array(0.0)))
            current_max_force = RobotConfig.RANDOM_PUSH_MAX_FORCE * curriculum_disturbance_scale
            rng_push_trigger, rng_push_dir = jax.random.split(rng_push, 2)
            is_push_step = jax.random.uniform(rng_push_trigger) < 0.03
            push_force_raw = jax.random.uniform(
                rng_push_dir, shape=(3,),
                minval=jp.array([-0.5, -1.0, -0.2]),
                maxval=jp.array([0.5, 1.0, 0.2])
            )
            push_force_norm = push_force_raw / (jp.linalg.norm(push_force_raw) + 1e-6)
            push_force = jp.where(is_push_step, push_force_norm * current_max_force, jp.zeros(3))
        else:
            is_push_step = jp.array(False)
            push_force = jp.zeros(3)

        # Standing-only policy must not generate stepping incentives.
        # This task is intentionally a static balancing task; no walking controller is allowed.
        if getattr(RobotConfig, 'ALLOW_WALKING', False) or getattr(RobotConfig, 'ALLOW_STEPPING', False):
            raise ValueError("Walking/stepping is forbidden for this task.")
        
        qfrc_applied = jp.zeros(self._mjx_model.nv)
        if self._mjx_model.nq >= 7:
            qfrc_applied = qfrc_applied.at[0:3].set(push_force)
            
        # Domain Randomization: friction & damping
        joint_vel = state.pipeline_state.qvel[6:] if self._mjx_model.nq >= 7 else state.pipeline_state.qvel
        damping_torque = -info['dr_damping'] * joint_vel
        friction_torque = -info['dr_friction'] * jp.sign(joint_vel)
        
        if self._mjx_model.nq >= 7:
            qfrc_applied = qfrc_applied.at[6:].add(damping_torque + friction_torque)
        else:
            qfrc_applied = qfrc_applied.add(damping_torque + friction_torque)

        # Physics simulation
        def physics_step(carry, _):
            d = carry.replace(ctrl=applied_action, qfrc_applied=qfrc_applied)
            d = mjx.step(self._mjx_model, d)
            return d, None
            
        mjx_data, _ = jax.lax.scan(physics_step, state.pipeline_state, (), length=RobotConfig.CONTROL_DECIMATION)
        
        # Reward and termination
        # 外乱検出：前ステップでプッシュがあったか
        was_disturbed = jp.array(is_push_step, dtype=jp.bool_)
        
        # 外乱復帰ステップ計数
        disturbance_recovery_steps = jp.where(
            is_push_step,
            jp.array(0),  # 新しい外乱
            info.get('disturbance_recovery_steps', jp.array(0)) + 1
        )
        
        reward, done, metrics, current_potential = self._reward_system.compute(
            mjx_data, applied_action, info['last_action'], info['double_last_action'],
            info['triple_last_action'], cbf_penalty, info['last_potential'], info['step'],
            info.get('reference_action', jp.zeros(self._mjx_model.nu)),
            servo_temp=info.get('servo_temp', None), 
            supply_volt=info.get('supply_volt', 11.1),
            global_step=jp.array(info.get('global_step', 0), dtype=jp.int32),
            gait_phase=jp.asarray(info.get('phase', 0.0), dtype=jp.float32),
            was_disturbed=was_disturbed,
            disturbance_recovery_steps=disturbance_recovery_steps,
            training_progress=info.get('training_progress', jp.array(0.0)),
        )
        info['last_potential'] = current_potential
        
        info['triple_last_action'] = info['double_last_action']
        info['double_last_action'] = info['last_action']
        info['last_action'] = applied_action
        info['step'] += 1
        env_steps = jp.asarray(info.get('_env_steps', info.get('global_step', 0)), dtype=jp.int32) + 1
        info['_env_steps'] = env_steps
        info['global_step'] = env_steps
        info['phase'] = (info.get('phase', 0.0) + RobotConfig.CONTROL_DT / RobotConfig.GAIT_PERIOD) % 1.0
        info['disturbance_recovery_steps'] = disturbance_recovery_steps
        info['was_disturbed'] = was_disturbed
        if '_env_steps' not in state.info:
            info['training_progress'] = jp.clip(
                jp.asarray(env_steps, dtype=jp.float32) / float(max(RobotConfig.TOTAL_TRAINING_STEPS_ESTIMATE, 1)),
                0.0,
                1.0,
            )
        
        terminated = done
        truncated = info['step'] >= RobotConfig.MAX_EPISODE_STEPS
        done = jp.logical_or(terminated, truncated)
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['time_out'] = truncated.astype(jp.float32)
        
        rng_noise = jax.random.PRNGKey(info['step'])
        obs, info = self._get_obs(mjx_data, info, rng_noise)
        
        return state.replace(pipeline_state=mjx_data, obs=obs, reward=reward,
                             done=done.astype(jp.float32), metrics=metrics, info=info)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any], rng: jax.Array) -> Tuple[jax.Array, Dict[str, Any]]:
        if self._mjx_model.nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            base_lin_vel = data.qvel[0:3]
            base_ang_vel = data.qvel[3:6]
            joint_pos = data.qpos[7:]
            joint_vel = data.qvel[6:]
        else:
            base_pos = base_quat = base_lin_vel = base_ang_vel = jp.zeros(3)
            base_quat = jp.array([1., 0., 0., 0.])
            joint_pos = data.qpos
            joint_vel = data.qvel
            
        rpy = quat_to_euler(base_quat)
        
        nsensor = getattr(self._mjx_model, 'nsensordata', 0)
        if nsensor >= 8:
            fsr_data = data.sensordata[:8]
        elif nsensor > 0:
            pad_len = 8 - nsensor
            fsr_data = jp.concatenate([data.sensordata, jp.zeros(pad_len)])
        else:
            fsr_data = jp.zeros(8)
            
        foot_positions = jp.array(RobotConfig.FSR_POSITIONS)
        total_p = jp.sum(fsr_data) + 1e-6
        zmp_x = jp.sum(foot_positions[:, 0] * fsr_data) / total_p
        zmp_y = jp.sum(foot_positions[:, 1] * fsr_data) / total_p
        zmp = jp.array([zmp_x, zmp_y])
        
        obs_components = [base_pos, rpy, base_lin_vel, base_ang_vel, joint_pos, joint_vel, fsr_data, zmp]
        raw_obs = jp.concatenate(obs_components)
        
        rng_obs, rng_pos, rng_vel = jax.random.split(rng, 3)
        noise = jax.random.normal(rng_obs, raw_obs.shape) * self.obs_noise
        noisy_obs = raw_obs + noise
        
        pos_noise = jax.random.normal(rng_pos, (3,)) * RobotConfig.NOISE_BASE_POS
        vel_noise = jax.random.normal(rng_vel, (3,)) * RobotConfig.NOISE_LIN_VEL
        noisy_obs = noisy_obs.at[0:3].add(pos_noise)
        noisy_obs = noisy_obs.at[6:9].add(vel_noise)
        
        phase = info.get('phase', 0.0)
        phase_obs = jp.array([jp.sin(2 * jp.pi * phase), jp.cos(2 * jp.pi * phase)])
        
        from robot.gait_generator import jax_get_reference_trajectory
        ref_angles = jax_get_reference_trajectory(phase, self._mjx_model.nu)
        info['reference_action'] = ref_angles
        
        if not RobotConfig.USE_REFERENCE_GAIT:
            ref_angles_obs = jp.zeros_like(ref_angles)
        else:
            ref_angles_obs = ref_angles
        
        base_obs = jp.concatenate([noisy_obs, phase_obs, ref_angles_obs])
        
        obs_hist = info.get('obs_history', jp.zeros((RobotConfig.HISTORY_LEN, RobotConfig.BASE_OBS_DIM)))
        obs_hist = jp.roll(obs_hist, shift=-1, axis=0)
        obs_hist = obs_hist.at[-1].set(base_obs)
        info['obs_history'] = obs_hist
        
        flat_obs_hist = obs_hist.flatten()
        flat_act_hist = info.get('action_history', jp.zeros((RobotConfig.HISTORY_LEN, self._mjx_model.nu))).flatten()
        
        servo_temp = info.get('servo_temp', jp.zeros(self._mjx_model.nu))
        supply_volt = jp.array([info.get('supply_volt', 11.1)])
        
        final_obs = jp.concatenate([base_obs, flat_obs_hist, flat_act_hist, servo_temp, supply_volt])
        return final_obs, info


envs.register_environment('senpuu_maru_mjx', SenpuuMaruMJXEnv)
