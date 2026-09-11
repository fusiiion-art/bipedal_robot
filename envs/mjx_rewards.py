"""固定足立位タスク用の報酬関数 (MJXRewardSystem.compute())。

成功判定は episode_alive 単独ではなく、以下の論理積で評価する
(改良規約 §11 参照):
  alive AND both_feet_contact AND upright AND height_ok
  AND no_illegal_contact AND slip_ok AND torque_ok AND recovered_in_time

報酬の主要成分:
  - r_alive: 生存ボーナス
  - r_upright / r_com_stab: 姿勢・重心安定性
  - r_capture_point / r_recovery / r_disturbance_recovery: 外乱回復系
    (Phase 0では外乱無効のため寄与は限定的)
  - soft_penalty / safety_penalty: エネルギー・滑らかさ・CBF安全項

NaN/Inf検出:
  total_reward が clip される前に jp.isfinite で検査し、結果を
  metrics['reward_is_finite'] (1.0=正常, 0.0=非有限値検出) として返す。
  JAX JITトレース内でPythonのraiseは使えないため、フラグ経由で
  呼び出し側 (train/train_mjx.py の progress_callback) に非有限値の
  発生を伝える設計 (改良規約 §18 即時停止条件)。

注意: このファイルは envs/mjx_env.py の step() (vmap/jit内部) から
呼ばれるため、全ての引数は単一環境のスカラー(バッチ次元なし)である。
"""

import jax
import jax.numpy as jp
from typing import Tuple, Dict

from mujoco import mjx

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler
from envs.stability_metrics import StabilityMetrics

"""
================================================================================
v2.2 (2026-09 レビュー: JAX/Braxバッチ次元誤解の修正と足裏基準高さの厳密化)
================================================================================
[FIX] JAX vmap境界の誤解解消
    旧実装にあった「training_progress は shape (num_envs,) を想定」という
    設計は、Braxの仕様上完全に誤りでした。mjx_rewards.py は mjx_env.py の
    step() (vmap内部) から呼ばれるため、すべての変数はすでに単一環境の
    スカラー(バッチ次元なし)としてスライスされています。
    要素ごとの処理（batch-wise）を想定したロジックを削除し、
    正しいスカラー処理としてクリーンアップしました。

[FIX] Contract Violation B (足裏基準の高さ判定) 修正
    caveat.md の「高さは足裏を基準にする」という契約に違反し、
    終了判定(is_low)や p_barrier_height でワールド絶対座標系 Z=0 からの
    base_pos[2] が使われていました。
    両足(data.xpos)のうち低い方のZ座標を基準点とし、重心との「相対高さ」
    (relative_height) を評価・判定に使用するよう修正しました。
================================================================================
"""

class MJXRewardSystem:
    def __init__(self, model: mjx.Model, weights: dict, left_foot_id: int, right_foot_id: int):
        self._model = model
        self._nq = model.nq
        self._nu = model.nu
        self._weights = weights
        self._left_foot_id = left_foot_id
        self._right_foot_id = right_foot_id

        foot_support_radius = getattr(RobotConfig, 'FOOT_SUPPORT_RADIUS', 0.06)
        self._stability = StabilityMetrics(
            left_foot_id, right_foot_id, RobotConfig.COM_HEIGHT,
            foot_support_radius=foot_support_radius,
        )

    def compute_potential(self, data: mjx.Data, lambda_phase: jax.Array = None) -> jax.Array:
        if self._nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            rpy = quat_to_euler(base_quat)
        else:
            base_pos = jp.zeros(3)
            rpy = jp.zeros(3)

        gravity_projection = jp.cos(rpy[0]) * jp.cos(rpy[1])
        p_upright = jp.exp(-5.0 * (1.0 - gravity_projection))

        pos_err = jp.sum(jp.square(base_pos[0:2]))
        yaw_err = jp.square(rpy[2])
        p_target = jp.exp(-2.0 * pos_err - 1.0 * yaw_err)

        w = self._weights
        lp = 1.0 if lambda_phase is None else lambda_phase
        return p_upright * w['upright'] + p_target * w['target_pose'] * lp

    def _get_curriculum_disturbance_scale(self, training_progress: jax.Array) -> jax.Array:
        """
        カリキュラム学習: 学習進捗率(スカラー)に応じて外乱強度を段階的に増加。
        """
        schedule = RobotConfig.CURRICULUM_SCHEDULE_FRACTIONS
        keys = sorted(schedule.keys())

        scale = jp.array(schedule[keys[0]], dtype=jp.float32)
        for key in keys:
            scale = jp.where(training_progress >= key, jp.array(schedule[key], dtype=jp.float32), scale)

        return jp.clip(scale, 0.0, 1.0)

    def _compute_adaptive_reward_scaling(self, servo_temp: jax.Array, supply_volt: float) -> Dict[str, jax.Array]:
        max_servo_temp = jp.max(servo_temp)
        temp_stress = jp.clip((max_servo_temp - 60.0) / 20.0, 0.0, 1.0)
        volt_stress = jp.clip((10.5 - supply_volt) / 2.0, 0.0, 1.0)
        stress = jp.maximum(temp_stress, volt_stress)

        return {
            'recovery': 1.0 + stress * 0.2,
            'energy': 1.0 - stress * 0.3,
            'smoothness': 1.0 - stress * 0.3,
        }

    def _compute_disturbance_recovery_bonus(
        self,
        was_disturbed: jax.Array,
        disturbance_recovery_steps: jax.Array,
        stability_index: jax.Array,
        window_steps: float = None,
        stability_threshold: float = None,
    ) -> jax.Array:
        if window_steps is None:
            window_steps = getattr(RobotConfig, 'RECOVERY_BONUS_WINDOW_STEPS', 50)
        if stability_threshold is None:
            stability_threshold = getattr(RobotConfig, 'RECOVERY_BONUS_STABILITY_THRESHOLD', 0.7)

        steps = jp.maximum(disturbance_recovery_steps.astype(jp.float32), 0.0)
        urgency = jp.exp(-jp.log(2.0) * steps / jp.maximum(window_steps, 1.0))

        outer_cutoff = window_steps * 6.0
        is_recovering = jp.logical_and(
            disturbance_recovery_steps >= 0,
            disturbance_recovery_steps < outer_cutoff,
        )
        is_stable = stability_index > stability_threshold

        bonus = jp.where(
            jp.logical_and(is_recovering, is_stable),
            urgency * stability_index * 2.0,
            0.0,
        )
        return jp.clip(bonus, 0.0, 10.0)

    @staticmethod
    def _log_barrier_lower(x: jax.Array, x_min: jax.Array, margin: jax.Array, clip_val: jax.Array) -> jax.Array:
        gap = x - x_min
        m = jp.maximum(margin, 1e-6)
        in_margin = jp.logical_and(gap > 0.0, gap < m)
        penalty = -jp.log(jp.clip(gap / m, 1e-4, 1.0))
        val = jp.where(in_margin, penalty, 0.0)
        val = jp.where(gap <= 0.0, clip_val, val)
        return jp.clip(val, 0.0, clip_val)

    @staticmethod
    def _log_barrier_upper(x: jax.Array, x_max: jax.Array, margin: jax.Array, clip_val: jax.Array) -> jax.Array:
        gap = x_max - x
        m = jp.maximum(margin, 1e-6)
        in_margin = jp.logical_and(gap > 0.0, gap < m)
        penalty = -jp.log(jp.clip(gap / m, 1e-4, 1.0))
        val = jp.where(in_margin, penalty, 0.0)
        val = jp.where(gap <= 0.0, clip_val, val)
        return jp.clip(val, 0.0, clip_val)

    def compute(
        self,
        data: mjx.Data,
        action: jax.Array,
        last_action: jax.Array,
        double_last_action: jax.Array,
        triple_last_action: jax.Array,
        cbf_penalty: jax.Array,
        last_potential: jax.Array,
        step: jax.Array,
        reference_action: jax.Array,
        servo_temp: jax.Array = None,
        supply_volt: float = 11.1,
        global_step: jax.Array = None,
        gait_phase: float = 0.0,
        was_disturbed: jax.Array = None,
        disturbance_recovery_steps: jax.Array = None,
        training_progress: jax.Array = None,
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array], jax.Array]:

        if global_step is None:
            global_step = step
        if servo_temp is None:
            servo_temp = jp.zeros(self._nu)
        if was_disturbed is None:
            was_disturbed = jp.array(False)
        if disturbance_recovery_steps is None:
            disturbance_recovery_steps = jp.array(1000)

        # --- 1. 状態抽出 ---
        if self._nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            base_lin_vel = data.qvel[0:3]
            base_ang_vel = data.qvel[3:6]
            rpy = quat_to_euler(base_quat)
            torques = data.actuator_force
            joint_pos = data.qpos[7:]
            joint_vel = data.qvel[6:]
        else:
            base_pos = jp.zeros(3)
            rpy = jp.zeros(3)
            base_lin_vel = jp.zeros(3)
            base_ang_vel = jp.zeros(3)
            torques = jp.zeros(self._nu)
            joint_pos = data.qpos
            joint_vel = data.qvel

        subtree_com = getattr(data, 'subtree_com', None)
        com_pos = subtree_com[0] if subtree_com is not None else base_pos

        base_qacc = getattr(data, 'qacc', None)
        if base_qacc is not None and self._nq >= 7:
            com_accel = base_qacc[0:3]
        else:
            com_accel = jp.array([0.0, 0.0, -9.81])

        # --- 2. 足裏位置と相対高さの計算 (Contract Violation B 修正) ---
        left_foot_pos = data.xpos[self._left_foot_id]
        right_foot_pos = data.xpos[self._right_foot_id]

        lowest_foot_z = jp.minimum(left_foot_pos[2], right_foot_pos[2])
        # 高さは足裏を基準とする（caveat契約遵守）
        relative_height = base_pos[2] - lowest_foot_z

        # --- 3. カリキュラム学習による外乱スケーリング ---
        tp = training_progress if training_progress is not None else jp.array(0.0)
        curriculum_disturbance_scale = self._get_curriculum_disturbance_scale(tp)

        # --- 4. λ_phase の計算 ---
        lw = getattr(RobotConfig, 'LAMBDA_PHASE_WEIGHTS', None) or {
            'tilt': 5.0, 'ang_vel': 0.5, 'lin_vel_err': 1.5, 'disturbance_flag': 4.0,
        }
        z_thresh = getattr(RobotConfig, 'LAMBDA_PHASE_Z_THRESH', 0.3)
        decay_k = getattr(RobotConfig, 'LAMBDA_PHASE_DECAY_K', 10.0)

        tilt_err = jp.sqrt(jp.square(rpy[0]) + jp.square(rpy[1]))
        ang_vel_norm = jp.linalg.norm(base_ang_vel)
        lin_vel_norm = jp.linalg.norm(base_lin_vel[0:2])
        disturbance_flag = jp.where(was_disturbed, 1.0, 0.0)

        z = (
            lw['tilt'] * tilt_err +
            lw['ang_vel'] * ang_vel_norm +
            lw['lin_vel_err'] * lin_vel_norm +
            lw['disturbance_flag'] * disturbance_flag
        )
        lambda_phase = jp.clip(
            jp.exp(-decay_k * jp.maximum(0.0, z - z_thresh)),
            0.0,
            1.0,
        )

        # --- 5. 終了判定 (足裏相対高さを利用) ---
        is_fallen_roll = jp.abs(rpy[0]) > RobotConfig.TERMINATION_ROLL
        is_fallen_pitch = jp.abs(rpy[1]) > RobotConfig.TERMINATION_PITCH
        is_low = relative_height < RobotConfig.TERMINATION_HEIGHT
        done = jp.logical_or(jp.logical_or(is_fallen_roll, is_fallen_pitch), is_low)

        # --- 6. 高度な安定性メトリクス計算 ---
        has_sensors = data.sensordata.shape[0] > 0
        left_foot_force = jp.where(has_sensors, jp.clip(jp.mean(jp.abs(data.sensordata[0:4] + 1e-6)), 0.0, 100.0), 0.5)
        right_foot_force = jp.where(has_sensors, jp.clip(jp.mean(jp.abs(data.sensordata[4:8] + 1e-6)), 0.0, 100.0), 0.5)
        contact_threshold = getattr(RobotConfig, 'FOOT_CONTACT_THRESHOLD', 0.05)
        both_feet_contact = jp.logical_and(
            left_foot_force > contact_threshold,
            right_foot_force > contact_threshold,
        )

        cp_margin_norm_dist = getattr(RobotConfig, 'CP_MARGIN_NORM_DIST', 0.15)
        stability_index, stability_metrics = self._stability.compute_unified_stability_index(
            com_pos, base_lin_vel, com_accel, rpy, base_ang_vel,
            left_foot_pos, right_foot_pos,
            left_foot_force, right_foot_force,
            gait_phase=gait_phase,
            cp_margin_norm_dist=cp_margin_norm_dist,
        )

        # --- 7. 報酬の計算 ---
        r_alive = 1.0

        current_potential = self.compute_potential(data, lambda_phase)
        gamma = 0.99
        r_pbrs = gamma * current_potential - last_potential

        body_vel_xy = jp.linalg.norm(base_lin_vel[0:2])
        body_yaw_rate = jp.abs(base_ang_vel[2])

        r_com_stab = jp.exp(-10.0 * (base_lin_vel[0]**2 + base_lin_vel[1]**2))
        r_upright = jp.exp(-30.0 * tilt_err**2)
        r_still = jp.exp(-20.0 * (body_vel_xy**2 + body_yaw_rate**2))
        r_target_pose = jp.exp(-5.0 * (base_pos[0]**2 + base_pos[1]**2 + rpy[2]**2))
        r_both_feet_contact = both_feet_contact.astype(jp.float32)

        p_cp = stability_metrics['cp_point']
        swing_is_left = left_foot_force < right_foot_force
        swing_foot_2d = jp.where(swing_is_left, left_foot_pos[0:2], right_foot_pos[0:2])
        stance_foot_2d = jp.where(swing_is_left, right_foot_pos[0:2], left_foot_pos[0:2])

        cp_dist_swing = jp.linalg.norm(swing_foot_2d - p_cp)
        cp_dist_stance = jp.linalg.norm(stance_foot_2d - p_cp)
        best_cp_dist = jp.minimum(cp_dist_swing, cp_dist_stance)

        r_cp_far = jp.exp(-2.5 * best_cp_dist)
        r_cp_near = jp.exp(-15.0 * best_cp_dist ** 2)
        r_capture_point = 0.6 * r_cp_far + 0.4 * r_cp_near

        tilt_vec = jp.array([rpy[0], rpy[1]])
        ang_vel_xy = jp.array([base_ang_vel[0], base_ang_vel[1]])
        tilt_dir = tilt_vec / (jp.linalg.norm(tilt_vec) + 1e-6)
        recovery_rate = -jp.dot(tilt_dir, ang_vel_xy)
        recovery_gate = jp.tanh(tilt_err / 0.15)
        r_recovery = jp.clip(jp.maximum(0.0, recovery_rate), 0.0, 5.0) * recovery_gate

        r_impedance = jp.exp(-0.01 * jp.sum(jp.square(torques))) * stability_index

        r_disturbance_recovery = self._compute_disturbance_recovery_bonus(
            was_disturbed, disturbance_recovery_steps, stability_index
        )

        # --- 8. ペナルティ ---
        p_ang_momentum_z = jp.square(base_ang_vel[2])
        p_ang_momentum_xy = jp.square(base_ang_vel[0]) + jp.square(base_ang_vel[1])

        p_energy = jp.clip(jp.sum(jp.square(torques)), 0.0, 100.0)
        p_smoothness = jp.clip(jp.sum(jp.square(action - last_action)), 0.0, 100.0)

        foot_translation = jp.linalg.norm(base_pos[0:2])
        step_penalty = jp.clip(body_vel_xy * 10.0 + body_yaw_rate * 4.0 + foot_translation * 3.0, 0.0, 20.0)

        drift_multiplier = lambda_phase * (1.0 - stability_index * 0.3)
        p_drift = jp.clip(jp.sum(jp.square(base_pos[0:2])), 0.0, 100.0) * drift_multiplier

        p_slip = jp.clip((jp.linalg.norm(base_lin_vel) * jp.mean(jp.abs(joint_vel))) ** 2, 0.0, 100.0)

        foot_span = jp.linalg.norm(right_foot_pos[0:2] - left_foot_pos[0:2])
        stance_width_penalty = jp.clip(jp.maximum(0.0, foot_span - 0.16) * 20.0, 0.0, 20.0)

        no_step_penalty = jp.where(
            getattr(RobotConfig, 'ALLOW_WALKING', False) or getattr(RobotConfig, 'ALLOW_STEPPING', False),
            100.0,
            0.0,
        )

        # 高さバリアは足裏基準の相対高さを使用
        h_margin = getattr(RobotConfig, 'BARRIER_HEIGHT_MARGIN', 0.05)
        h_clip = getattr(RobotConfig, 'BARRIER_HEIGHT_CLIP', 5.0)
        p_barrier_height = self._log_barrier_lower(
            relative_height, RobotConfig.TERMINATION_HEIGHT, h_margin, h_clip
        )

        torque_margin_ratio = getattr(RobotConfig, 'BARRIER_TORQUE_MARGIN_RATIO', 0.15)
        t_clip = getattr(RobotConfig, 'BARRIER_TORQUE_CLIP', 5.0)
        torque_margin = RobotConfig.MOTOR_MAX_TORQUE * torque_margin_ratio
        p_barrier_torque = jp.mean(
            self._log_barrier_upper(jp.abs(torques), RobotConfig.MOTOR_MAX_TORQUE, torque_margin, t_clip)
        )

        # --- 9. アダプティブ報酬スケーリング ---
        adaptive_scaling = self._compute_adaptive_reward_scaling(servo_temp, supply_volt)

        # --- 10. ペナルティスケジューリング ---
        warmup_steps = getattr(RobotConfig, 'PENALTY_INTRA_EPISODE_WARMUP_STEPS', 30)
        intra_ep_scale = jp.clip(step / jp.maximum(warmup_steps, 1), 0.0, 1.0)
        progress_scale = 1.0 if training_progress is None else jp.clip(training_progress, 0.0, 1.0)
        penalty_scale = intra_ep_scale * progress_scale

        safety_warmup_steps = getattr(RobotConfig, 'SAFETY_PENALTY_WARMUP_STEPS', 10)
        safety_scale = jp.clip(step / jp.maximum(safety_warmup_steps, 1), 0.0, 1.0)

        # --- 11. 報酬の統合 ---
        w = self._weights

        soft_penalty = (
            p_ang_momentum_z * w['ang_momentum_z'] +
            p_ang_momentum_xy * w['ang_momentum_xy'] * lambda_phase +
            p_energy * w['energy'] * adaptive_scaling['energy'] +
            p_smoothness * w['smoothness'] * adaptive_scaling['smoothness'] +
            p_drift * w['drift'] +
            p_slip * w['slip'] * lambda_phase +
            stance_width_penalty * w.get('stance_width', 0.5) +
            step_penalty +
            no_step_penalty
        ) * penalty_scale

        safety_penalty = (
            cbf_penalty * w['cbf'] +
            p_barrier_height * w.get('barrier_height', 1.0) +
            p_barrier_torque * w.get('barrier_torque', 1.0)
        ) * safety_scale

        total_reward = (
            r_alive * w['alive'] +
            r_pbrs +
            r_upright * w['upright'] +
            r_still * w['com_stab'] +
            r_target_pose * w['target_pose'] +
            r_both_feet_contact * w.get('both_feet_contact', 0.0) +

            lambda_phase * (
                r_com_stab * w['com_stab']
            ) +

            (1.0 - lambda_phase) * (
                r_capture_point * w['capture_point'] * adaptive_scaling['recovery'] +
                r_recovery * w['recovery'] * adaptive_scaling['recovery'] +
                r_impedance * w['impedance'] +
                r_disturbance_recovery
            ) -

            soft_penalty - safety_penalty
        )

        # --- NaN/Inf 検出（改良規約 §18: 即時停止条件） ---
        # clip前のtotal_rewardが非有限になっていないかをJAX互換の方法で検査する。
        # ここでは Python の if/raise は使わない (JIT トレースを壊すため)。
        # 代わりに jnp.isfinite の結果を metrics に float(0.0/1.0) として記録し、
        # 呼び出し側 (train/train_mjx.py の progress_callback) が
        # 学習ループの外側(非JIT領域)でこのフラグを見て停止判定を行う。
        reward_is_finite = jp.all(jp.isfinite(total_reward)).astype(jp.float32)

        total_reward = jp.clip(total_reward, -300.0, 300.0)
        total_reward = jp.where(done, w['fall_penalty'], total_reward)

        safe_step = jp.maximum(step, 1).astype(jp.float32)
        reward_per_step = total_reward
        total_penalty_value = soft_penalty + safety_penalty

        metrics = {
            'alive': r_alive,
            'total_reward': total_reward,
            'reward': total_reward,
            'reward_per_step': reward_per_step,
            'total_penalty': total_penalty_value,
            'lambda_phase': lambda_phase,
            'r_cp': r_capture_point,
            'r_recovery': r_recovery,
            'r_com_stab': r_com_stab,
            'both_feet_contact': r_both_feet_contact,
            'pbrs_reward': r_pbrs,
            'potential': current_potential,
            'fall_penalty': jp.where(done, w['fall_penalty'], 0.0),
            'stability_index': stability_index,
            'curriculum_scale': curriculum_disturbance_scale,
            'disturbance_recovery_bonus': r_disturbance_recovery,
            'zmp_margin': stability_metrics['zmp_margin'],
            'foot_balance': stability_metrics['foot_balance'],
            'barrier_height': p_barrier_height,
            'barrier_torque': p_barrier_torque,
            # NaN/Inf診断用フラグ (1.0=正常, 0.0=非有限値を検出)
            'reward_is_finite': reward_is_finite,
        }

        return total_reward, done, metrics, current_potential