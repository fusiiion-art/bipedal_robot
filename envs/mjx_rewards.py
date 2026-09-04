import jax
import jax.numpy as jp
from typing import Tuple, Dict

from mujoco import mjx

from robot.config import RobotConfig
from robot.math_utils import quat_to_euler
from envs.stability_metrics import StabilityMetrics

"""
================================================================================
v2 (2026-07 レビュー) での主な修正点
================================================================================
[CRITICAL-1] カリキュラム外乱スケジュール (①③関連)
    _get_curriculum_disturbance_scale() 自体のロジックは正しいが、
    envs/mjx_env.py 側で info['global_step'] が reset() の度に 0 へ
    初期化される実装になっているため、実際にこの関数へ渡る global_step は
    [0, MAX_EPISODE_STEPS=1000] の範囲しか取り得ず、CURRICULUM_SCHEDULE の
    100000/500000/2000000 という閾値に到達することが構造的に不可能。
    結果として外乱強度は学習全体を通して常に最弱ティア(5%=0.5N)に
    固定される。本ファイル内だけでは完全に修正できないため、
    envs/mjx_env.py 側の必須追加パッチをレビュー本文の④で示す。
    (このファイルの関数自体は、正しい global_step が入力された時点で
    正常に機能するよう据え置いている。)

[CRITICAL-2] λ_phase(s) の合成状態変数 z(s) (①関連)
    旧実装は z = tilt_err*5 + ang_vel_norm*0.5 のみで構成されており、
    報酬関数_詳細仕様書.md が定義した z(s)=α|θ|+β|v_err|+η|F_ext| のうち
    速度偏差項・外力検知項が欠落していた。傾き角は外力印加から
    数ステップ遅れて立ち上がるため、押された直後の1〜数ステップは
    λ_phase≈1のまま(=タスク優先のまま)残ってしまい、回復報酬への
    切り替えが遅延する「反応の空白期間」が生じていた。
    → 重心水平速度ノルムと was_disturbed フラグを z(s) に追加し、
      外力印加の瞬間にフェーズ遷移を先行させる。

[FIX] compute_potential() の λ_phase 折り込み (①関連)
    PBRSのtarget_pose(SE2追従)成分は従来λ_phaseと無関係に常時有効で
    あり、drift等の陽的ペナルティをλ_phaseで緩めても、PBRSという
    "裏口"から位置追従の圧力が回復動作と衝突していた。
    λ_phase(s) は状態sのみの関数であるため、これをΦ(s)の内部に
    折り込んでもΦ全体は依然として「状態のみに依存するスカラー関数」
    のままであり、Ng et al. (1999) のポリシー不変性定理は保持される。

[FIX] r_capture_point の多重ゲート撤廃 (②関連)
    旧: exp(-10*d^2) * stability_metrics['cp_margin'] は、CPが現在の
    両足支持基底(半幅7〜10cm程度)を外れた瞬間に完全に0となり、
    大外乱で必要な30〜50cm級のクロスオーバーステップを学習させる
    勾配が消失していた。乗算ゲートを撤廃し、遠距離では緩やかな
    指数減衰・近距離では鋭いガウス、の二段構成に変更。
    さらに接地荷重比から遊脚(Swing Leg)を推定し、CPは
    「遊脚が向かうべき目標」として評価する。

[FIX] r_recovery のオシレーション助長リスク (①関連)
    旧: -(rpy・ang_vel) は傾き角の大きさでスケールされるため、
    「大きく傾いてから勢いよく戻る」ほど無制限に高い報酬を得られ、
    減衰しない往復運動(お辞儀運動)を助長しうる報酬ハッキング経路が
    存在した。傾き方向の単位ベクトルとの内積に正規化し、
    tanhゲートでノイズレベルの微小傾きを無視するよう変更。

[FIX] penalty_scale が「エピソード内経過時間」に固定されていた問題 (①関連)
    旧: penalty_scale = clip(step/500, 0, 1) の step は info['step']
    (エピソード内カウンタ)であり、MAX_EPISODE_STEPS=1000の
    半分に相当する。これは「学習の進行に応じてペナルティを
    引き上げる」という設計意図(学習側説明書.md/報酬関数_詳細仕様書.md)
    に反し、学習が何百万ステップ進んでも "エピソード開始5秒間は
    energy/smoothness/drift/slip/cbf/ang_momentumが常にペナルティ0"
    という状態が恒久的に繰り返されていた。CBFペナルティまで
    同じスケールで抑制されていたのは安全上望ましくない。
    → 短いエピソード内グレース(物理リセット直後の過渡応答許容)と
      学習全体の進行度(training_progress、外部から供給されることを
      想定)を分離。CBF/ハードウェア系の安全項は別の高速ランプ
      (safety_scale) を用い、恒久的な5秒間フリーパスを解消。

[NEW] 緩和対数バリア関数 (Relaxed Log-Barrier) (①関連)
    報酬関数_詳細仕様書.md が提案する「安全域では0、限界近傍でのみ
    対数的に増加する」バリアを、高さ(転倒しきい値近傍)と
    関節トルク(モータ最大トルク近傍)の2軸で新規追加。
    既存の p_energy(全域に効く二次ペナルティ)とは独立に機能し、
    ハードウェア限界近傍でのみ勾配を発生させることで、
    通常域での探索を妨げない。

[FIX] 外乱復帰ボーナスの時定数 (①③関連)
    旧: max_bonus_steps=2 は 100Hz環境で20msに相当し、実際の
    プッシュリカバリー動作(重心移動・ステップ完了まで数百ms)が
    完了するには短すぎ、ボーナスがほぼ発火しない設計だった。
    ハードな時間窓によるカットオフを、滑らかな指数減衰の
    「緊急度」係数に置き換え、時定数を RobotConfig 側で
    調整可能にした(既定値: 半減期50ステップ=0.5秒)。

[NEW] data.subtree_com の利用 (②関連)
    Capture Point/ZMP計算に胴体位置(qpos[0:3])ではなく、
    可能であれば全身重心(data.subtree_com[0])を用いるよう変更。
    姿勢(upright/target_pose/drift)や終了判定は従来通り胴体基準を
    維持し、挙動の破壊的変更を避けている。フィールドが存在しない
    MJXビルドでも安全にフォールバックする。
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

        # Advanced stability metrics
        foot_support_radius = getattr(RobotConfig, 'FOOT_SUPPORT_RADIUS', 0.06)
        self._stability = StabilityMetrics(
            left_foot_id, right_foot_id, RobotConfig.COM_HEIGHT,
            foot_support_radius=foot_support_radius,
        )

    # ------------------------------------------------------------------
    # PBRS Potential
    # ------------------------------------------------------------------
    def compute_potential(self, data: mjx.Data, lambda_phase: jax.Array = None) -> jax.Array:
        """
        PBRS用のポテンシャル関数。

        [FIX] target_pose(SE2追従)成分を λ_phase(s) で変調する。
        λ_phase は状態のみの関数のため、Φ(s)全体もスカラー関数のまま
        保たれ、PBRSのポリシー不変性は失われない。
        lambda_phase=None の場合は従来通り常時フル重み(=1.0)として
        後方互換動作する(envs/mjx_env.py の reset() 呼び出しなど)。
        """
        if self._nq >= 7:
            base_pos = data.qpos[0:3]
            base_quat = data.qpos[3:7]
            rpy = quat_to_euler(base_quat)
        else:
            base_pos = jp.zeros(3)
            rpy = jp.zeros(3)

        # Upright potential
        gravity_projection = jp.cos(rpy[0]) * jp.cos(rpy[1])
        p_upright = jp.exp(-5.0 * (1.0 - gravity_projection))

        # Target Pose potential (SE2)
        pos_err = jp.sum(jp.square(base_pos[0:2]))
        yaw_err = jp.square(rpy[2])
        p_target = jp.exp(-2.0 * pos_err - 1.0 * yaw_err)

        w = self._weights
        lp = 1.0 if lambda_phase is None else lambda_phase
        return p_upright * w['upright'] + p_target * w['target_pose'] * lp

    # ------------------------------------------------------------------
    # Curriculum (外乱強度スケール)
    # ------------------------------------------------------------------
    def _get_curriculum_disturbance_scale(self, training_progress: jax.Array) -> jax.Array:
        """
        カリキュラム学習: 学習進捗率に応じて外乱強度を段階的に増加。

        [FIX] CRITICAL-1: training_progress (0.0~1.0) ベースに変更。
        CURRICULUM_SCHEDULE_FRACTIONS を使用し、学習全体の進行度に
        応じて正しくスケーリングされる。
        """
        schedule = RobotConfig.CURRICULUM_SCHEDULE_FRACTIONS
        keys = sorted(schedule.keys())

        scale = schedule[keys[0]]
        for key in keys:
            scale = jp.where(training_progress >= key, schedule[key], scale)

        return jp.clip(scale, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Adaptive Reward Scaling (熱/電圧)
    # ------------------------------------------------------------------
    def _compute_adaptive_reward_scaling(
        self,
        servo_temp: jax.Array,
        supply_volt: float
    ) -> Dict[str, jax.Array]:
        """
        サーボ温度・電圧低下時のアダプティブ報酬スケーリング。
        高温または低電圧時は recovery報酬をブースト、energy/smoothness
        ペナルティを削減する。
        """
        max_servo_temp = jp.max(servo_temp)

        temp_stress = jp.clip((max_servo_temp - 60.0) / 20.0, 0.0, 1.0)
        volt_stress = jp.clip((10.5 - supply_volt) / 2.0, 0.0, 1.0)
        stress = jp.maximum(temp_stress, volt_stress)

        return {
            'recovery': 1.0 + stress * 0.2,
            'energy': 1.0 - stress * 0.3,
            'smoothness': 1.0 - stress * 0.3,
        }

    # ------------------------------------------------------------------
    # Post-Disturbance Recovery Bonus [FIXED: smooth decay instead of hard 2-step window]
    # ------------------------------------------------------------------
    def _compute_disturbance_recovery_bonus(
        self,
        was_disturbed: jax.Array,
        disturbance_recovery_steps: jax.Array,
        stability_index: jax.Array,
        window_steps: float = None,
        stability_threshold: float = None,
    ) -> jax.Array:
        """
        外乱検出 → 復帰成功時のボーナス報酬。

        [FIX] 旧実装は max_bonus_steps=2 (20ms@100Hz) という極めて
        短いハード時間窓の中で stability_index>0.7 を満たさない限り
        ボーナスが一切発火しない設計だった。実際のプッシュリカバリー
        動作が完了するまでには数百msかかるのが通例であり、この
        窓は物理的に短すぎる。ハードカットオフを撤廃し、
        経過ステップ数に応じた滑らかな指数減衰(半減期=window_steps)の
        「緊急度」係数に置き換える。境界でのゲーミングも同時に回避。
        """
        if window_steps is None:
            window_steps = getattr(RobotConfig, 'RECOVERY_BONUS_WINDOW_STEPS', 50)
        if stability_threshold is None:
            stability_threshold = getattr(RobotConfig, 'RECOVERY_BONUS_STABILITY_THRESHOLD', 0.7)

        steps = jp.maximum(disturbance_recovery_steps.astype(jp.float32), 0.0)
        urgency = jp.exp(-jp.log(2.0) * steps / jp.maximum(window_steps, 1.0))

        # 「直近に外乱があった」ことを示す外側カットオフ。
        # (info側のセンチネル値=1000のような「外乱なし」を確実に除外する)
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

    # ------------------------------------------------------------------
    # Relaxed Log-Barrier (新規)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_barrier_lower(x: jax.Array, x_min: jax.Array, margin: jax.Array, clip_val: jax.Array) -> jax.Array:
        """下限バリア: x >= x_min+margin で0、x_min<x<x_min+margin で対数的に増加。"""
        gap = x - x_min
        m = jp.maximum(margin, 1e-6)
        in_margin = jp.logical_and(gap > 0.0, gap < m)
        penalty = -jp.log(jp.clip(gap / m, 1e-4, 1.0))
        val = jp.where(in_margin, penalty, 0.0)
        val = jp.where(gap <= 0.0, clip_val, val)
        return jp.clip(val, 0.0, clip_val)

    @staticmethod
    def _log_barrier_upper(x: jax.Array, x_max: jax.Array, margin: jax.Array, clip_val: jax.Array) -> jax.Array:
        """上限バリア: x <= x_max-margin で0、x_max-margin<x<x_max で対数的に増加。"""
        gap = x_max - x
        m = jp.maximum(margin, 1e-6)
        in_margin = jp.logical_and(gap > 0.0, gap < m)
        penalty = -jp.log(jp.clip(gap / m, 1e-4, 1.0))
        val = jp.where(in_margin, penalty, 0.0)
        val = jp.where(gap <= 0.0, clip_val, val)
        return jp.clip(val, 0.0, clip_val)

    # ------------------------------------------------------------------
    # Main compute()
    # ------------------------------------------------------------------
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

        # デフォルト値の設定
        if global_step is None:
            global_step = step  # フォールバック
        if servo_temp is None:
            servo_temp = jp.zeros(self._nu)
        if was_disturbed is None:
            was_disturbed = jp.array(False)
        if disturbance_recovery_steps is None:
            disturbance_recovery_steps = jp.array(1000)  # 「外乱なし」センチネル

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

        # [NEW] 可能であれば全身重心(subtree_com)をCapture Point/ZMP計算に使用。
        # 存在しないMJXビルドでは胴体位置にフォールバック(trace時のPython分岐
        # なので安全)。姿勢/drift/終了判定は従来通り base_pos を使用する。
        subtree_com = getattr(data, 'subtree_com', None)
        com_pos = subtree_com[0] if subtree_com is not None else base_pos

        # [NEW] 重心の実加速度。qacc は free joint の並進成分について
        # ワールド座標系(qvel[0:3]と同じ規約)であることを前提とする
        # (MuJoCo標準規約。ang成分[3:6]はローカル座標系である点に注意)。
        base_qacc = getattr(data, 'qacc', None)
        if base_qacc is not None and self._nq >= 7:
            com_accel = base_qacc[0:3]
        else:
            com_accel = jp.array([0.0, 0.0, -9.81])

        # --- 2. カリキュラム学習による外乱スケーリング (メトリクス記録用) ---
        # [FIX] CRITICAL-1: training_progress ベースに変更
        tp = training_progress if training_progress is not None else jp.array(0.0)
        curriculum_disturbance_scale = self._get_curriculum_disturbance_scale(tp)

        # --- 3. λ_phase の計算 [FIX: v_err と外乱フラグを追加] ---
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

        # --- 4. 終了判定 ---
        is_fallen_roll = jp.abs(rpy[0]) > RobotConfig.TERMINATION_ROLL
        is_fallen_pitch = jp.abs(rpy[1]) > RobotConfig.TERMINATION_PITCH
        is_low = base_pos[2] < RobotConfig.TERMINATION_HEIGHT
        done = jp.logical_or(jp.logical_or(is_fallen_roll, is_fallen_pitch), is_low)

        # --- 5. 高度な安定性メトリクス計算 ---
        left_foot_pos = data.xpos[self._left_foot_id]
        right_foot_pos = data.xpos[self._right_foot_id]

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

        # --- 6. 報酬の計算 ---
        r_alive = 1.0

        # PBRS (Potential Based Reward Shaping) [FIX: lambda_phaseをΦ内部へ折り込み]
        current_potential = self.compute_potential(data, lambda_phase)
        gamma = 0.99
        r_pbrs = gamma * current_potential - last_potential

        # タスク報酬 (λ_phase に比例)
        body_vel_xy = jp.linalg.norm(base_lin_vel[0:2])
        body_yaw_rate = jp.abs(base_ang_vel[2])

        r_com_stab = jp.exp(-10.0 * (base_lin_vel[0]**2 + base_lin_vel[1]**2))
        r_upright = jp.exp(-30.0 * tilt_err**2)
        r_still = jp.exp(-20.0 * (body_vel_xy**2 + body_yaw_rate**2))
        r_target_pose = jp.exp(-5.0 * (base_pos[0]**2 + base_pos[1]**2 + rpy[2]**2))
        r_both_feet_contact = both_feet_contact.astype(jp.float32)

        # --- Capture Point 報酬 [FIX: 多重ゲート撤廃 + swing-foot対応] ---
        p_cp = stability_metrics['cp_point']  # StabilityMetrics側と重複計算せず再利用(DRY)

        # 接地荷重の小さいほうを遊脚(Swing Leg)とみなす
        swing_is_left = left_foot_force < right_foot_force
        swing_foot_2d = jp.where(swing_is_left, left_foot_pos[0:2], right_foot_pos[0:2])
        stance_foot_2d = jp.where(swing_is_left, right_foot_pos[0:2], left_foot_pos[0:2])

        cp_dist_swing = jp.linalg.norm(swing_foot_2d - p_cp)
        cp_dist_stance = jp.linalg.norm(stance_foot_2d - p_cp)
        best_cp_dist = jp.minimum(cp_dist_swing, cp_dist_stance)

        # 遠距離: 緩やかな指数減衰(大外乱でも勾配が消えない誘導成分)
        # 近距離: 鋭いガウス(着地精度ボーナス)
        r_cp_far = jp.exp(-2.5 * best_cp_dist)
        r_cp_near = jp.exp(-15.0 * best_cp_dist ** 2)
        r_capture_point = 0.6 * r_cp_far + 0.4 * r_cp_near

        # --- 姿勢回復報酬 [FIX: 振幅非依存の正規化 + tanhゲート] ---
        tilt_vec = jp.array([rpy[0], rpy[1]])
        ang_vel_xy = jp.array([base_ang_vel[0], base_ang_vel[1]])
        tilt_dir = tilt_vec / (jp.linalg.norm(tilt_vec) + 1e-6)
        recovery_rate = -jp.dot(tilt_dir, ang_vel_xy)
        recovery_gate = jp.tanh(tilt_err / 0.15)
        r_recovery = jp.clip(jp.maximum(0.0, recovery_rate), 0.0, 5.0) * recovery_gate

        # インピーダンス報酬を統合安定性指標でスケーリング
        r_impedance = jp.exp(-0.01 * jp.sum(jp.square(torques))) * stability_index

        # 外乱復帰ボーナス [FIX: smooth decay]
        r_disturbance_recovery = self._compute_disturbance_recovery_bonus(
            was_disturbed, disturbance_recovery_steps, stability_index
        )

        # --- 7. ペナルティ ---
        p_ang_momentum_z = jp.square(base_ang_vel[2])
        p_ang_momentum_xy = jp.square(base_ang_vel[0]) + jp.square(base_ang_vel[1])

        p_energy = jp.clip(jp.sum(jp.square(torques)), 0.0, 100.0)
        p_smoothness = jp.clip(jp.sum(jp.square(action - last_action)), 0.0, 100.0)

        # Walking/stepping forbidden: penalize body motion and foot translation to keep static support.
        body_vel_xy = jp.linalg.norm(base_lin_vel[0:2])
        body_yaw_rate = jp.abs(base_ang_vel[2])
        foot_translation = jp.linalg.norm(base_pos[0:2])
        step_penalty = jp.clip(body_vel_xy * 10.0 + body_yaw_rate * 4.0 + foot_translation * 3.0, 0.0, 20.0)

        # Drift ペナルティを外乱時にミュート(stability_index も考慮)
        drift_multiplier = lambda_phase * (1.0 - stability_index * 0.3)
        p_drift = jp.clip(jp.sum(jp.square(base_pos[0:2])), 0.0, 100.0) * drift_multiplier

        p_slip = jp.clip((jp.linalg.norm(base_lin_vel) * jp.mean(jp.abs(joint_vel))) ** 2, 0.0, 100.0)

        # 片脚・両脚の足幅が極端に広がると、停止・固着の局所最適に落ちやすいため
        # ある程度の幅を超えると減点する。これにより「足を広げて止まる」挙動を抑制する。
        foot_span = jp.linalg.norm(right_foot_pos[0:2] - left_foot_pos[0:2])
        stance_width_penalty = jp.clip(jp.maximum(0.0, foot_span - 0.16) * 20.0, 0.0, 20.0)

        # Static standing constraints: no stepping, no foot translation, no single-foot lift.
        no_step_penalty = jp.where(
            getattr(RobotConfig, 'ALLOW_WALKING', False) or getattr(RobotConfig, 'ALLOW_STEPPING', False),
            100.0,
            0.0,
        )

        # [NEW] 緩和対数バリア: 高さ(転倒近傍) / トルク(モータ限界近傍)
        h_margin = getattr(RobotConfig, 'BARRIER_HEIGHT_MARGIN', 0.05)
        h_clip = getattr(RobotConfig, 'BARRIER_HEIGHT_CLIP', 5.0)
        p_barrier_height = self._log_barrier_lower(
            base_pos[2], RobotConfig.TERMINATION_HEIGHT, h_margin, h_clip
        )

        torque_margin_ratio = getattr(RobotConfig, 'BARRIER_TORQUE_MARGIN_RATIO', 0.15)
        t_clip = getattr(RobotConfig, 'BARRIER_TORQUE_CLIP', 5.0)
        torque_margin = RobotConfig.MOTOR_MAX_TORQUE * torque_margin_ratio
        p_barrier_torque = jp.mean(
            self._log_barrier_upper(jp.abs(torques), RobotConfig.MOTOR_MAX_TORQUE, torque_margin, t_clip)
        )

        # --- 8. アダプティブ報酬スケーリング ---
        adaptive_scaling = self._compute_adaptive_reward_scaling(servo_temp, supply_volt)

        # --- 9. ペナルティスケジューリング [FIX: エピソード内時間と学習進度を分離] ---
        warmup_steps = getattr(RobotConfig, 'PENALTY_INTRA_EPISODE_WARMUP_STEPS', 30)
        intra_ep_scale = jp.clip(step / jp.maximum(warmup_steps, 1), 0.0, 1.0)
        progress_scale = 1.0 if training_progress is None else jp.clip(training_progress, 0.0, 1.0)
        penalty_scale = intra_ep_scale * progress_scale

        # CBF/ハードウェア安全項は独立した高速ランプ(恒久的な5秒フリーパスを解消)
        safety_warmup_steps = getattr(RobotConfig, 'SAFETY_PENALTY_WARMUP_STEPS', 10)
        safety_scale = jp.clip(step / jp.maximum(safety_warmup_steps, 1), 0.0, 1.0)

        # --- 10. 報酬の統合 ---
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

        total_reward = jp.clip(total_reward, -300.0, 300.0)
        total_reward = jp.where(done, w['fall_penalty'], total_reward)

        # 1ステップあたりの平均報酬（エピソード長に依存しない報酬の質指標）
        safe_step = jp.maximum(step, 1).astype(jp.float32)
        reward_per_step = total_reward  # 現在ステップの即時報酬（累積ではなくステップ毎の値）

        # ペナルティ合計（監視用）
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
        }

        return total_reward, done, metrics, current_potential
