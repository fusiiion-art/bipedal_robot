import numpy as np
import math
from robot.config import RobotConfig
from robot.kinematics import LegKinematics

class GaitGenerator:
    """
    人間らしい歩行軌道を生成するクラス (Open Loop Control)
    強化学習の「お手本(Reference)」や、実機の動作テストに使用する。
    """
    def __init__(self):
        self.ik = LegKinematics()
        self.dt = RobotConfig.CONTROL_DT
        
        # 歩行パラメータ (調整箇所)
        self.period = 1.0       # 1歩にかかる時間 [秒]
        self.step_height = 0.04 # 足を上げる高さ [m] (4cm)
        self.step_length = 0.10 # 歩幅 [m] (10cm)
        self.sway_width = 0.03  # 重心移動の幅 [m] (3cm)
        self.stand_height = 0.23 # 直立時の腰の高さ [m] (23cm)
        
        self.phase = 0.0 # 歩行位相 (0.0 ~ 1.0)

    def update(self):
        """時間を進めて位相を更新"""
        self.phase += self.dt / self.period
        if self.phase >= 1.0:
            self.phase -= 1.0

    def get_target_angles(self):
        """
        現在の位相における、全身の関節角度を計算する
        Returns:
            np.ndarray: 全関節の目標角度 (1次元配列)
        """
        # --- 1. 足先軌道の生成 (サイクロイド軌道などを簡易化) ---
        phase_r = self.phase
        phase_l = (self.phase + 0.5) % 1.0
        
        def get_foot_pos(p):
            x, y, z = 0.0, 0.0, -self.stand_height
            if 0.0 <= p < 0.5:
                progress = (p - 0.25) * 4.0
                x = -progress * (self.step_length / 2)
                z = -self.stand_height
            else:
                swing_p = (p - 0.5) * 2.0
                x = -math.cos(swing_p * math.pi) * (self.step_length / 2)
                z = -self.stand_height + math.sin(swing_p * math.pi) * self.step_height
            return x, y, z

        # --- 2. 重心移動 (Sway) ---
        body_y_offset = self.sway_width * math.sin(self.phase * 2 * math.pi)
        
        # --- 3. IKを解く ---
        rx, ry, rz = get_foot_pos(phase_r)
        ry -= body_y_offset 
        r_angles = self.ik.solve_leg(rx, ry, rz, is_right=True)
        
        lx, ly, lz = get_foot_pos(phase_l)
        ly -= body_y_offset
        l_angles = self.ik.solve_leg(lx, ly, lz, is_right=False)
        
        return np.concatenate([r_angles, l_angles])

# テスト実行
if __name__ == "__main__":
    gait = GaitGenerator()
    print("Testing Gait Generator...")
    for i in range(5):
        gait.update()
        angles = gait.get_target_angles()
        print(f"Phase {gait.phase:.2f}: Angles[0]={angles[0]:.3f} (Right Hip Yaw)")


# ===================================================================
# JAX Native版: MJX環境 (JITコンパイル対応) 用の歩容リファレンス生成
# envs/mjx_env.py の中から直接呼び出される
# ===================================================================
try:
    import jax.numpy as jp
except Exception:
    jp = None

# --- サイクロイド歩容パラメータ (調整箇所) ---
STEP_LENGTH = 0.10   # 歩幅 [m] (10cm)
STEP_HEIGHT = 0.04   # 足上げ高さ [m] (4cm)
STAND_HEIGHT = 0.23  # 直立時の腰高さ [m]
# 簡易リンク長 (旋風丸の大腿/下腿)
THIGH_LEN = 0.12     # [m]
KNEE_LEN  = 0.12     # [m]

def _cycloid_foot_trajectory(swing_phase: float) -> tuple:
    """
    サイクロイド曲線によるスイング脚の足先軌道
    
    パラメータ方程式:
        x(θ) = SL / (2π) * (θ - sin θ)
        z(θ) = SH * (1 - cos θ) / 2    (高さ方向は 0→SH→0)
    
    ここで θ = swing_phase * 2π (0.0 ~ 1.0 を 0 ~ 2π にマッピング)
    
    これにより「ゆっくり動き出し→中間で最速→ゆっくり止まる」という
    ジャーク最小化特性を持つ滑らかな軌道が生成される。
    """
    theta = swing_phase * 2.0 * jp.pi  # 0 ~ 2π
    
    # 前後方向 (x): サイクロイドの正式な式
    # -SL/2 (後ろ) から +SL/2 (前) へ移動
    x = STEP_LENGTH / (2.0 * jp.pi) * (theta - jp.sin(theta)) - STEP_LENGTH / 2.0
    
    # 上下方向 (z): cos成分で 0 → STEP_HEIGHT → 0
    z = STEP_HEIGHT * (1.0 - jp.cos(theta)) / 2.0
    
    return x, z

def _simple_ik_leg(foot_x: float, foot_z: float) -> tuple:
    """
    簡易2リンク逆運動学 (Hip Pitch + Knee)
    足先座標(x, z)から股関節角度と膝角度を計算する。
    """
    # 足先のスタンス位置からの相対座標
    target_z = -(STAND_HEIGHT) + foot_z  # 負方向が下
    dist = jp.sqrt(foot_x**2 + target_z**2)
    dist = jp.clip(dist, 0.01, THIGH_LEN + KNEE_LEN - 0.001)
    
    # 余弦定理で膝角度を算出
    cos_knee = (THIGH_LEN**2 + KNEE_LEN**2 - dist**2) / (2.0 * THIGH_LEN * KNEE_LEN)
    cos_knee = jp.clip(cos_knee, -1.0, 1.0)
    knee_angle = jp.pi - jp.arccos(cos_knee)  # 0=伸展, >0=屈曲
    
    # 股関節角度
    alpha = jp.arctan2(foot_x, -target_z)
    cos_beta = (THIGH_LEN**2 + dist**2 - KNEE_LEN**2) / (2.0 * THIGH_LEN * dist)
    cos_beta = jp.clip(cos_beta, -1.0, 1.0)
    beta = jp.arccos(cos_beta)
    hip_angle = alpha - beta
    
    return hip_angle, knee_angle


def jax_get_reference_trajectory(phase: float, num_joints: int) -> "jp.ndarray":
    """
    歩行位相 (0.0 ~ 1.0) に基づいて、サイクロイド軌道ベースの
    理想的な関節角リファレンスを生成する。
    (JAX JIT互換: Python制御フローを含まないため vmap/scan 内で利用可能)
    
    残差RL のベースラインとして使用。AIはこの軌道からの「補正値」のみを出力する。
    
    歩行サイクル:
      phase 0.0 ~ 0.5: 右脚=立脚(Stance), 左脚=遊脚(Swing)
      phase 0.5 ~ 1.0: 右脚=遊脚(Swing), 左脚=立脚(Stance)
    """
    ref_angles = jp.zeros(num_joints)
    
    # 各脚の位相 (左右は半周期ずれ)
    p_r = phase
    p_l = (phase + 0.5) % 1.0
    
    # --- 右脚 ---
    # 遊脚期 (p > 0.5): サイクロイド軌道
    swing_phase_r = jp.clip((p_r - 0.5) * 2.0, 0.0, 1.0)
    is_swing_r = p_r > 0.5
    
    cx_r, cz_r = _cycloid_foot_trajectory(swing_phase_r)
    # 立脚期は足先を後ろへ流す (x = -progress * SL/2)
    stance_x_r = -(p_r * 2.0 - 0.5) * STEP_LENGTH / 2.0
    foot_x_r = jp.where(is_swing_r, cx_r, stance_x_r)
    foot_z_r = jp.where(is_swing_r, cz_r, 0.0)
    
    hip_r, knee_r = _simple_ik_leg(foot_x_r, foot_z_r)
    ankle_r = -(hip_r + knee_r)  # 足裏水平維持
    
    # --- 左脚 ---
    swing_phase_l = jp.clip((p_l - 0.5) * 2.0, 0.0, 1.0)
    is_swing_l = p_l > 0.5
    
    cx_l, cz_l = _cycloid_foot_trajectory(swing_phase_l)
    stance_x_l = -(p_l * 2.0 - 0.5) * STEP_LENGTH / 2.0
    foot_x_l = jp.where(is_swing_l, cx_l, stance_x_l)
    foot_z_l = jp.where(is_swing_l, cz_l, 0.0)
    
    hip_l, knee_l = _simple_ik_leg(foot_x_l, foot_z_l)
    ankle_l = -(hip_l + knee_l)
    
    # 旋風丸のインデックス割り当て
    # Right: HipPitch=2, Knee=3, AnklePitch=4
    # Left:  HipPitch=7, Knee=8, AnklePitch=9
    ref_angles = ref_angles.at[2].set(hip_r)
    ref_angles = ref_angles.at[3].set(-knee_r)  # 膝は負方向が屈曲(MuJoCoの慣例)
    ref_angles = ref_angles.at[4].set(ankle_r)
    
    ref_angles = ref_angles.at[7].set(hip_l)
    ref_angles = ref_angles.at[8].set(-knee_l)
    ref_angles = ref_angles.at[9].set(ankle_l)
    
    return ref_angles


def numpy_get_reference_trajectory(phase: float, num_joints: int) -> np.ndarray:
    """
    歩行位相 (0.0 ~ 1.0) に基づいて、サイクロイド軌道ベースの
    理想的な関節角リファレンスを生成する (実機NumPy版)。
    学習環境の jax_get_reference_trajectory と 100% 完全互換。
    """
    ref_angles = np.zeros(num_joints)
    
    # 各脚の位相 (左右は半周期ずれ)
    p_r = phase
    p_l = (phase + 0.5) % 1.0
    
    # --- 右脚 ---
    # 遊脚期 (p > 0.5): サイクロイド軌道
    swing_phase_r = np.clip((p_r - 0.5) * 2.0, 0.0, 1.0)
    is_swing_r = p_r > 0.5
    
    # 右脚遊脚軌道 (サイクロイド)
    theta_r = swing_phase_r * 2.0 * np.pi
    cx_r = STEP_LENGTH / (2.0 * np.pi) * (theta_r - np.sin(theta_r)) - STEP_LENGTH / 2.0
    cz_r = STEP_HEIGHT * (1.0 - np.cos(theta_r)) / 2.0
    
    # 立脚期は足先を後ろへ流す (x = -progress * SL/2)
    stance_x_r = -(p_r * 2.0 - 0.5) * STEP_LENGTH / 2.0
    foot_x_r = cx_r if is_swing_r else stance_x_r
    foot_z_r = cz_r if is_swing_r else 0.0
    
    # 2リンク幾何逆運動学 (IK)
    target_z_r = -STAND_HEIGHT + foot_z_r
    dist_r = np.sqrt(foot_x_r**2 + target_z_r**2)
    dist_r = np.clip(dist_r, 0.01, THIGH_LEN + KNEE_LEN - 0.001)
    
    cos_knee_r = (THIGH_LEN**2 + KNEE_LEN**2 - dist_r**2) / (2.0 * THIGH_LEN * KNEE_LEN)
    cos_knee_r = np.clip(cos_knee_r, -1.0, 1.0)
    knee_r = np.pi - np.arccos(cos_knee_r)
    
    alpha_r = np.arctan2(foot_x_r, -target_z_r)
    cos_beta_r = (THIGH_LEN**2 + dist_r**2 - KNEE_LEN**2) / (2.0 * THIGH_LEN * dist_r)
    cos_beta_r = np.clip(cos_beta_r, -1.0, 1.0)
    beta_r = np.arccos(cos_beta_r)
    hip_r = alpha_r - beta_r
    
    ankle_r = -(hip_r + knee_r)  # 足裏水平維持
    
    # --- 左脚 ---
    # 遊脚期 (p > 0.5): サイクロイド軌道
    swing_phase_l = np.clip((p_l - 0.5) * 2.0, 0.0, 1.0)
    is_swing_l = p_l > 0.5
    
    # 左脚遊脚軌道 (サイクロイド)
    theta_l = swing_phase_l * 2.0 * np.pi
    cx_l = STEP_LENGTH / (2.0 * np.pi) * (theta_l - np.sin(theta_l)) - STEP_LENGTH / 2.0
    cz_l = STEP_HEIGHT * (1.0 - np.cos(theta_l)) / 2.0
    
    # 立脚期は足先を後ろへ流す (x = -progress * SL/2)
    stance_x_l = -(p_l * 2.0 - 0.5) * STEP_LENGTH / 2.0
    foot_x_l = cx_l if is_swing_l else stance_x_l
    foot_z_l = cz_l if is_swing_l else 0.0
    
    # 2リンク幾何逆運動学 (IK)
    target_z_l = -STAND_HEIGHT + foot_z_l
    dist_l = np.sqrt(foot_x_l**2 + target_z_l**2)
    dist_l = np.clip(dist_l, 0.01, THIGH_LEN + KNEE_LEN - 0.001)
    
    cos_knee_l = (THIGH_LEN**2 + KNEE_LEN**2 - dist_l**2) / (2.0 * THIGH_LEN * KNEE_LEN)
    cos_knee_l = np.clip(cos_knee_l, -1.0, 1.0)
    knee_l = np.pi - np.arccos(cos_knee_l)
    
    alpha_l = np.arctan2(foot_x_l, -target_z_l)
    cos_beta_l = (THIGH_LEN**2 + dist_l**2 - KNEE_LEN**2) / (2.0 * THIGH_LEN * dist_l)
    cos_beta_l = np.clip(cos_beta_l, -1.0, 1.0)
    beta_l = np.arccos(cos_beta_l)
    hip_l = alpha_l - beta_l
    
    ankle_l = -(hip_l + knee_l)  # 足裏水平維持
    
    # 旋風丸のインデックスマッピング
    ref_angles[2] = hip_r
    ref_angles[3] = -knee_r  # 膝は負方向が屈曲
    ref_angles[4] = ankle_r
    
    ref_angles[7] = hip_l
    ref_angles[8] = -knee_l
    ref_angles[9] = ankle_l
    
    return ref_angles