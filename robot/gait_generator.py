"""
robot/gait_generator.py — サイクロイド歩行軌道 + 逆運動学

【修正対応 (2026-09-08)】
- [GAIT-1 FIXED] リンク長を config.py から参照（0.12m に統一）
- [GAIT-2 FIXED] STAND_HEIGHT も config.py に一元化
- LegKinematics との完全な互換性を確保

サイクロイド軌道の特性:
- ジャーク最小化（足の着地がスムーズ）
- 同期性が高い（両脚の協調動作が安定）
- 倒立振子モデルと整合しやすい
"""

import numpy as np
import jax.numpy as jp
from typing import Tuple, Optional

from robot.config import RobotConfig
from robot.kinematics import LegKinematics


# ============================================================
# JAX版サイクロイド軌道生成（学習環境用）
# ============================================================

def jax_cycloid_trajectory(
    phase: float,
    foot_height: float,
    step_length: float,
    stand_height: float
) -> Tuple[jp.ndarray, jp.ndarray]:
    """
    サイクロイド軌道：X-Z平面での足先位置と高さ。
    
    【数学背景】
    サイクロイドは、半径rの円が直線上を転がるとき、
    円周上の一点が描く軌跡。ジャーク最小化特性により、
    関節角の加加速度が最小化され、滑らかな足運動を実現。
    
    Args:
        phase: [0, 1) の周期的フェーズ
        foot_height: 遊脚中の最大高さ [m]
        step_length: 1周期での歩幅 [m]
        stand_height: 直立時の腰高さ [m]
    
    Returns:
        (x_traj [m], z_traj [m]): 足先のX-Z位置
    """
    # サイクロイド軌跡の半周期を 0.5 の phase で表現
    phase_mod = (phase % 1.0) * 2.0  # [0, 2)
    
    # 右脚: phase 0.0-1.0 で遊脚、1.0-2.0 で接地
    # 左脚は phase 0.5-1.5 で遊脚、1.5-0.5 で接地（半周期ずれ）
    
    # 遊脚フェーズ判定（0-1: swing, 1-2: stance）
    is_swing = phase_mod < 1.0
    phase_swing = jp.clip(phase_mod, 0.0, 1.0)  # [0, 1]
    phase_stance = jp.clip(phase_mod - 1.0, 0.0, 1.0)  # [0, 1]
    
    # ===== Swing Phase (遊脚) =====
    # サイクロイド曲線: x = r(θ - sin(θ)), z = r(1 - cos(θ))
    # θ: 転がる円の角度 [0, π]
    theta_swing = phase_swing * np.pi
    
    # サイクロイド：
    # - 水平移動: step_length の距離を移動
    # - 垂直移動: 最大 foot_height まで上昇して着地
    x_swing = (step_length / 2.0) * (theta_swing - jp.sin(theta_swing)) / np.pi
    z_swing = (foot_height / np.pi) * (1.0 - jp.cos(theta_swing))
    
    # ===== Stance Phase (接地) =====
    # 接地時は足が地面に固定（X, Z 共に変化なし）
    # または徐々に後方へ移動（参考文献により異なる）
    # ここでは簡略化して、接地時は最終位置を保持
    x_stance = step_length / 2.0  # 遊脚で移動した分
    z_stance = 0.0  # 地面に接触
    
    # スイッチング
    x_traj = jp.where(is_swing, x_swing - step_length / 2.0, -step_length / 2.0)
    z_traj = jp.where(is_swing, z_swing, z_stance)
    
    return x_traj, z_traj


def _simple_ik_leg(
    target_x: float,
    target_z: float,
    thigh_len: float = None,
    knee_len: float = None,
) -> Tuple[float, float, float]:
    """
    2リンク平面逆運動学（股関節ピッチと膝関節）。
    
    [GAIT-1 FIXED] リンク長を config.py から参照（デフォルト値付き）
    
    Args:
        target_x, target_z: 足先の目標位置 [m]
        thigh_len: 大腿長 [m]（デフォルト: config.py 参照）
        knee_len: 下腿長 [m]（デフォルト: config.py 参照）
    
    Returns:
        (hip_pitch, knee_angle, ankle_pitch): 関節角 [rad]
    """
    if thigh_len is None:
        thigh_len = RobotConfig.GAIT_THIGH_LEN
    if knee_len is None:
        knee_len = RobotConfig.GAIT_KNEE_LEN
    
    # 目標位置から股関節から足先までの距離を計算
    L = np.sqrt(target_x**2 + target_z**2)
    
    # 到達可能範囲のチェック
    max_len = thigh_len + knee_len
    if L > max_len:
        L = max_len
    elif L < abs(thigh_len - knee_len):
        L = abs(thigh_len - knee_len)
    
    # 余弦定理で膝角度を計算
    cos_knee = (thigh_len**2 + knee_len**2 - L**2) / (2 * thigh_len * knee_len)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    knee_angle = np.arccos(cos_knee)
    
    # 股関節ピッチ角を計算
    alpha = np.arctan2(target_z, target_x)
    beta = np.arcsin(knee_len * np.sin(knee_angle) / L)
    hip_pitch = alpha + beta
    
    # 足首角度：足底を水平に保つ
    ankle_pitch = -(hip_pitch + knee_angle)
    
    return hip_pitch, knee_angle, ankle_pitch


def jax_get_reference_trajectory(phase: float, num_joints: int = 20) -> jp.ndarray:
    """
    JAX版リファレンス軌道生成（学習環境用）。
    
    [GAIT-1 FIXED] config.py のパラメータを使用して、
    gait_generator.py の定数を消去。
    
    学習環境 mjx_env.py で 100Hz (CONTROL_DT=10ms) で呼び出されることを想定。
    
    Args:
        phase: [0, 1) の周期的フェーズ（mjx_env で計算）
        num_joints: 関節数（デフォルト: 20）
    
    Returns:
        ref_angles: 各関節の理想角度 [rad] shape=(num_joints,)
    """
    # [GAIT-2 FIXED] config.py から参照
    stand_height = RobotConfig.GAIT_STAND_HEIGHT
    step_height = RobotConfig.GAIT_STEP_HEIGHT
    step_length = RobotConfig.GAIT_STEP_LENGTH
    thigh_len = RobotConfig.GAIT_THIGH_LEN
    knee_len = RobotConfig.GAIT_KNEE_LEN
    
    # 左右の脚位相（半周期ずれ）
    phase_r = phase  # 右脚：phase を直接使用
    phase_l = (phase + 0.5) % 1.0  # 左脚：0.5 位相ずれ
    
    # サイクロイド軌道で右脚の足先目標を計算
    x_r, z_r = jax_cycloid_trajectory(phase_r, step_height, step_length, stand_height)
    
    # サイクロイド軌道で左脚の足先目標を計算
    x_l, z_l = jax_cycloid_trajectory(phase_l, step_height, step_length, stand_height)
    
    # 逆運動学（NumPy の _simple_ik_leg を使用するため一度 NumPy に戻す）
    # JAX JIT 互換性のため、JAX版 IK も別途実装すること（後述）
    x_r_np = float(x_r)
    z_r_np = float(z_r)
    x_l_np = float(x_l)
    z_l_np = float(z_l)
    
    hip_pitch_r, knee_r, ankle_pitch_r = _simple_ik_leg(x_r_np, z_r_np, thigh_len, knee_len)
    hip_pitch_l, knee_l, ankle_pitch_l = _simple_ik_leg(x_l_np, z_l_np, thigh_len, knee_len)
    
    # リファレンス軌道ベクトル（20関節のデフォルト）
    ref_angles = jp.zeros(num_joints)
    
    if num_joints >= 12:
        # 右脚インデックス
        ref_angles = ref_angles.at[2].set(jp.array(hip_pitch_r))    # right_hip_pitch
        ref_angles = ref_angles.at[3].set(jp.array(knee_r))         # right_knee
        ref_angles = ref_angles.at[4].set(jp.array(ankle_pitch_r))  # right_ankle_pitch
        
        # 左脚インデックス
        ref_angles = ref_angles.at[8].set(jp.array(hip_pitch_l))    # left_hip_pitch
        ref_angles = ref_angles.at[9].set(jp.array(knee_l))         # left_knee
        ref_angles = ref_angles.at[10].set(jp.array(ankle_pitch_l)) # left_ankle_pitch
    
    return ref_angles


# ============================================================
# NumPy版サイクロイド軌道生成（テスト・実機用）
# ============================================================

class GaitGenerator:
    """NumPy ベースのサイクロイド歩行軌道生成クラス。"""
    
    def __init__(self):
        self.ik = LegKinematics()
        # [GAIT-2 FIXED] config.py から参照
        self.stand_height = RobotConfig.GAIT_STAND_HEIGHT
        self.step_height = RobotConfig.GAIT_STEP_HEIGHT
        self.step_length = RobotConfig.GAIT_STEP_LENGTH
        self.thigh_len = RobotConfig.GAIT_THIGH_LEN
        self.knee_len = RobotConfig.GAIT_KNEE_LEN
    
    def get_foot_position(self, phase: float, right_leg: bool = True) -> Tuple[float, float]:
        """
        サイクロイド軌道から足先位置を計算。
        
        Args:
            phase: [0, 1) のフェーズ
            right_leg: True なら右脚、False なら左脚
        
        Returns:
            (x, z): 足先のX-Z位置 [m]
        """
        if not right_leg:
            phase = (phase + 0.5) % 1.0  # 左脚は半周期ずれ
        
        # サイクロイド軌跡
        phase_mod = phase * 2.0  # [0, 2)
        is_swing = phase_mod < 1.0
        phase_swing = np.clip(phase_mod, 0.0, 1.0)
        
        # サイクロイド
        theta = phase_swing * np.pi
        x_swing = (self.step_length / 2.0) * (theta - np.sin(theta)) / np.pi
        z_swing = (self.step_height / np.pi) * (1.0 - np.cos(theta))
        
        x = x_swing - self.step_length / 2.0 if is_swing else -self.step_length / 2.0
        z = z_swing if is_swing else 0.0
        
        return x, z
    
    def get_joint_angles(self, phase: float) -> np.ndarray:
        """
        指定された位相での各関節の目標角度を取得。
        
        Args:
            phase: [0, 1) のフェーズ
        
        Returns:
            angles: shape=(20,) の関節角度 [rad]
        """
        angles = np.zeros(20)
        
        # 右脚の足先位置
        x_r, z_r = self.get_foot_position(phase, right_leg=True)
        hip_r, knee_r, ankle_r = _simple_ik_leg(x_r, z_r, self.thigh_len, self.knee_len)
        
        # 左脚の足先位置
        x_l, z_l = self.get_foot_position(phase, right_leg=False)
        hip_l, knee_l, ankle_l = _simple_ik_leg(x_l, z_l, self.thigh_len, self.knee_len)
        
        # 右脚への割り当て
        angles[2] = hip_r    # right_hip_pitch
        angles[3] = knee_r   # right_knee
        angles[4] = ankle_r  # right_ankle_pitch
        
        # 左脚への割り当て
        angles[8] = hip_l    # left_hip_pitch
        angles[9] = knee_l   # left_knee
        angles[10] = ankle_l # left_ankle_pitch
        
        return angles


def numpy_get_reference_trajectory(phase: float, num_joints: int = 20) -> np.ndarray:
    """
    NumPy版リファレンス軌道生成（テスト・実機用）。
    
    [GAIT-1 FIXED] config.py のパラメータを使用。
    [GAIT-2 FIXED] GaitGenerator クラスと統一された実装。
    
    Args:
        phase: [0, 1) の周期的フェーズ
        num_joints: 関節数（デフォルト: 20）
    
    Returns:
        ref_angles: 各関節の理想角度 [rad] shape=(num_joints,)
    """
    gen = GaitGenerator()
    ref_angles = gen.get_joint_angles(phase)
    
    # 必要に応じてリサイズ
    if num_joints < 20:
        ref_angles = ref_angles[:num_joints]
    
    return ref_angles


# ============================================================
# テスト & デバッグ
# ============================================================

if __name__ == "__main__":
    print("[Test] Gait Generator Validation")
    print(f"Config Thigh Length: {RobotConfig.GAIT_THIGH_LEN} m")
    print(f"Config Knee Length: {RobotConfig.GAIT_KNEE_LEN} m")
    print(f"Config Stand Height: {RobotConfig.GAIT_STAND_HEIGHT} m")
    print()
    
    # NumPy版テスト
    print("[NumPy Test] Full cycle (0.0 to 1.0 phase)")
    gen = GaitGenerator()
    for phase_val in np.linspace(0.0, 1.0, 5, endpoint=False):
        angles = numpy_get_reference_trajectory(phase_val, num_joints=20)
        x_r, z_r = gen.get_foot_position(phase_val, right_leg=True)
        x_l, z_l = gen.get_foot_position(phase_val, right_leg=False)
        print(f"Phase {phase_val:.2f}: "
              f"Right foot ({x_r:+.3f}, {z_r:+.3f}m), "
              f"Left foot ({x_l:+.3f}, {z_l:+.3f}m), "
              f"Hip_R={angles[2]*57.3:+.1f}°")
    
    print("\n[JAX Test] Full cycle (NumPy を経由)")
    for phase_val in np.linspace(0.0, 1.0, 5, endpoint=False):
        ref_jax = jax_get_reference_trajectory(phase_val, num_joints=20)
        print(f"Phase {phase_val:.2f}: Hip_R={float(ref_jax[2])*57.3:+.1f}°")