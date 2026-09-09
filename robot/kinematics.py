"""
robot/kinematics.py — 解析的逆運動学 (Analytical IK) ソルバー

【修正対応 (2026-09-08)】
- [KIN-1 FIXED] リンク長の単位を [mm] から [m] に統一
- [KIN-1 FIXED] max_len チェックの次元を [m] で統一（mm での計算から改修）
- [GAIT-1 FIXED] config.py との連携で 0.12m に統一

数学基盤:
  2リンク平面IK（関節 yaw は独立なため省略）。
  股関節ピッチ + 膝関節 + 足首ピッチの 3軸に相当する
  水平（X）・鉛直（Z）の 2DoF 問題として定式化。
  足平を水平に保つ制約（ankle_pitch = -(hip_pitch + knee)）を導入。
"""

import numpy as np
from typing import Tuple

from robot.config import RobotConfig


class LegKinematics:
    """
    両脚の解析的逆運動学ソルバー。
    
    【旋風丸ロボット対応】
    股関節 (Pitch) -> 膝 -> 足首 (Pitch) の 3リンク配置。
    Yaw/Roll は別途制御（本ソルバーでは X-Z 平面のみ）。
    
    【物理仕様】
    - L_THIGH: 大腿リンク（股関節～膝）= 0.12 m [FIXED]
    - L_SHIN: 下腿リンク（膝～足首）= 0.12 m [FIXED]
    - 足首位置（Z=0）から股関節位置（Z=h）まで逆運動学を計算。
    
    使用方法:
      ik = LegKinematics()
      hip, knee, ankle = ik.solve_leg(x_target, z_target)
    """
    
    # [KIN-1 FIXED] リンク長を [m] で定義（[mm] ではなく）
    # config.py の GAIT_THIGH_LEN, GAIT_KNEE_LEN と同期
    L_THIGH = RobotConfig.GAIT_THIGH_LEN    # [m] 0.12
    L_SHIN  = RobotConfig.GAIT_KNEE_LEN     # [m] 0.12
    
    def __init__(self):
        """
        初期化: リンク長を config.py から読み込む。
        """
        self.L_THIGH = RobotConfig.GAIT_THIGH_LEN
        self.L_SHIN = RobotConfig.GAIT_KNEE_LEN
    
    def solve_leg(
        self,
        x_target: float,
        z_target: float,
        clamp: bool = True
    ) -> Tuple[float, float, float]:
        """
        2リンク平面逆運動学を解く。
        
        座標系定義:
          - 股関節を原点 (0, 0)
          - X軸: 前方向（ロボット進行方向）
          - Z軸: 下方向（足裏方向）
        
        【数学】
          足先目標位置 (x_target, z_target) に対して、
          股関節ピッチ角 θ₁、膝角度 θ₂、足首ピッチ角 θ₃ を求める。
          
          余弦定理で膝角度を計算し、幾何計算で股関節角度を導出。
          足平を水平に保つ制約: θ₃ = -(θ₁ + θ₂)
        
        Args:
            x_target: 足先目標のX座標 [m]
            z_target: 足先目標のZ座標 [m]（負が下方）
            clamp: True なら到達範囲外の目標位置を制限
        
        Returns:
            (hip_pitch, knee_angle, ankle_pitch): 関節角 [rad]
            
        例外:
            到達不可能な位置が指定された場合、
            clamp=True なら最大リーチ位置へ移動。
            clamp=False なら数値不安定性により不正な値が返される可能性。
        """
        
        # === Step 1: 股関節から足先までの距離を計算 ===
        # L = sqrt(x² + z²)
        L = np.sqrt(x_target**2 + z_target**2)
        
        # === Step 2: 到達範囲の判定と制限 ===
        # [KIN-1 FIXED] max_len を [m] 単位で正確に計算
        max_len = self.L_THIGH + self.L_SHIN  # [m] 0.24
        min_len = np.abs(self.L_THIGH - self.L_SHIN)  # [m] 0.0
        
        if clamp:
            L = np.clip(L, min_len, max_len)
        else:
            # clamp=False の場合も、数値安定性のため小さなマージンを追加
            if L > max_len:
                L = max_len * 0.9999
            elif L < min_len + 1e-6:
                L = min_len + 1e-6
        
        # === Step 3: 余弦定理で膝関節角を計算 ===
        # cos(θ₂) = (L₁² + L₂² - L²) / (2 * L₁ * L₂)
        cos_knee = (self.L_THIGH**2 + self.L_SHIN**2 - L**2) / (2.0 * self.L_THIGH * self.L_SHIN)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)  # 数値誤差対策
        
        knee_angle = np.arccos(cos_knee)  # [rad] [0, π]
        
        # === Step 4: 股関節ピッチ角を計算 ===
        # α = atan2(z_target, x_target) : 足先までのベアリング角
        alpha = np.arctan2(z_target, x_target)
        
        # β : 大腿と目標方向のなす角
        # sin(β) = (L_shin * sin(π - θ₂)) / L = (L_shin * sin(θ₂)) / L
        sin_beta = self.L_SHIN * np.sin(knee_angle) / (L + 1e-8)
        sin_beta = np.clip(sin_beta, -1.0, 1.0)
        beta = np.arcsin(sin_beta)
        
        # 股関節ピッチ = α + β
        hip_pitch = alpha + beta
        
        # === Step 5: 足首ピッチ角を計算（足平水平制約） ===
        # 足平を水平に保つには: ankle_pitch = -(hip_pitch + knee_angle)
        ankle_pitch = -(hip_pitch + knee_angle)
        
        return hip_pitch, knee_angle, ankle_pitch
    
    def solve_leg_batch(
        self,
        x_targets: np.ndarray,
        z_targets: np.ndarray,
        clamp: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        バッチ処理版逆運動学。複数の目標位置を一括計算。
        
        Args:
            x_targets: shape=(N,) の X座標配列 [m]
            z_targets: shape=(N,) の Z座標配列 [m]
            clamp: True なら到達範囲外を制限
        
        Returns:
            (hip_pitches, knee_angles, ankle_pitches): 各々 shape=(N,)
        """
        hip_pitches = np.zeros_like(x_targets)
        knee_angles = np.zeros_like(x_targets)
        ankle_pitches = np.zeros_like(x_targets)
        
        for i in range(len(x_targets)):
            h, k, a = self.solve_leg(x_targets[i], z_targets[i], clamp=clamp)
            hip_pitches[i] = h
            knee_angles[i] = k
            ankle_pitches[i] = a
        
        return hip_pitches, knee_angles, ankle_pitches
    
    def forward_kinematics(
        self,
        hip_pitch: float,
        knee_angle: float
    ) -> Tuple[float, float]:
        """
        順運動学: 関節角から足先位置を計算（検証用）。
        
        Args:
            hip_pitch: 股関節ピッチ角 [rad]
            knee_angle: 膝関節角 [rad]
        
        Returns:
            (x, z): 足先位置 [m]
        """
        # 大腿の先端（膝の位置）
        knee_x = self.L_THIGH * np.sin(hip_pitch)
        knee_z = self.L_THIGH * np.cos(hip_pitch)
        
        # 下腿の終端（足先の位置）
        # 膝関節での旋回: 膝角度だけ大腿からの角度が変わる
        leg_pitch = hip_pitch + knee_angle
        
        x = knee_x + self.L_SHIN * np.sin(leg_pitch)
        z = knee_z + self.L_SHIN * np.cos(leg_pitch)
        
        return x, z
    
    @staticmethod
    def get_ankle_pitch_for_horizontal_foot(hip_pitch: float, knee_angle: float) -> float:
        """
        足平を水平に保つための足首ピッチ角を計算。
        
        Args:
            hip_pitch: 股関節ピッチ角 [rad]
            knee_angle: 膝関節角 [rad]
        
        Returns:
            ankle_pitch: 足首ピッチ角 [rad]
        """
        return -(hip_pitch + knee_angle)


# ============================================================
# ユーティリティ関数
# ============================================================

def ik_reach_check(
    x_target: float,
    z_target: float,
    thigh_len: float = None,
    shin_len: float = None
) -> bool:
    """
    指定された位置が逆運動学で到達可能かを判定。
    
    Args:
        x_target: 目標X座標 [m]
        z_target: 目標Z座標 [m]
        thigh_len: 大腿長 [m]（デフォルト: config.py）
        shin_len: 下腿長 [m]（デフォルト: config.py）
    
    Returns:
        True なら到達可能、False なら不可能
    """
    if thigh_len is None:
        thigh_len = RobotConfig.GAIT_THIGH_LEN
    if shin_len is None:
        shin_len = RobotConfig.GAIT_KNEE_LEN
    
    L = np.sqrt(x_target**2 + z_target**2)
    max_len = thigh_len + shin_len
    min_len = np.abs(thigh_len - shin_len)
    
    return min_len <= L <= max_len


def ik_safety_clamp(
    x_target: float,
    z_target: float,
    thigh_len: float = None,
    shin_len: float = None
) -> Tuple[float, float]:
    """
    目標位置を到達範囲内に制限。
    
    Args:
        x_target: 目標X座標 [m]
        z_target: 目標Z座標 [m]
        thigh_len: 大腿長 [m]（デフォルト: config.py）
        shin_len: 下腿長 [m]（デフォルト: config.py）
    
    Returns:
        (x_clamped, z_clamped): 制限後の位置 [m]
    """
    if thigh_len is None:
        thigh_len = RobotConfig.GAIT_THIGH_LEN
    if shin_len is None:
        shin_len = RobotConfig.GAIT_KNEE_LEN
    
    L = np.sqrt(x_target**2 + z_target**2)
    max_len = thigh_len + shin_len
    min_len = np.abs(thigh_len - shin_len)
    
    if L > max_len:
        scale = max_len / (L + 1e-8)
    elif L < min_len:
        scale = min_len / (L + 1e-8)
    else:
        scale = 1.0
    
    return x_target * scale, z_target * scale


# ============================================================
# テスト & デバッグ
# ============================================================

if __name__ == "__main__":
    print("[Test] Leg Kinematics Validation")
    print(f"Thigh Length: {RobotConfig.GAIT_THIGH_LEN} m")
    print(f"Shin Length: {RobotConfig.GAIT_KNEE_LEN} m")
    print(f"Max Reach: {RobotConfig.GAIT_THIGH_LEN + RobotConfig.GAIT_KNEE_LEN} m")
    print()
    
    ik = LegKinematics()
    
    # テストケース
    test_cases = [
        (0.0, -0.24),    # 最大リーチ（真下）
        (0.12, -0.12),   # 中程度
        (0.0, -0.15),    # 直立相当
    ]
    
    print("[Inverse Kinematics Test]")
    for x_t, z_t in test_cases:
        if ik_reach_check(x_t, z_t):
            h, k, a = ik.solve_leg(x_t, z_t)
            print(f"Target ({x_t:+.2f}, {z_t:+.2f})m: "
                  f"Hip={np.degrees(h):+.1f}°, "
                  f"Knee={np.degrees(k):+.1f}°, "
                  f"Ankle={np.degrees(a):+.1f}°")
            
            # 順運動学で検証
            x_calc, z_calc = ik.forward_kinematics(h, k)
            print(f"  → FK check: ({x_calc:+.3f}, {z_calc:+.3f})m")
        else:
            print(f"Target ({x_t:+.2f}, {z_t:+.2f})m: Out of reach")
    
    print("\n[Safety Clamping Test]")
    x_bad, z_bad = 0.3, -0.2  # 到達不可
    x_safe, z_safe = ik_safety_clamp(x_bad, z_bad)
    print(f"Bad target ({x_bad:+.2f}, {z_bad:+.2f})m")
    print(f"Clamped to ({x_safe:+.3f}, {z_safe:+.3f})m")