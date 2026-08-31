import numpy as np

class LegKinematics:
    """
    解析的逆運動学 (Analytical IK) ソルバー
    幾何学計算のみで、足先の座標から関節角度を逆算する。
    """
    def __init__(self):
        # 設計値 [mm] (Fusion 360の寸法に合わせて修正してください)
        self.L_THIGH = 100.0  # 股関節～膝の長さ
        self.L_SHIN  = 100.0  # 膝～足首の長さ
        self.L_HIP_OFFSET = 30.0 # 胴体中心から股関節までの横幅

    def solve_leg(self, x, y, z, is_right=True):
        """
        片足のIKを解く
        Args:
            x, y, z: 足先座標 (股関節中心を原点(0,0,0)とした相対座標)
            is_right: 右足ならTrue
        Returns:
            angles: [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll] [rad]
        """
        # ※ ここでは簡易的に 3自由度 (HipPitch, Knee, AnklePitch) のみを計算する例
        #    実際はRoll/Yawも含めた回転行列計算が必要だが、まずは「屈伸」ができることが重要
        
        # 1. 膝の角度 (余弦定理)
        # L^2 = x^2 + z^2 (yは無視)
        L_sq = x**2 + z**2
        L = np.sqrt(L_sq)
        
        # 届かない場合はクリップ
        max_len = self.L_THIGH + self.L_SHIN
        if L > max_len:
            L = max_len
            z = -np.sqrt(L**2 - x**2) # zを調整

        # cos(膝角度)
        cos_knee = (L_sq - self.L_THIGH**2 - self.L_SHIN**2) / (2 * self.L_THIGH * self.L_SHIN)
        knee_angle = np.arccos(np.clip(cos_knee, -1.0, 1.0))
        
        # 膝は後ろには曲がらないので符号調整 (ロボットによる)
        # 旋風丸(鳥足でない)なら正の値
        
        # 2. 股関節ピッチ
        # alpha: 足先方向の角度, beta: 三角形の内角
        alpha = np.arctan2(-x, -z) # 鉛直下向きを0とする
        cos_beta = (L_sq + self.L_THIGH**2 - self.L_SHIN**2) / (2 * L * self.L_THIGH)
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
        
        hip_pitch = alpha - beta
        
        # 3. 足首ピッチ
        # 足裏を常に地面と平行にする場合: Hip + Knee + Ankle = 0
        ankle_pitch = -(hip_pitch + knee_angle)
        
        # 簡易計算なのでYaw/Rollは0
        return np.array([0, 0, hip_pitch, knee_angle, ankle_pitch, 0])

# テスト実行
if __name__ == "__main__":
    ik = LegKinematics()
    # 股関節の真下 150mm の位置に足を置く計算
    angles = ik.solve_leg(0, 0, -150)
    print(f"Target: (0, 0, -150)")
    print(f"Angles: {np.degrees(angles).round(2)}")