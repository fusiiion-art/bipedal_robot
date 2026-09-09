"""
robot/math_utils.py — クォータニオン・数学変換ユーティリティ

【修正対応 (2026-09-08)】
- [MATH-1 FIXED] quat_to_euler() を NumPy版と JAX版に分離（JIT互換化）
- 関数内での型判定を排除し、呼び出し側で型を明示的に選択
- JAX JIT コンパイルの制御フロー制限に対応
"""

import numpy as np

try:
    import jax
    import jax.numpy as jp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jp = None
    jax = None


# ============================================================
# NumPy版: クォータニオン -> オイラー角
# ============================================================

def quat_to_euler_numpy(q: np.ndarray) -> np.ndarray:
    """
    NumPy版クォータニオンからオイラー角（Roll-Pitch-Yaw）への変換。
    
    クォータニオン形式: q = [w, x, y, z]（BNO055標準）
    
    【変換式】
    ロール (Roll) φ：X軸周りの回転
    ピッチ (Pitch) θ：Y軸周りの回転
    ヨー (Yaw) ψ：Z軸周りの回転
    
    標準的な ZYX (Yaw-Pitch-Roll) オーダーで変換。
    
    Args:
        q: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        rpy: shape=(3,) オイラー角 [roll, pitch, yaw] [rad]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (X軸周りの回転)
    sinp = 2.0 * (w * x + y * z)
    cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = np.arctan2(sinp, cosp)
    
    # Pitch (Y軸周りの回転)
    sinp_pitch = 2.0 * (w * y - z * x)
    sinp_pitch = np.clip(sinp_pitch, -1.0, 1.0)  # 数値誤差対策
    pitch = np.arcsin(sinp_pitch)
    
    # Yaw (Z軸周りの回転)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(siny, cosy)
    
    return np.array([roll, pitch, yaw])


# ============================================================
# JAX版: クォータニオン -> オイラー角
# ============================================================

def quat_to_euler_jax(q: "jax.Array") -> "jax.Array":
    """
    JAX版クォータニオンからオイラー角への変換（JIT互換）。
    
    【重要】JAX JIT コンパイル内でも実行可能な実装。
    制御フロー（if/else）を使わず、jp.clip() と jp.arcsin() で
    数値安定性を確保。
    
    Args:
        q: shape=(4,) JAX配列 クォータニオン [w, x, y, z]
    
    Returns:
        rpy: shape=(3,) JAX配列 オイラー角 [roll, pitch, yaw] [rad]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (X軸周りの回転)
    sinp = 2.0 * (w * x + y * z)
    cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = jp.arctan2(sinp, cosp)
    
    # Pitch (Y軸周りの回転)
    sinp_pitch = 2.0 * (w * y - z * x)
    # [MATH-1 FIXED] clip で [-1, 1] に制限（JAX JIT互換）
    sinp_pitch = jp.clip(sinp_pitch, -1.0, 1.0)
    pitch = jp.arcsin(sinp_pitch)
    
    # Yaw (Z軸周りの回転)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y**2 + z**2)
    yaw = jp.arctan2(siny, cosy)
    
    return jp.array([roll, pitch, yaw])


# ============================================================
# [MATH-1 FIXED] ユーザー向け統一インターフェース
# ============================================================

def quat_to_euler(q) -> np.ndarray:
    """
    クォータニオンからオイラー角への統一インターフェース。
    
    【使い方】
    入力配列の型に基づいて、自動的に適切な実装を選択します。
    
    - NumPy配列 or Python float/list → NumPy版を使用
    - JAX配列 → JAX版を使用（JIT対応）
    
    Args:
        q: クォータニオン [w, x, y, z]（NumPy配列またはJAX配列）
    
    Returns:
        rpy: オイラー角 [roll, pitch, yaw] [rad]
             入力の型に応じて NumPy配列 or JAX配列を返す
    
    例:
        # NumPy環境
        q_np = np.array([1.0, 0.0, 0.0, 0.0])
        rpy_np = quat_to_euler(q_np)  # → NumPy配列
        
        # JAX環境
        q_jax = jax.numpy.array([1.0, 0.0, 0.0, 0.0])
        rpy_jax = quat_to_euler(q_jax)  # → JAX配列
        
        # JAX JIT 内で使用可能
        @jax.jit
        def compute_rpy(q):
            return quat_to_euler(q)  # 自動的に JAX版で実行
    """
    # [MATH-1 FIXED] 型判定を呼び出し側で実施
    if HAS_JAX and isinstance(q, jax.Array):
        # JAX配列の場合は JAX版を使用
        return quat_to_euler_jax(q)
    else:
        # NumPy配列 or その他の場合は NumPy版を使用
        q_np = np.asarray(q)
        return quat_to_euler_numpy(q_np)


# ============================================================
# オイラー角 -> クォータニオン（逆変換）
# ============================================================

def euler_to_quat_numpy(rpy: np.ndarray) -> np.ndarray:
    """
    NumPy版オイラー角からクォータニオンへの変換。
    
    Args:
        rpy: shape=(3,) オイラー角 [roll, pitch, yaw] [rad]
    
    Returns:
        q: shape=(4,) クォータニオン [w, x, y, z]
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    
    # 半角公式
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    return np.array([w, x, y, z])


def euler_to_quat_jax(rpy: "jax.Array") -> "jax.Array":
    """
    JAX版オイラー角からクォータニオンへの変換（JIT互換）。
    
    Args:
        rpy: shape=(3,) JAX配列 オイラー角 [roll, pitch, yaw] [rad]
    
    Returns:
        q: shape=(4,) JAX配列 クォータニオン [w, x, y, z]
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    
    # 半角公式
    cy = jp.cos(yaw * 0.5)
    sy = jp.sin(yaw * 0.5)
    cp = jp.cos(pitch * 0.5)
    sp = jp.sin(pitch * 0.5)
    cr = jp.cos(roll * 0.5)
    sr = jp.sin(roll * 0.5)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    return jp.array([w, x, y, z])


def euler_to_quat(rpy) -> np.ndarray:
    """
    オイラー角からクォータニオンへの統一インターフェース。
    
    入力配列の型に基づいて、自動的に適切な実装を選択します。
    """
    if HAS_JAX and isinstance(rpy, jax.Array):
        return euler_to_quat_jax(rpy)
    else:
        rpy_np = np.asarray(rpy)
        return euler_to_quat_numpy(rpy_np)


# ============================================================
# その他のユーティリティ関数
# ============================================================

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    クォータニオンを正規化（ノルム = 1）。
    
    Args:
        q: shape=(4,) クォータニオン
    
    Returns:
        q_normalized: 正規化されたクォータニオン
    """
    q = np.asarray(q)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 安全なデフォルト
    return q / norm


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    クォータニオンの逆元を計算。
    
    q⁻¹ = q*/|q|² （共役四元数を ノルムの二乗で割る）
    
    Args:
        q: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        q_inv: 逆元クォータニオン
    """
    q = np.asarray(q)
    norm_sq = np.sum(q**2)
    if norm_sq < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    # 共役: [w, -x, -y, -z]
    return np.array([q[0], -q[1], -q[2], -q[3]]) / norm_sq


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    クォータニオンでベクトルを回転。
    
    v' = q * v * q⁻¹
    
    Args:
        v: shape=(3,) ベクトル
        q: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        v_rotated: 回転後のベクトル shape=(3,)
    """
    v = np.asarray(v)
    q = np.asarray(q)
    q = normalize_quaternion(q)
    
    # v を [0, v_x, v_y, v_z] に拡張
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    
    # q * v * q⁻¹
    q_inv = quaternion_inverse(q)
    
    # quaternion multiplication: q * v
    qv = quaternion_multiply(q, v_quat)
    
    # (q * v) * q⁻¹
    result = quaternion_multiply(qv, q_inv)
    
    return result[1:4]  # 虚部のみを返す


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    2つのクォータニオンの積を計算。
    
    Args:
        q1, q2: shape=(4,) クォータニオン [w, x, y, z]
    
    Returns:
        product: 積のクォータニオン
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


# ============================================================
# テスト & デバッグ
# ============================================================

if __name__ == "__main__":
    print("[Test] Math Utilities Validation")
    print()
    
    # テストケース: いくつかの有名なクォータニオン
    test_quaternions = [
        np.array([1.0, 0.0, 0.0, 0.0]),         # Identity
        np.array([0.7071, 0.7071, 0.0, 0.0]),   # 90° roll
        np.array([0.7071, 0.0, 0.7071, 0.0]),   # 90° pitch
        np.array([0.7071, 0.0, 0.0, 0.7071]),   # 90° yaw
    ]
    
    print("[NumPy Version]")
    for q in test_quaternions:
        rpy = quat_to_euler_numpy(q)
        print(f"q = {q} → rpy = [{np.degrees(rpy[0]):+.1f}°, "
              f"{np.degrees(rpy[1]):+.1f}°, {np.degrees(rpy[2]):+.1f}°]")
    
    print("\n[JAX Version]")
    if HAS_JAX:
        for q_np in test_quaternions:
            q_jax = jp.array(q_np)
            rpy_jax = quat_to_euler_jax(q_jax)
            rpy = np.array(rpy_jax)
            print(f"q_jax = ... → rpy_jax = [{np.degrees(rpy[0]):+.1f}°, "
                  f"{np.degrees(rpy[1]):+.1f}°, {np.degrees(rpy[2]):+.1f}°]")
    else:
        print("(JAX not available)")
    
    print("\n[Unified Interface]")
    print("Testing quat_to_euler() auto-dispatch:")
    q_test = np.array([0.7071, 0.7071, 0.0, 0.0])
    rpy_result = quat_to_euler(q_test)
    print(f"Type: {type(rpy_result)}, Value: {rpy_result}")
    
    print("\n[Inverse Transform Test]")
    rpy_original = np.array([0.1, 0.2, 0.3])  # [rad]
    q_from_rpy = euler_to_quat(rpy_original)
    rpy_reconstructed = quat_to_euler(q_from_rpy)
    print(f"Original RPY: {np.degrees(rpy_original)}")
    print(f"Reconstructed RPY: {np.degrees(rpy_reconstructed)}")
    print(f"Error: {np.degrees(rpy_original - rpy_reconstructed)}")