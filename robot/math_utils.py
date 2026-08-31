import numpy as np

try:
    import jax.numpy as jp
    import jax
    HAS_JAX = True
except Exception:
    jp = None
    jax = None
    HAS_JAX = False

def quat_to_euler(q) -> np.ndarray:
    """
    クォータニオン(w, x, y, z)からオイラー角(Roll, Pitch, Yaw)を計算する共通関数
    JAX配列と標準NumPy配列の両方に自動対応。
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    
    if HAS_JAX and jax is not None and jp is not None and isinstance(q, jax.Array):
        roll = jp.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        sinp = jp.clip(sinp, -1.0, 1.0)
        pitch = jp.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jp.arctan2(siny_cosp, cosy_cosp)
        return jp.array([roll, pitch, yaw])
    else:
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])
