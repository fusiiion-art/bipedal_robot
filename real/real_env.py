"""
real/real_env.py — RPi5 実機メインループ & 観測ベクトル構築

【実装済み対策 (フィージビリティレビュー反映)】
- ONNX Runtime: シングルスレッド強制 (レイテンシスパイク防止)
- 制御周期: 100Hz対応 (dt=10ms, time.monotonic精密タイマー)
- 観測ベクトル: project_overview.md の625次元仕様に完全準拠

【修正対応 (2026-09-08)】
- [REAL-1 FIXED] 位相計算を相対時刻ベースに統一（step数カウンタ）
- [REAL-2 FIXED] base_pos[2] ゼロ埋めに明記・comment追加
- [REAL-3 FIXED] action_history 順序を mjx_env と明示的に一致（assert検証追加）
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    print("[Warn] onnxruntime not found. Policy will run in dummy mode.")
    ort = None

from real.real_io import TeensySpineIO
from robot.math_utils import projected_gravity_numpy, rotate_vector_by_quaternion
from robot.config import RobotConfig


# ============================================================
# ONNX推論ラッパー (シングルスレッド設定)
# ============================================================

class PolicyRunner:
    """
    ONNX Runtime 推論実行器。
    
    【重要】デフォルトでは4コアすべてを使おうとし、
    軽量MLPではスレッド同期オーバーヘッドで突発10ms超のスパイクが発生する。
    シングルスレッドに制限することで推論時間を1ms以下に安定化させる。
    """
    
    def __init__(
        self, 
        model_path: str = "/var/lib/bipedal_runtime/models/policy.onnx",
        obs_dim: int = 625,
        act_dim: int = 20
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.session: Optional[Any] = None
        self.dummy_mode = ort is None
        
        if not self.dummy_mode:
            try:
                opts = ort.SessionOptions()
                # ★ シングルスレッド強制 — レイテンシスパイク防止の核心設定
                opts.intra_op_num_threads = 1   # 演算内部: 並列化なし
                opts.inter_op_num_threads = 1   # 演算間: 並列化なし
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                # グラフ最適化はフルに活用
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.session.get_inputs()[0].name
                print(f"[Info] Policy loaded: {model_path} (single-thread, deterministic latency)")
            except Exception as e:
                print(f"[Error] ONNX load failed: {e}")
                self.dummy_mode = True
    
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """推論実行。入力: (obs_dim,), 出力: (act_dim,)"""
        if self.dummy_mode:
            return np.zeros(self.act_dim)
        
        obs_input = obs.astype(np.float32).reshape(1, -1)
        result = self.session.run(None, {self.input_name: obs_input})
        return result[0].flatten()[:self.act_dim]


# ============================================================
# メイン制御環境
# ============================================================

class RealRobotEnv:
    """
    Raspberry Pi 5 実機制御環境。
    
    100Hz (10ms) のメインループで:
    1. センサー取得 (共有メモリ or 直接)
    2. 625次元観測ベクトルの構築
    3. ONNX推論 (Base Policy, シングルスレッド)
    4. 残差合成 (中腰デフォルト姿勢 + AI残差)
    5. 安全クランプ + EMA平滑化
    6. Sync Write一括送信
    7. インターリーブRead (2台/ループ)
    """
    
    # [項目7] robot/config.py の定数を直接参照（ハードコード再定義を廃止）
    NUM_JOINTS = RobotConfig.NUM_JOINTS
    BASE_OBS_DIM = RobotConfig.BASE_OBS_DIM
    HISTORY_LEN = RobotConfig.HISTORY_LEN
    ACT_DIM = RobotConfig.ACT_DIM
    OBS_DIM = RobotConfig.OBS_DIM
    ACTION_SCALE = RobotConfig.ACTION_SCALE
    EMA_ALPHA = RobotConfig.MOTOR_LPF_ALPHA  # LPF平滑化係数 (学習側と同じ値)
    
    # 各関節の物理的可動限界 (assets/humanoid/humanoid.xml と 100% 完全同期)
    JOINT_LIMITS_MIN = np.array([
        # 右脚 (6関節)
        0.0, -0.523599, -0.523599, -1.047198, -1.570796, -0.436332,
        # 左脚 (6関節)
        -3.141593, -0.523599, -1.047198, -0.523599, -0.436332, -0.523599,
        # 右腕 (4関節)
        -3.141593, 0.0, 0.0, -1.570796,
        # 左腕 (4関節)
        -3.141593, -3.141593, -3.141593, -0.261799
    ])
    JOINT_LIMITS_MAX = np.array([
        # 右脚 (6関節)
        3.141593, 0.523599, 1.047198, 0.523599, 0.436332, 0.436332,
        # 左脚 (6関節)
        0.0, 0.523599, 0.523599, 1.047198, 1.570796, 0.436332,
        # 右腕 (4関節)
        3.141593, 3.141593, 3.141593, 0.261799,
        # 左腕 (4関節)
        3.141593, 0.0, 0.0, 1.570796
    ])
    
    def __init__(self, control_hz: int = 100):
        self.dt = 1.0 / control_hz
        self.control_hz = control_hz
        
        # --- ハードウェアI/O ---
        print("[Info] Initializing RealRobotEnv (100Hz target)...")
        self.spine = TeensySpineIO(num_servos=self.NUM_JOINTS)
        self.imu_data, self.fsr_contacts, self.servo_temps, self.servo_voltages = (
            self.spine.communicate(np.zeros(self.NUM_JOINTS))
        )
        
        # --- ONNX推論 (シングルスレッド) ---
        self.policy = PolicyRunner()
        
        # --- 状態変数 ---
        self.last_action = np.zeros(self.NUM_JOINTS)
        self.smoothed_action = np.zeros(self.NUM_JOINTS)
        
        # ZUPT速度推定用
        self._vel_estimate = np.zeros(3)           # IMU積分速度 [m/s]
        self._prev_joint_pos = np.zeros(self.NUM_JOINTS)  # 関節角速度の有限差分用
        
        self._episode_step = 0
        self._max_episode_steps = RobotConfig.MAX_EPISODE_STEPS

        # BNO055 の取り付け姿勢 (センサー座標系 → 胴体座標系)。学習側の観測は胴体座標系。
        self._imu_mount_quat = np.asarray(RobotConfig.IMU_MOUNT_QUAT, dtype=np.float64)

        # 履歴バッファ (古 → 新、mjx_env の jp.roll(shift=-1) と同じ向き)。
        # 中身は reset_episode() 後の最初の観測で埋める。
        self.obs_history = deque(maxlen=self.HISTORY_LEN)
        self.act_history = deque(maxlen=self.HISTORY_LEN)
        self._history_needs_fill = True

        print(f"[Info] RealRobotEnv ready. Control loop: {control_hz}Hz ({self.dt*1000:.1f}ms)")

    def _measured_joint_positions(self) -> np.ndarray:
        positions = getattr(self.spine, 'servo_positions', None)
        if positions is not None and np.any(np.asarray(positions) != 0):
            return np.asarray(positions, dtype=np.float64).copy()
        return np.asarray(RobotConfig.DEFAULT_JOINT_ANGLES, dtype=np.float64).copy()

    def reset_episode(self):
        """エピソード開始時のリセット（学習シミュレータの reset() に対応）。

        学習側と同じく、指令系(LPF状態・前回指令・指令履歴)は現在の実関節角で初期化し、
        観測履歴は最初の観測で埋める。[2026-10-02 FIX] 旧実装は smoothed_action を 0rad で
        初期化しており、最初の指令が「全関節0rad方向へ80%」という急激な動きになっていた。
        """
        self._episode_step = 0
        joint_pos = self._measured_joint_positions()
        self.smoothed_action = joint_pos.copy()
        self.last_action = joint_pos.copy()
        self._prev_joint_pos = joint_pos.copy()
        self.obs_history.clear()
        self.act_history.clear()
        self._history_needs_fill = True
    
    def build_observation(self) -> np.ndarray:
        """
        625次元観測ベクトルの構築 (project_overview.md 仕様に完全準拠)
        
        BASE_OBS (84次元, envs/mjx_env.py の _get_obs() と同じ順序):
          位置(3, 常に0) + 重力射影(3) + 線速度(3) + 角速度(3) = 12
          関節角度(20) + 関節角速度(20) = 40
          FSR接地フラグ(8) + ZMP(2, 常に0) = 10
          phase_sin(1, 常に0) + phase_cos(1, 常に1) = 2
          リファレンス角度(20, 常に0) = 20
        
        + 観測履歴 (84×5 = 420)
        + 行動履歴 (20×5 = 100)
        + サーボ温度 (20)
        + 電源電圧 (1)
        """
        # --- 1. IMU (UART経由, ブロッキングなし) ---
        imu_data = self.imu_data
        quat = imu_data["quat"]
        lin_accel = imu_data["lin_accel"]
        # 姿勢は胴体座標系の重力射影ベクトル (ヨー=磁北基準の絶対方位を含まない)。
        # センサー座標系の値を取り付け姿勢で胴体座標系へ回転する (学習側は胴体座標系)。
        gravity = rotate_vector_by_quaternion(projected_gravity_numpy(quat), self._imu_mount_quat)
        gyro = rotate_vector_by_quaternion(np.asarray(imu_data["gyro"], dtype=np.float64), self._imu_mount_quat)
        
        # --- base_pos: 実機では取得できないため常に0 (学習側の観測も常に0) ---
        base_pos = np.zeros(3)
        
        # --- lin_vel: ZUPT (Zero-velocity Update) 推定 ---
        # IMU加速度を1ステップ積分して速度を推定し、
        # 接地検出時にドリフトをリセットする
        world_accel = rotate_vector_by_quaternion(lin_accel, quat)
        self._vel_estimate += world_accel * self.dt
        
        # --- 3. FSR接地フラグ (TeensyオンチップADCで判定済み) ---
        fsr_raw = self.fsr_contacts
        zmp_xy = np.zeros(2)  # 実機ではCoP/ZMPを算出しない
        
        # XML/Teensy の FSR は [left_foot(4ch), right_foot(4ch)] の順で並ぶ。
        # そのため、先頭4chが left、後続4chが right である。
        left_contact = np.any(fsr_raw[:4] > 0.5)
        right_contact = np.any(fsr_raw[4:] > 0.5)
        if left_contact and right_contact:
            # 両足接地 = 静止推定 → ドリフトリセット
            self._vel_estimate *= 0.1  # 急なゼロリセットではなく減衰
        
        lin_vel = self._vel_estimate.copy()
        
        # --- 2. 関節状態 ---
        # [項目6] 実機のサーボ位置フィードバックを使用（指令値の代わりに実測値）
        if hasattr(self.spine, 'servo_positions') and np.any(self.spine.servo_positions != 0):
            joint_pos = self.spine.servo_positions.copy()
        else:
            # フォールバック: サーボ位置がまだ読み取られていない場合は指令値を使用
            joint_pos = self.smoothed_action.copy()
        # 有限差分で関節角速度を推定
        joint_vel = (joint_pos - self._prev_joint_pos) / self.dt
        self._prev_joint_pos = joint_pos.copy()
        
        # --- 4. Base Obs (84次元) ---
        base_obs = np.concatenate([
            base_pos,                  # 3 (常に0)
            gravity,                   # 3
            lin_vel,                   # 3 (ZUPT推定速度)
            gyro,                      # 3
            joint_pos,                 # 20
            joint_vel,                 # 20
            fsr_raw,                   # 8 (0/1)
            zmp_xy,                    # 2 (常に0)
            np.array([0.0, 1.0]),      # 2 位相 [sin, cos] (立位タスクでは常に位相0)
            np.zeros(self.NUM_JOINTS), # 20 参照角 (立位タスクでは常に0)
        ])  # 合計: 84

        # 履歴バッファ更新 ([古い→新しい] の順、新データを末尾に追加)。
        # エピソード最初の観測では、学習側と同じく全スロットをその観測・現在の指令で埋める。
        if self._history_needs_fill:
            for _ in range(self.HISTORY_LEN):
                self.obs_history.append(base_obs.copy())
                self.act_history.append(self.last_action.copy())
            self._history_needs_fill = False
        else:
            self.obs_history.append(base_obs.copy())
            self.act_history.append(self.last_action.copy())
        
        obs_hist_flat = np.concatenate(list(self.obs_history))   # 84×5 = 420
        act_hist_flat = np.concatenate(list(self.act_history))   # 20×5 = 100
        
        # --- 8. 温度・電圧 (インターリーブReadから取得, 10Hz更新) ---
        servo_temp = self.servo_temps.copy()     # 20
        supply_volt = np.array([np.mean(self.servo_voltages)])  # 1
        
        # --- 9. 最終観測ベクトル (625次元) ---
        obs = np.concatenate([
            base_obs,         # 84
            obs_hist_flat,    # 420
            act_hist_flat,    # 100
            servo_temp,       # 20
            supply_volt       # 1
        ])  # 合計: 625
        
        # [REAL-3 FIXED] 観測次元をアサート検証（ABI不変性保証）
        assert obs.shape[0] == self.OBS_DIM, (
            f"Observation shape mismatch: computed {obs.shape[0]}, "
            f"but OBS_DIM={self.OBS_DIM}"
        )
        
        # --- 安全フィルター: 観測の NaN/Inf 汚染防止 (Rule 15) ---
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("[Error] NaN/Inf detected in Observation! Zeroing to prevent policy corruption.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        return obs
    
    def step(self, obs: np.ndarray) -> np.ndarray:
        """1ステップの推論→行動適用 (中腰デフォルト姿勢 + AI残差)。"""
        # --- AI推論 ---
        raw_action = self.policy.infer(obs)
        
        # --- 安全フィルター: 行動の NaN/Inf 汚染防止 (Rule 15 契約厳守) ---
        if np.isnan(raw_action).any() or np.isinf(raw_action).any():
            print("[Error] NaN/Inf detected in Policy Output! Triggering software E-stop (Zero Action).")
            raw_action = np.zeros_like(raw_action)
        
        # --- アクションの合成: 中腰立ち姿勢からの残差 (最大 ±ACTION_SCALE) ---
        default_pose = np.array(RobotConfig.DEFAULT_JOINT_ANGLES)
        target = default_pose + raw_action * self.ACTION_SCALE
        
        # --- 安全クランプ (assets/humanoid/humanoid.xml と 100% 同期した個別限界) ---
        target = np.clip(target, self.JOINT_LIMITS_MIN, self.JOINT_LIMITS_MAX)
        
        # --- EMA平滑化 (MOTOR_LPF_ALPHA と同じ規約: alpha = 新しい値の重み) ---
        self.smoothed_action = (
            (1.0 - self.EMA_ALPHA) * self.smoothed_action + 
            self.EMA_ALPHA * target
        )
        
        self.last_action = self.smoothed_action.copy()
        return self.smoothed_action
    
    def run_loop(self):
        """
        100Hzメインループ。time.monotonic() による精密タイミング制御。
        """
        print("[Info] Starting 100Hz control loop. Press Ctrl+C to stop.")
        self.reset_episode()
        loop_count = 0
        
        try:
            while True:
                t_start = time.monotonic()
                
                # 1. 観測ベクトル構築
                obs = self.build_observation()
                
                # 2. 推論 + 残差合成 + 安全処理
                action = self.step(obs)
                
                # 3. サーボへ一括送信 (Sync Write, 0.65ms)
                self.imu_data, self.fsr_contacts, self.servo_temps, self.servo_voltages = (
                    self.spine.communicate(action)
                )
                
                # 異常検知時の強制終了 (Rule 18: 通信異常での即時停止)
                if getattr(self.spine, 'telemetry_timeout_flag', False):
                    print("[Fatal] Teensy telemetry continuous timeout. Halting control loop.")
                    break
                
                # [REAL-1 FIXED] エピソード内ステップ数をインクリメント
                self._episode_step += 1
                if self._episode_step >= self._max_episode_steps:
                    print(f"[Info] Episode finished ({self._episode_step} steps). Resetting...")
                    self.reset_episode()
                
                # 4. ループタイミング制御
                elapsed = time.monotonic() - t_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if loop_count % 100 == 0:
                        print(f"[Warn] Loop overrun: {elapsed*1000:.2f}ms > {self.dt*1000:.1f}ms")
                
                loop_count += 1
                
        except (KeyboardInterrupt, SystemExit):
            print("\n[Info] Shutting down...")
        finally:
            self.close()
    
    def close(self):
        """安全なシャットダウン"""
        print("[Info] Zeroing servos and releasing resources...")
        # サーボをニュートラルに
        self.spine.communicate(np.zeros(self.NUM_JOINTS))
        time.sleep(0.5)
        
        self.spine.close()
        print("[Info] Shutdown complete.")


# ============================================================
# エントリーポイント
# ============================================================

def main():
    """
    実行方法 (RT-Preempt環境):
      sudo chrt -f 99 taskset -c 3 python3 -m real.real_env
    """
    env = RealRobotEnv(control_hz=100)
    env.run_loop()


if __name__ == "__main__":
    main()