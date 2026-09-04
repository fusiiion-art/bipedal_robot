"""
real/real_env.py — RPi5 実機メインループ & 観測ベクトル構築

【実装済み対策 (フィージビリティレビュー反映)】
- ONNX Runtime: シングルスレッド強制 (レイテンシスパイク防止)
- 共有メモリ: atexit + signal + try...finally の3重防御
- 制御周期: 100Hz対応 (dt=10ms, time.monotonic精密タイマー)
- 観測ベクトル: project_overview.md の625次元仕様に完全準拠
"""

import os
import time
import signal
import atexit
import struct
import numpy as np
from typing import Dict, Any, Optional
from collections import deque
from multiprocessing import shared_memory

try:
    import onnxruntime as ort
except ImportError:
    print("[Warn] onnxruntime not found. Policy will run in dummy mode.")
    ort = None

from real.real_io import TeensySpineIO
from robot.math_utils import quat_to_euler
from robot.config import RobotConfig
from robot.gait_generator import numpy_get_reference_trajectory


# ============================================================
# 共有メモリ管理 (3重防御でリーク防止)
# ============================================================

class SharedMemoryManager:
    """
    POSIX共有メモリの安全なライフサイクル管理。
    
    【問題】クラッシュ時にPythonの__del__は呼ばれない→/dev/shm にゾンビが残る
    【対策】atexit + signal + systemd ExecStopPost の3重防御
    """
    
    def __init__(self):
        self._blocks: list = []
    
    def create(self, name: str, size: int) -> shared_memory.SharedMemory:
        """共有メモリブロックを作成し、自動クリーンアップを登録"""
        # 既存のゾンビブロックがあれば先に削除
        try:
            old = shared_memory.SharedMemory(name=name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        self._blocks.append(shm)
        return shm
    
    def attach(self, name: str) -> shared_memory.SharedMemory:
        """既存ブロックにアタッチ (RMAワーカー用)"""
        shm = shared_memory.SharedMemory(name=name, create=False)
        self._blocks.append(shm)
        return shm
    
    def cleanup(self):
        """全ブロックを安全に解放"""
        for shm in self._blocks:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self._blocks.clear()
        print("[Info] Shared memory cleaned up.")
    
    def register_signal_handlers(self):
        """SIGTERM/SIGINT でもクリーンアップを保証"""
        atexit.register(self.cleanup)
        
        def _handler(signum, frame):
            self.cleanup()
            raise SystemExit(0)
        
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)


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
    4. 残差RL合成 (サイクロイド・リファレンス + AI残差)
    5. 安全クランプ + EMA平滑化
    6. Sync Write一括送信
    7. インターリーブRead (2台/ループ)
    """
    
    # robot/config.py 準拠の定数
    NUM_JOINTS = 20
    BASE_OBS_DIM = 84    # 12 + 40 + 10 + 2 + 20
    HISTORY_LEN = 5
    ACT_DIM = 20
    OBS_DIM = 625        # BASE_OBS + HISTORY(520) + TEMP(20) + VOLT(1)
    ACTION_SCALE = np.deg2rad(90)
    RESIDUAL_SCALE = 0.5
    GAIT_PERIOD = 1.0
    EMA_ALPHA = 0.8      # LPF平滑化係数
    
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
        
        # --- 共有メモリ管理 (3重防御) ---
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.register_signal_handlers()
        
        # RMA適応器との共有メモリ: 8次元 latent vector (float32 × 8 = 32 bytes)
        self.shm_latent = self.shm_manager.create("robot_rma_latent", 4 * 8)
        self.latent_vector = np.zeros(8, dtype=np.float32)
        
        # --- ONNX推論 (シングルスレッド) ---
        self.policy = PolicyRunner()
        
        # --- 状態変数 ---
        self.last_action = np.zeros(self.NUM_JOINTS)
        self.smoothed_action = np.zeros(self.NUM_JOINTS)
        
        # ZUPT速度推定用
        self._vel_estimate = np.zeros(3)           # IMU積分速度 [m/s]
        self._prev_joint_pos = np.zeros(self.NUM_JOINTS)  # 関節角速度の有限差分用
        
        # 履歴バッファ (FIFO: 過去5ステップ)
        self.obs_history = deque(
            [np.zeros(self.BASE_OBS_DIM) for _ in range(self.HISTORY_LEN)],
            maxlen=self.HISTORY_LEN
        )
        self.act_history = deque(
            [np.zeros(self.ACT_DIM) for _ in range(self.HISTORY_LEN)],
            maxlen=self.HISTORY_LEN
        )
        
        # 歩行開始時刻
        self.start_time = time.monotonic()
        
        print(f"[Info] RealRobotEnv ready. Control loop: {control_hz}Hz ({self.dt*1000:.1f}ms)")
    
    def _compute_gait_phase(self) -> float:
        """現在の歩行位相 [0, 1) を計算"""
        t = time.monotonic() - self.start_time
        return (t % self.GAIT_PERIOD) / self.GAIT_PERIOD
    
    def _get_reference_trajectory(self, phase: float) -> np.ndarray:
        """
        サイクロイド・リファレンス軌道 (gait_generator.py のロジック実機NumPy共通版)。
        学習環境の jax_get_reference_trajectory と 100% 完全な整合性を担保。
        """
        return numpy_get_reference_trajectory(phase, self.NUM_JOINTS)
    
    def build_observation(self) -> np.ndarray:
        """
        625次元観測ベクトルの構築 (project_overview.md 仕様に完全準拠)
        
        BASE_OBS (84次元):
          位置(3) + RPY(3) + 線速度(3) + 角速度(3) = 12
          関節角度(20) + 関節角速度(20) = 40
          FSR(8) + ZMP(2) = 10
          phase_sin(1) + phase_cos(1) = 2
          リファレンス角度(20) = 20
        
        + 観測履歴 (84×5 = 420)
        + 行動履歴 (20×5 = 100)
        + サーボ温度 (20)
        + 電源電圧 (1)
        """
        # --- 1. IMU (UART経由, ブロッキングなし) ---
        imu_data = self.imu_data
        quat = imu_data["quat"]
        gyro = imu_data["gyro"]
        lin_accel = imu_data["lin_accel"]
        rpy = quat_to_euler(quat)  # roll, pitch, yaw
        
        # --- base_pos: 高さ(Z)のみ脚IKから粗推定、X/Yはゼロ ---
        # 学習側で NOISE_BASE_POS=0.1m の大ノイズDR済みのため
        # 実機側はゼロ埋めでも破綻しない設計
        base_pos = np.zeros(3)
        # base_pos[2] は将来的に脚のIKから推定可能:
        #   z_est ≈ L_thigh * cos(knee_angle) + L_shin * cos(ankle_angle)
        
        # --- lin_vel: ZUPT (Zero-velocity Update) 推定 ---
        # IMU加速度を1ステップ積分して速度を推定し、
        # 接地検出時にドリフトをリセットする
        self._vel_estimate += lin_accel * self.dt
        
        # --- 3. FSR接地フラグ (TeensyオンチップADCで判定済み) ---
        fsr_raw = self.fsr_contacts
        zmp_xy = np.zeros(2)  # 実機ではCoP/ZMPを算出しない
        
        # ZUPT: FSRが両足とも接地を検出 → 速度をゼロリセット
        right_contact = np.any(fsr_raw[:4] > 0.5)
        left_contact = np.any(fsr_raw[4:] > 0.5)
        if right_contact and left_contact:
            # 両足接地 = 静止推定 → ドリフトリセット
            self._vel_estimate *= 0.1  # 急なゼロリセットではなく減衰
        
        lin_vel = self._vel_estimate.copy()
        
        # --- 2. 関節状態 ---
        joint_pos = self.smoothed_action.copy()  # 簡易: 指令値 ≈ 実角度
        # 有限差分で関節角速度を推定
        joint_vel = (joint_pos - self._prev_joint_pos) / self.dt
        self._prev_joint_pos = joint_pos.copy()
        
        # --- 4. 歩行位相 ---
        phase = self._compute_gait_phase()
        phase_obs = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        
        # --- 5. リファレンス軌道 ---
        ref_angles = self._get_reference_trajectory(phase)
        
        # お手本無しのとき、観測のお手本情報(ref_angles)を0にリセットして、AIから目標の軌跡を完全に隠す
        # これにより、ロボットは自身の状態のみを頼りに歩行する（ただし、全体の次元数は変えないため、デプロイメント契約は壊れない）
        if not RobotConfig.USE_REFERENCE_GAIT:
            ref_angles_obs = np.zeros_like(ref_angles)
        else:
            ref_angles_obs = ref_angles
        
        # --- 6. Base Obs (84次元) ---
        base_obs = np.concatenate([
            base_pos,        # 3
            rpy,             # 3
            lin_vel,         # 3 (ZUPT推定速度)
            gyro,            # 3
            joint_pos,       # 20
            joint_vel,       # 20
            fsr_raw,         # 8
            zmp_xy,          # 2
            phase_obs,       # 2
            ref_angles_obs   # 20
        ])  # 合計: 84
        
        # --- 7. 履歴バッファ更新 (FIFO) ---
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
        
        return obs
    
    def step(self, obs: np.ndarray) -> np.ndarray:
        """
        1ステップの推論→行動適用。
        
        USE_REFERENCE_GAIT が True の場合は残差強化学習、False の場合はお手本無しのダイレクト強化学習を実行。
        """
        # --- AI推論 ---
        raw_action = self.policy.infer(obs)
        
        # --- アクションの合成 (USE_REFERENCE_GAITスイッチによるダイレクト/残差の切り替え) ---
        if RobotConfig.USE_REFERENCE_GAIT:
            # AIの出力は「残差」として扱う（元の最大50%に制限）
            phase = self._compute_gait_phase()
            ref_angles = self._get_reference_trajectory(phase)
            residual = raw_action * self.ACTION_SCALE * self.RESIDUAL_SCALE
            target = ref_angles + residual
        else:
            # 「お手本無し」の場合：AIの出力を、安定した「中腰立ち姿勢」からの直接変位（最大±90度）として解釈
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
        self.start_time = time.monotonic()
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
                
                # 4. RMA latent を共有メモリから読み取り (非同期更新)
                self.latent_vector[:] = np.frombuffer(
                    self.shm_latent.buf[:32], dtype=np.float32
                )
                
                # 6. ループタイミング制御
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
        self.shm_manager.cleanup()
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
