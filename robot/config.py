import numpy as np
import os
from pathlib import Path

class RobotConfig:
    """
    Sim-to-Real 二足歩行ロボット '旋風丸' 共通仕様書
    Target Hardware: 
    - Controller: Raspberry Pi 5 (16GB) + アクティブクーラー
    - Servo Driver: Hiwonder BusLinker V3.0 (x4, UART 1Mbps)
    - Actuator: Hiwonder HX-30HM (x20)
    - IMU: BNO055 (UART接続 — I2Cクロックストレッチング回避)
    - FSR判定: Teensy 4.1オンチップADCで読み取り、閾値判定した8ch二値信号
    - 足裏: FSR402 (x8)
    """

    # --- 1. Project Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    MUJOCO_MODEL_PATH = BASE_DIR / "assets" / "humanoid" / "humanoid.xml"
    OUTPUT_DIR = BASE_DIR / "log"

    # --- 1.1. FSR Hardware Layout ---
    # 実機ではTeensy側で接地判定するため、位置はシミュレーション専用。
    FSR_POSITIONS = np.array([
        [-0.08, -0.04], [0.08, -0.04], [-0.08, 0.04], [0.08, 0.04],  # Right foot
        [-0.08,  0.04], [0.08,  0.04], [-0.08, -0.04], [0.08, -0.04],  # Left foot
    ])
    FSR_CONTACT_THRESHOLD = 0.5
    
    # --- 2. Hardware Specs ---
    ROBOT_NAME = "SenpuuMaru_GIY_Type"
    
    # Actuator: Hiwonder HX-30HM Serial Bus Servo (Magnetic Encoder)
    # Spec: 30kg.cm (11.1V) -> 2.94 N.m
    MOTOR_MAX_TORQUE = 3.0       # [N.m] HX-30HMに合わせて修正
    MOTOR_MAX_VELOCITY = 6.5     # [rad/s] (0.19sec/60deg @11.1V)
    MOTOR_VOLTAGE = 11.1         # [V]
    
    # 関節定義 (Fusion 360のURDFとIDを一致させること)
    # 旋風丸の本稼働用設定 (20 DOF)
    JOINT_NAMES = [
        # 右脚 (6関節)
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        # 左脚 (6関節)
        "left_hip_yaw",  "left_hip_roll",  "left_hip_pitch",  "left_knee",  "left_ankle_pitch",  "left_ankle_roll",
        # 右腕 (4関節)
        "right_shoulder_roll", "right_shoulder_pitch", "right_elbow", "right_wrist_pitch",
        # 左腕 (4関節)
        "left_shoulder_roll", "left_shoulder_pitch", "left_elbow", "left_wrist_pitch"
    ]
    
    # --- Actuator Reality Gap (LPF) ---
    MOTOR_LPF_ALPHA = 0.8  # 1st-order Low-Pass Filter coefficient for HX-30HM
    
    NUM_JOINTS = len(JOINT_NAMES)
    INIT_JOINT_ANGLES = np.zeros(NUM_JOINTS)

    # --- お手本（Reference Trajectory）使用のトグルスイッチ ---
    # True: サイクロイド歩行軌道に基づく「残差強化学習 (Residual RL)」
    # False: 「お手本無し強化学習 (Direct RL)」 - 物理法則と報酬だけで自発的歩行を獲得
    USE_REFERENCE_GAIT = False  # お手本無しでやりたい場合は False に設定！

    # お手本無しの学習を劇的に安定させる「中腰デフォルト姿勢 (Default Standing Joint Angles)」
    # ユーザーが設定したXMLの可動域に合わせて、左右で符号を反転（右は膝マイナス、左は膝プラス等）
    DEFAULT_JOINT_ANGLES = np.array([
        # 右脚 (yaw, roll, pitch, knee, ankle_pitch, ankle_roll)
        0.0, 0.0, 0.29, -0.58, -0.29, 0.0,
        # 左脚 (yaw, roll, pitch, knee, ankle_pitch, ankle_roll)
        0.0, 0.0, -0.29, 0.58, 0.29, 0.0,
        # 右腕 (shoulder_roll, shoulder_pitch, elbow, wrist_pitch)
        0.0, 0.0, 0.0, 0.0,
        # 左腕 (shoulder_roll, shoulder_pitch, elbow, wrist_pitch)
        0.0, 0.0, 0.0, 0.0
    ])

    # --- 3. Control Specs ---
    SIM_DT = 1.0 / 400.0     # シミュレーション刻み (2.5ms)
    CONTROL_DECIMATION = 4
    CONTROL_DT = SIM_DT * CONTROL_DECIMATION # 100Hz (10msループ)
    
    # PD制御ゲイン (Sim用) — 外乱耐性のため剛性を引き上げ
    KP = 40.0
    KD = 1.0

    # --- 4. Sim-to-Real Gap Mitigation ---
    # センサーノイズ (実測値に合わせて後で調整)
    NOISE_ANGULAR_POS = np.deg2rad(0.1)  # 磁気エンコーダなので精度UP! ノイズ減
    NOISE_ANGULAR_VEL = np.deg2rad(1.0)
    NOISE_IMU_ANGLE   = np.deg2rad(1.0)
    NOISE_IMU_GYRO    = np.deg2rad(2.0)
    
    # base_pos / lin_vel の大ノイズ (実機ではIMU積分ドリフトで不正確)
    # 学習時にこれらを「信頼できない」特徴量として扱わせるためのDR
    NOISE_BASE_POS    = 0.1   # [m]  — 実機ではゼロ埋め or VIO推定のためドリフト大
    NOISE_LIN_VEL     = 0.5   # [m/s] — IMU積分だと数秒でm/sオーダーのエラー
    
    LATENCY_STEPS = 1 # 1Mbps通信なので遅延は少ないはず

    RANDOM_MASS_SCALE = [0.97, 1.03]  # Phase 1: DR範囲を縮小して基本直立に集中
    RANDOM_FRICTION = [0.7, 1.1]      # Phase 1: 摩擦変動を控えめに
    RANDOM_COM_OFFSET = [-0.02, 0.02]  # Phase 1: 重心偏差を最小化
    RANDOM_PUSH_MAX_FORCE = 0.0  # Phase 0: Gate 0 / Gate A を先に確定し、外乱導入は後に行う
    DISTURBANCE_CURRICULUM = False  # Phase 0 では外乱を無効化して静止直立を安定化させる
    PUSH_DIRECTIONS = 8  # 水平方向を8方位で評価
    PUSH_DURATION_STEPS = 1  # 100Hz制御での印加時間（既定10ms）
    PUSH_FORCE_LEVELS = [0.0, 1.0, 2.0, 3.0]  # [N] 評価時に明示的に掃引する値
    
    # --- カリキュラム学習: 外乱強度スケジュール ---
    #
    # [CRITICAL] envs/mjx_env.py の info['global_step'] は reset() の度に
    # 0へ初期化される実装になっており、MAX_EPISODE_STEPS=1000の制約から
    # 実際にこの関数へ渡る global_step は [0, 1000] の範囲しか取り得ない。
    # 従って以下の絶対ステップ数による閾値(100000等)には理論上絶対に
    # 到達できず、外乱強度は学習全体を通して常に最弱ティアに固定される
    # (envs/mjx_env.py 側の必須追加パッチについてはレビュー本文④を参照)。
    # この点を修正した上で使うことを前提に、値自体もやや前倒しすぎていた
    # ため引き上げている(Direct RLは基本歩行の確立に時間がかかるため)。
    CURRICULUM_SCHEDULE = {
        0: 0.05,          # 基本動作習得
        200000: 0.20,     # 軽い外乱への耐性
        800000: 0.45,     # 中程度外乱への耐性
        2000000: 0.75,    # 強い外乱への耐性
        4000000: 1.00,    # 最大外乱耐性
    }

    # [NEW] 上記の絶対ステップ版に代えて、「総学習ステップ数に対する割合」で
    # カリキュラムを定義する版。USE_REFERENCE_GAIT の有無で総学習ステップ数が
    # 大きく変わっても(10M vs 20〜30M)、同じ相対カリキュラムが自動的に機能する。
    # global_step の代わりに正しく機能する「学習進捗率(0.0〜1.0)」さえ
    # 供給できれば、resolve_curriculum_schedule() で絶対ステップ辞書へ変換できる。
    CURRICULUM_SCHEDULE_FRACTIONS = {
        0.00: 0.00,
        0.10: 0.10,
        0.25: 0.30,
        0.50: 0.60,
        0.75: 1.00,
    }
    # USE_REFERENCE_GAIT=True: 学習側説明書.md の目安(10Mステップ)
    # USE_REFERENCE_GAIT=False (Direct RL): 20〜30Mステップ推奨のため長めに設定
    TOTAL_TRAINING_STEPS_ESTIMATE = 10_000_000 if USE_REFERENCE_GAIT else 25_000_000

    # [NEW] ドメインランダマイゼーションの「範囲」自体もカリキュラム化する場合の
    # スキャフォールド(保守性のパラドックス対策)。envs/mjx_env.py の reset() 内で
    # 学習進捗率を使って easy<->hard を線形補間して消費することを想定。
    # (現状は未接続。RANDOM_MASS_SCALE 等の既存定数は "hard" 側の値と一致させてある)
    DR_CURRICULUM_RANGES = {
        "mass_scale":  {"easy": [0.97, 1.03], "hard": list(RANDOM_MASS_SCALE)},
        "friction":    {"easy": [0.6, 1.1],   "hard": list(RANDOM_FRICTION)},
        "com_offset":  {"easy": [-0.03, 0.03], "hard": list(RANDOM_COM_OFFSET)},
    }
    
    # 熱・電圧のシミュレーションパラメータ
    RANDOM_TEMP = [20.0, 80.0]  # ℃
    RANDOM_VOLT = [9.0, 12.6]   # V

    PRIVILEGED_OBS_DIM = 5 + NUM_JOINTS + 1 # mass, fric, com(3) + temp(N), volt(1)

    # --- 5. RL Settings ---
    # 歩行周期 (秒)
    GAIT_PERIOD = 1.0
    
    # 新アーキテクチャ(RMA/遅延補償対応)における観測空間定義
    HISTORY_LEN = 5 # 過去Nステップの観測と行動(50ms分@100Hz)
    
    # Base観測: 12(胴体) + N*2(関節角/速度) + 10(FSR/ZMP) + 2(位相) + N(理想軌道)
    BASE_OBS_DIM = 12 + (NUM_JOINTS * 2) + 10 + 2 + NUM_JOINTS
    
    # 行動次元
    ACT_DIM = NUM_JOINTS
    
    # サーボ温度(N)とシステム電圧(1)
    SERVO_TEMP_DIM = NUM_JOINTS
    SUPPLY_VOLTAGE_DIM = 1
    
    # 最終的な平坦化されたOBS次元:
    # 履歴バッファに入っている各ステップの観測(Base)と行動を合わせたものの履歴長
    HISTORY_DIM = (BASE_OBS_DIM + ACT_DIM) * HISTORY_LEN
    
    # 現在の観測次元の拡張 (RMA向け) = Base(現在) + 履歴 + 温度 + 電圧
    OBS_DIM = BASE_OBS_DIM + HISTORY_DIM + SERVO_TEMP_DIM + SUPPLY_VOLTAGE_DIM
    
    # 行動空間: ±30度 (Phase 1: 初期探索で暴走しないよう縮小。Phase 2以降で拡大)
    ACTION_SCALE = np.deg2rad(30)

    # === Standing-only mission constraints ===
    # 目的は自律歩行ではなく、外乱に耐えながらその場直立を維持すること。
    # 歩行、踏み出し、支持基底面の変更はいかなる外乱条件でも禁止。
    ALLOW_WALKING = False
    ALLOW_STEPPING = False
    TARGET_VEL_X = 0.0
    TARGET_VEL_Y = 0.0
    TARGET_YAW_RATE = 0.0
    MAX_FOOT_TRANSLATION = 0.005  # [m], 5 mm 未満を許容
    MAX_FOOT_YAW_ROT = np.deg2rad(3.0)
    MAX_SINGLE_FOOT_LIFT = 0.0
    ALLOW_ARM_SWING = True
    ARM_SWING_LIMIT_DEG = 12.0
    FOOT_CONTACT_THRESHOLD = 0.05  # [N] シミュレーション上の各足の最小接触力
    
    # ======================================================
    # 次世代・外乱耐性特化 報酬ウェイト (Phase-Dependent Architecture)
    # ======================================================
    COM_HEIGHT = 0.17            # [FIX] 中腰姿勢での実測CoM高 (旧0.28は高すぎた)
    
    REWARD_WEIGHTS = {
        # Phase 1: 静止直立で確実に正報酬を出すため、安定性と生存を強く重視する。
        "alive": 25.0,
        "fall_penalty": -30.0,

        # 安定維持を最優先
        "upright": 12.0,
        "target_pose": 4.0,
        "com_stab": 10.0,
        "both_feet_contact": 8.0,

        # 外乱が無い Phase 0/1 では回復ボーナスは控えめにする
        "capture_point": 0.5,
        "impedance": 0.2,
        "recovery": 0.5,

        # ペナルティは大きく下げて、PTPな振動で負値が吹き上がらないようにする
        "ang_momentum_z": 0.01,
        "ang_momentum_xy": 0.01,
        "cbf": 0.2,
        "symmetry": 0.0,
        "energy": 0.00005,
        "smoothness": 0.0001,
        "drift": 0.005,
        "slip": 0.01,
        "stance_width": 0.01,

        # 緩和対数バリアも安全域では大きく効かせない
        "barrier_height": 0.2,
        "barrier_torque": 0.1,
    }

    TERMINATION_HEIGHT = 0.10  # [FIX] 初期高さ z=0.165m に合わせて調整 (旧0.15は近すぎた)
    TERMINATION_PITCH = np.deg2rad(45) 
    TERMINATION_ROLL  = np.deg2rad(45)
    
    # 最大エピソード長 (Phase 1: 5秒。短いエピソードで高速学習サイクル)
    MAX_EPISODE_STEPS = 500 

    # ======================================================
    # [NEW] λ_phase(s) 合成状態変数 z(s) の重み
    # z(s) = tilt*|θ_err| + ang_vel*|ω| + lin_vel_err*|v_xy| + disturbance_flag*1{外乱検知}
    # 旧実装は tilt/ang_vel のみで構成されており、外力印加直後
    # (傾きがまだ立ち上がっていない数ステップ)にλ_phaseが1のまま残る
    # 「反応の空白期間」が生じていた。lin_vel_err と disturbance_flag を
    # 追加し、外力印加の瞬間にフェーズ遷移を先行させる。
    # ======================================================
    LAMBDA_PHASE_WEIGHTS = {
        "tilt": 5.0,
        "ang_vel": 0.5,
        "lin_vel_err": 1.5,
        "disturbance_flag": 4.0,
    }
    LAMBDA_PHASE_Z_THRESH = 0.3
    LAMBDA_PHASE_DECAY_K = 10.0

    # --- Capture Point / ZMP マージン計算パラメータ ---
    FOOT_SUPPORT_RADIUS = 0.06   # [m] 足平の実効支持半径。URDF実寸に要調整
    CP_MARGIN_NORM_DIST = 0.15   # [m]

    # --- 外乱復帰ボーナスの時定数 ---
    # 旧: 2ステップ(20ms)は短すぎたため、滑らかな指数減衰の半減期に変更
    RECOVERY_BONUS_WINDOW_STEPS = 50          # 半減期(0.5秒 @100Hz)
    RECOVERY_BONUS_STABILITY_THRESHOLD = 0.4

    # --- ペナルティスケジューリング ---
    # 旧: penalty_scale = clip(step/500,0,1) はエピソード内経過時間
    # (info['step'])に基づいており、MAX_EPISODE_STEPSの半分に相当する
    # 5秒間、学習終盤まで恒久的にペナルティが消失していた。
    # 短いエピソード内グレース(物理リセット直後の過渡応答許容)に短縮し、
    # 学習全体の進行度は training_progress (外部供給) で分離する。
    PENALTY_INTRA_EPISODE_WARMUP_STEPS = 30   # 0.3秒
    # CBF/バリア等ハードウェア安全項は独立した高速ランプ
    SAFETY_PENALTY_WARMUP_STEPS = 10          # 0.1秒

    # --- 緩和対数バリア関数パラメータ ---
    BARRIER_HEIGHT_MARGIN = 0.05        # TERMINATION_HEIGHTからのマージン[m]
    BARRIER_HEIGHT_CLIP = 5.0
    BARRIER_TORQUE_MARGIN_RATIO = 0.15  # MOTOR_MAX_TORQUEに対する比率
    BARRIER_TORQUE_CLIP = 5.0

    @classmethod
    def resolve_curriculum_schedule(cls, total_steps: int = None) -> dict:
        """
        CURRICULUM_SCHEDULE_FRACTIONS を絶対ステップ数の辞書へ変換する。
        train_mjx.py 側で実際の総学習ステップ数(またはその推定値)が
        確定した時点で呼び出し、正しく機能する global_step 相当の値と
        併せて envs/mjx_env.py へ供給することを推奨する。
        """
        total = total_steps if total_steps is not None else cls.TOTAL_TRAINING_STEPS_ESTIMATE
        return {int(frac * total): scale for frac, scale in cls.CURRICULUM_SCHEDULE_FRACTIONS.items()}

    # --- 6. MJX Training Settings ---
    # GPU VRAM等に合わせて調整
    MJX_NUM_ENVS = 2048
    MJX_BATCH_SIZE = 1024
    MJX_UNROLL_LENGTH = 20
    MJX_LEARNING_RATE = 1e-4  # 学習崩壊を防ぐため低めに設定

    @classmethod
    def print_config(cls):
        print(f"=== Robot Configuration: {cls.ROBOT_NAME} ===")
        print(f"Joints: {cls.NUM_JOINTS}")
        print(f"Max Torque: {cls.MOTOR_MAX_TORQUE} Nm (HX-30HM)")