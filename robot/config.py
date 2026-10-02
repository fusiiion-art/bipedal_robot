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
    
    【修正対応 (2026-09-08)】
    - [CONFIG-1 FIXED] CURRICULUM_SCHEDULE を廃止、CURRICULUM_SCHEDULE_FRACTIONS に統一
    - [CONFIG-3 FIXED] INITIAL_HEIGHT を明記、TERMINATION_HEIGHT の根拠を記載
    - [GAIT-2 FIXED] 歩容パラメータを config.py に一元化
    """

    # --- 1. Project Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    MUJOCO_MODEL_PATH = BASE_DIR / "assets" / "humanoid" / "humanoid.xml"

    # --- 1.1. FSR Hardware Layout ---
    # MuJoCo の <sensor> は IMU(gyro 3/accel 3/quat 4) の後に 8ch FSR touch が
    # left-foot 4ch(FL FR BL BR) → right-foot 4ch の順に並ぶ (sensordata[10:18])。
    # 実機は Teensy が各FSRを閾値判定した 0/1 フラグを送るため、方策の観測も
    # 同じ閾値で二値化する (docs/master_plan.md §1.3「足裏接触（バイナリ）」)。
    FSR_SENSOR_SLICE = slice(10, 18)
    FSR_CONTACT_THRESHOLD = 0.2  # [N] 1センサーあたりの接地判定しきい値 (FSR402 の作動荷重 ≈0.2N)

    # BNO055 の取り付け姿勢 (センサー座標系 → 胴体座標系の回転, クォータニオン [w,x,y,z])。
    # assets/humanoid/humanoid.xml の imu_bno055_site euler="-90 0 0" と一致させること
    # (tests/test_seed_readiness.py で検証)。実機の取り付け向きは要確認。
    IMU_MOUNT_QUAT = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])
    
    # --- 2. Hardware Specs ---
    
    # Actuator: Hiwonder HX-30HM Serial Bus Servo (Magnetic Encoder)
    # Spec: 30kg.cm (11.1V) -> 2.94 N.m
    # [2026-09-29] カタログ値(30kg.cm@11.1V=2.94N.m)に合わせ、assets/humanoid/humanoid.xml の
    # 全アクチュエータ forcerange と一致させる(tests/test_reset_and_cbf_contract.py で検証)。
    MOTOR_MAX_TORQUE = 2.94      # [N.m] HX-30HMに合わせて修正
    # [2026-09-29] 旧値6.5は注記の仕様(60deg/0.19s = 5.51rad/s)と不一致だった。
    MOTOR_MAX_VELOCITY = np.deg2rad(60) / 0.19  # [rad/s] ≈5.51 (0.19sec/60deg @11.1V)
    
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
    # サーボの位置分解能: 0〜1000 カウントで ±120° (real/real_io.py の sync_write_positions)。
    # 指令の変化がこれ未満ならサーボは動かない、として指令のデッドバンドに使う
    # (旧実装は根拠のない 0.02rad ≈ 5カウントで、1.15°未満の微修正ができなかった)。
    SERVO_POSITION_RESOLUTION = np.deg2rad(240.0 / 1000.0)  # [rad] ≈0.0042
    
    NUM_JOINTS = len(JOINT_NAMES)

    # 中腰デフォルト姿勢 (Default Standing Joint Angles)。行動は この姿勢からの残差 Δq。
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
    # lin_vel の大ノイズ (実機ではIMU積分(ZUPT)推定のため不正確)。
    # 学習時にこれを「信頼できない」特徴量として扱わせるためのDR。
    # (base_pos は実機で取得できないため観測では常に0。真値は critic の特権観測にのみ入る)
    NOISE_LIN_VEL     = 0.5   # [m/s] — IMU積分だと数秒でm/sオーダーのエラー

    # 初期状態分布 (docs/master_plan.md §1.6)。nominal姿勢の関節角・関節角速度に一様ノイズを
    # 加え、低い方の足裏が nominal と同じ高さで接地するよう胴体高さを補正して spawn する。
    # 0 にすると従来どおり常に同一姿勢から開始する(この場合ゼロ行動でも直立を維持できるため、
    # Gate A の評価が「学習した方策」と「何もしない方策」を区別できない)。
    INIT_JOINT_POS_NOISE = 0.05   # [rad] 一様 ±
    INIT_JOINT_VEL_NOISE = 0.2    # [rad/s] 一様 ±
    
    RANDOM_MASS_SCALE = [0.97, 1.03]  # Phase 1: DR範囲を縮小して基本直立に集中
    RANDOM_FRICTION = [0.7, 1.1]      # Phase 1: 摩擦変動を控えめに
    RANDOM_COM_OFFSET = [-0.02, 0.02]  # Phase 1: 重心偏差を最小化
    RANDOM_PUSH_MAX_FORCE = 0.0  # Phase 0: Gate 0 / Gate A を先に確定し、外乱導入は後に行う
    DISTURBANCE_CURRICULUM = False  # Phase 0 では外乱を無効化して静止直立を安定化させる
    # [2026-09-26追加] Gate B robustness envelope評価(scratch/gate_b_eval.py)が
    # 既定で掃引する外力[N]のリスト。master_plan.md §0.1の目標外乱スペック
    # (突っつきインパルス目標値・Phase-1物理限界値)が未記入のため、根拠のない
    # 数値をここで勝手に補わず既定は空リストにしている。実行時は必ず
    # `--force-levels`で明示的に指定すること(§9.2: 外乱上限を解析限界以上へ
    # 自動拡大してはならない、の原則に従う)。
    PUSH_FORCE_LEVELS = []
    
    # 熱・電圧のシミュレーションパラメータ
    RANDOM_TEMP = [25.0, 80.0]  # ℃ (下限は熱モデルの外気温 AMBIENT_TEMP。これ未満は初回更新で25℃に丸められていた)
    RANDOM_VOLT = [9.0, 12.6]   # V

    # --- 5. RL Settings ---
    HISTORY_LEN = 5 # 過去Nステップの観測と行動(50ms分@100Hz)
    
    # Base観測 (84):
    #   base_pos(3, 常に0) + 重力射影ベクトル(3) + 線速度(3) + 角速度(3)
    #   + 関節角(N) + 関節角速度(N) + FSR接地フラグ(8) + ZMP(2, 常に0)
    #   + 位相(2, 常に[0,1]) + 参照角(N, 常に0)
    # 常に定数のチャネルは歩行用の旧設計の名残だが、実機ONNXとの625次元契約を保つため残している。
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
    
    # critic のみが使う特権観測 (docs/master_plan.md §1.3 asymmetric actor-critic)。
    # envs/mjx_env.py の _get_privileged_obs() の内訳と一致させること。
    #   真の base_pos(3) + 重力射影(3) + 線速度(3) + 角速度(3) + 関節角/速度(2N)
    #   + FSR連続値(8) + 足裏水平変位(4) + DR値(質量1+摩擦1+重心3+関節粘性N+関節摩擦N)
    #   + 外力(3) + 外乱フラグ(1)
    PRIVILEGED_EXTRA_DIM = 12 + 2 * NUM_JOINTS + 8 + 4 + (5 + 2 * NUM_JOINTS) + 4
    PRIVILEGED_OBS_DIM = OBS_DIM + PRIVILEGED_EXTRA_DIM

    # 行動空間: ±30度 (Phase 1: 初期探索で暴走しないよう縮小。Phase 2以降で拡大)
    ACTION_SCALE = np.deg2rad(30)

    # === Standing-only mission (docs/master_plan.md: standing_fixed_feet が唯一のスコープ) ===
    # 目的は自律歩行ではなく、外乱に耐えながら両足接地のままその場直立を維持すること。
    # [2026-10-01 方針決定] 目的は直立姿勢の維持であり、足裏が初期位置から多少ずれることは許容する。
    # ただし閾値を外すと「足を滑らせて逃げる」立ち方も合格になるため、歩行・踏み出しの歯止めとして
    # 20mm を残す(旧5mm)。Gate A の slip_ok 判定(scratch/phase0_eval_diagnostics.py)で使用。
    MAX_FOOT_TRANSLATION = 0.020  # [m], 20 mm 以下を許容

    # [T2 2026-10-02] Gate A 成功条件 (master_plan §0.3 の upright / torque_ok / height_ok /
    # no_illegal_contact)。値は人間承認済み (2026-10-02)。判定は scratch/phase0_eval_diagnostics.py。
    GATE_A_MAX_TILT_DEG = 10.0             # 胴体z軸とworld鉛直のなす角の最大値 [deg]
    GATE_A_MAX_TORQUE_SAT_RATE = 0.01      # |τ| >= 0.98·MOTOR_MAX_TORQUE の関節を含むstepの割合
    GATE_A_MAX_REL_HEIGHT_DROP = 0.02      # 初期の足裏相対高さからの許容低下 [m]
    GATE_A_ILLEGAL_CONTACT_STEPS = 2       # 床と足裏以外の接触がこのstep数以上連続したら違反
    FOOT_CONTACT_THRESHOLD = 0.5  # [N] シミュレーション上の各足FSR4センサー平均の最小接触力 (足全体で2.0N以上)
    
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
        # com_stab: r_still(常時) および r_com_stab(安定時ボーナス)の重み (安定時は実質20.0相当)
        "com_stab": 10.0,
        "both_feet_contact": 8.0,
        # foot_balance: 左右荷重バランス (1.0=完全均等50:50, 片足脱力ローカルミニマム排除)
        "foot_balance": 6.0,

        # 外乱が無い Phase 0/1 では回復ボーナスは控えめにする
        "capture_point": 0.5,
        "impedance": 0.2,
        "recovery": 0.5,

        # ペナルティは大きく下げて、PTPな振動で負値が吹き上がらないようにする
        # 注: step_penalty (0~20) は踏み出し抑止のハード制約として
        #     REWARD_WEIGHTS を介さず mjx_rewards.py 内で直接加算される (実質重み1.0)。
        "ang_momentum_z": 0.01,
        "ang_momentum_xy": 0.01,
        "cbf": 0.2,
        "energy": 0.00005,
        "smoothness": 0.0001,
        "drift": 0.005,
        # slip: 足裏並進速度 (data.cvel) に対する滑りペナルティ重み
        "slip": 2.0,
        "stance_width": 0.01,

        # 緩和対数バリアも安全域では大きく効かせない
        "barrier_height": 0.2,
        "barrier_torque": 0.1,
    }

    # --- [CONFIG-3 FIXED] 初期高さを明記、終了条件を根拠付き ---
    # mjx_env.py の reset() で qpos[2] = 0.1773 として設定される
    # 注意: この値自体はワールド座標系での胴体初期位置だが、
    # TERMINATION_HEIGHT は envs/mjx_rewards.py::compute() 内で
    # 「足裏を基準にした相対高さ (base_pos[2] - lowest_foot_z)」との
    # 比較にのみ使われる（絶対座標の閾値ではない。"Contract Violation B"
    # 対応で相対高さ判定に統一済み）。以下の引き算は「7cmというマージン量」
    # を求めるための便宜的な計算であり、絶対座標の意味は持たない。
    INITIAL_HEIGHT = 0.1773  # [m] 直立姿勢での胴体初期位置（ワールド座標Z）
    
    # 転倒判定の高さマージン。足裏基準の相対高さがこの値を下回ったら終了。
    # 根拠: 中腰姿勢（膝屈曲）での安定限界に相当するマージンとして
    # INITIAL_HEIGHT - 0.07 を流用している（比較対象は相対高さ）。
    TERMINATION_HEIGHT = INITIAL_HEIGHT - 0.07  # = 0.1073m（相対高さの閾値）
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
    # [2026-09-29] reset直後の過渡応答(指令系のゼロ初期化が原因)は envs/mjx_env.py の
    # reset() 修正で解消したため撤廃(0 = 猶予なし)。Gate A は step 1 から足裏変位を測る。
    PENALTY_INTRA_EPISODE_WARMUP_STEPS = 0
    # [2026-09-29] ソフトペナルティは training_progress がこの割合に達した時点で満額にする。
    # 旧実装は学習全体(進捗0→1)で線形に立ち上げており、満額になるのは学習の最後だけだった。
    PENALTY_PROGRESS_RAMP_FRACTION = 0.3
    # CBF/バリア等ハードウェア安全項は独立した高速ランプ
    SAFETY_PENALTY_WARMUP_STEPS = 10          # 0.1秒

    # --- 緩和対数バリア関数パラメータ ---
    BARRIER_HEIGHT_MARGIN = 0.05        # TERMINATION_HEIGHTからのマージン[m]
    BARRIER_HEIGHT_CLIP = 5.0
    BARRIER_TORQUE_MARGIN_RATIO = 0.15  # MOTOR_MAX_TORQUEに対する比率
    BARRIER_TORQUE_CLIP = 5.0

    # ======================================================
    # [CONFIG-1 FIXED] カリキュラム学習: 外乱強度スケジュール
    # ======================================================
    # 【設計】学習進捗率 (0.0~1.0) に基づく相対スケジュール。
    # 絶対ステップ数による CURRICULUM_SCHEDULE は廃止。
    # 
    # 理由: 総学習ステップ数(--steps)を変えても同じ相対カリキュラムが機能する。
    # 
    # 供給元: training_progress (envs/training_wrapper.py の TrainingProgressWrapper)
    # 詳細: envs/mjx_rewards.py の curriculum_disturbance_scale() を参照
    CURRICULUM_SCHEDULE_FRACTIONS = {
        0.00: 0.00,  # 学習開始時: 外乱なし
        0.10: 0.10,  # 10%進捗: 微弱外乱
        0.25: 0.30,  # 25%進捗: 軽い外乱
        0.50: 0.60,  # 50%進捗: 中程度外乱
        0.75: 1.00,  # 75%進捗: 最大外乱
    }
