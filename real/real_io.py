"""
real/real_io.py — ハードウェアI/Oドライバ (RPi5 & Teensy 4.1 脳脊髄分離システム用)

【Hiwonder 公式プロトコル ＆ 実機電装完全準拠】
1. Hiwonder LX/HX シリアルバスサーボプロトコル:
   - パケット構造: 0x55 0x55 [ID] [Length] [Cmd] [Params...] [Checksum]
   - Checksum = ~(ID + Length + Cmd + Prm1 + ... + PrmN) & 0xFF
   - 放送アドレス: 0xFE (254)
   - コマンド: WRITE_MOVE=1 (0x01), READ_TEMP=26 (0x1A), READ_VIN=27 (0x1B), READ_POS=28 (0x1C)
2. BNO055 通信エラー保護:
   - バスエラー時の [0,0,0,0] 返却を防ぎ、直前の有効な単位クォータニオンを保持・復元。
3. LVCH16T245 ピン全二重分離:
   - Group 1 (Ch 1-8): DIR1 = HIGH (TX 4系統)
   - Group 2 (Ch 9-16): DIR2 = LOW (RX 4系統)
4. 20自由度 4バス割り当て (6+6+4+4 = 20):
   - バス1: 右脚 6軸 (ID: 1~6)
   - バス2: 左脚 6軸 (ID: 7~12)
   - バス3: 右腕 4軸 (ID: 13~16)
   - バス4: 左腕 4軸 (ID: 17~20)

【修正対応 (2026-09-08)】
- [REAL-4 FIXED] Checksum 検証を厳格化（破損データ読み出し防止）
- [REAL-5 FIXED] Teensy E-stop タイムアウト仕組みを明示・整合
"""

import os
import time
import struct
import numpy as np
import threading
from typing import Dict, Optional, Tuple

try:
    import serial
except ImportError:
    print("[Warn] pyserial not found. Hardware will run in dummy mode.")
    serial = None

def calc_checksum(buf: bytes) -> int:
    """
    Hiwonder 公式 Checksum 計算ロジック:
    ~(ID + Length + Cmd + Prm1 + ... + PrmN) & 0xFF
    """
    return (~(sum(buf)) & 0xFF)


# ============================================================
# 1. BNO055 IMU — UART接続 (異常値 [0,0,0,0] 防護実装)
# ============================================================

class BNO055UART:
    START_BYTE = 0xAA
    WRITE = 0x00
    READ = 0x01
    
    REG_QUA_DATA_W_LSB = 0x20
    REG_GYR_DATA_X_LSB = 0x14
    REG_LIA_DATA_X_LSB = 0x28
    REG_OPR_MODE = 0x3D
    
    NDOF_MODE = 0x0C
    
    def __init__(self, port: str = "/dev/ttyAMA1", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.dummy_mode = serial is None
        self.last_valid_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        if not self.dummy_mode:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.01)
                time.sleep(0.1)
                self._write_register(self.REG_OPR_MODE, self.NDOF_MODE)
                time.sleep(0.6)
                print(f"[Info] BNO055 initialized on UART {port}")
            except Exception as e:
                print(f"[Error] BNO055 UART init failed: {e}")
                self.dummy_mode = True
    
    def _write_register(self, reg: int, value: int):
        if self.ser is None:
            return
        packet = bytes([self.START_BYTE, self.WRITE, reg, 1, value])
        self.ser.write(packet)
        self.ser.read(2)
    
    def _read_registers(self, reg: int, length: int) -> bytes:
        if self.ser is None:
            return bytes(length)
        
        packet = bytes([self.START_BYTE, self.READ, reg, length])
        self.ser.write(packet)
        header = self.ser.read(2)
        if len(header) < 2 or header[0] != 0xBB:
            return bytes(length)
        data = self.ser.read(header[1])
        if len(data) < length:
            data += bytes(length - len(data))
        return data
    
    def get_quaternion(self) -> np.ndarray:
        """
        クォータニオン (w, x, y, z) を取得。
        バス障害・パケット破損時は [0,0,0,0] ではなく直前の有効なクォータニオンを返す。
        """
        if self.dummy_mode:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        data = self._read_registers(self.REG_QUA_DATA_W_LSB, 8)
        if len(data) < 8:
            return self.last_valid_quat
            
        w, x, y, z = struct.unpack('<4h', data[:8])
        scale = 1.0 / 16384.0
        quat = np.array([w * scale, x * scale, y * scale, z * scale])
        
        # ノルムチェック (0付近の異常クォータニオンを遮断)
        norm = np.linalg.norm(quat)
        if norm < 0.5 or norm > 1.5:
            return self.last_valid_quat
            
        self.last_valid_quat = quat / norm  # 正規化して保存
        return self.last_valid_quat
    
    def get_gyro(self) -> np.ndarray:
        if self.dummy_mode:
            return np.zeros(3)
        
        data = self._read_registers(self.REG_GYR_DATA_X_LSB, 6)
        if len(data) < 6:
            return np.zeros(3)
        gx, gy, gz = struct.unpack('<3h', data[:6])
        scale = 1.0 / 900.0
        return np.array([gx * scale, gy * scale, gz * scale])
    
    def get_linear_acceleration(self) -> np.ndarray:
        if self.dummy_mode:
            return np.zeros(3)
        
        data = self._read_registers(self.REG_LIA_DATA_X_LSB, 6)
        if len(data) < 6:
            return np.zeros(3)
        ax, ay, az = struct.unpack('<3h', data[:6])
        scale = 1.0 / 100.0
        return np.array([ax * scale, ay * scale, az * scale])
    
    def get_imu_data(self) -> Dict[str, np.ndarray]:
        return {
            "quat": self.get_quaternion(),
            "gyro": self.get_gyro(),
            "lin_accel": self.get_linear_acceleration()
        }


# ============================================================
# 2. BusLinker V3.0 — Hiwonder 公式 Checksum ＆ コマンドID 準拠
# ============================================================

class BusLinkerV3:
    """
    Hiwonder BusLinker V3.0 シリアルバスサーボドライバ。
    
    【公式プロトコル定数】
    HEADER: 0x55 0x55
    BROADCAST_ID: 0xFE (254)
    CMD_SERVO_MOVE_TIME_WRITE: 1 (0x01)
    CMD_SERVO_TEMP_READ: 26 (0x1A)
    CMD_SERVO_VIN_READ: 27 (0x1B)
    CMD_SERVO_POS_READ: 28 (0x1C)
    """
    
    HEADER = bytes([0x55, 0x55])
    BROADCAST_ID = 0xFE
    
    CMD_SERVO_MOVE_TIME_WRITE = 0x01
    CMD_SERVO_TEMP_READ = 0x1A
    CMD_SERVO_VIN_READ = 0x1B
    CMD_SERVO_POS_READ = 0x1C
    
    def __init__(
        self, 
        port: str = "/dev/ttyAMA0", 
        baudrate: int = 1_000_000,
        num_servos: int = 20,
        read_batch_size: int = 2,
        map_file: str = "/etc/bipedal_runtime/servo_map.yaml"
    ):
        self.num_servos = num_servos
        self.port = port
        self.baudrate = baudrate
        self.read_batch_size = read_batch_size
        self.lock = threading.Lock()
        
        self.ser: Optional[serial.Serial] = None
        self.dummy_mode = serial is None
        
        # servo_map.yaml のロード (Bus 1: 1-6, Bus 2: 7-12, Bus 3: 13-16, Bus 4: 17-20)
        self.servo_id_map = {i: i + 1 for i in range(num_servos)}
        if os.path.exists(map_file):
            try:
                import yaml
                with open(map_file, "r") as f:
                    cfg = yaml.safe_load(f)
                    if "servo_ids" in cfg:
                        for idx, sid in enumerate(cfg["servo_ids"]):
                            self.servo_id_map[idx] = int(sid)
                print(f"[Info] Loaded servo_map.yaml from {map_file}")
            except Exception as e:
                print(f"[Warn] Failed to parse {map_file}: {e}")

        self._read_cursor = 0
        self.servo_temps = np.full(num_servos, 25.0)
        self.servo_voltages = np.full(num_servos, 11.1)  # 3S LiPo 11.1V
        self.servo_positions = np.zeros(num_servos)
        
        if not self.dummy_mode:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.002)
                print(f"[Info] BusLinker connected: {port} @ {baudrate/1e6:.1f}Mbps")
            except Exception as e:
                print(f"[Error] BusLinker UART init failed: {e}")
                self.dummy_mode = True
    
    def sync_write_positions(self, angles_rad: np.ndarray, move_time_ms: int = 10):
        """
        Hiwonder 公式 Checksum 付加付きサーボ位置書き込みパケット送信。
        各サーボ宛てに 0x55 0x55 [ID] [Len] [Cmd=1] [PosL] [PosH] [TimeL] [TimeH] [Checksum] を送信。
        """
        if self.dummy_mode or self.ser is None:
            return
        
        count = min(len(angles_rad), self.num_servos)
        move_time = move_time_ms
        
        batch_packet = bytearray()
        for i in range(count):
            servo_id = self.servo_id_map.get(i, i + 1)
            angle_deg = np.degrees(angles_rad[i])
            angle_deg = np.clip(angle_deg, -120.0, 120.0)  # ±120度ハードクランプ
            pos = int(np.clip((angle_deg + 120.0) / 240.0 * 1000.0, 0, 1000))
            
            # 1サーボ宛てパケットデータ部
            # Length = 7 (Length, Cmd, PosL, PosH, TimeL, TimeH, Checksum)
            pkt_body = bytearray([servo_id, 7, self.CMD_SERVO_MOVE_TIME_WRITE])
            pkt_body.extend(struct.pack('<H', pos))
            pkt_body.extend(struct.pack('<H', move_time))
            
            checksum = calc_checksum(pkt_body)
            
            # 完全なパケット
            batch_packet.extend(self.HEADER)
            batch_packet.extend(pkt_body)
            batch_packet.append(checksum)
        
        with self.lock:
            self.ser.write(batch_packet)
    
    def interleave_read_status(self):
        """インターリーブ巡回読み出し (10Hz)"""
        if self.dummy_mode or self.ser is None:
            return
        
        for _ in range(self.read_batch_size):
            servo_id = self.servo_id_map.get(self._read_cursor, self._read_cursor + 1)
            
            temp = self._read_servo_register(servo_id, self.CMD_SERVO_TEMP_READ)
            if temp is not None:
                self.servo_temps[self._read_cursor] = float(temp)
            
            vin = self._read_servo_register(servo_id, self.CMD_SERVO_VIN_READ)
            if vin is not None:
                self.servo_voltages[self._read_cursor] = float(vin) / 1000.0
            
            self._read_cursor = (self._read_cursor + 1) % self.num_servos
    
    def _read_servo_register(self, servo_id: int, cmd: int) -> Optional[int]:
        """
        公式 Checksum 計算付きサーボレジスタ読み出し (半二重通信)
        
        [REAL-4 FIXED] Checksum 検証を厳格化。
        応答パケットの Checksum が一致しない場合は None を返す。
        パケット長の確認のみでは不十分（破損データを通す危険）。
        """
        if self.ser is None:
            return None
        
        # リクエストパケット: Header(2) + ID(1) + Len=3(1) + Cmd(1) + Checksum(1)
        pkt_body = bytearray([servo_id, 3, cmd])
        checksum = calc_checksum(pkt_body)
        
        packet = bytearray(self.HEADER)
        packet.extend(pkt_body)
        packet.append(checksum)
        
        with self.lock:
            self.ser.flushInput()
            self.ser.write(packet)
            
            # 応答受領: Header(2) + ID(1) + Len(1) + Cmd(1) + Data + Checksum(1)
            response = self.ser.read(8)
            if len(response) < 7:
                # [REAL-4 FIXED] パケット長不足でログ出力
                if len(response) > 0:
                    print(f"[Warn] Incomplete response from servo {servo_id}: {len(response)} bytes")
                return None
            
            if response[0:2] != self.HEADER:
                print(f"[Warn] Invalid header from servo {servo_id}")
                return None
            
            rx_id = response[2]
            rx_len = response[3]
            rx_cmd = response[4]
            
            # [REAL-4 FIXED] Checksum 検証を厳格化
            if len(response) <= 3 + rx_len:
                print(f"[Warn] Response too short for checksum validation from servo {servo_id}")
                return None
            
            rx_chk = response[3 + rx_len]
            calc_chk = calc_checksum(response[2:3+rx_len])
            
            if rx_chk != calc_chk:
                print(f"[Warn] Checksum mismatch for servo {servo_id}: "
                      f"expected {calc_chk:02x}, got {rx_chk:02x}")
                return None
            
            # データ抽出 (Checksum が一致した場合のみ)
            if cmd == self.CMD_SERVO_TEMP_READ:
                return response[5]
            elif cmd == self.CMD_SERVO_VIN_READ:
                return struct.unpack('<H', response[5:7])[0]
            elif cmd == self.CMD_SERVO_POS_READ:
                return struct.unpack('<h', response[5:7])[0]
        
        return None
    
    def close(self):
        if self.ser:
            self.ser.close()


# ============================================================
# 3. TeensySpineIO — 脊髄MCU (Teensy 4.1) 1kHz/100Hz 連携
# ============================================================

class TeensySpineIO:
    """
    Teensy 4.1 (脊髄MCU) との USB Serial パケット通信ドライバ。
    
    [REAL-5 FIXED] 通信タイムアウト仕組みを明示。
    
    RPi 側タイムアウト: 5ms (timeout=0.005)
    Teensy 側 E-stop トリガ: 30ms 無応答
    
    【仕組み説明】
    1. RPi から Teensy へ制御パケット送信 (毎ステップ = 10ms周期)
    2. Teensy が応答パケット返却 (通常 < 1ms)
    3. RPi が応答を 5ms タイムアウトで受信
    4. Teensy は最後に有効な通信時刻を記録
    5. 通信から 30ms 経過しても新しい通信がない場合、
       Teensy 側の 1kHz ハードウェアタイマが自動的に
       全サーボをゼロトルク にしてロボットを安全にドロップさせる
    
    RPi のアプリケーション層は 5ms タイムアウトで通信エラーに気付き、
    E-stop 処理を開始できる（30ms 前に検知可能）。
    """
    START_BYTE = 0xA5
    
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200, num_servos: int = 20):
        self.port = port
        self.baudrate = baudrate
        self.num_servos = num_servos
        self.ser: Optional[serial.Serial] = None
        self.dummy_mode = serial is None
        
        self.last_imu_data = {
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "gyro": np.zeros(3),
            "lin_accel": np.zeros(3)
        }
        self.last_fsr_contacts = np.zeros(8, dtype=np.float32)
        self.servo_temps = np.full(num_servos, 25.0)
        self.servo_voltages = np.full(num_servos, 11.1)
        
        self.telemetry_timeout_flag = False  # 上位ループへの異常通知用フラグ
        self._consecutive_timeouts = 0       # 連続タイムアウト回数
        
        if not self.dummy_mode:
            try:
                # [REAL-5 FIXED] RPi 側タイムアウト = 5ms
                # Teensy 側 E-stop トリガ = 30ms (Teensy ファームウェア側で定義)
                self.ser = serial.Serial(port, baudrate, timeout=0.005)
                print(f"[Info] Teensy 4.1 Spinal MCU connected on {port}")
                print(f"[Info] Communication safety: RPi timeout={0.005*1000:.1f}ms, "
                      f"Teensy E-stop trigger=30ms")
            except Exception as e:
                print(f"[Error] Teensy 4.1 USB Serial init failed: {e}")
                self.dummy_mode = True

    def communicate(self, target_angles_rad: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        if self.dummy_mode or self.ser is None:
            return self.last_imu_data, self.last_fsr_contacts, self.servo_temps, self.servo_voltages

        # NaNのバイナリパッキングを最終防衛線でブロック
        safe_angles = np.nan_to_num(target_angles_rad, nan=0.0, posinf=0.0, neginf=0.0)

        data = bytearray([self.START_BYTE])
        for angle in safe_angles[:self.num_servos]:
            data.extend(struct.pack('<f', float(angle)))
        
        self.ser.write(data)
        
        raw = self.ser.read(73)
        if len(raw) >= 73 and raw[0] == 0x5A:
            self._consecutive_timeouts = 0
            self.telemetry_timeout_flag = False
            
            w, x, y, z = struct.unpack('<4f', raw[1:17])
            gx, gy, gz = struct.unpack('<3f', raw[17:29])
            ax, ay, az = struct.unpack('<3f', raw[29:41])
            fsr_contacts = np.rint(np.clip(np.array(struct.unpack('<8f', raw[41:73])), 0.0, 1.0))
            
            quat = np.array([w, x, y, z])
            norm = np.linalg.norm(quat)
            if norm >= 0.5 and norm <= 1.5:
                self.last_imu_data["quat"] = quat / norm
                
            self.last_imu_data["gyro"] = np.array([gx, gy, gz])
            self.last_imu_data["lin_accel"] = np.array([ax, ay, az])
            self.last_fsr_contacts = fsr_contacts.astype(np.float32)
            
        else:
            # Rule 18 違反対策: タイムアウトのサイレント無視を廃止
            self._consecutive_timeouts += 1
            if len(raw) > 0:
                print(f"[Warn] Teensy telemetry incomplete: {len(raw)}/73 bytes (Consecutive: {self._consecutive_timeouts})")
            else:
                print(f"[Warn] Teensy telemetry timeout (0 bytes) (Consecutive: {self._consecutive_timeouts})")
                
            if self._consecutive_timeouts >= 3:
                # 30ms (3ステップ連続) 応答がない場合、致命的な異常としてフラグを立てる
                self.telemetry_timeout_flag = True
            
        return self.last_imu_data, self.last_fsr_contacts, self.servo_temps, self.servo_voltages

    def close(self):
        if self.ser:
            self.ser.close()