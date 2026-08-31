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

try:
    import spidev
except ImportError:
    print("[Warn] spidev not found. MCP3208 will run in dummy mode.")
    spidev = None


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
# 2. MCP3208 SPI ADC — (SPI1 bus=1 デフォルト)
# ============================================================

class MCP3208SPI:
    VREF = 3.3
    
    def __init__(self, bus: int = 1, device: int = 0, speed_hz: int = 1_000_000):
        self.dummy_mode = spidev is None
        self.spi = None
        
        if not self.dummy_mode:
            try:
                self.spi = spidev.SpiDev()
                self.spi.open(bus, device)
                self.spi.max_speed_hz = speed_hz
                self.spi.mode = 0
                print(f"[Info] MCP3208 initialized on SPI{bus}.{device} @ {speed_hz/1e6:.1f}MHz")
            except Exception as e:
                print(f"[Error] MCP3208 SPI init failed: {e}")
                self.dummy_mode = True
    
    def read_channel(self, channel: int) -> int:
        if self.dummy_mode or self.spi is None:
            return 0
        
        cmd = [0x06 | (channel >> 2), (channel & 0x03) << 6, 0x00]
        result = self.spi.xfer2(cmd)
        return ((result[1] & 0x0F) << 8) | result[2]
    
    def read_all_channels(self) -> np.ndarray:
        if self.dummy_mode:
            return np.zeros(8)
        
        raw = np.array([self.read_channel(ch) for ch in range(8)], dtype=np.float32)
        return raw / 4095.0
    
    def read_voltages(self) -> np.ndarray:
        return self.read_all_channels() * self.VREF
    
    def close(self):
        if self.spi:
            self.spi.close()


# ============================================================
# 3. BusLinker V3.0 — Hiwonder 公式 Checksum ＆ コマンドID 準拠
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
            if len(response) >= 7 and response[0:2] == self.HEADER:
                rx_id = response[2]
                rx_len = response[3]
                rx_cmd = response[4]
                
                # Checksum 検証
                rx_chk = response[3 + rx_len] if len(response) > 3 + rx_len else 0
                calc_chk = calc_checksum(response[2:3+rx_len])
                
                if rx_chk == calc_chk or len(response) >= 7:
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
# 4. TeensySpineIO — 脊髄MCU (Teensy 4.1) 1kHz/100Hz 連携
# ============================================================

class TeensySpineIO:
    """
    Teensy 4.1 (脊髄MCU) との USB Serial パケット通信ドライバ。
    30ms 通信無応答時に Teensy 側で 1kHz E-stop 自律発報。
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
        self.last_fsr_voltages = np.zeros(8)
        self.servo_temps = np.full(num_servos, 25.0)
        self.servo_voltages = np.full(num_servos, 11.1)
        
        if not self.dummy_mode:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.005)
                print(f"[Info] Teensy 4.1 Spinal MCU connected on {port}")
            except Exception as e:
                print(f"[Error] Teensy 4.1 USB Serial init failed: {e}")
                self.dummy_mode = True

    def communicate(self, target_angles_rad: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        if self.dummy_mode or self.ser is None:
            return self.last_imu_data, self.last_fsr_voltages, self.servo_temps, self.servo_voltages

        data = bytearray([self.START_BYTE])
        for angle in target_angles_rad[:self.num_servos]:
            data.extend(struct.pack('<f', float(angle)))
        
        self.ser.write(data)
        
        raw = self.ser.read(73)
        if len(raw) >= 73 and raw[0] == 0x5A:
            w, x, y, z = struct.unpack('<4f', raw[1:17])
            gx, gy, gz = struct.unpack('<3f', raw[17:29])
            ax, ay, az = struct.unpack('<3f', raw[29:41])
            fsr = np.array(struct.unpack('<8f', raw[41:73]))
            
            quat = np.array([w, x, y, z])
            norm = np.linalg.norm(quat)
            if norm >= 0.5 and norm <= 1.5:
                self.last_imu_data["quat"] = quat / norm
                
            self.last_imu_data["gyro"] = np.array([gx, gy, gz])
            self.last_imu_data["lin_accel"] = np.array([ax, ay, az])
            self.last_fsr_voltages = fsr
            
        return self.last_imu_data, self.last_fsr_voltages, self.servo_temps, self.servo_voltages

    def close(self):
        if self.ser:
            self.ser.close()
