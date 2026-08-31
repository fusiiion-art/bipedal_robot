import jax
import jax.numpy as jp
from typing import NamedTuple

from robot.config import RobotConfig


class ActuatorState(NamedTuple):
    """サーボモータの内部状態"""
    temperature: jax.Array  # [℃] 各関節の現在温度 shape=(nu,)
    supply_voltage: float   # [V] 現在の供給電圧


class HX30HMModel:
    """
    Hiwonder HX-30HM シリアルバスサーボの物理モデル
    - 熱ダレ (Thermal Derating): 温度上昇に伴うトルク/速度の低下
    - 電圧降下 (Voltage Derating): バッテリー残量低下に伴う性能低下
    - 温度更新: 消費電力に比例した発熱と、環境への放熱の簡易モデル
    
    Spec: HX-30HM
    - Stall Torque: 30 kg.cm (11.1V) = 2.94 N.m
    - No-load Speed: 0.19 sec/60deg (11.1V) = 315 deg/s = 5.5 rad/s
    - Operating Voltage: 6.0 ~ 12.6V
    - Operating Temperature: -5℃ ~ 85℃
    """
    
    # 定格電圧
    NOMINAL_VOLTAGE = 11.1  # [V]
    
    # 熱モデルパラメータ
    THERMAL_RESISTANCE = 0.08   # [℃/W] サーボの熱抵抗（小型サーボの概算）
    THERMAL_MASS = 50.0         # [J/℃] サーボの熱容量
    AMBIENT_TEMP = 25.0         # [℃] 環境温度
    MAX_SAFE_TEMP = 85.0        # [℃] 安全動作上限温度
    DERATING_START_TEMP = 60.0  # [℃] 熱ダレ開始温度
    
    # モータ効率（電気→機械変換効率）
    MOTOR_EFFICIENCY = 0.3  # 小型ホビーサーボの概算
    
    @staticmethod
    def compute_thermal_derating(temperature: jax.Array) -> jax.Array:
        """
        温度に基づくトルク/速度の低減係数を計算する。
        60℃以下: 100% 出力
        60℃〜85℃: 線形に低下 (100% → 30%)
        85℃以上: 30% に制限
        
        Returns:
            derating: [0.3, 1.0] の範囲の係数 shape=(nu,)
        """
        temp_range = HX30HMModel.MAX_SAFE_TEMP - HX30HMModel.DERATING_START_TEMP  # 25℃
        excess = jp.clip(temperature - HX30HMModel.DERATING_START_TEMP, 0.0, temp_range)
        derating = 1.0 - 0.7 * (excess / temp_range)  # 1.0 → 0.3
        return jp.clip(derating, 0.3, 1.0)
    
    @staticmethod
    def compute_voltage_derating(supply_voltage: float) -> jax.Array:
        """
        供給電圧に基づくトルク/速度の低減係数を計算する。
        トルクは電圧にほぼ比例し、速度も電圧に比例する。
        
        Returns:
            derating: [0.0, 1.2] の範囲の係数（過電圧で微増も許容）
        """
        ratio = supply_voltage / HX30HMModel.NOMINAL_VOLTAGE
        return jp.clip(ratio, 0.5, 1.2)
    
    @staticmethod
    def update_temperature(
        state: ActuatorState,
        torque: jax.Array,
        dt: float
    ) -> ActuatorState:
        """
        1制御ステップ分の温度更新を行う。
        
        簡易熱モデル:
          dT/dt = (P_heat - P_cool) / C_thermal
          P_heat = torque^2 * R_motor / efficiency  (銅損の概算)
          P_cool = (T - T_ambient) / R_thermal      (放熱)
        
        Args:
            state: 現在のアクチュエータ状態
            torque: 各関節のトルク [N.m] shape=(nu,)
            dt: 制御ステップ時間 [s]
        
        Returns:
            new_state: 温度が更新された新しい状態
        """
        # 発熱量 (I^2 * R に相当、トルクの2乗に比例)
        motor_resistance = 2.0  # [Ω] 概算のモータ巻線抵抗
        power_heat = jp.square(torque) * motor_resistance / HX30HMModel.MOTOR_EFFICIENCY
        
        # 放熱量
        power_cool = (state.temperature - HX30HMModel.AMBIENT_TEMP) / HX30HMModel.THERMAL_RESISTANCE
        
        # 温度変化
        dT = (power_heat - power_cool) * dt / HX30HMModel.THERMAL_MASS
        new_temp = state.temperature + dT
        
        # 温度をクリップ（環境温度以下にはならない）
        new_temp = jp.clip(new_temp, HX30HMModel.AMBIENT_TEMP, 120.0)
        
        return ActuatorState(
            temperature=new_temp,
            supply_voltage=state.supply_voltage
        )
