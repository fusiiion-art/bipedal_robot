import numpy as np
from robot.config import RobotConfig

class SensorFusion:
    """
    Fuses data from BNO055 (IMU/UART) and MCP3208 (FSR/SPI) to estimate Zero-Moment Point (ZMP) 
    and reliable ground contact flags for the Control Barrier Functions (CBF).
    """
    def __init__(self, fsr_positions: np.ndarray = None):
        """
        fsr_positions: Array of shape (8, 2) containing the (x, y) coordinates of the 8 FSRs
        relative to the foot center. If None, uses default SenpuuMaru positions.
        """
        if fsr_positions is None:
            self.fsr_pos = RobotConfig.FSR_POSITIONS
        else:
            self.fsr_pos = fsr_positions

    def get_zmp_and_contact(self, fsr_pressures: np.ndarray) -> tuple:
        """
        Calculates the ZMP and binary contact flags based on foot pressure.
        
        fsr_pressures: Array of 8 normalized pressure values [0.0, 1.0]
        
        Returns:
            zmp_xy: (x, y) coordinates of the Zero-Moment Point
            contact_flags: (right_contact, left_contact) boolean tuple
        """
        if len(fsr_pressures) != 8:
            return np.zeros(2), (False, False)

        right_pressures = fsr_pressures[:4]
        left_pressures = fsr_pressures[4:]

        right_total = np.sum(right_pressures)
        left_total = np.sum(left_pressures)
        total_pressure = right_total + left_total

        # Contact flags using a simple threshold
        threshold = 0.1
        right_contact = right_total > threshold
        left_contact = left_total > threshold

        if total_pressure < 0.05:
            # Airborne
            return np.zeros(2), (False, False)

        # Weighted average for ZMP
        zmp_x = np.average(self.fsr_pos[:, 0], weights=fsr_pressures)
        zmp_y = np.average(self.fsr_pos[:, 1], weights=fsr_pressures)
        
        return np.array([zmp_x, zmp_y]), (right_contact, left_contact)
