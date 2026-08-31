import jax
import jax.numpy as jp
from mujoco import mjx
from robot.config import RobotConfig

class CBFSafetyFilter:
    """
    Control Barrier Function (CBF) Safety Layer for Bipedal Posture Control.
    
    In JAX/MJX training, running a full QP solver per step per environment is very slow.
    This provides a simplified, differentiable margin-based clamping mechanism that mimics CBF,
    ensuring that nominal actions pushing the system towards unsafe states (e.g., instability,
    joint limits) are heavily penalized or clipped.
    
    During real-world deployment on Raspberry Pi, a strict QP-based CBF should replace this.
    See: safety/cbf_realworld.py (future)
    """
    
    def __init__(self):
        self.max_torque = RobotConfig.MOTOR_MAX_TORQUE
        self.joint_pos_margin = 0.1 # rad buffer from hard limits
        self.max_vel = RobotConfig.MOTOR_MAX_VELOCITY
        
    def filter_action(self, nominal_action: jax.Array, limit_lower: jax.Array, limit_upper: jax.Array) -> jax.Array:
        """
        Takes the RL's nominal action and projects it to a safe set.
        """
        safe_action = jp.clip(nominal_action, limit_lower + self.joint_pos_margin, limit_upper - self.joint_pos_margin)
        return safe_action

    def compute_cbf_penalty(self, nominal_action: jax.Array, limit_lower: jax.Array, limit_upper: jax.Array) -> jax.Array:
        """
        Calculates a penalty based on how much the CBF had to intervene.
        """
        safe_lower = limit_lower + self.joint_pos_margin
        safe_upper = limit_upper - self.joint_pos_margin
        
        k = 10.0 # steepness parameter
        upper_violation = jax.nn.softplus(k * (nominal_action - safe_upper))
        lower_violation = jax.nn.softplus(k * (safe_lower - nominal_action))
        
        return jp.sum(upper_violation + lower_violation)
