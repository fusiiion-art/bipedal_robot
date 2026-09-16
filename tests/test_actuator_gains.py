import numpy as np

from envs.mjx_env import SenpuuMaruMJXEnv
from robot.config import RobotConfig


def test_robot_config_gains_are_applied_to_mjx_model():
    env = SenpuuMaruMJXEnv()
    model = env._mjx_model

    assert np.allclose(np.asarray(model.actuator_gainprm)[:, 0], RobotConfig.KP)
    assert np.allclose(np.asarray(model.actuator_biasprm)[:, 1], -RobotConfig.KP)
    assert np.allclose(np.asarray(model.actuator_biasprm)[:, 2], -RobotConfig.KD)


def test_robot_config_gains_are_applied_to_brax_system():
    env = SenpuuMaruMJXEnv()
    actuator = env.sys.actuator

    assert np.allclose(np.asarray(actuator.gain), RobotConfig.KP)
    assert np.allclose(np.asarray(actuator.bias_q), -RobotConfig.KP)
    assert np.allclose(np.asarray(actuator.bias_qd), -RobotConfig.KD)