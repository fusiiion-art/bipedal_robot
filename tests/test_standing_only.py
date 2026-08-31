from robot.config import RobotConfig


def test_standing_only_constraints():
    assert RobotConfig.ALLOW_WALKING is False
    assert RobotConfig.USE_REFERENCE_GAIT is False
    assert RobotConfig.TARGET_VEL_X == 0.0
    assert RobotConfig.TARGET_VEL_Y == 0.0
    assert RobotConfig.TARGET_YAW_RATE == 0.0
