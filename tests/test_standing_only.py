from robot.config import RobotConfig


def test_standing_only_constraints():
    # 歩行関連の設定(お手本軌道・目標速度・歩容)は 2026-10-02 に削除済み。
    # 立位タスクを歩行へ切り替えるスイッチが復活していないことを確認する。
    for name in (
        "ALLOW_WALKING", "ALLOW_STEPPING", "USE_REFERENCE_GAIT",
        "TARGET_VEL_X", "TARGET_VEL_Y", "TARGET_YAW_RATE", "GAIT_PERIOD",
    ):
        assert not hasattr(RobotConfig, name), name
