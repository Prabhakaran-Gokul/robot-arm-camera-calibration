from robot_arm_camera_calibration.core.config import (
    ArucoMarkerConfig,
    CharucoBoardConfig,
    TargetConfig,
)
from robot_arm_camera_calibration.targets.aruco import ArucoMarkerTarget
from robot_arm_camera_calibration.targets.base import CalibrationTarget
from robot_arm_camera_calibration.targets.charuco import CharucoBoardTarget


def build_target(config: TargetConfig) -> CalibrationTarget:
    if isinstance(config, ArucoMarkerConfig):
        return ArucoMarkerTarget(config.dictionary, config.marker_id, config.marker_length_m)
    if isinstance(config, CharucoBoardConfig):
        return CharucoBoardTarget(
            config.dictionary,
            config.squares_x,
            config.squares_y,
            config.square_length_m,
            config.marker_length_m,
        )
    raise TypeError(f"Unsupported target config type: {type(config).__name__}")
