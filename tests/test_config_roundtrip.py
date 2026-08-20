from pathlib import Path

import pytest

from robot_arm_camera_calibration.core.config import (
    ArucoMarkerConfig,
    CalibrationConfig,
    CameraConfig,
    CharucoBoardConfig,
    JogLimits,
    WorkspaceLimits,
)


def test_aruco_config_roundtrip(tmp_path: Path) -> None:
    config = CalibrationConfig(
        robot_ip="192.168.1.10",
        target=ArucoMarkerConfig(marker_id=7, marker_length_m=0.04),
        camera=CameraConfig(intrinsics_source="device"),
        workspace_limits=WorkspaceLimits((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8)),
        jog_limits=JogLimits(max_step_m=0.01),
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    loaded = CalibrationConfig.from_yaml(path)

    assert loaded.robot_ip == config.robot_ip
    assert loaded.target == config.target
    assert loaded.workspace_limits == config.workspace_limits
    assert loaded.jog_limits == config.jog_limits


def test_charuco_config_roundtrip(tmp_path: Path) -> None:
    config = CalibrationConfig(
        robot_ip="10.0.0.5", target=CharucoBoardConfig(squares_x=6, squares_y=8)
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    loaded = CalibrationConfig.from_yaml(path)

    assert loaded.target == config.target


def test_workspace_limits_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_xyz_m"):
        WorkspaceLimits((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))
