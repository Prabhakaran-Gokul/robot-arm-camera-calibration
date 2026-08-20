from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.transform import Transform


def test_calibration_result_roundtrip(tmp_path: Path) -> None:
    camera_pose_in_base = Transform.from_rotation_translation(
        Rotation.from_euler("xyz", [12, -5, 30], degrees=True).as_matrix(),
        np.array([0.5, -0.3, 0.8]),
    )
    marker_pose_in_gripper = Transform.from_rotation_translation(
        Rotation.from_euler("xyz", [0, 0, 45], degrees=True).as_matrix(),
        np.array([0.0, 0.0, 0.03]),
    )
    result = CalibrationResult(
        camera_pose_in_base=camera_pose_in_base,
        marker_pose_in_gripper=marker_pose_in_gripper,
        method="cv2.calibrateRobotWorldHandEye",
        num_samples=18,
        created_at=datetime.now(UTC),
        translation_residual_rmse_m=0.0012,
        rotation_residual_rmse_deg=0.08,
        metadata={"dictionary": "DICT_5X5_100"},
    )
    path = tmp_path / "calibration_result.yaml"
    result.to_yaml(path)

    loaded = CalibrationResult.from_yaml(path)

    np.testing.assert_allclose(
        loaded.camera_pose_in_base.matrix, result.camera_pose_in_base.matrix, atol=1e-10
    )
    np.testing.assert_allclose(
        loaded.marker_pose_in_gripper.matrix, result.marker_pose_in_gripper.matrix, atol=1e-10
    )
    assert loaded.method == result.method
    assert loaded.num_samples == result.num_samples
    assert loaded.created_at == result.created_at
    assert loaded.translation_residual_rmse_m == result.translation_residual_rmse_m
    assert loaded.rotation_residual_rmse_deg == result.rotation_residual_rmse_deg
    assert loaded.metadata == result.metadata
