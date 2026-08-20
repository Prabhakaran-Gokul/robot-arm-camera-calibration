from collections.abc import Sequence

import numpy as np

from robot_arm_camera_calibration.core.samples import CalibrationSample
from robot_arm_camera_calibration.core.transform import Transform


def compute_residuals(
    samples: Sequence[CalibrationSample],
    camera_pose_in_base: Transform,
    marker_pose_in_gripper: Transform,
) -> tuple[float, float]:
    """Per-sample: predict target_pose_in_camera from the solved transforms and compare to what
    was actually observed. Far more physically meaningful than a generic reprojection error."""
    base_pose_in_camera = camera_pose_in_base.inverse()
    translation_errors_m = []
    rotation_errors_deg = []
    for sample in samples:
        predicted = base_pose_in_camera @ sample.gripper_pose_in_base @ marker_pose_in_gripper
        observed = sample.target_pose_in_camera
        error = observed.inverse() @ predicted
        translation_errors_m.append(np.linalg.norm(error.translation))
        angle_rad = np.arccos(np.clip((np.trace(error.rotation) - 1) / 2, -1.0, 1.0))
        rotation_errors_deg.append(np.degrees(angle_rad))

    translation_rmse = float(np.sqrt(np.mean(np.square(translation_errors_m))))
    rotation_rmse = float(np.sqrt(np.mean(np.square(rotation_errors_deg))))
    return translation_rmse, rotation_rmse
