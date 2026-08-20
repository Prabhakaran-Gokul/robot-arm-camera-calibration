"""Eye-to-hand hand-eye solver built on cv2.calibrateRobotWorldHandEye.

cv2.calibrateRobotWorldHandEye is designed for the *opposite* physical setup
from ours: a camera mounted on the gripper (eye-in-hand) observing a static
target that defines a "world" frame. Its algebra (A_i X = Z B_i) is agnostic
to physical labels, so it can be reused for our eye-to-hand rig (fixed base,
fixed camera, marker on the gripper) by relabeling which physical object
plays which role: our fixed *camera* takes the "world" role (both are the
static member of their respective rigid pair), and our moving *marker* takes
the "cam" role (both move rigidly together with the gripper). Working through
that relabeling against the exact parameter semantics from
`help(cv2.calibrateRobotWorldHandEye)` gives:

    R/t_world2cam[i]  := inverse(target_pose_in_camera[i])   (marker->camera, inverted)
    R/t_base2gripper[i] := inverse(gripper_pose_in_base[i])  (base->gripper, inverted)
    camera_pose_in_base    = inverse(R/t_base2world output)
    marker_pose_in_gripper = inverse(R/t_gripper2cam output)

This wiring is a derivation, not a guaranteed-correct fact — it is pinned
down empirically by the synthetic ground-truth round-trip in
tests/test_solver_synthetic.py, which is the actual source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import cv2
import numpy as np

from robot_arm_camera_calibration.core.errors import InsufficientSamplesError
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.samples import CalibrationSample
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.solvers.base import HandEyeSolver

_MIN_SAMPLES = 3


class OpenCVRobotWorldHandEyeSolver(HandEyeSolver):
    def __init__(self, method: int = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH) -> None:
        self._method = method

    def solve(self, samples: Sequence[CalibrationSample]) -> CalibrationResult:
        if len(samples) < _MIN_SAMPLES:
            raise InsufficientSamplesError(
                f"calibrateRobotWorldHandEye needs >= {_MIN_SAMPLES} samples, got {len(samples)}"
            )

        r_world2cam, t_world2cam, r_base2gripper, t_base2gripper = [], [], [], []
        for sample in samples:
            r, t = sample.target_pose_in_camera.inverse().as_rvec_tvec()
            r_world2cam.append(r)
            t_world2cam.append(t)
            r, t = sample.gripper_pose_in_base.inverse().as_rvec_tvec()
            r_base2gripper.append(r)
            t_base2gripper.append(t)

        r_bw, t_bw, r_gc, t_gc = cv2.calibrateRobotWorldHandEye(
            r_world2cam, t_world2cam, r_base2gripper, t_base2gripper, method=self._method
        )

        camera_pose_in_base = Transform.from_rotation_translation(r_bw, t_bw).inverse()
        marker_pose_in_gripper = Transform.from_rotation_translation(r_gc, t_gc).inverse()

        translation_rmse, rotation_rmse = _residuals(
            samples, camera_pose_in_base, marker_pose_in_gripper
        )

        return CalibrationResult(
            camera_pose_in_base=camera_pose_in_base,
            marker_pose_in_gripper=marker_pose_in_gripper,
            method="cv2.calibrateRobotWorldHandEye",
            num_samples=len(samples),
            created_at=datetime.now(UTC),
            translation_residual_rmse_m=translation_rmse,
            rotation_residual_rmse_deg=rotation_rmse,
        )


def _residuals(
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
