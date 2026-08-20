"""Eye-to-hand solver built on cv2.calibrateHandEye (Tsai/Park/... relative-motion methods).

Unlike OpenCVRobotWorldHandEyeSolver, this only needs relative motion between pairs of samples
to solve the camera extrinsic, which is why this family of methods (Tsai, Park, Horaud, ...) is
the long-established industry standard (e.g. ROS's easy_handeye) and often converges well with
fewer, less rotationally-diverse samples than the robot-world formulation needs.

The tradeoff: it is a hard mathematical fact, not an implementation gap, that this relative-
motion formulation is blind to the EE-to-marker offset — it algebraically cancels out of every
equation used to solve for the camera extrinsic, so it never appears in cv2.calibrateHandEye's
output. This solver recovers it afterward as a separate closed-form step (no new unknowns, just
algebra) once the camera extrinsic is known, so it still returns both outputs.

Eye-to-hand wiring (verified against cv2.calibrateHandEye's own docstring, not guessed — its
documented "eye-to-hand" equation is ^g{T_b} · ^b{T_c} · ^c{T_t} = constant across samples,
since a rigidly-mounted marker's pose in the gripper frame can't change):

    R/t_gripper2base[i] := inverse(gripper_pose_in_base[i])  (base-in-gripper-frame, inverted)
    R/t_target2cam[i]   := target_pose_in_camera[i]          (direct PnP output, unmodified)
    camera_pose_in_base  = output R/t_cam2gripper, used as-is (despite the parameter's name,
                            the eye-to-hand adaptation makes this represent base_T_cam directly)

Pinned down empirically by the synthetic ground-truth round-trip in
tests/test_solver_synthetic.py, same as the robot-world solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.errors import InsufficientSamplesError
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.samples import CalibrationSample
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.solvers._residuals import compute_residuals
from robot_arm_camera_calibration.solvers.base import HandEyeSolver

_MIN_SAMPLES = 3


class OpenCVHandEyeSolver(HandEyeSolver):
    def __init__(self, method: int = cv2.CALIB_HAND_EYE_PARK) -> None:
        self._method = method

    def solve(self, samples: Sequence[CalibrationSample]) -> CalibrationResult:
        if len(samples) < _MIN_SAMPLES:
            raise InsufficientSamplesError(
                f"calibrateHandEye needs >= {_MIN_SAMPLES} samples, got {len(samples)}"
            )

        base_pose_in_grippers = [s.gripper_pose_in_base.inverse() for s in samples]
        r_gripper2base = [t.rotation for t in base_pose_in_grippers]
        t_gripper2base = [t.translation for t in base_pose_in_grippers]
        r_target2cam = [s.target_pose_in_camera.rotation for s in samples]
        t_target2cam = [s.target_pose_in_camera.translation for s in samples]

        r_bc, t_bc = cv2.calibrateHandEye(
            r_gripper2base, t_gripper2base, r_target2cam, t_target2cam, method=self._method
        )
        camera_pose_in_base = Transform.from_rotation_translation(r_bc, t_bc)

        marker_pose_in_gripper = _average_marker_pose_in_gripper(
            samples, base_pose_in_grippers, camera_pose_in_base
        )

        translation_rmse, rotation_rmse = compute_residuals(
            samples, camera_pose_in_base, marker_pose_in_gripper
        )

        return CalibrationResult(
            camera_pose_in_base=camera_pose_in_base,
            marker_pose_in_gripper=marker_pose_in_gripper,
            method=f"cv2.calibrateHandEye(method={self._method})",
            num_samples=len(samples),
            created_at=datetime.now(UTC),
            translation_residual_rmse_m=translation_rmse,
            rotation_residual_rmse_deg=rotation_rmse,
        )


def _average_marker_pose_in_gripper(
    samples: Sequence[CalibrationSample],
    base_pose_in_grippers: Sequence[Transform],
    camera_pose_in_base: Transform,
) -> Transform:
    """Per sample: gripper_T_base @ base_T_camera @ camera_T_target = gripper_T_target, which
    is constant (the marker is rigidly mounted) — average these per-sample estimates."""
    estimates = [
        base_pose_in_gripper @ camera_pose_in_base @ sample.target_pose_in_camera
        for sample, base_pose_in_gripper in zip(samples, base_pose_in_grippers, strict=True)
    ]
    mean_rotation = Rotation.from_matrix([e.rotation for e in estimates]).mean().as_matrix()
    mean_translation = np.mean([e.translation for e in estimates], axis=0)
    return Transform.from_rotation_translation(mean_rotation, mean_translation)
