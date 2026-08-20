"""Blocking-gate tests: prove each solver's OpenCV wiring is correct by forward-simulating fake
samples from a known ground truth and checking the solver recovers it. This is the actual source
of truth for each solver's input/output convention, not the docstring derivations in
solvers/opencv_robot_world.py and solvers/opencv_hand_eye.py."""

from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.samples import CalibrationSample
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.solvers.base import HandEyeSolver
from robot_arm_camera_calibration.solvers.opencv_hand_eye import OpenCVHandEyeSolver
from robot_arm_camera_calibration.solvers.opencv_robot_world import (
    OpenCVRobotWorldHandEyeSolver,
)

_NUM_SAMPLES = 20
_SOLVERS = [
    pytest.param(OpenCVRobotWorldHandEyeSolver(), id="robot-world"),
    pytest.param(OpenCVHandEyeSolver(), id="hand-eye-park"),
    pytest.param(OpenCVHandEyeSolver(method=0), id="hand-eye-tsai"),
]


def _random_transform(rng: np.random.Generator, translation_scale: float) -> Transform:
    rotation = Rotation.random(rng=rng).as_matrix()
    translation = rng.uniform(-translation_scale, translation_scale, 3)
    return Transform.from_rotation_translation(rotation, translation)


def _simulate_samples(
    rng: np.random.Generator,
    camera_pose_in_base: Transform,
    marker_pose_in_gripper: Transform,
    *,
    noise_std_m: float = 0.0,
    noise_std_deg: float = 0.0,
) -> list[CalibrationSample]:
    base_pose_in_camera = camera_pose_in_base.inverse()
    samples = []
    for i in range(_NUM_SAMPLES):
        gripper_pose_in_base = _random_transform(rng, translation_scale=0.4)
        target_pose_in_camera = base_pose_in_camera @ gripper_pose_in_base @ marker_pose_in_gripper
        if noise_std_m or noise_std_deg:
            target_pose_in_camera = _perturb(rng, target_pose_in_camera, noise_std_m, noise_std_deg)
        samples.append(
            CalibrationSample(
                index=i,
                timestamp=datetime.now(UTC),
                gripper_pose_in_base=gripper_pose_in_base,
                target_pose_in_camera=target_pose_in_camera,
            )
        )
    return samples


def _perturb(
    rng: np.random.Generator, transform: Transform, noise_std_m: float, noise_std_deg: float
) -> Transform:
    noise_rotation = Rotation.from_rotvec(rng.normal(0, np.radians(noise_std_deg), 3)).as_matrix()
    noise = Transform.from_rotation_translation(noise_rotation, rng.normal(0, noise_std_m, 3))
    return transform @ noise


def _rotation_angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


@pytest.mark.parametrize("solver", _SOLVERS)
def test_solver_recovers_ground_truth_noiseless(solver: HandEyeSolver) -> None:
    rng = np.random.default_rng(42)
    true_camera_pose_in_base = _random_transform(rng, translation_scale=1.0)
    true_marker_pose_in_gripper = _random_transform(rng, translation_scale=0.05)
    samples = _simulate_samples(rng, true_camera_pose_in_base, true_marker_pose_in_gripper)

    result = solver.solve(samples)

    np.testing.assert_allclose(
        result.camera_pose_in_base.matrix, true_camera_pose_in_base.matrix, atol=1e-6
    )
    np.testing.assert_allclose(
        result.marker_pose_in_gripper.matrix, true_marker_pose_in_gripper.matrix, atol=1e-6
    )
    assert result.translation_residual_rmse_m == pytest.approx(0.0, abs=1e-6)
    assert result.rotation_residual_rmse_deg == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("solver", _SOLVERS)
def test_solver_is_robust_to_small_noise(solver: HandEyeSolver) -> None:
    rng = np.random.default_rng(7)
    true_camera_pose_in_base = _random_transform(rng, translation_scale=1.0)
    true_marker_pose_in_gripper = _random_transform(rng, translation_scale=0.05)
    samples = _simulate_samples(
        rng,
        true_camera_pose_in_base,
        true_marker_pose_in_gripper,
        noise_std_m=0.0005,
        noise_std_deg=0.1,
    )

    result = solver.solve(samples)

    np.testing.assert_allclose(
        result.camera_pose_in_base.translation, true_camera_pose_in_base.translation, atol=0.01
    )
    angle_error_deg = _rotation_angle_error_deg(
        result.camera_pose_in_base.rotation, true_camera_pose_in_base.rotation
    )
    assert angle_error_deg < 1.0
    assert result.translation_residual_rmse_m < 0.005
    assert result.rotation_residual_rmse_deg < 1.0
