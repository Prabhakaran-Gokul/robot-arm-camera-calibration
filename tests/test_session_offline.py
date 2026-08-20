"""Fully offline integration test: MockRobotArm + MockCamera + a test-only analytic target
that computes the marker pose directly from ground truth (instead of image processing, which
is already covered by test_targets_synthetic.py) to exercise the CalibrationSession wiring."""

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.cameras.base import CameraIntrinsics
from robot_arm_camera_calibration.cameras.mock import MockCamera
from robot_arm_camera_calibration.core.config import ArucoMarkerConfig, CalibrationConfig
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.mock import MockRobotArm
from robot_arm_camera_calibration.session import CalibrationSession
from robot_arm_camera_calibration.solvers.opencv_robot_world import OpenCVRobotWorldHandEyeSolver
from robot_arm_camera_calibration.targets.base import CalibrationTarget, TargetDetection


class _AnalyticTarget(CalibrationTarget):
    def __init__(
        self, robot: MockRobotArm, camera_pose_in_base: Transform, marker_pose_in_gripper: Transform
    ) -> None:
        self._robot = robot
        self._base_pose_in_camera = camera_pose_in_base.inverse()
        self._marker_pose_in_gripper = marker_pose_in_gripper

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics,
        *,
        annotate: bool = False,
    ) -> TargetDetection | None:
        del image, intrinsics, annotate
        gripper_pose_in_base = self._robot.get_tcp_pose()
        pose = self._base_pose_in_camera @ gripper_pose_in_base @ self._marker_pose_in_gripper
        return TargetDetection(pose_in_camera=pose, reprojection_rms_px=0.0)


def test_offline_session_recovers_ground_truth(tmp_path) -> None:
    rng = np.random.default_rng(3)
    true_camera_pose_in_base = Transform.from_rotation_translation(
        Rotation.random(rng=rng).as_matrix(), rng.uniform(-1, 1, 3)
    )
    true_marker_pose_in_gripper = Transform.from_rotation_translation(
        Rotation.random(rng=rng).as_matrix(), rng.uniform(-0.05, 0.05, 3)
    )

    robot = MockRobotArm()
    robot.connect()
    camera = MockCamera()
    camera.connect()
    target = _AnalyticTarget(robot, true_camera_pose_in_base, true_marker_pose_in_gripper)
    solver = OpenCVRobotWorldHandEyeSolver()
    config = CalibrationConfig(
        robot_ip="mock", target=ArucoMarkerConfig(), results_dir=tmp_path / "results"
    )
    session = CalibrationSession(robot, camera, target, solver, config)

    for _ in range(15):
        pose = Transform.from_rotation_translation(
            Rotation.random(rng=rng).as_matrix(), rng.uniform(-0.3, 0.3, 3)
        )
        robot.servo_to_pose(pose, speed=0.1, acceleration=0.5)
        session.capture_sample()

    assert len(session.samples) == 15

    result = session.run_calibration()
    np.testing.assert_allclose(
        result.camera_pose_in_base.matrix, true_camera_pose_in_base.matrix, atol=1e-6
    )
    np.testing.assert_allclose(
        result.marker_pose_in_gripper.matrix, true_marker_pose_in_gripper.matrix, atol=1e-6
    )

    saved_path = session.save_result(result)
    assert saved_path.exists()


def test_remove_and_clear_samples() -> None:
    robot = MockRobotArm()
    robot.connect()
    camera = MockCamera()
    camera.connect()
    target = _AnalyticTarget(robot, Transform.identity(), Transform.identity())
    session = CalibrationSession(
        robot,
        camera,
        target,
        OpenCVRobotWorldHandEyeSolver(),
        CalibrationConfig(robot_ip="mock", target=ArucoMarkerConfig()),
    )

    session.capture_sample()
    session.capture_sample()
    assert len(session.samples) == 2

    session.remove_sample(0)
    assert len(session.samples) == 1

    session.clear_samples()
    assert len(session.samples) == 0
