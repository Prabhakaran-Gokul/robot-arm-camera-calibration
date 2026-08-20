import numpy as np
import pytest

from robot_arm_camera_calibration.core.config import JogLimits, WorkspaceLimits
from robot_arm_camera_calibration.core.errors import JogLimitViolationError
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.jog import JogController
from robot_arm_camera_calibration.robots.mock import MockRobotArm


def _controller(**jog_limit_overrides: float) -> tuple[JogController, MockRobotArm]:
    robot = MockRobotArm(initial_pose=Transform.identity())
    robot.connect()
    workspace = WorkspaceLimits((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0))
    jog_limits = JogLimits(**jog_limit_overrides) if jog_limit_overrides else JogLimits()
    return JogController(robot, workspace, jog_limits), robot


def test_nudge_moves_within_bounds() -> None:
    controller, robot = _controller(max_step_m=0.05)
    target = controller.nudge(translation_delta_m=np.array([0.02, 0.0, 0.0]))
    np.testing.assert_allclose(target.translation, [0.02, 0.0, 0.0])
    np.testing.assert_allclose(robot.get_tcp_pose().translation, [0.02, 0.0, 0.0])


def test_nudge_clamps_oversized_step() -> None:
    controller, _robot = _controller(max_step_m=0.01)
    target = controller.nudge(translation_delta_m=np.array([1.0, 0.0, 0.0]))
    assert np.linalg.norm(target.translation) == pytest.approx(0.01)


def test_nudge_rejects_motion_outside_workspace() -> None:
    controller, robot = _controller(max_step_m=1.0)
    with pytest.raises(JogLimitViolationError):
        controller.nudge(translation_delta_m=np.array([10.0, 0.0, 0.0]))
    np.testing.assert_allclose(robot.get_tcp_pose().translation, [0.0, 0.0, 0.0])


def test_nudge_rotation_does_not_move_position() -> None:
    controller, _robot = _controller()
    target = controller.nudge(rotation_delta_deg=np.array([0.0, 0.0, 5.0]))
    np.testing.assert_allclose(target.translation, [0.0, 0.0, 0.0], atol=1e-10)
    assert not np.allclose(target.rotation, np.eye(3))
