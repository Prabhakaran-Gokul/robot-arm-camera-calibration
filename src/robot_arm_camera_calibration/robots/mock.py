from __future__ import annotations

import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.base import RobotArm


class MockRobotArm(RobotArm):
    """In-memory robot for offline development and tests. Motions apply instantly."""

    def __init__(self, initial_pose: Transform | None = None, num_joints: int = 6) -> None:
        self._connected = False
        self._pose = initial_pose if initial_pose is not None else Transform.identity()
        self._joint_positions: npt.NDArray[np.float64] = np.zeros(num_joints)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_tcp_pose(self) -> Transform:
        return self._pose

    def get_joint_positions(self) -> npt.NDArray[np.float64]:
        return self._joint_positions.copy()

    def servo_to_pose(self, target: Transform, *, speed: float, acceleration: float) -> None:
        del speed, acceleration
        self._pose = target

    def move_to_joint_positions(
        self, joint_positions: npt.NDArray[np.float64], *, speed: float, acceleration: float
    ) -> None:
        del speed, acceleration
        self._joint_positions = np.asarray(joint_positions, dtype=np.float64)

    def stop(self) -> None:
        pass
