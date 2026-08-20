from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.core.transform import Transform


class RobotArm(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_tcp_pose(self) -> Transform:
        """Current gripper pose expressed in the base frame."""

    @abstractmethod
    def get_joint_positions(self) -> npt.NDArray[np.float64]:
        """Radians, in URDF joint order."""

    @abstractmethod
    def servo_to_pose(self, target: Transform, *, speed: float, acceleration: float) -> None:
        """Non-blocking: updates the streaming setpoint. Does not wait for arrival."""

    @abstractmethod
    def move_to_joint_positions(
        self, joint_positions: npt.NDArray[np.float64], *, speed: float, acceleration: float
    ) -> None:
        """Blocking point-to-point move."""

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self) -> RobotArm:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()
