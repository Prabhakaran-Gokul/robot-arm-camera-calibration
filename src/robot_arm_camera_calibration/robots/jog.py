from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.config import JogLimits, WorkspaceLimits
from robot_arm_camera_calibration.core.errors import JogLimitViolationError
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.base import RobotArm
from robot_arm_camera_calibration.robots.safety import clamp_step

_ZERO3 = np.zeros(3)


class JogController:
    """Cartesian jog: translation and rotation deltas are both expressed in the base frame,
    with rotation applied about the TCP's current position (not the base origin)."""

    def __init__(
        self, robot: RobotArm, workspace_limits: WorkspaceLimits, jog_limits: JogLimits
    ) -> None:
        self._robot = robot
        self._workspace_limits = workspace_limits
        self._jog_limits = jog_limits

    def nudge(
        self,
        translation_delta_m: npt.NDArray[np.float64] | None = None,
        rotation_delta_deg: npt.NDArray[np.float64] | None = None,
    ) -> Transform:
        translation_delta = clamp_step(
            np.asarray(
                translation_delta_m if translation_delta_m is not None else _ZERO3, dtype=float
            ),
            self._jog_limits.max_step_m,
        )
        rotation_delta = clamp_step(
            np.asarray(
                rotation_delta_deg if rotation_delta_deg is not None else _ZERO3, dtype=float
            ),
            self._jog_limits.max_step_deg,
        )

        current = self._robot.get_tcp_pose()
        new_translation = current.translation + translation_delta
        new_translation_tuple = (
            float(new_translation[0]),
            float(new_translation[1]),
            float(new_translation[2]),
        )
        if not self._workspace_limits.contains(new_translation_tuple):
            raise JogLimitViolationError(
                f"Jog target {new_translation_tuple} is outside workspace bounds "
                f"{self._workspace_limits.min_xyz_m}..{self._workspace_limits.max_xyz_m}"
            )

        delta_rotation = Rotation.from_euler("xyz", rotation_delta, degrees=True).as_matrix()
        target = Transform.from_rotation_translation(
            delta_rotation @ current.rotation, new_translation
        )
        self._robot.servo_to_pose(
            target,
            speed=self._jog_limits.speed_m_s,
            acceleration=self._jog_limits.acceleration_m_s2,
        )
        return target
