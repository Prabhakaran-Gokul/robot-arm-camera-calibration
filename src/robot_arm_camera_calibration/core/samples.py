from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.core.transform import Transform


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    index: int
    timestamp: datetime
    gripper_pose_in_base: Transform
    target_pose_in_camera: Transform
    joint_positions: npt.NDArray[np.float64] | None = None
