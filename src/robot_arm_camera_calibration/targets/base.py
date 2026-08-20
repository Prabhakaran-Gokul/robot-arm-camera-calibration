from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.cameras.base import CameraIntrinsics
from robot_arm_camera_calibration.core.transform import Transform


@dataclass(frozen=True, slots=True)
class TargetDetection:
    pose_in_camera: Transform
    reprojection_rms_px: float
    annotated_image: npt.NDArray[np.uint8] | None = None


class CalibrationTarget(ABC):
    @abstractmethod
    def detect(
        self,
        image: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics,
        *,
        annotate: bool = False,
    ) -> TargetDetection | None:
        """Returns None (not an exception) when the target isn't visible in `image`."""
