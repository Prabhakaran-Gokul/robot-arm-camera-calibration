from abc import ABC, abstractmethod
from collections.abc import Sequence

from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.samples import CalibrationSample


class HandEyeSolver(ABC):
    @abstractmethod
    def solve(self, samples: Sequence[CalibrationSample]) -> CalibrationResult: ...
