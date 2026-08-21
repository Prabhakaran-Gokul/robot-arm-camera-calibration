from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: npt.NDArray[np.float64]

    @property
    def camera_matrix(self) -> npt.NDArray[np.float64]:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


class Camera(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics: ...

    @abstractmethod
    def get_color_frame(self) -> npt.NDArray[np.uint8]:
        """HxWx3, BGR."""

    @abstractmethod
    def get_point_cloud(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        """(points, colors): points are Nx3 meters in camera frame; colors are Nx3 RGB uint8,
        one per point, in the same order."""

    def __enter__(self) -> Camera:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()
