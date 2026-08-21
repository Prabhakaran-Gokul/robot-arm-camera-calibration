from __future__ import annotations

import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.cameras.base import Camera, CameraIntrinsics


class MockCamera(Camera):
    """In-memory camera for offline development and tests. Returns a fixed/settable frame."""

    def __init__(self, intrinsics: CameraIntrinsics | None = None) -> None:
        self._connected = False
        self._intrinsics = intrinsics or CameraIntrinsics(
            width=640, height=480, fx=800.0, fy=800.0, cx=320.0, cy=240.0, dist_coeffs=np.zeros(5)
        )
        self._color_frame: npt.NDArray[np.uint8] = np.full(
            (self._intrinsics.height, self._intrinsics.width, 3), 255, dtype=np.uint8
        )
        self._point_cloud: npt.NDArray[np.float64] = np.zeros((0, 3))
        self._point_cloud_colors: npt.NDArray[np.uint8] = np.zeros((0, 3), dtype=np.uint8)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def get_color_frame(self) -> npt.NDArray[np.uint8]:
        return self._color_frame

    def get_point_cloud(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        return self._point_cloud, self._point_cloud_colors

    def set_color_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        self._color_frame = frame

    def set_point_cloud(
        self, points: npt.NDArray[np.float64], colors: npt.NDArray[np.uint8] | None = None
    ) -> None:
        self._point_cloud = points
        self._point_cloud_colors = (
            colors if colors is not None else np.full((len(points), 3), 200, dtype=np.uint8)
        )
