# pyright: reportAttributeAccessIssue=false
# pyrealsense2 ships no type stubs at all, so its own attributes are untyped here; every other
# boundary in this file keeps normal type checking.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs

from robot_arm_camera_calibration.cameras.base import Camera, CameraIntrinsics


class RealSenseCamera(Camera):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_number: str | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._serial_number = serial_number
        self._pipeline: rs.pipeline | None = None
        self._align = rs.align(rs.stream.color)
        self._pointcloud = rs.pointcloud()
        self._intrinsics: CameraIntrinsics | None = None

    def connect(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        if self._serial_number is not None:
            config.enable_device(self._serial_number)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        profile = pipeline.start(config)

        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsics = CameraIntrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            dist_coeffs=np.array(intrinsics.coeffs, dtype=np.float64),
        )
        self._pipeline = pipeline

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._intrinsics = None

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def get_intrinsics(self) -> CameraIntrinsics:
        assert self._intrinsics is not None, "Camera is not connected"
        return self._intrinsics

    def get_color_frame(self) -> npt.NDArray[np.uint8]:
        frames = self._wait_for_aligned_frames()
        return np.asanyarray(frames.get_color_frame().get_data())

    def get_point_cloud(self) -> npt.NDArray[np.float64]:
        frames = self._wait_for_aligned_frames()
        depth_frame = frames.get_depth_frame()
        self._pointcloud.map_to(frames.get_color_frame())
        points = self._pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        return vertices[vertices[:, 2] > 0].astype(np.float64)

    def _wait_for_aligned_frames(self) -> rs.composite_frame:
        assert self._pipeline is not None, "Camera is not connected"
        return self._align.process(self._pipeline.wait_for_frames())
