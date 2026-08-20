"""Verification viser app: overlays the live RealSense point cloud (transformed into the
robot's base frame via a solved CalibrationResult) on the live URDF mesh. If calibration is
accurate, the point cloud should form a "skin" around the robot mesh."""

from __future__ import annotations

import threading
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from robot_arm_camera_calibration.apps._viser_helpers import load_urdf
from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.robots.base import RobotArm

_TICK_HZ = 15.0
_POINT_COLOR = (80, 170, 255)


class VerifyApp:
    def __init__(
        self,
        robot: RobotArm,
        camera: Camera,
        result: CalibrationResult,
        urdf_description: str = "ur5e_description",
        port: int = 8080,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._result = result

        self._server = viser.ViserServer(port=port)
        self._server.scene.world_axes.visible = True
        self._viser_urdf = ViserUrdf(self._server, load_urdf(urdf_description))
        self._point_cloud = self._server.scene.add_point_cloud(
            "/camera_point_cloud",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=_POINT_COLOR,
            point_size=0.003,
        )

        self._robot.connect()
        self._camera.connect()

    def run(self) -> None:
        threading.Thread(target=self._tick_loop, daemon=True).start()
        self._server.sleep_forever()

    def _tick_loop(self) -> None:
        # See CollectionApp._tick_loop: this polls live hardware indefinitely in the
        # background, so one bad frame must not silently kill live updates for the session.
        while True:
            try:
                self._tick()
            except Exception as error:
                print(f"[verify] tick failed, skipping this frame: {error}")
            time.sleep(1.0 / _TICK_HZ)

    def _tick(self) -> None:
        self._viser_urdf.update_cfg(self._robot.get_joint_positions())
        points_in_camera = self._camera.get_point_cloud()
        if points_in_camera.size == 0:
            return
        points_in_base = self._result.camera_pose_in_base.apply(points_in_camera)
        self._point_cloud.points = points_in_base.astype(np.float32)
