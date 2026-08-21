"""Verification viser app: overlays the live RealSense point cloud (transformed into the
robot's base frame via a solved CalibrationResult) on the live URDF mesh. If calibration is
accurate, the point cloud should form a "skin" around the robot mesh.

Includes manual-correction buttons (Manual Correction panel) that apply a candidate fix on top
of the loaded result and update the point cloud live, so a suspected sign/axis bug can be
confirmed or ruled out by eye — does any candidate actually make the cloud hug the mesh? — before
touching any code. This is a diagnostic aid, not a replacement for fixing the real cause.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from robot_arm_camera_calibration.apps._viser_helpers import load_urdf, transform_to_wxyz_position
from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.base import RobotArm

_TICK_HZ = 15.0
_POINT_COLOR = (80, 170, 255)


def _flip_about_base_axis(axis: str) -> Transform:
    rotation = Rotation.from_euler(axis, 180, degrees=True).as_matrix()
    return Transform.from_rotation_translation(rotation, np.zeros(3))


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
        self._camera_pose_in_base = result.camera_pose_in_base
        self._correction_label = "none"

        self._server = viser.ViserServer(port=port)
        self._server.scene.world_axes.visible = True
        self._viser_urdf = ViserUrdf(self._server, load_urdf(urdf_description))
        self._camera_frustum = self._server.scene.add_camera_frustum(
            "/camera", fov=0.8, aspect=4 / 3, scale=0.08
        )
        self._point_cloud = self._server.scene.add_point_cloud(
            "/camera_point_cloud",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=_POINT_COLOR,
            point_size=0.003,
        )
        self._build_gui()

        self._robot.connect()
        self._camera.connect()

    def _build_gui(self) -> None:
        gui = self._server.gui
        with gui.add_folder("Manual Correction"):
            self._correction_markdown = gui.add_markdown(self._correction_status_text())
            gui.add_button("Flip 180° about base X").on_click(
                lambda _: self._apply_correction("flip X", _flip_about_base_axis("x"))
            )
            gui.add_button("Flip 180° about base Y").on_click(
                lambda _: self._apply_correction("flip Y", _flip_about_base_axis("y"))
            )
            gui.add_button("Flip 180° about base Z").on_click(
                lambda _: self._apply_correction("flip Z", _flip_about_base_axis("z"))
            )
            gui.add_button("Negate translation").on_click(lambda _: self._negate_translation())
            gui.add_button("Use inverse").on_click(lambda _: self._use_inverse())
            gui.add_button("Reset to solved result").on_click(lambda _: self._reset_correction())

    def _apply_correction(self, label: str, correction: Transform) -> None:
        self._camera_pose_in_base = correction @ self._camera_pose_in_base
        self._correction_label = f"{self._correction_label} + {label}"
        self._correction_markdown.content = self._correction_status_text()

    def _negate_translation(self) -> None:
        self._camera_pose_in_base = Transform.from_rotation_translation(
            self._camera_pose_in_base.rotation, -self._camera_pose_in_base.translation
        )
        self._correction_label = f"{self._correction_label} + negate translation"
        self._correction_markdown.content = self._correction_status_text()

    def _use_inverse(self) -> None:
        self._camera_pose_in_base = self._camera_pose_in_base.inverse()
        self._correction_label = f"{self._correction_label} + inverse"
        self._correction_markdown.content = self._correction_status_text()

    def _reset_correction(self) -> None:
        self._camera_pose_in_base = self._result.camera_pose_in_base
        self._correction_label = "none"
        self._correction_markdown.content = self._correction_status_text()

    def _correction_status_text(self) -> str:
        x, y, z = self._camera_pose_in_base.translation
        return (
            f"**Correction:** {self._correction_label}  \n"
            f"**camera_pose_in_base translation:** ({x:.3f}, {y:.3f}, {z:.3f})"
        )

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
        wxyz, position = transform_to_wxyz_position(self._camera_pose_in_base)
        self._camera_frustum.wxyz = wxyz
        self._camera_frustum.position = position

        points_in_camera = self._camera.get_point_cloud()
        if points_in_camera.size == 0:
            return
        points_in_base = self._camera_pose_in_base.apply(points_in_camera)
        self._point_cloud.points = points_in_base.astype(np.float32)
