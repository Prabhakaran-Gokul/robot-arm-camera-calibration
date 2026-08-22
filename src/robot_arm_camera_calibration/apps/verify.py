"""Verification viser app: overlays the live RealSense point cloud (transformed into the
robot's base frame via a solved CalibrationResult, colored with the camera's own RGB image) on
the live URDF mesh. If calibration is accurate, the point cloud should form a "skin" around the
robot mesh.

The camera's placement is driven by a draggable viser transform-controls gizmo ("/camera_gizmo"),
not a fixed value: it starts at the solved result but can be translated/rotated by hand in the
browser to correct small residual error by eye, with the point cloud following live as you drag.
The camera frustum is parented under the gizmo ("/camera_gizmo/frustum") so it rides along for
free via viser's scene graph, without needing to be repositioned manually each tick.

The Manual Correction buttons apply a candidate fix (e.g. a 180-degree flip about a base axis) on
top of the gizmo's current pose — useful for confirming or ruling out a suspected systematic bug
by eye before touching any code, as opposed to freeform dragging for fine adjustment.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from robot_arm_camera_calibration.apps._viser_helpers import (
    load_urdf,
    transform_to_wxyz_position,
    wxyz_position_to_transform,
)
from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.base import RobotArm

_TICK_HZ = 15.0
_GIZMO_SCALE = 0.2


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
        result_path: Path | None = None,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._result = result
        self._result_path = result_path

        self._server = viser.ViserServer(port=port)
        self._server.scene.world_axes.visible = True
        self._viser_urdf = ViserUrdf(self._server, load_urdf(urdf_description))

        wxyz, position = transform_to_wxyz_position(result.camera_pose_in_base)
        self._camera_gizmo = self._server.scene.add_transform_controls(
            "/camera_gizmo", scale=_GIZMO_SCALE, wxyz=wxyz, position=position
        )
        self._server.scene.add_camera_frustum("/camera_gizmo/frustum", fov=0.8, aspect=4 / 3)
        self._camera_gizmo.on_update(lambda _: self._refresh_status())

        self._point_cloud = self._server.scene.add_point_cloud(
            "/camera_point_cloud",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.003,
        )
        self._build_gui()
        self._refresh_status()

        self._robot.connect()
        self._camera.connect()

    def _build_gui(self) -> None:
        gui = self._server.gui
        with gui.add_folder("Manual Correction"):
            self._status_markdown = gui.add_markdown("")
            gui.add_button("Flip 180° about base X").on_click(
                lambda _: self._apply_correction(_flip_about_base_axis("x"))
            )
            gui.add_button("Flip 180° about base Y").on_click(
                lambda _: self._apply_correction(_flip_about_base_axis("y"))
            )
            gui.add_button("Flip 180° about base Z").on_click(
                lambda _: self._apply_correction(_flip_about_base_axis("z"))
            )
            gui.add_button("Negate translation").on_click(lambda _: self._negate_translation())
            gui.add_button("Use inverse").on_click(lambda _: self._use_inverse())
            gui.add_button("Reset to solved result").on_click(lambda _: self._reset_correction())
            gui.add_button("Save adjusted result").on_click(lambda _: self._save_adjusted())
            self._save_status_markdown = gui.add_markdown("")

    def _current_camera_pose_in_base(self) -> Transform:
        return wxyz_position_to_transform(
            np.asarray(self._camera_gizmo.wxyz), np.asarray(self._camera_gizmo.position)
        )

    def _set_camera_pose_in_base(self, pose: Transform) -> None:
        wxyz, position = transform_to_wxyz_position(pose)
        self._camera_gizmo.wxyz = wxyz
        self._camera_gizmo.position = position
        self._refresh_status()

    def _apply_correction(self, correction: Transform) -> None:
        self._set_camera_pose_in_base(correction @ self._current_camera_pose_in_base())

    def _negate_translation(self) -> None:
        current = self._current_camera_pose_in_base()
        self._set_camera_pose_in_base(
            Transform.from_rotation_translation(current.rotation, -current.translation)
        )

    def _use_inverse(self) -> None:
        self._set_camera_pose_in_base(self._current_camera_pose_in_base().inverse())

    def _reset_correction(self) -> None:
        self._set_camera_pose_in_base(self._result.camera_pose_in_base)

    def _refresh_status(self) -> None:
        x, y, z = self._current_camera_pose_in_base().translation
        self._status_markdown.content = (
            f"**camera_pose_in_base translation:** ({x:.3f}, {y:.3f}, {z:.3f})  \n"
            "Drag the gizmo in the scene, or use a button below."
        )

    def _save_adjusted(self) -> None:
        adjusted = CalibrationResult(
            camera_pose_in_base=self._current_camera_pose_in_base(),
            marker_pose_in_gripper=self._result.marker_pose_in_gripper,
            method=f"{self._result.method} (manually adjusted)",
            num_samples=self._result.num_samples,
            created_at=datetime.now(UTC),
            translation_residual_rmse_m=self._result.translation_residual_rmse_m,
            rotation_residual_rmse_deg=self._result.rotation_residual_rmse_deg,
            metadata={
                **self._result.metadata,
                "manually_adjusted": True,
                "note": "residual RMSE above is from the original solve, not this adjusted pose",
            },
        )
        if self._result_path is not None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            path = self._result_path.parent / f"{self._result_path.stem}_adjusted_{timestamp}.yaml"
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            path = Path("results") / f"calibration_adjusted_{timestamp}.yaml"
        adjusted.to_yaml(path)
        self._save_status_markdown.content = f"Saved to `{path}`"

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

        points_in_camera, colors = self._camera.get_point_cloud()
        if points_in_camera.size == 0:
            return
        points_in_base = self._current_camera_pose_in_base().apply(points_in_camera)
        self._point_cloud.points = points_in_base.astype(np.float32)
        self._point_cloud.colors = colors
