"""Interactive viser app for eye-to-hand calibration: jog the robot, capture samples with
the target visible, run the solve, and save the result."""

from __future__ import annotations

import threading
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from robot_arm_camera_calibration.apps._viser_helpers import (
    load_urdf,
    rotation_diversity_deg,
    transform_to_wxyz_position,
)
from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.core.config import CalibrationConfig
from robot_arm_camera_calibration.core.errors import CalibrationError
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.robots.base import RobotArm
from robot_arm_camera_calibration.robots.jog import JogController
from robot_arm_camera_calibration.session import CalibrationSession
from robot_arm_camera_calibration.solvers.base import HandEyeSolver
from robot_arm_camera_calibration.targets.base import CalibrationTarget

_TICK_HZ = 15.0
_JOG_AXES = (
    ("X", np.array([1.0, 0.0, 0.0])),
    ("Y", np.array([0.0, 1.0, 0.0])),
    ("Z", np.array([0.0, 0.0, 1.0])),
)


class CollectionApp:
    def __init__(
        self,
        config: CalibrationConfig,
        robot: RobotArm,
        camera: Camera,
        target: CalibrationTarget,
        solver: HandEyeSolver,
        urdf_description: str = "ur5e_description",
        port: int = 8080,
    ) -> None:
        self._config = config
        self._robot = robot
        self._camera = camera
        self._target = target
        self._urdf_description = urdf_description
        self._session = CalibrationSession(robot, camera, target, solver, config)
        self._jog = JogController(robot, config.workspace_limits, config.jog_limits)
        self._last_result: CalibrationResult | None = None
        self._sample_row_folder: viser.GuiFolderHandle | None = None
        self._sample_row_buttons: dict[int, viser.GuiButtonHandle] = {}

        self._server = viser.ViserServer(port=port)
        self._viser_urdf: ViserUrdf | None = None
        self._target_frame = self._server.scene.add_frame(
            "/target", visible=False, axes_length=0.08
        )
        self._build_gui()

    def run(self) -> None:
        threading.Thread(target=self._tick_loop, daemon=True).start()
        self._server.sleep_forever()

    def _build_gui(self) -> None:
        gui = self._server.gui

        with gui.add_folder("Connection"):
            self._status_markdown = gui.add_markdown("**Status:** disconnected")
            connect_button = gui.add_button("Connect")
            self._disconnect_button = gui.add_button("Disconnect", disabled=True)
            connect_button.on_click(lambda _: self._on_connect())
            self._disconnect_button.on_click(lambda _: self._on_disconnect())

        with gui.add_folder("Jog"):
            self._step_m = gui.add_slider(
                "Step (m)",
                min=0.001,
                max=self._config.jog_limits.max_step_m,
                step=0.001,
                initial_value=self._config.jog_limits.max_step_m / 2,
            )
            self._step_deg = gui.add_slider(
                "Step (deg)",
                min=0.5,
                max=self._config.jog_limits.max_step_deg,
                step=0.5,
                initial_value=self._config.jog_limits.max_step_deg / 2,
            )
            for axis_name, axis in _JOG_AXES:
                for sign, label in ((1.0, f"+{axis_name}"), (-1.0, f"-{axis_name}")):
                    gui.add_button(f"Move {label}").on_click(
                        lambda _, a=axis, s=sign: self._on_translate(s * a)
                    )
            for axis_name, axis in _JOG_AXES:
                for sign, label in ((1.0, f"+R{axis_name}"), (-1.0, f"-R{axis_name}")):
                    gui.add_button(f"Rotate {label}").on_click(
                        lambda _, a=axis, s=sign: self._on_rotate(s * a)
                    )
            self._jog_status_markdown = gui.add_markdown("")

        with gui.add_folder("Samples"):
            gui.add_button("Capture Sample").on_click(lambda _: self._on_capture())
            gui.add_button("Clear All").on_click(lambda _: self._on_clear_samples())
            self._sample_row_folder = gui.add_folder("Captured", expand_by_default=False)
            self._samples_markdown = gui.add_markdown("No samples captured yet.")

        with gui.add_folder("Calibrate"):
            gui.add_button("Run Calibration").on_click(lambda _: self._on_calibrate())
            self._result_markdown = gui.add_markdown("")
            gui.add_button("Save Result").on_click(lambda _: self._on_save())

    def _on_connect(self) -> None:
        self._robot.connect()
        self._camera.connect()
        self._viser_urdf = ViserUrdf(self._server, load_urdf(self._urdf_description))
        self._status_markdown.content = "**Status:** connected"
        self._disconnect_button.disabled = False

    def _on_disconnect(self) -> None:
        self._robot.disconnect()
        self._camera.disconnect()
        self._status_markdown.content = "**Status:** disconnected"
        self._disconnect_button.disabled = True

    def _on_translate(self, direction: np.ndarray) -> None:
        self._jog_nudge(translation_delta_m=direction * self._step_m.value)

    def _on_rotate(self, direction: np.ndarray) -> None:
        self._jog_nudge(rotation_delta_deg=direction * self._step_deg.value)

    def _jog_nudge(
        self,
        translation_delta_m: np.ndarray | None = None,
        rotation_delta_deg: np.ndarray | None = None,
    ) -> None:
        if not self._robot.is_connected:
            self._jog_status_markdown.content = "Connect to the robot before jogging."
            return
        try:
            self._jog.nudge(translation_delta_m, rotation_delta_deg)
            self._jog_status_markdown.content = ""
        except CalibrationError as error:
            self._jog_status_markdown.content = f"⚠️ {error}"

    def _on_capture(self) -> None:
        try:
            sample = self._session.capture_sample()
        except CalibrationError as error:
            self._samples_markdown.content = f"⚠️ Capture failed: {error}"
            return

        assert self._sample_row_folder is not None
        with self._sample_row_folder:
            x, y, z = sample.gripper_pose_in_base.translation
            row = self._server.gui.add_button(f"Delete #{sample.index} ({x:.3f}, {y:.3f}, {z:.3f})")
        row.on_click(lambda _, index=sample.index: self._on_delete_sample(index))
        self._sample_row_buttons[sample.index] = row
        self._refresh_samples_summary()

    def _on_delete_sample(self, index: int) -> None:
        self._session.remove_sample(index)
        self._sample_row_buttons.pop(index).remove()
        self._refresh_samples_summary()

    def _on_clear_samples(self) -> None:
        self._session.clear_samples()
        for row in self._sample_row_buttons.values():
            row.remove()
        self._sample_row_buttons.clear()
        self._refresh_samples_summary()

    def _refresh_samples_summary(self) -> None:
        samples = self._session.samples
        if not samples:
            self._samples_markdown.content = "No samples captured yet."
            return
        diversity = rotation_diversity_deg([s.gripper_pose_in_base.rotation for s in samples])
        diversity_note = (
            "⚠️ low — vary orientation, not just position" if diversity < 30.0 else "good"
        )
        self._samples_markdown.content = (
            f"**{len(samples)} samples** — rotational diversity: "
            f"{diversity:.0f}° ({diversity_note})"
        )

    def _on_calibrate(self) -> None:
        try:
            result = self._session.run_calibration()
        except CalibrationError as error:
            self._result_markdown.content = f"⚠️ {error}"
            return
        self._last_result = result
        self._result_markdown.content = (
            f"**Translation RMSE:** {result.translation_residual_rmse_m * 1000:.2f} mm  \n"
            f"**Rotation RMSE:** {result.rotation_residual_rmse_deg:.3f}°"
        )
        wxyz, position = transform_to_wxyz_position(result.camera_pose_in_base)
        self._server.scene.add_frame(
            "/solved_camera", wxyz=wxyz, position=position, axes_length=0.1
        )

    def _on_save(self) -> None:
        if self._last_result is None:
            self._result_markdown.content = "⚠️ Run calibration before saving."
            return
        path = self._session.save_result(self._last_result)
        self._result_markdown.content += f"  \nSaved to `{path}`"

    def _tick_loop(self) -> None:
        while True:
            self._tick()
            time.sleep(1.0 / _TICK_HZ)

    def _tick(self) -> None:
        if not self._robot.is_connected:
            return
        if self._viser_urdf is not None:
            self._viser_urdf.update_cfg(self._robot.get_joint_positions())

        if not self._camera.is_connected:
            return
        image = self._camera.get_color_frame()
        detection = self._target.detect(image, self._camera.get_intrinsics())
        if detection is None:
            self._target_frame.visible = False
            return
        wxyz, position = transform_to_wxyz_position(detection.pose_in_camera)
        self._target_frame.wxyz = wxyz
        self._target_frame.position = position
        self._target_frame.visible = True
