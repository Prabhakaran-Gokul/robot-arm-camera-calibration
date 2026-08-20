from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.core.config import CalibrationConfig
from robot_arm_camera_calibration.core.errors import TargetNotDetectedError
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.core.samples import CalibrationSample
from robot_arm_camera_calibration.robots.base import RobotArm
from robot_arm_camera_calibration.solvers.base import HandEyeSolver
from robot_arm_camera_calibration.targets.base import CalibrationTarget


class CalibrationSession:
    def __init__(
        self,
        robot: RobotArm,
        camera: Camera,
        target: CalibrationTarget,
        solver: HandEyeSolver,
        config: CalibrationConfig,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._target = target
        self._solver = solver
        self._config = config
        self._samples: list[CalibrationSample] = []

    @property
    def samples(self) -> list[CalibrationSample]:
        return list(self._samples)

    def capture_sample(self) -> CalibrationSample:
        gripper_pose_in_base = self._robot.get_tcp_pose()
        image = self._camera.get_color_frame()
        detection = self._target.detect(image, self._camera.get_intrinsics())
        if detection is None:
            raise TargetNotDetectedError("Calibration target is not visible in the current frame")

        sample = CalibrationSample(
            index=len(self._samples),
            timestamp=datetime.now(UTC),
            gripper_pose_in_base=gripper_pose_in_base,
            target_pose_in_camera=detection.pose_in_camera,
            joint_positions=self._robot.get_joint_positions(),
        )
        self._samples.append(sample)
        return sample

    def remove_sample(self, index: int) -> None:
        self._samples = [s for s in self._samples if s.index != index]

    def clear_samples(self) -> None:
        self._samples.clear()

    def run_calibration(self) -> CalibrationResult:
        return self._solver.solve(self._samples)

    def save_result(self, result: CalibrationResult, path: Path | None = None) -> Path:
        if path is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            path = self._config.results_dir / f"calibration_{timestamp}.yaml"
        result.to_yaml(path)
        return path
