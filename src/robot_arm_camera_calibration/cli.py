from __future__ import annotations

import argparse
from pathlib import Path

from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.cameras.mock import MockCamera
from robot_arm_camera_calibration.core.config import CalibrationConfig
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.robots.base import RobotArm
from robot_arm_camera_calibration.robots.mock import MockRobotArm
from robot_arm_camera_calibration.solvers.opencv_robot_world import OpenCVRobotWorldHandEyeSolver
from robot_arm_camera_calibration.targets.factory import build_target


def _build_robot_and_camera(config: CalibrationConfig, mock: bool) -> tuple[RobotArm, Camera]:
    if mock:
        return MockRobotArm(), MockCamera()

    from robot_arm_camera_calibration.cameras.realsense import RealSenseCamera
    from robot_arm_camera_calibration.robots.ur5e import UR5eArm

    return UR5eArm(config.robot_ip), RealSenseCamera()


def _collect(args: argparse.Namespace) -> None:
    from robot_arm_camera_calibration.apps.collect import CollectionApp

    config = CalibrationConfig.from_yaml(Path(args.config))
    robot, camera = _build_robot_and_camera(config, args.mock)
    target = build_target(config.target)
    CollectionApp(
        config, robot, camera, target, OpenCVRobotWorldHandEyeSolver(), port=args.port
    ).run()


def _verify(args: argparse.Namespace) -> None:
    from robot_arm_camera_calibration.apps.verify import VerifyApp

    config = CalibrationConfig.from_yaml(Path(args.config))
    robot, camera = _build_robot_and_camera(config, args.mock)
    result = CalibrationResult.from_yaml(Path(args.result))
    VerifyApp(robot, camera, result, port=args.port).run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="robot-calib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Run the calibration collection app")
    collect_parser.add_argument("--config", required=True, help="Path to a CalibrationConfig YAML")
    collect_parser.add_argument("--mock", action="store_true", help="Use in-memory mock hardware")
    collect_parser.add_argument("--port", type=int, default=8080)
    collect_parser.set_defaults(func=_collect)

    verify_parser = subparsers.add_parser("verify", help="Run the calibration verification app")
    verify_parser.add_argument("--config", required=True, help="Path to a CalibrationConfig YAML")
    verify_parser.add_argument(
        "--result", required=True, help="Path to a saved CalibrationResult YAML"
    )
    verify_parser.add_argument("--mock", action="store_true", help="Use in-memory mock hardware")
    verify_parser.add_argument("--port", type=int, default=8080)
    verify_parser.set_defaults(func=_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
