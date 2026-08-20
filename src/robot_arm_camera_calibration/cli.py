from __future__ import annotations

import argparse
from pathlib import Path

from robot_arm_camera_calibration.cameras.base import Camera
from robot_arm_camera_calibration.cameras.mock import MockCamera
from robot_arm_camera_calibration.core.config import CalibrationConfig
from robot_arm_camera_calibration.core.results import CalibrationResult
from robot_arm_camera_calibration.robots.base import RobotArm
from robot_arm_camera_calibration.robots.mock import MockRobotArm
from robot_arm_camera_calibration.solvers.base import HandEyeSolver
from robot_arm_camera_calibration.solvers.opencv_hand_eye import OpenCVHandEyeSolver
from robot_arm_camera_calibration.solvers.opencv_robot_world import OpenCVRobotWorldHandEyeSolver
from robot_arm_camera_calibration.targets.factory import build_target

_SOLVERS = {
    "robot-world": OpenCVRobotWorldHandEyeSolver,
    "hand-eye-park": lambda: OpenCVHandEyeSolver(),
    "hand-eye-tsai": lambda: OpenCVHandEyeSolver(method=0),
}


def _build_solver(name: str) -> HandEyeSolver:
    return _SOLVERS[name]()


def _build_robot_and_camera(
    config: CalibrationConfig, mock: bool, control_enabled: bool = True
) -> tuple[RobotArm, Camera]:
    if mock:
        return MockRobotArm(), MockCamera()

    from robot_arm_camera_calibration.cameras.realsense import RealSenseCamera
    from robot_arm_camera_calibration.robots.ur5e import UR5eArm

    return UR5eArm(config.robot_ip, control_enabled=control_enabled), RealSenseCamera()


def _collect(args: argparse.Namespace) -> None:
    from robot_arm_camera_calibration.apps.collect import CollectionApp

    config = CalibrationConfig.from_yaml(Path(args.config))
    robot, camera = _build_robot_and_camera(
        config, args.mock, control_enabled=not args.teach_pendant
    )
    target = build_target(config.target)
    CollectionApp(
        config,
        robot,
        camera,
        target,
        _build_solver(args.solver),
        port=args.port,
        jog_enabled=not args.teach_pendant,
    ).run()


def _verify(args: argparse.Namespace) -> None:
    from robot_arm_camera_calibration.apps.verify import VerifyApp

    config = CalibrationConfig.from_yaml(Path(args.config))
    # VerifyApp only reads robot state, never commands motion, so it never needs control.
    robot, camera = _build_robot_and_camera(config, args.mock, control_enabled=False)
    result = CalibrationResult.from_yaml(Path(args.result))
    VerifyApp(robot, camera, result, port=args.port).run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="robot-calib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Run the calibration collection app")
    collect_parser.add_argument("--config", required=True, help="Path to a CalibrationConfig YAML")
    collect_parser.add_argument("--mock", action="store_true", help="Use in-memory mock hardware")
    collect_parser.add_argument(
        "--teach-pendant",
        action="store_true",
        help="Drive the robot from the teach pendant instead of viser jog controls; "
        "the app only reads poses to capture samples and never commands motion",
    )
    collect_parser.add_argument(
        "--solver",
        choices=sorted(_SOLVERS),
        default="robot-world",
        help="Hand-eye solver: robot-world (default, calibrateRobotWorldHandEye) solves the "
        "camera extrinsic and EE-to-marker offset jointly; hand-eye-park/hand-eye-tsai "
        "(calibrateHandEye) solve the extrinsic from relative motions, then recover the "
        "EE-to-marker offset as a separate closed-form step",
    )
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
