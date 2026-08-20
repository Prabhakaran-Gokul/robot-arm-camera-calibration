# robot-arm-camera-calibration

ROS-independent Python library for robot–camera extrinsic calibration. Ships with UR5e
(`ur_rtde`) and Intel RealSense (`pyrealsense2`) support, and an **eye-to-hand** workflow: robot
base and camera are both fixed, an ArUco marker or ChArUco board is mounted on the end effector,
and the unknown end-effector-to-marker offset is solved jointly with the base-to-camera extrinsic.

Two solver families are available (`--solver`, see `robot-calib collect --help`):
- `robot-world` (default): `cv2.calibrateRobotWorldHandEye`, solves both unknowns jointly from
  absolute per-sample geometry.
- `hand-eye-park` / `hand-eye-tsai`: `cv2.calibrateHandEye` (the classic Tsai/Park family used by
  ROS's `easy_handeye`), solves the camera extrinsic from relative motion between sample pairs,
  then recovers the EE-to-marker offset as a separate closed-form step. In practice `robot-world`
  is slightly more accurate for this rig in our own synthetic benchmarks, with `hand-eye-tsai`
  noticeably behind `hand-eye-park` — Tsai's reputation for accuracy comes from setups that don't
  need the joint EE-marker solve at all. Neither family fixes samples with poor rotational axis
  coverage; that's a data problem, not a solver problem (see the app's axis-coverage warning).

Architecture is built around small ABCs (`RobotArm`, `Camera`, `CalibrationTarget`,
`HandEyeSolver`) so new arms, cameras, calibration targets, or solvers (e.g. eye-in-hand) can be
added without touching existing code.

## Install

```bash
uv sync --extra dev            # core lib + dev tools (ruff, pyright, pytest), no hardware SDKs
uv sync --extra dev --extra hardware   # + ur_rtde and pyrealsense2 for real hardware
```

## Usage

```bash
# Interactive calibration collection, against mock hardware (no robot/camera needed)
uv run robot-calib collect --config configs/example_ur5e_realsense.yaml --mock

# Against a real UR5e + RealSense
uv run robot-calib collect --config configs/example_ur5e_realsense.yaml

# Verify a saved calibration by overlaying the live point cloud on the robot mesh
uv run robot-calib verify --config configs/example_ur5e_realsense.yaml --result results/calibration_....yaml
```

Both commands open a [viser](https://viser.studio) UI at `http://localhost:8080`.

## Development

```bash
uv run pytest       # all tests run offline against synthetic data and mock hardware
uv run ruff check .
uv run ruff format .
uv run pyright
```

`tests/test_solver_synthetic.py` is the load-bearing test: for each solver, it proves the
OpenCV input/output wiring is correct by forward-simulating samples from a known ground truth and
checking the solver recovers it exactly — see `solvers/opencv_robot_world.py` and
`solvers/opencv_hand_eye.py` for the derivations.

## Layout

```
src/robot_arm_camera_calibration/
├── core/       # Transform (SE3), CalibrationSample/Result/Config, errors
├── robots/     # RobotArm ABC, UR5eArm, MockRobotArm, jog control + safety limits
├── cameras/    # Camera ABC, RealSenseCamera, MockCamera
├── targets/    # CalibrationTarget ABC, ArucoMarkerTarget, CharucoBoardTarget
├── solvers/    # HandEyeSolver ABC, OpenCVRobotWorldHandEyeSolver, OpenCVHandEyeSolver
├── session.py  # CalibrationSession orchestrator
├── apps/       # viser collection + verification apps
└── cli.py      # `robot-calib collect` / `robot-calib verify`
```
