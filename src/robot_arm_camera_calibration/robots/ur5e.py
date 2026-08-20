"""ur_rtde-based UR5e driver.

servoL is a real-time streaming primitive: it must be called continuously at a fixed control
rate or the controller's watchdog faults. connect() spawns a background thread that streams the
current target pose every control cycle (holding position between jog commands re-sends the
same pose, satisfying the watchdog); servo_to_pose() just swaps the lock-protected target that
loop is reading. moveJ is blocking and would conflict with a concurrently-streaming servoL, so
move_to_joint_positions() pauses the streaming loop for its duration.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import numpy.typing as npt
import rtde_control
import rtde_receive

from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.base import RobotArm

_CONTROL_HZ = 500.0
_LOOKAHEAD_TIME_S = 0.1
_GAIN = 300


class UR5eArm(RobotArm):
    def __init__(self, robot_ip: str) -> None:
        self._robot_ip = robot_ip
        self._rtde_c: rtde_control.RTDEControlInterface | None = None
        self._rtde_r: rtde_receive.RTDEReceiveInterface | None = None
        self._lock = threading.Lock()
        self._target_pose: Transform | None = None
        self._servo_speed = 0.05
        self._servo_acceleration = 0.3
        self._paused = threading.Event()
        self._stop_event = threading.Event()
        self._servo_thread: threading.Thread | None = None

    def connect(self) -> None:
        self._rtde_c = rtde_control.RTDEControlInterface(self._robot_ip)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self._robot_ip)
        self._target_pose = self.get_tcp_pose()
        self._stop_event.clear()
        self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
        self._servo_thread.start()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=1.0)
        if self._rtde_c is not None:
            self._rtde_c.servoStop()
            self._rtde_c.stopScript()
        self._rtde_c = None
        self._rtde_r = None

    @property
    def is_connected(self) -> bool:
        return self._rtde_c is not None and self._rtde_r is not None

    def get_tcp_pose(self) -> Transform:
        assert self._rtde_r is not None, "Robot is not connected"
        return Transform.from_ur_pose(self._rtde_r.getActualTCPPose())

    def get_joint_positions(self) -> npt.NDArray[np.float64]:
        assert self._rtde_r is not None, "Robot is not connected"
        return np.array(self._rtde_r.getActualQ(), dtype=np.float64)

    def servo_to_pose(self, target: Transform, *, speed: float, acceleration: float) -> None:
        with self._lock:
            self._target_pose = target
            self._servo_speed = speed
            self._servo_acceleration = acceleration

    def move_to_joint_positions(
        self, joint_positions: npt.NDArray[np.float64], *, speed: float, acceleration: float
    ) -> None:
        assert self._rtde_c is not None, "Robot is not connected"
        self._paused.set()
        try:
            self._rtde_c.moveJ(
                np.asarray(joint_positions, dtype=np.float64).tolist(), speed, acceleration
            )
        finally:
            with self._lock:
                self._target_pose = self.get_tcp_pose()
            self._paused.clear()

    def stop(self) -> None:
        if self._rtde_c is not None:
            self._rtde_c.servoStop()

    def _servo_loop(self) -> None:
        assert self._rtde_c is not None
        dt = 1.0 / _CONTROL_HZ
        while not self._stop_event.is_set():
            if self._paused.is_set():
                time.sleep(dt)
                continue
            period_start = self._rtde_c.initPeriod()
            with self._lock:
                target, speed, acceleration = (
                    self._target_pose,
                    self._servo_speed,
                    self._servo_acceleration,
                )
            if target is not None:
                self._rtde_c.servoL(
                    target.as_ur_pose(), speed, acceleration, dt, _LOOKAHEAD_TIME_S, _GAIN
                )
            self._rtde_c.waitPeriod(period_start)
