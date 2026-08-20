"""ur_rtde-based UR5e driver.

servoL is a real-time streaming primitive: it must be called continuously at a fixed control
rate or the controller's watchdog faults. connect() spawns a background thread that streams the
current target pose every control cycle (holding position between jog commands re-sends the
same pose, satisfying the watchdog); servo_to_pose() just swaps the lock-protected target that
loop is reading. moveJ is blocking and would conflict with a concurrently-streaming servoL, so
move_to_joint_positions() pauses the streaming loop for its duration.

RTDEControlInterface uploads and runs an external-control script on the robot as soon as it
connects, which takes the teach pendant out of manual jog/freedrive control while active.
control_enabled=False skips creating it entirely, leaving the pendant in full control, while
RTDEReceiveInterface (read-only) still works for capturing sample poses.

RTDE connections can drop mid-session — most commonly when the robot's control state changes
on the controller side, e.g. switching into freedrive/manual control from the pendant — and
ur_rtde surfaces this as a raw exception (a boost.asio "End of file" is typical) rather than a
clean error. Both interfaces expose isConnected()/reconnect() for exactly this; every call that
touches them checks and transparently reconnects first, so a transient drop doesn't leave the
robot's reported pose frozen or crash the caller.
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
    def __init__(self, robot_ip: str, control_enabled: bool = True) -> None:
        self._robot_ip = robot_ip
        self._control_enabled = control_enabled
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
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self._robot_ip)
        if not self._control_enabled:
            return
        self._rtde_c = rtde_control.RTDEControlInterface(self._robot_ip)
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
        if self._rtde_r is None or not self._rtde_r.isConnected():
            return False
        if not self._control_enabled:
            return True
        return self._rtde_c is not None and self._rtde_c.isConnected()

    def get_tcp_pose(self) -> Transform:
        self._ensure_receive_connected()
        assert self._rtde_r is not None
        return Transform.from_ur_pose(self._rtde_r.getActualTCPPose())

    def get_joint_positions(self) -> npt.NDArray[np.float64]:
        self._ensure_receive_connected()
        assert self._rtde_r is not None
        return np.array(self._rtde_r.getActualQ(), dtype=np.float64)

    def _ensure_receive_connected(self) -> None:
        assert self._rtde_r is not None, "Robot is not connected"
        if not self._rtde_r.isConnected():
            self._rtde_r.reconnect()

    def _ensure_control_connected(self) -> None:
        assert self._rtde_c is not None, (
            "Robot control is disabled (control_enabled=False) — connect with "
            "control_enabled=True to jog, or drive the robot from the teach pendant instead"
        )
        if not self._rtde_c.isConnected():
            self._rtde_c.reconnect()

    def servo_to_pose(self, target: Transform, *, speed: float, acceleration: float) -> None:
        assert self._rtde_c is not None, (
            "Robot control is disabled (control_enabled=False) — connect with "
            "control_enabled=True to jog, or drive the robot from the teach pendant instead"
        )
        with self._lock:
            self._target_pose = target
            self._servo_speed = speed
            self._servo_acceleration = acceleration

    def move_to_joint_positions(
        self, joint_positions: npt.NDArray[np.float64], *, speed: float, acceleration: float
    ) -> None:
        self._ensure_control_connected()
        assert self._rtde_c is not None
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
            try:
                self._ensure_control_connected()
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
            except Exception as error:
                print(f"[UR5eArm] servo loop iteration failed, retrying: {error}")
                time.sleep(dt)
