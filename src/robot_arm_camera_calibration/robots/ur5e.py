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
on the controller side — and ur_rtde surfaces this as a raw exception (a boost.asio "End of
file" is typical) rather than a clean error. Both interfaces expose isConnected()/reconnect()
for exactly this; every call that touches them checks and transparently reconnects first, so a
transient drop doesn't leave the robot's reported pose frozen or crash the caller.

There is no freedrive support here: RTDEControlInterface's teachMode() looks like it should let
the robot be pushed by hand without leaving Remote mode, but per UR's own forum (and the ISO
10218-1 "Single Point of Control" principle their engineers cite) it does not actually engage
while the robot's system-wide Remote Control setting is on — the same restriction that blocks
the pendant's own freedrive button, since only one authority (pendant/local or external/remote)
can hold motion control at a time. That's a firmware/safety-level restriction, not something to
work around here. For manual guidance, use control_enabled=False (below) and toggle Remote
Control off on the pendant for that session instead.

RTDEControlInterface itself is not thread-safe (ur_rtde's own docs say so): calling any of its
methods concurrently from two threads — e.g. the background servo loop mid-servoL() while a GUI
callback calls moveJ() — corrupts its internal state, surfacing on the pendant as "another
thread is already controlling the robot" and in the log as "RTDE control script is not running".
_rtde_lock serializes every direct call into it (_rtde_r_lock does the same for
RTDEReceiveInterface); _state_lock is a separate, unrelated lock that only protects the plain
_target_pose/_servo_speed/_servo_acceleration Python variables.

A protective/emergency stop halts the running control script on the controller — reconnect()
alone (a socket-level reconnect) does not bring it back, since the script itself is no longer
running; reuploadScript() re-uploads and restarts it, which _ensure_control_connected() does
whenever isProgramRunning() reports false. There is deliberately no way to clear the stop itself
from software: is_protective_stopped/is_emergency_stopped only report the condition so the
caller can surface it, since actually clearing a safety stop must stay a physical, human action
on the pendant.

isConnected()/isProgramRunning() can themselves raise instead of cleanly returning false once
the underlying socket is already broken — _safe_check() treats that the same as a false answer
so reconnection is still attempted rather than the exception propagating out of a routine status
check. Genuinely unrecoverable failures (reconnect()/reuploadScript() itself raising) still
propagate; callers (the viser app's GUI handlers) are responsible for catching and surfacing
those rather than crashing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import rtde_control
import rtde_receive

from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.base import RobotArm

_CONTROL_HZ = 500.0
_LOOKAHEAD_TIME_S = 0.1
_GAIN = 300
_SAFETY_STOP_POLL_INTERVAL_S = 0.5


def _safe_check(query: Callable[[], bool]) -> bool:
    """A "still connected?"-style query can itself raise instead of cleanly returning False
    when the underlying socket is already broken — treat that the same as a False answer."""
    try:
        return query()
    except Exception:
        return False


class UR5eArm(RobotArm):
    def __init__(self, robot_ip: str, control_enabled: bool = True) -> None:
        self._robot_ip = robot_ip
        self._control_enabled = control_enabled
        self._rtde_c: rtde_control.RTDEControlInterface | None = None
        self._rtde_r: rtde_receive.RTDEReceiveInterface | None = None
        self._state_lock = threading.Lock()
        self._rtde_lock = threading.Lock()
        self._rtde_r_lock = threading.Lock()
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
            with self._rtde_lock:
                self._rtde_c.servoStop()
                self._rtde_c.stopScript()
        self._rtde_c = None
        self._rtde_r = None

    @property
    def is_connected(self) -> bool:
        if self._rtde_r is None:
            return False
        with self._rtde_r_lock:
            if not _safe_check(self._rtde_r.isConnected):
                return False
        if not self._control_enabled:
            return True
        if self._rtde_c is None:
            return False
        with self._rtde_lock:
            return _safe_check(self._rtde_c.isConnected)

    @property
    def is_protective_stopped(self) -> bool:
        if self._rtde_r is None:
            return False
        with self._rtde_r_lock:
            return _safe_check(self._rtde_r.isProtectiveStopped)

    @property
    def is_emergency_stopped(self) -> bool:
        if self._rtde_r is None:
            return False
        with self._rtde_r_lock:
            return _safe_check(self._rtde_r.isEmergencyStopped)

    def get_tcp_pose(self) -> Transform:
        self._ensure_receive_connected()
        assert self._rtde_r is not None
        with self._rtde_r_lock:
            pose = self._rtde_r.getActualTCPPose()
        return Transform.from_ur_pose(pose)

    def get_joint_positions(self) -> npt.NDArray[np.float64]:
        self._ensure_receive_connected()
        assert self._rtde_r is not None
        with self._rtde_r_lock:
            q = self._rtde_r.getActualQ()
        return np.array(q, dtype=np.float64)

    def _ensure_receive_connected(self) -> None:
        assert self._rtde_r is not None, "Robot is not connected"
        with self._rtde_r_lock:
            if not _safe_check(self._rtde_r.isConnected):
                self._rtde_r.reconnect()

    def _ensure_control_connected(self) -> None:
        assert self._rtde_c is not None, (
            "Robot control is disabled (control_enabled=False) — connect with "
            "control_enabled=True to jog, or drive the robot from the teach pendant instead"
        )
        with self._rtde_lock:
            if not _safe_check(self._rtde_c.isConnected):
                self._rtde_c.reconnect()
            if not _safe_check(self._rtde_c.isProgramRunning):
                self._rtde_c.reuploadScript()

    def servo_to_pose(self, target: Transform, *, speed: float, acceleration: float) -> None:
        assert self._rtde_c is not None, (
            "Robot control is disabled (control_enabled=False) — connect with "
            "control_enabled=True to jog, or drive the robot from the teach pendant instead"
        )
        with self._state_lock:
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
            with self._rtde_lock:
                self._rtde_c.moveJ(
                    np.asarray(joint_positions, dtype=np.float64).tolist(), speed, acceleration
                )
        finally:
            with self._state_lock:
                self._target_pose = self.get_tcp_pose()
            self._paused.clear()

    def stop(self) -> None:
        if self._rtde_c is not None:
            with self._rtde_lock:
                self._rtde_c.servoStop()

    def _servo_loop(self) -> None:
        assert self._rtde_c is not None
        dt = 1.0 / _CONTROL_HZ
        was_safety_stopped = False
        while not self._stop_event.is_set():
            if self._paused.is_set():
                time.sleep(dt)
                continue
            try:
                if self.is_protective_stopped or self.is_emergency_stopped:
                    if not was_safety_stopped:
                        print(
                            "[UR5eArm] Robot is protective/emergency stopped — waiting for it "
                            "to be cleared and re-enabled on the pendant."
                        )
                        was_safety_stopped = True
                    time.sleep(_SAFETY_STOP_POLL_INTERVAL_S)
                    continue
                if was_safety_stopped:
                    print("[UR5eArm] Safety stop cleared, resuming.")
                    was_safety_stopped = False

                self._ensure_control_connected()
                # Re-check: _paused may have been set by another thread (e.g.
                # move_to_joint_positions()) between the check above and here, and must not be
                # missed right before servoL.
                if self._paused.is_set():
                    continue
                with self._state_lock:
                    target, speed, acceleration = (
                        self._target_pose,
                        self._servo_speed,
                        self._servo_acceleration,
                    )
                with self._rtde_lock:
                    period_start = self._rtde_c.initPeriod()
                    if target is not None:
                        self._rtde_c.servoL(
                            target.as_ur_pose(), speed, acceleration, dt, _LOOKAHEAD_TIME_S, _GAIN
                        )
                    self._rtde_c.waitPeriod(period_start)
            except Exception as error:
                print(f"[UR5eArm] servo loop iteration failed, retrying: {error}")
                time.sleep(dt)
