"""Regression test for the UR controller-Base <-> URDF base_link frame correction.

Per UR's own ur_description documentation, the controller's native "Base" frame shares its
origin with base_link but is rotated 180 degrees about Z from it. Missing or reversing this
correction doesn't break calibration's math (it still converges with a good residual, since
every pose involved is self-consistently in the wrong frame) — it just makes the result look
mirrored through the base origin when compared against the rendered mesh, which is exactly what
was reported and diagnosed via the viser verify app's manual-correction buttons.

Doesn't need real hardware — only the rtde_control/rtde_receive packages need to be importable,
since UR5eArm imports them at module level.
"""

import numpy as np
import pytest

pytest.importorskip("rtde_control")

from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.robots.ur5e import _BASE_LINK_ROTATION_CORRECTION


def test_correction_is_pure_rotation_about_z_with_no_translation() -> None:
    assert np.allclose(_BASE_LINK_ROTATION_CORRECTION.translation, [0.0, 0.0, 0.0])
    # 180 degrees about Z: negates X and Y, leaves Z unchanged.
    rotated = _BASE_LINK_ROTATION_CORRECTION.rotation @ np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(rotated, [-1.0, -1.0, 1.0], atol=1e-10)


def test_correction_is_its_own_inverse() -> None:
    identity = _BASE_LINK_ROTATION_CORRECTION @ _BASE_LINK_ROTATION_CORRECTION
    np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-10)


def test_correction_negates_xy_translation_preserves_z() -> None:
    raw_controller_pose = Transform.from_ur_pose([0.5, -0.2, 0.3, 1.2, -0.5, 0.1])

    base_link_pose = _BASE_LINK_ROTATION_CORRECTION @ raw_controller_pose

    np.testing.assert_allclose(
        base_link_pose.translation,
        [
            -raw_controller_pose.translation[0],
            -raw_controller_pose.translation[1],
            raw_controller_pose.translation[2],
        ],
        atol=1e-10,
    )


def test_round_trip_recovers_original_controller_pose() -> None:
    raw_controller_pose = Transform.from_ur_pose([0.5, -0.2, 0.3, 1.2, -0.5, 0.1])

    base_link_pose = _BASE_LINK_ROTATION_CORRECTION @ raw_controller_pose
    recovered = _BASE_LINK_ROTATION_CORRECTION @ base_link_pose

    np.testing.assert_allclose(recovered.matrix, raw_controller_pose.matrix, atol=1e-10)
