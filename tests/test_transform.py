import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.transform import Transform


def test_identity_is_neutral() -> None:
    t = Transform.identity()
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(t.apply(points), points)


def test_inverse_composes_to_identity() -> None:
    rotation = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t = Transform.from_rotation_translation(rotation, np.array([0.1, -0.2, 0.3]))
    identity = t @ t.inverse()
    np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-10)


def test_composition_matches_manual_chaining() -> None:
    rng = np.random.default_rng(0)
    a = Transform.from_rotation_translation(
        Rotation.from_euler("xyz", rng.uniform(-180, 180, 3), degrees=True).as_matrix(),
        rng.uniform(-1, 1, 3),
    )
    b = Transform.from_rotation_translation(
        Rotation.from_euler("xyz", rng.uniform(-180, 180, 3), degrees=True).as_matrix(),
        rng.uniform(-1, 1, 3),
    )
    point_in_c = rng.uniform(-1, 1, 3)
    point_in_a_via_compose = (a @ b).apply(point_in_c)
    point_in_a_via_manual = a.apply(b.apply(point_in_c))
    np.testing.assert_allclose(point_in_a_via_compose, point_in_a_via_manual, atol=1e-10)


def test_rvec_tvec_roundtrip() -> None:
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([0.5, -0.4, 1.2])
    t = Transform.from_rvec_tvec(rvec, tvec)
    rvec_out, tvec_out = t.as_rvec_tvec()
    np.testing.assert_allclose(rvec_out, rvec, atol=1e-10)
    np.testing.assert_allclose(tvec_out, tvec, atol=1e-10)


def test_ur_pose_roundtrip() -> None:
    pose = [0.3, -0.1, 0.4, 0.05, 1.2, -0.7]
    t = Transform.from_ur_pose(pose)
    np.testing.assert_allclose(t.as_ur_pose(), pose, atol=1e-10)
