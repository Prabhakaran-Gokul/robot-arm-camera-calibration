import numpy as np
import numpy.typing as npt
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.transform import Transform


def load_urdf(description_name: str) -> yourdfpy.URDF:
    return load_robot_description(description_name)


def transform_to_wxyz_position(
    transform: Transform,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """viser poses scene nodes with a wxyz quaternion + xyz position, not a 4x4 matrix."""
    x, y, z, w = Rotation.from_matrix(transform.rotation).as_quat()
    tx, ty, tz = transform.translation
    return (float(w), float(x), float(y), float(z)), (float(tx), float(ty), float(tz))


def rotation_diversity_deg(rotations: list[npt.NDArray[np.float64]]) -> float:
    """Max pairwise rotation angle (degrees) across a set of rotation matrices — a cheap proxy
    for whether captured samples vary enough in orientation for hand-eye calibration to work."""
    if len(rotations) < 2:
        return 0.0
    max_angle = 0.0
    for i, r_i in enumerate(rotations):
        for r_j in rotations[i + 1 :]:
            relative = r_i.T @ r_j
            angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)))
            max_angle = max(max_angle, float(angle))
    return max_angle
