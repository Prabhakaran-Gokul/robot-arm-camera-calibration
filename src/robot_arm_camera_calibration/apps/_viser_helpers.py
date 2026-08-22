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


def wxyz_position_to_transform(
    wxyz: npt.NDArray[np.float64], position: npt.NDArray[np.float64]
) -> Transform:
    """Inverse of transform_to_wxyz_position — reads a live-dragged viser handle's pose back
    into a Transform."""
    w, x, y, z = wxyz
    rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    return Transform.from_rotation_translation(rotation, np.asarray(position))


def rotation_axis_coverage(rotations: list[npt.NDArray[np.float64]]) -> float:
    """0-1 score for how well the sample set's relative-rotation AXES span 3D space.

    The Tsai-Lenz observability condition for hand-eye calibration requires the samples'
    relative rotations to span at least two non-parallel axes — a large angle between two
    samples means little if every sample rotates about roughly the same axis, since the
    underlying linear solve is still near-singular. Near 0 means the axes are collinear or
    coplanar (degenerate, even with large individual angles); 1.0 means isotropic coverage
    across all three axes (the smallest and largest eigenvalues of the axis scatter match)."""
    axes = []
    for i, r_i in enumerate(rotations):
        for r_j in rotations[i + 1 :]:
            rotvec = Rotation.from_matrix(r_i.T @ r_j).as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle > 1e-6:
                axes.append(rotvec / angle)
    if len(axes) < 2:
        return 0.0
    axes_array = np.array(axes)
    scatter = axes_array.T @ axes_array / len(axes_array)
    eigenvalues = np.linalg.eigvalsh(scatter)
    return float(eigenvalues.min() / eigenvalues.max())
