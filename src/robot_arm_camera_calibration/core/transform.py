"""Rigid transform (SE(3)) type used everywhere in this library.

Convention: `Transform` wraps a 4x4 homogeneous matrix T such that
`p_a = T @ p_b` when T maps points expressed in frame `b` into frame `a`.
Values of this type are named `<b>_pose_in_<a>` (e.g. `gripper_pose_in_base`)
rather than ambiguous `a2b`/`b2a` abbreviations, since direction mixups are
the single most common source of hand-eye calibration bugs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


@dataclass(frozen=True, slots=True)
class Transform:
    matrix: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.matrix.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {self.matrix.shape}")

    @classmethod
    def identity(cls) -> Transform:
        return cls(np.eye(4))

    @classmethod
    def from_rotation_translation(
        cls, rotation: npt.ArrayLike, translation: npt.ArrayLike
    ) -> Transform:
        """Accepts loosely-typed input (e.g. cv2's MatLike) and normalizes to float64;
        this is the boundary where results from cv2/scipy enter the library's own types."""
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return cls(matrix)

    @classmethod
    def from_rvec_tvec(cls, rvec: npt.ArrayLike, tvec: npt.ArrayLike) -> Transform:
        rotation = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
        return cls.from_rotation_translation(
            rotation, np.asarray(tvec, dtype=np.float64).reshape(3)
        )

    @classmethod
    def from_ur_pose(cls, pose: Sequence[float]) -> Transform:
        x, y, z, rx, ry, rz = pose
        rotation = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
        return cls.from_rotation_translation(rotation, np.array([x, y, z]))

    @classmethod
    def from_xyz_quat_xyzw(
        cls, xyz: npt.NDArray[np.float64], quat_xyzw: npt.NDArray[np.float64]
    ) -> Transform:
        rotation = Rotation.from_quat(quat_xyzw).as_matrix()
        return cls.from_rotation_translation(rotation, np.asarray(xyz).reshape(3))

    @property
    def rotation(self) -> npt.NDArray[np.float64]:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> npt.NDArray[np.float64]:
        return self.matrix[:3, 3]

    def as_rvec_tvec(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rvec = Rotation.from_matrix(self.rotation).as_rotvec()
        return rvec, self.translation.copy()

    def as_ur_pose(self) -> list[float]:
        rvec, tvec = self.as_rvec_tvec()
        return [*tvec.tolist(), *rvec.tolist()]

    def as_xyz_quat_xyzw(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        quat = Rotation.from_matrix(self.rotation).as_quat()
        return self.translation.copy(), quat

    def inverse(self) -> Transform:
        rotation_t = self.rotation.T
        return Transform.from_rotation_translation(rotation_t, -rotation_t @ self.translation)

    def __matmul__(self, other: Transform) -> Transform:
        return Transform(self.matrix @ other.matrix)

    def apply(self, points_local: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        points_local = np.atleast_2d(points_local)
        return points_local @ self.rotation.T + self.translation
