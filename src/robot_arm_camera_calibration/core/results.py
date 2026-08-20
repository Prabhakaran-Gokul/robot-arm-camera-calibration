from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.core.transform import Transform


def _transform_to_yaml_dict(transform: Transform) -> dict[str, Any]:
    quat = Rotation.from_matrix(transform.rotation).as_quat()
    return {
        "matrix": transform.matrix.tolist(),
        "translation_m": transform.translation.tolist(),
        "quaternion_xyzw": quat.tolist(),
    }


def _transform_from_yaml_dict(data: dict[str, Any]) -> Transform:
    return Transform(np.array(data["matrix"], dtype=np.float64))


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    camera_pose_in_base: Transform
    marker_pose_in_gripper: Transform
    method: str
    num_samples: int
    created_at: datetime
    translation_residual_rmse_m: float
    rotation_residual_rmse_deg: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self, path: Path) -> None:
        data = {
            "camera_pose_in_base": _transform_to_yaml_dict(self.camera_pose_in_base),
            "marker_pose_in_gripper": _transform_to_yaml_dict(self.marker_pose_in_gripper),
            "method": self.method,
            "num_samples": self.num_samples,
            "created_at": self.created_at.isoformat(),
            "translation_residual_rmse_m": self.translation_residual_rmse_m,
            "rotation_residual_rmse_deg": self.rotation_residual_rmse_deg,
            "metadata": self.metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> CalibrationResult:
        data = yaml.safe_load(path.read_text())
        return cls(
            camera_pose_in_base=_transform_from_yaml_dict(data["camera_pose_in_base"]),
            marker_pose_in_gripper=_transform_from_yaml_dict(data["marker_pose_in_gripper"]),
            method=data["method"],
            num_samples=data["num_samples"],
            created_at=datetime.fromisoformat(data["created_at"]),
            translation_residual_rmse_m=data["translation_residual_rmse_m"],
            rotation_residual_rmse_deg=data["rotation_residual_rmse_deg"],
            metadata=data.get("metadata", {}),
        )
