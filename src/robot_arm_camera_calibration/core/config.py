from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    """Axis-aligned bounding box, in base-frame meters, jogging must stay within."""

    min_xyz_m: tuple[float, float, float]
    max_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(lo >= hi for lo, hi in zip(self.min_xyz_m, self.max_xyz_m, strict=True)):
            raise ValueError("workspace min_xyz_m must be < max_xyz_m on every axis")

    def contains(self, xyz_m: tuple[float, float, float]) -> bool:
        bounds = zip(self.min_xyz_m, xyz_m, self.max_xyz_m, strict=True)
        return all(lo <= v <= hi for lo, v, hi in bounds)


@dataclass(frozen=True, slots=True)
class JogLimits:
    max_step_m: float = 0.02
    max_step_deg: float = 5.0
    speed_m_s: float = 0.05
    acceleration_m_s2: float = 0.3

    def __post_init__(self) -> None:
        if self.max_step_m <= 0 or self.max_step_deg <= 0:
            raise ValueError("jog step limits must be positive")
        if self.speed_m_s <= 0 or self.acceleration_m_s2 <= 0:
            raise ValueError("jog speed/acceleration must be positive")


@dataclass(frozen=True, slots=True)
class ArucoMarkerConfig:
    kind: Literal["aruco"] = "aruco"
    dictionary: str = "DICT_5X5_100"
    marker_id: int = 0
    marker_length_m: float = 0.05


@dataclass(frozen=True, slots=True)
class CharucoBoardConfig:
    kind: Literal["charuco"] = "charuco"
    dictionary: str = "DICT_5X5_100"
    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.03
    marker_length_m: float = 0.022


TargetConfig = ArucoMarkerConfig | CharucoBoardConfig


@dataclass(frozen=True, slots=True)
class CameraConfig:
    intrinsics_source: Literal["device"] | str = "device"
    """Either "device" (read from the connected RealSense) or a path to a YAML intrinsics file."""


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    robot_ip: str
    target: TargetConfig
    camera: CameraConfig = field(default_factory=CameraConfig)
    workspace_limits: WorkspaceLimits = field(
        default_factory=lambda: WorkspaceLimits((-1.0, -1.0, 0.0), (1.0, 1.0, 1.2))
    )
    jog_limits: JogLimits = field(default_factory=JogLimits)
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        if not isinstance(self.results_dir, Path):
            object.__setattr__(self, "results_dir", Path(self.results_dir))

    def to_yaml(self, path: Path) -> None:
        data = asdict(self)
        data["results_dir"] = str(self.results_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> CalibrationConfig:
        data = yaml.safe_load(path.read_text())
        target_data = dict(data["target"])
        target: TargetConfig
        if target_data["kind"] == "aruco":
            target = ArucoMarkerConfig(**target_data)
        elif target_data["kind"] == "charuco":
            target = CharucoBoardConfig(**target_data)
        else:
            raise ValueError(f"Unknown target kind: {target_data['kind']!r}")

        return cls(
            robot_ip=data["robot_ip"],
            target=target,
            camera=CameraConfig(**data.get("camera", {})),
            workspace_limits=WorkspaceLimits(
                min_xyz_m=tuple(data["workspace_limits"]["min_xyz_m"]),
                max_xyz_m=tuple(data["workspace_limits"]["max_xyz_m"]),
            ),
            jog_limits=JogLimits(**data.get("jog_limits", {})),
            results_dir=Path(data.get("results_dir", "results")),
        )
