"""Object randomization reporting for v80 physics profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .physics_profile import PROFILES, PhysicsProfile
from .v80_reports import V80_PARTS, write_csv, write_json


@dataclass(frozen=True)
class RandomizationSpec:
    part_name: str
    physics_profile: str
    mass_scale_min: float
    mass_scale_max: float
    friction_scale_min: float
    friction_scale_max: float
    pose_noise_m: float
    yaw_noise_rad: float
    final_export_allowed: bool


def spec_for_part(part_name: str, profile: PhysicsProfile) -> RandomizationSpec:
    if profile.randomized:
        mass_min, mass_max = 0.8 * profile.mass_scale, 1.25 * profile.mass_scale
        friction_min = 0.75 * min(profile.static_friction_scale, profile.dynamic_friction_scale)
        friction_max = 1.35 * max(profile.static_friction_scale, profile.dynamic_friction_scale)
        pose_noise_m = 0.006
        yaw_noise_rad = 0.20
    else:
        mass_min = mass_max = profile.mass_scale
        friction_min = min(profile.static_friction_scale, profile.dynamic_friction_scale)
        friction_max = max(profile.static_friction_scale, profile.dynamic_friction_scale)
        pose_noise_m = 0.0
        yaw_noise_rad = 0.0
    return RandomizationSpec(
        part_name=part_name,
        physics_profile=profile.name,
        mass_scale_min=mass_min,
        mass_scale_max=mass_max,
        friction_scale_min=friction_min,
        friction_scale_max=friction_max,
        pose_noise_m=pose_noise_m,
        yaw_noise_rad=yaw_noise_rad,
        final_export_allowed=profile.final_training_export_allowed,
    )


def build_object_randomization_rows(parts: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for part in parts or V80_PARTS:
        for profile in PROFILES.values():
            rows.append(spec_for_part(part, profile).__dict__)
    return rows


def write_object_randomization_report(run_dir: str | Path, parts: list[str] | None = None) -> dict[str, str]:
    rows = build_object_randomization_rows(parts)
    csv_path = write_csv(Path(run_dir) / "object_randomization_report.csv", rows)
    json_path = write_json(Path(run_dir) / "object_randomization_report.json", rows)
    return {"object_randomization_report_csv": str(csv_path), "object_randomization_report_json": str(json_path)}
