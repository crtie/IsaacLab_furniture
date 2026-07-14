"""Explicit physics profiles for v80 grasp training and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .v80_reports import write_json


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    mass_scale: float = 1.0
    static_friction_scale: float = 1.0
    dynamic_friction_scale: float = 1.0
    restitution: float | None = None
    linear_damping: float | None = None
    angular_damping: float | None = None
    contact_offset_m: float | None = None
    rest_offset_m: float | None = None
    solver_position_iterations: int | None = None
    solver_velocity_iterations: int | None = None
    randomized: bool = False
    sticky_after_support_validation: bool = False
    final_training_export_allowed: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, PhysicsProfile] = {
    "canonical": PhysicsProfile(
        name="canonical",
        notes="Repo default masses/frictions; required for final validation.",
    ),
    "train_easy_high_friction": PhysicsProfile(
        name="train_easy_high_friction",
        mass_scale=1.5,
        static_friction_scale=1.8,
        dynamic_friction_scale=1.6,
        linear_damping=0.02,
        angular_damping=0.02,
        final_training_export_allowed=False,
        notes="Curriculum profile only; not final training evidence.",
    ),
    "train_randomized": PhysicsProfile(
        name="train_randomized",
        randomized=True,
        final_training_export_allowed=False,
        notes="Domain-randomized training profile; final rows must revalidate canonical.",
    ),
    "validation_randomized": PhysicsProfile(
        name="validation_randomized",
        randomized=True,
        final_training_export_allowed=False,
        notes="Robustness validation profile.",
    ),
    "validation_sticky_after_support": PhysicsProfile(
        name="validation_sticky_after_support",
        sticky_after_support_validation=True,
        final_training_export_allowed=False,
        notes="Validation-only sticky stabilizer after verified support/contact.",
    ),
}


def get_physics_profile(name: str) -> PhysicsProfile:
    try:
        return PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown v80 physics profile: {name}") from exc


def write_physics_profile_config(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "physics_profile_config.json"
    write_json(path, {name: profile.to_dict() for name, profile in PROFILES.items()})
    return path


def apply_profile_to_scalar(value: float, profile: PhysicsProfile, *, field: str) -> float:
    if field == "mass":
        return float(value) * profile.mass_scale
    if field == "static_friction":
        return float(value) * profile.static_friction_scale
    if field == "dynamic_friction":
        return float(value) * profile.dynamic_friction_scale
    return float(value)
