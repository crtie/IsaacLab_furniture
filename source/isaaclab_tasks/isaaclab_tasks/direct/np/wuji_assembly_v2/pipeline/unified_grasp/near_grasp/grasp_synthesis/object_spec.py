"""Validated object contracts for privileged grasp synthesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SUPPORTED_PARTS = ("Plug2", "Rod", "Backrest", "Screw1", "Frame")
SUPPORTED_GEOMETRY = ("mesh", "capsule_z", "cylinder_z", "box", "box_union")
SUPPORTED_CONTACT_MODES = ("antipodal_pair", "three_point")


def _vec(values: Sequence[float], width: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {width} finite values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class ContactRegion:
    region_id: str
    center_object: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]
    allowed_normal_object: tuple[float, float, float] | None = None
    normal_axis_object: tuple[float, float, float] | None = None
    normal_abs_dot_max: float | None = None
    allowed_finger_groups: tuple[str, ...] = ()
    contact_modes: tuple[str, ...] = ("antipodal_pair",)
    section_axis_object: tuple[float, float, float] | None = None
    section_tolerance_m: float = 0.0005
    local_thickness_range_m: tuple[float, float] = (0.0, float("inf"))
    approach_directions_object: tuple[tuple[float, float, float], ...] = ()
    sampling_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_object", _vec(self.center_object, 3, "contact region center"))
        extents = _vec(self.half_extents_m, 3, "contact region half extents")
        if min(extents) < 0.0:
            raise ValueError("contact region extents cannot be negative")
        object.__setattr__(self, "half_extents_m", extents)
        if self.allowed_normal_object is not None:
            normal = np.asarray(_vec(self.allowed_normal_object, 3, "contact region normal"))
            norm = float(np.linalg.norm(normal))
            if norm <= 1.0e-9:
                raise ValueError("contact region normal cannot be zero")
            object.__setattr__(self, "allowed_normal_object", tuple(float(value) for value in normal / norm))
        if self.normal_axis_object is not None:
            axis = np.asarray(_vec(self.normal_axis_object, 3, "contact region normal axis"))
            norm = float(np.linalg.norm(axis))
            if norm <= 1.0e-9:
                raise ValueError("contact region normal axis cannot be zero")
            object.__setattr__(self, "normal_axis_object", tuple(float(value) for value in axis / norm))
            if self.normal_abs_dot_max is None or not 0.0 <= self.normal_abs_dot_max <= 1.0:
                raise ValueError("normal_abs_dot_max must be in [0,1] when a normal axis is set")
        for group in self.allowed_finger_groups:
            _validate_finger_group(group)
        if not self.contact_modes or any(mode not in SUPPORTED_CONTACT_MODES for mode in self.contact_modes):
            raise ValueError("contact region has an unsupported contact mode")
        if self.section_axis_object is not None:
            axis = np.asarray(_vec(self.section_axis_object, 3, "contact region section axis"))
            norm = float(np.linalg.norm(axis))
            if norm <= 1.0e-9:
                raise ValueError("contact region section axis cannot be zero")
            object.__setattr__(self, "section_axis_object", tuple(float(value) for value in axis / norm))
        if self.section_tolerance_m < 0.0:
            raise ValueError("contact region section tolerance cannot be negative")
        thickness = _vec(self.local_thickness_range_m, 2, "contact region thickness range")
        if thickness[0] < 0.0 or thickness[0] >= thickness[1]:
            raise ValueError("contact region thickness range is invalid")
        object.__setattr__(self, "local_thickness_range_m", thickness)
        directions = tuple(_unit_vec(value, "contact region approach direction") for value in self.approach_directions_object)
        object.__setattr__(self, "approach_directions_object", directions)
        if not np.isfinite(self.sampling_weight) or self.sampling_weight <= 0.0:
            raise ValueError("contact region sampling weight must be positive")


@dataclass(frozen=True)
class AnalyticGeometry:
    kind: str
    center_object: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    half_length_m: float = 0.0
    half_extents_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_GEOMETRY:
            raise ValueError(f"unsupported geometry kind {self.kind!r}")
        object.__setattr__(self, "center_object", _vec(self.center_object, 3, "geometry center"))
        object.__setattr__(self, "half_extents_m", _vec(self.half_extents_m, 3, "geometry half extents"))
        if self.radius_m < 0.0 or self.half_length_m < 0.0:
            raise ValueError("geometry dimensions cannot be negative")
        if self.kind in {"capsule_z", "cylinder_z"} and (self.radius_m <= 0.0 or self.half_length_m <= 0.0):
            raise ValueError(f"{self.kind} requires positive radius and half length")
        if self.kind == "box" and min(self.half_extents_m) <= 0.0:
            raise ValueError("box geometry requires positive half extents")


@dataclass(frozen=True)
class ObjectGraspSpec:
    schema_version: int
    part_name: str
    asset_usd: str
    visual_mesh: str
    asset_scale: tuple[float, float, float]
    geometry: AnalyticGeometry
    allowed_contact_regions: tuple[ContactRegion, ...]
    allowed_finger_groups: tuple[str, ...]
    approach_directions_object: tuple[tuple[float, float, float], ...]
    nominal_mass_kg: float
    nominal_static_friction: float
    nominal_dynamic_friction: float
    target_normal_force_range_n: tuple[float, float]
    lift_direction_world: tuple[float, float, float]
    candidate_fallback_order: tuple[str, ...]
    canonical_object_pos: tuple[float, float, float]
    canonical_object_quat_wxyz: tuple[float, float, float, float]
    planning_geometry_only: bool = True
    runtime_asset_audit_required: bool = True

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 3}:
            raise ValueError("unsupported ObjectGraspSpec schema")
        if self.part_name not in SUPPORTED_PARTS:
            raise ValueError(f"unsupported part {self.part_name!r}")
        object.__setattr__(self, "asset_scale", _vec(self.asset_scale, 3, "asset scale"))
        if min(self.asset_scale) <= 0.0:
            raise ValueError("asset scale must be positive")
        if not self.allowed_contact_regions or not self.allowed_finger_groups or not self.approach_directions_object:
            raise ValueError("spec requires contact regions, finger groups, and approach directions")
        for group in self.allowed_finger_groups:
            _validate_finger_group(group)
        object.__setattr__(
            self,
            "approach_directions_object",
            tuple(_unit_vec(values, "approach direction") for values in self.approach_directions_object),
        )
        if self.nominal_mass_kg <= 0.0 or min(self.nominal_static_friction, self.nominal_dynamic_friction) < 0.0:
            raise ValueError("nominal mass and friction are invalid")
        force_range = _vec(self.target_normal_force_range_n, 2, "target force range")
        if not (0.05 <= force_range[0] < force_range[1] < 1.0):
            raise ValueError("target force range must stay inside the safe physical band")
        object.__setattr__(self, "target_normal_force_range_n", force_range)
        lift = np.asarray(_vec(self.lift_direction_world, 3, "lift direction"))
        lift_norm = float(np.linalg.norm(lift))
        if lift_norm <= 1.0e-9:
            raise ValueError("lift direction cannot be zero")
        object.__setattr__(self, "lift_direction_world", tuple(float(value) for value in lift / lift_norm))
        object.__setattr__(self, "canonical_object_pos", _vec(self.canonical_object_pos, 3, "canonical object pos"))
        quat = np.asarray(_vec(self.canonical_object_quat_wxyz, 4, "canonical object quaternion"))
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1.0e-9:
            raise ValueError("canonical object quaternion cannot be zero")
        object.__setattr__(self, "canonical_object_quat_wxyz", tuple(float(value) for value in quat / quat_norm))

    @property
    def target_force_n(self) -> float:
        return 0.5 * sum(self.target_normal_force_range_n)

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path, *, repo_root: str | Path | None = None) -> "ObjectGraspSpec":
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("ObjectGraspSpec YAML root must be a mapping")
        root = Path(repo_root).resolve() if repo_root is not None else source.parents[3]

        def resolve(value: str) -> str:
            candidate = Path(value)
            return str(candidate if candidate.is_absolute() else (root / candidate).resolve())

        geometry_row = dict(payload["geometry"])
        boxes = tuple(
            (tuple(float(v) for v in row["center"]), tuple(float(v) for v in row["half_extents_m"]))
            for row in geometry_row.get("boxes", ())
        )
        geometry = AnalyticGeometry(
            kind=str(geometry_row["kind"]),
            center_object=tuple(geometry_row.get("center_object", (0.0, 0.0, 0.0))),
            radius_m=float(geometry_row.get("radius_m", 0.0)),
            half_length_m=float(geometry_row.get("half_length_m", 0.0)),
            half_extents_m=tuple(geometry_row.get("half_extents_m", (0.0, 0.0, 0.0))),
            boxes=boxes,
        )
        regions = tuple(
            ContactRegion(
                region_id=str(row["id"]),
                center_object=tuple(row["center_object"]),
                half_extents_m=tuple(row["half_extents_m"]),
                allowed_normal_object=tuple(row["allowed_normal_object"]) if row.get("allowed_normal_object") is not None else None,
                normal_axis_object=tuple(row["normal_axis_object"]) if row.get("normal_axis_object") is not None else None,
                normal_abs_dot_max=float(row["normal_abs_dot_max"]) if row.get("normal_abs_dot_max") is not None else None,
                allowed_finger_groups=tuple(str(value) for value in row.get("allowed_finger_groups", ())),
                contact_modes=tuple(str(value) for value in row.get("contact_modes", ("antipodal_pair",))),
                section_axis_object=tuple(row["section_axis_object"]) if row.get("section_axis_object") is not None else None,
                section_tolerance_m=float(row.get("section_tolerance_m", 0.0005)),
                local_thickness_range_m=tuple(row.get("local_thickness_range_m", (0.0, float("inf")))),
                approach_directions_object=tuple(tuple(value) for value in row.get("approach_directions_object", ())),
                sampling_weight=float(row.get("sampling_weight", 1.0)),
            )
            for row in payload["allowed_contact_regions"]
        )
        return cls(
            schema_version=int(payload["schema_version"]),
            part_name=str(payload["part_name"]),
            asset_usd=resolve(str(payload["asset_usd"])),
            visual_mesh=resolve(str(payload["visual_mesh"])),
            asset_scale=tuple(payload["asset_scale"]),
            geometry=geometry,
            allowed_contact_regions=regions,
            allowed_finger_groups=tuple(str(value) for value in payload["allowed_finger_groups"]),
            approach_directions_object=tuple(tuple(value) for value in payload["approach_directions_object"]),
            nominal_mass_kg=float(payload["nominal_mass_kg"]),
            nominal_static_friction=float(payload["nominal_static_friction"]),
            nominal_dynamic_friction=float(payload["nominal_dynamic_friction"]),
            target_normal_force_range_n=tuple(payload["target_normal_force_range_n"]),
            lift_direction_world=tuple(payload["lift_direction_world"]),
            candidate_fallback_order=tuple(str(value) for value in payload["candidate_fallback_order"]),
            canonical_object_pos=tuple(payload["canonical_object_pos"]),
            canonical_object_quat_wxyz=tuple(payload["canonical_object_quat_wxyz"]),
            planning_geometry_only=bool(payload.get("planning_geometry_only", True)),
            runtime_asset_audit_required=bool(payload.get("runtime_asset_audit_required", True)),
        )

    def audit_static_assets(self, *, mismatch_abs_m: float = 0.001, mismatch_fraction: float = 0.02) -> dict[str, Any]:
        usd = Path(self.asset_usd)
        visual = Path(self.visual_mesh)
        row: dict[str, Any] = {
            "part_name": self.part_name,
            "asset_usd": str(usd),
            "asset_usd_exists": usd.is_file(),
            "visual_mesh": str(visual),
            "visual_mesh_exists": visual.is_file(),
            "planning_geometry_only": self.planning_geometry_only,
            "runtime_asset_audit_required": self.runtime_asset_audit_required,
            "spec_hash": self.content_hash,
        }
        for label, candidate in (("asset", usd), ("visual", visual)):
            if candidate.is_file():
                row[f"{label}_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if visual.is_file() and visual.suffix.lower() in {".obj", ".stl", ".ply", ".off"}:
            import trimesh

            mesh = trimesh.load_mesh(visual, process=False)
            extent = np.asarray(mesh.extents, dtype=np.float64) * np.asarray(self.asset_scale, dtype=np.float64)
            row.update(
                {
                    "visual_extent_m": extent.tolist(),
                    "visual_watertight": bool(mesh.is_watertight),
                    "visual_vertex_count": int(len(mesh.vertices)),
                    "visual_face_count": int(len(mesh.faces)),
                }
            )
            expected = _geometry_extent(self.geometry)
            if expected is not None:
                tolerance = np.maximum(float(mismatch_abs_m), float(mismatch_fraction) * np.maximum(extent, expected))
                mismatch = np.abs(extent - expected)
                row["analytic_extent_m"] = expected.tolist()
                row["extent_mismatch_m"] = mismatch.tolist()
                row["analytic_geometry_matches_visual"] = bool(np.all(mismatch <= tolerance))
        row["static_asset_audit_ok"] = bool(row["asset_usd_exists"] and row["visual_mesh_exists"])
        return row


def _validate_finger_group(group: str) -> None:
    try:
        digits = tuple(int(value) for value in group)
    except ValueError as exc:
        raise ValueError(f"invalid finger group {group!r}") from exc
    if len(digits) < 2 or len(set(digits)) != len(digits) or min(digits) < 1 or max(digits) > 5:
        raise ValueError(f"invalid finger group {group!r}")


def _unit_vec(values: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = np.asarray(_vec(values, 3, name), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        raise ValueError(f"{name} cannot be zero")
    return tuple(float(value) for value in vector / norm)


def _geometry_extent(geometry: AnalyticGeometry) -> np.ndarray | None:
    if geometry.kind == "box":
        return 2.0 * np.asarray(geometry.half_extents_m, dtype=np.float64)
    if geometry.kind == "cylinder_z":
        return np.asarray((2.0 * geometry.radius_m, 2.0 * geometry.radius_m, 2.0 * geometry.half_length_m))
    if geometry.kind == "capsule_z":
        return np.asarray(
            (2.0 * geometry.radius_m, 2.0 * geometry.radius_m, 2.0 * (geometry.half_length_m + geometry.radius_m))
        )
    if geometry.kind == "box_union" and geometry.boxes:
        lower = []
        upper = []
        for center, half in geometry.boxes:
            center_array = np.asarray(center)
            half_array = np.asarray(half)
            lower.append(center_array - half_array)
            upper.append(center_array + half_array)
        return np.max(np.stack(upper), axis=0) - np.min(np.stack(lower), axis=0)
    return None
