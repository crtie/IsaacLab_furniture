"""Morphology and object contracts for contact-adaptive closure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HandMorphologySpec:
    hand_name: str
    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    finger_names: tuple[str, ...]
    finger_joint_groups: Mapping[str, tuple[int, ...]]
    fingertip_body_names: tuple[str, ...]
    preshape_q: tuple[float, ...]
    close_reference_q: tuple[float, ...]
    contact_capable_fingers: tuple[str, ...]
    palm_support_body: str
    action_columns: tuple[int, ...]
    morphology_calibration_source: str
    calibration_identity: str

    def validate(self) -> None:
        width = len(self.joint_names)
        fields = (self.joint_limits, self.preshape_q, self.close_reference_q, self.action_columns)
        if width != 20 or any(len(item) != width for item in fields):
            raise ValueError("Wuji morphology requires aligned 20-joint fields")
        if len(self.finger_names) != len(self.fingertip_body_names):
            raise ValueError("finger/tip body count mismatch")
        covered = sorted(index for values in self.finger_joint_groups.values() for index in values)
        if covered != list(range(width)):
            raise ValueError("finger joint groups must cover each hand joint exactly once")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointEffectProbe:
    joint_name: str
    local_joint_index: int
    action_column: int
    action_value: float
    sign: int
    command_delta_rad: float
    app_target_delta_rad: float
    runtime_target_delta_rad: float
    actual_delta_rad: float
    settle_lag_steps: int
    other_joint_actual_peak_rad: float
    crosstalk_ratio: float
    fingertip_position_delta_m: tuple[tuple[float, float, float], ...]
    fingertip_orientation_delta_rotvec: tuple[tuple[float, float, float], ...]
    target_chain_ok: bool
    response_ok: bool


@dataclass(frozen=True)
class MorphologyCalibration:
    schema_version: int
    hand_signature: str
    probe_delta_rad: float
    joint_effect_probes: tuple[JointEffectProbe, ...]
    effect_matrix_30x20: tuple[tuple[float, ...], ...]
    crosstalk_limit: float
    calibration_valid: bool
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)

    def validate_for(self, spec: HandMorphologySpec) -> None:
        if self.hand_signature != spec.calibration_identity:
            raise ValueError("morphology calibration signature mismatch")
        matrix = np.asarray(self.effect_matrix_30x20, dtype=np.float64)
        if matrix.shape != (30, 20) or not np.isfinite(matrix).all():
            raise ValueError("effect matrix must be finite 30x20")
        if not self.calibration_valid:
            raise ValueError("morphology calibration is not valid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectGraspSpec:
    part_name: str
    grasp_family: str
    active_finger_groups: tuple[tuple[str, ...], ...]
    approach_frame: str
    acquisition_plan_path: str
    preshape_profile: str
    closure_prior_name: str
    closure_axes_by_finger: Mapping[str, tuple[float, float, float]]
    force_band_n: tuple[float, float]
    hard_abort_force_n: float
    preclose_object_motion_limit_m: float
    palm_support_allowed: bool
    lift_direction_xyz: tuple[float, float, float]
    lift_target_m: float
    closure_frame: str = "object_local"
    pinch_axis_object: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_center_offset_object: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nominal_contact_offset_m: float = 0.003
    finger_side_by_name: Mapping[str, float] = field(default_factory=dict)
    finger_approach_gain: Mapping[str, float] = field(default_factory=dict)
    pad_normal_target_by_finger: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    allowed_contact_bodies: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, morphology: HandMorphologySpec) -> None:
        known = set(morphology.finger_names)
        if not self.active_finger_groups or any(len(group) < 2 for group in self.active_finger_groups):
            raise ValueError("object spec requires at least one multi-finger group")
        if any(finger not in known for group in self.active_finger_groups for finger in group):
            raise ValueError("object spec references unknown finger")
        if self.force_band_n[0] <= 0.0 or self.force_band_n[1] >= self.hard_abort_force_n:
            raise ValueError("invalid force guard ordering")
        if self.closure_frame != "object_local":
            raise ValueError("closure_frame must be object_local")
        axis = np.asarray(self.pinch_axis_object, dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1.0e-9:
            raise ValueError("pinch_axis_object must be a finite nonzero vector")
        if len(self.contact_center_offset_object) != 3 or self.nominal_contact_offset_m <= 0.0:
            raise ValueError("invalid object-relative contact geometry")
        for group in self.active_finger_groups:
            if any(finger not in self.finger_side_by_name for finger in group):
                raise ValueError("every active finger requires an explicit pinch side")
            if any(float(self.finger_approach_gain.get(finger, 0.0)) <= 0.0 for finger in group):
                raise ValueError("every active finger requires a positive approach gain")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def morphology_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_wuji_morphology_spec(adapter: Any, *, calibration_source: str = "") -> HandMorphologySpec:
    names = tuple(str(item) for item in adapter.get_wuji_joint_names())
    limits = adapter.get_hand_joint_limits()
    preshape = adapter.get_hand_joint_pose("dex_hand_preshape_pose")
    close = adapter.get_hand_joint_pose("dex_hand_close_pose")
    if not limits.get("ok") or not preshape.get("ok") or not close.get("ok"):
        raise RuntimeError("runtime Wuji morphology fields unavailable")
    lower = tuple(float(item) for item in limits["lower"])
    upper = tuple(float(item) for item in limits["upper"])
    groups: dict[str, tuple[int, ...]] = {}
    for finger in range(1, 6):
        groups[f"finger{finger}"] = tuple(
            index for index, name in enumerate(names) if name.startswith(f"right_finger{finger}_joint")
        )
    base = adapter.env
    tip_names = tuple(
        str(item)
        for item in (
            list(getattr(base, "dex_fingertip_true_body_names", []) or [])
            or list(getattr(base, "dex_fingertip_anchor_body_names", []) or [])
        )
    )
    identity_payload = {
        "hand_name": "WujiFloatingHand",
        "joint_names": names,
        "joint_limits": list(zip(lower, upper)),
        "tip_names": tip_names,
        "preshape_q": preshape["pose"],
        "action_columns": list(range(6, 26)),
        "action_scale": float(getattr(base, "dex_hand_action_scale", 1.0)),
        "asset": str(getattr(getattr(getattr(base, "_robot", None), "cfg", None), "prim_path", "")),
    }
    spec = HandMorphologySpec(
        hand_name="WujiFloatingHand",
        joint_names=names,
        joint_limits=tuple(zip(lower, upper)),
        finger_names=tuple(f"finger{index}" for index in range(1, 6)),
        finger_joint_groups=groups,
        fingertip_body_names=tip_names,
        preshape_q=tuple(float(item) for item in preshape["pose"]),
        close_reference_q=tuple(float(item) for item in close["pose"]),
        contact_capable_fingers=tuple(f"finger{index}" for index in range(1, 6)),
        palm_support_body="palm_link",
        action_columns=tuple(range(6, 26)),
        morphology_calibration_source=str(calibration_source),
        calibration_identity=morphology_signature(identity_payload),
    )
    spec.validate()
    return spec


def default_object_grasp_specs() -> dict[str, ObjectGraspSpec]:
    common = {
        "approach_frame": "object_local",
        "preshape_profile": "runtime_dex_hand_preshape_pose",
        "closure_prior_name": "other_hand_closure_prior",
        "force_band_n": (0.05, 1.0),
        "hard_abort_force_n": 5.0,
        "preclose_object_motion_limit_m": 0.005,
        "palm_support_allowed": False,
        "lift_direction_xyz": (0.0, 0.0, 1.0),
        "lift_target_m": 0.010,
        "closure_frame": "object_local",
        "pinch_axis_object": (0.0, 1.0, 0.0),
        "contact_center_offset_object": (0.0, 0.0, 0.0),
        "nominal_contact_offset_m": 0.003,
        "finger_approach_gain": {
            "finger2": 1.0,
            "finger3": 1.0,
            "finger4": 1.0,
        },
    }
    return {
        "Screw1": ObjectGraspSpec(
            part_name="Screw1",
            grasp_family="small_object",
            active_finger_groups=(("finger3", "finger4"),),
            acquisition_plan_path=(
                "debug_runs/screw1_grasp_baseline_v2_probe_handoff_fix/validated_acquisition_plan.json"
            ),
            closure_axes_by_finger={"finger3": (0.0, -1.0, 0.0), "finger4": (0.0, 1.0, 0.0)},
            finger_side_by_name={"finger3": -1.0, "finger4": 1.0},
            allowed_contact_bodies=("Screw1",),
            **common,
        ),
        "Plug2": ObjectGraspSpec(
            part_name="Plug2",
            grasp_family="thin_object",
            active_finger_groups=(("finger2", "finger3"), ("finger3", "finger4")),
            acquisition_plan_path="",
            closure_axes_by_finger={
                "finger2": (0.0, -1.0, 0.0),
                "finger3": (0.0, 1.0, 0.0),
                "finger4": (0.0, 1.0, 0.0),
            },
            finger_side_by_name={"finger2": -1.0, "finger3": 1.0, "finger4": 1.0},
            allowed_contact_bodies=("Plug2",),
            **common,
        ),
    }


def effect_matrix_from_probes(probes: Sequence[JointEffectProbe], joint_count: int = 20) -> np.ndarray:
    matrix = np.zeros((30, joint_count), dtype=np.float64)
    grouped: dict[int, list[np.ndarray]] = {}
    for probe in probes:
        if not probe.response_ok or abs(probe.actual_delta_rad) <= 1.0e-8:
            continue
        pos = np.asarray(probe.fingertip_position_delta_m, dtype=np.float64).reshape(5, 3)
        rot = np.asarray(probe.fingertip_orientation_delta_rotvec, dtype=np.float64).reshape(5, 3)
        derivative = np.concatenate((pos, rot), axis=1).reshape(30) / probe.actual_delta_rad
        grouped.setdefault(probe.local_joint_index, []).append(derivative)
    for index, values in grouped.items():
        matrix[:, index] = np.mean(np.stack(values), axis=0)
    return matrix


def calibration_from_dict(data: Mapping[str, Any]) -> MorphologyCalibration:
    probes = tuple(JointEffectProbe(**dict(item)) for item in list(data.get("joint_effect_probes") or []))
    return MorphologyCalibration(
        schema_version=int(data.get("schema_version", 1)),
        hand_signature=str(data.get("hand_signature") or ""),
        probe_delta_rad=float(data.get("probe_delta_rad", 0.002)),
        joint_effect_probes=probes,
        effect_matrix_30x20=tuple(
            tuple(float(value) for value in row) for row in list(data.get("effect_matrix_30x20") or [])
        ),
        crosstalk_limit=float(data.get("crosstalk_limit", 0.25)),
        calibration_valid=bool(data.get("calibration_valid")),
        failure_reasons=tuple(str(item) for item in list(data.get("failure_reasons") or [])),
    )
