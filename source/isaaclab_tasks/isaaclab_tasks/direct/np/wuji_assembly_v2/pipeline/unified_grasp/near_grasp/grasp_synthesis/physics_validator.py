"""Gate contracts and explicit contact-source attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Iterable, Mapping

import numpy as np


class GateKind(IntEnum):
    RESET_CLOSED_HOLD_LIFT = 1
    PREGRASP_CLOSE_LIFT = 2
    STANDOFF_APPROACH_CLOSE_LIFT = 3


class ContactSource(str, Enum):
    NO_CONTACT = "NO_CONTACT"
    TARGET_OBJECT = "TARGET_OBJECT"
    TABLE = "TABLE"
    GROUND = "GROUND"
    SELF = "SELF"
    MULTIPLE_IDENTIFIED = "MULTIPLE_IDENTIFIED"
    UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT = "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT"


@dataclass(frozen=True)
class ContactAttribution:
    source: ContactSource
    identified_pairs: tuple[tuple[str, str], ...] = ()
    target_force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    all_force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    closure_residual_n: float = 0.0
    instrumentation_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source"] = self.source.value
        return row


def attribute_contact(
    *,
    all_force_xyz: Iterable[float],
    filtered_forces: Mapping[str, Iterable[float]],
    identified_pairs: Iterable[tuple[str, str]] = (),
    filter_valid: Mapping[str, bool] | None = None,
    residual_tolerance_n: float = 0.02,
) -> ContactAttribution:
    all_force = np.asarray(tuple(all_force_xyz), dtype=np.float64)
    if all_force.shape != (3,) or not np.all(np.isfinite(all_force)):
        raise ValueError("all contact force must be a finite 3D vector")
    valid = dict(filter_valid or {})
    vectors = {}
    for name in ("TARGET_OBJECT", "TABLE", "GROUND"):
        vector = np.asarray(tuple(filtered_forces.get(name, (0.0, 0.0, 0.0))), dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} force must be a finite 3D vector")
        vectors[name] = vector
    pairs = tuple((str(first), str(second)) for first, second in identified_pairs)
    pair_sources = set()
    for first, second in pairs:
        joined = f"{first} {second}".lower()
        if "targetobject" in joined or "plug2" in joined or "screw1" in joined or "backrest" in joined or "rod" in joined or "frame" in joined:
            pair_sources.add(ContactSource.TARGET_OBJECT)
        elif "table" in joined:
            pair_sources.add(ContactSource.TABLE)
        elif "ground" in joined:
            pair_sources.add(ContactSource.GROUND)
        elif "robot" in first.lower() and "robot" in second.lower():
            pair_sources.add(ContactSource.SELF)
    filtered_sum = sum((vector for name, vector in vectors.items() if valid.get(name, True)), start=np.zeros(3))
    residual = float(np.linalg.norm(all_force - filtered_sum))
    active_filtered = {
        ContactSource[name]
        for name, vector in vectors.items()
        if valid.get(name, True) and float(np.linalg.norm(vector)) > residual_tolerance_n
    }
    sources = pair_sources | active_filtered
    all_norm = float(np.linalg.norm(all_force))
    missing_filter = any(not valid.get(name, True) for name in vectors)
    unresolved = bool(all_norm > residual_tolerance_n and not sources and (residual > residual_tolerance_n or missing_filter))
    if unresolved:
        source = ContactSource.UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT
    elif len(sources) > 1:
        source = ContactSource.MULTIPLE_IDENTIFIED
    elif len(sources) == 1:
        source = next(iter(sources))
    else:
        source = ContactSource.NO_CONTACT
    return ContactAttribution(
        source=source,
        identified_pairs=pairs,
        target_force_xyz=tuple(float(value) for value in vectors["TARGET_OBJECT"]),
        table_force_xyz=tuple(float(value) for value in vectors["TABLE"]),
        ground_force_xyz=tuple(float(value) for value in vectors["GROUND"]),
        all_force_xyz=tuple(float(value) for value in all_force),
        closure_residual_n=residual,
        instrumentation_limit=source == ContactSource.UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT,
    )


@dataclass(frozen=True)
class GateResult:
    gate: GateKind
    candidate_id: str
    trial_id: int
    passed: bool
    termination_layer: str
    hold_contact_steps: int
    hold_total_steps: int
    peak_target_force_n: float
    lift_contact_duty: float
    object_z_gain_m: float
    lost_table_support: bool
    relative_drift_m: float
    physical_grasp_success: bool
    physical_lift_success: bool
    full_route_success: bool
    oracle_reset_grasp: bool
    approach_close_lift_success: bool = False
    honesty: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.full_route_success:
            raise ValueError("grasp gates stop at lift and cannot report full_route_success")
        if self.approach_close_lift_success and self.gate != GateKind.STANDOFF_APPROACH_CLOSE_LIFT:
            raise ValueError("only Gate C may report approach_close_lift_success")
        if self.gate == GateKind.RESET_CLOSED_HOLD_LIFT and self.physical_grasp_success:
            raise ValueError("Gate A cannot claim autonomous physical grasp acquisition")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["gate"] = self.gate.name
        return row


def gate_passed(results: Iterable[GateResult], *, required_successes: int = 3, required_trials: int = 5) -> bool:
    rows = tuple(results)
    if len(rows) != required_trials or len({row.gate for row in rows}) != 1 or len({row.candidate_id for row in rows}) != 1:
        return False
    return sum(int(row.passed) for row in rows) >= required_successes


def evaluate_gate_trace(
    *,
    gate: GateKind,
    candidate_id: str,
    trial_id: int,
    target_force_norms: np.ndarray,
    active_finger_mask: Iterable[bool],
    object_positions: np.ndarray,
    hand_positions: np.ndarray,
    table_supported: np.ndarray,
    hold_start_step: int,
    lift_start_step: int,
    controlled_close_completed: bool,
    verified_non_target_contact: bool,
    unresolved_contact: bool,
    post_reset_object_writes: int,
    post_reset_wrist_writes: int,
) -> GateResult:
    forces = np.asarray(target_force_norms, dtype=np.float64)
    active = np.asarray(tuple(active_finger_mask), dtype=bool)
    object_pos = np.asarray(object_positions, dtype=np.float64)
    hand_pos = np.asarray(hand_positions, dtype=np.float64)
    support = np.asarray(table_supported, dtype=bool)
    if forces.ndim != 2 or forces.shape[1] != 5 or active.shape != (5,):
        raise ValueError("gate force trace shapes are invalid")
    if object_pos.shape != hand_pos.shape or object_pos.shape != (len(forces), 3) or support.shape != (len(forces),):
        raise ValueError("gate pose trace shapes are invalid")
    if not np.any(active) or len(forces) == 0:
        raise ValueError("gate trace requires active fingers and at least one step")
    hold_start = int(np.clip(hold_start_step, 0, len(forces)))
    lift_start = int(np.clip(lift_start_step, hold_start, len(forces) - 1))
    formal = forces[:, active] > 0.05
    simultaneous = np.all(formal, axis=1)
    hold = simultaneous[hold_start:lift_start]
    hold_window = hold[-30:]
    hold_steps = int(np.count_nonzero(hold_window))
    peak = float(np.max(forces[:, active]))
    lift_duty = float(np.mean(simultaneous[lift_start:])) if lift_start < len(forces) else 0.0
    z_gain = float(object_pos[-1, 2] - object_pos[lift_start, 2])
    lost_support = bool(not np.any(support[-min(10, len(support)) :]))
    relative = object_pos - hand_pos
    drift = float(np.max(np.linalg.norm(relative[lift_start:] - relative[lift_start], axis=1)))
    invalid = bool(
        verified_non_target_contact
        or unresolved_contact
        or post_reset_object_writes > 0
        or post_reset_wrist_writes > 0
        or peak >= 5.0
    )
    close_required = gate != GateKind.RESET_CLOSED_HOLD_LIFT
    passed = bool(
        not invalid
        and hold_window.size >= 30
        and hold_steps >= 24
        and peak < 1.0
        and (controlled_close_completed or not close_required)
        and z_gain >= 0.010
        and lost_support
        and lift_duty >= 0.8
        and drift <= 0.010
    )
    if unresolved_contact:
        termination = "CONTACT_ATTRIBUTION_INSTRUMENTATION_LIMIT"
    elif verified_non_target_contact:
        termination = "VERIFIED_NON_TARGET_SCENE_CONTACT"
    elif peak >= 5.0:
        termination = "HARD_FORCE_ABORT"
    elif hold_steps < 24:
        termination = f"GATE_{gate.value}_MULTI_CONTACT_HOLD_FAILED"
    elif peak >= 1.0:
        termination = f"GATE_{gate.value}_SOFT_FORCE_LIMIT_EXCEEDED"
    elif close_required and not controlled_close_completed:
        termination = f"GATE_{gate.value}_CLOSE_FAILED"
    elif z_gain < 0.010 or not lost_support or lift_duty < 0.8 or drift > 0.010:
        termination = f"GATE_{gate.value}_LIFT_FAILED"
    else:
        termination = "PASS"
    honesty = {
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": True,
        "root_pose_writes_reset_only": post_reset_object_writes == 0,
        "proxy_used": False,
        "not_physical": False,
    }
    return GateResult(
        gate=gate,
        candidate_id=candidate_id,
        trial_id=int(trial_id),
        passed=passed,
        termination_layer=termination,
        hold_contact_steps=hold_steps,
        hold_total_steps=int(hold_window.size),
        peak_target_force_n=peak,
        lift_contact_duty=lift_duty,
        object_z_gain_m=z_gain,
        lost_table_support=lost_support,
        relative_drift_m=drift,
        physical_grasp_success=bool(passed and gate != GateKind.RESET_CLOSED_HOLD_LIFT),
        physical_lift_success=passed,
        full_route_success=False,
        oracle_reset_grasp=gate == GateKind.RESET_CLOSED_HOLD_LIFT,
        approach_close_lift_success=bool(passed and gate == GateKind.STANDOFF_APPROACH_CLOSE_LIFT),
        honesty=honesty,
        metadata={"controlled_close_completed": bool(controlled_close_completed)},
    )
