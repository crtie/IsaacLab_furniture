"""Pure data contracts for Frame physical cage proof and delivery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class CageTopology(str, Enum):
    HOOK_THROUGH_FRAME = "HOOK_THROUGH_FRAME"
    THREE_FINGER_WRAP = "THREE_FINGER_WRAP"
    TWO_FINGER_BRACKET = "TWO_FINGER_BRACKET"


class CageState(str, Enum):
    PRECONTACT = "PRECONTACT"
    FIRST_CONTACT = "FIRST_CONTACT"
    BALANCE_CONTACTS = "BALANCE_CONTACTS"
    CAGE_LOCKED = "CAGE_LOCKED"
    LIFT = "LIFT"
    HOLD = "HOLD"
    DONE = "DONE"
    FAILED = "FAILED"


class FailureCode(str, Enum):
    NONE = ""
    IK_UNREACHABLE = "IK_UNREACHABLE"
    PATH_COLLISION = "PATH_COLLISION"
    CAGE_MARGIN_INVALID = "CAGE_MARGIN_INVALID"
    NO_TARGET_CONTACT = "NO_TARGET_CONTACT"
    NON_TARGET_COLLISION = "NON_TARGET_COLLISION"
    FORCE_ABORT = "FORCE_ABORT"
    OBJECT_ESCAPED = "OBJECT_ESCAPED"
    WRIST_TRACKING_FAILED = "WRIST_TRACKING_FAILED"
    INSTRUMENTATION_UNRESOLVED = "INSTRUMENTATION_UNRESOLVED"
    FORBIDDEN_STATE_WRITE = "FORBIDDEN_STATE_WRITE"
    TIME_BUDGET_EXHAUSTED = "TIME_BUDGET_EXHAUSTED"


class DeliveryLevel(str, Enum):
    NONE = "NONE"
    C = "C"
    B = "B"
    A = "A"


class FrameStage4State(str, Enum):
    RESET_PRELOAD = "RESET_PRELOAD"
    SETTLE_ON_FORK = "SETTLE_ON_FORK"
    SUPPORT_CONFIRM = "SUPPORT_CONFIRM"
    LIFT = "LIFT"
    PREINSERT = "PREINSERT"
    INSERT = "INSERT"
    INSERT_HOLD = "INSERT_HOLD"
    RELEASE_TRANSFER = "RELEASE_TRANSFER"
    RETREAT = "RETREAT"
    DONE = "DONE"
    FAILED = "FAILED"


class FrameStage4Failure(str, Enum):
    NONE = ""
    RESET_PRELOAD = "RESET_PRELOAD"
    SUPPORT_NOT_ACQUIRED = "SUPPORT_NOT_ACQUIRED"
    ILLEGAL_CONTACT = "ILLEGAL_CONTACT"
    CONTACT_INSTRUMENTATION_UNAVAILABLE = "CONTACT_INSTRUMENTATION_UNAVAILABLE"
    HARD_FORCE_ABORT = "HARD_FORCE_ABORT"
    FRAME_JUMP = "FRAME_JUMP"
    LIFT_SUPPORT_LOST = "LIFT_SUPPORT_LOST"
    PREINSERT_FAILED = "PREINSERT_FAILED"
    INSERT_FAILED = "INSERT_FAILED"
    RELEASE_UNSTABLE = "RELEASE_UNSTABLE"
    FORBIDDEN_STATE_WRITE = "FORBIDDEN_STATE_WRITE"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class FrameBar:
    bar_id: str
    center_object: tuple[float, float, float]
    axes_object: tuple[tuple[float, float, float], ...]
    half_extents_m: tuple[float, float, float]
    axis_index: int
    local_thickness_m: float
    surface_bounds_object: tuple[tuple[float, float, float], tuple[float, float, float]]
    hole_relation: str
    source_vertex_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlFeature:
    finger_index: int
    link_name: str
    local_support_vertex: tuple[float, float, float]
    target_position_object: tuple[float, float, float]
    target_normal_object: tuple[float, float, float]
    surface_id: str
    role: str


@dataclass(frozen=True)
class CageCandidate:
    candidate_id: str
    topology: CageTopology
    bar_id: str
    finger_group: str
    preclose_q26: tuple[float, ...]
    closed_q26: tuple[float, ...]
    standoff_q26: tuple[float, ...]
    intended_contact_links: tuple[str, ...]
    intended_contact_surfaces: tuple[str, ...]
    control_features: tuple[ControlFeature, ...]
    approach_direction_object: tuple[float, float, float]
    cage_margin_m: float
    gravity_escape_gap_m: float
    bar_local_thickness_m: float
    forbidden_penetration_m: float
    self_collision_free: bool
    table_collision_free: bool
    preclose_path_free: bool
    solver_status: str
    failure_reason: str
    source_seed_id: str
    runtime_mesh_sha256: str
    max_contact_residual_m: float
    joint_limit_margin_rad: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("preclose_q26", "closed_q26", "standoff_q26"):
            if len(getattr(self, name)) != 26:
                raise ValueError(f"{name} must contain 26 values")
        if len(self.control_features) != len(self.finger_group):
            raise ValueError("control feature count must match finger group")

    @property
    def geometry_valid(self) -> bool:
        return bool(
            self.solver_status == "VALID"
            and not self.failure_reason
            and self.cage_margin_m > 0.0
            and self.gravity_escape_gap_m < self.bar_local_thickness_m - 0.001
            and self.forbidden_penetration_m <= 0.0001
            and self.self_collision_free
            and self.table_collision_free
            and self.preclose_path_free
            and self.joint_limit_margin_rad > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["topology"] = self.topology.value
        payload["geometry_valid"] = self.geometry_valid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CageCandidate":
        data = dict(payload)
        data.pop("geometry_valid", None)
        data["topology"] = CageTopology(str(data["topology"]))
        data["control_features"] = tuple(ControlFeature(**row) for row in data["control_features"])
        tuple_fields = (
            "preclose_q26",
            "closed_q26",
            "standoff_q26",
            "intended_contact_links",
            "intended_contact_surfaces",
            "approach_direction_object",
        )
        for name in tuple_fields:
            data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class CageProofResult:
    candidate_id: str
    trial_index: int
    passed: bool
    failure_code: FailureCode
    topology: CageTopology
    lift_delta_m: float
    table_support_final: bool
    intended_contact_duty: float
    peak_force_n: float
    relative_drift_m: float
    hold_steps: int
    post_reset_object_root_writes: int
    post_reset_wrist_state_writes: int
    trace_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_code"] = self.failure_code.value
        payload["topology"] = self.topology.value
        return payload


@dataclass(frozen=True)
class RootWriteAudit:
    reset_object_root_writes: int = 0
    reset_wrist_state_writes: int = 0
    post_reset_object_root_writes: int = 0
    post_reset_wrist_state_writes: int = 0
    locked: bool = False
    rejected_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContactEvidence:
    physics_frame_id: int
    actor0: str
    actor1: str
    source: str
    finger_index: int = 0
    link_name: str = ""
    contact_count: int = 0
    force_n: float = 0.0
    impulse_ns: float = 0.0
    positions_world: tuple[tuple[float, float, float], ...] = ()
    separations_m: tuple[float, ...] = ()


@dataclass(frozen=True)
class ForkSupportPose:
    q26: tuple[float, ...]
    fork_q20: tuple[float, ...]
    hook_q20: tuple[float, ...]
    wrist_q6: tuple[float, ...]
    frame_reset_position: tuple[float, float, float]
    frame_reset_quat_wxyz: tuple[float, float, float, float]
    fixed_reset_position: tuple[float, float, float]
    fixed_reset_quat_wxyz: tuple[float, float, float, float]
    support_links: tuple[str, ...]
    support_points_hand: tuple[tuple[float, float, float], ...]
    support_points_world: tuple[tuple[float, float, float], ...]
    signed_gap_m: float
    normal_offset_m: float
    partial_hook_fraction: float
    fk_position_error_m: float
    fk_rotation_error_deg: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        finite_vector(self.q26, 26, "q26")
        finite_vector(self.fork_q20, 20, "fork_q20")
        finite_vector(self.hook_q20, 20, "hook_q20")
        finite_vector(self.wrist_q6, 6, "wrist_q6")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameStage4Observation:
    physics_frame_id: int
    state: FrameStage4State
    joint_pos26: tuple[float, ...]
    joint_target26: tuple[float, ...]
    frame_position: tuple[float, float, float]
    frame_quat_wxyz: tuple[float, float, float, float]
    fixed_position: tuple[float, float, float]
    fixed_quat_wxyz: tuple[float, float, float, float]
    palm_position: tuple[float, float, float]
    palm_quat_wxyz: tuple[float, float, float, float]
    support_force_n: tuple[float, ...]
    contact_evidence: tuple[ContactEvidence, ...] = ()


@dataclass(frozen=True)
class FrameStage4Command:
    state: FrameStage4State
    target_q26: tuple[float, ...]
    terminal: bool = False
    failure: FrameStage4Failure = FrameStage4Failure.NONE


@dataclass(frozen=True)
class FrameStage4Result:
    delivery_level: DeliveryLevel
    physical_support_acquired: bool
    physical_lift_success: bool
    preinsert_success: bool
    physical_insert_success: bool
    release_stable: bool
    failure: FrameStage4Failure
    physics_frames: int
    frame_lift_m: float
    max_relative_position_drift_m: float
    max_relative_rotation_drift_deg: float
    peak_force_n: float
    post_reset_object_root_writes: int
    post_reset_wrist_state_writes: int
    post_reset_fixed_asset_root_writes: int = 0
    video_path: str = ""
    trace_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delivery_level"] = self.delivery_level.value
        payload["failure"] = self.failure.value
        return payload


class ResetWriteGate:
    """Fail closed if a root/joint-state write is requested after reset."""

    def __init__(self) -> None:
        self.locked = False
        self.reset_object_root_writes = 0
        self.reset_wrist_state_writes = 0
        self.post_reset_object_root_writes = 0
        self.post_reset_wrist_state_writes = 0
        self.rejected_operations: list[str] = []

    def record(self, operation: str, *, wrist_state: bool = False) -> None:
        if self.locked:
            if wrist_state:
                self.post_reset_wrist_state_writes += 1
            else:
                self.post_reset_object_root_writes += 1
            self.rejected_operations.append(str(operation))
            raise RuntimeError(f"forbidden post-reset state write: {operation}")
        if wrist_state:
            self.reset_wrist_state_writes += 1
        else:
            self.reset_object_root_writes += 1

    def lock(self) -> None:
        self.locked = True

    def audit(self) -> RootWriteAudit:
        return RootWriteAudit(
            reset_object_root_writes=self.reset_object_root_writes,
            reset_wrist_state_writes=self.reset_wrist_state_writes,
            post_reset_object_root_writes=self.post_reset_object_root_writes,
            post_reset_wrist_state_writes=self.post_reset_wrist_state_writes,
            locked=self.locked,
            rejected_operations=tuple(self.rejected_operations),
        )


def video_allowed(
    delivery_level: DeliveryLevel,
    *,
    continuous_rollout: bool,
    sticky_used: bool,
    snap_used: bool,
    proxy_used: bool,
    post_reset_root_writes: int,
) -> bool:
    return bool(
        delivery_level in {DeliveryLevel.A, DeliveryLevel.B, DeliveryLevel.C}
        and continuous_rollout
        and not sticky_used
        and not snap_used
        and not proxy_used
        and int(post_reset_root_writes) == 0
    )


def stage4_delivery_video_allowed(
    delivery_level: DeliveryLevel,
    *,
    continuous_rollout: bool,
    trace_matched: bool,
    sticky_used: bool,
    snap_used: bool,
    fixed_joint_used: bool,
    proxy_used: bool,
    object_follow_used: bool,
    post_reset_root_writes: int,
) -> bool:
    return bool(
        delivery_level in {DeliveryLevel.A, DeliveryLevel.B}
        and continuous_rollout
        and trace_matched
        and not sticky_used
        and not snap_used
        and not fixed_joint_used
        and not proxy_used
        and not object_follow_used
        and int(post_reset_root_writes) == 0
    )


def video_evidence_matches_numeric_rollout(video_evidence_attempt: str, numeric_final_attempt: str) -> bool:
    """Return true only when published video and numeric evidence use one rollout."""

    return bool(video_evidence_attempt and video_evidence_attempt == numeric_final_attempt)


def finite_vector(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    import math

    row = tuple(float(value) for value in values)
    if len(row) != int(length) or not all(math.isfinite(value) for value in row):
        raise ValueError(f"{name} must be a finite {length}D vector")
    return row
