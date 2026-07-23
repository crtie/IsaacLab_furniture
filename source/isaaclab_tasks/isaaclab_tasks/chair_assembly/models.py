"""Public result and task models for the packaged chair assembly workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ResultCode(str, Enum):
    OK = "OK"
    POLICY_UNAVAILABLE = "POLICY_UNAVAILABLE"
    POLICY_INVALID = "POLICY_INVALID"
    MISSING_CALIBRATION = "MISSING_CALIBRATION"
    CONTACT_UNAVAILABLE = "CONTACT_UNAVAILABLE"
    GRASP_NOT_VERIFIED = "GRASP_NOT_VERIFIED"
    INSERT_NOT_VERIFIED = "INSERT_NOT_VERIFIED"
    SYSTEM_VALIDATION_COMPLETE = "SYSTEM_VALIDATION_COMPLETE"
    SYSTEM_VALIDATION_FAILED = "SYSTEM_VALIDATION_FAILED"
    RUNTIME_ERROR = "RUNTIME_ERROR"


@dataclass(frozen=True)
class AssemblyTarget:
    stage_id: int
    target_id: int
    target_name: str
    part_name: str
    insert_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TargetResult:
    stage_id: int
    target_id: int
    target_name: str
    status: str
    reason: str = ""
    assistance_operations: list[str] = field(default_factory=list)
    policy_ids: dict[str, str] = field(default_factory=dict)
    sticky_used: bool = False
    snap_used: bool = False
    teacher_motion_used: bool = False
    root_pose_writes_used: bool = False
    physical_grasp_success: bool = False
    physical_lift_success: bool = False
    physical_insert_success: bool = False
    oracle_visual_only: bool = False
    not_physical: bool = True
    bc_training_eligible: bool = False


@dataclass
class AssemblyReport:
    backend: str
    robot: str
    variant: str
    status: str
    result_code: str
    targets_planned: int
    targets_completed: int
    target_results: list[TargetResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    sticky_used: bool = False
    snap_used: bool = False
    teacher_motion_used: bool = False
    root_pose_writes_used: bool = False
    physical_grasp_success: bool = False
    physical_lift_success: bool = False
    physical_insert_success: bool = False
    oracle_visual_only: bool = False
    not_physical: bool = True
    bc_training_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

