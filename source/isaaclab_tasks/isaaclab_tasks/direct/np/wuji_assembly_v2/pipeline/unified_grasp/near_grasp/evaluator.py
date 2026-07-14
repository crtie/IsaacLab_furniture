"""Strict physical evaluator independent of training reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


CONTACT_THRESHOLD_N = 0.05
SOFT_FORCE_LIMIT_N = 1.0
HARD_FORCE_ABORT_N = 5.0


@dataclass(frozen=True)
class EvaluationInput:
    target_force_norms: np.ndarray
    active_finger_mask: Sequence[bool]
    object_positions: np.ndarray
    hand_positions: np.ndarray
    table_top_z_m: float
    close_start_step: int
    lift_start_step: int
    controlled_close_completed: bool
    non_target_contact: bool = False
    unresolved_contact_truth: bool = False
    sticky_used: bool = False
    proxy_used: bool = False
    snap_used: bool = False
    teacher_motion_used: bool = False
    reset_root_pose_write_used: bool = True
    post_reset_object_writes: int = 0
    post_reset_wrist_state_writes: int = 0
    flyout: bool = False
    penetration: bool = False
    runtime_error: str = ""
    action_jerk: np.ndarray | None = None
    object_supported_by_table: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalEvaluation:
    valid_candidate: bool
    hard_invalid: bool
    invalid_reasons: tuple[str, ...]
    physical_lift_success: bool
    stable_close: bool
    simultaneous_contact_duty: float
    lift_contact_duty: float
    peak_target_force_n: float
    safe_force: bool
    object_z_gain_m: float
    table_clearance_m: float
    lost_table_support: bool
    lateral_displacement_m: float
    relative_drift_m: float
    jerk_metric: float
    target_filtered_success_evidence: bool
    honesty: dict[str, bool]
    metadata: dict[str, Any]

    def ranking_key(self) -> tuple[float, ...]:
        """Compatibility ranking; new search code uses SearchMetrics directly."""

        from .grasp_synthesis.grasp_energy import SearchMetrics

        return SearchMetrics.from_physical_evaluation(self).ranking_key()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_candidate": self.valid_candidate,
            "hard_invalid": self.hard_invalid,
            "invalid_reasons": list(self.invalid_reasons),
            "physical_lift_success": self.physical_lift_success,
            "stable_close": self.stable_close,
            "simultaneous_contact_duty": self.simultaneous_contact_duty,
            "lift_contact_duty": self.lift_contact_duty,
            "peak_target_force_n": self.peak_target_force_n,
            "safe_force": self.safe_force,
            "object_z_gain_m": self.object_z_gain_m,
            "table_clearance_m": self.table_clearance_m,
            "lost_table_support": self.lost_table_support,
            "lateral_displacement_m": self.lateral_displacement_m,
            "relative_drift_m": self.relative_drift_m,
            "jerk_metric": self.jerk_metric,
            "target_filtered_success_evidence": self.target_filtered_success_evidence,
            **self.honesty,
            "metadata": self.metadata,
        }


class StrictPhysicalEvaluator:
    """Evaluate traces without reading reward totals or geometry-only scores."""

    def __init__(
        self,
        *,
        contact_threshold_n: float = CONTACT_THRESHOLD_N,
        soft_force_limit_n: float = SOFT_FORCE_LIMIT_N,
        hard_force_abort_n: float = HARD_FORCE_ABORT_N,
        lift_height_m: float = 0.010,
        lift_contact_duty: float = 0.8,
        relative_drift_m: float = 0.010,
        stable_hold_steps: int = 30,
        stable_hold_required_steps: int = 24,
    ):
        self.contact_threshold_n = float(contact_threshold_n)
        self.soft_force_limit_n = float(soft_force_limit_n)
        self.hard_force_abort_n = float(hard_force_abort_n)
        self.lift_height_m = float(lift_height_m)
        self.required_lift_contact_duty = float(lift_contact_duty)
        self.max_relative_drift_m = float(relative_drift_m)
        self.stable_hold_steps = int(stable_hold_steps)
        self.stable_hold_required_steps = int(stable_hold_required_steps)

    def evaluate(self, evidence: EvaluationInput) -> PhysicalEvaluation:
        forces = np.asarray(evidence.target_force_norms, dtype=np.float64)
        object_pos = np.asarray(evidence.object_positions, dtype=np.float64)
        hand_pos = np.asarray(evidence.hand_positions, dtype=np.float64)
        active = np.asarray(evidence.active_finger_mask, dtype=bool)
        if forces.ndim != 2 or forces.shape[1] != 5 or object_pos.shape != hand_pos.shape or object_pos.ndim != 2 or object_pos.shape[1] != 3:
            raise ValueError("EvaluationInput trace shapes are invalid")
        if object_pos.shape[0] != forces.shape[0] or active.shape != (5,) or not np.any(active):
            raise ValueError("EvaluationInput trace lengths or active-finger mask are invalid")
        if not np.all(np.isfinite(forces)) or not np.all(np.isfinite(object_pos)) or not np.all(np.isfinite(hand_pos)):
            raise ValueError("EvaluationInput contains non-finite physics evidence")

        step_count = forces.shape[0]
        close_start = int(np.clip(evidence.close_start_step, 0, max(step_count - 1, 0)))
        lift_start = int(np.clip(evidence.lift_start_step, close_start, max(step_count - 1, 0)))
        active_contact = forces[:, active] > self.contact_threshold_n
        simultaneous = np.all(active_contact, axis=1)
        simultaneous_duty = float(np.mean(simultaneous))
        hold_start = max(close_start, lift_start - self.stable_hold_steps)
        hold = simultaneous[hold_start:lift_start]
        stable_close = bool(
            evidence.controlled_close_completed
            and hold.size >= self.stable_hold_steps
            and int(np.count_nonzero(hold[-self.stable_hold_steps :])) >= self.stable_hold_required_steps
        )
        lift_contact = simultaneous[lift_start:]
        lift_duty = float(np.mean(lift_contact)) if lift_contact.size else 0.0
        peak_force = float(np.max(forces[:, active])) if forces.size else 0.0
        safe_force = bool(peak_force < self.soft_force_limit_n)

        close_baseline = object_pos[lift_start]
        final_object = object_pos[-1]
        z_gain = float(final_object[2] - close_baseline[2])
        clearance = float(final_object[2] - evidence.table_top_z_m)
        lateral = float(np.linalg.norm(final_object[:2] - object_pos[0, :2]))
        relative = object_pos - hand_pos
        relative_drift = float(np.max(np.linalg.norm(relative[lift_start:] - relative[lift_start], axis=1)))
        if evidence.object_supported_by_table is None:
            lost_table_support = bool(clearance > 0.002 and z_gain >= self.lift_height_m)
        else:
            support = np.asarray(evidence.object_supported_by_table, dtype=bool).reshape(-1)
            if support.shape[0] != step_count:
                raise ValueError("object_supported_by_table length mismatch")
            lost_table_support = bool(not np.any(support[-min(10, step_count) :]))
        jerk = 0.0
        if evidence.action_jerk is not None:
            jerk_array = np.asarray(evidence.action_jerk, dtype=np.float64)
            jerk = float(np.mean(np.linalg.norm(jerk_array.reshape(jerk_array.shape[0], -1), axis=1)))

        invalid_reasons = []
        hard_flags = {
            "sticky_used": evidence.sticky_used,
            "proxy_used": evidence.proxy_used,
            "post_reset_object_write": evidence.post_reset_object_writes > 0,
            "post_reset_wrist_state_write": evidence.post_reset_wrist_state_writes > 0,
            "hard_force_abort": peak_force >= self.hard_force_abort_n,
            "verified_non_target_contact": evidence.non_target_contact,
            "flyout": evidence.flyout,
            "penetration": evidence.penetration,
            "runtime_error": bool(evidence.runtime_error),
        }
        invalid_reasons.extend(name for name, value in hard_flags.items() if value)
        if evidence.unresolved_contact_truth:
            invalid_reasons.append("unresolved_contact_truth")
        hard_invalid = bool(any(hard_flags.values()) or evidence.unresolved_contact_truth)
        target_evidence = bool(np.any(forces[:, active] > self.contact_threshold_n))
        lift_success = bool(
            not hard_invalid
            and stable_close
            and safe_force
            and target_evidence
            and z_gain >= self.lift_height_m
            and lost_table_support
            and lift_duty >= self.required_lift_contact_duty
            and relative_drift <= self.max_relative_drift_m
        )
        honesty = {
            "sticky_used": bool(evidence.sticky_used),
            "snap_used": bool(evidence.snap_used),
            "teacher_motion_used": bool(evidence.teacher_motion_used),
            "root_pose_writes_used": bool(
                evidence.reset_root_pose_write_used or evidence.post_reset_object_writes > 0
            ),
            "root_pose_writes_reset_only": bool(
                evidence.reset_root_pose_write_used and evidence.post_reset_object_writes == 0
            ),
            "physical_grasp_success": bool(stable_close and target_evidence and not hard_invalid),
            "physical_lift_success": lift_success,
            "physical_insert_success": False,
            "oracle_visual_only": False,
            "not_physical": not lift_success,
        }
        return PhysicalEvaluation(
            valid_candidate=not hard_invalid,
            hard_invalid=hard_invalid,
            invalid_reasons=tuple(invalid_reasons),
            physical_lift_success=lift_success,
            stable_close=stable_close,
            simultaneous_contact_duty=simultaneous_duty,
            lift_contact_duty=lift_duty,
            peak_target_force_n=peak_force,
            safe_force=safe_force,
            object_z_gain_m=z_gain,
            table_clearance_m=clearance,
            lost_table_support=lost_table_support,
            lateral_displacement_m=lateral,
            relative_drift_m=relative_drift,
            jerk_metric=jerk,
            target_filtered_success_evidence=target_evidence,
            honesty=honesty,
            metadata={**evidence.metadata, "runtime_error": evidence.runtime_error},
        )
