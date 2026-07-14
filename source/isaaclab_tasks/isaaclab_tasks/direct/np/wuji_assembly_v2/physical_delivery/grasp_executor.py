"""Action-driven mechanical cage proof for one Frame candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CageCandidate, CageProofResult, CageState, CageTopology, FailureCode


@dataclass(frozen=True)
class CageExecution:
    result: CageProofResult
    trace: tuple[Mapping[str, Any], ...]
    contacts: tuple[Mapping[str, Any], ...]
    reset_cache_audit: Mapping[str, Any]


class CageContactPolicy:
    """Per-finger point-Jacobian controller with force latch/reacquisition."""

    def __init__(self, config: Mapping[str, Any], active_mask: Sequence[bool]):
        self.cage = dict(config["cage"])
        self.force = dict(config["force"])
        self.active = np.asarray(active_mask, dtype=bool)
        self.latched = np.zeros(5, dtype=bool)
        self.below_keep = np.zeros(5, dtype=np.int64)

    def command(
        self,
        *,
        tip_positions: np.ndarray,
        targets: np.ndarray,
        normals: np.ndarray,
        forces: np.ndarray,
        jacobians: np.ndarray,
        hand_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        pair_contact: np.ndarray,
    ) -> np.ndarray:
        entered = self.active & pair_contact & (forces > float(self.force["formal_contact_n"]))
        self.latched |= entered
        below = self.active & self.latched & (
            (~pair_contact) | (forces < float(self.force["keep_contact_n"]))
        )
        self.below_keep[below] += 1
        self.below_keep[~below] = 0
        lost = self.below_keep >= int(self.force["loss_samples"])
        self.latched[lost] = False
        self.below_keep[lost] = 0

        delta = np.zeros(20, dtype=np.float64)
        for finger in np.flatnonzero(self.active):
            normal = np.asarray(normals[finger], dtype=np.float64)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            position_error = np.asarray(targets[finger]) - np.asarray(tip_positions[finger])
            if self.latched[finger]:
                tangential = position_error - float(np.dot(position_error, normal)) * normal
                force_error = float(self.force["target_n"]) - float(forces[finger])
                cartesian = 0.45 * tangential - 0.0005 * force_error * normal
            else:
                cartesian = 0.45 * position_error
            columns = np.asarray([finger + 5 * joint for joint in range(4)], dtype=np.int64)
            jacobian = np.asarray(jacobians[finger], dtype=np.float64)[:, columns]
            damping = float(self.cage["dls_damping"])
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + (damping**2) * np.eye(3), cartesian
            )
            delta[columns] = dq
        limit = float(self.cage["runtime_joint_step_limit_rad"])
        delta = np.clip(delta, -limit, limit)
        return np.clip(hand_q + delta, lower, upper) - hand_q


def run_cage_proof(
    env: Any,
    candidate: CageCandidate,
    config: Mapping[str, Any],
    *,
    trial_index: int,
    seed: int,
) -> CageExecution:
    """Run one fresh-reset proof without any post-reset state writes."""

    active_mask = _active_mask(candidate.finger_group)
    reset_q = np.asarray(candidate.preclose_q26, dtype=np.float64).reshape(1, 26)
    env.configure_privileged_control(
        reset_q,
        active_mask.reshape(1, 5),
        candidate_close_q20=np.asarray(candidate.closed_q26, dtype=np.float64).reshape(1, 26)[:, 6:],
        reset_seeds=[int(seed)],
    )
    env.reset()
    reset_cache = env.clear_reset_contact_cache_for_privileged_control()
    env.set_privileged_joint_targets(reset_q)
    policy = CageContactPolicy(config, active_mask)
    cage_cfg = config["cage"]
    force_cfg = config["force"]
    max_steps = int(config["runtime"]["candidate_trial_steps"])
    state = CageState.PRECONTACT
    state_steps = 0
    lock_steps = 0
    lift_progress = 0.0
    tracking_losses = 0
    applied_target = reset_q[0].copy()
    initial_object = None
    lift_relative = None
    trace: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    force_history: list[np.ndarray] = []
    intended_history: list[bool] = []
    failure = FailureCode.NONE
    table_support_final = True
    peak_force = 0.0
    max_drift = 0.0
    hold_completed = 0

    for frame_id in range(max_steps):
        env.step(env._zero_action)
        snapshot = env.privileged_snapshot()
        events = env.consume_forensic_contact_events()[0]
        if initial_object is None:
            initial_object = snapshot["object_pos_local"][0].copy()
        target_frames, normals = _runtime_targets(snapshot, candidate)
        pair_contact = _intended_pair_contact(candidate, events)
        table_support = _object_table_contact(events)
        table_support_final = table_support
        identified_non_target = _non_target_robot_contact(candidate, events)
        pair_instrumentation_mismatch = bool(
            np.any((snapshot["target_force_norms"][0] > float(force_cfg["formal_contact_n"])) & ~pair_contact)
        )
        unresolved = bool(snapshot["unresolved_contact"][0]) or not bool(snapshot["contact_report_available"][0])
        unresolved |= pair_instrumentation_mismatch
        current_forces = np.asarray(snapshot["target_force_norms"][0], dtype=np.float64)
        peak_force = max(peak_force, float(np.max(current_forces, initial=0.0)))
        hard_abort = peak_force >= float(force_cfg["hard_abort_n"])
        post_object_writes = int(snapshot["post_reset_object_writes"][0])
        post_wrist_writes = int(snapshot["post_reset_wrist_state_writes"][0])
        if post_object_writes or post_wrist_writes:
            failure = FailureCode.FORBIDDEN_STATE_WRITE
        elif unresolved:
            failure = FailureCode.INSTRUMENTATION_UNRESOLVED
        elif identified_non_target:
            failure = FailureCode.NON_TARGET_COLLISION
        elif hard_abort:
            failure = FailureCode.FORCE_ABORT

        actual_q = np.asarray(snapshot["joint_pos26"][0], dtype=np.float64)
        target_q = np.asarray(snapshot["joint_target26"][0], dtype=np.float64)
        wrist_error = np.abs(target_q[:6] - actual_q[:6])
        tracking_bad = bool(
            np.max(wrist_error[:3], initial=0.0) > float(cage_cfg["wrist_translation_error_limit_m"])
            or np.max(wrist_error[3:], initial=0.0)
            > np.deg2rad(float(cage_cfg["wrist_rotation_error_limit_deg"]))
        )
        tracking_losses = tracking_losses + 1 if tracking_bad else 0
        if tracking_losses >= int(cage_cfg["wrist_tracking_loss_samples"]):
            failure = FailureCode.WRIST_TRACKING_FAILED

        next_target = target_q.copy()
        hand_delta = np.zeros(20, dtype=np.float64)
        if failure == FailureCode.NONE:
            hand_delta = policy.command(
                tip_positions=snapshot["tip_pos_local"][0],
                targets=target_frames,
                normals=normals,
                forces=current_forces,
                jacobians=snapshot["tip_hand_jacobians"][0],
                hand_q=actual_q[6:],
                lower=snapshot["joint_lower26"][0, 6:],
                upper=snapshot["joint_upper26"][0, 6:],
                pair_contact=pair_contact,
            )
            next_target[6:] = actual_q[6:] + hand_delta

        topology_locked = _topology_locked(candidate, policy.latched)
        any_intended = bool(np.any(pair_contact & active_mask))
        if state == CageState.PRECONTACT and any_intended:
            state = CageState.FIRST_CONTACT
            state_steps = 0
        elif state == CageState.FIRST_CONTACT:
            if topology_locked:
                state = CageState.BALANCE_CONTACTS
                state_steps = 0
        elif state == CageState.BALANCE_CONTACTS:
            lock_steps = lock_steps + 1 if topology_locked else 0
            if lock_steps >= int(cage_cfg["cage_lock_steps"]):
                state = CageState.CAGE_LOCKED
                state_steps = 0
                lift_relative = snapshot["object_pos_local"][0] - snapshot["palm_pos_local"][0]
        elif state == CageState.CAGE_LOCKED:
            state = CageState.LIFT
            state_steps = 0
        elif state == CageState.LIFT:
            increment = min(
                float(cage_cfg["lift_increment_m"]),
                float(cage_cfg["lift_distance_m"]) - lift_progress,
            )
            next_target[2] += max(increment, 0.0)
            lift_progress += max(increment, 0.0)
            if lift_progress >= float(cage_cfg["lift_distance_m"]) - 1.0e-9:
                state = CageState.HOLD
                state_steps = 0
        elif state == CageState.HOLD:
            hold_completed += 1
            if hold_completed >= int(cage_cfg["post_lift_hold_steps"]):
                state = CageState.DONE

        if state in {CageState.PRECONTACT, CageState.FIRST_CONTACT} and frame_id >= 180 and not any_intended:
            failure = FailureCode.NO_TARGET_CONTACT
        if lift_relative is not None:
            relative = snapshot["object_pos_local"][0] - snapshot["palm_pos_local"][0]
            max_drift = max(max_drift, float(np.linalg.norm(relative - lift_relative)))
            if state in {CageState.LIFT, CageState.HOLD} and (
                max_drift > float(cage_cfg["relative_drift_limit_m"]) or not topology_locked
            ):
                failure = FailureCode.OBJECT_ESCAPED

        target_errors = np.linalg.norm(target_frames - snapshot["tip_pos_local"][0], axis=1)
        runtime_cage_margin = float(candidate.cage_margin_m - np.max(target_errors[active_mask], initial=0.0))
        if state in {CageState.LIFT, CageState.HOLD} and runtime_cage_margin <= 0.0:
            failure = FailureCode.OBJECT_ESCAPED

        state_steps += 1
        terminal = failure != FailureCode.NONE or state == CageState.DONE
        event_rows = [event.to_dict() for event in events]
        row = {
            "physics_frame_id": frame_id,
            "candidate_id": candidate.candidate_id,
            "trial_index": int(trial_index),
            "state": state.value,
            "applied_target26": target_q.tolist(),
            "next_target26": next_target.tolist(),
            "actual_joint26": actual_q.tolist(),
            "joint_error26": (target_q - actual_q).tolist(),
            "object_position": snapshot["object_pos_local"][0].tolist(),
            "object_quat_wxyz": snapshot["object_quat_wxyz"][0].tolist(),
            "object_linear_velocity": snapshot["object_lin_vel"][0].tolist(),
            "object_angular_velocity": snapshot["object_ang_vel"][0].tolist(),
            "palm_position": snapshot["palm_pos_local"][0].tolist(),
            "tip_positions": snapshot["tip_pos_local"][0].tolist(),
            "target_frame_positions": target_frames.tolist(),
            "target_normals_world": normals.tolist(),
            "target_force_norms": current_forces.tolist(),
            "target_force_xyz": snapshot["target_force_xyz"][0].tolist(),
            "all_force_xyz": snapshot["all_force_xyz"][0].tolist(),
            "table_force_xyz": snapshot["table_force_xyz"][0].tolist(),
            "ground_force_xyz": snapshot["ground_force_xyz"][0].tolist(),
            "latched_contacts": policy.latched.tolist(),
            "intended_pair_contacts": pair_contact.tolist(),
            "table_support": table_support,
            "runtime_cage_margin_m": runtime_cage_margin,
            "relative_drift_m": max_drift,
            "lift_progress_m": lift_progress,
            "peak_force_n": peak_force,
            "post_reset_object_root_writes": post_object_writes,
            "post_reset_wrist_state_writes": post_wrist_writes,
            "terminal": terminal,
            "failure_code": failure.value,
        }
        trace.append(row)
        contact_rows.append({"physics_frame_id": frame_id, "events": event_rows})
        force_history.append(current_forces.copy())
        intended_history.append(topology_locked)
        applied_target = next_target.copy()
        if terminal:
            break
        env.set_privileged_joint_targets(next_target.reshape(1, 26))

    final_snapshot = env.privileged_snapshot()
    if failure == FailureCode.NONE and state != CageState.DONE:
        failure = FailureCode.NO_TARGET_CONTACT if not any(intended_history) else FailureCode.OBJECT_ESCAPED
    lift_delta = float(final_snapshot["object_pos_local"][0, 2] - initial_object[2]) if initial_object is not None else 0.0
    contact_duty = float(np.mean(intended_history)) if intended_history else 0.0
    passed = bool(
        failure == FailureCode.NONE
        and state == CageState.DONE
        and lift_delta >= float(cage_cfg["lift_distance_m"])
        and not table_support_final
        and contact_duty >= float(cage_cfg["required_contact_duty"])
        and max_drift <= float(cage_cfg["relative_drift_limit_m"])
        and peak_force < float(force_cfg["hard_abort_n"])
        and int(final_snapshot["post_reset_object_writes"][0]) == 0
        and int(final_snapshot["post_reset_wrist_state_writes"][0]) == 0
    )
    if not passed and failure == FailureCode.NONE:
        failure = FailureCode.OBJECT_ESCAPED
    result = CageProofResult(
        candidate_id=candidate.candidate_id,
        trial_index=int(trial_index),
        passed=passed,
        failure_code=failure,
        topology=candidate.topology,
        lift_delta_m=lift_delta,
        table_support_final=table_support_final,
        intended_contact_duty=contact_duty,
        peak_force_n=peak_force,
        relative_drift_m=max_drift,
        hold_steps=hold_completed,
        post_reset_object_root_writes=int(final_snapshot["post_reset_object_writes"][0]),
        post_reset_wrist_state_writes=int(final_snapshot["post_reset_wrist_state_writes"][0]),
        metadata={
            "final_state": state.value,
            "physics_frames": len(trace),
            "first_physics_frame_recorded": bool(trace and trace[0]["physics_frame_id"] == 0),
            "sticky_used": False,
            "snap_used": False,
            "proxy_used": False,
            "teacher_used": False,
            "object_follow_used": False,
        },
    )
    if trace:
        trace[-1]["terminal"] = True
        trace[-1]["failure_code"] = failure.value
        trace[-1]["passed"] = passed
    return CageExecution(result, tuple(trace), tuple(contact_rows), reset_cache)


def _runtime_targets(snapshot: Mapping[str, np.ndarray], candidate: CageCandidate) -> tuple[np.ndarray, np.ndarray]:
    object_pos = np.asarray(snapshot["object_pos_local"][0], dtype=np.float64)
    object_rotation = _quat_wxyz_matrix(snapshot["object_quat_wxyz"][0])
    tip_rotations = np.asarray([_quat_wxyz_matrix(row) for row in snapshot["tip_quat_wxyz"][0]])
    targets = np.asarray(snapshot["tip_pos_local"][0], dtype=np.float64).copy()
    normals = np.zeros((5, 3), dtype=np.float64)
    for feature in candidate.control_features:
        finger = int(feature.finger_index) - 1
        contact = object_pos + object_rotation @ np.asarray(feature.target_position_object, dtype=np.float64)
        support = tip_rotations[finger] @ np.asarray(feature.local_support_vertex, dtype=np.float64)
        targets[finger] = contact - support
        normals[finger] = object_rotation @ np.asarray(feature.target_normal_object, dtype=np.float64)
    return targets, normals


def _intended_pair_contact(candidate: CageCandidate, events: Sequence[Any]) -> np.ndarray:
    result = np.zeros(5, dtype=bool)
    intended = set(candidate.intended_contact_links)
    for event in events:
        first = str(event.actor0)
        second = str(event.actor1)
        if "TargetObject" not in first and "TargetObject" not in second:
            continue
        for feature in candidate.control_features:
            if feature.link_name in intended and (feature.link_name in first or feature.link_name in second):
                result[int(feature.finger_index) - 1] = True
    return result


def _non_target_robot_contact(candidate: CageCandidate, events: Sequence[Any]) -> bool:
    intended = set(candidate.intended_contact_links)
    for event in events:
        first = str(event.actor0)
        second = str(event.actor1)
        if "Robot" not in first and "Robot" not in second:
            continue
        if "Table" in first or "Table" in second or "/ground" in first or "/ground" in second:
            return True
        if "Robot" in first and "Robot" in second:
            return True
        if "TargetObject" in first or "TargetObject" in second:
            robot_path = first if "Robot" in first else second
            if not any(link in robot_path for link in intended):
                return True
    return False


def _object_table_contact(events: Sequence[Any]) -> bool:
    return any(
        (("TargetObject" in str(event.actor0) and "Table" in str(event.actor1))
        or ("TargetObject" in str(event.actor1) and "Table" in str(event.actor0)))
        for event in events
    )


def _topology_locked(candidate: CageCandidate, latched: np.ndarray) -> bool:
    active = [int(value) - 1 for value in candidate.finger_group]
    if candidate.topology in {CageTopology.HOOK_THROUGH_FRAME, CageTopology.TWO_FINGER_BRACKET}:
        return bool(all(latched[index] for index in active))
    signs = [np.sign(candidate.control_features[offset].target_normal_object[1]) for offset in range(len(active))]
    positive = any(latched[index] and signs[offset] > 0 for offset, index in enumerate(active))
    negative = any(latched[index] and signs[offset] < 0 for offset, index in enumerate(active))
    return bool(positive and negative and sum(bool(latched[index]) for index in active) >= 2)


def _active_mask(group: str) -> np.ndarray:
    mask = np.zeros(5, dtype=bool)
    for value in str(group):
        index = int(value) - 1
        if not 0 <= index < 5:
            raise ValueError(f"invalid finger group {group}")
        mask[index] = True
    return mask


def _quat_wxyz_matrix(quat: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    return Rotation.from_quat((q[1], q[2], q[3], q[0])).as_matrix()
