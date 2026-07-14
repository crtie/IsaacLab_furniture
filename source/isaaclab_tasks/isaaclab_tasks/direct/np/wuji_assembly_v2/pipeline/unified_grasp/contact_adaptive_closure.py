"""Morphology-independent contact-adaptive closure policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

try:
    from .hand_morphology import HandMorphologySpec, MorphologyCalibration, ObjectGraspSpec
except ImportError:  # Support direct, Kit-free unit loading.
    from hand_morphology import HandMorphologySpec, MorphologyCalibration, ObjectGraspSpec


PRESHAPE = "PRESHAPE"
ENCLOSURE_ADVANCE = "ENCLOSURE_ADVANCE"
FIRST_CONTACT_HOLD = "FIRST_CONTACT_HOLD"
REMAINING_FINGER_ADVANCE = "REMAINING_FINGER_ADVANCE"
MULTI_CONTACT_BALANCE = "MULTI_CONTACT_BALANCE"
CONTROLLED_CLOSE = "CONTROLLED_CLOSE"
POST_CLOSE_HOLD = "POST_CLOSE_HOLD"
SLOW_LIFT = "SLOW_LIFT"
COMPLETE = "COMPLETE"
ABORT = "ABORT"


@dataclass(frozen=True)
class ClosureControllerConfig:
    contact_threshold_n: float = 0.05
    contact_keep_n: float = 0.035
    contact_loss_steps: int = 3
    soft_force_max_n: float = 1.0
    hard_abort_force_n: float = 5.0
    nominal_joint_step_rad: float = 0.001
    correction_joint_step_rad: float = 0.0005
    dual_hold_window_steps: int = 30
    dual_hold_required_steps: int = 24
    post_close_hold_steps: int = 30
    max_local_effect_updates: int = 2
    dls_damping: float = 1.0e-3


@dataclass(frozen=True)
class ClosureObservation:
    hand_q: tuple[float, ...]
    target_forces_n: Mapping[str, float]
    target_force_xyz_n: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    object_displacement_m: float = 0.0
    object_speed_mps: float = 0.0
    identified_non_target_contact_force_n: float = 0.0
    unresolved_unfiltered_residual_n: float = 0.0
    workspace_clamp_m: float = 0.0
    table_supported: bool = True
    object_lift_m: float = 0.0
    object_to_hand_relative_drift_m: float = 0.0


@dataclass(frozen=True)
class ClosureCommand:
    state: str
    hand_delta_q: tuple[float, ...]
    wrist_delta_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wrist_delta_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_kind: str = "HOLD"
    termination_reason: str = ""
    active_contacts: tuple[str, ...] = ()
    local_effect_update_requested: bool = False


class ContactAdaptiveClosure:
    """Generic closure engine; all object differences arrive through a spec."""

    def __init__(
        self,
        morphology: HandMorphologySpec,
        calibration: MorphologyCalibration,
        object_spec: ObjectGraspSpec,
        *,
        active_finger_group: tuple[str, ...] | None = None,
        config: ClosureControllerConfig | None = None,
        initial_state: str = PRESHAPE,
    ) -> None:
        morphology.validate()
        calibration.validate_for(morphology)
        object_spec.validate(morphology)
        self.morphology = morphology
        self.calibration = calibration
        self.object_spec = object_spec
        self.config = config or ClosureControllerConfig(
            soft_force_max_n=object_spec.force_band_n[1],
            hard_abort_force_n=object_spec.hard_abort_force_n,
        )
        self.active_fingers = active_finger_group or object_spec.active_finger_groups[0]
        self.state = initial_state
        self._latched = {finger: False for finger in self.active_fingers}
        self._below_keep = {finger: 0 for finger in self.active_fingers}
        self._dual_history: list[bool] = []
        self._first_contact_seen = False
        self._post_close_steps = 0
        self._effect_updates = 0
        self._close_direction = self._build_close_direction()
        self._effect = np.asarray(calibration.effect_matrix_30x20, dtype=np.float64)

    def update(self, observation: ClosureObservation) -> ClosureCommand:
        cfg = self.config
        q = np.asarray(observation.hand_q, dtype=np.float64)
        if q.shape != (20,) or not np.isfinite(q).all():
            return self._abort("invalid_hand_state")
        forces = {finger: float(observation.target_forces_n.get(finger, 0.0)) for finger in self.active_fingers}
        contacts = self._update_contacts(forces)
        first_contact_now = bool(contacts) and not self._first_contact_seen
        if contacts:
            self._first_contact_seen = True
        if self._first_contact_seen:
            self._dual_history.append(len(contacts) == len(self.active_fingers))
            self._dual_history = self._dual_history[-cfg.dual_hold_window_steps :]
        peak = max(forces.values(), default=0.0)
        if peak >= cfg.hard_abort_force_n:
            return self._abort("hard_force_abort", contacts)
        if observation.identified_non_target_contact_force_n > cfg.contact_threshold_n:
            return self._abort("non_target_contact", contacts)
        if observation.workspace_clamp_m > 1.0e-9:
            return self._abort("workspace_clamp", contacts)
        if observation.object_displacement_m > self.object_spec.preclose_object_motion_limit_m and self.state not in {
            SLOW_LIFT,
            COMPLETE,
        }:
            return self._abort("preclose_object_motion_limit", contacts)
        if peak >= cfg.soft_force_max_n:
            return self._release_overforce(forces, contacts)

        if self.state == PRESHAPE:
            if len(contacts) < len(self.active_fingers):
                return self._abort("precontact_executor_required", contacts)
            self.state = CONTROLLED_CLOSE
            return self._command(np.zeros(20), "PRECONTACT_HANDOFF_ACCEPTED", contacts)
        if len(contacts) == 0:
            return self._abort("all_target_contact_lost", contacts)
        if len(contacts) == 1:
            return self._command(np.zeros(20), "CONTACT_LOSS_HOLD", contacts)

        if self.state in {CONTROLLED_CLOSE, POST_CLOSE_HOLD, SLOW_LIFT}:
            return self._post_contact_update(observation, forces, contacts)
        self.state = MULTI_CONTACT_BALANCE
        if len(self._dual_history) >= cfg.dual_hold_window_steps and sum(self._dual_history) >= cfg.dual_hold_required_steps:
            self.state = CONTROLLED_CLOSE
            return self._command(np.zeros(20), "DUAL_HOLD_ACQUIRED", contacts)
        return self._command(self._balance_delta(forces), "MULTI_CONTACT_BALANCE", contacts)

    def _post_contact_update(
        self, observation: ClosureObservation, forces: Mapping[str, float], contacts: tuple[str, ...]
    ) -> ClosureCommand:
        if self.state == CONTROLLED_CLOSE:
            target = np.asarray(self.morphology.close_reference_q) - np.asarray(observation.hand_q)
            active = self._mask_for_fingers(set(self.active_fingers))
            delta = self._bounded(target * active)
            if np.max(np.abs(delta)) <= 1.0e-6 or all(force >= self.object_spec.force_band_n[0] for force in forces.values()):
                self.state = POST_CLOSE_HOLD
                self._post_close_steps = 0
                return self._command(np.zeros(20), "CLOSE_SUPPORT_ACQUIRED", contacts)
            return self._command(delta, "CONTROLLED_CLOSE_ADVANCE", contacts)
        if self.state == POST_CLOSE_HOLD:
            self._post_close_steps += 1
            if self._post_close_steps >= self.config.post_close_hold_steps:
                self.state = SLOW_LIFT
                return self._command(np.zeros(20), "POST_CLOSE_HOLD_ACQUIRED", contacts)
            return self._command(self._balance_delta(forces), "POST_CLOSE_ACTIVE_HOLD", contacts)
        if self.state == SLOW_LIFT:
            if (
                observation.object_lift_m >= self.object_spec.lift_target_m
                and not observation.table_supported
                and observation.object_to_hand_relative_drift_m <= 0.010
            ):
                self.state = COMPLETE
                return self._command(np.zeros(20), "PHYSICAL_LIFT_COMPLETE", contacts)
            return ClosureCommand(
                state=self.state,
                hand_delta_q=tuple(float(x) for x in self._balance_delta(forces)),
                wrist_delta_xyz_m=tuple(0.00025 * float(x) for x in self.object_spec.lift_direction_xyz),
                action_kind="SLOW_LIFT_ADVANCE",
                active_contacts=contacts,
            )
        return self._command(np.zeros(20), "HOLD", contacts)

    def _update_contacts(self, forces: Mapping[str, float]) -> tuple[str, ...]:
        cfg = self.config
        for finger, force in forces.items():
            if force > cfg.contact_threshold_n:
                self._latched[finger] = True
                self._below_keep[finger] = 0
            elif self._latched[finger] and force < cfg.contact_keep_n:
                self._below_keep[finger] += 1
                if self._below_keep[finger] >= cfg.contact_loss_steps:
                    self._latched[finger] = False
            else:
                self._below_keep[finger] = 0
        return tuple(finger for finger in self.active_fingers if self._latched[finger])

    def _build_close_direction(self) -> np.ndarray:
        delta = np.asarray(self.morphology.close_reference_q) - np.asarray(self.morphology.preshape_q)
        return delta * self._mask_for_fingers(set(self.active_fingers))

    def _mask_for_fingers(self, fingers: set[str]) -> np.ndarray:
        mask = np.zeros(20, dtype=np.float64)
        for finger in fingers:
            mask[list(self.morphology.finger_joint_groups[finger])] = 1.0
        return mask

    def _remaining_finger_delta(self, contacted: set[str]) -> np.ndarray:
        remaining = set(self.active_fingers) - contacted
        return self._bounded(self._close_direction * self._mask_for_fingers(remaining))

    def _balance_delta(self, forces: Mapping[str, float]) -> np.ndarray:
        desired = np.zeros(30, dtype=np.float64)
        target = 0.5 * sum(self.object_spec.force_band_n)
        for finger in self.active_fingers:
            tip_index = int(finger.replace("finger", "")) - 1
            axis = np.asarray(self.object_spec.closure_axes_by_finger[finger], dtype=np.float64)
            desired[tip_index * 6 : tip_index * 6 + 3] = axis * (target - forces[finger]) * 1.0e-4
        active_indices = sorted(
            index for finger in self.active_fingers for index in self.morphology.finger_joint_groups[finger]
        )
        jacobian = self._effect[:, active_indices]
        lhs = jacobian.T @ jacobian + self.config.dls_damping * np.eye(len(active_indices))
        rhs = jacobian.T @ desired
        solution = np.linalg.solve(lhs, rhs) if active_indices else np.zeros(0)
        delta = np.zeros(20, dtype=np.float64)
        delta[active_indices] = solution
        return self._bounded(delta, self.config.correction_joint_step_rad)

    def _release_overforce(self, forces: Mapping[str, float], contacts: tuple[str, ...]) -> ClosureCommand:
        finger = max(forces, key=forces.get)
        delta = -self._close_direction * self._mask_for_fingers({finger})
        return self._command(self._bounded(delta, self.config.correction_joint_step_rad), "RELEASE_OVERFORCE", contacts)

    def _bounded(self, delta: np.ndarray, limit: float | None = None) -> np.ndarray:
        bound = self.config.nominal_joint_step_rad if limit is None else float(limit)
        return np.clip(np.asarray(delta, dtype=np.float64), -bound, bound)

    def _command(self, delta: np.ndarray, kind: str, contacts: tuple[str, ...]) -> ClosureCommand:
        return ClosureCommand(
            state=self.state,
            hand_delta_q=tuple(float(x) for x in delta),
            action_kind=kind,
            active_contacts=contacts,
        )

    def _abort(self, reason: str, contacts: tuple[str, ...] = ()) -> ClosureCommand:
        self.state = ABORT
        return ClosureCommand(
            state=self.state,
            hand_delta_q=(0.0,) * 20,
            action_kind="ABORT",
            termination_reason=reason,
            active_contacts=contacts,
        )
