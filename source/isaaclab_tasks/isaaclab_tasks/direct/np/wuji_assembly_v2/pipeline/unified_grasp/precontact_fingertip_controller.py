"""Object-relative fingertip control before physical grasp contact."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .closure_prior import ClosurePrior
    from .hand_morphology import HandMorphologySpec, ObjectGraspSpec
except ImportError:  # Support direct, Kit-free unit loading.
    from closure_prior import ClosurePrior
    from hand_morphology import HandMorphologySpec, ObjectGraspSpec


PRESHAPE = "PRESHAPE"
PRECONTACT_TIP_ALIGNMENT = "PRECONTACT_TIP_ALIGNMENT"
PRECONTACT_PINCH_APPROACH = "PRECONTACT_PINCH_APPROACH"
FIRST_CONTACT_FORCE_HOLD = "FIRST_CONTACT_FORCE_HOLD"
SECOND_FINGER_APPROACH = "SECOND_FINGER_APPROACH"
MULTI_CONTACT_BALANCE = "MULTI_CONTACT_BALANCE"
CONTROLLED_CLOSE = "CONTROLLED_CLOSE"
POST_CLOSE_HOLD = "POST_CLOSE_HOLD"
SLOW_LIFT = "SLOW_LIFT"
ABORTED = "ABORTED"


def quat_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1.0e-12:
        raise ValueError("object quaternion must be finite wxyz")
    w, x, y, z = q / np.linalg.norm(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def damped_least_squares(jacobian: np.ndarray, desired_tip_delta: np.ndarray, damping: float) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=np.float64)
    desired = np.asarray(desired_tip_delta, dtype=np.float64)
    if matrix.ndim != 2 or desired.shape != (matrix.shape[0],):
        raise ValueError("DLS dimensions do not agree")
    lhs = matrix @ matrix.T + float(damping) ** 2 * np.eye(matrix.shape[0], dtype=np.float64)
    return matrix.T @ np.linalg.solve(lhs, desired)


@dataclass(frozen=True)
class LocalActiveJacobian:
    active_fingers: tuple[str, ...]
    active_joint_indices: tuple[int, ...]
    matrix_6x8: tuple[tuple[float, ...], ...]
    positive_effects: tuple[tuple[float, ...], ...] = ()
    negative_effects: tuple[tuple[float, ...], ...] = ()
    joint_restore_error_rad: float = 0.0
    tip_restore_error_m: float = 0.0
    object_motion_m: float = 0.0
    target_actual_lag_steps: tuple[int, ...] = ()
    crosstalk_ratio: tuple[float, ...] = ()
    parked_prior_matrix_6x8: tuple[tuple[float, ...], ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.matrix_6x8, dtype=np.float64)

    def diagnostics(self, desired_motion: Sequence[float] | None = None) -> dict[str, Any]:
        matrix = self.matrix
        singular = np.linalg.svd(matrix, compute_uv=False) if matrix.shape == (6, 8) else np.zeros(0)
        rank = int(np.linalg.matrix_rank(matrix, tol=1.0e-7)) if matrix.shape == (6, 8) else 0
        minimum = float(singular[-1]) if singular.size else 0.0
        condition = float(singular[0] / minimum) if minimum > 0.0 else float("inf")
        projection_residual = 0.0
        if desired_motion is not None:
            desired = np.asarray(desired_motion, dtype=np.float64)
            if desired.shape != (6,):
                raise ValueError("desired_motion must contain six Cartesian components")
            projected = matrix @ np.linalg.lstsq(matrix, desired, rcond=None)[0]
            projection_residual = float(np.linalg.norm(desired - projected) / max(np.linalg.norm(desired), 1.0e-12))
        valid = bool(
            matrix.shape == (6, 8)
            and np.isfinite(matrix).all()
            and rank == 6
            and minimum >= 1.0e-4
            and condition <= 100.0
            and projection_residual <= 0.20
            and self.joint_restore_error_rad <= 2.0e-4
            and self.tip_restore_error_m <= 2.0e-4
            and self.object_motion_m <= 5.0e-4
        )
        return {
            "rank": rank,
            "singular_values": [float(value) for value in singular],
            "minimum_singular_value_m_per_rad": minimum,
            "condition_number": condition,
            "desired_motion_projection_residual": projection_residual,
            "joint_restore_error_rad": float(self.joint_restore_error_rad),
            "tip_restore_error_m": float(self.tip_restore_error_m),
            "object_motion_m": float(self.object_motion_m),
            "controllable": valid,
        }


@dataclass(frozen=True)
class PriorScheduleSample:
    runtime_step: int
    source_frame: float
    phase: str
    progress: float
    latent_speed_scale: float
    hold_requested: bool


class PriorScheduler:
    """Timing-only schedule from the final source segment (frames 134-261)."""

    def __init__(self, prior: ClosurePrior, max_steps: int = 180) -> None:
        self._step = 0
        self._max_steps = max(1, int(max_steps))
        latent = np.asarray(prior.normalized_latent_trajectory, dtype=np.float64)
        if latent.ndim != 2 or latent.shape[0] <= 134:
            raise ValueError("closure prior lacks the required post-reset latent segment")
        self._latent = latent
        self._phases = tuple(phase for phase in prior.phases if phase.end_frame >= 134 and phase.start_frame <= 261)

    @property
    def accepted_steps(self) -> int:
        return self._step

    @property
    def exhausted(self) -> bool:
        return self._step >= self._max_steps

    def peek(self) -> PriorScheduleSample:
        ratio = min(self._step / max(self._max_steps - 1, 1), 1.0)
        source = 134.0 + ratio * (261.0 - 134.0)
        frame = int(np.clip(round(source), 134, min(261, self._latent.shape[0] - 1)))
        phase = next((item.kind for item in self._phases if item.start_frame <= frame <= item.end_frame), "motion")
        previous = max(134, frame - 1)
        derivative = self._latent[frame, :2] - self._latent[previous, :2]
        segment_derivatives = np.diff(self._latent[134 : min(262, self._latent.shape[0]), :2], axis=0)
        reference = float(np.median(np.linalg.norm(segment_derivatives, axis=1))) if segment_derivatives.size else 1.0
        speed = float(np.clip(np.linalg.norm(derivative) / max(reference, 1.0e-9), 0.5, 1.5))
        return PriorScheduleSample(
            runtime_step=self._step,
            source_frame=source,
            phase=phase,
            progress=ratio,
            latent_speed_scale=speed,
            hold_requested=phase == "hold",
        )

    def accept(self) -> None:
        self._step = min(self._step + 1, self._max_steps)


@dataclass(frozen=True)
class PreContactObservation:
    hand_q: tuple[float, ...]
    hand_target_q: tuple[float, ...]
    fingertip_positions: Mapping[str, tuple[float, float, float]]
    object_position: tuple[float, float, float]
    object_quaternion_wxyz: tuple[float, float, float, float]
    target_forces_n: Mapping[str, float]
    object_displacement_m: float = 0.0
    object_speed_mps: float = 0.0
    workspace_clamp_m: float = 0.0
    identified_non_target_contact: bool = False
    unresolved_unfiltered_residual_n: float = 0.0


@dataclass(frozen=True)
class PreContactCommand:
    state: str
    hand_delta_q: tuple[float, ...]
    desired_tip_delta: tuple[float, ...]
    predicted_tip_delta: tuple[float, ...]
    active_contacts: tuple[str, ...]
    action_kind: str
    prior_phase: str
    prior_progress: float
    latent_speed_scale: float
    hold_requested: bool
    accepted_for_schedule: bool = False
    ready_for_close: bool = False
    termination_reason: str = ""


@dataclass(frozen=True)
class PreContactControllerConfig:
    contact_threshold_n: float = 0.05
    soft_force_max_n: float = 1.0
    hard_abort_force_n: float = 5.0
    alignment_cartesian_cap_m: float = 0.00015
    approach_cartesian_cap_m: float = 0.00020
    joint_step_cap_rad: float = 0.002
    absolute_joint_step_cap_rad: float = 0.003
    alignment_tolerance_m: float = 0.00035
    dls_damping: float = 1.0e-4
    dual_hold_window_steps: int = 30
    dual_hold_required_steps: int = 24
    max_relinearizations: int = 2
    unsafe_object_speed_mps: float = 0.05
    joint_limit_margin_rad: float = 0.002
    preshape_regularization_gain: float = 0.01


class PreContactFingertipController:
    """Generic object-frame controller; part names never affect policy."""

    def __init__(
        self,
        morphology: HandMorphologySpec,
        object_spec: ObjectGraspSpec,
        local_jacobian: LocalActiveJacobian,
        prior: ClosurePrior,
        *,
        active_finger_group: tuple[str, ...] | None = None,
        config: PreContactControllerConfig | None = None,
    ) -> None:
        morphology.validate()
        object_spec.validate(morphology)
        self.morphology = morphology
        self.object_spec = object_spec
        self.active_fingers = active_finger_group or object_spec.active_finger_groups[0]
        self.local_jacobian = local_jacobian
        self.config = config or PreContactControllerConfig()
        self.prior = PriorScheduler(prior)
        self.state = PRESHAPE
        self._dual_history: list[bool] = []
        self._first_contact_hold_done = False
        self._bad_prediction_steps = 0
        self.relinearization_count = 0

    def object_relative_targets(self, observation: PreContactObservation) -> tuple[dict[str, np.ndarray], np.ndarray]:
        rotation = quat_wxyz_to_matrix(observation.object_quaternion_wxyz)
        axis = rotation @ np.asarray(self.object_spec.pinch_axis_object, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        center = np.asarray(observation.object_position, dtype=np.float64) + rotation @ np.asarray(
            self.object_spec.contact_center_offset_object, dtype=np.float64
        )
        targets = {
            finger: center
            + float(self.object_spec.finger_side_by_name[finger])
            * float(self.object_spec.nominal_contact_offset_m)
            * axis
            for finger in self.active_fingers
        }
        return targets, axis

    def update(self, observation: PreContactObservation) -> PreContactCommand:
        cfg = self.config
        forces = {finger: float(observation.target_forces_n.get(finger, 0.0)) for finger in self.active_fingers}
        contacts = tuple(finger for finger in self.active_fingers if forces[finger] > cfg.contact_threshold_n)
        if max(forces.values(), default=0.0) >= cfg.hard_abort_force_n:
            return self._abort("hard_force_abort", contacts)
        if max(forces.values(), default=0.0) >= cfg.soft_force_max_n:
            return self._abort("soft_force_limit", contacts)
        if observation.identified_non_target_contact:
            return self._abort("identified_non_target_scene_contact", contacts)
        if observation.workspace_clamp_m > 1.0e-9:
            return self._abort("workspace_clamp", contacts)
        if observation.object_displacement_m > self.object_spec.preclose_object_motion_limit_m:
            return self._abort("preclose_object_motion_limit", contacts)
        if observation.object_speed_mps > cfg.unsafe_object_speed_mps:
            return self._abort("unsafe_object_velocity", contacts)

        sample = self.prior.peek()
        if self.prior.exhausted and len(contacts) < len(self.active_fingers):
            return self._abort("precontact_schedule_exhausted", contacts)
        if not contacts:
            self._first_contact_hold_done = False
        if self.state == PRESHAPE:
            self.state = PRECONTACT_TIP_ALIGNMENT
            return self._command(np.zeros(8), np.zeros(6), contacts, "PRESHAPE_REACHED", sample, accepted=True)
        if sample.hold_requested and not contacts:
            return self._command(np.zeros(8), np.zeros(6), contacts, "PRIOR_HOLD", sample, accepted=True)

        targets, axis = self.object_relative_targets(observation)
        current = {finger: np.asarray(observation.fingertip_positions[finger], dtype=np.float64) for finger in self.active_fingers}
        errors = {finger: targets[finger] - current[finger] for finger in self.active_fingers}
        if not contacts and self.state == PRECONTACT_TIP_ALIGNMENT:
            desired_rows = [errors[finger] - axis * float(np.dot(errors[finger], axis)) for finger in self.active_fingers]
            if max(float(np.linalg.norm(row)) for row in desired_rows) <= cfg.alignment_tolerance_m:
                self.state = PRECONTACT_PINCH_APPROACH
            else:
                return self._solve(
                    observation, desired_rows, contacts, "OBJECT_RELATIVE_TIP_ALIGNMENT", sample, cfg.alignment_cartesian_cap_m
                )
        if len(contacts) == len(self.active_fingers):
            self.state = MULTI_CONTACT_BALANCE
            self._dual_history.append(True)
            self._dual_history = self._dual_history[-cfg.dual_hold_window_steps :]
            ready = bool(
                len(self._dual_history) == cfg.dual_hold_window_steps
                and sum(self._dual_history) >= cfg.dual_hold_required_steps
                and observation.object_speed_mps <= 0.01
                and observation.object_displacement_m <= self.object_spec.preclose_object_motion_limit_m
            )
            return self._command(
                np.zeros(8), np.zeros(6), contacts, "ACTIVE_DUAL_CONTACT_HOLD", sample, accepted=True, ready=ready
            )
        self._dual_history.append(False)
        self._dual_history = self._dual_history[-cfg.dual_hold_window_steps :]
        if len(contacts) == 1:
            if not self._first_contact_hold_done:
                self._first_contact_hold_done = True
                self.state = FIRST_CONTACT_FORCE_HOLD
                return self._command(
                    np.zeros(8), np.zeros(6), contacts, "FIRST_CONTACT_FORCE_HOLD", sample, accepted=True
                )
            self.state = SECOND_FINGER_APPROACH
        else:
            self.state = PRECONTACT_PINCH_APPROACH
        desired_rows = []
        for finger in self.active_fingers:
            if finger in contacts:
                desired_rows.append(np.zeros(3))
                continue
            gain = float(self.object_spec.finger_approach_gain[finger])
            desired_rows.append(errors[finger] * gain)
        return self._solve(
            observation, desired_rows, contacts, "OBJECT_RELATIVE_PINCH_APPROACH", sample, cfg.approach_cartesian_cap_m
        )

    def _solve(
        self,
        observation: PreContactObservation,
        rows: Sequence[np.ndarray],
        contacts: tuple[str, ...],
        kind: str,
        sample: PriorScheduleSample,
        cartesian_cap: float,
    ) -> PreContactCommand:
        desired = np.concatenate([np.asarray(row, dtype=np.float64) for row in rows])
        phase_gain = 0.5 if sample.phase == "secondary_adjustment" else 1.0
        desired *= float(sample.latent_speed_scale) * phase_gain
        for index in range(len(self.active_fingers)):
            block = desired[index * 3 : index * 3 + 3]
            norm = float(np.linalg.norm(block))
            if norm > cartesian_cap:
                desired[index * 3 : index * 3 + 3] = block * (cartesian_cap / norm)
        active_columns = [
            col
            for col, joint in enumerate(self.local_jacobian.active_joint_indices)
            if not any(joint in self.morphology.finger_joint_groups[finger] for finger in contacts)
        ]
        jacobian = self.local_jacobian.matrix
        delta = np.zeros(8, dtype=np.float64)
        if active_columns:
            delta[np.asarray(active_columns)] = damped_least_squares(
                jacobian[:, active_columns], desired, self.config.dls_damping
            )
        q = np.asarray(observation.hand_target_q, dtype=np.float64)
        preshape = np.asarray(self.morphology.preshape_q, dtype=np.float64)
        lower = np.asarray([item[0] for item in self.morphology.joint_limits], dtype=np.float64)
        upper = np.asarray([item[1] for item in self.morphology.joint_limits], dtype=np.float64)
        for column, joint in enumerate(self.local_jacobian.active_joint_indices):
            if column not in active_columns:
                continue
            delta[column] -= self.config.preshape_regularization_gain * (q[joint] - preshape[joint])
            delta[column] = np.clip(
                delta[column],
                lower[joint] + self.config.joint_limit_margin_rad - q[joint],
                upper[joint] - self.config.joint_limit_margin_rad - q[joint],
            )
        limit = min(self.config.joint_step_cap_rad, self.config.absolute_joint_step_cap_rad)
        delta = np.clip(delta, -limit, limit)
        predicted = jacobian @ delta
        return self._command(delta, desired, contacts, kind, sample, predicted=predicted, accepted=False)

    def evaluate_step(
        self,
        command: PreContactCommand,
        actual_tip_delta: Sequence[float],
        error_before: float,
        error_after: float,
    ) -> dict[str, Any]:
        actual = np.asarray(actual_tip_delta, dtype=np.float64)
        predicted = np.asarray(command.predicted_tip_delta, dtype=np.float64)
        prediction_dot = float(np.dot(predicted, actual))
        improved = bool(error_after < error_before - 1.0e-7 and prediction_dot >= 0.0)
        self._bad_prediction_steps = 0 if improved else self._bad_prediction_steps + 1
        if improved:
            self.prior.accept()
        request = self._bad_prediction_steps >= 3
        return {
            "accepted": improved,
            "prediction_dot": prediction_dot,
            "error_before_m": float(error_before),
            "error_after_m": float(error_after),
            "error_reduction_m": float(error_before - error_after),
            "bad_prediction_steps": self._bad_prediction_steps,
            "rollback_requested": request,
            "relinearization_requested": request and self.relinearization_count < self.config.max_relinearizations,
        }

    def replace_jacobian(self, jacobian: LocalActiveJacobian) -> None:
        self.local_jacobian = jacobian
        self.relinearization_count += 1
        self._bad_prediction_steps = 0

    def _command(
        self,
        active_delta: np.ndarray,
        desired: np.ndarray,
        contacts: tuple[str, ...],
        kind: str,
        sample: PriorScheduleSample,
        *,
        predicted: np.ndarray | None = None,
        accepted: bool,
        ready: bool = False,
    ) -> PreContactCommand:
        full = np.zeros(20, dtype=np.float64)
        full[list(self.local_jacobian.active_joint_indices)] = active_delta
        if accepted:
            self.prior.accept()
        return PreContactCommand(
            state=self.state,
            hand_delta_q=tuple(float(value) for value in full),
            desired_tip_delta=tuple(float(value) for value in desired),
            predicted_tip_delta=tuple(float(value) for value in (predicted if predicted is not None else np.zeros(6))),
            active_contacts=contacts,
            action_kind=kind,
            prior_phase=sample.phase,
            prior_progress=sample.progress,
            latent_speed_scale=sample.latent_speed_scale,
            hold_requested=sample.hold_requested,
            accepted_for_schedule=accepted,
            ready_for_close=ready,
        )

    def _abort(self, reason: str, contacts: tuple[str, ...]) -> PreContactCommand:
        self.state = ABORTED
        sample = self.prior.peek()
        command = self._command(np.zeros(8), np.zeros(6), contacts, "ABORT", sample, accepted=False, ready=False)
        return replace(command, termination_reason=reason)
