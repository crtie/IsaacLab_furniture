"""Object-space fingertip control with force feedback and reacquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContactControllerConfig:
    position_gain: float = 0.45
    force_gain_m_per_n: float = 0.0005
    damping: float = 0.01
    joint_step_limit_rad: float = 0.001
    wrist_translation_limit_m: float = 0.00025
    wrist_rotation_limit_rad: float = np.deg2rad(0.25)
    formal_contact_n: float = 0.05
    keep_contact_n: float = 0.035
    loss_samples: int = 3
    soft_force_n: float = 1.0
    hard_force_n: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_contact_n < self.formal_contact_n < self.soft_force_n < self.hard_force_n:
            raise ValueError("contact-controller force thresholds are invalid")
        if self.loss_samples < 1 or min(self.position_gain, self.force_gain_m_per_n, self.damping) <= 0.0:
            raise ValueError("contact-controller gains are invalid")


@dataclass(frozen=True)
class ContactControllerObservation:
    active_finger_mask: np.ndarray
    current_tip_positions: np.ndarray
    target_contact_positions: np.ndarray
    surface_normals_world: np.ndarray
    target_force_norms: np.ndarray
    target_force_n: float | np.ndarray
    fingertip_jacobians: np.ndarray
    current_hand_q: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    wrist_error6: np.ndarray | None = None
    non_target_contact: bool = False
    unresolved_contact: bool = False


@dataclass(frozen=True)
class ContactControllerCommand:
    hand_delta20: np.ndarray
    wrist_delta6: np.ndarray
    latched_contact: np.ndarray
    reacquiring: np.ndarray
    hard_abort: bool
    rejected: bool
    reason: str


class ObjectSpaceContactController:
    """Damped least-squares fingertip controller for arbitrary finger groups."""

    def __init__(self, config: ContactControllerConfig | None = None):
        self.config = config or ContactControllerConfig()
        self._latched = np.zeros(5, dtype=bool)
        self._below_keep = np.zeros(5, dtype=np.int64)

    @property
    def latched_contact(self) -> np.ndarray:
        return self._latched.copy()

    def reset(self) -> None:
        self._latched[:] = False
        self._below_keep[:] = 0

    def step(self, observation: ContactControllerObservation) -> ContactControllerCommand:
        active = np.asarray(observation.active_finger_mask, dtype=bool)
        tips = np.asarray(observation.current_tip_positions, dtype=np.float64)
        targets = np.asarray(observation.target_contact_positions, dtype=np.float64)
        normals = np.asarray(observation.surface_normals_world, dtype=np.float64)
        forces = np.asarray(observation.target_force_norms, dtype=np.float64)
        jacobians = np.asarray(observation.fingertip_jacobians, dtype=np.float64)
        q = np.asarray(observation.current_hand_q, dtype=np.float64)
        lower = np.asarray(observation.joint_lower, dtype=np.float64)
        upper = np.asarray(observation.joint_upper, dtype=np.float64)
        if active.shape != (5,) or tips.shape != (5, 3) or targets.shape != (5, 3) or normals.shape != (5, 3):
            raise ValueError("contact-controller fingertip shapes are invalid")
        if forces.shape != (5,) or jacobians.shape != (5, 3, 20) or q.shape != (20,) or lower.shape != (20,) or upper.shape != (20,):
            raise ValueError("contact-controller joint shapes are invalid")
        if not all(np.all(np.isfinite(value)) for value in (tips, targets, normals, forces, jacobians, q, lower, upper)):
            raise ValueError("contact-controller observation contains non-finite values")

        self._update_latches(active, forces)
        reacquiring = active & ~self._latched
        if float(np.max(forces[active], initial=0.0)) >= self.config.hard_force_n:
            return self._empty_command(reacquiring, hard_abort=True, rejected=True, reason="HARD_FORCE_ABORT")
        if observation.non_target_contact:
            return self._empty_command(reacquiring, hard_abort=False, rejected=True, reason="VERIFIED_NON_TARGET_CONTACT")
        if observation.unresolved_contact:
            return self._empty_command(
                reacquiring,
                hard_abort=False,
                rejected=True,
                reason="UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT",
            )

        target_force = np.broadcast_to(np.asarray(observation.target_force_n, dtype=np.float64), (5,))
        hand_delta = np.zeros(20, dtype=np.float64)
        for finger in np.flatnonzero(active):
            normal = normals[finger]
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            position_error = targets[finger] - tips[finger]
            force_error = float(target_force[finger] - forces[finger])
            cartesian = self.config.position_gain * position_error - self.config.force_gain_m_per_n * force_error * normal
            if self._latched[finger]:
                tangential = cartesian - float(np.dot(cartesian, normal)) * normal
                cartesian = tangential - self.config.force_gain_m_per_n * force_error * normal
            local_columns = np.asarray([finger + 5 * joint for joint in range(4)], dtype=np.int64)
            jacobian = jacobians[finger][:, local_columns]
            damping_matrix = jacobian @ jacobian.T + (self.config.damping**2) * np.eye(3)
            hand_delta[local_columns] = jacobian.T @ np.linalg.solve(damping_matrix, cartesian)

        hand_delta = np.clip(hand_delta, -self.config.joint_step_limit_rad, self.config.joint_step_limit_rad)
        next_q = np.clip(q + hand_delta, lower, upper)
        hand_delta = next_q - q
        wrist_delta = np.zeros(6, dtype=np.float64)
        if observation.wrist_error6 is not None:
            wrist_error = np.asarray(observation.wrist_error6, dtype=np.float64)
            if wrist_error.shape != (6,) or not np.all(np.isfinite(wrist_error)):
                raise ValueError("wrist error must be a finite 6D vector")
            wrist_delta[:3] = np.clip(
                wrist_error[:3], -self.config.wrist_translation_limit_m, self.config.wrist_translation_limit_m
            )
            wrist_delta[3:] = np.clip(
                wrist_error[3:], -self.config.wrist_rotation_limit_rad, self.config.wrist_rotation_limit_rad
            )
        reason = "SOFT_FORCE_REGULATION" if float(np.max(forces[active], initial=0.0)) > self.config.soft_force_n else "COMMAND"
        return ContactControllerCommand(
            hand_delta20=hand_delta,
            wrist_delta6=wrist_delta,
            latched_contact=self._latched.copy(),
            reacquiring=reacquiring,
            hard_abort=False,
            rejected=False,
            reason=reason,
        )

    def _update_latches(self, active: np.ndarray, forces: np.ndarray) -> None:
        entered = active & (forces > self.config.formal_contact_n)
        self._latched |= entered
        below = active & self._latched & (forces < self.config.keep_contact_n)
        self._below_keep[below] += 1
        self._below_keep[~below] = 0
        lost = self._below_keep >= self.config.loss_samples
        self._latched[lost] = False
        self._below_keep[lost] = 0
        self._latched[~active] = False
        self._below_keep[~active] = 0

    def _empty_command(self, reacquiring: np.ndarray, *, hard_abort: bool, rejected: bool, reason: str) -> ContactControllerCommand:
        return ContactControllerCommand(
            hand_delta20=np.zeros(20, dtype=np.float64),
            wrist_delta6=np.zeros(6, dtype=np.float64),
            latched_contact=self._latched.copy(),
            reacquiring=reacquiring.copy(),
            hard_abort=hard_abort,
            rejected=rejected,
            reason=reason,
        )
