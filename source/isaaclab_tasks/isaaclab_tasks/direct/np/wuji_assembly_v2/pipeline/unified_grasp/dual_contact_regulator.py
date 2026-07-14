"""Pure policy for bounded Screw1 dual-contact regulation.

The policy owns contact hysteresis and state transitions.  Isaac-specific
action execution, target snapshots, and transactional rollback stay in the
v2 baseline adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


REGULATE_FIRST_ANCHOR = "REGULATE_FIRST_ANCHOR"
ACQUIRE_SECOND_CONTACT = "ACQUIRE_SECOND_CONTACT"
DUAL_CONTACT_REGULATION = "DUAL_CONTACT_REGULATION"
CONTROLLED_CLOSE = "CONTROLLED_CLOSE"
SLOW_LIFT = "SLOW_LIFT"
COMPLETE = "COMPLETE"
ABORTED = "ABORTED"


@dataclass(frozen=True)
class DualContactRegulatorConfig:
    contact_on_n: float = 0.05
    contact_keep_n: float = 0.035
    contact_loss_confirm_steps: int = 3
    anchor_force_min_n: float = 0.08
    anchor_force_max_n: float = 0.30
    anchor_force_target_n: float = 0.15
    soft_force_max_n: float = 1.0
    hard_abort_force_n: float = 5.0
    finger_step_rad: float = 0.001
    anchor_loss_handoff_steps: int = 8
    anchor_reacquire_attempts: int = 4
    handoff_contact_confirm_steps: int = 3
    handoff_max: int = 1
    stagnation_steps: int = 12
    stagnation_min_improvement_n: float = 0.005
    wrist_translation_probe_m: tuple[float, float] = (0.00010, 0.00025)
    wrist_rotation_probe_deg: tuple[float, float] = (0.10, 0.25)
    wrist_axis_order: tuple[str, ...] = ("y", "z", "x", "pitch", "roll", "yaw")


@dataclass(frozen=True)
class DualContactObservation:
    finger3_force_n: float
    finger4_force_n: float
    object_displacement_m: float = 0.0
    object_speed_mps: float = 0.0
    non_target_force_n: float = 0.0
    workspace_clamp_m: float = 0.0

    def force(self, finger: str) -> float:
        return self.finger3_force_n if finger == "finger3" else self.finger4_force_n


@dataclass(frozen=True)
class DualContactCommand:
    state: str
    action_kind: str
    anchor_finger: str
    opposing_finger: str
    finger_deltas: Mapping[str, float] = field(default_factory=dict)
    termination_reason: str = ""
    handoff: bool = False
    request_wrist_probe: bool = False
    formal_finger3_contact: bool = False
    formal_finger4_contact: bool = False
    latched_finger3_contact: bool = False
    latched_finger4_contact: bool = False


@dataclass(frozen=True)
class WristProbeCandidate:
    axis: str
    sign: int
    translation_xyz_m: tuple[float, float, float]
    rotation_rpy_deg: tuple[float, float, float]
    amplitude_level: int


class DualContactRegulator:
    """Deterministic contact policy with no simulator dependencies."""

    def __init__(
        self,
        config: DualContactRegulatorConfig | None = None,
        *,
        anchor_finger: str = "finger3",
        preferred_finger_directions: Mapping[str, float] | None = None,
        initial_anchor_latched: bool = True,
    ) -> None:
        self.config = config or DualContactRegulatorConfig()
        self.anchor_finger = anchor_finger
        self.opposing_finger = "finger4" if anchor_finger == "finger3" else "finger3"
        self.preferred_finger_directions = {
            "finger3": 1.0,
            "finger4": 1.0,
            **dict(preferred_finger_directions or {}),
        }
        self.state = REGULATE_FIRST_ANCHOR
        self._latched = {"finger3": False, "finger4": False}
        self._latched[anchor_finger] = bool(initial_anchor_latched)
        self._below_keep = {"finger3": 0, "finger4": 0}
        self._formal_streak = {"finger3": 0, "finger4": 0}
        self._anchor_loss_steps = 0
        self._reacquire_attempts = 0
        self._handoff_count = 0
        self._stagnation_count = 0
        self._best_min_force = 0.0

    @property
    def handoff_count(self) -> int:
        return self._handoff_count

    def set_state(self, state: str) -> None:
        self.state = state

    def update(self, observation: DualContactObservation, *, allow_wrist: bool) -> DualContactCommand:
        cfg = self.config
        forces = {
            "finger3": float(observation.finger3_force_n),
            "finger4": float(observation.finger4_force_n),
        }
        formal = {finger: force > cfg.contact_on_n for finger, force in forces.items()}
        for finger in ("finger3", "finger4"):
            self._formal_streak[finger] = self._formal_streak[finger] + 1 if formal[finger] else 0
            if formal[finger]:
                self._latched[finger] = True
                self._below_keep[finger] = 0
            elif self._latched[finger] and forces[finger] < cfg.contact_keep_n:
                self._below_keep[finger] += 1
                if self._below_keep[finger] >= cfg.contact_loss_confirm_steps:
                    self._latched[finger] = False
            else:
                self._below_keep[finger] = 0

        peak = max(forces.values())
        if peak >= cfg.hard_abort_force_n:
            self.state = ABORTED
            return self._command("ABORT", formal, "hard_force_abort")
        if observation.non_target_force_n > cfg.contact_on_n:
            self.state = ABORTED
            return self._command("ABORT", formal, "non_target_contact")
        if observation.workspace_clamp_m > 1.0e-9:
            self.state = ABORTED
            return self._command("ABORT", formal, "workspace_clamp")
        if observation.object_displacement_m > 0.005:
            self.state = ABORTED
            return self._command("ABORT", formal, "object_displacement_limit")

        anchor_force = forces[self.anchor_finger]
        opposing_force = forces[self.opposing_finger]
        if peak > cfg.soft_force_max_n:
            finger = "finger3" if forces["finger3"] >= forces["finger4"] else "finger4"
            return self._finger_command("RELEASE_OVERFORCE", finger, -self._direction(finger), formal)

        if formal["finger3"] and formal["finger4"]:
            self.state = DUAL_CONTACT_REGULATION
            self._anchor_loss_steps = 0
            self._reacquire_attempts = 0
            return self._dual_regulation_command(forces, formal)

        min_force = min(forces.values())
        if min_force >= self._best_min_force + cfg.stagnation_min_improvement_n:
            self._best_min_force = min_force
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        if not self._latched[self.anchor_finger]:
            self._anchor_loss_steps += 1
            if self._reacquire_attempts < cfg.anchor_reacquire_attempts:
                self._reacquire_attempts += 1
                return self._finger_command(
                    "REACQUIRE_ANCHOR", self.anchor_finger, self._direction(self.anchor_finger), formal
                )
            can_handoff = bool(
                self._handoff_count < cfg.handoff_max
                and self._anchor_loss_steps >= cfg.anchor_loss_handoff_steps
                and self._formal_streak[self.opposing_finger] >= cfg.handoff_contact_confirm_steps
            )
            if can_handoff:
                self.anchor_finger, self.opposing_finger = self.opposing_finger, self.anchor_finger
                self._handoff_count += 1
                self._anchor_loss_steps = 0
                self._reacquire_attempts = 0
                self.state = REGULATE_FIRST_ANCHOR
                return self._command("HANDOFF", formal, handoff=True)
            self.state = ABORTED
            return self._command("ABORT", formal, "anchor_reacquisition_failed")

        self._anchor_loss_steps = 0
        self._reacquire_attempts = 0
        if anchor_force < cfg.anchor_force_min_n:
            self.state = REGULATE_FIRST_ANCHOR
            return self._finger_command(
                "INCREASE_ANCHOR_FORCE", self.anchor_finger, self._direction(self.anchor_finger), formal
            )
        if anchor_force > cfg.anchor_force_max_n:
            self.state = REGULATE_FIRST_ANCHOR
            return self._finger_command(
                "DECREASE_ANCHOR_FORCE", self.anchor_finger, -self._direction(self.anchor_finger), formal
            )

        self.state = ACQUIRE_SECOND_CONTACT
        if allow_wrist and self._stagnation_count >= cfg.stagnation_steps:
            self._stagnation_count = 0
            return self._command("REQUEST_WRIST_PROBE", formal, request_wrist_probe=True)
        if opposing_force <= cfg.anchor_force_max_n:
            return self._finger_command(
                "ADVANCE_OPPOSING", self.opposing_finger, self._direction(self.opposing_finger), formal
            )
        return self._command("HOLD", formal)

    def wrist_probe_candidates(self, amplitude_level: int = 0) -> list[WristProbeCandidate]:
        level = max(0, min(1, int(amplitude_level)))
        translation = self.config.wrist_translation_probe_m[level]
        rotation = self.config.wrist_rotation_probe_deg[level]
        candidates: list[WristProbeCandidate] = []
        for axis in self.config.wrist_axis_order:
            for sign in (1, -1):
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if axis in ("x", "y", "z"):
                    xyz[("x", "y", "z").index(axis)] = sign * translation
                else:
                    rpy[("roll", "pitch", "yaw").index(axis)] = sign * rotation
                candidates.append(WristProbeCandidate(axis, sign, tuple(xyz), tuple(rpy), level))
        return candidates

    def score_wrist_probe(
        self,
        before: DualContactObservation,
        after: DualContactObservation,
        *,
        anchor_finger: str | None = None,
    ) -> tuple[float, ...]:
        anchor = anchor_finger or self.anchor_finger
        opposing = "finger4" if anchor == "finger3" else "finger3"
        before_min = min(before.finger3_force_n, before.finger4_force_n)
        after_min = min(after.finger3_force_n, after.finger4_force_n)
        dual = float(after.finger3_force_n > self.config.contact_on_n and after.finger4_force_n > self.config.contact_on_n)
        anchor_error = abs(after.force(anchor) - self.config.anchor_force_target_n)
        return (
            dual,
            after_min - before_min,
            after.force(opposing) - before.force(opposing),
            -anchor_error,
            -after.object_displacement_m,
            -after.object_speed_mps,
            -max(after.finger3_force_n, after.finger4_force_n),
        )

    def _dual_regulation_command(
        self, forces: Mapping[str, float], formal: Mapping[str, bool]
    ) -> DualContactCommand:
        cfg = self.config
        for finger in ("finger3", "finger4"):
            if forces[finger] > cfg.anchor_force_max_n:
                return self._finger_command("DECREASE_DUAL_FORCE", finger, -self._direction(finger), formal)
        for finger in ("finger3", "finger4"):
            if forces[finger] < cfg.anchor_force_min_n:
                return self._finger_command("INCREASE_DUAL_FORCE", finger, self._direction(finger), formal)
        return self._command("HOLD_DUAL", formal)

    def _direction(self, finger: str) -> float:
        value = float(self.preferred_finger_directions.get(finger, 1.0))
        return self.config.finger_step_rad if value >= 0.0 else -self.config.finger_step_rad

    def _finger_command(
        self, action_kind: str, finger: str, delta: float, formal: Mapping[str, bool]
    ) -> DualContactCommand:
        return self._command(action_kind, formal, finger_deltas={finger: float(delta)})

    def _command(
        self,
        action_kind: str,
        formal: Mapping[str, bool],
        termination_reason: str = "",
        *,
        finger_deltas: Mapping[str, float] | None = None,
        handoff: bool = False,
        request_wrist_probe: bool = False,
    ) -> DualContactCommand:
        return DualContactCommand(
            state=self.state,
            action_kind=action_kind,
            anchor_finger=self.anchor_finger,
            opposing_finger=self.opposing_finger,
            finger_deltas=dict(finger_deltas or {}),
            termination_reason=termination_reason,
            handoff=handoff,
            request_wrist_probe=request_wrist_probe,
            formal_finger3_contact=bool(formal["finger3"]),
            formal_finger4_contact=bool(formal["finger4"]),
            latched_finger3_contact=bool(self._latched["finger3"]),
            latched_finger4_contact=bool(self._latched["finger4"]),
        )
