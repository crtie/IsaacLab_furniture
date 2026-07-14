"""Low-dimensional programs and templates for near-grasp physics search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROGRAM_DIM = 16
WRIST_SLICE = slice(0, 6)
LATENT_SLICE = slice(6, 12)
TIMING_SLICE = slice(12, 16)
WUJI_HAND_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}" for joint in range(1, 5) for finger in range(1, 6)
)


@dataclass(frozen=True)
class GraspProgramBounds:
    """Physical bounds for all 16 continuous program parameters."""

    lower: tuple[float, ...] = (
        -0.015,
        -0.015,
        -0.008,
        -np.deg2rad(15.0),
        -np.deg2rad(15.0),
        -np.deg2rad(15.0),
        -2.5,
        -2.5,
        -2.5,
        -2.5,
        -2.5,
        -2.5,
        40.0,
        0.25,
        8.0,
        8.0,
    )
    upper: tuple[float, ...] = (
        0.015,
        0.015,
        0.008,
        np.deg2rad(15.0),
        np.deg2rad(15.0),
        np.deg2rad(15.0),
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        140.0,
        1.5,
        48.0,
        64.0,
    )

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != (PROGRAM_DIM,) or upper.shape != (PROGRAM_DIM,):
            raise ValueError(f"GraspProgram bounds must both have shape ({PROGRAM_DIM},)")
        if np.any(lower >= upper):
            raise ValueError("Every GraspProgram lower bound must be below its upper bound")
        return lower, upper

    def clip(self, values: Sequence[float]) -> np.ndarray:
        lower, upper = self.arrays()
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (PROGRAM_DIM,):
            raise ValueError(f"Expected {PROGRAM_DIM} values, got shape {vector.shape}")
        return np.clip(vector, lower, upper)


@dataclass(frozen=True)
class GraspProgram:
    """A 16D continuous program plus one categorical acquisition template."""

    values: tuple[float, ...]
    template_id: int

    def __post_init__(self) -> None:
        if len(self.values) != PROGRAM_DIM:
            raise ValueError(f"GraspProgram requires {PROGRAM_DIM} values")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=np.float64))):
            raise ValueError("GraspProgram contains a non-finite value")
        if int(self.template_id) < 0:
            raise ValueError("template_id must be non-negative")

    @classmethod
    def from_vector(
        cls,
        values: Sequence[float],
        template_id: int,
        *,
        bounds: GraspProgramBounds | None = None,
    ) -> "GraspProgram":
        vector = (bounds or GraspProgramBounds()).clip(values)
        vector[12] = np.rint(vector[12])
        vector[14:16] = np.rint(vector[14:16])
        return cls(tuple(float(value) for value in vector), int(template_id))

    @property
    def wrist_residual_xyz_m(self) -> tuple[float, float, float]:
        return tuple(self.values[:3])

    @property
    def wrist_residual_rpy_rad(self) -> tuple[float, float, float]:
        return tuple(self.values[3:6])

    @property
    def hand_latent6(self) -> tuple[float, ...]:
        return tuple(self.values[LATENT_SLICE])

    @property
    def approach_steps(self) -> int:
        return int(round(self.values[12]))

    @property
    def closure_speed_multiplier(self) -> float:
        return float(self.values[13])

    @property
    def first_contact_hold_steps(self) -> int:
        return int(round(self.values[14]))

    @property
    def lift_delay_steps(self) -> int:
        return int(round(self.values[15]))

    def to_dict(self) -> dict[str, object]:
        return {
            "continuous16": list(self.values),
            "template_id": int(self.template_id),
            "wrist_residual_xyz_m": list(self.wrist_residual_xyz_m),
            "wrist_residual_rpy_rad": list(self.wrist_residual_rpy_rad),
            "hand_latent6": list(self.hand_latent6),
            "approach_steps": self.approach_steps,
            "closure_speed_multiplier": self.closure_speed_multiplier,
            "first_contact_hold_steps": self.first_contact_hold_steps,
            "lift_delay_steps": self.lift_delay_steps,
        }


@dataclass(frozen=True)
class GraspTemplate:
    """Categorical coarse seed, preshape profile, and active-finger mask."""

    template_id: int
    name: str
    dex_pose_relative_object_xyz_m: tuple[float, float, float]
    dex_quat_wxyz: tuple[float, float, float, float]
    preshape_profile: str
    active_finger_mask: tuple[bool, bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if len(self.active_finger_mask) != 5 or not any(self.active_finger_mask):
            raise ValueError("active_finger_mask must select at least one of five fingers")
        quat = np.asarray(self.dex_quat_wxyz, dtype=np.float64)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1.0e-6:
            raise ValueError("dex_quat_wxyz must be a finite non-zero quaternion")


def default_screw1_templates() -> tuple[GraspTemplate, ...]:
    """Templates around the frozen caging plan, without broad trajectory search."""

    # The relative center and quaternion are measured from the validated
    # bank_xcorr_rpy1 plan.  The three y tiers are the already-audited coarse
    # seeds; CEM searches only bounded residuals around them.
    base_xz = (-0.035, 0.008)
    quat = (0.0028643785382516675, -0.9846748480757688, -0.016233453999339422, -0.17361945131177692)
    rows: list[GraspTemplate] = []
    for y_index, y_m in enumerate((-0.027, -0.012, -0.042)):
        for profile_index, (profile, mask) in enumerate(
            (
                ("retargeted_pinch", (False, False, True, True, False)),
                ("retargeted_straddle", (False, True, True, True, False)),
            )
        ):
            template_id = y_index * 2 + profile_index
            rows.append(
                GraspTemplate(
                    template_id=template_id,
                    name=f"screw1_frozen_y{y_m:+.3f}_{profile}",
                    dex_pose_relative_object_xyz_m=(base_xz[0], y_m, base_xz[1]),
                    dex_quat_wxyz=quat,
                    preshape_profile=profile,
                    active_finger_mask=mask,
                )
            )
    return tuple(rows)


def grasp_templates_from_config(rows: Sequence[Mapping[str, Any]]) -> tuple[GraspTemplate, ...]:
    templates = tuple(
        GraspTemplate(
            template_id=int(row["id"]),
            name=str(row["name"]),
            dex_pose_relative_object_xyz_m=tuple(float(value) for value in row["relative_xyz_m"]),
            dex_quat_wxyz=tuple(float(value) for value in row["quat_wxyz"]),
            preshape_profile=str(row["preshape"]),
            active_finger_mask=tuple(bool(value) for value in row["active_fingers"]),
        )
        for row in rows
    )
    if [template.template_id for template in templates] != list(range(len(templates))):
        raise ValueError("configured template IDs must be contiguous from zero")
    return templates


class ProgramPhase(IntEnum):
    COARSE_ACQUISITION = 0
    WRIST_RESIDUAL_APPROACH = 1
    LATENT_CLOSURE = 2
    FIRST_CONTACT_HOLD = 3
    REMAINING_FINGER_PROGRESSION = 4
    MULTI_CONTACT_HOLD = 5
    CONTROLLED_CLOSE = 6
    SLOW_LIFT = 7
    COMPLETE = 8
    INVALID = 9


class ProgramTermination(IntEnum):
    NONE = 0
    COMPLETE = 1
    HARD_FORCE_ABORT = 2
    UNRESOLVED_CONTACT = 3
    LATENT_CLOSURE_EXHAUSTED = 4
    REMAINING_CONTACT_TIMEOUT = 5
    LIFT_CONTACT_LOSS = 6
    FLYOUT = 7
    PENETRATION = 8
    EPISODE_TIMEOUT = 9


class FirstContactFreezer:
    """Freeze contacted fingers with hysteresis and bounded reacquisition."""

    def __init__(
        self,
        joint_names: Iterable[str] = WUJI_HAND_JOINT_NAMES,
        threshold_n: float = 0.05,
        keep_threshold_n: float = 0.035,
        loss_samples: int = 3,
    ):
        self.joint_names = tuple(str(name) for name in joint_names)
        self.threshold_n = float(threshold_n)
        self.keep_threshold_n = float(keep_threshold_n)
        self.loss_samples = int(loss_samples)
        if not 0.0 < self.keep_threshold_n < self.threshold_n or self.loss_samples < 1:
            raise ValueError("FirstContactFreezer hysteresis is invalid")
        self._finger_by_joint = np.asarray([_finger_index(name) for name in self.joint_names], dtype=np.int64)
        self._latched = np.zeros(5, dtype=bool)
        self._below_keep = np.zeros(5, dtype=np.int64)

    @property
    def latched_fingers(self) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self._latched)

    def reset(self) -> None:
        self._latched[:] = False
        self._below_keep[:] = 0

    def apply(
        self,
        nominal_delta20: Sequence[float],
        target_force_norms5: Sequence[float],
        active_finger_mask5: Sequence[bool],
    ) -> np.ndarray:
        delta = np.asarray(nominal_delta20, dtype=np.float64).copy()
        forces = np.asarray(target_force_norms5, dtype=np.float64)
        active = np.asarray(active_finger_mask5, dtype=bool)
        if delta.shape != (len(self.joint_names),) or forces.shape != (5,) or active.shape != (5,):
            raise ValueError("FirstContactFreezer received an invalid shape")
        self._latched |= active & (forces > self.threshold_n)
        below = active & self._latched & (forces < self.keep_threshold_n)
        self._below_keep[below] += 1
        self._below_keep[~below] = 0
        lost = self._below_keep >= self.loss_samples
        self._latched[lost] = False
        self._below_keep[lost] = 0
        self._latched[~active] = False
        for joint_index, finger_index in enumerate(self._finger_by_joint):
            if not active[finger_index] or self._latched[finger_index]:
                delta[joint_index] = 0.0
        return delta


def resolve_preshape_profile(
    profile: str,
    preshape_q20: Sequence[float],
    close_reference_q20: Sequence[float],
    active_finger_mask5: Sequence[bool],
) -> np.ndarray:
    """Resolve a template profile into an executable morphology-correct target."""

    preshape = np.asarray(preshape_q20, dtype=np.float64)
    close = np.asarray(close_reference_q20, dtype=np.float64)
    active = np.asarray(active_finger_mask5, dtype=bool)
    if preshape.shape != (20,) or close.shape != (20,) or active.shape != (5,):
        raise ValueError("preshape profile inputs have invalid shapes")
    factors = {"retargeted_pinch": 0.12, "retargeted_straddle": 0.04}
    if profile not in factors:
        raise ValueError(f"unknown preshape profile {profile!r}")
    joint_active = np.asarray([active[_finger_index(name)] for name in WUJI_HAND_JOINT_NAMES], dtype=bool)
    target = preshape.copy()
    target[joint_active] += factors[profile] * (close[joint_active] - preshape[joint_active])
    return target


def _finger_index(joint_name: str) -> int:
    marker = "right_finger"
    start = joint_name.find(marker)
    if start < 0:
        raise ValueError(f"Cannot resolve Wuji finger from joint name {joint_name!r}")
    digit = joint_name[start + len(marker) : start + len(marker) + 1]
    if digit not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Cannot resolve Wuji finger from joint name {joint_name!r}")
    return int(digit) - 1
