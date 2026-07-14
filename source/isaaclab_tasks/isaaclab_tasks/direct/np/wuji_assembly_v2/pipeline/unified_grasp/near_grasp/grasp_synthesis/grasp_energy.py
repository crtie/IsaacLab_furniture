"""Geometry energy and search-only ranking, separate from strict success."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SearchMetrics:
    """Search guidance that cannot establish physical success."""

    hard_invalid: bool = False
    gate_progress: int = 0
    safe_target_contact_count: int = 0
    min_active_finger_contact_duty: float = 0.0
    mean_surface_distance_m: float = float("inf")
    opposition_quality: float = 0.0
    force_closure_quality: float = 0.0
    force_balance_error_n: float = float("inf")
    penetration_m: float = float("inf")
    table_collision_force_n: float = float("inf")
    relative_drift_m: float = float("inf")
    jerk_metric: float = float("inf")
    valid_contact_attribution: bool = True
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.gate_progress not in range(4):
            raise ValueError("gate_progress must be 0..3")
        if self.safe_target_contact_count < 0 or self.safe_target_contact_count > 5:
            raise ValueError("safe target contact count must be 0..5")
        if not 0.0 <= self.min_active_finger_contact_duty <= 1.0:
            raise ValueError("contact duty must be in [0,1]")

    def ranking_key(self) -> tuple[float, ...]:
        """Lexicographic key; safe contact always outranks no contact."""

        finite_penalty = 1.0e6

        def negative(value: float) -> float:
            return -float(value) if np.isfinite(value) else -finite_penalty

        return (
            float(not self.hard_invalid and self.valid_contact_attribution),
            float(self.gate_progress),
            float(self.safe_target_contact_count),
            float(self.min_active_finger_contact_duty),
            negative(self.mean_surface_distance_m),
            float(self.opposition_quality),
            float(self.force_closure_quality),
            negative(self.force_balance_error_n),
            negative(self.penetration_m),
            negative(self.table_collision_force_n),
            negative(self.relative_drift_m),
            negative(self.jerk_metric),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_physical_evaluation(cls, evaluation: Any) -> "SearchMetrics":
        target_evidence = bool(getattr(evaluation, "target_filtered_success_evidence", False))
        safe = bool(getattr(evaluation, "safe_force", False)) and not bool(getattr(evaluation, "hard_invalid", False))
        duty = float(getattr(evaluation, "simultaneous_contact_duty", 0.0))
        return cls(
            hard_invalid=bool(getattr(evaluation, "hard_invalid", False)),
            gate_progress=3 if bool(getattr(evaluation, "physical_lift_success", False)) else (2 if bool(getattr(evaluation, "stable_close", False)) else 0),
            safe_target_contact_count=int(target_evidence and safe),
            min_active_finger_contact_duty=duty,
            mean_surface_distance_m=0.0 if target_evidence else float("inf"),
            opposition_quality=duty,
            force_closure_quality=duty,
            force_balance_error_n=float(getattr(evaluation, "peak_target_force_n", 0.0)) if target_evidence else float("inf"),
            penetration_m=0.0,
            table_collision_force_n=0.0,
            relative_drift_m=float(getattr(evaluation, "relative_drift_m", float("inf"))),
            jerk_metric=float(getattr(evaluation, "jerk_metric", float("inf"))),
            valid_contact_attribution="unresolved_contact_truth" not in getattr(evaluation, "invalid_reasons", ()),
        )


@dataclass(frozen=True)
class GraspEnergyBreakdown:
    contact_distance: float
    opposition: float
    force_closure: float
    hand_object_penetration: float
    self_collision: float
    table_collision: float
    joint_limit: float
    path_collision: float

    @property
    def total(self) -> float:
        return float(
            50.0 * self.contact_distance
            + 4.0 * self.opposition
            + 4.0 * self.force_closure
            + 200.0 * self.hand_object_penetration
            + 200.0 * self.self_collision
            + 250.0 * self.table_collision
            + 2.0 * self.joint_limit
            + 250.0 * self.path_collision
        )

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


@dataclass(frozen=True)
class GravityWrenchResult:
    feasible: bool
    force_residual_n: float
    torque_residual_nm: float
    normal_forces_n: tuple[float, ...]
    friction_cone_edges: int
    solver_status: str
    quality: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gravity_wrench_feasibility(
    positions: Sequence[Sequence[float]],
    normals: Sequence[Sequence[float]],
    *,
    center_of_mass: Sequence[float],
    mass_kg: float,
    friction: float,
    normal_force_range_n: Sequence[float],
    gravity_object: Sequence[float] = (0.0, 0.0, -9.81),
    cone_edges: int = 8,
    force_tolerance_n: float = 1.0e-3,
    torque_tolerance_nm: float = 1.0e-5,
) -> GravityWrenchResult:
    """Check whether bounded friction-cone forces can balance gravity."""

    from scipy.optimize import linprog

    points = np.asarray(positions, dtype=np.float64)
    outward = np.asarray(normals, dtype=np.float64)
    center = np.asarray(center_of_mass, dtype=np.float64)
    gravity = np.asarray(gravity_object, dtype=np.float64)
    force_range = np.asarray(normal_force_range_n, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or outward.shape != points.shape or len(points) < 2:
        raise ValueError("gravity wrench requires matching contact [N,3] arrays")
    if center.shape != (3,) or gravity.shape != (3,) or force_range.shape != (2,):
        raise ValueError("gravity wrench center, gravity, or force range is invalid")
    if mass_kg <= 0.0 or friction < 0.0 or cone_edges < 4 or not 0.0 < force_range[0] < force_range[1]:
        raise ValueError("gravity wrench physical parameters are invalid")
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1.0e-12)
    generators = []
    contact_columns: list[list[int]] = [[] for _ in range(len(points))]
    for contact_index, (point, normal) in enumerate(zip(points, outward)):
        reference = np.asarray((0.0, 0.0, 1.0))
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.asarray((1.0, 0.0, 0.0))
        tangent_first = np.cross(normal, reference)
        tangent_first /= max(float(np.linalg.norm(tangent_first)), 1.0e-12)
        tangent_second = np.cross(normal, tangent_first)
        for edge in range(int(cone_edges)):
            angle = 2.0 * np.pi * float(edge) / float(cone_edges)
            direction = -normal + float(friction) * (
                np.cos(angle) * tangent_first + np.sin(angle) * tangent_second
            )
            contact_columns[contact_index].append(len(generators))
            generators.append(np.concatenate((direction, np.cross(point - center, direction))))
    wrench_matrix = np.stack(generators, axis=1)
    gravity_force = float(mass_kg) * gravity
    required = np.concatenate((-gravity_force, np.zeros(3, dtype=np.float64)))
    generator_count = wrench_matrix.shape[1]
    variable_count = generator_count + 12
    inequality_rows = []
    inequality_bounds = []
    for columns in contact_columns:
        row = np.zeros(variable_count, dtype=np.float64)
        row[columns] = 1.0
        inequality_rows.extend((row, -row))
        inequality_bounds.extend((force_range[1], -force_range[0]))
    scale = np.asarray(
        (force_tolerance_n, force_tolerance_n, force_tolerance_n, torque_tolerance_nm, torque_tolerance_nm, torque_tolerance_nm),
        dtype=np.float64,
    )
    scaled_wrench = wrench_matrix / scale.reshape(-1, 1)
    equality = np.concatenate((scaled_wrench, np.eye(6), -np.eye(6)), axis=1)
    objective = np.concatenate((np.full(generator_count, 1.0e-6), np.ones(12)))
    result = linprog(
        objective,
        A_ub=np.stack(inequality_rows),
        b_ub=np.asarray(inequality_bounds),
        A_eq=equality,
        b_eq=required / scale,
        bounds=(0.0, None),
        method="highs",
    )
    coefficients = (
        np.asarray(result.x[:generator_count], dtype=np.float64)
        if result.success
        else np.zeros(generator_count)
    )
    residual = wrench_matrix @ coefficients - required
    normal_forces = tuple(float(np.sum(coefficients[columns])) for columns in contact_columns)
    force_residual = float(np.linalg.norm(residual[:3]))
    torque_residual = float(np.linalg.norm(residual[3:]))
    feasible = bool(result.success and force_residual <= force_tolerance_n and torque_residual <= torque_tolerance_nm)
    if feasible:
        midpoint = 0.5 * float(np.sum(force_range))
        half_width = 0.5 * float(force_range[1] - force_range[0])
        margin = min(1.0 - abs(value - midpoint) / max(half_width, 1.0e-12) for value in normal_forces)
        quality = float(np.clip(0.5 + 0.5 * margin, 0.0, 1.0))
    else:
        quality = 0.0
    return GravityWrenchResult(
        feasible=feasible,
        force_residual_n=force_residual,
        torque_residual_nm=torque_residual,
        normal_forces_n=normal_forces,
        friction_cone_edges=int(cone_edges),
        solver_status=str(result.message),
        quality=quality,
    )


def opposition_quality(normals: Sequence[Sequence[float]]) -> float:
    values = np.asarray(normals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        raise ValueError("opposition quality requires at least two 3D normals")
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
    pair_scores = []
    for first in range(values.shape[0]):
        for second in range(first + 1, values.shape[0]):
            pair_scores.append(0.5 * (1.0 - float(np.dot(values[first], values[second]))))
    return float(np.clip(max(pair_scores), 0.0, 1.0))


def force_closure_proxy(
    positions: Sequence[Sequence[float]],
    normals: Sequence[Sequence[float]],
    *,
    friction: float,
) -> float:
    """Return a bounded wrench-conditioning proxy, never a success signal."""

    points = np.asarray(positions, dtype=np.float64)
    directions = -np.asarray(normals, dtype=np.float64)
    if points.shape != directions.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("force closure inputs must have matching [N,3] shape")
    center = np.mean(points, axis=0)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
    wrenches = np.concatenate((directions, np.cross(points - center, directions)), axis=1).T
    singular = np.linalg.svd(wrenches, compute_uv=False)
    conditioning = float(singular[-1] / max(singular[0], 1.0e-12)) if singular.size else 0.0
    return float(np.clip(conditioning * (1.0 + max(float(friction), 0.0)), 0.0, 1.0))
