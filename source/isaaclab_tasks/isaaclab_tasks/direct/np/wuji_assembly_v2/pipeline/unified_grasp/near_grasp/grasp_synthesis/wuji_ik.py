"""Wuji Pinocchio IK and optimized grasp-candidate contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contact_sampler import CollisionMesh, ContactSet
from .grasp_energy import GraspEnergyBreakdown
from ..grasp_program import WUJI_HAND_JOINT_NAMES


DEFAULT_TIP_FRAMES = tuple(f"right_finger{index}_link4" for index in range(1, 6))
DEFAULT_RUNTIME_JOINT_NAMES = (
    "wrist_x",
    "wrist_y",
    "wrist_z",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
) + WUJI_HAND_JOINT_NAMES


def closed_pose_collision_is_valid(
    scene_clearance: Mapping[str, Any],
    self_penetration_squared_m2: float,
    *,
    self_penetration_tolerance_m: float = 0.00005,
) -> bool:
    """Return collision validity without mixing in contact-target residuals."""

    return bool(
        scene_clearance.get("valid", False)
        and float(self_penetration_squared_m2) <= float(self_penetration_tolerance_m) ** 2
    )


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    part_name: str
    contact_sample_id: int
    finger_group: str
    contact_positions_object: tuple[tuple[float, float, float], ...]
    contact_normals_object: tuple[tuple[float, float, float], ...]
    closed_joint_q26: tuple[float, ...]
    pregrasp_joint_q26_by_opening: Mapping[str, tuple[float, ...]]
    standoff_joint_q26_by_distance: Mapping[str, tuple[float, ...]]
    approach_direction_object: tuple[float, float, float]
    target_force_n: float
    optimization_success: bool
    optimization_iterations: int
    tip_error_m: float
    energy: Mapping[str, float]
    input_hash: str
    collision_free_path: bool
    gate_eligibility: Mapping[str, bool]
    schema_version: int = 3
    reachability_success: bool = False
    contact_residual_success: bool = False
    closed_pose_collision_success: bool = False
    pregrasp_path_success: bool = False
    force_closure_success: bool = False
    physics_gate_a_eligible: bool = False
    seed_provider: str = ""
    solver_name: str = ""
    solver_version: str = ""
    per_tip_residual_m: tuple[float, ...] = ()
    target_pad_signed_distance_m: tuple[float, ...] = ()
    non_target_penetration_m: float = float("inf")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 3}:
            raise ValueError("unsupported GraspCandidate schema")
        if len(self.closed_joint_q26) != 26:
            raise ValueError("Wuji grasp candidate must contain 26 joint values")
        if len(self.contact_positions_object) != len(self.finger_group):
            raise ValueError("contact count must match finger group")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraspCandidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            part_name=str(payload["part_name"]),
            contact_sample_id=int(payload["contact_sample_id"]),
            finger_group=str(payload["finger_group"]),
            contact_positions_object=tuple(tuple(float(v) for v in row) for row in payload["contact_positions_object"]),
            contact_normals_object=tuple(tuple(float(v) for v in row) for row in payload["contact_normals_object"]),
            closed_joint_q26=tuple(float(v) for v in payload["closed_joint_q26"]),
            pregrasp_joint_q26_by_opening={str(k): tuple(float(v) for v in row) for k, row in payload["pregrasp_joint_q26_by_opening"].items()},
            standoff_joint_q26_by_distance={str(k): tuple(float(v) for v in row) for k, row in payload["standoff_joint_q26_by_distance"].items()},
            approach_direction_object=tuple(float(v) for v in payload["approach_direction_object"]),
            target_force_n=float(payload["target_force_n"]),
            optimization_success=bool(payload["optimization_success"]),
            optimization_iterations=int(payload["optimization_iterations"]),
            tip_error_m=float(payload["tip_error_m"]),
            energy=dict(payload["energy"]),
            input_hash=str(payload["input_hash"]),
            collision_free_path=bool(payload["collision_free_path"]),
            gate_eligibility=dict(
                payload.get(
                    "gate_eligibility",
                    {
                        "gate_a": bool(payload["optimization_success"]),
                        "gate_b": bool(payload["collision_free_path"]),
                        "gate_c": bool(payload["collision_free_path"]),
                    },
                )
            ),
            schema_version=int(payload.get("schema_version", 1)),
            reachability_success=bool(payload.get("reachability_success", False)),
            contact_residual_success=bool(payload.get("contact_residual_success", False)),
            closed_pose_collision_success=bool(payload.get("closed_pose_collision_success", False)),
            pregrasp_path_success=bool(payload.get("pregrasp_path_success", False)),
            force_closure_success=bool(payload.get("force_closure_success", False)),
            physics_gate_a_eligible=bool(payload.get("physics_gate_a_eligible", False)),
            seed_provider=str(payload.get("seed_provider", "")),
            solver_name=str(payload.get("solver_name", "")),
            solver_version=str(payload.get("solver_version", "")),
            per_tip_residual_m=tuple(float(value) for value in payload.get("per_tip_residual_m", ())),
            target_pad_signed_distance_m=tuple(
                float(value) for value in payload.get("target_pad_signed_distance_m", ())
            ),
            non_target_penetration_m=float(payload.get("non_target_penetration_m", float("inf"))),
            metadata=dict(payload.get("metadata", {})),
        )


class WujiKinematicModel:
    """Pinocchio model with explicit runtime-order and fingertip checks."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        tip_frames: Sequence[str] = DEFAULT_TIP_FRAMES,
        package_dirs: Sequence[str | Path] = (),
        runtime_joint_names: Sequence[str] = DEFAULT_RUNTIME_JOINT_NAMES,
    ):
        import pinocchio as pin

        self.pin = pin
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        if self.model.nq != 26 or self.model.nv != 26:
            raise ValueError(f"Wuji URDF must expose 26 DOF, got nq={self.model.nq}, nv={self.model.nv}")
        self.pin_joint_names = tuple(str(self.model.names[index]) for index in range(1, self.model.njoints))
        self.runtime_joint_names = tuple(str(name) for name in runtime_joint_names)
        if len(self.runtime_joint_names) != 26 or set(self.runtime_joint_names) != set(self.pin_joint_names):
            raise ValueError("runtime and Pinocchio Wuji joint sets do not match")
        self._pin_from_runtime = np.asarray(
            [self.runtime_joint_names.index(name) for name in self.pin_joint_names], dtype=np.int64
        )
        self._runtime_from_pin = np.argsort(self._pin_from_runtime)
        self.tip_frames = tuple(str(name) for name in tip_frames)
        missing = [name for name in self.tip_frames if not self.model.existFrame(name)]
        if missing:
            raise ValueError(f"Wuji URDF is missing fingertip frames: {missing}")
        self.tip_frame_ids = tuple(int(self.model.getFrameId(name)) for name in self.tip_frames)
        self.tip_collision_vertices = _load_tip_collision_vertices(self.urdf_path, self.tip_frames)
        self.geometry_model = None
        self.geometry_error = ""
        try:
            self.geometry_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.urdf_path),
                pin.GeometryType.COLLISION,
                [str(Path(value).resolve()) for value in package_dirs],
            )
            self.geometry_model.addAllCollisionPairs()
            for pair in list(self.geometry_model.collisionPairs):
                if _adjacent_geometry_pair(self.model, self.geometry_model, pair):
                    self.geometry_model.removeCollisionPair(pair)
            self.geometry_data = pin.GeometryData(self.geometry_model)
        except Exception as exc:
            self.geometry_model = None
            self.geometry_data = None
            self.geometry_error = f"{type(exc).__name__}:{exc}"
        self._object_collision_mesh: CollisionMesh | None = None
        self._table_collision_mesh: CollisionMesh | None = None
        self._object_bvh = None
        self._table_bvh = None

    @property
    def collision_model_available(self) -> bool:
        return bool(self.geometry_model is not None and self.geometry_data is not None)

    @property
    def collision_pair_count(self) -> int:
        return len(self.geometry_model.collisionPairs) if self.geometry_model is not None else 0

    @property
    def urdf_sha256(self) -> str:
        return hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()

    def validate_runtime_joint_order(self, runtime_joint_names: Sequence[str]) -> dict[str, Any]:
        runtime = tuple(str(name) for name in runtime_joint_names)
        expected = self.runtime_joint_names
        valid = runtime == expected
        return {
            "valid": valid,
            "runtime_joint_names": list(runtime),
            "expected_runtime_joint_names": list(expected),
            "pinocchio_joint_names": list(self.pin_joint_names),
            "pinocchio_reordering_required": self.pin_joint_names != self.runtime_joint_names,
            "pin_from_runtime_indices": self._pin_from_runtime.tolist(),
            "collision_model_available": self.collision_model_available,
            "collision_pair_count": self.collision_pair_count,
            "collision_model_error": self.geometry_error,
            "first_mismatch": next(
                (
                    {"index": index, "runtime": runtime[index], "pinocchio": expected[index]}
                    for index in range(min(len(runtime), len(expected)))
                    if runtime[index] != expected[index]
                ),
                None,
            ),
        }

    def tip_positions_and_jacobians(self, q26: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        positions, _rotations, jacobians = self.tip_poses_and_jacobians(q26)
        return positions, jacobians

    def tip_poses_and_jacobians(self, q26: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_runtime = np.asarray(q26, dtype=np.float64)
        if q_runtime.shape != (26,) or not np.all(np.isfinite(q_runtime)):
            raise ValueError("q26 must be a finite 26D vector")
        q = q_runtime[self._pin_from_runtime]
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        positions = []
        rotations = []
        jacobians = []
        for frame_id in self.tip_frame_ids:
            positions.append(self.data.oMf[frame_id].translation.copy())
            rotations.append(self.data.oMf[frame_id].rotation.copy())
            jacobian6 = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            jacobians.append(jacobian6[:3, self._runtime_from_pin].copy())
        return np.asarray(positions), np.asarray(rotations), np.asarray(jacobians)

    def pad_points_and_jacobians(
        self,
        q26: Sequence[float],
        finger_indices: Sequence[int],
        outward_normals_world: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        """Return support-vertex world positions and point Jacobians in runtime order."""

        q_runtime = np.asarray(q26, dtype=np.float64)
        fingers = np.asarray(finger_indices, dtype=np.int64)
        normals = np.asarray(outward_normals_world, dtype=np.float64)
        if q_runtime.shape != (26,) or normals.shape != (len(fingers), 3):
            raise ValueError("pad point inputs have invalid dimensions")
        q = q_runtime[self._pin_from_runtime]
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        points = []
        rotations = []
        point_jacobians = []
        vertices = []
        for finger, normal in zip(fingers, normals):
            frame_id = self.tip_frame_ids[int(finger)]
            placement = self.data.oMf[frame_id]
            rotation = placement.rotation.copy()
            vertex = self.pad_support_vertex(int(finger), rotation, normal)
            offset = rotation @ vertex
            jacobian6 = pin.getFrameJacobian(
                self.model,
                self.data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            point_jacobian_pin = jacobian6[:3] - _skew(offset) @ jacobian6[3:]
            points.append(placement.translation.copy() + offset)
            rotations.append(rotation)
            point_jacobians.append(point_jacobian_pin[:, self._runtime_from_pin].copy())
            vertices.append(vertex)
        return np.asarray(points), np.asarray(rotations), np.asarray(point_jacobians), tuple(vertices)

    def pad_support_distance(self, finger_index: int, frame_rotation: np.ndarray, outward_normal_world: np.ndarray) -> float:
        vertex = self.pad_support_vertex(finger_index, frame_rotation, outward_normal_world)
        world_offset = np.asarray(frame_rotation, dtype=np.float64) @ vertex
        return max(0.0, -float(np.dot(world_offset, outward_normal_world)))

    def pad_support_vertex(
        self,
        finger_index: int,
        frame_rotation: np.ndarray,
        outward_normal_world: np.ndarray,
    ) -> np.ndarray:
        local_direction = np.asarray(frame_rotation, dtype=np.float64).T @ np.asarray(
            outward_normal_world, dtype=np.float64
        )
        vertices = self.tip_collision_vertices[int(finger_index)]
        return vertices[int(np.argmin(vertices @ local_direction))].copy()

    def set_scene_collision_meshes(self, object_mesh: CollisionMesh, table_mesh: CollisionMesh) -> None:
        if object_mesh.coordinate_frame != "object_local":
            raise ValueError("object collision mesh must be object-local")
        if table_mesh.coordinate_frame != "env_local":
            raise ValueError("table collision mesh must be env-local")
        self._object_collision_mesh = object_mesh
        self._table_collision_mesh = table_mesh
        self._object_bvh = _build_hppfcl_bvh(object_mesh)
        self._table_bvh = _build_hppfcl_bvh(table_mesh)

    @property
    def scene_collision_available(self) -> bool:
        return self._object_bvh is not None and self._table_bvh is not None

    def scene_clearance(
        self,
        q26: Sequence[float],
        *,
        object_position_world: Sequence[float],
        object_rotation_world: np.ndarray,
        active_fingers: Sequence[int],
    ) -> dict[str, Any]:
        if not self.scene_collision_available or self.geometry_model is None or self.geometry_data is None:
            raise RuntimeError("full hand-object/table collision geometry is unavailable")
        import hppfcl

        q = np.asarray(q26, dtype=np.float64)[self._pin_from_runtime]
        self.pin.updateGeometryPlacements(self.model, self.data, self.geometry_model, self.geometry_data, q)
        object_transform = hppfcl.Transform3f(
            np.asarray(object_rotation_world, dtype=np.float64), np.asarray(object_position_world, dtype=np.float64)
        )
        table_transform = hppfcl.Transform3f()
        active_names = {
            marker
            for index in active_fingers
            for marker in (
                f"right_finger{int(index) + 1}_link4",
                f"right_finger{int(index) + 1}_tip_link",
            )
        }
        object_rows = []
        table_rows = []
        for index, geometry_object in enumerate(self.geometry_model.geometryObjects):
            placement = self.geometry_data.oMg[index]
            transform = hppfcl.Transform3f(placement.rotation, placement.translation)
            name = str(geometry_object.name)
            active_tip = any(marker in name for marker in active_names)
            object_clearance, object_penetration = _fcl_clearance(
                geometry_object.geometry, transform, self._object_bvh, object_transform
            )
            table_clearance, table_penetration = _fcl_clearance(
                geometry_object.geometry, transform, self._table_bvh, table_transform
            )
            object_rows.append(
                {
                    "link_geometry": name,
                    "active_target_tip": active_tip,
                    "clearance_m": object_clearance,
                    "penetration_m": object_penetration,
                }
            )
            table_rows.append(
                {
                    "link_geometry": name,
                    "clearance_m": table_clearance,
                    "penetration_m": table_penetration,
                }
            )
        active_object = [row for row in object_rows if row["active_target_tip"]]
        nonactive_object = [row for row in object_rows if not row["active_target_tip"]]
        active_max_penetration = max((float(row["penetration_m"]) for row in active_object), default=0.0)
        nonactive_min_clearance = min((float(row["clearance_m"]) for row in nonactive_object), default=float("inf"))
        table_min_clearance = min((float(row["clearance_m"]) for row in table_rows), default=float("inf"))
        total_object_penetration = float(sum(float(row["penetration_m"]) for row in object_rows))
        maximum_object_penetration = max((float(row["penetration_m"]) for row in object_rows), default=0.0)
        penalty_m2 = (
            max(active_max_penetration - 0.0001, 0.0) ** 2
            + max(0.0005 - nonactive_min_clearance, 0.0) ** 2
            + max(0.0005 - table_min_clearance, 0.0) ** 2
        )
        return {
            "active_tip_max_penetration_m": active_max_penetration,
            "nonactive_hand_object_min_clearance_m": nonactive_min_clearance,
            "whole_hand_table_min_clearance_m": table_min_clearance,
            "hand_object_penetration_max_m": maximum_object_penetration,
            "hand_object_penetration_sum_m": total_object_penetration,
            "penalty_m2": float(penalty_m2),
            "object_pairs": object_rows,
            "table_pairs": table_rows,
            "valid": bool(
                active_max_penetration <= 0.0001
                and nonactive_min_clearance >= 0.0005
                and table_min_clearance >= 0.0005
            ),
        }

    def fingertip_mesh_min_z(self, q26: Sequence[float]) -> np.ndarray:
        positions, rotations, _ = self.tip_poses_and_jacobians(q26)
        return np.asarray(
            [
                float(np.min((vertices @ rotation.T + position.reshape(1, 3))[:, 2]))
                for vertices, rotation, position in zip(self.tip_collision_vertices, rotations, positions)
            ],
            dtype=np.float64,
        )

    def self_collision(self, q26: Sequence[float]) -> bool:
        return self.self_collision_penetration(q26)[0] > 0.0

    def self_collision_penetration(self, q26: Sequence[float]) -> tuple[float, tuple[tuple[str, str], ...]]:
        if self.geometry_model is None or self.geometry_data is None:
            return 0.0, ()
        q = np.asarray(q26, dtype=np.float64)[self._pin_from_runtime]
        self.pin.updateGeometryPlacements(self.model, self.data, self.geometry_model, self.geometry_data, q)
        self.pin.computeCollisions(self.geometry_model, self.geometry_data, False)
        squared_depth = 0.0
        pairs = []
        for pair, result in zip(self.geometry_model.collisionPairs, self.geometry_data.collisionResults):
            if not result.isCollision():
                continue
            first = self.geometry_model.geometryObjects[pair.first].name
            second = self.geometry_model.geometryObjects[pair.second].name
            pairs.append((str(first), str(second)))
            for contact in result.getContacts():
                squared_depth += max(float(contact.penetration_depth), 0.0) ** 2
        return squared_depth, tuple(pairs)

    def optimize_contact_set_staged(
        self,
        *,
        part_name: str,
        contact_set: ContactSet,
        object_position_world: Sequence[float],
        object_rotation_world: np.ndarray,
        seed_candidates: Sequence[tuple[str, Sequence[float]]],
        joint_lower26: Sequence[float],
        joint_upper26: Sequence[float],
        target_force_n: float,
        input_hash: str,
        dls_iterations: int = 120,
        least_squares_evaluations: int = 256,
    ) -> GraspCandidate:
        """Solve reachability, collision feasibility, and path feasibility in separate stages."""

        from scipy.optimize import least_squares, minimize

        if not seed_candidates:
            raise ValueError("staged IK requires at least one seed")
        if not self.scene_collision_available:
            raise RuntimeError("staged IK requires audited object and table collision meshes")
        object_position = np.asarray(object_position_world, dtype=np.float64)
        rotation = np.asarray(object_rotation_world, dtype=np.float64)
        lower = np.asarray(joint_lower26, dtype=np.float64)
        upper = np.asarray(joint_upper26, dtype=np.float64)
        if object_position.shape != (3,) or rotation.shape != (3, 3) or lower.shape != (26,) or upper.shape != (26,):
            raise ValueError("staged IK input dimensions are invalid")
        contact_object = np.asarray([row.position_object for row in contact_set.contacts], dtype=np.float64)
        normal_object = np.asarray([row.normal_object for row in contact_set.contacts], dtype=np.float64)
        contact_world = object_position + contact_object @ rotation.T
        normal_world = normal_object @ rotation.T
        fingers = np.asarray([int(value) - 1 for value in contact_set.finger_group], dtype=np.int64)
        active_hand_local = np.asarray(
            [index for index in range(20) if index % 5 in set(fingers.tolist())], dtype=np.int64
        )
        active_columns = active_hand_local + 6
        optimized_columns = np.concatenate((np.arange(6, dtype=np.int64), active_columns))

        def pad_state(q: np.ndarray):
            pads, rotations, jacobians, vertices = self.pad_points_and_jacobians(q, fingers, normal_world)
            residual = pads - contact_world
            return residual, pads, rotations, jacobians, vertices

        attempts: list[dict[str, Any]] = []
        solved: list[dict[str, Any]] = []
        for provider, seed_values in seed_candidates[:12]:
            seed = np.clip(np.asarray(seed_values, dtype=np.float64), lower, upper)
            if seed.shape != (26,) or not np.all(np.isfinite(seed)):
                raise ValueError(f"invalid staged IK seed from {provider}")
            seed_lower = lower.copy()
            seed_upper = upper.copy()
            rotation_window = np.deg2rad(20.0)
            seed_lower[3:6] = np.maximum(seed_lower[3:6], seed[3:6] - rotation_window)
            seed_upper[3:6] = np.minimum(seed_upper[3:6], seed[3:6] + rotation_window)
            q = seed.copy()
            best_error = float("inf")
            stagnant = 0
            iterations = 0
            termination = "DLS_MAX_ITERATIONS"
            for iteration in range(int(dls_iterations)):
                residual, _pads, _rotations, jacobians, _vertices = pad_state(q)
                norms = np.linalg.norm(residual, axis=1)
                error = float(np.max(norms))
                iterations = iteration + 1
                if error <= 0.001:
                    termination = "DLS_CONTACT_RESIDUAL_REACHED"
                    break
                stacked_j = jacobians[:, :, optimized_columns].reshape(-1, len(optimized_columns))
                stacked_error = residual.reshape(-1)
                damping = 0.003 if error > 0.005 else 0.001
                dq = -stacked_j.T @ np.linalg.solve(
                    stacked_j @ stacked_j.T + (damping**2) * np.eye(stacked_j.shape[0]),
                    stacked_error,
                )
                dq[:3] = np.clip(dq[:3], -0.002, 0.002)
                dq[3:6] = np.clip(dq[3:6], -np.deg2rad(2.0), np.deg2rad(2.0))
                dq[6:] = np.clip(dq[6:], -0.02, 0.02)
                q[optimized_columns] = np.clip(
                    q[optimized_columns] + dq,
                    seed_lower[optimized_columns],
                    seed_upper[optimized_columns],
                )
                if best_error - error <= 1.0e-5:
                    stagnant += 1
                else:
                    stagnant = 0
                    best_error = error
                if stagnant >= 10:
                    termination = "DLS_STAGNATED"
                    break
            residual = pad_state(q)[0]
            dls_error = float(np.max(np.linalg.norm(residual, axis=1)))
            per_tip = np.linalg.norm(pad_state(q)[0], axis=1)
            max_error = float(np.max(per_tip))
            attempts.append(
                {
                    "seed_provider": provider,
                    "dls_error_m": dls_error,
                    "max_tip_error_m": max_error,
                    "per_tip_residual_m": per_tip.tolist(),
                    "iterations": iterations,
                    "least_squares_used": False,
                    "termination": termination,
                }
            )
            solved.append(
                {
                    "max_error": max_error,
                    "provider": str(provider),
                    "q": q.copy(),
                    "seed": seed.copy(),
                    "seed_lower": seed_lower,
                    "seed_upper": seed_upper,
                    "iterations": iterations,
                    "termination": termination,
                    "attempt_index": len(attempts) - 1,
                }
            )

        # Stage A has one bounded nonlinear fallback per contact target, not one
        # fallback for every seed.  Running 256 evaluations for all 12 starts
        # silently multiplies the declared target budget by twelve.
        solved.sort(key=lambda row: (row["max_error"], row["provider"]))
        best = solved[0]
        if float(best["max_error"]) > 0.001:
            seed = np.asarray(best["seed"], dtype=np.float64)
            seed_lower = np.asarray(best["seed_lower"], dtype=np.float64)
            seed_upper = np.asarray(best["seed_upper"], dtype=np.float64)

            def smooth_residual(active_q: np.ndarray) -> np.ndarray:
                candidate = seed.copy()
                candidate[optimized_columns] = active_q
                tip_residual = pad_state(candidate)[0].reshape(-1)
                regularization = 1.0e-3 * (active_q - seed[optimized_columns])
                return np.concatenate((tip_residual, regularization))

            def smooth_jacobian(active_q: np.ndarray) -> np.ndarray:
                candidate = seed.copy()
                candidate[optimized_columns] = active_q
                point_jacobians = pad_state(candidate)[3][:, :, optimized_columns]
                regularization = 1.0e-3 * np.eye(len(optimized_columns), dtype=np.float64)
                return np.vstack((point_jacobians.reshape(-1, len(optimized_columns)), regularization))

            result = least_squares(
                smooth_residual,
                np.asarray(best["q"], dtype=np.float64)[optimized_columns],
                jac=smooth_jacobian,
                bounds=(seed_lower[optimized_columns], seed_upper[optimized_columns]),
                max_nfev=int(least_squares_evaluations),
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
            least_squares_q = seed.copy()
            least_squares_q[optimized_columns] = result.x
            per_tip = np.linalg.norm(pad_state(least_squares_q)[0], axis=1)
            least_squares_error = float(np.max(per_tip))
            attempt = attempts[int(best["attempt_index"])]
            attempt.update(
                {
                    "least_squares_used": True,
                    "least_squares_nfev": int(result.nfev),
                    "least_squares_max_tip_error_m": least_squares_error,
                    "least_squares_per_tip_residual_m": per_tip.tolist(),
                    "least_squares_reported_success": bool(result.success),
                }
            )
            if least_squares_error < float(best["max_error"]):
                best["max_error"] = least_squares_error
                best["q"] = least_squares_q
                best["iterations"] = int(best["iterations"]) + int(result.nfev)
                best["termination"] = f"LEAST_SQUARES:{result.status}:{result.message}"
                attempt["max_tip_error_m"] = least_squares_error
                attempt["per_tip_residual_m"] = per_tip.tolist()
                attempt["iterations"] = int(best["iterations"])
                attempt["termination"] = best["termination"]
            else:
                attempt["least_squares_rejected"] = True

        stage_b_seed_evaluations = []
        reachable_solutions = [row for row in solved if float(row["max_error"]) <= 0.001]
        for row in reachable_solutions:
            seed_q = np.asarray(row["q"], dtype=np.float64)
            scene = self.scene_clearance(
                seed_q,
                object_position_world=object_position,
                object_rotation_world=rotation,
                active_fingers=fingers,
            )
            self_penetration, self_pairs = self.self_collision_penetration(seed_q)
            row["stage_b_scene"] = scene
            row["stage_b_self_penetration"] = self_penetration
            row["stage_b_self_pairs"] = self_pairs
            row["stage_b_entry_cost"] = float(scene["penalty_m2"]) + self_penetration
            stage_b_seed_evaluations.append(
                {
                    "seed_provider": row["provider"],
                    "max_tip_error_m": float(row["max_error"]),
                    "scene_valid": bool(scene["valid"]),
                    "scene_penalty_m2": float(scene["penalty_m2"]),
                    "self_penetration_squared_m2": self_penetration,
                    "self_collision_pairs": [list(pair) for pair in self_pairs],
                }
            )
        if reachable_solutions:
            reachable_solutions.sort(
                key=lambda row: (
                    not (
                        bool(row["stage_b_scene"]["valid"])
                        and float(row["stage_b_self_penetration"]) <= (0.00005**2)
                    ),
                    float(row["stage_b_entry_cost"]),
                    float(row["max_error"]),
                    str(row["provider"]),
                )
            )
            best = reachable_solutions[0]

        stage_a_error = float(best["max_error"])
        provider = str(best["provider"])
        q_reachable = np.asarray(best["q"], dtype=np.float64)
        stage_b_lower = np.asarray(best["seed_lower"], dtype=np.float64)
        stage_b_upper = np.asarray(best["seed_upper"], dtype=np.float64)
        stage_a_iterations = int(best["iterations"])
        stage_a_termination = str(best["termination"])
        reachability_success = bool(stage_a_error <= 0.003)
        contact_residual_success = bool(stage_a_error <= 0.001)
        q_closed = q_reachable.copy()
        stage_b_initial_scene = self.scene_clearance(
            q_reachable,
            object_position_world=object_position,
            object_rotation_world=rotation,
            active_fingers=fingers,
        )
        stage_b_initial_self_penetration, stage_b_initial_self_pairs = self.self_collision_penetration(q_reachable)
        continuation_rows: list[dict[str, Any]] = []
        if contact_residual_success and not (
            bool(stage_b_initial_scene["valid"])
            and stage_b_initial_self_penetration <= (0.00005**2)
        ):
            seed_for_regularization = q_reachable.copy()

            def collision_objective(active_q: np.ndarray, weight: float) -> float:
                q = seed_for_regularization.copy()
                q[optimized_columns] = active_q
                residual = pad_state(q)[0]
                self_penetration, _ = self.self_collision_penetration(q)
                scene = self.scene_clearance(
                    q,
                    object_position_world=object_position,
                    object_rotation_world=rotation,
                    active_fingers=fingers,
                )
                regularization = active_q - q_reachable[optimized_columns]
                return float(
                    np.sum(residual * residual)
                    + weight * (self_penetration + float(scene["penalty_m2"]))
                    + 1.0e-5 * np.sum(regularization * regularization)
                )

            for weight in (1.0e2, 1.0e4, 1.0e6):
                result = minimize(
                    lambda active_q, w=weight: collision_objective(active_q, w),
                    q_closed[optimized_columns],
                    method="SLSQP",
                    bounds=list(zip(stage_b_lower[optimized_columns], stage_b_upper[optimized_columns])),
                    options={"maxiter": 48, "ftol": 1.0e-12, "disp": False},
                )
                q_closed[optimized_columns] = result.x
                current_error = float(np.max(np.linalg.norm(pad_state(q_closed)[0], axis=1)))
                scene = self.scene_clearance(
                    q_closed,
                    object_position_world=object_position,
                    object_rotation_world=rotation,
                    active_fingers=fingers,
                )
                self_penetration, pairs = self.self_collision_penetration(q_closed)
                continuation_rows.append(
                    {
                        "weight": weight,
                        "reported_success": bool(result.success),
                        "message": str(result.message),
                        "iterations": int(getattr(result, "nit", 0)),
                        "tip_error_m": current_error,
                        "scene_valid": bool(scene["valid"]),
                        "self_penetration_squared_m2": self_penetration,
                        "self_collision_pairs": [list(pair) for pair in pairs],
                    }
                )

        residual, pads, closed_rotations, _jacobians, support_vertices = pad_state(q_closed)
        per_tip_residual = np.linalg.norm(residual, axis=1)
        tip_error = float(np.max(per_tip_residual))
        signed_distances = tuple(float(np.dot(value, normal)) for value, normal in zip(residual, normal_world))
        self_penetration, self_collision_pairs = self.self_collision_penetration(q_closed)
        closed_scene = self.scene_clearance(
            q_closed,
            object_position_world=object_position,
            object_rotation_world=rotation,
            active_fingers=fingers,
        )
        closed_collision_success = closed_pose_collision_is_valid(
            closed_scene,
            self_penetration,
        )
        wrench = dict(contact_set.gravity_wrench)
        force_closure_success = bool(wrench.get("feasible", False))

        pregrasps: dict[str, tuple[float, ...]] = {}
        pregrasp_ok: dict[str, bool] = {}
        for opening_m in (0.003, 0.005, 0.008):
            q_open = q_closed.copy()
            for _ in range(24):
                pad_points, _rotations, point_jacobians, _vertices = self.pad_points_and_jacobians(
                    q_open, fingers, normal_world
                )
                desired = pads + normal_world * opening_m
                error = (pad_points - desired).reshape(-1)
                jacobian = point_jacobians[:, :, active_columns].reshape(-1, len(active_columns))
                dq = -jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1.0e-6 * np.eye(jacobian.shape[0]), error
                )
                q_open[active_columns] = np.clip(
                    q_open[active_columns] + np.clip(dq, -0.02, 0.02),
                    lower[active_columns],
                    upper[active_columns],
                )
            key = f"{int(round(opening_m * 1000))}mm"
            pregrasps[key] = tuple(float(value) for value in q_open)
            scene = self.scene_clearance(
                q_open,
                object_position_world=object_position,
                object_rotation_world=rotation,
                active_fingers=fingers,
            )
            self_open, _ = self.self_collision_penetration(q_open)
            pregrasp_ok[key] = bool(scene["valid"] and self_open <= (0.00005**2))

        standoffs: dict[str, tuple[float, ...]] = {}
        standoff_ok: dict[str, bool] = {}
        approach_world = rotation @ np.asarray(contact_set.approach_direction_object, dtype=np.float64)
        for distance_m in (0.020, 0.030, 0.040):
            q_standoff = np.asarray(pregrasps["5mm"], dtype=np.float64).copy()
            q_standoff[:3] -= approach_world * distance_m
            q_standoff = np.clip(q_standoff, lower, upper)
            key = f"{int(round(distance_m * 1000))}mm"
            standoffs[key] = tuple(float(value) for value in q_standoff)
            scene = self.scene_clearance(
                q_standoff,
                object_position_world=object_position,
                object_rotation_world=rotation,
                active_fingers=fingers,
            )
            self_standoff, _ = self.self_collision_penetration(q_standoff)
            standoff_ok[key] = bool(scene["valid"] and self_standoff <= (0.00005**2))

        pregrasp_path_success = bool(
            closed_collision_success
            and pregrasp_ok.get("5mm", False)
            and _interpolated_collision_free(
                self,
                np.asarray(pregrasps["5mm"]),
                q_closed,
                object_position,
                rotation,
                fingers,
                samples=21,
            )
        )
        transit = np.asarray(next(value for name, value in seed_candidates if name == provider), dtype=np.float64).copy()
        transit[:6] = np.asarray(standoffs["30mm"], dtype=np.float64)[:6]
        transit_to_preshape = bool(
            standoff_ok.get("30mm", False)
            and _interpolated_collision_free(
                self,
                transit,
                np.asarray(standoffs["30mm"]),
                object_position,
                rotation,
                fingers,
                samples=21,
            )
        )
        standoff_path_success = bool(
            pregrasp_path_success
            and transit_to_preshape
            and _interpolated_collision_free(
                self,
                np.asarray(standoffs["30mm"]),
                np.asarray(pregrasps["5mm"]),
                object_position,
                rotation,
                fingers,
                samples=21,
            )
        )
        optimization_success = bool(
            reachability_success
            and tip_error <= 0.001
            and closed_collision_success
            and force_closure_success
        )
        gate_a = optimization_success
        gate_b = gate_a and pregrasp_path_success
        gate_c = gate_b and standoff_path_success
        energy = GraspEnergyBreakdown(
            contact_distance=float(np.mean(per_tip_residual)),
            opposition=1.0 - contact_set.opposition_quality,
            force_closure=1.0 - float(wrench.get("quality", 0.0)),
            hand_object_penetration=float(closed_scene["hand_object_penetration_max_m"]),
            self_collision=float(self_penetration > (0.00005**2)),
            table_collision=float(float(closed_scene["whole_hand_table_min_clearance_m"]) < 0.0005),
            joint_limit=float(np.mean(np.square((q_closed - q_reachable) / np.maximum(upper - lower, 1.0e-6)))),
            path_collision=float(not standoff_path_success),
        )
        candidate_payload = {
            "schema_version": 3,
            "solver": "staged_pad_dls_collision_continuation_v3_1",
            "part_name": part_name,
            "sample_id": contact_set.sample_id,
            "group": contact_set.finger_group,
            "contacts": contact_object.tolist(),
            "fallback_tier": contact_set.fallback_tier,
            "approach_direction_object": list(contact_set.approach_direction_object),
            "q": q_closed.tolist(),
            "input_hash": input_hash,
        }
        candidate_id = hashlib.sha256(json.dumps(candidate_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return GraspCandidate(
            candidate_id=candidate_id,
            part_name=part_name,
            contact_sample_id=contact_set.sample_id,
            finger_group=contact_set.finger_group,
            contact_positions_object=tuple(tuple(float(value) for value in row) for row in contact_object),
            contact_normals_object=tuple(tuple(float(value) for value in row) for row in normal_object),
            closed_joint_q26=tuple(float(value) for value in q_closed),
            pregrasp_joint_q26_by_opening=pregrasps,
            standoff_joint_q26_by_distance=standoffs,
            approach_direction_object=contact_set.approach_direction_object,
            target_force_n=float(target_force_n),
            optimization_success=optimization_success,
            optimization_iterations=int(stage_a_iterations + sum(row["iterations"] for row in continuation_rows)),
            tip_error_m=tip_error,
            energy=energy.to_dict(),
            input_hash=input_hash,
            collision_free_path=standoff_path_success,
            gate_eligibility={"gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c},
            schema_version=3,
            reachability_success=reachability_success,
            contact_residual_success=bool(tip_error <= 0.001),
            closed_pose_collision_success=closed_collision_success,
            pregrasp_path_success=pregrasp_path_success,
            force_closure_success=force_closure_success,
            physics_gate_a_eligible=gate_a,
            seed_provider=provider,
            solver_name="staged_pad_dls_collision_continuation",
            solver_version="3.1",
            per_tip_residual_m=tuple(float(value) for value in per_tip_residual),
            target_pad_signed_distance_m=signed_distances,
            non_target_penetration_m=float(closed_scene["hand_object_penetration_max_m"]),
            metadata={
                "stage_a_attempts": attempts,
                "stage_a_termination": stage_a_termination,
                "stage_b_seed_evaluations": stage_b_seed_evaluations,
                "stage_b_entry_q26": q_reachable.tolist(),
                "stage_b_wrist_lower": stage_b_lower[:6].tolist(),
                "stage_b_wrist_upper": stage_b_upper[:6].tolist(),
                "stage_b_initial_scene_clearance": stage_b_initial_scene,
                "stage_b_initial_self_penetration_squared_m2": stage_b_initial_self_penetration,
                "stage_b_initial_self_collision_pairs": [list(pair) for pair in stage_b_initial_self_pairs],
                "stage_b_continuation": continuation_rows,
                "gravity_wrench": wrench,
                "fallback_tier": contact_set.fallback_tier,
                "axis_delta_m": contact_set.axis_delta_m,
                "raycast_hit": contact_set.raycast_hit,
                "pad_support_vertex_local": [vertex.tolist() for vertex in support_vertices],
                "pad_support_world_offset": [
                    (closed_rotations[index] @ vertex).tolist() for index, vertex in enumerate(support_vertices)
                ],
                "pad_contact_positions_world": pads.tolist(),
                "frame_contact_targets_world": contact_world.tolist(),
                "scene_clearance": closed_scene,
                "self_collision_penetration_squared_m2": self_penetration,
                "self_collision_pairs": [list(pair) for pair in self_collision_pairs],
                "pregrasp_collision_free": pregrasp_ok,
                "standoff_collision_free": standoff_ok,
                "transit_open_q26": transit.tolist(),
                "transit_to_preshape_collision_free": transit_to_preshape,
                "object_collision_mesh_sha256": self._object_collision_mesh.sha256,
                "table_collision_mesh_sha256": self._table_collision_mesh.sha256,
                "pinocchio_urdf_sha256": self.urdf_sha256,
            },
        )

    def optimize_contact_set(
        self,
        *,
        part_name: str,
        contact_set: ContactSet,
        object_position_world: Sequence[float],
        object_rotation_world: np.ndarray,
        q_seed26: Sequence[float],
        joint_lower26: Sequence[float],
        joint_upper26: Sequence[float],
        table_top_z_m: float,
        target_force_n: float,
        input_hash: str,
        max_iterations: int = 32,
    ) -> GraspCandidate:
        from scipy.optimize import minimize

        object_position = np.asarray(object_position_world, dtype=np.float64)
        rotation = np.asarray(object_rotation_world, dtype=np.float64)
        q_seed = np.asarray(q_seed26, dtype=np.float64)
        lower = np.asarray(joint_lower26, dtype=np.float64)
        upper = np.asarray(joint_upper26, dtype=np.float64)
        if object_position.shape != (3,) or rotation.shape != (3, 3) or q_seed.shape != (26,):
            raise ValueError("IK input dimensions are invalid")
        if not self.scene_collision_available:
            raise RuntimeError("forensic IK requires audited object and table collision meshes")
        contact_object = np.asarray([row.position_object for row in contact_set.contacts], dtype=np.float64)
        normal_object = np.asarray([row.normal_object for row in contact_set.contacts], dtype=np.float64)
        contact_world = object_position + contact_object @ rotation.T
        normal_world = normal_object @ rotation.T
        fingers = np.asarray([int(value) - 1 for value in contact_set.finger_group], dtype=np.int64)
        active_hand_local = np.asarray(
            [index for index in range(20) if index % 5 in set(fingers.tolist())],
            dtype=np.int64,
        )
        active_columns = active_hand_local + 6
        optimized_columns = np.concatenate((np.arange(6, dtype=np.int64), active_columns))

        def expand(active_q: np.ndarray) -> np.ndarray:
            q = np.clip(q_seed, lower, upper).copy()
            q[optimized_columns] = active_q
            return q

        def residuals(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
            tips, rotations, jacobians = self.tip_poses_and_jacobians(q)
            frame_targets = contact_world.copy()
            support_vertices = []
            for contact_index, finger in enumerate(fingers):
                vertex = self.pad_support_vertex(finger, rotations[finger], normal_world[contact_index])
                support_vertices.append(vertex)
                frame_targets[contact_index] -= rotations[finger] @ vertex
            return tips[fingers] - frame_targets, tips, jacobians, frame_targets, support_vertices

        def objective(active_q: np.ndarray) -> float:
            q = expand(active_q)
            residual, _tips, _, _, _ = residuals(q)
            regularization = 1.0e-3 * (active_q - q_seed[optimized_columns])
            self_penetration, _ = self.self_collision_penetration(q)
            scene = self.scene_clearance(
                q,
                object_position_world=object_position,
                object_rotation_world=rotation,
                active_fingers=fingers,
            )
            return float(
                np.sum(residual * residual)
                + 1.0e4 * self_penetration
                + 1.0e4 * float(scene["penalty_m2"])
                + np.sum(regularization * regularization)
            )

        result = minimize(
            objective,
            np.clip(q_seed, lower, upper)[optimized_columns],
            method="SLSQP",
            bounds=list(zip(lower[optimized_columns], upper[optimized_columns])),
            options={"maxiter": int(max_iterations), "ftol": 1.0e-12, "disp": False},
        )
        q_closed = expand(np.asarray(result.x, dtype=np.float64))
        residual, tips, jacobians, frame_targets, support_vertices = residuals(q_closed)
        _, closed_rotations, _ = self.tip_poses_and_jacobians(q_closed)
        pad_support_distances = tuple(
            self.pad_support_distance(finger, closed_rotations[finger], normal_world[index])
            for index, finger in enumerate(fingers)
        )
        support_world_offsets = tuple(
            closed_rotations[finger] @ support_vertices[index] for index, finger in enumerate(fingers)
        )
        tip_error = float(np.max(np.linalg.norm(residual, axis=1)))
        self_penetration, self_collision_pairs = self.self_collision_penetration(q_closed)
        self_collision = self_penetration > 0.0
        fingertip_min_z = self.fingertip_mesh_min_z(q_closed)
        closed_scene = self.scene_clearance(
            q_closed,
            object_position_world=object_position,
            object_rotation_world=rotation,
            active_fingers=fingers,
        )
        table_collision = bool(float(closed_scene["whole_hand_table_min_clearance_m"]) < 0.0005)
        pregrasps = {}
        pregrasp_ok = {}
        for opening_m in (0.003, 0.005, 0.008):
            desired = np.zeros((len(fingers), 3), dtype=np.float64)
            desired[:] = normal_world * opening_m
            stacked_j = np.concatenate([jacobians[finger][:, active_columns] for finger in fingers], axis=0)
            stacked_dx = desired.reshape(-1)
            dq_active = stacked_j.T @ np.linalg.solve(
                stacked_j @ stacked_j.T + 1.0e-4 * np.eye(stacked_j.shape[0]),
                stacked_dx,
            )
            q_open = q_closed.copy()
            q_open[active_columns] += dq_active
            q_open = np.clip(q_open, lower, upper)
            key = f"{int(round(opening_m * 1000))}mm"
            pregrasps[key] = tuple(float(value) for value in q_open)
            pregrasp_scene = self.scene_clearance(
                q_open,
                object_position_world=object_position,
                object_rotation_world=rotation,
                active_fingers=fingers,
            )
            pregrasp_ok[key] = not self.self_collision(q_open) and bool(pregrasp_scene["valid"])
        standoffs = {}
        standoff_ok = {}
        approach_world = rotation @ np.asarray(contact_set.approach_direction_object, dtype=np.float64)
        for distance_m in (0.020, 0.030, 0.040):
            q_standoff = np.asarray(pregrasps["5mm"], dtype=np.float64).copy()
            q_standoff[:3] -= approach_world * distance_m
            q_standoff = np.clip(q_standoff, lower, upper)
            key = f"{int(round(distance_m * 1000))}mm"
            standoffs[key] = tuple(float(value) for value in q_standoff)
            standoff_scene = self.scene_clearance(
                q_standoff,
                object_position_world=object_position,
                object_rotation_world=rotation,
                active_fingers=fingers,
            )
            standoff_ok[key] = not self.self_collision(q_standoff) and bool(standoff_scene["valid"])
        closed_ok = self.collision_model_available and not self_collision and bool(closed_scene["valid"])
        gate_b_ok = closed_ok and pregrasp_ok["5mm"] and _interpolated_collision_free(
            self,
            np.asarray(pregrasps["5mm"]),
            q_closed,
            object_position,
            rotation,
            fingers,
        )
        gate_c_ok = gate_b_ok and standoff_ok["30mm"] and _interpolated_collision_free(
            self,
            np.asarray(standoffs["30mm"]),
            np.asarray(pregrasps["5mm"]),
            object_position,
            rotation,
            fingers,
        )
        path_ok = gate_c_ok
        wrench = dict(contact_set.gravity_wrench)
        closure = float(wrench.get("quality", 0.0))
        energy = GraspEnergyBreakdown(
            contact_distance=float(np.mean(np.linalg.norm(residual, axis=1))),
            opposition=1.0 - contact_set.opposition_quality,
            force_closure=1.0 - closure,
            hand_object_penetration=float(closed_scene["hand_object_penetration_max_m"]),
            self_collision=float(self_collision),
            table_collision=float(table_collision),
            joint_limit=float(np.mean(np.square((q_closed - q_seed) / np.maximum(upper - lower, 1.0e-6)))),
            path_collision=float(not path_ok),
        )
        candidate_payload = {
            "part_name": part_name,
            "sample_id": contact_set.sample_id,
            "group": contact_set.finger_group,
            "contacts": contact_object.tolist(),
            "fallback_tier": contact_set.fallback_tier,
            "approach_direction_object": list(contact_set.approach_direction_object),
            "q": q_closed.tolist(),
            "input_hash": input_hash,
        }
        candidate_id = hashlib.sha256(json.dumps(candidate_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return GraspCandidate(
            candidate_id=candidate_id,
            part_name=part_name,
            contact_sample_id=contact_set.sample_id,
            finger_group=contact_set.finger_group,
            contact_positions_object=tuple(tuple(float(value) for value in row) for row in contact_object),
            contact_normals_object=tuple(tuple(float(value) for value in row) for row in normal_object),
            closed_joint_q26=tuple(float(value) for value in q_closed),
            pregrasp_joint_q26_by_opening=pregrasps,
            standoff_joint_q26_by_distance=standoffs,
            approach_direction_object=contact_set.approach_direction_object,
            target_force_n=float(target_force_n),
            optimization_success=bool(tip_error <= 0.003 and closed_ok and bool(wrench.get("feasible", False))),
            optimization_iterations=int(getattr(result, "nit", 0)),
            tip_error_m=tip_error,
            energy=energy.to_dict(),
            input_hash=input_hash,
            collision_free_path=path_ok,
            gate_eligibility={"gate_a": closed_ok, "gate_b": gate_b_ok, "gate_c": gate_c_ok},
            metadata={
                "optimizer": "SLSQP",
                "optimizer_message": str(result.message),
                "optimizer_reported_success": bool(result.success),
                "optimized_runtime_joint_indices": optimized_columns.tolist(),
                "pinocchio_urdf_sha256": self.urdf_sha256,
                "force_closure_acceptance": "gravity_wrench_friction_cone_lp",
                "gravity_wrench": wrench,
                "fallback_tier": contact_set.fallback_tier,
                "axis_delta_m": contact_set.axis_delta_m,
                "raycast_hit": contact_set.raycast_hit,
                "pad_support_distance_m": list(pad_support_distances),
                "pad_support_vertex_local": [vertex.tolist() for vertex in support_vertices],
                "pad_support_world_offset": [value.tolist() for value in support_world_offsets],
                "frame_contact_targets_world": frame_targets.tolist(),
                "fingertip_mesh_min_z_world": fingertip_min_z.tolist(),
                "collision_model_available": self.collision_model_available,
                "collision_pair_count": self.collision_pair_count,
                "self_collision_penetration_squared_m2": self_penetration,
                "self_collision_pairs": [list(pair) for pair in self_collision_pairs],
                "scene_clearance": closed_scene,
                "object_collision_mesh_sha256": self._object_collision_mesh.sha256,
                "table_collision_mesh_sha256": self._table_collision_mesh.sha256,
                "pregrasp_collision_free": pregrasp_ok,
                "standoff_collision_free": standoff_ok,
            },
        )


def _adjacent_geometry_pair(model, geometry_model, pair) -> bool:
    first_joint = int(geometry_model.geometryObjects[pair.first].parentJoint)
    second_joint = int(geometry_model.geometryObjects[pair.second].parentJoint)
    return bool(
        first_joint == second_joint
        or int(model.parents[first_joint]) == second_joint
        or int(model.parents[second_joint]) == first_joint
    )


def _interpolated_collision_free(
    model: WujiKinematicModel,
    start: np.ndarray,
    end: np.ndarray,
    object_position: np.ndarray,
    object_rotation: np.ndarray,
    active_fingers: np.ndarray,
    *,
    samples: int = 11,
) -> bool:
    for alpha in np.linspace(0.0, 1.0, int(samples)):
        q = (1.0 - alpha) * start + alpha * end
        if model.self_collision(q):
            return False
        if not model.scene_clearance(
            q,
            object_position_world=object_position,
            object_rotation_world=object_rotation,
            active_fingers=active_fingers,
        )["valid"]:
            return False
    return True


def _skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _build_hppfcl_bvh(mesh: CollisionMesh):
    import hppfcl

    vertices = hppfcl.StdVec_Vec3f()
    for value in mesh.vertices:
        vertices.append(np.asarray(value, dtype=np.float64))
    triangles = hppfcl.StdVec_Triangle()
    for first, second, third in mesh.faces:
        triangles.append(hppfcl.Triangle(int(first), int(second), int(third)))
    model = hppfcl.BVHModelOBBRSS()
    if model.beginModel(len(mesh.faces), len(mesh.vertices)) != 0:
        raise RuntimeError("HPP-FCL beginModel failed")
    if model.addSubModel(vertices, triangles) != 0 or model.endModel() != 0:
        raise RuntimeError("HPP-FCL collision mesh construction failed")
    return model


def _fcl_clearance(first_geometry, first_transform, second_geometry, second_transform) -> tuple[float, float]:
    import hppfcl

    collision_request = hppfcl.CollisionRequest()
    collision_request.enable_contact = True
    collision_request.num_max_contacts = 32
    collision_result = hppfcl.CollisionResult()
    collision_count = hppfcl.collide(
        first_geometry,
        first_transform,
        second_geometry,
        second_transform,
        collision_request,
        collision_result,
    )
    if collision_count:
        contacts = collision_result.getContacts()
        penetration = max((max(float(contact.penetration_depth), 0.0) for contact in contacts), default=0.0)
        return -penetration, penetration
    distance_request = hppfcl.DistanceRequest()
    distance_request.enable_nearest_points = True
    distance_result = hppfcl.DistanceResult()
    distance = float(
        hppfcl.distance(
            first_geometry,
            first_transform,
            second_geometry,
            second_transform,
            distance_request,
            distance_result,
        )
    )
    return distance, 0.0


def _load_tip_collision_vertices(urdf_path: Path, tip_frames: Sequence[str]) -> tuple[np.ndarray, ...]:
    import trimesh
    from scipy.spatial.transform import Rotation
    import xml.etree.ElementTree as ET

    mesh_root = urdf_path.parent.parent / "meshes" / "right"
    urdf_root = ET.parse(urdf_path).getroot()
    vertices = []
    for frame in tip_frames:
        path = mesh_root / f"{frame}.STL"
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = trimesh.load_mesh(path, process=False)
        values = np.asarray(mesh.vertices, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
            raise ValueError(f"invalid fingertip collision mesh {path}")
        tip_link = frame.replace("_link4", "_tip_link")
        tip_path = mesh_root / f"{tip_link}.STL"
        if not tip_path.is_file():
            raise FileNotFoundError(tip_path)
        fixed_joint = next(
            (
                joint
                for joint in urdf_root.findall("joint")
                if joint.get("type") == "fixed"
                and joint.find("parent") is not None
                and joint.find("child") is not None
                and joint.find("parent").get("link") == frame
                and joint.find("child").get("link") == tip_link
            ),
            None,
        )
        if fixed_joint is None:
            raise ValueError(f"missing fixed fingertip joint {frame} -> {tip_link}")
        origin = fixed_joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(f"invalid fixed fingertip transform {frame} -> {tip_link}")
        tip_mesh = trimesh.load_mesh(tip_path, process=False)
        tip_values = np.asarray(tip_mesh.vertices, dtype=np.float64)
        if tip_values.ndim != 2 or tip_values.shape[1] != 3 or len(tip_values) == 0:
            raise ValueError(f"invalid fingertip collision mesh {tip_path}")
        transformed_tip = tip_values @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
        vertices.append(np.concatenate((values, transformed_tip), axis=0))
    return tuple(vertices)
