"""Deterministic, bounded mechanical-cage candidates for the Chair4 Frame."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CageCandidate, CageTopology, ControlFeature, FrameBar


_REFERENCE_IDS = (
    "251b607e0b5e084a",
    "63d08904442ba5ea",
    "8ecad8a93bbcd10e",
)


@dataclass(frozen=True)
class _Template:
    topology: CageTopology
    bar_id: str
    finger_group: str
    approach_sign: int
    hook_finger: int = 0


class FrameCagePlanner:
    """Generate twelve topology candidates from one audited runtime mesh."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        config: Mapping[str, Any],
        runtime_mesh_path: str | Path,
        runtime_audit_path: str | Path,
        runtime_seed_path: str | Path,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.config = dict(config)
        self.cage_cfg = dict(config["cage"])
        self.mesh_path = Path(runtime_mesh_path).resolve()
        self.audit_path = Path(runtime_audit_path).resolve()
        self.seed_path = Path(runtime_seed_path).resolve()
        self.audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
        self.seed = json.loads(self.seed_path.read_text(encoding="utf-8"))
        mesh = np.load(self.mesh_path)
        self.vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        self.faces = np.asarray(mesh["faces"], dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or self.faces.ndim != 2:
            raise ValueError("Frame runtime collision mesh arrays are invalid")
        self.mesh_sha256 = hashlib.sha256(self.vertices.tobytes() + self.faces.tobytes()).hexdigest()
        expected = str(self.audit.get("target_object", {}).get("mesh_sha256") or "")
        if expected and self.mesh_sha256 != expected:
            raise ValueError("runtime Frame mesh hash does not match audit")
        self.bars, self.hole_bounds = extract_frame_bars(self.vertices, self.faces)
        self.source_hole_open = ray_through_central_hole(self.vertices, self.faces, self.hole_bounds)

        from ..pipeline.unified_grasp.near_grasp.grasp_synthesis.contact_sampler import CollisionMesh
        from ..pipeline.unified_grasp.near_grasp.grasp_synthesis.wuji_ik import WujiKinematicModel

        urdf = self.repo_root / str(config["assets"]["wuji_urdf"])
        self.model = WujiKinematicModel(urdf, package_dirs=[urdf.parent])
        table_path = self.mesh_path.with_name("table_collision_mesh.npz")
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        table = np.load(table_path)
        object_mesh = CollisionMesh(
            vertices=self.vertices,
            faces=self.faces,
            sha256=self.mesh_sha256,
            coordinate_frame="object_local",
            source_prims=tuple(self.audit.get("target_object", {}).get("source_prims", ())),
        )
        table_vertices = np.asarray(table["vertices"], dtype=np.float64)
        table_faces = np.asarray(table["faces"], dtype=np.int64)
        table_mesh = CollisionMesh(
            vertices=table_vertices,
            faces=table_faces,
            sha256=hashlib.sha256(table_vertices.tobytes() + table_faces.tobytes()).hexdigest(),
            coordinate_frame="env_local",
            source_prims=tuple(self.audit.get("table", {}).get("source_prims", ())),
        )
        self.model.set_scene_collision_meshes(object_mesh, table_mesh)
        self.lower = np.asarray(self.seed["joint_lower26"], dtype=np.float64)
        self.upper = np.asarray(self.seed["joint_upper26"], dtype=np.float64)
        self.default_q = np.asarray(self.seed["q_seed26"], dtype=np.float64)
        self.object_position = np.asarray(
            self.audit.get("canonical_object_position", (-0.32467, -0.01761, 1.29095)), dtype=np.float64
        )
        self.object_rotation = np.eye(3, dtype=np.float64)
        self.references = _load_reference_candidates(
            self.repo_root / str(config["assets"]["old_frame_candidates"])
        )

    def generate(self) -> tuple[CageCandidate, ...]:
        templates = _candidate_templates()
        cap = int(self.config["runtime"]["candidate_cap"])
        if len(templates) > cap:
            raise ValueError("Frame cage template count exceeds configured cap")
        return tuple(self._solve_template(index, template) for index, template in enumerate(templates))

    def geometry_audit(self) -> dict[str, Any]:
        return {
            "runtime_mesh_path": str(self.mesh_path),
            "runtime_mesh_sha256": self.mesh_sha256,
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "bars": [bar.to_dict() for bar in self.bars.values()],
            "central_hole_bounds_object": self.hole_bounds.tolist(),
            "source_mesh_hole_raycast_open": self.source_hole_open,
            "physx_scene_query_hole_open": self.audit.get("physx_scene_query_hole_open"),
            "central_hole_is_contact_surface": False,
        }

    def _solve_template(self, index: int, template: _Template) -> CageCandidate:
        fingers = tuple(int(value) - 1 for value in template.finger_group)
        positions, normals, roles, surfaces = self._contact_geometry(template)
        approach = np.asarray((0.0, float(template.approach_sign), 0.0), dtype=np.float64)
        reference_id = "251b607e0b5e084a" if template.approach_sign > 0 else "63d08904442ba5ea"
        reference = self.references[reference_id]
        seed = np.asarray(reference["closed_joint_q26"], dtype=np.float64).copy()
        active_columns = [6 + finger + 5 * joint for finger in fingers for joint in range(4)]
        inactive_columns = sorted(set(range(6, 26)) - set(active_columns))
        seed = np.clip(seed, self.lower + 0.002, self.upper - 0.002)
        exact_reference_path = bool(
            template.topology == CageTopology.TWO_FINGER_BRACKET and template.finger_group == "23"
        )
        if exact_reference_path:
            q_closed = seed
            residual, vertices = self._pad_residual(q_closed, fingers, positions, normals)
        else:
            seed[inactive_columns] = self.default_q[inactive_columns]
            q_closed, residual, vertices = self._solve_pad_targets(seed, fingers, positions, normals)
        hook_runtime_valid = bool(
            template.topology != CageTopology.HOOK_THROUGH_FRAME
            or (
                self.source_hole_open
                and self.audit.get("physx_scene_query_hole_open") is True
            )
        )
        continuation: list[dict[str, Any]] = []
        preclose_continuation: list[dict[str, Any]] = []
        if hook_runtime_valid and not exact_reference_path:
            q_closed, continuation = self._collision_continuation(q_closed, fingers, positions, normals)
            residual, vertices = self._pad_residual(q_closed, fingers, positions, normals)

        if exact_reference_path:
            preclose_positions = positions + normals * float(self.cage_cfg["preclose_clearance_m"])
            preclose_seed = np.clip(
                np.asarray(reference["pregrasp_joint_q26_by_opening"]["3mm"], dtype=np.float64),
                self.lower + 0.002,
                self.upper - 0.002,
            )
            q_preclose, preclose_continuation = self._collision_continuation(
                preclose_seed, fingers, preclose_positions, normals
            )
            preclose_residual, _ = self._pad_residual(q_preclose, fingers, preclose_positions, normals)
            q_standoff = q_preclose.copy()
            q_standoff[:3] -= approach * float(self.cage_cfg["standoff_distance_m"])
            q_standoff = np.clip(q_standoff, self.lower + 0.002, self.upper - 0.002)
        else:
            preclose_positions = positions + normals * float(self.cage_cfg["preclose_clearance_m"])
            q_preclose, preclose_residual, _ = self._solve_pad_targets(q_closed, fingers, preclose_positions, normals)
            q_standoff = q_preclose.copy()
            q_standoff[:3] -= approach * float(self.cage_cfg["standoff_distance_m"])
            q_standoff = np.clip(q_standoff, self.lower + 0.002, self.upper - 0.002)

        self_penetration, self_pairs = self.model.self_collision_penetration(q_closed)
        closed_scene = self.model.scene_clearance(
            q_closed,
            object_position_world=self.object_position,
            object_rotation_world=self.object_rotation,
            active_fingers=fingers,
        )
        forbidden_penetration = max(
            (
                float(row["penetration_m"])
                for row in closed_scene["object_pairs"]
                if not bool(row["active_target_tip"])
            ),
            default=0.0,
        )
        self_free = bool(self_penetration <= float(self.cage_cfg["self_penetration_limit_m"]) ** 2)
        table_free = bool(
            float(closed_scene["whole_hand_table_min_clearance_m"])
            >= float(self.cage_cfg["forbidden_clearance_m"])
        )
        path_free, path_reason = self._path_is_free(q_preclose, q_closed, fingers)
        residual_max = float(np.max(residual))
        preclose_projection = self._preclose_projection(q_preclose, fingers, positions, normals)
        preclose_gap_valid = bool(
            np.all(preclose_projection >= 0.002 - 1.0e-6)
            and np.all(preclose_projection <= 0.003 + 1.0e-6)
        )
        escape_gap = max(residual_max, max(0.0, float(np.max(np.abs(preclose_projection))) - 0.003))
        bar = self.bars[template.bar_id]
        cage_margin = float(bar.local_thickness_m - float(self.cage_cfg["cage_safety_margin_m"]) - escape_gap)
        joint_margin = float(np.min(np.minimum(q_closed - self.lower, self.upper - q_closed)))
        enclosure_valid = _enclosure_valid(template.topology, positions, normals, bar, self.hole_bounds)
        failure = ""
        if residual_max > 0.006:
            failure = "IK_UNREACHABLE"
        elif not hook_runtime_valid:
            failure = "CAGE_MARGIN_INVALID:PHYSX_HOLE_NOT_VERIFIED_OPEN"
        elif not enclosure_valid or cage_margin <= 0.0:
            failure = "CAGE_MARGIN_INVALID"
        elif not self_free:
            failure = "PATH_COLLISION:SELF"
        elif not table_free or forbidden_penetration > float(self.cage_cfg["target_penetration_limit_m"]):
            failure = "PATH_COLLISION:SCENE"
        elif not path_free:
            failure = f"PATH_COLLISION:{path_reason}"
        elif not preclose_gap_valid:
            failure = "PATH_COLLISION:PRECLOSE_GAP_OUT_OF_RANGE"
        elif joint_margin <= 0.0001:
            failure = "IK_UNREACHABLE:JOINT_LIMIT"
        solver_status = "VALID" if not failure else "REJECTED"
        features = tuple(
            ControlFeature(
                finger_index=finger + 1,
                link_name=f"right_finger{finger + 1}_link4",
                local_support_vertex=tuple(float(value) for value in vertex),
                target_position_object=tuple(float(value) for value in position),
                target_normal_object=tuple(float(value) for value in normal),
                surface_id=surface,
                role=role,
            )
            for finger, vertex, position, normal, surface, role in zip(
                fingers, vertices, positions, normals, surfaces, roles
            )
        )
        payload = {
            "schema": 1,
            "index": index,
            "topology": template.topology.value,
            "bar": template.bar_id,
            "group": template.finger_group,
            "approach": template.approach_sign,
            "mesh": self.mesh_sha256,
            "q": np.round(q_closed, 9).tolist(),
        }
        candidate_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return CageCandidate(
            candidate_id=candidate_id,
            topology=template.topology,
            bar_id=template.bar_id,
            finger_group=template.finger_group,
            preclose_q26=tuple(float(value) for value in q_preclose),
            closed_q26=tuple(float(value) for value in q_closed),
            standoff_q26=tuple(float(value) for value in q_standoff),
            intended_contact_links=tuple(feature.link_name for feature in features),
            intended_contact_surfaces=tuple(surfaces),
            control_features=features,
            approach_direction_object=tuple(float(value) for value in approach),
            cage_margin_m=cage_margin,
            gravity_escape_gap_m=escape_gap,
            bar_local_thickness_m=bar.local_thickness_m,
            forbidden_penetration_m=forbidden_penetration,
            self_collision_free=self_free,
            table_collision_free=table_free,
            preclose_path_free=path_free,
            solver_status=solver_status,
            failure_reason=failure,
            source_seed_id=reference_id,
            runtime_mesh_sha256=self.mesh_sha256,
            max_contact_residual_m=residual_max,
            joint_limit_margin_rad=joint_margin,
            metadata={
                "template_index": index,
                "self_collision_pairs": [list(pair) for pair in self_pairs],
                "scene_clearance": closed_scene,
                "preclose_residual_m": preclose_residual.tolist(),
                "preclose_normal_projection_m": preclose_projection.tolist(),
                "preclose_gap_valid": preclose_gap_valid,
                "source_mesh_hole_open": self.source_hole_open,
                "physx_scene_query_hole_open": self.audit.get("physx_scene_query_hole_open"),
                "collision_continuation": continuation,
                "preclose_collision_continuation": preclose_continuation,
            },
        )

    def _contact_geometry(
        self, template: _Template
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
        bar = self.bars[template.bar_id]
        lower = np.asarray(bar.surface_bounds_object[0])
        upper = np.asarray(bar.surface_bounds_object[1])
        center = np.asarray(bar.center_object)
        front_y = upper[1]
        back_y = lower[1]
        upper_z = upper[2] - 0.40 * (upper[2] - lower[2])
        sign = float(template.approach_sign)
        if template.topology == CageTopology.HOOK_THROUGH_FRAME:
            hole_lower, hole_upper = self.hole_bounds
            x = hole_lower[0] + 0.32 * (hole_upper[0] - hole_lower[0])
            underside_z = self.bars["top_horizontal"].surface_bounds_object[0][2]
            support_y = front_y if sign > 0 else back_y
            support_normal = (0.0, 1.0 if sign > 0 else -1.0, 0.0)
            positions = np.asarray(((x, support_y, underside_z + 0.004), (x, center[1], underside_z)), dtype=np.float64)
            normals = np.asarray((support_normal, (0.0, 0.0, -1.0)), dtype=np.float64)
            roles = ("OUTER_STABILIZER", "GRAVITY_HOOK")
            surfaces = ("top_horizontal_outer", "top_horizontal_underside")
            return positions, normals, roles, surfaces
        first_y, second_y = (front_y, back_y) if sign > 0 else (back_y, front_y)
        first_normal = (0.0, 1.0 if sign > 0 else -1.0, 0.0)
        second_normal = (0.0, -1.0 if sign > 0 else 1.0, 0.0)
        if template.topology == CageTopology.TWO_FINGER_BRACKET:
            if template.finger_group == "23":
                reference_id = "251b607e0b5e084a" if template.approach_sign > 0 else "63d08904442ba5ea"
                reference = self.references[reference_id]
                return (
                    np.asarray(reference["contact_positions_object"], dtype=np.float64),
                    np.asarray(reference["contact_normals_object"], dtype=np.float64),
                    ("BRACKET_SIDE_A", "BRACKET_SIDE_B"),
                    (f"{bar.bar_id}_side_a", f"{bar.bar_id}_side_b"),
                )
            positions = np.asarray(((center[0], first_y, upper_z), (center[0], second_y, upper_z)), dtype=np.float64)
            normals = np.asarray((first_normal, second_normal), dtype=np.float64)
            roles = ("BRACKET_SIDE_A", "BRACKET_SIDE_B")
            surfaces = (f"{bar.bar_id}_side_a", f"{bar.bar_id}_side_b")
            return positions, normals, roles, surfaces
        positions = np.asarray(
            (
                (center[0], first_y, upper_z),
                (center[0], second_y, upper_z + 0.012),
                (center[0], second_y, upper_z - 0.012),
            ),
            dtype=np.float64,
        )
        normals = np.asarray((first_normal, second_normal, second_normal), dtype=np.float64)
        roles = ("WRAP_OPPOSING", "WRAP_PAIR_UPPER", "WRAP_PAIR_LOWER")
        surfaces = tuple(f"{bar.bar_id}_wrap_{index}" for index in range(3))
        return positions, normals, roles, surfaces

    def _solve_pad_targets(
        self,
        seed: np.ndarray,
        fingers: Sequence[int],
        targets_object: np.ndarray,
        normals_object: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        q = np.asarray(seed, dtype=np.float64).copy()
        targets = self.object_position.reshape(1, 3) + targets_object @ self.object_rotation.T
        normals = normals_object @ self.object_rotation.T
        active = np.asarray(
            list(range(6)) + [6 + int(finger) + 5 * joint for finger in fingers for joint in range(4)],
            dtype=np.int64,
        )
        damping = float(self.cage_cfg["dls_damping"])
        step_limit = float(self.cage_cfg["offline_joint_step_limit_rad"])
        vertices: tuple[np.ndarray, ...] = ()
        residual = np.full(len(fingers), np.inf)
        for _ in range(int(self.cage_cfg["dls_iterations"])):
            points, _rotations, jacobians, vertices = self.model.pad_points_and_jacobians(q, fingers, normals)
            error = targets - points
            residual = np.linalg.norm(error, axis=1)
            if float(np.max(residual)) <= 0.0005:
                break
            matrix = jacobians[:, :, active].reshape(-1, len(active))
            vector = error.reshape(-1)
            dq = matrix.T @ np.linalg.solve(
                matrix @ matrix.T + (damping**2) * np.eye(matrix.shape[0]), vector
            )
            q[active] += np.clip(dq, -step_limit, step_limit)
            q = np.clip(q, self.lower + 0.002, self.upper - 0.002)
        points, _rotations, _jacobians, vertices = self.model.pad_points_and_jacobians(q, fingers, normals)
        residual = np.linalg.norm(targets - points, axis=1)
        return q, residual, vertices

    def _pad_residual(
        self,
        q: np.ndarray,
        fingers: Sequence[int],
        targets_object: np.ndarray,
        normals_object: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        targets = self.object_position.reshape(1, 3) + targets_object @ self.object_rotation.T
        normals = normals_object @ self.object_rotation.T
        points, _rotations, _jacobians, vertices = self.model.pad_points_and_jacobians(q, fingers, normals)
        return np.linalg.norm(targets - points, axis=1), vertices

    def _collision_continuation(
        self,
        q_seed: np.ndarray,
        fingers: Sequence[int],
        targets_object: np.ndarray,
        normals_object: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        from scipy.optimize import minimize

        q = np.asarray(q_seed, dtype=np.float64).copy()
        active = np.asarray(
            list(range(6)) + [6 + int(finger) + 5 * joint for finger in fingers for joint in range(4)],
            dtype=np.int64,
        )
        reference = q.copy()
        targets = self.object_position.reshape(1, 3) + targets_object @ self.object_rotation.T
        normals = normals_object @ self.object_rotation.T
        rows: list[dict[str, Any]] = []

        def objective(values: np.ndarray, weight: float) -> float:
            candidate = q.copy()
            candidate[active] = values
            points = self.model.pad_points_and_jacobians(candidate, fingers, normals)[0]
            residual = points - targets
            self_penetration, _ = self.model.self_collision_penetration(candidate)
            scene = self.model.scene_clearance(
                candidate,
                object_position_world=self.object_position,
                object_rotation_world=self.object_rotation,
                active_fingers=fingers,
            )
            regularization = candidate[active] - reference[active]
            return float(
                np.sum(residual * residual)
                + weight * (self_penetration + float(scene["penalty_m2"]))
                + 1.0e-5 * np.sum(regularization * regularization)
            )

        for weight in (1.0e2, 1.0e4, 1.0e6):
            result = minimize(
                lambda values, current_weight=weight: objective(values, current_weight),
                q[active],
                method="SLSQP",
                bounds=list(zip(self.lower[active] + 0.002, self.upper[active] - 0.002)),
                options={"maxiter": 16, "ftol": 1.0e-12, "disp": False},
            )
            q[active] = result.x
            residual, _ = self._pad_residual(q, fingers, targets_object, normals_object)
            scene = self.model.scene_clearance(
                q,
                object_position_world=self.object_position,
                object_rotation_world=self.object_rotation,
                active_fingers=fingers,
            )
            self_penetration, pairs = self.model.self_collision_penetration(q)
            rows.append(
                {
                    "weight": weight,
                    "reported_success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(getattr(result, "nit", 0)),
                    "max_contact_residual_m": float(np.max(residual)),
                    "scene_valid": bool(scene["valid"]),
                    "scene_penalty_m2": float(scene["penalty_m2"]),
                    "self_penetration_squared_m2": float(self_penetration),
                    "self_collision_pairs": [list(pair) for pair in pairs],
                }
            )
            if bool(scene["valid"]) and self_penetration <= float(self.cage_cfg["self_penetration_limit_m"]) ** 2:
                break
        return q, rows

    def _preclose_projection(
        self, q: np.ndarray, fingers: Sequence[int], contacts: np.ndarray, normals: np.ndarray
    ) -> np.ndarray:
        target_world = self.object_position.reshape(1, 3) + contacts
        points = self.model.pad_points_and_jacobians(q, fingers, normals)[0]
        return np.sum((points - target_world) * normals, axis=1)

    def _path_is_free(self, start: np.ndarray, end: np.ndarray, fingers: Sequence[int]) -> tuple[bool, str]:
        for index, alpha in enumerate(np.linspace(0.0, 1.0, int(self.cage_cfg["path_samples"]))):
            q = (1.0 - alpha) * start + alpha * end
            self_penetration, _ = self.model.self_collision_penetration(q)
            if self_penetration > float(self.cage_cfg["self_penetration_limit_m"]) ** 2:
                return False, f"SELF_AT_{index}"
            scene = self.model.scene_clearance(
                q,
                object_position_world=self.object_position,
                object_rotation_world=self.object_rotation,
                active_fingers=fingers,
            )
            if not scene["valid"]:
                return False, f"SCENE_AT_{index}"
        return True, ""


def extract_frame_bars(vertices: np.ndarray, faces: np.ndarray | None = None) -> tuple[dict[str, FrameBar], np.ndarray]:
    """Fit four bars by clustering triangles that span each principal bar axis."""

    values = np.asarray(vertices, dtype=np.float64)
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    if faces is None:
        raise ValueError("Frame bar extraction requires runtime mesh faces")
    triangles = values[np.asarray(faces, dtype=np.int64)]
    spans = np.ptp(triangles, axis=1)
    horizontal_faces = np.flatnonzero(spans[:, 0] > 0.10)
    vertical_faces = np.flatnonzero(spans[:, 2] > 0.10)
    if len(horizontal_faces) < 4 or len(vertical_faces) < 4:
        raise ValueError("runtime Frame mesh does not expose four long-axis bar surfaces")
    horizontal_centers = np.mean(triangles[horizontal_faces, :, 2], axis=1)
    vertical_centers = np.mean(triangles[vertical_faces, :, 0], axis=1)
    horizontal_groups = _split_two_groups(horizontal_faces, horizontal_centers)
    vertical_groups = _split_two_groups(vertical_faces, vertical_centers)
    horizontal_groups.sort(key=lambda group: float(np.mean(triangles[group, :, 2])), reverse=True)
    vertical_groups.sort(key=lambda group: float(np.mean(triangles[group, :, 0])))
    groups = {
        "left_vertical": (vertical_groups[0], 2, "left_of_hole"),
        "right_vertical": (vertical_groups[1], 2, "right_of_hole"),
        "top_horizontal": (horizontal_groups[0], 0, "above_hole"),
        "bottom_horizontal": (horizontal_groups[1], 0, "below_hole"),
    }
    bars: dict[str, FrameBar] = {}
    for bar_id, (face_ids, axis_index, relation) in groups.items():
        vertex_ids = np.unique(np.asarray(faces, dtype=np.int64)[face_ids].reshape(-1))
        selected = values[vertex_ids]
        if len(selected) < 8:
            raise ValueError(f"runtime Frame mesh does not contain enough vertices for {bar_id}")
        fitted_lower = np.min(selected, axis=0)
        fitted_upper = np.max(selected, axis=0)
        center = 0.5 * (fitted_lower + fitted_upper)
        half = 0.5 * (fitted_upper - fitted_lower)
        cross = [2.0 * half[index] for index in range(3) if index != axis_index]
        thickness = float(min(cross))
        bars[bar_id] = FrameBar(
            bar_id=bar_id,
            center_object=tuple(float(value) for value in center),
            axes_object=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            half_extents_m=tuple(float(value) for value in half),
            axis_index=axis_index,
            local_thickness_m=thickness,
            surface_bounds_object=(
                tuple(float(value) for value in fitted_lower),
                tuple(float(value) for value in fitted_upper),
            ),
            hole_relation=relation,
            source_vertex_count=int(len(selected)),
        )
    hole = np.asarray(
        (
            (bars["left_vertical"].surface_bounds_object[1][0], lower[1], bars["bottom_horizontal"].surface_bounds_object[1][2]),
            (bars["right_vertical"].surface_bounds_object[0][0], upper[1], bars["top_horizontal"].surface_bounds_object[0][2]),
        ),
        dtype=np.float64,
    )
    if np.any(hole[1] <= hole[0]):
        raise ValueError("runtime Frame central hole bounds are invalid")
    return bars, hole


def _split_two_groups(face_ids: np.ndarray, centers: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(centers)
    sorted_centers = centers[order]
    gaps = np.diff(sorted_centers)
    if len(gaps) == 0:
        raise ValueError("cannot split a single Frame bar surface")
    split = int(np.argmax(gaps)) + 1
    if split == 0 or split == len(order):
        raise ValueError("Frame bar surface split is degenerate")
    return [face_ids[order[:split]], face_ids[order[split:]]]


def ray_through_central_hole(vertices: np.ndarray, faces: np.ndarray, hole_bounds: np.ndarray) -> bool:
    center = 0.5 * (hole_bounds[0] + hole_bounds[1])
    origin = center.copy()
    origin[1] = float(np.min(vertices[:, 1]) - 0.01)
    distance = float(np.max(vertices[:, 1]) - origin[1] + 0.01)
    hits = _ray_triangle_distances(origin, np.asarray((0.0, 1.0, 0.0)), vertices, faces)
    return not bool(np.any((hits >= 0.0) & (hits <= distance)))


def _ray_triangle_distances(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1.0e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    s = np.broadcast_to(origin, triangles[:, 0].shape) - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("j,ij->i", direction, q)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0)
    return t[hit]


def _candidate_templates() -> tuple[_Template, ...]:
    return (
        _Template(CageTopology.HOOK_THROUGH_FRAME, "top_horizontal", "23", 1, 3),
        _Template(CageTopology.HOOK_THROUGH_FRAME, "top_horizontal", "23", -1, 3),
        _Template(CageTopology.HOOK_THROUGH_FRAME, "top_horizontal", "34", 1, 4),
        _Template(CageTopology.HOOK_THROUGH_FRAME, "top_horizontal", "34", -1, 4),
        _Template(CageTopology.THREE_FINGER_WRAP, "left_vertical", "234", 1),
        _Template(CageTopology.THREE_FINGER_WRAP, "left_vertical", "234", -1),
        _Template(CageTopology.THREE_FINGER_WRAP, "right_vertical", "234", 1),
        _Template(CageTopology.THREE_FINGER_WRAP, "right_vertical", "234", -1),
        _Template(CageTopology.TWO_FINGER_BRACKET, "left_vertical", "23", 1),
        _Template(CageTopology.TWO_FINGER_BRACKET, "left_vertical", "23", -1),
        _Template(CageTopology.TWO_FINGER_BRACKET, "left_vertical", "34", 1),
        _Template(CageTopology.TWO_FINGER_BRACKET, "left_vertical", "34", -1),
    )


def _enclosure_valid(
    topology: CageTopology, positions: np.ndarray, normals: np.ndarray, bar: FrameBar, hole_bounds: np.ndarray
) -> bool:
    if topology == CageTopology.HOOK_THROUGH_FRAME:
        hook = positions[-1]
        return bool(
            hole_bounds[0, 0] < hook[0] < hole_bounds[1, 0]
            and hook[2] <= bar.surface_bounds_object[0][2] + 0.0001
            and float(np.dot(normals[-1], (0.0, 0.0, -1.0))) > 0.8
        )
    side_signs = np.sign(normals[:, 1])
    if not (np.any(side_signs > 0) and np.any(side_signs < 0)):
        return False
    center = np.asarray(bar.center_object)
    low = np.min(positions[:, 1])
    high = np.max(positions[:, 1])
    if not low <= center[1] <= high:
        return False
    return topology != CageTopology.THREE_FINGER_WRAP or len(positions) == 3


def _load_reference_candidates(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in _REFERENCE_IDS:
            rows[candidate_id] = row
    missing = sorted(set(_REFERENCE_IDS) - set(rows))
    if missing:
        raise ValueError(f"missing Frame reference candidates: {missing}")
    return rows
