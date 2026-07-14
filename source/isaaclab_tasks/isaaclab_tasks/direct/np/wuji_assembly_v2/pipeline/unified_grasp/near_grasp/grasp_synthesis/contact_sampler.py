"""Deterministic same-section raycast contacts for forensic grasp synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .grasp_energy import GravityWrenchResult, gravity_wrench_feasibility, opposition_quality
from .object_spec import ContactRegion, ObjectGraspSpec


@dataclass(frozen=True)
class CollisionMesh:
    vertices: np.ndarray
    faces: np.ndarray
    sha256: str
    coordinate_frame: str
    source_prims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("collision mesh vertices must have shape [N,3]")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
            raise ValueError("collision mesh faces must have shape [M,3]")
        if np.min(faces) < 0 or np.max(faces) >= len(vertices):
            raise ValueError("collision mesh face indices are invalid")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    @property
    def face_normals(self) -> np.ndarray:
        triangles = self.triangles
        values = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)

    @property
    def face_areas2(self) -> np.ndarray:
        triangles = self.triangles
        return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


@dataclass(frozen=True)
class ContactSample:
    position_object: tuple[float, float, float]
    normal_object: tuple[float, float, float]
    region_id: str


@dataclass(frozen=True)
class ContactSet:
    sample_id: int
    finger_group: str
    contacts: tuple[ContactSample, ...]
    approach_direction_object: tuple[float, float, float]
    opposition_quality: float
    force_closure_quality: float
    source: str
    fallback_tier: str = "antipodal_pair"
    raycast_hit: bool = False
    axis_delta_m: float = float("inf")
    gravity_wrench: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def geometric_rank_key(self) -> tuple[float, ...]:
        return (
            float(bool(self.gravity_wrench.get("feasible", False))),
            self.opposition_quality,
            self.force_closure_quality,
            -self.axis_delta_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_collision_mesh(path: str | Path, *, metadata_path: str | Path | None = None) -> CollisionMesh:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
    digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
    coordinate_frame = "object_local"
    prims: tuple[str, ...] = ()
    if metadata_path is not None:
        import json

        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        coordinate_frame = str(metadata.get("coordinate_frame", coordinate_frame))
        prims = tuple(str(value) for value in metadata.get("source_prims", ()))
        expected = str(metadata.get("mesh_sha256", digest))
        if expected != digest:
            raise ValueError(f"collision mesh hash mismatch: expected {expected}, got {digest}")
    return CollisionMesh(vertices, faces, digest, coordinate_frame, prims)


def collision_mesh_from_visual(spec: ObjectGraspSpec) -> CollisionMesh:
    """Compatibility helper for pure tests; forensic runs must use runtime export."""

    import trimesh

    loaded = trimesh.load_mesh(spec.visual_mesh, process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"visual mesh is not triangular: {spec.visual_mesh}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64) * np.asarray(spec.asset_scale, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
    return CollisionMesh(vertices, faces, digest, "visual_mesh_test_fallback", (str(spec.visual_mesh),))


def crop_collision_mesh(
    mesh: CollisionMesh,
    *,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> CollisionMesh:
    """Keep audited triangles whose AABBs intersect a conservative workspace."""

    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
        raise ValueError("collision crop bounds are invalid")
    triangles = mesh.triangles
    keep = np.all(np.max(triangles, axis=1) >= low, axis=1) & np.all(
        np.min(triangles, axis=1) <= high, axis=1
    )
    selected = mesh.faces[keep]
    if len(selected) == 0:
        raise ValueError("collision crop removed every triangle")
    used = np.unique(selected.reshape(-1))
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    vertices = mesh.vertices[used]
    faces = remap[selected]
    digest = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
    return CollisionMesh(
        vertices=vertices,
        faces=faces,
        sha256=digest,
        coordinate_frame=mesh.coordinate_frame,
        source_prims=mesh.source_prims,
    )


def sample_contact_sets(
    spec: ObjectGraspSpec,
    *,
    count: int,
    seed: int,
    collision_mesh: CollisionMesh | None = None,
    axis_delta_limit_m: float = 0.0005,
    center_of_mass_object: tuple[float, float, float] | None = None,
    mass_kg: float | None = None,
    friction: float | None = None,
) -> tuple[ContactSet, ...]:
    if count <= 0:
        raise ValueError("contact sample count must be positive")
    mesh = collision_mesh or collision_mesh_from_visual(spec)
    if mesh.coordinate_frame not in {"object_local", "visual_mesh_test_fallback"}:
        raise ValueError("target collision mesh must be in object-local coordinates")
    rng = np.random.default_rng(int(seed))
    triangles = mesh.triangles
    normals = mesh.face_normals
    area2 = mesh.face_areas2
    valid_faces = area2 > 1.0e-12
    if not np.any(valid_faces):
        raise RuntimeError("collision mesh has no non-degenerate faces")
    triangles = triangles[valid_faces]
    normals = normals[valid_faces]
    raycast_index = _RaycastIndex(triangles, normals)
    probability = area2[valid_faces] / np.sum(area2[valid_faces])
    rows = []
    rejected: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(count * 200, 4000)
    gravity_object = _gravity_in_object_frame(spec)
    max_region_weight = max(float(region.sampling_weight) for region in spec.allowed_contact_regions)
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        first_face = int(rng.choice(len(triangles), p=probability))
        first_point = _sample_triangle(triangles[first_face], rng)
        first_normal = normals[first_face]
        first_region = _matching_region(spec.allowed_contact_regions, first_point, first_normal)
        if first_region is None:
            rejected["first_contact_outside_allowed_region"] += 1
            continue
        if rng.random() > float(first_region.sampling_weight) / max_region_weight:
            rejected["region_sampling_weight"] += 1
            continue
        approaches = first_region.approach_directions_object or spec.approach_directions_object
        seed_direction = np.asarray(approaches[(attempts - 1) % len(approaches)], dtype=np.float64)
        surface_seed_origin = np.asarray(first_region.center_object, dtype=np.float64).copy()
        surface_seed_jitter_m = 0.0
        if attempts % 4 and first_region.section_axis_object is not None:
            section_axis = np.asarray(first_region.section_axis_object, dtype=np.float64)
            tangent = np.cross(seed_direction, section_axis)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1.0e-9:
                tangent /= tangent_norm
                half_extents = np.asarray(first_region.half_extents_m, dtype=np.float64)
                limits = [
                    half_extents[axis] / abs(tangent[axis])
                    for axis in range(3)
                    if abs(tangent[axis]) > 1.0e-9
                ]
                max_jitter = 0.5 * min(limits)
                surface_seed_jitter_m = float(rng.uniform(-max_jitter, max_jitter))
                surface_seed_origin += surface_seed_jitter_m * tangent
        first_hit = raycast_index.opposite(
            surface_seed_origin,
            -seed_direction,
            max_distance_m=float(first_region.local_thickness_range_m[1]) + 1.0e-4,
        )
        if first_hit is None:
            rejected["region_center_surface_raycast_miss"] += 1
            continue
        first_point, first_normal, first_face = first_hit
        if _matching_region(spec.allowed_contact_regions, first_point, first_normal) != first_region:
            rejected["region_center_surface_outside_allowed_region"] += 1
            continue
        hit = raycast_index.opposite(
            first_point,
            first_normal,
            max_distance_m=float(first_region.local_thickness_range_m[1]) + 1.0e-4,
        )
        if hit is None:
            rejected["opposite_raycast_miss"] += 1
            continue
        second_point, second_normal, second_face = hit
        section_axis = np.asarray(first_region.section_axis_object, dtype=np.float64) if first_region.section_axis_object else None
        axis_delta = (
            abs(float(np.dot(second_point - first_point, section_axis)))
            if section_axis is not None
            else 0.0
        )
        section_limit = min(float(axis_delta_limit_m), float(first_region.section_tolerance_m))
        thickness = float(np.linalg.norm(second_point - first_point))
        if axis_delta > section_limit or not (
            first_region.local_thickness_range_m[0] <= thickness <= first_region.local_thickness_range_m[1]
        ):
            rejected["section_or_thickness"] += 1
            continue
        points = np.asarray((first_point, second_point))
        pair_normals = np.asarray((first_normal, second_normal))
        regions = [_matching_region(spec.allowed_contact_regions, point, normal) for point, normal in zip(points, pair_normals)]
        if any(region is None for region in regions):
            rejected["opposite_contact_outside_allowed_region"] += 1
            continue
        if any(region.region_id != first_region.region_id for region in regions if region is not None):
            rejected["opposite_contact_region_mismatch"] += 1
            continue
        opposition = opposition_quality(pair_normals)
        if opposition < 0.75:
            rejected["opposition_quality"] += 1
            continue
        wrench = gravity_wrench_feasibility(
            points,
            pair_normals,
            center_of_mass=center_of_mass_object or spec.geometry.center_object,
            mass_kg=float(mass_kg if mass_kg is not None else spec.nominal_mass_kg),
            friction=float(friction if friction is not None else spec.nominal_dynamic_friction),
            normal_force_range_n=spec.target_normal_force_range_n,
            gravity_object=gravity_object,
        )
        third = None
        if "three_point" in first_region.contact_modes:
            third = _third_contact_same_section(
                first_point,
                first_normal,
                second_point,
                second_normal,
                triangles,
                normals,
                spec,
                first_region,
                rng,
                section_limit,
                center_of_mass_object or spec.geometry.center_object,
                float(mass_kg if mass_kg is not None else spec.nominal_mass_kg),
                float(friction if friction is not None else spec.nominal_dynamic_friction),
            )
        # A heavy object may be infeasible for the pair but feasible for the
        # region's explicit three-point mode.  Preserve the pair only as the
        # deterministic base needed to materialize that fallback.
        if not wrench.feasible and third is None:
            rejected["gravity_wrench_pair_and_three_point_infeasible"] += 1
            continue
        groups = first_region.allowed_finger_groups or spec.allowed_finger_groups
        pair_groups = tuple(group for group in groups if len(group) == 2)
        if not pair_groups:
            rejected["no_pair_finger_group"] += 1
            continue
        rows.append(
            ContactSet(
                sample_id=len(rows),
                finger_group=pair_groups[0],
                contacts=tuple(
                    ContactSample(tuple(float(value) for value in point), tuple(float(value) for value in normal), region.region_id)
                    for point, normal, region in zip(points, pair_normals, regions)
                ),
                approach_direction_object=approaches[0],
                opposition_quality=opposition,
                force_closure_quality=wrench.quality,
                source="runtime_collision_mesh_raycast" if collision_mesh is not None else "visual_mesh_test_raycast",
                fallback_tier="antipodal_pair",
                raycast_hit=True,
                axis_delta_m=axis_delta,
                gravity_wrench=wrench.to_dict(),
                metadata={
                    "collision_mesh_sha256": mesh.sha256,
                    "first_face": first_face,
                    "second_face": second_face,
                    "third_contact": third,
                    "region_id": first_region.region_id,
                    "region_finger_groups": list(groups),
                    "region_approach_directions": [list(value) for value in approaches],
                    "local_thickness_m": thickness,
                    "section_axis_object": list(first_region.section_axis_object) if first_region.section_axis_object else None,
                    "surface_seed_origin_object": surface_seed_origin.tolist(),
                    "surface_seed_jitter_m": surface_seed_jitter_m,
                },
            )
        )
    if len(rows) != count:
        raise RuntimeError(
            f"contact sampler produced {len(rows)}/{count} valid sets after {attempts} attempts; "
            f"rejections={dict(sorted(rejected.items()))}"
        )
    return tuple(rows)


def fallback_variants(base: ContactSet, spec: ObjectGraspSpec) -> tuple[ContactSet, ...]:
    """Materialize the ordered fallback cascade from ObjectGraspSpec."""

    variants = []
    groups = tuple(str(value) for value in base.metadata.get("region_finger_groups", spec.allowed_finger_groups))
    approaches = tuple(
        tuple(float(component) for component in value)
        for value in base.metadata.get("region_approach_directions", spec.approach_directions_object)
    )
    for tier in spec.candidate_fallback_order:
        if tier == "antipodal_pair":
            variants.append(base)
        elif tier == "three_point":
            third = base.metadata.get("third_contact")
            three_groups = tuple(group for group in groups if len(group) == 3)
            if third is None or not three_groups:
                continue
            third_sample = ContactSample(
                position_object=tuple(float(value) for value in third["position_object"]),
                normal_object=tuple(float(value) for value in third["normal_object"]),
                region_id=str(third["region_id"]),
            )
            contacts = base.contacts + (third_sample,)
            points = np.asarray([row.position_object for row in contacts])
            normals = np.asarray([row.normal_object for row in contacts])
            wrench = gravity_wrench_feasibility(
                points,
                normals,
                center_of_mass=spec.geometry.center_object,
                mass_kg=spec.nominal_mass_kg,
                friction=spec.nominal_dynamic_friction,
                normal_force_range_n=spec.target_normal_force_range_n,
                gravity_object=_gravity_in_object_frame(spec),
            )
            if wrench.feasible:
                variants.append(
                    _replace_contact_set(
                        base,
                        finger_group=three_groups[0],
                        contacts=contacts,
                        fallback_tier=tier,
                        opposition=opposition_quality(normals),
                        wrench=wrench,
                    )
                )
        elif tier == "alternate_finger_group":
            variants.extend(
                _replace_contact_set(base, finger_group=group, fallback_tier=tier)
                for group in groups
                if len(group) == len(base.contacts) and group != base.finger_group
            )
        elif tier == "alternate_approach":
            for direction in approaches[1:]:
                variants.append(_replace_contact_set(base, approach_direction=direction, fallback_tier=tier))
        else:
            raise ValueError(f"unsupported candidate fallback tier {tier!r}")
    if not variants:
        raise ValueError("candidate fallback order produced no executable variants")
    return tuple(variants)


def retain_geometric_candidates(samples: tuple[ContactSet, ...], *, count: int) -> tuple[ContactSet, ...]:
    if count <= 0:
        raise ValueError("retain count must be positive")
    ranked = sorted(samples, key=lambda row: (row.geometric_rank_key, -row.sample_id), reverse=True)
    return tuple(ranked[:count])


def _replace_contact_set(
    base: ContactSet,
    *,
    finger_group: str | None = None,
    contacts: tuple[ContactSample, ...] | None = None,
    approach_direction: tuple[float, float, float] | None = None,
    fallback_tier: str,
    opposition: float | None = None,
    wrench: GravityWrenchResult | None = None,
) -> ContactSet:
    return ContactSet(
        sample_id=base.sample_id,
        finger_group=finger_group or base.finger_group,
        contacts=contacts or base.contacts,
        approach_direction_object=approach_direction or base.approach_direction_object,
        opposition_quality=base.opposition_quality if opposition is None else opposition,
        force_closure_quality=base.force_closure_quality if wrench is None else wrench.quality,
        source=base.source,
        fallback_tier=fallback_tier,
        raycast_hit=base.raycast_hit,
        axis_delta_m=base.axis_delta_m,
        gravity_wrench=base.gravity_wrench if wrench is None else wrench.to_dict(),
        metadata=dict(base.metadata),
    )


def _raycast_opposite(
    point: np.ndarray,
    outward_normal: np.ndarray,
    triangles: np.ndarray,
    face_normals: np.ndarray,
    *,
    face_indices: np.ndarray | None = None,
    max_distance_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if face_indices is None:
        selected_triangles = triangles
        selected_normals = face_normals
        selected_indices = np.arange(len(triangles), dtype=np.int64)
    else:
        selected_indices = np.asarray(face_indices, dtype=np.int64)
        if selected_indices.ndim != 1:
            raise ValueError("raycast face indices must be one-dimensional")
        if len(selected_indices) == 0:
            return None
        selected_triangles = triangles[selected_indices]
        selected_normals = face_normals[selected_indices]
    direction = -np.asarray(outward_normal, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    origin = np.asarray(point, dtype=np.float64) + direction * 1.0e-7
    edge_first = selected_triangles[:, 1] - selected_triangles[:, 0]
    edge_second = selected_triangles[:, 2] - selected_triangles[:, 0]
    pvec = np.cross(np.broadcast_to(direction, edge_second.shape), edge_second)
    determinant = np.einsum("ij,ij->i", edge_first, pvec)
    valid = np.abs(determinant) > 1.0e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    tvec = origin.reshape(1, 3) - selected_triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge_first)
    v = np.einsum("j,ij->i", direction, qvec) * inverse
    distance = np.einsum("ij,ij->i", edge_second, qvec) * inverse
    valid &= (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1.0 + 1.0e-9) & (distance > 1.0e-5)
    if max_distance_m is not None:
        valid &= distance <= float(max_distance_m) + 1.0e-9
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return None
    selected_index = int(indices[np.argmin(distance[indices])])
    face_index = int(selected_indices[selected_index])
    return origin + distance[selected_index] * direction, selected_normals[selected_index], face_index


class _RaycastIndex:
    """Deterministic broad phase for exact finite-segment triangle raycasts."""

    def __init__(self, triangles: np.ndarray, face_normals: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.face_normals = np.asarray(face_normals, dtype=np.float64)
        self.centroids = np.mean(self.triangles, axis=1)
        self.triangle_radii = np.max(
            np.linalg.norm(self.triangles - self.centroids[:, None, :], axis=2),
            axis=1,
        )
        # A few exported collision faces can span most of an object.  Using the
        # global maximum as every KD query margin degenerates to a full scan.
        # Large faces are checked separately with their own conservative bound.
        self.regular_radius = max(float(np.quantile(self.triangle_radii, 0.99)), 1.0e-6)
        self.large_indices = np.flatnonzero(self.triangle_radii > self.regular_radius)
        self.tree = cKDTree(self.centroids)

    def opposite(
        self,
        point: np.ndarray,
        outward_normal: np.ndarray,
        *,
        max_distance_m: float,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        if max_distance_m <= 0.0:
            raise ValueError("raycast maximum distance must be positive")
        direction = -np.asarray(outward_normal, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        origin = np.asarray(point, dtype=np.float64) + direction * 1.0e-7
        midpoint = origin + 0.5 * float(max_distance_m) * direction
        half_length = 0.5 * float(max_distance_m)
        query_radius = half_length + self.regular_radius + 1.0e-9
        regular_indices = np.asarray(self.tree.query_ball_point(midpoint, query_radius), dtype=np.int64)
        if len(self.large_indices):
            large_distance = np.linalg.norm(self.centroids[self.large_indices] - midpoint, axis=1)
            large_indices = self.large_indices[
                large_distance <= half_length + self.triangle_radii[self.large_indices] + 1.0e-9
            ]
            indices = np.unique(np.concatenate((regular_indices, large_indices)))
        else:
            indices = regular_indices
        return _raycast_opposite(
            point,
            outward_normal,
            self.triangles,
            self.face_normals,
            face_indices=indices,
            max_distance_m=max_distance_m,
        )


def _third_contact_same_section(
    first_point: np.ndarray,
    first_normal: np.ndarray,
    second_point: np.ndarray,
    second_normal: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray,
    spec: ObjectGraspSpec,
    region: ContactRegion,
    rng: np.random.Generator,
    axis_delta_limit_m: float,
    center_of_mass: tuple[float, float, float],
    mass_kg: float,
    friction: float,
) -> dict[str, Any] | None:
    axis = np.asarray(region.section_axis_object or (0.0, 0.0, 1.0), dtype=np.float64)
    target_section = float(np.dot(first_point, axis))
    projection = triangles @ axis
    axial = (np.min(projection, axis=1) <= target_section) & (np.max(projection, axis=1) >= target_section)
    side = np.abs(normals @ axis) <= 0.35
    indices = np.flatnonzero(axial & side)
    if len(indices) == 0:
        return None
    for _ in range(min(64, max(8, len(indices)))):
        face_index = int(indices[int(rng.integers(0, len(indices)))])
        point = _triangle_point_at_section(triangles[face_index], axis, target_section)
        if point is None:
            continue
        if min(float(np.linalg.norm(point - first_point)), float(np.linalg.norm(point - second_point))) <= 0.001:
            continue
        normal = normals[face_index]
        region = _matching_region(spec.allowed_contact_regions, point, normal)
        if region is None or abs(float(np.dot(point - first_point, axis))) > float(axis_delta_limit_m):
            continue
        wrench = gravity_wrench_feasibility(
            (first_point, second_point, point),
            (first_normal, second_normal, normal),
            center_of_mass=center_of_mass,
            mass_kg=mass_kg,
            friction=friction,
            normal_force_range_n=spec.target_normal_force_range_n,
            gravity_object=_gravity_in_object_frame(spec),
        )
        if wrench.feasible:
            return {
                "position_object": [float(value) for value in point],
                "normal_object": [float(value) for value in normal],
                "region_id": region.region_id,
                "face": face_index,
                "gravity_wrench": wrench.to_dict(),
            }
    return None


def _triangle_point_at_section(triangle: np.ndarray, axis: np.ndarray, target: float) -> np.ndarray | None:
    intersections = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start = triangle[first]
        end = triangle[second]
        start_value = float(np.dot(start, axis))
        end_value = float(np.dot(end, axis))
        delta = end_value - start_value
        if abs(delta) <= 1.0e-12:
            if abs(start_value - target) <= 1.0e-9:
                intersections.extend((start, end))
            continue
        ratio = (target - start_value) / delta
        if -1.0e-9 <= ratio <= 1.0 + 1.0e-9:
            intersections.append(start + ratio * (end - start))
    unique = []
    for point in intersections:
        if not any(np.linalg.norm(point - existing) <= 1.0e-9 for existing in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    return 0.5 * (unique[0] + unique[1])


def _sample_triangle(triangle: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    first, second = rng.random(2)
    sqrt_first = np.sqrt(first)
    return (1.0 - sqrt_first) * triangle[0] + sqrt_first * (1.0 - second) * triangle[1] + sqrt_first * second * triangle[2]


def _matching_region(regions: tuple[ContactRegion, ...], point: np.ndarray, normal: np.ndarray) -> ContactRegion | None:
    for region in regions:
        center = np.asarray(region.center_object)
        half = np.asarray(region.half_extents_m)
        if np.all(np.abs(point - center) <= half + 1.0e-9):
            allowed_direction = True
            if region.allowed_normal_object is not None:
                allowed_direction = float(np.dot(normal, np.asarray(region.allowed_normal_object))) >= 0.5
            allowed_axis = True
            if region.normal_axis_object is not None:
                allowed_axis = abs(float(np.dot(normal, np.asarray(region.normal_axis_object)))) <= float(
                    region.normal_abs_dot_max
                )
            if allowed_direction and allowed_axis:
                return region
    return None


def _gravity_in_object_frame(spec: ObjectGraspSpec) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat = np.asarray(spec.canonical_object_quat_wxyz, dtype=np.float64)
    rotation = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
    return rotation.T @ np.asarray((0.0, 0.0, -9.81))
