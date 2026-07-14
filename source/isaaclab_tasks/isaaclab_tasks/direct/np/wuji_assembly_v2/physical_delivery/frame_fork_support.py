"""Deterministic three-finger fork preload for ChairAssembly4 Frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ForkSupportPose, finite_vector


_FINGERS = (2, 3, 4)
_LINK_LEVELS = (3, 4)


class FrameForkSupportBuilder:
    """Construct one fixed fork pose; this class deliberately has no search API."""

    def __init__(self, urdf_path: str | Path, geometry_audit_path: str | Path):
        from ..pipeline.unified_grasp.near_grasp.grasp_synthesis.wuji_ik import WujiKinematicModel

        self.urdf_path = Path(urdf_path).resolve()
        self.geometry_audit_path = Path(geometry_audit_path).resolve()
        self.geometry = json.loads(self.geometry_audit_path.read_text(encoding="utf-8"))
        self.model = WujiKinematicModel(self.urdf_path, package_dirs=[self.urdf_path.parent])
        self._mesh_vertices = _load_link_vertices(self.urdf_path)
        self._top_bar = next(row for row in self.geometry["bars"] if row["bar_id"] == "top_horizontal")
        self._hole_bounds = np.asarray(self.geometry["central_hole_bounds_object"], dtype=np.float64)

    def build(
        self,
        *,
        hand_preshape_q20: Sequence[float],
        hand_close_q20: Sequence[float],
        frame_position: Sequence[float],
        frame_quat_wxyz: Sequence[float],
        fixed_position: Sequence[float],
        fixed_quat_wxyz: Sequence[float],
        insertion_axis_fixed: Sequence[float],
        target_gap_m: float = 0.001,
        height_adjust_m: float = 0.0,
        normal_adjust_m: float = 0.0,
        partial_hook_fraction: float = 0.0,
    ) -> ForkSupportPose:
        preshape = np.asarray(finite_vector(hand_preshape_q20, 20, "hand_preshape_q20"))
        close = np.asarray(finite_vector(hand_close_q20, 20, "hand_close_q20"))
        frame_pos = np.asarray(finite_vector(frame_position, 3, "frame_position"))
        fixed_pos = np.asarray(finite_vector(fixed_position, 3, "fixed_position"))
        frame_rotation = _quat_wxyz_matrix(frame_quat_wxyz)
        fixed_rotation = _quat_wxyz_matrix(fixed_quat_wxyz)
        axis_world = fixed_rotation @ np.asarray(finite_vector(insertion_axis_fixed, 3, "axis"))
        axis_world = _unit(axis_world)

        base_q = np.concatenate((np.zeros(6, dtype=np.float64), preshape))
        placements = self._frame_placements(base_q)
        finger_centers = {
            finger: 0.5
            * (
                placements[f"right_finger{finger}_link3"][0]
                + placements[f"right_finger{finger}_link4"][0]
            )
            for finger in _FINGERS
        }
        spread_hand = _unit(finger_centers[4] - finger_centers[2])
        longitudinal_hand = _unit(
            np.mean(
                [
                    placements[f"right_finger{finger}_link4"][0]
                    - placements[f"right_finger{finger}_link3"][0]
                    for finger in _FINGERS
                ],
                axis=0,
            )
        )
        source = _orthonormal_basis(spread_hand, longitudinal_hand)
        target_spread = _unit(frame_rotation[:, 0])
        target_longitudinal = -axis_world
        target = _orthonormal_basis(target_spread, target_longitudinal)
        wrist_rotation = target @ source.T
        wrist_rpy = _serial_xyz_from_matrix(wrist_rotation)

        oriented_q = base_q.copy()
        oriented_q[3:6] = wrist_rpy
        support = self._upper_support_points(oriented_q)
        support_mean = np.mean([row[2] for row in support], axis=0)
        bar_corners = _bounds_corners(np.asarray(self._top_bar["surface_bounds_object"], dtype=np.float64))
        bar_world = frame_pos + bar_corners @ frame_rotation.T
        underside_z = float(np.min(bar_world[:, 2]))
        bar_center_world = frame_pos + np.mean(bar_corners, axis=0) @ frame_rotation.T
        target_mean = bar_center_world.copy()
        target_mean[2] = underside_z - float(target_gap_m)
        translation = target_mean - support_mean
        translation += np.array((0.0, 0.0, float(height_adjust_m)), dtype=np.float64)
        translation += axis_world * float(normal_adjust_m)

        q26 = oriented_q.copy()
        q26[:3] = translation
        final_support = self._upper_support_points(q26)
        support_world = np.asarray([row[2] for row in final_support], dtype=np.float64)
        final_mean = np.mean(support_world, axis=0)
        signed_gap = underside_z - float(final_mean[2])

        hook = preshape.copy()
        fraction = float(np.clip(partial_hook_fraction, 0.0, 0.30))
        for finger in _FINGERS:
            for joint in (2, 3):
                column = (finger - 1) + 5 * joint
                hook[column] = preshape[column] + fraction * (close[column] - preshape[column])

        palm_pos, palm_rotation = self._frame_placements(q26)["right_palm_link"]
        rotation_error = _rotation_error_deg(palm_rotation, wrist_rotation)
        position_error = float(np.linalg.norm(final_mean - target_mean - axis_world * float(normal_adjust_m)))
        hole_min, hole_max = self._hole_bounds
        support_object = (support_world - frame_pos) @ frame_rotation
        return ForkSupportPose(
            q26=tuple(float(value) for value in q26),
            fork_q20=tuple(float(value) for value in preshape),
            hook_q20=tuple(float(value) for value in hook),
            wrist_q6=tuple(float(value) for value in q26[:6]),
            frame_reset_position=tuple(float(value) for value in frame_pos),
            frame_reset_quat_wxyz=tuple(float(value) for value in frame_quat_wxyz),
            fixed_reset_position=tuple(float(value) for value in fixed_pos),
            fixed_reset_quat_wxyz=tuple(float(value) for value in fixed_quat_wxyz),
            support_links=tuple(row[0] for row in final_support),
            support_points_hand=tuple(tuple(float(value) for value in row[1]) for row in final_support),
            support_points_world=tuple(tuple(float(value) for value in row[2]) for row in final_support),
            signed_gap_m=float(signed_gap),
            normal_offset_m=float(normal_adjust_m),
            partial_hook_fraction=fraction,
            fk_position_error_m=position_error,
            fk_rotation_error_deg=rotation_error,
            metadata={
                "construction": "ONE_FIXED_THREE_FINGER_FORK_SUPPORT",
                "candidate_generation_used": False,
                "frame_spread_axis_world": target_spread.tolist(),
                "finger_longitudinal_axis_world": target_longitudinal.tolist(),
                "support_direction_world": [0.0, 0.0, 1.0],
                "axis_world": axis_world.tolist(),
                "bar_underside_world_z": underside_z,
                "support_points_object": support_object.tolist(),
                "hole_bounds_object": self._hole_bounds.tolist(),
                "support_points_within_hole_x": bool(
                    np.all((support_object[:, 0] >= hole_min[0]) & (support_object[:, 0] <= hole_max[0]))
                ),
            },
        )

    def _frame_placements(self, q26: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        q_pin = np.asarray(q26, dtype=np.float64)[self.model._pin_from_runtime]
        pin = self.model.pin
        pin.forwardKinematics(self.model.model, self.model.data, q_pin)
        pin.updateFramePlacements(self.model.model, self.model.data)
        names = ["right_palm_link"] + [
            f"right_finger{finger}_link{level}" for finger in _FINGERS for level in _LINK_LEVELS
        ]
        result = {}
        for name in names:
            frame_id = self.model.model.getFrameId(name)
            placement = self.model.data.oMf[frame_id]
            result[name] = (placement.translation.copy(), placement.rotation.copy())
        return result

    def _upper_support_points(self, q26: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
        placements = self._frame_placements(q26)
        rows = []
        for finger in _FINGERS:
            choices = []
            for level in _LINK_LEVELS:
                name = f"right_finger{finger}_link{level}"
                position, rotation = placements[name]
                vertices = self._mesh_vertices[name]
                world = position + vertices @ rotation.T
                index = int(np.argmax(world[:, 2]))
                choices.append((name, vertices[index].copy(), world[index].copy()))
            rows.append(max(choices, key=lambda row: float(row[2][2])))
        return rows


def stage4_goal_transform(
    fixed_position: Sequence[float],
    fixed_quat_wxyz: Sequence[float],
    pose_to_base: Sequence[Sequence[float]],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_matrix(fixed_quat_wxyz)
    transform[:3, 3] = np.asarray(fixed_position, dtype=np.float64)
    return transform @ np.asarray(pose_to_base, dtype=np.float64).reshape(4, 4)


def partial_hook_q20(
    preshape_q20: Sequence[float], close_q20: Sequence[float], fraction: float = 0.25
) -> np.ndarray:
    preshape = np.asarray(finite_vector(preshape_q20, 20, "preshape_q20"))
    close = np.asarray(finite_vector(close_q20, 20, "close_q20"))
    result = preshape.copy()
    for finger in _FINGERS:
        for joint in (2, 3):
            column = (finger - 1) + 5 * joint
            result[column] += float(fraction) * (close[column] - preshape[column])
    return result


def _load_link_vertices(urdf_path: Path) -> dict[str, np.ndarray]:
    import trimesh

    mesh_root = urdf_path.parent.parent / "meshes" / "right"
    rows = {}
    for finger in _FINGERS:
        for level in _LINK_LEVELS:
            name = f"right_finger{finger}_link{level}"
            mesh = trimesh.load_mesh(mesh_root / f"{name}.STL", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                raise ValueError(f"invalid collision mesh for {name}")
            rows[name] = vertices
    return rows


def _orthonormal_basis(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x = _unit(first)
    z = _unit(second - float(np.dot(second, x)) * x)
    y = _unit(np.cross(z, x))
    z = _unit(np.cross(x, y))
    return np.column_stack((x, y, z))


def _serial_xyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    pitch = float(np.arcsin(np.clip(value[0, 2], -1.0, 1.0)))
    cosine = float(np.cos(pitch))
    if abs(cosine) < 1.0e-8:
        roll = float(np.arctan2(value[2, 1], value[1, 1]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(-value[1, 2], value[2, 2]))
        yaw = float(np.arctan2(-value[0, 1], value[0, 0]))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _quat_wxyz_matrix(quat: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    values = np.asarray(finite_vector(quat, 4, "quat"), dtype=np.float64)
    values /= max(float(np.linalg.norm(values)), 1.0e-12)
    return Rotation.from_quat((values[1], values[2], values[3], values[0])).as_matrix()


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(np.asarray(first).T @ np.asarray(second)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _bounds_corners(bounds: np.ndarray) -> np.ndarray:
    lower, upper = np.asarray(bounds, dtype=np.float64)
    return np.asarray(
        [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
        dtype=np.float64,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("cannot normalize zero vector")
    return value / norm


def fork_pose_summary(pose: ForkSupportPose) -> Mapping[str, Any]:
    return pose.to_dict()
