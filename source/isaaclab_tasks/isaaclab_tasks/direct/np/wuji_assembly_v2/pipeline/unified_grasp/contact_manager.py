"""Contact signal collection for v80 unified grasping."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_csv, write_json, write_jsonl


FINGER_COUNT = 5


@dataclass
class ContactState:
    part_name: str
    object_id: int = 0
    active_finger_group: str = ""
    per_finger_contact_force_norm: list[float] = field(default_factory=list)
    per_finger_unfiltered_force_norm: list[float] = field(default_factory=list)
    per_finger_target_filtered_force_norm: list[float] = field(default_factory=list)
    per_finger_target_filtered_force_xyz: list[list[float]] = field(default_factory=list)
    active_target_filtered_force_count: int = 0
    target_object_contact_force_peak_n: float = 0.0
    target_contact_evidence_source: str = "target_filtered_contact_sensor_unavailable"
    target_filtered_force_available: bool = False
    target_filter_names: list[str] = field(default_factory=list)
    target_filter_index: int = -1
    distance_only_success_used: bool = False
    non_active_sensor_success_used: bool = False
    per_finger_contact_points: list[list[float]] = field(default_factory=list)
    per_finger_contact_normals: list[list[float]] = field(default_factory=list)
    per_finger_surface_distance_m: list[float] = field(default_factory=list)
    effective_contact_count_force: int = 0
    effective_contact_count_distance: int = 0
    support_contact_count: int = 0
    object_motion_before_contact_m: float = 0.0
    object_velocity_norm: float = 0.0
    object_angular_velocity_norm: float = 0.0
    table_collision: bool = False
    penetration_depth_m: float = 0.0
    contact_sensor_available: bool = False
    contact_sensor_api_available: bool = False
    contact_signal_nonzero: bool = False
    contact_force_peak_n: float = 0.0
    contact_evidence_source: str = "contact_sensor_unavailable"
    contact_sensor_status: str = "CONTACT_SENSOR_API_NOT_AVAILABLE"
    final_contact_success_allowed: bool = False
    per_finger_force_xyz: list[list[float]] = field(default_factory=list)
    per_finger_tip_positions: list[list[float]] = field(default_factory=list)
    per_finger_tip_linvel: list[list[float]] = field(default_factory=list)
    contact_point_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _slice_env_row(value: Any, env_index: int | None) -> Any:
    if value is None:
        return None
    if env_index is None:
        return value
    try:
        if hasattr(value, "detach"):
            if getattr(value, "ndim", 0) >= 2:
                return value[int(env_index)]
            return value
        if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes, dict)):
            if value and isinstance(value[0], (list, tuple)):
                return value[int(env_index)]
    except Exception:
        return value
    return value


def _as_list(value: Any, count: int = FINGER_COUNT, env_index: int | None = None) -> list[float]:
    value = _slice_env_row(value, env_index)
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "reshape") and not isinstance(value, (list, tuple)):
            value = value.reshape(-1).tolist()
        out = [float(item) for item in list(value)]
    except Exception:
        return []
    return [item if math.isfinite(item) else 0.0 for item in out[:count]]


def _as_vec_list(value: Any, count: int = FINGER_COUNT, env_index: int | None = None) -> list[list[float]]:
    value = _slice_env_row(value, env_index)
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            data = value.detach().cpu()
            if data.ndim == 3:
                data = data[0]
            rows = data.reshape(-1, 3).tolist()
        else:
            rows = value
        return [[float(x), float(y), float(z)] for x, y, z in list(rows)[:count]]
    except Exception:
        return []


def _as_matrix_list(
    value: Any,
    *,
    rows: int = FINGER_COUNT,
    cols: int = FINGER_COUNT,
    env_index: int | None = None,
) -> list[list[float]]:
    value = _slice_env_row(value, env_index)
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            data = value.detach().cpu()
            if data.ndim == 1:
                data = data.reshape(rows, -1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            matrix = data.tolist()
        else:
            matrix = value
        out: list[list[float]] = []
        for row in list(matrix)[:rows]:
            out.append([float(item) if math.isfinite(float(item)) else 0.0 for item in list(row)[:cols]])
        return out
    except Exception:
        return []


def _as_vec_matrix_for_filter(
    value: Any,
    filter_index: int,
    *,
    rows: int = FINGER_COUNT,
    env_index: int | None = None,
) -> list[list[float]]:
    value = _slice_env_row(value, env_index)
    if value is None or filter_index < 0:
        return []
    try:
        if hasattr(value, "detach"):
            data = value.detach().cpu()
            if data.ndim == 3:
                selected = data[:rows, int(filter_index), :].reshape(-1, 3).tolist()
            elif data.ndim == 2:
                selected = data.reshape(-1, 3).tolist()
            else:
                return []
        else:
            selected = [row[int(filter_index)] for row in list(value)[:rows]]
        return [[float(x), float(y), float(z)] for x, y, z in selected[:rows]]
    except Exception:
        return []


def _active_sensor_indices(group: str) -> list[int]:
    indices: list[int] = []
    for char in str(group or ""):
        if not char.isdigit():
            continue
        index = int(char) - 1
        if 0 <= index < FINGER_COUNT and index not in indices:
            indices.append(index)
    return indices


def _float_attr(env: Any, names: tuple[str, ...], default: float = 0.0, env_index: int | None = None) -> float:
    env = _base_env(env)
    for name in names:
        value = getattr(env, name, None)
        if value is None:
            continue
        try:
            value = _slice_env_row(value, env_index)
            if hasattr(value, "detach"):
                value = value.detach().cpu().reshape(-1)[0].item()
            return float(value)
        except Exception:
            continue
    return float(default)


def _bool_attr(env: Any, names: tuple[str, ...], default: bool = False, env_index: int | None = None) -> bool:
    env = _base_env(env)
    for name in names:
        value = getattr(env, name, None)
        if value is None:
            continue
        try:
            value = _slice_env_row(value, env_index)
            if hasattr(value, "detach"):
                value = bool(value.detach().cpu().reshape(-1)[0].item())
            return bool(value)
        except Exception:
            continue
    return bool(default)


class ContactManager:
    """Read real contact signals when the Isaac backend exposes them."""

    force_threshold_n: float = 0.05
    distance_threshold_m: float = 0.004

    def __init__(self) -> None:
        self.backend: Any | None = None

    def attach_backend(self, backend: Any) -> None:
        self.backend = backend

    def read_contact_state(
        self,
        env: Any | None = None,
        *,
        env_index: int | None = None,
        part_name: str,
        object_id: int = 0,
        active_finger_group: str = "",
    ) -> ContactState:
        base = _base_env(env) if env is not None else object()
        force_attr = getattr(base, "dex_fingertip_force_norm", None)
        flag_attr = getattr(base, "dex_fingertip_contact_flag", None)
        force_xyz_attr = getattr(base, "dex_fingertip_force_xyz", None)
        target_force_attr = getattr(base, "dex_fingertip_target_force_norm", None)
        target_force_xyz_attr = getattr(base, "dex_fingertip_target_force_xyz", None)
        target_filter_names = [str(item) for item in list(getattr(base, "dex_fingertip_target_filter_names", []) or [])]
        try:
            target_filter_index = target_filter_names.index(str(part_name))
        except ValueError:
            target_filter_index = -1
        force_norm = _as_list(force_attr, env_index=env_index)
        contact_flags = _as_list(flag_attr, env_index=env_index)
        force_xyz = _as_vec_list(force_xyz_attr, env_index=env_index)
        target_matrix = _as_matrix_list(target_force_attr, env_index=env_index)
        target_force_norm = [
            float(row[target_filter_index])
            for row in target_matrix[:FINGER_COUNT]
            if 0 <= target_filter_index < len(row)
        ]
        target_force_xyz = _as_vec_matrix_for_filter(
            target_force_xyz_attr,
            target_filter_index,
            env_index=env_index,
        )
        contact_points = _as_vec_list(getattr(base, "dex_fingertip_contact_points", None), env_index=env_index)
        if not contact_points:
            contact_points = _as_vec_list(getattr(base, "dex_fingertip_pos", None), env_index=env_index)
            contact_point_source = "fingertip_body_position"
        else:
            contact_point_source = "contact_sensor_point"
        normal_attr = getattr(base, "dex_fingertip_contact_normals", None)
        contact_normals = _as_vec_list(normal_attr, env_index=env_index)
        if not contact_normals and force_xyz:
            contact_normals = []
            for vec in force_xyz:
                norm = math.sqrt(sum(float(item) * float(item) for item in vec))
                contact_normals.append([float(item) / norm if norm > 1.0e-8 else 0.0 for item in vec])
        distance_attr = getattr(base, "dex_fingertip_surface_distance_m", None)
        if distance_attr is None:
            distance_attr = getattr(base, "dex_fingertip_surface_distances_m", None)
        distances = _as_list(distance_attr, env_index=env_index)
        fingertip_pos = _as_vec_list(getattr(base, "dex_fingertip_pos", None), env_index=env_index)
        fingertip_linvel = _as_vec_list(getattr(base, "dex_fingertip_linvel", None), env_index=env_index)

        force_available = force_attr is not None or flag_attr is not None or force_xyz_attr is not None
        target_available = bool(
            target_filter_index >= 0
            and len(target_force_norm) >= FINGER_COUNT
            and _float_attr(base, ("dex_target_force_valid",), env_index=env_index) > 0.5
        )
        if not force_norm and contact_flags:
            force_norm = [1.0 if flag > 0.5 else 0.0 for flag in contact_flags]
        peak_force = max([0.0, *[float(value) for value in force_norm]])
        target_peak = max([0.0, *[float(value) for value in target_force_norm]])
        active_indices = _active_sensor_indices(active_finger_group)
        if active_indices:
            active_target_count = sum(
                1
                for index in active_indices
                if 0 <= index < len(target_force_norm) and target_force_norm[index] >= self.force_threshold_n
            )
        else:
            active_target_count = sum(1 for value in target_force_norm if value >= self.force_threshold_n)
        signal_nonzero = bool(peak_force > 0.0 or target_peak > 0.0 or any(flag > 0.0 for flag in contact_flags))
        force_count = sum(1 for value in force_norm if value >= self.force_threshold_n)
        dist_count = sum(1 for value in distances if 0.0 <= value <= self.distance_threshold_m)
        evidence = "force_contact_sensor" if force_available else "distance_fallback_diagnostic"
        sensor_status = "force_contact_sensor_available" if force_available else "CONTACT_SENSOR_API_NOT_AVAILABLE"
        target_evidence = "ContactSensor.force_matrix_w" if target_available else "target_filtered_contact_sensor_unavailable"

        return ContactState(
            part_name=part_name,
            object_id=int(object_id),
            active_finger_group=active_finger_group,
            per_finger_contact_force_norm=force_norm,
            per_finger_unfiltered_force_norm=force_norm,
            per_finger_target_filtered_force_norm=target_force_norm,
            per_finger_target_filtered_force_xyz=target_force_xyz,
            active_target_filtered_force_count=int(active_target_count if target_available else 0),
            target_object_contact_force_peak_n=float(target_peak if target_available else 0.0),
            target_contact_evidence_source=target_evidence,
            target_filtered_force_available=target_available,
            target_filter_names=target_filter_names,
            target_filter_index=int(target_filter_index),
            distance_only_success_used=False,
            non_active_sensor_success_used=False,
            per_finger_contact_points=contact_points,
            per_finger_contact_normals=contact_normals,
            per_finger_surface_distance_m=distances,
            effective_contact_count_force=int(force_count),
            effective_contact_count_distance=int(dist_count),
            support_contact_count=int(force_count if force_available else 0),
            object_motion_before_contact_m=_float_attr(base, ("object_motion_before_contact_m", "part_motion_before_contact_m"), env_index=env_index),
            object_velocity_norm=_float_attr(base, ("object_velocity_norm", "part_velocity_norm"), env_index=env_index),
            object_angular_velocity_norm=_float_attr(base, ("object_angular_velocity_norm", "part_angular_velocity_norm"), env_index=env_index),
            table_collision=_bool_attr(base, ("table_collision", "hand_table_collision"), env_index=env_index),
            penetration_depth_m=_float_attr(base, ("penetration_depth_m", "max_penetration_depth_m"), env_index=env_index),
            contact_sensor_available=force_available,
            contact_sensor_api_available=force_available,
            contact_signal_nonzero=signal_nonzero,
            contact_force_peak_n=peak_force,
            contact_evidence_source=evidence,
            contact_sensor_status=sensor_status,
            final_contact_success_allowed=bool(force_available and signal_nonzero),
            per_finger_force_xyz=force_xyz,
            per_finger_tip_positions=fingertip_pos,
            per_finger_tip_linvel=fingertip_linvel,
            contact_point_source=contact_point_source,
        )


def probe_contact_signals(
    run_dir: str | Path,
    *,
    env: Any | None = None,
    parts: list[str] | None = None,
) -> dict[str, Any]:
    manager = ContactManager()
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for index, part in enumerate(parts or V80_PARTS):
        state = manager.read_contact_state(env or object(), part_name=part, object_id=index)
        row = state.to_dict()
        row.update(
            {
                "probe_attempted": True,
                "controlled_fingertip_probe_executed": env is not None,
                "fallback_contact_final_physical": False,
                "final_contact_success_allowed": bool(state.final_contact_success_allowed),
            }
        )
        rows.append(row)
        trace_rows.append({"probe_step": 0, **row})
    run_path = Path(run_dir)
    csv_path = write_csv(run_path / "contact_signal_audit.csv", rows)
    json_path = write_json(run_path / "contact_signal_audit.json", rows)
    trace_csv = write_csv(run_path / "contact_probe_trace.csv", trace_rows)
    trace_jsonl = write_jsonl(run_path / "contact_probe_trace.jsonl", trace_rows)
    return {
        "rows": rows,
        "contact_signal_audit_csv": str(csv_path),
        "contact_signal_audit_json": str(json_path),
        "contact_probe_trace_csv": str(trace_csv),
        "contact_probe_trace_jsonl": str(trace_jsonl),
    }
