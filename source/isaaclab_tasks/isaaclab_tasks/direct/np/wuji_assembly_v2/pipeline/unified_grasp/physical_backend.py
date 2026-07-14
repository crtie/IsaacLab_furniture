"""Physical IsaacLab backend adapter for v81 unified grasp RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contact_manager import ContactManager
from .unified_action_mapper import UnifiedActionMapper
from .v80_reports import V80_PARTS, write_csv, write_json, write_jsonl

try:  # pragma: no cover - runtime depends on Isaac environment
    import torch
except Exception:  # pragma: no cover
    torch = None


PART_TASK_MAP = {
    "Plug2": {"task": "Isaac-WujiFloating-Chair2-Direct-v0", "task_idx": 5, "object_attrs": ("_plug2", "_held_asset")},
    "Rod": {"task": "Isaac-WujiFloating-Chair2-Direct-v0", "task_idx": 3, "object_attrs": ("_rod_asset", "_held_asset")},
    "Backrest": {"task": "Isaac-Franka-Chair1-Direct-v0", "task_idx": 3, "object_attrs": ("_backrest_asset", "_held_asset")},
    "Frame": {"task": "Isaac-Franka-Chair4-Direct-v0", "task_idx": 1, "object_attrs": ("_plug1", "_held_asset")},
    "Screw1": {"task": "Isaac-Franka-Chair5-Direct-v0", "task_idx": 1, "object_attrs": ("_screw1", "_held_asset")},
}


@dataclass
class BackendEnvSlot:
    global_env_index: int
    part_name: str
    task_name: str
    local_env_index: int
    env: Any


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1)[0].item()
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _tensor_row(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return value[int(index)].detach().cpu()
        return value[int(index)]
    except Exception:
        return None


class IsaacUnifiedGraspPhysicalBackend:
    """Wrap real IsaacLab env instances behind the v80/v81 vector API."""

    required_methods = ("reset_envs", "step_envs", "step_env", "get_contact_state", "get_part_distribution", "close")

    def __init__(
        self,
        *,
        parts: list[str] | None = None,
        num_envs: int = 8,
        physics_profile: str = "canonical",
        device: str = "cuda:0",
        require_contact_sensor: bool = True,
        create_envs: bool = True,
    ) -> None:
        self.parts = [part for part in (parts or V80_PARTS) if part in V80_PARTS]
        self.num_envs = max(1, int(num_envs))
        self.physics_profile = str(physics_profile)
        self.device = device
        self.require_contact_sensor = bool(require_contact_sensor)
        self.create_envs_requested = bool(create_envs)
        self.contact_manager = ContactManager()
        self.contact_manager.attach_backend(self)
        self.action_mapper = UnifiedActionMapper()
        self.envs_by_part: dict[str, Any] = {}
        self.slots: list[BackendEnvSlot] = []
        self.creation_rows: list[dict[str, Any]] = []
        self.last_metrics: list[dict[str, Any]] = []
        self.action_mapping_rows: list[dict[str, Any]] = []
        self.object_identity_rows: list[dict[str, Any]] = []
        self.backend_ready = False
        self.contact_sensor_configured = False
        self.object_write_by_policy_detected = False
        self.object_write_reset_only = False
        self.sticky_action_available_to_policy = False
        self.proxy_action_available_to_policy = False
        self.route_selection_available_to_policy = False
        self.blocker = ""
        if create_envs:
            self._create_backend_pool()
            self._probe_backend_readiness()
        else:
            self.blocker = "physical_backend_creation_disabled"

    @property
    def physical_backend_ready(self) -> bool:
        return bool(self.backend_ready)

    def _create_backend_pool(self) -> None:
        try:
            import gymnasium as gym  # noqa: WPS433
            import isaaclab_tasks.direct.np  # noqa: F401,WPS433
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: WPS433
        except Exception as exc:
            self.blocker = f"PHYSICAL_BACKEND_IMPORT_FAILED:{type(exc).__name__}:{exc}"
            for part in self.parts:
                self.creation_rows.append(self._creation_row(part, False, self.blocker))
            return

        counts = self._balanced_counts()
        global_index = 0
        for part in self.parts:
            task_name = PART_TASK_MAP.get(part, {}).get("task", "")
            local_count = int(counts.get(part, 0))
            if local_count <= 0:
                continue
            if not task_name:
                reason = "NO_TASK_MAPPING_FOR_PART"
                self.creation_rows.append(self._creation_row(part, False, reason))
                continue
            try:
                env_cfg = parse_env_cfg(task_name, device=self.device, num_envs=local_count)
                env = gym.make(task_name, cfg=env_cfg)
                self.envs_by_part[part] = env
                self.creation_rows.append(self._creation_row(part, True, "created", task_name, local_count))
                for local_index in range(local_count):
                    self.slots.append(
                        BackendEnvSlot(
                            global_env_index=global_index,
                            part_name=part,
                            task_name=task_name,
                            local_env_index=local_index,
                            env=env,
                        )
                    )
                    global_index += 1
            except Exception as exc:
                reason = f"GYM_MAKE_FAILED:{type(exc).__name__}:{exc}"
                self.creation_rows.append(self._creation_row(part, False, reason, task_name, local_count))
        self.num_envs = len(self.slots) if self.slots else self.num_envs

    def _balanced_counts(self) -> dict[str, int]:
        counts = {part: 0 for part in self.parts}
        for index in range(self.num_envs):
            counts[self.parts[index % len(self.parts)]] += 1
        return counts

    def _creation_row(
        self,
        part: str,
        ok: bool,
        reason: str,
        task_name: str = "",
        local_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "part_name": part,
            "task_name": task_name or PART_TASK_MAP.get(part, {}).get("task", ""),
            "backend_env_created": bool(ok),
            "local_env_count": int(local_count),
            "reason": reason,
            "physics_profile": self.physics_profile,
        }

    def _probe_backend_readiness(self) -> None:
        if not self.slots:
            self.backend_ready = False
            self.contact_sensor_configured = False
            self.blocker = self.blocker or "NO_PHYSICAL_ENV_SLOTS_CREATED"
            return
        try:
            self.reset_envs()
            zero_actions = [[0.0] * 16 for _ in range(len(self.slots))]
            metrics = self.step_envs(zero_actions)
            self.last_metrics = metrics
            self.contact_sensor_configured = any(bool(row.get("contact_sensor_api_available")) for row in metrics)
            self.backend_ready = bool(metrics) and (self.contact_sensor_configured or not self.require_contact_sensor)
            if not self.backend_ready:
                self.blocker = "CONTACT_SENSOR_API_NOT_AVAILABLE"
        except Exception as exc:
            self.backend_ready = False
            self.contact_sensor_configured = False
            self.blocker = f"BACKEND_RESET_STEP_PROBE_FAILED:{type(exc).__name__}:{exc}"

    def reset_envs(self, env_ids: list[int] | None = None) -> list[dict[str, Any]]:
        selected = self._selected_slots(env_ids)
        rows = []
        for env in set(slot.env for slot in selected):
            try:
                env.reset()
                self.object_write_reset_only = True
            except Exception as exc:
                raise RuntimeError(f"backend_env_reset_failed:{type(exc).__name__}:{exc}") from exc
        for slot in selected:
            rows.append(self._metrics_for_slot(slot, phase="after_reset"))
        self.last_metrics = rows
        return rows

    def step_envs(self, actions: list[list[float]] | Any) -> list[dict[str, Any]]:
        action_rows = self._normalize_actions(actions)
        for env, slots in self._slots_by_env().items():
            if torch is None:
                raise RuntimeError("torch_unavailable_for_physical_backend_step")
            base = getattr(env, "unwrapped", env)
            device = getattr(base, "device", self.device)
            group_actions = [action_rows[slot.global_env_index] for slot in slots]
            action_tensor, audit_rows = self.action_mapper.map_batch(
                env,
                group_actions,
                device=device,
                env_indices=[slot.global_env_index for slot in slots],
            )
            slot_by_index = {slot.global_env_index: slot for slot in slots}
            for audit in audit_rows:
                slot = slot_by_index.get(int(audit.get("env_index", -1)))
                if slot is not None:
                    audit.update({"part_name": slot.part_name, "task_name": slot.task_name})
                self.action_mapping_rows.append(audit)
            try:
                env.step(action_tensor)
            except Exception as exc:
                raise RuntimeError(f"backend_env_step_failed:{type(exc).__name__}:{exc}") from exc
        rows = [self._metrics_for_slot(slot, phase="after_step") for slot in self.slots]
        self.last_metrics = rows
        return rows

    def step_env(self, env_index: int, action: list[float]) -> dict[str, Any]:
        full_actions = [[0.0] * 16 for _ in range(len(self.slots))]
        if 0 <= int(env_index) < len(full_actions):
            full_actions[int(env_index)] = list(action)
        rows = self.step_envs(full_actions)
        return rows[int(env_index)] if 0 <= int(env_index) < len(rows) else {}

    def _selected_slots(self, env_ids: list[int] | None) -> list[BackendEnvSlot]:
        if env_ids is None:
            return list(self.slots)
        wanted = {int(item) for item in env_ids}
        return [slot for slot in self.slots if slot.global_env_index in wanted]

    def _slots_by_env(self) -> dict[Any, list[BackendEnvSlot]]:
        grouped: dict[Any, list[BackendEnvSlot]] = {}
        for slot in self.slots:
            grouped.setdefault(slot.env, []).append(slot)
        return grouped

    def _normalize_actions(self, actions: list[list[float]] | Any) -> list[list[float]]:
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().tolist()
        if not actions:
            actions = [[0.0] * 16]
        if isinstance(actions[0], (int, float)):
            actions = [actions]
        rows = []
        for index in range(len(self.slots)):
            raw = list(actions[index % len(actions)])
            raw = [max(-1.0, min(1.0, float(value))) for value in raw[:16]]
            raw.extend([0.0] * (16 - len(raw)))
            rows.append(raw)
        return rows

    def _env_action_width(self, env: Any) -> int:
        action_space = getattr(env, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if shape:
            return int(shape[-1])
        base = getattr(env, "unwrapped", env)
        return int(getattr(base, "num_actions", 26) or 26)

    def get_contact_state(self, env_index: int, part_name: str | None = None, active_finger_group: str = "") -> Any:
        if not self.slots or not (0 <= int(env_index) < len(self.slots)):
            return self.contact_manager.read_contact_state(None, part_name=part_name or "", object_id=int(env_index))
        slot = self.slots[int(env_index)]
        return self.contact_manager.read_contact_state(
            slot.env,
            env_index=slot.local_env_index,
            part_name=part_name or slot.part_name,
            object_id=slot.global_env_index,
            active_finger_group=active_finger_group,
        )

    def get_metrics(self) -> list[dict[str, Any]]:
        return list(self.last_metrics)

    def get_part_distribution(self) -> dict[str, int]:
        return {part: sum(1 for slot in self.slots if slot.part_name == part) for part in V80_PARTS}

    def part_backend_ready(self, part_name: str) -> bool:
        return bool(self.backend_ready and any(slot.part_name == part_name for slot in self.slots))

    def part_blocker(self, part_name: str) -> str:
        if self.part_backend_ready(part_name):
            return ""
        row = next((item for item in self.creation_rows if item.get("part_name") == part_name), {})
        return str(row.get("reason") or self.blocker or "NO_PHYSICAL_ENV_SLOTS_CREATED_FOR_PART")

    def _metrics_for_slot(self, slot: BackendEnvSlot, *, phase: str) -> dict[str, Any]:
        state = self.get_contact_state(slot.global_env_index, slot.part_name).to_dict()
        force_peak = max([0.0, *[float(v) for v in state.get("per_finger_contact_force_norm", [])]])
        force_count = int(state.get("effective_contact_count_force") or 0)
        distance_count = int(state.get("effective_contact_count_distance") or 0)
        active = self._active_distance(slot.env, slot.local_env_index)
        identity = self._object_identity_for_slot(slot)
        self._record_object_identity(identity)
        controlled_gate = bool(
            identity.get("object_identity_verified")
            and force_peak > float(self.contact_manager.force_threshold_n)
            and force_count >= 1
            and not self.object_write_by_policy_detected
            and not self.sticky_action_available_to_policy
        )
        return {
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "part_name": slot.part_name,
            "task_name": slot.task_name,
            "phase": phase,
            "active_distance_m": active,
            "force_contact_count": force_count,
            "distance_contact_count": distance_count,
            "support_gate_ok": bool(force_count >= (2 if slot.part_name in {"Plug2", "Screw1"} else 1)),
            "object_motion_before_contact_m": float(state.get("object_motion_before_contact_m") or 0.0),
            "penetration_depth_m": float(state.get("penetration_depth_m") or 0.0),
            "table_collision": bool(state.get("table_collision")),
            "hold_success": False,
            "lift_success": False,
            "contact_sensor_available": bool(state.get("contact_sensor_available")),
            "contact_sensor_api_available": bool(state.get("contact_sensor_api_available")),
            "contact_signal_nonzero": bool(state.get("contact_signal_nonzero")),
            "nonzero_force_contact_observed": bool(state.get("contact_signal_nonzero")),
            "contact_evidence_source": state.get("contact_evidence_source", ""),
            "force_contact_probe_peak_n": force_peak,
            "controlled_contact_gate_pass": controlled_gate,
            "object_identity_verified": bool(identity.get("object_identity_verified")),
            "active_held_asset_name": identity.get("active_held_asset_name", ""),
            "active_held_asset_usd": identity.get("active_held_asset_usd", ""),
            "expected_part_name": identity.get("expected_part_name", ""),
            "object_identity_blocker": identity.get("object_identity_blocker", ""),
            "object_write_by_policy_detected": bool(self.object_write_by_policy_detected),
            "object_write_reset_only": bool(self.object_write_reset_only),
            "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
            "proxy_action_available_to_policy": bool(self.proxy_action_available_to_policy),
            "route_selection_available_to_policy": bool(self.route_selection_available_to_policy),
        }

    def _object_identity_for_slot(self, slot: BackendEnvSlot) -> dict[str, Any]:
        base = getattr(slot.env, "unwrapped", slot.env)
        part_map = PART_TASK_MAP.get(slot.part_name, {})
        expected_attrs = tuple(part_map.get("object_attrs", ()))
        expected_task_idx = part_map.get("task_idx", "")
        held = getattr(base, "_held_asset", None)
        active_attr = ""
        for attr in expected_attrs:
            if attr == "_held_asset":
                continue
            if getattr(base, attr, None) is held:
                active_attr = attr
                break
        if not active_attr and held is not None and "_held_asset" in expected_attrs:
            active_attr = "_held_asset"
        active_name = active_attr or type(held).__name__ if held is not None else ""
        usd_path = self._asset_usd_path(base, held, active_attr)
        prim_path = self._asset_prim_path(held)
        task_idx = getattr(getattr(base, "cfg_task", None), "task_idx", "")
        task_idx_matches = bool(expected_task_idx == "" or str(task_idx) == str(expected_task_idx))
        verified = bool(
            held is not None
            and active_attr
            and active_attr in expected_attrs
            and active_attr != "_held_asset"
            and task_idx_matches
        )
        blockers = []
        if held is None:
            blockers.append("NO_ACTIVE_HELD_ASSET")
        elif not active_attr:
            blockers.append(f"HELD_ASSET_NOT_IN_EXPECTED_ATTRS:{expected_attrs}")
        elif active_attr == "_held_asset":
            blockers.append("ONLY_GENERIC_HELD_ASSET_ATTR_MATCHED")
        elif not task_idx_matches:
            blockers.append(f"TASK_IDX_MISMATCH:expected={expected_task_idx}:actual={task_idx}")
        if held is not None and not task_idx_matches and not any(item.startswith("TASK_IDX_MISMATCH") for item in blockers):
            blockers.append(f"TASK_IDX_MISMATCH:expected={expected_task_idx}:actual={task_idx}")
        return {
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "part_name": slot.part_name,
            "task_name": slot.task_name,
            "expected_part_name": slot.part_name,
            "task_idx": task_idx,
            "expected_task_idx": expected_task_idx,
            "task_idx_matches_expected_part": task_idx_matches,
            "part_task_map_object_attrs": ",".join(expected_attrs),
            "object_attr_used": active_attr,
            "object_identity_verified": verified,
            "active_held_asset_name": active_name,
            "active_held_asset_usd": usd_path,
            "active_held_asset_prim_path": prim_path,
            "object_identity_blocker": ";".join(blockers),
        }

    def _asset_usd_path(self, base: Any, held: Any, active_attr: str) -> str:
        for candidate in (
            getattr(getattr(getattr(held, "cfg", None), "spawn", None), "usd_path", ""),
            getattr(getattr(getattr(getattr(base, "cfg_task", None), active_attr.lstrip("_"), None), "spawn", None), "usd_path", ""),
        ):
            if candidate:
                return str(candidate)
        return ""

    def _asset_prim_path(self, held: Any) -> str:
        return str(getattr(getattr(held, "cfg", None), "prim_path", "") or "")

    def _record_object_identity(self, row: dict[str, Any]) -> None:
        key = (row.get("env_index"), row.get("part_name"))
        self.object_identity_rows = [
            existing
            for existing in self.object_identity_rows
            if (existing.get("env_index"), existing.get("part_name")) != key
        ]
        self.object_identity_rows.append(row)

    def _active_distance(self, env: Any, local_index: int) -> float:
        base = getattr(env, "unwrapped", env)
        for name in ("active_max_dist_m", "grasp_tip_surface_min", "prev_grasp_tip_surface_min"):
            value = _tensor_row(getattr(base, name, None), local_index)
            if value is not None:
                return _to_float(value, 0.05)
        return 0.05

    def run_controlled_contact_probe(self, *, steps: int = 12) -> list[dict[str, Any]]:
        return self.run_contact_actuation_probe(close_steps=4, approach_steps=max(1, int(steps)))

    def run_contact_actuation_probe(self, *, close_steps: int = 4, approach_steps: int = 12) -> list[dict[str, Any]]:
        if not self.slots:
            return [
                {
                    "part_name": part,
                    "controlled_fingertip_probe_executed": False,
                    "contact_sensor_available": False,
                    "contact_sensor_api_available": False,
                    "nonzero_force_contact_observed": False,
                    "controlled_contact_gate_pass": False,
                    "contact_evidence_source": "CONTACT_SENSOR_API_NOT_AVAILABLE",
                    "blocker": self.blocker or "NO_PHYSICAL_ENV_SLOTS_CREATED",
                }
                for part in self.parts
            ]
        rows: list[dict[str, Any]] = []
        self.action_mapping_rows = []
        self.reset_envs()
        total_steps = max(1, int(close_steps)) + max(1, int(approach_steps))
        for step_index in range(total_steps):
            action = [[0.0] * 16 for _ in self.slots]
            phase = "close" if step_index < max(1, int(close_steps)) else "approach"
            for row in action:
                if phase == "approach":
                    row[2] = -0.35
                for col in range(6, 16):
                    row[col] = 0.75
            metrics = self.step_envs(action)
            for item in metrics:
                rows.append(
                    {
                        "probe_step": step_index,
                        "probe_phase": phase,
                        "controlled_fingertip_probe_executed": True,
                        **item,
                    }
                )
        return rows

    def close(self) -> None:
        for env in self.envs_by_part.values():
            try:
                env.close()
            except Exception:
                pass


def write_backend_creation_report(run_dir: str | Path, backend: IsaacUnifiedGraspPhysicalBackend) -> dict[str, str]:
    run_path = Path(run_dir)
    csv_path = write_csv(run_path / "physical_backend_creation_report.csv", backend.creation_rows)
    json_path = write_json(run_path / "physical_backend_creation_report.json", backend.creation_rows)
    return {"physical_backend_creation_report_csv": str(csv_path), "physical_backend_creation_report_json": str(json_path)}


def write_controlled_contact_probe(run_dir: str | Path, backend: IsaacUnifiedGraspPhysicalBackend) -> dict[str, Any]:
    rows = backend.run_controlled_contact_probe()
    run_path = Path(run_dir)
    trace_csv = write_csv(run_path / "contact_probe_trace.csv", rows)
    trace_jsonl = write_jsonl(run_path / "contact_probe_trace.jsonl", rows)
    summary: list[dict[str, Any]] = []
    for part in backend.parts:
        part_rows = [row for row in rows if row.get("part_name") == part]
        part_ready = backend.part_backend_ready(part)
        peak = max([0.0, *[float(row.get("force_contact_probe_peak_n") or 0.0) for row in part_rows]])
        contact_available = any(bool(row.get("contact_sensor_api_available") or row.get("contact_sensor_available")) for row in part_rows)
        signal_nonzero = any(bool(row.get("nonzero_force_contact_observed") or row.get("contact_signal_nonzero")) for row in part_rows)
        gate_pass = any(bool(row.get("controlled_contact_gate_pass")) for row in part_rows)
        executed = any(bool(row.get("controlled_fingertip_probe_executed")) for row in part_rows)
        summary.append(
            {
                "part_name": part,
                "physical_backend_ready": part_ready,
                "controlled_fingertip_probe_executed": executed,
                "contact_sensor_available": contact_available,
                "contact_sensor_api_available": contact_available,
                "contact_signal_nonzero": signal_nonzero,
                "nonzero_force_contact_observed": signal_nonzero,
                "controlled_contact_gate_pass": gate_pass,
                "contact_evidence_source": "force_contact_sensor" if contact_available else "CONTACT_SENSOR_API_NOT_AVAILABLE",
                "force_contact_probe_peak_n": peak,
                "object_motion_during_probe_m": max([0.0, *[float(row.get("object_motion_before_contact_m") or 0.0) for row in part_rows]]),
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "object_write_reset_only": bool(backend.object_write_reset_only),
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "blocker": "" if contact_available else backend.part_blocker(part) or "CONTACT_SENSOR_API_NOT_AVAILABLE",
            }
        )
    audit_csv = write_csv(run_path / "contact_signal_audit.csv", summary)
    audit_json = write_json(run_path / "contact_signal_audit.json", summary)
    return {
        "rows": summary,
        "trace_rows": rows,
        "contact_probe_trace_csv": str(trace_csv),
        "contact_probe_trace_jsonl": str(trace_jsonl),
        "contact_signal_audit_csv": str(audit_csv),
        "contact_signal_audit_json": str(audit_json),
    }


def write_v82_contact_actuation_probe(run_dir: str | Path, backend: IsaacUnifiedGraspPhysicalBackend) -> dict[str, Any]:
    rows = backend.run_contact_actuation_probe(close_steps=4, approach_steps=12)
    run_path = Path(run_dir)
    trace_csv = write_csv(run_path / "controlled_contact_probe_trace.csv", rows)
    trace_jsonl = write_jsonl(run_path / "controlled_contact_probe_trace.jsonl", rows)
    action_csv = write_csv(run_path / "action_mapping_audit.csv", backend.action_mapping_rows)
    action_json = write_json(run_path / "action_mapping_audit.json", backend.action_mapping_rows)
    identity_csv = write_csv(run_path / "object_identity_audit.csv", backend.object_identity_rows)
    identity_json = write_json(run_path / "object_identity_audit.json", backend.object_identity_rows)
    summary: list[dict[str, Any]] = []
    for part in backend.parts:
        part_rows = [row for row in rows if row.get("part_name") == part]
        identity = next((row for row in backend.object_identity_rows if row.get("part_name") == part), {})
        action_rows = [row for row in backend.action_mapping_rows if row.get("part_name") == part]
        peak = max([0.0, *[float(row.get("force_contact_probe_peak_n") or 0.0) for row in part_rows]])
        force_count_max = max([0, *[int(row.get("force_contact_count") or 0) for row in part_rows]])
        mapped_cols = ",".join(sorted({str(row.get("mapped_close_cols", "")) for row in action_rows if row.get("mapped_close_cols")}))
        isaac_action_width = max([0, *[int(row.get("isaac_action_dim") or 0) for row in action_rows]])
        width_ok = any(int(row.get("isaac_action_dim") or 0) == 26 for row in action_rows)
        close_cols_ok = any(str(row.get("mapped_close_cols") or "") == "16,17,18,19,20,21,22,23,24,25" for row in action_rows)
        close_commanded = any(bool(row.get("metric_close_dof_commanded")) for row in action_rows)
        object_ok = bool(identity.get("object_identity_verified"))
        gate_pass = bool(
            object_ok
            and width_ok
            and close_cols_ok
            and close_commanded
            and peak > float(backend.contact_manager.force_threshold_n)
            and force_count_max >= 1
            and not backend.object_write_by_policy_detected
            and not backend.sticky_action_available_to_policy
        )
        blocker = ""
        if not backend.part_backend_ready(part):
            blocker = backend.part_blocker(part)
        elif not object_ok:
            blocker = str(identity.get("object_identity_blocker") or "object_identity_not_verified")
        elif not width_ok or not close_cols_ok or not close_commanded:
            blocker = "policy_to_isaac_action_mapping_failed"
        elif peak <= float(backend.contact_manager.force_threshold_n) or force_count_max < 1:
            blocker = "controlled_force_contact_not_observed"
        summary.append(
            {
                "part_name": part,
                "physical_backend_ready": bool(backend.part_backend_ready(part)),
                "object_identity_verified": object_ok,
                "active_held_asset_name": identity.get("active_held_asset_name", ""),
                "active_held_asset_usd": identity.get("active_held_asset_usd", ""),
                "expected_part_name": identity.get("expected_part_name", part),
                "object_identity_blocker": identity.get("object_identity_blocker", ""),
                "policy_action_dim": 16,
                "isaac_action_width": isaac_action_width,
                "mapped_close_cols": mapped_cols,
                "metric_close_dof_commanded": close_commanded,
                "contact_sensor_api_available": any(bool(row.get("contact_sensor_api_available")) for row in part_rows),
                "nonzero_force_contact_observed": any(bool(row.get("nonzero_force_contact_observed")) for row in part_rows),
                "force_contact_probe_peak_n": peak,
                "effective_contact_count_force_max": force_count_max,
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "controlled_contact_gate_pass": gate_pass,
                "status": "PASS" if gate_pass else "CONTACT_ACTUATION_NO_FORCE_CONTACT",
                "blocker": blocker,
            }
        )
    summary_csv = write_csv(run_path / "contact_gate_summary.csv", summary)
    summary_json = write_json(run_path / "contact_gate_summary.json", summary)
    design = {
        "design_name": "v82_single_simulation_context_unified_backend",
        "status": "prepared_not_implemented",
        "multi_gym_make_success_path_allowed": False,
        "reason": "Isaac SimulationContext is singleton; v81 multi-gym backend can only be diagnostic.",
        "next_action": "build one DirectRLEnv scene that spawns all requested objects and switches active held-object state per env.",
    }
    design_path = write_json(run_path / "single_simulation_context_backend_design.json", design)
    return {
        "rows": summary,
        "trace_rows": rows,
        "action_mapping_rows": backend.action_mapping_rows,
        "object_identity_rows": backend.object_identity_rows,
        "controlled_contact_probe_trace_csv": str(trace_csv),
        "controlled_contact_probe_trace_jsonl": str(trace_jsonl),
        "action_mapping_audit_csv": str(action_csv),
        "action_mapping_audit_json": str(action_json),
        "object_identity_audit_csv": str(identity_csv),
        "object_identity_audit_json": str(identity_json),
        "contact_gate_summary_csv": str(summary_csv),
        "contact_gate_summary_json": str(summary_json),
        "single_simulation_context_backend_design_json": str(design_path),
    }
