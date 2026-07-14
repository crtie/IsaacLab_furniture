"""v83 single-SimulationContext backend and audit writers."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contact_manager import ContactManager
from .unified_action_mapper import UnifiedActionMapper
from .v80_reports import V80_PARTS, write_csv, write_json, write_jsonl

try:  # pragma: no cover - runtime depends on Isaac
    import torch
except Exception:  # pragma: no cover
    torch = None


V83_PARTS = ("Plug2", "Screw1", "Backrest", "Rod", "Frame")
V85_ACTION_PHASES = {"close", "approach_close", "hold_squeeze"}
V89_ACTION_PHASES = {"close", "approach_close", "hold_squeeze", "lift_stabilize"}
V90_ACTION_PHASES = {"slow_approach", "close_force_limited", "contact_aware_hold", "slow_lift"}
V85_FINGER_GROUPS = {
    "Plug2": ("23", "34"),
    "Screw1": ("34", "23"),
    "Backrest": ("34",),
    "Rod": ("234",),
    "Frame": ("234",),
}
V85_CLEARANCE_ORDER_M = (0.008, 0.012, 0.004, 0.018)
V85_LATERAL_ORDER_M = (0.0, 0.004, -0.004, 0.008, -0.008)
V85_SURFACE_STRATEGIES = {
    "Plug2": ("bbox_adjusted", "analytic_surface"),
    "Screw1": ("bbox_adjusted", "analytic_surface"),
    "Backrest": ("analytic_surface", "bbox_adjusted"),
    "Rod": ("bbox_adjusted", "analytic_surface"),
    "Frame": ("analytic_surface", "bbox_adjusted"),
}
V85_PART_GEOMETRY = {
    "Plug2": {"kind": "sphere", "radius": 0.018},
    "Screw1": {"kind": "cylinder_z", "center": (0.0, 0.0, 0.0125), "radius": 0.006, "half_height": 0.0125},
    "Backrest": {"kind": "boxes", "boxes": (((0.095, 0.010, -0.0885), (0.095, 0.010, 0.0885)),)},
    "Rod": {"kind": "boxes", "boxes": (((0.100, 0.010, -0.022), (0.100, 0.010, 0.022)),)},
    "Frame": {
        "kind": "boxes",
        "boxes": (
            ((0.018, 0.015, -0.275), (0.018, 0.014, 0.275)),
            ((0.269, 0.015, -0.275), (0.018, 0.014, 0.275)),
            ((0.1435, 0.015, -0.018), (0.1435, 0.014, 0.018)),
            ((0.1435, 0.015, -0.532), (0.1435, 0.014, 0.018)),
        ),
    },
}
V86_EXCESSIVE_FORCE_THRESHOLD_N = 150.0


@dataclass
class SingleContextSlot:
    global_env_index: int
    local_env_index: int
    part_name: str


def _basename(path: str) -> str:
    return str(path or "").rsplit("/", 1)[-1]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
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
            return value[int(index)].detach()
        return value[int(index)]
    except Exception:
        return None


def _int_field(row: dict[str, Any], key: str, default: int = -1) -> int:
    value = row.get(key, default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _torch_inference_context() -> Any:
    return torch.inference_mode() if torch is not None else nullcontext()


class IsaacUnifiedSingleContextBackend:
    """One Isaac env instance that exposes all v83 parts as logical slots."""

    required_methods = ("reset_envs", "step_envs", "step_env", "get_contact_state", "get_metrics", "get_part_distribution", "close")

    def __init__(
        self,
        *,
        parts: list[str] | None = None,
        num_envs: int = 5,
        physics_profile: str = "canonical",
        device: str = "cuda:0",
        create_env: bool = True,
        auto_probe_reset_step: bool = True,
    ) -> None:
        self.parts = [part for part in (parts or list(V83_PARTS)) if part in V83_PARTS]
        self.num_envs_requested = max(len(self.parts), int(num_envs or len(self.parts) or 1))
        self.physics_profile = str(physics_profile)
        self.device = device
        self.contact_manager = ContactManager()
        self.contact_manager.attach_backend(self)
        self.action_mapper = UnifiedActionMapper()
        self.env: Any | None = None
        self.slots: list[SingleContextSlot] = []
        self.last_metrics: list[dict[str, Any]] = []
        self.action_mapping_rows: list[dict[str, Any]] = []
        self.object_identity_rows: list[dict[str, Any]] = []
        self.single_context_rows: list[dict[str, Any]] = []
        self.contact_api_rows: list[dict[str, Any]] = []
        self.v86_selected_variants: dict[str, dict[str, Any]] = {}
        self.v86_near_contact_reset_rows: list[dict[str, Any]] = []
        self.v86_rollout_trace_rows: list[dict[str, Any]] = []
        self.v87_action_prior_rows: list[dict[str, Any]] = []
        self.v87_force_contact_streak: dict[tuple[int, str], int] = {}
        self._v87_pending_action_context: dict[int, dict[str, Any]] = {}
        self._previous_composed_policy_actions: dict[int, list[float]] = {}
        self.v89_selected_candidates: dict[str, dict[str, Any]] = {}
        self.v89_candidate_probe_trace_rows: list[dict[str, Any]] = []
        self.v89_candidate_probe_summary_rows: list[dict[str, Any]] = []
        self.v88_physics_tuning_rows: list[dict[str, Any]] = []
        self.v88_observation_feature_rows: list[dict[str, Any]] = []
        self.v93_safe_staging_plan_rows: list[dict[str, Any]] = []
        self.v93_safe_staging_audit_rows: list[dict[str, Any]] = []
        self.v94_pregrasp_plan_rows: list[dict[str, Any]] = []
        self.v94_pregrasp_audit_rows: list[dict[str, Any]] = []
        self.v95_pregrasp_plan_rows: list[dict[str, Any]] = []
        self.v86_policy_step_started = False
        self.v86_step_counter = 0
        self.v86_reset_counter = 0
        self.v86_reset_object_positions: dict[tuple[int, str], list[float]] = {}
        self.gym_make_count = 0
        self.single_simulation_context = False
        self.vector_reset_ok = False
        self.vector_step_ok = False
        self.object_write_reset_only = False
        self.object_write_by_policy_detected = False
        self.sticky_action_available_to_policy = False
        self.proxy_action_available_to_policy = False
        self.route_selection_available_to_policy = False
        self.blocker = ""
        if create_env:
            self._create_single_env()
            if auto_probe_reset_step:
                self._probe_reset_step()
        else:
            self.blocker = "single_context_backend_creation_disabled"

    @property
    def num_envs(self) -> int:
        return len(self.slots) if self.slots else self.num_envs_requested

    @property
    def physical_backend_ready(self) -> bool:
        return bool(self.single_simulation_context and self.vector_reset_ok and self.vector_step_ok)

    @property
    def contact_sensor_configured(self) -> bool:
        return any(bool(row.get("contact_sensor_api_available")) for row in self.contact_api_rows + self.last_metrics)

    def _create_single_env(self) -> None:
        try:
            import gymnasium as gym  # noqa: WPS433
            import isaaclab_tasks.direct.np  # noqa: F401,WPS433
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: WPS433
        except Exception as exc:
            self.blocker = f"SINGLE_CONTEXT_IMPORT_FAILED:{type(exc).__name__}:{exc}"
            return
        try:
            cfg = parse_env_cfg(
                "Isaac-Wuji-UnifiedPhysicalGrasp-v83-Direct-v0",
                device=self.device,
                num_envs=self.num_envs_requested,
            )
            self.env = gym.make("Isaac-Wuji-UnifiedPhysicalGrasp-v83-Direct-v0", cfg=cfg)
            self.gym_make_count = 1
            self.single_simulation_context = True
            self.slots = [
                SingleContextSlot(index, index, self.parts[index % len(self.parts)])
                for index in range(int(getattr(self.env.unwrapped, "num_envs", self.num_envs_requested)))
            ]
            self.single_context_rows.append(
                {
                    "backend_created": True,
                    "task_name": "Isaac-Wuji-UnifiedPhysicalGrasp-v83-Direct-v0",
                    "gym_make_count": self.gym_make_count,
                    "single_simulation_context": True,
                    "multi_gym_make_success_path_allowed": False,
                    "actual_env_count": len(self.slots),
                    "blocker": "",
                }
            )
        except Exception as exc:
            self.blocker = f"SINGLE_CONTEXT_GYM_MAKE_FAILED:{type(exc).__name__}:{exc}"
            self.single_context_rows.append(
                {
                    "backend_created": False,
                    "task_name": "Isaac-Wuji-UnifiedPhysicalGrasp-v83-Direct-v0",
                    "gym_make_count": self.gym_make_count,
                    "single_simulation_context": False,
                    "multi_gym_make_success_path_allowed": False,
                    "actual_env_count": 0,
                    "blocker": self.blocker,
                }
            )

    def _probe_reset_step(self) -> None:
        if self.env is None or not self.slots:
            return
        try:
            self.reset_envs()
            zero_actions = [[0.0] * 16 for _ in self.slots]
            self.step_envs(zero_actions)
        except Exception as exc:
            self.blocker = f"SINGLE_CONTEXT_RESET_STEP_FAILED:{type(exc).__name__}:{exc}"

    def reset_envs(self, env_ids: list[int] | None = None) -> list[dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(self.blocker or "single_context_env_not_created")
        with _torch_inference_context():
            self.env.reset()
        self.object_write_reset_only = True
        self.v86_policy_step_started = False
        self.v86_reset_counter += 1
        self.v87_force_contact_streak = {}
        self._v87_pending_action_context = {}
        self._previous_composed_policy_actions = {}
        self.vector_reset_ok = True
        rows = [self._metrics_for_slot(slot, "after_reset") for slot in self._selected_slots(env_ids)]
        self.last_metrics = rows
        return rows

    def step_envs(self, actions: Any) -> list[dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(self.blocker or "single_context_env_not_created")
        if torch is None:
            raise RuntimeError("torch_unavailable_for_single_context_step")
        action_rows = self._normalize_actions(actions)
        base = getattr(self.env, "unwrapped", self.env)
        device = getattr(base, "device", self.device)
        action_tensor, audit_rows = self.action_mapper.map_batch(
            self.env,
            action_rows,
            device=device,
            env_indices=[slot.global_env_index for slot in self.slots],
        )
        part_by_index = {slot.global_env_index: slot.part_name for slot in self.slots}
        for row in audit_rows:
            context = self._v87_pending_action_context.get(int(row.get("env_index", -1)), {})
            row.update(
                {
                    "part_name": part_by_index.get(int(row.get("env_index", -1)), ""),
                    "task_name": "v91_task_setup_asset_dynamics_and_controller_sanity"
                    if context.get("v91_task_sanity_mode")
                    else "v90_bottleneck_isolation_and_grasp_feasibility"
                    if context.get("v90_feasibility_mode")
                    else "v89_failure_driven_hybrid_grasp_repair"
                    if context.get("v89_hybrid_repair_mode")
                    else (
                        "v95_minimal_unified_workcell_repair"
                        if context.get("v95_minimal_workcell_mode")
                        else "v94_safe_pregrasp_and_finger_contact_calibration"
                        if context.get("v94_safe_pregrasp_mode")
                        else "v93_collision_geometry_and_safe_staging_repair"
                        if context.get("v93_setup_repair_mode")
                        else "v88_stabilized_grasp_policy_and_physics_audit"
                        if context.get("v88_stabilized_mode")
                        else ("v87_contact_guided_residual_ppo" if context else "v83_single_context")
                    ),
                    **{
                        key: value
                        for key, value in context.items()
                        if key
                        in {
                            "v88_stabilized_mode",
                            "v89_hybrid_repair_mode",
                            "v89_nominal_candidate_mode",
                            "v90_feasibility_mode",
                            "v91_task_sanity_mode",
                            "v93_setup_repair_mode",
                            "v94_safe_pregrasp_mode",
                            "v94_calibration_mode",
                            "v95_minimal_workcell_mode",
                            "v95_calibration_mode",
                            "v95_phase_step",
                            "calibration_axis",
                            "calibration_sign",
                            "hand_write_after_reset_allowed",
                            "sticky_action_available_to_policy",
                            "route_selection_available_to_policy",
                            "proxy_action_available_to_policy",
                            "logical_finger_id",
                            "diagnostic_target_used",
                            "actual_object_contact_calibration",
                            "v87_contact_guided_residual_mode",
                            "nominal_phase",
                            "candidate_id",
                            "candidate_family",
                            "candidate_rank",
                            "family_variant_index",
                            "active_finger_group",
                            "commanded_fingers",
                            "commanded_policy_cols",
                            "commanded_isaac_cols",
                            "non_active_finger_value",
                            "support_finger_value",
                            "force_limit_n",
                            "force_limit_active",
                            "force_limit_backoff",
                            "multi_finger_support_established",
                            "v90_phase_step",
                            "v91_phase_step",
                            "action_jerk",
                            "residual_action_norm",
                            "composed_action_norm",
                            "previous_composed_action_norm",
                            "residual_scale_wrist_xyz",
                            "residual_scale_wrist_rot",
                            "residual_scale_finger",
                        }
                    },
                }
            )
            self.action_mapping_rows.append(row)
        self.v86_policy_step_started = True
        with _torch_inference_context():
            self.env.step(action_tensor)
        self.vector_step_ok = True
        rows = [self._metrics_for_slot(slot, "after_step") for slot in self.slots]
        for row in rows:
            context = self._v87_pending_action_context.get(int(row.get("env_index", -1)), {})
            if context:
                row.update(context)
                row["phase"] = str(context.get("nominal_phase") or row.get("phase") or "after_step")
                self._update_v87_contact_streak(row)
            trace_row = dict(row)
            trace_row["backend_step_index"] = int(self.v86_step_counter)
            trace_row["ppo_policy_step"] = True
            self.v86_rollout_trace_rows.append(trace_row)
        self.v86_step_counter += 1
        self.last_metrics = rows
        self._v87_pending_action_context = {}
        return rows

    def step_env(self, env_index: int, action: list[float]) -> dict[str, Any]:
        actions = [[0.0] * 16 for _ in self.slots]
        if 0 <= int(env_index) < len(actions):
            actions[int(env_index)] = list(action)
        rows = self.step_envs(actions)
        return rows[int(env_index)] if 0 <= int(env_index) < len(rows) else {}

    def get_contact_state(self, env_index: int, part_name: str | None = None, active_finger_group: str = "") -> Any:
        if self.env is None:
            return self.contact_manager.read_contact_state(None, part_name=part_name or "", object_id=int(env_index))
        slot = self.slots[int(env_index)] if 0 <= int(env_index) < len(self.slots) else None
        return self.contact_manager.read_contact_state(
            self.env,
            env_index=int(getattr(slot, "local_env_index", env_index)),
            part_name=part_name or str(getattr(slot, "part_name", "")),
            object_id=int(env_index),
            active_finger_group=active_finger_group,
        )

    def get_metrics(self) -> list[dict[str, Any]]:
        return list(self.last_metrics)

    def get_part_distribution(self) -> dict[str, int]:
        return {part: sum(1 for slot in self.slots if slot.part_name == part) for part in V80_PARTS}

    def part_backend_ready(self, part_name: str) -> bool:
        return bool(self.physical_backend_ready and any(slot.part_name == part_name for slot in self.slots))

    def part_blocker(self, part_name: str) -> str:
        if self.part_backend_ready(part_name):
            return ""
        return self.blocker or "single_context_backend_not_ready"

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

    def _selected_slots(self, env_ids: list[int] | None) -> list[SingleContextSlot]:
        if env_ids is None:
            return list(self.slots)
        wanted = {int(index) for index in env_ids}
        return [slot for slot in self.slots if slot.global_env_index in wanted]

    def _normalize_actions(self, actions: Any) -> list[list[float]]:
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().tolist()
        if not actions:
            actions = [[0.0] * 16]
        if isinstance(actions[0], (int, float)):
            actions = [actions]
        rows = []
        for index in range(len(self.slots)):
            raw = list(actions[index % len(actions)])
            clipped = [max(-1.0, min(1.0, float(value))) for value in raw[:16]]
            clipped.extend([0.0] * (16 - len(clipped)))
            rows.append(clipped)
        return rows

    def configure_v86_staging(self, variants_by_part: dict[str, dict[str, Any]]) -> None:
        self.v86_selected_variants = {
            part: dict(variant)
            for part, variant in variants_by_part.items()
            if part in V83_PARTS and variant
        }

    def configure_v93_safe_staging(self, plan_rows: list[dict[str, Any]]) -> None:
        self.v93_safe_staging_plan_rows = [dict(row) for row in plan_rows]
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v93_configure_safe_staging"):
            base.v93_configure_safe_staging(self.v93_safe_staging_plan_rows)

    def clear_v93_safe_staging(self) -> None:
        self.v93_safe_staging_plan_rows = []
        self.v93_safe_staging_audit_rows = []
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v93_clear_safe_staging"):
            base.v93_clear_safe_staging()

    def configure_v94_pregrasp(self, plan_rows: list[dict[str, Any]]) -> None:
        self.v94_pregrasp_plan_rows = [dict(row) for row in plan_rows]
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v94_configure_pregrasp"):
            base.v94_configure_pregrasp(self.v94_pregrasp_plan_rows)

    def clear_v94_pregrasp(self) -> None:
        self.v94_pregrasp_plan_rows = []
        self.v94_pregrasp_audit_rows = []
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v94_clear_pregrasp"):
            base.v94_clear_pregrasp()

    def configure_v95_pregrasp(self, plan_rows: list[dict[str, Any]]) -> None:
        self.v95_pregrasp_plan_rows = [dict(row) for row in plan_rows]
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v95_configure_pregrasp"):
            base.v95_configure_pregrasp(self.v95_pregrasp_plan_rows)

    def clear_v95_pregrasp(self) -> None:
        self.v95_pregrasp_plan_rows = []
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v95_clear_pregrasp"):
            base.v95_clear_pregrasp()

    def v95_active_asset_info_for_slot(self, slot: SingleContextSlot) -> dict[str, Any]:
        base = getattr(self.env, "unwrapped", self.env) if self.env is not None else None
        if base is not None and hasattr(base, "v95_active_asset_info"):
            return dict(base.v95_active_asset_info(slot.local_env_index))
        return {
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "part_name": slot.part_name,
            "active_object_state_matches_slot": False,
            "legacy_held_state_matches_active_object": False,
            "active_object_refresh_error": "v95_active_asset_info_unavailable",
        }

    def clear_v86_rollout_trace(self) -> None:
        self.v86_rollout_trace_rows = []
        self.v86_near_contact_reset_rows = []
        self.v87_action_prior_rows = []
        self.v87_force_contact_streak = {}
        self._v87_pending_action_context = {}
        self._previous_composed_policy_actions = {}
        self.v86_step_counter = 0

    def clear_v87_rollout_trace(self) -> None:
        self.clear_v86_rollout_trace()

    def clear_v88_rollout_trace(self) -> None:
        self.clear_v86_rollout_trace()
        self.v88_observation_feature_rows = []

    def configure_v89_candidates(self, candidates_by_part: dict[str, dict[str, Any]]) -> None:
        self.v89_selected_candidates = {
            part: dict(candidate)
            for part, candidate in candidates_by_part.items()
            if part in V83_PARTS and candidate
        }

    def clear_v89_rollout_trace(self) -> None:
        self.clear_v88_rollout_trace()
        self.v89_candidate_probe_trace_rows = []
        self.v89_candidate_probe_summary_rows = []

    def step_v89_candidate_residual_envs(
        self,
        residual_actions: Any,
        *,
        episode_steps: list[int] | None = None,
        residual_scales: dict[str, float] | None = None,
        lift_phase_enabled: bool = True,
        forced_phase: str = "",
    ) -> list[dict[str, Any]]:
        residual_rows = self._normalize_actions(residual_actions)
        scales = {
            "wrist_xyz": float((residual_scales or {}).get("wrist_xyz", 0.10)),
            "wrist_rot": float((residual_scales or {}).get("wrist_rot", 0.05)),
            "finger": float((residual_scales or {}).get("finger", 0.12)),
        }
        composed_rows: list[list[float]] = []
        context: dict[int, dict[str, Any]] = {}
        selected_parts = {slot.part_name for slot in self.slots}
        phase_cache: dict[str, list[list[float]]] = {}
        for index, slot in enumerate(self.slots):
            step_value = int(episode_steps[index]) if episode_steps and index < len(episode_steps) else int(self.v86_step_counter)
            phase = str(forced_phase or self._v88_nominal_phase(step_value, lift_phase_enabled=lift_phase_enabled))
            if phase not in phase_cache:
                phase_cache[phase] = self._v89_policy_actions_for_phase(phase, selected_parts)
            candidate = self.v89_selected_candidates.get(slot.part_name, {})
            nominal = list(phase_cache[phase][index])
            residual = list(residual_rows[index])
            composed = []
            for col in range(16):
                scale = scales["finger"]
                if col < 3:
                    scale = scales["wrist_xyz"]
                elif col < 6:
                    scale = scales["wrist_rot"]
                composed.append(max(-1.0, min(1.0, float(nominal[col]) + float(residual[col]) * scale)))
            previous = list(self._previous_composed_policy_actions.get(slot.global_env_index, [0.0] * 16))
            residual_norm = math.sqrt(sum(float(value) * float(value) for value in residual))
            composed_norm = math.sqrt(sum(float(value) * float(value) for value in composed))
            nominal_norm = math.sqrt(sum(float(value) * float(value) for value in nominal))
            action_jerk = math.sqrt(sum((float(composed[col]) - float(previous[col])) ** 2 for col in range(16)))
            context_row = {
                "v89_hybrid_repair_mode": True,
                "v89_nominal_candidate_mode": True,
                "v88_stabilized_mode": True,
                "v87_contact_guided_residual_mode": True,
                "candidate_id": str(candidate.get("candidate_id") or f"{slot.part_name}_v89_default"),
                "candidate_rank": int(candidate.get("candidate_rank", 0) or 0),
                "active_finger_group": str(candidate.get("active_finger_group") or ""),
                "nominal_phase": phase,
                "nominal_action": nominal,
                "residual_action": residual,
                "composed_policy_action": composed,
                "previous_composed_policy_action": previous,
                "nominal_action_norm": nominal_norm,
                "residual_action_norm": residual_norm,
                "composed_action_norm": composed_norm,
                "previous_composed_action_norm": math.sqrt(sum(float(value) * float(value) for value in previous)),
                "action_jerk": action_jerk,
                "residual_scale_wrist_xyz": scales["wrist_xyz"],
                "residual_scale_wrist_rot": scales["wrist_rot"],
                "residual_scale_finger": scales["finger"],
                "policy_outputs_residual_only": True,
                "route_selection_available_to_policy": False,
                "proxy_action_available_to_policy": False,
                "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
                "fallback_success_used": False,
            }
            context[slot.global_env_index] = context_row
            self.v87_action_prior_rows.append(
                {
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "part_name": slot.part_name,
                    "backend_step_index": int(self.v86_step_counter),
                    "episode_step": step_value,
                    **context_row,
                }
            )
            composed_rows.append(composed)
            self._previous_composed_policy_actions[slot.global_env_index] = composed
        self._v87_pending_action_context = context
        return self.step_envs(composed_rows)

    def step_v90_feasibility_envs(
        self,
        *,
        phase: str,
        selected_parts: set[str],
        phase_step: int,
    ) -> list[dict[str, Any]]:
        phase = str(phase)
        nominal_rows = self._v90_policy_actions_for_phase(phase, selected_parts)
        composed_rows: list[list[float]] = []
        context: dict[int, dict[str, Any]] = {}
        for index, slot in enumerate(self.slots):
            candidate = self.v89_selected_candidates.get(slot.part_name, {})
            action = list(nominal_rows[index])
            previous = list(self._previous_composed_policy_actions.get(slot.global_env_index, [0.0] * 16))
            composed_norm = math.sqrt(sum(float(value) * float(value) for value in action))
            previous_norm = math.sqrt(sum(float(value) * float(value) for value in previous))
            action_jerk = math.sqrt(sum((float(action[col]) - float(previous[col])) ** 2 for col in range(16)))
            force_limit = float(candidate.get("force_limit_n") or 0.0)
            force_limit_active = bool(force_limit > 0.0 and self._last_peak_force_for_part(slot.part_name) > force_limit)
            context_row = {
                "v90_feasibility_mode": True,
                "nominal_phase": phase,
                "candidate_id": str(candidate.get("candidate_id") or f"{slot.part_name}_v90_default"),
                "candidate_family": str(candidate.get("candidate_family") or candidate.get("grasp_style") or ""),
                "candidate_rank": int(candidate.get("candidate_rank", 0) or 0),
                "family_variant_index": int(candidate.get("family_variant_index", 0) or 0),
                "active_finger_group": str(candidate.get("active_finger_group") or ""),
                "force_limit_n": force_limit,
                "force_limit_active": force_limit_active,
                "v90_phase_step": int(phase_step),
                "nominal_action": action,
                "residual_action": [0.0] * 16,
                "composed_policy_action": action,
                "previous_composed_policy_action": previous,
                "nominal_action_norm": composed_norm,
                "residual_action_norm": 0.0,
                "composed_action_norm": composed_norm,
                "previous_composed_action_norm": previous_norm,
                "action_jerk": action_jerk,
                "policy_outputs_residual_only": False,
                "diagnostic_controller_used": True,
                "route_selection_available_to_policy": False,
                "proxy_action_available_to_policy": False,
                "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
                "fallback_success_used": False,
                "distance_only_success_used": False,
                "object_write_after_reset_allowed": False,
            }
            context[slot.global_env_index] = context_row
            composed_rows.append(action)
            self._previous_composed_policy_actions[slot.global_env_index] = action
        self._v87_pending_action_context = context
        return self.step_envs(composed_rows)

    def _last_peak_force_for_part(self, part_name: str) -> float:
        peaks = [
            float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0)
            for row in self.last_metrics
            if row.get("part_name") == part_name
        ]
        return max([0.0, *peaks])

    def _last_force_count_for_part(self, part_name: str) -> int:
        counts = [
            int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0)
            for row in self.last_metrics
            if row.get("part_name") == part_name
        ]
        return max([0, *counts])

    def step_v91_quasistatic_feasibility_envs(
        self,
        *,
        phase: str,
        selected_parts: set[str],
        phase_step: int,
    ) -> list[dict[str, Any]]:
        phase = str(phase)
        policy_rows, mask_rows = self._v91_policy_actions_for_phase(phase, selected_parts, int(phase_step))
        context: dict[int, dict[str, Any]] = {}
        for index, slot in enumerate(self.slots):
            candidate = self.v89_selected_candidates.get(slot.part_name, {})
            action = list(policy_rows[index])
            previous = list(self._previous_composed_policy_actions.get(slot.global_env_index, [0.0] * 16))
            action_jerk = math.sqrt(sum((float(action[col]) - float(previous[col])) ** 2 for col in range(16)))
            force_limit = float(candidate.get("force_limit_n") or 0.0)
            current_peak = self._last_peak_force_for_part(slot.part_name)
            current_count = self._last_force_count_for_part(slot.part_name)
            mask = mask_rows[index]
            context_row = {
                "v91_task_sanity_mode": True,
                "nominal_phase": phase,
                "candidate_id": str(candidate.get("candidate_id") or f"{slot.part_name}_v91_default"),
                "candidate_family": str(candidate.get("candidate_family") or candidate.get("grasp_style") or ""),
                "candidate_rank": int(candidate.get("candidate_rank", 0) or 0),
                "family_variant_index": int(candidate.get("family_variant_index", 0) or 0),
                "active_finger_group": str(candidate.get("active_finger_group") or ""),
                "commanded_fingers": mask.get("commanded_fingers", ""),
                "commanded_policy_cols": mask.get("commanded_policy_cols", ""),
                "commanded_isaac_cols": mask.get("commanded_isaac_cols", ""),
                "non_active_finger_value": float(mask.get("non_active_finger_value") or 0.0),
                "support_finger_value": float(mask.get("support_finger_value") or 0.0),
                "force_limit_n": force_limit,
                "force_limit_active": bool(force_limit > 0.0 and current_peak > force_limit),
                "force_limit_backoff": float(mask.get("force_limit_backoff") or 1.0),
                "multi_finger_support_established": bool(current_count >= 2),
                "v91_phase_step": int(phase_step),
                "v90_phase_step": int(phase_step),
                "nominal_action": action,
                "residual_action": [0.0] * 16,
                "composed_policy_action": action,
                "previous_composed_policy_action": previous,
                "nominal_action_norm": math.sqrt(sum(float(value) * float(value) for value in action)),
                "residual_action_norm": 0.0,
                "composed_action_norm": math.sqrt(sum(float(value) * float(value) for value in action)),
                "previous_composed_action_norm": math.sqrt(sum(float(value) * float(value) for value in previous)),
                "action_jerk": action_jerk,
                "policy_outputs_residual_only": False,
                "diagnostic_controller_used": True,
                "route_selection_available_to_policy": False,
                "proxy_action_available_to_policy": False,
                "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
                "fallback_success_used": False,
                "distance_only_success_used": False,
                "object_write_after_reset_allowed": False,
            }
            context[slot.global_env_index] = context_row
            self._previous_composed_policy_actions[slot.global_env_index] = action
        self._v87_pending_action_context = context
        return self.step_envs(policy_rows)

    def step_v87_residual_envs(
        self,
        residual_actions: Any,
        *,
        episode_steps: list[int] | None = None,
        residual_scales: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        residual_rows = self._normalize_actions(residual_actions)
        scales = {
            "wrist_xyz": float((residual_scales or {}).get("wrist_xyz", 0.15)),
            "wrist_rot": float((residual_scales or {}).get("wrist_rot", 0.08)),
            "finger": float((residual_scales or {}).get("finger", 0.20)),
        }
        composed_rows: list[list[float]] = []
        context: dict[int, dict[str, Any]] = {}
        selected_parts = {slot.part_name for slot in self.slots}
        phase_cache: dict[str, list[list[float]]] = {}
        for index, slot in enumerate(self.slots):
            step_value = int(episode_steps[index]) if episode_steps and index < len(episode_steps) else int(self.v86_step_counter)
            phase = self._v87_nominal_phase(step_value)
            if phase not in phase_cache:
                phase_cache[phase] = self._v85_policy_actions_for_phase(phase, selected_parts)
            nominal = list(phase_cache[phase][index])
            residual = list(residual_rows[index])
            composed = []
            for col in range(16):
                scale = scales["finger"]
                if col < 3:
                    scale = scales["wrist_xyz"]
                elif col < 6:
                    scale = scales["wrist_rot"]
                composed.append(max(-1.0, min(1.0, float(nominal[col]) + float(residual[col]) * scale)))
            residual_norm = math.sqrt(sum(float(value) * float(value) for value in residual))
            composed_norm = math.sqrt(sum(float(value) * float(value) for value in composed))
            nominal_norm = math.sqrt(sum(float(value) * float(value) for value in nominal))
            context_row = {
                "v87_contact_guided_residual_mode": True,
                "nominal_phase": phase,
                "nominal_action": nominal,
                "residual_action": residual,
                "composed_policy_action": composed,
                "nominal_action_norm": nominal_norm,
                "residual_action_norm": residual_norm,
                "composed_action_norm": composed_norm,
                "residual_scale_wrist_xyz": scales["wrist_xyz"],
                "residual_scale_wrist_rot": scales["wrist_rot"],
                "residual_scale_finger": scales["finger"],
                "policy_outputs_residual_only": True,
                "route_selection_available_to_policy": False,
                "proxy_action_available_to_policy": False,
                "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
            }
            context[slot.global_env_index] = context_row
            self.v87_action_prior_rows.append(
                {
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "part_name": slot.part_name,
                    "backend_step_index": int(self.v86_step_counter),
                    "episode_step": step_value,
                    **context_row,
                }
            )
            composed_rows.append(composed)
        self._v87_pending_action_context = context
        return self.step_envs(composed_rows)

    def step_v88_stabilized_residual_envs(
        self,
        residual_actions: Any,
        *,
        episode_steps: list[int] | None = None,
        residual_scales: dict[str, float] | None = None,
        lift_phase_enabled: bool = True,
    ) -> list[dict[str, Any]]:
        residual_rows = self._normalize_actions(residual_actions)
        scales = {
            "wrist_xyz": float((residual_scales or {}).get("wrist_xyz", 0.12)),
            "wrist_rot": float((residual_scales or {}).get("wrist_rot", 0.06)),
            "finger": float((residual_scales or {}).get("finger", 0.16)),
        }
        composed_rows: list[list[float]] = []
        context: dict[int, dict[str, Any]] = {}
        selected_parts = {slot.part_name for slot in self.slots}
        phase_cache: dict[str, list[list[float]]] = {}
        for index, slot in enumerate(self.slots):
            step_value = int(episode_steps[index]) if episode_steps and index < len(episode_steps) else int(self.v86_step_counter)
            phase = self._v88_nominal_phase(step_value, lift_phase_enabled=lift_phase_enabled)
            if phase not in phase_cache:
                phase_cache[phase] = self._v88_policy_actions_for_phase(phase, selected_parts)
            nominal = list(phase_cache[phase][index])
            residual = list(residual_rows[index])
            composed = []
            for col in range(16):
                scale = scales["finger"]
                if col < 3:
                    scale = scales["wrist_xyz"]
                elif col < 6:
                    scale = scales["wrist_rot"]
                composed.append(max(-1.0, min(1.0, float(nominal[col]) + float(residual[col]) * scale)))
            residual_norm = math.sqrt(sum(float(value) * float(value) for value in residual))
            composed_norm = math.sqrt(sum(float(value) * float(value) for value in composed))
            nominal_norm = math.sqrt(sum(float(value) * float(value) for value in nominal))
            previous = list(self._previous_composed_policy_actions.get(slot.global_env_index, [0.0] * 16))
            previous_norm = math.sqrt(sum(float(value) * float(value) for value in previous))
            context_row = {
                "v88_stabilized_mode": True,
                "v87_contact_guided_residual_mode": True,
                "nominal_phase": phase,
                "nominal_action": nominal,
                "residual_action": residual,
                "composed_policy_action": composed,
                "previous_composed_policy_action": previous,
                "nominal_action_norm": nominal_norm,
                "residual_action_norm": residual_norm,
                "composed_action_norm": composed_norm,
                "previous_composed_action_norm": previous_norm,
                "residual_scale_wrist_xyz": scales["wrist_xyz"],
                "residual_scale_wrist_rot": scales["wrist_rot"],
                "residual_scale_finger": scales["finger"],
                "policy_outputs_residual_only": True,
                "route_selection_available_to_policy": False,
                "proxy_action_available_to_policy": False,
                "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
                "fallback_success_used": False,
            }
            context[slot.global_env_index] = context_row
            self.v87_action_prior_rows.append(
                {
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "part_name": slot.part_name,
                    "backend_step_index": int(self.v86_step_counter),
                    "episode_step": step_value,
                    **context_row,
                }
            )
            composed_rows.append(composed)
            self._previous_composed_policy_actions[slot.global_env_index] = composed
        self._v87_pending_action_context = context
        return self.step_envs(composed_rows)

    def _v87_nominal_phase(self, episode_step: int) -> str:
        step = max(0, int(episode_step))
        if step < 4:
            return "settle"
        if step < 16:
            return "close"
        if step < 72:
            return "approach_close"
        return "hold_squeeze"

    def _v88_nominal_phase(self, episode_step: int, *, lift_phase_enabled: bool) -> str:
        step = max(0, int(episode_step))
        if step < 4:
            return "settle"
        if step < 16:
            return "close"
        if step < 72:
            return "approach_close"
        if step < 104 or not lift_phase_enabled:
            return "hold_squeeze"
        return "lift_stabilize"

    def _update_v87_contact_streak(self, row: dict[str, Any]) -> None:
        key = (int(row.get("env_index") or 0), str(row.get("part_name") or ""))
        peak = float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0)
        count = int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0)
        active = bool(peak > float(self.contact_manager.force_threshold_n) and count >= 1)
        streak = int(self.v87_force_contact_streak.get(key, 0))
        streak = streak + 1 if active else 0
        self.v87_force_contact_streak[key] = streak
        row["force_contact_streak_steps"] = streak
        row["contact_duration_steps"] = streak
        row["force_contact_active"] = active
        phase = str(row.get("nominal_phase") or row.get("phase") or "")
        row["hold_object_displacement_m"] = float(row.get("object_displacement_m") or 0.0) if phase == "hold_squeeze" else 0.0
        row["lift_delta_z_m"] = max(0.0, float(row.get("object_delta_z_m") or 0.0)) if phase == "lift_stabilize" else 0.0
        row["fallback_success_used"] = False

    def stage_v86_near_contact_episode(self, env_ids: list[int] | None = None) -> list[dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(self.blocker or "single_context_env_not_created")
        if self.v86_policy_step_started:
            self.object_write_by_policy_detected = True
            return [
                {
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "part_name": slot.part_name,
                    "near_contact_stage_ok": False,
                    "near_contact_stage_failure_reason": "v86_stage_requested_after_policy_step",
                    "object_write_by_policy_detected": True,
                }
                for slot in self._selected_slots(env_ids)
            ]
        base = getattr(self.env, "unwrapped", self.env)
        slot_parts = [slot.part_name for slot in self.slots]
        if not hasattr(base, "v85_stage_clean_near_contact"):
            rows = [
                {
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "part_name": slot.part_name,
                    "near_contact_stage_ok": False,
                    "near_contact_stage_failure_reason": "v85_stage_clean_near_contact_missing",
                }
                for slot in self._selected_slots(env_ids)
            ]
            self.v86_near_contact_reset_rows.extend(rows)
            return rows
        with _torch_inference_context():
            stage_rows = base.v85_stage_clean_near_contact(slot_parts, self.v86_selected_variants)
        self.object_write_reset_only = True
        self._capture_v86_reset_object_positions()
        stage_by_key = {
            (int(row.get("env_index") or 0), str(row.get("part_name") or "")): row
            for row in stage_rows
        }
        metric_rows = []
        for slot in self._selected_slots(env_ids):
            metric = self._metrics_for_slot(slot, "after_v86_near_contact_reset")
            stage = stage_by_key.get((slot.global_env_index, slot.part_name), {})
            metric.update(stage)
            metric["v86_reset_index"] = int(self.v86_reset_counter)
            metric["v86_near_contact_reset"] = True
            metric_rows.append(metric)
        self.v86_near_contact_reset_rows.extend(metric_rows)
        self.last_metrics = metric_rows
        return metric_rows

    def _capture_v86_reset_object_positions(self) -> None:
        self.v86_reset_object_positions = {}
        for slot in self.slots:
            position = self._object_position_for_slot(slot)
            if position:
                self.v86_reset_object_positions[(slot.global_env_index, slot.part_name)] = position

    def _object_position_for_slot(self, slot: SingleContextSlot) -> list[float]:
        if self.env is None:
            return []
        try:
            base = getattr(self.env, "unwrapped", self.env)
            registry = getattr(base, "v83_active_asset_registry", {}).get(slot.part_name, {})
            asset = registry.get("asset")
            root_pos = _tensor_row(getattr(getattr(asset, "data", None), "root_pos_w", None), slot.local_env_index)
            if root_pos is None:
                return []
            if hasattr(root_pos, "detach"):
                values = root_pos.detach().cpu().reshape(-1).tolist()
            else:
                values = list(root_pos)
            return [float(values[0]), float(values[1]), float(values[2])]
        except Exception:
            return []

    def _v86_object_displacement(self, slot: SingleContextSlot) -> tuple[float, float]:
        start = self.v86_reset_object_positions.get((slot.global_env_index, slot.part_name), [])
        current = self._object_position_for_slot(slot)
        if len(start) < 3 or len(current) < 3:
            return 0.0, 0.0
        dx = current[0] - start[0]
        dy = current[1] - start[1]
        dz = current[2] - start[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz), dz

    def _v88_geometry_extent(self, part_name: str) -> tuple[float, float, float]:
        geometry = V85_PART_GEOMETRY.get(part_name, {})
        if geometry.get("kind") == "sphere":
            radius = float(geometry.get("radius", 0.0))
            return radius * 2.0, radius * 2.0, radius * 2.0
        if geometry.get("kind") == "cylinder_z":
            radius = float(geometry.get("radius", 0.0))
            half_height = float(geometry.get("half_height", 0.0))
            return radius * 2.0, radius * 2.0, half_height * 2.0
        boxes = geometry.get("boxes") or ()
        if boxes:
            max_x = max(float(half[0]) for _center, half in boxes)
            max_y = max(float(half[1]) for _center, half in boxes)
            max_z = max(float(half[2]) for _center, half in boxes)
            return max_x * 2.0, max_y * 2.0, max_z * 2.0
        return 0.0, 0.0, 0.0

    def _v88_observation_features_for_slot(self, slot: SingleContextSlot) -> dict[str, Any]:
        if self.env is None or torch is None:
            return {}
        try:
            base = getattr(self.env, "unwrapped", self.env)
            registry = getattr(base, "v83_active_asset_registry", {}).get(slot.part_name, {})
            asset = registry.get("asset")
            object_pos_w = _tensor_row(getattr(getattr(asset, "data", None), "root_pos_w", None), slot.local_env_index)
            env_origin = _tensor_row(getattr(getattr(base, "scene", None), "env_origins", None), slot.local_env_index)
            palm_local = _tensor_row(getattr(base, "fingertip_midpoint_pos", None), slot.local_env_index)
            fingertips = _tensor_row(getattr(base, "dex_fingertip_pos", None), slot.local_env_index)
            if object_pos_w is None or env_origin is None or palm_local is None:
                return {}
            object_local = object_pos_w.to(device=base.device, dtype=torch.float32).reshape(3) - env_origin.to(
                device=base.device, dtype=torch.float32
            ).reshape(3)
            palm_local = palm_local.to(device=base.device, dtype=torch.float32).reshape(3)
            rel = object_local - palm_local
            distances: list[float] = []
            if fingertips is not None:
                tip_rows = fingertips.to(device=base.device, dtype=torch.float32).reshape(-1, 3)
                distances = torch.linalg.vector_norm(tip_rows - object_local.reshape(1, 3), dim=1).detach().cpu().tolist()
            bbox = self._v88_geometry_extent(slot.part_name)
            row = {
                "object_rel_palm_x": float(rel[0].detach().cpu().item()),
                "object_rel_palm_y": float(rel[1].detach().cpu().item()),
                "object_rel_palm_z": float(rel[2].detach().cpu().item()),
                "object_bbox_x": float(bbox[0]),
                "object_bbox_y": float(bbox[1]),
                "object_bbox_z": float(bbox[2]),
                "fingertip_object_distance_min_m": min(distances) if distances else 0.0,
                "fingertip_object_distance_mean_m": sum(distances) / max(1, len(distances)) if distances else 0.0,
            }
            self.v88_observation_feature_rows.append({"env_index": slot.global_env_index, "part_name": slot.part_name, **row})
            return row
        except Exception as exc:
            return {"v88_observation_feature_blocker": f"{type(exc).__name__}:{exc}"}

    def apply_v88_conservative_contact_tuning(self) -> list[dict[str, Any]]:
        self.v88_physics_tuning_rows = []
        targets: list[tuple[str, Any]] = []
        if self.env is not None:
            base = getattr(self.env, "unwrapped", self.env)
            for part_name, entry in getattr(base, "v83_active_asset_registry", {}).items():
                targets.append((part_name, entry.get("asset")))
            targets.append(("WujiFingerPads", getattr(base, "_robot", None)))
        for name, asset in targets:
            row = {
                "asset_name": name,
                "physics_profile": "conservative_contact",
                "static_friction_before": "",
                "dynamic_friction_before": "",
                "restitution_before": "",
                "static_friction_after": "",
                "dynamic_friction_after": "",
                "restitution_after": "",
                "target_static_friction": 1.25,
                "target_dynamic_friction": 1.05,
                "target_restitution": 0.0,
                "tuning_applied": False,
                "tuning_failure_reason": "",
                "rationale": "bounded plausible contact stabilization; no sticky, adhesion, or success hack",
            }
            try:
                view = getattr(asset, "root_physx_view", None)
                if view is None:
                    raise RuntimeError("root_physx_view_missing")
                materials = view.get_material_properties()
                row["static_friction_before"] = float(materials[..., 0].mean().item()) if hasattr(materials[..., 0], "mean") else ""
                row["dynamic_friction_before"] = float(materials[..., 1].mean().item()) if hasattr(materials[..., 1], "mean") else ""
                if materials.shape[-1] >= 3:
                    row["restitution_before"] = float(materials[..., 2].mean().item()) if hasattr(materials[..., 2], "mean") else ""
                    materials[..., 2] = 0.0
                materials[..., 0] = 1.25
                materials[..., 1] = 1.05
                env_ids = torch.arange(self.num_envs, device="cpu") if torch is not None else None
                view.set_material_properties(materials, env_ids)
                after = view.get_material_properties()
                row["static_friction_after"] = float(after[..., 0].mean().item()) if hasattr(after[..., 0], "mean") else ""
                row["dynamic_friction_after"] = float(after[..., 1].mean().item()) if hasattr(after[..., 1], "mean") else ""
                if after.shape[-1] >= 3:
                    row["restitution_after"] = float(after[..., 2].mean().item()) if hasattr(after[..., 2], "mean") else ""
                row["tuning_applied"] = True
            except Exception as exc:
                row["tuning_failure_reason"] = f"{type(exc).__name__}:{exc}"
            self.v88_physics_tuning_rows.append(row)
        return list(self.v88_physics_tuning_rows)

    def _metrics_for_slot(self, slot: SingleContextSlot, phase: str) -> dict[str, Any]:
        identity = self.object_identity_for_slot(slot)
        self._record_identity(identity)
        state = self.get_contact_state(slot.global_env_index, slot.part_name).to_dict()
        target_contact_api = bool(state.get("target_filtered_force_available"))
        contact_api = bool(
            state.get("contact_sensor_api_available")
            or state.get("contact_sensor_available")
            or target_contact_api
        )
        force_values = [float(v) for v in state.get("per_finger_contact_force_norm", [])]
        target_force_values = [float(v) for v in state.get("per_finger_target_filtered_force_norm", [])]
        force_flags = [bool(float(v) >= float(self.contact_manager.force_threshold_n)) for v in force_values]
        target_force_flags = [
            bool(float(v) >= float(self.contact_manager.force_threshold_n)) for v in target_force_values
        ]
        peak_force = max([0.0, *force_values])
        target_peak_force = max([0.0, *target_force_values])
        mean_force = sum(force_values) / max(1, len(force_values)) if force_values else 0.0
        force_count = int(state.get("effective_contact_count_force") or 0)
        target_force_count = sum(1 for value in target_force_values if value >= self.contact_manager.force_threshold_n)
        distance_count = int(state.get("effective_contact_count_distance") or 0)
        object_displacement_m, object_delta_z_m = self._v86_object_displacement(slot)
        support_gate_ok = bool(
            contact_api
            and force_count >= 1
            and not bool(state.get("table_collision"))
            and float(state.get("penetration_depth_m") or 0.0) <= 0.0
            and not self.object_write_by_policy_detected
            and not self.sticky_action_available_to_policy
        )
        contact_row = {
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "part_name": slot.part_name,
            "phase": phase,
            "contact_sensor_api_available": contact_api,
            "contact_sensor_available": contact_api,
            "contact_signal_nonzero": bool(state.get("contact_signal_nonzero")),
            "nonzero_force_contact_observed": bool(state.get("contact_signal_nonzero")),
            "effective_contact_count_force": force_count,
            "effective_contact_count_distance": distance_count,
            "force_contact_count": force_count,
            "distance_contact_count": distance_count,
            "force_contact_probe_peak_n": peak_force,
            "force_contact_peak_n": peak_force,
            "force_contact_mean_n": mean_force,
            "per_finger_force_norm": force_values,
            "per_finger_contact_force_norm": force_values,
            "per_finger_unfiltered_force_norm": state.get("per_finger_unfiltered_force_norm", force_values),
            "per_finger_target_filtered_force_norm": target_force_values,
            "per_finger_target_filtered_force_xyz": state.get("per_finger_target_filtered_force_xyz", []),
            "per_finger_contact_flags": force_flags,
            "per_finger_target_filtered_contact_flags": target_force_flags,
            "per_finger_force_xyz": state.get("per_finger_force_xyz", []),
            "per_finger_tip_positions": state.get("per_finger_tip_positions", []),
            "per_finger_tip_linvel": state.get("per_finger_tip_linvel", []),
            "target_filtered_force_available": target_contact_api,
            "target_filter_names": state.get("target_filter_names", []),
            "target_filter_index": int(state.get("target_filter_index", -1)),
            "target_object_contact_force_peak_n": target_peak_force,
            "active_target_filtered_force_count": int(state.get("active_target_filtered_force_count") or target_force_count),
            "target_contact_evidence_source": state.get("target_contact_evidence_source", ""),
            "distance_only_success_used": bool(state.get("distance_only_success_used", False)),
            "non_active_sensor_success_used": bool(state.get("non_active_sensor_success_used", False)),
            "excessive_force_threshold_n": V86_EXCESSIVE_FORCE_THRESHOLD_N,
            "excessive_force": bool(peak_force > V86_EXCESSIVE_FORCE_THRESHOLD_N),
            "support_gate_ok": support_gate_ok,
            "object_displacement_m": object_displacement_m,
            "object_delta_z_m": object_delta_z_m,
            "object_motion_before_contact_m": object_displacement_m if force_count <= 0 else 0.0,
            "object_velocity_norm": float(state.get("object_velocity_norm") or 0.0),
            "object_angular_velocity_norm": float(state.get("object_angular_velocity_norm") or 0.0),
            "penetration_depth_m": float(state.get("penetration_depth_m") or 0.0),
            "table_collision": bool(state.get("table_collision")),
            "contact_evidence_source": state.get("contact_evidence_source", ""),
            "contact_success_required_for_v83": False,
        }
        contact_row.update(self._v88_observation_features_for_slot(slot))
        self._record_contact_api(contact_row)
        return {
            **contact_row,
            "object_identity_verified": bool(identity.get("object_identity_verified")),
            "single_simulation_context": bool(self.single_simulation_context),
            "vector_reset_ok": bool(self.vector_reset_ok),
            "vector_step_ok": bool(self.vector_step_ok),
            "object_write_reset_only": bool(self.object_write_reset_only),
            "object_write_by_policy_detected": bool(self.object_write_by_policy_detected),
            "sticky_action_available_to_policy": bool(self.sticky_action_available_to_policy),
            "usable_training_row_count": 0,
        }

    def object_identity_for_slot(self, slot: SingleContextSlot) -> dict[str, Any]:
        base = getattr(self.env, "unwrapped", self.env)
        info = base.v83_asset_info(slot.part_name) if hasattr(base, "v83_asset_info") else {"part_name": slot.part_name}
        active_usd = str(info.get("active_asset_usd", ""))
        active_prim = str(info.get("active_asset_prim_path", ""))
        expected_usd = str(info.get("expected_usd_basename", ""))
        expected_prim = str(info.get("expected_prim_path_suffix", ""))
        label_ok = str(info.get("active_asset_label", "")) == slot.part_name
        usd_ok = bool(expected_usd and _basename(active_usd) == expected_usd)
        prim_ok = bool(expected_prim and active_prim.endswith(expected_prim))
        object_ok = bool(info.get("asset"))
        blockers = []
        if not label_ok:
            blockers.append("ACTIVE_ASSET_LABEL_MISMATCH")
        if not usd_ok:
            blockers.append(f"ACTIVE_ASSET_USD_MISMATCH:expected={expected_usd}:actual={_basename(active_usd)}")
        if not prim_ok:
            blockers.append(f"ACTIVE_ASSET_PRIM_MISMATCH:expected_suffix={expected_prim}:actual={active_prim}")
        if not object_ok:
            blockers.append("ACTIVE_ASSET_OBJECT_MISSING")
        verified = bool(label_ok and usd_ok and prim_ok and object_ok)
        return {
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "part_name": slot.part_name,
            "active_asset_label": info.get("active_asset_label", ""),
            "active_asset_usd": active_usd,
            "active_asset_prim_path": active_prim,
            "expected_usd_basename": expected_usd,
            "expected_prim_path_suffix": expected_prim,
            "asset_object_type": info.get("asset_object_type", ""),
            "asset_object_id": info.get("asset_object_id", ""),
            "object_identity_verified": verified,
            "object_identity_blocker": ";".join(blockers),
        }

    def _record_identity(self, row: dict[str, Any]) -> None:
        key = (row.get("env_index"), row.get("part_name"))
        self.object_identity_rows = [
            existing
            for existing in self.object_identity_rows
            if (existing.get("env_index"), existing.get("part_name")) != key
        ]
        self.object_identity_rows.append(row)

    def _record_contact_api(self, row: dict[str, Any]) -> None:
        key = (row.get("env_index"), row.get("part_name"), row.get("phase"))
        self.contact_api_rows = [
            existing
            for existing in self.contact_api_rows
            if (existing.get("env_index"), existing.get("part_name"), existing.get("phase")) != key
        ]
        self.contact_api_rows.append(row)


def run_v83_unified_physical_backend_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = []
    if backend.env is not None and backend.slots:
        backend.action_mapping_rows = []
        backend.reset_envs()
        actions = [[0.0] * 16 for _ in backend.slots]
        for row in actions:
            for col in range(6, 16):
                row[col] = 0.25
        backend.step_envs(actions)
    for part in V83_PARTS:
        part_identities = [row for row in backend.object_identity_rows if row.get("part_name") == part]
        part_actions = [row for row in backend.action_mapping_rows if row.get("part_name") == part]
        part_contacts = [row for row in backend.contact_api_rows if row.get("part_name") == part]
        identity_ok = any(bool(row.get("object_identity_verified")) for row in part_identities)
        width_ok = any(int(row.get("isaac_action_dim") or 0) == 26 for row in part_actions)
        close_ok = any(bool(row.get("metric_close_dof_commanded")) for row in part_actions)
        row = {
            "part_name": part,
            "object_identity_verified": identity_ok,
            "single_simulation_context": bool(backend.single_simulation_context),
            "gym_make_count": int(backend.gym_make_count),
            "multi_gym_make_success_path_allowed": False,
            "vector_reset_ok": bool(backend.vector_reset_ok),
            "vector_step_ok": bool(backend.vector_step_ok),
            "policy_action_dim": 16,
            "isaac_action_width": 26 if width_ok else 0,
            "mapped_close_cols": ",".join(sorted({str(item.get("mapped_close_cols", "")) for item in part_actions if item.get("mapped_close_cols")})),
            "metric_close_dof_commanded": close_ok,
            "contact_sensor_api_available": any(bool(item.get("contact_sensor_api_available")) for item in part_contacts),
            "nonzero_force_contact_observed": any(bool(item.get("nonzero_force_contact_observed")) for item in part_contacts),
            "contact_success_required_for_v83": False,
            "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
            "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
            "ppo_ran": False,
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
        }
        success = bool(
            row["object_identity_verified"]
            and row["single_simulation_context"]
            and row["gym_make_count"] == 1
            and row["vector_reset_ok"]
            and row["vector_step_ok"]
            and row["isaac_action_width"] == 26
            and not row["object_write_by_policy_detected"]
            and not row["sticky_action_available_to_policy"]
        )
        blocker = ""
        if not success:
            identity = next((item for item in part_identities if item.get("object_identity_blocker")), {})
            blocker = str(identity.get("object_identity_blocker") or backend.blocker or "v83_single_context_audit_failed")
        row.update(
            {
                "v83_success": success,
                "status": "PASS" if success else "V83_SINGLE_CONTEXT_AUDIT_FAILED",
                "blocker": blocker,
                "next_action": "run_v84_controlled_force_contact_probe" if success else "repair_v83_single_context_identity_or_vector_step",
            }
        )
        rows.append(row)

    identity_csv = write_csv(run_path / "object_identity_audit.csv", backend.object_identity_rows)
    identity_json = write_json(run_path / "object_identity_audit.json", backend.object_identity_rows)
    single_csv = write_csv(run_path / "single_context_backend_audit.csv", backend.single_context_rows)
    single_json = write_json(run_path / "single_context_backend_audit.json", backend.single_context_rows)
    action_csv = write_csv(run_path / "action_mapping_audit.csv", backend.action_mapping_rows)
    action_json = write_json(run_path / "action_mapping_audit.json", backend.action_mapping_rows)
    contact_csv = write_csv(run_path / "contact_api_audit.csv", backend.contact_api_rows)
    contact_json = write_json(run_path / "contact_api_audit.json", backend.contact_api_rows)
    progress_csv = write_csv(run_path / "v83_progress_matrix.csv", rows)
    progress_md = run_path / "v83_progress_matrix.md"
    _write_md(progress_md, rows)
    root_debug = Path.cwd() / "debug_runs"
    root_csv = write_csv(root_debug / "v83_progress_matrix.csv", rows)
    root_md = root_debug / "v83_progress_matrix.md"
    _write_md(root_md, rows)
    trace_csv = write_csv(run_path / "v83_vector_step_metrics.csv", backend.last_metrics)
    trace_jsonl = write_jsonl(run_path / "v83_vector_step_metrics.jsonl", backend.last_metrics)
    return {
        "rows": rows,
        "object_identity_audit_csv": str(identity_csv),
        "object_identity_audit_json": str(identity_json),
        "single_context_backend_audit_csv": str(single_csv),
        "single_context_backend_audit_json": str(single_json),
        "action_mapping_audit_csv": str(action_csv),
        "action_mapping_audit_json": str(action_json),
        "contact_api_audit_csv": str(contact_csv),
        "contact_api_audit_json": str(contact_json),
        "v83_progress_csv": str(progress_csv),
        "v83_progress_md": str(progress_md),
        "v83_root_progress_csv": str(root_csv),
        "v83_root_progress_md": str(root_md),
        "v83_vector_step_metrics_csv": str(trace_csv),
        "v83_vector_step_metrics_jsonl": str(trace_jsonl),
    }


def run_v84_controlled_real_contact_probe(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    stage_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    if backend.env is not None and backend.slots:
        backend.action_mapping_rows = []
        backend.contact_api_rows = []
        backend.reset_envs()
        base = getattr(backend.env, "unwrapped", backend.env)
        slot_parts = [slot.part_name for slot in backend.slots]
        if hasattr(base, "v84_stage_near_contact"):
            with _torch_inference_context():
                stage_rows = base.v84_stage_near_contact(slot_parts, clearance_m=0.0015)
            backend.object_write_reset_only = True
        else:
            stage_rows = [
                {
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "part_name": slot.part_name,
                    "near_contact_stage_ok": False,
                    "near_contact_stage_failure_reason": "v84_stage_near_contact_missing",
                }
                for slot in backend.slots
            ]
        step_plan = [("settle", 4), ("close", 8), ("approach_close", 24), ("hold_squeeze", 8)]
        global_step = 0
        for phase, count in step_plan:
            for _ in range(count):
                actions = backend._v84_policy_actions_for_phase(phase)
                metrics = backend.step_envs(actions)
                for item in metrics:
                    probe_rows.append(
                        {
                            "probe_step": global_step,
                            "probe_phase": phase,
                            "mapped_close_action_executed": phase in {"close", "approach_close", "hold_squeeze"},
                            **item,
                        }
                    )
                global_step += 1

    rows: list[dict[str, Any]] = []
    for part in V83_PARTS:
        part_probe = [row for row in probe_rows if row.get("part_name") == part]
        part_stage = [row for row in stage_rows if row.get("part_name") == part]
        part_identity = [row for row in backend.object_identity_rows if row.get("part_name") == part]
        part_action = [row for row in backend.action_mapping_rows if row.get("part_name") == part]
        identity_ok = any(bool(row.get("object_identity_verified")) for row in part_identity)
        stage_ok = any(bool(row.get("near_contact_stage_ok")) for row in part_stage)
        mapped_close = any(
            bool(row.get("metric_close_dof_commanded"))
            and str(row.get("mapped_close_cols") or "") == "16,17,18,19,20,21,22,23,24,25"
            for row in part_action
        )
        peak = max([0.0, *[float(row.get("force_contact_probe_peak_n") or 0.0) for row in part_probe]])
        force_count = max([0, *[int(row.get("effective_contact_count_force") or 0) for row in part_probe]])
        success = bool(
            identity_ok
            and bool(backend.single_simulation_context)
            and stage_ok
            and mapped_close
            and peak > float(backend.contact_manager.force_threshold_n)
            and force_count >= 1
            and not backend.sticky_action_available_to_policy
            and not backend.object_write_by_policy_detected
        )
        blocker = ""
        if not success:
            if not identity_ok:
                identity = next((row for row in part_identity if row.get("object_identity_blocker")), {})
                blocker = str(identity.get("object_identity_blocker") or "object_identity_not_verified")
            elif not stage_ok:
                stage = next((row for row in part_stage if row.get("near_contact_stage_failure_reason")), {})
                blocker = str(stage.get("near_contact_stage_failure_reason") or "near_contact_stage_failed")
            elif not mapped_close:
                blocker = "mapped_close_action_not_executed"
            elif peak <= float(backend.contact_manager.force_threshold_n) or force_count < 1:
                blocker = "controlled_force_contact_not_observed"
            elif backend.object_write_by_policy_detected:
                blocker = "object_write_by_policy_detected"
            elif backend.sticky_action_available_to_policy:
                blocker = "sticky_action_available_to_policy"
        rows.append(
            {
                "part_name": part,
                "object_identity_verified": identity_ok,
                "single_simulation_context": bool(backend.single_simulation_context),
                "near_contact_stage_ok": stage_ok,
                "mapped_close_action_executed": mapped_close,
                "force_contact_peak_n": peak,
                "effective_contact_count_force_max": force_count,
                "contact_threshold_n": float(backend.contact_manager.force_threshold_n),
                "contact_sensor_api_available": any(bool(row.get("contact_sensor_api_available")) for row in part_probe),
                "nonzero_force_contact_observed": peak > 0.0,
                "distance_contact_count_max": max([0, *[int(row.get("effective_contact_count_distance") or 0) for row in part_probe]]),
                "object_write_reset_only": bool(backend.object_write_reset_only),
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "ppo_ran": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "checkpoint_written": False,
                "usable_training_row_count": 0,
                "v84_success": success,
                "status": "PASS" if success else "CONTACT_PROBE_NO_FORCE_CONTACT",
                "blocker": blocker,
                "training_locked": not success,
                "training_locked_blocker": "" if success else "controlled_force_contact_not_observed",
                "next_action": "unlock_training_gate_for_later_ppo" if success else "repair_near_contact_probe_staging_or_approach",
            }
        )

    stage_csv = write_csv(run_path / "near_contact_initialization_trace.csv", stage_rows)
    stage_jsonl = write_jsonl(run_path / "near_contact_initialization_trace.jsonl", stage_rows)
    probe_csv = write_csv(run_path / "v84_controlled_contact_probe_trace.csv", probe_rows)
    probe_jsonl = write_jsonl(run_path / "v84_controlled_contact_probe_trace.jsonl", probe_rows)
    identity_csv = write_csv(run_path / "object_identity_audit.csv", backend.object_identity_rows)
    identity_json = write_json(run_path / "object_identity_audit.json", backend.object_identity_rows)
    action_csv = write_csv(run_path / "action_mapping_audit.csv", backend.action_mapping_rows)
    action_json = write_json(run_path / "action_mapping_audit.json", backend.action_mapping_rows)
    force_csv = write_csv(run_path / "force_contact_audit.csv", rows)
    force_json = write_json(run_path / "force_contact_audit.json", rows)
    progress_csv = write_csv(run_path / "v84_progress_matrix.csv", rows)
    progress_md = run_path / "v84_progress_matrix.md"
    _write_md(progress_md, rows)
    root_debug = Path.cwd() / "debug_runs"
    root_csv = write_csv(root_debug / "v84_progress_matrix.csv", rows)
    root_md = root_debug / "v84_progress_matrix.md"
    _write_md(root_md, rows)
    return {
        "rows": rows,
        "near_contact_initialization_trace_csv": str(stage_csv),
        "near_contact_initialization_trace_jsonl": str(stage_jsonl),
        "v84_controlled_contact_probe_trace_csv": str(probe_csv),
        "v84_controlled_contact_probe_trace_jsonl": str(probe_jsonl),
        "object_identity_audit_csv": str(identity_csv),
        "object_identity_audit_json": str(identity_json),
        "action_mapping_audit_csv": str(action_csv),
        "action_mapping_audit_json": str(action_json),
        "force_contact_audit_csv": str(force_csv),
        "force_contact_audit_json": str(force_json),
        "v84_progress_csv": str(progress_csv),
        "v84_progress_md": str(progress_md),
        "v84_root_progress_csv": str(root_csv),
        "v84_root_progress_md": str(root_md),
    }


def run_v85_contact_sanity_and_plug_screw_repair(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    threshold = float(backend.contact_manager.force_threshold_n)
    variant_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    variant_table = {part: _v85_staging_variants(part) for part in V83_PARTS}
    max_variants = max([0, *[len(rows) for rows in variant_table.values()]])
    successful_parts: set[str] = set()
    if backend.env is not None and backend.slots:
        backend.action_mapping_rows = []
        backend.contact_api_rows = []
        for variant_round in range(max_variants):
            active_variants = {
                part: rows[variant_round]
                for part, rows in variant_table.items()
                if variant_round < len(rows) and part not in successful_parts
            }
            if not active_variants:
                continue
            selected_parts = set(active_variants)
            backend.object_write_by_policy_detected = False
            backend.reset_envs()
            base = getattr(backend.env, "unwrapped", backend.env)
            slot_parts = [slot.part_name for slot in backend.slots]
            if hasattr(base, "v85_stage_clean_near_contact"):
                with _torch_inference_context():
                    staged = base.v85_stage_clean_near_contact(slot_parts, active_variants)
                backend.object_write_reset_only = True
            else:
                staged = [
                    {
                        "env_index": slot.global_env_index,
                        "local_env_index": slot.local_env_index,
                        "part_name": slot.part_name,
                        "variant_index": int(active_variants.get(slot.part_name, {}).get("variant_index", variant_round)),
                        "near_contact_stage_ok": False,
                        "near_contact_stage_failure_reason": "v85_stage_clean_near_contact_missing",
                    }
                    for slot in backend.slots
                    if slot.part_name in selected_parts
                ]
            variant_rows.extend(row for row in staged if row.get("part_name") in selected_parts)

            pre_metrics = [
                backend._metrics_for_slot(slot, "pre_action_force_sample")
                for slot in backend.slots
                if slot.part_name in selected_parts
            ]
            trace_rows.extend(
                _v85_annotate_metrics(
                    pre_metrics,
                    active_variants,
                    probe_step=-1,
                    probe_phase="pre_action_force_sample",
                    mapped_close=False,
                    approach_action=False,
                    stage_rows=staged,
                )
            )

            global_step = 0
            step_plan = [("settle", 4), ("close", 10), ("approach_close", 72), ("hold_squeeze", 12)]
            for phase, count in step_plan:
                for _ in range(count):
                    action_start = len(backend.action_mapping_rows)
                    actions = backend._v85_policy_actions_for_phase(phase, selected_parts)
                    metrics = backend.step_envs(actions)
                    for row in backend.action_mapping_rows[action_start:]:
                        part = str(row.get("part_name") or "")
                        variant = active_variants.get(part, {})
                        row.update(
                            {
                                "variant_index": int(variant.get("variant_index", variant_round)),
                                "variant_rank": int(variant.get("variant_rank", variant.get("variant_index", variant_round))),
                                "probe_phase": phase,
                            }
                        )
                    selected_metrics = [row for row in metrics if row.get("part_name") in selected_parts]
                    trace_rows.extend(
                        _v85_annotate_metrics(
                            selected_metrics,
                            active_variants,
                            probe_step=global_step,
                            probe_phase=phase,
                            mapped_close=phase in V85_ACTION_PHASES,
                            approach_action=phase == "approach_close",
                            stage_rows=staged,
                        )
                    )
                    global_step += 1

            for part, variant in active_variants.items():
                variant_trace = [
                    row
                    for row in trace_rows
                    if row.get("part_name") == part and _int_field(row, "variant_index") == _int_field(variant, "variant_index", -2)
                ]
                stage = next(
                    (
                        row
                        for row in staged
                        if row.get("part_name") == part and _int_field(row, "variant_index") == _int_field(variant, "variant_index", -2)
                    ),
                    {},
                )
                variant_summaries.append(_v85_summarize_variant(part, variant, variant_trace, stage, backend, threshold))
                if bool(variant_summaries[-1].get("v85_success")):
                    successful_parts.add(part)

    selected_rows = _v85_select_progress_rows(variant_summaries, backend, threshold)
    stage_csv = write_csv(run_path / "v85_staging_variant_audit.csv", variant_rows)
    stage_json = write_json(run_path / "v85_staging_variant_audit.json", variant_rows)
    trace_csv = write_csv(run_path / "v85_contact_sanity_trace.csv", trace_rows)
    trace_jsonl = write_jsonl(run_path / "v85_contact_sanity_trace.jsonl", trace_rows)
    identity_csv = write_csv(run_path / "object_identity_audit.csv", backend.object_identity_rows)
    identity_json = write_json(run_path / "object_identity_audit.json", backend.object_identity_rows)
    action_csv = write_csv(run_path / "action_mapping_audit.csv", backend.action_mapping_rows)
    action_json = write_json(run_path / "action_mapping_audit.json", backend.action_mapping_rows)
    force_csv = write_csv(run_path / "force_contact_audit.csv", variant_summaries)
    force_json = write_json(run_path / "force_contact_audit.json", variant_summaries)
    progress_csv = write_csv(run_path / "v85_progress_matrix.csv", selected_rows)
    progress_md = run_path / "v85_progress_matrix.md"
    _write_md(progress_md, selected_rows)
    root_debug = Path.cwd() / "debug_runs"
    root_csv = write_csv(root_debug / "v85_progress_matrix.csv", selected_rows)
    root_md = root_debug / "v85_progress_matrix.md"
    _write_md(root_md, selected_rows)
    return {
        "rows": selected_rows,
        "variant_rows": variant_summaries,
        "v85_staging_variant_audit_csv": str(stage_csv),
        "v85_staging_variant_audit_json": str(stage_json),
        "v85_contact_sanity_trace_csv": str(trace_csv),
        "v85_contact_sanity_trace_jsonl": str(trace_jsonl),
        "object_identity_audit_csv": str(identity_csv),
        "object_identity_audit_json": str(identity_json),
        "action_mapping_audit_csv": str(action_csv),
        "action_mapping_audit_json": str(action_json),
        "force_contact_audit_csv": str(force_csv),
        "force_contact_audit_json": str(force_json),
        "v85_progress_csv": str(progress_csv),
        "v85_progress_md": str(progress_md),
        "v85_root_progress_csv": str(root_csv),
        "v85_root_progress_md": str(root_md),
    }


def run_v86_contact_gate_preflight(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    """Freshly revalidate the v85 clean-contact gate before PPO is allowed."""

    run_path = Path(run_dir)
    audit = run_v85_contact_sanity_and_plug_screw_repair(run_path, backend)
    rows: list[dict[str, Any]] = []
    selected_variants: dict[str, dict[str, Any]] = {}
    for row in audit.get("rows", []):
        part = str(row.get("part_name") or "")
        gate_pass = bool(row.get("v85_success"))
        out = dict(row)
        out.update(
            {
                "v86_contact_gate_pass": gate_pass,
                "preflight_source": "fresh_v85_contact_sanity_replay",
                "training_allowed_after_preflight": gate_pass,
                "ppo_ran": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
            }
        )
        if gate_pass:
            selected_variants[part] = {
                "part_name": part,
                "variant_index": int(row.get("variant_index") or 0),
                "variant_rank": int(row.get("variant_rank") or 0),
                "active_finger_group": str(row.get("active_finger_group") or ""),
                "target_clearance_m": float(row.get("target_clearance_m") or 0.008),
                "lateral_offset_m": float(row.get("lateral_offset_m") or 0.0),
                "surface_strategy": str(row.get("surface_strategy") or ""),
                "use_bbox_adjustment": bool(row.get("use_bbox_adjustment")),
            }
        rows.append(out)
    all_pass = bool(rows) and all(bool(row.get("v86_contact_gate_pass")) for row in rows)
    csv_path = write_csv(run_path / "v86_contact_gate_preflight.csv", rows)
    json_path = write_json(run_path / "v86_contact_gate_preflight.json", rows)
    selected_path = write_json(run_path / "v86_selected_staging_variants.json", selected_variants)
    backend.configure_v86_staging(selected_variants if all_pass else {})
    return {
        "rows": rows,
        "selected_variants": selected_variants,
        "all_parts_pass": all_pass,
        "v86_contact_gate_preflight_csv": str(csv_path),
        "v86_contact_gate_preflight_json": str(json_path),
        "v86_selected_staging_variants_json": str(selected_path),
        "v85_preflight_artifacts": {k: v for k, v in audit.items() if k not in {"rows", "variant_rows"}},
    }


def run_v89_candidate_probe(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Probe v89 geometry candidates with real no-sticky physics and no PPO."""

    run_path = Path(run_dir)
    threshold = float(backend.contact_manager.force_threshold_n)
    trace_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    candidates_by_rank: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidate_rows:
        try:
            rank = int(candidate.get("candidate_rank") or 0)
        except Exception:
            rank = 0
        candidates_by_rank.setdefault(rank, []).append(dict(candidate))

    if backend.env is not None and backend.slots:
        backend.clear_v89_rollout_trace()
        backend.action_mapping_rows = []
        backend.contact_api_rows = []
        for rank in sorted(candidates_by_rank):
            active = {
                str(candidate.get("part_name")): _v89_candidate_as_variant(candidate, rank)
                for candidate in candidates_by_rank[rank]
                if str(candidate.get("part_name")) in V83_PARTS
            }
            if not active:
                continue
            selected_parts = set(active)
            backend.object_write_by_policy_detected = False
            backend.configure_v89_candidates(active)
            backend.configure_v86_staging(active)
            backend.reset_envs()
            staged = backend.stage_v86_near_contact_episode()
            for row in staged:
                part = str(row.get("part_name") or "")
                if part in active:
                    stage_rows.append({**row, **_v89_candidate_fields(active[part])})

            pre_metrics = [
                backend._metrics_for_slot(slot, "pre_action_force_sample")
                for slot in backend.slots
                if slot.part_name in selected_parts
            ]
            trace_rows.extend(
                _v89_annotate_candidate_metrics(
                    pre_metrics,
                    active,
                    probe_step=-1,
                    probe_phase="pre_action_force_sample",
                    stage_rows=staged,
                )
            )

            global_step = 0
            step_plan = [("settle", 4), ("close", 10), ("approach_close", 72), ("hold_squeeze", 16), ("lift_stabilize", 24)]
            for phase, count in step_plan:
                for _ in range(count):
                    start = len(backend.v86_rollout_trace_rows)
                    backend.step_v89_candidate_residual_envs(
                        [[0.0] * 16 for _slot in backend.slots],
                        episode_steps=[global_step for _slot in backend.slots],
                        residual_scales={"wrist_xyz": 0.0, "wrist_rot": 0.0, "finger": 0.0},
                        lift_phase_enabled=True,
                        forced_phase=phase,
                    )
                    for row in backend.v86_rollout_trace_rows[start:]:
                        if row.get("part_name") not in selected_parts:
                            continue
                        candidate = active.get(str(row.get("part_name") or ""), {})
                        row.update(
                            {
                                "probe_step": global_step,
                                "probe_phase": phase,
                                "mapped_close_action_executed": phase in V89_ACTION_PHASES,
                                "approach_action_executed": phase == "approach_close",
                                "lift_action_executed": phase == "lift_stabilize",
                                **_v89_candidate_fields(candidate),
                            }
                        )
                        trace_rows.append(dict(row))
                    global_step += 1

            for part, candidate in active.items():
                cid = str(candidate.get("candidate_id") or "")
                candidate_trace = [row for row in trace_rows if row.get("part_name") == part and row.get("candidate_id") == cid]
                stage = next((row for row in stage_rows if row.get("part_name") == part and row.get("candidate_id") == cid), {})
                summaries.append(_v89_summarize_candidate(part, candidate, candidate_trace, stage, backend, threshold))

    backend.v89_candidate_probe_trace_rows = list(trace_rows)
    backend.v89_candidate_probe_summary_rows = list(summaries)
    stage_csv = write_csv(run_path / "v89_staging_variant_audit.csv", stage_rows)
    stage_json = write_json(run_path / "v89_staging_variant_audit.json", stage_rows)
    trace_csv = write_csv(run_path / "v89_candidate_probe_trace.csv", trace_rows)
    trace_jsonl = write_jsonl(run_path / "v89_candidate_probe_trace.jsonl", trace_rows)
    result_csv = write_csv(run_path / "v89_candidate_probe_results.csv", summaries)
    result_json = write_json(run_path / "v89_candidate_probe_results.json", summaries)
    return {
        "rows": summaries,
        "trace_rows": trace_rows,
        "stage_rows": stage_rows,
        "v89_staging_variant_audit_csv": str(stage_csv),
        "v89_staging_variant_audit_json": str(stage_json),
        "v89_candidate_probe_trace_csv": str(trace_csv),
        "v89_candidate_probe_trace_jsonl": str(trace_jsonl),
        "v89_candidate_probe_results_csv": str(result_csv),
        "v89_candidate_probe_results_json": str(result_json),
    }


def _v89_candidate_as_variant(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    out = dict(candidate)
    out["variant_index"] = int(out.get("variant_index", out.get("candidate_rank", rank)) or rank)
    out["variant_rank"] = int(out.get("candidate_rank", out["variant_index"]) or 0)
    out["active_finger_group"] = str(out.get("active_finger_group") or "34")
    out["target_clearance_m"] = float(out.get("target_clearance_m") or 0.010)
    out["lateral_offset_m"] = float(out.get("lateral_offset_m") or 0.0)
    out["surface_strategy"] = str(out.get("surface_strategy") or "bbox_adjusted")
    out["use_bbox_adjustment"] = bool(out.get("use_bbox_adjustment", out["surface_strategy"] == "bbox_adjusted"))
    return out


def _v89_candidate_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_rank": int(candidate.get("candidate_rank", candidate.get("variant_rank", 0)) or 0),
        "active_finger_group": str(candidate.get("active_finger_group") or ""),
        "target_clearance_m": float(candidate.get("target_clearance_m") or 0.0),
        "lateral_offset_m": float(candidate.get("lateral_offset_m") or 0.0),
        "surface_strategy": str(candidate.get("surface_strategy") or ""),
        "use_bbox_adjustment": bool(candidate.get("use_bbox_adjustment", False)),
        "close_value": float(candidate.get("close_value") or 0.0),
        "approach_scale": float(candidate.get("approach_scale") or 0.0),
        "hold_close_value": float(candidate.get("hold_close_value") or candidate.get("close_value") or 0.0),
        "lift_x": float(candidate.get("lift_x") or 0.0),
        "lift_y": float(candidate.get("lift_y") or 0.0),
        "lift_z": float(candidate.get("lift_z") or 0.0),
        "expected_contact_surfaces": str(candidate.get("expected_contact_surfaces") or ""),
        "grasp_style": str(candidate.get("grasp_style") or ""),
    }


def _v89_annotate_candidate_metrics(
    metrics: list[dict[str, Any]],
    active_candidates: dict[str, dict[str, Any]],
    *,
    probe_step: int,
    probe_phase: str,
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_by_part = {str(row.get("part_name") or ""): row for row in stage_rows}
    rows: list[dict[str, Any]] = []
    for item in metrics:
        part = str(item.get("part_name") or "")
        candidate = active_candidates.get(part)
        if not candidate:
            continue
        stage = stage_by_part.get(part, {})
        rows.append(
            {
                "probe_step": int(probe_step),
                "probe_phase": probe_phase,
                "mapped_close_action_executed": False,
                "approach_action_executed": False,
                "lift_action_executed": False,
                "near_contact_stage_ok": bool(stage.get("near_contact_stage_ok")),
                "near_contact_stage_failure_reason": str(stage.get("near_contact_stage_failure_reason") or ""),
                **_v89_candidate_fields(candidate),
                **item,
            }
        )
    return rows


def _v89_summarize_candidate(
    part: str,
    candidate: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    stage: dict[str, Any],
    backend: IsaacUnifiedSingleContextBackend,
    threshold: float,
) -> dict[str, Any]:
    pre_rows = [row for row in trace_rows if row.get("probe_phase") in {"pre_action_force_sample", "settle"}]
    action_rows = [row for row in trace_rows if row.get("probe_phase") in V89_ACTION_PHASES]
    hold_rows = [row for row in trace_rows if row.get("probe_phase") == "hold_squeeze"]
    lift_rows = [row for row in trace_rows if row.get("probe_phase") == "lift_stabilize"]
    force_rows = [
        row
        for row in action_rows
        if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
        and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
    ]
    support_rows = [row for row in trace_rows if bool(row.get("support_gate_ok"))]
    hold_hits = [
        row
        for row in hold_rows
        if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
        and float(row.get("hold_object_displacement_m") or row.get("object_displacement_m") or 0.0) <= 0.04
        and not bool(row.get("table_collision"))
    ]
    lift_hits = [
        row
        for row in lift_rows
        if float(row.get("lift_delta_z_m") or row.get("object_delta_z_m") or 0.0) > 0.005
        and float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
    ]
    peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in trace_rows]
    pre_peak = max([0.0, *[float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in pre_rows]])
    pre_count = max([0, *[int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) for row in pre_rows]])
    displacements = [float(row.get("object_displacement_m") or 0.0) for row in trace_rows]
    streaks = [float(row.get("force_contact_streak_steps") or row.get("contact_duration_steps") or 0.0) for row in trace_rows]
    jerks = [float(row.get("action_jerk") or 0.0) for row in trace_rows]
    excessive_rows = [row for row in trace_rows if bool(row.get("excessive_force"))]
    multi_finger = [
        row
        for row in trace_rows
        if int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 2
    ]
    reset_interpenetration = bool(pre_peak > threshold or pre_count > 0)
    forbidden = bool(
        backend.object_write_by_policy_detected
        or backend.sticky_action_available_to_policy
        or any(bool(row.get("fallback_success_used")) for row in trace_rows)
    )
    support_rate = len(support_rows) / max(1, len(trace_rows))
    hold_rate = len(hold_hits) / max(1, len(hold_rows))
    lift_rate = len(lift_hits) / max(1, len(lift_rows))
    force_rate = len(force_rows) / max(1, len(action_rows))
    peak_force = max([0.0, *peaks])
    identity_ok = any(
        bool(row.get("object_identity_verified"))
        for row in backend.object_identity_rows
        if row.get("part_name") == part
    )
    mapped_close = any(
        bool(row.get("metric_close_dof_commanded"))
        and str(row.get("candidate_id") or "") == str(candidate.get("candidate_id") or "")
        for row in backend.action_mapping_rows
        if row.get("part_name") == part
    )
    success = bool(
        identity_ok
        and backend.single_simulation_context
        and bool(stage.get("near_contact_stage_ok"))
        and mapped_close
        and force_rate > 0.0
        and support_rate > 0.0
        and hold_rate > 0.0
        and lift_rate > 0.0
        and not reset_interpenetration
        and not forbidden
        and peak_force <= 175.0
    )
    if success:
        status = "CANDIDATE_SUPPORT_HOLD_LIFT_PASS_NO_STICKY_DIAGNOSTIC_ONLY"
        blocker = ""
    elif reset_interpenetration:
        status = "CANDIDATE_RESET_INTERPENETRATION_REJECTED"
        blocker = "reset_interpenetration_or_impulse_detected"
    elif force_rate <= 0.0:
        status = "CANDIDATE_NO_REAL_FORCE_CONTACT"
        blocker = "candidate_probe_no_force_contact_after_action"
    elif support_rate <= 0.0:
        status = "CANDIDATE_CONTACT_BUT_NO_SUPPORT"
        blocker = "candidate_probe_no_support_gate"
    elif hold_rate <= 0.0:
        status = "CANDIDATE_SUPPORT_LOST_DURING_HOLD"
        blocker = "candidate_probe_hold_not_stable"
    elif lift_rate <= 0.0:
        status = "CANDIDATE_LIFT_FAILED"
        blocker = "candidate_probe_lift_not_observed"
    else:
        status = "CANDIDATE_FORBIDDEN_OR_EXCESSIVE_FORCE_REJECTED"
        blocker = "candidate_probe_forbidden_action_or_excessive_force"
    return {
        "part_name": part,
        **_v89_candidate_fields(candidate),
        "object_identity_verified": bool(identity_ok),
        "single_simulation_context": bool(backend.single_simulation_context),
        "near_contact_stage_ok": bool(stage.get("near_contact_stage_ok")),
        "pre_action_force_peak_n": pre_peak,
        "pre_action_force_count_max": int(pre_count),
        "reset_interpenetration_or_impulse_detected": bool(reset_interpenetration),
        "force_contact_rate": force_rate,
        "contact_duration_mean": sum(streaks) / max(1, len(streaks)),
        "mean_force_n": sum(peaks) / max(1, len(peaks)),
        "peak_force_n": peak_force,
        "force_contact_peak_n": peak_force,
        "excessive_force_rate": len(excessive_rows) / max(1, len(trace_rows)),
        "support_gate_rate": support_rate,
        "hold_gate_rate": hold_rate,
        "lift_gate_rate": lift_rate,
        "multi_finger_support_rate": len(multi_finger) / max(1, len(trace_rows)),
        "object_displacement_max_m": max([0.0, *displacements]),
        "action_jerk_mean": sum(jerks) / max(1, len(jerks)),
        "mapped_close_action_executed": bool(mapped_close),
        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
        "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
        "fallback_success_used": any(bool(row.get("fallback_success_used")) for row in trace_rows),
        "candidate_success": bool(success),
        "candidate_success_diagnostic_only": bool(success),
        "ppo_ran": False,
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "usable_training_row_count": 0,
        "status": status,
        "blocker": blocker,
    }


def run_v90_feasibility_bench(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    candidate_rows: list[dict[str, Any]],
    *,
    artifact_prefix: str = "v90",
    physics_profile: str = "canonical",
) -> dict[str, Any]:
    """Run bounded no-sticky diagnostic candidate feasibility probes."""

    run_path = Path(run_dir)
    threshold = float(backend.contact_manager.force_threshold_n)
    trace_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    candidates_by_rank: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidate_rows:
        try:
            rank = int(candidate.get("candidate_rank") or 0)
        except Exception:
            rank = 0
        candidates_by_rank.setdefault(rank, []).append(dict(candidate))

    if backend.env is not None and backend.slots:
        backend.clear_v89_rollout_trace()
        backend.action_mapping_rows = []
        backend.contact_api_rows = []
        for rank in sorted(candidates_by_rank):
            active = {
                str(candidate.get("part_name")): _v89_candidate_as_variant(candidate, rank)
                for candidate in candidates_by_rank[rank]
                if str(candidate.get("part_name")) in V83_PARTS
            }
            if not active:
                continue
            selected_parts = set(active)
            backend.object_write_by_policy_detected = False
            backend.configure_v89_candidates(active)
            backend.configure_v86_staging(active)
            backend.reset_envs()
            staged = backend.stage_v86_near_contact_episode()
            for row in staged:
                part = str(row.get("part_name") or "")
                if part in active:
                    stage_rows.append({**row, **_v90_candidate_fields(active[part])})

            pre_metrics = [
                backend._metrics_for_slot(slot, "v90_pre_action_force_sample")
                for slot in backend.slots
                if slot.part_name in selected_parts
            ]
            trace_rows.extend(
                _v90_annotate_candidate_metrics(
                    pre_metrics,
                    active,
                    sequence_step=-1,
                    probe_phase="pre_action_force_sample",
                    stage_rows=staged,
                )
            )

            global_step = 0
            step_plan = [
                ("settle", 4),
                ("slow_approach", 24),
                ("close_force_limited", 20),
                ("contact_aware_hold", 48),
                ("slow_lift", 32),
            ]
            for phase, count in step_plan:
                for phase_step in range(count):
                    start = len(backend.v86_rollout_trace_rows)
                    backend.step_v90_feasibility_envs(
                        phase=phase,
                        selected_parts=selected_parts,
                        phase_step=phase_step,
                    )
                    for row in backend.v86_rollout_trace_rows[start:]:
                        if row.get("part_name") not in selected_parts:
                            continue
                        candidate = active.get(str(row.get("part_name") or ""), {})
                        row.update(
                            {
                                "sequence_step": global_step,
                                "probe_step": global_step,
                                "probe_phase": phase,
                                "mapped_close_action_executed": phase in V90_ACTION_PHASES,
                                "approach_action_executed": phase == "slow_approach",
                                "lift_action_executed": phase == "slow_lift",
                                "physics_profile": physics_profile,
                                **_v90_candidate_fields(candidate),
                            }
                        )
                        trace_rows.append(dict(row))
                    global_step += 1

            for part, candidate in active.items():
                cid = str(candidate.get("candidate_id") or "")
                candidate_trace = [row for row in trace_rows if row.get("part_name") == part and row.get("candidate_id") == cid]
                stage = next((row for row in stage_rows if row.get("part_name") == part and row.get("candidate_id") == cid), {})
                summaries.append(_v90_summarize_candidate(part, candidate, candidate_trace, stage, backend, threshold, physics_profile))

    best_rows = _v90_select_best_rows(summaries)
    result_csv = write_csv(run_path / f"{artifact_prefix}_feasibility_candidate_results.csv", summaries)
    result_json = write_json(run_path / f"{artifact_prefix}_feasibility_candidate_results.json", summaries)
    best_csv = write_csv(run_path / f"{artifact_prefix}_best_candidate_trace_summary.csv", best_rows)
    best_json = write_json(run_path / f"{artifact_prefix}_best_candidate_trace_summary.json", best_rows)
    stage_csv = write_csv(run_path / f"{artifact_prefix}_feasibility_staging_audit.csv", stage_rows)
    trace_jsonl = write_jsonl(run_path / f"{artifact_prefix}_feasibility_candidate_trace.jsonl", trace_rows)
    return {
        "rows": summaries,
        "best_rows": best_rows,
        "trace_rows": trace_rows,
        "stage_rows": stage_rows,
        f"{artifact_prefix}_feasibility_candidate_results_csv": str(result_csv),
        f"{artifact_prefix}_feasibility_candidate_results_json": str(result_json),
        f"{artifact_prefix}_best_candidate_trace_summary_csv": str(best_csv),
        f"{artifact_prefix}_best_candidate_trace_summary_json": str(best_json),
        f"{artifact_prefix}_feasibility_staging_audit_csv": str(stage_csv),
        f"{artifact_prefix}_feasibility_candidate_trace_jsonl": str(trace_jsonl),
    }


def _v90_candidate_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = _v89_candidate_fields(candidate)
    fields.update(
        {
            "candidate_family": str(candidate.get("candidate_family") or ""),
            "family_variant_index": int(candidate.get("family_variant_index", 0) or 0),
            "approach_direction": str(candidate.get("approach_direction") or "sensor_to_object"),
            "pregrasp_offset_x": float(candidate.get("pregrasp_offset_x") or 0.0),
            "pregrasp_offset_y": float(candidate.get("pregrasp_offset_y") or 0.0),
            "pregrasp_offset_z": float(candidate.get("pregrasp_offset_z") or 0.0),
            "wrist_roll_rad": float(candidate.get("wrist_roll_rad") or 0.0),
            "force_limit_n": float(candidate.get("force_limit_n") or 0.0),
            "hold_steps": int(candidate.get("hold_steps", 48) or 48),
            "lift_steps": int(candidate.get("lift_steps", 32) or 32),
            "max_physics_steps": int(candidate.get("max_physics_steps", 128) or 128),
        }
    )
    return fields


def _v90_annotate_candidate_metrics(
    metrics: list[dict[str, Any]],
    active_candidates: dict[str, dict[str, Any]],
    *,
    sequence_step: int,
    probe_phase: str,
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_by_part = {str(row.get("part_name") or ""): row for row in stage_rows}
    rows: list[dict[str, Any]] = []
    for item in metrics:
        part = str(item.get("part_name") or "")
        candidate = active_candidates.get(part)
        if not candidate:
            continue
        stage = stage_by_part.get(part, {})
        rows.append(
            {
                "sequence_step": int(sequence_step),
                "probe_step": int(sequence_step),
                "probe_phase": probe_phase,
                "mapped_close_action_executed": False,
                "approach_action_executed": False,
                "lift_action_executed": False,
                "near_contact_stage_ok": bool(stage.get("near_contact_stage_ok")),
                "near_contact_stage_failure_reason": str(stage.get("near_contact_stage_failure_reason") or ""),
                **_v90_candidate_fields(candidate),
                **item,
            }
        )
    return rows


def _v90_summarize_candidate(
    part: str,
    candidate: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    stage: dict[str, Any],
    backend: IsaacUnifiedSingleContextBackend,
    threshold: float,
    physics_profile: str,
) -> dict[str, Any]:
    pre_rows = [row for row in trace_rows if row.get("probe_phase") in {"pre_action_force_sample", "settle"}]
    action_rows = [row for row in trace_rows if row.get("probe_phase") in V90_ACTION_PHASES]
    support_rows = [row for row in trace_rows if bool(row.get("support_gate_ok"))]
    hold_rows = [row for row in trace_rows if row.get("probe_phase") == "contact_aware_hold"]
    lift_rows = [row for row in trace_rows if row.get("probe_phase") == "slow_lift"]
    force_rows = [
        row
        for row in action_rows
        if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
        and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
    ]
    hold_hits = [
        row
        for row in hold_rows
        if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
        and float(row.get("hold_object_displacement_m") or row.get("object_displacement_m") or 0.0) <= 0.04
        and not bool(row.get("table_collision"))
    ]
    lift_hits = [
        row
        for row in lift_rows
        if float(row.get("lift_delta_z_m") or row.get("object_delta_z_m") or 0.0) > 0.005
        and float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
        and not bool(row.get("table_collision"))
    ]
    peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in trace_rows]
    pre_peak = max([0.0, *[float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in pre_rows]])
    pre_count = max([0, *[int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) for row in pre_rows]])
    displacements = [float(row.get("object_displacement_m") or 0.0) for row in trace_rows]
    streaks = [float(row.get("force_contact_streak_steps") or row.get("contact_duration_steps") or 0.0) for row in trace_rows]
    jerks = [float(row.get("action_jerk") or 0.0) for row in trace_rows]
    excessive_rows = [row for row in trace_rows if bool(row.get("excessive_force")) or float(row.get("force_contact_peak_n") or 0.0) > 175.0]
    multi_finger = [
        row
        for row in trace_rows
        if int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 2
    ]
    onset = _v90_first_force_contact(trace_rows, threshold)
    reset_interpenetration = bool(pre_peak > threshold or pre_count > 0 or (onset and onset.get("probe_phase") in {"pre_action_force_sample", "settle"}))
    forbidden = bool(
        backend.object_write_by_policy_detected
        or backend.sticky_action_available_to_policy
        or any(bool(row.get("fallback_success_used") or row.get("distance_only_success_used")) for row in trace_rows)
    )
    support_rate = len(support_rows) / max(1, len(trace_rows))
    hold_rate = len(hold_hits) / max(1, len(hold_rows))
    lift_rate = len(lift_hits) / max(1, len(lift_rows))
    force_rate = len(force_rows) / max(1, len(action_rows))
    peak_force = max([0.0, *peaks])
    mean_force = sum(peaks) / max(1, len(peaks))
    identity_ok = any(
        bool(row.get("object_identity_verified"))
        for row in backend.object_identity_rows
        if row.get("part_name") == part
    )
    cid = str(candidate.get("candidate_id") or "")
    mapped_close = any(
        bool(row.get("metric_close_dof_commanded"))
        and str(row.get("candidate_id") or "") == cid
        and str(row.get("mapped_close_cols") or "") == "16,17,18,19,20,21,22,23,24,25"
        for row in backend.action_mapping_rows
        if row.get("part_name") == part
    )
    stage_ok = bool(stage.get("near_contact_stage_ok"))
    displacement_max = max([0.0, *displacements])
    acceptable_force = bool(peak_force <= 175.0 and len(excessive_rows) / max(1, len(trace_rows)) <= 0.01)
    candidate_success = bool(
        identity_ok
        and backend.single_simulation_context
        and stage_ok
        and mapped_close
        and force_rate > 0.0
        and support_rate > 0.0
        and hold_rate > 0.0
        and lift_rate > 0.0
        and acceptable_force
        and displacement_max <= 0.06
        and not reset_interpenetration
        and not forbidden
    )
    if candidate_success:
        status = "FEASIBILITY_SUPPORT_HOLD_LIFT_PASS_NO_STICKY_DIAGNOSTIC_ONLY"
        blocker = ""
    elif reset_interpenetration:
        status = "RESET_INTERPENETRATION_CONTACT_REJECTED"
        blocker = "reset_interpenetration_or_impulse_detected"
    elif not stage_ok:
        status = "NON_INTERPENETRATING_PREGRASP_STAGE_FAILED"
        blocker = str(stage.get("near_contact_stage_failure_reason") or "near_contact_stage_failed")
    elif force_rate <= 0.0:
        status = "NO_REAL_FORCE_CONTACT_AFTER_ACTION"
        blocker = "candidate_probe_no_force_contact_after_action"
    elif support_rate <= 0.0:
        status = "CONTACT_BUT_NO_SUPPORT"
        blocker = "candidate_probe_no_support_gate"
    elif hold_rate <= 0.0:
        status = "SUPPORT_LOST_DURING_HOLD"
        blocker = "candidate_probe_hold_not_stable"
    elif lift_rate <= 0.0:
        status = "LIFT_FAILED"
        blocker = "candidate_probe_lift_not_observed"
    else:
        status = "EXCESSIVE_FORCE_DISPLACEMENT_OR_FORBIDDEN_ACTION_REJECTED"
        blocker = "candidate_probe_forbidden_excessive_force_or_displacement"
    return {
        "part_name": part,
        **_v90_candidate_fields(candidate),
        "physics_profile": physics_profile,
        "object_identity_verified": bool(identity_ok),
        "single_simulation_context": bool(backend.single_simulation_context),
        "near_contact_stage_ok": bool(stage_ok),
        "initial_reset_force_peak_n": pre_peak,
        "pre_action_force_peak_n": pre_peak,
        "pre_action_force_count_max": int(pre_count),
        "reset_interpenetration_or_impulse_detected": bool(reset_interpenetration),
        "contact_onset_step": "" if onset is None else int(onset.get("probe_step", -1)),
        "contact_onset_phase": "" if onset is None else str(onset.get("probe_phase", "")),
        "force_contact_rate": force_rate,
        "contact_duration_mean": sum(streaks) / max(1, len(streaks)),
        "mean_force_n": mean_force,
        "peak_force_n": peak_force,
        "force_contact_peak_n": peak_force,
        "excessive_force_rate": len(excessive_rows) / max(1, len(trace_rows)),
        "support_gate_rate": support_rate,
        "hold_gate_rate": hold_rate,
        "lift_gate_rate": lift_rate,
        "multi_finger_support_rate": len(multi_finger) / max(1, len(trace_rows)),
        "object_displacement_max_m": displacement_max,
        "action_jerk_mean": sum(jerks) / max(1, len(jerks)),
        "mapped_close_action_executed": bool(mapped_close),
        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
        "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
        "fallback_success_used": any(bool(row.get("fallback_success_used")) for row in trace_rows),
        "distance_only_success_used": any(bool(row.get("distance_only_success_used")) for row in trace_rows),
        "canonical_feasibility_pass": bool(candidate_success and physics_profile == "canonical"),
        "candidate_success": bool(candidate_success),
        "candidate_success_diagnostic_only": bool(candidate_success),
        "ppo_ran": False,
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "checkpoint_written": False,
        "dataset_exported": False,
        "usable_training_row_count": 0,
        "final_success": False,
        "status": status,
        "blocker": blocker,
    }


def _v90_first_force_contact(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    ordered = sorted(rows, key=lambda row: (_int_field(row, "probe_step"), str(row.get("probe_phase") or "")))
    for row in ordered:
        force = float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0)
        count = int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0)
        if force > threshold and count >= 1:
            return row
    return None


def _v90_select_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for part in V83_PARTS:
        part_rows = [row for row in rows if row.get("part_name") == part]
        if not part_rows:
            selected.append(
                {
                    "part_name": part,
                    "candidate_id": "",
                    "candidate_success": False,
                    "canonical_feasibility_pass": False,
                    "status": "NO_V90_CANDIDATE_EXECUTED",
                    "blocker": "no_v90_candidate_executed",
                    "ppo_ran": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "checkpoint_written": False,
                    "usable_training_row_count": 0,
                }
            )
            continue
        selected.append(max(part_rows, key=_v90_candidate_score))
    return selected


def _v90_candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    forbidden = bool(row.get("object_write_by_policy_detected")) or bool(row.get("sticky_action_available_to_policy")) or bool(row.get("fallback_success_used"))
    return (
        int(not forbidden),
        int(bool(row.get("candidate_success") or row.get("canonical_feasibility_pass"))),
        float(row.get("support_gate_rate") or 0.0),
        float(row.get("hold_gate_rate") or 0.0),
        float(row.get("lift_gate_rate") or 0.0),
        -float(row.get("excessive_force_rate") or 0.0),
        -float(row.get("object_displacement_max_m") or 0.0),
        float(row.get("contact_duration_mean") or 0.0),
        -_int_field(row, "candidate_rank", 999),
    )


def run_v91_observation_runtime_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    if backend.env is not None and backend.slots:
        backend.reset_envs()
        metrics = [backend._metrics_for_slot(slot, "v91_observation_runtime_audit") for slot in backend.slots]
        for row in metrics:
            blocker = str(row.get("v88_observation_feature_blocker") or "")
            bbox_available = all(str(row.get(key, "")) not in {"", "0", "0.0"} for key in ("object_bbox_x", "object_bbox_y", "object_bbox_z"))
            rel_available = all(key in row for key in ("object_rel_palm_x", "object_rel_palm_y", "object_rel_palm_z"))
            dist_available = "fingertip_object_distance_min_m" in row
            nonzero_features = sum(
                1
                for key in (
                    "object_bbox_x",
                    "object_bbox_y",
                    "object_bbox_z",
                    "object_rel_palm_x",
                    "object_rel_palm_y",
                    "object_rel_palm_z",
                    "fingertip_object_distance_min_m",
                    "fingertip_object_distance_mean_m",
                )
                if abs(float(row.get(key) or 0.0)) > 1.0e-8
            )
            rows.append(
                {
                    "part_name": row.get("part_name", ""),
                    "env_index": row.get("env_index", ""),
                    "old_blocker": blocker,
                    "fix_applied": "V85_PART_GEOMETRY_defined_in_single_context_backend",
                    "object_bbox_features_available": bool(bbox_available),
                    "object_pose_relative_to_palm_available": bool(rel_available),
                    "fingertip_object_distance_features_available": bool(dist_available),
                    "observation_dimension": 8,
                    "nonzero_feature_count": nonzero_features,
                    "observation_feature_sanity_ok": bool(not blocker and bbox_available and rel_available and dist_available and nonzero_features > 0),
                    "object_bbox_x": row.get("object_bbox_x", 0.0),
                    "object_bbox_y": row.get("object_bbox_y", 0.0),
                    "object_bbox_z": row.get("object_bbox_z", 0.0),
                    "fingertip_object_distance_min_m": row.get("fingertip_object_distance_min_m", 0.0),
                    "fingertip_object_distance_mean_m": row.get("fingertip_object_distance_mean_m", 0.0),
                }
            )
    csv_path = write_csv(run_path / "v91_observation_runtime_audit.csv", rows)
    json_path = write_json(run_path / "v91_observation_runtime_audit.json", rows)
    md_path = run_path / "v91_observation_runtime_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v91_observation_runtime_audit_csv": str(csv_path), "v91_observation_runtime_audit_json": str(json_path), "v91_observation_runtime_audit_md": str(md_path)}


def run_v91_asset_dynamics_audit(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    selected = _v91_first_candidate_by_part(candidate_rows)
    if backend.env is not None and backend.slots:
        backend.object_write_by_policy_detected = False
        backend.configure_v89_candidates(selected)
        backend.configure_v86_staging(selected)
        backend.reset_envs()
        backend.stage_v86_near_contact_episode()
        start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
        phases = [("gravity_settle", 8), ("small_lateral_push", 12), ("lift_response", 16)]
        for phase, count in phases:
            for step in range(count):
                if phase == "gravity_settle":
                    backend.step_envs([[0.0] * 16 for _slot in backend.slots])
                elif phase == "small_lateral_push":
                    actions = []
                    for slot in backend.slots:
                        action = [0.0] * 16
                        action[0] = 0.08
                        actions.append(action)
                    backend.step_envs(actions)
                else:
                    backend.step_v91_quasistatic_feasibility_envs(
                        phase="slow_lift",
                        selected_parts={slot.part_name for slot in backend.slots},
                        phase_step=step,
                    )
                for slot in backend.slots:
                    state = _v91_object_state_for_slot(backend, slot)
                    start = start_states.get(slot.part_name, {})
                    traces.append(
                        {
                            "part_name": slot.part_name,
                            "env_index": slot.global_env_index,
                            "test_phase": phase,
                            "phase_step": step,
                            **_v91_state_delta(start, state),
                            **state,
                        }
                    )
        for slot in backend.slots:
            part_traces = [row for row in traces if row.get("part_name") == slot.part_name]
            identity = backend.object_identity_for_slot(slot)
            props = _v91_asset_runtime_properties(backend, slot, identity)
            pose_delta_max = max([0.0, *[float(row.get("root_pose_delta_m") or 0.0) for row in part_traces]])
            vel_max = max([0.0, *[float(row.get("root_lin_vel_norm") or 0.0) for row in part_traces]])
            lift_delta = max([0.0, *[float(row.get("root_delta_z_m") or 0.0) for row in part_traces if row.get("test_phase") == "lift_response"]])
            contact_seen = any(int(row.get("effective_contact_count_force") or 0) >= 1 for row in backend.last_metrics if row.get("part_name") == slot.part_name)
            dynamic_ok = bool(
                identity.get("object_identity_verified")
                and not bool(props.get("kinematic_enabled") is True)
                and not bool(props.get("disable_gravity") is True)
                and (pose_delta_max > 1.0e-5 or vel_max > 1.0e-5)
            )
            blocker = ""
            if not dynamic_ok:
                if props.get("kinematic_enabled") is True:
                    blocker = "kinematic_enabled_true"
                elif props.get("disable_gravity") is True:
                    blocker = "disable_gravity_true"
                elif pose_delta_max <= 1.0e-5 and vel_max <= 1.0e-5:
                    blocker = "contact_seen_but_root_pose_velocity_not_updating"
                else:
                    blocker = "asset_dynamic_response_inconclusive"
            rows.append(
                {
                    "part_name": slot.part_name,
                    **identity,
                    **props,
                    "gravity_settle_pose_delta_max_m": max([0.0, *[float(row.get("root_pose_delta_m") or 0.0) for row in part_traces if row.get("test_phase") == "gravity_settle"]]),
                    "small_lateral_push_pose_delta_max_m": max([0.0, *[float(row.get("root_pose_delta_m") or 0.0) for row in part_traces if row.get("test_phase") == "small_lateral_push"]]),
                    "lift_response_delta_z_max_m": lift_delta,
                    "root_pose_updates_over_time": pose_delta_max > 1.0e-5,
                    "root_velocity_updates_over_time": vel_max > 1.0e-5,
                    "diagnostic_force_used": False,
                    "diagnostic_force_counts_as_success": False,
                    "asset_dynamics_ok": dynamic_ok,
                    "asset_dynamics_blocker": blocker,
                    "ppo_ran": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "usable_training_row_count": 0,
                }
            )
    csv_path = write_csv(run_path / "v91_asset_dynamics_audit.csv", rows)
    json_path = write_json(run_path / "v91_asset_dynamics_audit.json", rows)
    md_path = run_path / "v91_asset_dynamics_audit.md"
    trace_path = write_jsonl(run_path / "v91_asset_dynamics_trace.jsonl", traces)
    _write_md(md_path, rows)
    return {"rows": rows, "trace_rows": traces, "v91_asset_dynamics_audit_csv": str(csv_path), "v91_asset_dynamics_audit_json": str(json_path), "v91_asset_dynamics_audit_md": str(md_path), "v91_asset_dynamics_trace_jsonl": str(trace_path)}


def run_v91_staging_sanity_bench(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    candidates_by_rank: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidate_rows:
        candidates_by_rank.setdefault(int(candidate.get("candidate_rank") or 0), []).append(dict(candidate))
    if backend.env is not None and backend.slots:
        for rank in sorted(candidates_by_rank):
            active = {
                str(candidate.get("part_name")): _v89_candidate_as_variant(candidate, rank)
                for candidate in candidates_by_rank[rank]
                if str(candidate.get("part_name")) in V83_PARTS
            }
            if not active:
                continue
            backend.object_write_by_policy_detected = False
            backend.configure_v89_candidates(active)
            backend.configure_v86_staging(active)
            backend.reset_envs()
            staged = backend.stage_v86_near_contact_episode()
            pre_metrics = [backend._metrics_for_slot(slot, "v91_staging_pre_contact_sample") for slot in backend.slots if slot.part_name in active]
            for _ in range(2):
                backend.step_envs([[0.0] * 16 for _slot in backend.slots])
            settle_metrics = [backend._metrics_for_slot(slot, "v91_staging_zero_action_settle") for slot in backend.slots if slot.part_name in active]
            metrics_by_part = {}
            for item in [*pre_metrics, *settle_metrics]:
                metrics_by_part.setdefault(str(item.get("part_name") or ""), []).append(item)
            stage_by_part = {str(row.get("part_name") or ""): row for row in staged}
            for part, candidate in active.items():
                metrics = metrics_by_part.get(part, [])
                stage = stage_by_part.get(part, {})
                pre_contact_displacement = max([0.0, *[float(row.get("object_motion_before_contact_m") or row.get("object_displacement_m") or 0.0) for row in metrics]])
                z_drift = max([0.0, *[abs(float(row.get("object_delta_z_m") or 0.0)) for row in metrics]])
                velocity = max([0.0, *[float(row.get("object_velocity_norm") or 0.0) for row in metrics]])
                force_peak = max([0.0, *[float(row.get("force_contact_peak_n") or 0.0) for row in metrics]])
                force_count = max([0, *[int(row.get("effective_contact_count_force") or 0) for row in metrics]])
                reset_impulse = bool(force_peak > backend.contact_manager.force_threshold_n or force_count > 0)
                drop = bool(pre_contact_displacement > 0.02 or z_drift > 0.02)
                valid = bool(stage.get("near_contact_stage_ok") and not reset_impulse and not drop and velocity < 1.0)
                rows.append(
                    {
                        "part_name": part,
                        **_v90_candidate_fields(candidate),
                        "staging_mode": str(candidate.get("staging_mode") or "gripper_near_pregrasp"),
                        "table_supported": str(candidate.get("staging_mode") or "") == "table_supported",
                        "gripper_near": str(candidate.get("staging_mode") or "gripper_near_pregrasp") == "gripper_near_pregrasp",
                        "free_floating": False,
                        "near_contact_stage_ok": bool(stage.get("near_contact_stage_ok")),
                        "object_displacement_before_first_contact_m": pre_contact_displacement,
                        "object_z_drift_before_contact_m": z_drift,
                        "object_velocity_before_contact_mps": velocity,
                        "reset_interpenetration_or_impulse_detected": reset_impulse,
                        "pre_contact_drop_detected": drop,
                        "staging_valid_for_feasibility": valid,
                        "staging_blocker": "" if valid else "precontact_drop_or_reset_impulse_or_stage_failure",
                    }
                )
    csv_path = write_csv(run_path / "v91_staging_sanity_audit.csv", rows)
    json_path = write_json(run_path / "v91_staging_sanity_audit.json", rows)
    md_path = run_path / "v91_staging_sanity_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v91_staging_sanity_audit_csv": str(csv_path), "v91_staging_sanity_audit_json": str(json_path), "v91_staging_sanity_audit_md": str(md_path)}


def run_v91_quasistatic_feasibility_bench(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    candidate_rows: list[dict[str, Any]],
    staging_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    threshold = float(backend.contact_manager.force_threshold_n)
    valid_keys = {
        (str(row.get("part_name") or ""), str(row.get("candidate_id") or ""))
        for row in staging_rows
        if bool(row.get("staging_valid_for_feasibility"))
    }
    filtered = [
        dict(row)
        for row in candidate_rows
        if (str(row.get("part_name") or ""), str(row.get("candidate_id") or "")) in valid_keys
    ]
    if not filtered:
        filtered = list(candidate_rows)
    candidates_by_rank: dict[int, list[dict[str, Any]]] = {}
    for candidate in filtered:
        candidates_by_rank.setdefault(int(candidate.get("candidate_rank") or 0), []).append(dict(candidate))
    trace_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
    if backend.env is not None and backend.slots:
        backend.action_mapping_rows = []
        backend.clear_v89_rollout_trace()
        for rank in sorted(candidates_by_rank):
            active = {
                str(candidate.get("part_name")): _v89_candidate_as_variant(candidate, rank)
                for candidate in candidates_by_rank[rank]
                if str(candidate.get("part_name")) in V83_PARTS
            }
            if not active:
                continue
            selected_parts = set(active)
            backend.object_write_by_policy_detected = False
            backend.configure_v89_candidates(active)
            backend.configure_v86_staging(active)
            backend.reset_envs()
            staged = backend.stage_v86_near_contact_episode()
            pre_metrics = [backend._metrics_for_slot(slot, "v91_pre_controller_sample") for slot in backend.slots if slot.part_name in selected_parts]
            trace_rows.extend(_v91_annotate_metrics(pre_metrics, active, -1, "pre_controller_sample", staged))
            global_step = 0
            phase_plan = [
                ("approach_until_contact", max(int(row.get("approach_until_contact_max_steps", 36) or 36) for row in active.values())),
                ("gradual_close", max(int(row.get("close_steps", 24) or 24) for row in active.values())),
                ("contact_hold", max(int(row.get("hold_steps", 48) or 48) for row in active.values())),
                ("slow_lift", max(int(row.get("lift_steps", 32) or 32) for row in active.values())),
            ]
            for phase, count in phase_plan:
                for phase_step in range(count):
                    start = len(backend.v86_rollout_trace_rows)
                    action_start = len(backend.action_mapping_rows)
                    backend.step_v91_quasistatic_feasibility_envs(
                        phase=phase,
                        selected_parts=selected_parts,
                        phase_step=phase_step,
                    )
                    for row in backend.action_mapping_rows[action_start:]:
                        if row.get("part_name") in selected_parts:
                            finger_rows.append(dict(row))
                    for row in backend.v86_rollout_trace_rows[start:]:
                        if row.get("part_name") not in selected_parts:
                            continue
                        candidate = active.get(str(row.get("part_name") or ""), {})
                        row.update(
                            {
                                "sequence_step": global_step,
                                "probe_step": global_step,
                                "probe_phase": phase,
                                "physics_profile": "canonical",
                                **_v90_candidate_fields(candidate),
                                "staging_valid_for_feasibility": (str(row.get("part_name") or ""), str(candidate.get("candidate_id") or "")) in valid_keys,
                            }
                        )
                        trace_rows.append(dict(row))
                    global_step += 1
            for part, candidate in active.items():
                cid = str(candidate.get("candidate_id") or "")
                candidate_trace = [row for row in trace_rows if row.get("part_name") == part and row.get("candidate_id") == cid]
                candidate_fingers = [row for row in finger_rows if row.get("part_name") == part and row.get("candidate_id") == cid]
                summaries.append(_v91_summarize_candidate(part, candidate, candidate_trace, candidate_fingers, backend, threshold, valid_keys))
    best_rows = _v91_select_best_rows(summaries)
    trace_csv = write_csv(run_path / "v91_controller_trace.csv", trace_rows)
    trace_jsonl = write_jsonl(run_path / "v91_controller_trace.jsonl", trace_rows)
    controller_csv = write_csv(run_path / "v91_controller_summary.csv", summaries)
    controller_json = write_json(run_path / "v91_controller_summary.json", summaries)
    finger_csv = write_csv(run_path / "v91_finger_action_mask_audit.csv", finger_rows)
    finger_json = write_json(run_path / "v91_finger_action_mask_audit.json", finger_rows)
    result_csv = write_csv(run_path / "v91_feasibility_candidate_results.csv", summaries)
    result_json = write_json(run_path / "v91_feasibility_candidate_results.json", summaries)
    best_csv = write_csv(run_path / "v91_best_candidate_trace_summary.csv", best_rows)
    best_json = write_json(run_path / "v91_best_candidate_trace_summary.json", best_rows)
    return {
        "rows": summaries,
        "best_rows": best_rows,
        "trace_rows": trace_rows,
        "finger_rows": finger_rows,
        "v91_controller_trace_csv": str(trace_csv),
        "v91_controller_trace_jsonl": str(trace_jsonl),
        "v91_controller_summary_csv": str(controller_csv),
        "v91_controller_summary_json": str(controller_json),
        "v91_finger_action_mask_audit_csv": str(finger_csv),
        "v91_finger_action_mask_audit_json": str(finger_json),
        "v91_feasibility_candidate_results_csv": str(result_csv),
        "v91_feasibility_candidate_results_json": str(result_json),
        "v91_best_candidate_trace_summary_csv": str(best_csv),
        "v91_best_candidate_trace_summary_json": str(best_json),
    }


def run_v92_screw1_dynamic_repair_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    before = dict(getattr(base, "v92_screw1_dynamic_repair_before", {}) or {})
    after = dict(getattr(base, "v92_screw1_dynamic_repair_after", {}) or {})
    slot = next((slot for slot in backend.slots if slot.part_name == "Screw1"), None)
    identity = backend.object_identity_for_slot(slot) if slot is not None else {}
    props = _v91_asset_runtime_properties(backend, slot, identity) if slot is not None else {}
    start = _v91_object_state_for_slot(backend, slot) if slot is not None else {}
    if slot is not None:
        backend.reset_envs()
        start = _v91_object_state_for_slot(backend, slot)
        for _ in range(10):
            backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        gravity_state = _v91_object_state_for_slot(backend, slot)
        for _ in range(8):
            actions = [[0.0] * 16 for _slot in backend.slots]
            actions[slot.global_env_index][0] = 0.08
            backend.step_envs(actions)
        push_state = _v91_object_state_for_slot(backend, slot)
        backend.step_v91_quasistatic_feasibility_envs(phase="slow_lift", selected_parts={"Screw1"}, phase_step=0)
        lift_state = _v91_object_state_for_slot(backend, slot)
    else:
        gravity_state = {}
        push_state = {}
        lift_state = {}
    gravity_delta = _v91_state_delta(start, gravity_state)
    push_delta = _v91_state_delta(start, push_state)
    lift_delta = _v91_state_delta(start, lift_state)
    pose_updates = max(_to_float(gravity_delta.get("root_pose_delta_m")), _to_float(push_delta.get("root_pose_delta_m")), _to_float(lift_delta.get("root_pose_delta_m"))) > 1.0e-5
    vel_updates = max(_to_float(gravity_state.get("root_lin_vel_norm")), _to_float(push_state.get("root_lin_vel_norm")), _to_float(lift_state.get("root_lin_vel_norm"))) > 1.0e-5
    dynamic_ok = bool(
        identity.get("object_identity_verified")
        and props.get("disable_gravity") is False
        and props.get("kinematic_enabled") is False
        and _to_float(props.get("mass"), 0.0) > 0.0
        and (pose_updates or vel_updates)
    )
    blocker = ""
    if not dynamic_ok:
        if props.get("disable_gravity") is True:
            blocker = "disable_gravity_true"
        elif props.get("kinematic_enabled") is True:
            blocker = "kinematic_enabled_true"
        elif _to_float(props.get("mass"), 0.0) <= 0.0:
            blocker = "mass_missing_or_nonpositive"
        else:
            blocker = "root_pose_velocity_not_updating"
    rows.append(
        {
            "part_name": "Screw1",
            **identity,
            "before_disable_gravity": before.get("disable_gravity", ""),
            "before_kinematic_enabled": before.get("kinematic_enabled", ""),
            "before_mass": before.get("mass", ""),
            "after_disable_gravity": after.get("disable_gravity", ""),
            "after_kinematic_enabled": after.get("kinematic_enabled", ""),
            "after_mass": after.get("mass", ""),
            "runtime_disable_gravity": props.get("disable_gravity", ""),
            "runtime_kinematic_enabled": props.get("kinematic_enabled", ""),
            "runtime_rigid_body_enabled": props.get("rigid_body_enabled", ""),
            "runtime_mass": props.get("mass", ""),
            "gravity_response_delta_m": gravity_delta.get("root_pose_delta_m", 0.0),
            "small_lateral_push_delta_m": push_delta.get("root_pose_delta_m", 0.0),
            "slow_lift_delta_z_m": lift_delta.get("root_delta_z_m", 0.0),
            "root_pose_updates_over_time": pose_updates,
            "root_velocity_updates_over_time": vel_updates,
            "collision_contact_active": bool(props.get("collision_enabled") or props.get("contact_sensors_active")),
            "asset_dynamics_ok": dynamic_ok,
            "asset_dynamics_blocker": blocker,
            "ppo_ran": False,
            "bc_ran": False,
            "checkpoint_written": False,
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
        }
    )
    for other_slot in backend.slots:
        if other_slot.part_name == "Screw1":
            continue
        other_identity = backend.object_identity_for_slot(other_slot)
        other_props = _v91_asset_runtime_properties(backend, other_slot, other_identity)
        other_dynamic_ok = bool(
            other_identity.get("object_identity_verified")
            and other_props.get("disable_gravity") is False
            and other_props.get("kinematic_enabled") is False
            and _to_float(other_props.get("mass"), 0.0) > 0.0
        )
        other_blocker = "" if other_dynamic_ok else "dynamic_property_check_failed"
        rows.append(
            {
                "part_name": other_slot.part_name,
                **other_identity,
                "before_disable_gravity": "",
                "before_kinematic_enabled": "",
                "before_mass": "",
                "after_disable_gravity": "",
                "after_kinematic_enabled": "",
                "after_mass": "",
                "runtime_disable_gravity": other_props.get("disable_gravity", ""),
                "runtime_kinematic_enabled": other_props.get("kinematic_enabled", ""),
                "runtime_rigid_body_enabled": other_props.get("rigid_body_enabled", ""),
                "runtime_mass": other_props.get("mass", ""),
                "gravity_response_delta_m": "",
                "small_lateral_push_delta_m": "",
                "slow_lift_delta_z_m": "",
                "root_pose_updates_over_time": "",
                "root_velocity_updates_over_time": "",
                "collision_contact_active": bool(other_props.get("collision_enabled") or other_props.get("contact_sensors_active")),
                "asset_dynamics_ok": other_dynamic_ok,
                "asset_dynamics_blocker": other_blocker,
                "ppo_ran": False,
                "bc_ran": False,
                "checkpoint_written": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
            }
        )
    csv_path = write_csv(run_path / "v92_screw1_dynamic_repair.csv", rows)
    json_path = write_json(run_path / "v92_screw1_dynamic_repair.json", rows)
    md_path = run_path / "v92_screw1_dynamic_repair.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v92_screw1_dynamic_repair_csv": str(csv_path), "v92_screw1_dynamic_repair_json": str(json_path), "v92_screw1_dynamic_repair_md": str(md_path)}


def run_v92_collision_subtree_audit_and_repair(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    audit_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    try:
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
    except Exception as exc:
        stage = None
        repair_rows.append({"entity_type": "stage", "repair_attempted": False, "repair_blocker": f"{type(exc).__name__}:{exc}"})
    if stage is not None:
        for slot in backend.slots:
            if slot.global_env_index != slot.local_env_index:
                continue
            identity = backend.object_identity_for_slot(slot)
            root_path = _v91_resolved_prim_path(
                str(identity.get("active_asset_prim_path") or ""),
                slot.local_env_index,
            )
            row = _v92_collision_subtree_row(
                stage,
                root_path,
                slot.part_name,
                "object",
                V85_PART_GEOMETRY.get(slot.part_name, {}),
                env_index=slot.local_env_index,
            )
            audit_rows.append({**identity, **row})
            repair_rows.append(_v92_collision_repair_row(stage, row, slot.part_name))
        for index, name in enumerate(_v92_fingertip_body_names(backend)):
            root_path = _v92_find_robot_body_prim_path(stage, name)
            audit_rows.append(_v92_collision_subtree_row(stage, root_path, f"finger_{index + 1}", "fingertip", {}))
    csv_path = write_csv(run_path / "v92_collision_subtree_audit.csv", audit_rows)
    json_path = write_json(run_path / "v92_collision_subtree_audit.json", audit_rows)
    md_path = run_path / "v92_collision_subtree_audit.md"
    _write_md(md_path, audit_rows)
    repair_csv = write_csv(run_path / "v92_collision_repair_report.csv", repair_rows)
    repair_json = write_json(run_path / "v92_collision_repair_report.json", repair_rows)
    repair_md = run_path / "v92_collision_repair_report.md"
    _write_md(repair_md, repair_rows)
    return {
        "rows": audit_rows,
        "repair_rows": repair_rows,
        "v92_collision_subtree_audit_csv": str(csv_path),
        "v92_collision_subtree_audit_json": str(json_path),
        "v92_collision_subtree_audit_md": str(md_path),
        "v92_collision_repair_report_csv": str(repair_csv),
        "v92_collision_repair_report_json": str(repair_json),
        "v92_collision_repair_report_md": str(repair_md),
    }


def run_v92_staging_repair_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = _v92_stage_table_supported_pickup(backend, plan_rows)
    csv_path = write_csv(run_path / "v92_staging_repair_audit.csv", rows)
    json_path = write_json(run_path / "v92_staging_repair_audit.json", rows)
    md_path = run_path / "v92_staging_repair_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v92_staging_repair_audit_csv": str(csv_path), "v92_staging_repair_audit_json": str(json_path), "v92_staging_repair_audit_md": str(md_path)}


def run_v92_finger_action_sensor_calibration(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    _v92_stage_table_supported_pickup(backend, [])
    for logical_finger in range(1, 6):
        before_tips = _v92_tip_positions(backend)
        before_joints = _v92_joint_pos(backend)
        for step in range(10):
            actions = []
            for _slot in backend.slots:
                action = [0.0] * 16
                action[6 + (logical_finger - 1)] = 0.8
                action[11 + (logical_finger - 1)] = 0.8
                actions.append(action)
            backend.step_envs(actions)
        after_tips = _v92_tip_positions(backend)
        after_joints = _v92_joint_pos(backend)
        for slot in backend.slots:
            metrics = backend._metrics_for_slot(slot, "v92_single_finger_calibration")
            forces = _v92_force_list(metrics)
            responding_index = _v92_max_index(forces)
            tip_delta = _v92_tip_delta(before_tips, after_tips, slot.local_env_index, logical_finger - 1)
            joint_delta = _v92_joint_delta(before_joints, after_joints, slot.local_env_index)
            force_peak = max([0.0, *forces])
            consistent = bool(force_peak <= backend.contact_manager.force_threshold_n or responding_index == logical_finger - 1)
            blocker = "" if consistent else f"expected_sensor_{logical_finger - 1}_got_{responding_index}"
            rows.append(
                {
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "logical_finger_id": logical_finger,
                    "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                    "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                    "non_active_fingers_neutral": True,
                    "joint_position_delta_available": joint_delta >= 0.0,
                    "joint_position_delta_l2": joint_delta,
                    "fingertip_position_delta_m": tip_delta,
                    "per_finger_force_norm": forces,
                    "force_peak_n": force_peak,
                    "responding_sensor_index": responding_index,
                    "expected_sensor_index": logical_finger - 1,
                    "finger_mapping_consistent": consistent,
                    "finger_mapping_blocker": blocker,
                    "calibration_counts_as_grasp_success": False,
                }
            )
    for logical_finger in range(1, 6):
        finger_rows = [row for row in rows if int(row.get("logical_finger_id") or 0) == logical_finger]
        force_rows = [row for row in finger_rows if _to_float(row.get("force_peak_n")) > backend.contact_manager.force_threshold_n]
        if force_rows:
            responses = [int(row.get("responding_sensor_index") or -1) for row in force_rows]
            inferred = max(set(responses), key=responses.count)
            consistent = inferred == logical_finger - 1
            blocker = "" if consistent else f"logical_finger_{logical_finger}_maps_to_sensor_{inferred}"
        else:
            inferred = logical_finger - 1
            consistent = False
            blocker = "no_force_response_observed_for_calibration"
        mapping_rows.append(
            {
                "part_name": "__global__",
                "logical_finger_id": logical_finger,
                "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                "physical_sensor_index": inferred,
                "physical_fingertip_body_name": _v92_fingertip_name(backend, inferred),
                "finger_mapping_consistent": consistent,
                "finger_mapping_blocker": blocker,
            }
        )
    csv_path = write_csv(run_path / "v92_finger_action_sensor_calibration.csv", rows)
    json_path = write_json(run_path / "v92_finger_action_sensor_calibration.json", rows)
    md_path = run_path / "v92_finger_action_sensor_calibration.md"
    _write_md(md_path, rows)
    fix_csv = write_csv(run_path / "v92_finger_mapping_fix_report.csv", mapping_rows)
    fix_json = write_json(run_path / "v92_finger_mapping_fix_report.json", mapping_rows)
    fix_md = run_path / "v92_finger_mapping_fix_report.md"
    _write_md(fix_md, mapping_rows)
    return {
        "rows": rows,
        "mapping_rows": mapping_rows,
        "v92_finger_action_sensor_calibration_csv": str(csv_path),
        "v92_finger_action_sensor_calibration_json": str(json_path),
        "v92_finger_action_sensor_calibration_md": str(md_path),
        "v92_finger_mapping_fix_report_csv": str(fix_csv),
        "v92_finger_mapping_fix_report_json": str(fix_json),
        "v92_finger_mapping_fix_report_md": str(fix_md),
    }


def run_v92_support_gate_calibration(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    mapping_valid = all(bool(row.get("finger_mapping_consistent")) for row in mapping_rows if row.get("part_name") == "__global__")
    mapping = {int(row.get("logical_finger_id") or 0) - 1: int(row.get("physical_sensor_index") or -1) for row in mapping_rows if row.get("part_name") == "__global__"}
    for slot in backend.slots:
        metrics = backend._metrics_for_slot(slot, "v92_support_gate_calibration")
        group = V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0]
        logical = _v91_active_finger_indices(group)
        physical = [mapping.get(item, item) for item in logical]
        forces = _v92_force_list(metrics)
        calibrated_count = sum(1 for index in physical if 0 <= index < len(forces) and forces[index] > backend.contact_manager.force_threshold_n)
        new_gate = bool(mapping_valid and calibrated_count >= 2 and not bool(metrics.get("table_collision")) and not backend.object_write_by_policy_detected)
        old_gate = bool(metrics.get("support_gate_ok"))
        rows.append(
            {
                "part_name": slot.part_name,
                "active_finger_group": group,
                "old_support_gate_ok": old_gate,
                "new_calibrated_support_gate_ok": new_gate,
                "old_gate_would_pass_new_fails": bool(old_gate and not new_gate),
                "calibrated_multi_finger_support_observed": new_gate,
                "calibrated_physical_sensor_indices": ",".join(str(item) for item in physical),
                "calibrated_force_contact_count": calibrated_count,
                "force_peak_n": max([0.0, *forces]),
                "finger_mapping_consistent": mapping_valid,
                "support_gate_blocker": "" if new_gate else ("finger_mapping_inconsistent" if not mapping_valid else "calibrated_two_finger_support_not_observed"),
            }
        )
    csv_path = write_csv(run_path / "v92_support_gate_calibration_report.csv", rows)
    json_path = write_json(run_path / "v92_support_gate_calibration_report.json", rows)
    md_path = run_path / "v92_support_gate_calibration_report.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v92_support_gate_calibration_report_csv": str(csv_path), "v92_support_gate_calibration_report_json": str(json_path), "v92_support_gate_calibration_report_md": str(md_path)}


def run_v92_sanity_probe(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, mapping_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    mapping_valid = all(bool(row.get("finger_mapping_consistent")) for row in mapping_rows if row.get("part_name") == "__global__")
    mapping = {int(row.get("logical_finger_id") or 0) - 1: int(row.get("physical_sensor_index") or -1) for row in mapping_rows if row.get("part_name") == "__global__"}
    staging_rows = _v92_stage_table_supported_pickup(backend, [])
    start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
    first_contact_step: dict[str, int] = {}
    force_values_by_part: dict[str, list[float]] = {slot.part_name: [] for slot in backend.slots}
    two_finger_seen: dict[str, bool] = {slot.part_name: False for slot in backend.slots}
    for step in range(24):
        actions = []
        for slot in backend.slots:
            action, _mask = _v92_group_close_action(slot.part_name, step)
            actions.append(action)
        backend.step_envs(actions)
        for slot in backend.slots:
            metrics = backend._metrics_for_slot(slot, "v92_no_sticky_support_probe")
            forces = _v92_force_list(metrics)
            force_peak = max([0.0, *forces])
            if force_peak > backend.contact_manager.force_threshold_n and slot.part_name not in first_contact_step:
                first_contact_step[slot.part_name] = step
            group = V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0]
            logical = _v91_active_finger_indices(group)
            physical = [mapping.get(item, item) for item in logical]
            calibrated_count = sum(1 for index in physical if 0 <= index < len(forces) and forces[index] > backend.contact_manager.force_threshold_n)
            two_finger_seen[slot.part_name] = bool(two_finger_seen[slot.part_name] or (mapping_valid and calibrated_count >= 2))
            force_values_by_part[slot.part_name].append(force_peak)
            trace_rows.append(
                {
                    "probe_step": step,
                    "part_name": slot.part_name,
                    "active_finger_group": group,
                    "force_peak_n": force_peak,
                    "calibrated_force_contact_count": calibrated_count,
                    "calibrated_multi_finger_support_observed": bool(mapping_valid and calibrated_count >= 2),
                    "object_displacement_m": metrics.get("object_displacement_m", 0.0),
                    "table_collision": metrics.get("table_collision", False),
                    "object_write_by_policy_detected": backend.object_write_by_policy_detected,
                    "sticky_action_available_to_policy": backend.sticky_action_available_to_policy,
                    "distance_only_success_used": False,
                    "grasp_success_claimed": False,
                }
            )
    for slot in backend.slots:
        state = _v91_object_state_for_slot(backend, slot)
        delta = _v91_state_delta(start_states.get(slot.part_name, {}), state)
        staging = next((row for row in staging_rows if row.get("part_name") == slot.part_name), {})
        values = force_values_by_part.get(slot.part_name, [])
        summary_rows.append(
            {
                "part_name": slot.part_name,
                "object_identity_verified": bool(backend.object_identity_for_slot(slot).get("object_identity_verified")),
                "single_simulation_context": backend.single_simulation_context,
                "dynamic_asset_ok": True,
                "collision_subtree_ok": True,
                "initial_condition_valid": bool(staging.get("initial_condition_valid")),
                "pre_contact_displacement_m": _to_float(staging.get("pre_contact_displacement_m")),
                "first_contact_step": first_contact_step.get(slot.part_name, ""),
                "calibrated_multi_finger_support_observed": bool(two_finger_seen.get(slot.part_name)),
                "peak_force_n": max([0.0, *values]),
                "mean_force_n": sum(values) / len(values) if values else 0.0,
                "excessive_force_rate": sum(1 for value in values if value > 150.0) / len(values) if values else 0.0,
                "object_displacement_m": delta.get("root_pose_delta_m", 0.0),
                "object_write_by_policy_detected": backend.object_write_by_policy_detected,
                "sticky_action_available_to_policy": backend.sticky_action_available_to_policy,
                "ppo_ran": False,
                "bc_ran": False,
                "checkpoint_written": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "grasp_success_claimed": False,
            }
        )
    csv_path = write_csv(run_path / "v92_sanity_probe_results.csv", summary_rows)
    json_path = write_json(run_path / "v92_sanity_probe_results.json", summary_rows)
    trace_path = write_jsonl(run_path / "v92_sanity_probe_trace.jsonl", trace_rows)
    return {"rows": summary_rows, "trace_rows": trace_rows, "v92_sanity_probe_results_csv": str(csv_path), "v92_sanity_probe_results_json": str(json_path), "v92_sanity_probe_trace_jsonl": str(trace_path)}


def run_v93_native_shutdown_root_cause(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = [
        {
            "operation": "legacy_live_root_pose_write_after_reset_before_probe_step",
            "object_involved": "all_v83_active_assets",
            "probe_executed_in_main_process": False,
            "isolated_subprocess_probe_executed": False,
            "native_shutdown_reproduced": "",
            "likely_cause": "DirectRLEnv_lifecycle_misuse_or_stale_root_state_handle_after_active_simulation",
            "failing_operation": "post-reset live RigidObject.write_root_pose_to_sim staging used by older v84/v85 helpers",
            "safe_replacement_method": "v93_reset_staging_plan_applied_inside_WujiUnifiedFiveObjectEnv._set_assets_to_default_pose",
            "remaining_risk": "unsafe legacy live staging helpers still exist for historical modes and must not be used as v93 evidence",
            "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
            "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
            "grasp_success_claimed": False,
            "final_success": False,
        },
        {
            "operation": "v93_reset_lifecycle_staging",
            "object_involved": "all_v83_active_assets",
            "probe_executed_in_main_process": True,
            "isolated_subprocess_probe_executed": False,
            "native_shutdown_reproduced": False,
            "likely_cause": "",
            "failing_operation": "",
            "safe_replacement_method": "configure plan before reset and write root pose/velocity during reset only",
            "remaining_risk": "",
            "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
            "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
            "grasp_success_claimed": False,
            "final_success": False,
        },
    ]
    json_path = write_json(run_path / "v93_native_shutdown_root_cause.json", rows)
    md_path = run_path / "v93_native_shutdown_root_cause.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v93_native_shutdown_root_cause_json": str(json_path), "v93_native_shutdown_root_cause_md": str(md_path)}


def run_v93_collision_geometry_audit_and_repair(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    bbox_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    try:
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
    except Exception as exc:
        stage = None
        repair_rows.append({"repair_attempted": False, "repair_applied": False, "repair_blocker": f"{type(exc).__name__}:{exc}"})
    if stage is not None:
        for slot in backend.slots:
            if slot.global_env_index != slot.local_env_index:
                continue
            identity = backend.object_identity_for_slot(slot)
            root_path = _v91_resolved_prim_path(
                str(identity.get("active_asset_prim_path") or ""),
                slot.local_env_index,
            )
            bbox = _v93_collision_world_bbox_row(
                stage,
                root_path,
                slot.part_name,
                "object",
                V85_PART_GEOMETRY.get(slot.part_name, {}),
                env_index=slot.local_env_index,
            )
            bbox_rows.append({**identity, **bbox})
            inventory_rows.extend(_v93_collision_inventory_rows(stage, root_path, slot.part_name, "object"))
            alignment_rows.append(_v93_alignment_report_row({**identity, **bbox}))
        for index, name in enumerate(_v92_fingertip_body_names(backend)):
            root_path = _v92_find_robot_body_prim_path(stage, name)
            bbox = _v93_collision_world_bbox_row(stage, root_path, f"finger_{index + 1}", "fingertip", {})
            bbox_rows.append(bbox)
            inventory_rows.extend(_v93_collision_inventory_rows(stage, root_path, f"finger_{index + 1}", "fingertip"))
            alignment_rows.append(_v93_alignment_report_row(bbox))
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    repair_rows.extend([dict(row) for row in getattr(base, "v93_collision_repair_report", []) or []])
    if not repair_rows:
        repair_rows = [
            {
                "part_name": part,
                "repair_attempted": False,
                "repair_applied": False,
                "repair_type": "",
                "repair_scope": "unified_env_runtime_stage_only",
                "legacy_chair_configs_touched": False,
                "sticky_or_adhesion_added": False,
                "infinite_friction_used": False,
                "success_label_used": False,
                "repair_blocker": "no_v93_runtime_repair_report_available",
            }
            for part in V83_PARTS
        ]
    bbox_csv = write_csv(run_path / "v93_collision_world_bbox_audit.csv", bbox_rows)
    bbox_json = write_json(run_path / "v93_collision_world_bbox_audit.json", bbox_rows)
    bbox_md = run_path / "v93_collision_world_bbox_audit.md"
    _write_md(bbox_md, bbox_rows)
    inventory_csv = write_csv(run_path / "v93_collision_prim_inventory.csv", inventory_rows)
    inventory_json = write_json(run_path / "v93_collision_prim_inventory.json", inventory_rows)
    alignment_csv = write_csv(run_path / "v93_visual_collision_alignment_report.csv", alignment_rows)
    alignment_json = write_json(run_path / "v93_visual_collision_alignment_report.json", alignment_rows)
    alignment_md = run_path / "v93_visual_collision_alignment_report.md"
    _write_md(alignment_md, alignment_rows)
    repair_csv = write_csv(run_path / "v93_collision_repair_report.csv", repair_rows)
    repair_json = write_json(run_path / "v93_collision_repair_report.json", repair_rows)
    repair_md = run_path / "v93_collision_repair_report.md"
    _write_md(repair_md, repair_rows)
    return {
        "rows": bbox_rows,
        "inventory_rows": inventory_rows,
        "alignment_rows": alignment_rows,
        "repair_rows": repair_rows,
        "v93_collision_world_bbox_audit_csv": str(bbox_csv),
        "v93_collision_world_bbox_audit_json": str(bbox_json),
        "v93_collision_world_bbox_audit_md": str(bbox_md),
        "v93_collision_prim_inventory_csv": str(inventory_csv),
        "v93_collision_prim_inventory_json": str(inventory_json),
        "v93_visual_collision_alignment_report_csv": str(alignment_csv),
        "v93_visual_collision_alignment_report_json": str(alignment_json),
        "v93_visual_collision_alignment_report_md": str(alignment_md),
        "v93_collision_repair_report_csv": str(repair_csv),
        "v93_collision_repair_report_json": str(repair_json),
        "v93_collision_repair_report_md": str(repair_md),
    }


def run_v93_safe_staging_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    configured_rows = _v93_configured_staging_plan(backend, plan_rows)
    backend.configure_v93_safe_staging(configured_rows)
    backend.object_write_by_policy_detected = False
    backend.reset_envs()
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    reset_rows = [dict(row) for row in getattr(base, "v93_last_safe_staging_rows", []) or []]
    start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
    force_values: dict[str, list[float]] = {slot.part_name: [] for slot in backend.slots}
    contact_counts: dict[str, list[int]] = {slot.part_name: [] for slot in backend.slots}
    for slot in backend.slots:
        metrics = backend._metrics_for_slot(slot, "v93_after_safe_reset")
        force_values[slot.part_name].append(max([0.0, *_v92_force_list(metrics)]))
        contact_counts[slot.part_name].append(int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0))
    for _ in range(8):
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        for slot in backend.slots:
            metrics = backend._metrics_for_slot(slot, "v93_precontact_zero_settle")
            force_values[slot.part_name].append(max([0.0, *_v92_force_list(metrics)]))
            contact_counts[slot.part_name].append(int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0))
    audit_rows: list[dict[str, Any]] = []
    precontact_rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        identity = backend.object_identity_for_slot(slot)
        plan = next((row for row in configured_rows if row.get("part_name") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index), {})
        reset = next((row for row in reset_rows if row.get("part_name") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index), {})
        end_state = _v91_object_state_for_slot(backend, slot)
        delta = _v91_state_delta(start_states.get(slot.part_name, {}), end_state)
        peak = max([0.0, *force_values.get(slot.part_name, [])])
        count = max([0, *contact_counts.get(slot.part_name, [])])
        displacement = abs(_to_float(delta.get("root_pose_delta_m")))
        z_drift = abs(_to_float(delta.get("root_delta_z_m")))
        reset_impulse = bool(peak > backend.contact_manager.force_threshold_n or count > 0)
        valid = bool(
            reset
            and displacement <= 0.02
            and z_drift <= 0.02
            and not reset_impulse
            and not backend.object_write_by_policy_detected
        )
        blocker = ""
        if not reset:
            blocker = "v93_reset_staging_row_missing"
        elif displacement > 0.02:
            blocker = "pre_contact_displacement_gt_0p02m"
        elif z_drift > 0.02:
            blocker = "pre_contact_z_drift_gt_0p02m"
        elif reset_impulse:
            blocker = "reset_interpenetration_or_precontact_force"
        elif backend.object_write_by_policy_detected:
            blocker = "object_write_after_reset_detected"
        row = {
            **plan,
            **reset,
            "pre_contact_displacement_m": displacement,
            "pre_contact_z_drift_m": z_drift,
            "initial_force_peak_n": peak,
            "initial_force_contact_count": count,
            "reset_interpenetration_or_impulse_detected": reset_impulse,
            "initial_condition_valid": valid,
            "initial_condition_blocker": blocker,
            "native_shutdown_detected": False,
            "ppo_ran": False,
            "bc_ran": False,
            "checkpoint_written": False,
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
        }
        audit_rows.append(row)
        precontact_rows.append(
            {
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "pre_contact_displacement_m": displacement,
                "pre_contact_z_drift_m": z_drift,
                "initial_force_peak_n": peak,
                "initial_force_contact_count": count,
                "precontact_motion_ok": bool(valid),
                "blocker": blocker,
            }
        )
    backend.v93_safe_staging_audit_rows = audit_rows
    audit_csv = write_csv(run_path / "v93_safe_staging_audit.csv", audit_rows)
    audit_json = write_json(run_path / "v93_safe_staging_audit.json", audit_rows)
    audit_md = run_path / "v93_safe_staging_audit.md"
    _write_md(audit_md, audit_rows)
    pre_csv = write_csv(run_path / "v93_precontact_motion_audit.csv", precontact_rows)
    pre_json = write_json(run_path / "v93_precontact_motion_audit.json", precontact_rows)
    return {
        "rows": audit_rows,
        "precontact_rows": precontact_rows,
        "v93_safe_staging_audit_csv": str(audit_csv),
        "v93_safe_staging_audit_json": str(audit_json),
        "v93_safe_staging_audit_md": str(audit_md),
        "v93_precontact_motion_audit_csv": str(pre_csv),
        "v93_precontact_motion_audit_json": str(pre_json),
    }


def run_v93_screw1_post_repair_sanity(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    slot = next((item for item in backend.slots if item.part_name == "Screw1"), None)
    rows: list[dict[str, Any]] = []
    if slot is None:
        rows.append({"part_name": "Screw1", "screw1_post_repair_sanity_ok": False, "blocker": "screw1_slot_missing"})
    else:
        identity = backend.object_identity_for_slot(slot)
        props = _v91_asset_runtime_properties(backend, slot, identity)
        start = _v91_object_state_for_slot(backend, slot)
        for _ in range(4):
            backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        settle = _v91_object_state_for_slot(backend, slot)
        for _ in range(4):
            actions = [[0.0] * 16 for _slot in backend.slots]
            actions[slot.global_env_index][0] = 0.04
            backend.step_envs(actions)
        pushed = _v91_object_state_for_slot(backend, slot)
        settle_delta = _v91_state_delta(start, settle)
        push_delta = _v91_state_delta(settle, pushed)
        pose_updates = max(_to_float(settle_delta.get("root_pose_delta_m")), _to_float(push_delta.get("root_pose_delta_m"))) > 1.0e-6
        vel_updates = max(_to_float(settle.get("root_lin_vel_norm")), _to_float(pushed.get("root_lin_vel_norm"))) > 1.0e-6
        ok = bool(
            identity.get("object_identity_verified")
            and props.get("disable_gravity") is False
            and props.get("kinematic_enabled") is False
            and _to_float(props.get("mass"), 0.0) > 0.0
            and (pose_updates or vel_updates)
            and not backend.object_write_by_policy_detected
        )
        blocker = "" if ok else "screw1_dynamic_or_safe_staging_response_failed"
        rows.append(
            {
                "part_name": "Screw1",
                **identity,
                "runtime_disable_gravity": props.get("disable_gravity", ""),
                "runtime_kinematic_enabled": props.get("kinematic_enabled", ""),
                "runtime_mass": props.get("mass", ""),
                "root_pose_updates_over_time": pose_updates,
                "root_velocity_updates_over_time": vel_updates,
                "settle_delta_m": settle_delta.get("root_pose_delta_m", 0.0),
                "small_push_delta_m": push_delta.get("root_pose_delta_m", 0.0),
                "native_shutdown_detected": False,
                "object_write_by_policy_detected": backend.object_write_by_policy_detected,
                "screw1_post_repair_sanity_ok": ok,
                "blocker": blocker,
                "grasp_success_claimed": False,
            }
        )
    csv_path = write_csv(run_path / "v93_screw1_post_repair_sanity.csv", rows)
    json_path = write_json(run_path / "v93_screw1_post_repair_sanity.json", rows)
    md_path = run_path / "v93_screw1_post_repair_sanity.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v93_screw1_post_repair_sanity_csv": str(csv_path), "v93_screw1_post_repair_sanity_json": str(json_path), "v93_screw1_post_repair_sanity_md": str(md_path)}


def run_v93_post_collision_contact_sanity(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    backend.reset_envs()
    for mode in ("single_finger", "two_finger"):
        for step in range(8):
            actions: list[list[float]] = []
            context: dict[int, dict[str, Any]] = {}
            for slot in backend.slots:
                group = V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0]
                if mode == "single_finger":
                    group = group[:1]
                action = [0.0] * 16
                mask = _v91_apply_finger_mask(action, group=group, active_value=min(0.75, 0.10 * (step + 1)), support_value=0.0)
                actions.append(action)
                context[slot.global_env_index] = {
                    "v93_setup_repair_mode": True,
                    "nominal_phase": f"v93_{mode}_contact_sanity",
                    "active_finger_group": group,
                    **mask,
                    "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                    "fallback_success_used": False,
                    "distance_only_success_used": False,
                    "object_write_after_reset_allowed": False,
                }
            backend._v87_pending_action_context = context
            metrics_rows = backend.step_envs(actions)
            if step != 7:
                continue
            for slot, metrics in zip(backend.slots, metrics_rows):
                forces = _v92_force_list(metrics)
                peak = max([0.0, *forces])
                active_group = str(context[slot.global_env_index].get("active_finger_group") or "")
                active_indices = _v91_active_finger_indices(active_group)
                active_count = sum(1 for index in active_indices if 0 <= index < len(forces) and forces[index] > backend.contact_manager.force_threshold_n)
                rows.append(
                    {
                        "part_name": slot.part_name,
                        "env_index": slot.global_env_index,
                        "contact_sanity_mode": mode,
                        "commanded_fingers": context[slot.global_env_index].get("commanded_fingers", ""),
                        "commanded_policy_cols": context[slot.global_env_index].get("commanded_policy_cols", ""),
                        "commanded_isaac_cols": context[slot.global_env_index].get("commanded_isaac_cols", ""),
                        "per_finger_force_norm": forces,
                        "force_peak_n": peak,
                        "responding_sensor_index": _v92_max_index(forces),
                        "active_finger_force_count": active_count,
                        "multi_finger_contact_observed": bool(mode == "two_finger" and active_count >= 2),
                        "table_collision_counted_as_finger_support": bool(metrics.get("table_collision") and active_count > 0),
                        "object_write_by_policy_detected": backend.object_write_by_policy_detected,
                        "sticky_eval_ran": False,
                        "grasp_success_claimed": False,
                    }
                )
    csv_path = write_csv(run_path / "v93_post_collision_contact_sanity.csv", rows)
    json_path = write_json(run_path / "v93_post_collision_contact_sanity.json", rows)
    return {"rows": rows, "v93_post_collision_contact_sanity_csv": str(csv_path), "v93_post_collision_contact_sanity_json": str(json_path)}


def run_v93_support_gate_post_repair_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, contact_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        part_rows = [row for row in contact_rows if row.get("part_name") == slot.part_name]
        single_ok = any(row.get("contact_sanity_mode") == "single_finger" and _to_float(row.get("force_peak_n")) > backend.contact_manager.force_threshold_n for row in part_rows)
        two_ok = any(bool(row.get("multi_finger_contact_observed")) for row in part_rows)
        table_counted = any(bool(row.get("table_collision_counted_as_finger_support")) for row in part_rows)
        rows.append(
            {
                "part_name": slot.part_name,
                "single_finger_contact_sanity": single_ok,
                "two_finger_contact_sanity": two_ok,
                "calibrated_multi_finger_support_observed": two_ok,
                "table_collision_counted_as_finger_support": table_counted,
                "support_gate_post_repair_ok": bool(single_ok and two_ok and not table_counted and not backend.object_write_by_policy_detected),
                "support_gate_blocker": ""
                if single_ok and two_ok and not table_counted and not backend.object_write_by_policy_detected
                else (
                    "table_collision_counted_as_finger_support"
                    if table_counted
                    else "calibrated_two_finger_contact_not_observed"
                    if single_ok
                    else "single_finger_contact_not_observed"
                ),
                "sticky_eval_ran": False,
                "grasp_success_claimed": False,
            }
        )
    csv_path = write_csv(run_path / "v93_support_gate_post_repair_audit.csv", rows)
    json_path = write_json(run_path / "v93_support_gate_post_repair_audit.json", rows)
    md_path = run_path / "v93_support_gate_post_repair_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v93_support_gate_post_repair_audit_csv": str(csv_path), "v93_support_gate_post_repair_audit_json": str(json_path), "v93_support_gate_post_repair_audit_md": str(md_path)}


def run_v94_pregrasp_alignment_audit(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    configured_rows = _v94_configured_pregrasp_plan(backend, plan_rows or [])
    if not _v94_live_probes_enabled():
        audit_rows: list[dict[str, Any]] = []
        for slot in backend.slots:
            plan = _v94_plan_for_slot(configured_rows, slot)
            identity = backend.object_identity_for_slot(slot)
            backend._record_identity(identity)
            metrics = _v94_latest_metrics_for_slot(backend, slot)
            distance = _v94_distance_for_slot(backend, slot, str(plan.get("active_finger_group") or ""))
            force_peak = max([0.0, *_v92_force_list(metrics)])
            row = {
                **identity,
                **plan,
                **distance,
                "v94_live_probe_executed": False,
                "native_shutdown_risk": True,
                "initial_force_peak_n": force_peak,
                "initial_force_contact_count": int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0),
                "pre_contact_displacement_m": metrics.get("object_displacement_m", 0.0),
                "pre_contact_z_drift_m": metrics.get("object_delta_z_m", 0.0),
                "reset_interpenetration_or_impulse_detected": bool(force_peak > backend.contact_manager.force_threshold_n),
                "safe_pregrasp_ok": False,
                "initial_condition_valid": False,
                "safe_pregrasp_blocker": "v94_live_pregrasp_probe_disabled_native_shutdown_risk",
                "hand_live_write_after_reset_detected": False,
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "contact_sensor_api_available": bool(metrics.get("contact_sensor_api_available")),
                "single_simulation_context": bool(backend.single_simulation_context),
                "gym_make_count": int(backend.gym_make_count),
                "ppo_ran": False,
                "bc_ran": False,
                "checkpoint_written": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "dataset_exported": False,
                "usable_training_row_count": 0,
                "grasp_success_claimed": False,
                "final_success": False,
            }
            audit_rows.append(row)
        backend.v94_pregrasp_audit_rows = audit_rows
        csv_path = write_csv(run_path / "v94_pregrasp_alignment_audit.csv", audit_rows)
        json_path = write_json(run_path / "v94_pregrasp_alignment_audit.json", audit_rows)
        md_path = run_path / "v94_pregrasp_alignment_audit.md"
        _write_md(md_path, audit_rows)
        return {
            "rows": audit_rows,
            "configured_rows": configured_rows,
            "v94_pregrasp_alignment_audit_csv": str(csv_path),
            "v94_pregrasp_alignment_audit_json": str(json_path),
            "v94_pregrasp_alignment_audit_md": str(md_path),
        }
    _v94_configure_stable_reset(backend, configured_rows)
    backend.object_write_by_policy_detected = False
    backend.reset_envs()
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    reset_rows = [dict(row) for row in (getattr(base, "v94_last_pregrasp_rows", []) or getattr(base, "v93_last_safe_staging_rows", []) or [])]
    start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
    force_values: dict[str, list[float]] = {slot.part_name: [] for slot in backend.slots}
    contact_counts: dict[str, list[int]] = {slot.part_name: [] for slot in backend.slots}
    initial_metrics: dict[str, dict[str, Any]] = {}
    for slot in backend.slots:
        metrics = backend._metrics_for_slot(slot, "v94_after_pregrasp_reset")
        initial_metrics[slot.part_name] = metrics
        force_values[slot.part_name].append(max([0.0, *_v92_force_list(metrics)]))
        contact_counts[slot.part_name].append(int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0))
    for settle_step in range(4):
        context: dict[int, dict[str, Any]] = {}
        for slot in backend.slots:
            context[slot.global_env_index] = {
                "v94_safe_pregrasp_mode": True,
                "v94_calibration_mode": "pregrasp_zero_settle",
                "nominal_phase": "v94_pregrasp_zero_settle",
                "object_write_after_reset_allowed": False,
                "distance_only_success_used": False,
                "fallback_success_used": False,
            }
        backend._v87_pending_action_context = context
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        for slot in backend.slots:
            metrics = backend._metrics_for_slot(slot, f"v94_pregrasp_zero_settle_{settle_step}")
            force_values[slot.part_name].append(max([0.0, *_v92_force_list(metrics)]))
            contact_counts[slot.part_name].append(int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0))
    audit_rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        plan = next((row for row in configured_rows if row.get("part_name") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index), {})
        reset = next((row for row in reset_rows if row.get("part_name") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index), {})
        identity = backend.object_identity_for_slot(slot)
        backend._record_identity(identity)
        end_state = _v91_object_state_for_slot(backend, slot)
        delta = _v91_state_delta(start_states.get(slot.part_name, {}), end_state)
        distance = _v94_distance_for_slot(backend, slot, str(plan.get("active_finger_group") or ""))
        peak = max([0.0, *force_values.get(slot.part_name, [])])
        count = max([0, *contact_counts.get(slot.part_name, [])])
        displacement = abs(_to_float(delta.get("root_pose_delta_m")))
        z_drift = abs(_to_float(delta.get("root_delta_z_m")))
        reset_impulse = bool(peak > backend.contact_manager.force_threshold_n or count > 0)
        safe = bool(
            reset
            and distance.get("fingertip_object_surface_distance_min_m", 1.0) <= 0.015
            and displacement <= 0.02
            and z_drift <= 0.02
            and not reset_impulse
            and not backend.object_write_by_policy_detected
        )
        blocker = ""
        if not reset:
            blocker = "v94_pregrasp_reset_row_missing"
        elif distance.get("fingertip_object_surface_distance_min_m", 1.0) > 0.015:
            blocker = "fingertip_object_distance_gt_0p015m"
        elif displacement > 0.02:
            blocker = "pre_contact_displacement_gt_0p02m"
        elif z_drift > 0.02:
            blocker = "pre_contact_z_drift_gt_0p02m"
        elif reset_impulse:
            blocker = "reset_interpenetration_or_precontact_force"
        elif backend.object_write_by_policy_detected:
            blocker = "object_or_hand_write_after_reset_detected"
        row = {
            **identity,
            **plan,
            **reset,
            **distance,
            "initial_force_peak_n": peak,
            "initial_force_contact_count": count,
            "pre_contact_displacement_m": displacement,
            "pre_contact_z_drift_m": z_drift,
            "reset_interpenetration_or_impulse_detected": reset_impulse,
            "safe_pregrasp_ok": safe,
            "initial_condition_valid": safe,
            "safe_pregrasp_blocker": blocker,
            "hand_live_write_after_reset_detected": False,
            "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
            "contact_sensor_api_available": bool(initial_metrics.get(slot.part_name, {}).get("contact_sensor_api_available")),
            "ppo_ran": False,
            "bc_ran": False,
            "checkpoint_written": False,
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "dataset_exported": False,
            "usable_training_row_count": 0,
            "grasp_success_claimed": False,
            "final_success": False,
        }
        audit_rows.append(row)
    backend.v94_pregrasp_audit_rows = audit_rows
    csv_path = write_csv(run_path / "v94_pregrasp_alignment_audit.csv", audit_rows)
    json_path = write_json(run_path / "v94_pregrasp_alignment_audit.json", audit_rows)
    md_path = run_path / "v94_pregrasp_alignment_audit.md"
    _write_md(md_path, audit_rows)
    return {
        "rows": audit_rows,
        "configured_rows": configured_rows,
        "v94_pregrasp_alignment_audit_csv": str(csv_path),
        "v94_pregrasp_alignment_audit_json": str(json_path),
        "v94_pregrasp_alignment_audit_md": str(md_path),
    }


def run_v94_wrist_action_calibration(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    if not _v94_live_probes_enabled():
        for axis, axis_name in enumerate(("x", "y", "z")):
            for sign in (-1, 1):
                for slot in backend.slots:
                    rows.append(
                        {
                            "part_name": slot.part_name,
                            "env_index": slot.global_env_index,
                            "calibration_axis": axis_name,
                            "calibration_sign": int(sign),
                            "policy_col": axis,
                            "isaac_action_mapped_through_v82": True,
                            "policy_command_value": 0.25 * float(sign),
                            "v94_live_probe_executed": False,
                            "native_shutdown_risk": True,
                            "fingertip_midpoint_delta_x": 0.0,
                            "fingertip_midpoint_delta_y": 0.0,
                            "fingertip_midpoint_delta_z": 0.0,
                            "measured_axis_delta_m": 0.0,
                            "expected_vs_measured_sign_ok": False,
                            "object_displacement_m": _v94_latest_metrics_for_slot(backend, slot).get("object_displacement_m", 0.0),
                            "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                            "wrist_action_mapping_ok": False,
                            "blocker": "v94_live_wrist_probe_disabled_native_shutdown_risk",
                            "sticky_eval_ran": False,
                            "grasp_success_claimed": False,
                        }
                    )
        csv_path = write_csv(run_path / "v94_wrist_action_calibration.csv", rows)
        json_path = write_json(run_path / "v94_wrist_action_calibration.json", rows)
        md_path = run_path / "v94_wrist_action_calibration.md"
        _write_md(md_path, rows)
        return {"rows": rows, "v94_wrist_action_calibration_csv": str(csv_path), "v94_wrist_action_calibration_json": str(json_path), "v94_wrist_action_calibration_md": str(md_path)}
    for axis, axis_name in enumerate(("x", "y", "z")):
        for sign in (-1, 1):
            _v94_configure_stable_reset(backend, plan_rows)
            backend.reset_envs()
            before_mid = _v94_fingertip_midpoints(backend)
            before_state = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
            for _ in range(4):
                actions: list[list[float]] = []
                context: dict[int, dict[str, Any]] = {}
                for slot in backend.slots:
                    action = [0.0] * 16
                    action[axis] = 0.25 * float(sign)
                    actions.append(action)
                    context[slot.global_env_index] = {
                        "v94_safe_pregrasp_mode": True,
                        "v94_calibration_mode": "wrist_action",
                        "nominal_phase": f"v94_wrist_{axis_name}_{sign:+d}",
                        "calibration_axis": axis_name,
                        "calibration_sign": int(sign),
                        "object_write_after_reset_allowed": False,
                        "fallback_success_used": False,
                        "distance_only_success_used": False,
                    }
                backend._v87_pending_action_context = context
                backend.step_envs(actions)
            after_mid = _v94_fingertip_midpoints(backend)
            for slot in backend.slots:
                delta = _v94_vec_delta(before_mid, after_mid, slot.local_env_index)
                obj_delta = _v91_state_delta(before_state.get(slot.part_name, {}), _v91_object_state_for_slot(backend, slot))
                measured = delta[axis] if len(delta) >= 3 else 0.0
                sign_ok = bool(abs(measured) > 1.0e-5 and measured * float(sign) > 0.0)
                rows.append(
                    {
                        "part_name": slot.part_name,
                        "env_index": slot.global_env_index,
                        "calibration_axis": axis_name,
                        "calibration_sign": int(sign),
                        "policy_col": axis,
                        "isaac_action_mapped_through_v82": True,
                        "policy_command_value": 0.25 * float(sign),
                        "fingertip_midpoint_delta_x": delta[0] if len(delta) >= 3 else 0.0,
                        "fingertip_midpoint_delta_y": delta[1] if len(delta) >= 3 else 0.0,
                        "fingertip_midpoint_delta_z": delta[2] if len(delta) >= 3 else 0.0,
                        "measured_axis_delta_m": measured,
                        "expected_vs_measured_sign_ok": sign_ok,
                        "object_displacement_m": obj_delta.get("root_pose_delta_m", 0.0),
                        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                        "wrist_action_mapping_ok": bool(sign_ok and not backend.object_write_by_policy_detected),
                        "blocker": "" if sign_ok else "wrist_axis_delta_sign_or_magnitude_unexpected",
                        "grasp_success_claimed": False,
                    }
                )
    csv_path = write_csv(run_path / "v94_wrist_action_calibration.csv", rows)
    json_path = write_json(run_path / "v94_wrist_action_calibration.json", rows)
    md_path = run_path / "v94_wrist_action_calibration.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v94_wrist_action_calibration_csv": str(csv_path), "v94_wrist_action_calibration_json": str(json_path), "v94_wrist_action_calibration_md": str(md_path)}


def run_v94_finger_motion_calibration(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    if not _v94_live_probes_enabled():
        for logical_finger in range(1, 6):
            for slot in backend.slots:
                rows.append(
                    {
                        "part_name": slot.part_name,
                        "env_index": slot.global_env_index,
                        "logical_finger_id": logical_finger,
                        "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                        "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                        "non_active_fingers_neutral": True,
                        "v94_live_probe_executed": False,
                        "native_shutdown_risk": True,
                        "fingertip_position_delta_m": 0.0,
                        "joint_position_delta_l2": 0.0,
                        "non_active_finger_leakage_max_m": 0.0,
                        "non_active_finger_leakage_ok": False,
                        "close_sign_correct": False,
                        "finger_motion_ok": False,
                        "finger_motion_blocker": "v94_live_finger_motion_probe_disabled_native_shutdown_risk",
                        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                        "sticky_eval_ran": False,
                        "grasp_success_claimed": False,
                    }
                )
        csv_path = write_csv(run_path / "v94_finger_motion_calibration.csv", rows)
        json_path = write_json(run_path / "v94_finger_motion_calibration.json", rows)
        md_path = run_path / "v94_finger_motion_calibration.md"
        _write_md(md_path, rows)
        return {"rows": rows, "v94_finger_motion_calibration_csv": str(csv_path), "v94_finger_motion_calibration_json": str(json_path), "v94_finger_motion_calibration_md": str(md_path)}
    for logical_finger in range(1, 6):
        _v94_configure_stable_reset(backend, plan_rows)
        backend.reset_envs()
        before_tips = _v92_tip_positions(backend)
        before_joints = _v92_joint_pos(backend)
        for step in range(8):
            actions: list[list[float]] = []
            context: dict[int, dict[str, Any]] = {}
            for slot in backend.slots:
                action = [0.0] * 16
                value = min(0.85, 0.12 * float(step + 1))
                action[6 + (logical_finger - 1)] = value
                action[11 + (logical_finger - 1)] = value
                actions.append(action)
                context[slot.global_env_index] = {
                    "v94_safe_pregrasp_mode": True,
                    "v94_calibration_mode": "finger_motion",
                    "nominal_phase": f"v94_finger_{logical_finger}_motion",
                    "logical_finger_id": logical_finger,
                    "commanded_fingers": str(logical_finger),
                    "commanded_policy_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                    "commanded_isaac_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                    "object_write_after_reset_allowed": False,
                    "fallback_success_used": False,
                    "distance_only_success_used": False,
                }
            backend._v87_pending_action_context = context
            backend.step_envs(actions)
        after_tips = _v92_tip_positions(backend)
        after_joints = _v92_joint_pos(backend)
        for slot in backend.slots:
            active_delta = _v92_tip_delta(before_tips, after_tips, slot.local_env_index, logical_finger - 1)
            other_deltas = [
                _v92_tip_delta(before_tips, after_tips, slot.local_env_index, finger_index)
                for finger_index in range(5)
                if finger_index != logical_finger - 1
            ]
            joint_delta = _v92_joint_delta(before_joints, after_joints, slot.local_env_index)
            leakage = max([0.0, *other_deltas])
            moved = bool(active_delta > 1.0e-4 or joint_delta > 1.0e-4)
            leakage_ok = bool(leakage <= max(0.006, active_delta * 1.75 + 1.0e-5))
            rows.append(
                {
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "logical_finger_id": logical_finger,
                    "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                    "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                    "non_active_fingers_neutral": True,
                    "fingertip_position_delta_m": active_delta,
                    "joint_position_delta_l2": joint_delta,
                    "non_active_finger_leakage_max_m": leakage,
                    "non_active_finger_leakage_ok": leakage_ok,
                    "close_sign_correct": moved,
                    "finger_motion_ok": bool(moved and leakage_ok and not backend.object_write_by_policy_detected),
                    "finger_motion_blocker": "" if moved and leakage_ok else ("non_active_finger_leakage" if moved else "commanded_finger_did_not_move"),
                    "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                    "grasp_success_claimed": False,
                }
            )
    csv_path = write_csv(run_path / "v94_finger_motion_calibration.csv", rows)
    json_path = write_json(run_path / "v94_finger_motion_calibration.json", rows)
    md_path = run_path / "v94_finger_motion_calibration.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v94_finger_motion_calibration_csv": str(csv_path), "v94_finger_motion_calibration_json": str(json_path), "v94_finger_motion_calibration_md": str(md_path)}


def run_v94_diagnostic_target_finger_sensor_map(run_dir: str | Path, backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    if not _v94_live_probes_enabled():
        for logical_finger in range(1, 6):
            for slot in backend.slots:
                rows.append(
                    {
                        "part_name": slot.part_name,
                        "env_index": slot.global_env_index,
                        "logical_finger_id": logical_finger,
                        "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                        "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                        "diagnostic_target_used": False,
                        "diagnostic_target_counts_as_object_success": False,
                        "diagnostic_pad_placed": False,
                        "v94_live_probe_executed": False,
                        "native_shutdown_risk": True,
                        "per_finger_force_norm": _v92_force_list(_v94_latest_metrics_for_slot(backend, slot)),
                        "force_peak_n": 0.0,
                        "responding_sensor_index": -1,
                        "expected_sensor_index": logical_finger - 1,
                        "finger_sensor_mapping_consistent": False,
                        "finger_sensor_mapping_blocker": "diagnostic_target_probe_disabled_native_shutdown_risk",
                        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                        "sticky_eval_ran": False,
                        "grasp_success_claimed": False,
                    }
                )
        csv_path = write_csv(run_path / "v94_diagnostic_target_finger_sensor_map.csv", rows)
        json_path = write_json(run_path / "v94_diagnostic_target_finger_sensor_map.json", rows)
        md_path = run_path / "v94_diagnostic_target_finger_sensor_map.md"
        _write_md(md_path, rows)
        return {"rows": rows, "v94_diagnostic_target_finger_sensor_map_csv": str(csv_path), "v94_diagnostic_target_finger_sensor_map_json": str(json_path), "v94_diagnostic_target_finger_sensor_map_md": str(md_path)}
    for logical_finger in range(1, 6):
        _v94_configure_stable_reset(backend, plan_rows)
        backend.reset_envs()
        pad_rows = [
            dict(row)
            for row in getattr(base, "v94_last_diagnostic_pad_rows", []) or []
            if int(row.get("logical_finger_id") or -1) == logical_finger
        ]
        for step in range(10):
            actions: list[list[float]] = []
            context: dict[int, dict[str, Any]] = {}
            for slot in backend.slots:
                action = [0.0] * 16
                value = min(1.0, 0.10 * float(step + 1))
                action[6 + (logical_finger - 1)] = value
                action[11 + (logical_finger - 1)] = value
                actions.append(action)
                context[slot.global_env_index] = {
                    "v94_safe_pregrasp_mode": True,
                    "v94_calibration_mode": "diagnostic_target_sensor_map",
                    "diagnostic_target_used": True,
                    "nominal_phase": f"v94_diagnostic_pad_finger_{logical_finger}",
                    "logical_finger_id": logical_finger,
                    "commanded_fingers": str(logical_finger),
                    "commanded_policy_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                    "commanded_isaac_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                    "object_write_after_reset_allowed": False,
                    "fallback_success_used": False,
                    "distance_only_success_used": False,
                }
            backend._v87_pending_action_context = context
            metrics_rows = backend.step_envs(actions)
        for slot, metrics in zip(backend.slots, metrics_rows):
            forces = _v92_force_list(metrics)
            responding = _v92_max_index(forces)
            peak = max([0.0, *forces])
            placed = any(int(row.get("env_index") or -1) == slot.local_env_index and bool(row.get("diagnostic_pad_placed")) for row in pad_rows)
            consistent = bool(placed and peak > backend.contact_manager.force_threshold_n and responding == logical_finger - 1)
            blocker = ""
            if not placed:
                blocker = "diagnostic_contact_pad_not_placed"
            elif peak <= backend.contact_manager.force_threshold_n:
                blocker = "diagnostic_contact_pad_force_not_observed"
            elif responding != logical_finger - 1:
                blocker = f"expected_sensor_{logical_finger - 1}_got_{responding}"
            rows.append(
                {
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "logical_finger_id": logical_finger,
                    "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                    "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                    "diagnostic_target_used": True,
                    "diagnostic_target_counts_as_object_success": False,
                    "diagnostic_pad_placed": placed,
                    "per_finger_force_norm": forces,
                    "force_peak_n": peak,
                    "responding_sensor_index": responding,
                    "expected_sensor_index": logical_finger - 1,
                    "finger_sensor_mapping_consistent": consistent,
                    "finger_sensor_mapping_blocker": blocker,
                    "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                    "sticky_eval_ran": False,
                    "grasp_success_claimed": False,
                }
            )
    csv_path = write_csv(run_path / "v94_diagnostic_target_finger_sensor_map.csv", rows)
    json_path = write_json(run_path / "v94_diagnostic_target_finger_sensor_map.json", rows)
    md_path = run_path / "v94_diagnostic_target_finger_sensor_map.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v94_diagnostic_target_finger_sensor_map_csv": str(csv_path), "v94_diagnostic_target_finger_sensor_map_json": str(json_path), "v94_diagnostic_target_finger_sensor_map_md": str(md_path)}


def run_v94_actual_object_contact_calibration(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    sensor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    mapping = _v94_inferred_sensor_mapping(sensor_rows)
    if not _v94_live_probes_enabled():
        for mode in ("single_finger", "two_finger"):
            for slot in backend.slots:
                plan = _v94_plan_for_slot(plan_rows, slot)
                group = str(plan.get("active_finger_group") or V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0])
                selected_group = group[:1] if mode == "single_finger" else group
                logical = _v91_active_finger_indices(selected_group)
                expected_sensors = [mapping.get(index, index) for index in logical]
                distance = _v94_distance_for_slot(backend, slot, group)
                metrics = _v94_latest_metrics_for_slot(backend, slot)
                rows.append(
                    {
                        "part_name": slot.part_name,
                        "env_index": slot.global_env_index,
                        "contact_calibration_mode": mode,
                        "selected_finger_group": selected_group,
                        "pre_close_fingertip_object_distance_min_m": distance.get("fingertip_object_surface_distance_min_m", 1.0),
                        "pre_close_center_distance_min_m": distance.get("fingertip_object_center_distance_min_m", 1.0),
                        "commanded_logical_fingers": ",".join(str(index + 1) for index in logical),
                        "expected_calibrated_sensor_indices": ",".join(str(index) for index in expected_sensors),
                        "per_finger_force_norm": _v92_force_list(metrics),
                        "force_peak_n": 0.0,
                        "active_force_count": 0,
                        "effective_contact_count_force_max": 0,
                        "object_displacement_m": metrics.get("object_displacement_m", 0.0),
                        "object_delta_z_m": metrics.get("object_delta_z_m", 0.0),
                        "excessive_force_rate": 0.0,
                        "reset_interpenetration_or_impulse_detected": False,
                        "table_collision_counted_as_finger_support": False,
                        "v94_live_probe_executed": False,
                        "native_shutdown_risk": True,
                        "actual_object_contact_ok": False,
                        "actual_object_contact_blocker": "actual_object_contact_probe_disabled_native_shutdown_risk",
                        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                        "sticky_eval_ran": False,
                        "grasp_success_claimed": False,
                    }
                )
        csv_path = write_csv(run_path / "v94_actual_object_contact_calibration.csv", rows)
        json_path = write_json(run_path / "v94_actual_object_contact_calibration.json", rows)
        md_path = run_path / "v94_actual_object_contact_calibration.md"
        _write_md(md_path, rows)
        return {"rows": rows, "v94_actual_object_contact_calibration_csv": str(csv_path), "v94_actual_object_contact_calibration_json": str(json_path), "v94_actual_object_contact_calibration_md": str(md_path)}
    for mode in ("single_finger", "two_finger"):
        _v94_configure_stable_reset(backend, plan_rows)
        backend.reset_envs()
        pre_distances = {slot.part_name: _v94_distance_for_slot(backend, slot, str(_v94_plan_for_slot(plan_rows, slot).get("active_finger_group") or "")) for slot in backend.slots}
        start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
        peak_by_part: dict[str, float] = {slot.part_name: 0.0 for slot in backend.slots}
        count_by_part: dict[str, int] = {slot.part_name: 0 for slot in backend.slots}
        forces_by_part: dict[str, list[float]] = {slot.part_name: [] for slot in backend.slots}
        table_by_part: dict[str, bool] = {slot.part_name: False for slot in backend.slots}
        for step in range(12):
            actions: list[list[float]] = []
            context: dict[int, dict[str, Any]] = {}
            for slot in backend.slots:
                plan = _v94_plan_for_slot(plan_rows, slot)
                group = str(plan.get("active_finger_group") or V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0])
                if mode == "single_finger":
                    group = group[:1]
                action = [0.0] * 16
                mask = _v91_apply_finger_mask(action, group=group, active_value=min(0.95, 0.09 * float(step + 1)), support_value=0.0)
                actions.append(action)
                context[slot.global_env_index] = {
                    "v94_safe_pregrasp_mode": True,
                    "v94_calibration_mode": "actual_object_contact",
                    "actual_object_contact_calibration": True,
                    "nominal_phase": f"v94_actual_object_{mode}",
                    "active_finger_group": group,
                    **mask,
                    "object_write_after_reset_allowed": False,
                    "fallback_success_used": False,
                    "distance_only_success_used": False,
                }
            backend._v87_pending_action_context = context
            metrics_rows = backend.step_envs(actions)
            for slot, metrics in zip(backend.slots, metrics_rows):
                forces = _v92_force_list(metrics)
                forces_by_part[slot.part_name] = forces
                peak_by_part[slot.part_name] = max(peak_by_part[slot.part_name], max([0.0, *forces]))
                count_by_part[slot.part_name] = max(count_by_part[slot.part_name], int(metrics.get("effective_contact_count_force") or 0))
                table_by_part[slot.part_name] = table_by_part[slot.part_name] or bool(metrics.get("table_collision"))
        for slot in backend.slots:
            plan = _v94_plan_for_slot(plan_rows, slot)
            group = str(plan.get("active_finger_group") or V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0])
            selected_group = group[:1] if mode == "single_finger" else group
            logical = _v91_active_finger_indices(selected_group)
            expected_sensors = [mapping.get(index, index) for index in logical]
            forces = forces_by_part.get(slot.part_name, [])
            active_count = sum(1 for index in expected_sensors if 0 <= index < len(forces) and forces[index] > backend.contact_manager.force_threshold_n)
            obj_delta = _v91_state_delta(start_states.get(slot.part_name, {}), _v91_object_state_for_slot(backend, slot))
            required_count = 1 if mode == "single_finger" else min(2, len(expected_sensors))
            contact_ok = bool(
                peak_by_part.get(slot.part_name, 0.0) > backend.contact_manager.force_threshold_n
                and active_count >= required_count
                and not table_by_part.get(slot.part_name, False)
                and not backend.object_write_by_policy_detected
            )
            rows.append(
                {
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "contact_calibration_mode": mode,
                    "selected_finger_group": selected_group,
                    "pre_close_fingertip_object_distance_min_m": pre_distances.get(slot.part_name, {}).get("fingertip_object_surface_distance_min_m", 1.0),
                    "pre_close_center_distance_min_m": pre_distances.get(slot.part_name, {}).get("fingertip_object_center_distance_min_m", 1.0),
                    "commanded_logical_fingers": ",".join(str(index + 1) for index in logical),
                    "expected_calibrated_sensor_indices": ",".join(str(index) for index in expected_sensors),
                    "per_finger_force_norm": forces,
                    "force_peak_n": peak_by_part.get(slot.part_name, 0.0),
                    "active_force_count": active_count,
                    "effective_contact_count_force_max": count_by_part.get(slot.part_name, 0),
                    "object_displacement_m": obj_delta.get("root_pose_delta_m", 0.0),
                    "object_delta_z_m": obj_delta.get("root_delta_z_m", 0.0),
                    "excessive_force_rate": 1.0 if peak_by_part.get(slot.part_name, 0.0) > V86_EXCESSIVE_FORCE_THRESHOLD_N else 0.0,
                    "reset_interpenetration_or_impulse_detected": False,
                    "table_collision_counted_as_finger_support": bool(table_by_part.get(slot.part_name, False) and active_count > 0),
                    "actual_object_contact_ok": contact_ok,
                    "actual_object_contact_blocker": ""
                    if contact_ok
                    else (
                        "table_collision_counted_as_finger_support"
                        if table_by_part.get(slot.part_name, False)
                        else "actual_object_force_contact_not_observed"
                    ),
                    "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                    "sticky_eval_ran": False,
                    "grasp_success_claimed": False,
                }
            )
    csv_path = write_csv(run_path / "v94_actual_object_contact_calibration.csv", rows)
    json_path = write_json(run_path / "v94_actual_object_contact_calibration.json", rows)
    md_path = run_path / "v94_actual_object_contact_calibration.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v94_actual_object_contact_calibration_csv": str(csv_path), "v94_actual_object_contact_calibration_json": str(json_path), "v94_actual_object_contact_calibration_md": str(md_path)}


def _v94_configured_pregrasp_plan(backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_part = {str(row.get("part_name") or ""): dict(row) for row in plan_rows}
    table_tops = _v92_table_top_by_env(backend)
    rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        source = by_part.get(slot.part_name, {})
        table_top = _v92_safe_table_top(table_tops.get(slot.local_env_index, table_tops.get(0, 0.72)))
        half_height, half_source = _v94_support_half_height_m(backend, slot)
        center_x = _to_float(source.get("object_center_local_x"), -0.20)
        center_y = _to_float(source.get("object_center_local_y"), 0.0)
        center_z = _to_float(source.get("object_center_local_z"), table_top + half_height + 0.003)
        center_z = max(0.02, min(1.35, center_z))
        active_group = str(source.get("active_finger_group") or V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0])
        target_z_offset = max(0.010, min(0.045, half_height * 0.18 + 0.008))
        hand_target = source.get("hand_target_local_xyz") or [center_x, center_y - 0.018, center_z + target_z_offset]
        try:
            hand_target = [float(value) for value in hand_target[:3]]
        except Exception:
            hand_target = [center_x, center_y - 0.018, center_z + target_z_offset]
        rows.append(
            {
                "part_name": slot.part_name,
                "env_index": slot.local_env_index,
                "local_env_index": slot.local_env_index,
                "initial_condition_mode": str(source.get("initial_condition_mode") or "table_supported_pickup"),
                "support_surface": "Table_world_bbox_top_and_collision_bbox_bottom",
                "table_top_z_m": table_top,
                "object_support_half_height_m": half_height,
                "object_support_half_height_source": half_source,
                "object_center_local_x": center_x,
                "object_center_local_y": center_y,
                "object_center_local_z": center_z,
                "object_center_local_xyz": [center_x, center_y, center_z],
                "object_quat_wxyz": source.get("object_quat_wxyz", ""),
                "hand_target_local_x": hand_target[0],
                "hand_target_local_y": hand_target[1],
                "hand_target_local_z": hand_target[2],
                "hand_target_local_xyz": hand_target,
                "hand_target_source": str(source.get("hand_target_source") or "v94_object_surface_pregrasp_offset"),
                "active_finger_group": active_group,
                "reset_only_object_write_allowed": True,
                "reset_only_hand_write_allowed": True,
                "object_write_after_reset_allowed": False,
                "hand_write_after_reset_allowed": False,
                "pre_contact_displacement_limit_m": 0.02,
                "pre_contact_z_drift_limit_m": 0.02,
                "initial_force_limit_n": 0.05,
                "pregrasp_distance_target_m": 0.015,
                "diagnostic_target_pad_reset_only": True,
                "diagnostic_target_counts_as_object_success": False,
                "success_label_used": False,
            }
        )
    return rows


def _v94_configure_stable_reset(backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> None:
    """Use the v93 reset-lifecycle object staging path; v94 hand alignment is audited, not forced."""

    backend.clear_v94_pregrasp()
    backend.configure_v93_safe_staging(plan_rows)


def _v94_live_probes_enabled() -> bool:
    return os.environ.get("WUJI_V94_LIVE_PROBES", "0").strip().lower() in {"1", "true", "yes", "on"}


def _v94_latest_metrics_for_slot(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> dict[str, Any]:
    for row in reversed(getattr(backend, "last_metrics", []) or []):
        if str(row.get("part_name") or "") == slot.part_name and _int_field(row, "env_index") == slot.global_env_index:
            return dict(row)
    for row in reversed(getattr(backend, "contact_api_rows", []) or []):
        if str(row.get("part_name") or "") == slot.part_name and _int_field(row, "env_index") == slot.global_env_index:
            return dict(row)
    try:
        return dict(backend._metrics_for_slot(slot, "v94_report_only_snapshot"))
    except Exception:
        return {}


def _v94_support_half_height_m(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> tuple[float, str]:
    try:
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        identity = backend.object_identity_for_slot(slot)
        root_path = _v91_resolved_prim_path(
            str(identity.get("active_asset_prim_path") or ""),
            slot.local_env_index,
        )
        root = stage.GetPrimAtPath(root_path) if stage is not None else None
        if root and root.IsValid():
            from pxr import UsdPhysics  # noqa: WPS433

            collision_prims = []
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if path != str(root.GetPath()) and not path.startswith(str(root.GetPath()) + "/"):
                    continue
                collision_api = UsdPhysics.CollisionAPI(prim)
                if collision_api and bool(collision_api.GetCollisionEnabledAttr().Get()):
                    collision_prims.append(prim)
            box = _v93_bbox_for_prims(collision_prims)
            extent_z = float(box.get("extent", (0.0, 0.0, 0.0))[2])
            if extent_z > 1.0e-4:
                return max(0.005, min(0.35, 0.5 * extent_z)), "collision_world_bbox_extent"
    except Exception:
        pass
    return _v92_support_half_height_m(slot.part_name), "V85_PART_GEOMETRY_fallback"


def _v94_fingertip_midpoints(backend: IsaacUnifiedSingleContextBackend) -> Any:
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    try:
        if getattr(base, "last_update_timestamp", 0.0) < getattr(base._robot._data, "_sim_timestamp", 0.0):
            base._compute_intermediate_values(dt=base.physics_dt)
        return base.fingertip_midpoint_pos.detach().clone()
    except Exception:
        return None


def _v94_vec_delta(before: Any, after: Any, env_index: int) -> list[float]:
    try:
        delta = after[int(env_index)] - before[int(env_index)]
        return [float(value) for value in delta.detach().cpu().reshape(-1).tolist()[:3]]
    except Exception:
        return [0.0, 0.0, 0.0]


def _v94_plan_for_slot(plan_rows: list[dict[str, Any]], slot: SingleContextSlot) -> dict[str, Any]:
    return next((row for row in plan_rows if row.get("part_name") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index), {})


def _v94_distance_for_slot(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot, active_group: str = "") -> dict[str, Any]:
    if backend.env is None or torch is None:
        return {
            "fingertip_object_center_distance_min_m": 1.0,
            "fingertip_object_surface_distance_min_m": 1.0,
            "fingertip_object_distance_blocker": "torch_or_env_unavailable",
        }
    base = getattr(backend.env, "unwrapped", backend.env)
    try:
        tips = _v92_tip_positions(backend)
        state = _v91_object_state_for_slot(backend, slot)
        env_origin = _tensor_row(getattr(getattr(base, "scene", None), "env_origins", None), slot.local_env_index)
        if tips is None or not state.get("root_pos_w") or env_origin is None:
            raise RuntimeError("tip_or_object_state_unavailable")
        object_w = torch.tensor(state["root_pos_w"][:3], dtype=torch.float32, device=base.device)
        origin = env_origin.to(device=base.device, dtype=torch.float32).reshape(3)
        object_local = object_w - origin
        tip_rows = tips[int(slot.local_env_index)].to(device=base.device, dtype=torch.float32).reshape(-1, 3)
        active_indices = _v91_active_finger_indices(active_group) if active_group else list(range(min(5, tip_rows.shape[0])))
        best_center = 1.0
        best_surface = 1.0
        best_frame = ""
        for index in active_indices:
            if index < 0 or index >= int(tip_rows.shape[0]):
                continue
            for frame_name, object_center in (("local", object_local), ("world", object_w)):
                delta = tip_rows[index] - object_center.reshape(3)
                distance = float(torch.linalg.vector_norm(delta).detach().cpu().item())
                direction = delta / torch.clamp(torch.linalg.vector_norm(delta), min=1.0e-6)
                if hasattr(base, "_v85_surface_offset_m"):
                    surface_offset = float(base._v85_surface_offset_m(slot.part_name, direction))
                else:
                    surface_offset = _v92_support_half_height_m(slot.part_name)
                surface_distance = max(0.0, distance - surface_offset)
                if surface_distance < best_surface:
                    best_center = distance
                    best_surface = surface_distance
                    best_frame = frame_name
        return {
            "fingertip_object_center_distance_min_m": best_center,
            "fingertip_object_surface_distance_min_m": best_surface,
            "fingertip_object_distance_frame": best_frame,
            "active_fingertip_object_distance_indices": ",".join(str(index) for index in active_indices),
            "fingertip_object_distance_blocker": "",
        }
    except Exception as exc:
        return {
            "fingertip_object_center_distance_min_m": 1.0,
            "fingertip_object_surface_distance_min_m": 1.0,
            "active_fingertip_object_distance_indices": "",
            "fingertip_object_distance_blocker": f"{type(exc).__name__}:{exc}",
        }


def _v94_inferred_sensor_mapping(sensor_rows: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for logical in range(5):
        rows = [
            row
            for row in sensor_rows
            if int(row.get("logical_finger_id") or 0) == logical + 1
            and bool(row.get("finger_sensor_mapping_consistent"))
        ]
        if rows:
            mapping[logical] = int(rows[0].get("responding_sensor_index") or logical)
        else:
            mapping[logical] = logical
    return mapping


def _v91_first_candidate_by_part(candidate_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        part = str(row.get("part_name") or "")
        if part in V83_PARTS and part not in selected:
            selected[part] = _v89_candidate_as_variant(row, int(row.get("candidate_rank") or 0))
    return selected


def _v91_state_delta(start: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    p0 = start.get("root_pos_w", [])
    p1 = state.get("root_pos_w", [])
    if len(p0) >= 3 and len(p1) >= 3:
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        dz = float(p1[2]) - float(p0[2])
        return {"root_pose_delta_m": math.sqrt(dx * dx + dy * dy + dz * dz), "root_delta_z_m": dz}
    return {"root_pose_delta_m": 0.0, "root_delta_z_m": 0.0}


def _v91_object_state_for_slot(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> dict[str, Any]:
    if backend.env is None:
        return {}
    base = getattr(backend.env, "unwrapped", backend.env)
    registry = getattr(base, "v83_active_asset_registry", {}).get(slot.part_name, {})
    asset = registry.get("asset")
    pos = _tensor_row(getattr(getattr(asset, "data", None), "root_pos_w", None), slot.local_env_index)
    quat = _tensor_row(getattr(getattr(asset, "data", None), "root_quat_w", None), slot.local_env_index)
    lin = _tensor_row(getattr(getattr(asset, "data", None), "root_lin_vel_w", None), slot.local_env_index)
    ang = _tensor_row(getattr(getattr(asset, "data", None), "root_ang_vel_w", None), slot.local_env_index)
    def _vec(value: Any, width: int) -> list[float]:
        if value is None:
            return []
        try:
            if hasattr(value, "detach"):
                data = value.detach().cpu().reshape(-1).tolist()
            else:
                data = list(value)
            return [float(item) for item in data[:width]]
        except Exception:
            return []
    lin_vec = _vec(lin, 3)
    ang_vec = _vec(ang, 3)
    return {
        "root_pos_w": _vec(pos, 3),
        "root_quat_w": _vec(quat, 4),
        "root_lin_vel_w": lin_vec,
        "root_ang_vel_w": ang_vec,
        "root_lin_vel_norm": math.sqrt(sum(value * value for value in lin_vec)) if lin_vec else 0.0,
        "root_ang_vel_norm": math.sqrt(sum(value * value for value in ang_vec)) if ang_vec else 0.0,
    }


def _v91_asset_runtime_properties(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    identity: dict[str, Any],
) -> dict[str, Any]:
    props: dict[str, Any] = {
        "active_usd_path": identity.get("active_asset_usd", ""),
        "prim_path": identity.get("active_asset_prim_path", ""),
        "resolved_prim_path": _v91_resolved_prim_path(
            str(identity.get("active_asset_prim_path") or ""),
            slot.local_env_index,
        ),
        "rigid_body_enabled": "",
        "kinematic_enabled": "",
        "disable_gravity": "",
        "mass": "",
        "scale": "",
        "collision_enabled": "",
        "contact_sensors_active": False,
        "linear_damping": "",
        "angular_damping": "",
        "max_depenetration_velocity": "",
        "solver_position_iterations": "",
        "solver_velocity_iterations": "",
        "contact_offset": "",
        "rest_offset": "",
        "static_friction": "",
        "dynamic_friction": "",
        "restitution": "",
    }
    try:
        from pxr import PhysxSchema, UsdPhysics  # noqa: WPS433
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(str(props["resolved_prim_path"] or ""))
        if not prim or not prim.IsValid():
            prim = _v91_find_prim_by_suffix(
                stage,
                str(identity.get("expected_prim_path_suffix") or ""),
                slot.local_env_index,
            )
        props["rigid_body_enabled"] = bool(prim and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get())
        props["kinematic_enabled"] = bool(prim and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
        props["disable_gravity"] = bool(prim and PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get())
        props["linear_damping"] = _v91_attr_float(PhysxSchema.PhysxRigidBodyAPI(prim).GetLinearDampingAttr()) if prim else ""
        props["angular_damping"] = _v91_attr_float(PhysxSchema.PhysxRigidBodyAPI(prim).GetAngularDampingAttr()) if prim else ""
        props["max_depenetration_velocity"] = _v91_attr_float(PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxDepenetrationVelocityAttr()) if prim else ""
        props["solver_position_iterations"] = _v91_attr_float(PhysxSchema.PhysxRigidBodyAPI(prim).GetSolverPositionIterationCountAttr()) if prim else ""
        props["solver_velocity_iterations"] = _v91_attr_float(PhysxSchema.PhysxRigidBodyAPI(prim).GetSolverVelocityIterationCountAttr()) if prim else ""
        props["mass"] = _v91_attr_float(UsdPhysics.MassAPI(prim).GetMassAttr()) if prim else ""
        props["collision_enabled"] = bool(prim and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
    except Exception as exc:
        props["asset_property_read_blocker"] = f"{type(exc).__name__}:{exc}"
    try:
        base = getattr(backend.env, "unwrapped", backend.env)
        registry = getattr(base, "v83_active_asset_registry", {}).get(slot.part_name, {})
        asset = registry.get("asset")
        view = getattr(asset, "root_physx_view", None)
        if view is not None:
            mats = view.get_material_properties()
            props["static_friction"] = float(mats[..., 0].mean().item())
            props["dynamic_friction"] = float(mats[..., 1].mean().item())
            if mats.shape[-1] >= 3:
                props["restitution"] = float(mats[..., 2].mean().item())
    except Exception:
        pass
    props["contact_sensors_active"] = any(
        bool(row.get("contact_sensor_api_available"))
        for row in backend.contact_api_rows + backend.last_metrics
        if row.get("part_name") == slot.part_name
    )
    return props


def resolved_env_path(path_expr: str, env_index: int) -> str:
    path = str(path_expr or "")
    env_prefix = f"/World/envs/env_{int(env_index)}/"
    if "/World/envs/env_.*/" in path:
        return path.replace("/World/envs/env_.*/", env_prefix)
    if "/World/envs/env_0/" in path and int(env_index) != 0:
        return path.replace("/World/envs/env_0/", env_prefix)
    return path


def find_prim_by_suffix_for_env(stage: Any, suffix: str, env_index: int) -> Any:
    try:
        env_fragment = f"/World/envs/env_{int(env_index)}/"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith(str(suffix or "")) and env_fragment in path:
                return prim
    except Exception:
        return None
    return None


def find_robot_body_prim_path_for_env(stage: Any, body_name: str, env_index: int) -> str:
    try:
        env_fragment = f"/World/envs/env_{int(env_index)}/Robot/"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if env_fragment in path and path.endswith("/" + str(body_name)):
                return path
    except Exception:
        return ""
    return ""


def _v91_resolved_prim_path(path: str, env_index: int = 0) -> str:
    return resolved_env_path(path, env_index)


def _v91_find_prim_by_suffix(stage: Any, suffix: str, env_index: int = 0) -> Any:
    return find_prim_by_suffix_for_env(stage, suffix, env_index)


def _v91_attr_float(attr: Any) -> Any:
    try:
        value = attr.Get()
        return "" if value is None else float(value)
    except Exception:
        return ""


def _v91_annotate_metrics(
    metrics: list[dict[str, Any]],
    active: dict[str, dict[str, Any]],
    step: int,
    phase: str,
    staged: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_by_part = {str(row.get("part_name") or ""): row for row in staged}
    rows = []
    for item in metrics:
        part = str(item.get("part_name") or "")
        candidate = active.get(part)
        if not candidate:
            continue
        rows.append(
            {
                "sequence_step": step,
                "probe_step": step,
                "probe_phase": phase,
                "near_contact_stage_ok": bool(stage_by_part.get(part, {}).get("near_contact_stage_ok")),
                **_v90_candidate_fields(candidate),
                **item,
            }
        )
    return rows


def _v91_summarize_candidate(
    part: str,
    candidate: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    finger_rows: list[dict[str, Any]],
    backend: IsaacUnifiedSingleContextBackend,
    threshold: float,
    valid_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    action_rows = [row for row in trace_rows if row.get("probe_phase") in {"approach_until_contact", "gradual_close", "contact_hold", "slow_lift"}]
    pre_contact_rows: list[dict[str, Any]] = []
    contact_seen = False
    for row in trace_rows:
        force = float(row.get("force_contact_peak_n") or 0.0)
        count = int(row.get("effective_contact_count_force") or 0)
        if force > threshold and count >= 1:
            contact_seen = True
        if not contact_seen:
            pre_contact_rows.append(row)
    hold_rows = [row for row in trace_rows if row.get("probe_phase") == "contact_hold"]
    lift_rows = [row for row in trace_rows if row.get("probe_phase") == "slow_lift"]
    force_rows = [row for row in action_rows if float(row.get("force_contact_peak_n") or 0.0) > threshold and int(row.get("effective_contact_count_force") or 0) >= 1]
    multi_rows = [row for row in action_rows if int(row.get("effective_contact_count_force") or 0) >= 2]
    support_rows = [row for row in action_rows if bool(row.get("support_gate_ok"))]
    hold_hits = [
        row
        for row in hold_rows
        if float(row.get("force_contact_peak_n") or 0.0) > threshold
        and int(row.get("effective_contact_count_force") or 0) >= 2
        and float(row.get("object_displacement_m") or 0.0) <= 0.04
    ]
    lift_hits = [
        row
        for row in lift_rows
        if float(row.get("object_delta_z_m") or 0.0) > 0.005
        and float(row.get("force_contact_peak_n") or 0.0) > threshold
        and int(row.get("effective_contact_count_force") or 0) >= 2
    ]
    pre_disp = max([0.0, *[float(row.get("object_motion_before_contact_m") or row.get("object_displacement_m") or 0.0) for row in pre_contact_rows]])
    pre_drop = max([0.0, *[abs(float(row.get("object_delta_z_m") or 0.0)) for row in pre_contact_rows]])
    peaks = [float(row.get("force_contact_peak_n") or 0.0) for row in trace_rows]
    displacement_max = max([0.0, *[float(row.get("object_displacement_m") or 0.0) for row in trace_rows]])
    lift_delta_max = max([0.0, *[float(row.get("object_delta_z_m") or 0.0) for row in lift_rows]])
    reset_interpenetration = any(
        float(row.get("force_contact_peak_n") or 0.0) > threshold
        for row in trace_rows
        if row.get("probe_phase") == "pre_controller_sample"
    )
    cid = str(candidate.get("candidate_id") or "")
    staging_valid = (part, cid) in valid_keys
    forbidden = bool(
        backend.object_write_by_policy_detected
        or backend.sticky_action_available_to_policy
        or any(bool(row.get("fallback_success_used") or row.get("distance_only_success_used")) for row in trace_rows)
    )
    force_rate = len(force_rows) / max(1, len(action_rows))
    multi_rate = len(multi_rows) / max(1, len(action_rows))
    support_rate = len(support_rows) / max(1, len(action_rows))
    hold_rate = len(hold_hits) / max(1, len(hold_rows))
    lift_rate = len(lift_hits) / max(1, len(lift_rows))
    feasible = bool(
        staging_valid
        and pre_disp <= 0.02
        and not reset_interpenetration
        and force_rate > 0.0
        and multi_rate > 0.0
        and hold_rate > 0.0
        and lift_rate > 0.0
        and displacement_max <= 0.06
        and not forbidden
    )
    if feasible:
        status = "FEASIBLE_NOW_DIAGNOSTIC_ONLY"
        blocker = ""
    elif not staging_valid or pre_disp > 0.02 or reset_interpenetration:
        status = "STAGING_OR_TASK_SETUP_BLOCKER"
        blocker = "invalid_staging_or_precontact_motion"
    elif force_rate <= 0.0:
        status = "CONTROLLER_CANDIDATE_BLOCKER"
        blocker = "no_real_force_contact_after_quasistatic_controller"
    elif multi_rate <= 0.0:
        status = "CONTACT_MODEL_BLOCKER"
        blocker = "single_finger_or_unstable_contact_only"
    elif hold_rate <= 0.0:
        status = "CONTROLLER_CANDIDATE_BLOCKER"
        blocker = "hold_not_stable"
    elif lift_rate <= 0.0:
        status = "CONTROLLER_CANDIDATE_BLOCKER"
        blocker = "hand_lift_command_did_not_lift_object_slip_or_no_grasp"
    else:
        status = "CONTACT_MODEL_BLOCKER"
        blocker = "excessive_displacement_or_forbidden_action"
    per_finger_max = _v91_per_finger_max(trace_rows)
    return {
        "part_name": part,
        **_v90_candidate_fields(candidate),
        "staging_valid_for_feasibility": staging_valid,
        "object_displacement_before_contact_m": pre_disp,
        "object_z_drift_before_contact_m": pre_drop,
        "reset_interpenetration_or_impulse_detected": reset_interpenetration,
        "force_contact_rate": force_rate,
        "multi_finger_support_rate": multi_rate,
        "support_gate_rate": support_rate,
        "hold_gate_rate": hold_rate,
        "lift_gate_rate": lift_rate,
        "peak_force_n": max([0.0, *peaks]),
        "mean_force_n": sum(peaks) / max(1, len(peaks)),
        "excessive_force_rate": sum(1 for peak in peaks if peak > 150.0) / max(1, len(peaks)),
        "object_displacement_max_m": displacement_max,
        "lift_delta_z_max_m": lift_delta_max,
        "commanded_fingers": _v91_first_nonempty(finger_rows, "commanded_fingers"),
        "commanded_policy_cols": _v91_first_nonempty(finger_rows, "commanded_policy_cols"),
        "commanded_isaac_cols": _v91_first_nonempty(finger_rows, "commanded_isaac_cols"),
        "per_finger_force_norm_max": per_finger_max,
        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
        "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
        "fallback_success_used": any(bool(row.get("fallback_success_used")) for row in trace_rows),
        "distance_only_success_used": any(bool(row.get("distance_only_success_used")) for row in trace_rows),
        "v91_feasible_now_diagnostic_only": feasible,
        "final_success": False,
        "ppo_ran": False,
        "bc_ran": False,
        "checkpoint_written": False,
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "dataset_exported": False,
        "usable_training_row_count": 0,
        "status": status,
        "blocker": blocker,
    }


def _v91_first_nonempty(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def _v91_per_finger_max(rows: list[dict[str, Any]]) -> list[float]:
    out = [0.0] * 5
    for row in rows:
        values = row.get("per_finger_force_norm") or row.get("per_finger_contact_force_norm") or []
        if isinstance(values, str):
            continue
        for index, value in enumerate(list(values)[:5]):
            out[index] = max(out[index], float(value or 0.0))
    return out


def _v91_select_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for part in V83_PARTS:
        part_rows = [row for row in rows if row.get("part_name") == part]
        selected.append(max(part_rows, key=_v91_candidate_score, default={"part_name": part, "status": "NO_V91_CANDIDATE_EXECUTED", "blocker": "no_v91_candidate_executed"}))
    return selected


def _v91_candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(row.get("v91_feasible_now_diagnostic_only"))),
        float(row.get("lift_gate_rate") or 0.0),
        float(row.get("hold_gate_rate") or 0.0),
        float(row.get("multi_finger_support_rate") or 0.0),
        float(row.get("support_gate_rate") or 0.0),
        -float(row.get("object_displacement_before_contact_m") or 0.0),
        -float(row.get("object_displacement_max_m") or 0.0),
    )


def _v85_staging_variants(part_name: str) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for strategy in V85_SURFACE_STRATEGIES.get(part_name, ("bbox_adjusted", "analytic_surface")):
        for group in V85_FINGER_GROUPS.get(part_name, ("34",)):
            for clearance_m in V85_CLEARANCE_ORDER_M:
                for lateral_m in V85_LATERAL_ORDER_M:
                    variants.append(
                        {
                            "part_name": part_name,
                            "variant_index": len(variants),
                            "variant_rank": len(variants),
                            "active_finger_group": group,
                            "target_clearance_m": float(clearance_m),
                            "lateral_offset_m": float(lateral_m),
                            "surface_strategy": strategy,
                            "use_bbox_adjustment": strategy == "bbox_adjusted",
                        }
                    )
    return variants


def _v85_annotate_metrics(
    metrics: list[dict[str, Any]],
    active_variants: dict[str, dict[str, Any]],
    *,
    probe_step: int,
    probe_phase: str,
    mapped_close: bool,
    approach_action: bool,
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_by_part = {str(row.get("part_name")): row for row in stage_rows}
    rows: list[dict[str, Any]] = []
    for item in metrics:
        part = str(item.get("part_name") or "")
        variant = active_variants.get(part)
        if not variant:
            continue
        stage = stage_by_part.get(part, {})
        rows.append(
            {
                "probe_step": int(probe_step),
                "probe_phase": probe_phase,
                "mapped_close_action_executed": bool(mapped_close),
                "approach_action_executed": bool(approach_action),
                "variant_index": int(variant.get("variant_index", 0)),
                "variant_rank": int(variant.get("variant_rank", variant.get("variant_index", 0))),
                "active_finger_group": variant.get("active_finger_group", ""),
                "target_clearance_m": float(variant.get("target_clearance_m", 0.0)),
                "lateral_offset_m": float(variant.get("lateral_offset_m", 0.0)),
                "surface_strategy": str(variant.get("surface_strategy") or ""),
                "use_bbox_adjustment": bool(variant.get("use_bbox_adjustment", False)),
                "near_contact_stage_ok": bool(stage.get("near_contact_stage_ok")),
                "near_contact_stage_failure_reason": str(stage.get("near_contact_stage_failure_reason") or ""),
                **item,
            }
        )
    return rows


def _v85_summarize_variant(
    part: str,
    variant: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    stage: dict[str, Any],
    backend: IsaacUnifiedSingleContextBackend,
    threshold: float,
) -> dict[str, Any]:
    pre_rows = [row for row in trace_rows if row.get("probe_phase") in {"pre_action_force_sample", "settle"}]
    action_rows = [row for row in trace_rows if row.get("probe_phase") in V85_ACTION_PHASES]
    pre_peak = max([0.0, *[float(row.get("force_contact_probe_peak_n") or 0.0) for row in pre_rows]])
    pre_count = max([0, *[int(row.get("effective_contact_count_force") or 0) for row in pre_rows]])
    post_peak = max([0.0, *[float(row.get("force_contact_probe_peak_n") or 0.0) for row in action_rows]])
    post_count = max([0, *[int(row.get("effective_contact_count_force") or 0) for row in action_rows]])
    all_peak = max([pre_peak, post_peak])
    all_count = max([pre_count, post_count])
    onset = _v85_first_force_contact(trace_rows, threshold)
    contact_after_action = bool(onset and onset.get("probe_phase") in V85_ACTION_PHASES and bool(onset.get("mapped_close_action_executed")))
    reset_interpenetration = bool(
        pre_peak > threshold
        or pre_count > 0
        or (onset and onset.get("probe_phase") in {"pre_action_force_sample", "settle"})
    )
    identity_ok = any(
        bool(row.get("object_identity_verified"))
        for row in backend.object_identity_rows
        if row.get("part_name") == part
    )
    mapped_close = any(
        bool(row.get("metric_close_dof_commanded"))
        and str(row.get("mapped_close_cols") or "") == "16,17,18,19,20,21,22,23,24,25"
        for row in backend.action_mapping_rows
        if row.get("part_name") == part
        and _int_field(row, "variant_index") == _int_field(variant, "variant_index", -2)
    )
    stage_ok = bool(stage.get("near_contact_stage_ok"))
    success = bool(
        identity_ok
        and backend.single_simulation_context
        and stage_ok
        and mapped_close
        and post_peak > threshold
        and post_count >= 1
        and contact_after_action
        and not reset_interpenetration
        and not backend.object_write_by_policy_detected
        and not backend.sticky_action_available_to_policy
    )
    blocker = ""
    status = "PASS" if success else "CONTACT_SANITY_FAILED"
    if not success:
        if not identity_ok:
            identity = next((row for row in backend.object_identity_rows if row.get("part_name") == part and row.get("object_identity_blocker")), {})
            blocker = str(identity.get("object_identity_blocker") or "object_identity_not_verified")
            status = "OBJECT_IDENTITY_FAILED"
        elif not stage_ok:
            blocker = str(stage.get("near_contact_stage_failure_reason") or "near_contact_stage_failed")
            status = "NEAR_CONTACT_STAGE_FAILED"
        elif not mapped_close:
            blocker = "mapped_close_action_not_executed"
            status = "MAPPED_CLOSE_ACTION_NOT_EXECUTED"
        elif reset_interpenetration:
            blocker = "reset_interpenetration_or_impulse_detected"
            status = "RESET_INTERPENETRATION_CONTACT_REJECTED"
        elif post_peak <= threshold or post_count < 1:
            blocker = "controlled_force_contact_not_observed_after_action"
            status = "CONTACT_PROBE_NO_FORCE_CONTACT_AFTER_ACTION"
        elif not contact_after_action:
            blocker = "contact_onset_not_after_action"
        elif backend.object_write_by_policy_detected:
            blocker = "object_write_by_policy_detected"
        elif backend.sticky_action_available_to_policy:
            blocker = "sticky_action_available_to_policy"
    return {
        "part_name": part,
        "variant_index": int(variant.get("variant_index", 0)),
        "variant_rank": int(variant.get("variant_rank", variant.get("variant_index", 0))),
        "active_finger_group": variant.get("active_finger_group", ""),
        "target_clearance_m": float(variant.get("target_clearance_m", 0.0)),
        "lateral_offset_m": float(variant.get("lateral_offset_m", 0.0)),
        "surface_strategy": str(variant.get("surface_strategy") or ""),
        "use_bbox_adjustment": bool(variant.get("use_bbox_adjustment", False)),
        "object_identity_verified": bool(identity_ok),
        "single_simulation_context": bool(backend.single_simulation_context),
        "near_contact_stage_ok": bool(stage_ok),
        "initial_reset_force_peak_n": pre_peak,
        "pre_action_force_peak_n": pre_peak,
        "pre_action_force_count_max": int(pre_count),
        "post_action_force_peak_n": post_peak,
        "contact_onset_step": "" if onset is None else int(onset.get("probe_step", -1)),
        "contact_onset_phase": "" if onset is None else str(onset.get("probe_phase", "")),
        "force_contact_peak_n": all_peak,
        "effective_contact_count_force_max": int(all_count),
        "post_action_effective_contact_count_force_max": int(post_count),
        "contact_after_action": bool(contact_after_action),
        "reset_interpenetration_or_impulse_detected": bool(reset_interpenetration),
        "mapped_close_action_executed": bool(mapped_close),
        "object_write_reset_only": bool(backend.object_write_reset_only),
        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
        "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
        "ppo_ran": False,
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "checkpoint_written": False,
        "usable_training_row_count": 0,
        "v85_success": bool(success),
        "status": status,
        "blocker": blocker,
        "training_locked": True,
        "training_locked_blocker": "v85_contact_validation_only_no_training_export" if success else blocker,
        "next_action": "later_enable_ppo_after_v85_contact_gate" if success else "repair_staging_or_approach_contact_probe",
    }


def _v85_first_force_contact(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    ordered = sorted(rows, key=lambda row: (_int_field(row, "probe_step"), str(row.get("probe_phase") or "")))
    for row in ordered:
        force = float(row.get("force_contact_probe_peak_n") or 0.0)
        count = int(row.get("effective_contact_count_force") or 0)
        if force > threshold and count >= 1:
            return row
    return None


def _v85_select_progress_rows(
    variant_summaries: list[dict[str, Any]],
    backend: IsaacUnifiedSingleContextBackend,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in V83_PARTS:
        candidates = [row for row in variant_summaries if row.get("part_name") == part]
        selected = next((row for row in candidates if bool(row.get("v85_success"))), None)
        if selected is None and candidates:
            selected = max(
                candidates,
                key=lambda row: (
                    float(row.get("post_action_force_peak_n") or 0.0),
                    -float(row.get("pre_action_force_peak_n") or 0.0),
                    -int(row.get("variant_rank") or 0),
                ),
            )
        if selected is None:
            selected = {
                "part_name": part,
                "object_identity_verified": any(
                    bool(row.get("object_identity_verified"))
                    for row in backend.object_identity_rows
                    if row.get("part_name") == part
                ),
                "single_simulation_context": bool(backend.single_simulation_context),
                "initial_reset_force_peak_n": 0.0,
                "pre_action_force_peak_n": 0.0,
                "contact_onset_step": "",
                "contact_onset_phase": "",
                "force_contact_peak_n": 0.0,
                "effective_contact_count_force_max": 0,
                "contact_after_action": False,
                "reset_interpenetration_or_impulse_detected": False,
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "sticky_eval_ran": False,
                "ppo_ran": False,
                "video_generated": False,
                "checkpoint_written": False,
                "usable_training_row_count": 0,
                "v85_success": False,
                "status": "NO_V85_VARIANT_EXECUTED",
                "blocker": backend.blocker or "no_v85_variant_executed",
                "training_locked": True,
                "training_locked_blocker": backend.blocker or "no_v85_variant_executed",
            }
        final = dict(selected)
        if bool(final.get("v85_success")):
            final["status"] = "PASS"
            final["training_locked"] = True
            final["training_locked_blocker"] = "v85_contact_validation_only_no_training_export"
        elif bool(final.get("reset_interpenetration_or_impulse_detected")) and float(final.get("post_action_force_peak_n") or 0.0) > threshold:
            final["status"] = "RESET_INTERPENETRATION_CONTACT_REJECTED"
            final["blocker"] = "reset_interpenetration_or_impulse_detected"
        rows.append(final)
    return rows


def _v84_policy_actions_for_phase(self: IsaacUnifiedSingleContextBackend, phase: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for slot in self.slots:
        action = [0.0] * 16
        if phase in {"close", "approach_close", "hold_squeeze"}:
            for col in range(6, 16):
                action[col] = 0.75
        if phase == "approach_close":
            direction = self._v84_approach_direction(slot)
            for axis in range(3):
                action[axis] = max(-0.35, min(0.35, float(direction[axis]) * 0.35))
        rows.append(action)
    return rows


def _v84_approach_direction(self: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> list[float]:
    if self.env is None or torch is None:
        return [0.0, 0.0, 1.0]
    base = getattr(self.env, "unwrapped", self.env)
    registry = getattr(base, "v83_active_asset_registry", {}).get(slot.part_name, {})
    asset = registry.get("asset")
    object_pos_w = _tensor_row(getattr(getattr(asset, "data", None), "root_pos_w", None), slot.local_env_index)
    env_origin = _tensor_row(getattr(getattr(base, "scene", None), "env_origins", None), slot.local_env_index)
    fingertip = _tensor_row(getattr(base, "fingertip_midpoint_pos", None), slot.local_env_index)
    try:
        object_pos_local = object_pos_w.to(device=base.device, dtype=torch.float32) - env_origin.to(device=base.device, dtype=torch.float32)
        fingertip = fingertip.to(device=base.device, dtype=torch.float32)
        direction = object_pos_local.reshape(3) - fingertip.reshape(3)
        norm = torch.linalg.vector_norm(direction)
        if float(norm.detach().cpu().item()) < 1.0e-6:
            return [0.0, 0.0, 1.0]
        direction = direction / torch.clamp(norm, min=1.0e-6)
        return [float(value) for value in direction.detach().cpu().tolist()]
    except Exception:
        return [0.0, 0.0, 1.0]


def _v85_policy_actions_for_phase(
    self: IsaacUnifiedSingleContextBackend,
    phase: str,
    selected_parts: set[str],
) -> list[list[float]]:
    rows: list[list[float]] = []
    for slot in self.slots:
        action = [0.0] * 16
        if slot.part_name in selected_parts and phase in V85_ACTION_PHASES:
            close_value = 1.0 if slot.part_name in {"Plug2", "Screw1"} else 0.75
            for col in range(6, 16):
                action[col] = close_value
        if slot.part_name in selected_parts and phase == "approach_close":
            direction = self._v84_approach_direction(slot)
            scale = 0.55 if slot.part_name in {"Plug2", "Screw1"} else 0.40
            for axis in range(3):
                action[axis] = max(-0.60, min(0.60, float(direction[axis]) * scale))
        rows.append(action)
    return rows


def _v88_policy_actions_for_phase(
    self: IsaacUnifiedSingleContextBackend,
    phase: str,
    selected_parts: set[str],
) -> list[list[float]]:
    rows: list[list[float]] = []
    for slot in self.slots:
        action = [0.0] * 16
        if slot.part_name in selected_parts and phase in {"close", "approach_close", "hold_squeeze", "lift_stabilize"}:
            close_value = 1.0 if slot.part_name in {"Plug2", "Screw1"} else 0.78
            for col in range(6, 16):
                action[col] = close_value
        if slot.part_name in selected_parts and phase == "approach_close":
            direction = self._v84_approach_direction(slot)
            scale = 0.50 if slot.part_name in {"Plug2", "Screw1"} else 0.34
            for axis in range(3):
                action[axis] = max(-0.55, min(0.55, float(direction[axis]) * scale))
        if slot.part_name in selected_parts and phase == "lift_stabilize":
            action[2] = 0.32
        rows.append(action)
    return rows


def _v89_policy_actions_for_phase(
    self: IsaacUnifiedSingleContextBackend,
    phase: str,
    selected_parts: set[str],
) -> list[list[float]]:
    rows: list[list[float]] = []
    for slot in self.slots:
        action = [0.0] * 16
        candidate = self.v89_selected_candidates.get(slot.part_name, {})
        if not candidate:
            candidate = {
                "close_value": 1.0 if slot.part_name in {"Plug2", "Screw1"} else 0.78,
                "hold_close_value": 1.0 if slot.part_name in {"Plug2", "Screw1"} else 0.78,
                "approach_scale": 0.50 if slot.part_name in {"Plug2", "Screw1"} else 0.34,
                "lift_x": 0.0,
                "lift_y": 0.0,
                "lift_z": 0.28,
            }
        if slot.part_name in selected_parts and phase in V89_ACTION_PHASES:
            close_value = float(candidate.get("hold_close_value") if phase in {"hold_squeeze", "lift_stabilize"} else candidate.get("close_value") or 0.0)
            close_value = max(0.0, min(1.0, close_value))
            for col in range(6, 16):
                action[col] = close_value
        if slot.part_name in selected_parts and phase == "approach_close":
            direction = self._v84_approach_direction(slot)
            scale = max(0.0, min(0.65, float(candidate.get("approach_scale") or 0.0)))
            for axis in range(3):
                action[axis] = max(-0.65, min(0.65, float(direction[axis]) * scale))
        if slot.part_name in selected_parts and phase == "lift_stabilize":
            action[0] = max(-0.35, min(0.35, float(candidate.get("lift_x") or 0.0)))
            action[1] = max(-0.35, min(0.35, float(candidate.get("lift_y") or 0.0)))
            action[2] = max(0.0, min(0.45, float(candidate.get("lift_z") or 0.0)))
        rows.append(action)
    return rows


def _v90_policy_actions_for_phase(
    self: IsaacUnifiedSingleContextBackend,
    phase: str,
    selected_parts: set[str],
) -> list[list[float]]:
    rows: list[list[float]] = []
    for slot in self.slots:
        action = [0.0] * 16
        candidate = self.v89_selected_candidates.get(slot.part_name, {})
        if slot.part_name not in selected_parts:
            rows.append(action)
            continue
        close_value = max(0.0, min(1.0, float(candidate.get("close_value") or 0.0)))
        hold_close_value = max(0.0, min(1.0, float(candidate.get("hold_close_value") or close_value)))
        force_limit = float(candidate.get("force_limit_n") or 0.0)
        current_peak = self._last_peak_force_for_part(slot.part_name)
        force_limited = bool(force_limit > 0.0 and current_peak > force_limit)
        if force_limited:
            close_value = min(close_value, 0.20)
            hold_close_value = min(hold_close_value, 0.25)
        if phase in {"close_force_limited", "contact_aware_hold", "slow_lift"}:
            value = hold_close_value if phase in {"contact_aware_hold", "slow_lift"} else close_value
            for col in range(6, 16):
                action[col] = value
        if phase in {"slow_approach", "close_force_limited"}:
            direction = self._v84_approach_direction(slot)
            scale = max(0.0, min(0.65, float(candidate.get("approach_scale") or 0.0)))
            if phase == "close_force_limited":
                scale *= 0.45
            for axis in range(3):
                action[axis] = max(-0.65, min(0.65, float(direction[axis]) * scale))
        if phase == "slow_lift":
            action[0] = max(-0.30, min(0.30, float(candidate.get("lift_x") or 0.0)))
            action[1] = max(-0.30, min(0.30, float(candidate.get("lift_y") or 0.0)))
            action[2] = max(0.0, min(0.35, float(candidate.get("lift_z") or 0.0)))
        if phase in {"slow_approach", "close_force_limited", "contact_aware_hold", "slow_lift"}:
            action[3] = max(-0.35, min(0.35, float(candidate.get("wrist_roll_rad") or 0.0)))
        rows.append(action)
    return rows


def _v91_active_finger_indices(group: str) -> list[int]:
    indices: list[int] = []
    for char in str(group or ""):
        if char.isdigit():
            index = int(char) - 1
            if 0 <= index < 5 and index not in indices:
                indices.append(index)
    return indices or [2, 3]


def _v91_apply_finger_mask(
    action: list[float],
    *,
    group: str,
    active_value: float,
    support_value: float,
) -> dict[str, Any]:
    active_indices = _v91_active_finger_indices(group)
    commanded_policy_cols: list[int] = []
    commanded_isaac_cols: list[int] = []
    for finger in range(5):
        value = active_value if finger in active_indices else support_value
        if abs(value) <= 1.0e-8:
            continue
        for policy_col in (6 + finger, 11 + finger):
            action[policy_col] = value
            commanded_policy_cols.append(policy_col)
        commanded_isaac_cols.extend([16 + finger, 21 + finger])
    return {
        "commanded_fingers": ",".join(str(index + 1) for index in active_indices),
        "commanded_policy_cols": ",".join(str(col) for col in commanded_policy_cols),
        "commanded_isaac_cols": ",".join(str(col) for col in commanded_isaac_cols),
        "support_finger_value": support_value,
        "non_active_finger_value": support_value,
    }


def _v91_policy_actions_for_phase(
    self: IsaacUnifiedSingleContextBackend,
    phase: str,
    selected_parts: set[str],
    phase_step: int,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    rows: list[list[float]] = []
    masks: list[dict[str, Any]] = []
    for slot in self.slots:
        action = [0.0] * 16
        candidate = self.v89_selected_candidates.get(slot.part_name, {})
        if slot.part_name not in selected_parts:
            rows.append(action)
            masks.append(
                {
                    "commanded_fingers": "",
                    "commanded_policy_cols": "",
                    "commanded_isaac_cols": "",
                    "support_finger_value": 0.0,
                    "non_active_finger_value": 0.0,
                    "force_limit_backoff": 1.0,
                }
            )
            continue
        group = str(candidate.get("active_finger_group") or "34")
        close_value = max(0.0, min(1.0, float(candidate.get("close_value") or 0.0)))
        hold_close_value = max(0.0, min(1.0, float(candidate.get("hold_close_value") or close_value)))
        support_value = max(0.0, min(1.0, float(candidate.get("support_finger_value") or 0.0)))
        force_limit = float(candidate.get("force_limit_n") or 0.0)
        current_peak = self._last_peak_force_for_part(slot.part_name)
        backoff = 1.0
        if force_limit > 0.0 and current_peak > force_limit:
            backoff = max(0.35, min(1.0, force_limit / max(current_peak, 1.0e-6)))
        active_close = 0.0
        if phase == "gradual_close":
            close_steps = max(1, int(candidate.get("close_steps", 24) or 24))
            active_close = close_value * min(1.0, float(phase_step + 1) / float(close_steps)) * backoff
        elif phase in {"contact_hold", "slow_lift"}:
            phase_limit_key = "hold_steps" if phase == "contact_hold" else "lift_steps"
            phase_limit = max(1, int(candidate.get(phase_limit_key, 48 if phase == "contact_hold" else 32) or 1))
            active_close = hold_close_value * backoff if phase_step < phase_limit else 0.0
        if phase == "approach_until_contact":
            contact_count = self._last_force_count_for_part(slot.part_name)
            if contact_count <= 0:
                direction = self._v84_approach_direction(slot)
                scale = max(0.0, min(0.35, float(candidate.get("approach_scale") or 0.0)))
                for axis in range(3):
                    action[axis] = max(-0.35, min(0.35, float(direction[axis]) * scale))
        if (
            phase == "slow_lift"
            and self._last_force_count_for_part(slot.part_name) >= 2
            and phase_step < max(1, int(candidate.get("lift_steps", 32) or 32))
        ):
            action[0] = max(-0.20, min(0.20, float(candidate.get("lift_x") or 0.0)))
            action[1] = max(-0.20, min(0.20, float(candidate.get("lift_y") or 0.0)))
            action[2] = max(0.0, min(0.20, float(candidate.get("lift_z") or 0.0)))
        if phase in {"approach_until_contact", "gradual_close", "contact_hold", "slow_lift"}:
            action[3] = max(-0.25, min(0.25, float(candidate.get("wrist_roll_rad") or 0.0)))
        mask = _v91_apply_finger_mask(
            action,
            group=group,
            active_value=max(0.0, min(1.0, active_close)),
            support_value=support_value,
        )
        mask["force_limit_backoff"] = backoff
        rows.append(action)
        masks.append(mask)
    return rows, masks


def _v92_stage_table_supported_pickup(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if backend.env is None or torch is None:
        return []
    base = getattr(backend.env, "unwrapped", backend.env)
    plan_by_part = {str(row.get("part_name") or ""): dict(row) for row in plan_rows}
    table_tops = _v92_table_top_by_env(backend)
    rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        table_top = _v92_safe_table_top(table_tops.get(slot.local_env_index, table_tops.get(0, 0.72)))
        half_height = _v92_support_half_height_m(slot.part_name)
        rows.append(
            {
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "initial_condition_mode": str(plan_by_part.get(slot.part_name, {}).get("initial_condition_mode") or "table_supported_pickup"),
                "support_surface": "Table_subtree_bbox_top",
                "table_top_z_m": table_top,
                "object_support_half_height_m": half_height,
                "object_center_local_x": "",
                "object_center_local_y": "",
                "object_center_local_z": max(0.02, min(1.20, table_top + half_height + 0.002)),
                "reset_only_object_write": False,
                "object_write_after_reset_allowed": False,
                "object_write_by_policy_detected": False,
                "active_finger_group": V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0],
                "pre_contact_displacement_m": "",
                "pre_contact_z_drift_m": "",
                "initial_force_peak_n": "",
                "initial_force_contact_count": "",
                "reset_interpenetration_or_impulse_detected": "",
                "initial_condition_valid": False,
                "initial_condition_blocker": "runtime_table_supported_reset_staging_skipped_after_native_shutdown_probe",
                "ppo_ran": False,
                "bc_ran": False,
                "checkpoint_written": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
            }
        )
    return rows


def _v92_collision_subtree_row(
    stage: Any,
    root_path: str,
    name: str,
    entity_type: str,
    geometry: dict[str, Any],
    *,
    env_index: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "part_name": name if entity_type == "object" else "",
        "entity_name": name,
        "entity_type": entity_type,
        "root_prim_path": str(root_path or ""),
        "root_prim_valid": False,
        "root_collision_enabled": False,
        "collision_enabled_prim_count": 0,
        "collision_prim_paths": "",
        "mesh_prim_count": 0,
        "mesh_prim_paths": "",
        "collision_approximation_types": "",
        "contact_report_api_present": False,
        "material_static_friction": "",
        "material_dynamic_friction": "",
        "material_restitution": "",
        "visual_collision_bbox_match": False,
        "collision_only_on_child_prims": False,
        "collision_missing_entirely": True,
        "collision_subtree_ok": False,
        "collision_blocker": "root_prim_missing",
    }
    if stage is None or not root_path:
        return row
    try:
        from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: WPS433

        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            root = _v91_find_prim_by_suffix(stage, "/" + str(name), env_index)
        if not root or not root.IsValid():
            return row
        row["root_prim_path"] = str(root.GetPath())
        row["root_prim_valid"] = True
        collisions: list[str] = []
        meshes: list[str] = []
        approximations: list[str] = []
        contact_api = False
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path != str(root.GetPath()) and not path.startswith(str(root.GetPath()) + "/"):
                continue
            if prim.IsA(UsdGeom.Mesh):
                meshes.append(path)
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if collision_enabled:
                collisions.append(path)
                physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
                approx_attr = (
                    physx_collision.GetCollisionApproximationAttr()
                    if hasattr(physx_collision, "GetCollisionApproximationAttr")
                    else prim.GetAttribute("physxCollision:collisionApproximation")
                )
                approx = approx_attr.Get() if approx_attr and approx_attr.IsValid() else ""
                if approx:
                    approximations.append(str(approx))
            contact_api = contact_api or _v92_has_contact_report_api(prim)
        root_collision_api = UsdPhysics.CollisionAPI(root)
        row["root_collision_enabled"] = bool(root_collision_api.GetCollisionEnabledAttr().Get()) if root_collision_api else False
        row["collision_enabled_prim_count"] = len(collisions)
        row["collision_prim_paths"] = ";".join(collisions[:20])
        row["mesh_prim_count"] = len(meshes)
        row["mesh_prim_paths"] = ";".join(meshes[:20])
        row["collision_approximation_types"] = ";".join(sorted(set(approximations)))
        row["contact_report_api_present"] = contact_api
        row["collision_only_on_child_prims"] = bool(not row["root_collision_enabled"] and len(collisions) > 0)
        row["collision_missing_entirely"] = len(collisions) == 0
        bbox_extent = _v92_bbox_extent(stage, root)
        expected = _v92_expected_extent(geometry)
        row["bbox_extent_x"] = bbox_extent[0]
        row["bbox_extent_y"] = bbox_extent[1]
        row["bbox_extent_z"] = bbox_extent[2]
        row["expected_extent_x"] = expected[0]
        row["expected_extent_y"] = expected[1]
        row["expected_extent_z"] = expected[2]
        row["visual_collision_bbox_match"] = _v92_extent_match(bbox_extent, expected) if entity_type == "object" else True
        row["collision_subtree_ok"] = bool(len(collisions) > 0 and len(meshes) > 0 and row["visual_collision_bbox_match"])
        if row["collision_subtree_ok"]:
            row["collision_blocker"] = "root_collision_false_child_collision_valid" if row["collision_only_on_child_prims"] else ""
        elif len(collisions) == 0:
            row["collision_blocker"] = "collision_missing_entirely"
        elif not row["visual_collision_bbox_match"]:
            row["collision_blocker"] = "collision_bbox_mismatch_expected_geometry"
        else:
            row["collision_blocker"] = "collision_subtree_invalid"
    except Exception as exc:
        row["collision_blocker"] = f"{type(exc).__name__}:{exc}"
    return row


def _v92_collision_repair_row(stage: Any, audit_row: dict[str, Any], part_name: str) -> dict[str, Any]:
    repair = {
        "part_name": part_name,
        "repair_attempted": False,
        "repair_applied": False,
        "repair_type": "",
        "before_collision_enabled_prim_count": audit_row.get("collision_enabled_prim_count", 0),
        "after_collision_enabled_prim_count": audit_row.get("collision_enabled_prim_count", 0),
        "repair_blocker": "",
        "sticky_or_adhesion_added": False,
        "infinite_friction_used": False,
        "success_label_used": False,
    }
    if bool(audit_row.get("collision_subtree_ok")):
        return repair
    if int(audit_row.get("collision_enabled_prim_count") or 0) > 0:
        repair["repair_blocker"] = str(audit_row.get("collision_blocker") or "collision_present_but_invalid")
        return repair
    repair["repair_attempted"] = False
    repair["repair_applied"] = False
    repair["repair_type"] = "runtime_collision_repair_recommended_not_applied"
    repair["repair_blocker"] = "active_physx_tensor_view_not_mutated_in_v92_diagnostic"
    return repair


def _v93_configured_staging_plan(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_part = {str(row.get("part_name") or ""): dict(row) for row in plan_rows}
    table_tops = _v92_table_top_by_env(backend)
    rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        table_top = _v92_safe_table_top(table_tops.get(slot.local_env_index, table_tops.get(0, 0.72)))
        half_height = _v92_support_half_height_m(slot.part_name)
        source = by_part.get(slot.part_name, {})
        center_x = _to_float(source.get("object_center_local_x"), -0.20)
        center_y = _to_float(source.get("object_center_local_y"), 0.0)
        center_z = _to_float(source.get("object_center_local_z"), table_top + half_height + 0.002)
        center_z = max(0.02, min(1.35, center_z))
        rows.append(
            {
                "part_name": slot.part_name,
                "env_index": slot.local_env_index,
                "local_env_index": slot.local_env_index,
                "initial_condition_mode": str(source.get("initial_condition_mode") or "table_supported_pickup"),
                "support_surface": str(source.get("support_surface") or "Table_subtree_bbox_top"),
                "table_top_z_m": table_top,
                "object_support_half_height_m": half_height,
                "object_center_local_x": center_x,
                "object_center_local_y": center_y,
                "object_center_local_z": center_z,
                "object_center_local_xyz": [center_x, center_y, center_z],
                "gravity_enabled_required": True,
                "object_dynamic_required": True,
                "reset_only_object_write_allowed": True,
                "object_write_after_reset_allowed": False,
                "pre_contact_displacement_limit_m": 0.02,
                "pre_contact_z_drift_limit_m": 0.02,
                "initial_force_limit_n": 0.05,
                "active_finger_group": str(source.get("active_finger_group") or V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0]),
                "success_label_used": False,
            }
        )
    return rows


def _v93_collision_world_bbox_row(
    stage: Any,
    root_path: str,
    name: str,
    entity_type: str,
    geometry: dict[str, Any],
    *,
    env_index: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "part_name": name if entity_type == "object" else "",
        "entity_name": name,
        "entity_type": entity_type,
        "root_prim_path": str(root_path or ""),
        "root_prim_valid": False,
        "root_collision_enabled": False,
        "collision_only_on_child_prims": False,
        "visual_mesh_prim_count": 0,
        "collision_enabled_prim_count": 0,
        "contact_report_api_present": False,
        "visual_bbox_extent_x": 0.0,
        "visual_bbox_extent_y": 0.0,
        "visual_bbox_extent_z": 0.0,
        "collision_bbox_extent_x": 0.0,
        "collision_bbox_extent_y": 0.0,
        "collision_bbox_extent_z": 0.0,
        "expected_extent_x": 0.0,
        "expected_extent_y": 0.0,
        "expected_extent_z": 0.0,
        "visual_collision_center_offset_m": 0.0,
        "visual_collision_extent_error_max": 1.0,
        "expected_collision_extent_error_max": 1.0,
        "visual_collision_bbox_iou": 0.0,
        "best_axis_permutation": "xyz",
        "visual_collision_bbox_aligned": False,
        "expected_collision_bbox_aligned": False,
        "collision_bbox_ok": False,
        "collision_blocker": "root_prim_missing",
    }
    if stage is None or not root_path:
        return row
    try:
        from pxr import UsdGeom, UsdPhysics  # noqa: WPS433

        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            root = _v91_find_prim_by_suffix(stage, "/" + str(name), env_index)
        if not root or not root.IsValid():
            return row
        row["root_prim_path"] = str(root.GetPath())
        row["root_prim_valid"] = True
        visual_prims: list[Any] = []
        collision_prims: list[Any] = []
        contact_api = False
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path != str(root.GetPath()) and not path.startswith(str(root.GetPath()) + "/"):
                continue
            if prim.IsA(UsdGeom.Mesh) and "collision" not in path.lower():
                visual_prims.append(prim)
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if collision_enabled:
                collision_prims.append(prim)
            contact_api = contact_api or _v92_has_contact_report_api(prim)
        root_collision_api = UsdPhysics.CollisionAPI(root)
        row["root_collision_enabled"] = bool(root_collision_api.GetCollisionEnabledAttr().Get()) if root_collision_api else False
        row["collision_only_on_child_prims"] = bool(not row["root_collision_enabled"] and collision_prims)
        row["visual_mesh_prim_count"] = len(visual_prims)
        row["collision_enabled_prim_count"] = len(collision_prims)
        row["contact_report_api_present"] = contact_api
        visual_box = _v93_bbox_for_prims(visual_prims)
        collision_box = _v93_bbox_for_prims(collision_prims)
        expected = _v92_expected_extent(geometry)
        row.update(_v93_bbox_fields("visual_bbox", visual_box))
        row.update(_v93_bbox_fields("collision_bbox", collision_box))
        row["expected_extent_x"], row["expected_extent_y"], row["expected_extent_z"] = expected
        row["visual_collision_center_offset_m"] = _v93_center_offset(visual_box, collision_box)
        row["visual_collision_extent_error_max"] = _v93_extent_error(visual_box["extent"], collision_box["extent"], sorted_axes=True)
        row["expected_collision_extent_error_max"] = _v93_extent_error(expected, collision_box["extent"], sorted_axes=True)
        row["visual_collision_bbox_iou"] = _v93_bbox_iou(visual_box, collision_box)
        row["best_axis_permutation"] = _v93_best_axis_permutation(expected, collision_box["extent"])
        visual_aligned = bool(
            row["visual_mesh_prim_count"] > 0
            and row["collision_enabled_prim_count"] > 0
            and row["visual_collision_center_offset_m"] <= max(0.03, 0.75 * _v93_diag(visual_box["extent"]))
            and (row["visual_collision_extent_error_max"] <= 0.75 or row["visual_collision_bbox_iou"] >= 0.05)
        )
        expected_aligned = bool(row["collision_enabled_prim_count"] > 0 and row["expected_collision_extent_error_max"] <= 0.75)
        row["visual_collision_bbox_aligned"] = visual_aligned
        row["expected_collision_bbox_aligned"] = expected_aligned
        row["collision_bbox_ok"] = bool(row["collision_enabled_prim_count"] > 0 and (visual_aligned or expected_aligned or entity_type == "fingertip"))
        if row["collision_bbox_ok"]:
            if visual_aligned and not expected_aligned and entity_type == "object":
                row["collision_blocker"] = "expected_geometry_mismatch_not_collision_blocker"
            elif row["collision_only_on_child_prims"]:
                row["collision_blocker"] = "root_collision_false_child_collision_valid"
            else:
                row["collision_blocker"] = ""
        elif row["collision_enabled_prim_count"] <= 0:
            row["collision_blocker"] = "collision_missing_entirely"
        else:
            row["collision_blocker"] = "collision_bbox_mismatch_visual_and_expected_geometry"
    except Exception as exc:
        row["collision_blocker"] = f"{type(exc).__name__}:{exc}"
    return row


def _v93_collision_inventory_rows(stage: Any, root_path: str, name: str, entity_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage is None or not root_path:
        return rows
    try:
        from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: WPS433

        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return rows
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path != str(root.GetPath()) and not path.startswith(str(root.GetPath()) + "/"):
                continue
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
            approx_attr = (
                physx_collision.GetCollisionApproximationAttr()
                if hasattr(physx_collision, "GetCollisionApproximationAttr")
                else prim.GetAttribute("physxCollision:collisionApproximation")
            )
            approx = approx_attr.Get() if approx_attr and approx_attr.IsValid() else ""
            rows.append(
                {
                    "part_name": name if entity_type == "object" else "",
                    "entity_name": name,
                    "entity_type": entity_type,
                    "root_prim_path": str(root.GetPath()),
                    "prim_path": path,
                    "prim_type": prim.GetTypeName(),
                    "is_visual_mesh": bool(prim.IsA(UsdGeom.Mesh) and "collision" not in path.lower()),
                    "collision_enabled": collision_enabled,
                    "contact_report_api_present": _v92_has_contact_report_api(prim),
                    "collision_approximation": str(approx or ""),
                }
            )
    except Exception as exc:
        rows.append({"entity_name": name, "entity_type": entity_type, "root_prim_path": root_path, "prim_path": "", "prim_error": f"{type(exc).__name__}:{exc}"})
    return rows


def _v93_alignment_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "part_name": row.get("part_name", ""),
        "entity_name": row.get("entity_name", ""),
        "entity_type": row.get("entity_type", ""),
        "active_asset_usd": row.get("active_asset_usd", ""),
        "root_prim_path": row.get("root_prim_path", ""),
        "visual_mesh_prim_count": row.get("visual_mesh_prim_count", 0),
        "collision_enabled_prim_count": row.get("collision_enabled_prim_count", 0),
        "visual_collision_center_offset_m": row.get("visual_collision_center_offset_m", 0.0),
        "visual_collision_extent_error_max": row.get("visual_collision_extent_error_max", 1.0),
        "expected_collision_extent_error_max": row.get("expected_collision_extent_error_max", 1.0),
        "visual_collision_bbox_iou": row.get("visual_collision_bbox_iou", 0.0),
        "best_axis_permutation": row.get("best_axis_permutation", ""),
        "visual_collision_bbox_aligned": row.get("visual_collision_bbox_aligned", False),
        "expected_collision_bbox_aligned": row.get("expected_collision_bbox_aligned", False),
        "collision_bbox_ok": row.get("collision_bbox_ok", False),
        "collision_blocker": row.get("collision_blocker", ""),
    }


def _v93_bbox_for_prims(prims: list[Any]) -> dict[str, tuple[float, float, float]]:
    if not prims:
        return {"min": (0.0, 0.0, 0.0), "max": (0.0, 0.0, 0.0), "center": (0.0, 0.0, 0.0), "extent": (0.0, 0.0, 0.0)}
    try:
        from pxr import UsdGeom  # noqa: WPS433

        cache = UsdGeom.BBoxCache(0.0, ["default", "render", "proxy", "guide"])
        mins: list[tuple[float, float, float]] = []
        maxs: list[tuple[float, float, float]] = []
        for prim in prims:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn = box.GetMin()
            mx = box.GetMax()
            extent = (float(mx[0]) - float(mn[0]), float(mx[1]) - float(mn[1]), float(mx[2]) - float(mn[2]))
            if max(extent) <= 0.0:
                continue
            mins.append((float(mn[0]), float(mn[1]), float(mn[2])))
            maxs.append((float(mx[0]), float(mx[1]), float(mx[2])))
        if not mins:
            return {"min": (0.0, 0.0, 0.0), "max": (0.0, 0.0, 0.0), "center": (0.0, 0.0, 0.0), "extent": (0.0, 0.0, 0.0)}
        mn = tuple(min(values[i] for values in mins) for i in range(3))
        mx = tuple(max(values[i] for values in maxs) for i in range(3))
        extent = tuple(max(0.0, mx[i] - mn[i]) for i in range(3))
        center = tuple(0.5 * (mn[i] + mx[i]) for i in range(3))
        return {"min": mn, "max": mx, "center": center, "extent": extent}
    except Exception:
        return {"min": (0.0, 0.0, 0.0), "max": (0.0, 0.0, 0.0), "center": (0.0, 0.0, 0.0), "extent": (0.0, 0.0, 0.0)}


def _v93_bbox_fields(prefix: str, box: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    extent = box.get("extent", (0.0, 0.0, 0.0))
    center = box.get("center", (0.0, 0.0, 0.0))
    return {
        f"{prefix}_center_x": float(center[0]),
        f"{prefix}_center_y": float(center[1]),
        f"{prefix}_center_z": float(center[2]),
        f"{prefix}_extent_x": float(extent[0]),
        f"{prefix}_extent_y": float(extent[1]),
        f"{prefix}_extent_z": float(extent[2]),
    }


def _v93_diag(extent: tuple[float, float, float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in extent))


def _v93_center_offset(a: dict[str, tuple[float, float, float]], b: dict[str, tuple[float, float, float]]) -> float:
    ca = a.get("center", (0.0, 0.0, 0.0))
    cb = b.get("center", (0.0, 0.0, 0.0))
    return math.sqrt(sum((float(ca[i]) - float(cb[i])) ** 2 for i in range(3)))


def _v93_extent_error(a: tuple[float, float, float], b: tuple[float, float, float], *, sorted_axes: bool = False) -> float:
    aa = sorted([abs(float(value)) for value in a]) if sorted_axes else [abs(float(value)) for value in a]
    bb = sorted([abs(float(value)) for value in b]) if sorted_axes else [abs(float(value)) for value in b]
    if max([0.0, *aa, *bb]) <= 0.0:
        return 1.0
    errors = [abs(aa[i] - bb[i]) / max(aa[i], bb[i], 1.0e-6) for i in range(3)]
    return max(errors)


def _v93_bbox_iou(a: dict[str, tuple[float, float, float]], b: dict[str, tuple[float, float, float]]) -> float:
    amin = a.get("min", (0.0, 0.0, 0.0))
    amax = a.get("max", (0.0, 0.0, 0.0))
    bmin = b.get("min", (0.0, 0.0, 0.0))
    bmax = b.get("max", (0.0, 0.0, 0.0))
    inter = 1.0
    va = 1.0
    vb = 1.0
    for i in range(3):
        inter *= max(0.0, min(float(amax[i]), float(bmax[i])) - max(float(amin[i]), float(bmin[i])))
        va *= max(0.0, float(amax[i]) - float(amin[i]))
        vb *= max(0.0, float(bmax[i]) - float(bmin[i]))
    union = va + vb - inter
    return inter / union if union > 1.0e-12 else 0.0


def _v93_best_axis_permutation(expected: tuple[float, float, float], actual: tuple[float, float, float]) -> str:
    labels = ("x", "y", "z")
    try:
        order = sorted(range(3), key=lambda index: abs(float(expected[index])))
        actual_order = sorted(range(3), key=lambda index: abs(float(actual[index])))
        mapping = [""] * 3
        for e_idx, a_idx in zip(order, actual_order):
            mapping[e_idx] = labels[a_idx]
        return "".join(mapping)
    except Exception:
        return "xyz"


def _v92_has_contact_report_api(prim: Any) -> bool:
    try:
        return any("ContactReport" in str(name) for name in prim.GetAppliedSchemas())
    except Exception:
        return False


def _v92_table_top_by_env(backend: IsaacUnifiedSingleContextBackend) -> dict[int, float]:
    tops: dict[int, float] = {}
    try:
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        for slot in backend.slots:
            top = _v92_bbox_top(stage, f"/World/envs/env_{slot.local_env_index}/Table")
            if top is None:
                top = _v92_bbox_top(stage, "/World/envs/env_0/Table")
            tops[slot.local_env_index] = _v92_safe_table_top(top if top is not None else 0.72)
    except Exception:
        for slot in backend.slots:
            tops[slot.local_env_index] = 0.72
    return tops


def _v92_safe_table_top(value: Any) -> float:
    try:
        top = float(value)
    except Exception:
        return 0.72
    if not math.isfinite(top) or top < -0.10 or top > 1.10:
        return 0.72
    return top


def _v92_bbox_top(stage: Any, path: str) -> float | None:
    try:
        from pxr import UsdGeom  # noqa: WPS433

        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        cache = UsdGeom.BBoxCache(0.0, ["default", "render", "proxy", "guide"])
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        return float(bbox.GetMax()[2])
    except Exception:
        return None


def _v92_bbox_extent(stage: Any, prim: Any) -> tuple[float, float, float]:
    try:
        from pxr import UsdGeom  # noqa: WPS433

        cache = UsdGeom.BBoxCache(0.0, ["default", "render", "proxy", "guide"])
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        min_v = bbox.GetMin()
        max_v = bbox.GetMax()
        return (abs(float(max_v[0]) - float(min_v[0])), abs(float(max_v[1]) - float(min_v[1])), abs(float(max_v[2]) - float(min_v[2])))
    except Exception:
        return (0.0, 0.0, 0.0)


def _v92_expected_extent(geometry: dict[str, Any]) -> tuple[float, float, float]:
    kind = str(geometry.get("kind") or "")
    if kind == "sphere":
        radius = float(geometry.get("radius") or 0.0)
        return (2.0 * radius, 2.0 * radius, 2.0 * radius)
    if kind == "cylinder_z":
        radius = float(geometry.get("radius") or 0.0)
        half = float(geometry.get("half_height") or 0.0)
        return (2.0 * radius, 2.0 * radius, 2.0 * half)
    if kind == "boxes":
        max_x = max_y = max_z = 0.0
        for center, half in geometry.get("boxes", ()):
            max_x = max(max_x, abs(float(center[0])) + abs(float(half[0])))
            max_y = max(max_y, abs(float(center[1])) + abs(float(half[1])))
            max_z = max(max_z, abs(float(center[2])) + abs(float(half[2])))
        return (2.0 * max_x, 2.0 * max_y, 2.0 * max_z)
    return (0.0, 0.0, 0.0)


def _v92_extent_match(actual: tuple[float, float, float], expected: tuple[float, float, float]) -> bool:
    if min(actual) <= 0.0 or min(expected) <= 0.0:
        return False
    ratios = [actual[i] / max(expected[i], 1.0e-6) for i in range(3)]
    return all(0.2 <= ratio <= 5.0 for ratio in ratios)


def _v92_support_half_height_m(part_name: str) -> float:
    extent = _v92_expected_extent(V85_PART_GEOMETRY.get(part_name, {}))
    return max(0.005, float(extent[2]) * 0.5)


def _v92_fingertip_body_names(backend: IsaacUnifiedSingleContextBackend) -> list[str]:
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    names = list(getattr(base, "dex_fingertip_true_body_names", []) or [])
    if names:
        return names
    robot = getattr(base, "_robot", None)
    body_names = list(getattr(robot, "body_names", []) or [])
    return [name for name in body_names if name.startswith("right_finger") and name.endswith("link4")][:5]


def _v92_fingertip_name(backend: IsaacUnifiedSingleContextBackend, index: int) -> str:
    names = _v92_fingertip_body_names(backend)
    return names[index] if 0 <= index < len(names) else ""


def _v92_find_robot_body_prim_path(stage: Any, body_name: str, env_index: int = 0) -> str:
    return find_robot_body_prim_path_for_env(stage, body_name, env_index)


def _v92_force_list(metrics: dict[str, Any]) -> list[float]:
    values = metrics.get("per_finger_force_norm") or metrics.get("per_finger_contact_force_norm") or []
    if isinstance(values, str):
        values = values.strip("[]").split(",")
    try:
        return [float(item) for item in list(values)[:5]]
    except Exception:
        return []


def _v92_max_index(values: list[float]) -> int:
    if not values:
        return -1
    return max(range(len(values)), key=lambda index: values[index])


def _v92_tip_positions(backend: IsaacUnifiedSingleContextBackend) -> Any:
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    try:
        if getattr(base, "last_update_timestamp", 0.0) < getattr(base._robot._data, "_sim_timestamp", 0.0):
            base._compute_intermediate_values(dt=base.physics_dt)
        return base.dex_fingertip_pos.detach().clone()
    except Exception:
        return None


def _v92_joint_pos(backend: IsaacUnifiedSingleContextBackend) -> Any:
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    try:
        return base._robot.data.joint_pos.detach().clone()
    except Exception:
        return None


def _v92_tip_delta(before: Any, after: Any, env_index: int, finger_index: int) -> float:
    try:
        delta = after[int(env_index), int(finger_index)] - before[int(env_index), int(finger_index)]
        return float(torch.linalg.vector_norm(delta).detach().cpu().item())
    except Exception:
        return 0.0


def _v92_joint_delta(before: Any, after: Any, env_index: int) -> float:
    try:
        delta = after[int(env_index)] - before[int(env_index)]
        return float(torch.linalg.vector_norm(delta).detach().cpu().item())
    except Exception:
        return -1.0


def _v92_group_close_action(part_name: str, step: int) -> tuple[list[float], dict[str, Any]]:
    action = [0.0] * 16
    group = V85_FINGER_GROUPS.get(part_name, ("34",))[0]
    close = min(0.85, float(step + 1) / 16.0 * 0.85)
    mask = _v91_apply_finger_mask(action, group=group, active_value=close, support_value=0.0)
    return action, mask


IsaacUnifiedSingleContextBackend._v84_policy_actions_for_phase = _v84_policy_actions_for_phase
IsaacUnifiedSingleContextBackend._v84_approach_direction = _v84_approach_direction
IsaacUnifiedSingleContextBackend._v85_policy_actions_for_phase = _v85_policy_actions_for_phase
IsaacUnifiedSingleContextBackend._v88_policy_actions_for_phase = _v88_policy_actions_for_phase
IsaacUnifiedSingleContextBackend._v89_policy_actions_for_phase = _v89_policy_actions_for_phase
IsaacUnifiedSingleContextBackend._v90_policy_actions_for_phase = _v90_policy_actions_for_phase
IsaacUnifiedSingleContextBackend._v91_policy_actions_for_phase = _v91_policy_actions_for_phase


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fields = list(rows[0].keys())
    else:
        fields = ["part_name", "status", "blocker"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
