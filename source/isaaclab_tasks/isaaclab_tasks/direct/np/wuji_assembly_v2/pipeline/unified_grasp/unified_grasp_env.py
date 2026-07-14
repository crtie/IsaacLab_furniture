"""v80 unified multi-object grasp environment interface.

The class is intentionally honest about backend readiness. It exposes the
IsaacLab/Gym entry point and the v80 observation/action contract, but final
physical success is blocked unless a real Isaac contact backend is attached.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional at static compile time
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

try:  # pragma: no cover - torch availability is environment-specific
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - rsl-rl runtime dependency
    from tensordict import TensorDict
except Exception:  # pragma: no cover
    TensorDict = None

from .contact_manager import ContactManager
from .reward_terms import compute_reward_terms
from .v80_reports import V80_PARTS, write_csv, write_json, write_jsonl


@dataclass
class UnifiedGraspEnvCfg:
    num_envs: int = 16
    parts: list[str] = field(default_factory=lambda: list(V80_PARTS))
    part_sampler: str = "balanced"
    physics_profile: str = "canonical"
    policy_mode: str = "shared_multi_object_policy"
    max_episode_length: int = 48
    device: str = "cpu"
    use_diagnostic_dynamics: bool = False
    physical_backend_required: bool = True
    sticky_after_support_validation: bool = False
    v86_guarded_residual_mode: bool = False
    v87_contact_guided_residual_mode: bool = False
    v88_stabilized_mode: bool = False
    v89_hybrid_repair_mode: bool = False
    v89_nominal_candidate_mode: bool = False
    v88_observation_enhanced: bool = False
    v88_auto_reset_on_done: bool = False
    v88_nominal_lift_phase_enabled: bool = False
    use_v85_near_contact_resets: bool = False
    excessive_force_threshold_n: float = 150.0
    v87_residual_scale_wrist_xyz: float = 0.15
    v87_residual_scale_wrist_rot: float = 0.08
    v87_residual_scale_finger: float = 0.20


class UnifiedGraspEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    metadata = {"render_modes": []}

    observation_dim = 64
    action_dim = 16

    def __init__(self, cfg: UnifiedGraspEnvCfg | None = None, **kwargs: Any) -> None:
        self.cfg = cfg or UnifiedGraspEnvCfg(**{k: v for k, v in kwargs.items() if k in UnifiedGraspEnvCfg.__dataclass_fields__})
        self.num_envs = max(1, int(self.cfg.num_envs))
        self.parts = list(self.cfg.parts or V80_PARTS)
        self.contact_manager = ContactManager()
        self.physical_backend = kwargs.get("physical_backend")
        self.physical_backend_ready = self._validate_physical_backend(self.physical_backend)
        self.contact_manager.attach_backend(self.physical_backend)
        self.surrogate_training_used = bool(self.cfg.use_diagnostic_dynamics and not self.physical_backend_ready)
        self.dynamics_mode = "physical_backend" if self.physical_backend_ready else "surrogate_unit_test_only"
        self.object_writes_by_policy = False
        self.sticky_action_enabled = False
        self.proxy_action_enabled = False
        self.route_selection_enabled = False
        self.num_obs = self.observation_dim
        self.num_actions = self.action_dim
        self.max_episode_length = int(self.cfg.max_episode_length)
        self.device = self.cfg.device
        self.step_count = [0 for _ in range(self.num_envs)]
        self.episode_count = [0 for _ in range(self.num_envs)]
        self.part_for_env = [self._sample_part(index) for index in range(self.num_envs)]
        self._state = [self._initial_state(index) for index in range(self.num_envs)]
        if self.physical_backend_ready:
            self.num_envs = int(getattr(self.physical_backend, "num_envs", self.num_envs) or self.num_envs)
            self.step_count = [0 for _ in range(self.num_envs)]
            self.episode_count = [0 for _ in range(self.num_envs)]
            self.part_for_env = self._backend_parts()
            self._state = [self._initial_state(index) for index in range(self.num_envs)]
        if torch is not None:
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if spaces is not None:
            self.observation_space = spaces.Box(-float("inf"), float("inf"), shape=(self.observation_dim,), dtype="float32")
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype="float32")

    def _validate_physical_backend(self, backend: Any | None) -> bool:
        if backend is None:
            return False
        required = ("reset_envs", "step_envs", "step_env", "get_contact_state", "get_metrics", "get_part_distribution", "close")
        methods_ok = all(callable(getattr(backend, name, None)) for name in required)
        if not methods_ok:
            return False
        return bool(getattr(backend, "physical_backend_ready", False) and getattr(backend, "contact_sensor_configured", False))

    def _backend_parts(self) -> list[str]:
        slots = list(getattr(self.physical_backend, "slots", []) or [])
        if slots:
            return [str(getattr(slot, "part_name", self._sample_part(index))) for index, slot in enumerate(slots)]
        return [self._sample_part(index) for index in range(self.num_envs)]

    def _sample_part(self, env_index: int) -> str:
        if self.cfg.part_sampler == "round_robin":
            return self.parts[env_index % len(self.parts)]
        if self.cfg.part_sampler == "curriculum_weighted":
            return random.choice(self.parts)
        return self.parts[env_index % len(self.parts)]

    def _initial_state(self, env_index: int) -> dict[str, Any]:
        return {
            "env_index": env_index,
            "part_name": self.part_for_env[env_index],
            "active_distance_m": 0.05,
            "force_contact_count": 0.0,
            "distance_contact_count": 0.0,
            "force_contact_peak_n": 0.0,
            "force_contact_mean_n": 0.0,
            "excessive_force": False,
            "object_displacement_m": 0.0,
            "object_delta_z_m": 0.0,
            "support_gate_ok": False,
            "object_motion_before_contact_m": 0.0,
            "penetration_depth_m": 0.0,
            "table_collision": False,
            "hold_success": False,
            "lift_success": False,
            "contact_sensor_available": self.physical_backend_ready,
            "contact_evidence_source": "force_contact_sensor" if self.physical_backend_ready else "physical_backend_unavailable",
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
        for index in range(self.num_envs):
            self.step_count[index] = 0
            self.episode_count[index] += 1
            self.part_for_env[index] = self._backend_parts()[index] if self.physical_backend_ready else self._sample_part(index)
            self._state[index] = self._initial_state(index)
        if self.physical_backend_ready:
            reset_rows = self.physical_backend.reset_envs()
            if self.cfg.use_v85_near_contact_resets and callable(getattr(self.physical_backend, "stage_v86_near_contact_episode", None)):
                reset_rows = self.physical_backend.stage_v86_near_contact_episode()
            for row in reset_rows:
                env_index = int(row.get("env_index") or 0)
                if 0 <= env_index < self.num_envs:
                    self._state[env_index].update(row)
        if torch is not None:
            self.episode_length_buf[:] = 0
            self.reset_buf[:] = False
        obs = self._obs_batch()
        return obs, self._info()

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        actions = self._action_rows(action)
        backend_metrics: list[dict[str, Any]] = []
        if self.physical_backend_ready:
            if self.cfg.v89_hybrid_repair_mode and callable(
                getattr(self.physical_backend, "step_v89_candidate_residual_envs", None)
            ):
                backend_metrics = self.physical_backend.step_v89_candidate_residual_envs(
                    actions,
                    episode_steps=list(self.step_count),
                    residual_scales={
                        "wrist_xyz": float(self.cfg.v87_residual_scale_wrist_xyz),
                        "wrist_rot": float(self.cfg.v87_residual_scale_wrist_rot),
                        "finger": float(self.cfg.v87_residual_scale_finger),
                    },
                    lift_phase_enabled=bool(self.cfg.v88_nominal_lift_phase_enabled),
                )
            elif self.cfg.v88_stabilized_mode and callable(
                getattr(self.physical_backend, "step_v88_stabilized_residual_envs", None)
            ):
                backend_metrics = self.physical_backend.step_v88_stabilized_residual_envs(
                    actions,
                    episode_steps=list(self.step_count),
                    residual_scales={
                        "wrist_xyz": float(self.cfg.v87_residual_scale_wrist_xyz),
                        "wrist_rot": float(self.cfg.v87_residual_scale_wrist_rot),
                        "finger": float(self.cfg.v87_residual_scale_finger),
                    },
                    lift_phase_enabled=bool(self.cfg.v88_nominal_lift_phase_enabled),
                )
            elif self.cfg.v87_contact_guided_residual_mode and callable(
                getattr(self.physical_backend, "step_v87_residual_envs", None)
            ):
                backend_metrics = self.physical_backend.step_v87_residual_envs(
                    actions,
                    episode_steps=list(self.step_count),
                    residual_scales={
                        "wrist_xyz": float(self.cfg.v87_residual_scale_wrist_xyz),
                        "wrist_rot": float(self.cfg.v87_residual_scale_wrist_rot),
                        "finger": float(self.cfg.v87_residual_scale_finger),
                    },
                )
            else:
                backend_metrics = self.physical_backend.step_envs(actions)
            for row in backend_metrics:
                env_index = int(row.get("env_index") or 0)
                if 0 <= env_index < self.num_envs:
                    self._state[env_index].update(row)
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        for index, row in enumerate(actions):
            self.step_count[index] += 1
            if not self.physical_backend_ready and self.cfg.use_diagnostic_dynamics:
                self._diagnostic_step(index, row)
            elif not self.physical_backend_ready:
                self._state[index].update(
                    {
                        "failure_category": "ENV_NOT_VECTORIZEABLE",
                        "blocker": "physical_isaac_backend_not_configured",
                    }
                )
            contact = self._contact_dict(index)
            terms = compute_reward_terms(
                contact,
                part_name=self.part_for_env[index],
                close_achieved=self.step_count[index] > 4,
                support_gate_ok=bool(self._state[index].get("support_gate_ok")),
                hold_success=bool(self._state[index].get("hold_success")),
                lift_success=bool(self._state[index].get("lift_success")),
                action_norm=sum(abs(float(v)) for v in row) / max(1, len(row)),
                sticky_before_support=False,
                object_write_attempted=False,
                proxy_action_attempted=False,
            )
            rewards.append(float(terms["reward_total"]))
            done = self.step_count[index] >= self.max_episode_length
            terminated.append(bool(self._state[index].get("support_gate_ok") and self._state[index].get("lift_success")))
            truncated.append(done)
            self._state[index]["last_reward_terms"] = terms
        if torch is not None:
            self.episode_length_buf += 1
            done_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.device) | torch.tensor(truncated, dtype=torch.bool, device=self.device)
            self.reset_buf[:] = done_tensor
        obs = self._obs_batch()
        if self.cfg.v88_auto_reset_on_done and any(bool(item) for item in terminated + truncated):
            obs, _reset_info = self.reset()
        return obs, self._tensor_or_list(rewards), self._tensor_or_list(terminated), self._tensor_or_list(truncated), self._info()

    def _diagnostic_step(self, env_index: int, action: list[float]) -> None:
        state = self._state[env_index]
        action_mag = sum(abs(value) for value in action) / max(1, len(action))
        state["active_distance_m"] = max(0.0, float(state["active_distance_m"]) - 0.0015 + 0.0003 * action_mag)
        if state["active_distance_m"] < 0.010:
            state["distance_contact_count"] = min(2.0, state["distance_contact_count"] + 0.5)
        state["contact_sensor_available"] = False
        state["contact_evidence_source"] = "diagnostic_dynamics_distance_only"
        state["dynamics_mode"] = "surrogate_unit_test_only"
        state["failure_category"] = "CONTACT_SENSOR_UNAVAILABLE_FALLBACK_ONLY"
        state["blocker"] = "diagnostic_dynamics_is_not_final_contact_evidence"

    def _contact_dict(self, env_index: int) -> dict[str, Any]:
        state = self._state[env_index]
        return {
            "effective_contact_count_force": state.get("force_contact_count", 0.0),
            "effective_contact_count_distance": state.get("distance_contact_count", 0.0),
            "force_contact_peak_n": state.get("force_contact_peak_n", 0.0),
            "force_contact_mean_n": state.get("force_contact_mean_n", 0.0),
            "object_motion_before_contact_m": state.get("object_motion_before_contact_m", 0.0),
            "penetration_depth_m": state.get("penetration_depth_m", 0.0),
            "table_collision": state.get("table_collision", False),
            "contact_sensor_available": state.get("contact_sensor_available", False),
            "force_contact_streak_steps": state.get("force_contact_streak_steps", 0),
            "object_displacement_m": state.get("object_displacement_m", 0.0),
        }

    def _obs_for_env(self, env_index: int) -> list[float]:
        state = self._state[env_index]
        part_index = V80_PARTS.index(state["part_name"]) if state["part_name"] in V80_PARTS else 0
        values = [
            float(part_index) / max(1, len(V80_PARTS) - 1),
            min(float(state.get("active_distance_m", 0.05)), 0.10) / 0.10,
            min(float(state.get("force_contact_count", 0.0)), 5.0) / 5.0,
            min(float(state.get("distance_contact_count", 0.0)), 5.0) / 5.0,
            1.0 if state.get("support_gate_ok") else 0.0,
            min(float(state.get("object_motion_before_contact_m", 0.0)), 0.05) / 0.05,
            min(float(state.get("penetration_depth_m", 0.0)), 0.005) / 0.005,
            1.0 if state.get("table_collision") else 0.0,
            min(float(state.get("force_contact_peak_n", 0.0)), self.cfg.excessive_force_threshold_n)
            / max(1.0, float(self.cfg.excessive_force_threshold_n)),
            min(float(state.get("object_displacement_m", 0.0)), 0.05) / 0.05,
            max(-1.0, min(1.0, float(state.get("object_delta_z_m", 0.0)) / 0.02)),
            float(self.step_count[env_index]) / max(1.0, float(self.cfg.max_episode_length)),
            1.0 if self.physical_backend_ready else 0.0,
            1.0 if self.surrogate_training_used else 0.0,
            1.0 if state.get("contact_sensor_available") else 0.0,
            1.0 if self.cfg.v86_guarded_residual_mode else 0.0,
            1.0 if self.cfg.v87_contact_guided_residual_mode else 0.0,
            1.0 if self.cfg.v88_stabilized_mode else 0.0,
            1.0 if self.cfg.v89_hybrid_repair_mode else 0.0,
        ]
        if self.cfg.v88_observation_enhanced:
            one_hot = [1.0 if index == part_index else 0.0 for index in range(len(V80_PARTS))]
            values.extend(one_hot)
            values.extend(
                [
                    max(-1.0, min(1.0, float(state.get("object_rel_palm_x", 0.0)) / 0.25)),
                    max(-1.0, min(1.0, float(state.get("object_rel_palm_y", 0.0)) / 0.25)),
                    max(-1.0, min(1.0, float(state.get("object_rel_palm_z", 0.0)) / 0.25)),
                    min(float(state.get("object_bbox_x", 0.0)), 0.60) / 0.60,
                    min(float(state.get("object_bbox_y", 0.0)), 0.20) / 0.20,
                    min(float(state.get("object_bbox_z", 0.0)), 0.60) / 0.60,
                    min(float(state.get("fingertip_object_distance_min_m", 0.0)), 0.20) / 0.20,
                    min(float(state.get("fingertip_object_distance_mean_m", 0.0)), 0.30) / 0.30,
                    min(float(state.get("force_contact_streak_steps", 0.0)), 32.0) / 32.0,
                    min(float(state.get("contact_duration_steps", 0.0)), 32.0) / 32.0,
                    min(float(state.get("residual_action_norm", 0.0)), 4.0) / 4.0,
                    min(float(state.get("composed_action_norm", 0.0)), 4.0) / 4.0,
                    min(float(state.get("hold_object_displacement_m", 0.0)), 0.08) / 0.08,
                    min(max(float(state.get("lift_delta_z_m", 0.0)), 0.0), 0.04) / 0.04,
                ]
            )
        values.extend([0.0] * (self.observation_dim - len(values)))
        return values[: self.observation_dim]

    def _obs_batch(self) -> Any:
        rows = [self._obs_for_env(index) for index in range(self.num_envs)]
        return self._tensor_or_list(rows)

    def get_observations(self) -> dict[str, Any]:
        obs = self._obs_batch()
        if TensorDict is not None and torch is not None and hasattr(obs, "shape"):
            return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        return {"policy": obs}

    def _tensor_or_list(self, value: Any) -> Any:
        if torch is None:
            return value
        dtype = torch.bool if value and isinstance(value[0], bool) else torch.float32
        return torch.tensor(value, dtype=dtype)

    def _action_rows(self, action: Any) -> list[list[float]]:
        if hasattr(action, "detach"):
            action = action.detach().cpu().tolist()
        if not action:
            action = [[0.0] * self.action_dim for _ in range(self.num_envs)]
        if isinstance(action[0], (int, float)):
            action = [action]
        rows = []
        for index in range(self.num_envs):
            raw = list(action[index % len(action)])
            clipped = [max(-1.0, min(1.0, float(value))) for value in raw[: self.action_dim]]
            clipped.extend([0.0] * (self.action_dim - len(clipped)))
            rows.append(clipped)
        return rows

    def _info(self) -> dict[str, Any]:
        return {
            "physical_backend_ready": self.physical_backend_ready,
            "surrogate_training_used": self.surrogate_training_used,
            "dynamics_mode": self.dynamics_mode,
            "object_writes_by_policy": self.object_writes_by_policy,
            "sticky_action_enabled": self.sticky_action_enabled,
            "proxy_action_enabled": self.proxy_action_enabled,
            "route_selection_enabled": self.route_selection_enabled,
            "part_distribution": {part: self.part_for_env.count(part) for part in self.parts},
            "backend_blocker": str(getattr(self.physical_backend, "blocker", "")) if self.physical_backend is not None else "",
        }

    def close(self) -> None:
        return None


def run_env_smoke(
    run_dir: str | Path,
    *,
    num_envs: int = 16,
    steps: int = 8,
    parts: list[str] | None = None,
    physics_profile: str = "canonical",
    allow_diagnostic_dynamics: bool = False,
    physical_backend: Any | None = None,
) -> dict[str, Any]:
    cfg = UnifiedGraspEnvCfg(
        num_envs=num_envs,
        parts=parts or list(V80_PARTS),
        physics_profile=physics_profile,
        use_diagnostic_dynamics=allow_diagnostic_dynamics,
    )
    env = UnifiedGraspEnv(cfg, physical_backend=physical_backend)
    reset_obs, reset_info = env.reset(seed=0)
    reset_rows = [
        {
            "env_index": index,
            "part_name": env.part_for_env[index],
            "physics_profile": physics_profile,
            "physical_backend_ready": env.physical_backend_ready,
            "surrogate_training_used": env.surrogate_training_used,
        }
        for index in range(env.num_envs)
    ]
    step_rows: list[dict[str, Any]] = []
    for step_index in range(max(1, int(steps))):
        action = [[0.0] * env.action_dim for _ in range(env.num_envs)]
        obs, reward, terminated, truncated, info = env.step(action)
        for env_index in range(env.num_envs):
            state = env._state[env_index]
            step_rows.append(
                {
                    "step_index": step_index,
                    "env_index": env_index,
                    "part_name": state["part_name"],
                    "reward": float(reward[env_index]) if hasattr(reward, "__getitem__") else 0.0,
                    "active_distance_m": state.get("active_distance_m"),
                    "force_contact_count": state.get("force_contact_count"),
                    "distance_contact_count": state.get("distance_contact_count"),
                    "support_gate_ok": state.get("support_gate_ok"),
                    "failure_category": state.get("failure_category", ""),
                    "blocker": state.get("blocker", ""),
                    "action_bounded": True,
                    "object_writes_by_policy": False,
                    "sticky_action_enabled": False,
                }
            )
    episode_rows = []
    for part in cfg.parts:
        part_steps = [row for row in step_rows if row["part_name"] == part]
        part_env_count = sum(1 for item in reset_rows if item["part_name"] == part)
        part_vectorized_ok = bool(env.physical_backend_ready and part_env_count > 0)
        episode_rows.append(
            {
                "part_name": part,
                "episode_count": part_env_count,
                "step_count": len(part_steps),
                "physics_profile": physics_profile,
                "physical_backend_ready": env.physical_backend_ready,
                "rl_env_vectorized_ok": part_vectorized_ok,
                "actual_env_count": env.num_envs,
                "observations_finite": _obs_is_finite(reset_obs),
                "actions_bounded": True,
                "object_writes_by_policy": False,
                "sticky_action_enabled": False,
                "failure_category": "" if part_vectorized_ok else "ENV_NOT_VECTORIZEABLE",
                "blocker": ""
                if part_vectorized_ok
                else str(
                    getattr(physical_backend, "part_blocker", lambda _part: "")(part)
                    or getattr(physical_backend, "blocker", "")
                    or "physical_isaac_backend_not_configured"
                ),
            }
        )
    run_path = Path(run_dir)
    reset_csv = write_csv(run_path / "unified_env_reset_trace.csv", reset_rows)
    reset_jsonl = write_jsonl(run_path / "unified_env_reset_trace.jsonl", reset_rows)
    step_csv = write_csv(run_path / "unified_env_step_trace.csv", step_rows)
    step_jsonl = write_jsonl(run_path / "unified_env_step_trace.jsonl", step_rows)
    summary_csv = write_csv(run_path / "unified_env_episode_summary.csv", episode_rows)
    summary_json = write_json(run_path / "unified_env_episode_summary.json", episode_rows)
    return {
        "rows": episode_rows,
        "reset_info": reset_info,
        "unified_env_reset_trace_csv": str(reset_csv),
        "unified_env_reset_trace_jsonl": str(reset_jsonl),
        "unified_env_step_trace_csv": str(step_csv),
        "unified_env_step_trace_jsonl": str(step_jsonl),
        "unified_env_episode_summary_csv": str(summary_csv),
        "unified_env_episode_summary_json": str(summary_json),
    }


def _obs_is_finite(obs: Any) -> bool:
    try:
        if hasattr(obs, "detach"):
            return bool(torch.isfinite(obs).all().item()) if torch is not None else True
        for row in obs:
            for value in row:
                if not math.isfinite(float(value)):
                    return False
        return True
    except Exception:
        return False
