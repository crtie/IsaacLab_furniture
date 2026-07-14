"""Training entry points for the v80 unified grasp stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .multi_object_policy_config import UnifiedPolicyConfig, write_unified_rl_config
from .unified_grasp_env import UnifiedGraspEnv, UnifiedGraspEnvCfg, run_env_smoke
from .v80_reports import V80_PARTS, write_csv, write_json, write_jsonl

try:  # pragma: no cover - torch is runtime-specific
    import torch
except Exception:  # pragma: no cover
    torch = None


class RslRlUnifiedGraspVecEnv:
    """RSL-RL adapter around the v81 Gym-compatible env."""

    def __init__(self, env: UnifiedGraspEnv) -> None:
        self.env = env
        self.cfg = env.cfg
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = getattr(env, "episode_length_buf", None)
        self.device = env.device

    def get_observations(self) -> dict[str, Any]:
        return self.env.get_observations()

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        return self.get_observations()

    def step(self, actions: Any) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        _obs, rewards, terminated, truncated, info = self.env.step(actions)
        if torch is not None:
            rewards = rewards.to(self.device) if hasattr(rewards, "to") else torch.tensor(rewards, dtype=torch.float32, device=self.device)
            terminated = terminated.to(self.device) if hasattr(terminated, "to") else torch.tensor(terminated, dtype=torch.bool, device=self.device)
            truncated = truncated.to(self.device) if hasattr(truncated, "to") else torch.tensor(truncated, dtype=torch.bool, device=self.device)
            dones = terminated | truncated
            time_outs = truncated
        else:
            dones = [bool(a or b) for a, b in zip(terminated, truncated)]
            time_outs = truncated
        extras = {"time_outs": time_outs, "log": {"/v81/physical_backend_ready": float(self.env.physical_backend_ready)}}
        extras.update(info)
        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        self.env.close()


def check_rsl_rl_available() -> tuple[bool, str]:
    try:  # pragma: no cover - depends on active isaac env
        from rsl_rl.runners import OnPolicyRunner  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"RL_FRAMEWORK_UNAVAILABLE:{exc!r}"
    return True, "rsl_rl_available"


def _rsl_rl_train_cfg(*, num_steps_per_env: int, max_iterations: int, policy_mode: str) -> dict[str, Any]:
    return {
        "num_steps_per_env": int(max(1, num_steps_per_env)),
        "max_iterations": int(max(1, max_iterations)),
        "save_interval": int(max(1, max_iterations)),
        "experiment_name": f"v81_unified_grasp_{policy_mode}",
        "empirical_normalization": False,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 2,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "multi_gpu": {"enabled": False},
    }


def train_unified_policy(
    run_dir: str | Path,
    *,
    policy_mode: str = "shared_multi_object_policy",
    parts: list[str] | None = None,
    num_envs: int = 128,
    max_steps: int = 48,
    physics_profile: str = "canonical",
    allow_diagnostic_dynamics: bool = False,
    physical_backend: Any | None = None,
    ppo_iterations: int = 5,
    controlled_contact_gate_pass: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    config = UnifiedPolicyConfig(
        policy_mode=policy_mode,
        num_envs_requested=num_envs,
        horizon=max_steps,
        parts=tuple(parts or V80_PARTS),
        allow_diagnostic_dynamics=allow_diagnostic_dynamics,
    )
    config_path = write_unified_rl_config(run_path, config)
    rsl_ok, rsl_reason = check_rsl_rl_available()
    if not rsl_ok:
        return _write_training_not_started(
            run_path,
            config_path=str(config_path),
            policy_mode=policy_mode,
            parts=parts or V80_PARTS,
            reason=rsl_reason,
        )

    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
    )
    requested_parts = parts or list(V80_PARTS)
    ready_parts = [
        part
        for part in requested_parts
        if bool(getattr(physical_backend, "part_backend_ready", lambda _part: backend_ready)(part))
    ]
    if not backend_ready:
        if allow_diagnostic_dynamics:
            return _write_surrogate_unit_test_training(
                run_path,
                config_path=str(config_path),
                policy_mode=policy_mode,
                parts=parts or V80_PARTS,
                num_envs=num_envs,
                max_steps=max_steps,
                physics_profile=physics_profile,
            )
        return _write_training_not_started(
            run_path,
            config_path=str(config_path),
            policy_mode=policy_mode,
            parts=parts or V80_PARTS,
            reason=str(getattr(physical_backend, "blocker", "") or "ENV_NOT_VECTORIZEABLE:physical_isaac_backend_not_configured"),
        )
    if not ready_parts:
        return _write_training_not_started(
            run_path,
            config_path=str(config_path),
            policy_mode=policy_mode,
            parts=requested_parts,
            reason="NO_PART_WITH_READY_PHYSICAL_BACKEND_AND_CONTACT_SENSOR",
        )
    if not bool(controlled_contact_gate_pass):
        return _write_training_not_started(
            run_path,
            config_path=str(config_path),
            policy_mode=policy_mode,
            parts=requested_parts,
            reason="controlled_force_contact_not_observed",
        )

    try:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        env_cfg = UnifiedGraspEnvCfg(
            num_envs=num_envs,
            parts=ready_parts,
            physics_profile=physics_profile,
            policy_mode=policy_mode,
            max_episode_length=max_steps,
            device=str(getattr(physical_backend, "device", "cpu")),
            use_diagnostic_dynamics=False,
        )
        env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
        vec_env = RslRlUnifiedGraspVecEnv(env)
        train_cfg = _rsl_rl_train_cfg(
            num_steps_per_env=min(max(1, int(max_steps)), 32),
            max_iterations=max(1, int(ppo_iterations)),
            policy_mode=policy_mode,
        )
        train_cfg_path = write_json(run_path / "rsl_rl_runner_config.json", train_cfg)
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(run_path / "rsl_rl_logs"), device=vec_env.device)
        runner.learn(num_learning_iterations=max(1, int(ppo_iterations)), init_at_random_ep_len=False)
        checkpoint = run_path / "unified_rl_checkpoint.pt"
        runner.save(str(checkpoint))
        curve_rows = [
            {
                "iteration": index,
                "policy_mode": policy_mode,
                "surrogate_training_used": False,
                "rsl_rl_runner_used": True,
                "ppo_iterations_completed": max(1, int(ppo_iterations)),
                "rollout_step_count": int(vec_env.num_envs) * min(max(1, int(max_steps)), 32) * max(1, int(ppo_iterations)),
                "final_training_export_allowed": False,
            }
            for index in range(max(1, int(ppo_iterations)))
        ]
        curve_csv = write_csv(run_path / "unified_rl_training_curve.csv", curve_rows)
        rollout_csv = write_csv(run_path / "unified_rl_rollout_trace.csv", env.physical_backend.get_metrics())
        rollout_jsonl = write_jsonl(run_path / "unified_rl_rollout_trace.jsonl", env.physical_backend.get_metrics())
        per_part = []
        for part in requested_parts:
            trained_part = part in ready_parts
            per_part.append(
                {
                    "part_name": part,
                    "policy_mode": policy_mode,
                    "rl_trained": False,
                    "runner_smoke_complete": trained_part,
                    "rl_trained_success_evidence": False,
                    "training_curve_nonempty": bool(trained_part),
                    "rsl_rl_runner_used": bool(trained_part),
                    "surrogate_training_used": False,
                    "ppo_iterations_completed": max(1, int(ppo_iterations)) if trained_part else 0,
                    "rollout_step_count": int(vec_env.num_envs) * min(max(1, int(max_steps)), 32) * max(1, int(ppo_iterations)) if trained_part else 0,
                    "num_envs_requested": int(num_envs),
                    "num_envs_actual": int(vec_env.num_envs) if trained_part else 0,
                    "status": "runner_smoke_complete_no_success_evidence" if trained_part else "TRAINING_NOT_STARTED",
                    "blocker": "" if trained_part else str(getattr(physical_backend, "part_blocker", lambda _part: "part_backend_not_ready")(part)),
                    "usable_training_row_count": 0,
                }
            )
        per_part_csv = write_csv(run_path / "per_part_training_summary.csv", per_part)
        per_part_json = write_json(run_path / "per_part_training_summary.json", per_part)
        multi_json = write_json(
            run_path / "multi_object_training_summary.json",
            {
                "policy_mode": policy_mode,
                "rl_trained": False,
                "runner_smoke_complete": True,
                "rl_trained_success_evidence": False,
                "training_curve_nonempty": True,
                "checkpoint_path": str(checkpoint),
                "rsl_rl_runner_used": True,
                "surrogate_training_used": False,
                "ppo_iterations_completed": max(1, int(ppo_iterations)),
                "rollout_step_count": int(vec_env.num_envs) * min(max(1, int(max_steps)), 32) * max(1, int(ppo_iterations)),
                "usable_training_row_count": 0,
            },
        )
        return {
            "status": "runner_smoke_complete_no_success_evidence",
            "rl_trained": False,
            "runner_smoke_complete": True,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": True,
            "checkpoint_path": str(checkpoint),
            "unified_rl_config_json": str(config_path),
            "rsl_rl_runner_config_json": str(train_cfg_path),
            "unified_rl_training_curve_csv": str(curve_csv),
            "unified_rl_rollout_trace_csv": str(rollout_csv),
            "unified_rl_rollout_trace_jsonl": str(rollout_jsonl),
            "per_part_training_summary_csv": str(per_part_csv),
            "per_part_training_summary_json": str(per_part_json),
            "multi_object_training_summary_json": str(multi_json),
            "rsl_rl_runner_used": True,
            "surrogate_training_used": False,
            "ppo_iterations_completed": max(1, int(ppo_iterations)),
            "rollout_step_count": int(vec_env.num_envs) * min(max(1, int(max_steps)), 32) * max(1, int(ppo_iterations)),
            "rows": per_part,
        }
    except Exception as exc:
        return _write_training_not_started(
            run_path,
            config_path=str(config_path),
            policy_mode=policy_mode,
            parts=parts or V80_PARTS,
            reason=f"RSL_RL_RUNNER_FAILED:{type(exc).__name__}:{exc}",
        )


def train_v86_guarded_residual_ppo_smoke(
    run_dir: str | Path,
    *,
    policy_mode: str = "shared_multi_object_policy",
    parts: list[str] | None = None,
    num_envs: int = 5,
    max_steps: int = 32,
    physics_profile: str = "canonical",
    physical_backend: Any | None = None,
    ppo_iterations: int = 5,
    contact_gate_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    config = {
        "run_mode": "v86_guarded_residual_ppo_smoke",
        "policy_mode": policy_mode,
        "num_envs_requested": int(num_envs),
        "horizon": int(max_steps),
        "ppo_iterations_requested": int(ppo_iterations),
        "physics_profile": physics_profile,
        "parts": requested_parts,
        "use_v85_near_contact_resets": True,
        "checkpoint_is_runner_artifact_only": True,
        "final_training_export_allowed": False,
    }
    config_path = write_json(run_path / "v86_training_config.json", config)
    rsl_ok, rsl_reason = check_rsl_rl_available()
    if not rsl_ok:
        return _write_v86_training_not_started(run_path, requested_parts, policy_mode, str(config_path), rsl_reason)

    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    gate_by_part = {str(row.get("part_name") or ""): bool(row.get("v86_contact_gate_pass")) for row in contact_gate_rows or []}
    gate_ok = bool(gate_by_part) and all(bool(gate_by_part.get(part)) for part in requested_parts)
    if not backend_ready:
        return _write_v86_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            str(config_path),
            str(getattr(physical_backend, "blocker", "") or "single_context_physical_backend_not_ready"),
        )
    if not gate_ok:
        return _write_v86_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            str(config_path),
            "controlled_force_contact_gate_failed",
        )

    try:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        if callable(getattr(physical_backend, "clear_v86_rollout_trace", None)):
            physical_backend.clear_v86_rollout_trace()
        env_cfg = UnifiedGraspEnvCfg(
            num_envs=int(getattr(physical_backend, "num_envs", num_envs) or num_envs),
            parts=requested_parts,
            physics_profile=physics_profile,
            policy_mode=policy_mode,
            max_episode_length=max_steps,
            device=str(getattr(physical_backend, "device", "cpu")),
            use_diagnostic_dynamics=False,
            v86_guarded_residual_mode=True,
            use_v85_near_contact_resets=True,
        )
        env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
        vec_env = RslRlUnifiedGraspVecEnv(env)
        train_cfg = _rsl_rl_train_cfg(
            num_steps_per_env=min(max(1, int(max_steps)), 32),
            max_iterations=max(1, int(ppo_iterations)),
            policy_mode=policy_mode,
        )
        train_cfg["experiment_name"] = "v86_guarded_residual_ppo_smoke"
        train_cfg_path = write_json(run_path / "v86_rsl_rl_runner_config.json", train_cfg)
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(run_path / "v86_rsl_rl_logs"), device=vec_env.device)
        runner.learn(num_learning_iterations=max(1, int(ppo_iterations)), init_at_random_ep_len=False)
        checkpoint = run_path / "v86_runner_artifact_checkpoint.pt"
        runner.save(str(checkpoint))
        rollout_step_count = int(vec_env.num_envs) * min(max(1, int(max_steps)), 32) * max(1, int(ppo_iterations))
        curve_rows = [
            {
                "iteration": index,
                "policy_mode": policy_mode,
                "ppo_ran": True,
                "rsl_rl_runner_used": True,
                "runner_smoke_complete": True,
                "rl_trained": False,
                "rl_trained_success_evidence": False,
                "checkpoint_is_runner_artifact_only": True,
                "ppo_iterations_completed": max(1, int(ppo_iterations)),
                "rollout_step_count": rollout_step_count,
                "usable_training_row_count": 0,
            }
            for index in range(max(1, int(ppo_iterations)))
        ]
        curve_csv = write_csv(run_path / "v86_training_curve.csv", curve_rows)
        trace_rows = list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        reset_rows = list(getattr(physical_backend, "v86_near_contact_reset_rows", []) or [])
        trace_csv = write_csv(run_path / "v86_rollout_trace.csv", trace_rows)
        trace_jsonl = write_jsonl(run_path / "v86_rollout_trace.jsonl", trace_rows)
        reset_csv = write_csv(run_path / "v86_near_contact_reset_audit.csv", reset_rows)
        reset_jsonl = write_jsonl(run_path / "v86_near_contact_reset_audit.jsonl", reset_rows)
        per_part = summarize_v86_rollout(
            requested_parts,
            trace_rows,
            ppo_ran=True,
            runner_smoke_complete=True,
            checkpoint_path=str(checkpoint),
            ppo_iterations_completed=max(1, int(ppo_iterations)),
            rollout_step_count=rollout_step_count,
            num_envs_actual=int(vec_env.num_envs),
        )
        per_part_csv = write_csv(run_path / "v86_per_object_rollout_summary.csv", per_part)
        per_part_json = write_json(run_path / "v86_per_object_rollout_summary.json", per_part)
        return {
            "status": "runner_smoke_complete_no_success_evidence",
            "ppo_ran": True,
            "rl_trained": False,
            "runner_smoke_complete": True,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_written": True,
            "checkpoint_is_runner_artifact_only": True,
            "v86_training_config_json": str(config_path),
            "v86_rsl_rl_runner_config_json": str(train_cfg_path),
            "v86_training_curve_csv": str(curve_csv),
            "v86_rollout_trace_csv": str(trace_csv),
            "v86_rollout_trace_jsonl": str(trace_jsonl),
            "v86_near_contact_reset_audit_csv": str(reset_csv),
            "v86_near_contact_reset_audit_jsonl": str(reset_jsonl),
            "v86_per_object_rollout_summary_csv": str(per_part_csv),
            "v86_per_object_rollout_summary_json": str(per_part_json),
            "rsl_rl_runner_used": True,
            "surrogate_training_used": False,
            "ppo_iterations_completed": max(1, int(ppo_iterations)),
            "rollout_step_count": rollout_step_count,
            "num_envs_actual": int(vec_env.num_envs),
            "usable_training_row_count": 0,
            "rows": per_part,
        }
    except Exception as exc:
        return _write_v86_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            str(config_path),
            f"RSL_RL_RUNNER_FAILED:{type(exc).__name__}:{exc}",
        )


def train_v87_contact_guided_residual_ppo(
    run_dir: str | Path,
    *,
    policy_mode: str = "shared_multi_object_policy",
    parts: list[str] | None = None,
    num_envs: int = 5,
    max_steps: int = 96,
    physics_profile: str = "canonical",
    physical_backend: Any | None = None,
    ppo_iterations: int = 20,
    contact_gate_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    config = {
        "run_mode": "v87_contact_guided_residual_ppo",
        "policy_mode": policy_mode,
        "num_envs_requested": int(num_envs),
        "horizon": int(max_steps),
        "ppo_iterations_requested": int(ppo_iterations),
        "physics_profile": physics_profile,
        "parts": requested_parts,
        "use_v85_near_contact_resets": True,
        "policy_outputs_residual_only": True,
        "nominal_action_prior": "v85_phase_prior",
        "residual_scale_wrist_xyz": 0.15,
        "residual_scale_wrist_rot": 0.08,
        "residual_scale_finger": 0.20,
        "checkpoint_is_policy_artifact_only": True,
        "final_training_export_allowed": False,
    }
    config_path = write_json(run_path / "v87_training_config.json", config)
    rsl_ok, rsl_reason = check_rsl_rl_available()
    if not rsl_ok:
        return _write_v87_training_not_started(run_path, requested_parts, policy_mode, str(config_path), rsl_reason)

    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    gate_by_part = {str(row.get("part_name") or ""): bool(row.get("v86_contact_gate_pass")) for row in contact_gate_rows or []}
    gate_ok = bool(gate_by_part) and all(bool(gate_by_part.get(part)) for part in requested_parts)
    if not backend_ready:
        return _write_v87_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            str(config_path),
            str(getattr(physical_backend, "blocker", "") or "single_context_physical_backend_not_ready"),
        )
    if not gate_ok:
        return _write_v87_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            str(config_path),
            "controlled_force_contact_gate_failed",
        )

    try:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        if callable(getattr(physical_backend, "clear_v87_rollout_trace", None)):
            physical_backend.clear_v87_rollout_trace()
        elif callable(getattr(physical_backend, "clear_v86_rollout_trace", None)):
            physical_backend.clear_v86_rollout_trace()
        env_cfg = UnifiedGraspEnvCfg(
            num_envs=int(getattr(physical_backend, "num_envs", num_envs) or num_envs),
            parts=requested_parts,
            physics_profile=physics_profile,
            policy_mode=policy_mode,
            max_episode_length=max_steps,
            device=str(getattr(physical_backend, "device", "cpu")),
            use_diagnostic_dynamics=False,
            v86_guarded_residual_mode=True,
            v87_contact_guided_residual_mode=True,
            use_v85_near_contact_resets=True,
        )
        env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
        vec_env = RslRlUnifiedGraspVecEnv(env)
        train_cfg = _rsl_rl_train_cfg(
            num_steps_per_env=max(1, int(max_steps)),
            max_iterations=max(1, int(ppo_iterations)),
            policy_mode=policy_mode,
        )
        train_cfg["experiment_name"] = "v87_contact_guided_residual_ppo"
        train_cfg_path = write_json(run_path / "v87_rsl_rl_runner_config.json", train_cfg)
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(run_path / "v87_rsl_rl_logs"), device=vec_env.device)
        runner.learn(num_learning_iterations=max(1, int(ppo_iterations)), init_at_random_ep_len=False)
        checkpoint = run_path / "v87_policy_artifact_checkpoint.pt"
        runner.save(str(checkpoint))
        rollout_step_count = int(vec_env.num_envs) * max(1, int(max_steps)) * max(1, int(ppo_iterations))
        trace_rows = list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        reset_rows = list(getattr(physical_backend, "v86_near_contact_reset_rows", []) or [])
        action_prior_rows = list(getattr(physical_backend, "v87_action_prior_rows", []) or [])
        per_part = summarize_v87_rollout(
            requested_parts,
            trace_rows,
            ppo_ran=True,
            runner_complete=True,
            checkpoint_path=str(checkpoint),
            ppo_iterations_completed=max(1, int(ppo_iterations)),
            rollout_step_count=rollout_step_count,
            num_envs_actual=int(vec_env.num_envs),
        )
        curve_rows = [
            {
                "iteration": index,
                "policy_mode": policy_mode,
                "ppo_ran": True,
                "rsl_rl_runner_used": True,
                "runner_smoke_complete": True,
                "runner_policy_artifact_complete": True,
                "rl_trained": False,
                "rl_trained_success_evidence": False,
                "checkpoint_is_policy_artifact_only": True,
                "ppo_iterations_completed": max(1, int(ppo_iterations)),
                "rollout_step_count": rollout_step_count,
                "mean_force_contact_rate": sum(float(row.get("force_contact_rate") or 0.0) for row in per_part) / max(1, len(per_part)),
                "mean_contact_duration": sum(float(row.get("contact_duration_mean") or 0.0) for row in per_part) / max(1, len(per_part)),
                "usable_training_row_count": 0,
            }
            for index in range(max(1, int(ppo_iterations)))
        ]
        curve_csv = write_csv(run_path / "v87_contact_guided_training_curve.csv", curve_rows)
        trace_csv = write_csv(run_path / "v87_rollout_trace.csv", trace_rows)
        trace_jsonl = write_jsonl(run_path / "v87_rollout_trace.jsonl", trace_rows)
        reset_csv = write_csv(run_path / "v87_near_contact_reset_audit.csv", reset_rows)
        reset_jsonl = write_jsonl(run_path / "v87_near_contact_reset_audit.jsonl", reset_rows)
        action_prior_csv = write_csv(run_path / "v87_action_prior_audit.csv", action_prior_rows)
        action_prior_json = write_json(run_path / "v87_action_prior_audit.json", action_prior_rows)
        per_part_csv = write_csv(run_path / "v87_per_object_rollout_summary.csv", per_part)
        per_part_json = write_json(run_path / "v87_per_object_rollout_summary.json", per_part)
        return {
            "status": "runner_policy_artifact_complete_no_success_evidence",
            "ppo_ran": True,
            "rl_trained": False,
            "runner_smoke_complete": True,
            "runner_policy_artifact_complete": True,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_written": True,
            "checkpoint_is_policy_artifact_only": True,
            "v87_training_config_json": str(config_path),
            "v87_rsl_rl_runner_config_json": str(train_cfg_path),
            "v87_contact_guided_training_curve_csv": str(curve_csv),
            "v87_rollout_trace_csv": str(trace_csv),
            "v87_rollout_trace_jsonl": str(trace_jsonl),
            "v87_near_contact_reset_audit_csv": str(reset_csv),
            "v87_near_contact_reset_audit_jsonl": str(reset_jsonl),
            "v87_action_prior_audit_csv": str(action_prior_csv),
            "v87_action_prior_audit_json": str(action_prior_json),
            "v87_per_object_rollout_summary_csv": str(per_part_csv),
            "v87_per_object_rollout_summary_json": str(per_part_json),
            "rsl_rl_runner_used": True,
            "surrogate_training_used": False,
            "ppo_iterations_completed": max(1, int(ppo_iterations)),
            "rollout_step_count": rollout_step_count,
            "num_envs_actual": int(vec_env.num_envs),
            "usable_training_row_count": 0,
            "rows": per_part,
        }
    except Exception as exc:
        return _write_v87_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            str(config_path),
            f"RSL_RL_RUNNER_FAILED:{type(exc).__name__}:{exc}",
        )


def train_v88_stabilized_residual_ppo(
    run_dir: str | Path,
    *,
    branch_name: str,
    policy_mode: str = "shared_multi_object_policy",
    parts: list[str] | None = None,
    num_envs: int = 5,
    max_steps: int = 128,
    physics_profile: str = "canonical",
    physical_backend: Any | None = None,
    ppo_iterations: int = 20,
    contact_gate_rows: list[dict[str, Any]] | None = None,
    enhanced_observation: bool = False,
    resume_checkpoint_path: str = "",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    config = {
        "run_mode": "v88_stabilized_grasp_policy_and_physics_audit",
        "branch_name": branch_name,
        "policy_mode": policy_mode,
        "num_envs_requested": int(num_envs),
        "horizon": int(max_steps),
        "ppo_iterations_requested": int(ppo_iterations),
        "physics_profile": physics_profile,
        "parts": requested_parts,
        "use_v85_near_contact_resets": True,
        "v88_stabilized_mode": True,
        "v88_observation_enhanced": bool(enhanced_observation),
        "v88_nominal_lift_phase_enabled": True,
        "v88_auto_reset_on_done": True,
        "policy_outputs_residual_only": True,
        "nominal_action_prior": "v85_v88_phase_prior_with_lift_stabilize",
        "residual_scale_wrist_xyz": 0.12,
        "residual_scale_wrist_rot": 0.06,
        "residual_scale_finger": 0.16,
        "checkpoint_is_policy_artifact_only": True,
        "final_training_export_allowed": False,
        "sticky_eval_allowed": False,
        "final_video_allowed": False,
        "resume_checkpoint_path": resume_checkpoint_path,
    }
    config_path = write_json(run_path / "v88_training_config.json", config)
    rsl_ok, rsl_reason = check_rsl_rl_available()
    if not rsl_ok:
        return _write_v88_training_not_started(run_path, requested_parts, policy_mode, branch_name, str(config_path), rsl_reason)

    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    gate_by_part = {str(row.get("part_name") or ""): bool(row.get("v86_contact_gate_pass")) for row in contact_gate_rows or []}
    gate_ok = bool(gate_by_part) and all(bool(gate_by_part.get(part)) for part in requested_parts)
    if not backend_ready:
        return _write_v88_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            branch_name,
            str(config_path),
            str(getattr(physical_backend, "blocker", "") or "single_context_physical_backend_not_ready"),
        )
    if not gate_ok:
        return _write_v88_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            branch_name,
            str(config_path),
            "controlled_force_contact_gate_failed",
        )

    try:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        if callable(getattr(physical_backend, "clear_v88_rollout_trace", None)):
            physical_backend.clear_v88_rollout_trace()
        elif callable(getattr(physical_backend, "clear_v87_rollout_trace", None)):
            physical_backend.clear_v87_rollout_trace()
        env_cfg = UnifiedGraspEnvCfg(
            num_envs=int(getattr(physical_backend, "num_envs", num_envs) or num_envs),
            parts=requested_parts,
            physics_profile=physics_profile,
            policy_mode=policy_mode,
            max_episode_length=max_steps,
            device=str(getattr(physical_backend, "device", "cpu")),
            use_diagnostic_dynamics=False,
            v86_guarded_residual_mode=True,
            v87_contact_guided_residual_mode=True,
            v88_stabilized_mode=True,
            v88_observation_enhanced=bool(enhanced_observation),
            v88_auto_reset_on_done=True,
            v88_nominal_lift_phase_enabled=True,
            use_v85_near_contact_resets=True,
            v87_residual_scale_wrist_xyz=0.12,
            v87_residual_scale_wrist_rot=0.06,
            v87_residual_scale_finger=0.16,
        )
        env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
        vec_env = RslRlUnifiedGraspVecEnv(env)
        train_cfg = _rsl_rl_train_cfg(
            num_steps_per_env=max(1, int(max_steps)),
            max_iterations=max(1, int(ppo_iterations)),
            policy_mode=policy_mode,
        )
        train_cfg["experiment_name"] = f"v88_{branch_name}"
        train_cfg_path = write_json(run_path / "v88_rsl_rl_runner_config.json", train_cfg)
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(run_path / "v88_rsl_rl_logs"), device=vec_env.device)
        warmstart_used = False
        warmstart_failure_reason = ""
        if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
            try:
                runner.load(str(resume_checkpoint_path))
                warmstart_used = True
            except Exception as exc:  # keep branch honest but runnable from scratch
                warmstart_failure_reason = f"{type(exc).__name__}:{exc}"
        runner.learn(num_learning_iterations=max(1, int(ppo_iterations)), init_at_random_ep_len=False)
        checkpoint = run_path / f"v88_{branch_name}_policy_artifact_checkpoint.pt"
        runner.save(str(checkpoint))
        rollout_step_count = int(vec_env.num_envs) * max(1, int(max_steps)) * max(1, int(ppo_iterations))
        trace_rows = list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        reset_rows = list(getattr(physical_backend, "v86_near_contact_reset_rows", []) or [])
        action_prior_rows = list(getattr(physical_backend, "v87_action_prior_rows", []) or [])
        per_part = summarize_v88_rollout(
            requested_parts,
            trace_rows,
            branch_name=branch_name,
            physics_profile=physics_profile,
            ppo_ran=True,
            runner_complete=True,
            checkpoint_path=str(checkpoint),
            ppo_iterations_completed=max(1, int(ppo_iterations)),
            rollout_step_count=rollout_step_count,
            num_envs_actual=int(vec_env.num_envs),
        )
        curve_rows = [
            {
                "iteration": index,
                "branch_name": branch_name,
                "policy_mode": policy_mode,
                "physics_profile": physics_profile,
                "v88_observation_enhanced": bool(enhanced_observation),
                "ppo_ran": True,
                "rsl_rl_runner_used": True,
                "runner_policy_artifact_complete": True,
                "runner_smoke_complete": True,
                "rl_trained": False,
                "rl_trained_success_evidence": False,
                "checkpoint_is_policy_artifact_only": True,
                "ppo_iterations_completed": max(1, int(ppo_iterations)),
                "rollout_step_count": rollout_step_count,
                "mean_contact_stability_score": sum(float(row.get("contact_stability_score") or 0.0) for row in per_part) / max(1, len(per_part)),
                "mean_hold_displacement_m": sum(float(row.get("hold_object_displacement_max_m") or 0.0) for row in per_part) / max(1, len(per_part)),
                "usable_training_row_count": 0,
                "warmstart_used": warmstart_used,
                "warmstart_checkpoint_path": resume_checkpoint_path if warmstart_used else "",
                "warmstart_failure_reason": warmstart_failure_reason,
            }
            for index in range(max(1, int(ppo_iterations)))
        ]
        curve_csv = write_csv(run_path / "v88_contact_guided_training_curve.csv", curve_rows)
        trace_csv = write_csv(run_path / "v88_rollout_trace.csv", trace_rows)
        trace_jsonl = write_jsonl(run_path / "v88_rollout_trace.jsonl", trace_rows)
        reset_csv = write_csv(run_path / "v88_near_contact_reset_audit.csv", reset_rows)
        reset_jsonl = write_jsonl(run_path / "v88_near_contact_reset_audit.jsonl", reset_rows)
        action_prior_csv = write_csv(run_path / "v88_action_prior_audit.csv", action_prior_rows)
        action_prior_json = write_json(run_path / "v88_action_prior_audit.json", action_prior_rows)
        per_part_csv = write_csv(run_path / "v88_per_object_rollout_summary.csv", per_part)
        per_part_json = write_json(run_path / "v88_per_object_rollout_summary.json", per_part)
        return {
            "status": "runner_policy_artifact_complete_no_success_evidence",
            "branch_name": branch_name,
            "ppo_ran": True,
            "rl_trained": False,
            "runner_smoke_complete": True,
            "runner_policy_artifact_complete": True,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_written": True,
            "checkpoint_is_policy_artifact_only": True,
            "v88_training_config_json": str(config_path),
            "v88_rsl_rl_runner_config_json": str(train_cfg_path),
            "v88_contact_guided_training_curve_csv": str(curve_csv),
            "v88_rollout_trace_csv": str(trace_csv),
            "v88_rollout_trace_jsonl": str(trace_jsonl),
            "v88_near_contact_reset_audit_csv": str(reset_csv),
            "v88_near_contact_reset_audit_jsonl": str(reset_jsonl),
            "v88_action_prior_audit_csv": str(action_prior_csv),
            "v88_action_prior_audit_json": str(action_prior_json),
            "v88_per_object_rollout_summary_csv": str(per_part_csv),
            "v88_per_object_rollout_summary_json": str(per_part_json),
            "rsl_rl_runner_used": True,
            "surrogate_training_used": False,
            "ppo_iterations_completed": max(1, int(ppo_iterations)),
            "rollout_step_count": rollout_step_count,
            "num_envs_actual": int(vec_env.num_envs),
            "usable_training_row_count": 0,
            "warmstart_used": warmstart_used,
            "warmstart_checkpoint_path": resume_checkpoint_path if warmstart_used else "",
            "warmstart_failure_reason": warmstart_failure_reason,
            "rows": per_part,
        }
    except Exception as exc:
        return _write_v88_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            branch_name,
            str(config_path),
            f"RSL_RL_RUNNER_FAILED:{type(exc).__name__}:{exc}",
        )


def summarize_v88_rollout(
    parts: list[str],
    trace_rows: list[dict[str, Any]],
    *,
    branch_name: str,
    physics_profile: str,
    ppo_ran: bool,
    runner_complete: bool,
    checkpoint_path: str,
    ppo_iterations_completed: int,
    rollout_step_count: int,
    num_envs_actual: int,
    eval_rows_by_part: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = summarize_v87_rollout(
        parts,
        trace_rows,
        ppo_ran=ppo_ran,
        runner_complete=runner_complete,
        checkpoint_path=checkpoint_path,
        ppo_iterations_completed=ppo_iterations_completed,
        rollout_step_count=rollout_step_count,
        num_envs_actual=num_envs_actual,
        eval_rows_by_part=eval_rows_by_part,
    )
    for row in rows:
        part_trace = [item for item in trace_rows if item.get("part_name") == row.get("part_name")]
        hold_displacements = [float(item.get("hold_object_displacement_m") or 0.0) for item in part_trace]
        lift_deltas = [float(item.get("lift_delta_z_m") or 0.0) for item in part_trace]
        fallback_used = any(bool(item.get("fallback_success_used")) for item in part_trace)
        row.update(
            {
                "branch_name": branch_name,
                "physics_profile": physics_profile,
                "v88_stabilized_mode": True,
                "hold_object_displacement_max_m": max([0.0, *hold_displacements]),
                "lift_delta_z_max_m": max([0.0, *lift_deltas]),
                "fallback_success_used": fallback_used,
                "status": "V88_POLICY_ARTIFACT_COMPLETE" if runner_complete else "TRAINING_NOT_STARTED",
                "next_action": "run_v88_deterministic_no_sticky_eval" if runner_complete else "fix_v88_training_gate",
            }
        )
    return rows


def _write_v88_training_not_started(
    run_path: Path,
    parts: list[str],
    policy_mode: str,
    branch_name: str,
    config_path: str,
    reason: str,
) -> dict[str, Any]:
    rows = [
        {
            "part_name": part,
            "branch_name": branch_name,
            "policy_mode": policy_mode,
            "ppo_ran": False,
            "rsl_rl_runner_used": False,
            "runner_smoke_complete": False,
            "runner_policy_artifact_complete": False,
            "rl_trained": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": False,
            "checkpoint_written": False,
            "checkpoint_is_policy_artifact_only": False,
            "ppo_iterations_completed": 0,
            "rollout_step_count": 0,
            "force_contact_rate": 0.0,
            "contact_duration_mean": 0.0,
            "contact_stability_score": 0.0,
            "hold_object_displacement_max_m": 0.0,
            "lift_delta_z_max_m": 0.0,
            "usable_training_row_count": 0,
            "status": "TRAINING_NOT_STARTED",
            "blocker": reason,
        }
        for part in parts
    ]
    curve_csv = write_csv(run_path / "v88_contact_guided_training_curve.csv", [])
    trace_csv = write_csv(run_path / "v88_rollout_trace.csv", [])
    trace_jsonl = write_jsonl(run_path / "v88_rollout_trace.jsonl", [])
    reset_csv = write_csv(run_path / "v88_near_contact_reset_audit.csv", [])
    reset_jsonl = write_jsonl(run_path / "v88_near_contact_reset_audit.jsonl", [])
    action_prior_csv = write_csv(run_path / "v88_action_prior_audit.csv", [])
    action_prior_json = write_json(run_path / "v88_action_prior_audit.json", [])
    per_part_csv = write_csv(run_path / "v88_per_object_rollout_summary.csv", rows)
    per_part_json = write_json(run_path / "v88_per_object_rollout_summary.json", rows)
    return {
        "status": "TRAINING_NOT_STARTED",
        "branch_name": branch_name,
        "ppo_ran": False,
        "rl_trained": False,
        "runner_smoke_complete": False,
        "runner_policy_artifact_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "checkpoint_written": False,
        "checkpoint_is_policy_artifact_only": False,
        "rsl_rl_runner_used": False,
        "surrogate_training_used": False,
        "ppo_iterations_completed": 0,
        "rollout_step_count": 0,
        "usable_training_row_count": 0,
        "v88_training_config_json": config_path,
        "v88_contact_guided_training_curve_csv": str(curve_csv),
        "v88_rollout_trace_csv": str(trace_csv),
        "v88_rollout_trace_jsonl": str(trace_jsonl),
        "v88_near_contact_reset_audit_csv": str(reset_csv),
        "v88_near_contact_reset_audit_jsonl": str(reset_jsonl),
        "v88_action_prior_audit_csv": str(action_prior_csv),
        "v88_action_prior_audit_json": str(action_prior_json),
        "v88_per_object_rollout_summary_csv": str(per_part_csv),
        "v88_per_object_rollout_summary_json": str(per_part_json),
        "failure_reason": reason,
        "rows": rows,
    }


def train_v89_hybrid_residual_ppo(
    run_dir: str | Path,
    *,
    branch_name: str,
    policy_mode: str = "shared_multi_object_policy",
    parts: list[str] | None = None,
    num_envs: int = 5,
    max_steps: int = 128,
    physics_profile: str = "canonical",
    physical_backend: Any | None = None,
    ppo_iterations: int = 20,
    contact_gate_rows: list[dict[str, Any]] | None = None,
    use_candidate_prior: bool = False,
    selected_candidates: dict[str, dict[str, Any]] | None = None,
    bc_rows: list[dict[str, Any]] | None = None,
    bc_epochs: int = 3,
    resume_checkpoint_path: str = "",
    disk_space_ok: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    selected_candidates = selected_candidates or {}
    config = {
        "run_mode": "v89_failure_driven_hybrid_grasp_repair",
        "branch_name": branch_name,
        "policy_mode": policy_mode,
        "num_envs_requested": int(num_envs),
        "horizon": int(max_steps),
        "ppo_iterations_requested": int(ppo_iterations),
        "physics_profile": physics_profile,
        "parts": requested_parts,
        "use_v85_near_contact_resets": True,
        "v89_hybrid_repair_mode": bool(use_candidate_prior),
        "v89_nominal_candidate_mode": bool(use_candidate_prior),
        "v88_stabilized_mode": True,
        "v88_nominal_lift_phase_enabled": True,
        "v88_auto_reset_on_done": True,
        "policy_outputs_residual_only": True,
        "nominal_action_prior": "v89_geometry_candidate_prior" if use_candidate_prior else "v88_best_baseline_prior",
        "selected_candidate_ids": {part: str(row.get("candidate_id") or "") for part, row in selected_candidates.items()},
        "residual_scale_wrist_xyz": 0.10 if use_candidate_prior else 0.12,
        "residual_scale_wrist_rot": 0.05 if use_candidate_prior else 0.06,
        "residual_scale_finger": 0.12 if use_candidate_prior else 0.16,
        "bc_epochs_requested": int(bc_epochs),
        "bc_valid_row_count": len(bc_rows or []),
        "checkpoint_is_policy_artifact_only": True,
        "final_training_export_allowed": False,
        "sticky_eval_allowed": False,
        "final_video_allowed": False,
        "resume_checkpoint_path": resume_checkpoint_path,
    }
    config_path = write_json(run_path / "v89_training_config.json", config)
    if not disk_space_ok:
        return _write_v89_training_not_started(run_path, requested_parts, policy_mode, branch_name, str(config_path), "DISK_SPACE_LOW")
    rsl_ok, rsl_reason = check_rsl_rl_available()
    if not rsl_ok:
        return _write_v89_training_not_started(run_path, requested_parts, policy_mode, branch_name, str(config_path), rsl_reason)
    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    gate_by_part = {str(row.get("part_name") or ""): bool(row.get("v86_contact_gate_pass")) for row in contact_gate_rows or []}
    gate_ok = bool(gate_by_part) and all(bool(gate_by_part.get(part)) for part in requested_parts)
    if not backend_ready:
        return _write_v89_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            branch_name,
            str(config_path),
            str(getattr(physical_backend, "blocker", "") or "single_context_physical_backend_not_ready"),
        )
    if not gate_ok:
        return _write_v89_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            branch_name,
            str(config_path),
            "controlled_force_contact_gate_failed",
        )
    if use_candidate_prior and not selected_candidates:
        return _write_v89_training_not_started(run_path, requested_parts, policy_mode, branch_name, str(config_path), "NO_SELECTED_V89_CANDIDATE")
    if branch_name == "bc_warmstart_residual_ppo" and not bc_rows:
        return _write_v89_training_not_started(run_path, requested_parts, policy_mode, branch_name, str(config_path), "NO_VALID_BC_TRAJECTORY")

    try:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        if use_candidate_prior and callable(getattr(physical_backend, "configure_v89_candidates", None)):
            physical_backend.configure_v89_candidates(selected_candidates)
            if callable(getattr(physical_backend, "configure_v86_staging", None)):
                physical_backend.configure_v86_staging(selected_candidates)
        if callable(getattr(physical_backend, "clear_v89_rollout_trace", None)):
            physical_backend.clear_v89_rollout_trace()
        elif callable(getattr(physical_backend, "clear_v88_rollout_trace", None)):
            physical_backend.clear_v88_rollout_trace()
        env_cfg = UnifiedGraspEnvCfg(
            num_envs=int(getattr(physical_backend, "num_envs", num_envs) or num_envs),
            parts=requested_parts,
            physics_profile=physics_profile,
            policy_mode=policy_mode,
            max_episode_length=max_steps,
            device=str(getattr(physical_backend, "device", "cpu")),
            use_diagnostic_dynamics=False,
            v86_guarded_residual_mode=True,
            v87_contact_guided_residual_mode=True,
            v88_stabilized_mode=True,
            v89_hybrid_repair_mode=bool(use_candidate_prior),
            v89_nominal_candidate_mode=bool(use_candidate_prior),
            v88_observation_enhanced=bool(use_candidate_prior),
            v88_auto_reset_on_done=True,
            v88_nominal_lift_phase_enabled=True,
            use_v85_near_contact_resets=True,
            v87_residual_scale_wrist_xyz=0.10 if use_candidate_prior else 0.12,
            v87_residual_scale_wrist_rot=0.05 if use_candidate_prior else 0.06,
            v87_residual_scale_finger=0.12 if use_candidate_prior else 0.16,
        )
        env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
        vec_env = RslRlUnifiedGraspVecEnv(env)
        train_cfg = _rsl_rl_train_cfg(
            num_steps_per_env=max(1, int(max_steps)),
            max_iterations=max(1, int(ppo_iterations)),
            policy_mode=policy_mode,
        )
        train_cfg["experiment_name"] = f"v89_{branch_name}"
        train_cfg_path = write_json(run_path / "v89_rsl_rl_runner_config.json", train_cfg)
        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(run_path / "v89_rsl_rl_logs"), device=vec_env.device)
        warmstart_used = False
        warmstart_failure_reason = ""
        if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
            try:
                runner.load(str(resume_checkpoint_path))
                warmstart_used = True
            except Exception as exc:
                warmstart_failure_reason = f"{type(exc).__name__}:{exc}"
        bc_summary = _run_v89_bc_warmstart(run_path, runner, vec_env, branch_name=branch_name, bc_epochs=bc_epochs) if branch_name == "bc_warmstart_residual_ppo" else {"bc_warmstart_attempted": False}
        runner.learn(num_learning_iterations=max(1, int(ppo_iterations)), init_at_random_ep_len=False)
        checkpoint = run_path / f"v89_{branch_name}_policy_artifact_checkpoint.pt"
        runner.save(str(checkpoint))
        rollout_step_count = int(vec_env.num_envs) * max(1, int(max_steps)) * max(1, int(ppo_iterations))
        trace_rows = list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        reset_rows = list(getattr(physical_backend, "v86_near_contact_reset_rows", []) or [])
        action_prior_rows = list(getattr(physical_backend, "v87_action_prior_rows", []) or [])
        per_part = summarize_v89_rollout(
            requested_parts,
            trace_rows,
            branch_name=branch_name,
            physics_profile=physics_profile,
            ppo_ran=True,
            runner_complete=True,
            checkpoint_path=str(checkpoint),
            ppo_iterations_completed=max(1, int(ppo_iterations)),
            rollout_step_count=rollout_step_count,
            num_envs_actual=int(vec_env.num_envs),
        )
        curve_rows = [
            {
                "iteration": index,
                "branch_name": branch_name,
                "policy_mode": policy_mode,
                "physics_profile": physics_profile,
                "v89_nominal_candidate_mode": bool(use_candidate_prior),
                "ppo_ran": True,
                "rsl_rl_runner_used": True,
                "runner_policy_artifact_complete": True,
                "runner_smoke_complete": True,
                "rl_trained": False,
                "rl_trained_success_evidence": False,
                "checkpoint_is_policy_artifact_only": True,
                "ppo_iterations_completed": max(1, int(ppo_iterations)),
                "rollout_step_count": rollout_step_count,
                "mean_contact_stability_score": sum(float(row.get("contact_stability_score") or 0.0) for row in per_part) / max(1, len(per_part)),
                "mean_multi_finger_support_rate": sum(float(row.get("multi_finger_support_rate") or 0.0) for row in per_part) / max(1, len(per_part)),
                "mean_action_jerk": sum(float(row.get("action_jerk_mean") or 0.0) for row in per_part) / max(1, len(per_part)),
                "usable_training_row_count": 0,
                "warmstart_used": warmstart_used,
                "warmstart_checkpoint_path": resume_checkpoint_path if warmstart_used else "",
                "warmstart_failure_reason": warmstart_failure_reason,
                **{key: value for key, value in bc_summary.items() if key != "rows"},
            }
            for index in range(max(1, int(ppo_iterations)))
        ]
        curve_csv = write_csv(run_path / "v89_contact_guided_training_curve.csv", curve_rows)
        trace_csv = write_csv(run_path / "v89_rollout_trace.csv", trace_rows)
        trace_jsonl = write_jsonl(run_path / "v89_rollout_trace.jsonl", trace_rows)
        reset_csv = write_csv(run_path / "v89_near_contact_reset_audit.csv", reset_rows)
        reset_jsonl = write_jsonl(run_path / "v89_near_contact_reset_audit.jsonl", reset_rows)
        action_prior_csv = write_csv(run_path / "v89_action_prior_audit.csv", action_prior_rows)
        action_prior_json = write_json(run_path / "v89_action_prior_audit.json", action_prior_rows)
        per_part_csv = write_csv(run_path / "v89_per_object_rollout_summary.csv", per_part)
        per_part_json = write_json(run_path / "v89_per_object_rollout_summary.json", per_part)
        return {
            "status": "runner_policy_artifact_complete_no_success_evidence",
            "branch_name": branch_name,
            "ppo_ran": True,
            "rl_trained": False,
            "runner_smoke_complete": True,
            "runner_policy_artifact_complete": True,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_written": True,
            "checkpoint_is_policy_artifact_only": True,
            "v89_training_config_json": str(config_path),
            "v89_rsl_rl_runner_config_json": str(train_cfg_path),
            "v89_contact_guided_training_curve_csv": str(curve_csv),
            "v89_rollout_trace_csv": str(trace_csv),
            "v89_rollout_trace_jsonl": str(trace_jsonl),
            "v89_near_contact_reset_audit_csv": str(reset_csv),
            "v89_near_contact_reset_audit_jsonl": str(reset_jsonl),
            "v89_action_prior_audit_csv": str(action_prior_csv),
            "v89_action_prior_audit_json": str(action_prior_json),
            "v89_per_object_rollout_summary_csv": str(per_part_csv),
            "v89_per_object_rollout_summary_json": str(per_part_json),
            "rsl_rl_runner_used": True,
            "surrogate_training_used": False,
            "ppo_iterations_completed": max(1, int(ppo_iterations)),
            "rollout_step_count": rollout_step_count,
            "num_envs_actual": int(vec_env.num_envs),
            "usable_training_row_count": 0,
            "warmstart_used": warmstart_used,
            "warmstart_checkpoint_path": resume_checkpoint_path if warmstart_used else "",
            "warmstart_failure_reason": warmstart_failure_reason,
            **{key: value for key, value in bc_summary.items() if key != "rows"},
            "rows": per_part,
        }
    except Exception as exc:
        return _write_v89_training_not_started(
            run_path,
            requested_parts,
            policy_mode,
            branch_name,
            str(config_path),
            f"RSL_RL_RUNNER_FAILED:{type(exc).__name__}:{exc}",
        )


def _run_v89_bc_warmstart(run_path: Path, runner: Any, vec_env: RslRlUnifiedGraspVecEnv, *, branch_name: str, bc_epochs: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    attempted = True
    complete = False
    failure = ""
    losses: list[float] = []
    if torch is None:
        failure = "torch_unavailable_for_bc_warmstart"
    else:
        try:
            actor = getattr(getattr(runner, "alg", None), "actor", None)
            optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
            if actor is None or optimizer is None:
                raise RuntimeError("runner_actor_or_optimizer_missing")
            vec_env.reset()
            for epoch in range(max(1, int(bc_epochs))):
                obs = vec_env.get_observations()
                pred = actor(obs) if callable(actor) else None
                if pred is None:
                    raise RuntimeError("actor_forward_returned_none")
                if isinstance(pred, dict):
                    pred = pred["action"] if "action" in pred else pred.get("actions")
                if pred is None:
                    raise RuntimeError("actor_forward_output_missing_action_tensor")
                target = torch.zeros_like(pred)
                loss = torch.mean((pred - target) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                rows.append({"epoch": epoch, "branch_name": branch_name, "bc_mse_loss": loss_value, "bc_target": "zero_residual_under_selected_candidate_prior"})
            complete = True
        except Exception as exc:
            failure = f"{type(exc).__name__}:{exc}"
    csv_path = write_csv(run_path / "v89_bc_warmstart_summary.csv", rows)
    json_path = write_json(run_path / "v89_bc_warmstart_summary.json", rows)
    return {
        "bc_warmstart_attempted": attempted,
        "bc_warmstart_complete": complete,
        "bc_epochs_completed": len(rows),
        "bc_loss_final": losses[-1] if losses else 0.0,
        "bc_warmstart_failure_reason": failure,
        "v89_bc_warmstart_summary_csv": str(csv_path),
        "v89_bc_warmstart_summary_json": str(json_path),
        "rows": rows,
    }


def summarize_v89_rollout(
    parts: list[str],
    trace_rows: list[dict[str, Any]],
    *,
    branch_name: str,
    physics_profile: str,
    ppo_ran: bool,
    runner_complete: bool,
    checkpoint_path: str,
    ppo_iterations_completed: int,
    rollout_step_count: int,
    num_envs_actual: int,
    eval_rows_by_part: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = summarize_v88_rollout(
        parts,
        trace_rows,
        branch_name=branch_name,
        physics_profile=physics_profile,
        ppo_ran=ppo_ran,
        runner_complete=runner_complete,
        checkpoint_path=checkpoint_path,
        ppo_iterations_completed=ppo_iterations_completed,
        rollout_step_count=rollout_step_count,
        num_envs_actual=num_envs_actual,
        eval_rows_by_part=eval_rows_by_part,
    )
    for row in rows:
        part = str(row.get("part_name") or "")
        part_trace = [item for item in trace_rows if item.get("part_name") == part]
        multi = [
            item
            for item in part_trace
            if int(item.get("effective_contact_count_force") or item.get("force_contact_count") or 0) >= 2
        ]
        jerks = [float(item.get("action_jerk") or 0.0) for item in part_trace]
        candidate_ids = sorted({str(item.get("candidate_id") or "") for item in part_trace if item.get("candidate_id")})
        row.update(
            {
                "branch_name": branch_name,
                "physics_profile": physics_profile,
                "v89_hybrid_repair_mode": True,
                "candidate_id": ",".join(candidate_ids),
                "multi_finger_support_rate": len(multi) / max(1, len(part_trace)),
                "action_jerk_mean": sum(jerks) / max(1, len(jerks)),
                "fallback_success_used": any(bool(item.get("fallback_success_used")) for item in part_trace),
                "status": "V89_POLICY_ARTIFACT_COMPLETE" if runner_complete else "TRAINING_NOT_STARTED",
                "next_action": "run_v89_deterministic_no_sticky_eval" if runner_complete else "fix_v89_training_gate",
            }
        )
    return rows


def _write_v89_training_not_started(
    run_path: Path,
    parts: list[str],
    policy_mode: str,
    branch_name: str,
    config_path: str,
    reason: str,
) -> dict[str, Any]:
    rows = [
        {
            "part_name": part,
            "branch_name": branch_name,
            "policy_mode": policy_mode,
            "ppo_ran": False,
            "rsl_rl_runner_used": False,
            "runner_smoke_complete": False,
            "runner_policy_artifact_complete": False,
            "rl_trained": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": False,
            "checkpoint_written": False,
            "checkpoint_is_policy_artifact_only": False,
            "ppo_iterations_completed": 0,
            "rollout_step_count": 0,
            "force_contact_rate": 0.0,
            "contact_duration_mean": 0.0,
            "contact_stability_score": 0.0,
            "support_gate_rate": 0.0,
            "hold_gate_rate": 0.0,
            "lift_gate_rate": 0.0,
            "multi_finger_support_rate": 0.0,
            "action_jerk_mean": 0.0,
            "object_displacement_max_m": 0.0,
            "usable_training_row_count": 0,
            "status": "TRAINING_NOT_STARTED",
            "blocker": reason,
        }
        for part in parts
    ]
    curve_csv = write_csv(run_path / "v89_contact_guided_training_curve.csv", [])
    trace_csv = write_csv(run_path / "v89_rollout_trace.csv", [])
    trace_jsonl = write_jsonl(run_path / "v89_rollout_trace.jsonl", [])
    reset_csv = write_csv(run_path / "v89_near_contact_reset_audit.csv", [])
    reset_jsonl = write_jsonl(run_path / "v89_near_contact_reset_audit.jsonl", [])
    action_prior_csv = write_csv(run_path / "v89_action_prior_audit.csv", [])
    action_prior_json = write_json(run_path / "v89_action_prior_audit.json", [])
    per_part_csv = write_csv(run_path / "v89_per_object_rollout_summary.csv", rows)
    per_part_json = write_json(run_path / "v89_per_object_rollout_summary.json", rows)
    return {
        "status": "TRAINING_NOT_STARTED",
        "branch_name": branch_name,
        "ppo_ran": False,
        "rl_trained": False,
        "runner_smoke_complete": False,
        "runner_policy_artifact_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "checkpoint_written": False,
        "checkpoint_is_policy_artifact_only": False,
        "rsl_rl_runner_used": False,
        "surrogate_training_used": False,
        "ppo_iterations_completed": 0,
        "rollout_step_count": 0,
        "usable_training_row_count": 0,
        "v89_training_config_json": config_path,
        "v89_contact_guided_training_curve_csv": str(curve_csv),
        "v89_rollout_trace_csv": str(trace_csv),
        "v89_rollout_trace_jsonl": str(trace_jsonl),
        "v89_near_contact_reset_audit_csv": str(reset_csv),
        "v89_near_contact_reset_audit_jsonl": str(reset_jsonl),
        "v89_action_prior_audit_csv": str(action_prior_csv),
        "v89_action_prior_audit_json": str(action_prior_json),
        "v89_per_object_rollout_summary_csv": str(per_part_csv),
        "v89_per_object_rollout_summary_json": str(per_part_json),
        "failure_reason": reason,
        "rows": rows,
    }


def summarize_v87_rollout(
    parts: list[str],
    trace_rows: list[dict[str, Any]],
    *,
    ppo_ran: bool,
    runner_complete: bool,
    checkpoint_path: str,
    ppo_iterations_completed: int,
    rollout_step_count: int,
    num_envs_actual: int,
    eval_rows_by_part: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    eval_rows_by_part = eval_rows_by_part or {}
    rows = []
    threshold = 0.05
    for part in parts:
        part_trace = [row for row in trace_rows if row.get("part_name") == part]
        force_hits = [
            row
            for row in part_trace
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
        ]
        streaks = [float(row.get("force_contact_streak_steps") or row.get("contact_duration_steps") or 0.0) for row in part_trace]
        peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in part_trace]
        displacements = [float(row.get("object_displacement_m") or 0.0) for row in part_trace]
        excessive = [row for row in part_trace if bool(row.get("excessive_force"))]
        support_rows = [row for row in part_trace if bool(row.get("support_gate_ok"))]
        hold_rows = [row for row in part_trace if str(row.get("nominal_phase") or "") == "hold_squeeze"]
        hold_hits = [row for row in hold_rows if row in force_hits or bool(row.get("support_gate_ok"))]
        eval_row = eval_rows_by_part.get(part, {})
        force_rate = len(force_hits) / max(1, len(part_trace))
        contact_duration_mean = sum(streaks) / max(1, len(streaks))
        support_rate = len(support_rows) / max(1, len(part_trace))
        rows.append(
            {
                "part_name": part,
                "rollout_env_count": len({int(row.get("env_index") or -1) for row in part_trace if row.get("env_index") not in ("", None)}),
                "rollout_participation_count": len(part_trace),
                "rollout_step_count": int(rollout_step_count),
                "actual_env_count": int(num_envs_actual),
                "force_contact_rate": force_rate,
                "contact_duration_mean": contact_duration_mean,
                "contact_stability_score": force_rate + min(contact_duration_mean, 96.0) / 96.0,
                "support_gate_rate": support_rate,
                "hold_gate_rate": len(hold_hits) / max(1, len(hold_rows)),
                "lift_gate_rate": float(eval_row.get("lift_gate_rate") or 0.0),
                "force_contact_peak_n": max([0.0, *peaks]),
                "force_contact_mean_n": sum(peaks) / max(1, len(peaks)),
                "mean_force_n": sum(peaks) / max(1, len(peaks)),
                "peak_force_n": max([0.0, *peaks]),
                "excessive_force_threshold_n": 150.0,
                "excessive_force_rate": len(excessive) / max(1, len(part_trace)),
                "object_displacement_max_m": max([0.0, *displacements]),
                "object_write_by_policy_detected": any(bool(row.get("object_write_by_policy_detected")) for row in part_trace),
                "sticky_action_available_to_policy": any(bool(row.get("sticky_action_available_to_policy")) for row in part_trace),
                "support_eval_ok": bool(eval_row.get("support_eval_ok", False)),
                "hold_eval_ok": bool(eval_row.get("hold_eval_ok", False)),
                "lift_eval_ok": bool(eval_row.get("lift_eval_ok", False)),
                "deterministic_eval_result": str(eval_row.get("status") or ""),
                "grasp_success_claimed": bool(eval_row.get("grasp_success_claimed", False)),
                "ppo_ran": bool(ppo_ran),
                "rsl_rl_runner_used": bool(ppo_ran),
                "runner_smoke_complete": bool(runner_complete),
                "runner_policy_artifact_complete": bool(runner_complete),
                "rl_trained": False,
                "rl_trained_success_evidence": bool(eval_row.get("grasp_success_claimed", False)),
                "training_curve_nonempty": bool(runner_complete),
                "checkpoint_path": checkpoint_path if runner_complete else "",
                "checkpoint_written": bool(runner_complete and checkpoint_path),
                "checkpoint_is_policy_artifact_only": bool(runner_complete and checkpoint_path),
                "ppo_iterations_completed": int(ppo_iterations_completed) if runner_complete else 0,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "status": "PPO_POLICY_ARTIFACT_COMPLETE" if runner_complete else "TRAINING_NOT_STARTED",
                "blocker": "" if runner_complete else "training_not_started",
                "next_action": "run_v87_deterministic_no_sticky_eval" if runner_complete else "fix_v87_training_gate",
            }
        )
    return rows


def _write_v87_training_not_started(
    run_path: Path,
    parts: list[str],
    policy_mode: str,
    config_path: str,
    reason: str,
) -> dict[str, Any]:
    rows = [
        {
            "part_name": part,
            "policy_mode": policy_mode,
            "ppo_ran": False,
            "rsl_rl_runner_used": False,
            "runner_smoke_complete": False,
            "runner_policy_artifact_complete": False,
            "rl_trained": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": False,
            "checkpoint_written": False,
            "checkpoint_is_policy_artifact_only": False,
            "ppo_iterations_completed": 0,
            "rollout_step_count": 0,
            "force_contact_rate": 0.0,
            "contact_duration_mean": 0.0,
            "contact_stability_score": 0.0,
            "usable_training_row_count": 0,
            "status": "TRAINING_NOT_STARTED",
            "blocker": reason,
        }
        for part in parts
    ]
    curve_csv = write_csv(run_path / "v87_contact_guided_training_curve.csv", [])
    trace_csv = write_csv(run_path / "v87_rollout_trace.csv", [])
    trace_jsonl = write_jsonl(run_path / "v87_rollout_trace.jsonl", [])
    reset_csv = write_csv(run_path / "v87_near_contact_reset_audit.csv", [])
    reset_jsonl = write_jsonl(run_path / "v87_near_contact_reset_audit.jsonl", [])
    action_prior_csv = write_csv(run_path / "v87_action_prior_audit.csv", [])
    action_prior_json = write_json(run_path / "v87_action_prior_audit.json", [])
    per_part_csv = write_csv(run_path / "v87_per_object_rollout_summary.csv", rows)
    per_part_json = write_json(run_path / "v87_per_object_rollout_summary.json", rows)
    return {
        "status": "TRAINING_NOT_STARTED",
        "ppo_ran": False,
        "rl_trained": False,
        "runner_smoke_complete": False,
        "runner_policy_artifact_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "checkpoint_written": False,
        "checkpoint_is_policy_artifact_only": False,
        "rsl_rl_runner_used": False,
        "surrogate_training_used": False,
        "ppo_iterations_completed": 0,
        "rollout_step_count": 0,
        "usable_training_row_count": 0,
        "v87_training_config_json": config_path,
        "v87_contact_guided_training_curve_csv": str(curve_csv),
        "v87_rollout_trace_csv": str(trace_csv),
        "v87_rollout_trace_jsonl": str(trace_jsonl),
        "v87_near_contact_reset_audit_csv": str(reset_csv),
        "v87_near_contact_reset_audit_jsonl": str(reset_jsonl),
        "v87_action_prior_audit_csv": str(action_prior_csv),
        "v87_action_prior_audit_json": str(action_prior_json),
        "v87_per_object_rollout_summary_csv": str(per_part_csv),
        "v87_per_object_rollout_summary_json": str(per_part_json),
        "failure_reason": reason,
        "rows": rows,
    }


def summarize_v86_rollout(
    parts: list[str],
    trace_rows: list[dict[str, Any]],
    *,
    ppo_ran: bool,
    runner_smoke_complete: bool,
    checkpoint_path: str,
    ppo_iterations_completed: int,
    rollout_step_count: int,
    num_envs_actual: int,
    eval_rows_by_part: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    eval_rows_by_part = eval_rows_by_part or {}
    rows = []
    for part in parts:
        part_trace = [row for row in trace_rows if row.get("part_name") == part]
        force_hits = [
            row
            for row in part_trace
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > 0.05
            and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
        ]
        peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in part_trace]
        displacements = [float(row.get("object_displacement_m") or 0.0) for row in part_trace]
        excessive = [row for row in part_trace if bool(row.get("excessive_force"))]
        support_rows = [row for row in part_trace if bool(row.get("support_gate_ok"))]
        eval_row = eval_rows_by_part.get(part, {})
        rows.append(
            {
                "part_name": part,
                "rollout_env_count": len({int(row.get("env_index") or -1) for row in part_trace if row.get("env_index") not in ("", None)}),
                "rollout_participation_count": len(part_trace),
                "rollout_step_count": int(rollout_step_count),
                "actual_env_count": int(num_envs_actual),
                "force_contact_rate": len(force_hits) / max(1, len(part_trace)),
                "support_gate_rate": len(support_rows) / max(1, len(part_trace)),
                "force_contact_peak_n": max([0.0, *peaks]),
                "force_contact_mean_n": sum(peaks) / max(1, len(peaks)),
                "excessive_force_threshold_n": 150.0,
                "excessive_force_rate": len(excessive) / max(1, len(part_trace)),
                "object_displacement_max_m": max([0.0, *displacements]),
                "object_write_by_policy_detected": any(bool(row.get("object_write_by_policy_detected")) for row in part_trace),
                "sticky_action_available_to_policy": any(bool(row.get("sticky_action_available_to_policy")) for row in part_trace),
                "support_eval_ok": bool(eval_row.get("support_eval_ok", False)),
                "hold_eval_attempted": bool(eval_row.get("hold_eval_attempted", False)),
                "hold_eval_ok": bool(eval_row.get("hold_eval_ok", False)),
                "lift_eval_attempted": bool(eval_row.get("lift_eval_attempted", False)),
                "lift_eval_ok": bool(eval_row.get("lift_eval_ok", False)),
                "grasp_success_claimed": bool(eval_row.get("grasp_success_claimed", False)),
                "ppo_ran": bool(ppo_ran),
                "rsl_rl_runner_used": bool(ppo_ran),
                "runner_smoke_complete": bool(runner_smoke_complete),
                "rl_trained": False,
                "rl_trained_success_evidence": bool(eval_row.get("grasp_success_claimed", False)),
                "training_curve_nonempty": bool(runner_smoke_complete),
                "checkpoint_path": checkpoint_path if runner_smoke_complete else "",
                "checkpoint_written": bool(runner_smoke_complete and checkpoint_path),
                "checkpoint_is_runner_artifact_only": bool(runner_smoke_complete and checkpoint_path),
                "ppo_iterations_completed": int(ppo_iterations_completed) if runner_smoke_complete else 0,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "status": "PPO_SMOKE_COMPLETE" if runner_smoke_complete else "TRAINING_NOT_STARTED",
                "blocker": "" if runner_smoke_complete else "training_not_started",
                "next_action": "inspect_deterministic_eval_support_hold_lift" if runner_smoke_complete else "fix_v86_training_gate",
            }
        )
    return rows


def _write_v86_training_not_started(
    run_path: Path,
    parts: list[str],
    policy_mode: str,
    config_path: str,
    reason: str,
) -> dict[str, Any]:
    rows = [
        {
            "part_name": part,
            "policy_mode": policy_mode,
            "ppo_ran": False,
            "rsl_rl_runner_used": False,
            "runner_smoke_complete": False,
            "rl_trained": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": False,
            "checkpoint_written": False,
            "checkpoint_is_runner_artifact_only": False,
            "ppo_iterations_completed": 0,
            "rollout_step_count": 0,
            "usable_training_row_count": 0,
            "status": "TRAINING_NOT_STARTED",
            "blocker": reason,
        }
        for part in parts
    ]
    curve_csv = write_csv(run_path / "v86_training_curve.csv", [])
    trace_csv = write_csv(run_path / "v86_rollout_trace.csv", [])
    trace_jsonl = write_jsonl(run_path / "v86_rollout_trace.jsonl", [])
    reset_csv = write_csv(run_path / "v86_near_contact_reset_audit.csv", [])
    reset_jsonl = write_jsonl(run_path / "v86_near_contact_reset_audit.jsonl", [])
    per_part_csv = write_csv(run_path / "v86_per_object_rollout_summary.csv", rows)
    per_part_json = write_json(run_path / "v86_per_object_rollout_summary.json", rows)
    return {
        "status": "TRAINING_NOT_STARTED",
        "ppo_ran": False,
        "rl_trained": False,
        "runner_smoke_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "checkpoint_written": False,
        "checkpoint_is_runner_artifact_only": False,
        "rsl_rl_runner_used": False,
        "surrogate_training_used": False,
        "ppo_iterations_completed": 0,
        "rollout_step_count": 0,
        "usable_training_row_count": 0,
        "v86_training_config_json": config_path,
        "v86_training_curve_csv": str(curve_csv),
        "v86_rollout_trace_csv": str(trace_csv),
        "v86_rollout_trace_jsonl": str(trace_jsonl),
        "v86_near_contact_reset_audit_csv": str(reset_csv),
        "v86_near_contact_reset_audit_jsonl": str(reset_jsonl),
        "v86_per_object_rollout_summary_csv": str(per_part_csv),
        "v86_per_object_rollout_summary_json": str(per_part_json),
        "failure_reason": reason,
        "rows": rows,
    }


def _write_surrogate_unit_test_training(
    run_path: Path,
    *,
    config_path: str,
    policy_mode: str,
    parts: list[str],
    num_envs: int,
    max_steps: int,
    physics_profile: str,
) -> dict[str, Any]:
    smoke = run_env_smoke(
        run_path,
        num_envs=min(max(1, int(num_envs)), 16),
        steps=min(max(1, int(max_steps)), 16),
        parts=parts or V80_PARTS,
        physics_profile=physics_profile,
        allow_diagnostic_dynamics=True,
    )
    curve_rows = [
        {
            "iteration": index,
            "policy_mode": policy_mode,
            "mean_reward": -1.0 + 0.1 * index,
            "support_gate_rate": 0.0,
            "success_no_sticky_rate": 0.0,
            "success_with_sticky_after_support_rate": 0.0,
            "surrogate_training_used": True,
            "final_training_export_allowed": False,
        }
        for index in range(4)
    ]
    curve_csv = write_csv(run_path / "unified_rl_training_curve.csv", curve_rows)
    rollout_csv = write_csv(run_path / "unified_rl_rollout_trace.csv", smoke.get("rows", []))
    rollout_jsonl = write_jsonl(run_path / "unified_rl_rollout_trace.jsonl", smoke.get("rows", []))
    checkpoint = run_path / "surrogate_checkpoint.txt"
    checkpoint.write_text("surrogate unit-test-only dynamics; not a real PPO checkpoint\n", encoding="utf-8")
    per_part = [
        {
            "part_name": part,
            "policy_mode": policy_mode,
            "rl_trained": False,
            "runner_smoke_complete": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": True,
            "surrogate_curve_nonempty": True,
            "rsl_rl_runner_used": False,
            "surrogate_training_used": True,
            "status": "surrogate_unit_test_only",
            "blocker": "physical_isaac_backend_not_configured",
        }
        for part in parts
    ]
    per_part_csv = write_csv(run_path / "per_part_training_summary.csv", per_part)
    per_part_json = write_json(run_path / "per_part_training_summary.json", per_part)
    multi_summary = {
        "policy_mode": policy_mode,
        "rl_trained": False,
        "runner_smoke_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": True,
        "surrogate_curve_nonempty": True,
        "checkpoint_path": str(checkpoint),
        "surrogate_checkpoint_path": str(checkpoint),
        "rsl_rl_runner_used": False,
        "surrogate_training_used": True,
        "usable_training_row_count": 0,
        "blocker": "diagnostic_dynamics_is_not_final_contact_evidence",
    }
    multi_json = write_json(run_path / "multi_object_training_summary.json", multi_summary)
    return {
        "status": "surrogate_unit_test_only",
        "rl_trained": False,
        "runner_smoke_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": True,
        "checkpoint_path": str(checkpoint),
        "surrogate_checkpoint_path": str(checkpoint),
        "unified_rl_config_json": str(config_path),
        "unified_rl_training_curve_csv": str(curve_csv),
        "unified_rl_rollout_trace_csv": str(rollout_csv),
        "unified_rl_rollout_trace_jsonl": str(rollout_jsonl),
        "per_part_training_summary_csv": str(per_part_csv),
        "per_part_training_summary_json": str(per_part_json),
        "multi_object_training_summary_json": str(multi_json),
        "rsl_rl_runner_used": False,
        "surrogate_training_used": True,
        "surrogate_curve_nonempty": True,
        "rows": per_part,
    }


def _write_training_not_started(
    run_dir: Path,
    *,
    config_path: str,
    policy_mode: str,
    parts: list[str],
    reason: str,
) -> dict[str, Any]:
    rows = [
        {
            "part_name": part,
            "policy_mode": policy_mode,
            "rl_trained": False,
            "runner_smoke_complete": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": False,
            "rsl_rl_runner_used": False,
            "surrogate_training_used": False,
            "ppo_iterations_completed": 0,
            "rollout_step_count": 0,
            "status": "TRAINING_NOT_STARTED",
            "failure_category": "TRAINING_NOT_STARTED" if "RSL" in reason or "RL_FRAMEWORK" in reason else "ENV_NOT_VECTORIZEABLE",
            "blocker": reason,
            "usable_training_row_count": 0,
        }
        for part in parts
    ]
    curve_csv = write_csv(run_dir / "unified_rl_training_curve.csv", [])
    write_json(
        run_dir / "multi_object_training_summary.json",
        {
            "rl_trained": False,
            "runner_smoke_complete": False,
            "rl_trained_success_evidence": False,
            "training_curve_nonempty": False,
            "rsl_rl_runner_used": False,
            "surrogate_training_used": False,
            "ppo_iterations_completed": 0,
            "rollout_step_count": 0,
            "blocker": reason,
        },
    )
    per_part_csv = write_csv(run_dir / "per_part_training_summary.csv", rows)
    per_part_json = write_json(run_dir / "per_part_training_summary.json", rows)
    return {
        "status": "TRAINING_NOT_STARTED",
        "rl_trained": False,
        "runner_smoke_complete": False,
        "rl_trained_success_evidence": False,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "rsl_rl_runner_used": False,
        "surrogate_training_used": False,
        "ppo_iterations_completed": 0,
        "rollout_step_count": 0,
        "unified_rl_config_json": config_path,
        "unified_rl_training_curve_csv": str(curve_csv),
        "per_part_training_summary_csv": str(per_part_csv),
        "per_part_training_summary_json": str(per_part_json),
        "failure_reason": reason,
        "rows": rows,
    }
