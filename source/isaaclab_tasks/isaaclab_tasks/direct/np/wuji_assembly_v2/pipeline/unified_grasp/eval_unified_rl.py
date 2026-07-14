"""Evaluation helpers for v80 unified grasp policies."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from .train_unified_rl import RslRlUnifiedGraspVecEnv, _rsl_rl_train_cfg
from .unified_grasp_env import UnifiedGraspEnv, UnifiedGraspEnvCfg
from .v80_reports import V80_PARTS, write_csv, write_json, write_jsonl

try:  # pragma: no cover - runtime-specific
    import torch
except Exception:  # pragma: no cover
    torch = None


def evaluate_unified_policy(
    run_dir: str | Path,
    *,
    checkpoint_path: str = "",
    parts: list[str] | None = None,
    sticky_after_support: bool = False,
    eval_episode_count: int = 0,
    physical_backend: Any | None = None,
    physics_profile: str = "canonical",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    surrogate_checkpoint = bool(checkpoint_path and Path(checkpoint_path).name == "surrogate_checkpoint.txt")
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
    physical_rows: list[dict[str, Any]] = []
    eval_error = ""
    if checkpoint_exists and not surrogate_checkpoint and backend_ready:
        try:
            from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

            cfg_path = Path(checkpoint_path).parent / "rsl_rl_runner_config.json"
            if cfg_path.exists():
                train_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                train_cfg = _rsl_rl_train_cfg(num_steps_per_env=16, max_iterations=1, policy_mode="shared_multi_object_policy")
            env_cfg = UnifiedGraspEnvCfg(
                num_envs=int(getattr(physical_backend, "num_envs", 1) or 1),
                parts=ready_parts,
                physics_profile=physics_profile,
                max_episode_length=max(1, int(eval_episode_count or 1)),
                device=str(getattr(physical_backend, "device", "cpu")),
                sticky_after_support_validation=sticky_after_support,
            )
            env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
            vec_env = RslRlUnifiedGraspVecEnv(env)
            runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=vec_env.device)
            runner.load(checkpoint_path)
            policy = runner.get_inference_policy(device=vec_env.device)
            vec_env.reset()
            for episode in range(max(1, int(eval_episode_count))):
                obs = vec_env.get_observations()
                with torch.no_grad() if torch is not None else _nullcontext():
                    actions = policy(obs) if callable(policy) else None
                _next_obs, _reward, _done, extras = vec_env.step(actions)
                for metric in physical_backend.get_metrics():
                    physical_rows.append({"eval_episode": episode, **metric, "sticky_after_support": bool(sticky_after_support)})
        except Exception as exc:
            eval_error = f"RSL_RL_EVAL_FAILED:{type(exc).__name__}:{exc}"
    for part in requested_parts:
        if not checkpoint_exists:
            failure_category = "TRAINING_NOT_STARTED"
            blocker = "no_checkpoint_for_deterministic_eval"
        elif surrogate_checkpoint:
            failure_category = "TRAINING_NOT_STARTED"
            blocker = "diagnostic_checkpoint_rejected_surrogate_unit_test_only"
        elif not backend_ready or part not in ready_parts:
            failure_category = "ENV_NOT_VECTORIZEABLE"
            blocker = str(
                getattr(physical_backend, "part_blocker", lambda _part: "")(part)
                or getattr(physical_backend, "blocker", "")
                or "physical_backend_not_ready_for_eval"
            )
        elif eval_error:
            failure_category = "ENV_NOT_VECTORIZEABLE"
            blocker = eval_error
        else:
            failure_category = "POLICY_NO_CONTACT"
            blocker = "deterministic_policy_eval_no_support_gate_pass"
        part_rows = [row for row in physical_rows if row.get("part_name") == part]
        episode_count = int(
            eval_episode_count
            if checkpoint_exists and backend_ready and part in ready_parts and not eval_error and not surrogate_checkpoint
            else 0
        )
        support_rate = (
            sum(1 for row_item in part_rows if row_item.get("support_gate_ok")) / max(1, len(part_rows))
            if part_rows
            else 0.0
        )
        force_mean = (
            sum(float(row_item.get("force_contact_count") or 0.0) for row_item in part_rows) / max(1, len(part_rows))
            if part_rows
            else 0.0
        )
        distance_mean = (
            sum(float(row_item.get("distance_contact_count") or 0.0) for row_item in part_rows) / max(1, len(part_rows))
            if part_rows
            else 0.0
        )
        motion_rate = (
            sum(1 for row_item in part_rows if float(row_item.get("object_motion_before_contact_m") or 0.0) > 0.002) / max(1, len(part_rows))
            if part_rows
            else 0.0
        )
        penetration_rate = (
            sum(1 for row_item in part_rows if float(row_item.get("penetration_depth_m") or 0.0) > 0.0) / max(1, len(part_rows))
            if part_rows
            else 0.0
        )
        table_rate = (
            sum(1 for row_item in part_rows if bool(row_item.get("table_collision"))) / max(1, len(part_rows))
            if part_rows
            else 0.0
        )
        success_no_sticky = support_rate if not sticky_after_support else 0.0
        success_sticky = support_rate if sticky_after_support else 0.0
        row = {
            "part_name": part,
            "eval_mode": "deterministic_eval_sticky_after_support" if sticky_after_support else "deterministic_eval_no_sticky",
            "eval_episode_count": episode_count,
            "success_no_sticky_rate": success_no_sticky,
            "success_with_sticky_after_support_rate": success_sticky,
            "support_gate_rate": support_rate,
            "average_effective_contact_count_force": force_mean,
            "average_effective_contact_count_distance": distance_mean,
            "object_motion_before_contact_rate": motion_rate,
            "penetration_rate": penetration_rate,
            "table_collision_rate": table_rate,
            "lift_success_rate": 0.0,
            "hold_success_rate": 0.0,
            "video_clean_sample_count": 0,
            "usable_training_row_count": 0,
            "sticky_used_as_stabilizer": bool(sticky_after_support and support_rate > 0.0),
            "support_gate_ok": bool(support_rate > 0.0),
            "failure_category": failure_category,
            "blocker": blocker,
        }
        rows.append(row)
        failure_rows.append({"part_name": part, "failure_category": failure_category, "blocker": blocker})
    run_path = Path(run_dir)
    error_trace_path = run_path / "v86_deterministic_eval_error_traceback.txt"
    if error_trace_path.exists():
        error_trace_path.unlink()
    summary_csv = write_csv(run_path / "unified_rl_eval_summary.csv", rows)
    summary_json = write_json(run_path / "unified_rl_eval_summary.json", rows)
    failure_csv = write_csv(run_path / "unified_rl_failure_taxonomy.csv", failure_rows)
    failure_json = write_json(run_path / "unified_rl_failure_taxonomy.json", failure_rows)
    rollout_csv = write_csv(run_path / "unified_rl_eval_rollout_trace.csv", physical_rows)
    return {
        "rows": rows,
        "failure_rows": failure_rows,
        "unified_rl_eval_summary_csv": str(summary_csv),
        "unified_rl_eval_summary_json": str(summary_json),
        "unified_rl_failure_taxonomy_csv": str(failure_csv),
        "unified_rl_failure_taxonomy_json": str(failure_json),
        "unified_rl_eval_rollout_trace_csv": str(rollout_csv),
    }


def evaluate_v86_guarded_residual_policy(
    run_dir: str | Path,
    *,
    checkpoint_path: str = "",
    parts: list[str] | None = None,
    eval_step_count: int = 32,
    physical_backend: Any | None = None,
    physics_profile: str = "canonical",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    eval_error = ""
    if checkpoint_exists and backend_ready:
        try:
            from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

            if callable(getattr(physical_backend, "clear_v86_rollout_trace", None)):
                physical_backend.clear_v86_rollout_trace()
            cfg_path = Path(checkpoint_path).parent / "v86_rsl_rl_runner_config.json"
            if cfg_path.exists():
                train_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                train_cfg = _rsl_rl_train_cfg(num_steps_per_env=16, max_iterations=1, policy_mode="shared_multi_object_policy")
            env_cfg = UnifiedGraspEnvCfg(
                num_envs=int(getattr(physical_backend, "num_envs", 1) or 1),
                parts=requested_parts,
                physics_profile=physics_profile,
                max_episode_length=max(1, int(eval_step_count) + 24),
                device=str(getattr(physical_backend, "device", "cpu")),
                sticky_after_support_validation=False,
                v86_guarded_residual_mode=True,
                use_v85_near_contact_resets=True,
            )
            env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
            vec_env = RslRlUnifiedGraspVecEnv(env)
            runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=vec_env.device)
            runner.load(checkpoint_path)
            policy = runner.get_inference_policy(device=vec_env.device)
            vec_env.reset()
            for step_index in range(max(1, int(eval_step_count))):
                obs = vec_env.get_observations()
                with torch.inference_mode() if torch is not None else _nullcontext():
                    actions = policy(obs) if callable(policy) else None
                if hasattr(actions, "detach"):
                    actions = actions.detach().clone()
                start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
                vec_env.step(actions)
                _annotate_v86_eval_trace(physical_backend, start, "deterministic_policy", step_index)
            hold_steps = 12
            lift_steps = 12
            for step_index in range(hold_steps):
                start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
                env.step(_v86_eval_actions(env.num_envs, "hold"))
                _annotate_v86_eval_trace(physical_backend, start, "hold", step_index)
            for step_index in range(lift_steps):
                start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
                env.step(_v86_eval_actions(env.num_envs, "lift"))
                _annotate_v86_eval_trace(physical_backend, start, "lift", step_index)
            trace_rows = list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        except Exception as exc:
            eval_error = f"V86_DETERMINISTIC_EVAL_FAILED:{type(exc).__name__}:{exc}"
            error_trace_path.write_text(traceback.format_exc(), encoding="utf-8")

    threshold = 0.05
    for part in requested_parts:
        part_rows = [row for row in trace_rows if row.get("part_name") == part]
        policy_rows = [row for row in part_rows if row.get("eval_phase") == "deterministic_policy"]
        hold_rows = [row for row in part_rows if row.get("eval_phase") == "hold"]
        lift_rows = [row for row in part_rows if row.get("eval_phase") == "lift"]
        force_rows = [
            row
            for row in part_rows
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
        ]
        peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in part_rows]
        displacements = [float(row.get("object_displacement_m") or 0.0) for row in part_rows]
        support_ok = any(bool(row.get("support_gate_ok")) for row in policy_rows + hold_rows + lift_rows)
        hold_ok = sum(
            1
            for row in hold_rows
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            and not bool(row.get("table_collision"))
        ) >= 4
        lift_ok = any(
            float(row.get("object_delta_z_m") or 0.0) > 0.005
            and float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            for row in lift_rows
        )
        forbidden = bool(
            any(bool(row.get("object_write_by_policy_detected")) for row in part_rows)
            or any(bool(row.get("sticky_action_available_to_policy")) for row in part_rows)
        )
        success = bool(support_ok and hold_ok and lift_ok and not forbidden)
        if not checkpoint_exists:
            status = "EVAL_NOT_STARTED"
            blocker = "checkpoint_missing"
            failure_category = "TRAINING_NOT_STARTED"
        elif not backend_ready:
            status = "EVAL_NOT_STARTED"
            blocker = str(getattr(physical_backend, "blocker", "") or "single_context_backend_not_ready")
            failure_category = "ENV_NOT_VECTORIZEABLE"
        elif eval_error:
            status = "EVAL_FAILED"
            blocker = eval_error
            failure_category = "ENV_NOT_VECTORIZEABLE"
        elif success:
            status = "GRASP_CONTACT_SUPPORT_HOLD_LIFT_PASS_NO_STICKY"
            blocker = ""
            failure_category = ""
        elif not support_ok:
            status = "POLICY_NO_SUPPORT"
            blocker = "deterministic_eval_no_support_gate_pass"
            failure_category = "CONTACT_BUT_NO_SUPPORT" if force_rows else "POLICY_NO_CONTACT"
        elif not hold_ok:
            status = "HOLD_FAILED"
            blocker = "deterministic_eval_hold_not_stable"
            failure_category = "SLIP_AFTER_CONTACT"
        else:
            status = "LIFT_FAILED"
            blocker = "deterministic_eval_lift_not_observed"
            failure_category = "SLIP_AFTER_CONTACT"
        row = {
            "part_name": part,
            "eval_mode": "v86_deterministic_no_sticky",
            "eval_step_count": max(1, int(eval_step_count)) if checkpoint_exists and backend_ready and not eval_error else 0,
            "force_contact_rate": len(force_rows) / max(1, len(part_rows)),
            "support_eval_ok": bool(support_ok),
            "hold_eval_attempted": bool(checkpoint_exists and backend_ready and not eval_error),
            "hold_eval_ok": bool(hold_ok),
            "lift_eval_attempted": bool(checkpoint_exists and backend_ready and not eval_error),
            "lift_eval_ok": bool(lift_ok),
            "force_contact_peak_n": max([0.0, *peaks]),
            "force_contact_mean_n": sum(peaks) / max(1, len(peaks)),
            "excessive_force_threshold_n": 150.0,
            "excessive_force_rate": sum(1 for row_item in part_rows if bool(row_item.get("excessive_force"))) / max(1, len(part_rows)),
            "object_displacement_max_m": max([0.0, *displacements]),
            "object_write_by_policy_detected": any(bool(row_item.get("object_write_by_policy_detected")) for row_item in part_rows),
            "sticky_action_available_to_policy": any(bool(row_item.get("sticky_action_available_to_policy")) for row_item in part_rows),
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
            "grasp_success_claimed": bool(success),
            "rl_trained_success_evidence": bool(success),
            "status": status,
            "failure_category": failure_category,
            "blocker": blocker,
        }
        rows.append(row)
        failure_rows.append({"part_name": part, "failure_category": failure_category, "blocker": blocker})
    summary_csv = write_csv(run_path / "v86_deterministic_eval_summary.csv", rows)
    summary_json = write_json(run_path / "v86_deterministic_eval_summary.json", rows)
    trace_jsonl = write_jsonl(run_path / "v86_deterministic_eval_trace.jsonl", trace_rows)
    trace_csv = write_csv(run_path / "v86_deterministic_eval_trace.csv", trace_rows)
    failure_csv = write_csv(run_path / "v86_failure_taxonomy.csv", failure_rows)
    failure_json = write_json(run_path / "v86_failure_taxonomy.json", failure_rows)
    return {
        "rows": rows,
        "failure_rows": failure_rows,
        "v86_deterministic_eval_summary_csv": str(summary_csv),
        "v86_deterministic_eval_summary_json": str(summary_json),
        "v86_deterministic_eval_trace_csv": str(trace_csv),
        "v86_deterministic_eval_trace_jsonl": str(trace_jsonl),
        "v86_failure_taxonomy_csv": str(failure_csv),
        "v86_failure_taxonomy_json": str(failure_json),
    }


def evaluate_v87_contact_guided_policy(
    run_dir: str | Path,
    *,
    checkpoint_path: str = "",
    parts: list[str] | None = None,
    eval_step_count: int = 96,
    physical_backend: Any | None = None,
    physics_profile: str = "canonical",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    nominal_trace: list[dict[str, Any]] = []
    policy_trace: list[dict[str, Any]] = []
    nominal_error = ""
    policy_error = ""

    if backend_ready:
        try:
            nominal_trace = _run_v87_eval_rollout(
                requested_parts=requested_parts,
                physical_backend=physical_backend,
                physics_profile=physics_profile,
                eval_step_count=eval_step_count,
                checkpoint_path="",
                eval_mode="nominal_only_baseline",
            )
        except Exception as exc:
            nominal_error = f"V87_NOMINAL_ONLY_EVAL_FAILED:{type(exc).__name__}:{exc}"
            (run_path / "v87_nominal_only_eval_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")

    if checkpoint_exists and backend_ready:
        try:
            policy_trace = _run_v87_eval_rollout(
                requested_parts=requested_parts,
                physical_backend=physical_backend,
                physics_profile=physics_profile,
                eval_step_count=eval_step_count,
                checkpoint_path=checkpoint_path,
                eval_mode="policy_residual_eval",
            )
        except Exception as exc:
            policy_error = f"V87_POLICY_RESIDUAL_EVAL_FAILED:{type(exc).__name__}:{exc}"
            (run_path / "v87_deterministic_eval_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")

    nominal_rows, nominal_failures = _summarize_v87_eval(
        requested_parts,
        nominal_trace,
        eval_mode="nominal_only_baseline",
        checkpoint_exists=True,
        backend_ready=backend_ready,
        eval_error=nominal_error,
        eval_step_count=eval_step_count,
    )
    policy_rows, policy_failures = _summarize_v87_eval(
        requested_parts,
        policy_trace,
        eval_mode="policy_residual_eval",
        checkpoint_exists=checkpoint_exists,
        backend_ready=backend_ready,
        eval_error=policy_error,
        eval_step_count=eval_step_count,
    )
    nominal_csv = write_csv(run_path / "v87_nominal_only_eval_summary.csv", nominal_rows)
    nominal_json = write_json(run_path / "v87_nominal_only_eval_summary.json", nominal_rows)
    nominal_trace_csv = write_csv(run_path / "v87_nominal_only_eval_trace.csv", nominal_trace)
    nominal_trace_jsonl = write_jsonl(run_path / "v87_nominal_only_eval_trace.jsonl", nominal_trace)
    summary_csv = write_csv(run_path / "v87_deterministic_no_sticky_eval_summary.csv", policy_rows)
    summary_json = write_json(run_path / "v87_deterministic_no_sticky_eval_summary.json", policy_rows)
    trace_csv = write_csv(run_path / "v87_deterministic_no_sticky_eval_trace.csv", policy_trace)
    trace_jsonl = write_jsonl(run_path / "v87_deterministic_no_sticky_eval_trace.jsonl", policy_trace)
    failure_rows = policy_failures if checkpoint_exists else nominal_failures
    failure_csv = write_csv(run_path / "v87_failure_taxonomy.csv", failure_rows)
    failure_json = write_json(run_path / "v87_failure_taxonomy.json", failure_rows)
    return {
        "rows": policy_rows,
        "nominal_rows": nominal_rows,
        "failure_rows": failure_rows,
        "v87_nominal_only_eval_summary_csv": str(nominal_csv),
        "v87_nominal_only_eval_summary_json": str(nominal_json),
        "v87_nominal_only_eval_trace_csv": str(nominal_trace_csv),
        "v87_nominal_only_eval_trace_jsonl": str(nominal_trace_jsonl),
        "v87_deterministic_no_sticky_eval_summary_csv": str(summary_csv),
        "v87_deterministic_no_sticky_eval_summary_json": str(summary_json),
        "v87_deterministic_no_sticky_eval_trace_csv": str(trace_csv),
        "v87_deterministic_no_sticky_eval_trace_jsonl": str(trace_jsonl),
        "v87_failure_taxonomy_csv": str(failure_csv),
        "v87_failure_taxonomy_json": str(failure_json),
    }


def evaluate_v88_stabilized_policy(
    run_dir: str | Path,
    *,
    branch_name: str,
    checkpoint_path: str = "",
    parts: list[str] | None = None,
    eval_step_count: int = 128,
    physical_backend: Any | None = None,
    physics_profile: str = "canonical",
    enhanced_observation: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    nominal_trace: list[dict[str, Any]] = []
    policy_trace: list[dict[str, Any]] = []
    nominal_error = ""
    policy_error = ""

    if backend_ready:
        try:
            nominal_trace = _run_v88_eval_rollout(
                requested_parts=requested_parts,
                physical_backend=physical_backend,
                physics_profile=physics_profile,
                eval_step_count=eval_step_count,
                checkpoint_path="",
                eval_mode="nominal_only_baseline",
                enhanced_observation=enhanced_observation,
            )
        except Exception as exc:
            nominal_error = f"V88_NOMINAL_ONLY_EVAL_FAILED:{type(exc).__name__}:{exc}"
            (run_path / "v88_nominal_only_eval_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")

    if checkpoint_exists and backend_ready:
        try:
            policy_trace = _run_v88_eval_rollout(
                requested_parts=requested_parts,
                physical_backend=physical_backend,
                physics_profile=physics_profile,
                eval_step_count=eval_step_count,
                checkpoint_path=checkpoint_path,
                eval_mode="policy_residual_eval",
                enhanced_observation=enhanced_observation,
            )
        except Exception as exc:
            policy_error = f"V88_POLICY_RESIDUAL_EVAL_FAILED:{type(exc).__name__}:{exc}"
            (run_path / "v88_deterministic_eval_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")

    nominal_rows, nominal_failures = _summarize_v88_eval(
        requested_parts,
        nominal_trace,
        branch_name=branch_name,
        physics_profile=physics_profile,
        eval_mode="nominal_only_baseline",
        checkpoint_exists=True,
        backend_ready=backend_ready,
        eval_error=nominal_error,
        eval_step_count=eval_step_count,
    )
    policy_rows, policy_failures = _summarize_v88_eval(
        requested_parts,
        policy_trace,
        branch_name=branch_name,
        physics_profile=physics_profile,
        eval_mode="policy_residual_eval",
        checkpoint_exists=checkpoint_exists,
        backend_ready=backend_ready,
        eval_error=policy_error,
        eval_step_count=eval_step_count,
    )
    nominal_csv = write_csv(run_path / "v88_nominal_only_eval_summary.csv", nominal_rows)
    nominal_json = write_json(run_path / "v88_nominal_only_eval_summary.json", nominal_rows)
    nominal_trace_csv = write_csv(run_path / "v88_nominal_only_eval_trace.csv", nominal_trace)
    nominal_trace_jsonl = write_jsonl(run_path / "v88_nominal_only_eval_trace.jsonl", nominal_trace)
    summary_csv = write_csv(run_path / "v88_deterministic_eval_no_sticky.csv", policy_rows)
    summary_json = write_json(run_path / "v88_deterministic_eval_no_sticky.json", policy_rows)
    trace_csv = write_csv(run_path / "v88_deterministic_eval_no_sticky_trace.csv", policy_trace)
    trace_jsonl = write_jsonl(run_path / "v88_deterministic_eval_no_sticky_trace.jsonl", policy_trace)
    failure_rows = policy_failures if checkpoint_exists else nominal_failures
    failure_csv = write_csv(run_path / "v88_failure_taxonomy.csv", failure_rows)
    failure_json = write_json(run_path / "v88_failure_taxonomy.json", failure_rows)
    return {
        "rows": policy_rows,
        "nominal_rows": nominal_rows,
        "failure_rows": failure_rows,
        "v88_nominal_only_eval_summary_csv": str(nominal_csv),
        "v88_nominal_only_eval_summary_json": str(nominal_json),
        "v88_nominal_only_eval_trace_csv": str(nominal_trace_csv),
        "v88_nominal_only_eval_trace_jsonl": str(nominal_trace_jsonl),
        "v88_deterministic_eval_no_sticky_csv": str(summary_csv),
        "v88_deterministic_eval_no_sticky_json": str(summary_json),
        "v88_deterministic_eval_no_sticky_trace_csv": str(trace_csv),
        "v88_deterministic_eval_no_sticky_trace_jsonl": str(trace_jsonl),
        "v88_failure_taxonomy_csv": str(failure_csv),
        "v88_failure_taxonomy_json": str(failure_json),
    }


def evaluate_v89_hybrid_policy(
    run_dir: str | Path,
    *,
    branch_name: str,
    checkpoint_path: str = "",
    parts: list[str] | None = None,
    eval_step_count: int = 128,
    physical_backend: Any | None = None,
    physics_profile: str = "canonical",
    use_candidate_prior: bool = False,
    selected_candidates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    requested_parts = parts or list(V80_PARTS)
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    backend_ready = bool(
        physical_backend is not None
        and getattr(physical_backend, "physical_backend_ready", False)
        and getattr(physical_backend, "contact_sensor_configured", False)
        and getattr(physical_backend, "single_simulation_context", False)
    )
    nominal_trace: list[dict[str, Any]] = []
    policy_trace: list[dict[str, Any]] = []
    nominal_error = ""
    policy_error = ""
    if backend_ready:
        try:
            nominal_trace = _run_v89_eval_rollout(
                requested_parts=requested_parts,
                physical_backend=physical_backend,
                physics_profile=physics_profile,
                eval_step_count=eval_step_count,
                checkpoint_path="",
                eval_mode="nominal_candidate_baseline" if use_candidate_prior else "v88_best_nominal_baseline",
                use_candidate_prior=use_candidate_prior,
                selected_candidates=selected_candidates or {},
            )
        except Exception as exc:
            nominal_error = f"V89_NOMINAL_EVAL_FAILED:{type(exc).__name__}:{exc}"
            (run_path / "v89_nominal_eval_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    if checkpoint_exists and backend_ready:
        try:
            policy_trace = _run_v89_eval_rollout(
                requested_parts=requested_parts,
                physical_backend=physical_backend,
                physics_profile=physics_profile,
                eval_step_count=eval_step_count,
                checkpoint_path=checkpoint_path,
                eval_mode="policy_residual_eval",
                use_candidate_prior=use_candidate_prior,
                selected_candidates=selected_candidates or {},
            )
        except Exception as exc:
            policy_error = f"V89_POLICY_RESIDUAL_EVAL_FAILED:{type(exc).__name__}:{exc}"
            (run_path / "v89_deterministic_eval_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    nominal_rows, nominal_failures = _summarize_v88_eval(
        requested_parts,
        nominal_trace,
        branch_name=branch_name,
        physics_profile=physics_profile,
        eval_mode="nominal_candidate_baseline" if use_candidate_prior else "v88_best_nominal_baseline",
        checkpoint_exists=True,
        backend_ready=backend_ready,
        eval_error=nominal_error,
        eval_step_count=eval_step_count,
    )
    policy_rows, policy_failures = _summarize_v88_eval(
        requested_parts,
        policy_trace,
        branch_name=branch_name,
        physics_profile=physics_profile,
        eval_mode="policy_residual_eval",
        checkpoint_exists=checkpoint_exists,
        backend_ready=backend_ready,
        eval_error=policy_error,
        eval_step_count=eval_step_count,
    )
    _annotate_v89_eval_summaries(nominal_rows, nominal_trace, branch_name)
    _annotate_v89_eval_summaries(policy_rows, policy_trace, branch_name)
    failure_rows = policy_failures if checkpoint_exists else nominal_failures
    for row in failure_rows:
        row["branch_name"] = branch_name
    nominal_csv = write_csv(run_path / "v89_nominal_eval_summary.csv", nominal_rows)
    nominal_json = write_json(run_path / "v89_nominal_eval_summary.json", nominal_rows)
    nominal_trace_csv = write_csv(run_path / "v89_nominal_eval_trace.csv", nominal_trace)
    nominal_trace_jsonl = write_jsonl(run_path / "v89_nominal_eval_trace.jsonl", nominal_trace)
    summary_csv = write_csv(run_path / "v89_deterministic_eval_no_sticky.csv", policy_rows)
    summary_json = write_json(run_path / "v89_deterministic_eval_no_sticky.json", policy_rows)
    trace_csv = write_csv(run_path / "v89_deterministic_eval_no_sticky_trace.csv", policy_trace)
    trace_jsonl = write_jsonl(run_path / "v89_deterministic_eval_no_sticky_trace.jsonl", policy_trace)
    failure_csv = write_csv(run_path / "v89_failure_taxonomy.csv", failure_rows)
    failure_json = write_json(run_path / "v89_failure_taxonomy.json", failure_rows)
    return {
        "rows": policy_rows,
        "nominal_rows": nominal_rows,
        "failure_rows": failure_rows,
        "v89_nominal_eval_summary_csv": str(nominal_csv),
        "v89_nominal_eval_summary_json": str(nominal_json),
        "v89_nominal_eval_trace_csv": str(nominal_trace_csv),
        "v89_nominal_eval_trace_jsonl": str(nominal_trace_jsonl),
        "v89_deterministic_eval_no_sticky_csv": str(summary_csv),
        "v89_deterministic_eval_no_sticky_json": str(summary_json),
        "v89_deterministic_eval_no_sticky_trace_csv": str(trace_csv),
        "v89_deterministic_eval_no_sticky_trace_jsonl": str(trace_jsonl),
        "v89_failure_taxonomy_csv": str(failure_csv),
        "v89_failure_taxonomy_json": str(failure_json),
    }


def _run_v89_eval_rollout(
    *,
    requested_parts: list[str],
    physical_backend: Any,
    physics_profile: str,
    eval_step_count: int,
    checkpoint_path: str,
    eval_mode: str,
    use_candidate_prior: bool,
    selected_candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if use_candidate_prior and callable(getattr(physical_backend, "configure_v89_candidates", None)):
        physical_backend.configure_v89_candidates(selected_candidates)
        if callable(getattr(physical_backend, "configure_v86_staging", None)):
            physical_backend.configure_v86_staging(selected_candidates)
    if callable(getattr(physical_backend, "clear_v89_rollout_trace", None)):
        physical_backend.clear_v89_rollout_trace()
    elif callable(getattr(physical_backend, "clear_v88_rollout_trace", None)):
        physical_backend.clear_v88_rollout_trace()
    cfg_path = Path(checkpoint_path).parent / "v89_rsl_rl_runner_config.json" if checkpoint_path else Path("")
    if checkpoint_path and cfg_path.exists():
        train_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        train_cfg = _rsl_rl_train_cfg(num_steps_per_env=max(1, int(eval_step_count)), max_iterations=1, policy_mode="shared_multi_object_policy")
    env_cfg = UnifiedGraspEnvCfg(
        num_envs=int(getattr(physical_backend, "num_envs", 1) or 1),
        parts=requested_parts,
        physics_profile=physics_profile,
        max_episode_length=max(1, int(eval_step_count) + 4),
        device=str(getattr(physical_backend, "device", "cpu")),
        sticky_after_support_validation=False,
        v86_guarded_residual_mode=True,
        v87_contact_guided_residual_mode=True,
        v88_stabilized_mode=True,
        v89_hybrid_repair_mode=bool(use_candidate_prior),
        v89_nominal_candidate_mode=bool(use_candidate_prior),
        v88_observation_enhanced=bool(use_candidate_prior),
        v88_nominal_lift_phase_enabled=True,
        v88_auto_reset_on_done=False,
        use_v85_near_contact_resets=True,
        v87_residual_scale_wrist_xyz=0.10 if use_candidate_prior else 0.12,
        v87_residual_scale_wrist_rot=0.05 if use_candidate_prior else 0.06,
        v87_residual_scale_finger=0.12 if use_candidate_prior else 0.16,
    )
    env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
    vec_env = RslRlUnifiedGraspVecEnv(env)
    policy = None
    if checkpoint_path:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=vec_env.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=vec_env.device)
    vec_env.reset()
    for step_index in range(max(1, int(eval_step_count))):
        if policy is None:
            actions = [[0.0] * env.num_actions for _ in range(env.num_envs)]
        else:
            obs = vec_env.get_observations()
            with torch.no_grad() if torch is not None else _nullcontext():
                actions = policy(obs) if callable(policy) else [[0.0] * env.num_actions for _ in range(env.num_envs)]
            if hasattr(actions, "detach"):
                actions = actions.detach().clone()
        start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        vec_env.step(actions)
        _annotate_v89_eval_trace(physical_backend, start, eval_mode, "residual_policy" if policy is not None else "nominal_only", step_index)
    return list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])


def _annotate_v89_eval_summaries(rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], branch_name: str) -> None:
    for row in rows:
        part_rows = [item for item in trace_rows if item.get("part_name") == row.get("part_name")]
        multi = [
            item
            for item in part_rows
            if int(item.get("effective_contact_count_force") or item.get("force_contact_count") or 0) >= 2
        ]
        jerks = [float(item.get("action_jerk") or 0.0) for item in part_rows]
        candidate_ids = sorted({str(item.get("candidate_id") or "") for item in part_rows if item.get("candidate_id")})
        row.update(
            {
                "branch_name": branch_name,
                "candidate_id": ",".join(candidate_ids),
                "multi_finger_support_rate": len(multi) / max(1, len(part_rows)),
                "action_jerk_mean": sum(jerks) / max(1, len(jerks)),
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "final_success": False,
            }
        )


def _annotate_v89_eval_trace(physical_backend: Any, start_index: int, eval_mode: str, phase: str, step_index: int) -> None:
    rows = getattr(physical_backend, "v86_rollout_trace_rows", []) or []
    for row in rows[int(start_index) :]:
        row["eval_mode"] = eval_mode
        row["eval_phase"] = phase
        row["eval_step_index"] = int(step_index)
        row["sticky_eval_ran"] = False
        row["final_replay_ran"] = False
        row["video_generated"] = False


def _run_v88_eval_rollout(
    *,
    requested_parts: list[str],
    physical_backend: Any,
    physics_profile: str,
    eval_step_count: int,
    checkpoint_path: str,
    eval_mode: str,
    enhanced_observation: bool,
) -> list[dict[str, Any]]:
    if callable(getattr(physical_backend, "clear_v88_rollout_trace", None)):
        physical_backend.clear_v88_rollout_trace()
    elif callable(getattr(physical_backend, "clear_v87_rollout_trace", None)):
        physical_backend.clear_v87_rollout_trace()
    cfg_path = Path(checkpoint_path).parent / "v88_rsl_rl_runner_config.json" if checkpoint_path else Path("")
    if checkpoint_path and cfg_path.exists():
        train_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        train_cfg = _rsl_rl_train_cfg(num_steps_per_env=max(1, int(eval_step_count)), max_iterations=1, policy_mode="shared_multi_object_policy")
    env_cfg = UnifiedGraspEnvCfg(
        num_envs=int(getattr(physical_backend, "num_envs", 1) or 1),
        parts=requested_parts,
        physics_profile=physics_profile,
        max_episode_length=max(1, int(eval_step_count) + 4),
        device=str(getattr(physical_backend, "device", "cpu")),
        sticky_after_support_validation=False,
        v86_guarded_residual_mode=True,
        v87_contact_guided_residual_mode=True,
        v88_stabilized_mode=True,
        v88_observation_enhanced=bool(enhanced_observation),
        v88_nominal_lift_phase_enabled=True,
        v88_auto_reset_on_done=False,
        use_v85_near_contact_resets=True,
        v87_residual_scale_wrist_xyz=0.12,
        v87_residual_scale_wrist_rot=0.06,
        v87_residual_scale_finger=0.16,
    )
    env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
    vec_env = RslRlUnifiedGraspVecEnv(env)
    policy = None
    if checkpoint_path:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=vec_env.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=vec_env.device)
    vec_env.reset()
    for step_index in range(max(1, int(eval_step_count))):
        if policy is None:
            actions = [[0.0] * env.num_actions for _ in range(env.num_envs)]
        else:
            obs = vec_env.get_observations()
            with torch.no_grad() if torch is not None else _nullcontext():
                actions = policy(obs) if callable(policy) else [[0.0] * env.num_actions for _ in range(env.num_envs)]
            if hasattr(actions, "detach"):
                actions = actions.detach().clone()
        start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        vec_env.step(actions)
        _annotate_v88_eval_trace(physical_backend, start, eval_mode, "residual_policy" if policy is not None else "nominal_only", step_index)
    return list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])


def _summarize_v88_eval(
    parts: list[str],
    trace_rows: list[dict[str, Any]],
    *,
    branch_name: str,
    physics_profile: str,
    eval_mode: str,
    checkpoint_exists: bool,
    backend_ready: bool,
    eval_error: str,
    eval_step_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = 0.05
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for part in parts:
        part_rows = [row for row in trace_rows if row.get("part_name") == part]
        hold_rows = [row for row in part_rows if row.get("nominal_phase") == "hold_squeeze"]
        lift_rows = [row for row in part_rows if row.get("nominal_phase") == "lift_stabilize"]
        force_rows = [
            row
            for row in part_rows
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
        ]
        peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in part_rows]
        streaks = [float(row.get("force_contact_streak_steps") or row.get("contact_duration_steps") or 0.0) for row in part_rows]
        displacements = [float(row.get("object_displacement_m") or 0.0) for row in part_rows]
        hold_displacements = [float(row.get("hold_object_displacement_m") or 0.0) for row in part_rows]
        lift_deltas = [float(row.get("lift_delta_z_m") or row.get("object_delta_z_m") or 0.0) for row in lift_rows]
        support_rows = [row for row in part_rows if bool(row.get("support_gate_ok"))]
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
        forbidden = bool(
            any(bool(row.get("object_write_by_policy_detected")) for row in part_rows)
            or any(bool(row.get("sticky_action_available_to_policy")) for row in part_rows)
            or any(bool(row.get("fallback_success_used")) for row in part_rows)
        )
        support_rate = len(support_rows) / max(1, len(part_rows))
        hold_rate = len(hold_hits) / max(1, len(hold_rows))
        lift_rate = len(lift_hits) / max(1, len(lift_rows))
        force_rate = len(force_rows) / max(1, len(part_rows))
        contact_duration_mean = sum(streaks) / max(1, len(streaks))
        excessive_rate = sum(1 for item in part_rows if bool(item.get("excessive_force"))) / max(1, len(part_rows))
        peak_force = max([0.0, *peaks])
        hold_displacement_max = max([0.0, *hold_displacements, *displacements])
        success = bool(
            force_rate >= 0.25
            and support_rate > 0.0
            and hold_rate >= 0.5
            and lift_rate >= 0.25
            and excessive_rate <= 0.01
            and peak_force <= 175.0
            and hold_displacement_max <= 0.04
            and not forbidden
        )
        if not checkpoint_exists:
            status = "EVAL_NOT_STARTED"
            blocker = "checkpoint_missing"
            failure_category = "TRAINING_NOT_STARTED"
        elif not backend_ready:
            status = "EVAL_NOT_STARTED"
            blocker = "single_context_backend_not_ready"
            failure_category = "ENV_NOT_VECTORIZEABLE"
        elif eval_error:
            status = "EVAL_FAILED"
            blocker = eval_error
            failure_category = "ENV_NOT_VECTORIZEABLE"
        elif success:
            status = "GRASP_CONTACT_SUPPORT_HOLD_LIFT_PASS_NO_STICKY"
            blocker = ""
            failure_category = ""
        elif support_rate <= 0.0:
            status = "POLICY_NO_SUPPORT"
            blocker = "deterministic_eval_no_support_gate_pass"
            failure_category = "CONTACT_BUT_NO_SUPPORT" if force_rows else "POLICY_NO_CONTACT"
        elif hold_rate < 0.5:
            status = "HOLD_FAILED"
            blocker = "deterministic_eval_hold_not_stable"
            failure_category = "SLIP_AFTER_CONTACT"
        elif lift_rate < 0.25:
            status = "LIFT_FAILED"
            blocker = "deterministic_eval_lift_not_observed"
            failure_category = "SLIP_AFTER_CONTACT"
        elif excessive_rate > 0.01 or peak_force > 175.0:
            status = "EXCESSIVE_FORCE_REJECTED"
            blocker = "deterministic_eval_excessive_force"
            failure_category = "PENETRATION"
        elif hold_displacement_max > 0.04:
            status = "HOLD_DISPLACEMENT_TOO_HIGH"
            blocker = "deterministic_eval_hold_displacement_gt_0p04m"
            failure_category = "SLIP_AFTER_CONTACT"
        else:
            status = "FORBIDDEN_ACTION_OR_FALLBACK_REJECTED"
            blocker = "object_write_sticky_or_fallback_success_detected"
            failure_category = "STICKY_REJECTED_BEFORE_SUPPORT"
        row = {
            "part_name": part,
            "branch_name": branch_name,
            "physics_profile": physics_profile,
            "eval_mode": eval_mode,
            "eval_step_count": max(1, int(eval_step_count)) if checkpoint_exists and backend_ready and not eval_error else 0,
            "force_contact_rate": force_rate,
            "contact_duration_mean": contact_duration_mean,
            "contact_stability_score": force_rate + min(contact_duration_mean, 128.0) / 128.0,
            "support_gate_rate": support_rate,
            "hold_gate_rate": hold_rate,
            "lift_gate_rate": lift_rate,
            "support_eval_ok": support_rate > 0.0,
            "hold_eval_ok": hold_rate >= 0.5,
            "lift_eval_ok": lift_rate >= 0.25,
            "force_contact_peak_n": peak_force,
            "force_contact_mean_n": sum(peaks) / max(1, len(peaks)),
            "mean_force_n": sum(peaks) / max(1, len(peaks)),
            "peak_force_n": peak_force,
            "excessive_force_threshold_n": 150.0,
            "excessive_force_rate": excessive_rate,
            "object_displacement_max_m": max([0.0, *displacements]),
            "hold_object_displacement_max_m": hold_displacement_max,
            "lift_delta_z_max_m": max([0.0, *lift_deltas]),
            "object_write_by_policy_detected": any(bool(item.get("object_write_by_policy_detected")) for item in part_rows),
            "sticky_action_available_to_policy": any(bool(item.get("sticky_action_available_to_policy")) for item in part_rows),
            "fallback_success_used": any(bool(item.get("fallback_success_used")) for item in part_rows),
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
            "grasp_success_claimed": bool(success),
            "rl_trained_success_evidence": bool(success),
            "status": status,
            "failure_category": failure_category,
            "blocker": blocker,
        }
        rows.append(row)
        failures.append({"part_name": part, "branch_name": branch_name, "failure_category": failure_category, "blocker": blocker})
    return rows, failures


def _run_v87_eval_rollout(
    *,
    requested_parts: list[str],
    physical_backend: Any,
    physics_profile: str,
    eval_step_count: int,
    checkpoint_path: str,
    eval_mode: str,
) -> list[dict[str, Any]]:
    if callable(getattr(physical_backend, "clear_v87_rollout_trace", None)):
        physical_backend.clear_v87_rollout_trace()
    elif callable(getattr(physical_backend, "clear_v86_rollout_trace", None)):
        physical_backend.clear_v86_rollout_trace()
    cfg_path = Path(checkpoint_path).parent / "v87_rsl_rl_runner_config.json" if checkpoint_path else Path("")
    if checkpoint_path and cfg_path.exists():
        train_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        train_cfg = _rsl_rl_train_cfg(num_steps_per_env=max(1, int(eval_step_count)), max_iterations=1, policy_mode="shared_multi_object_policy")
    env_cfg = UnifiedGraspEnvCfg(
        num_envs=int(getattr(physical_backend, "num_envs", 1) or 1),
        parts=requested_parts,
        physics_profile=physics_profile,
        max_episode_length=max(1, int(eval_step_count) + 24),
        device=str(getattr(physical_backend, "device", "cpu")),
        sticky_after_support_validation=False,
        v86_guarded_residual_mode=True,
        v87_contact_guided_residual_mode=True,
        use_v85_near_contact_resets=True,
    )
    env = UnifiedGraspEnv(env_cfg, physical_backend=physical_backend)
    vec_env = RslRlUnifiedGraspVecEnv(env)
    policy = None
    if checkpoint_path:
        from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

        runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device=vec_env.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=vec_env.device)
    vec_env.reset()
    for step_index in range(max(1, int(eval_step_count))):
        if policy is None:
            actions = [[0.0] * env.num_actions for _ in range(env.num_envs)]
        else:
            obs = vec_env.get_observations()
            with torch.no_grad() if torch is not None else _nullcontext():
                actions = policy(obs) if callable(policy) else [[0.0] * env.num_actions for _ in range(env.num_envs)]
            if hasattr(actions, "detach"):
                actions = actions.detach().clone()
        start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        vec_env.step(actions)
        _annotate_v87_eval_trace(physical_backend, start, eval_mode, "residual_policy" if policy is not None else "nominal_only", step_index)
    for step_index in range(12):
        start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        env.step(_v87_eval_residual_actions(env.num_envs, "hold"))
        _annotate_v87_eval_trace(physical_backend, start, eval_mode, "hold", step_index)
    for step_index in range(12):
        start = len(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])
        env.step(_v87_eval_residual_actions(env.num_envs, "lift"))
        _annotate_v87_eval_trace(physical_backend, start, eval_mode, "lift", step_index)
    return list(getattr(physical_backend, "v86_rollout_trace_rows", []) or [])


def _summarize_v87_eval(
    parts: list[str],
    trace_rows: list[dict[str, Any]],
    *,
    eval_mode: str,
    checkpoint_exists: bool,
    backend_ready: bool,
    eval_error: str,
    eval_step_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = 0.05
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for part in parts:
        part_rows = [row for row in trace_rows if row.get("part_name") == part]
        hold_rows = [row for row in part_rows if row.get("eval_phase") == "hold" or row.get("nominal_phase") == "hold_squeeze"]
        lift_rows = [row for row in part_rows if row.get("eval_phase") == "lift"]
        force_rows = [
            row
            for row in part_rows
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            and int(row.get("effective_contact_count_force") or row.get("force_contact_count") or 0) >= 1
        ]
        peaks = [float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) for row in part_rows]
        streaks = [float(row.get("force_contact_streak_steps") or row.get("contact_duration_steps") or 0.0) for row in part_rows]
        displacements = [float(row.get("object_displacement_m") or 0.0) for row in part_rows]
        support_rows = [row for row in part_rows if bool(row.get("support_gate_ok"))]
        hold_hits = [
            row
            for row in hold_rows
            if float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
            and not bool(row.get("table_collision"))
        ]
        lift_hits = [
            row
            for row in lift_rows
            if float(row.get("object_delta_z_m") or 0.0) > 0.005
            and float(row.get("force_contact_peak_n") or row.get("force_contact_probe_peak_n") or 0.0) > threshold
        ]
        forbidden = bool(
            any(bool(row.get("object_write_by_policy_detected")) for row in part_rows)
            or any(bool(row.get("sticky_action_available_to_policy")) for row in part_rows)
        )
        support_rate = len(support_rows) / max(1, len(part_rows))
        hold_rate = len(hold_hits) / max(1, len(hold_rows))
        lift_rate = len(lift_hits) / max(1, len(lift_rows))
        force_rate = len(force_rows) / max(1, len(part_rows))
        contact_duration_mean = sum(streaks) / max(1, len(streaks))
        success = bool(support_rate > 0.0 and hold_rate > 0.0 and lift_rate > 0.0 and not forbidden)
        if not checkpoint_exists:
            status = "EVAL_NOT_STARTED"
            blocker = "checkpoint_missing"
            failure_category = "TRAINING_NOT_STARTED"
        elif not backend_ready:
            status = "EVAL_NOT_STARTED"
            blocker = "single_context_backend_not_ready"
            failure_category = "ENV_NOT_VECTORIZEABLE"
        elif eval_error:
            status = "EVAL_FAILED"
            blocker = eval_error
            failure_category = "ENV_NOT_VECTORIZEABLE"
        elif success:
            status = "GRASP_CONTACT_SUPPORT_HOLD_LIFT_PASS_NO_STICKY"
            blocker = ""
            failure_category = ""
        elif support_rate <= 0.0:
            status = "POLICY_NO_SUPPORT"
            blocker = "deterministic_eval_no_support_gate_pass"
            failure_category = "CONTACT_BUT_NO_SUPPORT" if force_rows else "POLICY_NO_CONTACT"
        elif hold_rate <= 0.0:
            status = "HOLD_FAILED"
            blocker = "deterministic_eval_hold_not_stable"
            failure_category = "SLIP_AFTER_CONTACT"
        else:
            status = "LIFT_FAILED"
            blocker = "deterministic_eval_lift_not_observed"
            failure_category = "SLIP_AFTER_CONTACT"
        row = {
            "part_name": part,
            "eval_mode": eval_mode,
            "eval_step_count": max(1, int(eval_step_count)) if checkpoint_exists and backend_ready and not eval_error else 0,
            "force_contact_rate": force_rate,
            "contact_duration_mean": contact_duration_mean,
            "contact_stability_score": force_rate + min(contact_duration_mean, 96.0) / 96.0,
            "support_gate_rate": support_rate,
            "hold_gate_rate": hold_rate,
            "lift_gate_rate": lift_rate,
            "support_eval_ok": support_rate > 0.0,
            "hold_eval_ok": hold_rate > 0.0,
            "lift_eval_ok": lift_rate > 0.0,
            "force_contact_peak_n": max([0.0, *peaks]),
            "force_contact_mean_n": sum(peaks) / max(1, len(peaks)),
            "mean_force_n": sum(peaks) / max(1, len(peaks)),
            "peak_force_n": max([0.0, *peaks]),
            "excessive_force_threshold_n": 150.0,
            "excessive_force_rate": sum(1 for item in part_rows if bool(item.get("excessive_force"))) / max(1, len(part_rows)),
            "object_displacement_max_m": max([0.0, *displacements]),
            "object_write_by_policy_detected": any(bool(item.get("object_write_by_policy_detected")) for item in part_rows),
            "sticky_action_available_to_policy": any(bool(item.get("sticky_action_available_to_policy")) for item in part_rows),
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
            "grasp_success_claimed": bool(success),
            "rl_trained_success_evidence": bool(success),
            "status": status,
            "failure_category": failure_category,
            "blocker": blocker,
        }
        rows.append(row)
        failures.append({"part_name": part, "failure_category": failure_category, "blocker": blocker})
    return rows, failures


def _v87_eval_residual_actions(num_envs: int, phase: str) -> list[list[float]]:
    rows = []
    for _index in range(max(1, int(num_envs))):
        action = [0.0] * 16
        if phase == "lift":
            action[2] = 1.0
        rows.append(action)
    return rows


def _annotate_v87_eval_trace(physical_backend: Any, start_index: int, eval_mode: str, phase: str, step_index: int) -> None:
    rows = getattr(physical_backend, "v86_rollout_trace_rows", []) or []
    for row in rows[int(start_index) :]:
        row["eval_mode"] = eval_mode
        row["eval_phase"] = phase
        row["eval_step_index"] = int(step_index)
        row["sticky_eval_ran"] = False


def _annotate_v88_eval_trace(physical_backend: Any, start_index: int, eval_mode: str, phase: str, step_index: int) -> None:
    rows = getattr(physical_backend, "v86_rollout_trace_rows", []) or []
    for row in rows[int(start_index) :]:
        row["eval_mode"] = eval_mode
        row["eval_phase"] = phase
        row["eval_step_index"] = int(step_index)
        row["sticky_eval_ran"] = False
        row["final_replay_ran"] = False
        row["video_generated"] = False


def _v86_eval_actions(num_envs: int, phase: str) -> list[list[float]]:
    rows = []
    for _index in range(max(1, int(num_envs))):
        action = [0.0] * 16
        if phase in {"hold", "lift"}:
            for col in range(6, 16):
                action[col] = 0.75
        if phase == "lift":
            action[2] = 0.25
        rows.append(action)
    return rows


def _annotate_v86_eval_trace(physical_backend: Any, start_index: int, phase: str, step_index: int) -> None:
    rows = getattr(physical_backend, "v86_rollout_trace_rows", []) or []
    for row in rows[int(start_index) :]:
        row["eval_phase"] = phase
        row["eval_step_index"] = int(step_index)
        row["sticky_eval_ran"] = False


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> bool:
        return False
