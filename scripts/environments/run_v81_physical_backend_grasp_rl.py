"""Run v81 physical-backend unified grasp RL gates.

This runner must be launched through IsaacLab (``./isaaclab.sh -p``).  It keeps
v80 surrogate dynamics as unit-test-only evidence and only starts real RSL-RL
when the physical backend and force contact API are both ready.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
for rel in ("source/isaaclab", "source/isaaclab_tasks", "source/isaaclab_assets", "source/isaaclab_rl"):
    path = str(REPO_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
PACKAGE_ROOT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


RUN_DIR_BY_MODE = {
    "v95_minimal_unified_workcell_repair": "debug_runs/v95_minimal_unified_workcell_repair",
    "v94_safe_pregrasp_and_finger_contact_calibration": "debug_runs/v94_safe_pregrasp_and_finger_contact_calibration",
    "v93_collision_geometry_and_safe_staging_repair": "debug_runs/v93_collision_geometry_and_safe_staging_repair",
    "v92_asset_task_setup_and_contact_calibration": "debug_runs/v92_asset_task_setup_and_contact_calibration",
    "v91_task_setup_asset_dynamics_and_controller_sanity": "debug_runs/v91_task_setup_asset_dynamics_and_controller_sanity",
    "v90_bottleneck_isolation_and_grasp_feasibility": "debug_runs/v90_bottleneck_isolation_and_grasp_feasibility",
    "v89_failure_driven_hybrid_grasp_repair": "debug_runs/v89_failure_driven_hybrid_grasp_repair",
    "v88_stabilized_grasp_policy_and_physics_audit": "debug_runs/v88_stabilized_grasp_policy_and_physics_audit",
    "v87_contact_guided_residual_ppo": "debug_runs/v87_contact_guided_residual_ppo",
    "v86_guarded_residual_ppo_smoke": "debug_runs/v86_guarded_residual_ppo_smoke",
    "v85_contact_sanity_and_plug_screw_repair": "debug_runs/v85_contact_sanity_and_plug_screw_repair",
    "v84_controlled_real_contact_probe": "debug_runs/v84_controlled_real_contact_probe",
    "v83_unified_physical_backend": "debug_runs/v83_unified_physical_backend",
    "v82_contact_actuation_probe": "debug_runs/v82_contact_actuation_probe",
    "physical_backend_probe_audit": "debug_runs/v81_physical_backend_probe_audit",
    "unified_env_real_vectorized_smoke": "debug_runs/v81_unified_env_real_vectorized_smoke",
    "real_residual_ppo_smoke": "debug_runs/v81_real_residual_ppo_smoke",
    "deterministic_eval_no_sticky": "debug_runs/v81_deterministic_eval_no_sticky",
    "deterministic_eval_sticky_after_support": "debug_runs/v81_deterministic_eval_sticky_after_support",
}


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run_mode", choices=tuple(RUN_DIR_BY_MODE), default="physical_backend_probe_audit")
parser.add_argument("--output_dir", default="")
parser.add_argument("--parts", default="Plug2,Screw1,Backrest,Rod,Frame")
parser.add_argument("--physics_profile", default="canonical")
parser.add_argument("--policy_mode", choices=("shared_multi_object_policy", "per_object_policy"), default="shared_multi_object_policy")
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=32)
parser.add_argument("--ppo_iterations", type=int, default=5)
parser.add_argument("--v88_branches", default="v87_baseline_continuation,physics_tuned_residual_ppo")
parser.add_argument("--v89_branches", default="baseline_v88_best_continuation,nominal_candidate_residual_ppo,bc_warmstart_residual_ppo")
parser.add_argument("--v89_bc_epochs", type=int, default=3)
parser.add_argument("--top_k_priors_per_part", type=int, default=32)
parser.add_argument("--program_prior_search_roots", default="debug_runs")
parser.add_argument("--allow_diagnostic_dynamics", nargs="?", const=True, default=False, type=_parse_bool)
parser.add_argument("--checkpoint_path", default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pipeline.unified_grasp.asset_audit import audit_assets  # noqa: E402
from pipeline.unified_grasp.curriculum import write_curriculum_artifacts  # noqa: E402
from pipeline.unified_grasp.eval_unified_rl import (  # noqa: E402
    evaluate_unified_policy,
    evaluate_v86_guarded_residual_policy,
    evaluate_v87_contact_guided_policy,
    evaluate_v88_stabilized_policy,
    evaluate_v89_hybrid_policy,
)
from pipeline.unified_grasp.grasp_prior_generator import write_grasp_prior_bank  # noqa: E402
from pipeline.unified_grasp.object_randomization import write_object_randomization_report  # noqa: E402
from pipeline.unified_grasp.physical_backend import (  # noqa: E402
    IsaacUnifiedGraspPhysicalBackend,
    write_backend_creation_report,
    write_controlled_contact_probe,
    write_v82_contact_actuation_probe,
)
from pipeline.unified_grasp.seed_bank import load_unified_seed_bank  # noqa: E402
from pipeline.unified_grasp.single_context_backend import (  # noqa: E402
    IsaacUnifiedSingleContextBackend,
    run_v94_actual_object_contact_calibration,
    run_v94_diagnostic_target_finger_sensor_map,
    run_v94_finger_motion_calibration,
    run_v94_pregrasp_alignment_audit,
    run_v94_wrist_action_calibration,
    run_v93_collision_geometry_audit_and_repair,
    run_v93_native_shutdown_root_cause,
    run_v93_post_collision_contact_sanity,
    run_v93_safe_staging_audit,
    run_v93_screw1_post_repair_sanity,
    run_v93_support_gate_post_repair_audit,
    run_v92_collision_subtree_audit_and_repair,
    run_v92_finger_action_sensor_calibration,
    run_v92_sanity_probe,
    run_v92_screw1_dynamic_repair_audit,
    run_v92_staging_repair_audit,
    run_v92_support_gate_calibration,
    run_v91_asset_dynamics_audit,
    run_v91_observation_runtime_audit,
    run_v91_quasistatic_feasibility_bench,
    run_v91_staging_sanity_bench,
    run_v90_feasibility_bench,
    run_v89_candidate_probe,
    run_v86_contact_gate_preflight,
    run_v85_contact_sanity_and_plug_screw_repair,
    run_v84_controlled_real_contact_probe,
    run_v83_unified_physical_backend_audit,
)
from pipeline.unified_grasp.train_unified_rl import (  # noqa: E402
    train_unified_policy,
    train_v86_guarded_residual_ppo_smoke,
    train_v87_contact_guided_residual_ppo,
    train_v88_stabilized_residual_ppo,
    train_v89_hybrid_residual_ppo,
)
from pipeline.unified_grasp.unified_grasp_env import run_env_smoke  # noqa: E402
from pipeline.unified_grasp.v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json  # noqa: E402
from pipeline.unified_grasp.v81_reports import (  # noqa: E402
    default_v81_progress_row,
    merge_v81_progress_rows,
    write_v81_progress_matrix,
    write_v81_source_archive_check,
)
from pipeline.unified_grasp.v88_audits import (  # noqa: E402
    write_v88_code_path_audit,
    write_v88_forbidden_artifact_scan,
    write_v88_observation_audit,
    write_v88_physics_asset_audit,
)
from pipeline.unified_grasp.v89_hybrid_repair import (  # noqa: E402
    V89_BRANCHES,
    load_v88_artifacts,
    select_best_candidates,
    valid_bc_rows,
    write_v89_code_growth_report,
    write_v89_failure_diagnosis,
    write_v89_forbidden_artifact_scan,
    write_v89_nominal_candidate_plan,
    write_v89_physics_decision,
    write_v89_progress_matrix,
)
from pipeline.unified_grasp.v90_bottleneck_isolation import (  # noqa: E402
    classify_v90_bottlenecks,
    write_v90_code_freeze_report,
    write_v90_disk_preflight,
    write_v90_feasibility_candidate_plan,
    write_v90_forbidden_artifact_scan,
    write_v90_physics_asset_bottleneck_report,
    write_v90_progress_matrix,
)
from pipeline.unified_grasp.v91_task_sanity import (  # noqa: E402
    classify_v91_decisions,
    write_v91_feasibility_candidate_plan,
    write_v91_forbidden_artifact_scan,
    write_v91_progress_matrix,
)
from pipeline.unified_grasp.v92_task_setup_calibration import (  # noqa: E402
    classify_v92_progress,
    write_v92_forbidden_artifact_scan,
    write_v92_task_initial_condition_plan,
)
from pipeline.unified_grasp.v93_collision_geometry_safe_staging import (  # noqa: E402
    classify_v93_readiness,
    write_v93_forbidden_artifact_scan,
    write_v93_safe_staging_plan,
)
from pipeline.unified_grasp.v94_safe_pregrasp_contact_calibration import (  # noqa: E402
    classify_v94_readiness,
    write_v94_collision_proxy_decision,
    write_v94_forbidden_artifact_scan,
    write_v94_v93_blocker_reclassification,
)
from pipeline.unified_grasp.minimal_workcell_gate import (  # noqa: E402
    run_v95_minimal_workcell_gate,
    write_v95_workcell_code_freeze,
)


def _parts(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    return [part for part in V80_PARTS if part in requested] or list(V80_PARTS)


def _run_dir() -> Path:
    return ensure_run_dir(args_cli.output_dir or RUN_DIR_BY_MODE[args_cli.run_mode])


def _foundation_progress(
    *,
    parts: list[str],
    physics_profile: str,
    seed_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    curriculum_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        rows.append(
            default_v81_progress_row(
                part,
                physics_profile=physics_profile,
                seed_count=sum(1 for row in seed_rows if row.get("part_name") == part),
                allowed_curriculum_count=sum(1 for row in curriculum_rows if row.get("part_name") == part),
                analytic_prior_unvalidated_count=sum(
                    1 for row in prior_rows if row.get("part_name") == part and bool(row.get("analytic_prior_unvalidated"))
                ),
                reachable_prior_physical_count=sum(
                    1 for row in prior_rows if row.get("part_name") == part and bool(row.get("reachable_prior_physical"))
                ),
                failure_category="ENV_NOT_VECTORIZEABLE",
                blocker="physical_backend_probe_not_completed",
                next_action="run_v81_physical_backend_probe_audit",
            )
        )
    return rows


def _progress_from_probe(parts: list[str], probe_rows: list[dict[str, Any]], backend: IsaacUnifiedGraspPhysicalBackend) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        probe = next((row for row in probe_rows if row.get("part_name") == part), {})
        contact_available = bool(probe.get("contact_sensor_available"))
        ready = bool(probe.get("physical_backend_ready") or backend.part_backend_ready(part))
        blocker = "" if ready and contact_available else str(probe.get("blocker") or backend.blocker or "CONTACT_SENSOR_API_NOT_AVAILABLE")
        rows.append(
            {
                "part_name": part,
                "physical_backend_ready": ready,
                "contact_sensor_available": contact_available,
                "controlled_fingertip_probe_executed": bool(probe.get("controlled_fingertip_probe_executed")),
                "contact_evidence_source": str(probe.get("contact_evidence_source") or "CONTACT_SENSOR_API_NOT_AVAILABLE"),
                "force_contact_probe_peak_n": float(probe.get("force_contact_probe_peak_n") or 0.0),
                "object_motion_during_probe_m": float(probe.get("object_motion_during_probe_m") or 0.0),
                "object_write_by_policy_detected": bool(probe.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": bool(probe.get("sticky_action_available_to_policy")),
                "failure_category": "" if ready and contact_available else "CONTACT_SENSOR_UNAVAILABLE_FALLBACK_ONLY",
                "blocker": blocker,
                "next_action": "run_v81_unified_env_real_vectorized_smoke" if ready and contact_available else "repair_physical_backend_contact_sensor",
            }
        )
    return rows


def _progress_from_env(parts: list[str], env_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        env = next((row for row in env_rows if row.get("part_name") == part), {})
        ok = bool(env.get("rl_env_vectorized_ok"))
        rows.append(
            {
                "part_name": part,
                "rl_env_vectorized_ok": ok,
                "actual_env_count": int(env.get("actual_env_count") or 0),
                "failure_category": "" if ok else "ENV_NOT_VECTORIZEABLE",
                "blocker": "" if ok else str(env.get("blocker") or "physical_backend_vector_step_failed"),
                "next_action": "run_v81_real_residual_ppo_smoke" if ok else "fix_vector_reset_step",
            }
        )
    return rows


def _progress_from_training(parts: list[str], result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    train_rows = result.get("rows", [])
    for part in parts:
        row = next((item for item in train_rows if item.get("part_name") == part), {})
        trained = bool(row.get("rl_trained"))
        rows.append(
            {
                "part_name": part,
                "rsl_rl_runner_used": bool(row.get("rsl_rl_runner_used")),
                "surrogate_training_used": bool(row.get("surrogate_training_used")),
                "rl_trained": trained,
                "ppo_iterations_completed": int(row.get("ppo_iterations_completed") or 0),
                "rollout_step_count": int(row.get("rollout_step_count") or 0),
                "training_curve_nonempty": bool(row.get("training_curve_nonempty")),
                "checkpoint_path": str(result.get("checkpoint_path") or "") if trained else "",
                "failure_category": "" if trained else "TRAINING_NOT_STARTED",
                "blocker": "" if trained else str(row.get("blocker") or result.get("failure_reason") or "training_not_started"),
                "next_action": "run_v81_deterministic_eval_no_sticky" if trained else "fix_real_ppo_gate",
            }
        )
    return rows


def _progress_from_eval(parts: list[str], result: dict[str, Any], *, sticky_after_support: bool) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        row = next((item for item in result.get("rows", []) if item.get("part_name") == part), {})
        update = {
            "part_name": part,
            "support_gate_rate": float(row.get("support_gate_rate") or 0.0),
            "force_contact_count_mean": float(row.get("average_effective_contact_count_force") or 0.0),
            "object_motion_before_contact_rate": float(row.get("object_motion_before_contact_rate") or 0.0),
            "penetration_rate": float(row.get("penetration_rate") or 0.0),
            "table_collision_rate": float(row.get("table_collision_rate") or 0.0),
            "hold_success_rate": float(row.get("hold_success_rate") or 0.0),
            "lift_success_rate": float(row.get("lift_success_rate") or 0.0),
            "usable_training_row_count": int(row.get("usable_training_row_count") or 0),
            "failure_category": str(row.get("failure_category") or "POLICY_NO_CONTACT"),
            "blocker": str(row.get("blocker") or ""),
            "next_action": "serial_final_replay_validation"
            if bool(row.get("support_gate_ok"))
            else (
                "fix_vector_reset_step"
                if str(row.get("failure_category") or "") == "ENV_NOT_VECTORIZEABLE"
                else "improve_policy_contact_support"
            ),
        }
        if sticky_after_support:
            update.update(
                {
                    "eval_episode_count_sticky_after_support": int(row.get("eval_episode_count") or 0),
                    "success_with_sticky_after_support_rate": float(row.get("success_with_sticky_after_support_rate") or 0.0),
                }
            )
        else:
            update.update(
                {
                    "eval_episode_count_no_sticky": int(row.get("eval_episode_count") or 0),
                    "success_no_sticky_rate": float(row.get("success_no_sticky_rate") or 0.0),
                }
            )
        rows.append(update)
    return rows


def _write_v82_progress_matrix(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    fields = [
        "part_name",
        "physical_backend_ready",
        "object_identity_verified",
        "active_held_asset_name",
        "active_held_asset_usd",
        "expected_part_name",
        "object_identity_blocker",
        "policy_action_dim",
        "isaac_action_width",
        "mapped_close_cols",
        "metric_close_dof_commanded",
        "contact_sensor_api_available",
        "nonzero_force_contact_observed",
        "force_contact_probe_peak_n",
        "effective_contact_count_force_max",
        "object_write_by_policy_detected",
        "sticky_action_available_to_policy",
        "controlled_contact_gate_pass",
        "ppo_ran",
        "sticky_eval_ran",
        "final_replay_ran",
        "usable_training_row_count",
        "status",
        "blocker",
        "next_action",
    ]
    matrix_rows = []
    for row in rows:
        gate = bool(row.get("controlled_contact_gate_pass"))
        matrix = {field: row.get(field, "") for field in fields}
        matrix.update(
            {
                "ppo_ran": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "usable_training_row_count": 0,
                "status": "PASS" if gate else str(row.get("status") or "CONTACT_ACTUATION_NO_FORCE_CONTACT"),
                "next_action": "unlock_training_gate" if gate else "repair_policy_to_isaac_contact_actuation",
            }
        )
        matrix_rows.append(matrix)
    csv_path = write_csv(run_dir / "v82_progress_matrix.csv", matrix_rows, fields)
    md_path = run_dir / "v82_progress_matrix.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in matrix_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    root_csv = write_csv(REPO_ROOT / "debug_runs/v82_progress_matrix.csv", matrix_rows, fields)
    root_md = REPO_ROOT / "debug_runs/v82_progress_matrix.md"
    root_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "v82_progress_csv": str(csv_path),
        "v82_progress_md": str(md_path),
        "v82_root_progress_csv": str(root_csv),
        "v82_root_progress_md": str(root_md),
    }


def _write_v86_progress_matrix(
    run_dir: Path,
    *,
    parts: list[str],
    preflight_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, str]:
    fields = [
        "part_name",
        "object_identity_verified",
        "single_simulation_context",
        "v86_contact_gate_pass",
        "rollout_env_count",
        "rollout_participation_count",
        "rollout_force_contact_rate",
        "eval_force_contact_rate",
        "force_contact_rate",
        "support_eval_ok",
        "hold_eval_attempted",
        "hold_eval_ok",
        "lift_eval_attempted",
        "lift_eval_ok",
        "force_contact_peak_n",
        "force_contact_mean_n",
        "excessive_force_threshold_n",
        "excessive_force_rate",
        "object_displacement_max_m",
        "object_write_by_policy_detected",
        "sticky_action_available_to_policy",
        "ppo_ran",
        "rsl_rl_runner_used",
        "runner_smoke_complete",
        "rl_trained",
        "rl_trained_success_evidence",
        "grasp_success_claimed",
        "checkpoint_written",
        "checkpoint_is_runner_artifact_only",
        "checkpoint_path",
        "ppo_iterations_completed",
        "rollout_step_count",
        "training_curve_nonempty",
        "sticky_eval_ran",
        "final_replay_ran",
        "video_generated",
        "usable_training_row_count",
        "status",
        "blocker",
        "next_action",
    ]
    matrix_rows = []
    for part in parts:
        preflight = next((row for row in preflight_rows if row.get("part_name") == part), {})
        train = next((row for row in train_rows if row.get("part_name") == part), {})
        eval_row = next((row for row in eval_rows if row.get("part_name") == part), {})
        eval_has_metrics = bool(int(eval_row.get("eval_step_count") or 0) > 0)
        rollout_force_rate = float(train.get("force_contact_rate") or 0.0)
        eval_force_rate = float(eval_row.get("force_contact_rate") or 0.0) if eval_has_metrics else 0.0
        rollout_peak = float(train.get("force_contact_peak_n") or 0.0)
        eval_peak = float(eval_row.get("force_contact_peak_n") or 0.0) if eval_has_metrics else 0.0
        rollout_mean = float(train.get("force_contact_mean_n") or 0.0)
        eval_mean = float(eval_row.get("force_contact_mean_n") or 0.0) if eval_has_metrics else 0.0
        rollout_disp = float(train.get("object_displacement_max_m") or 0.0)
        eval_disp = float(eval_row.get("object_displacement_max_m") or 0.0) if eval_has_metrics else 0.0
        runner_ok = bool(train.get("runner_smoke_complete"))
        grasp_success = bool(eval_row.get("grasp_success_claimed"))
        safety_ok = not bool(train.get("object_write_by_policy_detected")) and not bool(train.get("sticky_action_available_to_policy"))
        if grasp_success:
            status = "GRASP_EVAL_PASS_NO_STICKY"
            blocker = ""
            next_action = "review_before_any_later_final_replay_stage"
        elif runner_ok and safety_ok:
            status = "PPO_SMOKE_PASS_NO_GRASP_SUCCESS"
            blocker = str(eval_row.get("blocker") or "deterministic_eval_no_support_hold_lift_success")
            next_action = "inspect_policy_and_reward_before_longer_training"
        elif not bool(preflight.get("v86_contact_gate_pass")):
            status = "TRAINING_NOT_STARTED"
            blocker = str(preflight.get("blocker") or "controlled_force_contact_gate_failed")
            next_action = "repair_v85_contact_gate_regression"
        else:
            status = str(train.get("status") or "PPO_SMOKE_FAILED")
            blocker = str(train.get("blocker") or eval_row.get("blocker") or "v86_ppo_smoke_failed")
            next_action = "fix_v86_runner_or_backend_safety_gate"
        row = {
            "part_name": part,
            "object_identity_verified": bool(preflight.get("object_identity_verified")),
            "single_simulation_context": bool(preflight.get("single_simulation_context")),
            "v86_contact_gate_pass": bool(preflight.get("v86_contact_gate_pass")),
            "rollout_env_count": int(train.get("rollout_env_count") or 0),
            "rollout_participation_count": int(train.get("rollout_participation_count") or 0),
            "rollout_force_contact_rate": rollout_force_rate,
            "eval_force_contact_rate": eval_force_rate,
            "force_contact_rate": rollout_force_rate,
            "support_eval_ok": bool(eval_row.get("support_eval_ok")),
            "hold_eval_attempted": bool(eval_row.get("hold_eval_attempted")),
            "hold_eval_ok": bool(eval_row.get("hold_eval_ok")),
            "lift_eval_attempted": bool(eval_row.get("lift_eval_attempted")),
            "lift_eval_ok": bool(eval_row.get("lift_eval_ok")),
            "force_contact_peak_n": max(rollout_peak, eval_peak),
            "force_contact_mean_n": max(rollout_mean, eval_mean),
            "excessive_force_threshold_n": float(train.get("excessive_force_threshold_n") or eval_row.get("excessive_force_threshold_n") or 150.0),
            "excessive_force_rate": float((eval_row.get("excessive_force_rate") if eval_has_metrics else train.get("excessive_force_rate")) or 0.0),
            "object_displacement_max_m": max(rollout_disp, eval_disp),
            "object_write_by_policy_detected": bool(train.get("object_write_by_policy_detected") or eval_row.get("object_write_by_policy_detected")),
            "sticky_action_available_to_policy": bool(train.get("sticky_action_available_to_policy") or eval_row.get("sticky_action_available_to_policy")),
            "ppo_ran": bool(train.get("ppo_ran")),
            "rsl_rl_runner_used": bool(train.get("rsl_rl_runner_used")),
            "runner_smoke_complete": runner_ok,
            "rl_trained": False,
            "rl_trained_success_evidence": grasp_success,
            "grasp_success_claimed": grasp_success,
            "checkpoint_written": bool(train.get("checkpoint_written")),
            "checkpoint_is_runner_artifact_only": bool(train.get("checkpoint_is_runner_artifact_only")),
            "checkpoint_path": str(train.get("checkpoint_path") or ""),
            "ppo_iterations_completed": int(train.get("ppo_iterations_completed") or 0),
            "rollout_step_count": int(train.get("rollout_step_count") or 0),
            "training_curve_nonempty": bool(train.get("training_curve_nonempty")),
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "usable_training_row_count": 0,
            "status": status,
            "blocker": blocker,
            "next_action": next_action,
        }
        matrix_rows.append(row)
    csv_path = write_csv(run_dir / "v86_progress_matrix.csv", matrix_rows, fields)
    md_path = run_dir / "v86_progress_matrix.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in matrix_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    root_csv = write_csv(REPO_ROOT / "debug_runs/v86_progress_matrix.csv", matrix_rows, fields)
    root_md = REPO_ROOT / "debug_runs/v86_progress_matrix.md"
    root_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "rows": matrix_rows,
        "v86_progress_csv": str(csv_path),
        "v86_progress_md": str(md_path),
        "v86_root_progress_csv": str(root_csv),
        "v86_root_progress_md": str(root_md),
    }


def _read_v86_baseline_rows(parts: list[str]) -> tuple[dict[str, dict[str, Any]], bool]:
    path = REPO_ROOT / "debug_runs/v86_guarded_residual_ppo_smoke/v86_progress_matrix.csv"
    if not path.exists():
        return {part: {} for part in parts}, True
    rows: dict[str, dict[str, Any]] = {part: {} for part in parts}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            part = str(row.get("part_name") or "")
            if part in rows:
                rows[part] = dict(row)
    return rows, False


def _read_v87_baseline_rows(parts: list[str]) -> tuple[dict[str, dict[str, Any]], bool]:
    path = REPO_ROOT / "debug_runs/v87_contact_guided_residual_ppo/v87_progress_matrix.csv"
    if not path.exists():
        return {part: {} for part in parts}, True
    rows: dict[str, dict[str, Any]] = {part: {} for part in parts}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            part = str(row.get("part_name") or "")
            if part in rows:
                rows[part] = dict(row)
    return rows, False


def _write_v87_progress_matrix(
    run_dir: Path,
    *,
    parts: list[str],
    preflight_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    nominal_eval_rows: list[dict[str, Any]],
    policy_eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = [
        "part_name",
        "object_identity_verified",
        "single_simulation_context",
        "v85_contact_gate_pass",
        "rollout_env_count",
        "rollout_participation_count",
        "rollout_force_contact_rate",
        "nominal_force_contact_rate",
        "policy_eval_force_contact_rate",
        "force_contact_rate",
        "contact_duration_mean",
        "nominal_contact_duration_mean",
        "policy_eval_contact_duration_mean",
        "mean_force_n",
        "peak_force_n",
        "excessive_force_rate",
        "support_gate_rate",
        "hold_gate_rate",
        "lift_gate_rate",
        "object_displacement_max_m",
        "deterministic_eval_result",
        "v86_baseline_force_contact_rate",
        "v86_baseline_missing",
        "contact_stability_score",
        "nominal_contact_stability_score",
        "v86_contact_stability_score",
        "contact_stability_improved_vs_nominal",
        "contact_stability_improved_vs_v86",
        "object_write_by_policy_detected",
        "sticky_action_available_to_policy",
        "ppo_ran",
        "rsl_rl_runner_used",
        "runner_policy_artifact_complete",
        "rl_trained",
        "rl_trained_success_evidence",
        "grasp_success_claimed",
        "checkpoint_written",
        "checkpoint_is_policy_artifact_only",
        "checkpoint_path",
        "ppo_iterations_completed",
        "rollout_step_count",
        "training_curve_nonempty",
        "sticky_eval_ran",
        "final_replay_ran",
        "video_generated",
        "usable_training_row_count",
        "engineering_pass",
        "status",
        "blocker",
        "next_action",
    ]
    v86_rows, v86_missing = _read_v86_baseline_rows(parts)
    matrix_rows: list[dict[str, Any]] = []
    for part in parts:
        preflight = next((row for row in preflight_rows if row.get("part_name") == part), {})
        train = next((row for row in train_rows if row.get("part_name") == part), {})
        nominal = next((row for row in nominal_eval_rows if row.get("part_name") == part), {})
        policy = next((row for row in policy_eval_rows if row.get("part_name") == part), {})
        v86 = v86_rows.get(part, {})
        rollout_force = float(train.get("force_contact_rate") or 0.0)
        nominal_force = float(nominal.get("force_contact_rate") or 0.0)
        policy_force = float(policy.get("force_contact_rate") or 0.0)
        rollout_duration = float(train.get("contact_duration_mean") or 0.0)
        nominal_duration = float(nominal.get("contact_duration_mean") or 0.0)
        policy_duration = float(policy.get("contact_duration_mean") or 0.0)
        train_stability = float(train.get("contact_stability_score") or 0.0)
        nominal_stability = float(nominal.get("contact_stability_score") or 0.0)
        policy_stability = float(policy.get("contact_stability_score") or 0.0)
        v86_force = float(v86.get("rollout_force_contact_rate") or v86.get("force_contact_rate") or 0.0)
        v86_stability = v86_force
        improved_vs_nominal = bool(policy_stability > nominal_stability or train_stability > nominal_stability)
        improved_vs_v86 = bool(v86_missing or policy_stability > v86_stability or train_stability > v86_stability)
        runner_ok = bool(train.get("runner_policy_artifact_complete") or train.get("runner_smoke_complete"))
        safety_ok = not bool(train.get("object_write_by_policy_detected") or policy.get("object_write_by_policy_detected")) and not bool(
            train.get("sticky_action_available_to_policy") or policy.get("sticky_action_available_to_policy")
        )
        grasp_success = bool(policy.get("grasp_success_claimed"))
        engineering_pass = bool(runner_ok and safety_ok and (improved_vs_nominal or improved_vs_v86))
        if grasp_success:
            status = "GRASP_EVAL_PASS_NO_STICKY"
            blocker = ""
            next_action = "review_support_hold_lift_evidence_before_any_later_final_stage"
        elif engineering_pass:
            status = "V87_ENGINEERING_PASS_NO_GRASP_SUCCESS"
            blocker = str(policy.get("blocker") or "deterministic_eval_no_support_hold_lift_success")
            next_action = "inspect_v87_policy_eval_and_reward_before_longer_training"
        elif runner_ok and safety_ok:
            status = "PPO_RAN_NO_STABILITY_IMPROVEMENT"
            blocker = "contact_stability_not_improved_over_nominal_or_v86_baseline"
            next_action = "adjust_contact_guided_reward_or_residual_scales"
        elif not bool(preflight.get("v86_contact_gate_pass")):
            status = "TRAINING_NOT_STARTED"
            blocker = str(preflight.get("blocker") or "controlled_force_contact_gate_failed")
            next_action = "repair_v85_contact_gate_regression"
        else:
            status = str(train.get("status") or "V87_PPO_FAILED")
            blocker = str(train.get("blocker") or policy.get("blocker") or "v87_runner_or_backend_safety_gate_failed")
            next_action = "fix_v87_runner_or_backend_safety_gate"
        matrix_rows.append(
            {
                "part_name": part,
                "object_identity_verified": bool(preflight.get("object_identity_verified")),
                "single_simulation_context": bool(preflight.get("single_simulation_context")),
                "v85_contact_gate_pass": bool(preflight.get("v86_contact_gate_pass")),
                "rollout_env_count": int(train.get("rollout_env_count") or 0),
                "rollout_participation_count": int(train.get("rollout_participation_count") or 0),
                "rollout_force_contact_rate": rollout_force,
                "nominal_force_contact_rate": nominal_force,
                "policy_eval_force_contact_rate": policy_force,
                "force_contact_rate": max(rollout_force, policy_force),
                "contact_duration_mean": max(rollout_duration, policy_duration),
                "nominal_contact_duration_mean": nominal_duration,
                "policy_eval_contact_duration_mean": policy_duration,
                "mean_force_n": max(float(train.get("mean_force_n") or train.get("force_contact_mean_n") or 0.0), float(policy.get("mean_force_n") or 0.0)),
                "peak_force_n": max(float(train.get("peak_force_n") or train.get("force_contact_peak_n") or 0.0), float(policy.get("peak_force_n") or 0.0)),
                "excessive_force_rate": max(float(train.get("excessive_force_rate") or 0.0), float(policy.get("excessive_force_rate") or 0.0)),
                "support_gate_rate": float(policy.get("support_gate_rate") or train.get("support_gate_rate") or 0.0),
                "hold_gate_rate": float(policy.get("hold_gate_rate") or train.get("hold_gate_rate") or 0.0),
                "lift_gate_rate": float(policy.get("lift_gate_rate") or train.get("lift_gate_rate") or 0.0),
                "object_displacement_max_m": max(
                    float(train.get("object_displacement_max_m") or 0.0),
                    float(policy.get("object_displacement_max_m") or 0.0),
                ),
                "deterministic_eval_result": str(policy.get("status") or ""),
                "v86_baseline_force_contact_rate": v86_force,
                "v86_baseline_missing": bool(v86_missing),
                "contact_stability_score": max(train_stability, policy_stability),
                "nominal_contact_stability_score": nominal_stability,
                "v86_contact_stability_score": v86_stability,
                "contact_stability_improved_vs_nominal": improved_vs_nominal,
                "contact_stability_improved_vs_v86": improved_vs_v86,
                "object_write_by_policy_detected": bool(train.get("object_write_by_policy_detected") or policy.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": bool(train.get("sticky_action_available_to_policy") or policy.get("sticky_action_available_to_policy")),
                "ppo_ran": bool(train.get("ppo_ran")),
                "rsl_rl_runner_used": bool(train.get("rsl_rl_runner_used")),
                "runner_policy_artifact_complete": runner_ok,
                "rl_trained": False,
                "rl_trained_success_evidence": grasp_success,
                "grasp_success_claimed": grasp_success,
                "checkpoint_written": bool(train.get("checkpoint_written")),
                "checkpoint_is_policy_artifact_only": bool(train.get("checkpoint_is_policy_artifact_only")),
                "checkpoint_path": str(train.get("checkpoint_path") or ""),
                "ppo_iterations_completed": int(train.get("ppo_iterations_completed") or 0),
                "rollout_step_count": int(train.get("rollout_step_count") or 0),
                "training_curve_nonempty": bool(train.get("training_curve_nonempty")),
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "engineering_pass": engineering_pass,
                "status": status,
                "blocker": blocker,
                "next_action": next_action,
            }
        )
    csv_path = write_csv(run_dir / "v87_progress_matrix.csv", matrix_rows, fields)
    md_path = run_dir / "v87_progress_matrix.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in matrix_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    root_csv = write_csv(REPO_ROOT / "debug_runs/v87_progress_matrix.csv", matrix_rows, fields)
    root_md = REPO_ROOT / "debug_runs/v87_progress_matrix.md"
    root_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "rows": matrix_rows,
        "v87_progress_csv": str(csv_path),
        "v87_progress_md": str(md_path),
        "v87_root_progress_csv": str(root_csv),
        "v87_root_progress_md": str(root_md),
    }


def _write_v88_progress_matrix(
    run_dir: Path,
    *,
    parts: list[str],
    branch_results: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = [
        "part_name",
        "best_branch",
        "physics_profile",
        "object_identity_verified",
        "single_simulation_context",
        "v85_contact_gate_pass",
        "rollout_participation_count",
        "rollout_force_contact_rate",
        "nominal_force_contact_rate",
        "policy_eval_force_contact_rate",
        "force_contact_rate",
        "contact_duration_mean",
        "mean_force_n",
        "peak_force_n",
        "excessive_force_rate",
        "support_gate_rate",
        "hold_gate_rate",
        "lift_gate_rate",
        "object_displacement_max_m",
        "hold_object_displacement_max_m",
        "lift_delta_z_max_m",
        "deterministic_eval_result",
        "v87_baseline_force_contact_rate",
        "v87_baseline_support_gate_rate",
        "v87_baseline_hold_gate_rate",
        "v87_baseline_lift_gate_rate",
        "v87_baseline_hold_displacement_m",
        "v87_baseline_missing",
        "contact_stability_score",
        "nominal_contact_stability_score",
        "v87_contact_stability_score",
        "contact_stability_improved_vs_nominal",
        "contact_stability_improved_vs_v87",
        "hold_displacement_improved_vs_v87",
        "object_write_by_policy_detected",
        "sticky_action_available_to_policy",
        "fallback_success_used",
        "ppo_ran",
        "rsl_rl_runner_used",
        "runner_policy_artifact_complete",
        "rl_trained",
        "rl_trained_success_evidence",
        "grasp_progress",
        "grasp_success_claimed",
        "final_success",
        "checkpoint_written",
        "checkpoint_is_policy_artifact_only",
        "checkpoint_path",
        "ppo_iterations_completed",
        "rollout_step_count",
        "training_curve_nonempty",
        "sticky_eval_ran",
        "final_replay_ran",
        "video_generated",
        "usable_training_row_count",
        "engineering_pass",
        "status",
        "blocker",
        "next_action",
    ]
    v87_rows, v87_missing = _read_v87_baseline_rows(parts)
    matrix_rows: list[dict[str, Any]] = []
    branch_part_rows: list[dict[str, Any]] = []
    for branch in branch_results:
        branch_name = str(branch.get("branch_name") or "")
        physics_profile = str(branch.get("physics_profile") or "")
        preflight_rows = branch.get("preflight_rows", [])
        train_rows = branch.get("train_rows", [])
        nominal_rows = branch.get("nominal_eval_rows", [])
        policy_rows = branch.get("policy_eval_rows", [])
        for part in parts:
            preflight = next((row for row in preflight_rows if row.get("part_name") == part), {})
            train = next((row for row in train_rows if row.get("part_name") == part), {})
            nominal = next((row for row in nominal_rows if row.get("part_name") == part), {})
            policy = next((row for row in policy_rows if row.get("part_name") == part), {})
            train_stability = float(train.get("contact_stability_score") or 0.0)
            policy_stability = float(policy.get("contact_stability_score") or 0.0)
            nominal_stability = float(nominal.get("contact_stability_score") or 0.0)
            row = {
                "part_name": part,
                "best_branch": branch_name,
                "physics_profile": physics_profile,
                "object_identity_verified": bool(preflight.get("object_identity_verified")),
                "single_simulation_context": bool(preflight.get("single_simulation_context")),
                "v85_contact_gate_pass": bool(preflight.get("v86_contact_gate_pass")),
                "rollout_participation_count": int(train.get("rollout_participation_count") or 0),
                "rollout_force_contact_rate": float(train.get("force_contact_rate") or 0.0),
                "nominal_force_contact_rate": float(nominal.get("force_contact_rate") or 0.0),
                "policy_eval_force_contact_rate": float(policy.get("force_contact_rate") or 0.0),
                "force_contact_rate": max(float(train.get("force_contact_rate") or 0.0), float(policy.get("force_contact_rate") or 0.0)),
                "contact_duration_mean": max(float(train.get("contact_duration_mean") or 0.0), float(policy.get("contact_duration_mean") or 0.0)),
                "mean_force_n": max(float(train.get("mean_force_n") or 0.0), float(policy.get("mean_force_n") or 0.0)),
                "peak_force_n": max(float(train.get("peak_force_n") or train.get("force_contact_peak_n") or 0.0), float(policy.get("peak_force_n") or 0.0)),
                "excessive_force_rate": max(float(train.get("excessive_force_rate") or 0.0), float(policy.get("excessive_force_rate") or 0.0)),
                "support_gate_rate": float(policy.get("support_gate_rate") or train.get("support_gate_rate") or 0.0),
                "hold_gate_rate": float(policy.get("hold_gate_rate") or train.get("hold_gate_rate") or 0.0),
                "lift_gate_rate": float(policy.get("lift_gate_rate") or train.get("lift_gate_rate") or 0.0),
                "object_displacement_max_m": max(float(train.get("object_displacement_max_m") or 0.0), float(policy.get("object_displacement_max_m") or 0.0)),
                "hold_object_displacement_max_m": max(float(train.get("hold_object_displacement_max_m") or 0.0), float(policy.get("hold_object_displacement_max_m") or 0.0)),
                "lift_delta_z_max_m": max(float(train.get("lift_delta_z_max_m") or 0.0), float(policy.get("lift_delta_z_max_m") or 0.0)),
                "deterministic_eval_result": str(policy.get("status") or ""),
                "contact_stability_score": max(train_stability, policy_stability),
                "nominal_contact_stability_score": nominal_stability,
                "object_write_by_policy_detected": bool(train.get("object_write_by_policy_detected") or policy.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": bool(train.get("sticky_action_available_to_policy") or policy.get("sticky_action_available_to_policy")),
                "fallback_success_used": bool(train.get("fallback_success_used") or policy.get("fallback_success_used")),
                "ppo_ran": bool(train.get("ppo_ran")),
                "rsl_rl_runner_used": bool(train.get("rsl_rl_runner_used")),
                "runner_policy_artifact_complete": bool(train.get("runner_policy_artifact_complete") or train.get("runner_smoke_complete")),
                "rl_trained": False,
                "rl_trained_success_evidence": bool(policy.get("grasp_success_claimed")),
                "grasp_success_claimed": bool(policy.get("grasp_success_claimed")),
                "final_success": False,
                "checkpoint_written": bool(train.get("checkpoint_written")),
                "checkpoint_is_policy_artifact_only": bool(train.get("checkpoint_is_policy_artifact_only")),
                "checkpoint_path": str(train.get("checkpoint_path") or ""),
                "ppo_iterations_completed": int(train.get("ppo_iterations_completed") or 0),
                "rollout_step_count": int(train.get("rollout_step_count") or 0),
                "training_curve_nonempty": bool(train.get("training_curve_nonempty")),
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "branch_blocker": str(train.get("blocker") or policy.get("blocker") or preflight.get("blocker") or ""),
                "branch_status": str(policy.get("status") or train.get("status") or ""),
            }
            branch_part_rows.append(row)
    for part in parts:
        candidates = [row for row in branch_part_rows if row.get("part_name") == part]
        if not candidates:
            candidates = [{"part_name": part, "best_branch": "", "branch_blocker": "no_v88_branch_candidate"}]
        v87 = v87_rows.get(part, {})
        v87_force = float(v87.get("force_contact_rate") or v87.get("policy_eval_force_contact_rate") or 0.0)
        v87_support = float(v87.get("support_gate_rate") or 0.0)
        v87_hold = float(v87.get("hold_gate_rate") or 0.0)
        v87_lift = float(v87.get("lift_gate_rate") or 0.0)
        v87_hold_disp = float(v87.get("hold_object_displacement_max_m") or v87.get("object_displacement_max_m") or 0.0)
        v87_stability = float(v87.get("contact_stability_score") or v87_force)

        def score(row: dict[str, Any]) -> tuple[Any, ...]:
            safety_ok = not bool(row.get("object_write_by_policy_detected") or row.get("sticky_action_available_to_policy") or row.get("fallback_success_used"))
            raw_disp = row.get("hold_object_displacement_max_m")
            hold_disp_for_score = 999.0 if raw_disp in ("", None) else float(raw_disp)
            return (
                int(bool(row.get("runner_policy_artifact_complete")) and safety_ok),
                int(bool(row.get("grasp_success_claimed"))),
                float(row.get("support_gate_rate") or 0.0),
                float(row.get("hold_gate_rate") or 0.0),
                float(row.get("lift_gate_rate") or 0.0),
                -hold_disp_for_score,
                float(row.get("contact_duration_mean") or 0.0),
            )

        best = max(candidates, key=score)
        improved_vs_nominal = bool(float(best.get("contact_stability_score") or 0.0) > float(best.get("nominal_contact_stability_score") or 0.0))
        improved_vs_v87 = bool(v87_missing or float(best.get("contact_stability_score") or 0.0) > v87_stability)
        hold_disp = float(best.get("hold_object_displacement_max_m") or 0.0)
        hold_improved = bool(v87_missing or (v87_hold_disp > 0.0 and hold_disp < v87_hold_disp))
        safety_ok = not bool(best.get("object_write_by_policy_detected") or best.get("sticky_action_available_to_policy") or best.get("fallback_success_used"))
        runner_ok = bool(best.get("runner_policy_artifact_complete"))
        grasp_success = bool(best.get("grasp_success_claimed"))
        grasp_progress = bool(
            (float(best.get("support_gate_rate") or 0.0) > v87_support)
            or (float(best.get("hold_gate_rate") or 0.0) > v87_hold)
            or (float(best.get("lift_gate_rate") or 0.0) > v87_lift)
            or hold_improved
        )
        engineering_pass = bool(runner_ok and safety_ok)
        if grasp_success:
            status = "GRASP_EVAL_PASS_NO_STICKY_NONFINAL"
            blocker = ""
            next_action = "review_v88_evidence_before_any_later_serial_replay_request"
        elif engineering_pass and (improved_vs_nominal or improved_vs_v87 or grasp_progress):
            status = "V88_ENGINEERING_PASS_WITH_GRASP_PROGRESS"
            blocker = str(best.get("branch_blocker") or "deterministic_eval_no_full_support_hold_lift_success")
            next_action = "inspect_best_branch_eval_and_consider_longer_nonfinal_training"
        elif engineering_pass:
            status = "V88_ENGINEERING_PASS_NO_GRASP_PROGRESS"
            blocker = "ppo_ran_safely_but_no_stability_or_hold_lift_improvement"
            next_action = "adjust_observations_physics_reward_or_nominal_lift_phase"
        elif not bool(best.get("v85_contact_gate_pass")):
            status = "TRAINING_NOT_STARTED"
            blocker = str(best.get("branch_blocker") or "controlled_force_contact_gate_failed")
            next_action = "repair_v85_contact_gate_regression"
        else:
            status = str(best.get("branch_status") or "V88_PPO_FAILED")
            blocker = str(best.get("branch_blocker") or "v88_branch_failed")
            next_action = "fix_v88_runner_or_backend_safety_gate"
        final = {field: best.get(field, "") for field in fields}
        final.update(
            {
                "v87_baseline_force_contact_rate": v87_force,
                "v87_baseline_support_gate_rate": v87_support,
                "v87_baseline_hold_gate_rate": v87_hold,
                "v87_baseline_lift_gate_rate": v87_lift,
                "v87_baseline_hold_displacement_m": v87_hold_disp,
                "v87_baseline_missing": bool(v87_missing),
                "v87_contact_stability_score": v87_stability,
                "contact_stability_improved_vs_nominal": improved_vs_nominal,
                "contact_stability_improved_vs_v87": improved_vs_v87,
                "hold_displacement_improved_vs_v87": hold_improved,
                "rl_trained": False,
                "rl_trained_success_evidence": grasp_success,
                "grasp_progress": grasp_progress,
                "grasp_success_claimed": grasp_success,
                "final_success": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "engineering_pass": engineering_pass,
                "status": status,
                "blocker": blocker,
                "next_action": next_action,
            }
        )
        matrix_rows.append(final)
    csv_path = write_csv(run_dir / "v88_progress_matrix.csv", matrix_rows, fields)
    md_path = run_dir / "v88_progress_matrix.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in matrix_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    root_csv = write_csv(REPO_ROOT / "debug_runs/v88_progress_matrix.csv", matrix_rows, fields)
    root_md = REPO_ROOT / "debug_runs/v88_progress_matrix.md"
    root_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    branch_csv = write_csv(run_dir / "v88_branch_candidate_matrix.csv", branch_part_rows)
    return {
        "rows": matrix_rows,
        "branch_rows": branch_part_rows,
        "v88_progress_csv": str(csv_path),
        "v88_progress_md": str(md_path),
        "v88_root_progress_csv": str(root_csv),
        "v88_root_progress_md": str(root_md),
        "v88_branch_candidate_matrix_csv": str(branch_csv),
    }


def _write_dynamic_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parts = _parts(args_cli.parts)
    run_dir = _run_dir()
    backend: IsaacUnifiedGraspPhysicalBackend | IsaacUnifiedSingleContextBackend | None = None
    try:
        metadata: dict[str, Any] = {
            "run_mode": args_cli.run_mode,
            "parts": parts,
            "repo_root": str(REPO_ROOT),
            "physics_profile": args_cli.physics_profile,
            "diagnostic_dynamics_allowed": bool(args_cli.allow_diagnostic_dynamics),
        }
        asset = audit_assets(run_dir, parts=parts, physics_profile=args_cli.physics_profile)
        randomization = write_object_randomization_report(run_dir, parts=parts)
        seeds = load_unified_seed_bank(
            run_dir,
            search_roots=[item.strip() for item in args_cli.program_prior_search_roots.split(",") if item.strip()],
            parts=parts,
        )
        priors = write_grasp_prior_bank(run_dir, parts=parts, seeds=seeds["rows"], top_k_per_part=args_cli.top_k_priors_per_part)
        curriculum = write_curriculum_artifacts(run_dir, seeds=seeds["rows"], priors=priors["rows"], parts=parts, physics_profile=args_cli.physics_profile)
        progress_rows = _foundation_progress(
            parts=parts,
            physics_profile=args_cli.physics_profile,
            seed_rows=seeds["rows"],
            prior_rows=priors["rows"],
            curriculum_rows=curriculum["rows"],
        )

        if args_cli.run_mode == "v95_minimal_unified_workcell_repair":
            previous_v93_repair_flag = os.environ.get("WUJI_V93_COLLISION_REPAIR")
            previous_v94_pad_flag = os.environ.get("WUJI_V94_DIAGNOSTIC_PADS")
            previous_v94_live_flag = os.environ.get("WUJI_V94_LIVE_PROBES")
            os.environ["WUJI_V93_COLLISION_REPAIR"] = "0"
            os.environ["WUJI_V94_DIAGNOSTIC_PADS"] = "0"
            os.environ["WUJI_V94_LIVE_PROBES"] = "1"
            code_freeze = write_v95_workcell_code_freeze(run_dir, repo_root=REPO_ROOT)
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
                auto_probe_reset_step=False,
            )
            try:
                gate = run_v95_minimal_workcell_gate(run_dir, backend, repo_root=REPO_ROOT, parts=parts)
                identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
                identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
                action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
                action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
                contact_csv = write_csv(run_dir / "contact_api_audit.csv", getattr(backend, "contact_api_rows", []) or [])
                contact_json = write_json(run_dir / "contact_api_audit.json", getattr(backend, "contact_api_rows", []) or [])
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                backend_summary = {
                    "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                    "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                    "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                    "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                    "object_write_reset_only": bool(getattr(backend, "object_write_reset_only", False)),
                    "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                    "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                    "proxy_action_available_to_policy": bool(getattr(backend, "proxy_action_available_to_policy", False)),
                    "route_selection_available_to_policy": bool(getattr(backend, "route_selection_available_to_policy", False)),
                    "auto_probe_reset_step": False,
                }
                ready_count = sum(
                    1
                    for row in gate.get("rows", [])
                    if row.get("v95_decision") == "WORKCELL_READY_FOR_FEASIBILITY_CONTROLLER"
                )
                metadata.update(
                    {
                        "asset": {k: v for k, v in asset.items() if k != "rows"},
                        "randomization": randomization,
                        "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                        "priors": {k: v for k, v in priors.items() if k != "rows"},
                        "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                        "code_freeze": {k: v for k, v in code_freeze.items() if k != "payload"},
                        "minimal_workcell_gate": {
                            k: v
                            for k, v in gate.items()
                            if k
                            not in {
                                "rows",
                                "plan_rows",
                                "active_rows",
                                "reset_lifecycle_rows",
                                "pregrasp_rows",
                                "pregrasp_calibration_trace_rows",
                                "wrist_rows",
                                "finger_rows",
                                "contact_rows",
                            }
                        },
                        "object_identity_audit_csv": str(identity_csv),
                        "object_identity_audit_json": str(identity_json),
                        "action_mapping_audit_csv": str(action_csv),
                        "action_mapping_audit_json": str(action_json),
                        "contact_api_audit_csv": str(contact_csv),
                        "contact_api_audit_json": str(contact_json),
                        "forbidden_artifact_scan": {k: v for k, v in gate.get("forbidden_artifact_scan", {}).items() if k != "rows"},
                        "source_archive": source,
                        **backend_summary,
                        "v95_live_probe_started": Path(gate.get("v95_live_probe_started_json", "")).exists(),
                        "v95_live_probe_completed": Path(gate.get("v95_live_probe_completed_json", "")).exists(),
                        "workcell_ready_object_count": ready_count,
                        "ppo_ran": False,
                        "bc_ran": False,
                        "rsl_rl_runner_used": False,
                        "runner_policy_artifact_complete": False,
                        "rl_trained": False,
                        "rl_trained_success_evidence": False,
                        "checkpoint_written": False,
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "dataset_exported": False,
                        "usable_training_row_count": 0,
                        "grasp_success_claimed": False,
                        "final_success": False,
                        "training_locked": ready_count < len(parts),
                        "training_locked_blocker": "" if ready_count == len(parts) else "v95_minimal_workcell_gate_not_ready_for_all_objects",
                    }
                )
                metadata_path = write_json(run_dir / "v95_run_metadata.json", metadata)
                print(f"v95 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {gate.get('v95_workcell_progress_matrix_csv')}")
                return 0
            finally:
                backend.close()
                if previous_v93_repair_flag is None:
                    os.environ.pop("WUJI_V93_COLLISION_REPAIR", None)
                else:
                    os.environ["WUJI_V93_COLLISION_REPAIR"] = previous_v93_repair_flag
                if previous_v94_pad_flag is None:
                    os.environ.pop("WUJI_V94_DIAGNOSTIC_PADS", None)
                else:
                    os.environ["WUJI_V94_DIAGNOSTIC_PADS"] = previous_v94_pad_flag
                if previous_v94_live_flag is None:
                    os.environ.pop("WUJI_V94_LIVE_PROBES", None)
                else:
                    os.environ["WUJI_V94_LIVE_PROBES"] = previous_v94_live_flag

        if args_cli.run_mode == "v94_safe_pregrasp_and_finger_contact_calibration":
            previous_v93_repair_flag = os.environ.get("WUJI_V93_COLLISION_REPAIR")
            previous_v94_pad_flag = os.environ.get("WUJI_V94_DIAGNOSTIC_PADS")
            os.environ["WUJI_V93_COLLISION_REPAIR"] = "1"
            os.environ["WUJI_V94_DIAGNOSTIC_PADS"] = "0"
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
            )
            try:
                reclassification = write_v94_v93_blocker_reclassification(run_dir, parts=parts)
                pregrasp = run_v94_pregrasp_alignment_audit(run_dir, backend)
                pregrasp_plan_rows = pregrasp.get("configured_rows", [])
                wrist = run_v94_wrist_action_calibration(run_dir, backend, pregrasp_plan_rows)
                finger = run_v94_finger_motion_calibration(run_dir, backend, pregrasp_plan_rows)
                sensor = run_v94_diagnostic_target_finger_sensor_map(run_dir, backend, pregrasp_plan_rows)
                actual_contact = run_v94_actual_object_contact_calibration(run_dir, backend, pregrasp_plan_rows, sensor.get("rows", []))
                proxy = write_v94_collision_proxy_decision(run_dir, parts=parts)
                identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
                identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
                action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
                action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
                contact_csv = write_csv(run_dir / "contact_api_audit.csv", getattr(backend, "contact_api_rows", []) or [])
                contact_json = write_json(run_dir / "contact_api_audit.json", getattr(backend, "contact_api_rows", []) or [])
                backend_summary = {
                    "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                    "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                    "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                    "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                    "object_write_reset_only": bool(getattr(backend, "object_write_reset_only", False)),
                    "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                    "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                }
                readiness = classify_v94_readiness(
                    run_dir,
                    parts=parts,
                    pregrasp_rows=pregrasp.get("rows", []),
                    wrist_rows=wrist.get("rows", []),
                    finger_rows=finger.get("rows", []),
                    sensor_rows=sensor.get("rows", []),
                    object_contact_rows=actual_contact.get("rows", []),
                    proxy_rows=proxy.get("rows", []),
                    backend_summary=backend_summary,
                )
                forbidden_scan = write_v94_forbidden_artifact_scan(run_dir)
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                metadata.update(
                    {
                        "asset": {k: v for k, v in asset.items() if k != "rows"},
                        "randomization": randomization,
                        "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                        "priors": {k: v for k, v in priors.items() if k != "rows"},
                        "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                        "v93_blocker_reclassification": {k: v for k, v in reclassification.items() if k != "rows"},
                        "pregrasp_alignment_audit": {k: v for k, v in pregrasp.items() if k not in {"rows", "configured_rows"}},
                        "wrist_action_calibration": {k: v for k, v in wrist.items() if k != "rows"},
                        "finger_motion_calibration": {k: v for k, v in finger.items() if k != "rows"},
                        "diagnostic_target_finger_sensor_map": {k: v for k, v in sensor.items() if k != "rows"},
                        "actual_object_contact_calibration": {k: v for k, v in actual_contact.items() if k != "rows"},
                        "collision_proxy_decision": {k: v for k, v in proxy.items() if k != "rows"},
                        "readiness": {k: v for k, v in readiness.items() if k != "rows"},
                        "object_identity_audit_csv": str(identity_csv),
                        "object_identity_audit_json": str(identity_json),
                        "action_mapping_audit_csv": str(action_csv),
                        "action_mapping_audit_json": str(action_json),
                        "contact_api_audit_csv": str(contact_csv),
                        "contact_api_audit_json": str(contact_json),
                        "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                        "source_archive": source,
                        **backend_summary,
                        "v94_live_probes_enabled": os.environ.get("WUJI_V94_LIVE_PROBES", "0").strip().lower()
                        in {"1", "true", "yes", "on"},
                        "v94_live_probe_default": "disabled_to_avoid_native_shutdown_until_reset_lifecycle_is_repaired",
                        "ppo_ran": False,
                        "bc_ran": False,
                        "rsl_rl_runner_used": False,
                        "runner_policy_artifact_complete": False,
                        "rl_trained": False,
                        "rl_trained_success_evidence": False,
                        "checkpoint_written": False,
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "dataset_exported": False,
                        "usable_training_row_count": 0,
                        "grasp_success_claimed": False,
                        "final_success": False,
                        "training_locked": True,
                        "training_locked_blocker": "v94_diagnostic_only_ready_for_v95_gate",
                    }
                )
                metadata_path = write_json(run_dir / "v94_run_metadata.json", metadata)
                print(f"v94 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {readiness.get('v94_progress_matrix_csv')}")
                return 0
            finally:
                backend.close()
                if previous_v93_repair_flag is None:
                    os.environ.pop("WUJI_V93_COLLISION_REPAIR", None)
                else:
                    os.environ["WUJI_V93_COLLISION_REPAIR"] = previous_v93_repair_flag
                if previous_v94_pad_flag is None:
                    os.environ.pop("WUJI_V94_DIAGNOSTIC_PADS", None)
                else:
                    os.environ["WUJI_V94_DIAGNOSTIC_PADS"] = previous_v94_pad_flag

        if args_cli.run_mode == "v93_collision_geometry_and_safe_staging_repair":
            previous_v93_repair_flag = os.environ.get("WUJI_V93_COLLISION_REPAIR")
            os.environ["WUJI_V93_COLLISION_REPAIR"] = "1"
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
            )
            try:
                native = run_v93_native_shutdown_root_cause(run_dir, backend)
                collision = run_v93_collision_geometry_audit_and_repair(run_dir, backend)
                staging_plan = write_v93_safe_staging_plan(run_dir, parts=parts)
                staging = run_v93_safe_staging_audit(run_dir, backend, staging_plan.get("rows", []))
                screw1 = run_v93_screw1_post_repair_sanity(run_dir, backend)
                contact_sanity = run_v93_post_collision_contact_sanity(run_dir, backend)
                support_gate = run_v93_support_gate_post_repair_audit(run_dir, backend, contact_sanity.get("rows", []))
                identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
                identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
                action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
                action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
                contact_csv = write_csv(run_dir / "contact_api_audit.csv", getattr(backend, "contact_api_rows", []) or [])
                contact_json = write_json(run_dir / "contact_api_audit.json", getattr(backend, "contact_api_rows", []) or [])
                backend_summary = {
                    "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                    "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                    "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                    "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                    "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                    "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                }
                readiness = classify_v93_readiness(
                    run_dir,
                    parts=parts,
                    native_rows=native.get("rows", []),
                    collision_rows=[row for row in collision.get("rows", []) if row.get("entity_type") == "object"],
                    staging_rows=staging.get("rows", []),
                    screw1_rows=screw1.get("rows", []),
                    support_rows=support_gate.get("rows", []),
                    backend_summary=backend_summary,
                )
                forbidden_scan = write_v93_forbidden_artifact_scan(run_dir)
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                metadata.update(
                    {
                        "asset": {k: v for k, v in asset.items() if k != "rows"},
                        "randomization": randomization,
                        "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                        "priors": {k: v for k, v in priors.items() if k != "rows"},
                        "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                        "native_shutdown_root_cause": {k: v for k, v in native.items() if k != "rows"},
                        "collision_geometry_audit": {k: v for k, v in collision.items() if k not in {"rows", "inventory_rows", "alignment_rows", "repair_rows"}},
                        "safe_staging_plan": {k: v for k, v in staging_plan.items() if k != "rows"},
                        "safe_staging_audit": {k: v for k, v in staging.items() if k not in {"rows", "precontact_rows"}},
                        "screw1_post_repair_sanity": {k: v for k, v in screw1.items() if k != "rows"},
                        "post_collision_contact_sanity": {k: v for k, v in contact_sanity.items() if k != "rows"},
                        "support_gate_post_repair": {k: v for k, v in support_gate.items() if k != "rows"},
                        "readiness": {k: v for k, v in readiness.items() if k != "rows"},
                        "object_identity_audit_csv": str(identity_csv),
                        "object_identity_audit_json": str(identity_json),
                        "action_mapping_audit_csv": str(action_csv),
                        "action_mapping_audit_json": str(action_json),
                        "contact_api_audit_csv": str(contact_csv),
                        "contact_api_audit_json": str(contact_json),
                        "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                        "source_archive": source,
                        **backend_summary,
                        "ppo_ran": False,
                        "bc_ran": False,
                        "rsl_rl_runner_used": False,
                        "runner_policy_artifact_complete": False,
                        "rl_trained": False,
                        "rl_trained_success_evidence": False,
                        "checkpoint_written": False,
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "dataset_exported": False,
                        "usable_training_row_count": 0,
                        "grasp_success_claimed": False,
                        "final_success": False,
                        "training_locked": True,
                        "training_locked_blocker": "v93_diagnostic_only_ready_for_v94_gate",
                    }
                )
                metadata_path = write_json(run_dir / "v93_run_metadata.json", metadata)
                print(f"v93 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {readiness.get('v93_progress_matrix_csv')}")
                return 0
            finally:
                backend.close()
                if previous_v93_repair_flag is None:
                    os.environ.pop("WUJI_V93_COLLISION_REPAIR", None)
                else:
                    os.environ["WUJI_V93_COLLISION_REPAIR"] = previous_v93_repair_flag

        if args_cli.run_mode == "v92_asset_task_setup_and_contact_calibration":
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
            )
            screw1_repair = run_v92_screw1_dynamic_repair_audit(run_dir, backend)
            collision = run_v92_collision_subtree_audit_and_repair(run_dir, backend)
            initial_plan = write_v92_task_initial_condition_plan(run_dir, parts=parts)
            staging_repair = run_v92_staging_repair_audit(run_dir, backend, initial_plan.get("rows", []))
            finger_calibration = run_v92_finger_action_sensor_calibration(run_dir, backend)
            support_gate = run_v92_support_gate_calibration(run_dir, backend, finger_calibration.get("mapping_rows", []))
            sanity_probe = run_v92_sanity_probe(run_dir, backend, finger_calibration.get("mapping_rows", []))
            identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
            identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
            action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
            action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
            contact_csv = write_csv(run_dir / "contact_api_audit.csv", getattr(backend, "contact_api_rows", []) or [])
            contact_json = write_json(run_dir / "contact_api_audit.json", getattr(backend, "contact_api_rows", []) or [])
            backend_summary = {
                "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
            }
            progress = classify_v92_progress(
                run_dir,
                parts=parts,
                asset_rows=screw1_repair.get("rows", []),
                collision_rows=[row for row in collision.get("rows", []) if row.get("entity_type") == "object"],
                staging_rows=staging_repair.get("rows", []),
                mapping_rows=finger_calibration.get("mapping_rows", []),
                sanity_rows=sanity_probe.get("rows", []),
                backend_summary=backend_summary,
            )
            forbidden_scan = write_v92_forbidden_artifact_scan(run_dir)
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "asset": {k: v for k, v in asset.items() if k != "rows"},
                    "randomization": randomization,
                    "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                    "priors": {k: v for k, v in priors.items() if k != "rows"},
                    "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                    "screw1_dynamic_repair": {k: v for k, v in screw1_repair.items() if k != "rows"},
                    "collision_subtree_audit": {k: v for k, v in collision.items() if k not in {"rows", "repair_rows"}},
                    "task_initial_condition_plan": {k: v for k, v in initial_plan.items() if k != "rows"},
                    "staging_repair_audit": {k: v for k, v in staging_repair.items() if k != "rows"},
                    "finger_action_sensor_calibration": {k: v for k, v in finger_calibration.items() if k not in {"rows", "mapping_rows"}},
                    "support_gate_calibration": {k: v for k, v in support_gate.items() if k != "rows"},
                    "sanity_probe": {k: v for k, v in sanity_probe.items() if k not in {"rows", "trace_rows"}},
                    "progress": {k: v for k, v in progress.items() if k != "rows"},
                    "object_identity_audit_csv": str(identity_csv),
                    "object_identity_audit_json": str(identity_json),
                    "action_mapping_audit_csv": str(action_csv),
                    "action_mapping_audit_json": str(action_json),
                    "contact_api_audit_csv": str(contact_csv),
                    "contact_api_audit_json": str(contact_json),
                    "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                    "source_archive": source,
                    **backend_summary,
                    "ppo_ran": False,
                    "bc_ran": False,
                    "rsl_rl_runner_used": False,
                    "runner_policy_artifact_complete": False,
                    "rl_trained": False,
                    "rl_trained_success_evidence": False,
                    "checkpoint_written": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "dataset_exported": False,
                    "usable_training_row_count": 0,
                    "grasp_success_claimed": False,
                    "final_success": False,
                    "training_locked": True,
                    "training_locked_blocker": "v92_diagnostic_only_no_training_rows",
                }
            )
            metadata_path = write_json(run_dir / "v92_run_metadata.json", metadata)
            backend.close()
            print(f"v92 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {progress.get('v92_progress_matrix_csv')}")
            return 0

        if args_cli.run_mode == "v91_task_setup_asset_dynamics_and_controller_sanity":
            candidate_plan = write_v91_feasibility_candidate_plan(run_dir)
            candidate_rows = [row for row in candidate_plan.get("rows", []) if row.get("part_name") in parts]
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
            )
            observation_audit = run_v91_observation_runtime_audit(run_dir, backend)
            asset_dynamics = run_v91_asset_dynamics_audit(run_dir, backend, candidate_rows)
            staging_sanity = run_v91_staging_sanity_bench(run_dir, backend, candidate_rows)
            feasibility = run_v91_quasistatic_feasibility_bench(
                run_dir,
                backend,
                candidate_rows,
                staging_sanity.get("rows", []),
            )
            identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
            identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
            action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
            action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
            contact_csv = write_csv(run_dir / "contact_api_audit.csv", getattr(backend, "contact_api_rows", []) or [])
            contact_json = write_json(run_dir / "contact_api_audit.json", getattr(backend, "contact_api_rows", []) or [])
            decisions = classify_v91_decisions(
                run_dir,
                asset_rows=asset_dynamics.get("rows", []),
                staging_rows=staging_sanity.get("rows", []),
                candidate_rows=feasibility.get("rows", []),
            )
            backend_summary = {
                "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
            }
            progress = write_v91_progress_matrix(
                run_dir,
                decision_rows=decisions.get("rows", []),
                backend_summary=backend_summary,
            )
            forbidden_scan = write_v91_forbidden_artifact_scan(run_dir)
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "asset": {k: v for k, v in asset.items() if k != "rows"},
                    "randomization": randomization,
                    "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                    "priors": {k: v for k, v in priors.items() if k != "rows"},
                    "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                    "candidate_plan": {k: v for k, v in candidate_plan.items() if k != "rows"},
                    "observation_runtime_audit": {k: v for k, v in observation_audit.items() if k != "rows"},
                    "asset_dynamics_audit": {k: v for k, v in asset_dynamics.items() if k not in {"rows", "trace_rows"}},
                    "staging_sanity_audit": {k: v for k, v in staging_sanity.items() if k != "rows"},
                    "feasibility_bench": {k: v for k, v in feasibility.items() if k not in {"rows", "best_rows", "trace_rows", "finger_rows"}},
                    "per_object_decision": {k: v for k, v in decisions.items() if k != "rows"},
                    "progress": {k: v for k, v in progress.items() if k != "rows"},
                    "object_identity_audit_csv": str(identity_csv),
                    "object_identity_audit_json": str(identity_json),
                    "action_mapping_audit_csv": str(action_csv),
                    "action_mapping_audit_json": str(action_json),
                    "contact_api_audit_csv": str(contact_csv),
                    "contact_api_audit_json": str(contact_json),
                    "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                    "source_archive": source,
                    **backend_summary,
                    "ppo_ran": False,
                    "bc_ran": False,
                    "rsl_rl_runner_used": False,
                    "runner_policy_artifact_complete": False,
                    "rl_trained": False,
                    "rl_trained_success_evidence": False,
                    "checkpoint_written": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "dataset_exported": False,
                    "usable_training_row_count": 0,
                    "final_success": False,
                    "training_locked": True,
                    "training_locked_blocker": "v91_diagnostic_only_no_training_rows",
                }
            )
            metadata_path = write_json(run_dir / "v91_run_metadata.json", metadata)
            print(f"v91 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {progress.get('v91_progress_matrix_csv')}")
            return 0

        if args_cli.run_mode == "v90_bottleneck_isolation_and_grasp_feasibility":
            touched_files = [
                "scripts/environments/run_v81_physical_backend_grasp_rl.py",
                "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/single_context_backend.py",
                "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/v90_bottleneck_isolation.py",
            ]
            disk_preflight = write_v90_disk_preflight(run_dir, repo_root=REPO_ROOT, heavy_feasibility_ran=False)
            code_freeze = write_v90_code_freeze_report(run_dir, repo_root=REPO_ROOT, touched_files=touched_files)
            candidate_plan = write_v90_feasibility_candidate_plan(run_dir)
            candidate_rows = [row for row in candidate_plan.get("rows", []) if row.get("part_name") in parts]
            disk_space_ok = bool(disk_preflight.get("disk_space_ok_for_heavy_feasibility"))
            feasibility: dict[str, Any] = {
                "rows": [],
                "best_rows": [],
                "trace_rows": [],
                "stage_rows": [],
            }
            heavy_feasibility_ran = False
            if disk_space_ok:
                backend = IsaacUnifiedSingleContextBackend(
                    parts=parts,
                    num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                    physics_profile="canonical",
                    device=args_cli.device,
                    create_env=True,
                )
                feasibility = run_v90_feasibility_bench(
                    run_dir,
                    backend,
                    candidate_rows,
                    artifact_prefix="v90",
                    physics_profile="canonical",
                )
                heavy_feasibility_ran = bool(feasibility.get("rows"))
                identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
                identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
                action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
                action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
                contact_csv = write_csv(run_dir / "contact_api_audit.csv", getattr(backend, "contact_api_rows", []) or [])
                contact_json = write_json(run_dir / "contact_api_audit.json", getattr(backend, "contact_api_rows", []) or [])
            else:
                identity_csv = write_csv(run_dir / "object_identity_audit.csv", [])
                identity_json = write_json(run_dir / "object_identity_audit.json", [])
                action_csv = write_csv(run_dir / "action_mapping_audit.csv", [])
                action_json = write_json(run_dir / "action_mapping_audit.json", [])
                contact_csv = write_csv(run_dir / "contact_api_audit.csv", [])
                contact_json = write_json(run_dir / "contact_api_audit.json", [])
                write_csv(run_dir / "v90_feasibility_candidate_results.csv", [])
                write_json(run_dir / "v90_feasibility_candidate_results.json", [])
                write_csv(run_dir / "v90_best_candidate_trace_summary.csv", [])
                write_json(run_dir / "v90_best_candidate_trace_summary.json", [])

            physics_report = write_v90_physics_asset_bottleneck_report(
                run_dir,
                canonical_rows=feasibility.get("rows", []),
                diagnostic_rows=[],
                tuning_rows=getattr(backend, "v88_physics_tuning_rows", []) if backend is not None else [],
            )
            diagnosis = classify_v90_bottlenecks(
                run_dir,
                candidate_rows=feasibility.get("rows", []),
                physics_rows=physics_report.get("rows", []),
                repo_root=REPO_ROOT,
            )
            backend_summary = {
                "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                "gym_make_count": int(getattr(backend, "gym_make_count", 0)) if backend is not None else 0,
                "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
            }
            progress = write_v90_progress_matrix(
                run_dir,
                diagnosis_rows=diagnosis.get("rows", []),
                candidate_rows=feasibility.get("rows", []),
                backend_summary=backend_summary,
            )
            disk_preflight = write_v90_disk_preflight(
                run_dir,
                repo_root=REPO_ROOT,
                heavy_feasibility_ran=heavy_feasibility_ran,
            )
            forbidden_scan = write_v90_forbidden_artifact_scan(run_dir)
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "asset": {k: v for k, v in asset.items() if k != "rows"},
                    "randomization": randomization,
                    "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                    "priors": {k: v for k, v in priors.items() if k != "rows"},
                    "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                    "disk_preflight": {k: v for k, v in disk_preflight.items() if k != "cleanup_candidates"},
                    "code_freeze_report": {k: v for k, v in code_freeze.items() if k != "rows"},
                    "candidate_plan": {k: v for k, v in candidate_plan.items() if k != "rows"},
                    "feasibility_bench": {
                        k: v
                        for k, v in feasibility.items()
                        if k not in {"rows", "best_rows", "trace_rows", "stage_rows"}
                    },
                    "physics_asset_bottleneck_report": {k: v for k, v in physics_report.items() if k != "rows"},
                    "per_object_bottleneck_diagnosis": {k: v for k, v in diagnosis.items() if k != "rows"},
                    "progress": {k: v for k, v in progress.items() if k != "rows"},
                    "object_identity_audit_csv": str(identity_csv),
                    "object_identity_audit_json": str(identity_json),
                    "action_mapping_audit_csv": str(action_csv),
                    "action_mapping_audit_json": str(action_json),
                    "contact_api_audit_csv": str(contact_csv),
                    "contact_api_audit_json": str(contact_json),
                    "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                    "source_archive": source,
                    **backend_summary,
                    "disk_space_ok_for_heavy_feasibility": disk_space_ok,
                    "heavy_feasibility_ran": heavy_feasibility_ran,
                    "ppo_ran": False,
                    "rsl_rl_runner_used": False,
                    "runner_policy_artifact_complete": False,
                    "rl_trained": False,
                    "rl_trained_success_evidence": False,
                    "checkpoint_written": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "dataset_exported": False,
                    "usable_training_row_count": 0,
                    "final_success": False,
                    "training_locked": True,
                    "training_locked_blocker": "v90_diagnostic_only_no_training_rows",
                }
            )
            metadata_path = write_json(run_dir / "v90_run_metadata.json", metadata)
            print(f"v90 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {progress.get('v90_progress_matrix_csv')}")
            return 0

        if args_cli.run_mode == "v89_failure_driven_hybrid_grasp_repair":
            v89_branches = [
                item.strip()
                for item in str(args_cli.v89_branches or "").split(",")
                if item.strip() in set(V89_BRANCHES)
            ] or list(V89_BRANCHES)
            v89_max_steps = args_cli.max_steps if int(args_cli.max_steps) != 32 else 128
            v89_iterations = args_cli.ppo_iterations if int(args_cli.ppo_iterations) != 5 else 20
            free_bytes = shutil.disk_usage(run_dir.parent).free
            disk_space_ok = free_bytes >= 15 * 1024 * 1024 * 1024
            failure_diagnosis = write_v89_failure_diagnosis(run_dir, repo_root=REPO_ROOT)
            physics_decision = write_v89_physics_decision(run_dir, repo_root=REPO_ROOT)
            candidate_plan = write_v89_nominal_candidate_plan(run_dir)
            v88_artifacts = load_v88_artifacts(REPO_ROOT)
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
            )
            candidate_probe = run_v89_candidate_probe(run_dir, backend, candidate_plan.get("rows", []))
            selected_candidates = select_best_candidates(candidate_probe.get("rows", []))
            selected_json = write_json(run_dir / "v89_selected_nominal_candidates.json", selected_candidates)
            bc_candidate_rows = valid_bc_rows(candidate_probe.get("rows", []))
            bc_rows_json = write_json(run_dir / "v89_valid_bc_trace_sources.json", bc_candidate_rows)
            branch_results: list[dict[str, Any]] = []
            all_train_rows: list[dict[str, Any]] = []
            all_eval_rows: list[dict[str, Any]] = []
            all_nominal_rows: list[dict[str, Any]] = []
            all_preflight_rows: list[dict[str, Any]] = []
            branch_metadata: list[dict[str, Any]] = []
            for branch_name in v89_branches:
                branch_dir = ensure_run_dir(run_dir / "branches" / branch_name)
                use_candidate_prior = branch_name in {"nominal_candidate_residual_ppo", "bc_warmstart_residual_ppo"}
                warmstart_path = ""
                if branch_name == "baseline_v88_best_continuation":
                    candidate = (
                        REPO_ROOT
                        / "debug_runs/v88_stabilized_grasp_policy_and_physics_audit/branches/v87_baseline_continuation/v88_v87_baseline_continuation_policy_artifact_checkpoint.pt"
                    )
                    warmstart_path = str(candidate) if candidate.exists() else ""
                preflight = run_v86_contact_gate_preflight(branch_dir, backend)
                preflight_rows = [
                    {**row, "branch_name": branch_name, "physics_profile": "canonical"}
                    for row in preflight.get("rows", [])
                ]
                write_csv(branch_dir / "v89_contact_gate_preflight.csv", preflight_rows)
                write_json(branch_dir / "v89_contact_gate_preflight.json", preflight_rows)
                all_preflight_rows.extend(preflight_rows)
                train_result = train_v89_hybrid_residual_ppo(
                    branch_dir,
                    branch_name=branch_name,
                    policy_mode=args_cli.policy_mode,
                    parts=parts,
                    num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                    max_steps=v89_max_steps,
                    physics_profile="canonical",
                    physical_backend=backend,
                    ppo_iterations=v89_iterations,
                    contact_gate_rows=preflight.get("rows", []),
                    use_candidate_prior=use_candidate_prior,
                    selected_candidates=selected_candidates,
                    bc_rows=bc_candidate_rows,
                    bc_epochs=args_cli.v89_bc_epochs,
                    resume_checkpoint_path=warmstart_path,
                    disk_space_ok=disk_space_ok,
                )
                eval_result: dict[str, Any] = {"rows": [], "nominal_rows": []}
                if bool(train_result.get("runner_policy_artifact_complete")) and str(train_result.get("checkpoint_path") or ""):
                    eval_result = evaluate_v89_hybrid_policy(
                        branch_dir,
                        branch_name=branch_name,
                        checkpoint_path=str(train_result.get("checkpoint_path") or ""),
                        parts=parts,
                        eval_step_count=max(1, int(v89_max_steps)),
                        physical_backend=backend,
                        physics_profile="canonical",
                        use_candidate_prior=use_candidate_prior,
                        selected_candidates=selected_candidates,
                    )
                train_rows = train_result.get("rows", [])
                eval_rows = eval_result.get("rows", [])
                all_train_rows.extend(train_rows)
                all_eval_rows.extend(eval_rows)
                all_nominal_rows.extend(eval_result.get("nominal_rows", []))
                branch_results.append(
                    {
                        "branch_name": branch_name,
                        "physics_profile": "canonical",
                        "use_candidate_prior": use_candidate_prior,
                        "train_rows": train_rows,
                        "eval_rows": eval_rows,
                        "nominal_eval_rows": eval_result.get("nominal_rows", []),
                    }
                )
                branch_metadata.append(
                    {
                        "branch_name": branch_name,
                        "physics_profile": "canonical",
                        "use_candidate_prior": use_candidate_prior,
                        "warmstart_path": warmstart_path,
                        "preflight_all_parts_pass": bool(preflight.get("all_parts_pass")),
                        "candidate_prior_count": len(selected_candidates) if use_candidate_prior else 0,
                        "bc_valid_row_count": len(bc_candidate_rows),
                        "ppo_ran": bool(train_result.get("ppo_ran")),
                        "runner_policy_artifact_complete": bool(train_result.get("runner_policy_artifact_complete")),
                        "checkpoint_path": str(train_result.get("checkpoint_path") or ""),
                        "grasp_success_claimed_any_part": any(bool(row.get("grasp_success_claimed")) for row in eval_rows),
                        "blocker": str(train_result.get("failure_reason") or ""),
                    }
                )

            training_summary_csv = write_csv(run_dir / "v89_training_summary.csv", all_train_rows)
            training_summary_json = write_json(run_dir / "v89_training_summary.json", all_train_rows)
            eval_summary_csv = write_csv(run_dir / "v89_deterministic_eval_no_sticky.csv", all_eval_rows)
            eval_summary_json = write_json(run_dir / "v89_deterministic_eval_no_sticky.json", all_eval_rows)
            nominal_summary_csv = write_csv(run_dir / "v89_nominal_eval_summary.csv", all_nominal_rows)
            nominal_summary_json = write_json(run_dir / "v89_nominal_eval_summary.json", all_nominal_rows)
            preflight_csv = write_csv(run_dir / "v89_contact_gate_preflight.csv", all_preflight_rows)
            preflight_json = write_json(run_dir / "v89_contact_gate_preflight.json", all_preflight_rows)
            identity_csv = write_csv(run_dir / "object_identity_audit.csv", getattr(backend, "object_identity_rows", []) or [])
            identity_json = write_json(run_dir / "object_identity_audit.json", getattr(backend, "object_identity_rows", []) or [])
            action_csv = write_csv(run_dir / "action_mapping_audit.csv", getattr(backend, "action_mapping_rows", []) or [])
            action_json = write_json(run_dir / "action_mapping_audit.json", getattr(backend, "action_mapping_rows", []) or [])
            progress = write_v89_progress_matrix(
                run_dir,
                branch_results=branch_results,
                v88_progress_rows=v88_artifacts.get("progress", []),
            )
            code_growth = write_v89_code_growth_report(
                run_dir,
                touched_files=[
                    "scripts/environments/run_v81_physical_backend_grasp_rl.py",
                    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/single_context_backend.py",
                    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/unified_grasp_env.py",
                    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/train_unified_rl.py",
                    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/eval_unified_rl.py",
                    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/v89_hybrid_repair.py",
                ],
                used_modes=[args_cli.run_mode, *v89_branches],
                unused_modes=["sticky_eval", "final_replay", "video_generation", "dataset_export"],
            )
            forbidden_scan = write_v89_forbidden_artifact_scan(run_dir)
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "asset": {k: v for k, v in asset.items() if k != "rows"},
                    "randomization": randomization,
                    "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                    "priors": {k: v for k, v in priors.items() if k != "rows"},
                    "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                    "disk_space_free_gb": free_bytes / (1024 * 1024 * 1024),
                    "disk_space_ok_for_ppo_branches": bool(disk_space_ok),
                    "failure_diagnosis": {k: v for k, v in failure_diagnosis.items() if k != "rows"},
                    "physics_decision": {k: v for k, v in physics_decision.items() if k != "rows"},
                    "candidate_plan": {k: v for k, v in candidate_plan.items() if k != "rows"},
                    "candidate_probe": {k: v for k, v in candidate_probe.items() if k not in {"rows", "trace_rows", "stage_rows"}},
                    "selected_candidates_json": str(selected_json),
                    "valid_bc_trace_sources_json": str(bc_rows_json),
                    "branches": branch_metadata,
                    "training_summary_csv": str(training_summary_csv),
                    "training_summary_json": str(training_summary_json),
                    "deterministic_eval_no_sticky_csv": str(eval_summary_csv),
                    "deterministic_eval_no_sticky_json": str(eval_summary_json),
                    "nominal_eval_summary_csv": str(nominal_summary_csv),
                    "nominal_eval_summary_json": str(nominal_summary_json),
                    "contact_gate_preflight_csv": str(preflight_csv),
                    "contact_gate_preflight_json": str(preflight_json),
                    "object_identity_audit_csv": str(identity_csv),
                    "object_identity_audit_json": str(identity_json),
                    "action_mapping_audit_csv": str(action_csv),
                    "action_mapping_audit_json": str(action_json),
                    "progress": {k: v for k, v in progress.items() if k != "rows"},
                    "code_growth_report": {k: v for k, v in code_growth.items() if k != "rows"},
                    "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                    "source_archive": source,
                    "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                    "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                    "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                    "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                    "ppo_ran": any(bool(row.get("ppo_ran")) for row in all_train_rows),
                    "rsl_rl_runner_used": any(bool(row.get("rsl_rl_runner_used")) for row in all_train_rows),
                    "runner_policy_artifact_complete": any(bool(row.get("runner_policy_artifact_complete")) for row in all_train_rows),
                    "rl_trained": False,
                    "rl_trained_success_evidence": any(bool(row.get("grasp_success_claimed")) for row in all_eval_rows),
                    "checkpoint_written": any(bool(row.get("checkpoint_written")) for row in all_train_rows),
                    "checkpoint_is_policy_artifact_only": True,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "usable_training_row_count": 0,
                    "final_success": False,
                    "training_locked": True,
                    "training_locked_blocker": "v89_nonfinal_policy_artifact_only_no_training_export",
                    "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                    "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                }
            )
            metadata_path = write_json(run_dir / "v89_run_metadata.json", metadata)
            print(f"v89 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {progress.get('v89_progress_matrix_csv')}")
            return 0

        if args_cli.run_mode == "v88_stabilized_grasp_policy_and_physics_audit":
            v88_branches = [
                item.strip()
                for item in str(args_cli.v88_branches or "").split(",")
                if item.strip() in {"v87_baseline_continuation", "physics_tuned_residual_ppo"}
            ] or ["v87_baseline_continuation", "physics_tuned_residual_ppo"]
            v88_max_steps = args_cli.max_steps if int(args_cli.max_steps) != 32 else 128
            v88_iterations = args_cli.ppo_iterations if int(args_cli.ppo_iterations) != 5 else 20
            code_audit = write_v88_code_path_audit(run_dir, REPO_ROOT, branch_names=v88_branches)
            branch_results: list[dict[str, Any]] = []
            all_train_rows: list[dict[str, Any]] = []
            all_eval_rows: list[dict[str, Any]] = []
            all_nominal_rows: list[dict[str, Any]] = []
            all_preflight_rows: list[dict[str, Any]] = []
            all_physics_rows: list[dict[str, Any]] = []
            all_observation_feature_rows: list[dict[str, Any]] = []
            branch_metadata: list[dict[str, Any]] = []
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile="canonical",
                device=args_cli.device,
                create_env=True,
            )
            for branch_name in v88_branches:
                branch_dir = ensure_run_dir(run_dir / "branches" / branch_name)
                branch_profile = "conservative_contact" if branch_name == "physics_tuned_residual_ppo" else "canonical"
                enhanced_observation = branch_name == "physics_tuned_residual_ppo"
                warmstart_path = ""
                if branch_name == "v87_baseline_continuation":
                    candidate = REPO_ROOT / "debug_runs/v87_contact_guided_residual_ppo/v87_policy_artifact_checkpoint.pt"
                    warmstart_path = str(candidate) if candidate.exists() else ""
                branch_backend: IsaacUnifiedSingleContextBackend | None = backend
                preflight: dict[str, Any] = {"rows": [], "all_parts_pass": False}
                train_result: dict[str, Any] = {"rows": [], "status": "TRAINING_NOT_STARTED"}
                eval_result: dict[str, Any] = {"rows": [], "nominal_rows": []}
                tuning_rows: list[dict[str, Any]] = []
                physics_audit: dict[str, Any] = {"rows": []}
                try:
                    branch_backend.physics_profile = branch_profile
                    if branch_name == "physics_tuned_residual_ppo" and callable(
                        getattr(branch_backend, "apply_v88_conservative_contact_tuning", None)
                    ):
                        tuning_rows = branch_backend.apply_v88_conservative_contact_tuning()
                    preflight = run_v86_contact_gate_preflight(branch_dir, branch_backend)
                    preflight_rows = [
                        {**row, "branch_name": branch_name, "physics_profile": branch_profile}
                        for row in preflight.get("rows", [])
                    ]
                    write_csv(branch_dir / "v88_contact_gate_preflight.csv", preflight_rows)
                    write_json(branch_dir / "v88_contact_gate_preflight.json", preflight_rows)
                    write_json(branch_dir / "v88_selected_staging_variants.json", preflight.get("selected_variants", {}))
                    physics_audit = write_v88_physics_asset_audit(
                        branch_dir,
                        backend=branch_backend,
                        tuning_rows=tuning_rows,
                        physics_profile=branch_profile,
                    )
                    train_result = train_v88_stabilized_residual_ppo(
                        branch_dir,
                        branch_name=branch_name,
                        policy_mode=args_cli.policy_mode,
                        parts=parts,
                        num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                        max_steps=v88_max_steps,
                        physics_profile=branch_profile,
                        physical_backend=branch_backend,
                        ppo_iterations=v88_iterations,
                        contact_gate_rows=preflight.get("rows", []),
                        enhanced_observation=enhanced_observation,
                        resume_checkpoint_path=warmstart_path,
                    )
                    if bool(train_result.get("runner_policy_artifact_complete")) and str(train_result.get("checkpoint_path") or ""):
                        eval_result = evaluate_v88_stabilized_policy(
                            branch_dir,
                            branch_name=branch_name,
                            checkpoint_path=str(train_result.get("checkpoint_path") or ""),
                            parts=parts,
                            eval_step_count=max(1, int(v88_max_steps)),
                            physical_backend=branch_backend,
                            physics_profile=branch_profile,
                            enhanced_observation=enhanced_observation,
                        )
                    all_observation_feature_rows.extend(list(getattr(branch_backend, "v88_observation_feature_rows", []) or []))
                    all_preflight_rows.extend(preflight_rows)
                    all_train_rows.extend(train_result.get("rows", []))
                    all_eval_rows.extend(eval_result.get("rows", []))
                    all_nominal_rows.extend(eval_result.get("nominal_rows", []))
                    for row in physics_audit.get("rows", []):
                        all_physics_rows.append({**row, "branch_name": branch_name})
                    branch_results.append(
                        {
                            "branch_name": branch_name,
                            "physics_profile": branch_profile,
                            "enhanced_observation": enhanced_observation,
                            "preflight_rows": preflight_rows,
                            "train_rows": train_result.get("rows", []),
                            "nominal_eval_rows": eval_result.get("nominal_rows", []),
                            "policy_eval_rows": eval_result.get("rows", []),
                        }
                    )
                    branch_metadata.append(
                        {
                            "branch_name": branch_name,
                            "physics_profile": branch_profile,
                            "enhanced_observation": enhanced_observation,
                            "warmstart_path": warmstart_path,
                            "preflight_all_parts_pass": bool(preflight.get("all_parts_pass")),
                            "ppo_ran": bool(train_result.get("ppo_ran")),
                            "runner_policy_artifact_complete": bool(train_result.get("runner_policy_artifact_complete")),
                            "checkpoint_path": str(train_result.get("checkpoint_path") or ""),
                            "grasp_success_claimed_any_part": any(bool(row.get("grasp_success_claimed")) for row in eval_result.get("rows", [])),
                            "object_write_by_policy_detected": bool(getattr(branch_backend, "object_write_by_policy_detected", False)),
                            "sticky_action_available_to_policy": bool(getattr(branch_backend, "sticky_action_available_to_policy", False)),
                        }
                    )
                finally:
                    pass

            training_summary_csv = write_csv(run_dir / "v88_training_summary.csv", all_train_rows)
            training_summary_json = write_json(run_dir / "v88_training_summary.json", all_train_rows)
            eval_summary_csv = write_csv(run_dir / "v88_deterministic_eval_no_sticky.csv", all_eval_rows)
            eval_summary_json = write_json(run_dir / "v88_deterministic_eval_no_sticky.json", all_eval_rows)
            nominal_summary_csv = write_csv(run_dir / "v88_nominal_only_eval_summary.csv", all_nominal_rows)
            nominal_summary_json = write_json(run_dir / "v88_nominal_only_eval_summary.json", all_nominal_rows)
            preflight_csv = write_csv(run_dir / "v88_contact_gate_preflight.csv", all_preflight_rows)
            preflight_json = write_json(run_dir / "v88_contact_gate_preflight.json", all_preflight_rows)
            physics_csv = write_csv(run_dir / "v88_physics_asset_audit.csv", all_physics_rows)
            physics_json = write_json(run_dir / "v88_physics_asset_audit.json", all_physics_rows)
            physics_md = run_dir / "v88_physics_asset_audit.md"
            _write_dynamic_md(physics_md, all_physics_rows)
            observation_audit = write_v88_observation_audit(run_dir, enhanced_feature_rows=all_observation_feature_rows)
            progress = _write_v88_progress_matrix(run_dir, parts=parts, branch_results=branch_results)
            forbidden_scan = write_v88_forbidden_artifact_scan(run_dir)
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "asset": {k: v for k, v in asset.items() if k != "rows"},
                    "randomization": randomization,
                    "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                    "priors": {k: v for k, v in priors.items() if k != "rows"},
                    "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                    "branches": branch_metadata,
                    "code_path_audit": {k: v for k, v in code_audit.items() if k != "rows"},
                    "observation_audit": {k: v for k, v in observation_audit.items() if k != "rows"},
                    "physics_asset_audit": {
                        "v88_physics_asset_audit_csv": str(physics_csv),
                        "v88_physics_asset_audit_json": str(physics_json),
                        "v88_physics_asset_audit_md": str(physics_md),
                    },
                    "training_summary_csv": str(training_summary_csv),
                    "training_summary_json": str(training_summary_json),
                    "deterministic_eval_no_sticky_csv": str(eval_summary_csv),
                    "deterministic_eval_no_sticky_json": str(eval_summary_json),
                    "nominal_only_eval_summary_csv": str(nominal_summary_csv),
                    "nominal_only_eval_summary_json": str(nominal_summary_json),
                    "contact_gate_preflight_csv": str(preflight_csv),
                    "contact_gate_preflight_json": str(preflight_json),
                    "progress": {k: v for k, v in progress.items() if k not in {"rows", "branch_rows"}},
                    "forbidden_artifact_scan": {k: v for k, v in forbidden_scan.items() if k != "rows"},
                    "source_archive": source,
                    "ppo_ran": any(bool(row.get("ppo_ran")) for row in all_train_rows),
                    "rsl_rl_runner_used": any(bool(row.get("rsl_rl_runner_used")) for row in all_train_rows),
                    "runner_policy_artifact_complete": any(bool(row.get("runner_policy_artifact_complete")) for row in all_train_rows),
                    "rl_trained": False,
                    "rl_trained_success_evidence": any(bool(row.get("grasp_success_claimed")) for row in all_eval_rows),
                    "checkpoint_written": any(bool(row.get("checkpoint_written")) for row in all_train_rows),
                    "checkpoint_is_policy_artifact_only": True,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "usable_training_row_count": 0,
                    "training_locked": True,
                    "training_locked_blocker": "v88_nonfinal_policy_artifact_only_no_training_export",
                    "final_success": False,
                }
            )
            metadata_path = write_json(run_dir / "v88_run_metadata.json", metadata)
            print(f"v88 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {progress.get('v88_progress_csv')}")
            return 0

        if args_cli.run_mode in {
            "v87_contact_guided_residual_ppo",
            "v86_guarded_residual_ppo_smoke",
            "v83_unified_physical_backend",
            "v84_controlled_real_contact_probe",
            "v85_contact_sanity_and_plug_screw_repair",
        }:
            backend = IsaacUnifiedSingleContextBackend(
                parts=parts,
                num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                physics_profile=args_cli.physics_profile,
                device=args_cli.device,
                create_env=True,
            )
            if args_cli.run_mode == "v87_contact_guided_residual_ppo":
                v87_max_steps = args_cli.max_steps if int(args_cli.max_steps) != 32 else 96
                v87_iterations = args_cli.ppo_iterations if int(args_cli.ppo_iterations) != 5 else 20
                preflight = run_v86_contact_gate_preflight(run_dir, backend)
                v87_preflight_csv = write_csv(run_dir / "v87_contact_gate_preflight.csv", preflight.get("rows", []))
                v87_preflight_json = write_json(run_dir / "v87_contact_gate_preflight.json", preflight.get("rows", []))
                v87_selected_json = write_json(run_dir / "v87_selected_staging_variants.json", preflight.get("selected_variants", {}))
                train_result = train_v87_contact_guided_residual_ppo(
                    run_dir,
                    policy_mode=args_cli.policy_mode,
                    parts=parts,
                    num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                    max_steps=v87_max_steps,
                    physics_profile=args_cli.physics_profile,
                    physical_backend=backend,
                    ppo_iterations=v87_iterations,
                    contact_gate_rows=preflight.get("rows", []),
                )
                eval_result: dict[str, Any] = {}
                if bool(train_result.get("runner_policy_artifact_complete")) and str(train_result.get("checkpoint_path") or ""):
                    eval_result = evaluate_v87_contact_guided_policy(
                        run_dir,
                        checkpoint_path=str(train_result.get("checkpoint_path") or ""),
                        parts=parts,
                        eval_step_count=max(1, int(v87_max_steps)),
                        physical_backend=backend,
                        physics_profile=args_cli.physics_profile,
                    )
                progress = _write_v87_progress_matrix(
                    run_dir,
                    parts=parts,
                    preflight_rows=preflight.get("rows", []),
                    train_rows=train_result.get("rows", []),
                    nominal_eval_rows=eval_result.get("nominal_rows", []),
                    policy_eval_rows=eval_result.get("rows", []),
                )
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                metadata.update(
                    {
                        "v87_contact_gate_preflight": {
                            **{k: v for k, v in preflight.items() if k not in {"rows", "selected_variants"}},
                            "v87_contact_gate_preflight_csv": str(v87_preflight_csv),
                            "v87_contact_gate_preflight_json": str(v87_preflight_json),
                            "v87_selected_staging_variants_json": str(v87_selected_json),
                        },
                        "training": {k: v for k, v in train_result.items() if k != "rows"},
                        "eval": {k: v for k, v in eval_result.items() if k not in {"rows", "nominal_rows", "failure_rows"}},
                        "progress": {k: v for k, v in progress.items() if k != "rows"},
                        "source_archive": source,
                        "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                        "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                        "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                        "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                        "controlled_contact_gate_pass_all_parts": bool(preflight.get("all_parts_pass")),
                        "ppo_ran": bool(train_result.get("ppo_ran")),
                        "rsl_rl_runner_used": bool(train_result.get("rsl_rl_runner_used")),
                        "runner_policy_artifact_complete": bool(train_result.get("runner_policy_artifact_complete")),
                        "rl_trained": False,
                        "rl_trained_success_evidence": any(bool(row.get("grasp_success_claimed")) for row in eval_result.get("rows", [])),
                        "checkpoint_written": bool(train_result.get("checkpoint_written")),
                        "checkpoint_is_policy_artifact_only": bool(train_result.get("checkpoint_is_policy_artifact_only")),
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "usable_training_row_count": 0,
                        "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                        "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                        "training_locked": True,
                        "training_locked_blocker": "v87_nonfinal_policy_artifact_only_no_training_export",
                    }
                )
                metadata_path = write_json(run_dir / "v87_run_metadata.json", metadata)
                print(f"v87 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {progress.get('v87_progress_csv')}")
                return 0
            if args_cli.run_mode == "v86_guarded_residual_ppo_smoke":
                preflight = run_v86_contact_gate_preflight(run_dir, backend)
                train_result = train_v86_guarded_residual_ppo_smoke(
                    run_dir,
                    policy_mode=args_cli.policy_mode,
                    parts=parts,
                    num_envs=args_cli.num_envs if args_cli.num_envs > 0 else 5,
                    max_steps=args_cli.max_steps,
                    physics_profile=args_cli.physics_profile,
                    physical_backend=backend,
                    ppo_iterations=args_cli.ppo_iterations,
                    contact_gate_rows=preflight.get("rows", []),
                )
                eval_result: dict[str, Any] = {}
                if bool(train_result.get("runner_smoke_complete")) and str(train_result.get("checkpoint_path") or ""):
                    eval_result = evaluate_v86_guarded_residual_policy(
                        run_dir,
                        checkpoint_path=str(train_result.get("checkpoint_path") or ""),
                        parts=parts,
                        eval_step_count=max(1, min(args_cli.max_steps, 32)),
                        physical_backend=backend,
                        physics_profile=args_cli.physics_profile,
                    )
                progress = _write_v86_progress_matrix(
                    run_dir,
                    parts=parts,
                    preflight_rows=preflight.get("rows", []),
                    train_rows=train_result.get("rows", []),
                    eval_rows=eval_result.get("rows", []),
                )
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                metadata.update(
                    {
                        "v86_contact_gate_preflight": {k: v for k, v in preflight.items() if k not in {"rows", "selected_variants"}},
                        "training": {k: v for k, v in train_result.items() if k != "rows"},
                        "eval": {k: v for k, v in eval_result.items() if k not in {"rows", "failure_rows"}},
                        "progress": {k: v for k, v in progress.items() if k != "rows"},
                        "source_archive": source,
                        "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                        "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                        "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                        "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                        "controlled_contact_gate_pass_all_parts": bool(preflight.get("all_parts_pass")),
                        "ppo_ran": bool(train_result.get("ppo_ran")),
                        "rsl_rl_runner_used": bool(train_result.get("rsl_rl_runner_used")),
                        "runner_smoke_complete": bool(train_result.get("runner_smoke_complete")),
                        "rl_trained": False,
                        "rl_trained_success_evidence": any(bool(row.get("grasp_success_claimed")) for row in eval_result.get("rows", [])),
                        "checkpoint_written": bool(train_result.get("checkpoint_written")),
                        "checkpoint_is_runner_artifact_only": bool(train_result.get("checkpoint_is_runner_artifact_only")),
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "usable_training_row_count": 0,
                        "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                        "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                        "training_locked": True,
                        "training_locked_blocker": "v86_ppo_smoke_only_no_training_export",
                    }
                )
                metadata_path = write_json(run_dir / "v86_run_metadata.json", metadata)
                print(f"v86 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {progress.get('v86_progress_csv')}")
                return 0
            if args_cli.run_mode == "v84_controlled_real_contact_probe":
                audit = run_v84_controlled_real_contact_probe(run_dir, backend)
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                all_contact = bool(audit.get("rows")) and all(bool(row.get("v84_success")) for row in audit.get("rows", []))
                metadata.update(
                    {
                        "v84_controlled_real_contact_probe": {k: v for k, v in audit.items() if k != "rows"},
                        "source_archive": source,
                        "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                        "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                        "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                        "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                        "controlled_contact_gate_pass_all_parts": all_contact,
                        "ppo_ran": False,
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "checkpoint_written": False,
                        "usable_training_row_count": 0,
                        "training_locked_blocker": "" if all_contact else "controlled_force_contact_not_observed",
                    }
                )
                metadata_path = write_json(run_dir / "v84_run_metadata.json", metadata)
                print(f"v84 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {audit.get('v84_progress_csv')}")
                return 0
            if args_cli.run_mode == "v85_contact_sanity_and_plug_screw_repair":
                audit = run_v85_contact_sanity_and_plug_screw_repair(run_dir, backend)
                source = write_v81_source_archive_check(REPO_ROOT, run_dir)
                all_contact = bool(audit.get("rows")) and all(bool(row.get("v85_success")) for row in audit.get("rows", []))
                metadata.update(
                    {
                        "v85_contact_sanity_and_plug_screw_repair": {
                            k: v for k, v in audit.items() if k not in {"rows", "variant_rows"}
                        },
                        "source_archive": source,
                        "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                        "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                        "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                        "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                        "controlled_contact_gate_pass_all_parts": all_contact,
                        "ppo_ran": False,
                        "sticky_eval_ran": False,
                        "final_replay_ran": False,
                        "video_generated": False,
                        "checkpoint_written": False,
                        "usable_training_row_count": 0,
                        "training_locked": True,
                        "training_locked_blocker": "v85_contact_validation_only_no_training_export"
                        if all_contact
                        else "controlled_force_contact_not_observed_or_reset_sanity_failed",
                    }
                )
                metadata_path = write_json(run_dir / "v85_run_metadata.json", metadata)
                print(f"v85 run complete: {run_dir}")
                print(f"metadata: {metadata_path}")
                print(f"progress: {audit.get('v85_progress_csv')}")
                return 0
            audit = run_v83_unified_physical_backend_audit(run_dir, backend)
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "v83_unified_physical_backend": {k: v for k, v in audit.items() if k != "rows"},
                    "source_archive": source,
                    "single_simulation_context": bool(getattr(backend, "single_simulation_context", False)),
                    "gym_make_count": int(getattr(backend, "gym_make_count", 0)),
                    "vector_reset_ok": bool(getattr(backend, "vector_reset_ok", False)),
                    "vector_step_ok": bool(getattr(backend, "vector_step_ok", False)),
                    "ppo_ran": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "video_generated": False,
                    "usable_training_row_count": 0,
                    "training_locked_blocker": "controlled_force_contact_not_observed",
                }
            )
            metadata_path = write_json(run_dir / "v83_run_metadata.json", metadata)
            print(f"v83 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {audit.get('v83_progress_csv')}")
            return 0

        backend = IsaacUnifiedGraspPhysicalBackend(
            parts=parts,
            num_envs=args_cli.num_envs,
            physics_profile=args_cli.physics_profile,
            device=args_cli.device,
            require_contact_sensor=True,
            create_envs=True,
        )
        backend_report = write_backend_creation_report(run_dir, backend)
        if args_cli.run_mode == "v82_contact_actuation_probe":
            probe = write_v82_contact_actuation_probe(run_dir, backend)
            progress = _write_v82_progress_matrix(run_dir, probe["rows"])
            source = write_v81_source_archive_check(REPO_ROOT, run_dir)
            metadata.update(
                {
                    "backend": backend_report,
                    "contact_actuation_probe": {k: v for k, v in probe.items() if k not in {"rows", "trace_rows", "action_mapping_rows", "object_identity_rows"}},
                    "progress": progress,
                    "source_archive": source,
                    "ppo_ran": False,
                    "sticky_eval_ran": False,
                    "final_replay_ran": False,
                    "usable_training_row_count": 0,
                    "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                    "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                }
            )
            metadata_path = write_json(run_dir / "v82_run_metadata.json", metadata)
            print(f"v82 run complete: {run_dir}")
            print(f"metadata: {metadata_path}")
            print(f"progress: {progress.get('v82_progress_csv')}")
            return 0
        probe = write_controlled_contact_probe(run_dir, backend)
        progress_rows = merge_v81_progress_rows(progress_rows, _progress_from_probe(parts, probe["rows"], backend))

        env_result: dict[str, Any] = {}
        train_result: dict[str, Any] = {}
        eval_result: dict[str, Any] = {}
        if args_cli.run_mode in {
            "unified_env_real_vectorized_smoke",
            "real_residual_ppo_smoke",
            "deterministic_eval_no_sticky",
            "deterministic_eval_sticky_after_support",
        }:
            env_result = run_env_smoke(
                run_dir,
                num_envs=args_cli.num_envs,
                steps=min(args_cli.max_steps, 12),
                parts=parts,
                physics_profile=args_cli.physics_profile,
                allow_diagnostic_dynamics=False,
                physical_backend=backend,
            )
            progress_rows = merge_v81_progress_rows(progress_rows, _progress_from_env(parts, env_result["rows"]))

        if args_cli.run_mode == "real_residual_ppo_smoke":
            controlled_gate_pass = any(bool(row.get("controlled_contact_gate_pass")) for row in probe.get("rows", []))
            train_result = train_unified_policy(
                run_dir,
                policy_mode=args_cli.policy_mode,
                parts=parts,
                num_envs=args_cli.num_envs,
                max_steps=args_cli.max_steps,
                physics_profile=args_cli.physics_profile,
                allow_diagnostic_dynamics=args_cli.allow_diagnostic_dynamics,
                physical_backend=backend,
                ppo_iterations=args_cli.ppo_iterations,
                controlled_contact_gate_pass=controlled_gate_pass,
            )
            progress_rows = merge_v81_progress_rows(progress_rows, _progress_from_training(parts, train_result))

        if args_cli.run_mode in {"deterministic_eval_no_sticky", "deterministic_eval_sticky_after_support"}:
            sticky = args_cli.run_mode == "deterministic_eval_sticky_after_support"
            eval_result = evaluate_unified_policy(
                run_dir,
                checkpoint_path=args_cli.checkpoint_path,
                parts=parts,
                sticky_after_support=sticky,
                eval_episode_count=max(1, min(args_cli.max_steps, 16)),
                physical_backend=backend,
                physics_profile=args_cli.physics_profile,
            )
            progress_rows = merge_v81_progress_rows(progress_rows, _progress_from_eval(parts, eval_result, sticky_after_support=sticky))

        progress = write_v81_progress_matrix(run_dir, progress_rows, also_root=True)
        failure_rows = [
            {"part_name": row.get("part_name"), "failure_category": row.get("failure_category"), "blocker": row.get("blocker")}
            for row in progress_rows
        ]
        failure_csv = write_csv(run_dir / "v81_failure_taxonomy.csv", failure_rows)
        failure_json = write_json(run_dir / "v81_failure_taxonomy.json", failure_rows)
        source = write_v81_source_archive_check(REPO_ROOT, run_dir)
        metadata.update(
            {
                "asset": {k: v for k, v in asset.items() if k != "rows"},
                "randomization": randomization,
                "seeds": {k: v for k, v in seeds.items() if k != "rows"},
                "priors": {k: v for k, v in priors.items() if k != "rows"},
                "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
                "backend": backend_report,
                "contact_probe": {k: v for k, v in probe.items() if k not in {"rows", "trace_rows"}},
                "env": {k: v for k, v in env_result.items() if k != "rows"},
                "training": {k: v for k, v in train_result.items() if k != "rows"},
                "eval": {k: v for k, v in eval_result.items() if k not in {"rows", "failure_rows"}},
                "progress": progress,
                "failure_taxonomy_csv": str(failure_csv),
                "failure_taxonomy_json": str(failure_json),
                "source_archive": source,
                "object_write_reset_only": bool(getattr(backend, "object_write_reset_only", False)),
                "object_write_by_policy_detected": bool(getattr(backend, "object_write_by_policy_detected", False)),
                "sticky_action_available_to_policy": bool(getattr(backend, "sticky_action_available_to_policy", False)),
                "proxy_action_available_to_policy": bool(getattr(backend, "proxy_action_available_to_policy", False)),
                "route_selection_available_to_policy": bool(getattr(backend, "route_selection_available_to_policy", False)),
            }
        )
        metadata_path = write_json(run_dir / "v81_run_metadata.json", metadata)
        print(f"v81 run complete: {run_dir}")
        print(f"metadata: {metadata_path}")
        print(f"progress: {progress.get('v81_progress_csv')}")
        return 0
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
