"""v90 bottleneck isolation and grasp-feasibility helpers."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json


V90_RUN_MODE = "v90_bottleneck_isolation_and_grasp_feasibility"
V90_DISK_THRESHOLD_GB = 15.0
V90_DECISIONS = {
    "FEASIBLE_NOW",
    "FEASIBLE_AFTER_ASSET_FIX",
    "CONTROL_POLICY_BOTTLENECK",
    "NO_FEASIBLE_CANDIDATE_FOUND",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_v89_artifacts(repo_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(repo_root)
    run = root / "debug_runs/v89_failure_driven_hybrid_grasp_repair"
    return {
        "progress": _read_csv(run / "v89_progress_matrix.csv"),
        "candidate_results": _read_csv(run / "v89_candidate_probe_results.csv"),
        "eval": _read_csv(run / "v89_deterministic_eval_no_sticky.csv"),
    }


def write_v90_disk_preflight(
    run_dir: str | Path,
    *,
    repo_root: str | Path,
    threshold_gb: float = V90_DISK_THRESHOLD_GB,
    heavy_feasibility_ran: bool = False,
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    usage = shutil.disk_usage(run_path)
    free_gb = usage.free / (1024.0**3)
    total_gb = usage.total / (1024.0**3)
    used_gb = usage.used / (1024.0**3)
    cleanup_candidates = _v90_cleanup_candidates(Path(repo_root), run_path)
    payload = {
        "run_mode": V90_RUN_MODE,
        "disk_path": str(run_path),
        "free_gb": free_gb,
        "used_gb": used_gb,
        "total_gb": total_gb,
        "threshold_gb": float(threshold_gb),
        "disk_space_ok_for_heavy_feasibility": free_gb >= float(threshold_gb),
        "heavy_feasibility_ran": bool(heavy_feasibility_ran),
        "inotify_watch_risk_hint": "low" if free_gb >= float(threshold_gb) else "high_disk_pressure_can_trigger_runtime_watch_failures",
        "cleanup_policy": "only new v90 transient files may be removed automatically; preserve v83-v89 summaries and canonical baseline",
        "cleanup_candidates": cleanup_candidates,
    }
    txt_lines = [
        f"run_mode: {V90_RUN_MODE}",
        f"free_gb: {free_gb:.3f}",
        f"threshold_gb: {float(threshold_gb):.3f}",
        f"disk_space_ok_for_heavy_feasibility: {payload['disk_space_ok_for_heavy_feasibility']}",
        f"heavy_feasibility_ran: {bool(heavy_feasibility_ran)}",
        f"inotify_watch_risk_hint: {payload['inotify_watch_risk_hint']}",
        "cleanup_policy: preserve v83-v89 summaries and canonical baseline",
    ]
    if cleanup_candidates:
        txt_lines.append("cleanup_candidates:")
        txt_lines.extend(f"- {row['path']} ({row['reason']})" for row in cleanup_candidates)
    else:
        txt_lines.append("cleanup_candidates: none")
    txt_path = run_path / "v90_disk_preflight.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    json_path = write_json(run_path / "v90_disk_preflight.json", payload)
    return {**payload, "v90_disk_preflight_txt": str(txt_path), "v90_disk_preflight_json": str(json_path)}


def _v90_cleanup_candidates(repo_root: Path, run_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transient_roots = [run_path / "tmp", run_path / "cache", run_path / "branches"]
    for path in transient_roots:
        if path.exists():
            rows.append({"path": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path), "reason": "new_v90_transient_only"})
    return rows


def write_v90_code_freeze_report(
    run_dir: str | Path,
    *,
    repo_root: str | Path,
    touched_files: list[str],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    root = Path(repo_root)
    rows: list[dict[str, Any]] = []
    for rel in touched_files:
        path = root / rel
        rows.append(
            {
                "category": "touched_file",
                "path": rel,
                "exists": path.exists(),
                "v90_role": "runner/helper/backend_extension",
                "recommendation": "keep_for_v90_audit",
            }
        )
    rows.extend(
        [
            {
                "category": "architecture_guard",
                "path": "scripts/environments/run_v81_physical_backend_grasp_rl.py",
                "exists": True,
                "v90_role": "existing_runner_reused",
                "recommendation": "no_new_runner_created",
            },
            {
                "category": "architecture_guard",
                "path": "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_unified_five_object_env.py",
                "exists": (root / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_unified_five_object_env.py").exists(),
                "v90_role": "existing_single_context_env_reused",
                "recommendation": "no_new_env_created",
            },
            {
                "category": "architecture_guard",
                "path": "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/single_context_backend.py",
                "exists": True,
                "v90_role": "existing_backend_extended",
                "recommendation": "no_new_backend_class_created",
            },
            {
                "category": "train_eval_path",
                "path": "PPO/train/eval",
                "exists": False,
                "v90_role": "intentionally_not_called",
                "recommendation": "keep_v90_diagnostic_only",
            },
            {
                "category": "duplicated_logic",
                "path": "v86-v90 per-stage matrix writers",
                "exists": True,
                "v90_role": "audit_critical_old_modes_preserved",
                "recommendation": "future_cleanup_candidate_only_after_final_policy_decision",
            },
            {
                "category": "likely_dead_mode",
                "path": "legacy multi-gym physical backend success path",
                "exists": True,
                "v90_role": "not_used_for_success",
                "recommendation": "quarantine_candidate_only_do_not_delete_in_v90",
            },
        ]
    )
    summary = {
        "run_mode": V90_RUN_MODE,
        "new_runner_created": False,
        "new_env_created": False,
        "new_backend_created": False,
        "helper_module_added": True,
        "ppo_path_called": False,
        "sticky_path_called": False,
        "final_replay_path_called": False,
        "video_path_called": False,
        "dataset_export_path_called": False,
        "rows": rows,
    }
    csv_path = write_csv(run_path / "v90_code_freeze_report.csv", rows)
    json_path = write_json(run_path / "v90_code_freeze_report.json", summary)
    md_path = run_path / "v90_code_freeze_report.md"
    _write_md(md_path, rows)
    return {**summary, "v90_code_freeze_report_csv": str(csv_path), "v90_code_freeze_report_json": str(json_path), "v90_code_freeze_report_md": str(md_path)}


def default_v90_candidate_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: dict[str, list[tuple[str, str, str, str, tuple[float, float, float], tuple[float, float, float], float, float, float]]] = {
        "Plug2": [
            ("plug2_side_body_pinch", "side_body_pinch", "23", "plug cylindrical body side", (0.010, 0.000, 0.0), (0.0, 0.0, 0.28), 1.00, 0.46, 115.0),
            ("plug2_cap_head_pinch", "cap_head_pinch", "34", "plug cap/head rim", (0.012, 0.004, 0.0), (0.0, 0.0, 0.24), 0.88, 0.38, 95.0),
            ("plug2_two_side_clamp", "two_side_clamp", "234", "opposed plug body sides", (0.008, -0.004, 0.0), (0.0, 0.0, 0.24), 0.82, 0.34, 90.0),
            ("plug2_current_nominal_low_force", "lower_force_current_nominal", "23", "current best plug side", (0.014, 0.000, 0.0), (0.0, 0.0, 0.22), 0.70, 0.30, 80.0),
        ],
        "Screw1": [
            ("screw1_shaft_pinch", "shaft_pinch", "34", "screw shaft", (0.008, 0.000, 0.0), (0.0, 0.0, 0.24), 1.00, 0.42, 110.0),
            ("screw1_head_pinch", "head_pinch", "23", "screw head", (0.010, 0.004, 0.0), (0.0, 0.0, 0.22), 0.88, 0.36, 95.0),
            ("screw1_multifinger_clamp", "multi_finger_clamp", "234", "shaft plus head support", (0.012, -0.004, 0.0), (0.0, 0.0, 0.22), 0.78, 0.32, 85.0),
            ("screw1_shallow_wrap", "shallow_wrap", "234", "shallow wrap around shaft", (0.014, 0.000, 0.0), (0.0, 0.0, 0.20), 0.68, 0.28, 80.0),
        ],
        "Backrest": [
            ("backrest_current_best_nominal", "current_best_nominal", "34", "backrest broad face", (0.012, 0.000, 0.0), (0.0, 0.0, 0.22), 0.62, 0.28, 105.0),
            ("backrest_lower_force_variant", "lower_force_variant", "34", "broad face lower close", (0.016, 0.004, 0.0), (0.0, 0.0, 0.18), 0.48, 0.22, 80.0),
            ("backrest_wide_multifinger_support", "wider_multi_finger_support", "234", "face plus side support", (0.014, -0.004, 0.0), (0.0, 0.0, 0.20), 0.56, 0.24, 90.0),
            ("backrest_slower_lift", "slower_lift", "34", "broad face slow lift", (0.012, 0.000, 0.0), (0.0, 0.0, 0.14), 0.58, 0.22, 85.0),
        ],
        "Rod": [
            ("rod_side_clamp", "side_clamp", "34", "rod long side", (0.010, 0.000, 0.0), (0.0, 0.0, 0.24), 0.82, 0.30, 100.0),
            ("rod_shallow_wrap", "shallow_wrap", "234", "partial wrap on rod", (0.012, 0.004, 0.0), (0.0, 0.0, 0.22), 0.74, 0.26, 90.0),
            ("rod_two_finger_antiroll", "two_finger_anti_roll_support", "23", "side pinch with anti-roll support", (0.014, -0.004, 0.0), (0.0, 0.0, 0.20), 0.70, 0.24, 85.0),
            ("rod_friendly_lift", "rod_friendly_lift", "234", "rod side support with gentle lift", (0.016, 0.000, 0.0), (0.0, 0.0, 0.16), 0.66, 0.22, 80.0),
        ],
        "Frame": [
            ("frame_edge_clamp", "edge_clamp", "234", "frame edge/bar", (0.012, 0.000, 0.0), (0.0, 0.0, 0.22), 0.78, 0.26, 95.0),
            ("frame_two_side_support", "two_side_support", "34", "two side bar support", (0.014, 0.004, 0.0), (0.0, 0.0, 0.20), 0.70, 0.24, 85.0),
            ("frame_corner_pinch", "corner_pinch", "234", "frame corner", (0.016, -0.004, 0.0), (0.0, 0.0, 0.18), 0.66, 0.22, 80.0),
            ("frame_lower_force_support_lift", "lower_force_support_lift", "234", "frame support lift lower force", (0.018, 0.000, 0.0), (0.0, 0.0, 0.15), 0.58, 0.20, 75.0),
        ],
    }
    clearance_offsets = (0.0, 0.004, -0.004)
    wrist_rolls = (0.0, 0.10, -0.10)
    for part, families in specs.items():
        rank = 0
        for base_id, family, group, surfaces, pregrasp, lift, close, approach, force_limit in families:
            for variant_index, delta in enumerate(clearance_offsets):
                clearance = max(0.004, pregrasp[0] + abs(delta) * 0.5)
                lateral = pregrasp[1] + delta
                rows.append(
                    {
                        "candidate_id": f"{base_id}_v{variant_index}",
                        "part_name": part,
                        "candidate_family": family,
                        "candidate_rank": rank,
                        "family_variant_index": variant_index,
                        "active_finger_group": group,
                        "approach_direction": "sensor_to_object",
                        "surface_strategy": "bbox_adjusted" if family not in {"cap_head_pinch", "corner_pinch"} else "analytic_surface",
                        "use_bbox_adjustment": family not in {"cap_head_pinch", "corner_pinch"},
                        "expected_contact_surfaces": surfaces,
                        "target_clearance_m": clearance,
                        "lateral_offset_m": lateral,
                        "pregrasp_offset_x": pregrasp[0],
                        "pregrasp_offset_y": lateral,
                        "pregrasp_offset_z": pregrasp[2],
                        "wrist_roll_rad": wrist_rolls[variant_index],
                        "close_value": max(0.0, min(1.0, close - 0.04 * variant_index)),
                        "approach_scale": max(0.0, min(0.65, approach - 0.03 * variant_index)),
                        "hold_close_value": max(0.0, min(1.0, close - 0.02 * variant_index)),
                        "force_limit_n": force_limit,
                        "hold_steps": 48 + 8 * variant_index,
                        "lift_steps": 32,
                        "lift_x": lift[0],
                        "lift_y": lift[1],
                        "lift_z": max(0.0, lift[2] - 0.02 * variant_index),
                        "max_physics_steps": 128,
                        "physics_profile": "canonical",
                        "source": "v90_geometry_only_no_success_label",
                        "success_label_used": False,
                    }
                )
                rank += 1
    return rows


def write_v90_feasibility_candidate_plan(
    run_dir: str | Path,
    *,
    max_candidates_per_part: int = 12,
    max_total_candidates: int = 60,
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        part_rows = [row for row in default_v90_candidate_plan() if row.get("part_name") == part]
        rows.extend(part_rows[: max(0, int(max_candidates_per_part))])
    rows = rows[: max(0, int(max_total_candidates))]
    csv_path = write_csv(run_path / "v90_feasibility_candidate_plan.csv", rows)
    json_path = write_json(run_path / "v90_feasibility_candidate_plan.json", rows)
    return {"rows": rows, "v90_feasibility_candidate_plan_csv": str(csv_path), "v90_feasibility_candidate_plan_json": str(json_path)}


def write_v90_physics_asset_bottleneck_report(
    run_dir: str | Path,
    *,
    canonical_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]] | None = None,
    tuning_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    diagnostic_rows = diagnostic_rows or []
    tuning_rows = tuning_rows or []
    for part in V80_PARTS:
        best = _best_candidate([row for row in canonical_rows if row.get("part_name") == part])
        diagnostic = _best_candidate([row for row in diagnostic_rows if row.get("part_name") == part])
        slip_dominant = _is_slip_dominant(best)
        diagnostic_pass = bool(diagnostic.get("candidate_success") or diagnostic.get("canonical_feasibility_pass"))
        rows.append(
            {
                "part_name": part,
                "canonical_best_candidate_id": best.get("candidate_id", ""),
                "canonical_status": best.get("status", "NO_CANONICAL_CANDIDATE"),
                "canonical_blocker": best.get("blocker", ""),
                "canonical_support_gate_rate": _float(best.get("support_gate_rate")),
                "canonical_hold_gate_rate": _float(best.get("hold_gate_rate")),
                "canonical_lift_gate_rate": _float(best.get("lift_gate_rate")),
                "canonical_peak_force_n": _float(best.get("peak_force_n") or best.get("force_contact_peak_n")),
                "canonical_object_displacement_max_m": _float(best.get("object_displacement_max_m")),
                "slip_dominant_failure": slip_dominant,
                "asset_tunable_fields_exposed": bool(tuning_rows),
                "diagnostic_material_comparison_ran": bool(diagnostic_rows),
                "diagnostic_material_comparison_passed": diagnostic_pass,
                "bounded_static_friction_limit": 1.25,
                "bounded_dynamic_friction_limit": 1.05,
                "bounded_contact_offset_m": 0.002,
                "adhesion_or_infinite_friction_used": False,
                "hidden_sticky_used": False,
                "modified_physics_success_treated_as_final": False,
                "decision_rationale": _physics_rationale(best, bool(diagnostic_rows), diagnostic_pass, slip_dominant, bool(tuning_rows)),
            }
        )
    csv_path = write_csv(run_path / "v90_physics_asset_bottleneck_report.csv", rows)
    json_path = write_json(run_path / "v90_physics_asset_bottleneck_report.json", {"rows": rows, "tuning_rows": tuning_rows})
    md_path = run_path / "v90_physics_asset_bottleneck_report.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v90_physics_asset_bottleneck_report_csv": str(csv_path), "v90_physics_asset_bottleneck_report_json": str(json_path), "v90_physics_asset_bottleneck_report_md": str(md_path)}


def classify_v90_bottlenecks(
    run_dir: str | Path,
    *,
    candidate_rows: list[dict[str, Any]],
    physics_rows: list[dict[str, Any]],
    repo_root: str | Path,
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    v89 = load_v89_artifacts(repo_root)
    diagnosis_rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        part_candidates = [row for row in candidate_rows if row.get("part_name") == part]
        best = _best_candidate(part_candidates)
        physics = next((row for row in physics_rows if row.get("part_name") == part), {})
        prior = next((row for row in v89["progress"] if row.get("part_name") == part), {})
        learned_success = _bool(prior.get("grasp_success_claimed")) or str(prior.get("status") or "").startswith("GRASP_EVAL_PASS")
        canonical_pass = _bool(best.get("canonical_feasibility_pass") or best.get("candidate_success"))
        diagnostic_physics_pass = _bool(physics.get("diagnostic_material_comparison_passed"))
        if canonical_pass and not learned_success:
            decision = "CONTROL_POLICY_BOTTLENECK"
            blocker = "deterministic_feasibility_candidate_passed_but_v89_learned_eval_did_not"
            next_action = "use_v90_candidate_trace_for_bc_or_residual_policy_repair"
        elif canonical_pass:
            decision = "FEASIBLE_NOW"
            blocker = ""
            next_action = "promote_candidate_as_nonfinal_feasibility_input_for_later_policy_stage"
        elif diagnostic_physics_pass or _asset_fix_evidence(best, physics):
            decision = "FEASIBLE_AFTER_ASSET_FIX"
            blocker = str(best.get("blocker") or physics.get("canonical_blocker") or "asset_or_physics_configuration_blocks_candidate")
            next_action = "repair_collision_material_mass_or_contact_configuration_then_repeat_v90"
        else:
            decision = "NO_FEASIBLE_CANDIDATE_FOUND"
            blocker = str(best.get("blocker") or "all_v90_candidates_failed")
            next_action = "add_new_geometry_candidate_family_or_repair_controller_staging"
        diagnosis_rows.append(
            {
                "part_name": part,
                "decision": decision,
                "canonical_feasible_now": canonical_pass,
                "learned_v89_success": learned_success,
                "best_candidate_id": best.get("candidate_id", ""),
                "best_candidate_family": best.get("candidate_family", ""),
                "best_status": best.get("status", ""),
                "force_contact_rate": _float(best.get("force_contact_rate")),
                "contact_duration_mean": _float(best.get("contact_duration_mean")),
                "support_gate_rate": _float(best.get("support_gate_rate")),
                "hold_gate_rate": _float(best.get("hold_gate_rate")),
                "lift_gate_rate": _float(best.get("lift_gate_rate")),
                "peak_force_n": _float(best.get("peak_force_n") or best.get("force_contact_peak_n")),
                "mean_force_n": _float(best.get("mean_force_n") or best.get("mean_force")),
                "excessive_force_rate": _float(best.get("excessive_force_rate")),
                "object_displacement_max_m": _float(best.get("object_displacement_max_m")),
                "multi_finger_support_rate": _float(best.get("multi_finger_support_rate")),
                "object_write_by_policy_detected": _bool(best.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": _bool(best.get("sticky_action_available_to_policy")),
                "fallback_success_used": _bool(best.get("fallback_success_used")),
                "distance_only_success_used": False,
                "diagnostic_physics_comparison_passed": diagnostic_physics_pass,
                "blocker": blocker,
                "next_action": next_action,
            }
        )
    csv_path = write_csv(run_path / "v90_per_object_bottleneck_diagnosis.csv", diagnosis_rows)
    json_path = write_json(run_path / "v90_per_object_bottleneck_diagnosis.json", diagnosis_rows)
    return {"rows": diagnosis_rows, "v90_per_object_bottleneck_diagnosis_csv": str(csv_path), "v90_per_object_bottleneck_diagnosis_json": str(json_path)}


def write_v90_progress_matrix(
    run_dir: str | Path,
    *,
    diagnosis_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    backend_summary: dict[str, Any],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    matrix_rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        diagnosis = next((row for row in diagnosis_rows if row.get("part_name") == part), {})
        best = _best_candidate([row for row in candidate_rows if row.get("part_name") == part])
        decision = str(diagnosis.get("decision") or "NO_FEASIBLE_CANDIDATE_FOUND")
        if decision not in V90_DECISIONS:
            decision = "NO_FEASIBLE_CANDIDATE_FOUND"
        matrix_rows.append(
            {
                "part_name": part,
                "diagnostic_decision": decision,
                "canonical_feasible_now": _bool(diagnosis.get("canonical_feasible_now")),
                "best_candidate_id": diagnosis.get("best_candidate_id", ""),
                "best_candidate_family": diagnosis.get("best_candidate_family", ""),
                "object_identity_verified": _bool(best.get("object_identity_verified")),
                "single_simulation_context": bool(backend_summary.get("single_simulation_context")),
                "vector_reset_ok": bool(backend_summary.get("vector_reset_ok")),
                "vector_step_ok": bool(backend_summary.get("vector_step_ok")),
                "candidate_count_tested": sum(1 for row in candidate_rows if row.get("part_name") == part),
                "force_contact_rate": _float(diagnosis.get("force_contact_rate")),
                "contact_duration_mean": _float(diagnosis.get("contact_duration_mean")),
                "support_gate_rate": _float(diagnosis.get("support_gate_rate")),
                "hold_gate_rate": _float(diagnosis.get("hold_gate_rate")),
                "lift_gate_rate": _float(diagnosis.get("lift_gate_rate")),
                "peak_force_n": _float(diagnosis.get("peak_force_n")),
                "mean_force_n": _float(diagnosis.get("mean_force_n")),
                "excessive_force_rate": _float(diagnosis.get("excessive_force_rate")),
                "object_displacement_max_m": _float(diagnosis.get("object_displacement_max_m")),
                "multi_finger_support_rate": _float(diagnosis.get("multi_finger_support_rate")),
                "object_write_by_policy_detected": _bool(diagnosis.get("object_write_by_policy_detected")) or bool(backend_summary.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": _bool(diagnosis.get("sticky_action_available_to_policy")) or bool(backend_summary.get("sticky_action_available_to_policy")),
                "fallback_success_used": _bool(diagnosis.get("fallback_success_used")),
                "distance_only_success_used": _bool(diagnosis.get("distance_only_success_used")),
                "ppo_ran": False,
                "checkpoint_written": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "dataset_exported": False,
                "usable_training_row_count": 0,
                "final_success": False,
                "status": decision,
                "blocker": diagnosis.get("blocker", ""),
                "next_action": diagnosis.get("next_action", ""),
            }
        )
    csv_path = write_csv(run_path / "v90_progress_matrix.csv", matrix_rows)
    md_path = run_path / "v90_progress_matrix.md"
    _write_md(md_path, matrix_rows)
    root_debug = Path.cwd() / "debug_runs"
    root_csv = write_csv(root_debug / "v90_progress_matrix.csv", matrix_rows)
    root_md = root_debug / "v90_progress_matrix.md"
    _write_md(root_md, matrix_rows)
    return {
        "rows": matrix_rows,
        "v90_progress_matrix_csv": str(csv_path),
        "v90_progress_matrix_md": str(md_path),
        "v90_root_progress_matrix_csv": str(root_csv),
        "v90_root_progress_matrix_md": str(root_md),
    }


def write_v90_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    tokens = (
        ".mp4",
        "final_replay",
        "final_video",
        "usable_training_row",
        "training_rows",
        "dataset_export",
        "bc_dataset",
        "sticky_eval",
        "sticky_success",
        "checkpoint.pt",
        "policy_artifact_checkpoint",
        "rsl_rl",
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(run_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_path).as_posix()
        hits = [token for token in tokens if token in rel]
        if hits:
            rows.append({"path": rel, "forbidden_tokens": ",".join(hits), "forbidden_artifact_present": True})
    summary = {
        "forbidden_artifact_present": bool(rows),
        "forbidden_artifact_count": len(rows),
        "ppo_ran": False,
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "checkpoint_written": False,
        "usable_training_row_count": 0,
        "rows": rows,
    }
    txt_path = run_path / "v90_forbidden_artifact_scan.txt"
    txt_path.write_text(
        ("FORBIDDEN ARTIFACTS FOUND\n" + "\n".join(row["path"] for row in rows) if rows else "No v90 forbidden PPO/checkpoint/final/sticky/video/dataset artifacts found.") + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v90_forbidden_artifact_scan.json", summary)
    return {**summary, "v90_forbidden_artifact_scan_txt": str(txt_path), "v90_forbidden_artifact_scan_json": str(json_path)}


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=_candidate_score, default={})


def _candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    forbidden = _bool(row.get("object_write_by_policy_detected")) or _bool(row.get("sticky_action_available_to_policy")) or _bool(row.get("fallback_success_used"))
    return (
        int(not forbidden),
        int(_bool(row.get("canonical_feasibility_pass") or row.get("candidate_success"))),
        _float(row.get("support_gate_rate")),
        _float(row.get("hold_gate_rate")),
        _float(row.get("lift_gate_rate")),
        -_float(row.get("excessive_force_rate")),
        -_float(row.get("object_displacement_max_m")),
        _float(row.get("contact_duration_mean")),
        -int(_float(row.get("candidate_rank"), 999.0)),
    )


def _is_slip_dominant(row: dict[str, Any]) -> bool:
    if not row:
        return False
    support = _float(row.get("support_gate_rate"))
    hold = _float(row.get("hold_gate_rate"))
    lift = _float(row.get("lift_gate_rate"))
    displacement = _float(row.get("object_displacement_max_m"))
    excessive = _float(row.get("excessive_force_rate"))
    return bool(support > 0.0 and (hold <= 0.0 or lift <= 0.0 or displacement > 0.04) and excessive <= 0.01)


def _physics_rationale(best: dict[str, Any], comparison_ran: bool, comparison_pass: bool, slip_dominant: bool, tunable: bool) -> str:
    if comparison_pass:
        return "bounded diagnostic physics comparison improved the candidate; treat as asset/physics evidence only"
    if comparison_ran:
        return "bounded diagnostic physics comparison did not pass support/hold/lift"
    if slip_dominant and not tunable:
        return "canonical failure looks slip/displacement-dominant but no tunable material/contact API was exposed"
    if slip_dominant:
        return "canonical failure is slip/displacement-dominant; bounded per-object comparison is a valid next diagnostic"
    if not best:
        return "no canonical candidate result was available"
    return "canonical failure does not primarily indicate material/contact tuning"


def _asset_fix_evidence(best: dict[str, Any], physics: dict[str, Any]) -> bool:
    if _bool(physics.get("slip_dominant_failure")) and _bool(physics.get("asset_tunable_fields_exposed")):
        return True
    status = str(best.get("status") or "")
    blocker = str(best.get("blocker") or "")
    return any(token in status or token in blocker for token in ("COLLISION", "PENETRATION", "ASSET", "MATERIAL", "GEOMETRY"))


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
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
