"""v91 task-setup, dynamics, staging, and controller sanity helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json
from .v90_bottleneck_isolation import default_v90_candidate_plan


V91_RUN_MODE = "v91_task_setup_asset_dynamics_and_controller_sanity"
V91_DECISIONS = {
    "FEASIBLE_NOW_DIAGNOSTIC_ONLY",
    "ASSET_DYNAMICS_BLOCKER",
    "STAGING_OR_TASK_SETUP_BLOCKER",
    "CONTROLLER_CANDIDATE_BLOCKER",
    "CONTACT_MODEL_BLOCKER",
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


def write_v91_feasibility_candidate_plan(run_dir: str | Path, *, max_candidates_per_part: int = 8) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        part_rows = [row for row in default_v90_candidate_plan() if row.get("part_name") == part]
        part_rows = _two_variants_per_family(part_rows)[: int(max_candidates_per_part)]
        for rank, row in enumerate(part_rows):
            item = dict(row)
            item.update(
                {
                    "candidate_rank": rank,
                    "staging_mode": "gripper_near_pregrasp",
                    "support_finger_value": 0.0,
                    "close_steps": 24,
                    "approach_until_contact_max_steps": 36,
                    "hold_steps": int(item.get("hold_steps") or 48),
                    "lift_steps": int(item.get("lift_steps") or 32),
                    "pre_contact_displacement_limit_m": 0.02,
                    "lift_success_threshold_m": 0.005,
                    "source": "v91_bounded_v90_family_retest_no_success_label",
                    "success_label_used": False,
                }
            )
            rows.append(item)
    csv_path = write_csv(run_path / "v91_feasibility_candidate_plan.csv", rows)
    json_path = write_json(run_path / "v91_feasibility_candidate_plan.json", rows)
    return {"rows": rows, "v91_feasibility_candidate_plan_csv": str(csv_path), "v91_feasibility_candidate_plan_json": str(json_path)}


def _two_variants_per_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_family") or ""), []).append(dict(row))
    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        selected.extend(sorted(grouped[family], key=lambda row: int(row.get("family_variant_index", 0) or 0))[:2])
    return sorted(selected, key=lambda row: (str(row.get("candidate_family") or ""), int(row.get("family_variant_index", 0) or 0)))


def classify_v91_decisions(
    run_dir: str | Path,
    *,
    asset_rows: list[dict[str, Any]],
    staging_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        asset = next((row for row in asset_rows if row.get("part_name") == part), {})
        staging_part = [row for row in staging_rows if row.get("part_name") == part]
        candidates = [row for row in candidate_rows if row.get("part_name") == part]
        best = _best_candidate(candidates)
        dynamic_ok = _bool(asset.get("asset_dynamics_ok"))
        staging_ok_any = any(_bool(row.get("staging_valid_for_feasibility")) for row in staging_part)
        feasible = _bool(best.get("v91_feasible_now_diagnostic_only"))
        multi_support = _float(best.get("multi_finger_support_rate")) > 0.0
        best_precontact_displacement = _float(best.get("object_displacement_before_contact_m"))
        best_staging_blocked = (
            str(best.get("status") or "") == "STAGING_OR_TASK_SETUP_BLOCKER"
            or str(best.get("blocker") or "") == "invalid_staging_or_precontact_motion"
            or best_precontact_displacement > 0.02
            or _bool(best.get("reset_interpenetration_or_impulse_detected"))
        )
        if feasible:
            decision = "FEASIBLE_NOW_DIAGNOSTIC_ONLY"
            blocker = ""
            next_action = "use_v91_candidate_as_diagnostic_input_for_later_bc_or_ppo"
        elif not dynamic_ok:
            decision = "ASSET_DYNAMICS_BLOCKER"
            blocker = str(asset.get("asset_dynamics_blocker") or "asset_dynamics_failed")
            next_action = "repair_rigid_body_gravity_kinematic_mass_or_root_velocity_config"
        elif not staging_ok_any:
            decision = "STAGING_OR_TASK_SETUP_BLOCKER"
            blocker = "no_candidate_has_valid_precontact_staging"
            next_action = "repair_reset_staging_table_support_or_gripper_near_pregrasp"
        elif best_staging_blocked:
            decision = "STAGING_OR_TASK_SETUP_BLOCKER"
            blocker = str(best.get("blocker") or "invalid_staging_or_precontact_motion")
            next_action = "repair_reset_staging_table_support_or_gripper_near_pregrasp"
        elif _float(best.get("force_contact_rate")) > 0.0 and not multi_support:
            decision = "CONTACT_MODEL_BLOCKER"
            blocker = "real_contact_exists_but_multi_finger_support_missing"
            next_action = "repair_finger_mask_collision_or_contact_material_model"
        else:
            decision = "CONTROLLER_CANDIDATE_BLOCKER"
            blocker = str(best.get("blocker") or "controller_candidates_failed_hold_or_lift")
            next_action = "add_or_repair_quasistatic_candidate_family"
        rows.append(
            {
                "part_name": part,
                "decision": decision,
                "best_candidate_id": best.get("candidate_id", ""),
                "asset_dynamics_ok": dynamic_ok,
                "staging_valid_any_candidate": staging_ok_any,
                "force_contact_rate": _float(best.get("force_contact_rate")),
                "multi_finger_support_rate": _float(best.get("multi_finger_support_rate")),
                "support_gate_rate": _float(best.get("support_gate_rate")),
                "hold_gate_rate": _float(best.get("hold_gate_rate")),
                "lift_gate_rate": _float(best.get("lift_gate_rate")),
                "object_displacement_before_contact_m": best_precontact_displacement,
                "object_displacement_max_m": _float(best.get("object_displacement_max_m")),
                "lift_delta_z_max_m": _float(best.get("lift_delta_z_max_m")),
                "peak_force_n": _float(best.get("peak_force_n")),
                "object_write_by_policy_detected": _bool(best.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": _bool(best.get("sticky_action_available_to_policy")),
                "fallback_success_used": _bool(best.get("fallback_success_used")),
                "distance_only_success_used": _bool(best.get("distance_only_success_used")),
                "final_success": False,
                "usable_training_row_count": 0,
                "blocker": blocker,
                "next_action": next_action,
            }
        )
    csv_path = write_csv(run_path / "v91_per_object_decision.csv", rows)
    json_path = write_json(run_path / "v91_per_object_decision.json", rows)
    return {"rows": rows, "v91_per_object_decision_csv": str(csv_path), "v91_per_object_decision_json": str(json_path)}


def write_v91_progress_matrix(
    run_dir: str | Path,
    *,
    decision_rows: list[dict[str, Any]],
    backend_summary: dict[str, Any],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        decision = next((row for row in decision_rows if row.get("part_name") == part), {})
        status = str(decision.get("decision") or "CONTROLLER_CANDIDATE_BLOCKER")
        if status not in V91_DECISIONS:
            status = "CONTROLLER_CANDIDATE_BLOCKER"
        rows.append(
            {
                "part_name": part,
                "v91_decision": status,
                "best_candidate_id": decision.get("best_candidate_id", ""),
                "asset_dynamics_ok": _bool(decision.get("asset_dynamics_ok")),
                "staging_valid_any_candidate": _bool(decision.get("staging_valid_any_candidate")),
                "single_simulation_context": bool(backend_summary.get("single_simulation_context")),
                "vector_reset_ok": bool(backend_summary.get("vector_reset_ok")),
                "vector_step_ok": bool(backend_summary.get("vector_step_ok")),
                "force_contact_rate": _float(decision.get("force_contact_rate")),
                "multi_finger_support_rate": _float(decision.get("multi_finger_support_rate")),
                "support_gate_rate": _float(decision.get("support_gate_rate")),
                "hold_gate_rate": _float(decision.get("hold_gate_rate")),
                "lift_gate_rate": _float(decision.get("lift_gate_rate")),
                "object_displacement_before_contact_m": _float(decision.get("object_displacement_before_contact_m")),
                "object_displacement_max_m": _float(decision.get("object_displacement_max_m")),
                "lift_delta_z_max_m": _float(decision.get("lift_delta_z_max_m")),
                "peak_force_n": _float(decision.get("peak_force_n")),
                "ppo_ran": False,
                "bc_ran": False,
                "checkpoint_written": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "dataset_exported": False,
                "usable_training_row_count": 0,
                "final_success": False,
                "status": status,
                "blocker": decision.get("blocker", ""),
                "next_action": decision.get("next_action", ""),
            }
        )
    csv_path = write_csv(run_path / "v91_progress_matrix.csv", rows)
    md_path = run_path / "v91_progress_matrix.md"
    _write_md(md_path, rows)
    root_csv = write_csv(Path.cwd() / "debug_runs/v91_progress_matrix.csv", rows)
    root_md = Path.cwd() / "debug_runs/v91_progress_matrix.md"
    _write_md(root_md, rows)
    return {
        "rows": rows,
        "v91_progress_matrix_csv": str(csv_path),
        "v91_progress_matrix_md": str(md_path),
        "v91_root_progress_matrix_csv": str(root_csv),
        "v91_root_progress_matrix_md": str(root_md),
    }


def write_v91_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
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
    rows = []
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
        "bc_ran": False,
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "checkpoint_written": False,
        "dataset_exported": False,
        "usable_training_row_count": 0,
        "rows": rows,
    }
    txt_path = run_path / "v91_forbidden_artifact_scan.txt"
    txt_path.write_text(
        ("FORBIDDEN ARTIFACTS FOUND\n" + "\n".join(row["path"] for row in rows) if rows else "No v91 forbidden PPO/BC/checkpoint/final/sticky/video/dataset artifacts found.") + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v91_forbidden_artifact_scan.json", summary)
    return {**summary, "v91_forbidden_artifact_scan_txt": str(txt_path), "v91_forbidden_artifact_scan_json": str(json_path)}


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=_candidate_score, default={})


def _candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    forbidden = _bool(row.get("object_write_by_policy_detected")) or _bool(row.get("sticky_action_available_to_policy")) or _bool(row.get("fallback_success_used"))
    return (
        int(not forbidden),
        int(_bool(row.get("v91_feasible_now_diagnostic_only"))),
        _float(row.get("lift_gate_rate")),
        _float(row.get("hold_gate_rate")),
        _float(row.get("multi_finger_support_rate")),
        _float(row.get("support_gate_rate")),
        -_float(row.get("object_displacement_before_contact_m")),
        -_float(row.get("object_displacement_max_m")),
        -_float(row.get("excessive_force_rate")),
    )


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
