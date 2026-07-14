"""v92 asset, task setup, and contact calibration report helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json


V92_RUN_MODE = "v92_asset_task_setup_and_contact_calibration"
V92_DECISIONS = {
    "READY_FOR_V93_FEASIBILITY_SEARCH",
    "ASSET_DYNAMICS_BLOCKER",
    "COLLISION_CONTACT_BLOCKER",
    "INITIAL_CONDITION_BLOCKER",
    "FINGER_MAPPING_BLOCKER",
}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def write_v92_task_initial_condition_plan(run_dir: str | Path, parts: list[str] | None = None) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in parts or V80_PARTS:
        rows.append(
            {
                "part_name": part,
                "initial_condition_mode": "table_supported_pickup",
                "support_surface": "Table_subtree_bbox_top",
                "gravity_enabled_required": True,
                "object_dynamic_required": True,
                "reset_only_object_write_allowed": True,
                "object_write_after_reset_allowed": False,
                "pre_contact_displacement_limit_m": 0.02,
                "pre_contact_z_drift_limit_m": 0.02,
                "initial_force_limit_n": 0.05,
                "staging_fallback_mode": "gripper_near_pregrasp_only_if_contact_before_0p02m_drop",
                "success_label_used": False,
            }
        )
    csv_path = write_csv(run_path / "v92_task_initial_condition_plan.csv", rows)
    json_path = write_json(run_path / "v92_task_initial_condition_plan.json", rows)
    md_path = run_path / "v92_task_initial_condition_plan.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v92_task_initial_condition_plan_csv": str(csv_path), "v92_task_initial_condition_plan_json": str(json_path), "v92_task_initial_condition_plan_md": str(md_path)}


def classify_v92_progress(
    run_dir: str | Path,
    *,
    parts: list[str],
    asset_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
    staging_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    backend_summary: dict[str, Any],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    progress_rows: list[dict[str, Any]] = []
    for part in parts:
        asset = _find(asset_rows, part)
        collision = _find(collision_rows, part)
        staging = _find(staging_rows, part)
        sanity = _find(sanity_rows, part)
        mapping = _mapping_for_part(mapping_rows, part)
        dynamic_ok = _bool(asset.get("asset_dynamics_ok"))
        collision_ok = _bool(collision.get("collision_subtree_ok"))
        initial_ok = _bool(staging.get("initial_condition_valid"))
        mapping_ok = _bool(mapping.get("finger_mapping_consistent"))
        two_finger = _bool(sanity.get("calibrated_multi_finger_support_observed"))
        if dynamic_ok and collision_ok and initial_ok and mapping_ok and two_finger:
            decision = "READY_FOR_V93_FEASIBILITY_SEARCH"
            blocker = ""
            next_action = "run_v93_bounded_feasibility_search_no_training"
        elif not dynamic_ok:
            decision = "ASSET_DYNAMICS_BLOCKER"
            blocker = str(asset.get("asset_dynamics_blocker") or "asset_dynamics_failed")
            next_action = "repair_dynamic_rigid_body_gravity_kinematic_mass_or_root_response"
        elif not collision_ok:
            decision = "COLLISION_CONTACT_BLOCKER"
            blocker = str(collision.get("collision_blocker") or "collision_or_contact_subtree_invalid")
            next_action = "repair_collision_subtree_contact_report_or_material_config"
        elif not initial_ok:
            decision = "INITIAL_CONDITION_BLOCKER"
            blocker = str(staging.get("initial_condition_blocker") or "invalid_table_supported_initial_condition")
            next_action = "repair_table_supported_staging_and_precontact_motion"
        elif not mapping_ok:
            decision = "FINGER_MAPPING_BLOCKER"
            blocker = str(mapping.get("finger_mapping_blocker") or "finger_action_sensor_mapping_inconsistent")
            next_action = "repair_action_column_to_sensor_mapping_before_support_gate"
        else:
            decision = "FINGER_MAPPING_BLOCKER"
            blocker = "calibrated_two_finger_support_not_observed"
            next_action = "repair_two_finger_contact_calibration_before_v93"
        progress_rows.append(
            {
                "part_name": part,
                "v92_decision": decision,
                "object_identity_verified": _bool(asset.get("object_identity_verified") or sanity.get("object_identity_verified")),
                "single_simulation_context": _bool(backend_summary.get("single_simulation_context")),
                "gym_make_count": int(backend_summary.get("gym_make_count") or 0),
                "dynamic_asset_ok": dynamic_ok,
                "collision_subtree_ok": collision_ok,
                "initial_condition_valid": initial_ok,
                "pre_contact_displacement_m": _float(staging.get("pre_contact_displacement_m") or sanity.get("pre_contact_displacement_m")),
                "first_contact_step": sanity.get("first_contact_step", ""),
                "finger_mapping_consistent": mapping_ok,
                "calibrated_multi_finger_support_observed": two_finger,
                "peak_force_n": _float(sanity.get("peak_force_n")),
                "mean_force_n": _float(sanity.get("mean_force_n")),
                "excessive_force_rate": _float(sanity.get("excessive_force_rate")),
                "object_displacement_m": _float(sanity.get("object_displacement_m")),
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
                "training_locked": True,
                "status": decision,
                "blocker": blocker,
                "next_action": next_action,
            }
        )
    csv_path = write_csv(run_path / "v92_progress_matrix.csv", progress_rows)
    json_path = write_json(run_path / "v92_progress_matrix.json", progress_rows)
    md_path = run_path / "v92_progress_matrix.md"
    _write_md(md_path, progress_rows)
    root_csv = write_csv(Path.cwd() / "debug_runs/v92_progress_matrix.csv", progress_rows)
    root_md = Path.cwd() / "debug_runs/v92_progress_matrix.md"
    _write_md(root_md, progress_rows)
    return {
        "rows": progress_rows,
        "v92_progress_matrix_csv": str(csv_path),
        "v92_progress_matrix_json": str(json_path),
        "v92_progress_matrix_md": str(md_path),
        "v92_root_progress_matrix_csv": str(root_csv),
        "v92_root_progress_matrix_md": str(root_md),
    }


def write_v92_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
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
    txt_path = run_path / "v92_forbidden_artifact_scan.txt"
    txt_path.write_text(
        ("FORBIDDEN ARTIFACTS FOUND\n" + "\n".join(row["path"] for row in rows) if rows else "No v92 forbidden PPO/BC/checkpoint/final/sticky/video/dataset artifacts found.") + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v92_forbidden_artifact_scan.json", summary)
    return {**summary, "v92_forbidden_artifact_scan_txt": str(txt_path), "v92_forbidden_artifact_scan_json": str(json_path)}


def _find(rows: list[dict[str, Any]], part: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("part_name") == part), {})


def _mapping_for_part(rows: list[dict[str, Any]], part: str) -> dict[str, Any]:
    part_rows = [row for row in rows if row.get("part_name") in {part, "__global__"}]
    if not part_rows:
        return {}
    if any(not _bool(row.get("finger_mapping_consistent")) for row in part_rows):
        bad = next(row for row in part_rows if not _bool(row.get("finger_mapping_consistent")))
        return {"finger_mapping_consistent": False, "finger_mapping_blocker": bad.get("finger_mapping_blocker", "finger_mapping_inconsistent")}
    return {"finger_mapping_consistent": True, "finger_mapping_blocker": ""}


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
