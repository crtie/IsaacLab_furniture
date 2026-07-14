"""v93 collision geometry and safe staging report helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json


V93_RUN_MODE = "v93_collision_geometry_and_safe_staging_repair"
V93_DECISIONS = {
    "READY_FOR_V94_FEASIBILITY_SEARCH",
    "COLLISION_GEOMETRY_BLOCKER",
    "SAFE_STAGING_BLOCKER",
    "ASSET_DYNAMICS_BLOCKER",
    "FINGER_CONTACT_MAPPING_BLOCKER",
}
V93_ACTIVE_FINGER_GROUPS = {
    "Plug2": "23",
    "Screw1": "34",
    "Backrest": "34",
    "Rod": "234",
    "Frame": "234",
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


def write_v93_safe_staging_plan(run_dir: str | Path, parts: list[str] | None = None) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in parts or V80_PARTS:
        rows.append(
            {
                "part_name": part,
                "initial_condition_mode": "table_supported_pickup",
                "support_surface": "Table_subtree_bbox_top",
                "object_center_local_x": -0.20,
                "object_center_local_y": 0.0,
                "object_center_local_z": "",
                "active_finger_group": V93_ACTIVE_FINGER_GROUPS.get(part, "34"),
                "gravity_enabled_required": True,
                "object_dynamic_required": True,
                "reset_only_object_write_allowed": True,
                "object_write_after_reset_allowed": False,
                "pre_contact_displacement_limit_m": 0.02,
                "pre_contact_z_drift_limit_m": 0.02,
                "initial_force_limit_n": 0.05,
                "fallback_mode": "gripper_near_pregrasp_only_if_contact_before_0p02m_motion",
                "success_label_used": False,
            }
        )
    csv_path = write_csv(run_path / "v93_safe_staging_plan.csv", rows)
    json_path = write_json(run_path / "v93_safe_staging_plan.json", rows)
    md_path = run_path / "v93_safe_staging_plan.md"
    _write_md(md_path, rows)
    return {
        "rows": rows,
        "v93_safe_staging_plan_csv": str(csv_path),
        "v93_safe_staging_plan_json": str(json_path),
        "v93_safe_staging_plan_md": str(md_path),
    }


def classify_v93_readiness(
    run_dir: str | Path,
    *,
    parts: list[str],
    native_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
    staging_rows: list[dict[str, Any]],
    screw1_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    backend_summary: dict[str, Any],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    native_shutdown = any(_bool(row.get("native_shutdown_reproduced")) for row in native_rows)
    for part in parts:
        collision = _find(collision_rows, part)
        staging = _find(staging_rows, part)
        support = _find(support_rows, part)
        screw1 = _find(screw1_rows, part)
        object_identity_verified = _bool(collision.get("object_identity_verified"))
        single_context = _bool(backend_summary.get("single_simulation_context")) and int(backend_summary.get("gym_make_count") or 0) == 1
        dynamic_ok = _bool(screw1.get("screw1_post_repair_sanity_ok")) if part == "Screw1" else True
        collision_ok = _bool(collision.get("collision_bbox_ok"))
        staging_ok = _bool(staging.get("initial_condition_valid"))
        support_ok = _bool(support.get("support_gate_post_repair_ok"))
        if object_identity_verified and single_context and dynamic_ok and collision_ok and staging_ok and support_ok and not native_shutdown:
            decision = "READY_FOR_V94_FEASIBILITY_SEARCH"
            blocker = ""
            next_action = "run_v94_no_sticky_feasibility_search"
            ready = True
        elif not object_identity_verified or not single_context:
            decision = "COLLISION_GEOMETRY_BLOCKER"
            blocker = "object_identity_or_single_context_not_verified"
            next_action = "repair_identity_single_context_audit_before_v94"
            ready = False
        elif not dynamic_ok:
            decision = "ASSET_DYNAMICS_BLOCKER"
            blocker = str(screw1.get("blocker") or "dynamic_asset_sanity_failed")
            next_action = "repair_dynamic_rigid_body_response_before_v94"
            ready = False
        elif not collision_ok:
            decision = "COLLISION_GEOMETRY_BLOCKER"
            blocker = str(collision.get("collision_blocker") or "collision_geometry_not_verified")
            next_action = "repair_visual_collision_expected_bbox_alignment"
            ready = False
        elif not staging_ok or native_shutdown:
            decision = "SAFE_STAGING_BLOCKER"
            blocker = "native_shutdown_reproduced" if native_shutdown else str(staging.get("initial_condition_blocker") or "safe_staging_invalid")
            next_action = "repair_reset_lifecycle_safe_staging"
            ready = False
        else:
            decision = "FINGER_CONTACT_MAPPING_BLOCKER"
            blocker = str(support.get("support_gate_blocker") or "finger_contact_sanity_failed")
            next_action = "repair_calibrated_finger_contact_mapping"
            ready = False
        rows.append(
            {
                "part_name": part,
                "v93_decision": decision,
                "ready_for_v94": ready,
                "object_identity_verified": object_identity_verified,
                "single_simulation_context": single_context,
                "gym_make_count": int(backend_summary.get("gym_make_count") or 0),
                "active_usd": collision.get("active_asset_usd", ""),
                "dynamic_asset_ok": dynamic_ok,
                "collision_bbox_ok": collision_ok,
                "visual_collision_bbox_iou": _float(collision.get("visual_collision_bbox_iou")),
                "visual_collision_extent_error_max": _float(collision.get("visual_collision_extent_error_max")),
                "expected_collision_extent_error_max": _float(collision.get("expected_collision_extent_error_max")),
                "safe_staging_ok": staging_ok,
                "native_shutdown_reproduced_or_detected": native_shutdown or _bool(staging.get("native_shutdown_detected")),
                "pre_contact_displacement_m": _float(staging.get("pre_contact_displacement_m")),
                "pre_contact_z_drift_m": _float(staging.get("pre_contact_z_drift_m")),
                "reset_interpenetration_detected": _bool(staging.get("reset_interpenetration_or_impulse_detected")),
                "single_finger_contact_sanity": _bool(support.get("single_finger_contact_sanity")),
                "two_finger_contact_sanity": _bool(support.get("two_finger_contact_sanity")),
                "calibrated_multi_finger_support_observed": _bool(support.get("calibrated_multi_finger_support_observed")),
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
    readiness_csv = write_csv(run_path / "v93_per_object_readiness.csv", rows)
    readiness_json = write_json(run_path / "v93_per_object_readiness.json", rows)
    progress_csv = write_csv(run_path / "v93_progress_matrix.csv", rows)
    progress_md = run_path / "v93_progress_matrix.md"
    _write_md(progress_md, rows)
    root_csv = write_csv(Path.cwd() / "debug_runs/v93_progress_matrix.csv", rows)
    root_md = Path.cwd() / "debug_runs/v93_progress_matrix.md"
    _write_md(root_md, rows)
    return {
        "rows": rows,
        "v93_per_object_readiness_csv": str(readiness_csv),
        "v93_per_object_readiness_json": str(readiness_json),
        "v93_progress_matrix_csv": str(progress_csv),
        "v93_progress_matrix_md": str(progress_md),
        "v93_root_progress_matrix_csv": str(root_csv),
        "v93_root_progress_matrix_md": str(root_md),
    }


def write_v93_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
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
        "checkpoint",
        ".pt",
        "policy_artifact",
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
    txt_path = run_path / "v93_forbidden_artifact_scan.txt"
    txt_path.write_text(
        ("FORBIDDEN ARTIFACTS FOUND\n" + "\n".join(row["path"] for row in rows) if rows else "No v93 forbidden PPO/BC/checkpoint/final/sticky/video/dataset artifacts found.") + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v93_forbidden_artifact_scan.json", summary)
    return {**summary, "v93_forbidden_artifact_scan_txt": str(txt_path), "v93_forbidden_artifact_scan_json": str(json_path)}


def _find(rows: list[dict[str, Any]], part: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("part_name") == part), {})


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
