"""v94 safe-pregrasp and finger-contact calibration report helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json


V94_RUN_MODE = "v94_safe_pregrasp_and_finger_contact_calibration"
V93_RUN_DIR = Path("debug_runs/v93_collision_geometry_and_safe_staging_repair")
V92_RUN_DIR = Path("debug_runs/v92_asset_task_setup_and_contact_calibration")
V94_DECISIONS = {
    "READY_FOR_V95_FEASIBILITY_SEARCH",
    "SAFE_PREGRASP_BLOCKER",
    "WRIST_ACTION_MAPPING_BLOCKER",
    "FINGER_MOTION_BLOCKER",
    "FINGER_SENSOR_MAPPING_BLOCKER",
    "ACTUAL_OBJECT_CONTACT_BLOCKER",
    "COLLISION_PROXY_BLOCKER",
}


def write_v94_v93_blocker_reclassification(
    run_dir: str | Path,
    *,
    parts: list[str] | None = None,
    v93_run_dir: str | Path = V93_RUN_DIR,
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    v93_path = Path(v93_run_dir)
    progress_rows = _read_csv(v93_path / "v93_progress_matrix.csv")
    contact_rows = _read_csv(v93_path / "contact_api_audit.csv")
    sanity_rows = _read_csv(v93_path / "v93_post_collision_contact_sanity.csv")
    rows: list[dict[str, Any]] = []
    for part in parts or list(V80_PARTS):
        progress = _find(progress_rows, part)
        contact = _best_contact_row(contact_rows, part)
        sanity = _find(sanity_rows, part)
        min_distance = _float(contact.get("fingertip_object_distance_min_m"), 1.0)
        surface_distance = max(0.0, min_distance - 0.015)
        force_peak = max(
            _float(contact.get("force_contact_peak_n")),
            _float(contact.get("force_contact_probe_peak_n")),
            _float(sanity.get("force_peak_n")),
        )
        old_decision = str(progress.get("v93_decision") or progress.get("status") or "")
        far = min_distance > 0.015 and force_peak <= 0.05
        if old_decision == "FINGER_CONTACT_MAPPING_BLOCKER" and far:
            new_class = "no_contact_because_fingers_far_from_object"
            true_mapping_failure = False
            next_action = "repair_reset_safe_pregrasp_alignment_before_finger_mapping_claim"
        elif old_decision == "FINGER_CONTACT_MAPPING_BLOCKER":
            new_class = "possible_finger_mapping_failure_after_contact_range"
            true_mapping_failure = True
            next_action = "run_diagnostic_target_finger_sensor_mapping"
        else:
            new_class = "preserve_v93_blocker"
            true_mapping_failure = old_decision == "FINGER_CONTACT_MAPPING_BLOCKER"
            next_action = "repair_v93_blocker_before_v95"
        rows.append(
            {
                "part_name": part,
                "v93_decision": old_decision,
                "v93_blocker": progress.get("blocker", ""),
                "v93_force_peak_n": force_peak,
                "v93_fingertip_object_distance_min_m": min_distance,
                "v93_surface_adjusted_distance_estimate_m": surface_distance,
                "v93_single_finger_contact_sanity": _bool(progress.get("single_finger_contact_sanity")),
                "v93_two_finger_contact_sanity": _bool(progress.get("two_finger_contact_sanity")),
                "v94_reclassification": new_class,
                "true_finger_sensor_mapping_failure_evidence": true_mapping_failure,
                "distance_not_mapping_failure": bool(far),
                "success_label_used": False,
                "next_action": next_action,
            }
        )
    csv_path = write_csv(run_path / "v94_v93_blocker_reclassification.csv", rows)
    json_path = write_json(run_path / "v94_v93_blocker_reclassification.json", rows)
    md_path = run_path / "v94_v93_blocker_reclassification.md"
    _write_md(md_path, rows)
    return {
        "rows": rows,
        "v94_v93_blocker_reclassification_csv": str(csv_path),
        "v94_v93_blocker_reclassification_json": str(json_path),
        "v94_v93_blocker_reclassification_md": str(md_path),
    }


def write_v94_collision_proxy_decision(
    run_dir: str | Path,
    *,
    parts: list[str] | None = None,
    v92_run_dir: str | Path = V92_RUN_DIR,
    v93_run_dir: str | Path = V93_RUN_DIR,
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    v92_rows = _read_csv(Path(v92_run_dir) / "v92_collision_subtree_audit.csv")
    v93_bbox_rows = _read_csv(Path(v93_run_dir) / "v93_collision_world_bbox_audit.csv")
    v93_repair_rows = _read_csv(Path(v93_run_dir) / "v93_collision_repair_report.csv")
    rows: list[dict[str, Any]] = []
    for part in parts or list(V80_PARTS):
        native = _find(v92_rows, part)
        bbox = _find(v93_bbox_rows, part)
        repairs = [row for row in v93_repair_rows if row.get("part_name") == part]
        native_ok = _bool(native.get("collision_subtree_ok"))
        v93_ok = _bool(bbox.get("collision_bbox_ok"))
        proxy_applied = any(_bool(row.get("repair_applied")) for row in repairs)
        expected_error = _float(bbox.get("expected_collision_extent_error_max"), 1.0)
        visual_error = _float(bbox.get("visual_collision_extent_error_max"), 1.0)
        oversize_risk = bool(proxy_applied and (expected_error > 1.25 or visual_error > 1.25))
        acceptable = bool((native_ok or v93_ok) and not oversize_risk)
        rows.append(
            {
                "part_name": part,
                "native_usd_collision_adequate": native_ok,
                "v92_native_collision_blocker": native.get("collision_blocker", ""),
                "v93_collision_bbox_ok": v93_ok,
                "v93_runtime_proxy_required": bool(proxy_applied and not native_ok),
                "v93_runtime_proxy_applied": proxy_applied,
                "persistent_usd_edit_recommended": bool(proxy_applied and not native_ok),
                "proxy_scope": "diagnostic_unified_env_runtime_only" if proxy_applied else "",
                "proxy_acceptable_for_unified_env_diagnostics": acceptable,
                "oversize_or_offset_risk": oversize_risk,
                "visual_collision_extent_error_max": visual_error,
                "expected_collision_extent_error_max": expected_error,
                "collision_proxy_decision_ok": acceptable,
                "collision_proxy_blocker": "" if acceptable else "collision_proxy_or_native_collision_untrustworthy",
                "sticky_or_adhesion_added": False,
                "success_label_used": False,
            }
        )
    csv_path = write_csv(run_path / "v94_collision_proxy_decision.csv", rows)
    json_path = write_json(run_path / "v94_collision_proxy_decision.json", rows)
    md_path = run_path / "v94_collision_proxy_decision.md"
    _write_md(md_path, rows)
    return {
        "rows": rows,
        "v94_collision_proxy_decision_csv": str(csv_path),
        "v94_collision_proxy_decision_json": str(json_path),
        "v94_collision_proxy_decision_md": str(md_path),
    }


def classify_v94_readiness(
    run_dir: str | Path,
    *,
    parts: list[str],
    pregrasp_rows: list[dict[str, Any]],
    wrist_rows: list[dict[str, Any]],
    finger_rows: list[dict[str, Any]],
    sensor_rows: list[dict[str, Any]],
    object_contact_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    backend_summary: dict[str, Any],
) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for part in parts:
        pre = _find(pregrasp_rows, part)
        proxy = _find(proxy_rows, part)
        part_wrist = [row for row in wrist_rows if row.get("part_name") == part]
        part_fingers = [row for row in finger_rows if row.get("part_name") == part]
        part_sensor = [row for row in sensor_rows if row.get("part_name") == part]
        part_contact = [row for row in object_contact_rows if row.get("part_name") == part]
        single_contact = next((row for row in part_contact if row.get("contact_calibration_mode") == "single_finger"), {})
        two_contact = next((row for row in part_contact if row.get("contact_calibration_mode") == "two_finger"), {})
        wrist_ok = bool(part_wrist) and all(_bool(row.get("wrist_action_mapping_ok")) for row in part_wrist)
        finger_ok = bool(part_fingers) and all(_bool(row.get("finger_motion_ok")) for row in part_fingers)
        sensor_ok = bool(part_sensor) and all(_bool(row.get("finger_sensor_mapping_consistent")) for row in part_sensor)
        actual_ok = bool(_bool(single_contact.get("actual_object_contact_ok")) and _bool(two_contact.get("actual_object_contact_ok")))
        proxy_ok = _bool(proxy.get("collision_proxy_decision_ok"))
        pre_ok = _bool(pre.get("safe_pregrasp_ok"))
        single_context = _bool(backend_summary.get("single_simulation_context")) and int(backend_summary.get("gym_make_count") or 0) == 1
        identity_ok = _bool(pre.get("object_identity_verified")) or any(_bool(row.get("object_identity_verified")) for row in pregrasp_rows if row.get("part_name") == part)
        if not pre_ok or not single_context or not identity_ok:
            decision = "SAFE_PREGRASP_BLOCKER"
            blocker = pre.get("safe_pregrasp_blocker") or "identity_single_context_or_safe_pregrasp_failed"
            next_action = "repair_reset_safe_object_hand_pregrasp_alignment"
        elif not wrist_ok:
            decision = "WRIST_ACTION_MAPPING_BLOCKER"
            blocker = _first_blocker(part_wrist, "blocker", "wrist_action_mapping_failed")
            next_action = "repair_wrist_residual_axis_mapping_before_v95"
        elif not finger_ok:
            decision = "FINGER_MOTION_BLOCKER"
            blocker = _first_blocker(part_fingers, "finger_motion_blocker", "finger_motion_failed")
            next_action = "repair_finger_actuation_before_sensor_mapping"
        elif not sensor_ok:
            decision = "FINGER_SENSOR_MAPPING_BLOCKER"
            blocker = _first_blocker(part_sensor, "finger_sensor_mapping_blocker", "finger_sensor_mapping_inconsistent")
            next_action = "repair_logical_finger_to_force_sensor_mapping"
        elif not actual_ok:
            decision = "ACTUAL_OBJECT_CONTACT_BLOCKER"
            blocker = _first_blocker(part_contact, "actual_object_contact_blocker", "actual_object_contact_not_observed")
            next_action = "repair_pregrasp_or_collision_until_real_object_contact_passes"
        elif not proxy_ok:
            decision = "COLLISION_PROXY_BLOCKER"
            blocker = proxy.get("collision_proxy_blocker") or "collision_proxy_decision_failed"
            next_action = "repair_native_usd_collision_or_proxy_size_before_v95"
        else:
            decision = "READY_FOR_V95_FEASIBILITY_SEARCH"
            blocker = ""
            next_action = "run_v95_no_sticky_feasibility_search"
        ready = decision == "READY_FOR_V95_FEASIBILITY_SEARCH"
        rows.append(
            {
                "part_name": part,
                "v94_decision": decision,
                "ready_for_v95": ready,
                "object_identity_verified": identity_ok,
                "single_simulation_context": single_context,
                "gym_make_count": int(backend_summary.get("gym_make_count") or 0),
                "safe_pregrasp_ok": pre_ok,
                "fingertip_object_distance_min_m": pre.get("fingertip_object_surface_distance_min_m", ""),
                "pre_contact_displacement_m": pre.get("pre_contact_displacement_m", ""),
                "pre_contact_z_drift_m": pre.get("pre_contact_z_drift_m", ""),
                "initial_force_peak_n": pre.get("initial_force_peak_n", ""),
                "reset_interpenetration_detected": _bool(pre.get("reset_interpenetration_or_impulse_detected")),
                "wrist_action_mapping_ok": wrist_ok,
                "finger_motion_ok": finger_ok,
                "diagnostic_sensor_mapping_ok": sensor_ok,
                "actual_object_single_finger_contact_ok": _bool(single_contact.get("actual_object_contact_ok")),
                "actual_object_two_finger_contact_ok": _bool(two_contact.get("actual_object_contact_ok")),
                "actual_object_force_peak_n": max(_float(single_contact.get("force_peak_n")), _float(two_contact.get("force_peak_n"))),
                "actual_object_active_force_count": max(int(single_contact.get("active_force_count") or 0), int(two_contact.get("active_force_count") or 0)),
                "collision_proxy_decision_ok": proxy_ok,
                "native_usd_collision_adequate": _bool(proxy.get("native_usd_collision_adequate")),
                "v93_runtime_proxy_required": _bool(proxy.get("v93_runtime_proxy_required")),
                "object_write_by_policy_detected": _bool(backend_summary.get("object_write_by_policy_detected")),
                "sticky_action_available_to_policy": _bool(backend_summary.get("sticky_action_available_to_policy")),
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
    readiness_csv = write_csv(run_path / "v94_per_object_readiness.csv", rows)
    readiness_json = write_json(run_path / "v94_per_object_readiness.json", rows)
    progress_csv = write_csv(run_path / "v94_progress_matrix.csv", rows)
    progress_md = run_path / "v94_progress_matrix.md"
    _write_md(progress_md, rows)
    root_csv = write_csv(Path.cwd() / "debug_runs/v94_progress_matrix.csv", rows)
    root_md = Path.cwd() / "debug_runs/v94_progress_matrix.md"
    _write_md(root_md, rows)
    return {
        "rows": rows,
        "v94_per_object_readiness_csv": str(readiness_csv),
        "v94_per_object_readiness_json": str(readiness_json),
        "v94_progress_matrix_csv": str(progress_csv),
        "v94_progress_matrix_md": str(progress_md),
        "v94_root_progress_matrix_csv": str(root_csv),
        "v94_root_progress_matrix_md": str(root_md),
    }


def write_v94_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
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
    txt_path = run_path / "v94_forbidden_artifact_scan.txt"
    txt_path.write_text(
        (
            "FORBIDDEN ARTIFACTS FOUND\n" + "\n".join(row["path"] for row in rows)
            if rows
            else "No v94 forbidden PPO/BC/checkpoint/final/sticky/video/dataset artifacts found."
        )
        + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v94_forbidden_artifact_scan.json", summary)
    return {**summary, "v94_forbidden_artifact_scan_txt": str(txt_path), "v94_forbidden_artifact_scan_json": str(json_path)}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except Exception:
        return []


def _find(rows: list[dict[str, Any]], part: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("part_name") == part), {})


def _best_contact_row(rows: list[dict[str, Any]], part: str) -> dict[str, Any]:
    part_rows = [row for row in rows if row.get("part_name") == part]
    if not part_rows:
        return {}
    return min(part_rows, key=lambda row: _float(row.get("fingertip_object_distance_min_m"), 1.0))


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


def _first_blocker(rows: list[dict[str, Any]], key: str, default: str) -> str:
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            return value
    return default


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
