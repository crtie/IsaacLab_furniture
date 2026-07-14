"""Minimal executable workcell gate for v95.

This module intentionally stays thin: it configures reset-only pregrasp plans,
then uses the existing single-context backend, env reset lifecycle, contact
manager, and v82 action mapper to run live wrist/finger/contact probes.
"""

from __future__ import annotations

import math
import os
import time
import copy
from pathlib import Path
from typing import Any

from .single_context_backend import (
    V83_PARTS,
    V85_FINGER_GROUPS,
    V85_PART_GEOMETRY,
    IsaacUnifiedSingleContextBackend,
    SingleContextSlot,
    find_prim_by_suffix_for_env,
    _int_field,
    _tensor_row,
    _to_float,
    _v91_active_finger_indices,
    _v91_apply_finger_mask,
    _v91_object_state_for_slot,
    _v91_state_delta,
    _v92_force_list,
    _v92_joint_delta,
    _v92_joint_pos,
    _v92_max_index,
    _v92_support_half_height_m,
    _v92_table_top_by_env,
    _v92_tip_delta,
    _v92_tip_positions,
    _v94_distance_for_slot,
    _v94_fingertip_midpoints,
    _v94_support_half_height_m,
    _v94_vec_delta,
)
from .v80_reports import V80_PARTS, ensure_run_dir, write_csv, write_json, write_jsonl


V95_RUN_MODE = "v95_minimal_unified_workcell_repair"
V95_OUTPUT_DIR = "debug_runs/v95_minimal_unified_workcell_repair"
V95_DECISIONS = {
    "WORKCELL_READY_FOR_FEASIBILITY_CONTROLLER",
    "ACTIVE_OBJECT_RESET_BLOCKER",
    "PREGRASP_PLAN_NOT_APPLIED",
    "LEGACY_PLUG2_RESET_PATH_BLOCKER",
    "WRIST_ACTION_MAPPING_BLOCKER",
    "FINGER_MOTION_BLOCKER",
    "ACTUAL_OBJECT_CONTACT_BLOCKER",
    "NATIVE_COLLISION_ASSET_BLOCKER",
    "NATIVE_SHUTDOWN_OR_RESET_LIFECYCLE_BLOCKER",
}
V95_WRIST_RESET_SETTLE_STEPS = 8

_FORBIDDEN_ARTIFACT_PATTERNS = (
    ".mp4",
    ".pt",
    "checkpoint",
    "rsl",
    "final_replay",
    "training_row",
    "dataset",
    "sticky_eval",
    "sticky_success",
)
_FORBIDDEN_SOURCE_TOKENS = (
    "OnPolicyRunner(",
    "train_v86",
    "train_v87",
    "train_v88",
    "train_v89",
    "evaluate_unified_policy(",
    "evaluate_v86",
    "evaluate_v87",
    "evaluate_v88",
    "evaluate_v89",
    "run_v94_pregrasp_alignment_audit(",
    "run_v93_collision_geometry_audit_and_repair(",
    "write_final",
    "export_bc_dataset",
    "usable_training_row_count = 1",
)


def write_v95_workcell_code_freeze(run_dir: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Write the v95 code-freeze contract and current large-file line counts."""

    run_path = ensure_run_dir(run_dir)
    root = Path(repo_root)
    touched_files = [
        "scripts/environments/run_v81_physical_backend_grasp_rl.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/chair2_env.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_floating_chair2_env.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_unified_five_object_env.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/single_context_backend.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/minimal_workcell_gate.py",
    ]
    rows = []
    for rel in touched_files:
        path = root / rel
        rows.append(
            {
                "file": rel,
                "exists": path.exists(),
                "line_count": _line_count(path),
                "allowed_v95_touch": True,
            }
        )
    payload = {
        "run_mode": V95_RUN_MODE,
        "goal": "minimal_unified_workcell_repair_only",
        "code_freeze": True,
        "large_file_line_counts": rows,
        "allowed_touched_files": touched_files,
        "forbidden_next_step": "do_not_add_v96_v97_report_modules_until_workcell_gate_passes",
        "artifact_completeness_is_physical_success": False,
        "ppo_allowed": False,
        "bc_allowed": False,
        "candidate_search_allowed": False,
        "sticky_allowed": False,
        "video_allowed": False,
        "checkpoint_allowed": False,
        "dataset_export_allowed": False,
        "training_rows_allowed": False,
    }
    json_path = write_json(run_path / "v95_workcell_code_freeze.json", payload)
    md_path = run_path / "v95_workcell_code_freeze.md"
    lines = [
        "# v95 Workcell Code Freeze",
        "",
        "- Scope: minimal reset/pregrasp/wrist/finger/contact gate only.",
        "- Forbidden: PPO, BC, candidate search, sticky, final replay/video, checkpoints, dataset/training rows.",
        "- Artifact completeness is not physical success.",
        "",
        "| file | line_count |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['file']} | {row['line_count']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"payload": payload, "v95_workcell_code_freeze_json": str(json_path), "v95_workcell_code_freeze_md": str(md_path)}


def build_v95_pregrasp_plan(backend: IsaacUnifiedSingleContextBackend) -> list[dict[str, Any]]:
    """Build one reset-only table-supported pregrasp row per backend slot."""

    table_tops = _v92_table_top_by_env(backend)
    rows: list[dict[str, Any]] = []
    for slot in backend.slots:
        part_name = slot.part_name
        table_top = _safe_table_top(table_tops.get(slot.local_env_index, table_tops.get(0, 0.72)))
        half_height, half_source = _v95_support_half_height_m(backend, slot)
        center_x, center_y = _object_xy_for_slot(slot)
        center_z = max(0.02, min(1.35, table_top + half_height + 0.003))
        active_group = V85_FINGER_GROUPS.get(part_name, ("34",))[0]
        hand_target = _safe_hand_target()
        rows.append(
            {
                "part_name": part_name,
                "env_index": slot.local_env_index,
                "local_env_index": slot.local_env_index,
                "slot_index": slot.global_env_index,
                "expected_slot_part_name": V83_PARTS[slot.local_env_index % len(V83_PARTS)],
                "initial_condition_mode": "table_supported_pickup",
                "support_surface": "Table_world_bbox_top",
                "table_top_z_m": table_top,
                "object_support_half_height_m": half_height,
                "object_support_half_height_source": half_source,
                "object_center_local_x": center_x,
                "object_center_local_y": center_y,
                "object_center_local_z": center_z,
                "object_center_local_xyz": [center_x, center_y, center_z],
                "object_quat_wxyz": "",
                "hand_target_local_x": hand_target[0],
                "hand_target_local_y": hand_target[1],
                "hand_target_local_z": hand_target[2],
                "hand_target_local_xyz": hand_target,
                "hand_target_quat_wxyz": "",
                "hand_quat_source": "active_overhand_from_part",
                "hand_target_source": "v95_stage_a_safe_hand",
                "active_finger_group": active_group,
                "reset_only_object_write_allowed": True,
                "reset_only_hand_write_allowed": True,
                "object_write_after_reset_allowed": False,
                "hand_write_after_reset_allowed": False,
                "pre_contact_displacement_limit_m": 0.02,
                "pre_contact_z_drift_limit_m": 0.02,
                "initial_force_limit_n": 0.05,
                "pregrasp_distance_target_m": 0.015,
                "native_collision_proxy_allowed_for_pass": False,
                "success_label_used": False,
            }
        )
    return rows


def run_v95_minimal_workcell_gate(
    run_dir: str | Path,
    backend: IsaacUnifiedSingleContextBackend,
    *,
    repo_root: str | Path,
    parts: list[str] | None = None,
) -> dict[str, Any]:
    """Run the live v95 workcell gate and write required artifacts."""

    run_path = ensure_run_dir(run_dir)
    selected_parts = parts or list(V80_PARTS)
    started_path = write_json(
        run_path / "v95_live_probe_started.json",
        {
            "run_mode": V95_RUN_MODE,
            "started_unix_s": time.time(),
            "live_probe_required": True,
            "fallback_live_probe_disabled_path_allowed": False,
            "v93_collision_proxy_enabled": _env_enabled("WUJI_V93_COLLISION_REPAIR"),
            "v94_diagnostic_pads_enabled": _env_enabled("WUJI_V94_DIAGNOSTIC_PADS"),
        },
    )
    initial_plan_rows = build_v95_pregrasp_plan(backend)
    backend.clear_v93_safe_staging()
    backend.clear_v94_pregrasp()
    backend.object_write_by_policy_detected = False
    backend.sticky_action_available_to_policy = False
    backend.proxy_action_available_to_policy = False
    backend.route_selection_available_to_policy = False
    calibration = _calibrate_v95_pregrasp(backend, initial_plan_rows)
    plan_rows = calibration["plan_rows"]
    reset_rows = calibration["reset_rows"]
    hand_rows = calibration["hand_rows"]
    sample_rows = calibration["sample_rows"]
    trace_rows = calibration["trace_rows"]
    active_rows = _active_object_state_rows(backend)
    reset_lifecycle_rows = _reset_lifecycle_rows(backend, reset_rows, hand_rows, sample_rows)
    pregrasp_rows = _pregrasp_application_rows_from_sample(backend, plan_rows, reset_rows, hand_rows, sample_rows)
    pregrasp_by_part = {str(row.get("part_name") or ""): dict(row) for row in pregrasp_rows}
    write_csv(run_path / "v95_active_object_state_audit.csv", active_rows)
    write_json(run_path / "v95_active_object_state_audit.json", active_rows)
    write_csv(run_path / "v95_reset_lifecycle_gate.csv", reset_lifecycle_rows)
    write_json(run_path / "v95_reset_lifecycle_gate.json", reset_lifecycle_rows)
    write_csv(run_path / "v95_pregrasp_plan_application.csv", pregrasp_rows)
    write_json(run_path / "v95_pregrasp_plan_application.json", pregrasp_rows)
    trace_csv = write_csv(run_path / "v95_pregrasp_calibration_trace.csv", trace_rows)
    trace_json = write_json(run_path / "v95_pregrasp_calibration_trace.json", trace_rows)

    wrist_rows = _run_live_wrist_calibration(backend, plan_rows)
    write_csv(run_path / "v95_live_wrist_action_calibration.csv", wrist_rows)
    write_json(run_path / "v95_live_wrist_action_calibration.json", wrist_rows)
    finger_rows = _run_live_finger_motion_calibration(backend, plan_rows)
    write_csv(run_path / "v95_live_finger_motion_calibration.csv", finger_rows)
    write_json(run_path / "v95_live_finger_motion_calibration.json", finger_rows)
    screw1_asset_rows = _screw1_native_contact_asset_audit(backend)
    screw1_pre_trace_rows: list[dict[str, Any]] = []
    screw1_pre_summary_rows = [
        {
            "part_name": "Screw1",
            "status": "NOT_EXECUTED",
            "reason": "v95_target_filtered_contact_acquisition_replaces_signed_distance_truth_triage",
            "distance_only_success_used": False,
        }
    ]
    last_mile = {
        "plan_rows": plan_rows,
        "trace_rows": list(screw1_pre_trace_rows),
        "summary_rows": list(screw1_pre_summary_rows),
        "pregrasp_trace_rows": [],
        "correction_rows": [],
        "plan_changed": False,
        "skipped_reason": "v95_runtime_contact_truth_stack_repair_pauses_screw1_last_mile",
    }
    plan_rows = last_mile["plan_rows"]
    screw1_trace_rows = list(last_mile["trace_rows"])
    trace_rows.extend(last_mile.get("pregrasp_trace_rows", []))
    if bool(last_mile.get("plan_changed")):
        final_sample = _sample_v95_plan(
            backend,
            plan_rows,
            phase="screw1_last_mile_final_reset",
            iteration=0,
            settle_steps=2,
        )
        reset_rows = final_sample["reset_rows"]
        hand_rows = final_sample["hand_rows"]
        sample_rows = final_sample["sample_rows"]
        trace_rows.extend(final_sample["trace_rows"])
        reset_lifecycle_rows = _reset_lifecycle_rows(backend, reset_rows, hand_rows, sample_rows)
        pregrasp_rows = _pregrasp_application_rows_from_sample(backend, plan_rows, reset_rows, hand_rows, sample_rows)
        pregrasp_by_part = {str(row.get("part_name") or ""): dict(row) for row in pregrasp_rows}
        write_csv(run_path / "v95_reset_lifecycle_gate.csv", reset_lifecycle_rows)
        write_json(run_path / "v95_reset_lifecycle_gate.json", reset_lifecycle_rows)
        write_csv(run_path / "v95_pregrasp_plan_application.csv", pregrasp_rows)
        write_json(run_path / "v95_pregrasp_plan_application.json", pregrasp_rows)
        trace_csv = write_csv(run_path / "v95_pregrasp_calibration_trace.csv", trace_rows)
        trace_json = write_json(run_path / "v95_pregrasp_calibration_trace.json", trace_rows)
        wrist_rows = _run_live_wrist_calibration(backend, plan_rows)
        write_csv(run_path / "v95_live_wrist_action_calibration.csv", wrist_rows)
        write_json(run_path / "v95_live_wrist_action_calibration.json", wrist_rows)
        finger_rows = _run_live_finger_motion_calibration(backend, plan_rows)
        write_csv(run_path / "v95_live_finger_motion_calibration.csv", finger_rows)
        write_json(run_path / "v95_live_finger_motion_calibration.json", finger_rows)

    sensor_self_check_rows = [
        {
            "part_name": "Screw1",
            "status": "NOT_EXECUTED",
            "reason": "v95_target_filtered_contact_acquisition_uses_target_force_matrix_preflight",
            "object_contact_success_evidence_used": False,
            "distance_only_success_used": False,
        }
    ]
    native_pair_rows = _native_pair_contact_sanity_rows(sensor_self_check_rows)
    screw1_filter_coverage_rows = _screw1_target_filter_coverage_audit(backend)
    filtered_structure_rows = _filtered_sensor_structure_audit(backend)
    finger_contact_body_audit_rows = _finger_contact_body_audit(backend, filtered_structure_rows)
    cube_canary_rows = _run_cube_cube_contact_canary(backend)
    finger_reference_canary_rows = _run_finger_reference_contact_canary(
        backend,
        plan_rows,
        structure_rows=filtered_structure_rows,
        body_audit_rows=finger_contact_body_audit_rows,
        cube_canary_rows=cube_canary_rows,
    )
    finger_contact_body_audit_rows = _merge_finger_contact_body_audit_with_canary(
        finger_contact_body_audit_rows,
        finger_reference_canary_rows,
    )
    canary_ok, canary_blocker = _v95_canary_decision(
        filtered_structure_rows,
        cube_canary_rows,
        finger_reference_canary_rows,
    )
    reference_filtered_rows = [
        {
            "part_name": "Screw1",
            "status": "CANARY_PASSED" if canary_ok else "CANARY_FAILED",
            "reason": "v95_filtered_sensor_canary_replaces_reference_filtered_calibration_this_round",
            "reference_filtered_calibration_ok": bool(canary_ok),
            "canary_ok": bool(canary_ok),
            "blocker": "" if canary_ok else canary_blocker,
            "object_contact_success_evidence_used": False,
            "training_locked": True,
        }
    ]
    screw1_reference_contact_rows = _run_screw1_reference_contact_canary(
        backend,
        plan_rows,
        canary_ok=canary_ok,
        canary_blocker=canary_blocker,
        cube_canary_rows=cube_canary_rows,
        finger_reference_canary_rows=finger_reference_canary_rows,
        filter_coverage_rows=screw1_filter_coverage_rows,
    )
    screw1_reference_ok = _screw1_reference_contact_ok(screw1_reference_contact_rows)
    screw1_reference_status = _screw1_reference_contact_status(screw1_reference_contact_rows)
    if canary_ok and screw1_reference_ok:
        screw1_forced_contact_rows = _run_screw1_forced_target_contact_calibration(
            backend,
            plan_rows,
            finger_rows=finger_rows,
            finger_reference_canary_rows=finger_reference_canary_rows,
            reference_filtered_rows=reference_filtered_rows,
            filter_coverage_rows=screw1_filter_coverage_rows,
        )
        screw1_mirror_rows = _run_screw1_object_side_mirror_check(
            backend,
            forced_contact_rows=screw1_forced_contact_rows,
        )
    elif canary_ok:
        screw1_forced_contact_rows = [
            {
                "part_name": "Screw1",
                "status": "NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED",
                "forced_contact_status": "NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED",
                "reason": "screw1_reference_canary_failed_before_finger_forced_contact",
                "canary_ok": canary_ok,
                "screw1_reference_contact_canary_ok": False,
                "screw1_reference_contact_status": screw1_reference_status,
                "blocker": screw1_reference_status,
                "object_contact_success_evidence_used": False,
                "training_locked": True,
            }
        ]
        screw1_mirror_rows = [
            {
                "part_name": "Screw1",
                "status": "NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED",
                "reason": "object_side_mirror_check_blocked_by_screw1_reference_canary",
                "canary_ok": canary_ok,
                "screw1_reference_contact_canary_ok": False,
                "screw1_reference_contact_status": screw1_reference_status,
                "blocker": screw1_reference_status,
                "object_contact_success_evidence_used": False,
                "training_locked": True,
            }
        ]
    else:
        screw1_forced_contact_rows = [
            {
                "part_name": "Screw1",
                "status": "NOT_EXECUTED_CANARY_BLOCKED",
                "forced_contact_status": "NOT_EXECUTED_CANARY_BLOCKED",
                "reason": "canary_failed_before_screw1_forced_target_contact",
                "canary_ok": canary_ok,
                "blocker": canary_blocker,
                "object_contact_success_evidence_used": False,
                "training_locked": True,
            }
        ]
        screw1_mirror_rows = [
            {
                "part_name": "Screw1",
                "status": "NOT_EXECUTED_CANARY_BLOCKED",
                "reason": "object_side_mirror_check_blocked_by_canary",
                "canary_ok": canary_ok,
                "blocker": canary_blocker,
                "object_contact_success_evidence_used": False,
                "training_locked": True,
            }
        ]
    forced_status = _screw1_forced_contact_status(screw1_forced_contact_rows)
    forced_ok = _screw1_forced_contact_ok(screw1_forced_contact_rows)
    pair_contact_matrix_rows = _pair_contact_matrix_rows(
        cube_canary_rows=cube_canary_rows,
        finger_reference_canary_rows=finger_reference_canary_rows,
        screw1_reference_contact_rows=screw1_reference_contact_rows,
        screw1_forced_contact_rows=screw1_forced_contact_rows,
    )
    if not canary_ok:
        contact_rows, acquisition_trace_rows = _canary_blocked_contact_rows(
            backend,
            plan_rows,
            blocker=canary_blocker,
            canary_ok=False,
        )
    elif not screw1_reference_ok:
        contact_rows, acquisition_trace_rows = _forced_contact_blocked_contact_rows(
            backend,
            plan_rows,
            blocker="NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED",
            forced_contact_rows=screw1_forced_contact_rows,
        )
    else:
        contact_rows, acquisition_trace_rows = _forced_contact_blocked_contact_rows(
            backend,
            plan_rows,
            blocker=forced_status,
            forced_contact_rows=screw1_forced_contact_rows,
        )
    screw1_final_trace_rows: list[dict[str, Any]] = []
    screw1_summary_rows = [
        {
            "part_name": "Screw1",
            "status": forced_status if screw1_reference_ok else screw1_reference_status,
            "reason": "screw1_reference_contact_precedes_forced_and_ordinary_acquisition",
            "screw1_reference_contact_canary_ok": screw1_reference_ok,
            "screw1_reference_contact_status": screw1_reference_status,
            "forced_contact_calibration_ok": forced_ok,
            "ordinary_acquisition_executed_after_forced_ok": False,
            "ordinary_acquisition_next_allowed": bool(forced_ok),
            "distance_only_success_used": False,
            "object_contact_success_evidence_used": False,
            "training_locked": True,
        }
    ]
    screw1_trace_rows.extend(screw1_final_trace_rows)
    write_csv(run_path / "v95_actual_object_contact_gate.csv", contact_rows)
    write_json(run_path / "v95_actual_object_contact_gate.json", contact_rows)
    acquisition_trace_csv = write_csv(run_path / "v95_contact_acquisition_trace.csv", acquisition_trace_rows)
    acquisition_trace_jsonl = write_jsonl(run_path / "v95_contact_acquisition_trace.jsonl", acquisition_trace_rows)
    screw1_trace_csv = write_csv(run_path / "v95_screw1_contact_truth_trace.csv", screw1_trace_rows)
    screw1_trace_jsonl = write_jsonl(run_path / "v95_screw1_contact_truth_trace.jsonl", screw1_trace_rows)
    screw1_summary_csv = write_csv(run_path / "v95_screw1_contact_truth_summary.csv", screw1_summary_rows)
    screw1_summary_json = write_json(run_path / "v95_screw1_contact_truth_summary.json", screw1_summary_rows)
    sensor_self_check_csv = write_csv(run_path / "v95_active_sensor_self_check.csv", sensor_self_check_rows)
    sensor_self_check_json = write_json(run_path / "v95_active_sensor_self_check.json", sensor_self_check_rows)
    native_pair_csv = write_csv(run_path / "v95_native_pair_contact_sanity.csv", native_pair_rows)
    native_pair_json = write_json(run_path / "v95_native_pair_contact_sanity.json", native_pair_rows)
    screw1_asset_csv = write_csv(run_path / "v95_screw1_native_contact_asset_audit.csv", screw1_asset_rows)
    screw1_asset_json = write_json(run_path / "v95_screw1_native_contact_asset_audit.json", screw1_asset_rows)
    reference_filtered_csv = write_csv(run_path / "v95_reference_filtered_sensor_calibration.csv", reference_filtered_rows)
    reference_filtered_json = write_json(run_path / "v95_reference_filtered_sensor_calibration.json", reference_filtered_rows)
    screw1_reference_csv = write_csv(run_path / "v95_screw1_reference_contact_canary.csv", screw1_reference_contact_rows)
    screw1_reference_json = write_json(run_path / "v95_screw1_reference_contact_canary.json", screw1_reference_contact_rows)
    screw1_forced_csv = write_csv(run_path / "v95_screw1_forced_target_contact_calibration.csv", screw1_forced_contact_rows)
    screw1_forced_json = write_json(run_path / "v95_screw1_forced_target_contact_calibration.json", screw1_forced_contact_rows)
    screw1_mirror_csv = write_csv(run_path / "v95_screw1_object_side_mirror_check.csv", screw1_mirror_rows)
    screw1_mirror_json = write_json(run_path / "v95_screw1_object_side_mirror_check.json", screw1_mirror_rows)
    screw1_filter_csv = write_csv(run_path / "v95_screw1_target_filter_coverage_audit.csv", screw1_filter_coverage_rows)
    screw1_filter_json = write_json(run_path / "v95_screw1_target_filter_coverage_audit.json", screw1_filter_coverage_rows)
    filtered_structure_csv = write_csv(run_path / "v95_filtered_sensor_structure_audit.csv", filtered_structure_rows)
    filtered_structure_json = write_json(run_path / "v95_filtered_sensor_structure_audit.json", filtered_structure_rows)
    finger_body_audit_csv = write_csv(run_path / "v95_finger_contact_body_audit.csv", finger_contact_body_audit_rows)
    finger_body_audit_json = write_json(run_path / "v95_finger_contact_body_audit.json", finger_contact_body_audit_rows)
    cube_canary_csv = write_csv(run_path / "v95_cube_cube_contact_canary.csv", cube_canary_rows)
    cube_canary_json = write_json(run_path / "v95_cube_cube_contact_canary.json", cube_canary_rows)
    finger_reference_canary_csv = write_csv(run_path / "v95_finger_reference_contact_canary.csv", finger_reference_canary_rows)
    finger_reference_canary_json = write_json(run_path / "v95_finger_reference_contact_canary.json", finger_reference_canary_rows)
    pair_matrix_csv = write_csv(run_path / "v95_pair_contact_matrix.csv", pair_contact_matrix_rows)
    pair_matrix_json = write_json(run_path / "v95_pair_contact_matrix.json", pair_contact_matrix_rows)

    completed_path = write_json(
        run_path / "v95_live_probe_completed.json",
        {
            "run_mode": V95_RUN_MODE,
            "completed_unix_s": time.time(),
            "live_probe_completed": True,
            "live_probe_started_json": str(started_path),
            "native_shutdown": False,
            "live_probe_rows_written": bool(wrist_rows and finger_rows and contact_rows and filtered_structure_rows and cube_canary_rows),
            "filtered_contact_canary_ok": canary_ok,
            "filtered_contact_canary_blocker": "" if canary_ok else canary_blocker,
            "screw1_reference_contact_canary_ok": screw1_reference_ok,
            "screw1_reference_contact_status": screw1_reference_status,
            "screw1_forced_contact_status": forced_status,
            "screw1_forced_contact_calibration_ok": forced_ok,
            "ordinary_screw1_acquisition_executed": False,
            "ordinary_screw1_acquisition_next_allowed": bool(forced_ok),
            "finger_contact_body_audit_rows": len(finger_contact_body_audit_rows),
        },
    )
    progress_rows = _progress_rows(
        parts=selected_parts,
        backend=backend,
        active_rows=active_rows,
        reset_lifecycle_rows=reset_lifecycle_rows,
        pregrasp_rows=pregrasp_rows,
        wrist_rows=wrist_rows,
        finger_rows=finger_rows,
        contact_rows=contact_rows,
        pregrasp_by_part=pregrasp_by_part,
    )
    progress_csv = write_csv(run_path / "v95_workcell_progress_matrix.csv", progress_rows)
    progress_md = run_path / "v95_workcell_progress_matrix.md"
    _write_md(progress_md, progress_rows)
    forbidden = write_v95_forbidden_artifact_scan(run_path, repo_root=repo_root)
    return {
        "rows": progress_rows,
        "plan_rows": plan_rows,
        "active_rows": active_rows,
        "reset_lifecycle_rows": reset_lifecycle_rows,
        "pregrasp_rows": pregrasp_rows,
        "pregrasp_calibration_trace_rows": trace_rows,
        "wrist_rows": wrist_rows,
        "finger_rows": finger_rows,
        "contact_rows": contact_rows,
        "contact_acquisition_trace_row_count": len(acquisition_trace_rows),
        "screw1_contact_truth_trace_rows": screw1_trace_rows,
        "screw1_contact_truth_summary_rows": screw1_summary_rows,
        "screw1_last_mile_correction_rows": last_mile.get("correction_rows", []),
        "active_sensor_self_check_rows": sensor_self_check_rows,
        "native_pair_contact_sanity_rows": native_pair_rows,
        "screw1_native_contact_asset_audit_rows": screw1_asset_rows,
        "reference_filtered_sensor_calibration_rows": reference_filtered_rows,
        "screw1_reference_contact_canary_rows": screw1_reference_contact_rows,
        "screw1_forced_target_contact_calibration_rows": screw1_forced_contact_rows,
        "screw1_object_side_mirror_check_rows": screw1_mirror_rows,
        "screw1_target_filter_coverage_audit_rows": screw1_filter_coverage_rows,
        "filtered_sensor_structure_audit_rows": filtered_structure_rows,
        "finger_contact_body_audit_rows": finger_contact_body_audit_rows,
        "cube_cube_contact_canary_rows": cube_canary_rows,
        "finger_reference_contact_canary_rows": finger_reference_canary_rows,
        "pair_contact_matrix_rows": pair_contact_matrix_rows,
        "v95_live_probe_started_json": str(started_path),
        "v95_live_probe_completed_json": str(completed_path),
        "v95_workcell_progress_matrix_csv": str(progress_csv),
        "v95_workcell_progress_matrix_md": str(progress_md),
        "v95_pregrasp_calibration_trace_csv": str(trace_csv),
        "v95_pregrasp_calibration_trace_json": str(trace_json),
        "v95_contact_acquisition_trace_csv": str(acquisition_trace_csv),
        "v95_contact_acquisition_trace_jsonl": str(acquisition_trace_jsonl),
        "v95_screw1_contact_truth_trace_csv": str(screw1_trace_csv),
        "v95_screw1_contact_truth_trace_jsonl": str(screw1_trace_jsonl),
        "v95_screw1_contact_truth_summary_csv": str(screw1_summary_csv),
        "v95_screw1_contact_truth_summary_json": str(screw1_summary_json),
        "v95_active_sensor_self_check_csv": str(sensor_self_check_csv),
        "v95_active_sensor_self_check_json": str(sensor_self_check_json),
        "v95_native_pair_contact_sanity_csv": str(native_pair_csv),
        "v95_native_pair_contact_sanity_json": str(native_pair_json),
        "v95_screw1_native_contact_asset_audit_csv": str(screw1_asset_csv),
        "v95_screw1_native_contact_asset_audit_json": str(screw1_asset_json),
        "v95_reference_filtered_sensor_calibration_csv": str(reference_filtered_csv),
        "v95_reference_filtered_sensor_calibration_json": str(reference_filtered_json),
        "v95_screw1_reference_contact_canary_csv": str(screw1_reference_csv),
        "v95_screw1_reference_contact_canary_json": str(screw1_reference_json),
        "v95_screw1_forced_target_contact_calibration_csv": str(screw1_forced_csv),
        "v95_screw1_forced_target_contact_calibration_json": str(screw1_forced_json),
        "v95_screw1_object_side_mirror_check_csv": str(screw1_mirror_csv),
        "v95_screw1_object_side_mirror_check_json": str(screw1_mirror_json),
        "v95_screw1_target_filter_coverage_audit_csv": str(screw1_filter_csv),
        "v95_screw1_target_filter_coverage_audit_json": str(screw1_filter_json),
        "v95_filtered_sensor_structure_audit_csv": str(filtered_structure_csv),
        "v95_filtered_sensor_structure_audit_json": str(filtered_structure_json),
        "v95_finger_contact_body_audit_csv": str(finger_body_audit_csv),
        "v95_finger_contact_body_audit_json": str(finger_body_audit_json),
        "v95_cube_cube_contact_canary_csv": str(cube_canary_csv),
        "v95_cube_cube_contact_canary_json": str(cube_canary_json),
        "v95_finger_reference_contact_canary_csv": str(finger_reference_canary_csv),
        "v95_finger_reference_contact_canary_json": str(finger_reference_canary_json),
        "v95_pair_contact_matrix_csv": str(pair_matrix_csv),
        "v95_pair_contact_matrix_json": str(pair_matrix_json),
        "forbidden_artifact_scan": forbidden,
    }


def write_v95_forbidden_artifact_scan(run_dir: str | Path, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    run_path = ensure_run_dir(run_dir)
    artifact_hits = []
    for path in run_path.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(pattern in lower for pattern in _FORBIDDEN_ARTIFACT_PATTERNS):
            artifact_hits.append(str(path.relative_to(run_path)))
    source_hits = []
    if repo_root is not None:
        root = Path(repo_root)
        source_files = [
            root / "scripts/environments/run_v81_physical_backend_grasp_rl.py",
            root / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/minimal_workcell_gate.py",
        ]
        for path in source_files:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            scan_text = _v95_branch_scan_text(text) if path.name == "run_v81_physical_backend_grasp_rl.py" else text
            if path.name == "minimal_workcell_gate.py":
                scan_text = _strip_scanner_literal_tables(scan_text)
            for token in _FORBIDDEN_SOURCE_TOKENS:
                if token in scan_text:
                    source_hits.append({"file": str(path.relative_to(root)), "token": token})
    rows = [
        {
            "artifact_hits": artifact_hits,
            "source_hits": source_hits,
            "forbidden_artifact_count": len(artifact_hits),
            "forbidden_source_hit_count": len(source_hits),
            "forbidden_paths_absent": not artifact_hits and not source_hits,
            "ppo_ran": False,
            "bc_ran": False,
            "checkpoint_written": False,
            "sticky_eval_ran": False,
            "final_replay_ran": False,
            "video_generated": False,
            "dataset_exported": False,
            "usable_training_row_count": 0,
        }
    ]
    txt_path = run_path / "v95_forbidden_artifact_scan.txt"
    txt_path.write_text(
        "\n".join(
            [
                f"forbidden_artifact_count={len(artifact_hits)}",
                f"forbidden_source_hit_count={len(source_hits)}",
                "artifact_hits:",
                *artifact_hits,
                "source_hits:",
                *[f"{row['file']}:{row['token']}" for row in source_hits],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v95_forbidden_artifact_scan.json", rows)
    return {"rows": rows, "v95_forbidden_artifact_scan_txt": str(txt_path), "v95_forbidden_artifact_scan_json": str(json_path)}


def _calibrate_v95_pregrasp(
    backend: IsaacUnifiedSingleContextBackend,
    initial_plan_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    plan_rows = [copy.deepcopy(row) for row in initial_plan_rows]
    trace_rows: list[dict[str, Any]] = []
    final_sample: dict[str, Any] | None = None

    for row in plan_rows:
        row["hand_target_local_xyz"] = _safe_hand_target()
        row["hand_target_source"] = "v95_stage_a_safe_hand"
        row["hand_quat_source"] = "active_overhand_from_part"
    for iteration in range(6):
        sample = _sample_v95_plan(backend, plan_rows, phase="object_safe_place", iteration=iteration)
        trace_rows.extend(sample["trace_rows"])
        final_sample = sample
        if all(_sample_motion_force_ok(row, backend) for row in sample["sample_rows"]):
            break
        by_part = _rows_by_part(sample["sample_rows"])
        for row in plan_rows:
            _adjust_object_support_height(row, by_part.get(str(row.get("part_name") or ""), {}), backend)

    for row in plan_rows:
        row["hand_target_local_xyz"] = _initial_pregrasp_target(row)
        row["hand_target_source"] = "v95_stage_b_safe_pregrasp"
        row["hand_quat_source"] = "active_overhand_from_part"
    for iteration in range(6):
        sample = _sample_v95_plan(backend, plan_rows, phase="hand_safe_pregrasp", iteration=iteration)
        trace_rows.extend(sample["trace_rows"])
        final_sample = sample
        if all(_sample_motion_force_ok(row, backend) for row in sample["sample_rows"]):
            break
        by_part = _rows_by_part(sample["sample_rows"])
        for row in plan_rows:
            _adjust_hand_away_from_contact(row, by_part.get(str(row.get("part_name") or ""), {}), backend)

    distance_converged = False
    for iteration in range(8):
        sample = _sample_v95_plan(backend, plan_rows, phase="distance_closure", iteration=iteration)
        trace_rows.extend(sample["trace_rows"])
        final_sample = sample
        if all(_sample_pregrasp_ok(row, backend) for row in sample["sample_rows"]):
            distance_converged = True
            break
        by_part = _rows_by_part(sample["sample_rows"])
        for row in plan_rows:
            slot = _slot_for_plan_row(backend, row)
            if slot is None:
                continue
            adjustment = _distance_closure_adjustment(backend, slot, row, by_part.get(slot.part_name, {}))
            _apply_hand_adjustment(row, adjustment, backend)
            row["hand_target_source"] = "v95_stage_c_distance_closure"
    if not distance_converged:
        sample = _sample_v95_plan(backend, plan_rows, phase="distance_closure_final", iteration=8)
        trace_rows.extend(sample["trace_rows"])
        final_sample = sample

    if final_sample is None:
        raise RuntimeError("V95_PREGRASP_CALIBRATION_DID_NOT_RUN")
    return {
        "plan_rows": plan_rows,
        "reset_rows": final_sample["reset_rows"],
        "hand_rows": final_sample["hand_rows"],
        "sample_rows": final_sample["sample_rows"],
        "trace_rows": trace_rows,
    }


def _sample_v95_plan(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    phase: str,
    iteration: int,
    settle_steps: int = 2,
) -> dict[str, Any]:
    _assert_v95_plan_valid(backend, plan_rows, phase=phase)
    backend.configure_v95_pregrasp(plan_rows)
    backend.object_write_by_policy_detected = False
    backend.reset_envs()
    backend._capture_v86_reset_object_positions()
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    reset_rows = [dict(row) for row in getattr(base, "v95_last_pregrasp_rows", []) or []]
    hand_rows = [dict(row) for row in getattr(base, "v95_last_hand_reset_rows", []) or []]
    _assert_v95_reset_applied(backend, plan_rows, reset_rows, hand_rows, phase=phase)
    start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
    metric_trace: list[dict[str, Any]] = []
    for slot in backend.slots:
        metric_trace.append(_sample_metric_row(backend, slot, phase, iteration, sample_step=-1, start_states=start_states))
    for step in range(settle_steps):
        _set_context(backend, f"v95_{phase}_zero_settle", step=step)
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        for slot in backend.slots:
            metric_trace.append(_sample_metric_row(backend, slot, phase, iteration, sample_step=step, start_states=start_states))
    sample_rows = []
    for slot in backend.slots:
        part_rows = [row for row in metric_trace if row.get("part_name") == slot.part_name]
        latest = part_rows[-1] if part_rows else {}
        force_peak = max([0.0, *[_to_float(row.get("force_peak_n")) for row in part_rows]])
        force_count = max([0, *[_int_field(row, "force_count") for row in part_rows]])
        sample = dict(latest)
        sample.update(
            {
                "phase": phase,
                "iteration": int(iteration),
                "force_peak_n": force_peak,
                "force_count": force_count,
                "sample_motion_force_ok": bool(
                    force_peak <= backend.contact_manager.force_threshold_n
                    and force_count <= 0
                    and _to_float(latest.get("pre_contact_displacement_m")) <= 0.02
                    and abs(_to_float(latest.get("pre_contact_z_drift_signed_m"))) <= 0.02
                    and not backend.object_write_by_policy_detected
                ),
            }
        )
        sample["sample_pregrasp_ok"] = _sample_pregrasp_ok(sample, backend)
        sample["sample_blocker"] = _sample_blocker(sample, backend)
        sample_rows.append(sample)
    trace_rows = []
    for row in metric_trace:
        trace = dict(row)
        summary = _find_part_row(sample_rows, str(row.get("part_name") or ""))
        trace["sample_blocker"] = summary.get("sample_blocker", "")
        trace["sample_motion_force_ok"] = summary.get("sample_motion_force_ok", False)
        trace["sample_pregrasp_ok"] = summary.get("sample_pregrasp_ok", False)
        trace_rows.append(trace)
    return {"reset_rows": reset_rows, "hand_rows": hand_rows, "sample_rows": sample_rows, "trace_rows": trace_rows}


def _sample_metric_row(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    phase: str,
    iteration: int,
    *,
    sample_step: int,
    start_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = backend._metrics_for_slot(slot, f"v95_{phase}_{iteration}_{sample_step}")
    delta = _v91_state_delta(start_states.get(slot.part_name, {}), _v91_object_state_for_slot(backend, slot))
    plan = _find_slot_row(getattr(backend, "v95_pregrasp_plan_rows", []) or [], slot)
    distance = _v94_distance_for_slot(backend, slot, str(plan.get("active_finger_group") or ""))
    forces = _v92_force_list(metrics)
    return {
        "part_name": slot.part_name,
        "env_index": slot.global_env_index,
        "local_env_index": slot.local_env_index,
        "phase": phase,
        "iteration": int(iteration),
        "sample_step": int(sample_step),
        "sample_step_name": "immediate_after_reset" if sample_step < 0 else "post_reset_zero_settle",
        "hand_target_local_xyz": plan.get("hand_target_local_xyz", ""),
        "object_center_local_xyz": plan.get("object_center_local_xyz", ""),
        "active_finger_group": plan.get("active_finger_group", ""),
        "last_hand_adjustment_xyz": plan.get("last_hand_adjustment_xyz", ""),
        "last_hand_adjustment_norm_m": plan.get("last_hand_adjustment_norm_m", ""),
        "fingertip_object_surface_distance_min_m": distance.get("fingertip_object_surface_distance_min_m", 1.0),
        "fingertip_object_center_distance_min_m": distance.get("fingertip_object_center_distance_min_m", 1.0),
        "force_peak_n": max([0.0, *forces]),
        "force_count": int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0),
        "pre_contact_displacement_m": _to_float(delta.get("root_pose_delta_m")),
        "pre_contact_z_drift_signed_m": _to_float(delta.get("root_delta_z_m")),
        "pre_contact_z_drift_m": abs(_to_float(delta.get("root_delta_z_m"))),
        "table_collision": bool(metrics.get("table_collision")),
        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
        "distance_blocker": distance.get("fingertip_object_distance_blocker", ""),
    }


def _assert_v95_plan_valid(backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]], *, phase: str) -> None:
    if len(plan_rows) != len(backend.slots):
        raise RuntimeError(f"V95_PLAN_COUNT_MISMATCH:{phase}:{len(plan_rows)}!={len(backend.slots)}")
    seen: set[int] = set()
    for slot in backend.slots:
        row = _find_slot_row(plan_rows, slot)
        if not row:
            raise RuntimeError(f"V95_PLAN_ROW_MISSING:{phase}:{slot.local_env_index}:{slot.part_name}")
        env_index = _int_field(row, "env_index", -1)
        if env_index in seen:
            raise RuntimeError(f"V95_PLAN_DUPLICATE_ENV:{phase}:{env_index}")
        seen.add(env_index)
        expected = V83_PARTS[slot.local_env_index % len(V83_PARTS)]
        if str(row.get("part_name") or "") != slot.part_name or slot.part_name != expected:
            raise RuntimeError(f"V95_PLAN_PART_MISMATCH:{phase}:env{slot.local_env_index}:{row.get('part_name')}!={expected}")


def _assert_v95_reset_applied(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    reset_rows: list[dict[str, Any]],
    hand_rows: list[dict[str, Any]],
    *,
    phase: str,
) -> None:
    for slot in backend.slots:
        plan = _find_slot_row(plan_rows, slot)
        reset = _find_slot_row(reset_rows, slot)
        hand = _find_slot_row(hand_rows, slot)
        if not reset or not hand:
            raise RuntimeError(f"V95_RESET_ROW_MISSING:{phase}:env{slot.local_env_index}:{slot.part_name}")
        if not bool(reset.get("pregrasp_plan_applied")) or not bool(hand.get("pregrasp_plan_applied")):
            raise RuntimeError(f"V95_RESET_PLAN_NOT_APPLIED:{phase}:env{slot.local_env_index}:{slot.part_name}")
        if bool(hand.get("legacy_randomize_initial_state_called", True)):
            raise RuntimeError(f"V95_LEGACY_RANDOMIZE_CALLED:{phase}:env{slot.local_env_index}:{slot.part_name}")
        if str(plan.get("part_name") or "") != str(reset.get("part_name") or ""):
            raise RuntimeError(f"V95_RESET_PART_MISMATCH:{phase}:env{slot.local_env_index}")


def _sample_motion_force_ok(row: dict[str, Any], backend: IsaacUnifiedSingleContextBackend) -> bool:
    return bool(
        _to_float(row.get("force_peak_n")) <= backend.contact_manager.force_threshold_n
        and _int_field(row, "force_count", 0) <= 0
        and _to_float(row.get("pre_contact_displacement_m")) <= 0.02
        and abs(_to_float(row.get("pre_contact_z_drift_signed_m"))) <= 0.02
        and not _bool(row.get("table_collision"))
        and not _bool(row.get("object_write_by_policy_detected"))
    )


def _sample_pregrasp_ok(row: dict[str, Any], backend: IsaacUnifiedSingleContextBackend) -> bool:
    return bool(_sample_motion_force_ok(row, backend) and _to_float(row.get("fingertip_object_surface_distance_min_m"), 1.0) <= 0.015)


def _sample_blocker(row: dict[str, Any], backend: IsaacUnifiedSingleContextBackend) -> str:
    part_name = str(row.get("part_name") or "")
    phase = str(row.get("phase") or "")
    if _to_float(row.get("force_peak_n")) > backend.contact_manager.force_threshold_n or _int_field(row, "force_count", 0) > 0:
        if part_name == "Backrest" and phase == "object_safe_place":
            return "backrest_object_table_or_collision_initial_force_gt_0p05n"
        if part_name == "Backrest":
            return "backrest_hand_object_interpenetration_or_reset_force_gt_0p05n"
        return "reset_interpenetration_or_initial_force_gt_0p05n"
    if _to_float(row.get("pre_contact_displacement_m")) > 0.02:
        if part_name == "Backrest" and phase == "object_safe_place":
            return "backrest_object_table_support_pre_contact_displacement_gt_0p02m"
        if part_name == "Backrest":
            return "backrest_hand_pregrasp_or_workspace_pre_contact_displacement_gt_0p02m"
        return "pre_contact_displacement_gt_0p02m"
    if abs(_to_float(row.get("pre_contact_z_drift_signed_m"))) > 0.02:
        if part_name == "Backrest":
            return "backrest_object_height_or_orientation_z_drift_gt_0p02m"
        return "pre_contact_z_drift_gt_0p02m"
    if _bool(row.get("table_collision")):
        if part_name == "Backrest":
            return "backrest_table_collision_during_reset_settle"
        return "table_collision_during_reset_settle"
    if _to_float(row.get("fingertip_object_surface_distance_min_m"), 1.0) > 0.015:
        return "fingertip_object_distance_gt_0p015m"
    if _bool(row.get("object_write_by_policy_detected")):
        return "object_or_hand_write_after_reset_detected"
    return ""


def _pregrasp_application_rows_from_sample(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    reset_rows: list[dict[str, Any]],
    hand_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for slot in backend.slots:
        plan = _find_slot_row(plan_rows, slot)
        reset = _find_slot_row(reset_rows, slot)
        hand = _find_slot_row(hand_rows, slot)
        sample = _find_slot_row(sample_rows, slot)
        applied = bool(plan and reset and hand and reset.get("pregrasp_plan_applied") and hand.get("pregrasp_plan_applied"))
        parking_ok = bool(reset.get("inactive_object_parked", False))
        ok = bool(applied and parking_ok and _sample_pregrasp_ok(sample, backend))
        blocker = (
            ""
            if ok
            else (
                "v95_pregrasp_plan_not_applied_in_reset_lifecycle"
                if not applied
                else (reset.get("inactive_object_parking_blocker") or "inactive_object_parking_failed")
                if not parking_ok
                else _sample_blocker(sample, backend)
            )
        )
        rows.append(
            {
                **plan,
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "fingertip_object_center_distance_min_m": sample.get("fingertip_object_center_distance_min_m", 1.0),
                "fingertip_object_surface_distance_min_m": sample.get("fingertip_object_surface_distance_min_m", 1.0),
                "pregrasp_plan_applied": applied,
                "pregrasp_application_ok": ok,
                "pregrasp_application_blocker": blocker,
                "pre_contact_displacement_m": sample.get("pre_contact_displacement_m", 0.0),
                "pre_contact_z_drift_m": sample.get("pre_contact_z_drift_m", 0.0),
                "pre_contact_z_drift_signed_m": sample.get("pre_contact_z_drift_signed_m", 0.0),
                "initial_force_peak_n": sample.get("force_peak_n", 0.0),
                "initial_force_contact_count": sample.get("force_count", 0),
                "object_reset_source": reset.get("reset_lifecycle_source", ""),
                "hand_reset_source": hand.get("hand_write_phase", ""),
                "inactive_object_parked": bool(reset.get("inactive_object_parked", False)),
                "inactive_object_parked_count": int(reset.get("inactive_object_parked_count") or 0),
                "inactive_object_parked_parts": reset.get("inactive_object_parked_parts", ""),
                "inactive_object_parking_blocker": reset.get("inactive_object_parking_blocker", ""),
                "legacy_randomize_initial_state_called": bool(hand.get("legacy_randomize_initial_state_called", True)),
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "reset_evidence_window": "post_reset_zero_settle",
                "live_probe_executed": True,
                "native_shutdown": False,
            }
        )
    return rows


def _base_env(backend: IsaacUnifiedSingleContextBackend) -> Any:
    return getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None


def _enable_v95_action_control(backend: IsaacUnifiedSingleContextBackend) -> None:
    base = _base_env(backend)
    if base is None:
        return
    if hasattr(base, "v95_configure_action_control"):
        base.v95_configure_action_control(mode="anchored_delta", allow_commanded_hand_actions=True)
        return
    try:
        base.v95_action_control_mode = "anchored_delta"
        base.v95_allow_commanded_hand_actions = True
    except Exception:
        return


def _disable_v95_action_control(backend: IsaacUnifiedSingleContextBackend) -> None:
    base = _base_env(backend)
    if base is None:
        return
    if hasattr(base, "v95_clear_action_control"):
        base.v95_clear_action_control()
        return
    try:
        base.v95_action_control_mode = ""
        base.v95_allow_commanded_hand_actions = False
    except Exception:
        return


def _v95_control_diag_rows(backend: IsaacUnifiedSingleContextBackend) -> dict[str, dict[str, Any]]:
    base = _base_env(backend)
    rows: dict[str, dict[str, Any]] = {}
    for slot in backend.slots:
        action = _tensor_list(getattr(base, "v95_last_env_received_action", None), slot.local_env_index)
        anchor = _tensor_list(getattr(base, "v95_last_wrist_anchor_pos", None), slot.local_env_index)
        target_before = _tensor_list(getattr(base, "v95_last_wrist_target_before", None), slot.local_env_index)
        target_pre = _tensor_list(getattr(base, "v95_last_wrist_target_pre_clamp", None), slot.local_env_index)
        target_post = _tensor_list(getattr(base, "v95_last_wrist_target_post_clamp", None), slot.local_env_index)
        target_after = _tensor_list(getattr(base, "v95_last_wrist_target_after_generate", None), slot.local_env_index)
        target_delta = _tensor_list(getattr(base, "v95_last_wrist_target_delta", None), slot.local_env_index)
        rows[slot.part_name] = {
            "v95_action_control_mode": str(getattr(base, "v95_action_control_mode", "")),
            "v95_allow_commanded_hand_actions": bool(getattr(base, "v95_allow_commanded_hand_actions", False)),
            "v95_wrist_anchor_valid": bool(_tensor_bool(getattr(base, "v95_wrist_anchor_valid", None), slot.local_env_index)),
            "env_received_action_wrist_xyz": action[:3],
            "env_received_action_norm": _norm(action),
            "wrist_anchor_target_xyz": anchor[:3],
            "wrist_target_before_xyz": target_before[:3],
            "wrist_target_pre_clamp_xyz": target_pre[:3],
            "wrist_target_post_clamp_xyz": target_post[:3],
            "wrist_target_after_generate_xyz": target_after[:3],
            "wrist_target_delta_xyz": target_delta[:3],
            "wrist_target_delta_l2": _norm(target_delta),
            "workspace_clamp_delta_m": _tensor_float(getattr(base, "v95_last_workspace_clamp_delta", None), slot.local_env_index),
            "table_barrier_delta_z_m": _tensor_float(getattr(base, "v95_last_table_barrier_delta_z", None), slot.local_env_index),
            "wrist_joint_target_delta_l2": _tensor_float(
                getattr(base, "v95_last_wrist_joint_target_delta_l2", None),
                slot.local_env_index,
            ),
            "anchor_workspace_clamp_delta_m": _tensor_float(
                getattr(base, "v95_anchor_workspace_clamp_delta", None),
                slot.local_env_index,
            ),
        }
    return rows


def _v95_hand_diag_rows(backend: IsaacUnifiedSingleContextBackend) -> dict[str, dict[str, Any]]:
    base = _base_env(backend)
    rows: dict[str, dict[str, Any]] = {}
    for slot in backend.slots:
        rows[slot.part_name] = {
            "hand_raw_action": _tensor_list(getattr(base, "last_dex_hand_raw_action", None), slot.local_env_index),
            "hand_clamped_action": _tensor_list(getattr(base, "last_dex_hand_clamped_action", None), slot.local_env_index),
            "hand_target_delta": _tensor_list(getattr(base, "last_dex_hand_target_delta", None), slot.local_env_index),
            "hand_limiter_delta": _tensor_list(getattr(base, "last_dex_hand_limiter_delta", None), slot.local_env_index),
            "hand_limiter_applied": bool(getattr(base, "last_dex_hand_limiter_applied", False)),
            "hand_limiter_bypassed_for_v95": bool(getattr(base, "last_dex_hand_limiter_bypassed_for_v95", False)),
            "v95_allow_commanded_hand_actions": bool(getattr(base, "v95_allow_commanded_hand_actions", False)),
        }
    return rows


def _v95_hand_target_pos(backend: IsaacUnifiedSingleContextBackend) -> Any:
    base = _base_env(backend)
    try:
        return base.ctrl_target_joint_pos.detach().clone()
    except Exception:
        return None


def _active_object_state_rows(backend: IsaacUnifiedSingleContextBackend) -> list[dict[str, Any]]:
    rows = []
    for slot in backend.slots:
        identity = backend.object_identity_for_slot(slot)
        backend._record_identity(identity)
        active = backend.v95_active_asset_info_for_slot(slot)
        rows.append(
            {
                **identity,
                **active,
                "expected_slot_part_name": V83_PARTS[slot.local_env_index % len(V83_PARTS)],
                "slot_to_part_mapping": f"env{slot.local_env_index}->{slot.part_name}",
                "slot_mapping_round_robin": True,
                "single_simulation_context": bool(backend.single_simulation_context),
                "gym_make_count": int(backend.gym_make_count),
                "object_identity_verified": bool(identity.get("object_identity_verified")),
                "active_object_state_matches_slot": bool(active.get("active_object_state_matches_slot")),
                "legacy_held_state_matches_active_object": bool(active.get("legacy_held_state_matches_active_object")),
                "legacy_global_held_asset_is_plug2": bool(active.get("legacy_global_held_asset_is_plug2")),
                "v95_critical_path_uses_active_object_state": bool(active.get("v95_critical_path_uses_active_object_state")),
                "v95_derived_state_refresh_complete": bool(active.get("v95_derived_state_refresh_complete")),
                "v95_derived_state_refresh_fields": active.get("v95_derived_state_refresh_fields", ""),
            }
        )
    return rows


def _reset_lifecycle_rows(
    backend: IsaacUnifiedSingleContextBackend,
    reset_rows: list[dict[str, Any]],
    hand_rows: list[dict[str, Any]],
    reset_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for slot in backend.slots:
        reset = _find_slot_row(reset_rows, slot)
        hand = _find_slot_row(hand_rows, slot)
        metrics = _find_slot_row(reset_metrics, slot)
        force_peak = max([0.0, *_v92_force_list(metrics)])
        force_count = int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0)
        parking_ok = bool(reset.get("inactive_object_parked", False))
        sample_blocker = _sample_blocker(metrics, backend)
        reset_blocker = (
            reset.get("inactive_object_parking_blocker") or "inactive_object_parking_failed"
            if not parking_ok
            else sample_blocker
        )
        rows.append(
            {
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "pregrasp_plan_applied": bool(reset.get("pregrasp_plan_applied") and hand.get("pregrasp_plan_applied")),
                "object_reset_source": reset.get("reset_lifecycle_source", ""),
                "hand_reset_source": hand.get("hand_write_phase", ""),
                "reset_only_object_write": bool(reset.get("reset_only_object_write")),
                "reset_only_hand_write": bool(hand.get("hand_write_reset_only")),
                "inactive_object_parked": bool(reset.get("inactive_object_parked", False)),
                "inactive_object_parked_count": int(reset.get("inactive_object_parked_count") or 0),
                "inactive_object_parked_parts": reset.get("inactive_object_parked_parts", ""),
                "inactive_object_parking_blocker": reset.get("inactive_object_parking_blocker", ""),
                "legacy_randomize_initial_state_called": bool(hand.get("legacy_randomize_initial_state_called", True)),
                "object_write_after_reset_detected": bool(backend.object_write_by_policy_detected),
                "hand_live_write_after_reset_detected": False,
                "initial_force_peak_n": force_peak,
                "initial_force_contact_count": force_count,
                "initial_reset_force_ok": bool(force_peak <= backend.contact_manager.force_threshold_n and force_count <= 0),
                "reset_evidence_window": "post_reset_zero_settle",
                "post_reset_zero_settle_steps": 2,
                "pre_contact_displacement_m": metrics.get("pre_contact_displacement_m", 0.0),
                "pre_contact_z_drift_m": metrics.get("pre_contact_z_drift_m", 0.0),
                "reset_lifecycle_blocker": reset_blocker,
                "single_simulation_context": bool(backend.single_simulation_context),
                "gym_make_count": int(backend.gym_make_count),
                "auto_probe_reset_step_disabled_for_v95": True,
                "reset_lifecycle_ok": bool(
                    reset
                    and hand
                    and parking_ok
                    and not bool(hand.get("legacy_randomize_initial_state_called", True))
                    and force_peak <= backend.contact_manager.force_threshold_n
                    and force_count <= 0
                    and _to_float(metrics.get("pre_contact_displacement_m")) <= 0.02
                    and abs(_to_float(metrics.get("pre_contact_z_drift_signed_m"))) <= 0.02
                    and not bool(metrics.get("table_collision"))
                    and not backend.object_write_by_policy_detected
                ),
            }
        )
    return rows


def _pregrasp_application_rows(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    reset_rows: list[dict[str, Any]],
    hand_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    start_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
    for settle_step in range(2):
        _set_context(backend, "v95_pregrasp_zero_settle", step=settle_step)
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])
    for slot in backend.slots:
        plan = _find_slot_row(plan_rows, slot)
        reset = _find_slot_row(reset_rows, slot)
        hand = _find_slot_row(hand_rows, slot)
        metrics = backend._metrics_for_slot(slot, "v95_after_pregrasp_plan_application")
        delta = _v91_state_delta(start_states.get(slot.part_name, {}), _v91_object_state_for_slot(backend, slot))
        distance = _v94_distance_for_slot(backend, slot, str(plan.get("active_finger_group") or ""))
        force_peak = max([0.0, *_v92_force_list(metrics)])
        force_count = int(metrics.get("effective_contact_count_force") or metrics.get("force_contact_count") or 0)
        displacement = abs(_to_float(delta.get("root_pose_delta_m")))
        z_drift = abs(_to_float(delta.get("root_delta_z_m")))
        applied = bool(plan and reset and hand and reset.get("pregrasp_plan_applied") and hand.get("pregrasp_plan_applied"))
        ok = bool(
            applied
            and distance.get("fingertip_object_surface_distance_min_m", 1.0) <= 0.015
            and displacement <= 0.02
            and z_drift <= 0.02
            and force_peak <= backend.contact_manager.force_threshold_n
            and force_count <= 0
            and not backend.object_write_by_policy_detected
        )
        blocker = ""
        if not applied:
            blocker = "v95_pregrasp_plan_not_applied_in_reset_lifecycle"
        elif force_peak > backend.contact_manager.force_threshold_n or force_count > 0:
            blocker = "reset_interpenetration_or_initial_force_gt_0p05n"
        elif displacement > 0.02:
            blocker = "pre_contact_displacement_gt_0p02m"
        elif z_drift > 0.02:
            blocker = "pre_contact_z_drift_gt_0p02m"
        elif distance.get("fingertip_object_surface_distance_min_m", 1.0) > 0.015:
            blocker = "fingertip_object_distance_gt_0p015m"
        elif backend.object_write_by_policy_detected:
            blocker = "object_or_hand_write_after_reset_detected"
        rows.append(
            {
                **plan,
                **distance,
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "pregrasp_plan_applied": applied,
                "pregrasp_application_ok": ok,
                "pregrasp_application_blocker": blocker,
                "pre_contact_displacement_m": displacement,
                "pre_contact_z_drift_m": z_drift,
                "initial_force_peak_n": force_peak,
                "initial_force_contact_count": force_count,
                "object_reset_source": reset.get("reset_lifecycle_source", ""),
                "hand_reset_source": hand.get("hand_write_phase", ""),
                "legacy_randomize_initial_state_called": bool(hand.get("legacy_randomize_initial_state_called", True)),
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "live_probe_executed": True,
                "native_shutdown": False,
            }
        )
    return rows


def _run_live_wrist_calibration(backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    calibration_plan_rows = _wrist_calibration_plan_rows(plan_rows)
    for axis, axis_name in enumerate(("x", "y", "z")):
        for sign in (-1, 1):
            _reset_with_plan(backend, calibration_plan_rows, settle_steps=V95_WRIST_RESET_SETTLE_STEPS)
            zero_initial = _slot_metric_rows(backend, f"v95_wrist_{axis_name}_{sign:+d}_zero_initial")
            zero_before_mid = _v94_fingertip_midpoints(backend)
            zero_before_state = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
            zero_first_diag: dict[str, dict[str, Any]] = {}
            for step in range(4):
                _set_context(backend, f"v95_wrist_zero_baseline_{axis_name}_{sign:+d}", axis=axis_name, sign=sign, step=step)
                backend.step_envs([[0.0] * 16 for _slot in backend.slots])
                if step == 0:
                    zero_first_diag = _v95_control_diag_rows(backend)
            zero_last_diag = _v95_control_diag_rows(backend)
            zero_after_mid = _v94_fingertip_midpoints(backend)
            zero_delta_by_part = {
                slot.part_name: _v94_vec_delta(zero_before_mid, zero_after_mid, slot.local_env_index)
                for slot in backend.slots
            }
            zero_object_delta_by_part = {
                slot.part_name: _v91_state_delta(zero_before_state.get(slot.part_name, {}), _v91_object_state_for_slot(backend, slot))
                for slot in backend.slots
            }

            _reset_with_plan(backend, calibration_plan_rows, settle_steps=V95_WRIST_RESET_SETTLE_STEPS)
            command_initial = _slot_metric_rows(backend, f"v95_wrist_{axis_name}_{sign:+d}_command_initial")
            before_mid = _v94_fingertip_midpoints(backend)
            before_state = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
            command_first_diag: dict[str, dict[str, Any]] = {}
            for step in range(4):
                actions = []
                for _slot in backend.slots:
                    action = [0.0] * 16
                    action[axis] = 0.25 * float(sign)
                    actions.append(action)
                _set_context(backend, f"v95_wrist_{axis_name}_{sign:+d}", axis=axis_name, sign=sign, step=step)
                backend.step_envs(actions)
                if step == 0:
                    command_first_diag = _v95_control_diag_rows(backend)
            command_last_diag = _v95_control_diag_rows(backend)
            after_mid = _v94_fingertip_midpoints(backend)
            for slot in backend.slots:
                raw_delta = _v94_vec_delta(before_mid, after_mid, slot.local_env_index)
                zero_delta = zero_delta_by_part.get(slot.part_name, [0.0, 0.0, 0.0])
                corrected_delta = _vec_sub(raw_delta, zero_delta)
                obj_delta = _v91_state_delta(before_state.get(slot.part_name, {}), _v91_object_state_for_slot(backend, slot))
                zero_obj_delta = zero_object_delta_by_part.get(slot.part_name, {})
                zero_metrics = zero_initial.get(slot.part_name, {})
                command_metrics = command_initial.get(slot.part_name, {})
                zero_diag = zero_last_diag.get(slot.part_name, {})
                command_diag = command_first_diag.get(slot.part_name, command_last_diag.get(slot.part_name, {}))
                zero_force_peak = max([0.0, *_v92_force_list(zero_metrics)])
                command_initial_force_peak = max([0.0, *_v92_force_list(command_metrics)])
                zero_object_motion = _to_float(zero_obj_delta.get("root_pose_delta_m"))
                command_object_motion = _to_float(obj_delta.get("root_pose_delta_m"))
                measured = corrected_delta[axis] if len(corrected_delta) >= 3 else 0.0
                raw_axis_delta = raw_delta[axis] if len(raw_delta) >= 3 else 0.0
                zero_axis_delta = zero_delta[axis] if len(zero_delta) >= 3 else 0.0
                magnitude_ok = abs(measured) > 1.0e-5
                sign_ok = bool(magnitude_ok and measured * float(sign) > 0.0)
                env_action = list(command_diag.get("env_received_action_wrist_xyz") or [0.0, 0.0, 0.0])
                target_delta = list(command_diag.get("wrist_target_delta_xyz") or [0.0, 0.0, 0.0])
                target_axis_delta = target_delta[axis] if len(target_delta) >= 3 else 0.0
                mapper_action_received = bool(len(env_action) >= 3 and abs(env_action[axis] - 0.25 * float(sign)) <= 1.0e-4)
                target_changed = bool(abs(target_axis_delta) > 1.0e-5 and target_axis_delta * float(sign) > 0.0)
                clamp_delta = _to_float(command_diag.get("workspace_clamp_delta_m"))
                barrier_delta = abs(_to_float(command_diag.get("table_barrier_delta_z_m")))
                clamp_or_barrier = bool(clamp_delta > 1.0e-6 or barrier_delta > 1.0e-6)
                joint_target_changed = bool(_to_float(command_diag.get("wrist_joint_target_delta_l2")) > 1.0e-5)
                baseline_contact_free = bool(
                    zero_force_peak <= backend.contact_manager.force_threshold_n
                    and command_initial_force_peak <= backend.contact_manager.force_threshold_n
                    and zero_object_motion <= 0.02
                    and not bool(zero_metrics.get("table_collision"))
                    and not bool(command_metrics.get("table_collision"))
                    and not backend.object_write_by_policy_detected
                )
                drift_dominates = bool(abs(zero_axis_delta) > max(0.0005, abs(raw_axis_delta) * 0.50))
                command_object_motion_exceeded = bool(command_object_motion > 0.02)
                command_blocked_or_clamped = bool(
                    not magnitude_ok
                    or not mapper_action_received
                    or not target_changed
                    or not joint_target_changed
                    or clamp_or_barrier
                    or command_object_motion_exceeded
                )
                ok = bool(
                    sign_ok
                    and baseline_contact_free
                    and not drift_dominates
                    and not command_object_motion_exceeded
                    and not command_blocked_or_clamped
                )
                if not baseline_contact_free or drift_dominates:
                    blocker = "WRIST_BASELINE_CONTAMINATED"
                    blocker_layer = "baseline"
                elif not mapper_action_received:
                    blocker = "WRIST_ACTION_MAPPER_BLOCKER"
                    blocker_layer = "mapper"
                elif not target_changed:
                    blocker = "WRIST_ACTION_TARGET_BLOCKER"
                    blocker_layer = "target"
                elif clamp_or_barrier:
                    blocker = "WRIST_ACTION_CLAMP_OR_BARRIER_BLOCKER"
                    blocker_layer = "clamp_or_barrier"
                elif not joint_target_changed:
                    blocker = "WRIST_CONTROLLER_TARGET_DID_NOT_MOVE"
                    blocker_layer = "target_to_joint"
                elif command_object_motion_exceeded:
                    blocker = "WRIST_NO_CONTACT_OBJECT_MOTION_BLOCKER"
                    blocker_layer = "baseline"
                elif command_blocked_or_clamped:
                    blocker = "WRIST_CONTROLLER_GAIN_OR_DYNAMICS_BLOCKER"
                    blocker_layer = "controller_or_dynamics"
                elif not sign_ok:
                    blocker = "wrist_axis_delta_sign_or_magnitude_unexpected"
                    blocker_layer = "measured_motion"
                else:
                    blocker = ""
                    blocker_layer = ""
                rows.append(
                    {
                        "part_name": slot.part_name,
                        "env_index": slot.global_env_index,
                        "local_env_index": slot.local_env_index,
                        "calibration_axis": axis_name,
                        "calibration_sign": int(sign),
                        "policy_col": axis,
                        "policy_command_value": 0.25 * float(sign),
                        "isaac_action_col": axis,
                        "isaac_action_mapped_through_v82": True,
                        "wrist_calibration_pregrasp_mode": "safe_no_contact_reset",
                        "wrist_command_frame": "v95_anchored_delta_env_local_target_world_delta_measured",
                        "v95_action_control_mode": command_diag.get("v95_action_control_mode", ""),
                        "v95_wrist_anchor_valid": bool(command_diag.get("v95_wrist_anchor_valid")),
                        "env_received_action_wrist_xyz": env_action[:3],
                        "mapper_action_received": mapper_action_received,
                        "wrist_anchor_target_xyz": command_diag.get("wrist_anchor_target_xyz", []),
                        "wrist_target_before_xyz": command_diag.get("wrist_target_before_xyz", []),
                        "wrist_target_pre_clamp_xyz": command_diag.get("wrist_target_pre_clamp_xyz", []),
                        "wrist_target_post_clamp_xyz": command_diag.get("wrist_target_post_clamp_xyz", []),
                        "wrist_target_after_generate_xyz": command_diag.get("wrist_target_after_generate_xyz", []),
                        "wrist_target_delta_xyz": target_delta[:3],
                        "wrist_target_axis_delta_m": target_axis_delta,
                        "wrist_target_changed": target_changed,
                        "wrist_joint_target_delta_l2": command_diag.get("wrist_joint_target_delta_l2", 0.0),
                        "wrist_joint_target_changed": joint_target_changed,
                        "workspace_clamp_delta_m": clamp_delta,
                        "anchor_workspace_clamp_delta_m": command_diag.get("anchor_workspace_clamp_delta_m", 0.0),
                        "table_barrier_delta_z_m": barrier_delta,
                        "zero_action_anchor_target_delta_l2": zero_diag.get("wrist_target_delta_l2", 0.0),
                        "zero_action_workspace_clamp_delta_m": zero_diag.get("workspace_clamp_delta_m", 0.0),
                        "zero_action_table_barrier_delta_z_m": zero_diag.get("table_barrier_delta_z_m", 0.0),
                        "zero_baseline_delta_x": zero_delta[0] if len(zero_delta) >= 3 else 0.0,
                        "zero_baseline_delta_y": zero_delta[1] if len(zero_delta) >= 3 else 0.0,
                        "zero_baseline_delta_z": zero_delta[2] if len(zero_delta) >= 3 else 0.0,
                        "raw_command_delta_x": raw_delta[0] if len(raw_delta) >= 3 else 0.0,
                        "raw_command_delta_y": raw_delta[1] if len(raw_delta) >= 3 else 0.0,
                        "raw_command_delta_z": raw_delta[2] if len(raw_delta) >= 3 else 0.0,
                        "baseline_subtracted_delta_x": corrected_delta[0] if len(corrected_delta) >= 3 else 0.0,
                        "baseline_subtracted_delta_y": corrected_delta[1] if len(corrected_delta) >= 3 else 0.0,
                        "baseline_subtracted_delta_z": corrected_delta[2] if len(corrected_delta) >= 3 else 0.0,
                        "fingertip_midpoint_delta_x": corrected_delta[0] if len(corrected_delta) >= 3 else 0.0,
                        "fingertip_midpoint_delta_y": corrected_delta[1] if len(corrected_delta) >= 3 else 0.0,
                        "fingertip_midpoint_delta_z": corrected_delta[2] if len(corrected_delta) >= 3 else 0.0,
                        "measured_axis_delta_m": measured,
                        "raw_measured_axis_delta_m": raw_axis_delta,
                        "zero_baseline_axis_delta_m": zero_axis_delta,
                        "baseline_contact_free_ok": baseline_contact_free,
                        "zero_baseline_force_peak_n": zero_force_peak,
                        "command_initial_force_peak_n": command_initial_force_peak,
                        "zero_baseline_object_displacement_m": zero_object_motion,
                        "command_object_displacement_m": command_object_motion,
                        "baseline_drift_dominates_command": drift_dominates,
                        "command_object_motion_exceeded": command_object_motion_exceeded,
                        "workspace_or_stabilizer_interference": bool(clamp_or_barrier),
                        "command_blocked_or_clamped": command_blocked_or_clamped,
                        "wrist_blocker_layer": blocker_layer,
                        "wrist_axis_magnitude_valid": magnitude_ok,
                        "wrist_axis_sign_valid": sign_ok,
                        "wrist_action_mapping_ok": ok,
                        "object_displacement_m": obj_delta.get("root_pose_delta_m", 0.0),
                        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                        "live_probe_executed": True,
                        "native_shutdown": False,
                        "blocker": blocker,
                    }
                )
    return rows


def _wrist_calibration_plan_rows(plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in plan_rows:
        clone = copy.deepcopy(row)
        target = _safe_no_contact_hand_target(row)
        clone["hand_target_local_xyz"] = target
        clone["hand_target_local_x"] = target[0]
        clone["hand_target_local_y"] = target[1]
        clone["hand_target_local_z"] = target[2]
        clone["hand_target_source"] = "v95_wrist_safe_no_contact"
        clone["hand_quat_source"] = "active_overhand_from_part"
        rows.append(clone)
    return rows


def _run_live_finger_motion_calibration(backend: IsaacUnifiedSingleContextBackend, plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logical_finger in range(1, 6):
        _reset_with_plan(backend, plan_rows)
        before_tips = _v92_tip_positions(backend)
        before_joints = _v92_joint_pos(backend)
        before_targets = _v95_hand_target_pos(backend)
        for step in range(8):
            actions = []
            value = min(0.85, 0.12 * float(step + 1))
            for _slot in backend.slots:
                action = [0.0] * 16
                action[6 + (logical_finger - 1)] = value
                action[11 + (logical_finger - 1)] = value
                actions.append(action)
            _set_context(backend, f"v95_finger_{logical_finger}_motion", finger=logical_finger, step=step)
            backend.step_envs(actions)
        after_tips = _v92_tip_positions(backend)
        after_joints = _v92_joint_pos(backend)
        after_targets = _v95_hand_target_pos(backend)
        hand_diag_by_part = _v95_hand_diag_rows(backend)
        for slot in backend.slots:
            tip_deltas = [
                _v92_tip_delta(before_tips, after_tips, slot.local_env_index, finger_index)
                for finger_index in range(5)
            ]
            active_delta = tip_deltas[logical_finger - 1] if logical_finger - 1 < len(tip_deltas) else 0.0
            other_deltas = [value for index, value in enumerate(tip_deltas) if index != logical_finger - 1]
            active_joint_delta = _finger_close_joint_delta(before_joints, after_joints, slot.local_env_index, logical_finger - 1)
            non_active_joint_delta = max(
                [
                    0.0,
                    *[
                        _finger_close_joint_delta(before_joints, after_joints, slot.local_env_index, finger_index)
                        for finger_index in range(5)
                        if finger_index != logical_finger - 1
                    ],
                ]
            )
            active_target_delta = _finger_close_joint_delta(before_targets, after_targets, slot.local_env_index, logical_finger - 1)
            non_active_target_delta = max(
                [
                    0.0,
                    *[
                        _finger_close_joint_delta(before_targets, after_targets, slot.local_env_index, finger_index)
                        for finger_index in range(5)
                        if finger_index != logical_finger - 1
                    ],
                ]
            )
            global_joint_delta = _v92_joint_delta(before_joints, after_joints, slot.local_env_index)
            leakage = max([0.0, *other_deltas])
            hand_diag = hand_diag_by_part.get(slot.part_name, {})
            raw_values = _finger_local_values(hand_diag.get("hand_raw_action", []), logical_finger - 1)
            clamped_values = _finger_local_values(hand_diag.get("hand_clamped_action", []), logical_finger - 1)
            last_target_values = _finger_local_values(hand_diag.get("hand_target_delta", []), logical_finger - 1)
            limiter_values = _finger_local_values(hand_diag.get("hand_limiter_delta", []), logical_finger - 1)
            moved_tip_index = _v92_max_index(tip_deltas)
            active_target_dominant = bool(
                active_target_delta >= 1.0e-4
                and active_target_delta >= max(1.0e-4, non_active_target_delta * 1.20)
            )
            active_tip_dominant = bool(
                active_delta >= 2.0e-4
                and moved_tip_index == logical_finger - 1
                and active_delta >= max(1.0e-4, leakage * 1.20)
            )
            active_joint_dominant = bool(
                active_joint_delta >= 1.0e-4
                and active_joint_delta >= max(1.0e-4, non_active_joint_delta * 1.20)
            )
            moved = bool(active_target_dominant and active_tip_dominant and active_joint_dominant)
            leakage_ok = bool(moved and leakage <= max(0.0015, active_delta * 1.20 + 1.0e-5))
            inferred_sensor_index = moved_tip_index if moved and moved_tip_index == logical_finger - 1 else -1
            if not active_target_dominant:
                blocker = "finger_target_did_not_change_mapper_or_limiter"
                blocker_layer = "mapper_or_limiter"
            elif not active_joint_dominant:
                blocker = "finger_target_changed_but_joint_did_not_move"
                blocker_layer = "actuator_gain_or_dynamics"
            elif not active_tip_dominant:
                blocker = "finger_joint_moved_but_fingertip_not_dominant"
                blocker_layer = "kinematic_or_body_mapping"
            elif not leakage_ok:
                blocker = "non_active_finger_leakage"
                blocker_layer = "finger_leakage"
            else:
                blocker = ""
                blocker_layer = ""
            rows.append(
                {
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "logical_finger_id": logical_finger,
                    "policy_close_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                    "isaac_close_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                    "hand_local_close_cols": f"{10 + logical_finger - 1},{15 + logical_finger - 1}",
                    "raw_hand_action_local_close_values": raw_values,
                    "clamped_hand_action_local_close_values": clamped_values,
                    "last_step_hand_target_delta_local_close_values": last_target_values,
                    "hand_limiter_delta_local_close_values": limiter_values,
                    "hand_limiter_applied": bool(hand_diag.get("hand_limiter_applied")),
                    "hand_limiter_bypassed_for_v95": bool(hand_diag.get("hand_limiter_bypassed_for_v95")),
                    "v95_allow_commanded_hand_actions": bool(hand_diag.get("v95_allow_commanded_hand_actions")),
                    "non_active_fingers_neutral": True,
                    "per_finger_tip_delta_m": tip_deltas,
                    "fingertip_position_delta_m": active_delta,
                    "active_close_target_delta_l2": active_target_delta,
                    "non_active_close_target_delta_max_l2": non_active_target_delta,
                    "active_close_joint_delta_l2": active_joint_delta,
                    "non_active_close_joint_delta_max_l2": non_active_joint_delta,
                    "joint_position_delta_l2": global_joint_delta,
                    "non_active_finger_leakage_max_m": leakage,
                    "non_active_finger_leakage_ok": leakage_ok,
                    "moved_fingertip_index": moved_tip_index,
                    "actually_moved_logical_finger": moved_tip_index + 1 if moved_tip_index >= 0 else 0,
                    "active_target_motion_dominant": active_target_dominant,
                    "active_tip_motion_dominant": active_tip_dominant,
                    "active_joint_motion_dominant": active_joint_dominant,
                    "inferred_logical_to_sensor_index": inferred_sensor_index,
                    "sensor_mapping_status": "calibrated_from_active_motion" if inferred_sensor_index >= 0 else "uncalibrated_active_motion_not_dominant",
                    "close_sign_correct": bool(active_target_dominant and active_joint_dominant),
                    "finger_motion_ok": bool(moved and leakage_ok and not backend.object_write_by_policy_detected),
                    "finger_motion_blocker": blocker,
                    "finger_blocker_layer": blocker_layer,
                    "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                    "live_probe_executed": True,
                    "native_shutdown": False,
                }
            )
    return rows


def _screw1_native_contact_asset_audit(backend: IsaacUnifiedSingleContextBackend) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "asset_audit_ok": False, "blocker": "screw1_slot_missing"}]
    base = _base_env(backend)
    row: dict[str, Any] = {
        "part_name": "Screw1",
        "env_index": slot.global_env_index,
        "local_env_index": slot.local_env_index,
        "native_collision_proxy_used_as_pass_evidence": False,
        "runtime_proxy_allowed_for_pass": False,
    }
    identity = backend.object_identity_for_slot(slot)
    row.update(
        {
                "identity_verified": bool(identity.get("object_identity_verified")),
                "active_usd": identity.get("active_asset_usd", ""),
                "active_prim_path": identity.get("active_asset_prim_path", ""),
            }
        )
    try:
        after = dict(getattr(base, "v92_screw1_dynamic_repair_after", {}) or {})
        row.update(
            {
                "disable_gravity": after.get("disable_gravity", ""),
                "kinematic_enabled": after.get("kinematic_enabled", ""),
                "mass_kg": after.get("mass", ""),
                "config_contact_offset_m": after.get("contact_offset", ""),
                "config_rest_offset_m": after.get("rest_offset", ""),
            }
        )
        registry = getattr(base, "v83_active_asset_registry", {}).get("Screw1", {}) if base is not None else {}
        asset = registry.get("asset")
        row["asset_data_available"] = asset is not None
        stage_row = _screw1_stage_collision_row(slot, base)
        row.update(stage_row)
        sensor_names = list(getattr(base, "dex_fingertip_force_sensor_body_names", []) or [])
        row["active_fingertip_sensor_body_paths"] = ";".join(
            sensor_names[index] for index in (2, 3) if 0 <= index < len(sensor_names)
        )
        row["force_sensor_body_count"] = _tensor_float(getattr(base, "dex_force_sensor_body_count", None), slot.local_env_index)
        row["force_sensor_initialized_count"] = _tensor_float(
            getattr(base, "dex_force_sensor_initialized_count", None),
            slot.local_env_index,
        )
        chain_blocker = str(row.get("v95_contact_chain_blocker") or row.get("collision_blocker") or "")
        diagnostic_chain_blocker = chain_blocker if chain_blocker == "DIAGNOSTIC_ENV_OR_FRAME_MISMATCH" else ""
        asset_chain_blocker = "" if diagnostic_chain_blocker else chain_blocker
        row["diagnostic_contact_chain_blocker"] = diagnostic_chain_blocker
        row["asset_audit_ok"] = bool(
            row.get("identity_verified")
            and row.get("asset_data_available")
            and int(row.get("collision_enabled_prim_count") or 0) > 0
            and not asset_chain_blocker
        )
        row["blocker"] = "" if row["asset_audit_ok"] else (asset_chain_blocker or "screw1_native_asset_audit_failed")
    except Exception as exc:
        row["asset_audit_ok"] = False
        row["blocker"] = f"{type(exc).__name__}:{exc}"
    return [row]


def _run_active_sensor_self_check(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    finger_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "self_check_status": "NOT_EXECUTED", "blocker": "screw1_slot_missing"}]
    sensor_maps = _finger_sensor_maps(finger_rows).get("Screw1", {})
    base_row = _find_slot_row(plan_rows, slot)
    rows: list[dict[str, Any]] = []
    for logical_finger in (3, 4):
        row_plan = copy.deepcopy(base_row)
        table_top = _to_float(row_plan.get("table_top_z_m"), 0.72)
        row_plan["hand_target_local_x"] = 0.02
        row_plan["hand_target_local_y"] = -0.20
        row_plan["hand_target_local_z"] = table_top + 0.04
        row_plan["hand_target_local_xyz"] = [
            row_plan["hand_target_local_x"],
            row_plan["hand_target_local_y"],
            row_plan["hand_target_local_z"],
        ]
        row_plan["hand_target_source"] = "v95_active_sensor_self_check_table_reference"
        check_plan = [copy.deepcopy(row) for row in plan_rows]
        for index, existing in enumerate(check_plan):
            if str(existing.get("part_name") or "") == "Screw1":
                check_plan[index] = row_plan
        _reset_with_plan(backend, check_plan, settle_steps=4)
        tip_start = _active_tip_local_xyz(backend, slot, logical_finger - 1)
        for step in range(8):
            actions = [[0.0] * 16 for _slot in backend.slots]
            _v91_apply_finger_mask(
                actions[slot.local_env_index],
                group=str(logical_finger),
                active_value=min(0.65, 0.10 * float(step + 1)),
                support_value=0.0,
            )
            _set_context(backend, "v95_active_sensor_self_check_path_probe", finger=logical_finger, step=step)
            backend.step_envs(actions)
        tip_after = _active_tip_local_xyz(backend, slot, logical_finger - 1)
        close_path_delta = _vec_sub(tip_after, tip_start) if tip_start and tip_after else [0.0, 0.0, -1.0]
        close_path_delta_norm = _norm(close_path_delta)
        close_dir = _unit(close_path_delta) if close_path_delta_norm > 1.0e-5 else [0.0, 0.0, -1.0]
        _remove_v95_reference_pad(slot, logical_finger)
        _reset_with_plan(backend, check_plan, settle_steps=4)
        tip_reset = _active_tip_local_xyz(backend, slot, logical_finger - 1) or tip_start
        reference = _ensure_v95_reference_pad(
            backend,
            slot,
            logical_finger,
            tip_reset,
            close_dir,
            close_path_delta_norm,
        )
        _reset_with_plan(backend, check_plan, settle_steps=2)
        sensor_index = sensor_maps.get(logical_finger - 1, -1)
        active_peak = 0.0
        non_active_peak = 0.0
        active_count_peak = 0
        per_sensor_peak = [0.0] * 5
        initial_reference = _self_check_reference_distance_row(backend, slot, logical_finger - 1, table_top, reference=reference)
        best_reference_signed = _to_float(initial_reference.get("reference_signed_distance_m"), 1.0)
        reference_best_step = -1
        reference_geometry_reached = best_reference_signed <= 0.0015
        geometry_overlap_observed = _to_float(initial_reference.get("reference_signed_distance_m"), 1.0) <= 0.0
        nonzero_indices: set[int] = set()
        initial_metrics = backend._metrics_for_slot(slot, f"v95_active_sensor_self_check_f{logical_finger}_initial")
        for step in range(24):
            actions = [[0.0] * 16 for _slot in backend.slots]
            action = actions[slot.local_env_index]
            _v91_apply_finger_mask(
                action,
                group=str(logical_finger),
                active_value=min(0.95, 0.08 * float(step + 1)),
                support_value=0.0,
            )
            _set_context(backend, "v95_active_sensor_self_check", finger=logical_finger, step=step)
            backend.step_envs(actions)
            metrics = backend._metrics_for_slot(slot, f"v95_active_sensor_self_check_f{logical_finger}_{step}")
            forces = _v92_force_list(metrics)
            for index, value in enumerate(forces[:5]):
                per_sensor_peak[index] = max(per_sensor_peak[index], _to_float(value))
            reference_sample = _self_check_reference_distance_row(backend, slot, logical_finger - 1, table_top, reference=reference)
            reference_signed = _to_float(reference_sample.get("reference_signed_distance_m"), 1.0)
            if reference_signed < best_reference_signed:
                best_reference_signed = reference_signed
                reference_best_step = step
            reference_geometry_reached = bool(reference_geometry_reached or reference_signed <= 0.0015)
            geometry_overlap_observed = bool(geometry_overlap_observed or reference_signed <= 0.0)
            for index, value in enumerate(forces):
                if value > backend.contact_manager.force_threshold_n:
                    nonzero_indices.add(index)
            if 0 <= sensor_index < len(forces):
                active_peak = max(active_peak, _to_float(forces[sensor_index]))
                active_count_peak = max(
                    active_count_peak,
                    1 if _to_float(forces[sensor_index]) > backend.contact_manager.force_threshold_n else 0,
                )
            non_active_peak = max(
                [
                    non_active_peak,
                    *[
                        _to_float(value)
                        for index, value in enumerate(forces)
                        if index != sensor_index
                    ],
                ]
            )
        responding_sensor = _v92_max_index(per_sensor_peak) if max([0.0, *per_sensor_peak]) > backend.contact_manager.force_threshold_n else -1
        active_sensor_observed = active_peak > backend.contact_manager.force_threshold_n
        status = "ACTIVE_SENSOR_SELF_CHECK_PASSED" if active_sensor_observed else "ACTIVE_SENSOR_READOUT_FAILED"
        if sensor_index < 0:
            status = "FINGER_ACTION_SENSOR_MAPPING_UNCALIBRATED"
        mismatch = bool(non_active_peak > backend.contact_manager.force_threshold_n and active_peak <= backend.contact_manager.force_threshold_n)
        if mismatch:
            status = "SENSOR_MAPPING_MISMATCH"
        if reference_geometry_reached and active_peak <= backend.contact_manager.force_threshold_n and non_active_peak <= backend.contact_manager.force_threshold_n:
            status = "SENSOR_ATTACHMENT_OR_READOUT_FAILED"
        reference_distance_model_untrusted = bool(
            not reference_geometry_reached
            and active_sensor_observed
            and sensor_index >= 0
            and not mismatch
        )
        if reference_distance_model_untrusted:
            status = "ACTIVE_SENSOR_SELF_CHECK_PASSED_REFERENCE_DISTANCE_MODEL_UNTRUSTED"
        elif not reference_geometry_reached and sensor_index >= 0:
            status = "SELF_CHECK_GEOMETRY_DID_NOT_REACH_REFERENCE"
        native_pair_status = _native_pair_status_from_self_check(
            reference_geometry_reached=reference_geometry_reached,
            geometry_overlap_observed=geometry_overlap_observed,
            active_peak=active_peak,
            non_active_peak=non_active_peak,
            threshold=backend.contact_manager.force_threshold_n,
        )
        rows.append(
            {
                "part_name": "Screw1",
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "logical_finger_id": logical_finger,
                "expected_sensor_index": sensor_index,
                "self_check_reference": "v95_static_reference_pad_on_measured_close_path",
                "self_check_not_object_contact": True,
                "reference_pad_prim_path": reference.get("reference_pad_prim_path", ""),
                "reference_pad_spawned": bool(reference.get("reference_pad_spawned")),
                "reference_pad_contact_report_api": bool(reference.get("reference_pad_contact_report_api")),
                "reference_pad_collision_enabled": bool(reference.get("reference_pad_collision_enabled")),
                "reference_pad_half_extent_xyz": reference.get("reference_pad_half_extent_xyz", []),
                "reference_pad_center_local_xyz": reference.get("reference_pad_center_local_xyz", []),
                "reference_signed_distance_min_m": best_reference_signed,
                "reference_geometry_reached": reference_geometry_reached,
                "self_check_geometry_reached_reference": reference_geometry_reached,
                "reference_geometry_overlap_observed": geometry_overlap_observed,
                "reference_distance_model_untrusted": reference_distance_model_untrusted,
                "active_sensor_force_observed_despite_reference_point_clearance": reference_distance_model_untrusted,
                "reference_best_step": reference_best_step,
                "reference_setup_iterations": 1,
                "reference_initial_signed_distance_m": initial_reference.get("reference_signed_distance_m", 1.0),
                "reference_initial_tip_local_xyz": initial_reference.get("reference_tip_local_xyz", []),
                "reference_close_path_tip_start_local_xyz": tip_start,
                "reference_close_path_tip_after_local_xyz": tip_after,
                "reference_close_path_delta_local_xyz": close_path_delta,
                "reference_close_path_delta_norm_m": close_path_delta_norm,
                "reference_close_path_direction_local_xyz": close_dir,
                "reference_hand_target_local_z": row_plan.get("hand_target_local_z", ""),
                "commanded_policy_cols": f"{6 + logical_finger - 1},{11 + logical_finger - 1}",
                "commanded_isaac_cols": f"{16 + logical_finger - 1},{21 + logical_finger - 1}",
                "contact_sensor_api_available": bool(initial_metrics.get("contact_sensor_api_available")),
                "initial_force_peak_n": max([0.0, *_v92_force_list(initial_metrics)]),
                "active_sensor_force_peak_n": active_peak,
                "non_active_sensor_force_peak_n": non_active_peak,
                "per_sensor_force_peak_n": per_sensor_peak,
                "active_force_count_peak": active_count_peak,
                "nonzero_force_sensor_indices": sorted(nonzero_indices),
                "sensor_action_mapping_mismatch": mismatch,
                "responding_sensor_index": responding_sensor,
                "native_pair_contact_status": native_pair_status,
                "native_pair_contact_observed": native_pair_status
                in {
                    "PAIR_CONTACT_OBSERVED_BY_ACTIVE_SENSOR",
                    "ACTIVE_SENSOR_FORCE_OBSERVED_REFERENCE_DISTANCE_MODEL_UNTRUSTED",
                },
                "mapping_scan_status": "unique_sensor_mapping_observed"
                if (reference_geometry_reached or reference_distance_model_untrusted) and len(nonzero_indices) == 1
                else ("reference_not_reached" if not reference_geometry_reached else "no_unique_sensor_response"),
                "object_contact_success_evidence_used": False,
                "distance_only_success_used": False,
                "fallback_success_used": False,
                "self_check_status": status,
                "blocker": "" if status.startswith("ACTIVE_SENSOR_SELF_CHECK_PASSED") else status,
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "native_collision_proxy_used_as_pass_evidence": False,
                "live_probe_executed": True,
                "native_shutdown": False,
            }
        )
        _remove_v95_reference_pad(slot, logical_finger)
    return rows


def _run_screw1_contact_truth_triage(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    finger_rows: list[dict[str, Any]],
    wrist_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    sensor_self_check_rows: list[dict[str, Any]],
    asset_audit_rows: list[dict[str, Any]],
    trace_phase: str = "final",
    correction_pass: int = -1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [], [{"part_name": "Screw1", "blocker": "screw1_slot_missing", "actual_object_contact_ok": False}]
    sensor_map = _finger_sensor_maps(finger_rows).get("Screw1", {})
    wrist_ok = all(_bool(row.get("wrist_action_mapping_ok")) for row in wrist_rows if row.get("part_name") == "Screw1")
    trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    modes = (("single_finger", "3"), ("two_multi_finger", "34"))
    for mode, group in modes:
        logical = _v91_active_finger_indices(group)
        mapped = [sensor_map.get(index, -1) for index in logical]
        mapped = [index for index in mapped if 0 <= index < 5]
        mapping_ready = len(mapped) == len(logical)
        _reset_with_plan(backend, plan_rows, settle_steps=2)
        before_state = _v91_object_state_for_slot(backend, slot)
        initial_distance = _screw1_active_distance_row(backend, slot, logical)
        active_force_peak = 0.0
        summary_force_peak = _contact_force_for_mode(contact_rows, mode)
        non_active_force_peak = 0.0
        active_count_peak = 0
        nonzero_indices: set[int] = set()
        distance_blockers: set[str] = set()
        distance_valid_observed = False
        best_signed = _to_float(initial_distance.get("min_active_signed_surface_distance_m"), 1.0)
        best_abs = _to_float(initial_distance.get("min_active_abs_surface_distance_m"), 1.0)
        best_step = -1
        best_distance_row = dict(initial_distance)
        final_signed = best_signed
        final_abs = best_abs
        if str(initial_distance.get("distance_blocker") or ""):
            distance_blockers.add(str(initial_distance.get("distance_blocker") or ""))
        else:
            distance_valid_observed = True
        target_response_observed = False
        distance_decrease_peak = 0.0
        for step in range(24):
            actions = [[0.0] * 16 for _slot in backend.slots]
            if wrist_ok and mapping_ready:
                action = actions[slot.local_env_index]
                _v91_apply_finger_mask(
                    action,
                    group=group,
                    active_value=min(0.95, 0.08 * float(step + 1)),
                    support_value=0.0,
                )
            _set_context(backend, f"v95_screw1_contact_truth_{mode}", finger=0, step=step)
            backend.step_envs(actions)
            metrics = backend._metrics_for_slot(slot, f"v95_screw1_contact_truth_{mode}_{step}")
            distance = _screw1_active_distance_row(backend, slot, logical)
            state_delta = _v91_state_delta(before_state, _v91_object_state_for_slot(backend, slot))
            unfiltered_forces = _v92_force_list(metrics)
            target_forces = _v95_target_force_list(metrics)
            target_filtered_available = bool(metrics.get("target_filtered_force_available"))
            step_active_peak = max([0.0, *[target_forces[index] for index in mapped if 0 <= index < len(target_forces)]])
            step_non_active_peak = max(
                [
                    0.0,
                    *[
                        _to_float(value)
                        for index, value in enumerate(target_forces)
                        if index not in set(mapped)
                    ],
                ]
            )
            step_unfiltered_peak = max([0.0, *unfiltered_forces])
            active_force_peak = max(active_force_peak, step_active_peak)
            non_active_force_peak = max(non_active_force_peak, step_non_active_peak)
            active_count_peak = max(
                active_count_peak,
                sum(
                    1
                    for index in mapped
                    if 0 <= index < len(target_forces)
                    and target_forces[index] > backend.contact_manager.force_threshold_n
                ),
            )
            for index, value in enumerate(target_forces):
                if value > backend.contact_manager.force_threshold_n:
                    nonzero_indices.add(index)
            signed = _to_float(distance.get("min_active_signed_surface_distance_m"), 1.0)
            abs_dist = _to_float(distance.get("min_active_abs_surface_distance_m"), 1.0)
            if str(distance.get("distance_blocker") or ""):
                distance_blockers.add(str(distance.get("distance_blocker") or ""))
            else:
                distance_valid_observed = True
            if signed < best_signed:
                best_signed = signed
                best_abs = abs_dist
                best_step = step
                best_distance_row = dict(distance)
            else:
                best_abs = min(best_abs, abs_dist)
            final_signed = signed
            final_abs = abs_dist
            initial_signed = _to_float(initial_distance.get("min_active_signed_surface_distance_m"), 1.0)
            distance_decrease = initial_signed - signed
            distance_decrease_peak = max(distance_decrease_peak, distance_decrease)
            target_response = bool(
                _to_float(state_delta.get("root_pose_delta_m")) >= 1.0e-5
                or _to_float(metrics.get("object_velocity_norm")) >= 1.0e-4
            )
            target_response_observed = bool(target_response_observed or target_response)
            trace_rows.append(
                {
                    "part_name": "Screw1",
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "trace_phase": trace_phase,
                    "correction_pass": correction_pass,
                    "contact_gate_mode": mode,
                    "selected_finger_group": group,
                    "step": step,
                    "commanded_logical_finger_indices": logical,
                    "gate_active_sensor_indices": mapped,
                    "active_value": min(0.95, 0.08 * float(step + 1)) if wrist_ok and mapping_ready else 0.0,
                    "active_tip_world_xyz": distance.get("active_tip_world_xyz", []),
                    "active_tip_local_xyz": distance.get("active_tip_local_xyz", []),
                    "active_tip_distance_xyz": distance.get("active_tip_distance_xyz", []),
                    "closest_surface_point_local_xyz": distance.get("closest_surface_point_local_xyz", []),
                    "best_gap_vector_local_xyz": distance.get("best_gap_vector_local_xyz", []),
                    "best_gap_norm_m": distance.get("best_gap_norm_m", 1.0),
                    "closest_active_finger_index": distance.get("closest_active_finger_index", -1),
                    "screw1_root_pos_w": distance.get("screw1_root_pos_w", []),
                    "screw1_root_quat_w": distance.get("screw1_root_quat_w", []),
                    "env_origin_w": distance.get("env_origin_w", []),
                    "distance_frame": distance.get("distance_frame", ""),
                    "tip_tensor_shape": distance.get("tip_tensor_shape", ""),
                    "min_active_signed_surface_distance_m": signed,
                    "min_active_abs_surface_distance_m": abs_dist,
                    "per_active_signed_surface_distance_m": distance.get("per_active_signed_surface_distance_m", []),
                    "per_active_abs_surface_distance_m": distance.get("per_active_abs_surface_distance_m", []),
                    "per_active_distance_candidates": distance.get("per_active_distance_candidates", []),
                    "distance_blocker": distance.get("distance_blocker", ""),
                    "distance_decreased_from_initial_m": distance_decrease,
                    "preclose_signed_distance_m": initial_signed,
                    "close_toward_distance_m": distance_decrease_peak,
                    "reach_margin_m": distance_decrease_peak - initial_signed,
                    "distance_crossed_or_penetrated": signed <= 0.0,
                    "active_sensor_force_peak_n": step_active_peak,
                    "active_target_filtered_force_peak_n": step_active_peak,
                    "non_active_sensor_force_peak_n": step_non_active_peak,
                    "per_finger_force_norm": target_forces,
                    "per_finger_target_filtered_force_norm": target_forces,
                    "per_finger_unfiltered_force_norm": unfiltered_forces,
                    "unfiltered_force_peak_n": step_unfiltered_peak,
                    "target_filtered_force_available": target_filtered_available,
                    "target_contact_evidence_source": metrics.get("target_contact_evidence_source", ""),
                    "nonzero_force_sensor_indices": sorted(
                        index for index, value in enumerate(unfiltered_forces) if value > backend.contact_manager.force_threshold_n
                    ),
                    "nonzero_target_filtered_sensor_indices": sorted(
                        index for index, value in enumerate(target_forces) if value > backend.contact_manager.force_threshold_n
                    ),
                    "screw1_response_delta_m": state_delta.get("root_pose_delta_m", 0.0),
                    "screw1_velocity_norm": metrics.get("object_velocity_norm", 0.0),
                    "target_response_observed": target_response,
                    "contact_sensor_api_available": bool(metrics.get("contact_sensor_api_available")),
                    "table_collision": bool(metrics.get("table_collision")),
                    "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                    "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                    "native_collision_proxy_used_as_pass_evidence": False,
                    "distance_only_success_used": False,
                    "fallback_success_used": False,
                }
            )
        self_check_status = _self_check_status_for_group(sensor_self_check_rows, logical)
        asset_ok = bool(asset_audit_rows and asset_audit_rows[0].get("asset_audit_ok"))
        asset_blocker = str(
            (asset_audit_rows[0].get("v95_contact_chain_blocker") or asset_audit_rows[0].get("blocker") or "")
            if asset_audit_rows
            else ""
        )
        distance_trace_blocker = "" if distance_valid_observed else ";".join(sorted(distance_blockers)) or "no_valid_screw1_distance_trace"
        summary = _classify_screw1_truth(
            mode=mode,
            group=group,
            logical=logical,
            mapped=mapped,
            trace_phase=trace_phase,
            correction_pass=correction_pass,
            wrist_ok=wrist_ok,
            mapping_ready=mapping_ready,
            initial_signed=_to_float(initial_distance.get("min_active_signed_surface_distance_m"), 1.0),
            best_signed=best_signed,
            final_signed=final_signed,
            best_abs=best_abs,
            final_abs=final_abs,
            best_step=best_step,
            best_gap_vector_local_xyz=list(best_distance_row.get("best_gap_vector_local_xyz", [])),
            closest_surface_point_local_xyz=list(best_distance_row.get("closest_surface_point_local_xyz", [])),
            distance_decrease_peak=distance_decrease_peak,
            active_force_peak=active_force_peak,
            summary_force_peak=summary_force_peak,
            non_active_force_peak=non_active_force_peak,
            active_count_peak=active_count_peak,
            target_response_observed=target_response_observed,
            self_check_status=self_check_status,
            asset_ok=asset_ok,
            asset_blocker=asset_blocker,
            distance_trace_blocker=distance_trace_blocker,
            threshold=backend.contact_manager.force_threshold_n,
            object_write=bool(backend.object_write_by_policy_detected),
            sticky=bool(backend.sticky_action_available_to_policy),
        )
        summary_rows.append(summary)
    return trace_rows, summary_rows


def _apply_screw1_truth_summary(contact_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_by_mode = {str(row.get("contact_gate_mode") or ""): row for row in summary_rows}
    updated: list[dict[str, Any]] = []
    for row in contact_rows:
        clone = dict(row)
        if str(clone.get("part_name") or "") == "Screw1":
            summary = summary_by_mode.get(str(clone.get("contact_gate_mode") or ""))
            if summary:
                clone["screw1_truth_blocker"] = summary.get("blocker", "")
                clone["screw1_truth_classification"] = summary.get("truth_classification", "")
                clone["screw1_truth_active_force_peak_n"] = summary.get("active_force_peak_n", 0.0)
                clone["screw1_truth_best_signed_surface_distance_m"] = summary.get("best_signed_surface_distance_m", "")
                clone["screw1_truth_distance_decrease_peak_m"] = summary.get("distance_decrease_peak_m", "")
                clone["screw1_truth_reach_margin_m"] = summary.get("reach_margin_m", "")
                clone["screw1_truth_close_toward_distance_m"] = summary.get("close_toward_distance_m", "")
                clone["screw1_truth_best_step"] = summary.get("best_step", "")
                clone["screw1_truth_controller_or_workspace_limited"] = summary.get("controller_or_workspace_limited", False)
                clone["screw1_truth_distance_trace_blocker"] = summary.get("distance_trace_blocker", "")
                clone["screw1_truth_self_check_status"] = summary.get("self_check_status", "")
                clone["screw1_truth_asset_audit_ok"] = summary.get("asset_audit_ok", False)
                clone["screw1_truth_asset_contact_chain_blocker"] = summary.get("asset_contact_chain_blocker", "")
                if not _bool(summary.get("actual_object_contact_ok")):
                    clone["actual_object_contact_ok"] = False
                    clone["actual_object_force_contact"] = False
                    clone["target_object_contact_evidence_sufficient"] = False
                    clone["blocker"] = summary.get("blocker") or clone.get("blocker", "")
                else:
                    clone["screw1_truth_verified_target_filtered_contact"] = True
        updated.append(clone)
    return updated


def _apply_screw1_last_mile_correction(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    finger_rows: list[dict[str, Any]],
    wrist_rows: list[dict[str, Any]],
    asset_audit_rows: list[dict[str, Any]],
    initial_trace_rows: list[dict[str, Any]],
    initial_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    slot = _slot_by_part(backend, "Screw1")
    current_plan = [copy.deepcopy(row) for row in plan_rows]
    out: dict[str, Any] = {
        "plan_rows": current_plan,
        "trace_rows": list(initial_trace_rows),
        "summary_rows": list(initial_summary_rows),
        "pregrasp_trace_rows": [],
        "correction_rows": [],
        "plan_changed": False,
    }
    if slot is None:
        out["correction_rows"].append({"part_name": "Screw1", "correction_status": "NOT_EXECUTED", "blocker": "screw1_slot_missing"})
        return out

    summaries = list(initial_summary_rows)
    for pass_index in range(2):
        summary = _summary_for_mode(summaries, "single_finger") or (summaries[0] if summaries else {})
        if not _screw1_should_apply_last_mile_correction(summary, backend):
            out["correction_rows"].append(
                {
                    "part_name": "Screw1",
                    "correction_pass": pass_index,
                    "correction_status": "not_needed",
                    "source_blocker": summary.get("blocker", ""),
                    "best_signed_surface_distance_m": summary.get("best_signed_surface_distance_m", ""),
                    "reach_margin_m": summary.get("reach_margin_m", ""),
                }
            )
            break

        adjustment = _screw1_last_mile_adjustment(summary)
        if _norm(adjustment) <= 1.0e-6:
            out["correction_rows"].append(
                {
                    "part_name": "Screw1",
                    "correction_pass": pass_index,
                    "correction_status": "not_applied",
                    "blocker": "screw1_gap_vector_unusable",
                    "source_blocker": summary.get("blocker", ""),
                }
            )
            break

        candidate_plan = [copy.deepcopy(row) for row in current_plan]
        row = _find_slot_row_ref(candidate_plan, slot)
        if row is None:
            out["correction_rows"].append(
                {
                    "part_name": "Screw1",
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "correction_pass": pass_index,
                    "correction_status": "not_applied",
                    "blocker": "screw1_plan_row_missing",
                    "source_blocker": summary.get("blocker", ""),
                }
            )
            break
        before_target = _row_hand_target(row)
        _apply_hand_adjustment(row, adjustment, backend)
        row["hand_target_source"] = "v95_screw1_last_mile_pregrasp_correction"
        row["screw1_last_mile_correction_pass"] = pass_index
        row["screw1_last_mile_correction_source_best_signed_m"] = summary.get("best_signed_surface_distance_m", "")
        row["screw1_last_mile_correction_source_reach_margin_m"] = summary.get("reach_margin_m", "")
        row["screw1_last_mile_correction_gap_vector_local_xyz"] = summary.get("best_gap_vector_local_xyz", [])

        sample = _sample_v95_plan(
            backend,
            candidate_plan,
            phase="screw1_last_mile_correction",
            iteration=pass_index,
            settle_steps=4,
        )
        out["pregrasp_trace_rows"].extend(sample.get("trace_rows", []))
        screw_sample = _find_slot_row(sample.get("sample_rows", []), slot)
        sample_ok = _sample_motion_force_ok(screw_sample, backend)
        reset_blocker = "" if sample_ok else _sample_blocker(screw_sample, backend)
        distance_after = _screw1_active_distance_row(backend, slot, _v91_active_finger_indices("3"))
        out["correction_rows"].append(
            {
                "part_name": "Screw1",
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "correction_pass": pass_index,
                "correction_status": "applied_and_kept" if sample_ok else "rejected_reset_safety",
                "source_blocker": summary.get("blocker", ""),
                "source_truth_classification": summary.get("truth_classification", ""),
                "source_best_signed_surface_distance_m": summary.get("best_signed_surface_distance_m", ""),
                "source_reach_margin_m": summary.get("reach_margin_m", ""),
                "adjustment_local_xyz": adjustment,
                "adjustment_norm_m": _norm(adjustment),
                "hand_target_before_xyz": before_target,
                "hand_target_after_xyz": _row_hand_target(row),
                "post_reset_force_peak_n": screw_sample.get("force_peak_n", 0.0),
                "post_reset_force_count": screw_sample.get("force_count", 0),
                "post_reset_displacement_m": screw_sample.get("pre_contact_displacement_m", 0.0),
                "post_reset_z_drift_m": screw_sample.get("pre_contact_z_drift_m", 0.0),
                "post_reset_signed_distance_m": distance_after.get("min_active_signed_surface_distance_m", ""),
                "post_reset_distance_blocker": distance_after.get("distance_blocker", ""),
                "reset_safety_ok": sample_ok,
                "blocker": reset_blocker,
            }
        )
        if not sample_ok:
            break

        current_plan = candidate_plan
        out["plan_rows"] = current_plan
        out["plan_changed"] = True
        trace_rows, summaries = _run_screw1_contact_truth_triage(
            backend,
            current_plan,
            finger_rows=finger_rows,
            wrist_rows=wrist_rows,
            contact_rows=[],
            sensor_self_check_rows=[],
            asset_audit_rows=asset_audit_rows,
            trace_phase="last_mile_correction_check",
            correction_pass=pass_index,
        )
        out["trace_rows"].extend(trace_rows)
        out["summary_rows"] = summaries
        check_summary = _summary_for_mode(summaries, "single_finger") or (summaries[0] if summaries else {})
        if _to_float(check_summary.get("best_signed_surface_distance_m"), 1.0) <= 0.0015:
            break
    return out


def _screw1_target_filter_coverage_audit(backend: IsaacUnifiedSingleContextBackend) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "filter_coverage_ok": False, "blocker": "screw1_slot_missing"}]
    row: dict[str, Any] = {
        "part_name": "Screw1",
        "env_index": slot.global_env_index,
        "local_env_index": slot.local_env_index,
        "finger_side_filter_expr": "/World/envs/env_.*/Screw1",
        "collision_child_filter_candidate_expr": "/World/envs/env_.*/Screw1/geometry/mesh",
        "native_collision_proxy_used_as_pass_evidence": False,
    }
    try:
        import omni.usd  # noqa: WPS433
        from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            row.update({"filter_coverage_ok": False, "blocker": "stage_unavailable"})
            return [row]
        root_path = f"/World/envs/env_{slot.local_env_index}/Screw1"
        filter_path = root_path
        collision_child_path = f"{root_path}/geometry/mesh"
        root = stage.GetPrimAtPath(root_path)
        filter_prim = stage.GetPrimAtPath(filter_path)
        collision_child = stage.GetPrimAtPath(collision_child_path)
        collision_paths = []
        contact_report_paths = []
        rigid_body_paths = []
        if root and root.IsValid():
            for prim in Usd.PrimRange(root):
                collision_api = UsdPhysics.CollisionAPI(prim)
                try:
                    if collision_api and bool(collision_api.GetCollisionEnabledAttr().Get()):
                        collision_paths.append(str(prim.GetPath()))
                except Exception:
                    pass
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    contact_report_paths.append(str(prim.GetPath()))
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body_paths.append(str(prim.GetPath()))
        root_collision = False
        if root and root.IsValid():
            try:
                root_collision = bool(UsdPhysics.CollisionAPI(root).GetCollisionEnabledAttr().Get())
            except Exception:
                root_collision = False
        filter_collision = False
        filter_report = False
        filter_rigid_body = False
        if filter_prim and filter_prim.IsValid():
            try:
                filter_collision = bool(UsdPhysics.CollisionAPI(filter_prim).GetCollisionEnabledAttr().Get())
            except Exception:
                filter_collision = False
            filter_report = bool(filter_prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
            filter_rigid_body = bool(filter_prim.HasAPI(UsdPhysics.RigidBodyAPI))
        collision_child_valid = bool(collision_child and collision_child.IsValid())
        collision_child_collision = False
        collision_child_report = False
        if collision_child_valid:
            try:
                collision_child_collision = bool(UsdPhysics.CollisionAPI(collision_child).GetCollisionEnabledAttr().Get())
            except Exception:
                collision_child_collision = False
            collision_child_report = bool(collision_child.HasAPI(PhysxSchema.PhysxContactReportAPI))
        coverage_ok = bool(
            filter_prim
            and filter_prim.IsValid()
            and filter_rigid_body
            and collision_paths
        )
        row.update(
            {
                "stage_root_path": root_path,
                "stage_root_valid": bool(root and root.IsValid()),
                "resolved_filter_path": filter_path,
                "resolved_filter_valid": bool(filter_prim and filter_prim.IsValid()),
                "root_has_collision_api": root_collision,
                "filter_has_collision_api": filter_collision,
                "filter_has_contact_report_api": filter_report,
                "filter_has_rigid_body_api": filter_rigid_body,
                "collision_child_path": collision_child_path,
                "collision_child_valid": collision_child_valid,
                "collision_child_has_collision_api": collision_child_collision,
                "collision_child_has_contact_report_api": collision_child_report,
                "collision_enabled_prim_count": len(collision_paths),
                "resolved_collider_body_paths": ";".join(collision_paths),
                "contact_report_body_paths": ";".join(contact_report_paths),
                "rigid_body_paths": ";".join(rigid_body_paths),
                "filter_points_to_collision_child": False,
                "filter_covers_contact_report_body": bool(filter_path in contact_report_paths),
                "root_filter_would_be_ambiguous": False,
                "collision_child_filter_candidate_is_rigid_body": bool(collision_child and collision_child.HasAPI(UsdPhysics.RigidBodyAPI)),
                "collision_child_filter_candidate_runtime_supported": False,
                "recommended_filter_expr": "/World/envs/env_.*/Screw1",
                "filter_coverage_ok": coverage_ok,
                "blocker": "" if coverage_ok else "screw1_filter_rigid_body_or_collision_coverage_unavailable",
            }
        )
    except Exception as exc:
        row.update({"filter_coverage_ok": False, "blocker": f"{type(exc).__name__}:{exc}"})
    return [row]


def _init_contact_sensor_for_audit(sensor: Any) -> str:
    if sensor is None:
        return "sensor_missing"
    if not getattr(sensor, "is_initialized", False):
        try:
            sensor._initialize_impl()
            sensor._is_initialized = True
        except Exception as exc:
            return f"init_error={type(exc).__name__}:{exc}"
    return ""


def _sensor_cfg_value(sensor: Any, name: str, default: Any = "") -> Any:
    return getattr(getattr(sensor, "cfg", None), name, default)


def _sensor_body_names(sensor: Any) -> list[str]:
    try:
        return [str(item) for item in list(getattr(sensor, "body_names", []) or [])]
    except Exception:
        return []


def _sensor_body_leaf(sensor: Any) -> str:
    names = _sensor_body_names(sensor)
    if not names:
        return ""
    return str(names[0]).rstrip("/").split("/")[-1]


def _sensor_filter_count(sensor: Any) -> int:
    try:
        return int(getattr(getattr(sensor, "contact_physx_view", None), "filter_count", 0) or 0)
    except Exception:
        return 0


def _sensor_shape(value: Any) -> str:
    try:
        shape = getattr(value, "shape", None)
        return str(tuple(shape)) if shape is not None else ""
    except Exception:
        return ""


def _sensor_force_shapes(sensor: Any) -> tuple[str, str]:
    data = getattr(sensor, "data", None)
    return _sensor_shape(getattr(data, "net_forces_w", None)), _sensor_shape(getattr(data, "force_matrix_w", None))


def _concrete_filter_paths_for_env(filter_exprs: list[str], env_index: int) -> list[str]:
    out: list[str] = []
    try:
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
    except Exception:
        stage = None
    for expr in filter_exprs:
        concrete = (
            str(expr)
            .replace("{ENV_REGEX_NS}", f"/World/envs/env_{int(env_index)}")
            .replace("/env_.*/", f"/env_{int(env_index)}/")
            .replace("env_.*", f"env_{int(env_index)}")
        )
        if stage is not None:
            prim = stage.GetPrimAtPath(concrete)
            if prim and prim.IsValid():
                out.append(concrete)
                continue
        out.append(f"{concrete}:UNRESOLVED")
    return out


def _filtered_sensor_structure_audit(backend: IsaacUnifiedSingleContextBackend) -> list[dict[str, Any]]:
    base = _base_env(backend)
    if base is None:
        return [{"sensor_kind": "filtered", "filtered_sensor_structure_ok": False, "blocker": "env_unavailable"}]
    rows: list[dict[str, Any]] = []
    unfiltered_sensors = list(getattr(base, "_dex_fingertip_force_sensors", []) or [])
    target_sensors = list(getattr(base, "_dex_fingertip_target_force_sensors", []) or [])
    reference_sensors = list(getattr(base, "_dex_fingertip_reference_force_sensors", []) or [])
    reference_mirror_sensors = list(getattr(base, "_v95_reference_mirror_force_sensors", []) or [])
    body_names = list(getattr(base, "dex_fingertip_force_sensor_body_names", []) or [])
    for finger_idx in range(5):
        logical_finger = finger_idx + 1
        unfiltered = unfiltered_sensors[finger_idx] if finger_idx < len(unfiltered_sensors) else None
        unfiltered_init = _init_contact_sensor_for_audit(unfiltered)
        unfiltered_body_names = _sensor_body_names(unfiltered)
        unfiltered_leaf = _sensor_body_leaf(unfiltered)
        unfiltered_net_shape, _unfiltered_matrix_shape = _sensor_force_shapes(unfiltered)
        for sensor_kind, sensors, expected_filter_count in (
            ("target_filtered", target_sensors, len(V83_PARTS)),
            ("reference_filtered", reference_sensors, 1),
        ):
            sensor = sensors[finger_idx] if finger_idx < len(sensors) else None
            init_error = _init_contact_sensor_for_audit(sensor)
            body = _sensor_body_leaf(sensor)
            body_names_sensor = _sensor_body_names(sensor)
            filter_exprs = _string_list(_sensor_cfg_value(sensor, "filter_prim_paths_expr", []))
            _net_shape, matrix_shape = _sensor_force_shapes(sensor)
            filter_count = _sensor_filter_count(sensor)
            num_bodies = len(body_names_sensor)
            source_match = bool(unfiltered_leaf and body == unfiltered_leaf)
            one_body = num_bodies == 1
            count_ok = filter_count >= int(expected_filter_count)
            ok = bool(not init_error and not unfiltered_init and one_body and source_match and count_ok)
            blocker = ""
            if unfiltered_init:
                blocker = unfiltered_init
            elif init_error:
                blocker = init_error
            elif not one_body:
                blocker = "FILTERED_SENSOR_SOURCE_BODY_MISMATCH"
            elif not source_match:
                blocker = "FILTERED_SENSOR_SOURCE_BODY_MISMATCH"
            elif not count_ok:
                blocker = "FILTERED_SENSOR_FILTER_COUNT_INVALID"
            rows.append(
                {
                    "logical_finger_id": logical_finger,
                    "finger_index": finger_idx,
                    "expected_body_name": body_names[finger_idx] if finger_idx < len(body_names) else "",
                    "sensor_kind": sensor_kind,
                    "unfiltered_prim_path": _sensor_cfg_value(unfiltered, "prim_path", ""),
                    "filtered_prim_path": _sensor_cfg_value(sensor, "prim_path", ""),
                    "unfiltered_num_bodies": len(unfiltered_body_names),
                    "filtered_num_bodies": num_bodies,
                    "unfiltered_body_names": ";".join(unfiltered_body_names),
                    "filtered_body_names": ";".join(body_names_sensor),
                    "source_body_matches_unfiltered": source_match,
                    "filtered_sensor_one_source_body": one_body,
                    "filter_count": filter_count,
                    "expected_filter_count": int(expected_filter_count),
                    "filter_prim_paths_expr": ";".join(filter_exprs),
                    "resolved_filter_paths_env0": ";".join(_concrete_filter_paths_for_env(filter_exprs, 0)),
                    "unfiltered_net_forces_w_shape": unfiltered_net_shape,
                    "filtered_force_matrix_w_shape": matrix_shape,
                    "filtered_sensor_structure_ok": ok,
                    "blocker": blocker,
                    "training_locked": True,
                }
            )
        mirror = reference_mirror_sensors[finger_idx] if finger_idx < len(reference_mirror_sensors) else None
        mirror_init = _init_contact_sensor_for_audit(mirror)
        mirror_exprs = _string_list(_sensor_cfg_value(mirror, "filter_prim_paths_expr", []))
        mirror_net_shape, mirror_matrix_shape = _sensor_force_shapes(mirror)
        rows.append(
            {
                "logical_finger_id": logical_finger,
                "finger_index": finger_idx,
                "expected_body_name": body_names[finger_idx] if finger_idx < len(body_names) else "",
                "sensor_kind": "reference_mirror",
                "unfiltered_prim_path": _sensor_cfg_value(unfiltered, "prim_path", ""),
                "filtered_prim_path": _sensor_cfg_value(mirror, "prim_path", ""),
                "unfiltered_num_bodies": len(unfiltered_body_names),
                "filtered_num_bodies": len(_sensor_body_names(mirror)),
                "unfiltered_body_names": ";".join(unfiltered_body_names),
                "filtered_body_names": ";".join(_sensor_body_names(mirror)),
                "source_body_matches_unfiltered": True,
                "filtered_sensor_one_source_body": len(_sensor_body_names(mirror)) == 1,
                "filter_count": _sensor_filter_count(mirror),
                "expected_filter_count": 1,
                "filter_prim_paths_expr": ";".join(mirror_exprs),
                "resolved_filter_paths_env0": ";".join(_concrete_filter_paths_for_env(mirror_exprs, 0)),
                "unfiltered_net_forces_w_shape": mirror_net_shape,
                "filtered_force_matrix_w_shape": mirror_matrix_shape,
                "filtered_sensor_structure_ok": bool(
                    not mirror_init and len(_sensor_body_names(mirror)) == 1 and _sensor_filter_count(mirror) >= 1
                ),
                "blocker": mirror_init
                if mirror_init
                else ("" if len(_sensor_body_names(mirror)) == 1 and _sensor_filter_count(mirror) >= 1 else "REFERENCE_MIRROR_SENSOR_INVALID"),
                "training_locked": True,
            }
        )
    return rows


def _run_cube_cube_contact_canary(backend: IsaacUnifiedSingleContextBackend) -> list[dict[str, Any]]:
    base = _base_env(backend)
    threshold = float(backend.contact_manager.force_threshold_n)
    row: dict[str, Any] = {
        "canary_name": "cube_cube_contact_canary",
        "local_env_index": 0,
        "cube_cube_canary_ok": False,
        "cube_a_net_force_peak_n": 0.0,
        "cube_b_net_force_peak_n": 0.0,
        "cube_a_to_b_filtered_force_peak_n": 0.0,
        "cube_b_to_a_filtered_force_peak_n": 0.0,
        "cube_a_to_b_filter_count": 0,
        "cube_b_to_a_filter_count": 0,
        "cube_a_source_body_count": 0,
        "cube_b_source_body_count": 0,
        "force_threshold_n": threshold,
        "evidence_source": "ContactSensor.net_forces_w+force_matrix_w",
        "object_contact_success_evidence_used": False,
        "training_locked": True,
        "blocker": "",
    }
    if base is None or not hasattr(base, "v95_place_canary_cubes"):
        row["blocker"] = "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED"
        row["canary_reset_blocker"] = "v95_place_canary_cubes_unavailable"
        return [row]
    reset_info = base.v95_place_canary_cubes(local_env_index=0)
    row.update(reset_info)
    if not _bool(reset_info.get("canary_cube_reset_ok")):
        row["blocker"] = "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED"
        return [row]
    for step in range(180):
        _set_context(backend, "v95_cube_cube_contact_canary", step=step)
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        row["cube_a_net_force_peak_n"] = max(
            _to_float(row.get("cube_a_net_force_peak_n")),
            _tensor_float(getattr(base, "v95_canary_cube_a_net_force_norm", None), 0),
        )
        row["cube_b_net_force_peak_n"] = max(
            _to_float(row.get("cube_b_net_force_peak_n")),
            _tensor_float(getattr(base, "v95_canary_cube_b_net_force_norm", None), 0),
        )
        row["cube_a_to_b_filtered_force_peak_n"] = max(
            _to_float(row.get("cube_a_to_b_filtered_force_peak_n")),
            _tensor_float(getattr(base, "v95_canary_cube_ab_force_norm", None), 0),
        )
        row["cube_b_to_a_filtered_force_peak_n"] = max(
            _to_float(row.get("cube_b_to_a_filtered_force_peak_n")),
            _tensor_float(getattr(base, "v95_canary_cube_ba_force_norm", None), 0),
        )
    row["cube_a_to_b_filter_count"] = _tensor_float(getattr(base, "v95_canary_cube_ab_filter_count", None), 0)
    row["cube_b_to_a_filter_count"] = _tensor_float(getattr(base, "v95_canary_cube_ba_filter_count", None), 0)
    row["cube_a_source_body_count"] = _tensor_float(getattr(base, "v95_canary_cube_ab_body_count", None), 0)
    row["cube_b_source_body_count"] = _tensor_float(getattr(base, "v95_canary_cube_ba_body_count", None), 0)
    row["cube_cube_canary_ok"] = bool(
        _to_float(row.get("cube_a_net_force_peak_n")) > threshold
        and _to_float(row.get("cube_b_net_force_peak_n")) > threshold
        and _to_float(row.get("cube_a_to_b_filtered_force_peak_n")) > threshold
        and _to_float(row.get("cube_b_to_a_filtered_force_peak_n")) > threshold
        and _to_float(row.get("cube_a_to_b_filter_count")) >= 1
        and _to_float(row.get("cube_b_to_a_filter_count")) >= 1
        and _to_float(row.get("cube_a_source_body_count")) == 1
        and _to_float(row.get("cube_b_source_body_count")) == 1
    )
    row["blocker"] = "" if row["cube_cube_canary_ok"] else "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED"
    return [row]


def _move_v95_canary_cube_runtime(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    cube_name: str,
    center_local_xyz: list[float],
) -> dict[str, Any]:
    base = _base_env(backend)
    if base is not None and hasattr(base, "v95_move_canary_cube"):
        return dict(base.v95_move_canary_cube(slot.local_env_index, str(cube_name), center_local_xyz))
    return {
        "canary_cube_name": str(cube_name),
        "canary_cube_center_local_xyz": list(center_local_xyz[:3]),
        "canary_cube_blocker": "v95_move_canary_cube_unavailable",
        "canary_cube_rigid_object": False,
    }


def _v95_canary_cube_pose(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    cube_name: str,
) -> dict[str, Any]:
    base = _base_env(backend)
    if base is not None and hasattr(base, "v95_canary_cube_local_pose"):
        try:
            return dict(base.v95_canary_cube_local_pose(slot.local_env_index, str(cube_name)))
        except Exception as exc:
            return {"canary_cube_pose_blocker": f"{type(exc).__name__}:{exc}"}
    return {"canary_cube_pose_blocker": "v95_canary_cube_local_pose_unavailable"}


def _v95_screw1_reference_force_values(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
) -> dict[str, float]:
    base = _base_env(backend)
    return {
        "screw1_unfiltered_force_n": _tensor_float(
            getattr(base, "v95_screw1_reference_screw1_net_force_norm", None),
            slot.local_env_index,
        ),
        "reference_unfiltered_force_n": _tensor_float(
            getattr(base, "v95_screw1_reference_reference_net_force_norm", None),
            slot.local_env_index,
        ),
        "reference_to_screw1_filtered_force_n": _tensor_float(
            getattr(base, "v95_screw1_reference_filtered_force_norm", None),
            slot.local_env_index,
        ),
        "screw1_to_reference_mirror_force_n": _tensor_float(
            getattr(base, "v95_screw1_reference_mirror_force_norm", None),
            slot.local_env_index,
        ),
        "reference_filter_count": _tensor_float(
            getattr(base, "v95_screw1_reference_filter_count", None),
            slot.local_env_index,
        ),
        "mirror_filter_count": _tensor_float(
            getattr(base, "v95_screw1_reference_mirror_filter_count", None),
            slot.local_env_index,
        ),
        "reference_body_count": _tensor_float(
            getattr(base, "v95_screw1_reference_body_count", None),
            slot.local_env_index,
        ),
        "mirror_body_count": _tensor_float(
            getattr(base, "v95_screw1_reference_mirror_body_count", None),
            slot.local_env_index,
        ),
        "force_valid": _tensor_float(
            getattr(base, "v95_screw1_reference_force_valid", None),
            slot.local_env_index,
        ),
    }


def _v95_finger_body_snapshot(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_finger: int,
) -> dict[str, Any]:
    base = _base_env(backend)
    if base is not None and hasattr(base, "v95_finger_contact_body_snapshot"):
        try:
            return dict(base.v95_finger_contact_body_snapshot(slot.local_env_index, int(logical_finger)))
        except Exception as exc:
            return {"snapshot_blocker": f"{type(exc).__name__}:{exc}"}
    return {"snapshot_blocker": "v95_finger_contact_body_snapshot_unavailable"}


def _v95_reference_pad_pose(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_finger: int,
) -> dict[str, Any]:
    base = _base_env(backend)
    if base is not None and hasattr(base, "v95_reference_pad_local_pose"):
        try:
            return dict(base.v95_reference_pad_local_pose(slot.local_env_index, int(logical_finger)))
        except Exception as exc:
            return {"reference_pad_pose_blocker": f"{type(exc).__name__}:{exc}"}
    return {"reference_pad_pose_blocker": "v95_reference_pad_local_pose_unavailable"}


def _finger_structure_sources(
    structure_rows: list[dict[str, Any]],
    logical_finger: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in structure_rows:
        if _int_field(row, "logical_finger_id", -1) != int(logical_finger):
            continue
        out[str(row.get("sensor_kind") or "")] = dict(row)
    return out


def _finger_contact_body_audit(
    backend: IsaacUnifiedSingleContextBackend,
    structure_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "body_audit_ok": False, "blocker": "screw1_slot_missing"}]
    rows: list[dict[str, Any]] = []
    for logical_finger in range(1, 6):
        sources = _finger_structure_sources(structure_rows, logical_finger)
        target = sources.get("target_filtered", {})
        reference = sources.get("reference_filtered", {})
        mirror = sources.get("reference_mirror", {})
        snapshot = _v95_finger_body_snapshot(backend, slot, logical_finger)
        pad_pose = _v95_reference_pad_pose(backend, slot, logical_finger)
        source_local = _vec3(snapshot.get("source_body_runtime_local_xyz", []))
        true_tip_local = _vec3(snapshot.get("true_tip_runtime_local_xyz", []))
        anchor_local = _vec3(snapshot.get("link4_anchor_runtime_local_xyz", []))
        pad_local = _vec3(pad_pose.get("reference_pad_runtime_center_local_xyz", []))
        pad_available = bool(pad_pose.get("reference_pad_pose_available"))
        source_available = bool(source_local and _int_field(snapshot, "source_body_index", -1) >= 0)
        true_tip_available = bool(true_tip_local and _int_field(snapshot, "true_tip_body_index", -1) >= 0)
        anchor_available = bool(anchor_local and _int_field(snapshot, "link4_anchor_body_index", -1) >= 0)
        source_body = str(snapshot.get("source_body_name") or "")
        target_body = str(target.get("filtered_body_names") or "").split(";")[0]
        reference_body = str(reference.get("filtered_body_names") or "").split(";")[0]
        unfiltered_body = str(target.get("unfiltered_body_names") or reference.get("unfiltered_body_names") or "").split(";")[0]
        source_matches = bool(
            source_body
            and unfiltered_body == source_body
            and target_body == source_body
            and reference_body == source_body
            and _bool(target.get("source_body_matches_unfiltered", True))
            and _bool(reference.get("source_body_matches_unfiltered", True))
        )
        row = {
            "part_name": "Screw1",
            "logical_finger_id": logical_finger,
            "finger_index": logical_finger - 1,
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "unfiltered_sensor_body_names": unfiltered_body,
            "target_filtered_source_body_names": target_body,
            "reference_filtered_source_body_names": reference_body,
            "reference_mirror_source_body_names": str(mirror.get("filtered_body_names") or ""),
            "filtered_source_body_matches_unfiltered": source_matches,
            "robot_source_body_name": source_body,
            "robot_source_body_index": snapshot.get("source_body_index", -1),
            "source_body_runtime_local_xyz": snapshot.get("source_body_runtime_local_xyz", []),
            "source_body_runtime_world_xyz": snapshot.get("source_body_runtime_world_xyz", []),
            "link4_anchor_body_name": snapshot.get("link4_anchor_body_name", ""),
            "link4_anchor_body_index": snapshot.get("link4_anchor_body_index", -1),
            "link4_anchor_runtime_local_xyz": snapshot.get("link4_anchor_runtime_local_xyz", []),
            "true_tip_body_name": snapshot.get("true_tip_body_name", ""),
            "true_tip_body_index": snapshot.get("true_tip_body_index", -1),
            "true_tip_runtime_local_xyz": snapshot.get("true_tip_runtime_local_xyz", []),
            "source_to_link4_anchor_distance_m": snapshot.get("source_to_link4_anchor_distance_m", 0.0),
            "source_to_true_tip_distance_m": snapshot.get("source_to_true_tip_distance_m", 0.0),
            "reference_pad_runtime_center_local_xyz": pad_pose.get("reference_pad_runtime_center_local_xyz", []),
            "reference_pad_to_source_body_distance_m": _norm(_vec_sub(pad_local, source_local))
            if pad_available and source_available
            else "",
            "reference_pad_to_true_tip_distance_m": _norm(_vec_sub(pad_local, true_tip_local))
            if pad_available and true_tip_available
            else "",
            "reference_pad_to_link4_anchor_distance_m": _norm(_vec_sub(pad_local, anchor_local))
            if pad_available and anchor_available
            else "",
            "source_body_collision_enabled_prim_count": snapshot.get("source_body_collision_enabled_prim_count", 0),
            "source_body_contact_report_api": bool(snapshot.get("source_body_contact_report_api")),
            "source_body_has_effective_collider_or_reporter": bool(
                snapshot.get("source_body_has_effective_collider_or_reporter")
            ),
            "diagnostic_candidate_force_peaks_n": {},
            "old_self_check_actual_force_body_name": "",
            "old_self_check_force_source_classification": "not_yet_reproduced",
            "body_audit_ok": bool(
                source_matches
                and source_available
                and bool(snapshot.get("source_body_has_effective_collider_or_reporter"))
                and not str(snapshot.get("snapshot_blocker") or "")
            ),
            "blocker": str(snapshot.get("snapshot_blocker") or "")
            or ("" if source_matches else "FILTERED_SENSOR_SOURCE_BODY_MISMATCH")
            or ("" if bool(snapshot.get("source_body_has_effective_collider_or_reporter")) else "SOURCE_BODY_COLLIDER_OR_REPORTER_MISSING"),
            "object_contact_success_evidence_used": False,
            "training_locked": True,
        }
        rows.append(row)
    return rows


def _merge_finger_contact_body_audit_with_canary(
    audit_rows: list[dict[str, Any]],
    canary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canary_by_finger = {_int_field(row, "logical_finger_id", -1): row for row in canary_rows}
    merged: list[dict[str, Any]] = []
    for row in audit_rows:
        out = dict(row)
        logical = _int_field(row, "logical_finger_id", -1)
        canary = canary_by_finger.get(logical)
        if canary:
            out.update(
                {
                    "reference_pad_final_center_local_xyz": canary.get("reference_pad_center_local_xyz", []),
                    "reference_pad_final_to_source_body_distance_m": canary.get(
                        "reference_pad_to_source_body_distance_m", ""
                    ),
                    "reference_pad_bracket_offsets_local_xyz": canary.get("reference_bracket_offsets_local_xyz", []),
                    "active_unfiltered_force_peak_n": canary.get("active_unfiltered_force_peak_n", 0.0),
                    "finger_to_reference_filtered_force_peak_n": canary.get(
                        "finger_to_reference_filtered_force_peak_n", 0.0
                    ),
                    "reference_to_finger_mirror_force_peak_n": canary.get(
                        "reference_to_finger_mirror_force_peak_n", 0.0
                    ),
                    "active_unfiltered_force_observed": bool(canary.get("active_unfiltered_force_observed")),
                    "finger_to_reference_filtered_force_observed": bool(
                        canary.get("finger_to_reference_filtered_force_observed")
                    ),
                    "reference_to_finger_mirror_force_observed": bool(
                        canary.get("reference_to_finger_mirror_force_observed")
                    ),
                    "canary_force_below_object_threshold": bool(
                        canary.get("canary_force_below_object_threshold")
                    ),
                    "canary_detection_threshold_n": canary.get("canary_detection_threshold_n", ""),
                    "object_contact_force_threshold_n": canary.get("object_contact_force_threshold_n", ""),
                    "diagnostic_candidate_force_peaks_n": canary.get("diagnostic_candidate_force_peaks_n", {}),
                    "old_self_check_actual_force_body_name": canary.get("responding_diagnostic_body_name", ""),
                    "old_self_check_force_source_classification": canary.get(
                        "finger_reference_force_source_classification", ""
                    ),
                    "finger_reference_canary_ok": bool(canary.get("finger_reference_canary_ok")),
                    "finger_reference_canary_blocker": canary.get("blocker", ""),
                }
            )
        merged.append(out)
    return merged


def _v95_reference_mirror_force_list(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> list[float]:
    base = _base_env(backend)
    return _tensor_list(getattr(base, "v95_reference_mirror_force_norm", None), slot.local_env_index, width=5)


def _move_v95_reference_pad_runtime(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_finger: int,
    center_local_xyz: list[float],
) -> dict[str, Any]:
    base = _base_env(backend)
    if base is not None and hasattr(base, "v95_move_reference_pad"):
        return dict(base.v95_move_reference_pad(slot.local_env_index, int(logical_finger), center_local_xyz))
    return _move_v95_reference_pad(slot, logical_finger, center_local_xyz)


def _v95_diagnostic_body_force_peaks(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_finger: int,
) -> dict[str, float]:
    base = _base_env(backend)
    rows = list(getattr(base, "_v95_finger_body_audit_force_sensors", []) or []) if base is not None else []
    out: dict[str, float] = {}
    for item in rows:
        try:
            if int(item.get("logical_finger", -1)) != int(logical_finger):
                continue
            body_name = str(item.get("body_name") or "")
            candidate = str(item.get("candidate_kind") or body_name)
            sensor = item.get("sensor")
            if sensor is None:
                continue
            init_error = _init_contact_sensor_for_audit(sensor)
            key = f"{candidate}:{body_name}"
            if init_error:
                out[key] = 0.0
                continue
            forces = getattr(getattr(sensor, "data", None), "net_forces_w", None)
            row = _tensor_row(forces, slot.local_env_index)
            if hasattr(row, "detach"):
                try:
                    import torch  # noqa: WPS433

                    tensor = row.detach().to(dtype=torch.float32)
                    out[key] = float(torch.linalg.vector_norm(tensor.reshape(-1, 3), dim=-1).max().cpu().item())
                    continue
                except Exception:
                    pass
            values = []
            try:
                flat = list(row)
                for index in range(0, len(flat), 3):
                    values.append(_norm([flat[index], flat[index + 1], flat[index + 2]]))
            except Exception:
                values = []
            out[key] = max([0.0, *[_to_float(value) for value in values]])
        except Exception:
            continue
    return out


def _merge_force_peak_dict(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    out = {str(key): float(value) for key, value in left.items()}
    for key, value in right.items():
        text = str(key)
        out[text] = max(float(out.get(text, 0.0)), _to_float(value))
    return out


def _body_audit_for_finger(body_audit_rows: list[dict[str, Any]], logical_finger: int) -> dict[str, Any]:
    for row in body_audit_rows:
        if _int_field(row, "logical_finger_id", -1) == int(logical_finger):
            return dict(row)
    return {}


def _force_source_from_diagnostic_peaks(
    peaks: dict[str, float],
    contact_threshold: float,
    detection_threshold: float,
) -> tuple[str, str]:
    best_key = ""
    best_value = 0.0
    for key, value in peaks.items():
        force = _to_float(value)
        if force > best_value:
            best_key = str(key)
            best_value = force
    if best_value <= float(detection_threshold):
        return "", "no_diagnostic_body_force_observed"
    if best_value <= float(contact_threshold):
        return best_key, "diagnostic_body_unfiltered_force_observed_below_object_threshold"
    return best_key, "diagnostic_body_unfiltered_force_observed"


def _run_finger_reference_contact_canary(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    structure_rows: list[dict[str, Any]],
    body_audit_rows: list[dict[str, Any]],
    cube_canary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold = float(backend.contact_manager.force_threshold_n)
    canary_detection_threshold = max(1.0e-6, threshold * 0.002)
    cube_ok = bool(cube_canary_rows and all(_bool(row.get("cube_cube_canary_ok")) for row in cube_canary_rows))
    structure_ok = bool(
        structure_rows
        and all(
            _bool(row.get("filtered_sensor_structure_ok"))
            for row in structure_rows
            if str(row.get("sensor_kind") or "") in {"target_filtered", "reference_filtered"}
        )
    )
    slot = _slot_by_part(backend, "Screw1")
    rows: list[dict[str, Any]] = []
    bracket_offsets = (
        [0.006, 0.0, 0.0],
        [-0.006, 0.0, 0.0],
        [0.0, 0.006, 0.0],
        [0.0, -0.006, 0.0],
        [0.0, 0.0, 0.006],
        [0.0, 0.0, -0.006],
    )
    for logical_finger in (3, 4):
        finger_index = logical_finger - 1
        audit_row = _body_audit_for_finger(body_audit_rows, logical_finger)
        base_row = {
            "part_name": "Screw1",
            "logical_finger_id": logical_finger,
            "finger_index": finger_index,
            "env_index": slot.global_env_index if slot is not None else -1,
            "local_env_index": slot.local_env_index if slot is not None else -1,
            "finger_reference_canary_ok": False,
            "cube_cube_precondition_ok": cube_ok,
            "filtered_structure_precondition_ok": structure_ok,
            "reference_pad_prim_path": "",
            "reference_pad_placement_source": "contact_sensor_source_body_runtime_pose",
            "source_body_name": audit_row.get("robot_source_body_name", ""),
            "source_body_runtime_local_xyz": audit_row.get("source_body_runtime_local_xyz", []),
            "true_tip_runtime_local_xyz": audit_row.get("true_tip_runtime_local_xyz", []),
            "link4_anchor_runtime_local_xyz": audit_row.get("link4_anchor_runtime_local_xyz", []),
            "source_to_true_tip_distance_m": audit_row.get("source_to_true_tip_distance_m", ""),
            "source_to_link4_anchor_distance_m": audit_row.get("source_to_link4_anchor_distance_m", ""),
            "filtered_source_body_matches_unfiltered": bool(
                audit_row.get("filtered_source_body_matches_unfiltered", False)
            ),
            "source_body_has_effective_collider_or_reporter": bool(
                audit_row.get("source_body_has_effective_collider_or_reporter", False)
            ),
            "reference_bracket_offsets_local_xyz": [list(offset) for offset in bracket_offsets],
            "active_unfiltered_force_peak_n": 0.0,
            "finger_to_reference_filtered_force_peak_n": 0.0,
            "reference_to_finger_mirror_force_peak_n": 0.0,
            "non_active_unfiltered_force_peak_n": 0.0,
            "non_active_reference_filtered_force_peak_n": 0.0,
            "diagnostic_candidate_force_peaks_n": {},
            "responding_diagnostic_body_name": "",
            "finger_reference_force_source_classification": "not_observed",
            "reference_mirror_available": False,
            "reference_displacement_m": 0.0,
            "reference_pad_to_source_body_distance_m": "",
            "reference_pad_to_true_tip_distance_m": "",
            "calibration_force_peak_n": 0.0,
            "calibration_unsafe": False,
            "canary_detection_threshold_n": canary_detection_threshold,
            "object_contact_force_threshold_n": threshold,
            "active_unfiltered_force_observed": False,
            "finger_to_reference_filtered_force_observed": False,
            "reference_to_finger_mirror_force_observed": False,
            "canary_force_below_object_threshold": False,
            "evidence_source": "ContactSensor.force_matrix_w",
            "object_contact_success_evidence_used": False,
            "training_locked": True,
            "blocker": "",
        }
        if slot is None:
            base_row["blocker"] = "screw1_slot_missing"
            rows.append(base_row)
            continue
        if not structure_ok:
            base_row["blocker"] = "FILTERED_SENSOR_SOURCE_BODY_MISMATCH"
            rows.append(base_row)
            continue
        if not _bool(audit_row.get("body_audit_ok")):
            base_row["blocker"] = str(audit_row.get("blocker") or "V95_FINGER_CONTACT_BODY_AUDIT_FAILED")
            rows.append(base_row)
            continue
        if not cube_ok:
            base_row["blocker"] = "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED"
            rows.append(base_row)
            continue
        active_index = finger_index
        tested_offsets: list[list[float]] = []
        before_center = []
        after_center = []
        pad: dict[str, Any] = {}
        for attempt, offset_xyz in enumerate(bracket_offsets):
            tested_offsets.append([float(value) for value in offset_xyz])
            _reset_with_plan(backend, plan_rows, settle_steps=2)
            snapshot = _v95_finger_body_snapshot(backend, slot, logical_finger)
            source_local = _vec3(snapshot.get("source_body_runtime_local_xyz", []))
            true_tip_local = _vec3(snapshot.get("true_tip_runtime_local_xyz", []))
            if _int_field(snapshot, "source_body_index", -1) < 0:
                base_row["blocker"] = "finger_reference_source_body_pose_unavailable"
                break
            center = _vec_add(source_local, offset_xyz)
            pad = _move_v95_reference_pad_runtime(backend, slot, logical_finger, center)
            base_row.update(
                {
                    "reference_pad_prim_path": pad.get("reference_pad_prim_path", ""),
                    "reference_pad_center_local_xyz": pad.get("reference_pad_center_local_xyz", []),
                    "reference_pad_rigid_object": bool(pad.get("reference_pad_rigid_object", False)),
                    "reference_pad_collision_enabled": bool(pad.get("reference_pad_collision_enabled", False)),
                    "reference_pad_contact_report_api": bool(pad.get("reference_pad_contact_report_api", False)),
                    "reference_attempt_index": attempt,
                    "reference_attempt_offset_local_xyz": [float(value) for value in offset_xyz],
                    "source_body_runtime_local_xyz": source_local,
                    "true_tip_runtime_local_xyz": true_tip_local,
                    "reference_pad_to_source_body_distance_m": _norm(_vec_sub(center, source_local)),
                    "reference_pad_to_true_tip_distance_m": _norm(_vec_sub(center, true_tip_local))
                    if _int_field(snapshot, "true_tip_body_index", -1) >= 0
                    else "",
                }
            )
            if pad.get("reference_pad_blocker"):
                base_row["blocker"] = str(pad.get("reference_pad_blocker"))
                break
            before_center = list(pad.get("reference_pad_center_local_xyz", center))
            for step in range(8):
                _set_context(backend, "v95_finger_reference_contact_canary", finger=logical_finger, step=step)
                backend.step_envs([[0.0] * 16 for _slot in backend.slots])
                metrics = backend._metrics_for_slot(slot, f"v95_finger_reference_canary_f{logical_finger}_{attempt}_{step}")
                unfiltered = _v92_force_list(metrics)
                reference = _v95_reference_force_list(backend, slot)
                mirror = _v95_reference_mirror_force_list(backend, slot)
                diagnostic_peaks = _v95_diagnostic_body_force_peaks(backend, slot, logical_finger)
                base_row["diagnostic_candidate_force_peaks_n"] = _merge_force_peak_dict(
                    dict(base_row.get("diagnostic_candidate_force_peaks_n") or {}),
                    diagnostic_peaks,
                )
                base_row["active_unfiltered_force_peak_n"] = max(
                    _to_float(base_row.get("active_unfiltered_force_peak_n")),
                    _force_peak_for_indices(unfiltered, [active_index]),
                )
                base_row["finger_to_reference_filtered_force_peak_n"] = max(
                    _to_float(base_row.get("finger_to_reference_filtered_force_peak_n")),
                    _force_peak_for_indices(reference, [active_index]),
                )
                base_row["reference_to_finger_mirror_force_peak_n"] = max(
                    _to_float(base_row.get("reference_to_finger_mirror_force_peak_n")),
                    _force_peak_for_indices(mirror, [active_index]),
                )
                base_row["non_active_unfiltered_force_peak_n"] = max(
                    _to_float(base_row.get("non_active_unfiltered_force_peak_n")),
                    _force_peak_except_indices(unfiltered, [active_index]),
                )
                base_row["non_active_reference_filtered_force_peak_n"] = max(
                    _to_float(base_row.get("non_active_reference_filtered_force_peak_n")),
                    _force_peak_except_indices(reference, [active_index]),
                )
                base_row["reference_mirror_available"] = (
                    _tensor_float(getattr(_base_env(backend), "v95_reference_mirror_force_valid", None), slot.local_env_index) > 0.5
                )
                force_peak = max(
                    _to_float(base_row.get("active_unfiltered_force_peak_n")),
                    _to_float(base_row.get("finger_to_reference_filtered_force_peak_n")),
                    _to_float(base_row.get("reference_to_finger_mirror_force_peak_n")),
                )
                base_row["calibration_force_peak_n"] = force_peak
                if force_peak > 5.0:
                    base_row["calibration_unsafe"] = True
                    base_row["blocker"] = "FINGER_REFERENCE_CALIBRATION_UNSAFE"
                    break
            pad_pose = _v95_reference_pad_pose(backend, slot, logical_finger)
            after_center = list(pad_pose.get("reference_pad_runtime_center_local_xyz") or before_center)
            displacement = _norm(_vec_sub(after_center, before_center)) if before_center and after_center else 0.0
            base_row["reference_displacement_m"] = max(_to_float(base_row.get("reference_displacement_m")), displacement)
            if displacement > 0.02:
                base_row["calibration_unsafe"] = True
                base_row["blocker"] = "FINGER_REFERENCE_CALIBRATION_UNSAFE"
            if base_row["blocker"] or (
                threshold
                < _to_float(base_row.get("finger_to_reference_filtered_force_peak_n"))
                <= 5.0
            ):
                break
        responding_body, force_source = _force_source_from_diagnostic_peaks(
            dict(base_row.get("diagnostic_candidate_force_peaks_n") or {}),
            threshold,
            canary_detection_threshold,
        )
        base_row["responding_diagnostic_body_name"] = responding_body
        base_row["finger_reference_force_source_classification"] = force_source
        base_row["reference_attempt_offsets_local_xyz"] = tested_offsets
        active_unfiltered_peak = _to_float(base_row.get("active_unfiltered_force_peak_n"))
        reference_filtered_peak = _to_float(base_row.get("finger_to_reference_filtered_force_peak_n"))
        mirror_peak = _to_float(base_row.get("reference_to_finger_mirror_force_peak_n"))
        base_row["active_unfiltered_force_observed"] = bool(active_unfiltered_peak > canary_detection_threshold)
        base_row["finger_to_reference_filtered_force_observed"] = bool(
            reference_filtered_peak > canary_detection_threshold
        )
        base_row["reference_to_finger_mirror_force_observed"] = bool(mirror_peak > canary_detection_threshold)
        base_row["canary_force_below_object_threshold"] = bool(
            active_unfiltered_peak > canary_detection_threshold
            and reference_filtered_peak > canary_detection_threshold
            and (active_unfiltered_peak <= threshold or reference_filtered_peak <= threshold)
        )
        if not base_row["blocker"]:
            if active_unfiltered_peak <= canary_detection_threshold:
                base_row["blocker"] = "FINGER_REFERENCE_NO_UNFILTERED_CONTACT"
                if not responding_body:
                    base_row["finger_reference_force_source_classification"] = (
                        "reference_pad_did_not_touch_any_diagnostic_finger_body"
                    )
            elif reference_filtered_peak <= canary_detection_threshold:
                base_row["blocker"] = "FINGER_REFERENCE_FILTERED_FORCE_MISSING"
                if responding_body:
                    base_row["finger_reference_force_source_classification"] = (
                        "unfiltered_force_on_diagnostic_body_without_reference_filter"
                    )
            elif (
                _bool(base_row.get("reference_mirror_available"))
                and mirror_peak <= canary_detection_threshold
            ):
                base_row["blocker"] = "FINGER_REFERENCE_MIRROR_FORCE_MISSING"
                base_row["finger_reference_force_source_classification"] = (
                    "finger_reference_filtered_force_without_reference_side_mirror"
                )
            else:
                base_row["finger_reference_canary_ok"] = True
                if bool(base_row.get("canary_force_below_object_threshold")):
                    base_row["finger_reference_force_source_classification"] = (
                        "reference_filtered_contact_observed_below_object_threshold"
                    )
                else:
                    base_row["finger_reference_force_source_classification"] = "reference_filtered_contact_observed"
                base_row["blocker"] = ""
        rows.append(base_row)
        _move_v95_reference_pad_runtime(
            backend,
            slot,
            logical_finger,
            [1.45 + 0.03 * float(max(0, int(logical_finger) - 1)), 1.45, 1.25],
        )
    return rows


def _v95_canary_decision(
    structure_rows: list[dict[str, Any]],
    cube_canary_rows: list[dict[str, Any]],
    finger_reference_rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    if not structure_rows or not all(
        _bool(row.get("filtered_sensor_structure_ok"))
        for row in structure_rows
        if str(row.get("sensor_kind") or "") in {"target_filtered", "reference_filtered"}
    ):
        return False, "FILTERED_SENSOR_SOURCE_BODY_MISMATCH"
    if not cube_canary_rows or not all(_bool(row.get("cube_cube_canary_ok")) for row in cube_canary_rows):
        return False, "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED"
    if not finger_reference_rows or not all(_bool(row.get("finger_reference_canary_ok")) for row in finger_reference_rows):
        blocker = _first_blocker(finger_reference_rows, "blocker", "FINGER_REFERENCE_FILTERED_FORCE_MISSING")
        return False, blocker
    return True, ""


def _canary_blocked_contact_rows(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    blocker: str,
    canary_ok: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for mode in ("single_finger", "two_multi_finger"):
        for slot in backend.slots:
            group = _contact_mode_group(plan_rows, slot, mode)
            row = {
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "contact_gate_mode": mode,
                "active_finger_group": group,
                "actual_object_contact_ok": False,
                "actual_object_contact_probe_executed": False,
                "contact_acquisition_status": "NOT_EXECUTED_CANARY_ONLY"
                if canary_ok
                else "NOT_EXECUTED_CANARY_BLOCKED",
                "canary_ok": bool(canary_ok),
                "target_contact_evidence_source": "ContactSensor.force_matrix_w",
                "object_contact_success_evidence_used": False,
                "distance_only_success_used": False,
                "non_active_sensor_success_used": False,
                "proxy_success_used": False,
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "live_probe_executed": True,
                "force_peak_n": 0.0,
                "active_target_filtered_force_peak_n": 0.0,
                "target_object_contact_force_peak_n": 0.0,
                "target_filtered_force_available": False,
                "training_locked": True,
                "blocker": blocker,
            }
            rows.append(row)
            trace_rows.append(
                {
                    "phase": row["contact_acquisition_status"],
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "contact_gate_mode": mode,
                    "active_finger_group": group,
                    "canary_ok": bool(canary_ok),
                    "blocker": blocker,
                    "object_contact_success_evidence_used": False,
                    "distance_only_success_used": False,
                    "non_active_sensor_success_used": False,
                }
            )
    return rows, trace_rows


def _forced_contact_blocked_contact_rows(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    blocker: str,
    forced_contact_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    forced_peak = max([0.0, *[_to_float(row.get("active_target_filtered_force_peak_n", row.get("finger_side_target_filtered_force_peak_n"))) for row in forced_contact_rows]])
    forced_unfiltered = max([0.0, *[_to_float(row.get("unfiltered_active_force_peak_n")) for row in forced_contact_rows]])
    forced_mirror = max([0.0, *[_to_float(row.get("object_side_mirror_force_peak_n")) for row in forced_contact_rows]])
    forced_ok = _screw1_forced_contact_ok(forced_contact_rows)
    for mode in ("single_finger", "two_multi_finger"):
        for slot in backend.slots:
            group = _contact_mode_group(plan_rows, slot, mode)
            is_screw1 = slot.part_name == "Screw1"
            status = (
                "NOT_EXECUTED_ORDINARY_ACQUISITION_PAUSED_AFTER_FORCED_OK"
                if is_screw1 and forced_ok
                else "NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED"
                if is_screw1 and blocker == "NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED"
                else "NOT_EXECUTED_FORCED_TARGET_CONTACT_BLOCKED"
                if is_screw1
                else "NOT_EXECUTED_SCREW1_ONLY_REGRESSION_GUARD"
            )
            row_blocker = (
                "ORDINARY_ACQUISITION_PAUSED_AFTER_FORCED_CALIBRATION_OK"
                if is_screw1 and forced_ok
                else blocker
                if is_screw1
                else "NOT_EXECUTED_SCREW1_ONLY_REGRESSION_GUARD"
            )
            row = {
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "contact_gate_mode": mode,
                "active_finger_group": group,
                "actual_object_contact_ok": False,
                "actual_object_contact_probe_executed": False,
                "contact_acquisition_status": status,
                "canary_ok": True,
                "forced_contact_status": blocker,
                "ordinary_acquisition_executed_after_forced_ok": False,
                "ordinary_acquisition_next_allowed": bool(forced_ok and is_screw1),
                "target_contact_evidence_source": "ContactSensor.force_matrix_w",
                "object_contact_success_evidence_used": False,
                "distance_only_success_used": False,
                "non_active_sensor_success_used": False,
                "proxy_success_used": False,
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "live_probe_executed": True,
                "force_peak_n": 0.0,
                "active_target_filtered_force_peak_n": forced_peak if is_screw1 else 0.0,
                "unfiltered_active_force_peak_n": forced_unfiltered if is_screw1 else 0.0,
                "object_side_mirror_force_peak_n": forced_mirror if is_screw1 else 0.0,
                "target_object_contact_force_peak_n": 0.0,
                "target_filtered_force_available": False,
                "training_locked": True,
                "blocker": row_blocker,
            }
            rows.append(row)
            trace_rows.append(
                {
                    "phase": status,
                    "part_name": slot.part_name,
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "contact_gate_mode": mode,
                    "active_finger_group": group,
                    "canary_ok": True,
                    "forced_contact_status": blocker,
                    "ordinary_acquisition_executed_after_forced_ok": False,
                    "ordinary_acquisition_next_allowed": bool(forced_ok and is_screw1),
                    "object_contact_success_evidence_used": False,
                    "distance_only_success_used": False,
                    "non_active_sensor_success_used": False,
                    "proxy_success_used": False,
                    "blocker": row_blocker,
                }
            )
    return rows, trace_rows


def _run_reference_filtered_sensor_calibration(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    finger_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "reference_filtered_calibration_ok": False, "blocker": "screw1_slot_missing"}]
    sensor_map = _finger_sensor_maps(finger_rows).get("Screw1", {})
    rows: list[dict[str, Any]] = []
    for logical_finger in (3, 4):
        finger_index = logical_finger - 1
        _park_v95_reference_pad(slot, logical_finger)
        _reset_with_plan(backend, plan_rows, settle_steps=2)
        tip_start = _active_tip_local_xyz(backend, slot, finger_index)
        for step in range(6):
            actions = [[0.0] * 16 for _slot in backend.slots]
            _v91_apply_finger_mask(
                actions[slot.local_env_index],
                group=str(logical_finger),
                active_value=min(0.40, 0.07 * float(step + 1)),
                support_value=0.0,
            )
            _set_context(backend, "v95_reference_filtered_sensor_path_probe", finger=logical_finger, step=step)
            backend.step_envs(actions)
        tip_after = _active_tip_local_xyz(backend, slot, finger_index)
        close_delta = _vec_sub(tip_after, tip_start) if tip_start and tip_after else [0.0, 0.0, -1.0]
        close_delta_norm = _norm(close_delta)
        close_dir = _unit(close_delta) if close_delta_norm > 1.0e-5 else [0.0, 0.0, -1.0]
        _reset_with_plan(backend, plan_rows, settle_steps=2)
        tip_reset = _active_tip_local_xyz(backend, slot, finger_index) or tip_start
        pad_half_extent = 0.003
        initial_gap = 0.0005
        center_distance = pad_half_extent + initial_gap + min(0.0015, max(0.0005, close_delta_norm * 0.25))
        pad_center = _vec_add(tip_reset, [close_dir[index] * center_distance for index in range(3)]) if tip_reset else []
        pad = _move_v95_reference_pad(slot, logical_finger, pad_center)
        _reset_with_plan(backend, plan_rows, settle_steps=1)
        active_sensor = sensor_map.get(finger_index, finger_index)
        active_unfiltered_peak = 0.0
        active_reference_peak = 0.0
        non_active_unfiltered_peak = 0.0
        non_active_reference_peak = 0.0
        max_force_peak = 0.0
        native_pair_observed = False
        unsafe = False
        blocker = str(pad.get("reference_pad_blocker") or "")
        for step in range(8):
            actions = [[0.0] * 16 for _slot in backend.slots]
            if not blocker:
                _v91_apply_finger_mask(
                    actions[slot.local_env_index],
                    group=str(logical_finger),
                    active_value=min(0.45, 0.08 * float(step + 1)),
                    support_value=0.0,
                )
            _set_context(backend, "v95_reference_filtered_sensor_calibration", finger=logical_finger, step=step)
            backend.step_envs(actions)
            metrics = backend._metrics_for_slot(slot, f"v95_reference_filtered_f{logical_finger}_{step}")
            unfiltered = _v92_force_list(metrics)
            reference = _v95_reference_force_list(backend, slot)
            active_unfiltered_peak = max(active_unfiltered_peak, _force_peak_for_indices(unfiltered, [active_sensor]))
            active_reference_peak = max(active_reference_peak, _force_peak_for_indices(reference, [active_sensor]))
            non_active_unfiltered_peak = max(non_active_unfiltered_peak, _force_peak_except_indices(unfiltered, [active_sensor]))
            non_active_reference_peak = max(non_active_reference_peak, _force_peak_except_indices(reference, [active_sensor]))
            max_force_peak = max(max_force_peak, max([0.0, *unfiltered, *reference]))
            native_pair_observed = bool(native_pair_observed or active_reference_peak > backend.contact_manager.force_threshold_n)
            if max_force_peak > 5.0:
                unsafe = True
                blocker = "reference_filtered_calibration_unsafe_force_peak"
                break
        reference_valid = _tensor_float(getattr(_base_env(backend), "dex_reference_force_valid", None), slot.local_env_index) > 0.5
        if not blocker and not reference_valid:
            blocker = "reference_filtered_force_matrix_unavailable"
        elif not blocker and active_unfiltered_peak > backend.contact_manager.force_threshold_n and active_reference_peak <= backend.contact_manager.force_threshold_n:
            blocker = "unfiltered_force_without_reference_filtered_force"
        elif not blocker and active_reference_peak <= backend.contact_manager.force_threshold_n:
            blocker = "reference_filtered_force_not_observed"
        rows.append(
            {
                "part_name": "Screw1",
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "logical_finger_id": logical_finger,
                "expected_sensor_index": active_sensor,
                "reference_pad_prim_path": pad.get("reference_pad_prim_path", ""),
                "reference_pad_center_local_xyz": pad.get("reference_pad_center_local_xyz", []),
                "reference_pad_collision_enabled": bool(pad.get("reference_pad_collision_enabled")),
                "reference_pad_contact_report_api": bool(pad.get("reference_pad_contact_report_api")),
                "reference_pad_half_extent_m": pad_half_extent,
                "reference_pad_initial_gap_m": initial_gap,
                "measured_close_path_tip_start_local_xyz": tip_start,
                "measured_close_path_tip_after_local_xyz": tip_after,
                "measured_close_path_direction_local_xyz": close_dir,
                "measured_close_path_delta_norm_m": close_delta_norm,
                "reference_filtered_force_available": reference_valid,
                "reference_filter_count": _tensor_float(getattr(_base_env(backend), "dex_reference_force_filter_count", None), slot.local_env_index),
                "unfiltered_active_force_peak_n": active_unfiltered_peak,
                "reference_filtered_active_force_peak_n": active_reference_peak,
                "non_active_unfiltered_force_peak_n": non_active_unfiltered_peak,
                "non_active_reference_filtered_force_peak_n": non_active_reference_peak,
                "native_pair_contact_observed": native_pair_observed,
                "calibration_force_peak_n": max_force_peak,
                "calibration_unsafe": unsafe,
                "object_contact_success_evidence_used": False,
                "distance_only_success_used": False,
                "reference_filtered_calibration_ok": bool(
                    reference_valid
                    and active_reference_peak > backend.contact_manager.force_threshold_n
                    and not unsafe
                ),
                "blocker": blocker,
                "training_locked": True,
            }
        )
        _park_v95_reference_pad(slot, logical_finger)
    return rows


def _screw1_forced_contact_targets_from_reference_canary(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    finger_reference_canary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    z_offsets = _screw1_collision_root_z_offsets(backend, slot)
    bottom_offset = max(0.005, _to_float(z_offsets.get("bottom_offset_m"), 0.025))
    support_half_z = 0.003
    support_penetration = 0.0005
    targets: list[dict[str, Any]] = []
    for canary in finger_reference_canary_rows:
        if str(canary.get("part_name") or "") != "Screw1" or not _bool(canary.get("finger_reference_canary_ok")):
            continue
        logical_finger = _int_field(canary, "logical_finger_id", -1)
        if logical_finger not in (3, 4):
            continue
        source_local = _vec3(canary.get("source_body_runtime_local_xyz", []))
        base = _vec3(canary.get("reference_pad_center_local_xyz", []))
        if len(source_local) < 3 or len(base) < 3:
            continue
        normal = _normalize(_vec_sub(base, source_local))
        tangent_x = _orthogonal_unit(normal)
        tangent_y = _normalize(_cross(normal, tangent_x))
        bracket_offsets = [
            [0.0, 0.0, 0.0],
            [0.006 * value for value in normal],
            [-0.006 * value for value in normal],
            [0.003 * value for value in tangent_x],
            [-0.003 * value for value in tangent_x],
            [0.003 * value for value in tangent_y],
            [-0.003 * value for value in tangent_y],
        ]
        for bracket_index, offset in enumerate(bracket_offsets):
            center = _vec_add(base, offset)
            support_center = [
                center[0],
                center[1],
                center[2] - bottom_offset - support_half_z + support_penetration,
            ]
            targets.append(
                {
                    "forced_pose_source_finger_id": logical_finger,
                    "forced_pose_source_body": canary.get("source_body_name", ""),
                    "forced_pose_source_body_local_xyz": source_local,
                    "forced_reference_pad_center_local_xyz": base,
                    "forced_reference_attempt_offset_local_xyz": canary.get(
                        "reference_attempt_offset_local_xyz", []
                    ),
                    "forced_reference_contact_normal_local_xyz": normal,
                    "forced_reference_tangent_x_local_xyz": tangent_x,
                    "forced_reference_tangent_y_local_xyz": tangent_y,
                    "forced_bracket_index": bracket_index,
                    "forced_bracket_offset_local_xyz": offset,
                    "forced_object_center_local_xyz": center,
                    "forced_pose_to_reference_pad_distance_m": _norm(_vec_sub(center, base)),
                    "forced_pose_to_source_body_distance_m": _norm(_vec_sub(center, source_local)),
                    "screw1_support_pad_center_local_xyz": support_center,
                    "screw1_support_pad_half_extent_xyz": [0.03, 0.03, support_half_z],
                    "screw1_support_pad_penetration_m": support_penetration,
                    "screw1_support_bottom_offset_m": bottom_offset,
                    "screw1_support_offset_source": z_offsets.get("offset_source", ""),
                    "screw1_support_offset_blocker": z_offsets.get("offset_blocker", ""),
                }
            )
    return targets


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def _move_v95_screw1_support_pad_runtime(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    center_local_xyz: list[float],
) -> dict[str, Any]:
    base = _base_env(backend)
    if base is not None and hasattr(base, "v95_move_screw1_support_pad"):
        return dict(base.v95_move_screw1_support_pad(slot.local_env_index, center_local_xyz))
    return {
        "screw1_support_pad_center_local_xyz": list(center_local_xyz[:3]),
        "screw1_support_pad_blocker": "v95_move_screw1_support_pad_unavailable",
        "screw1_support_pad_rigid_object": False,
    }


def _park_v95_screw1_support_pad(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> None:
    _move_v95_screw1_support_pad_runtime(backend, slot, [1.65, 1.45, 1.25])


def _forced_screw1_plan(
    plan_rows: list[dict[str, Any]],
    *,
    forced_center: list[float],
    hand_target: list[float] | None,
) -> list[dict[str, Any]]:
    forced_plan = [copy.deepcopy(row) for row in plan_rows]
    for row in forced_plan:
        if str(row.get("part_name") or "") != "Screw1":
            continue
        row["object_center_local_xyz"] = list(forced_center)
        row["object_center_local_x"] = forced_center[0]
        row["object_center_local_y"] = forced_center[1]
        row["object_center_local_z"] = forced_center[2]
        row["initial_condition_mode"] = "v95_forced_target_contact_diagnostic"
        row["hand_target_source"] = (
            "v95_forced_object_only_safe_retreat"
            if hand_target is not None
            else "v95_forced_contact_reuses_reference_canary_hand"
        )
        row["forced_contact_diagnostic_only"] = True
        if hand_target is not None:
            row["hand_target_local_xyz"] = list(hand_target)
            row["hand_target_local_x"] = hand_target[0]
            row["hand_target_local_y"] = hand_target[1]
            row["hand_target_local_z"] = hand_target[2]
    return forced_plan


def _reset_forced_screw1_with_support(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    support_center_local_xyz: list[float],
) -> dict[str, Any]:
    _assert_v95_plan_valid(backend, plan_rows, phase="forced_screw1_reset")
    _disable_v95_action_control(backend)
    backend.configure_v95_pregrasp(plan_rows)
    backend.object_write_by_policy_detected = False
    backend.reset_envs()
    backend._capture_v86_reset_object_positions()
    base = _base_env(backend)
    reset_rows = [dict(row) for row in getattr(base, "v95_last_pregrasp_rows", []) or []]
    hand_rows = [dict(row) for row in getattr(base, "v95_last_hand_reset_rows", []) or []]
    _assert_v95_reset_applied(backend, plan_rows, reset_rows, hand_rows, phase="forced_screw1_reset")
    slot = _slot_by_part(backend, "Screw1")
    support = (
        _move_v95_screw1_support_pad_runtime(backend, slot, support_center_local_xyz)
        if slot is not None
        else {"screw1_support_pad_blocker": "screw1_slot_missing"}
    )
    _enable_v95_action_control(backend)
    return support


def _screw1_root_local_xyz(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> list[float]:
    state = _v91_object_state_for_slot(backend, slot)
    pos_w = _vec3(state.get("root_pos_w", []))
    base = _base_env(backend)
    origin = _tensor_list(getattr(getattr(base, "scene", None), "env_origins", None), slot.local_env_index, width=3)
    if len(pos_w) < 3 or len(origin) < 3:
        return []
    return _vec_sub(pos_w, origin)


def _screw1_collision_root_z_offsets(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
) -> dict[str, Any]:
    geometry = V85_PART_GEOMETRY.get("Screw1", {})
    center_offset = list(geometry.get("center", (0.0, 0.0, 0.0125)))
    half_height = _to_float(geometry.get("half_height"), 0.0125)
    fallback_top = _to_float(center_offset[2] if len(center_offset) >= 3 else 0.0125) + half_height
    out: dict[str, Any] = {
        "center_offset_x_m": 0.0,
        "center_offset_y_m": 0.0,
        "top_offset_m": fallback_top,
        "bottom_offset_m": fallback_top,
        "extent_z_m": 2.0 * half_height,
        "offset_source": "V85_PART_GEOMETRY_fallback",
        "offset_blocker": "",
    }
    try:
        import omni.usd  # noqa: WPS433
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        root_path = f"/World/envs/env_{slot.local_env_index}/Screw1"
        root = stage.GetPrimAtPath(root_path) if stage is not None else None
        if root is None or not root.IsValid():
            out["offset_blocker"] = "screw1_stage_root_unavailable"
            return out
        collision_prims = []
        for prim in Usd.PrimRange(root):
            collision_api = UsdPhysics.CollisionAPI(prim)
            if collision_api and bool(collision_api.GetCollisionEnabledAttr().Get()):
                collision_prims.append(prim)
        if not collision_prims:
            out["offset_blocker"] = "screw1_collision_prims_unavailable"
            return out
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = _bbox_for_prims(cache, collision_prims)
        center = _vec3(box.get("center", []))
        extent = _vec3(box.get("extent", []))
        if len(center) < 3 or len(extent) < 3 or extent[2] <= 1.0e-5:
            out["offset_blocker"] = "screw1_collision_bbox_unavailable"
            return out
        root_xform = UsdGeom.Xformable(root)
        root_translation = root_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        root_z = float(root_translation[2])
        root_x = float(root_translation[0])
        root_y = float(root_translation[1])
        bbox_min_z = float(center[2]) - 0.5 * float(extent[2])
        bbox_max_z = float(center[2]) + 0.5 * float(extent[2])
        top_offset = bbox_max_z - root_z
        bottom_offset = root_z - bbox_min_z
        if not math.isfinite(top_offset) or not math.isfinite(bottom_offset):
            out["offset_blocker"] = "screw1_collision_bbox_offset_nonfinite"
            return out
        out.update(
            {
                "center_offset_x_m": max(-0.05, min(0.05, float(center[0]) - root_x)),
                "center_offset_y_m": max(-0.05, min(0.05, float(center[1]) - root_y)),
                "top_offset_m": max(-0.05, min(0.10, float(top_offset))),
                "bottom_offset_m": max(-0.05, min(0.10, float(bottom_offset))),
                "extent_z_m": float(extent[2]),
                "offset_source": "env_specific_usd_collision_bbox_root_relative",
                "offset_blocker": "",
            }
        )
    except Exception as exc:
        out["offset_blocker"] = f"{type(exc).__name__}:{exc}"
    return out


def _run_screw1_reference_contact_canary(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    canary_ok: bool,
    canary_blocker: str,
    cube_canary_rows: list[dict[str, Any]],
    finger_reference_canary_rows: list[dict[str, Any]],
    filter_coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    threshold = float(backend.contact_manager.force_threshold_n)
    detection_threshold = max(1.0e-6, threshold * 0.002)
    reference_name = "V95CanaryCubeA"
    parked_name = "V95CanaryCubeB"
    row: dict[str, Any] = {
        "part_name": "Screw1",
        "env_index": slot.global_env_index if slot is not None else -1,
        "local_env_index": slot.local_env_index if slot is not None else -1,
        "screw1_reference_contact_canary_ok": False,
        "screw1_reference_contact_status": "NOT_EXECUTED",
        "reference_body": reference_name,
        "reference_prim_path": "",
        "screw1_prim_path": "",
        "hand_safe_retreat_used": True,
        "inactive_object_parked": True,
        "object_contact_success_evidence_used": False,
        "actual_object_contact_ok": False,
        "distance_only_success_used": False,
        "proxy_success_used": False,
        "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
        "training_locked": True,
        "force_detection_threshold_n": detection_threshold,
        "force_contact_threshold_n": threshold,
        "calibration_force_unsafe_threshold_n": 5.0,
        "displacement_unsafe_threshold_m": 0.02,
        "screw1_unfiltered_force_peak_n": 0.0,
        "reference_unfiltered_force_peak_n": 0.0,
        "reference_to_screw1_filtered_force_peak_n": 0.0,
        "screw1_to_reference_mirror_force_peak_n": 0.0,
        "screw1_displacement_m": 0.0,
        "screw1_z_drift_m": 0.0,
        "reference_displacement_m": 0.0,
        "calibration_unsafe": False,
        "filter_path": "/World/envs/env_.*/Screw1",
        "filter_coverage_ok": any(_bool(item.get("filter_coverage_ok")) for item in filter_coverage_rows),
        "filter_coverage_blocker": _first_blocker(filter_coverage_rows, "blocker", ""),
        "reference_filter_count": 0.0,
        "mirror_filter_count": 0.0,
        "reference_body_count": 0.0,
        "mirror_body_count": 0.0,
        "target_contact_evidence_source": "ContactSensor.force_matrix_w",
        "blocker": "",
        "verdict": "NOT_EXECUTED",
    }
    if slot is None:
        row.update(
            {
                "screw1_reference_contact_status": "SCREW1_SLOT_MISSING",
                "verdict": "SCREW1_SLOT_MISSING",
                "blocker": "screw1_slot_missing",
            }
        )
        return [row]
    row["screw1_prim_path"] = f"/World/envs/env_{slot.local_env_index}/Screw1"
    row["reference_prim_path"] = f"/World/envs/env_{slot.local_env_index}/{reference_name}"
    if not canary_ok:
        row.update(
            {
                "screw1_reference_contact_status": "NOT_EXECUTED_CANARY_BLOCKED",
                "verdict": "NOT_EXECUTED_CANARY_BLOCKED",
                "blocker": canary_blocker or "CANARY_FAILED_BEFORE_SCREW1_REFERENCE_CONTACT",
            }
        )
        return [row]
    cube_ok = bool(cube_canary_rows and all(_bool(item.get("cube_cube_canary_ok")) for item in cube_canary_rows))
    finger_reference_ok = bool(
        finger_reference_canary_rows
        and all(_bool(item.get("finger_reference_canary_ok")) for item in finger_reference_canary_rows)
    )
    if not cube_ok:
        row.update(
            {
                "screw1_reference_contact_status": "NOT_EXECUTED_CANARY_BLOCKED",
                "verdict": "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED",
                "blocker": "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED",
            }
        )
        return [row]
    if not finger_reference_ok:
        blocker = _first_blocker(finger_reference_canary_rows, "blocker", "FINGER_REFERENCE_CANARY_FAILED")
        row.update(
            {
                "screw1_reference_contact_status": "NOT_EXECUTED_FINGER_REFERENCE_BLOCKED",
                "verdict": blocker,
                "blocker": blocker,
            }
        )
        return [row]

    plan_row = _find_slot_row(plan_rows, slot)
    object_center = _vec3(plan_row.get("object_center_local_xyz", []))
    if len(object_center) < 3:
        object_center = [
            _to_float(plan_row.get("object_center_local_x")),
            _to_float(plan_row.get("object_center_local_y")),
            _to_float(plan_row.get("object_center_local_z")),
        ]
    if len(object_center) < 3:
        _reset_with_plan(backend, plan_rows, settle_steps=2)
        object_center = _screw1_root_local_xyz(backend, slot)
    if len(object_center) < 3:
        row.update(
            {
                "screw1_reference_contact_status": "SCREW1_REFERENCE_PLACEMENT_UNAVAILABLE",
                "verdict": "SCREW1_REFERENCE_PLACEMENT_UNAVAILABLE",
                "blocker": "screw1_plan_or_runtime_root_unavailable",
            }
        )
        return [row]

    z_offsets = _screw1_collision_root_z_offsets(backend, slot)
    top_offset = max(0.005, _to_float(z_offsets.get("top_offset_m"), 0.025))
    bottom_offset = max(0.005, _to_float(z_offsets.get("bottom_offset_m"), 0.025))
    support_half_z = 0.003
    support_penetration = 0.0005
    cube_half = 0.5 * 0.04
    reference_press_m = 0.002
    center_offset_x = _to_float(z_offsets.get("center_offset_x_m"), 0.0)
    center_offset_y = _to_float(z_offsets.get("center_offset_y_m"), 0.0)
    reference_center = [
        object_center[0] + center_offset_x,
        object_center[1] + center_offset_y,
        object_center[2] + top_offset + cube_half - reference_press_m,
    ]
    support_center = [
        object_center[0] + center_offset_x,
        object_center[1] + center_offset_y,
        object_center[2] - bottom_offset - support_half_z + support_penetration,
    ]
    row.update(
        {
            "screw1_planned_root_local_xyz": object_center,
            "reference_cube_planned_center_local_xyz": reference_center,
            "reference_cube_half_extent_m": cube_half,
            "reference_press_m": reference_press_m,
            "placement_gap_advisory_m": reference_center[2] - (object_center[2] + top_offset + cube_half),
            "placement_contact_advisory_ok": abs(
                reference_center[2] - (object_center[2] + top_offset + cube_half)
            )
            <= 0.004,
            "screw1_support_pad_center_local_xyz": support_center,
            "screw1_support_bottom_offset_m": bottom_offset,
            "screw1_support_offset_source": z_offsets.get("offset_source", ""),
            "screw1_support_offset_blocker": z_offsets.get("offset_blocker", ""),
        }
    )

    safe_hand_target = [0.34, 0.28, 1.18]
    object_plan = _forced_screw1_plan(plan_rows, forced_center=object_center, hand_target=safe_hand_target)
    support_info = _reset_forced_screw1_with_support(
        backend,
        object_plan,
        support_center_local_xyz=support_center,
    )
    row.update(
        {
            "screw1_support_pad_blocker": support_info.get("screw1_support_pad_blocker", ""),
            "screw1_support_pad_rigid_object": bool(support_info.get("screw1_support_pad_rigid_object", False)),
        }
    )
    reference_info = _move_v95_canary_cube_runtime(backend, slot, reference_name, reference_center)
    parked_info = _move_v95_canary_cube_runtime(backend, slot, parked_name, [1.35, -1.35, 1.25])
    row.update(
        {
            "reference_cube_move_blocker": reference_info.get("canary_cube_blocker", ""),
            "parked_cube_move_blocker": parked_info.get("canary_cube_blocker", ""),
            "reference_body_rigid_object": bool(reference_info.get("canary_cube_rigid_object", False)),
        }
    )
    if row.get("screw1_support_pad_blocker") or row.get("reference_cube_move_blocker") or row.get("parked_cube_move_blocker"):
        blocker = (
            str(row.get("screw1_support_pad_blocker") or "")
            or str(row.get("reference_cube_move_blocker") or "")
            or str(row.get("parked_cube_move_blocker") or "")
        )
        row.update(
            {
                "screw1_reference_contact_status": "SCREW1_REFERENCE_CALIBRATION_BODY_UNAVAILABLE",
                "verdict": "SCREW1_REFERENCE_CALIBRATION_BODY_UNAVAILABLE",
                "blocker": blocker,
            }
        )
        _park_v95_screw1_support_pad(backend, slot)
        _move_v95_canary_cube_runtime(backend, slot, reference_name, [1.25, -1.25, 1.20])
        _move_v95_canary_cube_runtime(backend, slot, parked_name, [1.30, -1.30, 1.25])
        return [row]

    before_state = _v91_object_state_for_slot(backend, slot)
    before_reference_pose = _v95_canary_cube_pose(backend, slot, reference_name)
    before_reference = _vec3(before_reference_pose.get("canary_cube_runtime_center_local_xyz", []))
    max_force = 0.0
    for step in range(32):
        _set_context(backend, "v95_screw1_reference_contact_canary", step=step)
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])
        forces = _v95_screw1_reference_force_values(backend, slot)
        row["screw1_unfiltered_force_peak_n"] = max(
            _to_float(row.get("screw1_unfiltered_force_peak_n")),
            _to_float(forces.get("screw1_unfiltered_force_n")),
        )
        row["reference_unfiltered_force_peak_n"] = max(
            _to_float(row.get("reference_unfiltered_force_peak_n")),
            _to_float(forces.get("reference_unfiltered_force_n")),
        )
        row["reference_to_screw1_filtered_force_peak_n"] = max(
            _to_float(row.get("reference_to_screw1_filtered_force_peak_n")),
            _to_float(forces.get("reference_to_screw1_filtered_force_n")),
        )
        row["screw1_to_reference_mirror_force_peak_n"] = max(
            _to_float(row.get("screw1_to_reference_mirror_force_peak_n")),
            _to_float(forces.get("screw1_to_reference_mirror_force_n")),
        )
        row["reference_filter_count"] = max(_to_float(row.get("reference_filter_count")), forces["reference_filter_count"])
        row["mirror_filter_count"] = max(_to_float(row.get("mirror_filter_count")), forces["mirror_filter_count"])
        row["reference_body_count"] = max(_to_float(row.get("reference_body_count")), forces["reference_body_count"])
        row["mirror_body_count"] = max(_to_float(row.get("mirror_body_count")), forces["mirror_body_count"])
        row["sensor_force_valid"] = bool(_bool(row.get("sensor_force_valid")) or forces["force_valid"] > 0.5)
        state_delta = _v91_state_delta(before_state, _v91_object_state_for_slot(backend, slot))
        row["screw1_displacement_m"] = max(_to_float(row.get("screw1_displacement_m")), _to_float(state_delta.get("root_pose_delta_m")))
        row["screw1_z_drift_m"] = max(_to_float(row.get("screw1_z_drift_m")), abs(_to_float(state_delta.get("root_delta_z_m"))))
        reference_pose = _v95_canary_cube_pose(backend, slot, reference_name)
        reference_now = _vec3(reference_pose.get("canary_cube_runtime_center_local_xyz", []))
        if before_reference and reference_now:
            row["reference_displacement_m"] = max(
                _to_float(row.get("reference_displacement_m")),
                _norm(_vec_sub(reference_now, before_reference)),
            )
        max_force = max(
            max_force,
            _to_float(row.get("screw1_unfiltered_force_peak_n")),
            _to_float(row.get("reference_unfiltered_force_peak_n")),
            _to_float(row.get("reference_to_screw1_filtered_force_peak_n")),
            _to_float(row.get("screw1_to_reference_mirror_force_peak_n")),
        )
        row["calibration_force_peak_n"] = max_force
        row["calibration_unsafe"] = bool(
            max_force > 5.0
            or _to_float(row.get("screw1_displacement_m")) > 0.02
            or _to_float(row.get("screw1_z_drift_m")) > 0.02
            or _to_float(row.get("reference_displacement_m")) > 0.02
        )
        if _bool(row.get("calibration_unsafe")) or _to_float(row.get("reference_to_screw1_filtered_force_peak_n")) > threshold:
            break

    filtered_peak = _to_float(row.get("reference_to_screw1_filtered_force_peak_n"))
    mirror_peak = _to_float(row.get("screw1_to_reference_mirror_force_peak_n"))
    unfiltered_peak = max(
        _to_float(row.get("screw1_unfiltered_force_peak_n")),
        _to_float(row.get("reference_unfiltered_force_peak_n")),
    )
    if _bool(row.get("calibration_unsafe")):
        verdict = "SCREW1_REFERENCE_OBJECT_UNSTABLE"
    elif filtered_peak > detection_threshold:
        verdict = "SCREW1_REFERENCE_CONTACT_STACK_WORKS"
        row["screw1_reference_contact_canary_ok"] = True
    elif mirror_peak > detection_threshold or unfiltered_peak > detection_threshold:
        verdict = "SCREW1_REFERENCE_FILTERED_OR_MIRROR_MISMATCH"
    elif not _bool(row.get("placement_contact_advisory_ok")):
        verdict = "SCREW1_REFERENCE_PLACEMENT_NOT_TOUCHING"
    else:
        verdict = "V95_SCREW1_CONTACT_STACK_NOT_TRUSTWORTHY_ASSET_OR_CONTACT_REPORT_REPAIR_REQUIRED"
    row["screw1_reference_contact_status"] = verdict
    row["verdict"] = verdict
    row["blocker"] = "" if _bool(row.get("screw1_reference_contact_canary_ok")) else verdict
    row["object_contact_success_evidence_used"] = False
    row["actual_object_contact_ok"] = False
    _park_v95_screw1_support_pad(backend, slot)
    _move_v95_canary_cube_runtime(backend, slot, reference_name, [1.25, -1.25, 1.20])
    _move_v95_canary_cube_runtime(backend, slot, parked_name, [1.30, -1.30, 1.25])
    return [row]


def _screw1_reference_contact_ok(rows: list[dict[str, Any]]) -> bool:
    return any(_bool(row.get("screw1_reference_contact_canary_ok")) for row in rows)


def _screw1_reference_contact_status(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        status = str(row.get("screw1_reference_contact_status") or row.get("verdict") or row.get("blocker") or "")
        if status:
            return status
    return "SCREW1_REFERENCE_CONTACT_NOT_EXECUTED"


def _pair_contact_matrix_rows(
    *,
    cube_canary_rows: list[dict[str, Any]],
    finger_reference_canary_rows: list[dict[str, Any]],
    screw1_reference_contact_rows: list[dict[str, Any]],
    screw1_forced_contact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cube = cube_canary_rows[0] if cube_canary_rows else {}
    rows.append(
        {
            "pair_name": "cube_to_cube",
            "source_body": cube.get("cube_a_prim_path", "V95CanaryCubeA"),
            "target_body": cube.get("cube_b_prim_path", "V95CanaryCubeB"),
            "unfiltered_force_peak_n": cube.get("cube_a_net_force_peak_n", 0.0),
            "filtered_force_peak_n": cube.get("cube_a_to_b_filtered_force_peak_n", 0.0),
            "mirror_force_peak_n": cube.get("cube_b_to_a_filtered_force_peak_n", 0.0),
            "source_displacement_m": 0.0,
            "target_displacement_m": 0.0,
            "unsafe": False,
            "verdict": "CUBE_CUBE_FILTERED_CONTACT_CANARY_PASSED"
            if _bool(cube.get("cube_cube_canary_ok"))
            else str(cube.get("blocker") or "CUBE_CUBE_FILTERED_CONTACT_CANARY_FAILED"),
        }
    )
    for logical_finger in (3, 4):
        canary = next(
            (
                row
                for row in finger_reference_canary_rows
                if _int_field(row, "logical_finger_id", -1) == logical_finger
            ),
            {},
        )
        rows.append(
            {
                "pair_name": f"finger{logical_finger}_to_reference",
                "source_body": canary.get("source_body_name", f"right_finger{logical_finger}_tip_link"),
                "target_body": canary.get("reference_pad_prim_path", f"v95_reference_pad_f{logical_finger}"),
                "unfiltered_force_peak_n": canary.get("active_unfiltered_force_peak_n", 0.0),
                "filtered_force_peak_n": canary.get("finger_to_reference_filtered_force_peak_n", 0.0),
                "mirror_force_peak_n": canary.get("reference_to_finger_mirror_force_peak_n", 0.0),
                "source_displacement_m": 0.0,
                "target_displacement_m": canary.get("reference_displacement_m", 0.0),
                "unsafe": bool(canary.get("calibration_unsafe")),
                "verdict": "FINGER_REFERENCE_CONTACT_STACK_WORKS"
                if _bool(canary.get("finger_reference_canary_ok"))
                else str(canary.get("blocker") or "FINGER_REFERENCE_CANARY_FAILED"),
            }
        )
    screw_ref = screw1_reference_contact_rows[0] if screw1_reference_contact_rows else {}
    rows.append(
        {
            "pair_name": "screw1_to_reference",
            "source_body": screw_ref.get("reference_prim_path", "/World/envs/env_1/V95CanaryCubeA"),
            "target_body": screw_ref.get("screw1_prim_path", "/World/envs/env_1/Screw1"),
            "unfiltered_force_peak_n": max(
                _to_float(screw_ref.get("screw1_unfiltered_force_peak_n")),
                _to_float(screw_ref.get("reference_unfiltered_force_peak_n")),
            ),
            "filtered_force_peak_n": screw_ref.get("reference_to_screw1_filtered_force_peak_n", 0.0),
            "mirror_force_peak_n": screw_ref.get("screw1_to_reference_mirror_force_peak_n", 0.0),
            "source_displacement_m": screw_ref.get("reference_displacement_m", 0.0),
            "target_displacement_m": screw_ref.get("screw1_displacement_m", 0.0),
            "unsafe": bool(screw_ref.get("calibration_unsafe")),
            "verdict": screw_ref.get("verdict")
            or screw_ref.get("screw1_reference_contact_status")
            or "SCREW1_REFERENCE_CONTACT_NOT_EXECUTED",
        }
    )
    screw_reference_ok = _screw1_reference_contact_ok(screw1_reference_contact_rows)
    for logical_finger in (3, 4):
        related = [
            row
            for row in screw1_forced_contact_rows
            if _int_field(row, "forced_contact_probe_finger_id", -1) == logical_finger
        ]
        if not screw_reference_ok:
            verdict = "NOT_EXECUTED_SCREW1_REFERENCE_CANARY_BLOCKED"
            unfiltered_peak = 0.0
            filtered_peak = 0.0
            mirror_peak = 0.0
            unsafe = False
            target_displacement = 0.0
            source_body = f"right_finger{logical_finger}_tip_link"
        else:
            verdict = _screw1_forced_contact_status(related) if related else "FORCED_CONTACT_NOT_EXECUTED"
            unfiltered_peak = max([0.0, *[_to_float(row.get("unfiltered_active_force_peak_n")) for row in related]])
            filtered_peak = max([0.0, *[_to_float(row.get("active_target_filtered_force_peak_n")) for row in related]])
            mirror_peak = max([0.0, *[_to_float(row.get("object_side_mirror_force_peak_n")) for row in related]])
            unsafe = any(_bool(row.get("calibration_unsafe")) for row in related)
            target_displacement = max([0.0, *[_to_float(row.get("screw1_response_delta_m")) for row in related]])
            source_body = next(
                (str(row.get("forced_pose_source_body")) for row in related if row.get("forced_pose_source_body")),
                f"right_finger{logical_finger}_tip_link",
            )
        rows.append(
            {
                "pair_name": f"finger{logical_finger}_to_screw1_forced",
                "source_body": source_body,
                "target_body": "/World/envs/env_1/Screw1",
                "unfiltered_force_peak_n": unfiltered_peak,
                "filtered_force_peak_n": filtered_peak,
                "mirror_force_peak_n": mirror_peak,
                "source_displacement_m": 0.0,
                "target_displacement_m": target_displacement,
                "unsafe": unsafe,
                "verdict": verdict,
            }
        )
    return rows


def _classify_screw1_forced_contact(
    *,
    target_peak: float,
    unfiltered_peak: float,
    mirror_peak: float,
    mirror_available: bool,
    unsafe: bool,
    object_only_stable_seen: bool,
    threshold: float,
    detection_threshold: float,
) -> str:
    if unsafe:
        return "forced_contact_calibration_unsafe"
    if _to_float(target_peak) > float(threshold) and _to_float(unfiltered_peak) > float(detection_threshold):
        if bool(mirror_available) and _to_float(mirror_peak) <= float(detection_threshold):
            return "SCREW1_FORCED_TARGET_FILTERED_WITH_MIRROR_MISSING"
        return "SCREW1_FORCED_TARGET_CONTACT_SENSOR_CHAIN_WORKS"
    if _to_float(target_peak) > float(detection_threshold):
        return "SCREW1_FORCED_TARGET_FILTERED_FORCE_OBSERVED_BELOW_OBJECT_THRESHOLD"
    if _to_float(unfiltered_peak) > float(detection_threshold):
        return "SCREW1_TARGET_FILTER_OR_COLLIDER_CONTACT_REPORT_BLOCKER"
    if not bool(object_only_stable_seen):
        return "FORCED_POSE_OBJECT_UNSTABLE"
    return "FORCED_CONTACT_POSE_NOT_TOUCHING_SOURCE_BODY"


def _screw1_forced_contact_status(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        status = str(row.get("forced_contact_status") or row.get("status") or row.get("blocker") or "")
        if status:
            return status
    return "SCREW1_FORCED_TARGET_CONTACT_NOT_EXECUTED"


def _screw1_forced_contact_ok(rows: list[dict[str, Any]]) -> bool:
    return any(_bool(row.get("forced_contact_calibration_ok")) for row in rows)


def _run_screw1_forced_target_contact_calibration(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    finger_rows: list[dict[str, Any]],
    finger_reference_canary_rows: list[dict[str, Any]],
    reference_filtered_rows: list[dict[str, Any]],
    filter_coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "forced_contact_calibration_ok": False, "blocker": "screw1_slot_missing"}]
    reference_ok = any(_bool(row.get("reference_filtered_calibration_ok")) for row in reference_filtered_rows)
    reference_blocker = _first_blocker(reference_filtered_rows, "blocker", "")
    filter_coverage_ok = any(_bool(row.get("filter_coverage_ok")) for row in filter_coverage_rows)
    filter_coverage_blocker = _first_blocker(filter_coverage_rows, "blocker", "")
    if not reference_ok:
        return [
            {
                "part_name": "Screw1",
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "forced_contact_calibration_ok": False,
                "forced_contact_status": "NOT_EXECUTED_CANARY_BLOCKED",
                "blocker": reference_blocker or "CANARY_FAILED_BEFORE_SCREW1_FORCED_CONTACT",
                "object_contact_success_evidence_used": False,
                "training_locked": True,
            }
        ]
    sensor_map = _finger_sensor_maps(finger_rows).get("Screw1", {})
    logical = _v91_active_finger_indices("34")
    active_sensors = [sensor_map.get(index, index) for index in logical]
    _reset_with_plan(backend, plan_rows, settle_steps=2)
    forced_targets = _screw1_forced_contact_targets_from_reference_canary(
        backend,
        slot,
        finger_reference_canary_rows,
    )
    if not forced_targets:
        return [
            {
                "part_name": "Screw1",
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "forced_contact_calibration_ok": False,
                "forced_contact_status": "SCREW1_FORCED_SOURCE_BODY_POSE_UNAVAILABLE",
                "blocker": "reference_canary_contact_template_unavailable",
                "training_locked": True,
            }
        ]
    rows: list[dict[str, Any]] = []
    target_peak = 0.0
    unfiltered_peak = 0.0
    mirror_peak = 0.0
    non_active_target_peak = 0.0
    non_active_unfiltered_peak = 0.0
    non_active_mirror_peak = 0.0
    unsafe = False
    blocker = ""
    object_only_stable_seen = False
    object_only_unstable_seen = False
    finger_contact_executed = False
    threshold = float(backend.contact_manager.force_threshold_n)
    detection_threshold = max(1.0e-6, threshold * 0.002)
    step_index = 0
    mirror_available = _v95_screw1_mirror_available(backend, slot)
    safe_hand_target = [0.34, 0.28, 1.18]
    for target in forced_targets:
        forced_center = list(target.get("forced_object_center_local_xyz") or [])
        if len(forced_center) < 3:
            continue
        support_center = list(target.get("screw1_support_pad_center_local_xyz") or [])
        if len(support_center) < 3:
            continue

        object_plan = _forced_screw1_plan(
            plan_rows,
            forced_center=forced_center,
            hand_target=safe_hand_target,
        )
        support_info = _reset_forced_screw1_with_support(
            backend,
            object_plan,
            support_center_local_xyz=support_center,
        )
        before_state = _v91_object_state_for_slot(backend, slot)
        target_object_only_ok = False
        target_object_only_unsafe = False
        for local_step in range(6):
            _set_context(
                backend,
                "v95_screw1_forced_object_only_stability",
                finger=target.get("forced_pose_source_finger_id", 0),
                step=step_index,
            )
            backend.step_envs([[0.0] * 16 for _slot in backend.slots])
            metrics = backend._metrics_for_slot(slot, f"v95_screw1_forced_object_only_{step_index}")
            target_forces = _v95_target_force_list(metrics)
            unfiltered = _v92_force_list(metrics)
            mirror = _v95_screw1_mirror_force_list(backend, slot)
            step_target = _force_peak_for_indices(target_forces, active_sensors)
            step_unfiltered = _force_peak_for_indices(unfiltered, active_sensors)
            step_mirror = _force_peak_for_indices(mirror, logical)
            target_peak = max(target_peak, step_target)
            unfiltered_peak = max(unfiltered_peak, step_unfiltered)
            mirror_peak = max(mirror_peak, step_mirror)
            mirror_available = bool(mirror_available or _v95_screw1_mirror_available(backend, slot))
            non_active_target_peak = max(non_active_target_peak, _force_peak_except_indices(target_forces, active_sensors))
            non_active_unfiltered_peak = max(non_active_unfiltered_peak, _force_peak_except_indices(unfiltered, active_sensors))
            non_active_mirror_peak = max(non_active_mirror_peak, _force_peak_except_indices(mirror, logical))
            state_delta = _v91_state_delta(before_state, _v91_object_state_for_slot(backend, slot))
            object_delta = _to_float(state_delta.get("root_pose_delta_m"))
            object_z_drift = abs(_to_float(state_delta.get("root_delta_z_m")))
            force_peak = max(step_target, step_unfiltered, step_mirror)
            support_blocker = str(support_info.get("screw1_support_pad_blocker") or "")
            target_object_only_unsafe = bool(
                support_blocker
                or object_delta > 0.02
                or object_z_drift > 0.02
                or force_peak > 5.0
            )
            forced_status = "FORCED_POSE_OBJECT_UNSTABLE" if target_object_only_unsafe else "OBJECT_ONLY_STABLE"
            if target_object_only_unsafe:
                unsafe = True
                object_only_unstable_seen = True
                blocker = "FORCED_POSE_OBJECT_UNSTABLE"
            else:
                object_only_stable_seen = True
            rows.append(
                {
                    "part_name": "Screw1",
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "phase": "object_only_stability",
                    "step": step_index,
                    "phase_step": local_step,
                    "forced_bracket_index": target.get("forced_bracket_index", ""),
                    "forced_contact_probe_finger_id": target.get("forced_pose_source_finger_id", ""),
                    "reference_filtered_precondition_ok": reference_ok,
                    "filtered_contact_canary_precondition_ok": reference_ok,
                    "forced_contact_diagnostic_only": True,
                    "finger_contact_phase_executed": False,
                    "object_only_stability_ok": not target_object_only_unsafe,
                    "forced_contact_status": forced_status,
                    "forced_pose_source_body": target.get("forced_pose_source_body", ""),
                    "forced_pose_source_body_local_xyz": target.get("forced_pose_source_body_local_xyz", []),
                    "forced_reference_pad_center_local_xyz": target.get("forced_reference_pad_center_local_xyz", []),
                    "forced_reference_attempt_offset_local_xyz": target.get(
                        "forced_reference_attempt_offset_local_xyz", []
                    ),
                    "forced_reference_contact_normal_local_xyz": target.get(
                        "forced_reference_contact_normal_local_xyz", []
                    ),
                    "forced_reference_tangent_x_local_xyz": target.get("forced_reference_tangent_x_local_xyz", []),
                    "forced_reference_tangent_y_local_xyz": target.get("forced_reference_tangent_y_local_xyz", []),
                    "forced_bracket_offset_local_xyz": target.get("forced_bracket_offset_local_xyz", []),
                    "forced_object_center_local_xyz": forced_center,
                    "forced_screw1_runtime_root_local_xyz": _screw1_root_local_xyz(backend, slot),
                    "forced_pose_to_reference_pad_distance_m": target.get(
                        "forced_pose_to_reference_pad_distance_m", ""
                    ),
                    "forced_pose_to_source_body_distance_m": target.get(
                        "forced_pose_to_source_body_distance_m", ""
                    ),
                    "screw1_support_pad_center_local_xyz": support_info.get(
                        "screw1_support_pad_center_local_xyz",
                        support_center,
                    ),
                    "screw1_support_pad_half_extent_xyz": target.get("screw1_support_pad_half_extent_xyz", []),
                    "screw1_support_pad_blocker": support_blocker,
                    "screw1_support_pad_rigid_object": bool(
                        support_info.get("screw1_support_pad_rigid_object", False)
                    ),
                    "screw1_support_bottom_offset_m": target.get("screw1_support_bottom_offset_m", ""),
                    "screw1_support_offset_source": target.get("screw1_support_offset_source", ""),
                    "forced_screw1_env_path": f"/World/envs/env_{slot.local_env_index}/Screw1",
                    "target_filter_coverage_ok": filter_coverage_ok,
                    "target_filter_coverage_blocker": filter_coverage_blocker,
                    "active_logical_finger_indices": logical,
                    "active_sensor_indices": active_sensors,
                    "finger_side_target_filtered_force_step_n": step_target,
                    "finger_side_target_filtered_force_peak_n": target_peak,
                    "active_target_filtered_force_peak_n": target_peak,
                    "unfiltered_active_force_step_n": step_unfiltered,
                    "unfiltered_active_force_peak_n": unfiltered_peak,
                    "object_side_mirror_force_step_n": step_mirror,
                    "object_side_mirror_force_peak_n": mirror_peak,
                    "non_active_target_filtered_force_peak_n": non_active_target_peak,
                    "non_active_unfiltered_force_peak_n": non_active_unfiltered_peak,
                    "non_active_mirror_force_peak_n": non_active_mirror_peak,
                    "screw1_response_delta_m": object_delta,
                    "screw1_z_drift_m": object_z_drift,
                    "screw1_velocity_norm": metrics.get("object_velocity_norm", 0.0),
                    "target_filter_path": "/World/envs/env_.*/Screw1",
                    "target_contact_evidence_source": metrics.get("target_contact_evidence_source", ""),
                    "mirror_sensor_available": mirror_available,
                    "calibration_force_peak_n": force_peak,
                    "calibration_unsafe": target_object_only_unsafe,
                    "unfiltered_force_observed": unfiltered_peak > detection_threshold,
                    "target_filtered_force_observed": target_peak > detection_threshold,
                    "force_detection_threshold_n": detection_threshold,
                    "force_contact_threshold_n": threshold,
                    "object_contact_success_evidence_used": False,
                    "actual_object_contact_ok": False,
                    "distance_only_success_used": False,
                    "training_locked": True,
                    "blocker": blocker if target_object_only_unsafe else "",
                }
            )
            step_index += 1
            if target_object_only_unsafe:
                break
        target_object_only_ok = not target_object_only_unsafe
        if not target_object_only_ok:
            unsafe = False
            continue

        contact_plan = _forced_screw1_plan(
            plan_rows,
            forced_center=forced_center,
            hand_target=None,
        )
        support_info = _reset_forced_screw1_with_support(
            backend,
            contact_plan,
            support_center_local_xyz=support_center,
        )
        before_state = _v91_object_state_for_slot(backend, slot)
        for local_step in range(8):
            actions = [[0.0] * 16 for _slot in backend.slots]
            _v91_apply_finger_mask(
                actions[slot.local_env_index],
                group="34",
                active_value=min(0.30, 0.04 * float(local_step + 1)),
                support_value=0.0,
            )
            _set_context(
                backend,
                "v95_screw1_forced_finger_contact_bracket",
                finger=target.get("forced_pose_source_finger_id", 0),
                step=step_index,
            )
            backend.step_envs(actions)
            finger_contact_executed = True
            metrics = backend._metrics_for_slot(slot, f"v95_screw1_forced_finger_contact_{step_index}")
            target_forces = _v95_target_force_list(metrics)
            unfiltered = _v92_force_list(metrics)
            mirror = _v95_screw1_mirror_force_list(backend, slot)
            step_target = _force_peak_for_indices(target_forces, active_sensors)
            step_unfiltered = _force_peak_for_indices(unfiltered, active_sensors)
            step_mirror = _force_peak_for_indices(mirror, logical)
            target_peak = max(target_peak, step_target)
            unfiltered_peak = max(unfiltered_peak, step_unfiltered)
            mirror_peak = max(mirror_peak, step_mirror)
            mirror_available = bool(mirror_available or _v95_screw1_mirror_available(backend, slot))
            non_active_target_peak = max(non_active_target_peak, _force_peak_except_indices(target_forces, active_sensors))
            non_active_unfiltered_peak = max(non_active_unfiltered_peak, _force_peak_except_indices(unfiltered, active_sensors))
            non_active_mirror_peak = max(non_active_mirror_peak, _force_peak_except_indices(mirror, logical))
            state_delta = _v91_state_delta(before_state, _v91_object_state_for_slot(backend, slot))
            object_delta = _to_float(state_delta.get("root_pose_delta_m"))
            object_z_drift = abs(_to_float(state_delta.get("root_delta_z_m")))
            force_peak = max(step_target, step_unfiltered, step_mirror)
            unsafe = bool(force_peak > 5.0 or object_delta > 0.02 or object_z_drift > 0.02)
            forced_status = _classify_screw1_forced_contact(
                target_peak=target_peak,
                unfiltered_peak=unfiltered_peak,
                mirror_peak=mirror_peak,
                mirror_available=mirror_available,
                unsafe=unsafe,
                object_only_stable_seen=object_only_stable_seen,
                threshold=threshold,
                detection_threshold=detection_threshold,
            )
            if unsafe:
                blocker = "forced_contact_calibration_unsafe"
            rows.append(
                {
                    "part_name": "Screw1",
                    "env_index": slot.global_env_index,
                    "local_env_index": slot.local_env_index,
                    "phase": "finger_contact_bracket",
                    "step": step_index,
                    "phase_step": local_step,
                    "forced_bracket_index": target.get("forced_bracket_index", ""),
                    "forced_contact_probe_finger_id": target.get("forced_pose_source_finger_id", ""),
                    "reference_filtered_precondition_ok": reference_ok,
                    "filtered_contact_canary_precondition_ok": reference_ok,
                    "forced_contact_diagnostic_only": True,
                    "finger_contact_phase_executed": True,
                    "object_only_stability_ok": True,
                    "forced_contact_status": forced_status,
                    "forced_pose_source_body": target.get("forced_pose_source_body", ""),
                    "forced_pose_source_body_local_xyz": target.get("forced_pose_source_body_local_xyz", []),
                    "forced_reference_pad_center_local_xyz": target.get("forced_reference_pad_center_local_xyz", []),
                    "forced_reference_attempt_offset_local_xyz": target.get(
                        "forced_reference_attempt_offset_local_xyz", []
                    ),
                    "forced_reference_contact_normal_local_xyz": target.get(
                        "forced_reference_contact_normal_local_xyz", []
                    ),
                    "forced_reference_tangent_x_local_xyz": target.get("forced_reference_tangent_x_local_xyz", []),
                    "forced_reference_tangent_y_local_xyz": target.get("forced_reference_tangent_y_local_xyz", []),
                    "forced_bracket_offset_local_xyz": target.get("forced_bracket_offset_local_xyz", []),
                    "forced_object_center_local_xyz": forced_center,
                    "forced_screw1_runtime_root_local_xyz": _screw1_root_local_xyz(backend, slot),
                    "forced_pose_to_reference_pad_distance_m": target.get(
                        "forced_pose_to_reference_pad_distance_m", ""
                    ),
                    "forced_pose_to_source_body_distance_m": target.get(
                        "forced_pose_to_source_body_distance_m", ""
                    ),
                    "screw1_support_pad_center_local_xyz": support_info.get(
                        "screw1_support_pad_center_local_xyz",
                        support_center,
                    ),
                    "screw1_support_pad_half_extent_xyz": target.get("screw1_support_pad_half_extent_xyz", []),
                    "screw1_support_pad_blocker": support_info.get("screw1_support_pad_blocker", ""),
                    "screw1_support_pad_rigid_object": bool(
                        support_info.get("screw1_support_pad_rigid_object", False)
                    ),
                    "screw1_support_bottom_offset_m": target.get("screw1_support_bottom_offset_m", ""),
                    "screw1_support_offset_source": target.get("screw1_support_offset_source", ""),
                    "forced_screw1_env_path": f"/World/envs/env_{slot.local_env_index}/Screw1",
                    "target_filter_coverage_ok": filter_coverage_ok,
                    "target_filter_coverage_blocker": filter_coverage_blocker,
                    "active_logical_finger_indices": logical,
                    "active_sensor_indices": active_sensors,
                    "finger_side_target_filtered_force_step_n": step_target,
                    "finger_side_target_filtered_force_peak_n": target_peak,
                    "active_target_filtered_force_peak_n": target_peak,
                    "unfiltered_active_force_step_n": step_unfiltered,
                    "unfiltered_active_force_peak_n": unfiltered_peak,
                    "object_side_mirror_force_step_n": step_mirror,
                    "object_side_mirror_force_peak_n": mirror_peak,
                    "non_active_target_filtered_force_peak_n": non_active_target_peak,
                    "non_active_unfiltered_force_peak_n": non_active_unfiltered_peak,
                    "non_active_mirror_force_peak_n": non_active_mirror_peak,
                    "screw1_response_delta_m": object_delta,
                    "screw1_z_drift_m": object_z_drift,
                    "screw1_velocity_norm": metrics.get("object_velocity_norm", 0.0),
                    "target_filter_path": "/World/envs/env_.*/Screw1",
                    "target_contact_evidence_source": metrics.get("target_contact_evidence_source", ""),
                    "mirror_sensor_available": mirror_available,
                    "calibration_force_peak_n": force_peak,
                    "calibration_unsafe": unsafe,
                    "unfiltered_force_observed": unfiltered_peak > detection_threshold,
                    "target_filtered_force_observed": target_peak > detection_threshold,
                    "force_detection_threshold_n": detection_threshold,
                    "force_contact_threshold_n": threshold,
                    "object_contact_success_evidence_used": False,
                    "actual_object_contact_ok": False,
                    "distance_only_success_used": False,
                    "training_locked": True,
                    "blocker": blocker,
                }
            )
            step_index += 1
            if unsafe or forced_status == "SCREW1_FORCED_TARGET_CONTACT_SENSOR_CHAIN_WORKS":
                break
        if unsafe or target_peak > threshold:
            break
    final = rows[-1] if rows else {}
    final_status = _classify_screw1_forced_contact(
        target_peak=target_peak,
        unfiltered_peak=unfiltered_peak,
        mirror_peak=mirror_peak,
        mirror_available=mirror_available,
        unsafe=unsafe,
        object_only_stable_seen=object_only_stable_seen,
        threshold=threshold,
        detection_threshold=detection_threshold,
    )
    final_ok = bool(
        reference_ok
        and filter_coverage_ok
        and final_status == "SCREW1_FORCED_TARGET_CONTACT_SENSOR_CHAIN_WORKS"
        and not _bool(final.get("calibration_unsafe"))
    )
    if rows:
        rows[-1]["forced_contact_calibration_ok"] = final_ok
        rows[-1]["ordinary_acquisition_executed_after_forced_ok"] = False
        rows[-1]["ordinary_acquisition_next_allowed"] = final_ok
        rows[-1]["forced_contact_status"] = final_status
        if not filter_coverage_ok:
            rows[-1]["blocker"] = filter_coverage_blocker or "SCREW1_TARGET_FILTER_COVERAGE_INVALID"
        elif final_status != "SCREW1_FORCED_TARGET_CONTACT_SENSOR_CHAIN_WORKS":
            rows[-1]["blocker"] = final_status
        else:
            rows[-1]["blocker"] = ""
        rows[-1]["object_only_stable_pose_seen"] = object_only_stable_seen
        rows[-1]["object_only_unstable_pose_seen"] = object_only_unstable_seen
        rows[-1]["finger_contact_phase_executed_any"] = finger_contact_executed
        _park_v95_screw1_support_pad(backend, slot)
    return rows


def _run_screw1_object_side_mirror_check(
    backend: IsaacUnifiedSingleContextBackend,
    *,
    forced_contact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot = _slot_by_part(backend, "Screw1")
    if slot is None:
        return [{"part_name": "Screw1", "mirror_check_ok": False, "blocker": "screw1_slot_missing"}]
    finger_peak = max([0.0, *[_to_float(row.get("finger_side_target_filtered_force_peak_n")) for row in forced_contact_rows]])
    mirror_peak = max([0.0, *[_to_float(row.get("object_side_mirror_force_peak_n")) for row in forced_contact_rows]])
    unfiltered_peak = max([0.0, *[_to_float(row.get("unfiltered_active_force_peak_n")) for row in forced_contact_rows]])
    threshold = float(backend.contact_manager.force_threshold_n)
    mirror_available = _v95_screw1_mirror_available(backend, slot)
    if not mirror_available:
        status = "MIRROR_SENSOR_UNAVAILABLE"
    elif finger_peak > threshold and mirror_peak > threshold:
        status = "FINGER_SIDE_AND_OBJECT_SIDE_FILTERED_FORCE_AGREE"
    elif mirror_peak > threshold and finger_peak <= threshold:
        status = "OBJECT_SIDE_FORCE_ONLY_FINGERTIP_FILTER_OR_MAPPING_BLOCKER"
    elif finger_peak > threshold and mirror_peak <= threshold:
        status = "FINGER_SIDE_FORCE_ONLY_OBJECT_SIDE_SENSOR_OR_REPORTING_BLOCKER"
    elif unfiltered_peak > threshold:
        status = "UNFILTERED_FORCE_WITHOUT_FILTERED_PAIR_FORCE"
    else:
        status = "NO_FORCED_CONTACT_FORCE_OBSERVED"
    return [
        {
            "part_name": "Screw1",
            "env_index": slot.global_env_index,
            "local_env_index": slot.local_env_index,
            "mirror_sensor_available": mirror_available,
            "mirror_filter_count": _tensor_float(getattr(_base_env(backend), "v95_screw1_mirror_filter_count", None), slot.local_env_index),
            "finger_side_target_filtered_force_peak_n": finger_peak,
            "object_side_mirror_force_peak_n": mirror_peak,
            "unfiltered_active_force_peak_n": unfiltered_peak,
            "mirror_consistency_status": status,
            "mirror_check_ok": status == "FINGER_SIDE_AND_OBJECT_SIDE_FILTERED_FORCE_AGREE",
            "object_contact_success_evidence_used": False,
            "distance_only_success_used": False,
            "training_locked": True,
            "blocker": "" if status == "FINGER_SIDE_AND_OBJECT_SIDE_FILTERED_FORCE_AGREE" else status,
        }
    ]


def _run_actual_object_contact_gate(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    finger_rows: list[dict[str, Any]],
    wrist_rows: list[dict[str, Any]],
    allowed_parts: set[str] | None = None,
    forced_contact_status: str = "",
    ordinary_acquisition_executed_after_forced_ok: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    sensor_maps = _finger_sensor_maps(finger_rows)
    finger_ok_by_part = _finger_motion_ok_by_part(finger_rows)
    wrist_ok_by_part = {
        slot.part_name: bool(
            [row for row in wrist_rows if row.get("part_name") == slot.part_name]
            and all(_bool(row.get("wrist_action_mapping_ok")) for row in wrist_rows if row.get("part_name") == slot.part_name)
        )
        for slot in backend.slots
    }
    for mode in ("single_finger", "two_multi_finger"):
        mode_rows, mode_trace = _run_target_filtered_contact_acquisition(
            backend,
            plan_rows,
            mode=mode,
            sensor_maps=sensor_maps,
            finger_ok_by_part=finger_ok_by_part,
            wrist_ok_by_part=wrist_ok_by_part,
            allowed_parts=allowed_parts,
            forced_contact_status=forced_contact_status,
            ordinary_acquisition_executed_after_forced_ok=ordinary_acquisition_executed_after_forced_ok,
        )
        rows.extend(mode_rows)
        trace_rows.extend(mode_trace)
    return rows, trace_rows


def _run_target_filtered_contact_acquisition(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    mode: str,
    sensor_maps: dict[str, dict[int, int]],
    finger_ok_by_part: dict[str, dict[int, bool]],
    wrist_ok_by_part: dict[str, bool],
    allowed_parts: set[str] | None = None,
    forced_contact_status: str = "",
    ordinary_acquisition_executed_after_forced_ok: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = float(backend.contact_manager.force_threshold_n)
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    _reset_with_plan(backend, plan_rows)
    before_states = {slot.part_name: _v91_object_state_for_slot(backend, slot) for slot in backend.slots}
    before_tips = _v92_tip_positions(backend)
    before_joints = _v92_joint_pos(backend)
    before_targets = _v95_hand_target_pos(backend)
    pre_distances = {
        slot.part_name: _v94_distance_for_slot(
            backend,
            slot,
            str(_find_slot_row(plan_rows, slot).get("active_finger_group") or ""),
        )
        for slot in backend.slots
    }
    states: dict[str, dict[str, Any]] = {}
    for slot in backend.slots:
        allowed = allowed_parts is None or slot.part_name in allowed_parts
        group = _contact_mode_group(plan_rows, slot, mode)
        logical = _v91_active_finger_indices(group)
        sensor_map = sensor_maps.get(slot.part_name, {})
        mapped = [sensor_map.get(index, -1) for index in logical]
        mapped = [index for index in mapped if 0 <= index < 5]
        mapping_valid = len(mapped) == len(logical)
        finger_ready = bool(
            mapping_valid
            and logical
            and all(bool(finger_ok_by_part.get(slot.part_name, {}).get(index, False)) for index in logical)
        )
        metrics = backend._metrics_for_slot(slot, f"v95_contact_acquisition_{mode}_preflight")
        preflight = _target_filter_preflight_for_slot(backend, slot, metrics)
        target_vec = _approach_vector_for_slot(backend, slot, logical)
        center_distance = _target_center_distance_for_slot(backend, slot, logical)
        blocker = ""
        done = False
        if not allowed:
            blocker = "NOT_EXECUTED_SCREW1_ONLY_REGRESSION_GUARD"
            done = True
        elif not bool(preflight.get("target_filter_preflight_ok")):
            blocker = "TARGET_FILTER_CONFIG_INVALID"
            done = True
        elif not bool(wrist_ok_by_part.get(slot.part_name, False)):
            blocker = "WRIST_OR_WORKSPACE_BLOCKED"
            done = True
        states[slot.part_name] = {
            "slot": slot,
            "allowed_for_ordinary_acquisition": allowed,
            "forced_contact_status": forced_contact_status,
            "ordinary_acquisition_executed_after_forced_ok": bool(
                ordinary_acquisition_executed_after_forced_ok and slot.part_name == "Screw1"
            ),
            "group": group,
            "logical_indices": logical,
            "mapped_indices": mapped,
            "mapping_valid": mapping_valid,
            "finger_ready": finger_ready,
            "wrist_ready": bool(wrist_ok_by_part.get(slot.part_name, False)),
            "preflight": preflight,
            "target_direction_xyz": target_vec.get("direction", [0.0, 0.0, 0.0]),
            "target_direction_valid": bool(target_vec.get("valid")),
            "initial_center_distance_m": center_distance,
            "best_center_distance_m": center_distance,
            "target_progress_peak_m": 0.0,
            "last_approach_action_xyz": [0.0, 0.0, 0.0],
            "per_finger_target_peak": [0.0] * 5,
            "per_finger_unfiltered_peak": [0.0] * 5,
            "active_target_force_peak_n": 0.0,
            "active_unfiltered_force_peak_n": 0.0,
            "unfiltered_force_peak_n": 0.0,
            "non_active_target_force_peak_n": 0.0,
            "non_active_unfiltered_force_peak_n": 0.0,
            "active_target_filtered_force_count": 0,
            "target_filtered_available": bool(metrics.get("target_filtered_force_available")),
            "contact_sensor_api_available": bool(metrics.get("contact_sensor_api_available")),
            "target_response_observed": False,
            "target_response_delta_m": 0.0,
            "target_response_velocity_norm": 0.0,
            "table_collision": bool(metrics.get("table_collision")),
            "workspace_or_clamp_blocked": False,
            "close_executed": False,
            "acquired_phase": "",
            "done": done,
            "blocker": blocker,
            "final_metrics": metrics,
        }
        trace_rows.append(
            _contact_acquisition_trace_row(
                backend,
                slot,
                mode=mode,
                phase="target_filter_preflight",
                step=-1,
                state=states[slot.part_name],
                metrics=metrics,
                action_xyz=[0.0, 0.0, 0.0],
                close_value=0.0,
                blocker=blocker,
            )
        )

    for step in range(18):
        actions = [[0.0] * 16 for _slot in backend.slots]
        for action_index, slot in enumerate(backend.slots):
            state = states[slot.part_name]
            if bool(state.get("done")):
                continue
            if not bool(state.get("target_direction_valid")):
                state["done"] = True
                state["blocker"] = "APPROACH_DIRECTION_NOT_TOWARD_TARGET"
                continue
            scale = min(0.75, 0.045 * float(step + 1))
            direction = [float(value) for value in list(state.get("target_direction_xyz") or [0.0, 0.0, 0.0])[:3]]
            action_xyz = [max(-0.75, min(0.75, float(value) * scale)) for value in direction]
            states[slot.part_name]["last_approach_action_xyz"] = action_xyz
            actions[action_index][0:3] = action_xyz
        _set_context(backend, f"v95_contact_acquisition_{mode}_anchored_approach", step=step)
        backend.step_envs(actions)
        diag = _v95_control_diag_rows(backend)
        for slot in backend.slots:
            state = states[slot.part_name]
            if bool(state.get("done")):
                continue
            metrics = backend._metrics_for_slot(slot, f"v95_contact_acquisition_{mode}_approach_{step}")
            _update_contact_acquisition_state(backend, state, metrics, before_states.get(slot.part_name, {}))
            current_distance = _target_center_distance_for_slot(backend, slot, state.get("logical_indices", []))
            if math.isfinite(current_distance):
                state["best_center_distance_m"] = min(_to_float(state.get("best_center_distance_m"), current_distance), current_distance)
                state["target_progress_peak_m"] = max(
                    _to_float(state.get("target_progress_peak_m")),
                    _to_float(state.get("initial_center_distance_m"), current_distance) - current_distance,
                )
            control = diag.get(slot.part_name, {})
            clamp_blocked = bool(
                _to_float(control.get("workspace_clamp_delta_m")) > 0.004
                or _to_float(control.get("table_barrier_delta_z_m")) > 0.001
            )
            state["workspace_or_clamp_blocked"] = bool(state.get("workspace_or_clamp_blocked") or clamp_blocked)
            blocker = ""
            if _to_float(state.get("active_target_force_peak_n")) > threshold:
                state["done"] = True
                state["blocker"] = "TARGET_CONTACT_ACQUIRED"
                state["acquired_phase"] = "anchored_approach"
                blocker = "TARGET_CONTACT_ACQUIRED"
            elif _to_float(state.get("unfiltered_force_peak_n")) > threshold:
                state["done"] = True
                state["blocker"] = "NON_TARGET_CONTACT_BEFORE_TARGET"
                blocker = "NON_TARGET_CONTACT_BEFORE_TARGET"
            elif clamp_blocked:
                state["done"] = True
                state["blocker"] = "WRIST_OR_WORKSPACE_BLOCKED"
                blocker = "WRIST_OR_WORKSPACE_BLOCKED"
            trace_rows.append(
                _contact_acquisition_trace_row(
                    backend,
                    slot,
                    mode=mode,
                    phase="anchored_approach",
                    step=step,
                    state=state,
                    metrics=metrics,
                    action_xyz=state.get("last_approach_action_xyz", [0.0, 0.0, 0.0]),
                    close_value=0.0,
                    blocker=blocker,
                )
            )

    for step in range(24):
        actions = [[0.0] * 16 for _slot in backend.slots]
        any_active = False
        for action_index, slot in enumerate(backend.slots):
            state = states[slot.part_name]
            if bool(state.get("done")):
                continue
            if not bool(state.get("finger_ready")):
                state["done"] = True
                state["blocker"] = "FINGER_CONTROL_BLOCKED"
                continue
            action = actions[action_index]
            action[0:3] = list(state.get("last_approach_action_xyz") or [0.0, 0.0, 0.0])[:3]
            close_value = min(0.95, 0.04 * float(step + 1))
            _v91_apply_finger_mask(action, group=str(state.get("group") or ""), active_value=close_value, support_value=0.0)
            state["close_executed"] = True
            any_active = True
        if not any_active:
            break
        _set_context(backend, f"v95_contact_acquisition_{mode}_controlled_close", step=step)
        backend.step_envs(actions)
        for slot in backend.slots:
            state = states[slot.part_name]
            if bool(state.get("done")):
                continue
            metrics = backend._metrics_for_slot(slot, f"v95_contact_acquisition_{mode}_close_{step}")
            _update_contact_acquisition_state(backend, state, metrics, before_states.get(slot.part_name, {}))
            current_distance = _target_center_distance_for_slot(backend, slot, state.get("logical_indices", []))
            if math.isfinite(current_distance):
                state["best_center_distance_m"] = min(_to_float(state.get("best_center_distance_m"), current_distance), current_distance)
                state["target_progress_peak_m"] = max(
                    _to_float(state.get("target_progress_peak_m")),
                    _to_float(state.get("initial_center_distance_m"), current_distance) - current_distance,
                )
            blocker = ""
            if _to_float(state.get("active_target_force_peak_n")) > threshold:
                state["done"] = True
                state["blocker"] = "TARGET_CONTACT_ACQUIRED"
                state["acquired_phase"] = "controlled_close"
                blocker = "TARGET_CONTACT_ACQUIRED"
            elif _to_float(state.get("unfiltered_force_peak_n")) > threshold:
                state["done"] = True
                state["blocker"] = "NON_TARGET_CONTACT_BEFORE_TARGET"
                blocker = "NON_TARGET_CONTACT_BEFORE_TARGET"
            trace_rows.append(
                _contact_acquisition_trace_row(
                    backend,
                    slot,
                    mode=mode,
                    phase="controlled_close",
                    step=step,
                    state=state,
                    metrics=metrics,
                    action_xyz=actions[backend.slots.index(slot)][0:3],
                    close_value=min(0.95, 0.04 * float(step + 1)),
                    blocker=blocker,
                )
            )

    after_tips = _v92_tip_positions(backend)
    after_joints = _v92_joint_pos(backend)
    after_targets = _v95_hand_target_pos(backend)
    for slot in backend.slots:
        state = states[slot.part_name]
        metrics = backend._metrics_for_slot(slot, f"v95_contact_acquisition_{mode}_summary")
        _update_contact_acquisition_state(backend, state, metrics, before_states.get(slot.part_name, {}))
        if str(state.get("blocker") or "") == "":
            state["blocker"] = _final_acquisition_blocker(backend, state)
        commanded_group = str(state.get("group") or "")
        commanded_indices = list(state.get("logical_indices") or [])
        active_indices = list(state.get("mapped_indices") or [])
        required_count = 1 if mode == "single_finger" else min(2, max(1, len(active_indices)))
        contact_active_target_delta = _group_close_joint_delta(before_targets, after_targets, slot.local_env_index, commanded_indices)
        contact_active_joint_delta = _group_close_joint_delta(before_joints, after_joints, slot.local_env_index, commanded_indices)
        contact_active_tip_delta = _group_tip_delta_max(before_tips, after_tips, slot.local_env_index, commanded_indices)
        non_active = [index for index in range(5) if index not in set(commanded_indices)]
        contact_non_active_joint_delta = _group_close_joint_delta(before_joints, after_joints, slot.local_env_index, non_active)
        contact_non_active_tip_delta = _group_tip_delta_max(before_tips, after_tips, slot.local_env_index, non_active)
        finger_precondition_ok = bool(state.get("finger_ready"))
        contact_commanded_fingers_moved = bool(
            finger_precondition_ok
            and (
                str(state.get("acquired_phase") or "") == "anchored_approach"
                or (
                    contact_active_target_delta >= 1.0e-4
                    and contact_active_joint_delta >= 1.0e-4
                    and contact_active_tip_delta >= 2.0e-4
                    and contact_active_tip_delta >= max(1.0e-4, contact_non_active_tip_delta * 1.20)
                )
            )
        )
        max_per_finger = list(state.get("per_finger_target_peak") or [0.0] * 5)[:5]
        unfiltered_per_finger = list(state.get("per_finger_unfiltered_peak") or [0.0] * 5)[:5]
        max_per_finger.extend([0.0] * max(0, 5 - len(max_per_finger)))
        unfiltered_per_finger.extend([0.0] * max(0, 5 - len(unfiltered_per_finger)))
        active_force_peak = _to_float(state.get("active_target_force_peak_n"))
        count = int(state.get("active_target_filtered_force_count") or 0)
        target_filtered_available = bool(state.get("target_filtered_available"))
        target_evidence = bool(
            target_filtered_available
            and active_force_peak > threshold
            and str(metrics.get("target_contact_evidence_source") or "ContactSensor.force_matrix_w") == "ContactSensor.force_matrix_w"
        )
        evidence_source = "ContactSensor.force_matrix_w" if target_evidence else "target_filtered_force_no_target_pair_response"
        if _to_float(state.get("unfiltered_force_peak_n")) > threshold and active_force_peak <= threshold:
            evidence_source = "unfiltered_force_without_target_filtered_force"
        sensor_mismatch = bool(
            _to_float(state.get("non_active_target_force_peak_n")) > threshold
            and active_force_peak <= threshold
        )
        unfiltered_non_target_contact = bool(
            _to_float(state.get("unfiltered_force_peak_n")) > threshold
            and active_force_peak <= threshold
        )
        preclose_distance = _to_float(
            pre_distances.get(slot.part_name, {}).get("fingertip_object_surface_distance_min_m"),
            1.0,
        )
        distance_untrusted = bool(preclose_distance <= 1.0e-6 and active_force_peak <= threshold)
        ok = bool(
            state.get("wrist_ready")
            and finger_precondition_ok
            and contact_commanded_fingers_moved
            and target_filtered_available
            and active_force_peak > threshold
            and count >= required_count
            and target_evidence
            and not bool(state.get("table_collision"))
            and not backend.object_write_by_policy_detected
            and not backend.sticky_action_available_to_policy
            and not _env_enabled("WUJI_V93_COLLISION_REPAIR")
        )
        blocker = "" if ok else str(state.get("blocker") or _final_acquisition_blocker(backend, state))
        if not ok and not finger_precondition_ok:
            blocker = "FINGER_CONTROL_BLOCKED"
        elif not ok and active_force_peak > threshold and count < required_count:
            blocker = "TARGET_FILTERED_FORCE_NOT_OBSERVED"
        rows.append(
            {
                "part_name": slot.part_name,
                "env_index": slot.global_env_index,
                "local_env_index": slot.local_env_index,
                "contact_gate_mode": mode,
                "contact_acquisition_primitive": "deterministic_target_filtered_v95",
                "contact_acquisition_status": state.get("blocker", ""),
                "forced_contact_status": forced_contact_status,
                "ordinary_acquisition_executed_after_forced_ok": bool(
                    ordinary_acquisition_executed_after_forced_ok and slot.part_name == "Screw1"
                ),
                "contact_acquired_phase": state.get("acquired_phase", ""),
                "target_filter_preflight_ok": bool(state.get("preflight", {}).get("target_filter_preflight_ok")),
                "target_filter_preflight_blocker": state.get("preflight", {}).get("target_filter_preflight_blocker", ""),
                "selected_finger_group": str(_find_slot_row(plan_rows, slot).get("active_finger_group") or ""),
                "commanded_logical_fingers": commanded_group,
                "commanded_logical_finger_indices": commanded_indices,
                "commanded_policy_cols": _policy_cols_for_group(commanded_group),
                "commanded_isaac_cols": _isaac_cols_for_group(commanded_group),
                "gate_active_sensor_indices": active_indices,
                "nonzero_force_sensor_indices": [
                    index for index, value in enumerate(unfiltered_per_finger) if value > threshold
                ],
                "finger_sensor_mapping_status": "calibrated" if state.get("mapping_valid") else "FINGER_ACTION_SENSOR_MAPPING_UNCALIBRATED",
                "wrist_precondition_ok": bool(state.get("wrist_ready")),
                "finger_motion_sensor_precondition_ok": finger_precondition_ok,
                "contact_close_commanded_fingers_moved": contact_commanded_fingers_moved,
                "contact_active_close_target_delta_l2": contact_active_target_delta,
                "contact_active_close_joint_delta_l2": contact_active_joint_delta,
                "contact_active_fingertip_delta_max_m": contact_active_tip_delta,
                "contact_non_active_close_joint_delta_l2": contact_non_active_joint_delta,
                "contact_non_active_fingertip_leakage_max_m": contact_non_active_tip_delta,
                "approach_direction_world_xyz": state.get("target_direction_xyz", []),
                "approach_target_progress_peak_m": state.get("target_progress_peak_m", 0.0),
                "initial_active_tip_target_center_distance_m": state.get("initial_center_distance_m", 0.0),
                "best_active_tip_target_center_distance_m": state.get("best_center_distance_m", 0.0),
                "preclose_fingertip_object_distance_min_m": preclose_distance,
                "force_peak_n": active_force_peak,
                "overall_force_peak_n": state.get("unfiltered_force_peak_n", 0.0),
                "target_object_contact_force_peak_n": active_force_peak,
                "target_filtered_force_available": target_filtered_available,
                "force_contact_threshold_n": threshold,
                "contact_sensor_api_available": bool(metrics.get("contact_sensor_api_available")),
                "active_force_count": count,
                "active_target_filtered_force_count": count,
                "required_active_force_count": required_count,
                "per_finger_force_norm": max_per_finger,
                "per_finger_unfiltered_force_norm": unfiltered_per_finger,
                "per_finger_target_filtered_force_norm": max_per_finger,
                "summary_per_finger_force_norm": _v92_force_list(metrics),
                "summary_per_finger_target_filtered_force_norm": _v95_target_force_list(metrics),
                "max_active_finger_force_norm": active_force_peak,
                "max_non_active_finger_force_norm": state.get("non_active_target_force_peak_n", 0.0),
                "max_active_unfiltered_finger_force_norm": state.get("active_unfiltered_force_peak_n", 0.0),
                "max_non_active_unfiltered_finger_force_norm": state.get("non_active_unfiltered_force_peak_n", 0.0),
                "responding_sensor_index": _v92_max_index(max_per_finger),
                "responding_unfiltered_sensor_index": _v92_max_index(unfiltered_per_finger),
                "nonzero_target_filtered_sensor_indices": [
                    index for index, value in enumerate(max_per_finger) if value > threshold
                ],
                "object_displacement_m": metrics.get("object_displacement_m", 0.0),
                "target_object_response_delta_m": state.get("target_response_delta_m", 0.0),
                "target_object_response_velocity_norm": state.get("target_response_velocity_norm", 0.0),
                "target_object_response_observed": bool(state.get("target_response_observed")),
                "target_object_contact_evidence_source": evidence_source,
                "target_object_contact_evidence_sufficient": target_evidence,
                "distance_evidence_trusted": not distance_untrusted,
                "sensor_action_mapping_mismatch": sensor_mismatch,
                "unfiltered_contact_not_target_object": unfiltered_non_target_contact,
                "excessive_force_rate": 1.0 if bool(metrics.get("excessive_force")) else 0.0,
                "table_collision_excluded": not bool(state.get("table_collision")),
                "reset_interpenetration_detected": False,
                "actual_object_force_contact": active_force_peak > threshold,
                "actual_object_contact_ok": ok,
                "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
                "distance_only_success_used": False,
                "non_active_sensor_success_used": False,
                "fallback_success_used": False,
                "native_collision_proxy_used_as_pass_evidence": False,
                "native_collision_trusted": bool(ok and not _env_enabled("WUJI_V93_COLLISION_REPAIR")),
                "live_probe_executed": True,
                "native_shutdown": False,
                "blocker": blocker,
            }
        )
    return rows, trace_rows


def _finger_motion_ok_by_part(finger_rows: list[dict[str, Any]]) -> dict[str, dict[int, bool]]:
    out: dict[str, dict[int, bool]] = {}
    for row in finger_rows:
        part = str(row.get("part_name") or "")
        logical = _int_field(row, "logical_finger_id", 0) - 1
        if not part or not (0 <= logical < 5):
            continue
        out.setdefault(part, {})[logical] = bool(row.get("finger_motion_ok"))
    return out


def _contact_mode_group(plan_rows: list[dict[str, Any]], slot: SingleContextSlot, mode: str) -> str:
    group = str(_find_slot_row(plan_rows, slot).get("active_finger_group") or V85_FINGER_GROUPS.get(slot.part_name, ("34",))[0])
    indices = _v91_active_finger_indices(group)
    if mode == "single_finger" and indices:
        return str(indices[0] + 1)
    return group


def _target_filter_preflight_for_slot(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    names = _string_list(metrics.get("target_filter_names"))
    target_forces = _v95_target_force_list(metrics)
    try:
        expected_index = names.index(slot.part_name)
    except ValueError:
        expected_index = -1
    reported_index = _int_field(metrics, "target_filter_index", -1)
    root_path, collision_count, collision_blocker = _stage_target_collision_summary(slot)
    available = bool(metrics.get("target_filtered_force_available"))
    source = str(metrics.get("target_contact_evidence_source") or "")
    force_matrix_ok = bool(available and source == "ContactSensor.force_matrix_w")
    filter_names_ok = bool(expected_index >= 0 and reported_index == expected_index)
    filter_shape_ok = bool(len(target_forces) >= 5 and len(names) >= len(V83_PARTS))
    root_ok = bool(root_path and collision_count > 0 and not collision_blocker)
    blocker = ""
    if not force_matrix_ok:
        blocker = "force_matrix_w_unavailable"
    elif not filter_names_ok:
        blocker = "target_filter_index_or_name_mismatch"
    elif not filter_shape_ok:
        blocker = "target_filter_shape_invalid"
    elif not root_ok:
        blocker = collision_blocker or "target_root_or_collision_missing"
    return {
        "target_filter_preflight_ok": bool(force_matrix_ok and filter_names_ok and filter_shape_ok and root_ok),
        "target_filter_preflight_blocker": blocker,
        "target_filtered_force_available": available,
        "target_contact_evidence_source": source,
        "target_filter_names": names,
        "target_filter_index": reported_index,
        "expected_target_filter_index": expected_index,
        "target_filter_shape_ok": filter_shape_ok,
        "target_stage_root_path": root_path,
        "target_collision_enabled_prim_count": collision_count,
        "target_collision_blocker": collision_blocker,
    }


def _stage_target_collision_summary(slot: SingleContextSlot) -> tuple[str, int, str]:
    try:
        import omni.usd  # noqa: WPS433
        from pxr import Usd, UsdPhysics  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return "", 0, "stage_unavailable"
        root_path = f"/World/envs/env_{slot.local_env_index}/{slot.part_name}"
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            root = find_prim_by_suffix_for_env(stage, f"/{slot.part_name}", slot.local_env_index)
            root_path = str(root.GetPath()) if root and root.IsValid() else ""
        if not root or not root.IsValid():
            return root_path, 0, "target_root_prim_missing"
        count = 0
        for prim in Usd.PrimRange(root):
            collision_api = UsdPhysics.CollisionAPI(prim)
            try:
                if collision_api and bool(collision_api.GetCollisionEnabledAttr().Get()):
                    count += 1
            except Exception:
                continue
        return str(root.GetPath()), count, "" if count > 0 else "target_collision_enabled_prim_missing"
    except Exception as exc:
        return "", 0, f"{type(exc).__name__}:{exc}"


def _approach_vector_for_slot(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_indices: list[int],
) -> dict[str, Any]:
    tip = _active_group_midpoint_world(backend, slot, logical_indices)
    obj = _active_object_root_pos_world(backend, slot)
    if len(tip) < 3 or len(obj) < 3:
        return {"valid": False, "direction": [0.0, 0.0, 0.0], "blocker": "tip_or_object_physx_pose_unavailable"}
    delta = _vec_sub(obj, tip)
    norm = _norm(delta)
    if norm <= 1.0e-6:
        return {"valid": False, "direction": [0.0, 0.0, 0.0], "blocker": "tip_already_at_object_center_direction_degenerate"}
    return {"valid": True, "direction": [value / norm for value in delta], "distance_m": norm, "blocker": ""}


def _target_center_distance_for_slot(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_indices: list[int],
) -> float:
    tip = _active_group_midpoint_world(backend, slot, logical_indices)
    obj = _active_object_root_pos_world(backend, slot)
    if len(tip) < 3 or len(obj) < 3:
        return float("inf")
    return _norm(_vec_sub(obj, tip))


def _active_group_midpoint_world(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_indices: list[int],
) -> list[float]:
    tips = _v95_tip_positions_world(backend)
    points: list[list[float]] = []
    for index in logical_indices:
        point = _matrix_vec(tips, slot.local_env_index, index)
        if len(point) >= 3:
            points.append(point[:3])
    if not points:
        return []
    return [
        sum(point[axis] for point in points) / float(len(points))
        for axis in range(3)
    ]


def _active_object_root_pos_world(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> list[float]:
    state = _v91_object_state_for_slot(backend, slot)
    pos = state.get("root_pos_w", [])
    try:
        return [float(pos[index]) for index in range(3)]
    except Exception:
        return []


def _matrix_vec(value: Any, row_index: int, col_index: int) -> list[float]:
    try:
        row = value[int(row_index), int(col_index)]
        if hasattr(row, "detach"):
            row = row.detach().cpu().reshape(-1).tolist()
        return [float(item) for item in list(row)[:3]]
    except Exception:
        return []


def _update_contact_acquisition_state(
    backend: IsaacUnifiedSingleContextBackend,
    state: dict[str, Any],
    metrics: dict[str, Any],
    before_state: dict[str, Any],
) -> None:
    threshold = float(backend.contact_manager.force_threshold_n)
    mapped = list(state.get("mapped_indices") or [])
    target_forces = _v95_target_force_list(metrics)
    unfiltered_forces = _v92_force_list(metrics)
    state["per_finger_target_peak"] = _max_force_lists(state.get("per_finger_target_peak", []), target_forces)
    state["per_finger_unfiltered_peak"] = _max_force_lists(state.get("per_finger_unfiltered_peak", []), unfiltered_forces)
    active_target = _force_peak_for_indices(target_forces, mapped)
    active_unfiltered = _force_peak_for_indices(unfiltered_forces, mapped)
    non_active_target = _force_peak_except_indices(target_forces, mapped)
    non_active_unfiltered = _force_peak_except_indices(unfiltered_forces, mapped)
    state["active_target_force_peak_n"] = max(_to_float(state.get("active_target_force_peak_n")), active_target)
    state["active_unfiltered_force_peak_n"] = max(_to_float(state.get("active_unfiltered_force_peak_n")), active_unfiltered)
    state["unfiltered_force_peak_n"] = max(_to_float(state.get("unfiltered_force_peak_n")), max([0.0, *unfiltered_forces]))
    state["non_active_target_force_peak_n"] = max(_to_float(state.get("non_active_target_force_peak_n")), non_active_target)
    state["non_active_unfiltered_force_peak_n"] = max(_to_float(state.get("non_active_unfiltered_force_peak_n")), non_active_unfiltered)
    state["active_target_filtered_force_count"] = max(
        int(state.get("active_target_filtered_force_count") or 0),
        sum(1 for index in mapped if 0 <= index < len(target_forces) and target_forces[index] > threshold),
    )
    state["target_filtered_available"] = bool(state.get("target_filtered_available") or metrics.get("target_filtered_force_available"))
    state["contact_sensor_api_available"] = bool(state.get("contact_sensor_api_available") or metrics.get("contact_sensor_api_available"))
    state_delta = _v91_state_delta(before_state, _v91_object_state_for_slot(backend, state["slot"]))
    response_delta = _to_float(state_delta.get("root_pose_delta_m"))
    response_velocity = _to_float(metrics.get("object_velocity_norm"))
    state["target_response_delta_m"] = max(_to_float(state.get("target_response_delta_m")), response_delta)
    state["target_response_velocity_norm"] = max(_to_float(state.get("target_response_velocity_norm")), response_velocity)
    state["target_response_observed"] = bool(
        state.get("target_response_observed")
        or response_delta >= 1.0e-5
        or response_velocity >= 1.0e-4
    )
    state["table_collision"] = bool(state.get("table_collision") or metrics.get("table_collision"))
    state["final_metrics"] = metrics


def _max_force_lists(left: Any, right: Any) -> list[float]:
    out = [0.0] * 5
    try:
        left_values = [float(value) for value in list(left)[:5]]
    except Exception:
        left_values = []
    try:
        right_values = [float(value) for value in list(right)[:5]]
    except Exception:
        right_values = []
    for index in range(5):
        out[index] = max(
            left_values[index] if index < len(left_values) else 0.0,
            right_values[index] if index < len(right_values) else 0.0,
        )
    return out


def _force_peak_for_indices(values: list[float], indices: list[int]) -> float:
    return max([0.0, *[_to_float(values[index]) for index in indices if 0 <= index < len(values)]])


def _force_peak_except_indices(values: list[float], indices: list[int]) -> float:
    active = set(int(index) for index in indices)
    return max([0.0, *[_to_float(value) for index, value in enumerate(values[:5]) if index not in active]])


def _final_acquisition_blocker(backend: IsaacUnifiedSingleContextBackend, state: dict[str, Any]) -> str:
    threshold = float(backend.contact_manager.force_threshold_n)
    if _to_float(state.get("active_target_force_peak_n")) > threshold:
        return "TARGET_CONTACT_ACQUIRED"
    if _to_float(state.get("unfiltered_force_peak_n")) > threshold:
        return "NON_TARGET_CONTACT_BEFORE_TARGET"
    if not bool(state.get("preflight", {}).get("target_filter_preflight_ok")):
        return "TARGET_FILTER_CONFIG_INVALID"
    if not bool(state.get("wrist_ready")) or bool(state.get("workspace_or_clamp_blocked")):
        return "WRIST_OR_WORKSPACE_BLOCKED"
    if not bool(state.get("target_direction_valid")) or _to_float(state.get("target_progress_peak_m")) <= 1.0e-5:
        return "APPROACH_DIRECTION_NOT_TOWARD_TARGET"
    if not bool(state.get("finger_ready")):
        return "FINGER_CONTROL_BLOCKED"
    if bool(state.get("close_executed")):
        return "CLOSE_REACH_EXHAUSTED_NO_TARGET_FORCE"
    return "TARGET_FILTERED_FORCE_NOT_OBSERVED"


def _contact_acquisition_trace_row(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    *,
    mode: str,
    phase: str,
    step: int,
    state: dict[str, Any],
    metrics: dict[str, Any],
    action_xyz: list[float],
    close_value: float,
    blocker: str,
) -> dict[str, Any]:
    current_distance = _target_center_distance_for_slot(backend, slot, state.get("logical_indices", []))
    target_forces = _v95_target_force_list(metrics)
    unfiltered_forces = _v92_force_list(metrics)
    active = list(state.get("mapped_indices") or [])
    active_target = _force_peak_for_indices(target_forces, active)
    unfiltered_peak = max([0.0, *unfiltered_forces])
    return {
        "part_name": slot.part_name,
        "env_index": slot.global_env_index,
        "local_env_index": slot.local_env_index,
        "contact_gate_mode": mode,
        "phase": phase,
        "step": int(step),
        "forced_contact_status": state.get("forced_contact_status", ""),
        "ordinary_acquisition_executed_after_forced_ok": bool(
            state.get("ordinary_acquisition_executed_after_forced_ok")
        ),
        "allowed_for_ordinary_acquisition": bool(state.get("allowed_for_ordinary_acquisition", True)),
        "commanded_group": state.get("group", ""),
        "commanded_logical_finger_indices": state.get("logical_indices", []),
        "active_sensor_indices": active,
        "target_filter_preflight_ok": bool(state.get("preflight", {}).get("target_filter_preflight_ok")),
        "target_filter_preflight_blocker": state.get("preflight", {}).get("target_filter_preflight_blocker", ""),
        "target_stage_root_path": state.get("preflight", {}).get("target_stage_root_path", ""),
        "target_collision_enabled_prim_count": state.get("preflight", {}).get("target_collision_enabled_prim_count", 0),
        "target_filter_names": state.get("preflight", {}).get("target_filter_names", []),
        "target_filter_index": state.get("preflight", {}).get("target_filter_index", -1),
        "target_direction_world_xyz": state.get("target_direction_xyz", []),
        "target_direction_valid": bool(state.get("target_direction_valid")),
        "policy_wrist_xyz": action_xyz[:3],
        "finger_close_value": float(close_value),
        "initial_center_distance_m": state.get("initial_center_distance_m", 0.0),
        "current_center_distance_m": current_distance,
        "target_progress_peak_m": state.get("target_progress_peak_m", 0.0),
        "active_target_filtered_force_step_n": active_target,
        "active_target_filtered_force_peak_n": state.get("active_target_force_peak_n", 0.0),
        "unfiltered_force_step_peak_n": unfiltered_peak,
        "unfiltered_force_peak_n": state.get("unfiltered_force_peak_n", 0.0),
        "non_active_target_filtered_force_peak_n": state.get("non_active_target_force_peak_n", 0.0),
        "per_finger_target_filtered_force_norm": target_forces,
        "per_finger_unfiltered_force_norm": unfiltered_forces,
        "active_target_filtered_force_count": state.get("active_target_filtered_force_count", 0),
        "target_filtered_force_available": bool(metrics.get("target_filtered_force_available")),
        "target_contact_evidence_source": metrics.get("target_contact_evidence_source", ""),
        "object_response_delta_m": state.get("target_response_delta_m", 0.0),
        "object_velocity_norm": metrics.get("object_velocity_norm", 0.0),
        "active_object_response_observed": bool(state.get("target_response_observed")),
        "table_collision": bool(metrics.get("table_collision")),
        "workspace_or_clamp_blocked": bool(state.get("workspace_or_clamp_blocked")),
        "object_write_by_policy_detected": bool(backend.object_write_by_policy_detected),
        "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
        "proxy_success_used": False,
        "distance_only_success_used": False,
        "non_active_sensor_success_used": False,
        "blocker": blocker or state.get("blocker", ""),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    try:
        return [str(item) for item in list(value)]
    except Exception:
        return []


def _move_v95_reference_pad(
    slot: SingleContextSlot,
    logical_finger: int,
    center_local_xyz: list[float],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "reference_pad_prim_path": "",
        "reference_pad_center_local_xyz": [],
        "reference_pad_collision_enabled": False,
        "reference_pad_contact_report_api": False,
        "reference_pad_half_extent_xyz": [],
        "reference_pad_blocker": "",
    }
    if len(center_local_xyz) < 3:
        out["reference_pad_blocker"] = "center_local_xyz_unavailable"
        return out
    try:
        import omni.usd  # noqa: WPS433
        from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            out["reference_pad_blocker"] = "stage_unavailable"
            return out
        path = f"/World/envs/env_{slot.local_env_index}/v95_reference_pad_f{int(logical_finger)}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            prim = cube.GetPrim()
            UsdGeom.Imageable(prim).MakeInvisible()
        xform_api = UsdGeom.XformCommonAPI(prim)
        center = [float(center_local_xyz[0]), float(center_local_xyz[1]), float(center_local_xyz[2])]
        half_extent = [0.003, 0.003, 0.003]
        xform_api.SetTranslate(tuple(center))
        xform_api.SetScale(tuple(value * 2.0 for value in half_extent))
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision_api.CreateContactOffsetAttr().Set(0.001)
        physx_collision_api.CreateRestOffsetAttr().Set(0.0)
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        out.update(
            {
                "reference_pad_prim_path": path,
                "reference_pad_center_local_xyz": center,
                "reference_pad_collision_enabled": True,
                "reference_pad_contact_report_api": True,
                "reference_pad_half_extent_xyz": half_extent,
            }
        )
    except Exception as exc:
        out["reference_pad_blocker"] = f"{type(exc).__name__}:{exc}"
    return out


def _park_v95_reference_pad(slot: SingleContextSlot, logical_finger: int) -> None:
    _move_v95_reference_pad(
        slot,
        logical_finger,
        [1.45 + 0.03 * float(max(0, int(logical_finger) - 1)), 1.45, 1.25],
    )


def _v95_reference_force_list(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> list[float]:
    base = _base_env(backend)
    return _tensor_list(getattr(base, "dex_fingertip_reference_force_norm", None), slot.local_env_index, width=5)


def _v95_screw1_mirror_force_list(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> list[float]:
    base = _base_env(backend)
    return _tensor_list(getattr(base, "v95_screw1_mirror_force_norm", None), slot.local_env_index, width=5)


def _v95_screw1_mirror_available(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> bool:
    base = _base_env(backend)
    return _tensor_float(getattr(base, "v95_screw1_mirror_force_valid", None), slot.local_env_index) > 0.5


def _progress_rows(
    *,
    parts: list[str],
    backend: IsaacUnifiedSingleContextBackend,
    active_rows: list[dict[str, Any]],
    reset_lifecycle_rows: list[dict[str, Any]],
    pregrasp_rows: list[dict[str, Any]],
    wrist_rows: list[dict[str, Any]],
    finger_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    pregrasp_by_part: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        active = _find_part_row(active_rows, part)
        reset = _find_part_row(reset_lifecycle_rows, part)
        pre = _find_part_row(pregrasp_rows, part)
        part_wrist = [row for row in wrist_rows if row.get("part_name") == part]
        part_fingers = [row for row in finger_rows if row.get("part_name") == part]
        part_contact = [row for row in contact_rows if row.get("part_name") == part]
        single = next((row for row in part_contact if row.get("contact_gate_mode") == "single_finger"), {})
        multi = next((row for row in part_contact if row.get("contact_gate_mode") == "two_multi_finger"), {})
        wrist_ok = bool(part_wrist) and all(_bool(row.get("wrist_action_mapping_ok")) for row in part_wrist)
        finger_ok = bool(part_fingers) and all(_bool(row.get("finger_motion_ok")) for row in part_fingers)
        single_ok = _bool(single.get("actual_object_contact_ok"))
        multi_ok = _bool(multi.get("actual_object_contact_ok"))
        native_collision = bool(single_ok and multi_ok and not _env_enabled("WUJI_V93_COLLISION_REPAIR"))
        identity_ok = _bool(active.get("object_identity_verified"))
        active_ok = _bool(active.get("active_object_state_matches_slot"))
        critical_path_ok = bool(
            _bool(active.get("v95_critical_path_uses_active_object_state"))
            and _bool(active.get("v95_derived_state_refresh_complete"))
        )
        legacy_ok = _bool(active.get("legacy_held_state_matches_active_object"))
        plan_ok = _bool(pre.get("pregrasp_plan_applied"))
        pre_ok = _bool(pre.get("pregrasp_application_ok"))
        live_ok = bool(
            part_wrist
            and part_fingers
            and part_contact
            and all(_bool(row.get("live_probe_executed")) for row in part_wrist + part_fingers + part_contact)
        )
        reset_ok = _bool(reset.get("reset_lifecycle_ok"))
        no_forbidden = bool(
            not backend.object_write_by_policy_detected
            and not backend.sticky_action_available_to_policy
            and not backend.proxy_action_available_to_policy
            and not backend.route_selection_available_to_policy
        )
        if not identity_ok or not active_ok:
            decision = "ACTIVE_OBJECT_RESET_BLOCKER"
            blocker = "active_object_identity_or_slot_state_failed"
        elif not critical_path_ok:
            decision = "LEGACY_PLUG2_RESET_PATH_BLOCKER"
            blocker = "v95_critical_path_active_object_refresh_incomplete"
        elif not plan_ok:
            decision = "PREGRASP_PLAN_NOT_APPLIED"
            blocker = "v95_reset_pregrasp_plan_missing_or_not_applied"
        elif not reset_ok or not pre_ok or not live_ok or not no_forbidden:
            decision = "NATIVE_SHUTDOWN_OR_RESET_LIFECYCLE_BLOCKER"
            blocker = pre.get("pregrasp_application_blocker") or reset.get("reset_lifecycle_ok", "reset_lifecycle_failed")
        elif not wrist_ok:
            decision = "WRIST_ACTION_MAPPING_BLOCKER"
            blocker = _first_blocker(part_wrist, "blocker", "wrist_action_mapping_failed")
        elif not finger_ok:
            decision = "FINGER_MOTION_BLOCKER"
            blocker = _first_blocker(part_fingers, "finger_motion_blocker", "finger_motion_failed")
        elif not single_ok or not multi_ok:
            decision = "ACTUAL_OBJECT_CONTACT_BLOCKER"
            blocker = _first_blocker(part_contact, "blocker", "actual_object_force_contact_failed")
        elif not native_collision:
            decision = "NATIVE_COLLISION_ASSET_BLOCKER"
            blocker = "native_collision_not_trusted_or_v93_runtime_proxy_enabled"
        else:
            decision = "WORKCELL_READY_FOR_FEASIBILITY_CONTROLLER"
            blocker = ""
        ready = decision == "WORKCELL_READY_FOR_FEASIBILITY_CONTROLLER"
        rows.append(
            {
                "part_name": part,
                "v95_decision": decision,
                "ready_for_feasibility_controller": ready,
                "object_identity_verified": identity_ok,
                "single_simulation_context": bool(backend.single_simulation_context),
                "gym_make_count": int(backend.gym_make_count),
                "active_object_state_matches_slot": active_ok,
                "legacy_held_state_matches_active_object": legacy_ok,
                "legacy_global_held_asset_is_plug2": _bool(active.get("legacy_global_held_asset_is_plug2")),
                "v95_critical_path_uses_active_object_state": _bool(active.get("v95_critical_path_uses_active_object_state")),
                "v95_derived_state_refresh_complete": _bool(active.get("v95_derived_state_refresh_complete")),
                "v95_derived_state_refresh_fields": active.get("v95_derived_state_refresh_fields", ""),
                "pregrasp_plan_applied": plan_ok,
                "live_probe_executed": live_ok,
                "native_shutdown": False,
                "native_collision_trusted": native_collision,
                "fingertip_object_preclose_distance_min_m": pre.get("fingertip_object_surface_distance_min_m", ""),
                "pre_contact_displacement_m": pre.get("pre_contact_displacement_m", ""),
                "pre_contact_z_drift_m": pre.get("pre_contact_z_drift_m", ""),
                "initial_force_peak_n": pre.get("initial_force_peak_n", ""),
                "wrist_axis_all_valid": wrist_ok,
                "finger_motion_all_valid": finger_ok,
                "actual_object_single_finger_force_contact": single_ok,
                "actual_object_two_multi_finger_force_contact": multi_ok,
                "single_finger_force_peak_n": single.get("force_peak_n", 0.0),
                "two_multi_finger_force_peak_n": multi.get("force_peak_n", 0.0),
                "object_write_after_reset_detected": bool(backend.object_write_by_policy_detected),
                "hand_write_after_reset_detected": False,
                "sticky_action_available_to_policy": bool(backend.sticky_action_available_to_policy),
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
                "training_locked": not ready,
                "status": "PASS" if ready else "BLOCKED",
                "blocker": blocker,
                "next_action": "unlock_v95_feasibility_controller_inputs" if ready else _next_action(decision),
                "pregrasp_source": pregrasp_by_part.get(part, {}).get("hand_target_source", ""),
            }
        )
    return rows


def _reset_with_plan(
    backend: IsaacUnifiedSingleContextBackend,
    plan_rows: list[dict[str, Any]],
    *,
    settle_steps: int = 2,
) -> None:
    _assert_v95_plan_valid(backend, plan_rows, phase="live_probe_reset")
    _disable_v95_action_control(backend)
    backend.configure_v95_pregrasp(plan_rows)
    backend.object_write_by_policy_detected = False
    backend.reset_envs()
    backend._capture_v86_reset_object_positions()
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    reset_rows = [dict(row) for row in getattr(base, "v95_last_pregrasp_rows", []) or []]
    hand_rows = [dict(row) for row in getattr(base, "v95_last_hand_reset_rows", []) or []]
    _assert_v95_reset_applied(backend, plan_rows, reset_rows, hand_rows, phase="live_probe_reset")
    _enable_v95_action_control(backend)
    _zero_settle(backend, steps=settle_steps)


def _zero_settle(backend: IsaacUnifiedSingleContextBackend, *, steps: int) -> None:
    for step in range(steps):
        _set_context(backend, "v95_zero_step_settle", step=step)
        backend.step_envs([[0.0] * 16 for _slot in backend.slots])


def _set_context(
    backend: IsaacUnifiedSingleContextBackend,
    phase: str,
    *,
    axis: str = "",
    sign: int = 0,
    finger: int = 0,
    step: int = 0,
) -> None:
    context: dict[int, dict[str, Any]] = {}
    for slot in backend.slots:
        row: dict[str, Any] = {
            "v95_minimal_workcell_mode": True,
            "v95_calibration_mode": phase,
            "nominal_phase": phase,
            "v95_phase_step": int(step),
            "object_write_after_reset_allowed": False,
            "hand_write_after_reset_allowed": False,
            "fallback_success_used": False,
            "distance_only_success_used": False,
            "sticky_action_available_to_policy": False,
            "route_selection_available_to_policy": False,
            "proxy_action_available_to_policy": False,
        }
        if axis:
            row["calibration_axis"] = axis
            row["calibration_sign"] = int(sign)
        if finger:
            row["logical_finger_id"] = int(finger)
            row["commanded_fingers"] = str(finger)
            row["commanded_policy_cols"] = f"{6 + finger - 1},{11 + finger - 1}"
            row["commanded_isaac_cols"] = f"{16 + finger - 1},{21 + finger - 1}"
        context[slot.global_env_index] = row
    backend._v87_pending_action_context = context


def _v95_support_half_height_m(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot) -> tuple[float, str]:
    try:
        half, source = _v94_support_half_height_m(backend, slot)
        if half > 0.0 and source:
            return half, f"v95_{source}"
    except Exception:
        pass
    return _v92_support_half_height_m(slot.part_name), "V85_PART_GEOMETRY_minimal_v95_fallback"


def _slot_metric_rows(backend: IsaacUnifiedSingleContextBackend, phase: str) -> dict[str, dict[str, Any]]:
    return {slot.part_name: backend._metrics_for_slot(slot, phase) for slot in backend.slots}


def _vec_sub(left: list[float], right: list[float]) -> list[float]:
    return [
        (float(left[index]) if index < len(left) else 0.0) - (float(right[index]) if index < len(right) else 0.0)
        for index in range(3)
    ]


def _safe_no_contact_hand_target(row: dict[str, Any]) -> list[float]:
    center = _row_center(row)
    half = _to_float(row.get("object_support_half_height_m"), 0.02)
    offset_xy = max(0.12, min(0.145, 0.12 + half * 0.04))
    z_offset = max(0.10, min(0.18, 0.09 + half * 0.40))
    return [
        center[0] + offset_xy,
        center[1] - offset_xy,
        center[2] + z_offset,
    ]


def _slot_by_part(backend: IsaacUnifiedSingleContextBackend, part_name: str) -> SingleContextSlot | None:
    for slot in backend.slots:
        if slot.part_name == part_name:
            return slot
    return None


def _v95_tip_positions_world(backend: IsaacUnifiedSingleContextBackend) -> Any:
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    try:
        if getattr(base, "last_update_timestamp", 0.0) < getattr(base._robot._data, "_sim_timestamp", 0.0):
            base._compute_intermediate_values(dt=base.physics_dt)
        indices = list(getattr(base, "dex_fingertip_true_body_indices", []) or [])
        if not indices:
            return None
        return base._robot.data.body_pos_w[:, indices, :].detach().clone()
    except Exception:
        return None


def _screw1_stage_collision_row(slot: SingleContextSlot, base: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage_root_path": "",
        "stage_root_valid": False,
        "collision_enabled_prim_count": 0,
        "collision_prim_paths": "",
        "collision_approximation_types": "",
        "contact_report_api_prim_count": 0,
        "visual_mesh_prim_count": 0,
        "visual_bbox_center_xyz": [],
        "visual_bbox_extent_xyz": [],
        "collision_bbox_center_xyz": [],
        "collision_bbox_extent_xyz": [],
        "signed_helper_bbox_center_xyz": [],
        "signed_helper_bbox_extent_xyz": [],
        "visual_collision_center_offset_m": 0.0,
        "visual_collision_extent_error_max": 1.0,
        "distance_helper_collision_center_offset_m": 1.0,
        "distance_helper_collision_extent_error_max": 1.0,
        "distance_helper_collision_aligned": False,
        "visual_collision_aligned": False,
        "contact_report_api_missing_on_screw1": False,
        "v95_contact_chain_blocker": "",
        "expected_collision_extent_error_max": 1.0,
        "collision_blocker": "",
        "diagnostic_runtime_frame_advisory": "",
    }
    try:
        import omni.usd  # noqa: WPS433
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            row["collision_blocker"] = "stage_unavailable"
            return row
        candidates = [f"/World/envs/env_{slot.local_env_index}/Screw1"]
        root = None
        for path in candidates:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                root = prim
                row["stage_root_path"] = path
                break
        if root is None:
            row["collision_blocker"] = "screw1_root_prim_missing"
            return row
        row["stage_root_valid"] = True
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        visual_prims = []
        collision_prims = []
        approximations = []
        contact_reports = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) and "collision" not in path.lower():
                visual_prims.append(prim)
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if collision_enabled:
                collision_prims.append(prim)
                physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
                approx_attr = (
                    physx_collision.GetCollisionApproximationAttr()
                    if hasattr(physx_collision, "GetCollisionApproximationAttr")
                    else prim.GetAttribute("physxCollision:collisionApproximation")
                )
                approx = approx_attr.Get() if approx_attr else ""
                if approx:
                    approximations.append(str(approx))
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                contact_reports.append(path)
        row["visual_mesh_prim_count"] = len(visual_prims)
        row["collision_enabled_prim_count"] = len(collision_prims)
        row["collision_prim_paths"] = ";".join(str(prim.GetPath()) for prim in collision_prims[:20])
        row["collision_approximation_types"] = ";".join(sorted(set(approximations)))
        row["contact_report_api_prim_count"] = len(contact_reports)
        visual_box = _bbox_for_prims(cache, visual_prims)
        collision_box = _bbox_for_prims(cache, collision_prims)
        row["visual_bbox_center_xyz"] = visual_box.get("center", [])
        row["visual_bbox_extent_xyz"] = visual_box.get("extent", [])
        row["collision_bbox_center_xyz"] = collision_box.get("center", [])
        row["collision_bbox_extent_xyz"] = collision_box.get("extent", [])
        row["visual_collision_center_offset_m"] = _center_offset(visual_box.get("center", []), collision_box.get("center", []))
        row["visual_collision_extent_error_max"] = _extent_error(visual_box.get("extent", []), collision_box.get("extent", []))
        row["visual_collision_aligned"] = bool(
            row["visual_collision_center_offset_m"] <= 0.005
            and row["visual_collision_extent_error_max"] <= 0.25
        )
        expected = [0.012, 0.012, 0.025]
        geometry = V85_PART_GEOMETRY.get("Screw1", {})
        if geometry:
            expected = [2.0 * _to_float(geometry.get("radius"), 0.006), 2.0 * _to_float(geometry.get("radius"), 0.006), 2.0 * _to_float(geometry.get("half_height"), 0.0125)]
        row["expected_collision_extent_error_max"] = _extent_error(expected, collision_box.get("extent", []))
        helper_box = _screw1_signed_helper_world_bbox(slot, base)
        row["signed_helper_bbox_center_xyz"] = helper_box.get("center", [])
        row["signed_helper_bbox_extent_xyz"] = helper_box.get("extent", [])
        row["distance_helper_collision_center_offset_m"] = _center_offset(helper_box.get("center", []), collision_box.get("center", []))
        row["distance_helper_collision_extent_error_max"] = _extent_error(helper_box.get("extent", []), collision_box.get("extent", []))
        row["distance_helper_collision_aligned"] = bool(
            row["distance_helper_collision_center_offset_m"] <= 0.010
            and row["distance_helper_collision_extent_error_max"] <= 0.35
        )
        row["contact_report_api_missing_on_screw1"] = len(contact_reports) <= 0
        if len(collision_prims) <= 0:
            row["collision_blocker"] = "collision_missing_entirely"
        elif row["expected_collision_extent_error_max"] > 1.25:
            row["collision_blocker"] = "collision_bbox_mismatch_expected_geometry"
        else:
            row["collision_blocker"] = ""
        if row["collision_blocker"]:
            row["v95_contact_chain_blocker"] = row["collision_blocker"]
        elif not row["visual_collision_aligned"]:
            row["v95_contact_chain_blocker"] = "VISUAL_COLLIDER_MISMATCH"
        elif not row["distance_helper_collision_aligned"]:
            row["v95_contact_chain_blocker"] = "DIAGNOSTIC_ENV_OR_FRAME_MISMATCH"
            row["diagnostic_runtime_frame_advisory"] = (
                "runtime signed helper bbox is PhysX-derived advisory; USD static bbox is not runtime contact truth"
            )
        else:
            row["v95_contact_chain_blocker"] = ""
    except Exception as exc:
        row["collision_blocker"] = f"{type(exc).__name__}:{exc}"
    return row


def _screw1_signed_helper_world_bbox(slot: SingleContextSlot, base: Any) -> dict[str, list[float]]:
    out = {"center": [], "extent": []}
    try:
        registry = getattr(base, "v83_active_asset_registry", {}).get("Screw1", {}) if base is not None else {}
        asset = registry.get("asset")
        pos = _tensor_row(getattr(getattr(asset, "data", None), "root_pos_w", None), slot.local_env_index)
        quat = _tensor_row(getattr(getattr(asset, "data", None), "root_quat_w", None), slot.local_env_index)
        root_pos = _vec3(pos)
        root_quat = _tensor_list(getattr(getattr(asset, "data", None), "root_quat_w", None), slot.local_env_index, width=4)
        if len(root_pos) < 3 or len(root_quat) < 4:
            return out
        geometry = V85_PART_GEOMETRY.get("Screw1", {})
        center_local = [float(value) for value in geometry.get("center", (0.0, 0.0, 0.0125))]
        radius = _to_float(geometry.get("radius"), 0.006)
        half_height = _to_float(geometry.get("half_height"), 0.0125)
        center = _vec_add(root_pos, _quat_apply_wxyz(root_quat, center_local))
        axis = _normalize(_quat_apply_wxyz(root_quat, [0.0, 0.0, 1.0]))
        extent = [
            2.0 * (abs(axis[index]) * half_height + radius * math.sqrt(max(0.0, 1.0 - axis[index] * axis[index])))
            for index in range(3)
        ]
        out["center"] = center
        out["extent"] = extent
    except Exception:
        pass
    return out


def _screw1_active_distance_row(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    active_indices: list[int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "screw1_root_pos_w": [],
        "screw1_root_quat_w": [],
        "env_origin_w": [],
        "active_tip_world_xyz": [],
        "active_tip_local_xyz": [],
        "active_tip_distance_xyz": [],
        "closest_surface_point_local_xyz": [],
        "best_gap_vector_local_xyz": [],
        "best_gap_norm_m": 1.0,
        "closest_active_finger_index": -1,
        "min_active_signed_surface_distance_m": 1.0,
        "min_active_abs_surface_distance_m": 1.0,
        "per_active_signed_surface_distance_m": [],
        "per_active_abs_surface_distance_m": [],
        "per_active_distance_candidates": [],
        "distance_frame": "",
        "tip_tensor_shape": "",
        "distance_blocker": "",
    }
    try:
        base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
        tips = _v95_tip_positions_world(backend)
        state = _v91_object_state_for_slot(backend, slot)
        root_pos = [float(value) for value in state.get("root_pos_w", [])[:3]]
        root_quat = [float(value) for value in state.get("root_quat_w", [])[:4]]
        env_origin = _tensor_row(getattr(getattr(base, "scene", None), "env_origins", None), slot.local_env_index)
        origin = _vec3(env_origin) if env_origin is not None else [0.0, 0.0, 0.0]
        row["env_origin_w"] = origin
        tip_rows, tip_shape = _tip_rows_for_env(tips, slot.local_env_index)
        row["tip_tensor_shape"] = tip_shape
        if tips is None:
            row["distance_blocker"] = "tip_tensor_unavailable"
            return row
        if len(root_pos) < 3 or len(root_quat) < 4:
            row["distance_blocker"] = "screw1_root_state_unavailable"
            return row
        if not tip_rows:
            row["screw1_root_pos_w"] = root_pos
            row["screw1_root_quat_w"] = root_quat
            row["distance_blocker"] = "tip_rows_unavailable"
            return row
        geometry = V85_PART_GEOMETRY.get("Screw1", {})
        center_local = [float(value) for value in geometry.get("center", (0.0, 0.0, 0.0125))]
        radius = _to_float(geometry.get("radius"), 0.006)
        half_height = _to_float(geometry.get("half_height"), 0.0125)
        center_world = _vec_add(root_pos, _quat_apply_wxyz(root_quat, center_local))
        axis = _normalize(_quat_apply_wxyz(root_quat, [0.0, 0.0, 1.0]))
        signed_values = []
        abs_values = []
        candidate_rows: list[dict[str, Any]] = []
        best = 1.0
        best_abs = 1.0
        best_tip_world: list[float] = []
        best_tip_local: list[float] = []
        best_tip_distance: list[float] = []
        best_surface_local: list[float] = []
        best_gap_local: list[float] = []
        best_frame = ""
        best_index = -1
        for finger_index in active_indices:
            tip = _tip_from_rows(tip_rows, finger_index)
            if not tip:
                continue
            tip_world = [float(value) for value in tip[:3]]
            tip_local = _vec_sub(tip_world, origin)
            signed = _signed_distance_to_cylinder(tip_world, center_world, axis, radius, half_height)
            surface = _nearest_point_on_cylinder_surface(tip_world, center_world, axis, radius, half_height)
            gap_world = _vec_sub(surface, tip_world)
            surface_local = _vec_sub(surface, origin)
            candidate_rows.append(
                {
                    "finger_index": int(finger_index),
                    "frame": "physx_world_tip_vs_physx_world_screw1_advisory",
                    "signed_surface_distance_m": signed,
                    "abs_surface_distance_m": abs(signed),
                    "tip_xyz": tip_world,
                    "closest_surface_point_xyz": [float(value) for value in surface[:3]],
                    "gap_vector_xyz": [float(value) for value in gap_world[:3]],
                }
            )
            signed = _to_float(signed, 1.0)
            best_for_finger = {
                "signed_surface_distance_m": signed,
                "abs_surface_distance_m": abs(signed),
                "tip_world_xyz": tip_world,
                "tip_local_xyz": tip_local,
                "tip_xyz": tip_world,
                "surface_local_xyz": surface_local,
                "gap_local_xyz": gap_world,
                "frame": "physx_world_tip_vs_physx_world_screw1_advisory",
            }
            signed_values.append(signed)
            abs_values.append(abs(signed))
            if abs(signed) < best_abs:
                best = signed
                best_abs = abs(signed)
                best_tip_world = list(best_for_finger.get("tip_world_xyz", []))
                best_tip_local = list(best_for_finger.get("tip_local_xyz", []))
                best_tip_distance = list(best_for_finger.get("tip_xyz", []))
                best_surface_local = list(best_for_finger.get("surface_local_xyz", []))
                best_gap_local = list(best_for_finger.get("gap_local_xyz", []))
                best_frame = str(best_for_finger.get("frame") or "")
                best_index = finger_index
        if best_index < 0:
            row["distance_blocker"] = "active_tip_indices_unavailable"
        row.update(
            {
                "screw1_root_pos_w": root_pos,
                "screw1_root_quat_w": root_quat,
                "active_tip_world_xyz": best_tip_world,
                "active_tip_local_xyz": best_tip_local,
                "active_tip_distance_xyz": best_tip_distance,
                "closest_surface_point_local_xyz": best_surface_local,
                "best_gap_vector_local_xyz": best_gap_local,
                "best_gap_norm_m": _norm(best_gap_local) if best_gap_local else 1.0,
                "closest_active_finger_index": best_index,
                "min_active_signed_surface_distance_m": best,
                "min_active_abs_surface_distance_m": best_abs,
                "per_active_signed_surface_distance_m": signed_values,
                "per_active_abs_surface_distance_m": abs_values,
                "per_active_distance_candidates": candidate_rows,
                "distance_frame": best_frame,
            }
        )
    except Exception as exc:
        row["distance_blocker"] = f"{type(exc).__name__}:{exc}"
    return row


def _classify_screw1_truth(
    *,
    mode: str,
    group: str,
    logical: list[int],
    mapped: list[int],
    trace_phase: str,
    correction_pass: int,
    wrist_ok: bool,
    mapping_ready: bool,
    initial_signed: float,
    best_signed: float,
    final_signed: float,
    best_abs: float,
    final_abs: float,
    best_step: int,
    best_gap_vector_local_xyz: list[float],
    closest_surface_point_local_xyz: list[float],
    distance_decrease_peak: float,
    active_force_peak: float,
    summary_force_peak: float,
    non_active_force_peak: float,
    active_count_peak: int,
    target_response_observed: bool,
    self_check_status: str,
    asset_ok: bool,
    asset_blocker: str,
    distance_trace_blocker: str,
    threshold: float,
    object_write: bool,
    sticky: bool,
) -> dict[str, Any]:
    distance_toward = distance_decrease_peak > 0.001
    near_or_contact = best_signed <= 0.0015
    crossed_or_penetrated = best_signed <= 0.0
    reach_margin = distance_decrease_peak - max(0.0, initial_signed)
    controller_or_workspace_limited = bool(reach_margin < 0.0 and distance_decrease_peak > 0.001)
    active_force = active_force_peak > threshold
    summary_force = summary_force_peak > threshold
    self_check_pass = self_check_status.startswith("ACTIVE_SENSOR_SELF_CHECK_PASSED")
    actual_ok = bool(
        wrist_ok
        and mapping_ready
        and active_force
        and not object_write
        and not sticky
    )
    if not wrist_ok:
        blocker = "WRIST_ACTION_CALIBRATION_NOT_PASSED"
        classification = "wrist_precondition_failed"
    elif not mapping_ready:
        blocker = "FINGER_ACTION_SENSOR_MAPPING_UNCALIBRATED"
        classification = "sensor_mapping_unready"
    elif object_write:
        blocker = "object_write_after_reset_detected"
        classification = "reset_lifecycle_failed"
    elif sticky:
        blocker = "sticky_available_to_policy"
        classification = "forbidden_sticky_path"
    elif non_active_force_peak > threshold and not active_force:
        blocker = "FINGER_SENSOR_OR_ACTION_MAPPING_MISMATCH"
        classification = "non_active_sensor_force_without_active_force"
    elif active_force and not summary_force:
        blocker = "CONTACT_FORCE_SAMPLING_WINDOW_BLOCKER"
        classification = "per_step_force_peak_summary_missed"
        actual_ok = False
    elif not self_check_pass and not active_force:
        blocker = self_check_status or "ACTIVE_SENSOR_SELF_CHECK_FAILED"
        classification = "active_sensor_self_check_failed"
    elif not active_force:
        blocker = "target_filtered_object_force_contact_not_observed"
        classification = "no_target_filtered_active_force"
    elif active_count_peak < max(1, min(2, len(mapped))):
        blocker = "active_target_filtered_force_count_below_required"
        classification = "insufficient_active_force_sensor_count"
    else:
        blocker = ""
        classification = "target_filtered_active_force_observed"
    return {
        "part_name": "Screw1",
        "trace_phase": trace_phase,
        "correction_pass": correction_pass,
        "contact_gate_mode": mode,
        "selected_finger_group": group,
        "commanded_logical_finger_indices": logical,
        "gate_active_sensor_indices": mapped,
        "preclose_signed_distance_m": initial_signed,
        "close_toward_distance_m": distance_decrease_peak,
        "reach_margin_m": reach_margin,
        "best_step": best_step,
        "best_gap_vector_local_xyz": best_gap_vector_local_xyz,
        "best_gap_norm_m": _norm(best_gap_vector_local_xyz),
        "closest_surface_point_local_xyz": closest_surface_point_local_xyz,
        "controller_or_workspace_limited": controller_or_workspace_limited,
        "initial_signed_surface_distance_m": initial_signed,
        "best_signed_surface_distance_m": best_signed,
        "final_signed_surface_distance_m": final_signed,
        "best_abs_surface_distance_m": best_abs,
        "final_abs_surface_distance_m": final_abs,
        "distance_decrease_peak_m": distance_decrease_peak,
        "active_fingertip_moved_toward_screw1": distance_toward,
        "distance_crossed_or_penetrated": crossed_or_penetrated,
        "active_force_peak_n": active_force_peak,
        "summary_force_peak_n": summary_force_peak,
        "non_active_force_peak_n": non_active_force_peak,
        "active_force_count_peak": active_count_peak,
        "target_response_observed": target_response_observed,
        "self_check_status": self_check_status,
        "asset_audit_ok": asset_ok,
        "asset_contact_chain_blocker": asset_blocker,
        "distance_trace_blocker": distance_trace_blocker,
        "actual_object_contact_ok": actual_ok and not blocker,
        "truth_classification": classification,
        "blocker": blocker,
        "distance_only_success_used": False,
        "fallback_success_used": False,
        "native_collision_proxy_used_as_pass_evidence": False,
        "self_check_used_as_object_contact_success": False,
        "grasp_success_claimed": False,
        "final_success": False,
    }


def _contact_force_for_mode(contact_rows: list[dict[str, Any]], mode: str) -> float:
    for row in contact_rows:
        if row.get("part_name") == "Screw1" and row.get("contact_gate_mode") == mode:
            return _to_float(row.get("target_object_contact_force_peak_n", row.get("force_peak_n")))
    return 0.0


def _v95_target_force_list(metrics: dict[str, Any]) -> list[float]:
    values = metrics.get("per_finger_target_filtered_force_norm") or []
    if isinstance(values, str):
        values = values.strip("[]").split(",")
    try:
        return [float(item) for item in list(values)[:5]]
    except Exception:
        return []


def _summary_for_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    return next((row for row in rows if str(row.get("contact_gate_mode") or "") == mode), {})


def _screw1_should_apply_last_mile_correction(summary: dict[str, Any], backend: IsaacUnifiedSingleContextBackend) -> bool:
    blocker = str(summary.get("blocker") or "")
    if blocker not in {"SCREW1_CLOSE_REACH_INSUFFICIENT", "SCREW1_CLOSE_GEOMETRY_DID_NOT_REACH_OBJECT"}:
        return False
    if str(summary.get("distance_trace_blocker") or ""):
        return False
    if not _bool(summary.get("active_fingertip_moved_toward_screw1")):
        return False
    if _to_float(summary.get("active_force_peak_n")) > backend.contact_manager.force_threshold_n:
        return False
    if _to_float(summary.get("best_signed_surface_distance_m"), 1.0) <= 0.0015:
        return False
    return _norm(list(summary.get("best_gap_vector_local_xyz") or [])) > 1.0e-5


def _screw1_last_mile_adjustment(summary: dict[str, Any]) -> list[float]:
    gap = [float(value) for value in list(summary.get("best_gap_vector_local_xyz") or [])[:3]]
    if len(gap) < 3:
        return [0.0, 0.0, 0.0]
    if _norm(gap) <= 1.0e-8:
        return [0.0, 0.0, 0.0]
    direction = _unit(gap)
    best_signed = _to_float(summary.get("best_signed_surface_distance_m"), 1.0)
    desired_penetration = 0.001
    step = min(0.006, max(0.0, best_signed + desired_penetration))
    return [value * step for value in direction]


def _self_check_status_for_group(rows: list[dict[str, Any]], logical_indices: list[int]) -> str:
    statuses = []
    for finger_index in logical_indices:
        logical = finger_index + 1
        row = next((item for item in rows if _int_field(item, "logical_finger_id", -1) == logical), {})
        statuses.append(str(row.get("self_check_status") or "NOT_EXECUTED"))
    if statuses and all(status.startswith("ACTIVE_SENSOR_SELF_CHECK_PASSED") for status in statuses):
        return "ACTIVE_SENSOR_SELF_CHECK_PASSED"
    if any(status in {"SENSOR_MAPPING_MISMATCH", "FINGER_SENSOR_OR_ACTION_MAPPING_MISMATCH"} for status in statuses):
        return "SENSOR_MAPPING_MISMATCH"
    if any(status == "FINGER_ACTION_SENSOR_MAPPING_UNCALIBRATED" for status in statuses):
        return "FINGER_ACTION_SENSOR_MAPPING_UNCALIBRATED"
    if any(status == "SELF_CHECK_GEOMETRY_DID_NOT_REACH_REFERENCE" for status in statuses):
        return "SELF_CHECK_GEOMETRY_DID_NOT_REACH_REFERENCE"
    if any(status == "ACTIVE_SENSOR_READOUT_FAILED" for status in statuses):
        return "ACTIVE_SENSOR_READOUT_FAILED"
    return "SENSOR_ATTACHMENT_OR_READOUT_FAILED"


def _native_pair_contact_sanity_rows(self_check_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in self_check_rows:
        geometry_reached = _bool(row.get("reference_geometry_reached")) or _bool(row.get("self_check_geometry_reached_reference"))
        active_peak = _to_float(row.get("active_sensor_force_peak_n"))
        non_active_peak = _to_float(row.get("non_active_sensor_force_peak_n"))
        status = str(row.get("native_pair_contact_status") or "")
        rows.append(
            {
                "part_name": row.get("part_name", "Screw1"),
                "logical_finger_id": row.get("logical_finger_id", ""),
                "expected_sensor_index": row.get("expected_sensor_index", ""),
                "reference_body": row.get("reference_pad_prim_path", ""),
                "reference_geometry_reached": geometry_reached,
                "reference_geometry_overlap_observed": _bool(row.get("reference_geometry_overlap_observed")),
                "reference_distance_model_untrusted": _bool(row.get("reference_distance_model_untrusted")),
                "native_pair_contact_observed": status
                in {
                    "PAIR_CONTACT_OBSERVED_BY_ACTIVE_SENSOR",
                    "ACTIVE_SENSOR_FORCE_OBSERVED_REFERENCE_DISTANCE_MODEL_UNTRUSTED",
                },
                "native_pair_contact_status": status or _native_pair_status_from_self_check(
                    reference_geometry_reached=geometry_reached,
                    geometry_overlap_observed=_bool(row.get("reference_geometry_overlap_observed")),
                    active_peak=active_peak,
                    non_active_peak=non_active_peak,
                    threshold=0.05,
                ),
                "active_sensor_force_peak_n": active_peak,
                "non_active_sensor_force_peak_n": non_active_peak,
                "nonzero_force_sensor_indices": row.get("nonzero_force_sensor_indices", []),
                "self_check_status": row.get("self_check_status", ""),
                "object_contact_success_evidence_used": False,
                "native_collision_proxy_used_as_pass_evidence": False,
                "training_row_used": False,
            }
        )
    return rows


def _native_pair_status_from_self_check(
    *,
    reference_geometry_reached: bool,
    geometry_overlap_observed: bool,
    active_peak: float,
    non_active_peak: float,
    threshold: float,
) -> str:
    if not reference_geometry_reached:
        if active_peak > threshold:
            return "ACTIVE_SENSOR_FORCE_OBSERVED_REFERENCE_DISTANCE_MODEL_UNTRUSTED"
        if non_active_peak > threshold:
            return "NON_ACTIVE_SENSOR_FORCE_OBSERVED_REFERENCE_DISTANCE_MODEL_UNTRUSTED"
        return "REFERENCE_GEOMETRY_NOT_REACHED"
    if active_peak > threshold:
        return "PAIR_CONTACT_OBSERVED_BY_ACTIVE_SENSOR"
    if non_active_peak > threshold:
        return "PAIR_CONTACT_OR_MAPPING_MISMATCH_NON_ACTIVE_SENSOR"
    if geometry_overlap_observed:
        return "GEOMETRY_OVERLAP_WITH_ZERO_SENSOR_FORCE"
    return "REFERENCE_REACHED_MARGIN_WITH_ZERO_SENSOR_FORCE"


def _active_tip_local_xyz(backend: IsaacUnifiedSingleContextBackend, slot: SingleContextSlot, finger_index: int) -> list[float]:
    tips = _v92_tip_positions(backend)
    tip_rows, _shape = _tip_rows_for_env(tips, slot.local_env_index)
    return _tip_from_rows(tip_rows, finger_index)


def _ensure_v95_reference_pad(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    logical_finger: int,
    tip_local_xyz: list[float],
    close_dir_local_xyz: list[float],
    close_path_delta_norm: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "reference_pad_prim_path": "",
        "reference_pad_spawned": False,
        "reference_pad_collision_enabled": False,
        "reference_pad_contact_report_api": False,
        "reference_pad_center_local_xyz": [],
        "reference_pad_half_extent_xyz": [],
        "reference_pad_blocker": "",
    }
    if len(tip_local_xyz) < 3:
        out["reference_pad_blocker"] = "tip_local_unavailable"
        return out
    try:
        import omni.usd  # noqa: WPS433
        from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            out["reference_pad_blocker"] = "stage_unavailable"
            return out
        direction = _unit(close_dir_local_xyz if _norm(close_dir_local_xyz) > 1.0e-8 else [0.0, 0.0, -1.0])
        half_extent = [0.006, 0.006, 0.006]
        path_distance = max(0.002, min(0.010, max(close_path_delta_norm * 0.55, 0.004)))
        center = _vec_add(tip_local_xyz, [direction[index] * path_distance for index in range(3)])
        path = f"/World/envs/env_{slot.local_env_index}/v95_active_sensor_reference_f{int(logical_finger)}"
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xform_api = UsdGeom.XformCommonAPI(cube)
        xform_api.SetTranslate(tuple(float(value) for value in center))
        xform_api.SetScale(tuple(float(value) for value in [value * 2.0 for value in half_extent]))
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision_api.CreateCollisionEnabledAttr().Set(True)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        physx_collision_api.CreateContactOffsetAttr().Set(0.002)
        physx_collision_api.CreateRestOffsetAttr().Set(0.0)
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim())
        report_api.CreateThresholdAttr().Set(0.0)
        out.update(
            {
                "reference_pad_prim_path": path,
                "reference_pad_spawned": bool(cube.GetPrim().IsValid()),
                "reference_pad_collision_enabled": True,
                "reference_pad_contact_report_api": True,
                "reference_pad_center_local_xyz": center,
                "reference_pad_half_extent_xyz": half_extent,
                "reference_pad_blocker": "",
            }
        )
    except Exception as exc:
        out["reference_pad_blocker"] = f"{type(exc).__name__}:{exc}"
    return out


def _remove_v95_reference_pad(slot: SingleContextSlot, logical_finger: int) -> None:
    try:
        import omni.usd  # noqa: WPS433

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        path = f"/World/envs/env_{slot.local_env_index}/v95_active_sensor_reference_f{int(logical_finger)}"
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
    except Exception:
        pass


def _self_check_reference_distance_row(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    finger_index: int,
    table_top_z: float,
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "reference_signed_distance_m": 1.0,
        "reference_tip_local_xyz": [],
        "reference_pad_center_local_xyz": list((reference or {}).get("reference_pad_center_local_xyz", [])),
        "reference_pad_half_extent_xyz": list((reference or {}).get("reference_pad_half_extent_xyz", [])),
        "reference_distance_blocker": "",
    }
    try:
        tips = _v92_tip_positions(backend)
        tip_rows, tip_shape = _tip_rows_for_env(tips, slot.local_env_index)
        tip = _tip_from_rows(tip_rows, finger_index)
        if not tip:
            out["reference_distance_blocker"] = f"tip_unavailable:{tip_shape}"
            return out
        out["reference_tip_local_xyz"] = tip
        center = list((reference or {}).get("reference_pad_center_local_xyz", []))
        half = list((reference or {}).get("reference_pad_half_extent_xyz", []))
        if len(center) >= 3 and len(half) >= 3:
            out["reference_signed_distance_m"] = _signed_distance_to_aabb(tip, center, half)
        else:
            out["reference_signed_distance_m"] = float(tip[2]) - float(table_top_z)
    except Exception as exc:
        out["reference_distance_blocker"] = f"{type(exc).__name__}:{exc}"
    return out


def _signed_distance_to_aabb(point: list[float], center: list[float], half_extent: list[float]) -> float:
    q = [abs(float(point[index]) - float(center[index])) - max(1.0e-6, float(half_extent[index])) for index in range(3)]
    outside = [max(value, 0.0) for value in q]
    outside_norm = _norm(outside)
    if outside_norm > 0.0:
        return outside_norm
    return max(q)


def _bbox_for_prims(cache: Any, prims: list[Any]) -> dict[str, list[float]]:
    if not prims:
        return {"center": [], "extent": []}
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for prim in prims:
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn = box.GetMin()
            mx = box.GetMax()
            for axis in range(3):
                mins[axis] = min(mins[axis], float(mn[axis]))
                maxs[axis] = max(maxs[axis], float(mx[axis]))
        except Exception:
            continue
    if not all(math.isfinite(value) for value in [*mins, *maxs]):
        return {"center": [], "extent": []}
    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    extent = [maxs[axis] - mins[axis] for axis in range(3)]
    return {"center": center, "extent": extent}


def _center_offset(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _extent_error(expected: list[float], actual: list[float]) -> float:
    if len(expected) < 3 or len(actual) < 3:
        return 1.0
    expected_sorted = sorted([max(abs(float(value)), 1.0e-6) for value in expected])
    actual_sorted = sorted([max(abs(float(value)), 1.0e-6) for value in actual])
    return max(
        abs(actual_sorted[index] - expected_sorted[index]) / expected_sorted[index]
        for index in range(3)
    )


def _tip_world_xyz(tips: Any, env_index: int, finger_index: int) -> list[float]:
    try:
        row = tips[int(env_index), int(finger_index)].detach().cpu().reshape(-1).tolist()
        return [float(value) for value in row[:3]]
    except Exception:
        return []


def _tip_rows_for_env(tips: Any, env_index: int) -> tuple[list[list[float]], str]:
    if tips is None:
        return [], ""
    shape = _shape_string(tips)
    try:
        source = tips
        if hasattr(source, "detach"):
            if len(getattr(source, "shape", ())) >= 3:
                source = source[int(env_index)]
            source = source.detach().cpu().reshape(-1, 3).tolist()
        else:
            source = list(source)
            if source and isinstance(source[0], (list, tuple)) and source[0] and isinstance(source[0][0], (list, tuple)):
                source = source[int(env_index)]
        rows = []
        for row in source:
            values = [float(value) for value in list(row)[:3]]
            if len(values) >= 3:
                rows.append(values)
        return rows, shape
    except Exception as exc:
        return [], f"{shape}|{type(exc).__name__}:{exc}"


def _tip_from_rows(rows: list[list[float]], finger_index: int) -> list[float]:
    index = int(finger_index)
    if 0 <= index < len(rows):
        return [float(value) for value in rows[index][:3]]
    return []


def _shape_string(value: Any) -> str:
    try:
        shape = getattr(value, "shape", None)
        if shape is not None:
            return "x".join(str(int(item)) for item in list(shape))
    except Exception:
        pass
    try:
        outer = len(value)
        inner = len(value[0]) if outer else 0
        inner2 = len(value[0][0]) if outer and inner and isinstance(value[0][0], (list, tuple)) else 0
        return "x".join(str(item) for item in (outer, inner, inner2) if item)
    except Exception:
        return ""


def _quat_apply_wxyz(quat: list[float], vec: list[float]) -> list[float]:
    if len(quat) < 4 or len(vec) < 3:
        return [0.0, 0.0, 0.0]
    w, x, y, z = [float(value) for value in quat[:4]]
    vx, vy, vz = [float(value) for value in vec[:3]]
    # q * v * q^-1 for unit quaternions in wxyz order.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _signed_distance_to_cylinder(
    point: list[float],
    center: list[float],
    axis: list[float],
    radius: float,
    half_height: float,
) -> float:
    rel = _vec_sub(point, center)
    axial = _dot(rel, axis)
    radial_vec = _vec_sub(rel, [axis[index] * axial for index in range(3)])
    radial = _norm(radial_vec)
    radial_excess = radial - float(radius)
    axial_excess = abs(axial) - float(half_height)
    outside_radial = max(radial_excess, 0.0)
    outside_axial = max(axial_excess, 0.0)
    if outside_radial > 0.0 or outside_axial > 0.0:
        return math.sqrt(outside_radial * outside_radial + outside_axial * outside_axial)
    return max(radial_excess, axial_excess)


def _nearest_point_on_cylinder_surface(
    point: list[float],
    center: list[float],
    axis: list[float],
    radius: float,
    half_height: float,
) -> list[float]:
    rel = _vec_sub(point, center)
    axial = max(-float(half_height), min(float(half_height), _dot(rel, axis)))
    radial_vec = _vec_sub(rel, [axis[index] * _dot(rel, axis) for index in range(3)])
    radial_norm = _norm(radial_vec)
    if radial_norm <= 1.0e-8:
        radial_dir = _orthogonal_unit(axis)
    else:
        radial_dir = [radial_vec[index] / radial_norm for index in range(3)]
    return _vec_add(center, _vec_add([axis[index] * axial for index in range(3)], [radial_dir[index] * float(radius) for index in range(3)]))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(min(len(left), len(right), 3)))


def _normalize(vec: list[float]) -> list[float]:
    norm = _norm(vec)
    if norm <= 1.0e-8:
        return [0.0, 0.0, 1.0]
    return [float(value) / norm for value in vec[:3]]


def _orthogonal_unit(vec: list[float]) -> list[float]:
    axis = _normalize(vec)
    candidate = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
    cross = [
        axis[1] * candidate[2] - axis[2] * candidate[1],
        axis[2] * candidate[0] - axis[0] * candidate[2],
        axis[0] * candidate[1] - axis[1] * candidate[0],
    ]
    return _normalize(cross)


def _vec_add(left: list[float], right: list[float]) -> list[float]:
    return [
        (float(left[index]) if index < len(left) else 0.0) + (float(right[index]) if index < len(right) else 0.0)
        for index in range(3)
    ]


def _finger_close_joint_delta(before: Any, after: Any, env_index: int, finger_index: int) -> float:
    cols = [16 + int(finger_index), 21 + int(finger_index)]
    try:
        env = int(env_index)
        values = []
        for col in cols:
            if col < int(before.shape[-1]) and col < int(after.shape[-1]):
                values.append(after[env, col] - before[env, col])
        if not values:
            return 0.0
        if hasattr(values[0], "detach"):
            import torch  # noqa: WPS433

            delta = torch.stack(values)
            return float(torch.linalg.vector_norm(delta).detach().cpu().item())
        return math.sqrt(sum(float(value) * float(value) for value in values))
    except Exception:
        return 0.0


def _group_close_joint_delta(before: Any, after: Any, env_index: int, finger_indices: list[int]) -> float:
    values = [
        _finger_close_joint_delta(before, after, env_index, finger_index)
        for finger_index in finger_indices
        if 0 <= int(finger_index) < 5
    ]
    return math.sqrt(sum(float(value) * float(value) for value in values)) if values else 0.0


def _group_tip_delta_max(before: Any, after: Any, env_index: int, finger_indices: list[int]) -> float:
    values = [
        _v92_tip_delta(before, after, env_index, finger_index)
        for finger_index in finger_indices
        if 0 <= int(finger_index) < 5
    ]
    return max([0.0, *[float(value) for value in values]])


def _finger_local_values(values: Any, finger_index: int) -> list[float]:
    cols = [10 + int(finger_index), 15 + int(finger_index)]
    try:
        row = values
        if hasattr(row, "detach"):
            row = row.detach().cpu().reshape(-1).tolist()
        row = list(row)
        return [float(row[col]) if col < len(row) else 0.0 for col in cols]
    except Exception:
        return [0.0, 0.0]


def _finger_sensor_maps(finger_rows: list[dict[str, Any]]) -> dict[str, dict[int, int]]:
    maps: dict[str, dict[int, int]] = {}
    for row in finger_rows:
        part = str(row.get("part_name") or "")
        logical = _int_field(row, "logical_finger_id", 0) - 1
        sensor = _int_field(row, "inferred_logical_to_sensor_index", -1)
        if not part or not (0 <= logical < 5) or not (0 <= sensor < 5):
            continue
        if not _bool(row.get("finger_motion_ok")):
            continue
        maps.setdefault(part, {})[logical] = sensor
    return maps


def _object_xy_for_slot(slot: SingleContextSlot) -> tuple[float, float]:
    # Keep all objects close to the hand workspace while preserving per-slot identity.
    offsets = {
        "Plug2": (-0.20, 0.000),
        "Screw1": (-0.20, 0.000),
        "Backrest": (-0.20, 0.000),
        "Rod": (-0.20, 0.000),
        "Frame": (-0.20, 0.000),
    }
    return offsets.get(slot.part_name, (-0.20, 0.0))


def _safe_hand_target() -> list[float]:
    return [0.0, -0.05, 1.05]


def _initial_pregrasp_target(row: dict[str, Any]) -> list[float]:
    center = _row_center(row)
    half = _to_float(row.get("object_support_half_height_m"), 0.02)
    return [
        center[0],
        center[1] - max(0.075, min(0.14, 0.06 + half * 0.10)),
        center[2] + max(0.040, min(0.10, 0.035 + half * 0.10)),
    ]


def _adjust_object_support_height(row: dict[str, Any], sample: dict[str, Any], backend: IsaacUnifiedSingleContextBackend) -> None:
    table_top = _to_float(row.get("table_top_z_m"), 0.72)
    half = _to_float(row.get("object_support_half_height_m"), 0.02)
    current = _to_float(row.get("object_center_local_z"), table_top + half + 0.003)
    is_backrest = str(row.get("part_name") or "") == "Backrest"
    lower = table_top + half + (-0.012 if is_backrest else 0.001)
    upper = table_top + half + (0.060 if is_backrest else 0.080)
    reason = "support_height_noop"
    before = current
    if _to_float(sample.get("force_peak_n")) > backend.contact_manager.force_threshold_n:
        current += 0.018 if is_backrest else 0.012
        reason = "raise_after_initial_force"
    elif _to_float(sample.get("pre_contact_displacement_m")) > 0.02 and _to_float(sample.get("pre_contact_z_drift_signed_m")) < -0.002:
        current -= min(0.018 if is_backrest else 0.010, abs(_to_float(sample.get("pre_contact_z_drift_signed_m"))) * (0.90 if is_backrest else 0.75))
        reason = "lower_after_freefall_z_drift"
    else:
        current += 0.008 if is_backrest else 0.006
        reason = "raise_after_planar_motion_or_table_contact"
    _set_object_center_z(row, max(lower, min(upper, current)))
    row["object_support_height_adjustment_reason"] = reason
    row["object_support_height_adjustment_m"] = _to_float(row.get("object_center_local_z")) - before


def _adjust_hand_away_from_contact(row: dict[str, Any], sample: dict[str, Any], backend: IsaacUnifiedSingleContextBackend) -> None:
    if _sample_motion_force_ok(sample, backend):
        return
    target = _row_hand_target(row)
    center = _row_center(row)
    away_y = -1.0 if target[1] <= center[1] else 1.0
    if str(row.get("part_name") or "") == "Backrest":
        adjustment = [0.0, away_y * 0.035, 0.025]
    else:
        adjustment = [0.0, away_y * 0.020, 0.015]
    _apply_hand_adjustment(row, adjustment, backend)


def _distance_closure_adjustment(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    row: dict[str, Any],
    sample: dict[str, Any],
) -> list[float]:
    if _to_float(sample.get("force_peak_n")) > backend.contact_manager.force_threshold_n or _int_field(sample, "force_count", 0) > 0:
        target = _row_hand_target(row)
        center = _row_center(row)
        away_y = -1.0 if target[1] <= center[1] else 1.0
        return [0.0, away_y * 0.015, 0.012]
    correction = _active_tip_distance_correction(backend, slot, row)
    if _norm(correction) <= 1.0e-8:
        target = _row_hand_target(row)
        center = _row_center(row)
        direction_y = 1.0 if target[1] < center[1] else -1.0
        return [0.0, direction_y * 0.012, 0.0]
    return correction


def _active_tip_distance_correction(
    backend: IsaacUnifiedSingleContextBackend,
    slot: SingleContextSlot,
    row: dict[str, Any],
) -> list[float]:
    tips = _v92_tip_positions(backend)
    if tips is None:
        return [0.0, 0.0, 0.0]
    base = getattr(backend.env, "unwrapped", backend.env) if backend.env is not None else None
    state = _v91_object_state_for_slot(backend, slot)
    object_w = state.get("root_pos_w", [])
    origin = _tensor_row(getattr(getattr(base, "scene", None), "env_origins", None), slot.local_env_index)
    if len(object_w) < 3 or origin is None:
        return [0.0, 0.0, 0.0]
    origin_vec = _vec3(origin)
    object_local = [float(object_w[i]) - origin_vec[i] for i in range(3)]
    object_world = [float(object_w[i]) for i in range(3)]
    try:
        tip_rows = tips[int(slot.local_env_index)].detach().cpu().reshape(-1, 3).tolist()
    except Exception:
        return [0.0, 0.0, 0.0]
    active_indices = _v91_active_finger_indices(str(row.get("active_finger_group") or ""))
    best: tuple[float, list[float]] | None = None
    for index in active_indices:
        if index < 0 or index >= len(tip_rows):
            continue
        tip = [float(value) for value in tip_rows[index][:3]]
        for object_center in (object_local, object_world):
            delta = [tip[i] - object_center[i] for i in range(3)]
            dist = _norm(delta)
            direction = _unit(delta)
            offset = _surface_offset_for_direction(slot.part_name, direction)
            surface_distance = max(0.0, dist - offset)
            if best is None or surface_distance < best[0]:
                desired = offset + 0.010
                correction = [direction[i] * (desired - dist) for i in range(3)]
                best = (surface_distance, correction)
    if best is None:
        return [0.0, 0.0, 0.0]
    return _clip_vec(best[1], 0.020)


def _apply_hand_adjustment(row: dict[str, Any], adjustment: list[float], backend: IsaacUnifiedSingleContextBackend) -> None:
    target = _row_hand_target(row)
    adjusted = [target[i] + float(adjustment[i]) for i in range(3)]
    table_top = _to_float(row.get("table_top_z_m"), 0.72)
    adjusted[2] = max(table_top + 0.045, min(table_top + 0.35, adjusted[2]))
    row["hand_target_local_xyz"] = adjusted
    row["hand_target_local_x"] = adjusted[0]
    row["hand_target_local_y"] = adjusted[1]
    row["hand_target_local_z"] = adjusted[2]
    row["last_hand_adjustment_xyz"] = [float(value) for value in adjustment[:3]]
    row["last_hand_adjustment_norm_m"] = _norm(adjustment)


def _set_object_center_z(row: dict[str, Any], z_value: float) -> None:
    center = _row_center(row)
    center[2] = float(z_value)
    row["object_center_local_xyz"] = center
    row["object_center_local_x"] = center[0]
    row["object_center_local_y"] = center[1]
    row["object_center_local_z"] = center[2]


def _row_center(row: dict[str, Any]) -> list[float]:
    raw = row.get("object_center_local_xyz") or [
        row.get("object_center_local_x", 0.0),
        row.get("object_center_local_y", 0.0),
        row.get("object_center_local_z", 0.0),
    ]
    return [float(value) for value in list(raw)[:3]]


def _row_hand_target(row: dict[str, Any]) -> list[float]:
    raw = row.get("hand_target_local_xyz") or [
        row.get("hand_target_local_x", 0.0),
        row.get("hand_target_local_y", 0.0),
        row.get("hand_target_local_z", 1.0),
    ]
    return [float(value) for value in list(raw)[:3]]


def _rows_by_part(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("part_name") or ""): dict(row) for row in rows}


def _slot_for_plan_row(backend: IsaacUnifiedSingleContextBackend, row: dict[str, Any]) -> SingleContextSlot | None:
    part_name = str(row.get("part_name") or "")
    env_index = _int_field(row, "env_index", -1)
    for slot in backend.slots:
        if slot.part_name == part_name and int(slot.local_env_index) == env_index:
            return slot
    return None


def _surface_offset_for_direction(part_name: str, direction: list[float]) -> float:
    geometry = V85_PART_GEOMETRY.get(part_name, {})
    kind = str(geometry.get("kind") or "")
    if kind == "sphere":
        return max(0.001, _to_float(geometry.get("radius"), 0.018))
    if kind == "cylinder_z":
        radius = _to_float(geometry.get("radius"), 0.006)
        half = _to_float(geometry.get("half_height"), 0.0125)
        radial = math.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
        return max(0.001, radial * radius + abs(direction[2]) * half)
    if kind == "boxes":
        best = 0.0
        for _center, half in geometry.get("boxes", ()):
            best = max(best, sum(abs(float(direction[i])) * abs(float(half[i])) for i in range(3)))
        return max(0.001, best)
    return 0.02


def _vec3(value: Any) -> list[float]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        return [float(item) for item in list(value)[:3]]
    except Exception:
        return [0.0, 0.0, 0.0]


def _tensor_list(value: Any, index: int, *, width: int | None = None) -> list[float]:
    row = _tensor_row(value, index)
    try:
        if hasattr(row, "detach"):
            row = row.detach().cpu().reshape(-1).tolist()
        out = [float(item) for item in list(row)]
    except Exception:
        out = []
    if width is not None:
        out = out[: int(width)]
        out.extend([0.0] * max(0, int(width) - len(out)))
    return out


def _tensor_float(value: Any, index: int, default: float = 0.0) -> float:
    return _to_float(_tensor_row(value, index), default)


def _tensor_bool(value: Any, index: int) -> bool:
    row = _tensor_row(value, index)
    if hasattr(row, "detach"):
        try:
            return bool(row.detach().cpu().reshape(-1)[0].item())
        except Exception:
            return False
    return bool(row)


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vec[:3]))


def _unit(vec: list[float]) -> list[float]:
    norm = _norm(vec)
    if norm <= 1.0e-8:
        return [0.0, -1.0, 0.0]
    return [float(value) / norm for value in vec[:3]]


def _clip_vec(vec: list[float], limit: float) -> list[float]:
    norm = _norm(vec)
    if norm <= float(limit) or norm <= 1.0e-8:
        return [float(value) for value in vec[:3]]
    scale = float(limit) / norm
    return [float(value) * scale for value in vec[:3]]


def _find_slot_row(rows: list[dict[str, Any]], slot: SingleContextSlot) -> dict[str, Any]:
    for row in rows:
        if str(row.get("part_name") or "") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index:
            return dict(row)
    for row in rows:
        if str(row.get("part_name") or "") == slot.part_name:
            return dict(row)
    return {}


def _find_slot_row_ref(rows: list[dict[str, Any]], slot: SingleContextSlot) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("part_name") or "") == slot.part_name and _int_field(row, "env_index") == slot.local_env_index:
            return row
    for row in rows:
        if str(row.get("part_name") or "") == slot.part_name:
            return row
    return None


def _find_part_row(rows: list[dict[str, Any]], part_name: str) -> dict[str, Any]:
    return next((dict(row) for row in rows if str(row.get("part_name") or "") == part_name), {})


def _first_blocker(rows: list[dict[str, Any]], key: str, default: str) -> str:
    return str(next((row.get(key) for row in rows if row.get(key)), default))


def _policy_cols_for_group(group: str) -> str:
    cols: list[str] = []
    for index in _v91_active_finger_indices(group):
        cols.extend([str(6 + index), str(11 + index)])
    return ",".join(cols)


def _isaac_cols_for_group(group: str) -> str:
    cols: list[str] = []
    for index in _v91_active_finger_indices(group):
        cols.extend([str(16 + index), str(21 + index)])
    return ",".join(cols)


def _next_action(decision: str) -> str:
    if decision == "ACTIVE_OBJECT_RESET_BLOCKER":
        return "repair_slot_to_active_asset_state_refresh"
    if decision == "PREGRASP_PLAN_NOT_APPLIED":
        return "repair_v95_reset_pregrasp_plan_lifecycle"
    if decision == "LEGACY_PLUG2_RESET_PATH_BLOCKER":
        return "remove_legacy_plug2_held_state_from_unified_reset"
    if decision == "WRIST_ACTION_MAPPING_BLOCKER":
        return "repair_wrist_action_mapping_before_any_training"
    if decision == "FINGER_MOTION_BLOCKER":
        return "repair_finger_action_mapping_or_joint_actuation"
    if decision == "ACTUAL_OBJECT_CONTACT_BLOCKER":
        return "repair_pregrasp_or_native_collision_until_force_contact_is_real"
    if decision == "NATIVE_COLLISION_ASSET_BLOCKER":
        return "repair_native_usd_collision_before_training_unlock"
    return "repair_reset_lifecycle_or_native_shutdown"


def _safe_table_top(value: Any) -> float:
    top = _to_float(value, 0.72)
    if not math.isfinite(top) or top < -0.10 or top > 1.10:
        return 0.72
    return top


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "pass"}
    return bool(value)


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return 0


def _v95_branch_scan_text(text: str) -> str:
    marker = 'if args_cli.run_mode == "v95_minimal_unified_workcell_repair":'
    start = text.find(marker)
    if start < 0:
        return ""
    next_marker = text.find('\n        if args_cli.run_mode == "v94_safe_pregrasp_and_finger_contact_calibration":', start + len(marker))
    return text[start:] if next_marker < 0 else text[start:next_marker]


def _strip_scanner_literal_tables(text: str) -> str:
    for marker in ("_FORBIDDEN_SOURCE_TOKENS = (", "_FORBIDDEN_ARTIFACT_PATTERNS = ("):
        start = text.find(marker)
        if start < 0:
            continue
        end = text.find(")\n", start)
        if end >= 0:
            text = text[:start] + text[end + 2 :]
    return text


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else ["part_name", "status", "blocker"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
