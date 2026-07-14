"""v89 failure-driven hybrid grasp repair helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_csv, write_json


V89_RUN_MODE = "v89_failure_driven_hybrid_grasp_repair"
V89_BRANCHES = ["baseline_v88_best_continuation", "nominal_candidate_residual_ppo", "bc_warmstart_residual_ppo"]


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


def _has_value(row: dict[str, Any], key: str) -> bool:
    return key in row and row.get(key) not in ("", None)


def _metric(primary: dict[str, Any], fallback: dict[str, Any], key: str, default: float = 0.0) -> float:
    if _has_value(primary, key):
        return _float(primary.get(key), default)
    return _float(fallback.get(key), default)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_v88_artifacts(repo_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(repo_root)
    run = root / "debug_runs/v88_stabilized_grasp_policy_and_physics_audit"
    return {
        "progress": _read_csv(run / "v88_progress_matrix.csv"),
        "branch_candidates": _read_csv(run / "v88_branch_candidate_matrix.csv"),
        "eval": _read_csv(run / "v88_deterministic_eval_no_sticky.csv"),
        "training": _read_csv(run / "v88_training_summary.csv"),
        "physics": _read_csv(run / "v88_physics_asset_audit.csv"),
    }


def write_v89_failure_diagnosis(run_dir: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    v88 = load_v88_artifacts(repo_root)
    rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        progress = next((row for row in v88["progress"] if row.get("part_name") == part), {})
        branch = str(progress.get("best_branch") or "v87_baseline_continuation")
        eval_row = next(
            (row for row in v88["eval"] if row.get("part_name") == part and row.get("branch_name") == branch),
            next((row for row in v88["eval"] if row.get("part_name") == part), {}),
        )
        training = next(
            (row for row in v88["training"] if row.get("part_name") == part and row.get("branch_name") == branch),
            next((row for row in v88["training"] if row.get("part_name") == part), {}),
        )
        force_rate = _float(eval_row.get("force_contact_rate") or progress.get("force_contact_rate"))
        duration = _float(eval_row.get("contact_duration_mean") or progress.get("contact_duration_mean"))
        support = _float(eval_row.get("support_gate_rate") or progress.get("support_gate_rate"))
        hold = _float(eval_row.get("hold_gate_rate") or progress.get("hold_gate_rate"))
        lift = _float(eval_row.get("lift_gate_rate") or progress.get("lift_gate_rate"))
        peak_force = _float(eval_row.get("peak_force_n") or progress.get("peak_force_n"))
        mean_force = _float(eval_row.get("mean_force_n") or progress.get("mean_force_n"))
        excessive = _float(eval_row.get("excessive_force_rate") or progress.get("excessive_force_rate"))
        disp_max = _float(eval_row.get("object_displacement_max_m") or progress.get("object_displacement_max_m"))
        hold_disp = _float(eval_row.get("hold_object_displacement_max_m") or progress.get("hold_object_displacement_max_m"))
        failure_mode, recommended = _classify_failure(part, force_rate, duration, support, hold, lift, peak_force, excessive, disp_max, hold_disp)
        rows.append(
            {
                "part_name": part,
                "best_v88_branch": branch,
                "force_contact_rate": force_rate,
                "contact_duration_mean": duration,
                "support_gate_rate": support,
                "hold_gate_rate": hold,
                "lift_gate_rate": lift,
                "mean_force": mean_force,
                "peak_force": peak_force,
                "excessive_force_rate": excessive,
                "object_displacement_mean": _float(training.get("object_displacement_mean_m")),
                "object_displacement_max": max(disp_max, hold_disp),
                "deterministic_no_sticky_result": str(eval_row.get("status") or progress.get("status") or ""),
                "main_failure_mode": failure_mode,
                "recommended_repair_action": recommended,
                "v88_blocker": str(eval_row.get("blocker") or progress.get("blocker") or ""),
            }
        )
    csv_path = write_csv(run_path / "v89_failure_diagnosis.csv", rows)
    json_path = write_json(run_path / "v89_failure_diagnosis.json", rows)
    md_path = run_path / "v89_failure_diagnosis.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v89_failure_diagnosis_csv": str(csv_path), "v89_failure_diagnosis_json": str(json_path), "v89_failure_diagnosis_md": str(md_path)}


def _classify_failure(
    part: str,
    force_rate: float,
    duration: float,
    support: float,
    hold: float,
    lift: float,
    peak_force: float,
    excessive: float,
    disp_max: float,
    hold_disp: float,
) -> tuple[str, str]:
    if excessive > 0.01 or peak_force > 175.0:
        return "force too high or impact-like", "lower close/approach magnitude and reject impact-only candidate contacts"
    if max(disp_max, hold_disp) > 0.04:
        return "excessive object displacement", "use lower-force candidate with more support fingers and shorter pre-lift hold"
    if force_rate < 0.10 or duration < 2.0:
        return "contact too brief", "try geometry candidate with closer reset clearance and more direct approach"
    if support > 0.0 and hold < 0.5:
        return "support achieved but hold lost", "increase multi-finger support duration and penalize contact loss/action jerk"
    if hold >= 0.5 and lift <= 0.0:
        return "hold achieved but lift failed", "repair lift vector/timing while keeping close pattern stable"
    if 0.0 < lift < 0.25:
        return "bad lift direction/timing", "use candidate-specific lift direction and longer hold before lift"
    if part in {"Rod", "Frame"}:
        return "likely collision/bbox/geometry issue", "prefer wrap/edge clamp candidates over direct push"
    if part in {"Plug2", "Screw1"}:
        return "single-point contact instead of multi-finger support", "prefer body/shaft multi-finger clamp candidates"
    return "observation/policy conditioning issue", "train candidate-conditioned residual policy with candidate id and geometry features"


def write_v89_physics_decision(run_dir: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    v88 = load_v88_artifacts(repo_root)
    branch_rows = v88["branch_candidates"]
    rows: list[dict[str, Any]] = []
    for part in V80_PARTS:
        baseline = next((row for row in branch_rows if row.get("part_name") == part and row.get("best_branch") == "v87_baseline_continuation"), {})
        tuned = next((row for row in branch_rows if row.get("part_name") == part and row.get("best_branch") == "physics_tuned_residual_ppo"), {})
        baseline_score = _float(baseline.get("support_gate_rate")) + _float(baseline.get("hold_gate_rate")) + _float(baseline.get("lift_gate_rate"))
        tuned_score = _float(tuned.get("support_gate_rate")) + _float(tuned.get("hold_gate_rate")) + _float(tuned.get("lift_gate_rate"))
        keep_tuned = bool(tuned and tuned_score > baseline_score and _float(tuned.get("peak_force_n")) <= 175.0)
        rows.append(
            {
                "part_name": part,
                "physics_change": "v88_conservative_contact",
                "decision": "keep_per_object_only" if keep_tuned else "disabled",
                "global_default": False,
                "baseline_score": baseline_score,
                "physics_tuned_score": tuned_score,
                "before_profile": "canonical",
                "after_profile": "conservative_contact",
                "rationale": "disabled because v88 physics_tuned did not beat baseline" if not keep_tuned else "kept only for this object because v88 branch improved gated evidence",
                "risks_hidden_adhesion_or_unrealistic_friction": False,
                "object_pose_write_after_reset_allowed": False,
            }
        )
    csv_path = write_csv(run_path / "v89_physics_decision.csv", rows)
    json_path = write_json(run_path / "v89_physics_decision.json", rows)
    md_path = run_path / "v89_physics_decision.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v89_physics_decision_csv": str(csv_path), "v89_physics_decision_json": str(json_path), "v89_physics_decision_md": str(md_path)}


def default_v89_candidate_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = {
        "Plug2": [
            ("plug2_side_pinch_23", "23", "bbox_adjusted", 0.008, 0.004, 1.0, 0.45, 0.28, "side pinch", "plug cylindrical side"),
            ("plug2_body_pinch_34", "34", "bbox_adjusted", 0.008, -0.004, 1.0, 0.45, 0.30, "body pinch", "plug body side"),
            ("plug2_cap_pinch_234", "234", "analytic_surface", 0.012, 0.0, 0.85, 0.35, 0.24, "cap/head pinch", "plug cap/head"),
        ],
        "Screw1": [
            ("screw1_shaft_pinch_34", "34", "bbox_adjusted", 0.006, 0.0, 1.0, 0.42, 0.26, "shaft pinch", "screw shaft"),
            ("screw1_head_pinch_23", "23", "analytic_surface", 0.008, 0.004, 0.95, 0.36, 0.24, "head pinch", "screw head"),
            ("screw1_multifinger_clamp_234", "234", "bbox_adjusted", 0.010, -0.004, 0.82, 0.30, 0.22, "multi-finger clamp", "shaft and head support"),
        ],
        "Backrest": [
            ("backrest_current_best_low_force_34", "34", "analytic_surface", 0.010, 0.0, 0.62, 0.28, 0.22, "current best lower-force", "backrest broad face"),
            ("backrest_lower_edge_support_34", "34", "bbox_adjusted", 0.012, 0.004, 0.70, 0.26, 0.22, "lower edge support", "backrest lower edge"),
            ("backrest_two_finger_support_234", "234", "bbox_adjusted", 0.014, -0.004, 0.72, 0.22, 0.20, "two-side support", "backrest face and edge"),
        ],
        "Rod": [
            ("rod_wrap_234", "234", "bbox_adjusted", 0.008, 0.004, 0.82, 0.28, 0.24, "wrap clamp", "rod long side"),
            ("rod_side_clamp_34", "34", "bbox_adjusted", 0.010, -0.004, 0.86, 0.30, 0.24, "side clamp", "rod side"),
            ("rod_side_clamp_23", "23", "analytic_surface", 0.012, 0.0, 0.78, 0.26, 0.20, "opposed side clamp", "rod opposite side"),
        ],
        "Frame": [
            ("frame_edge_clamp_234", "234", "analytic_surface", 0.010, 0.0, 0.78, 0.25, 0.22, "edge clamp", "frame edge"),
            ("frame_two_side_support_34", "34", "bbox_adjusted", 0.012, 0.004, 0.76, 0.24, 0.22, "two-side support", "frame bar side"),
            ("frame_corner_support_234", "234", "bbox_adjusted", 0.014, -0.004, 0.72, 0.22, 0.20, "corner support", "frame corner"),
        ],
    }
    for part, items in specs.items():
        for rank, (cid, group, strategy, clearance, lateral, close, approach, lift_z, label, surfaces) in enumerate(items):
            rows.append(
                {
                    "candidate_id": cid,
                    "part_name": part,
                    "candidate_rank": rank,
                    "active_finger_group": group,
                    "surface_strategy": strategy,
                    "use_bbox_adjustment": strategy == "bbox_adjusted",
                    "target_clearance_m": clearance,
                    "lateral_offset_m": lateral,
                    "close_value": close,
                    "approach_scale": approach,
                    "hold_close_value": close,
                    "lift_x": 0.0,
                    "lift_y": 0.0,
                    "lift_z": lift_z,
                    "expected_contact_surfaces": surfaces,
                    "grasp_style": label,
                    "source": "geometry_bbox_surface_only",
                    "success_label_used": False,
                }
            )
    return rows


def write_v89_nominal_candidate_plan(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = default_v89_candidate_plan()
    csv_path = write_csv(run_path / "v89_nominal_candidate_plan.csv", rows)
    json_path = write_json(run_path / "v89_nominal_candidate_plan.json", rows)
    return {"rows": rows, "v89_nominal_candidate_plan_csv": str(csv_path), "v89_nominal_candidate_plan_json": str(json_path)}


def select_best_candidates(candidate_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for part in V80_PARTS:
        rows = [row for row in candidate_rows if row.get("part_name") == part]
        if not rows:
            continue
        rows = sorted(rows, key=_candidate_score, reverse=True)
        selected[part] = dict(rows[0])
    return selected


def _candidate_score(row: dict[str, Any]) -> tuple[Any, ...]:
    forbidden = _bool(row.get("object_write_by_policy_detected")) or _bool(row.get("sticky_action_available_to_policy"))
    return (
        int(not forbidden),
        int(_bool(row.get("candidate_success"))),
        _float(row.get("support_gate_rate")),
        _float(row.get("hold_gate_rate")),
        _float(row.get("lift_gate_rate")),
        -_float(row.get("excessive_force_rate")),
        -_float(row.get("object_displacement_max_m")),
        _float(row.get("contact_duration_mean")),
    )


def valid_bc_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in candidate_rows
        if _float(row.get("support_gate_rate")) > 0.0
        and _float(row.get("hold_gate_rate")) > 0.0
        and not _bool(row.get("object_write_by_policy_detected"))
        and not _bool(row.get("sticky_action_available_to_policy"))
    ]


def write_v89_code_growth_report(
    run_dir: str | Path,
    *,
    touched_files: list[str],
    used_modes: list[str],
    unused_modes: list[str],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = []
    for path in touched_files:
        rows.append({"category": "touched_file", "name": path, "v89_used": True, "recommendation": "keep"})
    for mode in used_modes:
        rows.append({"category": "run_mode_used", "name": mode, "v89_used": True, "recommendation": "keep"})
    for mode in unused_modes:
        rows.append({"category": "run_mode_not_used", "name": mode, "v89_used": False, "recommendation": "future quarantine candidate only"})
    rows.append({"category": "duplicated_logic", "name": "v86/v87/v88/v89 branch matrix writers", "v89_used": True, "recommendation": "consolidate after v89 smoke passes"})
    csv_path = write_csv(run_path / "v89_code_growth_report.csv", rows)
    json_path = write_json(run_path / "v89_code_growth_report.json", rows)
    md_path = run_path / "v89_code_growth_report.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v89_code_growth_report_csv": str(csv_path), "v89_code_growth_report_json": str(json_path), "v89_code_growth_report_md": str(md_path)}


def write_v89_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    tokens = (".mp4", "final_replay", "final_video", "usable_training_row", "training_rows", "dataset_export", "bc_dataset", "sticky_eval", "sticky_success")
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
        "sticky_eval_ran": False,
        "final_replay_ran": False,
        "video_generated": False,
        "usable_training_row_count": 0,
        "rows": rows,
    }
    txt_path = run_path / "v89_forbidden_artifact_scan.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(
        ("FORBIDDEN ARTIFACTS FOUND\n" + "\n".join(row["path"] for row in rows) if rows else "No v89 forbidden final/sticky/video/dataset artifacts found.") + "\n",
        encoding="utf-8",
    )
    json_path = write_json(run_path / "v89_forbidden_artifact_scan.json", summary)
    return {**summary, "v89_forbidden_artifact_scan_txt": str(txt_path), "v89_forbidden_artifact_scan_json": str(json_path)}


def write_v89_progress_matrix(
    run_dir: str | Path,
    *,
    branch_results: list[dict[str, Any]],
    v88_progress_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = []
    for part in V80_PARTS:
        candidates = []
        for branch in branch_results:
            train = next((row for row in branch.get("train_rows", []) if row.get("part_name") == part), {})
            eval_row = next((row for row in branch.get("eval_rows", []) if row.get("part_name") == part), {})
            candidates.append({"branch_name": branch.get("branch_name", ""), "train": train, "eval": eval_row})
        best = max(candidates, key=lambda item: _branch_score(item["train"], item["eval"]), default={"branch_name": "", "train": {}, "eval": {}})
        train = best["train"]
        eval_row = best["eval"]
        v88 = next((row for row in v88_progress_rows if row.get("part_name") == part), {})
        force = _metric(eval_row, train, "force_contact_rate")
        support = _metric(eval_row, train, "support_gate_rate")
        hold = _metric(eval_row, train, "hold_gate_rate")
        lift = _metric(eval_row, train, "lift_gate_rate")
        v88_support = _float(v88.get("support_gate_rate"))
        v88_hold = _float(v88.get("hold_gate_rate"))
        v88_lift = _float(v88.get("lift_gate_rate"))
        success = _bool(eval_row.get("grasp_success_claimed"))
        progress = bool(support > v88_support or hold > v88_hold or lift > v88_lift)
        safety_ok = not _bool(eval_row.get("object_write_by_policy_detected") or train.get("object_write_by_policy_detected")) and not _bool(
            eval_row.get("sticky_action_available_to_policy") or train.get("sticky_action_available_to_policy")
        )
        engineering = bool(_bool(train.get("runner_policy_artifact_complete") or train.get("runner_smoke_complete")) and safety_ok)
        if success:
            status = "GRASP_EVAL_PASS_NO_STICKY_NONFINAL"
            blocker = ""
        elif engineering and progress:
            status = "V89_ENGINEERING_PASS_WITH_GRASP_PROGRESS"
            blocker = str(eval_row.get("blocker") or "deterministic_eval_no_full_support_hold_lift_success")
        elif engineering:
            status = "V89_ENGINEERING_PASS_NO_GRASP_PROGRESS"
            blocker = str(eval_row.get("blocker") or "no_material_progress_over_v88")
        else:
            status = str(train.get("status") or "TRAINING_NOT_STARTED")
            blocker = str(train.get("blocker") or eval_row.get("blocker") or "")
        rows.append(
            {
                "part_name": part,
                "best_branch": best["branch_name"],
                "force_contact_rate": force,
                "contact_duration_mean": _metric(eval_row, train, "contact_duration_mean"),
                "mean_force": _metric(eval_row, train, "mean_force_n"),
                "peak_force": _metric(eval_row, train, "peak_force_n"),
                "excessive_force_rate": _metric(eval_row, train, "excessive_force_rate"),
                "support_gate_rate": support,
                "hold_gate_rate": hold,
                "lift_gate_rate": lift,
                "object_displacement_max_m": _metric(eval_row, train, "object_displacement_max_m"),
                "multi_finger_support_rate": _metric(eval_row, train, "multi_finger_support_rate"),
                "v88_support_gate_rate": v88_support,
                "v88_hold_gate_rate": v88_hold,
                "v88_lift_gate_rate": v88_lift,
                "object_write_by_policy_detected": not safety_ok,
                "sticky_action_available_to_policy": _bool(eval_row.get("sticky_action_available_to_policy") or train.get("sticky_action_available_to_policy")),
                "fallback_success_used": _bool(eval_row.get("fallback_success_used") or train.get("fallback_success_used")),
                "ppo_ran": _bool(train.get("ppo_ran")),
                "runner_policy_artifact_complete": _bool(train.get("runner_policy_artifact_complete")),
                "checkpoint_path": str(train.get("checkpoint_path") or ""),
                "grasp_progress": progress,
                "grasp_success_claimed": success,
                "final_success": False,
                "sticky_eval_ran": False,
                "final_replay_ran": False,
                "video_generated": False,
                "usable_training_row_count": 0,
                "engineering_pass": engineering,
                "status": status,
                "blocker": blocker,
                "next_action": "inspect_v89_best_branch_eval" if engineering else "repair_v89_gate_or_candidate_probe",
            }
        )
    csv_path = write_csv(run_path / "v89_progress_matrix.csv", rows)
    md_path = run_path / "v89_progress_matrix.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v89_progress_matrix_csv": str(csv_path), "v89_progress_matrix_md": str(md_path)}


def _branch_score(train: dict[str, Any], eval_row: dict[str, Any]) -> tuple[Any, ...]:
    forbidden = _bool(eval_row.get("object_write_by_policy_detected") or train.get("object_write_by_policy_detected")) or _bool(
        eval_row.get("sticky_action_available_to_policy") or train.get("sticky_action_available_to_policy")
    )
    return (
        int(not forbidden),
        int(_bool(eval_row.get("grasp_success_claimed"))),
        _metric(eval_row, train, "support_gate_rate"),
        _metric(eval_row, train, "hold_gate_rate"),
        _metric(eval_row, train, "lift_gate_rate"),
        -_metric(eval_row, train, "object_displacement_max_m"),
        _metric(eval_row, train, "contact_duration_mean"),
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
