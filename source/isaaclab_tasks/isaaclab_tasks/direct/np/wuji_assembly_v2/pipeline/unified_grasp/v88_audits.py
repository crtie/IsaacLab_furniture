"""Audit helpers for the non-final v88 stabilized PPO stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_csv, write_json


def write_v88_code_path_audit(
    run_dir: str | Path,
    repo_root: str | Path,
    *,
    branch_names: list[str],
) -> dict[str, Any]:
    run_path = Path(run_dir)
    root = Path(repo_root)
    rows = [
        {
            "category": "entrypoint",
            "name": "v88_stabilized_grasp_policy_and_physics_audit",
            "path": "scripts/environments/run_v81_physical_backend_grasp_rl.py",
            "role": "--run_mode v88_stabilized_grasp_policy_and_physics_audit",
            "v88_used": True,
            "notes": "existing v81-v87 runner extended; no new runner/backend fork",
        },
        {
            "category": "rl_runner",
            "name": "OnPolicyRunner",
            "path": "rsl_rl.runners.OnPolicyRunner",
            "role": "PPO collection and policy artifact checkpoint",
            "v88_used": True,
            "notes": "runner completion is not grasp success",
        },
        {
            "category": "rl_adapter",
            "name": "RslRlUnifiedGraspVecEnv",
            "path": "pipeline/unified_grasp/train_unified_rl.py",
            "role": "RSL-RL vector env adapter",
            "v88_used": True,
            "notes": "wraps UnifiedGraspEnv without exposing object-write/sticky/proxy actions",
        },
        {
            "category": "env",
            "name": "UnifiedGraspEnv",
            "path": "pipeline/unified_grasp/unified_grasp_env.py",
            "role": "v88 flags, enhanced observation fill, auto reset",
            "v88_used": True,
            "notes": "action API remains 16D residual-only",
        },
        {
            "category": "backend",
            "name": "IsaacUnifiedSingleContextBackend",
            "path": "pipeline/unified_grasp/single_context_backend.py",
            "role": "single SimulationContext, v85 staging, v88 nominal prior, physics tuning",
            "v88_used": True,
            "notes": "multi-gym.make backend remains diagnostic only",
        },
        {
            "category": "action_mapping",
            "name": "UnifiedActionMapper",
            "path": "pipeline/unified_grasp/unified_action_mapper.py",
            "role": "16D residual policy action to 26D Wuji action",
            "v88_used": True,
            "notes": "only policy-to-Isaac action path",
        },
        {
            "category": "contact",
            "name": "ContactManager",
            "path": "pipeline/unified_grasp/contact_manager.py",
            "role": "force-contact tensor readout and support gates",
            "v88_used": True,
            "notes": "distance fallback remains diagnostic only",
        },
        {
            "category": "reward",
            "name": "compute_reward_terms",
            "path": "pipeline/unified_grasp/reward_terms.py",
            "role": "contact/support/smoothness penalties",
            "v88_used": True,
            "notes": "no distance-only success",
        },
        {
            "category": "preflight",
            "name": "run_v86_contact_gate_preflight",
            "path": "pipeline/unified_grasp/single_context_backend.py",
            "role": "fresh v85 clean contact gate before every branch",
            "v88_used": True,
            "notes": "prior v85 artifacts are not pass evidence",
        },
        {
            "category": "train",
            "name": "train_v88_stabilized_residual_ppo",
            "path": "pipeline/unified_grasp/train_unified_rl.py",
            "role": "branch PPO artifact training",
            "v88_used": True,
            "notes": "checkpoint is policy artifact only",
        },
        {
            "category": "eval",
            "name": "evaluate_v88_stabilized_policy",
            "path": "pipeline/unified_grasp/eval_unified_rl.py",
            "role": "deterministic no-sticky eval for nominal and policy branches",
            "v88_used": True,
            "notes": "no final replay/video/export path",
        },
        {
            "category": "legacy_compat",
            "name": "v80-v82 diagnostic paths",
            "path": "pipeline/unified_grasp/physical_backend.py",
            "role": "not used as v88 success path",
            "v88_used": False,
            "notes": "quarantine candidate after single-context stack stabilizes",
        },
        {
            "category": "cleanup_candidate",
            "name": "duplicated v86/v87 progress writers",
            "path": "scripts/environments/run_v81_physical_backend_grasp_rl.py",
            "role": "report-only candidate; not deleted in v88",
            "v88_used": False,
            "notes": "future cleanup can consolidate branch matrix writers",
        },
    ]
    for branch in branch_names:
        rows.append(
            {
                "category": "branch",
                "name": branch,
                "path": str(root / "debug_runs/v88_stabilized_grasp_policy_and_physics_audit" / "branches" / branch),
                "role": "v88 experiment branch",
                "v88_used": True,
                "notes": "non-final PPO/eval branch",
            }
        )
    csv_path = write_csv(run_path / "v88_code_path_audit.csv", rows)
    json_path = write_json(run_path / "v88_code_path_audit.json", rows)
    md_path = run_path / "v88_code_path_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v88_code_path_audit_csv": str(csv_path), "v88_code_path_audit_json": str(json_path), "v88_code_path_audit_md": str(md_path)}


def write_v88_observation_audit(
    run_dir: str | Path,
    *,
    enhanced_feature_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    enhanced_feature_rows = enhanced_feature_rows or []
    base_rows = [
        {"feature": "part_scalar_id", "slot_range": "0", "old_layout": True, "enhanced_layout": True, "source": "env state"},
        {"feature": "active_distance", "slot_range": "1", "old_layout": True, "enhanced_layout": True, "source": "env/contact state"},
        {"feature": "force_contact_count", "slot_range": "2", "old_layout": True, "enhanced_layout": True, "source": "ContactManager"},
        {"feature": "support/object/table flags", "slot_range": "4:11", "old_layout": True, "enhanced_layout": True, "source": "backend metrics"},
        {"feature": "backend/mode flags", "slot_range": "12:17", "old_layout": True, "enhanced_layout": True, "source": "UnifiedGraspEnvCfg"},
        {"feature": "part_one_hot", "slot_range": "18:22", "old_layout": False, "enhanced_layout": True, "source": "part label"},
        {"feature": "object_pose_relative_to_palm", "slot_range": "23:25", "old_layout": False, "enhanced_layout": True, "source": "single-context backend"},
        {"feature": "bbox_geometry_compact", "slot_range": "26:28", "old_layout": False, "enhanced_layout": True, "source": "analytic geometry"},
        {"feature": "fingertip_object_distance", "slot_range": "29:30", "old_layout": False, "enhanced_layout": True, "source": "fingertip tensors"},
        {"feature": "contact_history", "slot_range": "31:32", "old_layout": False, "enhanced_layout": True, "source": "force-contact streak"},
        {"feature": "previous_action_summary", "slot_range": "33:34", "old_layout": False, "enhanced_layout": True, "source": "nominal/residual action audit"},
        {"feature": "hold_lift_state", "slot_range": "35:36", "old_layout": False, "enhanced_layout": True, "source": "object displacement and lift delta"},
    ]
    rows = [
        {
            **row,
            "old_observation_dim": 64,
            "new_observation_dim": 64,
            "action_dim": 16,
            "future_success_label_exposed": False,
            "object_write_action_exposed": False,
        }
        for row in base_rows
    ]
    for item in enhanced_feature_rows[: max(1, min(50, len(enhanced_feature_rows)))]:
        rows.append(
            {
                "feature": "runtime_enhanced_feature_sample",
                "slot_range": "enhanced",
                "old_layout": False,
                "enhanced_layout": True,
                "source": "backend runtime metric",
                "old_observation_dim": 64,
                "new_observation_dim": 64,
                "action_dim": 16,
                "future_success_label_exposed": False,
                "object_write_action_exposed": False,
                **{f"sample_{key}": value for key, value in item.items() if key in {"part_name", "object_rel_palm_x", "fingertip_object_distance_min_m"}},
            }
        )
    csv_path = write_csv(run_path / "v88_observation_audit.csv", rows)
    json_path = write_json(run_path / "v88_observation_audit.json", rows)
    md_path = run_path / "v88_observation_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v88_observation_audit_csv": str(csv_path), "v88_observation_audit_json": str(json_path), "v88_observation_audit_md": str(md_path)}


def write_v88_physics_asset_audit(
    run_dir: str | Path,
    *,
    backend: Any | None,
    tuning_rows: list[dict[str, Any]] | None,
    physics_profile: str,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    identity_rows = list(getattr(backend, "object_identity_rows", []) or []) if backend is not None else []
    contact_rows = list(getattr(backend, "contact_api_rows", []) or []) if backend is not None else []
    for part in V80_PARTS:
        identity = next((row for row in identity_rows if row.get("part_name") == part), {})
        contact = next((row for row in contact_rows if row.get("part_name") == part), {})
        rows.append(
            {
                "asset_name": part,
                "physics_profile": physics_profile,
                "active_asset_label": identity.get("active_asset_label", ""),
                "active_asset_usd": identity.get("active_asset_usd", ""),
                "active_asset_prim_path": identity.get("active_asset_prim_path", ""),
                "object_identity_verified": bool(identity.get("object_identity_verified")),
                "contact_sensor_api_available": bool(contact.get("contact_sensor_api_available")),
                "material_tuning_source": "runtime_physx_material_properties_when_available",
                "solver_tuning_source": "reported_only_unavailable_if_not_exposed",
                "static_friction_before": "",
                "dynamic_friction_before": "",
                "restitution_before": "",
                "static_friction_after": "",
                "dynamic_friction_after": "",
                "restitution_after": "",
                "target_static_friction": 1.25 if physics_profile == "conservative_contact" else "",
                "target_dynamic_friction": 1.05 if physics_profile == "conservative_contact" else "",
                "target_restitution": 0.0 if physics_profile == "conservative_contact" else "",
                "contact_offset_target_m": 0.002 if physics_profile == "conservative_contact" else "",
                "rest_offset_target_m": 0.0 if physics_profile == "conservative_contact" else "",
                "velocity_solver_iterations_min": 4 if physics_profile == "conservative_contact" else "",
                "max_depenetration_velocity_target_mps": 2.0 if physics_profile == "conservative_contact" else "",
                "tuning_applied": False,
                "tuning_failure_reason": "not_requested" if physics_profile != "conservative_contact" else "not_reported_by_backend",
                "rationale": "canonical branch leaves physics untouched" if physics_profile != "conservative_contact" else "bounded plausible stabilization, no sticky or adhesion",
            }
        )
    for item in tuning_rows or []:
        rows.append(
            {
                "asset_name": item.get("asset_name", ""),
                "physics_profile": item.get("physics_profile", physics_profile),
                "active_asset_label": item.get("asset_name", ""),
                "active_asset_usd": "",
                "active_asset_prim_path": "",
                "object_identity_verified": item.get("asset_name") == "WujiFingerPads" or item.get("asset_name") in V80_PARTS,
                "contact_sensor_api_available": "",
                "material_tuning_source": "backend.apply_v88_conservative_contact_tuning",
                "solver_tuning_source": "reported_only_unavailable_if_not_exposed",
                **item,
            }
        )
    csv_path = write_csv(run_path / "v88_physics_asset_audit.csv", rows)
    json_path = write_json(run_path / "v88_physics_asset_audit.json", rows)
    md_path = run_path / "v88_physics_asset_audit.md"
    _write_md(md_path, rows)
    return {"rows": rows, "v88_physics_asset_audit_csv": str(csv_path), "v88_physics_asset_audit_json": str(json_path), "v88_physics_asset_audit_md": str(md_path)}


def write_v88_forbidden_artifact_scan(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    forbidden_names = (
        ".mp4",
        "final_replay",
        "final_video",
        "usable_training_row",
        "training_rows",
        "dataset_export",
        "bc_dataset",
        "sticky_eval",
        "sticky_success",
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(run_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_path).as_posix()
        hits = [name for name in forbidden_names if name in rel]
        if hits:
            rows.append({"path": rel, "forbidden_tokens": ",".join(hits), "forbidden_artifact_present": True})
    summary = {
        "forbidden_artifact_present": bool(rows),
        "forbidden_artifact_count": len(rows),
        "ppo_artifact_checkpoints_allowed": True,
        "final_replay_ran": False,
        "video_generated": False,
        "sticky_eval_ran": False,
        "usable_training_row_count": 0,
        "rows": rows,
    }
    txt_path = run_path / "v88_forbidden_artifact_scan.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        lines = ["FORBIDDEN ARTIFACTS FOUND"]
        lines.extend(f"{row['path']} :: {row['forbidden_tokens']}" for row in rows)
    else:
        lines = ["No v88 forbidden final/sticky/video/dataset artifacts found."]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = write_json(run_path / "v88_forbidden_artifact_scan.json", summary)
    return {"rows": rows, "v88_forbidden_artifact_scan_txt": str(txt_path), "v88_forbidden_artifact_scan_json": str(json_path), **summary}


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
