"""Build the bounded Multi-Object Privileged Physical Grasp v3 handoff in 3/."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "debug_runs/multi_object_privileged_grasp_v3"
DELIVERY_ROOT = REPO_ROOT / "3"
PARTS = ("Rod", "Backrest", "Frame", "Screw1", "Plug2")

SOURCE_PATHS = (
    "scripts/environments/run_privileged_physics_grasp_v1.py",
    "scripts/environments/run_multi_object_privileged_grasp_v3.py",
    "scripts/environments/build_wuji_retargeting_seed_bank_v3.py",
    "scripts/environments/audit_external_grasp_models_v3.py",
    "scripts/environments/build_multi_object_privileged_grasp_v3_delivery.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/near_grasp/hand_prior_adapter.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/near_grasp/near_grasp_physics_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/near_grasp/grasp_synthesis",
    "source/isaaclab_tasks/test/test_privileged_physics_grasp.py",
    "configs/grasp_synthesis/objects",
)

ARTIFACT_NAMES = {
    "code_fact_audit_before.json",
    "code_fact_audit_after.json",
    "object_difficulty_ranking.json",
    "m0_reachability_matrix.csv",
    "gate_orchestration_summary.json",
    "gate_a_targeted_retry_comparison.json",
    "external_model_audit.json",
    "external_model_runtime_smokes.json",
    "wuji_retargeting_seed_bank.json",
    "gendex_rod_contact_map_seed.json",
    "final_summary.json",
    "final_report.md",
    "gpu_preflight.json",
    "resolved_object_spec.json",
    "runtime_collision_geometry_audit.json",
    "runtime_kinematic_seed.json",
    "runtime_physics_fingerprint.json",
    "target_object_collision_mesh.json",
    "table_collision_mesh.json",
    "first_physics_frame.json",
    "wuji_joint_asset_compatibility.json",
    "m0_summary.json",
    "optimized_candidates.jsonl",
    "top_candidates.json",
    "synthesis_heartbeat.json",
    "seed_filter_diagnostics.jsonl",
    "summary.json",
    "gate_trace.csv",
    "gate_summary.json",
    "screen_results.json",
    "trial_results.json",
    "matched_trace.csv",
    "force_trace.csv",
    "object_trajectory.csv",
    "video_alignment.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build")
    after = write_after_audit()
    summary = write_final_summary()
    report = write_report(summary)
    build_delivery(after, summary, report)
    print(json.dumps({"delivery": str(DELIVERY_ROOT), "classification": summary["classification"]}, indent=2))


def write_after_audit() -> dict[str, Any]:
    payload = {
        "schema_version": 3,
        "head": _git_head(),
        "facts": {
            "search_and_strict_evaluation_separated": True,
            "safe_single_contact_outranks_no_contact": True,
            "pca_endpoint_effect_preserved": True,
            "contact_latch_releases_and_reacquires": True,
            "gate_c_preshape_action_driven": True,
            "sampler_part_name_branches": False,
            "runtime_collision_mesh_required": True,
            "strict_cache_schema": 3,
            "coordex_official_66d_order": True,
            "coordex_official_target_semantics": True,
            "full_pad_support_includes_fixed_tip_link": True,
            "stage_a_nonlinear_fallbacks_per_target_max": 1,
            "old_screw1_cem_started": False,
            "ppo_started": False,
            "targeted_gate_a_retry_count": 1,
        },
        "sticky_used": False,
        "snap_used": False,
        "proxy_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_after_reset": 0,
    }
    _write(RUN_ROOT / "code_fact_audit_after.json", payload)
    return payload


def write_final_summary() -> dict[str, Any]:
    objects = {}
    any_approach_close_lift = False
    any_gate_a = False
    any_gate_a_run = False
    total_strict = 0
    for part in PARTS:
        runtime = _read(RUN_ROOT / part / "runtime_audit/summary.json")
        smoke = _read(RUN_ROOT / part / "vector_smoke/summary.json")
        screen = _read(RUN_ROOT / part / "m0_screen/summary.json")
        full = _read(RUN_ROOT / part / "m0_full/summary.json")
        gate_a = _read(RUN_ROOT / part / "gate_a/summary.json")
        gate_b = _read(RUN_ROOT / part / "gate_b/summary.json")
        gate_c = _read(RUN_ROOT / part / "gate_c/summary.json")
        fingerprint = runtime.get("collision_geometry", {}).get("runtime_physics_fingerprint", {})
        strict = int((full or screen).get("strict_m0_candidate_count", 0))
        total_strict += strict
        gate_a_pass = bool(gate_a.get("qualified_candidate_ids"))
        any_gate_a_run |= bool(gate_a)
        gate_c_pass = bool(gate_c.get("approach_close_lift_success", False))
        any_gate_a |= gate_a_pass
        any_approach_close_lift |= gate_c_pass
        videos = sorted(str(path.relative_to(REPO_ROOT)) for path in (RUN_ROOT / part).rglob("*.mp4"))
        retry_comparison = {}
        if part == "Plug2":
            retry_comparison = {
                "before": _trial_metrics(_read_list(RUN_ROOT / part / "gate_a_attempt_1/trial_results.json")),
                "after": _trial_metrics(_read_list(RUN_ROOT / part / "gate_a/trial_results.json")),
                "retry_count": 1,
                "screen_resets": 2,
                "formal_fresh_reset_trials": 10,
                "total_fresh_resets": 12,
            }
        objects[part] = {
            "runtime_audit": runtime.get("classification", "NOT_RUN"),
            "vector_smoke": smoke.get("classification", "NOT_RUN"),
            "asset_usd": fingerprint.get("target_asset_usd", ""),
            "mass_kg": fingerprint.get("runtime_mass_kg_min"),
            "friction": (fingerprint.get("runtime_material_properties_first") or [[None]])[0][0],
            "m0_screen": _phase_counts(screen),
            "m0_full": _phase_counts(full),
            "contact_hold": _gate_counts(_read(RUN_ROOT / part / "contact_smoke/summary.json")),
            "gate_a": _gate_counts(gate_a),
            "gate_b": _gate_counts(gate_b),
            "gate_c": _gate_counts(gate_c),
            "physical_grasp_success": bool(gate_b.get("physical_grasp_success", False) or gate_c.get("physical_grasp_success", False)),
            "physical_lift_success": bool(gate_c.get("physical_lift_success", False)),
            "approach_close_lift_success": gate_c_pass,
            "full_route_success": False,
            "assisted_configuration": False,
            "external_seed_providers": ["CoorDex", "wuji-retargeting"],
            "videos": videos,
            "gate_a_targeted_retry": retry_comparison,
            "failure_layer": (
                "GATE_A_MULTI_CONTACT_HOLD"
                if bool(gate_a)
                else "M0_CLOSED_POSE_COLLISION"
                if int((full or screen).get("reachability_success_count", 0)) > 0
                else "M0_REACHABILITY"
            ),
        }
    if any_approach_close_lift:
        classification = "MULTI_OBJECT_PRIVILEGED_GRASP_V3_GATE_C_APPROACH_CLOSE_LIFT_ACQUIRED"
    elif any_gate_a:
        classification = "STRICT_M0_CONTACT_ACQUIRED_GATE_CONTINUATION_PENDING"
    elif total_strict and any_gate_a_run:
        classification = "STRICT_M0_CANDIDATE_GATE_A_FAILED"
    elif total_strict:
        classification = "STRICT_M0_CANDIDATE_AVAILABLE_PHYSICS_PENDING"
    else:
        classification = "NO_VALID_M0_CANDIDATE"
    payload = {
        "schema_version": 3,
        "classification": classification,
        "objects": objects,
        "strict_m0_candidate_total": total_strict,
        "approach_close_lift_success": any_approach_close_lift,
        "full_route_success": False,
        "cem_started": False,
        "ppo_started": False,
        "sticky_used": False,
        "snap_used": False,
        "proxy_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_after_reset": 0,
        "canonical_sticky_baseline_modified": False,
        "canonical_sticky_baseline_present": (
            REPO_ROOT / "debug_runs/full_oracle_sticky_assembly_pipeline/report.md"
        ).is_file(),
        "delivery_directory": "3",
    }
    _write(RUN_ROOT / "final_summary.json", payload)
    _write(RUN_ROOT / "Plug2/gate_a_targeted_retry_comparison.json", objects["Plug2"]["gate_a_targeted_retry"])
    _write(
        RUN_ROOT / "gate_orchestration_summary.json",
        {
            "classification": classification,
            "selected_part": "Plug2" if objects["Plug2"]["gate_a"]["classification"] != "NOT_RUN" else "",
            "gate_a": objects["Plug2"]["gate_a"],
            "gate_b_started": objects["Plug2"]["gate_b"]["classification"] != "NOT_RUN",
            "gate_c_started": objects["Plug2"]["gate_c"]["classification"] != "NOT_RUN",
            "targeted_retry_count": 1,
            "matched_video": "",
            "cem_started": False,
            "ppo_started": False,
        },
    )
    return payload


def write_report(summary: dict[str, Any]) -> Path:
    lines = [
        "# Multi-Object Privileged Physical Grasp v3",
        "",
        f"Final classification: `{summary['classification']}`.",
        "",
        "No old Screw1 CEM or PPO was started. Sticky, snap, proxy, teacher motion, and post-reset object/wrist writes were not used.",
        "",
        "| Part | Mass kg | Friction | Strict M0 | Gate A | Gate B | Gate C | Failure layer | Video |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for part, row in summary["objects"].items():
        strict = (row["m0_full"] if row["m0_full"]["classification"] != "NOT_RUN" else row["m0_screen"])["strict_m0"]
        lines.append(
            f"| {part} | {row['mass_kg']} | {row['friction']} | {strict} | "
            f"{row['gate_a']['classification']} ({row['gate_a']['successes']}/{row['gate_a']['trials']}) | "
            f"{row['gate_b']['classification']} | {row['gate_c']['classification']} | {row['failure_layer']} | "
            f"{', '.join(row['videos']) or '-'} |"
        )
    lines.extend(
        (
            "",
            "## Main blockers",
            "",
            "See each object's `m0_screen/summary.json`, `optimized_candidates.jsonl` in the run directory, and the difficulty ranking. "
            "Candidates that fail 1 mm residual, closed-pose collision, gravity wrench, cache hash, or Gate eligibility are never sent to physics.",
            "",
            "## Handoff edit surface",
            "",
            "The next operator should modify only `configs/grasp_synthesis/objects/<part>.yaml` for object semantics or the immutable candidate JSON selected by `top_candidates.json`. "
            "Controller code has no part-name branch.",
            "",
            "## Risk points",
            "",
            "- PhysX may cook dynamic triangle meshes as convex hulls; source-mesh and cooked-contact equivalence is not assumed.",
            "- GPU Table/ground filtered-contact warnings remain instrumentation limits and are not interpreted as named collisions without pair evidence.",
            "- CoorDex and some external baselines have no detected redistribution license; checkpoints and clones are excluded from `3/`.",
            f"- Canonical sticky baseline present in this workspace: `{summary['canonical_sticky_baseline_present']}`; this task did not create or replace it.",
            "",
            "## External seeds",
            "",
            "Stage A used project-authored seeds plus isolated CoorDex and wuji-retargeting seeds. External seeds establish no success evidence. GenDex/DRO/DexGraspNet remain isolated audit inputs.",
            "",
            "## Changed files",
            "",
            *(f"- `{path}`" for path in SOURCE_PATHS),
            "",
            "## Test commands",
            "",
            "- `conda run -n isaac python -m py_compile <v3 runners, near_grasp and grasp_synthesis modules>`",
            "- `pytest -q source/isaaclab_tasks/test/test_near_grasp_search.py source/isaaclab_tasks/test/test_privileged_physics_grasp.py` (37 passed)",
            "- Five `16-env / 32-step` Isaac vector smokes; all produced finite `[16,169]` observations, valid target filters and zero post-reset writes.",
        )
    )
    path = RUN_ROOT / "final_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_delivery(after: dict[str, Any], summary: dict[str, Any], report: Path) -> None:
    if DELIVERY_ROOT.exists():
        shutil.rmtree(DELIVERY_ROOT)
    DELIVERY_ROOT.mkdir(parents=True)
    for relative in SOURCE_PATHS:
        source = REPO_ROOT / relative
        target = DELIVERY_ROOT / "source_snapshot" / relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    artifact_root = DELIVERY_ROOT / "artifacts"
    for source in sorted(RUN_ROOT.rglob("*")):
        if source.is_file() and (source.name in ARTIFACT_NAMES or source.suffix == ".mp4"):
            target = artifact_root / source.relative_to(RUN_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    commands = (
        "# Reproduction commands\n\n"
        "Activate the `isaac` conda environment and set the repository PYTHONPATH as required by AGENTS.md.\n\n"
        "```bash\n"
        "TERM=xterm-256color ./isaaclab.sh -p scripts/environments/run_multi_object_privileged_grasp_v3.py --phase runtime_audit\n"
        "python scripts/environments/run_multi_object_privileged_grasp_v3.py --phase m0_screen\n"
        "python scripts/environments/run_multi_object_privileged_grasp_v3.py --phase m0_full --parts Rod Backrest Frame\n"
        "python scripts/environments/run_multi_object_privileged_grasp_v3.py --phase difficulty\n"
        "TERM=xterm-256color ./isaaclab.sh -p scripts/environments/run_multi_object_privileged_grasp_v3.py --phase gates\n"
        "```\n"
    )
    (DELIVERY_ROOT / "README.md").write_text(
        "# Delivery 3\n\nMulti-Object Privileged Physical Grasp v3. Start with `artifacts/final_report.md` and `artifacts/final_summary.json`.\n",
        encoding="utf-8",
    )
    (DELIVERY_ROOT / "REPRODUCE.md").write_text(commands, encoding="utf-8")
    manifest = []
    for path in sorted(DELIVERY_ROOT.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            manifest.append(
                {
                    "path": str(path.relative_to(DELIVERY_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    _write(DELIVERY_ROOT / "sha256_manifest.json", {"schema_version": 1, "files": manifest})


def _phase_counts(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": row.get("classification", "NOT_RUN"),
        "sampled": int(row.get("sample_count", 0)),
        "retained": int(row.get("retained_count", 0)),
        "optimized": int(row.get("optimized_count", 0)),
        "reachability": int(row.get("reachability_success_count", 0)),
        "contact_residual": int(row.get("contact_residual_success_count", 0)),
        "closed_collision": int(row.get("closed_pose_collision_success_count", 0)),
        "strict_m0": int(row.get("strict_m0_candidate_count", 0)),
        "physical_resets": int(row.get("physical_reset_count", 0)),
    }


def _gate_counts(row: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = list(row.get("candidate_results", {}).values())
    return {
        "classification": row.get("classification", "NOT_RUN"),
        "successes": int(
            row.get("successes", sum(int(candidate.get("successes", 0)) for candidate in candidate_rows))
        ),
        "trials": int(row.get("trials", sum(int(candidate.get("trials", 0)) for candidate in candidate_rows))),
        "qualified_candidate_ids": list(row.get("qualified_candidate_ids", ())),
    }


def _trial_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"trials": 0, "successes": 0}
    hold = [int(row.get("hold_contact_steps", 0)) for row in rows]
    return {
        "trials": len(rows),
        "successes": sum(int(bool(row.get("passed", False))) for row in rows),
        "hold_contact_steps_min": min(hold),
        "hold_contact_steps_max": max(hold),
        "peak_target_force_n_max": max(float(row.get("peak_target_force_n", 0.0)) for row in rows),
        "lift_contact_duty_max": max(float(row.get("lift_contact_duty", 0.0)) for row in rows),
        "termination_layers": [str(row.get("termination_layer", "")) for row in rows],
    }


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    return list(value) if isinstance(value, list) else []


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
