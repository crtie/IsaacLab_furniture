"""Build the final Plug2 Gate A forensic v2 comparison and report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CANDIDATES = REPO_ROOT / (
    "state_banks/grasp_synthesis/Plug2/"
    "d92179a09d69c7150262035dd99895b3d9570c868cde11b4ff44d0c1c37b7882/candidates.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="debug_runs/plug2_gate_a_forensic_v2")
    args = parser.parse_args()
    root = (REPO_ROOT / args.output_dir).resolve()
    part_root = root / "Plug2"
    synth_dir = part_root / "synthesize_forensic_v2"
    gate_dir = part_root / "gate_a_forensic_v2"
    synth = _load(synth_dir / "summary.json")
    gate = _load(gate_dir / "gate_a_forensic_summary.json")
    shortlist = _load(gate_dir / "gate_a_physics_shortlist.json")["candidates"]
    collision = _load(part_root / "forensic_audit/runtime_collision_geometry_audit.json")
    fk = _load(gate_dir / "fk_cross_engine_summary.json")
    old = [json.loads(line) for line in LEGACY_CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()][:16]
    comparison = [_comparison_row(rank, old[rank], shortlist[rank]) for rank in range(16)]
    _write_json(root / "legacy_vs_forensic_candidate_geometry.json", comparison)
    _write_csv(root / "legacy_vs_forensic_candidate_geometry.csv", [_csv_row(row) for row in comparison])
    first_frames = _load(gate_dir / "first_frame_collision_diagnostics.json")
    stale_peaks = []
    for result_path in gate_dir.glob("candidates/*/reset_*/result.json"):
        stale_peaks.append(float(_load(result_path)["metadata"]["reset_cache_audit"]["unfiltered_force_peak_n"]))
    termination_counts = Counter(
        termination
        for values in gate["candidate_results"].values()
        for termination in values["termination_layers"]
    )
    peak_values = [
        float(value)
        for values in gate["candidate_results"].values()
        for value in values["peak_target_force_n"]
    ]
    tip_errors = [float(candidate["tip_error_m"]) for candidate in shortlist]
    report = _report_text(
        synth=synth,
        gate=gate,
        collision=collision,
        fk=fk,
        first_frames=first_frames,
        stale_peaks=stale_peaks,
        termination_counts=termination_counts,
        peak_values=peak_values,
        tip_errors=tip_errors,
    )
    (root / "report.md").write_text(report, encoding="utf-8")
    final = {
        "classification": "PLUG2_GATE_A_FORENSIC_V2_FAILED",
        "sample_count": synth["sample_count"],
        "retained_count": synth["retained_count"],
        "optimization_call_count": synth["optimization_call_count"],
        "physics_candidate_count": gate["candidate_count"],
        "sequential_fresh_reset_count": gate["sequential_fresh_reset_count"],
        "qualified_candidate_ids": gate["qualified_candidate_ids"],
        "termination_counts": dict(termination_counts),
        "tip_error_m_min": min(tip_errors),
        "tip_error_m_max": max(tip_errors),
        "target_force_peak_n_max": max(peak_values),
        "fk_global_position_error_mm": fk["global_position_error_mm"],
        "fk_global_orientation_error_deg": fk["global_orientation_error_deg"],
        "matched_video": "",
        "cem_started": False,
        "ppo_started": False,
        "sticky_used": False,
        "snap_used": False,
        "proxy_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": True,
        "root_pose_writes_reset_only": True,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
    }
    _write_json(root / "final_summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


def _comparison_row(rank: int, legacy: dict[str, Any], forensic: dict[str, Any]) -> dict[str, Any]:
    old_contacts = np.asarray(legacy["contact_positions_object"], dtype=np.float64)
    old_normals = np.asarray(legacy["contact_normals_object"], dtype=np.float64)
    new_meta = forensic["metadata"]
    return {
        "rank": rank,
        "legacy_candidate_id": legacy["candidate_id"],
        "forensic_candidate_id": forensic["candidate_id"],
        "legacy_finger_group": legacy["finger_group"],
        "forensic_finger_group": forensic["finger_group"],
        "legacy_axis_delta_m_recomputed": float(np.ptp(old_contacts[:, 2])) if len(old_contacts) else "not_available",
        "forensic_axis_delta_m": new_meta["axis_delta_m"],
        "legacy_normal_min_dot_recomputed": _minimum_normal_dot(old_normals),
        "forensic_normal_min_dot_recomputed": _minimum_normal_dot(np.asarray(forensic["contact_normals_object"])),
        "legacy_raycast_hit": "not_recorded",
        "forensic_raycast_hit": new_meta["raycast_hit"],
        "legacy_gravity_wrench": "not_recorded",
        "forensic_gravity_wrench": new_meta["gravity_wrench"],
        "legacy_support_model": {
            "type": "scalar_only",
            "pad_support_distance_m": legacy.get("metadata", {}).get("pad_support_distance_m", "not_recorded"),
        },
        "forensic_support_model": {
            "type": "full_3d_vertex",
            "pad_support_vertex_local": new_meta["pad_support_vertex_local"],
            "pad_support_world_offset": new_meta["pad_support_world_offset"],
        },
        "legacy_hand_object_penetration": {
            "value": legacy.get("energy", {}).get("hand_object_penetration", "not_recorded"),
            "trust": "hardcoded_zero_not_physical_evidence",
        },
        "forensic_hand_object_penetration": {
            "value": forensic["energy"]["hand_object_penetration"],
            "scene_clearance": new_meta["scene_clearance"],
        },
        "legacy_tip_error_m": legacy["tip_error_m"],
        "forensic_tip_error_m": forensic["tip_error_m"],
        "legacy_gate_a_eligible": legacy["gate_eligibility"]["gate_a"],
        "forensic_gate_a_eligible": forensic["gate_eligibility"]["gate_a"],
        "forensic_optimizer_reported_success": new_meta["optimizer_reported_success"],
    }


def _minimum_normal_dot(normals: np.ndarray) -> float | str:
    if len(normals) < 2:
        return "not_available"
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return float(min(np.dot(normals[first], normals[second]) for first in range(len(normals)) for second in range(first + 1, len(normals))))


def _report_text(**values: Any) -> str:
    synth = values["synth"]
    gate = values["gate"]
    collision = values["collision"]
    fk = values["fk"]
    termination = values["termination_counts"]
    tip_errors = values["tip_errors"]
    peaks = values["peak_values"]
    stale = values["stale_peaks"]
    target = collision["target_object"]
    physics = collision["runtime_physics_fingerprint"]
    return f"""# Plug2 Gate A Forensic v2 Report

## Final Status

- `classification=PLUG2_GATE_A_FORENSIC_V2_FAILED`
- Gate A: 16 candidates x 5 sequential fresh resets = {gate['sequential_fresh_reset_count']} trials, all `0/5`
- Gate B/C, other objects, CEM and PPO were not started
- `physical_grasp_success=false`, `physical_lift_success=false`, `full_route_success=false`
- `matched_video=\"\"`; no MP4 was generated because no candidate reached Gate A `3/5`

## Candidate Synthesis

The completed synthesis used exactly `{synth['sample_count']}/{synth['retained_count']}/{synth['optimization_call_count']}/{synth['physics_candidate_count']}` with seed `20260713`. Raycast pairs were constrained to `<=0.5mm` axial mismatch and accepted only after an 8-edge friction-cone gravity-wrench LP.

No bounded SLSQP call reached the `<=3mm` contact target criterion. The 16 unique forensic poses had tip residuals `{min(tip_errors) * 1000.0:.2f}-{max(tip_errors) * 1000.0:.2f}mm`; they entered Gate A as forensic scene-clearance probes, not as validated optimized grasps.

Two pre-cache performance probes were aborted and archived after exposing impractical full-table/26D numerical-gradient cost. Neither wrote a candidate cache or Gate result. The completed run used 6D wrist plus active-finger optimization and a hashed crop of actual Table collision triangles.

## Gate A Evidence

- Terminations: `{dict(termination)}`
- Maximum target-filtered force: `{max(peaks):.3f}N`
- Candidate `359fd6eefd52e35b` produced five hard aborts with `5.47-13.10N`
- Other candidates failed multi-contact hold; most had zero force, while isolated safe single-finger peaks never established simultaneous contact
- Every trace starts at physics frame 0 and ends at its terminal frame; normal traces contain 124 frames and hard-abort traces are truncated earlier

Reset-forward diagnostic caches contained stale unfiltered peaks up to `{max(stale):.1f}N`. After the permitted pre-physics cache clear, recorded frame 0 values came from a real physics step. These stale values are not counted as collision or grasp evidence.

## Geometry And FK

- Runtime Plug2 collision mesh: `{target['vertex_count']}` vertices, `{target['face_count']}` faces, AABB `{target['aabb_min']} -> {target['aabb_max']}`
- Asset: `{physics['target_asset_usd']}`, scale `{physics['target_asset_scale']}`, mass `{physics['runtime_mass_kg_min']:.3f}kg`
- Runtime material vector: `{physics['runtime_material_properties_first'][0]}`
- Isaac/Pinocchio mean position error: `{fk['global_position_error_mm']['mean']:.9f}mm`; max `{fk['global_position_error_mm']['max']:.9f}mm`
- Isaac/Pinocchio mean orientation error: `{fk['global_orientation_error_deg']['mean']:.9f}deg`; max `{fk['global_orientation_error_deg']['max']:.9f}deg`

The cross-engine FK mapping is therefore consistent. The remaining blocker is candidate contact-target reach, followed by unsafe force on one candidate, not FK body mapping.

## Honesty And Assistance

- `sticky_used=false`, `snap_used=false`, `proxy_used=false`, `teacher_motion_used=false`
- Reset initialization writes were used; post-reset object and wrist state writes were zero
- Gate A remains `ORACLE RESET GRASP` and cannot claim autonomous physical grasp acquisition
- Unresolved residuals were not assigned to a specific collision body

## Key Artifacts

- `legacy_vs_forensic_candidate_geometry.csv/json`
- `Plug2/forensic_audit/runtime_collision_geometry_audit.json`
- `Plug2/gate_a_forensic_v2/first_frame_collision_diagnostics.csv/json`
- `Plug2/gate_a_forensic_v2/fk_cross_engine_summary.csv/json`
- `Plug2/gate_a_forensic_v2/candidates/<candidate>/reset_<0..4>/trace.csv`
- `Plug2/gate_a_forensic_v2/gate_a_forensic_summary.json`

## Risk Points

1. All SLSQP candidates remained 20-41mm from their intended pad contact targets; geometry eligibility alone must not be reused as optimized-grasp validity.
2. GPU Table/ground filtered-force warnings remain. Actor-pair events and explicit instrumentation-limit states are retained in every trace.
3. The one force-producing hard-abort candidate proves that scene-clearance checks do not by themselves bound dynamic reset impact.
"""


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        for key, value in row.items()
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
