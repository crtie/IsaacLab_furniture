"""Analytic grasp prior generator for v80."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_csv, write_json


PRIOR_KINDS = {
    "Plug2": ("two_sided_pinch", "straddle_pinch"),
    "Screw1": ("tight_pinch", "small_object_enclosure"),
    "Backrest": ("support_patch", "edge_cradle"),
    "Rod": ("rod_axis_cradle", "three_finger_enclosure"),
    "Frame": ("frame_bar_hook", "contact_patch"),
}

ACTIVE_GROUPS = {
    "Plug2": ("23", "34"),
    "Screw1": ("34", "23", "45"),
    "Backrest": ("34", "23"),
    "Rod": ("234", "345"),
    "Frame": ("234", "34"),
}


def _prior_score(part: str, kind_index: int, group_index: int) -> dict[str, float]:
    force_closure = max(0.0, 0.92 - 0.06 * kind_index - 0.03 * group_index)
    support = max(0.0, 0.86 - 0.04 * kind_index + (0.05 if part in {"Backrest", "Rod", "Frame"} else 0.0))
    reach_error = 0.0035 + 0.0015 * kind_index + 0.001 * group_index
    penetration = 0.0004 * kind_index
    clearance = 0.010 - 0.001 * group_index
    approach = 0.12 + 0.06 * kind_index + 0.02 * group_index
    return {
        "predicted_force_closure_score": round(force_closure, 5),
        "opposed_normal_score": round(force_closure - 0.05, 5),
        "contact_geometry_support_score": round(support, 5),
        "reachability_error_m": round(reach_error, 6),
        "penetration_risk": round(penetration, 6),
        "table_clearance_m": round(clearance, 6),
        "approach_collision_risk": round(approach, 5),
    }


def generate_grasp_priors(
    *,
    parts: list[str] | None = None,
    seeds: list[dict[str, Any]] | None = None,
    top_k_per_part: int = 32,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in parts or V80_PARTS:
        part_rows: list[dict[str, Any]] = []
        for kind_index, kind in enumerate(PRIOR_KINDS[part]):
            for group_index, group in enumerate(ACTIVE_GROUPS[part]):
                score = _prior_score(part, kind_index, group_index)
                part_rows.append(
                    {
                        "prior_id": f"{part}_analytic_{kind}_{group}",
                        "part_name": part,
                        "prior_source": "analytic_geometry",
                        "prior_type": kind,
                        "active_finger_group": group,
                        "surface_source": "analytic",
                        "object_center_proxy_used": False,
                        "seed_success_claimed": False,
                        "usable_training_row": False,
                        "analytic_prior_unvalidated": True,
                        "prior_validated_by_physics": False,
                        "prior_validation_source": "analytic_geometry_unvalidated",
                        "reachable_prior": False,
                        "reachable_prior_physical": False,
                        "reachable_prior_analytic_only": score["reachability_error_m"] <= (0.008 if part != "Screw1" else 0.006),
                        **score,
                    }
                )
        for seed in seeds or []:
            if seed.get("part_name") != part or seed.get("disallowed_evidence_detected"):
                continue
            reachable_physical = bool(seed.get("allowed_for_curriculum"))
            part_rows.append(
                {
                    "prior_id": f"{part}_recovered_{seed.get('seed_id')}",
                    "part_name": part,
                    "prior_source": "recovered_honest_seed",
                    "prior_type": "near_contact_seed",
                    "active_finger_group": str(seed.get("active_finger_group") or ""),
                    "surface_source": "serial_replay_artifact",
                    "object_center_proxy_used": False,
                    "seed_success_claimed": False,
                    "usable_training_row": False,
                    "analytic_prior_unvalidated": False,
                    "prior_validated_by_physics": reachable_physical,
                    "prior_validation_source": "serial_replay_seed_curriculum_gate" if reachable_physical else "serial_replay_seed_rejected",
                    "reachable_prior": reachable_physical,
                    "reachable_prior_physical": reachable_physical,
                    "reachable_prior_analytic_only": False,
                    "predicted_force_closure_score": 0.35 + 0.1 * min(int(seed.get("contact_count") or 0), 3),
                    "opposed_normal_score": 0.3,
                    "contact_geometry_support_score": 0.3,
                    "reachability_error_m": float(seed.get("active_max_m") or 0.05),
                    "penetration_risk": 0.0,
                    "table_clearance_m": 0.006,
                    "approach_collision_risk": 0.5,
                }
            )
        part_rows.sort(
            key=lambda row: (
                not bool(row["reachable_prior_physical"]),
                not bool(row["reachable_prior"]),
                float(row["approach_collision_risk"]),
                float(row["reachability_error_m"]),
                -float(row["predicted_force_closure_score"]),
            )
        )
        rows.extend(part_rows[: max(1, int(top_k_per_part))])
    return rows


def write_grasp_prior_bank(
    run_dir: str | Path,
    *,
    parts: list[str] | None = None,
    seeds: list[dict[str, Any]] | None = None,
    top_k_per_part: int = 32,
) -> dict[str, Any]:
    rows = generate_grasp_priors(parts=parts, seeds=seeds, top_k_per_part=top_k_per_part)
    run_path = Path(run_dir)
    csv_path = write_csv(run_path / "grasp_prior_bank_v80.csv", rows)
    json_path = write_json(run_path / "grasp_prior_bank_v80.json", rows)
    return {
        "rows": rows,
        "grasp_prior_bank_v80_csv": str(csv_path),
        "grasp_prior_bank_v80_json": str(json_path),
    }
