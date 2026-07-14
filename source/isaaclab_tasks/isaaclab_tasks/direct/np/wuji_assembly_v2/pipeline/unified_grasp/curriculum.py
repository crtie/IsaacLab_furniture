"""Curriculum state construction for v80."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_csv, write_json


CURRICULUM_MODES = [
    "near_contact_seed",
    "preclose_seed",
    "support_stabilization_seed",
    "randomized_pregrasp",
    "full_approach",
]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_curriculum_states(
    *,
    seeds: list[dict[str, Any]] | None = None,
    priors: list[dict[str, Any]] | None = None,
    parts: list[str] | None = None,
    physics_profile: str = "canonical",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed_parts = parts or V80_PARTS
    for seed in seeds or []:
        part = str(seed.get("part_name") or "")
        if part not in allowed_parts or not _bool(seed.get("allowed_for_curriculum")):
            continue
        contact_count = int(float(seed.get("contact_count") or 0))
        support_ok = _bool(seed.get("support_gate_ok"))
        mode = str(seed.get("curriculum_mode") or "")
        if not mode or mode == "blocked":
            if support_ok:
                mode = "support_stabilization_seed"
            elif contact_count > 0:
                mode = "near_contact_seed"
            else:
                mode = "preclose_seed"
        rows.append(
            {
                "curriculum_state_id": f"seed_{len(rows)}",
                "curriculum_mode": mode,
                "curriculum_only": True,
                "allowed_for_final_video": False,
                "allowed_for_training_export": False,
                "sticky_disabled_during_training": True,
                "source_seed_id": seed.get("seed_id", ""),
                "source_prior_id": "",
                "part_name": part,
                "physics_profile": physics_profile,
                "contact_count": contact_count,
                "support_gate_ok": support_ok,
                "active_max_m": float(seed.get("active_max_m") or 0.0),
                "allowed_for_rl_training": True,
                "curriculum_allowed_reason": seed.get("curriculum_allowed_reason", ""),
                "curriculum_reject_reason": "",
            }
        )
    for prior in priors or []:
        part = str(prior.get("part_name") or "")
        if part not in allowed_parts or not _bool(prior.get("reachable_prior_physical")):
            continue
        rows.append(
            {
                "curriculum_state_id": f"prior_{len(rows)}",
                "curriculum_mode": "randomized_pregrasp",
                "curriculum_only": True,
                "allowed_for_final_video": False,
                "allowed_for_training_export": False,
                "sticky_disabled_during_training": True,
                "source_seed_id": "",
                "source_prior_id": prior.get("prior_id", ""),
                "part_name": part,
                "physics_profile": physics_profile,
                "contact_count": 0,
                "support_gate_ok": False,
                "active_max_m": float(prior.get("reachability_error_m") or 0.05),
                "allowed_for_rl_training": True,
                "curriculum_allowed_reason": prior.get("prior_validation_source", ""),
                "curriculum_reject_reason": "",
            }
        )
    return rows


def write_curriculum_artifacts(
    run_dir: str | Path,
    *,
    seeds: list[dict[str, Any]] | None = None,
    priors: list[dict[str, Any]] | None = None,
    parts: list[str] | None = None,
    physics_profile: str = "canonical",
) -> dict[str, Any]:
    rows = build_curriculum_states(seeds=seeds, priors=priors, parts=parts, physics_profile=physics_profile)
    run_path = Path(run_dir)
    bank_csv = write_csv(run_path / "curriculum_state_bank.csv", rows)
    bank_json = write_json(run_path / "curriculum_state_bank.json", rows)
    schedule = {
        "curriculum_modes": CURRICULUM_MODES,
        "stage_order": ["near_contact_seed", "preclose_seed", "randomized_pregrasp", "full_approach"],
        "sticky_disabled_during_training": True,
        "state_count": len(rows),
    }
    schedule_path = write_json(run_path / "curriculum_schedule.json", schedule)
    return {
        "rows": rows,
        "curriculum_state_bank_csv": str(bank_csv),
        "curriculum_state_bank_json": str(bank_json),
        "curriculum_schedule_json": str(schedule_path),
    }
