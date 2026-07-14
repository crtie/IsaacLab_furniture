"""Shared action-driven acquisition-plan replay contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_acquisition_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"acquisition plan must be a JSON object: {plan_path}")
    required = ("plan_id", "acquisition_mode", "staged_wrist_waypoints", "final_target_pos_xyz")
    missing = [name for name in required if not data.get(name)]
    if missing:
        raise ValueError(f"acquisition plan missing fields: {','.join(missing)}")
    return data


def execute_acquisition_plan(
    env: Any,
    base: Any,
    env_index: int,
    v2_config: Any,
    alignment: Any,
    trace_rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    phase: str,
    trial_id: int | str,
) -> dict[str, Any]:
    """Replay through the audited v2 action path without live state writes."""

    from .screw1_grasp_baseline_v2 import _execute_acquisition_plan

    return _execute_acquisition_plan(
        env,
        base,
        env_index,
        v2_config,
        alignment,
        trace_rows,
        plan=plan,
        phase=phase,
        trial_id=trial_id,
    )


def translated_plan(
    source: dict[str, Any],
    *,
    plan_id: str,
    translation_xyz_m: tuple[float, float, float],
    active_finger_group: str,
) -> dict[str, Any]:
    """Translate only the final approach portion of a verified corridor plan."""

    plan = copy.deepcopy(source)
    delta = tuple(float(item) for item in translation_xyz_m)
    changed = 0
    for stage in list(plan.get("staged_wrist_waypoints") or []):
        metadata = dict(stage.get("metadata") or {})
        kind = str(metadata.get("seed_replay_stage_kind") or "")
        if kind not in {"final_contact_approach", "standoff_orientation"}:
            continue
        pos = list(stage.get("target_pos_xyz") or [])
        if len(pos) >= 3:
            stage["target_pos_xyz"] = [float(pos[index]) + delta[index] for index in range(3)]
            changed += 1
    final_pos = list(plan.get("final_target_pos_xyz") or [])
    if len(final_pos) >= 3:
        plan["final_target_pos_xyz"] = [float(final_pos[index]) + delta[index] for index in range(3)]
    if changed == 0:
        raise ValueError("source acquisition plan has no translatable final approach stages")
    plan["plan_id"] = str(plan_id)
    plan["active_finger_group"] = str(active_finger_group)
    plan["source_plan_id"] = str(source.get("plan_id") or "")
    plan["plan_translation_xyz_m"] = list(delta)
    plan["geometry_selection_only"] = True
    plan["success_claimed"] = False
    return plan
