"""Unified seed bank loader for v80 curriculum starts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_csv, write_json


DISALLOWED_SEED_TOKENS = ("oracle", "teacher", "cache", "dirty", "proxy")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _part_from_path_or_row(path: Path, row: dict[str, Any]) -> str:
    for key in ("part_name", "part", "object_name"):
        value = str(row.get(key) or "")
        if value:
            if value.lower() == "plug":
                return "Plug2"
            if value.lower() == "screw":
                return "Screw1"
            return value
    text = str(path).lower()
    if "backrest" in text:
        return "Backrest"
    if "screw" in text:
        return "Screw1"
    if "plug" in text:
        return "Plug2"
    if "rod" in text:
        return "Rod"
    if "frame" in text:
        return "Frame"
    return ""


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _is_disallowed(row: dict[str, Any]) -> bool:
    haystack = " ".join(str(value).lower() for value in row.values())
    return any(token in haystack for token in DISALLOWED_SEED_TOKENS)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _curriculum_gate(*, row: dict[str, Any], disallowed: bool, contact_count: int, active_max_m: float, support_ok: bool) -> tuple[bool, str, str, str]:
    if disallowed:
        return False, "", "disallowed_fake_or_teacher_like_evidence", "blocked"
    source_text = " ".join(str(row.get(key, "")).lower() for key in ("seed_source", "source", "status", "repair_source", "source_path"))
    approach_repair = "approach_repair" in source_text or "guarded_approach" in source_text
    if support_ok:
        return True, "serial_support_gate_pass", "", "support_stabilization_seed"
    if contact_count >= 1 and active_max_m <= 0.020:
        return True, "near_contact_with_real_contact_count", "", "near_contact_seed"
    if approach_repair and contact_count >= 1:
        return True, "approach_repair_curriculum_only", "", "approach_repair_curriculum_only"
    if approach_repair and contact_count <= 0:
        return False, "", "zero_contact_failed_approach_repair", "blocked"
    return False, "", "no_support_or_near_contact_evidence", "blocked"


def _seed_from_row(path: Path, row: dict[str, Any], index: int, source: str) -> dict[str, Any] | None:
    part = _part_from_path_or_row(path, row)
    if part not in V80_PARTS:
        return None
    disallowed = _is_disallowed(row)
    contact_count = int(_float_value(row.get("effective_contact_count") or row.get("serial_effective_contact_count"), 0.0))
    active = _float_value(
        row.get("active_max_m")
        or row.get("serial_active_max_m")
        or row.get("active_max_dist_m")
        or row.get("best_active_max_m"),
        1.0,
    )
    support_ok = _bool_value(row.get("support_gate_ok") or row.get("seed_serial_support_ok") or row.get("serial_support_gate_ok"))
    allowed, allow_reason, reject_reason, mode = _curriculum_gate(
        row=row,
        disallowed=disallowed,
        contact_count=contact_count,
        active_max_m=active,
        support_ok=support_ok,
    )
    return {
        "seed_id": f"{source}_{index}",
        "part_name": part,
        "seed_source": source,
        "source_path": str(path),
        "contact_count": contact_count,
        "active_max_m": active,
        "support_gate_ok": support_ok,
        "seed_success_claimed": False,
        "usable_training_row": False,
        "disallowed_evidence_detected": disallowed,
        "allowed_for_curriculum": bool(allowed),
        "curriculum_allowed_reason": allow_reason,
        "curriculum_reject_reason": reject_reason,
        "curriculum_mode": mode,
        "allowed_for_final": False,
        "raw": row,
    }


def load_unified_seed_bank(
    run_dir: str | Path,
    *,
    search_roots: list[str | Path] | None = None,
    parts: list[str] | None = None,
) -> dict[str, Any]:
    roots = [Path(root) for root in (search_roots or ["debug_runs"])]
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    patterns = ["target_seed_serial_replay.csv", "executable_seed_bank.csv", "best_grasp_program.json", "grasp_optimizer_best_replay_packet.json"]
    for root in roots:
        if not root.exists():
            rejects.append({"path": str(root), "reason": "search_root_missing"})
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.suffix == ".csv":
                    for row_index, row in enumerate(_read_csv(path)):
                        seed = _seed_from_row(path, row, len(rows) + row_index, path.stem)
                        if seed is None or seed["part_name"] not in (parts or V80_PARTS):
                            continue
                        rows.append(seed)
                else:
                    payload = _read_json(path)
                    if payload is None:
                        rejects.append({"path": str(path), "reason": "json_load_failed"})
                        continue
                    row = payload if isinstance(payload, dict) else {"payload_type": type(payload).__name__}
                    seed = _seed_from_row(path, row, len(rows), path.stem)
                    if seed is None or seed["part_name"] not in (parts or V80_PARTS):
                        continue
                    rows.append(seed)
    run_path = Path(run_dir)
    csv_path = write_csv(run_path / "unified_seed_bank.csv", rows)
    json_path = write_json(run_path / "unified_seed_bank.json", rows)
    summary = {
        "seed_count": len(rows),
        "allowed_curriculum_count": sum(1 for row in rows if row["allowed_for_curriculum"]),
        "parts": {part: sum(1 for row in rows if row["part_name"] == part) for part in V80_PARTS},
        "reject_count": len(rejects),
    }
    summary_path = write_json(run_path / "unified_seed_bank_summary.json", summary)
    rejects_path = write_csv(run_path / "unified_seed_bank_rejects.csv", rejects)
    return {
        "rows": rows,
        "summary": summary,
        "unified_seed_bank_csv": str(csv_path),
        "unified_seed_bank_json": str(json_path),
        "unified_seed_bank_summary_json": str(summary_path),
        "unified_seed_bank_rejects_csv": str(rejects_path),
    }
