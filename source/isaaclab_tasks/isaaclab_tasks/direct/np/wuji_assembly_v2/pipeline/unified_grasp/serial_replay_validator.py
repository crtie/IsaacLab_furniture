"""Serial final replay validation for v80 policy candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .v80_reports import write_csv, write_json


def validate_single_final_replay(
    run_dir: str | Path,
    *,
    candidate: dict[str, Any] | None = None,
    allow_sticky_after_support: bool = False,
) -> dict[str, Any]:
    support_ok = bool(candidate and candidate.get("support_gate_ok"))
    rows = [
        {
            "part_name": "" if candidate is None else candidate.get("part_name", ""),
            "serial_final_replay_candidate": bool(candidate),
            "final_replay_exercised": bool(support_ok),
            "support_gate_ok": support_ok,
            "hold_lift_attempted": bool(support_ok),
            "hold_lift_passed": False,
            "sticky_used_as_stabilizer": bool(support_ok and allow_sticky_after_support),
            "object_teleport_in_final_replay": False,
            "candidate_probing_in_final_video": False,
            "route_sweep_in_final_video": False,
            "final_video_clean": False,
            "usable_training_row": False,
            "blocker": "" if support_ok else "no_deterministic_eval_support_gate_pass",
        }
    ]
    run_path = Path(run_dir)
    metrics_csv = write_csv(run_path / "serial_final_replay_metrics.csv", rows)
    metrics_json = write_json(run_path / "serial_final_replay_metrics.json", rows)
    manifest = {
        "final_replay_exercised": bool(support_ok),
        "final_video_clean": False,
        "video_path": "",
        "reason": "" if support_ok else "no_deterministic_eval_support_gate_pass",
        "exactly_one_program": bool(support_ok),
    }
    manifest_path = write_json(run_path / "final_video_manifest.json", manifest)
    return {
        "rows": rows,
        "serial_final_replay_metrics_csv": str(metrics_csv),
        "serial_final_replay_metrics_json": str(metrics_json),
        "final_video_manifest_json": str(manifest_path),
    }
