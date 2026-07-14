"""v81 report helpers for physical backend and real-RL gates."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Iterable

from .v80_reports import V80_PARTS, ensure_run_dir, repo_root_from_file, write_csv, write_json


V81_PROGRESS_COLUMNS = [
    "physical_backend_ready",
    "contact_sensor_available",
    "controlled_fingertip_probe_executed",
    "contact_evidence_source",
    "force_contact_probe_peak_n",
    "object_motion_during_probe_m",
    "object_write_by_policy_detected",
    "sticky_action_available_to_policy",
    "physics_profile",
    "seed_count",
    "allowed_curriculum_count",
    "analytic_prior_unvalidated_count",
    "reachable_prior_physical_count",
    "rl_env_vectorized_ok",
    "actual_env_count",
    "rsl_rl_runner_used",
    "surrogate_training_used",
    "rl_trained",
    "ppo_iterations_completed",
    "rollout_step_count",
    "training_curve_nonempty",
    "checkpoint_path",
    "eval_episode_count_no_sticky",
    "success_no_sticky_rate",
    "eval_episode_count_sticky_after_support",
    "success_with_sticky_after_support_rate",
    "support_gate_rate",
    "force_contact_count_mean",
    "object_motion_before_contact_rate",
    "penetration_rate",
    "table_collision_rate",
    "hold_success_rate",
    "lift_success_rate",
    "final_replay_exercised",
    "final_video_clean",
    "usable_training_row_count",
    "failure_category",
    "blocker",
    "next_action",
]


def default_v81_progress_row(part_name: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "part_name": part_name,
        "physical_backend_ready": False,
        "contact_sensor_available": False,
        "controlled_fingertip_probe_executed": False,
        "contact_evidence_source": "not_probed",
        "force_contact_probe_peak_n": 0.0,
        "object_motion_during_probe_m": 0.0,
        "object_write_by_policy_detected": False,
        "sticky_action_available_to_policy": False,
        "physics_profile": "canonical",
        "seed_count": 0,
        "allowed_curriculum_count": 0,
        "analytic_prior_unvalidated_count": 0,
        "reachable_prior_physical_count": 0,
        "rl_env_vectorized_ok": False,
        "actual_env_count": 0,
        "rsl_rl_runner_used": False,
        "surrogate_training_used": False,
        "rl_trained": False,
        "ppo_iterations_completed": 0,
        "rollout_step_count": 0,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "eval_episode_count_no_sticky": 0,
        "success_no_sticky_rate": 0.0,
        "eval_episode_count_sticky_after_support": 0,
        "success_with_sticky_after_support_rate": 0.0,
        "support_gate_rate": 0.0,
        "force_contact_count_mean": 0.0,
        "object_motion_before_contact_rate": 0.0,
        "penetration_rate": 0.0,
        "table_collision_rate": 0.0,
        "hold_success_rate": 0.0,
        "lift_success_rate": 0.0,
        "final_replay_exercised": False,
        "final_video_clean": False,
        "usable_training_row_count": 0,
        "failure_category": "ENV_NOT_VECTORIZEABLE",
        "blocker": "physical_backend_not_run",
        "next_action": "run_v81_physical_backend_probe_audit",
    }
    row.update(overrides)
    return row


def merge_v81_progress_rows(*row_sets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {part: default_v81_progress_row(part) for part in V80_PARTS}
    for rows in row_sets:
        for row in rows:
            part = str(row.get("part_name") or row.get("part") or "")
            if not part:
                continue
            merged.setdefault(part, default_v81_progress_row(part)).update(row)
    return [merged[part] for part in V80_PARTS]


def write_v81_progress_matrix(run_dir: str | Path, rows: Iterable[dict[str, Any]], *, also_root: bool = True) -> dict[str, str]:
    run_path = ensure_run_dir(run_dir)
    final_rows = merge_v81_progress_rows(rows)
    fields = ["part_name", *V81_PROGRESS_COLUMNS]
    csv_path = write_csv(run_path / "v81_physical_rl_progress_matrix.csv", final_rows, fields)
    md_path = run_path / "v81_physical_rl_progress_matrix.md"
    _write_md(md_path, final_rows, fields)
    outputs = {"v81_progress_csv": str(csv_path), "v81_progress_md": str(md_path)}
    if also_root:
        root_debug = repo_root_from_file() / "debug_runs"
        root_csv = write_csv(root_debug / "v81_physical_rl_progress_matrix.csv", final_rows, fields)
        root_md = root_debug / "v81_physical_rl_progress_matrix.md"
        _write_md(root_md, final_rows, fields)
        outputs.update({"v81_root_progress_csv": str(root_csv), "v81_root_progress_md": str(root_md)})
    return outputs


def _write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v81_source_archive_check(repo_root: str | Path, run_dir: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    run_path = ensure_run_dir(run_dir)
    source_rel = [
        "scripts/environments/run_v80_unified_grasp_stack.py",
        "scripts/environments/run_v81_physical_backend_grasp_rl.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/__init__.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_unified_five_object_env.py",
    ]
    package_root = root / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
    if package_root.exists():
        source_rel.extend(str(path.relative_to(root)) for path in sorted(package_root.glob("*.py")))
    rows = []
    zip_path = run_path / "v81_physical_backend_source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in source_rel:
            path = root / rel
            exists = path.exists()
            rows.append({"path": rel, "exists": exists, "size_bytes": path.stat().st_size if exists else 0})
            if exists and path.is_file():
                zf.write(path, rel)
    manifest_csv = write_csv(run_path / "v81_physical_backend_source_manifest.csv", rows)
    manifest_json = write_json(run_path / "v81_physical_backend_source_manifest.json", rows)
    return {
        "source_archive_path": str(zip_path),
        "source_archive_manifest_csv": str(manifest_csv),
        "source_archive_manifest_json": str(manifest_json),
        "source_archive_complete": all(bool(row["exists"]) for row in rows),
    }
