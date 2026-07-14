"""Report and artifact helpers for the v80 unified grasp stack."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


V80_PARTS = ["Plug2", "Screw1", "Backrest", "Rod", "Frame"]

FAILURE_CATEGORIES = [
    "NO_CONTACT_SIGNAL",
    "CONTACT_SENSOR_UNAVAILABLE_FALLBACK_ONLY",
    "NO_REACHABLE_PRIOR",
    "POLICY_NO_CONTACT",
    "POLICY_ONE_FINGER_CHEAT",
    "EARLY_OBJECT_MOTION",
    "CONTACT_BUT_NO_SUPPORT",
    "SLIP_AFTER_CONTACT",
    "PENETRATION",
    "TABLE_COLLISION",
    "REPLAY_MISMATCH",
    "STICKY_REJECTED_BEFORE_SUPPORT",
    "VIDEO_FAKE_OR_NOT_EXERCISED",
    "TRAINING_NOT_STARTED",
    "ENV_NOT_VECTORIZEABLE",
]

PROGRESS_COLUMNS = [
    "asset_audit_ok",
    "contact_sensor_available",
    "contact_evidence_source",
    "physics_profile",
    "seed_count",
    "reachable_prior_count",
    "curriculum_state_count",
    "rl_env_vectorized_ok",
    "actual_env_count",
    "policy_mode",
    "rl_trained",
    "training_curve_nonempty",
    "checkpoint_path",
    "eval_episode_count",
    "success_no_sticky_rate",
    "success_with_sticky_after_support_rate",
    "support_gate_rate",
    "force_contact_count_mean",
    "distance_contact_count_mean",
    "object_motion_before_contact_rate",
    "penetration_rate",
    "table_collision_rate",
    "lift_success_rate",
    "sticky_used_as_stabilizer",
    "final_replay_exercised",
    "final_video_clean",
    "usable_training_row_count",
    "failure_category",
    "blocker",
    "next_action",
]

ARTIFACT_FILENAMES = {
    "progress_csv": "v80_unified_grasp_progress_matrix.csv",
    "progress_md": "v80_unified_grasp_progress_matrix.md",
    "failure_csv": "unified_rl_failure_taxonomy.csv",
    "failure_json": "unified_rl_failure_taxonomy.json",
}


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[8]


def ensure_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_plain(row), sort_keys=True) + "\n")
    return out


def _fieldnames(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    names: list[str] = []
    for name in preferred or []:
        if name not in names:
            names.append(name)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row_list = [_plain(dict(row)) for row in rows]
    fields = _fieldnames(row_list, fieldnames)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            encoded = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(encoded)
    return out


def default_progress_row(part_name: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "part_name": part_name,
        "asset_audit_ok": False,
        "contact_sensor_available": False,
        "contact_evidence_source": "not_probed",
        "physics_profile": "canonical",
        "seed_count": 0,
        "reachable_prior_count": 0,
        "curriculum_state_count": 0,
        "rl_env_vectorized_ok": False,
        "actual_env_count": 0,
        "policy_mode": "",
        "rl_trained": False,
        "training_curve_nonempty": False,
        "checkpoint_path": "",
        "eval_episode_count": 0,
        "success_no_sticky_rate": 0.0,
        "success_with_sticky_after_support_rate": 0.0,
        "support_gate_rate": 0.0,
        "force_contact_count_mean": 0.0,
        "distance_contact_count_mean": 0.0,
        "object_motion_before_contact_rate": 0.0,
        "penetration_rate": 0.0,
        "table_collision_rate": 0.0,
        "lift_success_rate": 0.0,
        "sticky_used_as_stabilizer": False,
        "final_replay_exercised": False,
        "final_video_clean": False,
        "usable_training_row_count": 0,
        "failure_category": "TRAINING_NOT_STARTED",
        "blocker": "v80_not_run",
        "next_action": "run_v80_foundation_audit",
    }
    row.update(overrides)
    return row


def merge_progress_rows(*row_sets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {part: default_progress_row(part) for part in V80_PARTS}
    for rows in row_sets:
        for row in rows:
            part = str(row.get("part_name") or row.get("part") or "")
            if not part:
                continue
            merged.setdefault(part, default_progress_row(part)).update(row)
    return [merged[part] for part in V80_PARTS]


def write_progress_matrix(run_dir: str | Path, rows: Iterable[dict[str, Any]], *, also_root: bool = True) -> dict[str, str]:
    run_path = ensure_run_dir(run_dir)
    final_rows = merge_progress_rows(rows)
    fields = ["part_name", *PROGRESS_COLUMNS]
    csv_path = write_csv(run_path / ARTIFACT_FILENAMES["progress_csv"], final_rows, fields)
    md_path = run_path / ARTIFACT_FILENAMES["progress_md"]
    _write_markdown_table(md_path, final_rows, fields)
    outputs = {"v80_progress_csv": str(csv_path), "v80_progress_md": str(md_path)}
    if also_root:
        root_debug = repo_root_from_file() / "debug_runs"
        root_csv = write_csv(root_debug / ARTIFACT_FILENAMES["progress_csv"], final_rows, fields)
        root_md = root_debug / ARTIFACT_FILENAMES["progress_md"]
        _write_markdown_table(root_md, final_rows, fields)
        outputs.update({"v80_root_progress_csv": str(root_csv), "v80_root_progress_md": str(root_md)})
    return outputs


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_taxonomy(run_dir: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    run_path = ensure_run_dir(run_dir)
    final_rows = []
    for row in rows:
        item = dict(row)
        if item.get("failure_category") not in FAILURE_CATEGORIES:
            item["failure_category"] = "TRAINING_NOT_STARTED"
        final_rows.append(item)
    csv_path = write_csv(run_path / ARTIFACT_FILENAMES["failure_csv"], final_rows)
    json_path = write_json(run_path / ARTIFACT_FILENAMES["failure_json"], final_rows)
    return {"failure_taxonomy_csv": str(csv_path), "failure_taxonomy_json": str(json_path)}


def write_source_archive_check(repo_root: str | Path, run_dir: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    run_path = ensure_run_dir(run_dir)
    source_rel = [
        "scripts/environments/run_v80_unified_grasp_stack.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/__init__.py",
    ]
    package_root = root / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
    if package_root.exists():
        source_rel.extend(str(path.relative_to(root)) for path in sorted(package_root.glob("*.py")))
    rows = []
    zip_path = run_path / "v80_unified_grasp_source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in source_rel:
            path = root / rel
            exists = path.exists()
            rows.append({"path": rel, "exists": exists, "size_bytes": path.stat().st_size if exists else 0})
            if exists and path.is_file():
                zf.write(path, rel)
    manifest_csv = write_csv(run_path / "v80_unified_grasp_source_manifest.csv", rows)
    manifest_json = write_json(run_path / "v80_unified_grasp_source_manifest.json", rows)
    return {
        "source_archive_path": str(zip_path),
        "source_archive_manifest_csv": str(manifest_csv),
        "source_archive_manifest_json": str(manifest_json),
        "source_archive_complete": all(bool(row["exists"]) for row in rows),
    }
