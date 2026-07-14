"""Build the AST/text dependency audit for the near-grasp handoff."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2"
TEST_ROOT = REPO_ROOT / "source/isaaclab_tasks/test"
SCRIPT_ROOT = REPO_ROOT / "scripts"
NEAR_ROOT = V2_ROOT / "pipeline/unified_grasp/near_grasp"

ACTIVE_SCRIPTS = {
    "scripts/environments/run_near_grasp_cem.py",
    "scripts/environments/replay_near_grasp_candidate.py",
    "scripts/environments/audit_near_grasp_code_paths.py",
    "scripts/environments/near_grasp_doctor.py",
    "scripts/environments/build_near_grasp_handoff.py",
}
LEGACY_NAMES = {
    "run_v80_unified_grasp_stack.py",
    "run_v81_physical_backend_grasp_rl.py",
    "run_v95_scripted_contact_baseline.py",
    "v80_reports.py",
    "v81_reports.py",
    "v88_audits.py",
    "v89_hybrid_repair.py",
    "v90_bottleneck_isolation.py",
    "v91_task_sanity.py",
    "v92_task_setup_calibration.py",
    "v93_collision_geometry_safe_staging.py",
    "v94_safe_pregrasp_contact_calibration.py",
    "scripted_contact_baseline.py",
    "screw1_grasp_baseline_v2.py",
    "contact_adaptive_grasp_baseline.py",
    "contact_adaptive_closure.py",
    "dual_contact_regulator.py",
    "precontact_fingertip_controller.py",
    "acquisition_plan_executor.py",
    "serial_replay_validator.py",
    "unified_grasp_env.py",
    "train_unified_rl.py",
    "eval_unified_rl.py",
    "reward_terms.py",
    "curriculum.py",
    "multi_object_policy_config.py",
    "video_log_alignment.py",
    "test_dual_contact_regulator.py",
    "test_precontact_fingertip_controller.py",
    "test_contact_adaptive_closure.py",
    "test_closure_prior.py",
}
DYNAMIC_MARKERS = (
    "importlib.",
    "__import__(",
    "spec_from_file_location",
    "SourceFileLoader",
    "runpy.",
    "exec(",
    "eval(",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="docs/CODE_PATH_AUDIT.md")
    parser.add_argument("--manifest", default="debug_runs/handoff_release/code_path_manifest.csv")
    parser.add_argument("--graph", default="debug_runs/handoff_release/import_graph.json")
    parser.add_argument("--legacy-manifest", default="legacy/LEGACY_MANIFEST.csv")
    args = parser.parse_args()
    files = _files()
    module_by_path, path_by_module = _module_maps(files)
    parsed = {path: _parse(path, module_by_path[path], path_by_module) for path in files}
    inbound: dict[Path, set[Path]] = defaultdict(set)
    edges = []
    for source, row in parsed.items():
        for target in row["resolved_paths"]:
            inbound[target].add(source)
            edges.append({"source": _rel(source), "target": _rel(target), "kind": "ast_import"})
    for target in files:
        name = target.name
        for source in files:
            if source != target and name in parsed[source]["text"]:
                inbound[target].add(source)
                if not any(edge["source"] == _rel(source) and edge["target"] == _rel(target) for edge in edges):
                    edges.append({"source": _rel(source), "target": _rel(target), "kind": "text_or_dynamic_reference"})
    rows = [_row(path, parsed[path], inbound[path]) for path in files]
    _write_csv(REPO_ROOT / args.manifest, rows)
    _write_csv(REPO_ROOT / args.legacy_manifest, [row for row in rows if row["classification"] == "LEGACY_REFERENCE"])
    _write_json(
        REPO_ROOT / args.graph,
        {
            "schema_version": 1,
            "nodes": [{"path": row["file"], "classification": row["classification"]} for row in rows],
            "edges": sorted(edges, key=lambda edge: (edge["source"], edge["target"], edge["kind"])),
            "active_entrypoint": "scripts/environments/run_near_grasp_cem.py",
        },
    )
    _write_docs(REPO_ROOT / args.docs, rows, edges)
    print(json.dumps(_summary(rows, edges), indent=2, sort_keys=True))


def _files() -> list[Path]:
    paths = {path for root in (SCRIPT_ROOT, V2_ROOT, TEST_ROOT) for path in root.rglob("*.py") if path.is_file()}
    wuji = REPO_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/wuji_hand.py"
    if wuji.is_file():
        paths.add(wuji)
    return sorted(paths)


def _module_maps(files: list[Path]) -> tuple[dict[Path, str], dict[str, Path]]:
    module_by_path: dict[Path, str] = {}
    path_by_module: dict[str, Path] = {}
    for path in files:
        relative = _rel(path)
        if relative.startswith("source/isaaclab_tasks/isaaclab_tasks/"):
            module = relative[len("source/isaaclab_tasks/") : -3].replace("/", ".")
        elif relative.startswith("source/isaaclab_assets/isaaclab_assets/"):
            module = relative[len("source/isaaclab_assets/") : -3].replace("/", ".")
        else:
            module = relative[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        module_by_path[path] = module
        path_by_module[module] = path
        near_marker = ".pipeline.unified_grasp.near_grasp"
        if near_marker in module:
            alias = "near_grasp" + module.split(near_marker, 1)[1]
            path_by_module[alias] = path
    return module_by_path, path_by_module


def _parse(path: Path, module: str, path_by_module: dict[str, Path]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    imports: list[str] = []
    resolved: set[Path] = set()
    parse_error = ""
    try:
        tree = ast.parse(text, filename=str(path))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(package, node.module or "", node.level)
                names = [base]
                names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            for name in names:
                imports.append(name)
                candidate = _resolve_module_path(name, path_by_module)
                if candidate is not None:
                    resolved.add(candidate)
    except SyntaxError as exc:
        parse_error = f"{exc.msg}:{exc.lineno}"
    return {
        "text": text,
        "imports": sorted(set(imports)),
        "resolved_paths": sorted(resolved),
        "parse_error": parse_error,
        "dynamic_markers": [marker for marker in DYNAMIC_MARKERS if marker in text],
    }


def _row(path: Path, parsed: dict[str, Any], inbound: set[Path]) -> dict[str, Any]:
    relative = _rel(path)
    classification = _classification(path, relative)
    runners = sorted(_rel(source) for source in inbound if source.parts[-3:-1] == ("scripts", "environments") or source.name.startswith("run_"))
    tests = sorted(_rel(source) for source in inbound if "test" in source.parts or source.name.startswith("test_"))
    replacement = "near_grasp package and handoff runbook" if classification == "LEGACY_REFERENCE" else ""
    action = {
        "ACTIVE_MAINLINE": "keep",
        "ACTIVE_SHARED": "keep",
        "LEGACY_REFERENCE": "keep_compat_location",
        "GENERATED": "exclude_generated",
        "UNKNOWN": "unknown",
    }[classification]
    return {
        "file": relative,
        "line_count": len(parsed["text"].splitlines()),
        "classification": classification,
        "inbound_imports": len(inbound),
        "inbound_files": ";".join(sorted(_rel(source) for source in inbound)),
        "outbound_imports": len(parsed["imports"]),
        "outbound_modules": ";".join(parsed["imports"]),
        "referenced_by_runner": bool(runners),
        "runner_references": ";".join(runners),
        "referenced_by_tests": bool(tests),
        "test_references": ";".join(tests),
        "dynamic_import_risk": bool(parsed["dynamic_markers"]),
        "dynamic_markers": ";".join(parsed["dynamic_markers"]),
        "parse_error": parsed["parse_error"],
        "replacement": replacement,
        "action": action,
        "archive_reason": "inbound runner/test/dynamic references remain" if classification == "LEGACY_REFERENCE" else "",
    }


def _classification(path: Path, relative: str) -> str:
    if path.name in {"__pycache__", ".pytest_cache"} or path.suffix == ".pyc":
        return "GENERATED"
    if relative in ACTIVE_SCRIPTS or path.is_relative_to(NEAR_ROOT) or path.name == "test_near_grasp_search.py":
        return "ACTIVE_MAINLINE"
    if path.name == "wuji_hand.py":
        return "ACTIVE_SHARED"
    if path.name in LEGACY_NAMES:
        return "LEGACY_REFERENCE"
    if "pipeline/unified_grasp" in relative and "near_grasp" not in relative:
        return "LEGACY_REFERENCE"
    return "UNKNOWN"


def _resolve_from(package: str, module: str, level: int) -> str:
    if level <= 0:
        return module
    parts = package.split(".") if package else []
    keep = max(0, len(parts) - level + 1)
    prefix = ".".join(parts[:keep])
    return f"{prefix}.{module}".strip(".")


def _resolve_module_path(name: str, path_by_module: dict[str, Path]) -> Path | None:
    candidate = name
    while candidate:
        if candidate in path_by_module:
            return path_by_module[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


def _summary(rows: list[dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, Any]:
    classes: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = classes.setdefault(row["classification"], {"files": 0, "loc": 0})
        bucket["files"] += 1
        bucket["loc"] += int(row["line_count"])
    return {"files": len(rows), "edges": len(edges), "classifications": classes}


def _write_docs(path: Path, rows: list[dict[str, Any]], edges: list[dict[str, str]]) -> None:
    summary = _summary(rows, edges)
    active = [row for row in rows if row["classification"] == "ACTIVE_MAINLINE"]
    legacy = [row for row in rows if row["classification"] == "LEGACY_REFERENCE"]
    unknown = [row for row in rows if row["classification"] == "UNKNOWN"]
    lines = [
        "# Code Path Audit",
        "",
        "Generated from AST imports, text/dynamic references, runner CLI files, and tests.",
        "",
        "## Summary",
        "",
        f"- Python files: {summary['files']}",
        f"- Import/reference edges: {summary['edges']}",
        f"- Active mainline: {len(active)} files / {sum(int(row['line_count']) for row in active)} LOC",
        f"- Legacy reference: {len(legacy)} files / {sum(int(row['line_count']) for row in legacy)} LOC",
        f"- Unknown: {len(unknown)} files; no UNKNOWN file may be moved or deleted.",
        "",
        "## Active Mainline",
        "",
        "`run_near_grasp_cem.py -> near_grasp_physics_env.py -> grasp_program.py -> hand_prior_adapter.py -> evaluator.py -> cem.py`",
        "",
    ]
    lines.extend(f"- `{row['file']}` ({row['line_count']} LOC)" for row in active)
    lines.extend(
        [
            "",
            "## Archive Decision",
            "",
            "No historical source is physically moved in this release. Legacy candidates retain runner/test/dynamic inbound references and are indexed under `legacy/` with action `keep_compat_location`.",
            "",
            "## Unknown Files",
            "",
        ]
    )
    lines.extend(f"- `{row['file']}`" for row in unknown)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
