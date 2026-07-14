"""Verify frozen evidence, build the failure gallery, and package the handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_GRASP_PARENT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
if str(NEAR_GRASP_PARENT) not in sys.path:
    sys.path.insert(0, str(NEAR_GRASP_PARENT))

from near_grasp.configuration import NearGraspRunConfig  # noqa: E402
from near_grasp.replay import evaluation_class, load_candidate_rows  # noqa: E402
from near_grasp.run_manifest import RunManifest, sha256_file  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs/near_grasp/screw1_cem_v1.yaml"
CONFIG = NearGraspRunConfig.load(CONFIG_PATH)
RELEASE_ROOT = REPO_ROOT / "debug_runs/handoff_release"
REPLAY_ROOT = RELEASE_ROOT / "replays"
GALLERY_ROOT = RELEASE_ROOT / "failure_gallery"
HANDOFF_ROOT = REPO_ROOT / "handoff/screw1_near_grasp_v0_1"
ZIP_PATH = REPO_ROOT / "handoff/screw1_near_grasp_v0_1_handoff.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--docs-check", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not any((args.verify_only, args.docs_check, args.build)):
        parser.error("choose --verify-only, --docs-check, or --build")
    verification = verify_current()
    if args.docs_check or args.build:
        docs = docs_check()
    else:
        docs = {}
    if args.build:
        gallery = build_gallery()
        bundle = build_bundle(verification, docs, gallery)
        final_report = write_final_report(verification, docs, gallery, bundle)
        payload = {"verification": verification, "docs": docs, "gallery": gallery, "bundle": bundle, "final_report": str(final_report)}
    else:
        payload = {"verification": verification, "docs": docs}
    print(json.dumps(payload, indent=2, sort_keys=True))
    passed = verification["passed"] and (not docs or docs["passed"])
    raise SystemExit(0 if passed else 2)


def verify_current() -> dict[str, Any]:
    artifact = REPO_ROOT / str(CONFIG.values["artifacts"]["candidate_results"])
    rows = load_candidate_rows(artifact)
    evaluations = [row["evaluation"] for row in rows]
    reasons = Counter(reason for evaluation in evaluations for reason in evaluation.get("invalid_reasons", []))
    actual = {
        "candidates": len(rows),
        "valid": sum(bool(row.get("valid_candidate")) for row in evaluations),
        "hard_invalid": sum(bool(row.get("hard_invalid")) for row in evaluations),
        "target_contact": sum(bool(row.get("target_filtered_success_evidence")) for row in evaluations),
        "dual_contact": sum(float(row.get("simultaneous_contact_duty", 0.0)) > 0.0 for row in evaluations),
        "stable_close": sum(bool(row.get("stable_close")) for row in evaluations),
        "physical_lift": sum(bool(row.get("physical_lift_success")) for row in evaluations),
        "unresolved": reasons["unresolved_contact_truth"],
        "flyout": reasons["flyout"],
        "hard_force_abort": reasons["hard_force_abort"],
    }
    expected = {
        "candidates": 1280,
        "valid": 910,
        "hard_invalid": 370,
        "target_contact": 2,
        "dual_contact": 0,
        "stable_close": 0,
        "physical_lift": 0,
        "unresolved": 362,
        "flyout": 6,
        "hard_force_abort": 2,
    }
    top = json.loads((REPO_ROOT / str(CONFIG.values["artifacts"]["top_programs"])).read_text(encoding="utf-8"))
    generations = list(csv.DictReader((REPO_ROOT / str(CONFIG.values["artifacts"]["generation_metrics"])).open(encoding="utf-8")))
    checks = {
        "counts_match": actual == expected,
        "generation_count": len(generations) == 5,
        "top_candidate_id": int(top["programs"][0]["candidate_id"]) == 373,
        "classification": CONFIG.values["classification"] == "NEAR_GRASP_SEARCH_INCOMPLETE",
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_cem_reexecuted": False,
    }
    result = {
        "schema_version": 1,
        "passed": all(bool(value) is True for key, value in checks.items() if key not in {"physical_grasp_success", "physical_lift_success", "full_cem_reexecuted"}),
        "actual": actual,
        "expected": expected,
        "checks": checks,
        "config_sha256": CONFIG.sha256,
        "candidate_artifact_sha256": sha256_file(artifact),
    }
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RELEASE_ROOT / "reproduction_verification.json", result)
    return result


def docs_check() -> dict[str, Any]:
    required = [
        "README.md",
        "readme/README.md",
        "readme/RUNBOOK.md",
        "readme/CODE_MAP.md",
        "readme/EXPERIMENT_HISTORY.md",
        "readme/EVIDENCE.md",
        "readme/DEBUG_REFERENCE.txt",
        "readme/experiment_ledger.json",
        "artifacts/physical_delivery/README.md",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/near_grasp/README.md",
        "legacy/LEGACY_MANIFEST.csv",
        "Makefile",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).is_file()]
    combined = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8", errors="replace") for path in required if (REPO_ROOT / path).is_file())
    forbidden = [
        token
        for token in (
            "physical_grasp_success=true",
            "physical_lift_success=true",
            "successful_grasp.mp4",
        )
        if token in combined
    ]
    required_phrases = [
        "NEAR_GRASP_SEARCH_INCOMPLETE",
        "no validated physical grasp",
        "make demo-current",
        "make reproduce-current",
        "full_route_success",
    ]
    absent_phrases = [phrase for phrase in required_phrases if phrase not in combined]
    result = {
        "passed": not missing and not forbidden and not absent_phrases,
        "required_files": required,
        "missing": missing,
        "forbidden_success_claims": forbidden,
        "missing_required_phrases": absent_phrases,
    }
    _write_json(RELEASE_ROOT / "docs_check.json", result)
    return result


def build_gallery() -> dict[str, Any]:
    GALLERY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_candidate_rows(REPO_ROOT / str(CONFIG.values["artifacts"]["candidate_results"]))
    by_id = {int(row["candidate_id"]): row for row in rows}
    candidate_ids = [373, 33, 707, 0, 837]
    metrics = []
    replay_rows: dict[int, dict[str, Any]] = {}
    attribution_payloads = {}
    for candidate_id in candidate_ids:
        original = by_id[candidate_id]
        replay_dir = _latest_replay(candidate_id)
        matched = _read_optional(replay_dir / "matched_evaluation.json") if replay_dir else {}
        summary = _read_optional(replay_dir / "replay_summary.json") if replay_dir else {}
        replay_rows[candidate_id] = {"dir": str(replay_dir) if replay_dir else "", "matched": matched, "summary": summary}
        evaluation = original["evaluation"]
        metrics.append(
            {
                "candidate_id": candidate_id,
                "role": {373: "best_valid", 33: "hard_abort_primary", 707: "hard_abort_secondary", 0: "unresolved", 837: "flyout"}[candidate_id],
                "reset_seed": original["reset_seed"],
                "template_id": original["program"]["template_id"],
                "original_class": evaluation_class(evaluation),
                "original_peak_force_n": evaluation.get("peak_target_force_n", 0.0),
                "original_lateral_displacement_m": evaluation.get("lateral_displacement_m", 0.0),
                "replay_class": matched.get("replay_class", "NOT_REPLAYED"),
                "matched_original_class": matched.get("matched_original_class", False),
                "replay_dir": str(replay_dir) if replay_dir else "",
            }
        )
        if replay_dir and (replay_dir / "replay_trace.csv").is_file():
            _plot_trace(candidate_id, replay_dir / "replay_trace.csv", GALLERY_ROOT / f"candidate_{candidate_id}_force_object.png")
        if replay_dir and (replay_dir / "contact_attribution.json").is_file():
            attribution_payloads[str(candidate_id)] = _read_optional(replay_dir / "contact_attribution.json")
        if replay_dir:
            for video in replay_dir.glob("*.mp4"):
                shutil.copy2(video, GALLERY_ROOT / f"candidate_{candidate_id}_{video.name}")
    _write_csv(GALLERY_ROOT / "candidate_metrics.csv", metrics)
    contact = {
        "classification": "NEAR_GRASP_SEARCH_INCOMPLETE",
        "original_unresolved_candidates": 362,
        "replay_attribution": attribution_payloads,
        "instrumentation_limit_label": "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT",
        "physical_success": False,
    }
    _write_json(GALLERY_ROOT / "contact_attribution.json", contact)
    _write_gallery_html(metrics)
    best_video = _find_video(373, "current_best_valid_failure.mp4")
    hard_video = _find_matching_hard_abort_video()
    video_validation = {
        "current_best_valid_failure": _validate_video(best_video) if best_video else {"passed": False, "error": "missing"},
        "target_contact_hard_abort": _validate_video(hard_video) if hard_video else {"passed": False, "error": "missing"},
    }
    video_validation["passed"] = all(row.get("passed", False) for row in video_validation.values())
    _write_json(RELEASE_ROOT / "video_validation.json", video_validation)
    return {
        "path": str(GALLERY_ROOT),
        "candidate_ids": candidate_ids,
        "replays": replay_rows,
        "contact_attribution": str(GALLERY_ROOT / "contact_attribution.json"),
        "video_validation": video_validation,
    }


def build_bundle(verification: dict[str, Any], docs: dict[str, Any], gallery: dict[str, Any]) -> dict[str, Any]:
    if HANDOFF_ROOT.exists():
        shutil.rmtree(HANDOFF_ROOT)
    HANDOFF_ROOT.mkdir(parents=True)
    resolved = CONFIG.write(RELEASE_ROOT / "resolved_config.yaml")
    manifest_path = _release_manifest(resolved)
    copy_map = {
        REPO_ROOT / "docs/CURRENT_STATUS.md": "CURRENT_STATUS.md",
        REPO_ROOT / "docs/ARCHITECTURE.md": "ARCHITECTURE.md",
        REPO_ROOT / "docs/RUNBOOK.md": "RUNBOOK.md",
        REPO_ROOT / "docs/HANDOFF_TO_SENIOR.md": "HANDOFF_TO_SENIOR.md",
        REPO_ROOT / "docs/SUCCESS_CRITERIA.md": "SUCCESS_CRITERIA.md",
        REPO_ROOT / "docs/KNOWN_ISSUES.md": "KNOWN_ISSUES.md",
        REPO_ROOT / "docs/CODE_PATH_AUDIT.md": "code_audit/CODE_PATH_AUDIT.md",
        REPO_ROOT / "docs/LEGACY_MAP.md": "LEGACY_MAP.md",
        REPO_ROOT / "docs/EXTERNAL_BASELINES.md": "EXTERNAL_BASELINES.md",
        REPO_ROOT / "README.md": "PROJECT_README.md",
        resolved: "resolved_config.yaml",
        manifest_path: "run_manifest.json",
        REPO_ROOT / str(CONFIG.values["artifacts"]["top_programs"]): "top_grasp_programs.json",
        REPO_ROOT / str(CONFIG.values["artifacts"]["generation_metrics"]): "cem_generation_metrics.csv",
        REPO_ROOT / str(CONFIG.values["artifacts"]["honesty_summary"]): "honesty_summary.json",
        REPO_ROOT / str(CONFIG.values["prior"]["artifact"]): "retargeted_wuji_pca6.json",
        REPO_ROOT / "legacy/LEGACY_MANIFEST.csv": "LEGACY_MANIFEST.csv",
        RELEASE_ROOT / "code_path_manifest.csv": "code_audit/code_path_manifest.csv",
        RELEASE_ROOT / "import_graph.json": "code_audit/import_graph.json",
        RELEASE_ROOT / "video_validation.json": "video_validation.json",
        GALLERY_ROOT / "contact_attribution.json": "contact_attribution.json",
        GALLERY_ROOT / "candidate_metrics.csv": "candidate_metrics.csv",
        GALLERY_ROOT / "failure_gallery.html": "failure_gallery/failure_gallery.html",
        RELEASE_ROOT / "reproduction_verification.json": "reproduction_verification.json",
    }
    best_video = _find_video(373, "current_best_valid_failure.mp4")
    hard_video = _find_matching_hard_abort_video()
    if best_video is None or hard_video is None:
        raise RuntimeError("matched best-valid and hard-abort videos are required before building the handoff")
    copy_map[best_video] = "current_best_valid_failure.mp4"
    copy_map[hard_video] = "target_contact_hard_abort.mp4"
    for gallery_file in sorted(GALLERY_ROOT.glob("candidate_*_force_object.png")):
        copy_map[gallery_file] = f"failure_gallery/{gallery_file.name}"
    for candidate_id, replay in gallery["replays"].items():
        replay_dir = Path(replay["dir"])
        for name in (
            "matched_evaluation.json",
            "replay_summary.json",
            "replay_trace.csv",
            "contact_attribution.json",
            "contact_attribution_trace.csv",
            "frame_trace_alignment.csv",
            "reset_snapshot.json",
            "reset_contact_guard_rearm.json",
            "run_manifest.json",
        ):
            source = replay_dir / name
            if source.is_file():
                copy_map[source] = f"replays/candidate_{candidate_id}/{name}"
    for source, relative in copy_map.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = HANDOFF_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (HANDOFF_ROOT / "README.md").write_text(
        "# Screw1 Near-Grasp v0.1 Handoff\n\n"
        "Status: `NEAR_GRASP_SEARCH_INCOMPLETE`; no physical grasp or lift success.\n\n"
        "Start with `RUNBOOK.md`, then compare the two failure videos with `candidate_metrics.csv` and `contact_attribution.json`.\n",
        encoding="utf-8",
    )
    bundle_rows = []
    for path in sorted(HANDOFF_ROOT.rglob("*")):
        if path.is_file() and path.name != "handoff_bundle_manifest.json":
            bundle_rows.append({"path": str(path.relative_to(HANDOFF_ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    bundle_manifest = {
        "schema_version": 1,
        "classification": "NEAR_GRASP_SEARCH_INCOMPLETE",
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "verification_passed": verification["passed"],
        "docs_check_passed": docs["passed"],
        "files": bundle_rows,
        "excluded": ["CoorDex checkpoint", "full debug_runs", "IsaacLab upstream source", "cache", "duplicate videos"],
    }
    bundle_manifest_path = HANDOFF_ROOT / "handoff_bundle_manifest.json"
    _write_json(bundle_manifest_path, bundle_manifest)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(HANDOFF_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(HANDOFF_ROOT.name) / path.relative_to(HANDOFF_ROOT))
    return {
        "directory": str(HANDOFF_ROOT),
        "zip": str(ZIP_PATH),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256_file(ZIP_PATH),
        "manifest": str(bundle_manifest_path),
        "file_count": len(bundle_rows) + 1,
    }


def write_final_report(
    verification: dict[str, Any],
    docs: dict[str, Any],
    gallery: dict[str, Any],
    bundle: dict[str, Any],
) -> Path:
    audit_rows = list(csv.DictReader((RELEASE_ROOT / "code_path_manifest.csv").open(encoding="utf-8")))
    active = [row for row in audit_rows if row["classification"] == "ACTIVE_MAINLINE"]
    legacy = [row for row in audit_rows if row["classification"] == "LEGACY_REFERENCE"]
    unknown = [row for row in audit_rows if row["classification"] == "UNKNOWN"]
    graph = _read_optional(RELEASE_ROOT / "import_graph.json")
    classification = {node["path"]: node["classification"] for node in graph.get("nodes", [])}
    active_legacy_edges = [
        edge
        for edge in graph.get("edges", [])
        if classification.get(edge["source"]) == "ACTIVE_MAINLINE" and classification.get(edge["target"]) == "LEGACY_REFERENCE"
    ]
    active_legacy_ast_edges = [edge for edge in active_legacy_edges if edge["kind"] == "ast_import"]
    active_legacy_text_edges = [edge for edge in active_legacy_edges if edge["kind"] != "ast_import"]
    before = _pre_cleanup_active_metrics()
    replay_summary = gallery["replays"]
    lines = [
        "# Screw1 Near-Grasp Handoff Release Report",
        "",
        "## 1. Codebase Metrics Before/After",
        "",
        f"- Before snapshot active files/LOC: {before['files']} / {before['loc']}",
        f"- After active files/LOC: {len(active)} / {sum(int(row['line_count']) for row in active)}",
        f"- Legacy reference files/LOC: {len(legacy)} / {sum(int(row['line_count']) for row in legacy)}",
        "- Physically moved or deleted source files: 0",
        "- Deleted generated files: 0; caches are excluded by ignore/package rules.",
        "",
        "## 2. Active Dependency Graph",
        "",
        "`runner -> NearGraspPhysicsEnv -> GraspProgram/HandPriorAdapter -> StrictPhysicalEvaluator -> MixedCem`",
        f"- Active-to-legacy AST import edges: {len(active_legacy_ast_edges)}",
        f"- Active-to-legacy audit text/reference edges: {len(active_legacy_text_edges)}",
        "- Runner uses the lightweight top-level `near_grasp` package path.",
        "",
        "## 3. Archived Modules",
        "",
        f"- Logically indexed legacy modules: {len(legacy)}; physical archive moves: 0.",
        "- Reason: old runners/tests/dynamic references remain; action is `keep_compat_location`.",
        f"- UNKNOWN files left untouched: {len(unknown)}.",
        "",
        "## 4. Current Reproducibility",
        "",
        f"- Frozen artifact verification: {'PASS' if verification['passed'] else 'FAIL'}",
        "- Doctor, tests, 16-env smoke, and candidate replay results are stored under `debug_runs/handoff_release/`.",
        "- Ruff: `SKIPPED` because it is not installed; no dependency was added.",
        f"- Config hash: `{verification['config_sha256']}`",
        "",
        "## 5. What The User Sees",
        "",
        f"- Candidate 373 replay: `{replay_summary[373]['matched'].get('replay_class', 'missing')}`; object remains on the table.",
        f"- Candidate 33 replay: `{replay_summary[33]['matched'].get('replay_class', 'missing')}` with strict hard-force rejection.",
        f"- Candidate 0 replay matched: `{replay_summary[0]['matched'].get('matched_original_class', False)}`.",
        f"- Candidate 707 replay matched: `{replay_summary[707]['matched'].get('matched_original_class', False)}`; mismatch retained.",
        f"- Candidate 837 replay matched: `{replay_summary[837]['matched'].get('matched_original_class', False)}`; mismatch retained.",
        "- Videos overlay forces, object motion, termination, and `PHYSICAL SUCCESS: FALSE`.",
        f"- H.264/frame/alignment validation: `{'PASS' if gallery['video_validation']['passed'] else 'FAIL'}`.",
        "",
        "## 6. Contact Attribution",
        "",
        "- Original unresolved candidates: 362.",
        "- Replay distinguishes Screw1-filtered force from the all-force vector.",
        "- Table/ground GPU filters are unsupported; unexplained nonzero residual remains `UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT` without collider inference.",
        "- Step-0 stale reset-forward contact values are recorded and re-armed without physics state writes.",
        "",
        "## 7. Documentation",
        "",
        f"- Documentation check: {'PASS' if docs['passed'] else 'FAIL'}",
        "- Root README, architecture, status, criteria, runbook, issues, legacy map, external audit, and senior handoff are present.",
        "",
        "## 8. Handoff Package",
        "",
        f"- Directory: `{bundle['directory']}`",
        f"- Zip: `{bundle['zip']}`",
        f"- Size: {bundle['zip_size_bytes']} bytes",
        f"- SHA256: `{bundle['zip_sha256']}`",
        f"- Manifest: `{bundle['manifest']}`",
        "",
        "## 9. Current Algorithm Status",
        "",
        "- `NEAR_GRASP_SEARCH_INCOMPLETE`",
        "- `physical_grasp_success=false`",
        "- `physical_lift_success=false`",
        "- CEM generations were not extended; PPO was not started.",
        "",
        "## 10. Recommended External-Help Question",
        "",
        "How should the frozen grasp templates, Wuji prior, or candidate initialization be changed so the current 16D program reaches a low-force multi-contact attraction region without weakening the strict evaluator or force guards?",
    ]
    path = RELEASE_ROOT / "final_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _release_manifest(resolved: Path) -> Path:
    source_paths = [
        REPO_ROOT / "scripts/environments/run_near_grasp_cem.py",
        REPO_ROOT / "scripts/environments/replay_near_grasp_candidate.py",
        *sorted((NEAR_GRASP_PARENT / "near_grasp").glob("*.py")),
    ]
    assets = [REPO_ROOT / relative for relative in CONFIG.values["assets"].values()]
    prior = REPO_ROOT / str(CONFIG.values["prior"]["artifact"])
    doctor = _read_optional(RELEASE_ROOT / "doctor_report.json")
    manifest = RunManifest.capture(
        repo_root=REPO_ROOT,
        run_id="screw1_near_grasp_v0_1_handoff",
        command=["make", "reproduce-current"],
        config_path=resolved,
        config_sha256=CONFIG.sha256,
        source_paths=source_paths,
        asset_paths=assets,
        prior_paths=[prior],
        runtime={"doctor_report": doctor, "full_cem_reexecuted": False, "ppo_started": False},
    )
    path = RELEASE_ROOT / "run_manifest.json"
    manifest.write(path)
    return path


def _plot_trace(candidate_id: int, trace_path: Path, output: Path) -> None:
    rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    if not rows:
        return
    steps = [int(float(row["step"])) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for finger in range(1, 6):
        axes[0].plot(steps, [float(row[f"finger{finger}_target_force_n"]) for row in rows], label=f"finger{finger}")
    axes[0].axhline(0.05, color="green", linestyle="--", label="contact 0.05N")
    axes[0].axhline(5.0, color="red", linestyle="--", label="hard abort 5N")
    axes[0].set_ylabel("Target force (N)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].plot(steps, [1000.0 * float(row["object_delta_z"]) for row in rows], label="delta z (mm)")
    axes[1].plot(
        steps,
        [1000.0 * (float(row["object_delta_x"]) ** 2 + float(row["object_delta_y"]) ** 2) ** 0.5 for row in rows],
        label="lateral (mm)",
    )
    axes[1].axhline(10.0, color="black", linestyle="--", label="lift criterion")
    axes[1].set_xlabel("Physics step")
    axes[1].set_ylabel("Object motion (mm)")
    axes[1].legend()
    fig.suptitle(f"Candidate {candidate_id} failure replay - physical success false")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _write_gallery_html(metrics: list[dict[str, Any]]) -> None:
    cards = []
    for row in metrics:
        candidate_id = int(row["candidate_id"])
        videos = sorted(GALLERY_ROOT.glob(f"candidate_{candidate_id}_*.mp4"))
        video_html = "".join(f'<video controls width="520" src="{html.escape(video.name)}"></video>' for video in videos)
        plot = GALLERY_ROOT / f"candidate_{candidate_id}_force_object.png"
        plot_html = f'<img width="700" src="{plot.name}" alt="force and object plot">' if plot.is_file() else "<p>Replay trace unavailable.</p>"
        cards.append(
            f"<section><h2>Candidate {candidate_id}: {html.escape(str(row['role']))}</h2>"
            f"<p>Original: {html.escape(str(row['original_class']))}; replay: {html.escape(str(row['replay_class']))}; success=false</p>"
            f"{video_html}{plot_html}</section>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Near-Grasp Failure Gallery</title>"
        "<style>body{font-family:sans-serif;max-width:1000px;margin:2rem auto;color:#222}section{border-top:1px solid #aaa;padding:1rem 0}video,img{display:block;margin:.5rem 0;max-width:100%}</style>"
        "</head><body><h1>Current Failure Gallery - Not Grasp Success</h1>"
        "<p>Classification: NEAR_GRASP_SEARCH_INCOMPLETE. All entries have physical success=false.</p>"
        + "".join(cards)
        + "</body></html>"
    )
    (GALLERY_ROOT / "failure_gallery.html").write_text(page, encoding="utf-8")


def _validate_video(path: Path) -> dict[str, Any]:
    replay_dir = path.parent
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC)) if opened else 0
    codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    decoded_frames = 0
    changing_frame_pairs = 0
    max_mean_abs_change = 0.0
    pixel_std_max = 0.0
    previous_gray = None
    while opened:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixel_std_max = max(pixel_std_max, float(np.std(gray)))
        if previous_gray is not None:
            change = float(np.mean(cv2.absdiff(previous_gray, gray)))
            max_mean_abs_change = max(max_mean_abs_change, change)
            changing_frame_pairs += int(change > 0.05)
        previous_gray = gray
        decoded_frames += 1
    capture.release()

    alignment = list(csv.DictReader((replay_dir / "frame_trace_alignment.csv").open(encoding="utf-8")))
    trace = list(csv.DictReader((replay_dir / "replay_trace.csv").open(encoding="utf-8")))
    matched = _read_optional(replay_dir / "matched_evaluation.json")
    run_id = str(matched.get("run_id", ""))
    alignment_run_ids = {row.get("run_id", "") for row in alignment}
    trace_run_ids = {row.get("run_id", "") for row in trace}
    frame_indices = [int(row["frame_index"]) for row in alignment] if alignment else []
    trace_indices = [int(row["trace_row"]) for row in alignment] if alignment else []
    replay_source = (REPO_ROOT / "scripts/environments/replay_near_grasp_candidate.py").read_text(encoding="utf-8")
    overlay_contract = all(
        phrase in replay_source
        for phrase in (
            "CURRENT FAILURE REPLAY - NOT GRASP SUCCESS",
            "PHYSICAL SUCCESS: FALSE",
        )
    )
    checks = {
        "opened": opened,
        "codec_h264": codec.lower() in {"h264", "avc1", "x264"},
        "decoded_nonempty": decoded_frames > 2,
        "dimensions": width == 960 and height == 540,
        "pixel_variation": pixel_std_max > 1.0 and changing_frame_pairs > 0,
        "declared_matches_decoded": declared_frames == decoded_frames,
        "alignment_count": len(alignment) == decoded_frames,
        "alignment_indices": frame_indices == list(range(decoded_frames)),
        "alignment_trace_bounds": bool(trace_indices) and max(trace_indices) < len(trace),
        "run_id_aligned": bool(run_id) and alignment_run_ids == {run_id} and trace_run_ids == {run_id},
        "physical_success_false": matched.get("physical_success") is False,
        "overlay_contract": overlay_contract,
    }
    return {
        "path": str(path),
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "declared_frames": declared_frames,
        "decoded_frames": decoded_frames,
        "pixel_std_max": pixel_std_max,
        "changing_frame_pairs": changing_frame_pairs,
        "max_mean_abs_change": max_mean_abs_change,
        "run_id": run_id,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _latest_replay(candidate_id: int) -> Path | None:
    candidates = [path for path in REPLAY_ROOT.glob(f"replay_candidate_{candidate_id}_*") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _find_video(candidate_id: int, name: str) -> Path | None:
    replay = _latest_replay(candidate_id)
    path = replay / name if replay else None
    return path if path and path.is_file() else None


def _find_matching_hard_abort_video() -> Path | None:
    for candidate_id in (33, 707):
        replay = _latest_replay(candidate_id)
        if replay is None:
            continue
        matched = _read_optional(replay / "matched_evaluation.json")
        video = replay / "target_contact_hard_abort.mp4"
        if matched.get("replay_class") == "TARGET_CONTACT_HARD_ABORT" and video.is_file():
            return video
    return None


def _pre_cleanup_active_metrics() -> dict[str, int]:
    archive = RELEASE_ROOT / "pre_cleanup_untracked_sources.tar.gz"
    names = {"scripts/environments/run_near_grasp_cem.py", "source/isaaclab_tasks/test/test_near_grasp_search.py"}
    prefix = "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/near_grasp/"
    files = 0
    loc = 0
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream.getmembers():
                if not member.isfile() or not member.name.endswith(".py"):
                    continue
                if member.name in names or member.name.startswith(prefix):
                    extracted = stream.extractfile(member)
                    if extracted is not None:
                        files += 1
                        loc += len(extracted.read().decode("utf-8", errors="replace").splitlines())
    return {"files": files, "loc": loc}


def _read_optional(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
