"""Replay one frozen near-grasp candidate with matched failure evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_GRASP_PARENT = (
    REPO_ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
)
if str(NEAR_GRASP_PARENT) not in sys.path:
    sys.path.insert(0, str(NEAR_GRASP_PARENT))

from near_grasp.configuration import NearGraspRunConfig  # noqa: E402
from near_grasp.replay import ReplayRequest, ReplayResult, attribute_contact_step, evaluation_class, load_candidate_rows, select_candidate  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=str(REPO_ROOT / "configs/near_grasp/screw1_cem_v1.yaml"))
parser.add_argument("--candidate-json", default="")
parser.add_argument("--candidate-id", type=int, default=None)
parser.add_argument("--mode", choices=("best-valid", "target-contact-abort", "exact"), default="best-valid")
parser.add_argument("--output-dir", default="debug_runs/handoff_release/replays")
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--alignment-debug", action="store_true")
parser.add_argument(
    "--contact-attribution",
    action="store_true",
    help="Add replay-only Table/ground filter sensors when the runtime supports them.",
)
parser.add_argument("--video-name", default="")
parser.add_argument("--max-steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.record_video)

RUN_CONFIG = NearGraspRunConfig.load(args_cli.config)
candidate_path = args_cli.candidate_json or str(
    REPO_ROOT / str(RUN_CONFIG.values["artifacts"]["candidate_results"])
)
REQUEST = ReplayRequest(
    candidate_json=candidate_path,
    mode=args_cli.mode,
    candidate_id=args_cli.candidate_id,
    record_video=bool(args_cli.record_video),
    alignment_debug=bool(args_cli.alignment_debug),
)
SELECTED = select_candidate(
    load_candidate_rows(REQUEST.candidate_json),
    mode=REQUEST.mode,
    candidate_id=REQUEST.candidate_id,
)
RUN_ID = (
    f"replay_candidate_{int(SELECTED['candidate_id'])}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)
OUTPUT_DIR = (REPO_ROOT / args_cli.output_dir / RUN_ID).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from near_grasp.grasp_program import GraspProgram, grasp_templates_from_config  # noqa: E402
from near_grasp.hand_prior_adapter import RetargetedPcaPriorAdapter, load_pca_artifact  # noqa: E402
from near_grasp.near_grasp_physics_env import (  # noqa: E402
    NearGraspPhysicsEnv,
    NearGraspPhysicsEnvCfg,
    apply_near_grasp_run_config,
)
from near_grasp.run_manifest import RunManifest  # noqa: E402


def main() -> None:
    env = None
    try:
        env = _make_env()
        _configure_prior(env)
        result = _run_replay(env)
        _write_json(OUTPUT_DIR / "replay_summary.json", result.to_dict())
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = {
            "classification": "REPLAY_IMPLEMENTATION_ERROR",
            "candidate_id": int(SELECTED["candidate_id"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "physical_success": False,
        }
        _write_json(OUTPUT_DIR / "replay_error.json", failure)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _make_env() -> NearGraspPhysicsEnv:
    cfg = NearGraspPhysicsEnvCfg()
    cfg.seed = int(SELECTED["reset_seed"])
    max_steps = int(args_cli.max_steps or RUN_CONFIG.values["scene"]["episode_steps"])
    apply_near_grasp_run_config(
        cfg,
        RUN_CONFIG.values,
        num_envs=1,
        device=str(args_cli.device),
        episode_steps=max_steps,
        replay=True,
        enable_replay_camera=bool(args_cli.record_video),
        enable_contact_attribution=bool(args_cli.contact_attribution),
    )
    return NearGraspPhysicsEnv(cfg=cfg, render_mode="rgb_array")


def _configure_prior(env: NearGraspPhysicsEnv) -> None:
    artifact_path = REPO_ROOT / str(RUN_CONFIG.values["prior"]["artifact"])
    artifact = load_pca_artifact(artifact_path)
    prior = RetargetedPcaPriorAdapter(
        artifact,
        joint_lower=env._lower[env._hand_ids].detach().cpu().numpy(),
        joint_upper=env._upper[env._hand_ids].detach().cpu().numpy(),
    )
    env.configure_prior(prior)


def _run_replay(env: NearGraspPhysicsEnv) -> ReplayResult:
    templates = grasp_templates_from_config(RUN_CONFIG.values["templates"])
    program = GraspProgram.from_vector(
        SELECTED["program"]["continuous16"],
        int(SELECTED["program"]["template_id"]),
    )
    candidate_id = int(SELECTED["candidate_id"])
    env.set_program_batch(
        [program],
        templates,
        reset_seeds=[int(SELECTED["reset_seed"])],
        candidate_ids=[candidate_id],
    )
    env.reset()
    _write_json(OUTPUT_DIR / "reset_snapshot.json", env.replay_snapshot(0))
    reset_rearm = env.rearm_replay_contact_guard_after_reset(0)
    _write_json(OUTPUT_DIR / "reset_contact_guard_rearm.json", reset_rearm)
    max_steps = int(args_cli.max_steps or RUN_CONFIG.values["scene"]["episode_steps"])
    trace_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    peak_force = 0.0
    for loop_step in range(max_steps):
        env.step(torch.zeros((1, 14), dtype=torch.float32, device=env.device))
        snapshot = env.replay_snapshot(0)
        forces = [float(value) for value in snapshot["target_force_n"]]
        peak_force = max(peak_force, max(forces, default=0.0))
        trace_row = _flatten_snapshot(snapshot, peak_force)
        trace_rows.append(trace_row)
        attribution_rows.extend(_attribution_rows(snapshot))
        if args_cli.record_video and loop_step % 2 == 0:
            frame = env.replay_rgb_frame(0)
            if frame is not None and frame.size:
                frames.append(_overlay(frame, snapshot, peak_force))
                alignment_rows.append(
                    {"run_id": RUN_ID, "frame_index": len(frames) - 1, "trace_row": len(trace_rows) - 1, "step": snapshot["step"]}
                )
        if candidate_id in env._completed:
            break
    evaluations = env.consume_completed_evaluations()
    replay_evaluation = evaluations.get(candidate_id)
    if replay_evaluation is None:
        raise RuntimeError(f"candidate {candidate_id} did not terminate within {max_steps} steps")
    replay_payload = replay_evaluation.to_dict()
    original_payload = dict(SELECTED["evaluation"])
    original_class = evaluation_class(original_payload)
    replay_class = evaluation_class(replay_payload)
    matched = original_class == replay_class

    trace_path = _write_csv(OUTPUT_DIR / "replay_trace.csv", trace_rows)
    attribution_path = _write_contact_attribution(attribution_rows)
    _write_csv(OUTPUT_DIR / "frame_trace_alignment.csv", alignment_rows)
    resolved_path = RUN_CONFIG.write(OUTPUT_DIR / "resolved_config.yaml")
    _write_manifest(resolved_path, env)

    video_path = ""
    if args_cli.record_video and frames:
        requested_name = args_cli.video_name or _default_video_name(replay_class)
        if REQUEST.mode == "target-contact-abort" and replay_class != "TARGET_CONTACT_HARD_ABORT":
            requested_name = "target_contact_replay_mismatch.mp4"
        target = OUTPUT_DIR / requested_name
        _write_video(target, frames, int(RUN_CONFIG.values["replay"]["fps"]))
        video_path = str(target)
    _write_json(
        OUTPUT_DIR / "matched_evaluation.json",
        {
            "run_id": RUN_ID,
            "candidate_id": candidate_id,
            "original_class": original_class,
            "replay_class": replay_class,
            "matched_original_class": matched,
            "original_evaluation": original_payload,
            "replay_evaluation": replay_payload,
            "physical_success": False,
            "classification": "NEAR_GRASP_SEARCH_INCOMPLETE",
            "reset_contact_guard_rearmed": bool(reset_rearm["rearmed"]),
        },
    )
    return ReplayResult(
        run_id=RUN_ID,
        candidate_id=candidate_id,
        original_evaluation=original_payload,
        replay_evaluation=replay_payload,
        classification="NEAR_GRASP_SEARCH_INCOMPLETE",
        matched_original_class=matched,
        trace_path=str(trace_path),
        video_path=video_path,
        contact_attribution_path=str(attribution_path),
        physical_success=False,
    )


def _flatten_snapshot(snapshot: dict[str, Any], peak_force: float) -> dict[str, Any]:
    row = {
        "run_id": RUN_ID,
        "step": snapshot["step"],
        "phase": snapshot["phase"],
        "termination_reason": snapshot["termination_reason"],
        "candidate_id": snapshot["candidate_id"],
        "template_id": snapshot["template_id"],
        "peak_target_force_n": peak_force,
        "post_reset_object_writes": snapshot["post_reset_object_writes"],
        "post_reset_wrist_state_writes": snapshot["post_reset_wrist_state_writes"],
    }
    for index, force in enumerate(snapshot["target_force_n"], start=1):
        row[f"finger{index}_target_force_n"] = force
    for axis, value in zip("xyz", snapshot["object_pos_local"]):
        row[f"object_{axis}"] = value
    for axis, value in zip("xyz", snapshot["object_delta_xyz"]):
        row[f"object_delta_{axis}"] = value
    return row


def _attribution_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for finger in range(5):
        result = attribute_contact_step(
            step=int(snapshot["step"]),
            all_force_xyz=snapshot["all_force_xyz"][finger],
            screw1_force_xyz=snapshot["target_force_xyz"][finger],
            table_force_xyz=snapshot["table_force_xyz"][finger],
            ground_force_xyz=snapshot["ground_force_xyz"][finger],
            filter_valid={
                "Screw1": snapshot["target_filter_valid"],
                "Table": snapshot["table_filter_valid"],
                "ground": snapshot["ground_filter_valid"],
            },
            threshold_n=float(RUN_CONFIG.values["force"]["formal_contact_n"]),
        )
        row = result.to_dict()
        row["run_id"] = RUN_ID
        row["finger"] = finger + 1
        row["identified_contacts"] = ";".join(result.identified_contacts)
        rows.append(row)
    return rows


def _write_contact_attribution(rows: list[dict[str, Any]]) -> Path:
    csv_path = _write_csv(OUTPUT_DIR / "contact_attribution_trace.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1
    payload = {
        "run_id": RUN_ID,
        "candidate_id": int(SELECTED["candidate_id"]),
        "counts": counts,
        "unresolved_count": counts.get("UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT", 0),
        "self_collision_enabled": False,
        "instrumentation_limit_label": "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT",
        "trace_csv": str(csv_path),
        "physical_success": False,
    }
    path = OUTPUT_DIR / "contact_attribution.json"
    _write_json(path, payload)
    return path


def _overlay(frame: np.ndarray, snapshot: dict[str, Any], peak_force: float) -> np.ndarray:
    canvas = np.ascontiguousarray(frame.copy())
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 172), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0.0, canvas)
    forces = " ".join(f"F{index + 1}:{force:.3f}N" for index, force in enumerate(snapshot["target_force_n"]))
    delta = snapshot["object_delta_xyz"]
    lines = [
        "CURRENT FAILURE REPLAY - NOT GRASP SUCCESS",
        f"candidate={snapshot['candidate_id']} template={snapshot['template_id']} phase={snapshot['phase']}",
        forces,
        f"peak={peak_force:.3f}N  delta_z={delta[2] * 1000.0:+.2f}mm  lateral={(delta[0] ** 2 + delta[1] ** 2) ** 0.5 * 1000.0:.2f}mm",
        f"termination={snapshot['termination_reason']}  PHYSICAL SUCCESS: FALSE",
    ]
    for index, line in enumerate(lines):
        color = (50, 80, 255) if index in {0, 4} else (240, 240, 240)
        cv2.putText(canvas, line, (18, 30 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.67, color, 2, cv2.LINE_AA)
    return canvas


def _default_video_name(replay_class: str) -> str:
    if REQUEST.mode == "best-valid":
        return "current_best_valid_failure.mp4"
    if replay_class == "TARGET_CONTACT_HARD_ABORT":
        return "target_contact_hard_abort.mp4"
    return f"candidate_{int(SELECTED['candidate_id'])}_failure.mp4"


def _write_manifest(resolved_path: Path, env: NearGraspPhysicsEnv) -> None:
    import isaaclab

    source_paths = [Path(__file__).resolve(), *sorted(NEAR_GRASP_PARENT.joinpath("near_grasp").glob("*.py"))]
    assets = [REPO_ROOT / value for value in RUN_CONFIG.values["assets"].values()]
    prior = REPO_ROOT / str(RUN_CONFIG.values["prior"]["artifact"])
    manifest = RunManifest.capture(
        repo_root=REPO_ROOT,
        run_id=RUN_ID,
        command=[sys.executable, *sys.argv],
        config_path=resolved_path,
        config_sha256=RUN_CONFIG.sha256,
        source_paths=source_paths,
        asset_paths=assets,
        prior_paths=[prior],
        runtime={
            "isaaclab_module": str(Path(isaaclab.__file__).resolve()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "physics_fingerprint": env.physics_fingerprint(),
            "candidate_id": int(SELECTED["candidate_id"]),
            "reset_seed": int(SELECTED["reset_seed"]),
        },
    )
    manifest.write(OUTPUT_DIR / "run_manifest.json")


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=int(fps), codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, tuple, dict)) else value
                    for key, value in row.items()
                }
            )
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
