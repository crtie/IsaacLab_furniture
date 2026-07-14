"""Run the single fixed Wuji Frame Stage 4 physical delivery trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
NP_DIR = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np"
UNIFIED_DIR = NP_DIR / "wuji_assembly_v2/pipeline/unified_grasp"
for value in (NP_DIR, UNIFIED_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from wuji_assembly_v2.physical_delivery.contracts import (  # noqa: E402
    video_evidence_matches_numeric_rollout,
)
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/physical_delivery/frame_stage4_rc2.yaml")
parser.add_argument("--output-dir", default="debug_runs/frame_stage4_delivery_rc2")
parser.add_argument("--no-directed-corrections", action="store_true")
parser.add_argument(
    "--build-delivery",
    action="store_true",
    help="Explicitly rebuild 5/frame_stage4_delivery_rc2 after the rollout.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

CONFIG_PATH = (REPO_ROOT / args.config).resolve()
OUTPUT = (REPO_ROOT / args.output_dir).resolve()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = _load_yaml(CONFIG_PATH)
    preflight = _gpu_preflight(float(config["runtime"]["min_gpu_free_gb"]))
    _write_json(OUTPUT / "gpu_preflight.json", preflight)
    if not preflight["sufficient"]:
        summary = {"classification": "HOST_GPU_BUSY", **_honesty(), **preflight}
        _write_json(OUTPUT / "final_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    env = None
    try:
        env, task = _make_env(config)
        runtime = env.runtime_fingerprint()
        _validate_assets(runtime)
        _write_json(OUTPUT / "runtime_fingerprint.json", runtime)
        attempts = []
        correction_log = []
        height_adjust = 0.0
        normal_adjust = 0.0
        partial_hook = float(config["fork"]["baseline_partial_hook_fraction"])
        used_corrections: set[str] = set()
        first_success = None
        first_execution = None
        selected = None
        max_corrections = 0 if args.no_directed_corrections else int(
            config["fork"]["maximum_directed_corrections"]
        )

        for attempt_index in range(max_corrections + 1):
            pose = _build_fork_pose(
                env,
                task,
                config,
                height_adjust=height_adjust,
                normal_adjust=normal_adjust,
                partial_hook=partial_hook,
            )
            attempt_name = "attempt_0" if attempt_index == 0 else f"correction_{attempt_index}"
            _write_json(OUTPUT / attempt_name / "fork_support_pose.json", pose.to_dict())
            execution = _run_rollout(env, task, config, pose, attempt_name, attempt_index)
            if first_execution is None:
                first_execution = execution
            attempts.append(execution["result"])
            if attempt_index == 0:
                shutil.copy2(execution["annotated_video"], OUTPUT / "videos/debug_first.mp4")
            result = execution["result"]
            if result["delivery_level"] in {"A", "B"}:
                first_success = execution
                selected = execution
                break
            if result["physical_support_acquired"] and result["preinsert_success"]:
                selected = execution
                break
            correction = _derive_correction(
                execution,
                pose,
                config,
                used_corrections,
            )
            if correction is None or attempt_index >= max_corrections:
                selected = execution
                break
            used_corrections.add(correction["kind"])
            if correction["kind"] == "height":
                height_adjust += float(correction["delta_m"])
            elif correction["kind"] == "normal":
                normal_adjust += float(correction["delta_m"])
            else:
                partial_hook = float(config["fork"]["fallback_partial_hook_fraction"])
            correction_log.append(correction)

        reproducible = False
        replay_result = None
        if first_success is not None:
            frozen = {
                "height_adjust_m": height_adjust,
                "normal_adjust_m": normal_adjust,
                "partial_hook_fraction": partial_hook,
            }
            _write_json(OUTPUT / "frozen_route.json", frozen)
            replay_pose = _build_fork_pose(
                env,
                task,
                config,
                height_adjust=height_adjust,
                normal_adjust=normal_adjust,
                partial_hook=partial_hook,
            )
            replay = _run_rollout(env, task, config, replay_pose, "reproduction", 100)
            replay_result = replay["result"]
            if replay_result["delivery_level"] in {"A", "B"}:
                selected = replay
                reproducible = True
            else:
                selected = first_success

        assert selected is not None
        published_evidence = selected if first_success is not None else first_execution
        assert published_evidence is not None
        _publish_selected(published_evidence, first_success is not None)
        final = dict(selected["result"])
        video_evidence_attempt = published_evidence["result"]["metadata"]["attempt"]
        numeric_final_attempt = selected["result"]["metadata"]["attempt"]
        final.update(
            {
                "classification": _classification(final),
                "reproducible": reproducible,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "reproduction_result": replay_result,
                "directed_corrections": correction_log,
                "runtime_fingerprint": runtime,
                "video_path": (
                    "videos/frame_stage4_delivery.mp4"
                    if final["delivery_level"] in {"A", "B"}
                    else "videos/debug_first.mp4"
                ),
                "video_evidence_attempt": video_evidence_attempt,
                "numeric_final_attempt": numeric_final_attempt,
                "video_matches_numeric_rollout": video_evidence_matches_numeric_rollout(
                    video_evidence_attempt, numeric_final_attempt
                ),
                "delivery_built": bool(args.build_delivery),
                **_honesty(),
            }
        )
        _write_json(OUTPUT / "final_summary.json", final)
        if args.build_delivery:
            _build_delivery(final, selected)
        print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = {
            "classification": "FRAME_STAGE4_RC2_RUNTIME_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            **_honesty(),
        }
        _write_json(OUTPUT / "runtime_error.json", failure)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _make_env(config: Mapping[str, Any]):
    from chair_tasks_cfg import ChairAssembly4
    from wuji_assembly_v2.physical_delivery.frame_stage4_env import FrameStage4Env, FrameStage4EnvCfg

    cfg = FrameStage4EnvCfg()
    cfg.seed = int(config["seed"])
    cfg.scene.num_envs = 1
    cfg.sim.device = str(args.device)
    cfg.sim.dt = float(config["runtime"]["dt"])
    cfg.episode_length_s = float(config["runtime"]["max_rollout_steps"] + 20) * cfg.sim.dt
    env = FrameStage4Env(cfg=cfg)
    return env, ChairAssembly4()


def _build_fork_pose(env, task, config, *, height_adjust: float, normal_adjust: float, partial_hook: float):
    from wuji_assembly_v2.physical_delivery.frame_fork_support import FrameForkSupportBuilder

    builder = FrameForkSupportBuilder(
        REPO_ROOT / config["assets"]["wuji_urdf"],
        REPO_ROOT / config["assets"]["frame_geometry_audit"],
    )
    return builder.build(
        hand_preshape_q20=env.hand_preshape_q.detach().cpu().numpy(),
        hand_close_q20=env.hand_close_q.detach().cpu().numpy(),
        frame_position=task.frame.init_state.pos,
        frame_quat_wxyz=task.frame.init_state.rot,
        fixed_position=task.fixed_asset.init_state.pos,
        fixed_quat_wxyz=task.fixed_asset.init_state.rot,
        insertion_axis_fixed=task.connection_cfg1.axis_t,
        target_gap_m=float(config["fork"]["target_gap_m"]),
        height_adjust_m=height_adjust,
        normal_adjust_m=normal_adjust,
        partial_hook_fraction=partial_hook,
    )


def _run_rollout(env, task, config, pose, name: str, attempt_index: int) -> dict[str, Any]:
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch

    from wuji_assembly_v2.physical_delivery.frame_stage4_controller import FrameStage4Controller
    from wuji_assembly_v2.physical_delivery.frame_stage4_evaluator import (
        FrameStage4Evaluator,
        attribute_frame_contacts,
    )

    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    video_dir = OUTPUT / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    raw_path = video_dir / f"{name}_raw.mp4"
    annotated_path = video_dir / ("debug_first_internal.mp4" if attempt_index == 0 else f"{name}.mp4")
    trace_path = directory / "full_route_trace.csv"
    contacts_path = directory / "contact_pairs.jsonl"
    alignment_path = directory / "video_alignment.json"

    env.configure_reset(pose.q26)
    env.reset()
    reset_cache = env.clear_reset_contact_cache()
    controller = FrameStage4Controller(
        config,
        pose,
        pose_to_base=task.connection_cfg1.pose_to_base,
        insertion_axis_fixed=task.connection_cfg1.axis_t,
    )
    evaluator = FrameStage4Evaluator()
    action = torch.as_tensor(pose.q26, dtype=torch.float32, device=env.device).reshape(1, 26)
    capture_every = int(config["video"]["capture_every_physics_steps"])
    fps = int(config["video"]["fps"])
    raw_writer = imageio.get_writer(raw_path, fps=fps, codec="libx264", quality=8, macro_block_size=2)
    annotated_writer = imageio.get_writer(
        annotated_path, fps=fps, codec="libx264", quality=8, macro_block_size=2
    )
    trace_rows = []
    contact_rows = []
    alignment = []
    command = None
    terminal_diagnostic_remaining = 0
    try:
        for _ in range(int(config["runtime"]["max_rollout_steps"])):
            env.step(action)
            snapshot = env.snapshot()
            events = env.consume_contact_events()
            contact_summary = attribute_frame_contacts(
                events, report_available=bool(snapshot["contact_report_available"])
            )
            diagnostic_hold = terminal_diagnostic_remaining > 0
            if not diagnostic_hold:
                evaluator.update(snapshot, relative_reference=controller.relative_reference)
                command = controller.command(snapshot, contact_summary)
            else:
                terminal_diagnostic_remaining -= 1
            pre_m, pre_deg, goal_m, goal_deg = controller.pose_errors(snapshot)
            gap = _measured_signed_gap(snapshot, pose, config)
            row = {
                "physics_frame_id": snapshot["physics_frame_id"],
                "attempt": name,
                "state": command.state.value,
                "actual_joint26": snapshot["joint_pos26"].tolist(),
                "applied_target26": snapshot["joint_target26"].tolist(),
                "next_target26": list(command.target_q26),
                "joint_error26": (snapshot["joint_target26"] - snapshot["joint_pos26"]).tolist(),
                "frame_position": snapshot["frame_position"].tolist(),
                "frame_quat_wxyz": snapshot["frame_quat_wxyz"].tolist(),
                "frame_linear_velocity": snapshot["frame_linear_velocity"].tolist(),
                "frame_angular_velocity": snapshot["frame_angular_velocity"].tolist(),
                "fixed_position": snapshot["fixed_position"].tolist(),
                "fixed_quat_wxyz": snapshot["fixed_quat_wxyz"].tolist(),
                "palm_position": snapshot["palm_position"].tolist(),
                "palm_quat_wxyz": snapshot["palm_quat_wxyz"].tolist(),
                "support_link_names": snapshot["support_link_names"],
                "support_link_positions": snapshot["support_link_positions"].tolist(),
                "support_force_n": snapshot["support_force_n"].tolist(),
                "support_force_xyz": snapshot["support_force_xyz"].tolist(),
                "frame_fixed_force_n": snapshot["frame_fixed_force_n"],
                "frame_fixed_force_xyz": snapshot["frame_fixed_force_xyz"].tolist(),
                "frame_table_force_n": snapshot["frame_table_force_n"],
                "frame_table_force_xyz": snapshot["frame_table_force_xyz"].tolist(),
                "support_fingers": list(contact_summary.support_fingers),
                "support_links": list(contact_summary.support_links),
                "hand_frame_contact": contact_summary.hand_frame_contact,
                "frame_fixed_contact": contact_summary.frame_fixed_contact,
                "frame_table_contact": contact_summary.frame_table_contact,
                "frame_ground_contact": contact_summary.frame_ground_contact,
                "illegal_contact": contact_summary.illegal_contact,
                "illegal_pairs": [list(pair) for pair in contact_summary.illegal_pairs],
                "signed_gap_m": gap,
                "preinsert_position_error_m": pre_m,
                "preinsert_rotation_error_deg": pre_deg,
                "goal_position_error_m": goal_m,
                "goal_rotation_error_deg": goal_deg,
                "relative_position_drift_m": controller.relative_drift(snapshot)[0],
                "relative_rotation_drift_deg": controller.relative_drift(snapshot)[1],
                "post_reset_object_root_writes": snapshot["post_reset_object_root_writes"],
                "post_reset_wrist_state_writes": snapshot["post_reset_wrist_state_writes"],
                "post_reset_fixed_asset_root_writes": snapshot["post_reset_fixed_asset_root_writes"],
                "terminal": command.terminal,
                "failure": command.failure.value,
                "diagnostic_post_terminal": diagnostic_hold,
            }
            trace_rows.append(row)
            event_payload = {
                "physics_frame_id": snapshot["physics_frame_id"],
                "events": [event.to_dict() for event in events],
            }
            contact_rows.append(event_payload)
            if int(snapshot["physics_frame_id"]) % capture_every == 0:
                frame = env.rgb_frame()
                if frame is not None:
                    raw_writer.append_data(frame)
                    annotated = _overlay(frame, row)
                    annotated_writer.append_data(annotated)
                    alignment.append(
                        {
                            "video_frame_index": len(alignment),
                            "physics_frame_id": snapshot["physics_frame_id"],
                            "state": command.state.value,
                        }
                    )
            action = torch.as_tensor(command.target_q26, dtype=torch.float32, device=env.device).reshape(1, 26)
            if command.terminal and not diagnostic_hold:
                terminal_diagnostic_remaining = int(config["runtime"]["terminal_diagnostic_steps"])
            elif diagnostic_hold and terminal_diagnostic_remaining <= 0:
                break
    finally:
        raw_writer.close()
        annotated_writer.close()

    if command is None:
        raise RuntimeError("rollout produced no physics frames")
    _write_trace(trace_path, trace_rows)
    _write_jsonl(contacts_path, contact_rows)
    _write_json(
        alignment_path,
        {
            "attempt": name,
            "trace_path": str(trace_path),
            "raw_video": str(raw_path),
            "annotated_video": str(annotated_path),
            "frames": alignment,
        },
    )
    result = evaluator.finalize(
        physical_support_acquired=controller.milestones.physical_support_acquired,
        physical_lift_success=controller.milestones.physical_lift_success,
        preinsert_success=controller.milestones.preinsert_success,
        physical_insert_success=controller.milestones.physical_insert_success,
        release_stable=controller.milestones.release_stable,
        failure=controller.failure,
        write_audit=env.root_write_audit(),
        trace_path=str(trace_path),
        video_path=str(annotated_path),
        metadata={
            "attempt": name,
            "reset_cache": reset_cache,
            "sticky_used": False,
            "snap_used": False,
            "fixed_joint_used": False,
            "proxy_used": False,
            "object_follow_used": False,
            "teacher_motion_used": False,
            "candidate_generation_used": False,
            "cem_started": False,
            "ppo_started": False,
        },
    ).to_dict()
    _write_json(directory / "result.json", result)
    _write_json(directory / "root_write_audit.json", env.root_write_audit())
    return {
        "result": result,
        "trace_path": trace_path,
        "contacts_path": contacts_path,
        "alignment_path": alignment_path,
        "raw_video": raw_path,
        "annotated_video": annotated_path,
        "trace_rows": trace_rows,
        "contact_rows": contact_rows,
    }


def _derive_correction(execution, pose, config, used: set[str]) -> dict[str, Any] | None:
    import numpy as np

    result = execution["result"]
    rows = execution["trace_rows"]
    gaps = [
        float(row["signed_gap_m"])
        for row in rows[:40]
        if not row.get("diagnostic_post_terminal") and np.isfinite(row["signed_gap_m"])
    ]
    measured_gap = float(np.median(gaps)) if gaps else float(pose.signed_gap_m)
    target = float(config["fork"]["target_gap_m"])
    if not result["physical_support_acquired"] and "height" not in used and not (
        float(config["fork"]["allowed_gap_min_m"])
        <= measured_gap
        <= float(config["fork"]["allowed_gap_max_m"])
    ):
        delta = float(
            np.clip(
                measured_gap - target,
                -float(config["fork"]["max_height_correction_m"]),
                float(config["fork"]["max_height_correction_m"]),
            )
        )
        return {"kind": "height", "delta_m": delta, "measured_gap_m": measured_gap}
    positions = []
    for payload in execution["contact_rows"]:
        for event in payload["events"]:
            actors = f"{event['actor0']} {event['actor1']}"
            if "/Robot/" in actors and "/Frame" in actors:
                positions.extend(event.get("positions_world", ()))
    if not result["physical_support_acquired"] and "normal" not in used and positions:
        centroid = np.mean(np.asarray(positions, dtype=np.float64), axis=0)
        expected = np.mean(np.asarray(pose.support_points_world, dtype=np.float64), axis=0)
        axis = np.asarray(pose.metadata["axis_world"], dtype=np.float64)
        delta = float(
            np.clip(
                np.dot(expected - centroid, axis),
                -float(config["fork"]["max_normal_correction_m"]),
                float(config["fork"]["max_normal_correction_m"]),
            )
        )
        if abs(delta) > 1.0e-5:
            return {"kind": "normal", "delta_m": delta, "contact_centroid_world": centroid.tolist()}
    if result["physical_support_acquired"] and not result["physical_lift_success"] and "hook" not in used:
        return {"kind": "hook", "fraction": float(config["fork"]["fallback_partial_hook_fraction"])}
    return None


def _measured_signed_gap(snapshot, pose, config) -> float:
    import numpy as np
    from scipy.spatial.transform import Rotation

    geometry = json.loads((REPO_ROOT / config["assets"]["frame_geometry_audit"]).read_text())
    bar = next(row for row in geometry["bars"] if row["bar_id"] == "top_horizontal")
    bounds = np.asarray(bar["surface_bounds_object"], dtype=np.float64)
    corners = np.asarray(
        [[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]], dtype=np.float64
    )
    quat = np.asarray(snapshot["frame_quat_wxyz"], dtype=np.float64)
    rotation = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
    bar_world = np.asarray(snapshot["frame_position"]) + corners @ rotation.T
    underside = float(np.min(bar_world[:, 2]))
    target_wrist = np.asarray(pose.wrist_q6[:3], dtype=np.float64)
    actual_wrist = np.asarray(snapshot["joint_pos26"][:3], dtype=np.float64)
    support = np.asarray(pose.support_points_world, dtype=np.float64) + (actual_wrist - target_wrist)
    return underside - float(np.mean(support[:, 2]))


def _overlay(frame, row):
    import cv2

    output = frame.copy()
    lines = [
        "WUJI FRAME STAGE 4 - PHYSICAL ROLLOUT",
        f"phase: {row['state']}",
        f"support fingers: {row['support_fingers']}  peak force: {max(row['support_force_n'], default=0.0):.3f} N",
        f"frame z: {row['frame_position'][2]:.3f} m  goal error: {1000.0*row['goal_position_error_m']:.1f} mm",
        "SNAP/STICKY/FIXED JOINT/OBJECT FOLLOW: NONE",
    ]
    shade = output.copy()
    cv2.rectangle(shade, (0, 0), (960, 165), (0, 0, 0), thickness=-1)
    cv2.addWeighted(shade, 0.68, output, 0.32, 0.0, output)
    for index, line in enumerate(lines):
        cv2.putText(output, line, (18, 30 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _publish_selected(selected: Mapping[str, Any], physical_insert_success: bool) -> None:
    artifacts = OUTPUT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["trace_path"], artifacts / "full_route_trace.csv")
    shutil.copy2(selected["contacts_path"], artifacts / "contact_pairs.jsonl")
    shutil.copy2(selected["alignment_path"], artifacts / "video_alignment.json")
    shutil.copy2(selected["trace_path"], artifacts / "pose_error_trace.csv")
    _write_json(artifacts / "root_write_audit.json", {
        key: selected["result"][key]
        for key in (
            "post_reset_object_root_writes",
            "post_reset_wrist_state_writes",
            "post_reset_fixed_asset_root_writes",
        )
    })
    if physical_insert_success:
        shutil.copy2(selected["raw_video"], OUTPUT / "videos/frame_stage4_delivery_raw.mp4")
        shutil.copy2(selected["annotated_video"], OUTPUT / "videos/frame_stage4_delivery.mp4")


def _build_delivery(final: Mapping[str, Any], selected: Mapping[str, Any]) -> None:
    destination = REPO_ROOT / "5/frame_stage4_delivery_rc2"
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "artifacts").mkdir(parents=True)
    (destination / "videos").mkdir(parents=True)
    for path in (OUTPUT / "artifacts").glob("*"):
        shutil.copy2(path, destination / "artifacts" / path.name)
    shutil.copy2(OUTPUT / "final_summary.json", destination / "artifacts/final_summary.json")
    shutil.copy2(OUTPUT / "runtime_fingerprint.json", destination / "artifacts/runtime_fingerprint.json")
    shutil.copy2(OUTPUT / "gpu_preflight.json", destination / "artifacts/gpu_preflight.json")
    for name in ("debug_first.mp4", "frame_stage4_delivery.mp4", "frame_stage4_delivery_raw.mp4"):
        source = OUTPUT / "videos" / name
        if source.is_file():
            shutil.copy2(source, destination / "videos" / name)
    snapshot_paths = [
        "configs/physical_delivery/frame_stage4_rc2.yaml",
        "scripts/environments/run_frame_stage4_delivery_rc2.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/physical_delivery/contracts.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/physical_delivery/frame_fork_support.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/physical_delivery/frame_stage4_env.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/physical_delivery/frame_stage4_controller.py",
        "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/physical_delivery/frame_stage4_evaluator.py",
        "source/isaaclab_tasks/test/test_frame_stage4_delivery_rc2.py",
    ]
    for relative in snapshot_paths:
        source = REPO_ROOT / relative
        if source.is_file():
            target = destination / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    _write_delivery_docs(destination, final)
    manifest = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            manifest.append(
                {
                    "path": str(path.relative_to(destination)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(destination / "sha256_manifest.json", {"schema_version": 1, "files": manifest})


def _write_delivery_docs(destination: Path, final: Mapping[str, Any]) -> None:
    command = "bash 5/frame_stage4_delivery_rc2/reproduce.sh"
    (destination / "README.md").write_text(
        "# Wuji Frame Stage 4 Delivery RC2\n\n"
        f"Delivery level: `{final['delivery_level']}`\n\n"
        "Run exactly one command from the repository root:\n\n"
        f"```bash\n{command}\n```\n",
        encoding="utf-8",
    )
    (destination / "STATUS.md").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (destination / "ARCHITECTURE.md").write_text(
        "# Architecture\n\nA single DirectRLEnv scene executes one deterministic three-finger fork route. "
        "All post-reset motion uses 26D articulation position targets; no object or wrist root pose is written.\n",
        encoding="utf-8",
    )
    runtime_command = (
        "TERM=xterm-256color ./isaaclab.sh -p scripts/environments/run_frame_stage4_delivery_rc2.py "
        "--headless --enable_cameras --device cuda:0"
    )
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../..\"\n"
        "source /home/CNS2025331827/miniconda3/etc/profile.d/conda.sh\n"
        "conda activate isaac\n"
        "export PYTHONPATH=\"$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:${PYTHONPATH:-}\"\n"
        + runtime_command
        + "\n"
    )
    (destination / "reproduce.sh").write_text(script, encoding="utf-8")
    os.chmod(destination / "reproduce.sh", 0o755)


def _classification(final: Mapping[str, Any]) -> str:
    return {
        "A": "FRAME_PHYSICAL_INSERT_RELEASE_STABLE",
        "B": "FRAME_PHYSICAL_INSERT_HELD",
        "C": "FRAME_PHYSICAL_TRANSPORT_PREINSERT",
        "NONE": "FRAME_STAGE4_PHYSICAL_ROUTE_FAILED",
    }[str(final["delivery_level"])]


def _validate_assets(runtime: Mapping[str, Any]) -> None:
    if Path(str(runtime["frame_asset"])).name != "frame_mirror.usd":
        raise RuntimeError("wrong moving Frame asset")
    if Path(str(runtime["fixed_asset"])).name != "frame_back_rod_rod.usd":
        raise RuntimeError("wrong Stage4 fixed asset")


def _gpu_preflight(min_free_gb: float) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    rows = []
    for line in result.stdout.splitlines() if result.returncode == 0 else []:
        values = [item.strip() for item in line.split(",")]
        if len(values) >= 5:
            rows.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "total_mib": int(values[2]),
                    "used_mib": int(values[3]),
                    "free_mib": int(values[4]),
                }
            )
    processes = subprocess.run(
        ["ps", "-eo", "pid,ppid,cmd"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    active = [line.strip() for line in processes.stdout.splitlines() if any(key in line.lower() for key in ("isaac", "kit"))]
    maximum = max((row["free_mib"] for row in rows), default=0)
    return {
        "sufficient": bool(result.returncode == 0 and maximum >= float(min_free_gb) * 1024.0),
        "minimum_free_gb": float(min_free_gb),
        "gpus": rows,
        "active_isaac_kit_processes": active,
        "processes_killed": False,
        "nvidia_smi_error": "" if result.returncode == 0 else result.stdout.strip(),
    }


def _honesty() -> dict[str, Any]:
    return {
        "sticky_used": False,
        "snap_used": False,
        "fixed_joint_used": False,
        "proxy_used": False,
        "object_follow_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used_after_reset": False,
        "cem_started": False,
        "ppo_started": False,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, tuple)) else value
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
