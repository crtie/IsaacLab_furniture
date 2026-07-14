"""Video/log/contact alignment helpers for the scripted Screw1 baseline."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Isaac runtime dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


PART_NAMES = ("Plug2", "Screw1", "Backrest", "Rod", "Frame")
FORCE_THRESHOLD_N = 0.05


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_status(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            proc = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, check=False)
            return (proc.stdout or proc.stderr or "").strip()
        except Exception as exc:
            return f"{type(exc).__name__}:{exc}"

    return {
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty_status": _run(["git", "status", "--short"]),
    }


def write_run_manifest(
    output_dir: Path,
    *,
    run_id: str,
    start_time: str,
    end_time: str | None,
    command_line: list[str],
    repo_root: Path,
    target_part: str,
    target_env_index: int | None = None,
    summary_path: Path | None = None,
    log_path: Path | None = None,
    video_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_mtime = None
    if video_path is not None and Path(video_path).is_file():
        video_mtime = Path(video_path).stat().st_mtime
    manifest = {
        "run_id": run_id,
        "start_time": start_time,
        "end_time": end_time or "",
        "git_commit_or_dirty_status_if_available": git_status(repo_root),
        "command_line": list(command_line),
        "output_dir": str(output_dir),
        "video_path": str(video_path or ""),
        "video_mtime": video_mtime,
        "summary_path": str(summary_path or output_dir / "rollout_summary.json"),
        "log_path": str(log_path or output_dir / "rollout_log.csv"),
        "target_part": str(target_part),
        "target_env_index": "" if target_env_index is None else int(target_env_index),
        "old_mp4_provenance": "unverified",
    }
    if extra:
        manifest.update(extra)
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


class AlignmentRecorder:
    """Collects video/log/action/contact alignment evidence for one baseline run."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        run_id: str,
        target_part: str,
        target_env_index: int,
        video_recorder: Any | None,
    ):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.target_part = str(target_part)
        self.target_env_index = int(target_env_index)
        self.video_recorder = video_recorder
        self.alignment_step_index = 0
        self.camera_info: dict[str, Any] = {}
        self.body_mapping: dict[str, Any] = {}
        self.contact_rows: list[dict[str, Any]] = []
        self._contact_sensors: list[dict[str, Any]] = []
        self._contact_sensor_setup_done = False
        self._contact_sensor_setup_error = ""
        self._markers: Any | None = None
        self._marker_setup_error = ""
        self._last_action_audit: dict[str, Any] = {}
        self._last_marker_positions: dict[str, list[float]] = {}

    def frame_count(self) -> int:
        if self.video_recorder is None:
            return -1
        return int(getattr(self.video_recorder, "_continuous_frame_count", -1) or 0)

    def capture_frame(self) -> int:
        if not self.enabled or self.video_recorder is None:
            return -1
        before = self.frame_count()
        try:
            write_frame = getattr(self.video_recorder, "_write_frame", None)
            if callable(write_frame):
                write_frame()
        except Exception:
            return -1
        after = self.frame_count()
        if after > before:
            return after - 1
        return after - 1 if after > 0 else -1

    def configure_camera(self, env: Any, base: Any, state: dict[str, Any]) -> None:
        if not self.enabled:
            return
        target = list(state.get("object_world_pos") or _world_from_local(base, self.target_env_index, state["object_local_pos"]))
        eye = [float(target[0]) + 0.82, float(target[1]) - 0.72, float(target[2]) + 0.48]
        origins = _tensor_matrix_any(getattr(getattr(base, "scene", None), "env_origins", None))
        nearest = _nearest_env_to_point(origins, target)
        try:
            sim = getattr(_base_env(env), "sim", None)
            if sim is not None and hasattr(sim, "set_camera_view"):
                sim.set_camera_view(eye=tuple(eye), target=tuple(target))
        except Exception as exc:
            self.camera_info["camera_set_error"] = f"{type(exc).__name__}:{exc}"
        self.camera_info.update(
            {
                "camera_position": eye,
                "camera_target": target,
                "nearest_env_to_camera_target": nearest,
                "target_env_origin": _list_get(origins, self.target_env_index, [0.0, 0.0, 0.0]),
                "all_env_origins": origins,
                "script_target_env": self.target_env_index,
                "target_part": self.target_part,
                "camera_env_mismatch": bool(nearest != self.target_env_index),
            }
        )

    def setup_body_mapping_and_sensors(self, base: Any) -> None:
        if not self.enabled or self._contact_sensor_setup_done:
            return
        self._contact_sensor_setup_done = True
        robot = getattr(base, "_robot", None)
        body_names = list(getattr(robot, "body_names", []) or [])
        groups = _group_robot_bodies(body_names)
        object_info = _object_info(base)
        self.body_mapping = {
            "robot_body_names": body_names,
            "finger3_tip_body_index": _safe_index(body_names, "right_finger3_tip_link"),
            "finger3_tip_body_name": "right_finger3_tip_link" if "right_finger3_tip_link" in body_names else "",
            "finger4_tip_body_index": _safe_index(body_names, "right_finger4_tip_link"),
            "finger4_tip_body_name": "right_finger4_tip_link" if "right_finger4_tip_link" in body_names else "",
            "finger3_proximal_body_indices_names": [
                [_safe_index(body_names, name), name] for name in groups.get("finger3_proximal", [])
            ],
            "finger4_proximal_body_indices_names": [
                [_safe_index(body_names, name), name] for name in groups.get("finger4_proximal", [])
            ],
            "palm_body_indices_names": [[_safe_index(body_names, name), name] for name in groups.get("palm", [])],
            "wrist_body_indices_names": [[_safe_index(body_names, name), name] for name in groups.get("wrist", [])],
            "proximal_finger_body_indices_names": [
                [_safe_index(body_names, name), name] for name in groups.get("finger_proximal_links", [])
            ],
            "body_groups": groups,
            "object_prim_paths_asset_names_env_slots": object_info,
        }
        write_json(self.output_dir / "body_mapping.json", self.body_mapping)
        self._setup_contact_sensors(base, groups)

    def setup_markers(self) -> None:
        if not self.enabled or self._markers is not None or self._marker_setup_error:
            return
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            colors = {
                "screw1_root": (1.0, 0.0, 0.0),
                "contact_target": (1.0, 1.0, 0.0),
                "finger3_tip": (0.0, 1.0, 0.0),
                "finger4_tip": (0.0, 1.0, 1.0),
                "palm": (0.0, 0.2, 1.0),
                "wrist": (1.0, 0.0, 1.0),
            }
            markers = {}
            for name, color in colors.items():
                markers[name] = sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            cfg = VisualizationMarkersCfg(prim_path="/World/Visuals/scripted_baseline_alignment", markers=markers)
            self._markers = VisualizationMarkers(cfg)
        except Exception as exc:
            self._marker_setup_error = f"{type(exc).__name__}:{exc}"

    def update_markers(self, base: Any, state: dict[str, Any], contact_target_local: list[float] | None = None) -> None:
        if not self.enabled:
            return
        self.setup_markers()
        origin = _tensor_vec_any(getattr(getattr(base, "scene", None), "env_origins", None), self.target_env_index)
        contact_target_local = contact_target_local or list(state.get("object_local_pos", [0.0, 0.0, 0.0]))
        positions = {
            "screw1_root": list(state.get("object_world_pos") or _add_vec(origin, state["object_local_pos"])),
            "contact_target": _add_vec(origin, contact_target_local),
            "finger3_tip": _add_vec(origin, state["finger3_tip_local_pos"]),
            "finger4_tip": _add_vec(origin, state["finger4_tip_local_pos"]),
            "palm": _add_vec(origin, state["palm_local_pos"]),
            "wrist": _wrist_world_pos(base, self.target_env_index, fallback=_add_vec(origin, state["palm_local_pos"])),
        }
        self._last_marker_positions = positions
        if self._markers is None:
            return
        try:
            translations = [positions[name] for name in positions]
            marker_indices = list(range(len(translations)))
            if torch is not None:
                translations = torch.tensor(translations, dtype=torch.float32, device=getattr(base, "device", None))
                marker_indices = torch.tensor(marker_indices, dtype=torch.long, device=getattr(base, "device", None))
            self._markers.visualize(translations=translations, marker_indices=marker_indices)
        except Exception as exc:
            self._marker_setup_error = f"{type(exc).__name__}:{exc}"

    def audit_action(self, mapped_action: Any) -> dict[str, Any]:
        if not self.enabled:
            return {}
        rows = _rows_from_action(mapped_action)
        norms = [_norm(row) for row in rows]
        target_norm = _list_get(norms, self.target_env_index, 0.0)
        non_target = [value for idx, value in enumerate(norms) if idx != self.target_env_index]
        active = [idx for idx, value in enumerate(norms) if float(value) > 1.0e-8]
        audit = {
            "target_env_action_norm": target_norm,
            "non_target_env_action_norm_max": max([0.0, *non_target]),
            "non_target_env_action_norm_sum": sum(non_target),
            "active_action_env_ids": ";".join(str(idx) for idx in active),
            "action_sent_only_to_env_1": bool(active == [self.target_env_index]),
        }
        self._last_action_audit = audit
        return audit

    def make_log_extra(self, frame_index: int, action_audit: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        audit = action_audit or self._last_action_audit or {}
        out = {
            "run_id": self.run_id,
            "alignment_step_index": self.alignment_step_index,
            "video_frame_index": int(frame_index) if frame_index is not None else -1,
            "sim_step_or_common_step_counter": "",
            **audit,
        }
        self.alignment_step_index += 1
        return out

    def record_contacts(self, phase: str, step: int, state: dict[str, Any], base: Any) -> None:
        if not self.enabled:
            return
        if not self._contact_sensor_setup_done:
            self.setup_body_mapping_and_sensors(base)
        self.contact_rows.extend(_existing_tip_contact_rows(phase, step, state, base, self.target_env_index, self.target_part))
        for record in self._contact_sensors:
            self.contact_rows.extend(_sensor_contact_rows(record, phase, step, self.target_env_index, self.target_part))

    def finalize(self, logs: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        contact_log_path = self.output_dir / "contact_alignment_log.csv"
        contact_summary_path = self.output_dir / "contact_alignment_summary.json"
        _write_csv(contact_log_path, self.contact_rows)
        contact_summary = summarize_contacts(
            self.contact_rows,
            camera_info=self.camera_info,
            action_rows=logs,
            setup_error=self._contact_sensor_setup_error,
            target_part=self.target_part,
        )
        write_json(contact_summary_path, contact_summary)
        aligned_video = self.output_dir / "videos" / "scripted_screw1_baseline_aligned.mp4"
        source_video = Path(str(summary.get("video_path") or ""))
        if not bool(summary.get("video_available")) or not source_video.is_file():
            overlay_info = {
                "aligned_video_written": False,
                "aligned_video_path": "",
                "aligned_video_frame_count": 0,
                "aligned_video_error": "source_video_unavailable_for_this_run",
            }
        else:
            overlay_info = write_aligned_video(
                source_video,
                aligned_video,
                logs,
                {
                    **summary,
                    **self.camera_info,
                    "run_id": self.run_id,
                    "contact_alignment": contact_summary,
                },
            )
        aligned_frame = self.output_dir / "final_frame_aligned.png"
        source_frame = Path(str(summary.get("final_frame_png") or ""))
        if not source_frame.is_file():
            frame_info = {
                "aligned_final_frame_written": False,
                "aligned_final_frame_path": "",
                "aligned_final_frame_error": "source_final_frame_unavailable_for_this_run",
            }
        else:
            frame_info = write_aligned_final_frame(
                source_frame,
                aligned_frame,
                {
                    **summary,
                    **self.camera_info,
                    "run_id": self.run_id,
                    "marker_positions_world": self._last_marker_positions,
                    "marker_setup_error": self._marker_setup_error,
                    "contact_alignment": contact_summary,
                },
            )
        return {
            "alignment_debug": True,
            "run_id": self.run_id,
            **self.camera_info,
            "body_mapping_json": str(self.output_dir / "body_mapping.json"),
            "contact_alignment_log_csv": str(contact_log_path),
            "contact_alignment_summary_json": str(contact_summary_path),
            "aligned_video_path": str(aligned_video),
            "aligned_final_frame_path": str(aligned_frame),
            "marker_setup_error": self._marker_setup_error,
            **overlay_info,
            **frame_info,
            **contact_summary,
        }

    def _setup_contact_sensors(self, base: Any, groups: dict[str, list[str]]) -> None:
        try:
            import copy
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import ContactSensor
            from pxr import PhysxSchema
        except Exception as exc:
            self._contact_sensor_setup_error = f"import_error:{type(exc).__name__}:{exc}"
            return
        body_names = []
        for group in (
            "finger3_tip",
            "finger4_tip",
            "finger3_proximal",
            "finger4_proximal",
            "palm",
            "wrist",
            "finger_proximal_links",
            "other_hand_links",
        ):
            body_names.extend(groups.get(group, []))
        body_names = list(dict.fromkeys(body_names))
        sensor_cfg_template = getattr(getattr(base, "cfg", None), "dex_fingertip_force_sensor", None)
        if sensor_cfg_template is None:
            self._contact_sensor_setup_error = "dex_fingertip_force_sensor_cfg_missing"
            return
        stage_errors = []
        for body_name in body_names:
            group = _classify_body(body_name)
            try:
                pattern = f"/World/envs/env_.*/Robot/{body_name}"
                for prim in sim_utils.find_matching_prims(pattern):
                    if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        api = PhysxSchema.PhysxContactReportAPI.Get(prim.GetStage(), prim.GetPrimPath())
                    else:
                        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                    api.CreateThresholdAttr().Set(0.0)
                cfg = copy.deepcopy(sensor_cfg_template)
                cfg.prim_path = pattern
                cfg.filter_prim_paths_expr = [f"/World/envs/env_.*/{part}" for part in PART_NAMES]
                sensor = ContactSensor(cfg)
                status = "created"
                try:
                    sensor._initialize_impl()
                    sensor._is_initialized = True
                    status = "initialized"
                except Exception as exc:
                    status = f"init_error:{type(exc).__name__}:{exc}"
                self._contact_sensors.append(
                    {
                        "body_name": body_name,
                        "hand_group": group,
                        "sensor": sensor,
                        "sensor_status": status,
                    }
                )
            except Exception as exc:
                stage_errors.append(f"{body_name}:{type(exc).__name__}:{exc}")
        if stage_errors:
            self._contact_sensor_setup_error = "; ".join(stage_errors[:8])


def summarize_contacts(
    rows: list[dict[str, Any]],
    *,
    camera_info: dict[str, Any],
    action_rows: list[dict[str, Any]],
    setup_error: str,
    target_part: str,
) -> dict[str, Any]:
    peaks: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    unresolved_peak = 0.0
    unresolved_reason = ""
    for row in rows:
        group = str(row.get("hand_group", ""))
        obj = str(row.get("object_name", ""))
        force = _float(row.get("force_norm_or_contact_strength", 0.0))
        key = (group, obj)
        if force > peaks.get(key, (0.0, {}))[0]:
            peaks[key] = (force, row)
        if obj == "unknown_object" and force > unresolved_peak:
            unresolved_peak = force
            unresolved_reason = str(row.get("sensor_status", ""))
    def peak(group: str, obj: str) -> float:
        return float(peaks.get((group, obj), (0.0, {}))[0])

    finger3 = peak("finger3_tip", target_part)
    finger4 = peak("finger4_tip", target_part)
    palm_wrist = max(peak("palm", target_part), peak("wrist", target_part))
    proximal_other = max(
        peak("finger3_proximal", target_part),
        peak("finger4_proximal", target_part),
        peak("finger_proximal_links", target_part),
        peak("other_hand_links", target_part),
    )
    non_target_peak = 0.0
    non_target_row = {}
    for (group, obj), (force, row) in peaks.items():
        if obj not in ("", target_part, "unknown_object") and force > non_target_peak:
            non_target_peak = force
            non_target_row = row
    threshold = FORCE_THRESHOLD_N
    codes = []
    if bool(camera_info.get("camera_env_mismatch")):
        codes.append("E")
    if max(finger3, finger4) > threshold:
        codes.append("A")
    if palm_wrist > threshold:
        codes.append("B")
    if proximal_other > threshold:
        codes.append("C")
    if non_target_peak > threshold:
        codes.append("D")
    debug_unavailable = bool(setup_error) or any("error" in str(row.get("sensor_status", "")) for row in rows)
    if debug_unavailable and not any(code in codes for code in ("A", "B", "C", "D")):
        codes.append("F")
    if not codes:
        codes.append("no_real_contact_visual_overlap_or_miss")

    action_active = set()
    non_target_action_max = 0.0
    for row in action_rows:
        active = str(row.get("active_action_env_ids", ""))
        for item in active.split(";"):
            if item.strip():
                try:
                    action_active.add(int(item))
                except Exception:
                    pass
        non_target_action_max = max(non_target_action_max, _float(row.get("non_target_env_action_norm_max", 0.0)))
    action_only_env1 = bool(action_active == {1} and non_target_action_max <= 1.0e-8)

    reason = "no real contact; visual overlap/motion looked like contact"
    if bool(camera_info.get("camera_env_mismatch")):
        reason = "camera saw wrong env"
    elif not action_only_env1:
        reason = "action moved wrong env"
    elif palm_wrist > threshold or proximal_other > threshold:
        reason = "contact body was palm/wrist/proximal, not finger3/finger4 tip_link"
    elif non_target_peak > threshold:
        reason = "contact object was non-target, not Screw1"
    elif debug_unavailable:
        reason = "debug contact sensors did not cover that body/object pair"

    next_fix = "pregrasp"
    if bool(camera_info.get("camera_env_mismatch")):
        next_fix = "camera/env selection"
    elif not action_only_env1:
        next_fix = "action scale"
    elif debug_unavailable:
        next_fix = "contact logging"
    elif palm_wrist > threshold or proximal_other > threshold:
        next_fix = "hand body selection"
    elif non_target_peak > threshold:
        next_fix = "target/contact surface selection"

    return {
        "finger3_tip_to_screw1_contact": bool(finger3 > threshold),
        "finger3_tip_to_screw1_peak_force_n": finger3,
        "finger4_tip_to_screw1_contact": bool(finger4 > threshold),
        "finger4_tip_to_screw1_peak_force_n": finger4,
        "palm_wrist_to_screw1_contact": bool(palm_wrist > threshold),
        "palm_wrist_to_screw1_peak_force_n": palm_wrist,
        "proximal_other_finger_to_screw1_contact": bool(proximal_other > threshold),
        "proximal_other_finger_to_screw1_peak_force_n": proximal_other,
        "any_hand_body_to_non_target_object_contact": bool(non_target_peak > threshold),
        "non_target_object_peak_force_n": non_target_peak,
        "non_target_object_peak_body": str(non_target_row.get("hand_body_name", "")),
        "non_target_object_peak_object": str(non_target_row.get("object_name", "")),
        "unknown_unresolved_contact": bool(unresolved_peak > threshold or debug_unavailable),
        "unknown_unresolved_peak_force_n": unresolved_peak,
        "unknown_unresolved_reason": unresolved_reason or setup_error,
        "all_body_object_resolved_contact_unavailable": setup_error,
        "video_hand_pressing_chair_edge_corresponds_to": ",".join(codes),
        "why_finger3_finger4_screw1_target_filtered_force_zero": reason,
        "action_sent_only_to_env_1": action_only_env1,
        "active_action_env_ids": sorted(action_active),
        "non_target_env_action_norm_max": non_target_action_max,
        "next_baseline_fix": next_fix,
    }


def write_aligned_video(source: Path, target: Path, logs: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return {"aligned_video_written": False, "aligned_video_error": f"cv2_unavailable:{type(exc).__name__}:{exc}"}
    if not source.is_file():
        return {"aligned_video_written": False, "aligned_video_error": "source_video_missing"}
    by_frame = {}
    last = {}
    for row in logs:
        frame = int(_float(row.get("video_frame_index", -1)))
        if frame >= 0:
            by_frame[frame] = row
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return {"aligned_video_written": False, "aligned_video_error": "cv2_open_source_failed"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    tmp = target.with_name(f"{target.stem}_raw{target.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"aligned_video_written": False, "aligned_video_error": "cv2_writer_open_failed"}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if index in by_frame:
            last = by_frame[index]
        _draw_overlay(frame, index, last, summary)
        writer.write(frame)
        index += 1
    cap.release()
    writer.release()
    transcode = _transcode_to_h264(tmp, target)
    if transcode.get("ok"):
        tmp.unlink(missing_ok=True)
    return {
        "aligned_video_written": bool(target.is_file()),
        "aligned_video_frame_count": index,
        "aligned_video_resolution": f"{width}x{height}",
        "aligned_video_error": "" if target.is_file() else str(transcode.get("error", "transcode_failed")),
    }


def write_aligned_final_frame(source: Path, target: Path, summary: dict[str, Any]) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return {"aligned_final_frame_written": False, "aligned_final_frame_error": f"cv2_unavailable:{type(exc).__name__}:{exc}"}
    if not source.is_file():
        return {"aligned_final_frame_written": False, "aligned_final_frame_error": "source_final_frame_missing"}
    frame = cv2.imread(str(source))
    if frame is None:
        return {"aligned_final_frame_written": False, "aligned_final_frame_error": "cv2_read_failed"}
    _draw_overlay(frame, -1, {}, summary, final_frame=True)
    y = 170
    markers = dict(summary.get("marker_positions_world", {}) or {})
    for name, pos in markers.items():
        text = f"{name}: world=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(target), frame)
    return {"aligned_final_frame_written": bool(ok), "aligned_final_frame_error": "" if ok else "cv2_write_failed"}


def _draw_overlay(frame: Any, frame_index: int, row: dict[str, Any], summary: dict[str, Any], final_frame: bool = False) -> None:
    import cv2  # type: ignore

    lines = [
        f"run_id={summary.get('run_id','')}",
        f"frame={frame_index} phase={row.get('phase','final' if final_frame else '')} step={row.get('step','')}",
        f"script_target_env={summary.get('script_target_env', summary.get('target_env_index', 1))} camera_target_env={summary.get('nearest_env_to_camera_target','')}",
        f"target_part={summary.get('target_part','Screw1')}",
        f"surface={row.get('surface_source', summary.get('surface_source',''))} target_mode={row.get('contact_target_mode','')}",
        f"normal=({row.get('contact_target_normal_x','')},{row.get('contact_target_normal_y','')},{row.get('contact_target_normal_z','')})",
        f"active_envs={row.get('active_action_env_ids','')} non_target_action_max={row.get('non_target_env_action_norm_max','')}",
        f"f3_surface={row.get('finger3_distance_to_real_surface_m', summary.get('finger3_min_distance_to_real_surface_m',''))} press_cmd={row.get('press_depth_commanded_m', summary.get('press_depth_commanded_m', summary.get('press_depth_commanded','')))} press_actual={row.get('press_depth_actual_m', summary.get('press_depth_actual_m', summary.get('press_depth_actual','')))}",
        f"press_mode={row.get('press_control_mode_used', summary.get('press_control_mode_used',''))} policy_norm={row.get('commanded_policy_xyz_norm', summary.get('press_policy_action_norm_peak',''))} mapped_norm={row.get('mapped_isaac_action_xyz_norm', summary.get('press_mapped_isaac_action_norm_peak',''))} ctrl_moved={row.get('ctrl_target_moved', summary.get('press_ctrl_target_moved',''))}",
        f"press_dot={row.get('finger3_delta_dot_press_dir_cumulative', summary.get('finger3_delta_dot_press_dir',''))} z_clamp={row.get('z_clamp_after_z_raise', summary.get('z_clamp_max_during_press',''))}",
        f"f3_tf={row.get('finger3_target_filtered_force_n', summary.get('finger3_target_filtered_force_peak_n',''))} f4_tf={row.get('finger4_target_filtered_force_n', summary.get('finger4_target_filtered_force_peak_n',''))}",
    ]
    x, y = 12, 24
    cv2.rectangle(frame, (0, 0), (1040, 250), (0, 0, 0), -1)
    for line in lines:
        cv2.putText(frame, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20


def _existing_tip_contact_rows(phase: str, step: int, state: dict[str, Any], base: Any, env_id: int, target_part: str) -> list[dict[str, Any]]:
    rows = []
    target_names = list(getattr(base, "dex_fingertip_target_filter_names", []) or PART_NAMES)
    matrix = getattr(base, "dex_fingertip_target_force_norm", None)
    vals = _tensor_matrix_any(matrix, env_id)
    for finger_index, group, body in (
        (2, "finger3_tip", "right_finger3_tip_link"),
        (3, "finger4_tip", "right_finger4_tip_link"),
    ):
        for obj_idx, obj in enumerate(target_names):
            force = 0.0
            if len(vals) > finger_index and len(vals[finger_index]) > obj_idx:
                force = float(vals[finger_index][obj_idx])
            rows.append(
                _contact_row(phase, step, env_id, group, body, obj, obj == target_part, force, "existing_tip_target_filtered_sensor", "ok")
            )
        unfiltered = _tensor_list_any(getattr(base, "dex_fingertip_force_norm", None), env_id)
        rows.append(
            _contact_row(
                phase,
                step,
                env_id,
                group,
                body,
                "unknown_object",
                False,
                _list_get(unfiltered, finger_index, 0.0),
                "existing_unfiltered_tip_sensor",
                "object_unresolved",
            )
        )
    return rows


def _sensor_contact_rows(record: dict[str, Any], phase: str, step: int, env_id: int, target_part: str) -> list[dict[str, Any]]:
    sensor = record.get("sensor")
    body = str(record.get("body_name", ""))
    group = str(record.get("hand_group", ""))
    status = str(record.get("sensor_status", ""))
    rows = []
    try:
        if sensor is None or "error" in status:
            return [_contact_row(phase, step, env_id, group, body, "unknown_object", False, 0.0, "debug_contact_sensor", status)]
        data = getattr(sensor, "data", None)
        matrix = getattr(data, "force_matrix_w", None)
        if torch is not None and torch.is_tensor(matrix) and matrix.ndim >= 4 and matrix.shape[1] > 0:
            force_matrix = matrix.detach().to("cpu")
            for obj_idx, obj in enumerate(PART_NAMES):
                force = 0.0
                if env_id < force_matrix.shape[0] and obj_idx < force_matrix.shape[2]:
                    force = float(torch.linalg.vector_norm(force_matrix[env_id, 0, obj_idx, :]).item())
                rows.append(_contact_row(phase, step, env_id, group, body, obj, obj == target_part, force, "debug_contact_sensor", status))
            return rows
        forces = getattr(data, "net_forces_w", None)
        force = 0.0
        if torch is not None and torch.is_tensor(forces) and forces.ndim >= 3 and env_id < forces.shape[0]:
            force = float(torch.linalg.vector_norm(forces.detach().to("cpu")[env_id, 0, :]).item())
        return [_contact_row(phase, step, env_id, group, body, "unknown_object", False, force, "rigid_body_net_contact_force", "force_matrix_unavailable")]
    except Exception as exc:
        return [_contact_row(phase, step, env_id, group, body, "unknown_object", False, 0.0, "debug_contact_sensor", f"read_error:{type(exc).__name__}:{exc}")]


def _contact_row(
    phase: str,
    step: int,
    env_id: int,
    hand_group: str,
    hand_body_name: str,
    object_name: str,
    is_target: bool,
    force: float,
    source: str,
    status: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "step": int(step),
        "env_id": int(env_id),
        "hand_group": hand_group,
        "hand_body_name": hand_body_name,
        "object_name": object_name,
        "is_target_object": bool(is_target),
        "force_norm_or_contact_strength": float(force),
        "raw_contact_state": "contact" if abs(float(force)) > 1.0e-8 else "no_contact",
        "source": source,
        "sensor_status": status,
    }


def _group_robot_bodies(body_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "finger3_tip": [],
        "finger4_tip": [],
        "finger3_proximal": [],
        "finger4_proximal": [],
        "palm": [],
        "wrist": [],
        "finger_proximal_links": [],
        "other_hand_links": [],
    }
    for name in body_names:
        group = _classify_body(name)
        if group in groups:
            groups[group].append(name)
    return groups


def _classify_body(name: str) -> str:
    if name == "right_finger3_tip_link":
        return "finger3_tip"
    if name == "right_finger4_tip_link":
        return "finger4_tip"
    if "palm" in name:
        return "palm"
    if "wrist" in name or name in {"panda_hand", "panda_link7"}:
        return "wrist"
    import re

    if re.match(r"right_finger3_link[123]$", name):
        return "finger3_proximal"
    if re.match(r"right_finger4_link[123]$", name):
        return "finger4_proximal"
    if re.match(r"right_finger\d+_link[123]$", name):
        return "finger_proximal_links"
    if name.startswith("right_finger") or "hand" in name:
        return "other_hand_links"
    return "untracked"


def _object_info(base: Any) -> list[dict[str, Any]]:
    out = []
    registry = getattr(base, "v83_active_asset_registry", {}) or {}
    for idx, name in enumerate(PART_NAMES):
        entry = dict(registry.get(name, {}) or {})
        asset = entry.get("asset")
        out.append(
            {
                "object_name": name,
                "env_slot": idx,
                "asset_class": type(asset).__name__ if asset is not None else "",
                "prim_path": str(getattr(getattr(asset, "cfg", None), "prim_path", entry.get("prim_path", ""))),
                "filter_expr": f"/World/envs/env_.*/{name}",
            }
        )
    return out


def _transcode_to_h264(source: Path, target: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            if source != target:
                shutil.copy2(source, target)
            return {"ok": bool(target.is_file()), "error": f"ffmpeg_unavailable:{type(exc).__name__}:{exc}"}
    tmp = target.with_name(f"{target.stem}_h264_tmp{target.suffix}")
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return {"ok": False, "error": (proc.stderr or proc.stdout or "ffmpeg_failed").strip().splitlines()[-1]}
    tmp.replace(target)
    return {"ok": True, "error": ""}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _nearest_env_to_point(origins: list[list[float]], point: list[float]) -> int:
    best_idx = -1
    best_dist = float("inf")
    for idx, origin in enumerate(origins):
        dist = math.sqrt((float(point[0]) - float(origin[0])) ** 2 + (float(point[1]) - float(origin[1])) ** 2)
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def _wrist_world_pos(base: Any, env_index: int, fallback: list[float]) -> list[float]:
    robot = getattr(base, "_robot", None)
    names = list(getattr(robot, "body_names", []) or [])
    candidates = [idx for idx, name in enumerate(names) if "wrist" in name or name in {"panda_hand", "panda_link7"}]
    if not candidates:
        return fallback
    return _tensor_vec_any(getattr(getattr(robot, "data", None), "body_pos_w", None), (env_index, candidates[0]))


def _world_from_local(base: Any, env_index: int, local: list[float]) -> list[float]:
    origin = _tensor_vec_any(getattr(getattr(base, "scene", None), "env_origins", None), env_index)
    return _add_vec(origin, local)


def _add_vec(a: list[float], b: list[float]) -> list[float]:
    return [float(_list_get(a, i, 0.0)) + float(_list_get(b, i, 0.0)) for i in range(3)]


def _rows_from_action(action: Any) -> list[list[float]]:
    if torch is not None and hasattr(action, "detach"):
        return action.detach().to("cpu").float().tolist()
    return [[float(value) for value in row] for row in list(action)]


def _tensor_vec_any(tensor: Any, index: Any, width: int = 3) -> list[float]:
    try:
        value = tensor[index]
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").flatten().tolist()
        else:
            value = list(value)
        return [float(x) for x in value[:width]]
    except Exception:
        return [0.0] * width


def _tensor_matrix_any(tensor: Any, index: Any | None = None) -> list[list[float]]:
    try:
        value = tensor if index is None else tensor[index]
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").tolist()
        return [[float(x) for x in row[:3]] if isinstance(row, (list, tuple)) else [float(row)] for row in list(value)]
    except Exception:
        return []


def _tensor_list_any(tensor: Any, index: int) -> list[float]:
    try:
        value = tensor[index]
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").flatten().tolist()
        return [float(x) for x in list(value)]
    except Exception:
        return []


def _safe_index(values: list[str], name: str) -> int:
    try:
        return int(values.index(name))
    except ValueError:
        return -1


def _list_get(values: Any, index: int, default: Any = 0.0) -> Any:
    try:
        return values[index]
    except Exception:
        return default


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if torch is not None and hasattr(value, "detach"):
        return _plain(value.detach().to("cpu").tolist())
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value
