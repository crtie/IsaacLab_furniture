"""Deterministic Screw1 contact/lift baseline.

This module is intentionally independent from the v95 workcell gate.  It uses
the existing reset-time parking hook, runtime PhysX tensors, and the
UnifiedActionMapper, but it does not write progress matrices or claim training
readiness.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .unified_action_mapper import UnifiedActionMapper
from .video_log_alignment import AlignmentRecorder, make_run_id

try:  # pragma: no cover - Isaac runtime dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


V83_PARTS = ("Plug2", "Screw1", "Backrest", "Rod", "Frame")
FINGER3_INDEX = 2
FINGER4_INDEX = 3
TARGET_FORCE_THRESHOLD_N = 0.05
RESULT_CLASSES = {
    "COARSE_PREGRASP_NOT_REACHED",
    "CONTACT_SEARCH_NO_TARGET_CONTACT",
    "CONTACT_SEARCH_STOPPED_BY_NON_TARGET_CONTACT",
    "CONTACT_SEARCH_STOPPED_BY_CLAMP",
    "CONTACT_SEARCH_CONTROLLER_NO_MOTION",
    "PREGRASP_NOT_NEAR_TARGET",
    "APPROACH_NOT_REDUCING_DISTANCE",
    "APPROACH_REDUCED_BUT_CLAMPED_NO_CONTACT",
    "APPROACH_STOPPED_BY_WORKSPACE_OR_TABLE_CLAMP",
    "NEAR_SURFACE_NO_FORCE",
    "SURFACE_ALIGNED_NO_FORCE",
    "PRECONTACT_REACHED_PRESS_NOT_ENTERED",
    "PRESS_ENTERED_NO_SURFACE_PROGRESS",
    "PRESS_ENTERED_NO_FORCE",
    "PRESS_BLOCKED_BY_Z_CLAMP",
    "PRESS_ACTION_NOT_REACHING_CONTROLLER",
    "PRESS_ACTION_FRAME_WRONG",
    "PRESS_CONTROLLER_TARGET_MOVED_TIP_STATIC",
    "PRESS_PROGRESS_NO_FORCE",
    "IK_PRESS_NO_SURFACE_PROGRESS",
    "SURFACE_DISTANCE_CLOSE_NO_FORCE",
    "TARGET_CONTACT_ACQUIRED",
    "TARGET_CONTACT_AND_CLOSE",
    "TARGET_CONTACT_CLOSE_AND_LIFT_ATTEMPTED",
    "STABLE_TWO_FINGER_CONTACT_ACQUIRED",
    "TWO_FINGER_CONTACT_OBJECT_MOVED",
    "SINGLE_FINGER_CONTACT_ONLY",
    "TWO_SEED_REFINEMENT_FORCE_LIMIT",
    "TWO_SEED_REFINEMENT_NO_TARGET_CONTACT",
    "TWO_SEED_REFINEMENT_STOPPED_BY_NON_TARGET_CONTACT",
    "RUNTIME_INTERFACE_MISSING",
}


@dataclass
class ScriptedBaselineConfig:
    part: str = "Screw1"
    output_dir: str = "debug_runs/scripted_baseline_screw1"
    max_approach_steps: int = 150
    max_close_steps: int = 50
    max_lift_steps: int = 60
    sanity_steps: int = 3
    approach_action_gain: float = 0.60
    approach_action_gain_after_stall: float = 0.90
    micro_press_action_gain: float = 0.10
    sanity_action_gain: float = 0.15
    lift_action_gain: float = 0.25
    pregrasp_gap_m: float = 0.035
    surface_pregrasp_gap_m: float = 0.015
    surface_precontact_gap_m: float = 0.015
    surface_press_depth_m: float = 0.003
    surface_table_clearance_m: float = 0.015
    precontact_reached_distance_m: float = 0.005
    real_surface_micro_close_distance_m: float = 0.005
    max_micro_press_steps: int = 20
    micro_press_step_m: float = 0.00075
    press_action_servo_probe_steps: int = 2
    press_ik_fallback_step_m: float = 0.002
    press_ik_fallback_max_steps: int = 16
    press_ik_fallback_max_total_m: float = 0.025
    press_ik_fallback_max_time_s: float = 0.20
    press_ik_fallback_pos_tol_m: float = 0.002
    press_min_tip_progress_before_ik_m: float = 0.001
    press_surface_reduction_before_ik_m: float = 0.0015
    press_efficiency_before_ik: float = 0.40
    press_ctrl_delta_before_ik_m: float = 0.002
    press_close_surface_distance_m: float = 0.003
    press_target_surface_distance_m: float = 0.001
    press_ik_min_surface_reduction_m: float = 0.002
    press_z_clamp_block_threshold_m: float = 0.005
    press_z_raise_m: float = 0.008
    contact_truth_extra_penetration_m: float = 0.001
    contact_truth_step_m: float = 0.0005
    contact_search_step_m: float = 0.0015
    contact_search_entry_radius_m: float = 0.030
    contact_search_radial_limit_m: float = 0.026
    contact_search_lateral_offsets_m: tuple[float, ...] = (0.0, 0.004, -0.004)
    contact_search_z_offsets_m: tuple[float, ...] = (0.0, -0.004, 0.004, -0.008, 0.008)
    contact_search_force_threshold_n: float = TARGET_FORCE_THRESHOLD_N
    contact_search_contact_only: bool = True
    contact_search_max_steps: int = 900
    contact_search_no_motion_steps: int = 5
    diagnostic_sweep_position_offsets_m: tuple[float, ...] = (-0.020, -0.010, 0.0, 0.010, 0.020)
    diagnostic_sweep_roll_offsets_deg: tuple[float, ...] = (0.0, 20.0, -20.0)
    diagnostic_sweep_pitch_offsets_deg: tuple[float, ...] = (0.0, 20.0, -20.0)
    diagnostic_sweep_yaw_offsets_deg: tuple[float, ...] = (0.0, 30.0, -30.0)
    diagnostic_sweep_ik_max_time_s: float = 0.08
    diagnostic_sweep_pos_tol_m: float = 0.003
    contact_refinement_seed_position_offset_m: tuple[float, float, float] = (0.020, 0.010, -0.010)
    contact_refinement_seed_orientation_rpy_deg: tuple[float, float, float] = (0.0, 20.0, 0.0)
    contact_refinement_seed_a_position_offset_m: tuple[float, float, float] = (0.020, 0.010, -0.010)
    contact_refinement_seed_a_orientation_rpy_deg: tuple[float, float, float] = (0.0, 20.0, 0.0)
    contact_refinement_seed_b_position_offset_m: tuple[float, float, float] = (0.015, 0.015, -0.010)
    contact_refinement_seed_b_orientation_rpy_deg: tuple[float, float, float] = (10.0, 20.0, 0.0)
    contact_refinement_position_offsets_m: tuple[float, ...] = (-0.005, 0.0, 0.005)
    contact_refinement_orientation_offsets_deg: tuple[float, ...] = (0.0, -10.0, 10.0)
    contact_refinement_fine_position_offsets_m: tuple[float, ...] = (-0.0025, 0.0, 0.0025)
    contact_refinement_fine_orientation_offsets_deg: tuple[float, ...] = (0.0, -5.0, 5.0)
    contact_refinement_force_threshold_n: float = TARGET_FORCE_THRESHOLD_N
    contact_refinement_reasonable_force_max_n: float = 20.0
    contact_refinement_reasonable_total_force_min_n: float = 0.10
    contact_refinement_reasonable_total_force_max_n: float = 20.0
    contact_refinement_hard_force_stop_n: float = 20.0
    contact_refinement_stable_object_motion_limit_m: float = 0.010
    contact_refinement_ik_max_time_s: float = 0.08
    contact_refinement_pos_tol_m: float = 0.003
    reanchor_z_clamp_threshold_m: float = 0.005
    reanchor_target_z_raise_m: float = 0.008
    contact_gap_m: float = 0.002
    tip_z_clearance_m: float = 0.050
    near_target_distance_m: float = 0.004
    screw1_surface_radius_m: float = 0.024
    screw1_surface_z_offset_m: float = 0.014
    table_safe_clearance_m: float = 0.025
    clamp_adjust_z_raise_m: float = 0.010
    clamp_adjust_min_table_clearance_m: float = 0.035
    pregrasp_fail_distance_m: float = 0.050
    pregrasp_goal_distance_m: float = 0.030
    near_surface_close_distance_m: float = 0.015
    post_stop_hold_frames: int = 45
    open_hand_steps: int = 3
    open_hand_command: float = -0.60
    approach_stall_window_steps: int = 25
    approach_gain_reduction_threshold_m: float = 0.003
    approach_fail_reduction_threshold_m: float = 0.005
    force_threshold_n: float = TARGET_FORCE_THRESHOLD_N
    force_close_without_contact_for_debug: bool = False
    alignment_debug: bool = False
    runtime_collision_debug: bool = False
    runtime_collision_sdf_num_points: int = 7
    run_id: str = ""


def run_scripted_contact_baseline(
    env: Any,
    cfg: ScriptedBaselineConfig,
    *,
    video_recorder: Any | None = None,
) -> dict[str, Any]:
    """Run a deterministic Screw1 baseline and write compact artifacts."""

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    mapper = UnifiedActionMapper()
    alignment: AlignmentRecorder | None = None
    summary: dict[str, Any] = {
        "target_part": cfg.part,
        "target_env_index": -1,
        "training_unlocked": False,
        "object_ready": False,
        "success_claimed": False,
        "canary_or_forced_pose_success_used": False,
        "distance_only_success_used": False,
        "unfiltered_only_success_used": False,
        "sticky_or_proxy_success_used": False,
        "result_class": "RUNTIME_INTERFACE_MISSING",
        "failure_reason": "",
    }

    try:
        if torch is None:
            raise RuntimeError("torch_unavailable")
        base = _base_env(env)
        target_env_index = _target_env_index(base, cfg.part)
        summary["target_env_index"] = target_env_index
        _assert_runtime_interfaces(base, cfg.part, target_env_index)
        alignment = AlignmentRecorder(
            enabled=bool(cfg.alignment_debug),
            output_dir=output_dir,
            run_id=str(cfg.run_id or make_run_id()),
            target_part=cfg.part,
            target_env_index=target_env_index,
            video_recorder=video_recorder,
        )

        env.reset()
        _refresh_runtime(base)
        first_state = read_state(base, target_env_index, cfg.part, cfg.force_threshold_n)
        reset_plan = _build_reset_pregrasp_plan(base, target_env_index, cfg, first_state)
        if hasattr(base, "v95_configure_pregrasp"):
            base.v95_configure_pregrasp(reset_plan)
        env.reset()
        _refresh_runtime(base)
        if hasattr(base, "v95_configure_action_control"):
            env_ids = torch.tensor([target_env_index], dtype=torch.long, device=base.device)
            base.v95_configure_action_control(mode="", allow_commanded_hand_actions=True, env_ids=env_ids)

        reset_state = _state_with_contact_target(
            base,
            target_env_index,
            cfg,
            read_state(base, target_env_index, cfg.part, cfg.force_threshold_n),
        )
        alignment.configure_camera(env, base, reset_state)
        alignment.setup_body_mapping_and_sensors(base)
        object_reset_pos = list(reset_state["object_local_pos"])
        alignment.update_markers(base, reset_state)
        reset_frame = _capture_video_frame(video_recorder, alignment)
        summary.update(
            {
                "run_id": alignment.run_id,
                "target_filter_names": reset_state.get("target_filter_names", []),
                "target_filter_index": reset_state.get("target_filter_index", -1),
                "finger3_source_body_name": reset_state.get("finger3_source_body_name", ""),
                "finger4_source_body_name": reset_state.get("finger4_source_body_name", ""),
                "dex_fingertip_true_source": reset_state.get("dex_fingertip_true_source", ""),
                "action_dim": _action_dim(env, base),
                "policy_action_dim": 16,
                "reset_plan_used": "v95_configure_pregrasp_reset_time_only",
                "object_write_after_reset_used": False,
            }
        )
        _append_log(
            logs,
            "reset",
            0,
            reset_state,
            cfg,
            object_reset_pos,
            extra={"phase_event": "after_reset", "video_frame_index": reset_frame},
            alignment=alignment,
            base=base,
        )

        pregrasp_state = _analytic_pregrasp(env, base, target_env_index, cfg, reset_state)
        alignment.update_markers(base, pregrasp_state, contact_target_local=pregrasp_state.get("contact_target_local_pos"))
        pregrasp_frame = _capture_video_frame(video_recorder, alignment)
        _append_log(
            logs,
            "pregrasp",
            0,
            pregrasp_state,
            cfg,
            object_reset_pos,
            extra={"phase_event": "after_analytic_pregrasp", "video_frame_index": pregrasp_frame},
            alignment=alignment,
            base=base,
        )
        reanchor = _reanchor_after_pregrasp(
            env,
            base,
            mapper,
            target_env_index,
            cfg,
            logs,
            object_reset_pos,
            alignment,
        )
        contact_adjustment = dict(reanchor.get("contact_adjustment", {}) or {})
        open_hand = _run_open_hand(
            env,
            base,
            mapper,
            target_env_index,
            cfg,
            logs,
            object_reset_pos,
            alignment,
            adjustment=contact_adjustment,
        )
        _configure_action_control(base, target_env_index, mode="", allow_commanded_hand_actions=True)
        post_open_state = _state_with_contact_target(
            base,
            target_env_index,
            cfg,
            read_state(base, target_env_index, cfg.part, cfg.force_threshold_n),
            adjustment=contact_adjustment,
        )
        search_entry = _contact_search_entry_target(base, target_env_index, cfg, post_open_state)
        post_open_state.update(
            {
                "contact_search_entry_local_xyz": search_entry["search_target_local_xyz"],
                "contact_search_side_dir_xy": search_entry["side_dir_xy"],
                "contact_search_base_z_m": search_entry["base_z_m"],
            }
        )

        pregrasp_check = _coarse_pregrasp_root_check(post_open_state, cfg)
        approach: dict[str, Any] = {
            "executed": False,
            "stop_reason": "not_executed_coarse_pregrasp_blocked",
            "steps": 0,
            "distance_reduced_during_approach": False,
            "min_finger3_distance_to_contact_target": pregrasp_check["pregrasp_finger3_distance"],
            "min_finger4_distance_to_contact_target": pregrasp_check["pregrasp_finger4_distance"],
            "near_surface_reached": False,
            "target_contact_observed": False,
        }
        close: dict[str, Any] = {
            "executed": False,
            "close_trigger": "not_executed",
            "limiter_applied": False,
            "limiter_bypassed_for_v95": False,
            "debug_forced_close_without_contact": False,
        }
        lift: dict[str, Any] = {"executed": False, "steps": 0, "skip_reason": "pregrasp_or_approach_blocked"}
        contact_truth: dict[str, Any] = {
            "executed": False,
            "contact_truth_class": "not_executed",
            "contact_truth_reason": "contact_driven_search_skips_controlled_penetration",
        }
        runtime_collision: dict[str, Any] = {
            "runtime_collision_debug": bool(cfg.runtime_collision_debug),
            "runtime_collision_executed": False,
            "runtime_collision_query_available": False,
            "runtime_collision_query_source": "",
            "runtime_collision_conclusion": "not_executed",
        }
        if pregrasp_check["pregrasp_near_enough"]:
            approach = _run_contact_search_acquisition(
                env,
                base,
                mapper,
                target_env_index,
                cfg,
                logs,
                object_reset_pos,
                alignment,
                adjustment=contact_adjustment,
            )
            if bool(cfg.runtime_collision_debug):
                runtime_state = _state_with_contact_target(
                    base,
                    target_env_index,
                    cfg,
                    read_state(base, target_env_index, cfg.part, cfg.force_threshold_n),
                    adjustment=contact_adjustment,
                )
                runtime_state.update(_finger3_tip_collision_geometry(base, target_env_index, runtime_state))
                runtime_state.update(_screw1_collision_truth_metadata(base, target_env_index, runtime_state))
                runtime_collision = _run_runtime_collision_debug(
                    base,
                    target_env_index,
                    cfg,
                    runtime_state,
                    output_dir,
                )
            if _max_target_force_from_logs(logs) > cfg.force_threshold_n:
                approach["target_contact_observed"] = True
            close_decision = (
                {"execute": False, "close_trigger": "not_executed", "skip_reason": "CONTACT_SEARCH_CONTACT_ONLY"}
                if bool(cfg.contact_search_contact_only)
                else _close_decision(approach, cfg)
            )
            if close_decision["execute"]:
                close = _run_close(
                    env,
                    base,
                    mapper,
                    target_env_index,
                    cfg,
                    logs,
                    object_reset_pos,
                    close_trigger=str(close_decision["close_trigger"]),
                    alignment=alignment,
                )
                post_close_force = _max_target_force_from_logs(logs)
                if post_close_force > cfg.force_threshold_n:
                    lift = _run_lift(env, base, mapper, target_env_index, cfg, logs, object_reset_pos, alignment)
                else:
                    lift = {"executed": False, "steps": 0, "skip_reason": "NO_CONTACT_NO_LIFT"}
            else:
                close["skip_reason"] = str(close_decision.get("skip_reason", ""))
                lift = {"executed": False, "steps": 0, "skip_reason": "NO_CONTACT_NO_LIFT"}

        final_state = _state_with_contact_target(
            base,
            target_env_index,
            cfg,
            read_state(base, target_env_index, cfg.part, cfg.force_threshold_n),
            adjustment=contact_adjustment,
        )
        final_state.update(_finger3_tip_collision_geometry(base, target_env_index, final_state))
        final_state.update(_screw1_collision_truth_metadata(base, target_env_index, final_state))
        final_state.update(_finger3_body_chain_truth_metadata(base, target_env_index, final_state))
        alignment.update_markers(base, final_state, contact_target_local=final_state.get("contact_target_local_pos"))
        final_frame_index = _capture_video_frame(video_recorder, alignment)
        _append_log(
            logs,
            "final",
            0,
            final_state,
            cfg,
            object_reset_pos,
            extra={"phase_event": "final", "video_frame_index": final_frame_index},
            alignment=alignment,
            base=base,
        )
        post_stop_hold_frames = _hold_video_frames(video_recorder, alignment, int(cfg.post_stop_hold_frames))

        max_f3_target = max([0.0, *[float(row.get("finger3_target_filtered_force_n", 0.0)) for row in logs]])
        max_f4_target = max([0.0, *[float(row.get("finger4_target_filtered_force_n", 0.0)) for row in logs]])
        max_f3_unfiltered = max([0.0, *[float(row.get("finger3_unfiltered_force_n", 0.0)) for row in logs]])
        max_f4_unfiltered = max([0.0, *[float(row.get("finger4_unfiltered_force_n", 0.0)) for row in logs]])
        close_target_delta_peak = max([0.0, *[abs(float(row.get("hand_target_delta_l2", 0.0))) for row in logs]])
        object_z_delta = _vec_delta_z(object_reset_pos, final_state["object_local_pos"])
        object_displacement = _distance(object_reset_pos, final_state["object_local_pos"])
        clamp_summary = _summarize_clamp_from_logs(logs)
        surface_summary = _summarize_surface_from_logs(logs, final_state)
        clamp_observed_anywhere = bool(
            approach.get("workspace_or_table_clamp_observed")
            or float(clamp_summary.get("workspace_clamp_max_m", 0.0) or 0.0) > 1.0e-4
            or float(clamp_summary.get("table_clamp_max_m", 0.0) or 0.0) > 1.0e-4
        )
        result_class, failure_reason = _classify_result(
            pregrasp=pregrasp_check,
            approach=approach,
            close=close,
            lift=lift,
            max_target=max(max_f3_target, max_f4_target),
            threshold=cfg.force_threshold_n,
            cfg=cfg,
        )
        summary.update(
            {
                "result_class": result_class,
                "failure_reason": failure_reason,
                "open_hand_executed": bool(open_hand.get("executed")),
                "open_hand_steps": int(open_hand.get("steps", 0)),
                "open_hand_stop_reason": str(open_hand.get("stop_reason", "")),
                "control_reanchored_after_pregrasp": bool(reanchor.get("control_reanchored_after_pregrasp")),
                "z_clamp_before_reanchor": float(reanchor.get("z_clamp_before_reanchor", 0.0) or 0.0),
                "z_clamp_after_reanchor": float(reanchor.get("z_clamp_after_reanchor", 0.0) or 0.0),
                "press_target_z_raised_after_reanchor": bool(reanchor.get("press_target_z_raised_after_reanchor")),
                "pregrasp_finger3_distance": pregrasp_check["pregrasp_finger3_distance"],
                "pregrasp_finger4_distance": pregrasp_check["pregrasp_finger4_distance"],
                "pregrasp_near_enough": bool(pregrasp_check["pregrasp_near_enough"]),
                "pregrasp_goal_met": bool(pregrasp_check["pregrasp_goal_met"]),
                "min_finger3_distance_to_contact_target": approach.get(
                    "min_finger3_distance_to_contact_target", pregrasp_check["pregrasp_finger3_distance"]
                ),
                "min_finger4_distance_to_contact_target": approach.get(
                    "min_finger4_distance_to_contact_target", pregrasp_check["pregrasp_finger4_distance"]
                ),
                "min_finger3_distance_to_precontact": approach.get(
                    "min_finger3_distance_to_precontact", pregrasp_check["pregrasp_finger3_distance"]
                ),
                "min_finger3_distance_to_real_surface": approach.get("min_finger3_distance_to_real_surface", ""),
                "min_finger4_distance_to_real_surface": approach.get("min_finger4_distance_to_real_surface", ""),
                "finger3_tip_at_min_surface_distance": approach.get("finger3_tip_at_min_surface_distance", []),
                "finger4_tip_at_min_surface_distance": approach.get("finger4_tip_at_min_surface_distance", []),
                "press_depth_attempted": approach.get("press_depth_attempted", 0.0),
                "press_stage_entered": bool(approach.get("press_stage_entered")),
                "press_start_reason": str(approach.get("press_start_reason", "")),
                "press_depth_commanded": float(approach.get("press_depth_commanded", 0.0) or 0.0),
                "press_depth_actual": float(approach.get("press_depth_actual", 0.0) or 0.0),
                "press_direction_flipped": bool(approach.get("press_direction_flipped")),
                "press_surface_progress": bool(approach.get("press_surface_progress")),
                "press_control_mode_used": str(approach.get("press_control_mode_used", "not_entered")),
                "press_policy_action_norm_peak": float(approach.get("press_policy_action_norm_peak", 0.0) or 0.0),
                "press_mapped_isaac_action_norm_peak": float(
                    approach.get("press_mapped_isaac_action_norm_peak", 0.0) or 0.0
                ),
                "press_ctrl_target_delta_peak": float(approach.get("press_ctrl_target_delta_peak", 0.0) or 0.0),
                "press_ctrl_target_moved": bool(approach.get("press_ctrl_target_moved")),
                "press_finger3_delta_dot_press_dir": float(
                    approach.get("press_finger3_delta_dot_press_dir", 0.0) or 0.0
                ),
                "finger3_delta_dot_press_dir": float(
                    approach.get("press_finger3_delta_dot_press_dir", 0.0) or 0.0
                ),
                "press_finger3_delta_dot_press_dir_peak": float(
                    approach.get("press_finger3_delta_dot_press_dir_peak", 0.0) or 0.0
                ),
                "z_clamp_before_press": float(approach.get("press_z_clamp_before_press", 0.0) or 0.0),
                "z_clamp_after_press_reanchor": float(approach.get("press_z_clamp_after_reanchor", 0.0) or 0.0),
                "z_clamp_after_z_raise": float(approach.get("press_z_clamp_after_z_raise", 0.0) or 0.0),
                "workspace_clamp_max_during_press": float(
                    approach.get("workspace_clamp_max_during_press", 0.0) or 0.0
                ),
                "press_blocked_by_z_clamp": bool(approach.get("press_blocked_by_z_clamp")),
                "press_action_not_reaching_controller": bool(approach.get("press_action_not_reaching_controller")),
                "press_action_frame_wrong": bool(approach.get("press_action_frame_wrong")),
                "press_controller_target_moved_tip_static": bool(
                    approach.get("press_controller_target_moved_tip_static")
                ),
                "press_target_z_raised_after_preflight": bool(approach.get("press_target_z_raised_after_preflight")),
                "press_target_minus_tip_dot_press_dir": float(
                    approach.get("press_target_minus_tip_dot_press_dir", 0.0) or 0.0
                ),
                "surface_point_minus_tip_dot_press_dir": float(
                    approach.get("surface_point_minus_tip_dot_press_dir", 0.0) or 0.0
                ),
                "fallback_triggered": bool(approach.get("fallback_triggered")),
                "fallback_reason": str(approach.get("fallback_reason", "")),
                "surface_distance_reduction_during_press": float(
                    approach.get("surface_distance_reduction_during_press", 0.0) or 0.0
                ),
                "action_probe_surface_reduction": float(approach.get("action_probe_surface_reduction", 0.0) or 0.0),
                "ik_surface_reduction": float(approach.get("ik_surface_reduction", 0.0) or 0.0),
                "press_start_surface_distance": float(approach.get("press_start_surface_distance", 0.0) or 0.0),
                "press_min_surface_distance": float(approach.get("press_min_surface_distance", 0.0) or 0.0),
                "press_final_surface_distance": float(approach.get("press_final_surface_distance", 0.0) or 0.0),
                "surface_distance_start_m": float(approach.get("press_start_surface_distance", 0.0) or 0.0),
                "surface_distance_final_m": float(approach.get("press_final_surface_distance", 0.0) or 0.0),
                "total_press_limit_m": float(approach.get("total_press_limit_m", cfg.press_ik_fallback_max_total_m) or 0.0),
                "per_step_press_m": float(approach.get("per_step_press_m", cfg.press_ik_fallback_step_m) or 0.0),
                "target_surface_distance_m": float(
                    approach.get("press_target_surface_distance", cfg.press_target_surface_distance_m) or 0.0
                ),
                "press_dir_surface_dir_dot": approach.get("press_dir_surface_dir_dot_last", ""),
                "press_dir_surface_dir_dot_last": approach.get("press_dir_surface_dir_dot_last", ""),
                "press_dir_surface_dir_dot_min": approach.get("press_dir_surface_dir_dot_min", ""),
                "press_dir_surface_dir_dot_mean": approach.get("press_dir_surface_dir_dot_mean", ""),
                "distance_reduced_during_approach": bool(approach.get("distance_reduced_during_approach")),
                "approach_executed": bool(approach.get("executed")),
                "approach_stop_reason": approach.get("stop_reason", ""),
                "approach_clamp_adjusted": bool(approach.get("clamp_adjusted")),
                "approach_clamp_adjustment_type": str(approach.get("clamp_adjustment_type", "")),
                "approach_clamp_adjustment_step": int(approach.get("clamp_adjustment_step", -1)),
                "geometry_distance_used_for_control": False,
                "geometry_distance_used_for_success": False,
                "contact_search_executed": bool(approach.get("contact_search_executed", False)),
                "contact_search_success": bool(approach.get("contact_search_success", False)),
                "contact_search_target_contact_observed": bool(
                    approach.get("contact_search_target_contact_observed", False)
                ),
                "contact_search_mode": str(approach.get("contact_search_mode", "")),
                "contact_search_reanchored_before_search": bool(
                    approach.get("contact_search_reanchored_before_search", False)
                ),
                "contact_search_zero_action_ctrl_delta_l2": float(
                    approach.get("contact_search_zero_action_ctrl_delta_l2", 0.0) or 0.0
                ),
                "contact_search_steps": int(approach.get("contact_search_steps", 0) or 0),
                "contact_search_attempted_pose_count": int(
                    approach.get("contact_search_attempted_pose_count", approach.get("contact_search_steps", 0)) or 0
                ),
                "contact_search_completed_planned_sweep": bool(
                    approach.get("contact_search_completed_planned_sweep", False)
                ),
                "contact_search_termination_reason": str(
                    approach.get("contact_search_termination_reason", approach.get("stop_reason", ""))
                ),
                "contact_search_total_commanded_m": float(
                    approach.get("contact_search_total_commanded_m", 0.0) or 0.0
                ),
                "contact_search_max_step_m": float(approach.get("contact_search_max_step_m", 0.0) or 0.0),
                "contact_search_planned_pose_count": int(approach.get("contact_search_planned_pose_count", 0) or 0),
                "contact_search_max_steps": int(approach.get("contact_search_max_steps", 0) or 0),
                "contact_search_reached_pose_count": int(approach.get("contact_search_reached_pose_count", 0) or 0),
                "contact_search_clamped_probe_count": int(approach.get("contact_search_clamped_probe_count", 0) or 0),
                "contact_search_controller_no_motion_count": int(
                    approach.get("contact_search_controller_no_motion_count", 0) or 0
                ),
                "contact_search_any_probe_clamped": bool(approach.get("contact_search_any_probe_clamped", False)),
                "contact_search_position_extent_xyz_m": approach.get("contact_search_position_extent_xyz_m", []),
                "contact_search_orientation_extent_rpy_deg": approach.get(
                    "contact_search_orientation_extent_rpy_deg", []
                ),
                "contact_search_position_offsets_m": approach.get("contact_search_position_offsets_m", []),
                "contact_search_orientation_offsets_rpy_deg": approach.get(
                    "contact_search_orientation_offsets_rpy_deg", []
                ),
                "contact_search_center_finger3_local_xyz": approach.get(
                    "contact_search_center_finger3_local_xyz", []
                ),
                "contact_search_center_finger4_local_xyz": approach.get(
                    "contact_search_center_finger4_local_xyz", []
                ),
                "contact_search_center_palm_local_xyz": approach.get("contact_search_center_palm_local_xyz", []),
                "contact_search_center_screw1_root_local_xyz": approach.get(
                    "contact_search_center_screw1_root_local_xyz", []
                ),
                "contact_search_raw_contact_found": bool(approach.get("contact_search_raw_contact_found", False)),
                "contact_search_contact_probe_id": approach.get("contact_search_contact_probe_id", ""),
                "contact_search_contact_position_offset_xyz_m": approach.get(
                    "contact_search_contact_position_offset_xyz_m", []
                ),
                "contact_search_contact_orientation_offset_rpy_deg": approach.get(
                    "contact_search_contact_orientation_offset_rpy_deg", []
                ),
                "contact_search_contact_orientation_offset_norm_deg": float(
                    approach.get("contact_search_contact_orientation_offset_norm_deg", 0.0) or 0.0
                ),
                "diagnostic_acquisition_judgement": _diagnostic_sweep_judgement(
                    approach, cfg.force_threshold_n
                ),
                "contact_search_non_target_blocked": bool(approach.get("contact_search_non_target_blocked", False)),
                "contact_refinement_executed": bool(approach.get("contact_refinement_executed", False)),
                "contact_refinement_probe_count": int(approach.get("contact_refinement_probe_count", 0) or 0),
                "contact_refinement_coarse_probe_count": int(
                    approach.get("contact_refinement_coarse_probe_count", 0) or 0
                ),
                "contact_refinement_fine_pass_executed": bool(
                    approach.get("contact_refinement_fine_pass_executed", False)
                ),
                "contact_refinement_fine_probe_count": int(
                    approach.get("contact_refinement_fine_probe_count", 0) or 0
                ),
                "contact_refinement_termination_reason": str(
                    approach.get("contact_refinement_termination_reason", "")
                ),
                "contact_refinement_seed_position_offset_xyz_m": approach.get(
                    "contact_refinement_seed_position_offset_xyz_m", []
                ),
                "contact_refinement_seed_orientation_rpy_deg": approach.get(
                    "contact_refinement_seed_orientation_rpy_deg", []
                ),
                "contact_refinement_seed_a_position_offset_xyz_m": approach.get(
                    "contact_refinement_seed_a_position_offset_xyz_m", []
                ),
                "contact_refinement_seed_a_orientation_rpy_deg": approach.get(
                    "contact_refinement_seed_a_orientation_rpy_deg", []
                ),
                "contact_refinement_seed_b_position_offset_xyz_m": approach.get(
                    "contact_refinement_seed_b_position_offset_xyz_m", []
                ),
                "contact_refinement_seed_b_orientation_rpy_deg": approach.get(
                    "contact_refinement_seed_b_orientation_rpy_deg", []
                ),
                "contact_refinement_interpolation_centers": approach.get(
                    "contact_refinement_interpolation_centers", []
                ),
                "contact_refinement_best_probe_id": approach.get("contact_refinement_best_probe_id", ""),
                "contact_refinement_best_pass": str(approach.get("contact_refinement_best_pass", "")),
                "contact_refinement_best_center_id": str(approach.get("contact_refinement_best_center_id", "")),
                "contact_refinement_best_interpolation_factor": float(
                    approach.get("contact_refinement_best_interpolation_factor", 0.0) or 0.0
                ),
                "contact_refinement_best_center_position_offset_xyz_m": approach.get(
                    "contact_refinement_best_center_position_offset_xyz_m", []
                ),
                "contact_refinement_best_center_orientation_rpy_deg": approach.get(
                    "contact_refinement_best_center_orientation_rpy_deg", []
                ),
                "contact_refinement_best_pose_offset_xyz_m": approach.get(
                    "contact_refinement_best_pose_offset_xyz_m", []
                ),
                "contact_refinement_best_local_position_offset_xyz_m": approach.get(
                    "contact_refinement_best_local_position_offset_xyz_m", []
                ),
                "contact_refinement_best_orientation_rpy_deg": approach.get(
                    "contact_refinement_best_orientation_rpy_deg", []
                ),
                "contact_refinement_best_local_orientation_offset_rpy_deg": approach.get(
                    "contact_refinement_best_local_orientation_offset_rpy_deg", []
                ),
                "contact_refinement_best_finger3_force_n": float(
                    approach.get("contact_refinement_best_finger3_force_n", 0.0) or 0.0
                ),
                "contact_refinement_best_finger4_force_n": float(
                    approach.get("contact_refinement_best_finger4_force_n", 0.0) or 0.0
                ),
                "contact_refinement_best_total_target_force_n": float(
                    approach.get("contact_refinement_best_total_target_force_n", 0.0) or 0.0
                ),
                "contact_refinement_best_max_target_force_n": float(
                    approach.get("contact_refinement_best_max_target_force_n", 0.0) or 0.0
                ),
                "contact_refinement_best_finger3_contact": bool(
                    approach.get("contact_refinement_best_finger3_contact", False)
                ),
                "contact_refinement_best_finger4_contact": bool(
                    approach.get("contact_refinement_best_finger4_contact", False)
                ),
                "contact_refinement_any_target_contact": bool(
                    approach.get("contact_refinement_any_target_contact", False)
                ),
                "two_finger_contact_acquired": bool(approach.get("two_finger_contact_acquired", False)),
                "force_level_reasonable": bool(approach.get("force_level_reasonable", False)),
                "best_contact_object_displacement_m": float(
                    approach.get("best_contact_object_displacement_m", 0.0) or 0.0
                ),
                "best_contact_object_motion_stable": bool(
                    approach.get("best_contact_object_motion_stable", False)
                ),
                "contact_refinement_reasonable_force_min_n": float(
                    approach.get("contact_refinement_reasonable_force_min_n", cfg.contact_refinement_force_threshold_n)
                    or 0.0
                ),
                "contact_refinement_reasonable_force_max_n": float(
                    approach.get(
                        "contact_refinement_reasonable_force_max_n",
                        cfg.contact_refinement_reasonable_force_max_n,
                    )
                    or 0.0
                ),
                "contact_refinement_reasonable_total_force_min_n": float(
                    approach.get(
                        "contact_refinement_reasonable_total_force_min_n",
                        cfg.contact_refinement_reasonable_total_force_min_n,
                    )
                    or 0.0
                ),
                "contact_refinement_reasonable_total_force_max_n": float(
                    approach.get(
                        "contact_refinement_reasonable_total_force_max_n",
                        cfg.contact_refinement_reasonable_total_force_max_n,
                    )
                    or 0.0
                ),
                "contact_refinement_hard_force_stop_n": float(
                    approach.get("contact_refinement_hard_force_stop_n", cfg.contact_refinement_hard_force_stop_n)
                    or 0.0
                ),
                "contact_refinement_stable_object_motion_limit_m": float(
                    approach.get(
                        "contact_refinement_stable_object_motion_limit_m",
                        cfg.contact_refinement_stable_object_motion_limit_m,
                    )
                    or 0.0
                ),
                "two_seed_contact_refinement_executed": bool(
                    approach.get("two_seed_contact_refinement_executed", False)
                ),
                "two_seed_contact_refinement_probe_count": int(
                    approach.get("two_seed_contact_refinement_probe_count", 0) or 0
                ),
                "two_seed_contact_refinement_termination_reason": str(
                    approach.get("two_seed_contact_refinement_termination_reason", "")
                ),
                "two_seed_contact_refinement_seed_a_position_offset_xyz_m": approach.get(
                    "two_seed_contact_refinement_seed_a_position_offset_xyz_m", []
                ),
                "two_seed_contact_refinement_seed_a_orientation_rpy_deg": approach.get(
                    "two_seed_contact_refinement_seed_a_orientation_rpy_deg", []
                ),
                "two_seed_contact_refinement_seed_b_position_offset_xyz_m": approach.get(
                    "two_seed_contact_refinement_seed_b_position_offset_xyz_m", []
                ),
                "two_seed_contact_refinement_seed_b_orientation_rpy_deg": approach.get(
                    "two_seed_contact_refinement_seed_b_orientation_rpy_deg", []
                ),
                "two_seed_contact_refinement_interpolation_centers": approach.get(
                    "two_seed_contact_refinement_interpolation_centers", []
                ),
                "two_seed_contact_refinement_position_offsets_m": approach.get(
                    "two_seed_contact_refinement_position_offsets_m", []
                ),
                "two_seed_contact_refinement_orientation_offsets_rpy_deg": approach.get(
                    "two_seed_contact_refinement_orientation_offsets_rpy_deg", []
                ),
                "two_seed_contact_refinement_search_position_extent_xyz_m": approach.get(
                    "two_seed_contact_refinement_search_position_extent_xyz_m", []
                ),
                "two_seed_contact_refinement_search_orientation_extent_rpy_deg": approach.get(
                    "two_seed_contact_refinement_search_orientation_extent_rpy_deg", []
                ),
                "two_seed_contact_refinement_best_two_finger_probe_id": approach.get(
                    "two_seed_contact_refinement_best_two_finger_probe_id", ""
                ),
                "two_seed_contact_refinement_best_two_finger_center_id": str(
                    approach.get("two_seed_contact_refinement_best_two_finger_center_id", "")
                ),
                "two_seed_contact_refinement_best_two_finger_interpolation_factor": float(
                    approach.get("two_seed_contact_refinement_best_two_finger_interpolation_factor", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_two_finger_pose_offset_xyz_m": approach.get(
                    "two_seed_contact_refinement_best_two_finger_pose_offset_xyz_m", []
                ),
                "two_seed_contact_refinement_best_two_finger_orientation_rpy_deg": approach.get(
                    "two_seed_contact_refinement_best_two_finger_orientation_rpy_deg", []
                ),
                "two_seed_contact_refinement_best_two_finger_finger3_force_n": float(
                    approach.get("two_seed_contact_refinement_best_two_finger_finger3_force_n", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_two_finger_finger4_force_n": float(
                    approach.get("two_seed_contact_refinement_best_two_finger_finger4_force_n", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_two_finger_total_target_force_n": float(
                    approach.get("two_seed_contact_refinement_best_two_finger_total_target_force_n", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_two_finger_object_displacement_m": float(
                    approach.get("two_seed_contact_refinement_best_two_finger_object_displacement_m", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_two_finger_stable": bool(
                    approach.get("two_seed_contact_refinement_best_two_finger_stable", False)
                ),
                "two_seed_contact_refinement_best_overall_probe_id": approach.get(
                    "two_seed_contact_refinement_best_overall_probe_id", ""
                ),
                "two_seed_contact_refinement_best_overall_center_id": str(
                    approach.get("two_seed_contact_refinement_best_overall_center_id", "")
                ),
                "two_seed_contact_refinement_best_overall_interpolation_factor": float(
                    approach.get("two_seed_contact_refinement_best_overall_interpolation_factor", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_overall_pose_offset_xyz_m": approach.get(
                    "two_seed_contact_refinement_best_overall_pose_offset_xyz_m", []
                ),
                "two_seed_contact_refinement_best_overall_orientation_rpy_deg": approach.get(
                    "two_seed_contact_refinement_best_overall_orientation_rpy_deg", []
                ),
                "two_seed_contact_refinement_best_overall_finger3_force_n": float(
                    approach.get("two_seed_contact_refinement_best_overall_finger3_force_n", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_overall_finger4_force_n": float(
                    approach.get("two_seed_contact_refinement_best_overall_finger4_force_n", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_overall_total_target_force_n": float(
                    approach.get("two_seed_contact_refinement_best_overall_total_target_force_n", 0.0) or 0.0
                ),
                "two_seed_contact_refinement_best_overall_object_displacement_m": float(
                    approach.get("two_seed_contact_refinement_best_overall_object_displacement_m", 0.0) or 0.0
                ),
                "stable_two_finger_contact_acquired": bool(
                    approach.get("stable_two_finger_contact_acquired", False)
                ),
                "raw_tip_screw1_peak_n": float(approach.get("raw_tip_screw1_peak_n", 0.0) or 0.0),
                "raw_any_screw1_peak_n": float(approach.get("raw_any_screw1_peak_n", 0.0) or 0.0),
                "raw_non_target_peak_n": float(approach.get("raw_non_target_peak_n", 0.0) or 0.0),
                "raw_unknown_object_peak_n": float(approach.get("raw_unknown_object_peak_n", 0.0) or 0.0),
                "unresolved_unfiltered_contact_observed": bool(
                    approach.get("unresolved_unfiltered_contact_observed", False)
                ),
                "unresolved_unfiltered_force_peak_n": float(
                    approach.get("unresolved_unfiltered_force_peak_n", 0.0) or 0.0
                ),
                "raw_screw1_peak_body": str(approach.get("raw_screw1_peak_body", "")),
                "raw_screw1_peak_object": str(approach.get("raw_screw1_peak_object", "")),
                "object_contact_state": str(approach.get("object_contact_state", "")),
                "contact_truth_executed": bool(contact_truth.get("executed")),
                "contact_truth_class": str(contact_truth.get("contact_truth_class", "")),
                "contact_truth_reason": str(contact_truth.get("contact_truth_reason", "")),
                "contact_truth_extra_penetration_commanded_m": float(
                    contact_truth.get("extra_penetration_commanded_m", 0.0) or 0.0
                ),
                "contact_truth_extra_penetration_actual_m": float(
                    contact_truth.get("extra_penetration_actual_m", 0.0) or 0.0
                ),
                "contact_truth_before_finger3_surface_distance_m": contact_truth.get(
                    "before_finger3_surface_distance_m", ""
                ),
                "contact_truth_after_finger3_surface_distance_m": contact_truth.get(
                    "after_finger3_surface_distance_m", ""
                ),
                "contact_truth_before_finger3_target_force_n": contact_truth.get(
                    "before_finger3_target_force_n", ""
                ),
                "contact_truth_after_finger3_target_force_n": contact_truth.get(
                    "after_finger3_target_force_n", ""
                ),
                "contact_truth_raw_tip_screw1_peak_n": contact_truth.get("raw_tip_screw1_peak_n", 0.0),
                "contact_truth_raw_any_screw1_peak_n": contact_truth.get("raw_any_screw1_peak_n", 0.0),
                "contact_truth_raw_non_target_peak_n": contact_truth.get("raw_non_target_peak_n", 0.0),
                "contact_truth_raw_sensor_available": bool(contact_truth.get("raw_sensor_available", False)),
                "contact_truth_excluded_classes": contact_truth.get("excluded_classes", []),
                "runtime_collision_debug": bool(runtime_collision.get("runtime_collision_debug", False)),
                "runtime_collision_executed": bool(runtime_collision.get("runtime_collision_executed", False)),
                "runtime_collision_probe_json": runtime_collision.get("runtime_collision_probe_json", ""),
                "runtime_collision_query_available": bool(
                    runtime_collision.get("runtime_collision_query_available", False)
                ),
                "runtime_collision_query_source": runtime_collision.get("runtime_collision_query_source", ""),
                "runtime_collision_conclusion": runtime_collision.get("runtime_collision_conclusion", ""),
                "runtime_collision_unavailable_reason": runtime_collision.get(
                    "runtime_collision_unavailable_reason", ""
                ),
                "runtime_collision_distance_to_runtime_collision_m": runtime_collision.get(
                    "distance_to_runtime_collision_m", ""
                ),
                "distance_to_runtime_collision_m": runtime_collision.get("distance_to_runtime_collision_m", ""),
                "finger3_collision_surface_to_runtime_collision_gap_m": runtime_collision.get(
                    "finger3_collision_surface_to_runtime_collision_gap_m", ""
                ),
                "screw1_sdf_distance_at_finger3_tip_m": runtime_collision.get(
                    "screw1_sdf_distance_at_finger3_tip_m", ""
                ),
                "screw1_sdf_gradient_at_finger3_tip_xyz": runtime_collision.get(
                    "screw1_sdf_gradient_at_finger3_tip_xyz", []
                ),
                "runtime_collision_sdf_available": bool(runtime_collision.get("sdf_available", False)),
                "runtime_collision_sdf_error": runtime_collision.get("sdf_error", ""),
                "runtime_collision_scene_query_available": bool(runtime_collision.get("scene_query_available", False)),
                "runtime_collision_scene_query_error": runtime_collision.get("scene_query_error", ""),
                "runtime_collision_scene_query_source": runtime_collision.get("scene_query_source", ""),
                "runtime_collision_scene_query_distance_m": runtime_collision.get("scene_query_distance_m", ""),
                "runtime_collision_scene_query_collision_path": runtime_collision.get(
                    "scene_query_collision_path", ""
                ),
                "runtime_collision_aabb_distance_m": runtime_collision.get("aabb_distance_m", ""),
                "runtime_collision_aabb_minus_runtime_distance_m": runtime_collision.get(
                    "aabb_minus_runtime_distance_m", ""
                ),
                "target_contact_observed": bool(max(max_f3_target, max_f4_target) > cfg.force_threshold_n),
                "finger3_target_filtered_force_peak_n": max_f3_target,
                "finger4_target_filtered_force_peak_n": max_f4_target,
                "finger3_screw1_force_peak": max_f3_target,
                "finger4_screw1_force_peak": max_f4_target,
                "finger3_unfiltered_force_peak_n": max_f3_unfiltered,
                "finger4_unfiltered_force_peak_n": max_f4_unfiltered,
                "unfiltered_force_peak_n": max(max_f3_unfiltered, max_f4_unfiltered),
                "non_target_contact_before_target": bool(approach.get("non_target_contact_before_target")),
                "workspace_or_table_clamp_observed": clamp_observed_anywhere,
                "close_executed": bool(close.get("executed")),
                "close_trigger": str(close.get("close_trigger", "not_executed")),
                "close_skip_reason": str(close.get("skip_reason", "")),
                "close_mode": "full_hand_distal_close_via_existing_mapper",
                "close_limiter_applied": bool(close.get("limiter_applied")),
                "close_limiter_bypassed_for_v95": bool(close.get("limiter_bypassed_for_v95")),
                "debug_forced_close_without_contact": bool(close.get("debug_forced_close_without_contact")),
                "close_target_delta_peak_l2": close_target_delta_peak,
                "lift_executed": bool(lift.get("executed")),
                "lift_steps": int(lift.get("steps", 0)),
                "lift_skip_reason": str(lift.get("skip_reason", "")),
                "object_z_delta_m": object_z_delta,
                "object_displacement_m": object_displacement,
                "post_stop_hold_frames_captured": post_stop_hold_frames,
                "next_baseline_fix": _next_baseline_fix(result_class, pregrasp_check, approach),
                **clamp_summary,
                **surface_summary,
                "ordinary_acquisition_or_v95_gate_called": False,
                "rl_ppo_bc_training_used": False,
                "checkpoint_or_dataset_written": False,
            }
        )
    except Exception as exc:
        summary["result_class"] = "RUNTIME_INTERFACE_MISSING"
        summary["failure_reason"] = f"{type(exc).__name__}:{exc}"
        try:
            base = _base_env(env)
            env_index = _target_env_index(base, cfg.part)
            state = read_state(base, env_index, cfg.part, cfg.force_threshold_n)
            _append_log(
                logs,
                "runtime_error",
                0,
                state,
                cfg,
                state.get("object_local_pos", [0.0, 0.0, 0.0]),
                alignment=alignment,
                base=base,
            )
        except Exception:
            pass

    rollout_log_path = output_dir / "rollout_log.csv"
    rollout_summary_path = output_dir / "rollout_summary.json"
    _write_csv(rollout_log_path, logs)
    _save_trajectory_plot(output_dir / "trajectory_plot.png", logs)
    trajectory_path = output_dir / "trajectory_plot.png"
    final_frame_info = _save_final_frame(env, output_dir / "final_frame.png")
    video_info = _finalize_video_artifact(video_recorder, output_dir)
    alignment_info: dict[str, Any] = {}
    baseline_next_fix = str(summary.get("next_baseline_fix", ""))
    if alignment is not None and alignment.enabled:
        alignment_info = alignment.finalize(logs, {**summary, **final_frame_info, **video_info})
    alignment_next_fix = str(alignment_info.get("next_baseline_fix", ""))
    summary.update(
        {
            "rollout_log_csv": str(rollout_log_path),
            "rollout_summary_json": str(rollout_summary_path),
            "trajectory_plot_png": str(trajectory_path),
            "trajectory_plot_path": str(trajectory_path),
            **final_frame_info,
            **video_info,
            **alignment_info,
        }
    )
    if alignment_next_fix:
        summary["alignment_contact_next_baseline_fix"] = alignment_next_fix
    if baseline_next_fix:
        summary["next_baseline_fix"] = baseline_next_fix
    if "final_frame_path" not in summary:
        summary["final_frame_path"] = summary.get("final_frame_png", "")
    rollout_summary_path.write_text(json.dumps(_plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def read_state(base: Any, env_index: int, part: str, force_threshold_n: float) -> dict[str, Any]:
    _refresh_runtime(base)
    registry = getattr(base, "v83_active_asset_registry", {})
    asset = dict(registry.get(part, {})).get("asset")
    if asset is None:
        raise RuntimeError(f"active_asset_missing_for_{part}")
    origin = _tensor_vec(getattr(getattr(base, "scene", None), "env_origins", None), env_index)
    object_w = _tensor_vec(getattr(getattr(asset, "data", None), "root_pos_w", None), env_index)
    object_local = _sub_vec(object_w, origin)
    object_quat = _tensor_vec(getattr(getattr(asset, "data", None), "root_quat_w", None), env_index, width=4)
    palm = _tensor_vec(getattr(base, "fingertip_midpoint_pos", None), env_index)
    tip_rows = _tensor_matrix(getattr(base, "dex_fingertip_pos", None), env_index)
    tip3 = tip_rows[FINGER3_INDEX] if len(tip_rows) > FINGER3_INDEX else [0.0, 0.0, 0.0]
    tip4 = tip_rows[FINGER4_INDEX] if len(tip_rows) > FINGER4_INDEX else [0.0, 0.0, 0.0]
    unfiltered = _tensor_list(getattr(base, "dex_fingertip_force_norm", None), env_index)
    target_matrix = _tensor_matrix(getattr(base, "dex_fingertip_target_force_norm", None), env_index)
    filter_names = [str(item) for item in list(getattr(base, "dex_fingertip_target_filter_names", []) or [])]
    target_index = filter_names.index(part) if part in filter_names else -1
    target_forces = []
    for row in target_matrix:
        target_forces.append(float(row[target_index]) if 0 <= target_index < len(row) else 0.0)
    palm_quat = _tensor_vec(getattr(base, "fingertip_midpoint_quat", None), env_index, width=4)
    ctrl_palm_quat = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4)
    hand_delta = _tensor_vec(getattr(base, "last_dex_hand_target_delta", None), env_index, width=20)
    limiter_delta = _tensor_vec(getattr(base, "last_dex_hand_limiter_delta", None), env_index, width=20)
    wrist_pre_clamp = _tensor_vec(getattr(base, "v95_last_wrist_target_pre_clamp", None), env_index)
    wrist_post_clamp = _tensor_vec(getattr(base, "v95_last_wrist_target_post_clamp", None), env_index)
    workspace_clamp_vec = _sub_vec(wrist_post_clamp, wrist_pre_clamp)
    table_barrier_delta_z = _tensor_scalar(getattr(base, "v95_last_table_barrier_delta_z", None), env_index)
    true_names = list(getattr(base, "dex_fingertip_true_body_names", []) or [])
    state = {
        "object_local_pos": object_local,
        "object_world_pos": object_w,
        "object_quat_wxyz": object_quat,
        "palm_local_pos": palm,
        "palm_quat_wxyz": palm_quat,
        "ctrl_target_palm_quat_wxyz": ctrl_palm_quat,
        "finger3_tip_local_pos": tip3,
        "finger4_tip_local_pos": tip4,
        "finger3_source_body_name": true_names[FINGER3_INDEX] if len(true_names) > FINGER3_INDEX else "",
        "finger4_source_body_name": true_names[FINGER4_INDEX] if len(true_names) > FINGER4_INDEX else "",
        "dex_fingertip_true_source": str(getattr(base, "dex_fingertip_true_source", "")),
        "finger3_unfiltered_force_n": _list_get(unfiltered, FINGER3_INDEX),
        "finger4_unfiltered_force_n": _list_get(unfiltered, FINGER4_INDEX),
        "finger3_target_filtered_force_n": _list_get(target_forces, FINGER3_INDEX),
        "finger4_target_filtered_force_n": _list_get(target_forces, FINGER4_INDEX),
        "target_filter_names": filter_names,
        "target_filter_index": int(target_index),
        "target_filtered_force_available": bool(
            target_index >= 0 and _tensor_scalar(getattr(base, "dex_target_force_valid", None), env_index) > 0.5
        ),
        "force_threshold_n": float(force_threshold_n),
        "wrist_target_delta_xyz": _tensor_vec(getattr(base, "v95_last_wrist_target_delta", None), env_index),
        "workspace_clamp_delta_m": _tensor_scalar(getattr(base, "v95_last_workspace_clamp_delta", None), env_index),
        "table_barrier_delta_z_m": table_barrier_delta_z,
        "wrist_target_pre_clamp_xyz": wrist_pre_clamp,
        "wrist_target_post_clamp_xyz": wrist_post_clamp,
        "workspace_clamp_vector_xyz": workspace_clamp_vec,
        "workspace_clamp_x": _list_get(workspace_clamp_vec, 0),
        "workspace_clamp_y": _list_get(workspace_clamp_vec, 1),
        "workspace_clamp_z": _list_get(workspace_clamp_vec, 2),
        "workspace_clamp_dominant_axis": _dominant_clamp_axis(workspace_clamp_vec, table_barrier_delta_z),
        "hand_target_delta_l2": _norm(hand_delta),
        "hand_limiter_delta_l2": _norm(limiter_delta),
        "last_dex_hand_limiter_applied": bool(getattr(base, "last_dex_hand_limiter_applied", False)),
        "last_dex_hand_limiter_bypassed_for_v95": bool(
            getattr(base, "last_dex_hand_limiter_bypassed_for_v95", False)
        ),
    }
    state["active_target_filtered_force_peak_n"] = max(
        state["finger3_target_filtered_force_n"], state["finger4_target_filtered_force_n"]
    )
    state["active_unfiltered_force_peak_n"] = max(
        state["finger3_unfiltered_force_n"], state["finger4_unfiltered_force_n"]
    )
    return state


def _analytic_pregrasp(
    env: Any,
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    reset_state: dict[str, Any],
) -> dict[str, Any]:
    state = _state_with_contact_target(base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.force_threshold_n))
    search_target = _contact_search_entry_target(base, env_index, cfg, state)
    contact_target = list(search_target["search_target_local_xyz"])
    tip3 = state["finger3_tip_local_pos"]
    tip4 = state["finger4_tip_local_pos"]
    palm = state["palm_local_pos"]
    tip3_offset = _sub_vec(tip3, palm)
    tip4_offset = _sub_vec(tip4, palm)
    desired_tip = list(contact_target)
    palm_from_tip3 = _sub_vec(desired_tip, tip3_offset)
    desired_palm = list(palm_from_tip3)
    try:
        target = torch.tensor(desired_palm, dtype=torch.float32, device=base.device)
        base.ctrl_target_fingertip_midpoint_pos[env_index, :] = target
        env_ids = torch.tensor([env_index], dtype=torch.long, device=base.device)
        if hasattr(base, "set_pos_inverse_kinematics"):
            base.set_pos_inverse_kinematics(env_ids=env_ids, max_time=0.35, pos_tol=0.003)
        else:
            base.generate_ctrl_signals()
            base.step_sim_no_action()
    except Exception:
        pass
    out = _state_with_contact_target(base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.force_threshold_n))
    out["pregrasp_desired_tip3_local_pos"] = desired_tip
    out["pregrasp_desired_tip4_local_pos"] = []
    out["pregrasp_desired_palm_local_pos"] = desired_palm
    out["pregrasp_tip3_offset_from_palm"] = tip3_offset
    out["pregrasp_tip4_offset_from_palm"] = tip4_offset
    out["contact_search_entry_local_xyz"] = contact_target
    out["contact_search_side_dir_xy"] = search_target["side_dir_xy"]
    out["contact_search_base_z_m"] = search_target["base_z_m"]
    out["finger3_distance_to_contact_search_entry_m"] = _distance(out["finger3_tip_local_pos"], contact_target)
    out["pregrasp_object_moved_m"] = _distance(reset_state["object_local_pos"], out["object_local_pos"])
    out["pregrasp_safe"] = bool(
        out["active_target_filtered_force_peak_n"] <= cfg.force_threshold_n
        and out["pregrasp_object_moved_m"] <= 0.02
    )
    return out


def _reanchor_after_pregrasp(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None = None,
) -> dict[str, Any]:
    before = _state_with_contact_target(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.force_threshold_n),
    )
    z_before = _z_clamp_amount(before)
    reanchored = False
    try:
        env_ids = torch.tensor([env_index], dtype=torch.long, device=base.device)
        base.ctrl_target_fingertip_midpoint_pos[env_index, :] = base.fingertip_midpoint_pos[env_index].detach().clone()
        base.ctrl_target_fingertip_midpoint_quat[env_index, :] = base.fingertip_midpoint_quat[env_index].detach().clone()
        if hasattr(base, "generate_ctrl_signals"):
            base.generate_ctrl_signals()
        _configure_action_control(base, env_index, mode="anchored_delta", allow_commanded_hand_actions=True)
        reanchored = True
    except Exception:
        pass
    audit = _step_policy(env, mapper, env_index, [0.0] * 16, alignment=alignment)
    after = _state_with_contact_target(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.force_threshold_n),
    )
    z_after = _z_clamp_amount(after)
    contact_adjustment: dict[str, Any] = {}
    raised = bool(z_after > float(cfg.reanchor_z_clamp_threshold_m))
    if raised:
        contact_adjustment["z_bias_m"] = float(cfg.reanchor_target_z_raise_m)
    _append_log(
        logs,
        "reanchor",
        0,
        after,
        cfg,
        object_reset_pos,
        extra={
            "control_reanchored_after_pregrasp": reanchored,
            "z_clamp_before_reanchor": z_before,
            "z_clamp_after_reanchor": z_after,
            "press_target_z_raised_after_reanchor": raised,
            **audit,
        },
        alignment=alignment,
        base=base,
    )
    return {
        "control_reanchored_after_pregrasp": reanchored,
        "z_clamp_before_reanchor": z_before,
        "z_clamp_after_reanchor": z_after,
        "press_target_z_raised_after_reanchor": raised,
        "contact_adjustment": contact_adjustment,
    }


def _configure_action_control(
    base: Any,
    env_index: int,
    *,
    mode: str,
    allow_commanded_hand_actions: bool,
) -> None:
    if not hasattr(base, "v95_configure_action_control") or torch is None:
        return
    env_ids = torch.tensor([env_index], dtype=torch.long, device=base.device)
    base.v95_configure_action_control(
        mode=mode,
        allow_commanded_hand_actions=allow_commanded_hand_actions,
        env_ids=env_ids,
    )


def _z_clamp_amount(state: dict[str, Any]) -> float:
    return max(
        abs(float(state.get("workspace_clamp_z", 0.0) or 0.0)),
        abs(float(state.get("table_barrier_delta_z_m", 0.0) or 0.0)),
    )


def _set_ctrl_target_to_measured_midpoint(base: Any, env_index: int) -> bool:
    if torch is None:
        return False
    try:
        base.ctrl_target_fingertip_midpoint_pos[env_index, :] = base.fingertip_midpoint_pos[env_index].detach().clone()
        base.ctrl_target_fingertip_midpoint_quat[env_index, :] = base.fingertip_midpoint_quat[env_index].detach().clone()
        if hasattr(base, "generate_ctrl_signals"):
            base.generate_ctrl_signals()
        return True
    except Exception:
        return False


def _quat_normalize_wxyz(quat: list[float] | tuple[float, ...] | None) -> list[float]:
    values = list(quat or [])
    if len(values) < 4:
        return [1.0, 0.0, 0.0, 0.0]
    out = [float(values[index]) for index in range(4)]
    norm = math.sqrt(sum(value * value for value in out))
    if norm <= 1.0e-8 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [value / norm for value in out]


def _quat_mul_wxyz(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = _quat_normalize_wxyz(a)
    bw, bx, by, bz = _quat_normalize_wxyz(b)
    return _quat_normalize_wxyz(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_from_euler_xyz_wxyz(roll_rad: float, pitch_rad: float, yaw_rad: float) -> list[float]:
    cr = math.cos(float(roll_rad) * 0.5)
    sr = math.sin(float(roll_rad) * 0.5)
    cp = math.cos(float(pitch_rad) * 0.5)
    sp = math.sin(float(pitch_rad) * 0.5)
    cy = math.cos(float(yaw_rad) * 0.5)
    sy = math.sin(float(yaw_rad) * 0.5)
    return _quat_normalize_wxyz(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _quat_for_rpy_offset(anchor_quat: list[float], rpy_deg: tuple[float, float, float]) -> list[float]:
    delta = _quat_from_euler_xyz_wxyz(
        math.radians(float(rpy_deg[0])),
        math.radians(float(rpy_deg[1])),
        math.radians(float(rpy_deg[2])),
    )
    return _quat_mul_wxyz(_quat_normalize_wxyz(anchor_quat), delta)


def _quat_angle_delta_wxyz(a: list[float], b: list[float]) -> float | str:
    if len(a) < 4 or len(b) < 4:
        return ""
    qa = _quat_normalize_wxyz(a)
    qb = _quat_normalize_wxyz(b)
    dot = abs(sum(qa[index] * qb[index] for index in range(4)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.acos(dot))


def _diagnostic_position_offsets(cfg: ScriptedBaselineConfig) -> list[list[float]]:
    values = [float(value) for value in tuple(cfg.diagnostic_sweep_position_offsets_m)]
    if 0.0 not in values:
        values.append(0.0)
    combos = [[x, y, z] for x in values for y in values for z in values]
    combos.sort(key=lambda item: (_norm(item), abs(item[2]), abs(item[1]), abs(item[0]), item[2], item[1], item[0]))
    return combos


def _diagnostic_orientation_offsets_deg(cfg: ScriptedBaselineConfig) -> list[tuple[float, float, float]]:
    roll_values = [float(value) for value in tuple(cfg.diagnostic_sweep_roll_offsets_deg) if abs(float(value)) > 1.0e-9]
    pitch_values = [float(value) for value in tuple(cfg.diagnostic_sweep_pitch_offsets_deg) if abs(float(value)) > 1.0e-9]
    yaw_values = [float(value) for value in tuple(cfg.diagnostic_sweep_yaw_offsets_deg) if abs(float(value)) > 1.0e-9]
    offsets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    offsets.extend((value, 0.0, 0.0) for value in roll_values)
    offsets.extend((0.0, value, 0.0) for value in pitch_values)
    offsets.extend((0.0, 0.0, value) for value in yaw_values)
    seen: set[tuple[float, float, float]] = set()
    unique: list[tuple[float, float, float]] = []
    for item in offsets:
        key = (float(item[0]), float(item[1]), float(item[2]))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _grid_position_offsets(values: tuple[float, ...]) -> list[list[float]]:
    vals = [float(value) for value in tuple(values)]
    if 0.0 not in vals:
        vals.append(0.0)
    combos = [[x, y, z] for x in vals for y in vals for z in vals]
    combos.sort(key=lambda item: (_norm(item), abs(item[2]), abs(item[1]), abs(item[0]), item[2], item[1], item[0]))
    return combos


def _axis_orientation_offsets_deg(values: tuple[float, ...]) -> list[tuple[float, float, float]]:
    vals = [float(value) for value in tuple(values) if abs(float(value)) > 1.0e-9]
    offsets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    offsets.extend((value, 0.0, 0.0) for value in vals)
    offsets.extend((0.0, value, 0.0) for value in vals)
    offsets.extend((0.0, 0.0, value) for value in vals)
    seen: set[tuple[float, float, float]] = set()
    unique: list[tuple[float, float, float]] = []
    for item in offsets:
        key = (float(item[0]), float(item[1]), float(item[2]))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _contact_refinement_force_flags(candidate: dict[str, Any], cfg: ScriptedBaselineConfig) -> dict[str, Any]:
    threshold = float(cfg.contact_refinement_force_threshold_n)
    per_finger_upper = float(cfg.contact_refinement_hard_force_stop_n)
    total_lower = float(cfg.contact_refinement_reasonable_total_force_min_n)
    total_upper = float(cfg.contact_refinement_reasonable_total_force_max_n)
    stable_limit = float(cfg.contact_refinement_stable_object_motion_limit_m)
    f3 = float(candidate.get("finger3_force_n", 0.0) or 0.0)
    f4 = float(candidate.get("finger4_force_n", 0.0) or 0.0)
    total_force = f3 + f4
    obj = float(candidate.get("object_displacement_from_refinement_start_m", 0.0) or 0.0)
    f3_contact = f3 > threshold
    f4_contact = f4 > threshold
    f3_reasonable = f3_contact and f3 <= per_finger_upper
    f4_reasonable = f4_contact and f4 <= per_finger_upper
    contacted = [value for value, active in ((f3, f3_contact), (f4, f4_contact)) if active]
    total_force_reasonable = bool(contacted and total_lower <= total_force <= total_upper and max(f3, f4) <= per_finger_upper)
    two_finger = bool(f3_contact and f4_contact)
    object_stable = bool(obj < stable_limit)
    return {
        "finger3_contact": f3_contact,
        "finger4_contact": f4_contact,
        "finger3_force_reasonable": f3_reasonable,
        "finger4_force_reasonable": f4_reasonable,
        "two_finger_contact": two_finger,
        "two_finger_force_reasonable": bool(two_finger and total_force_reasonable),
        "any_target_contact": bool(f3_contact or f4_contact),
        "single_finger_contact": bool((f3_contact or f4_contact) and not two_finger),
        "force_level_reasonable": total_force_reasonable,
        "total_force_reasonable": total_force_reasonable,
        "object_motion_stable": object_stable,
        "stable_two_finger_contact": bool(two_finger and total_force_reasonable and object_stable),
    }


def _contact_refinement_rank(candidate: dict[str, Any], cfg: ScriptedBaselineConfig) -> tuple[float, float, float, float]:
    flags = _contact_refinement_force_flags(candidate, cfg)
    f3 = float(candidate.get("finger3_force_n", 0.0) or 0.0)
    f4 = float(candidate.get("finger4_force_n", 0.0) or 0.0)
    max_force = max(f3, f4)
    total_force = f3 + f4
    obj = float(candidate.get("object_displacement_from_refinement_start_m", 0.0) or 0.0)
    total_lower = float(cfg.contact_refinement_reasonable_total_force_min_n)
    total_upper = float(cfg.contact_refinement_reasonable_total_force_max_n)
    if total_force < total_lower:
        force_band_penalty = total_lower - total_force
    elif total_force > total_upper:
        force_band_penalty = total_force - total_upper
    else:
        force_band_penalty = 0.0
    if flags["stable_two_finger_contact"]:
        priority = 0.0
    elif flags["two_finger_contact"]:
        priority = 1.0
    elif flags["any_target_contact"]:
        priority = 2.0
    else:
        priority = 3.0
    return (priority, obj, force_band_penalty, total_force if total_force > 0.0 else max_force)


def _contact_refinement_result_class(approach: dict[str, Any], cfg: ScriptedBaselineConfig) -> tuple[str, str]:
    termination = str(approach.get("contact_refinement_termination_reason", ""))
    if termination in {"contact_refinement_force_limit", "two_seed_refinement_force_limit"}:
        return (
            "TWO_SEED_REFINEMENT_FORCE_LIMIT",
            "two-seed refinement stopped because target-filtered force exceeded the 20N safety limit",
        )
    if termination == "raw_non_target_contact":
        return (
            "TWO_SEED_REFINEMENT_STOPPED_BY_NON_TARGET_CONTACT",
            "two-seed refinement stopped on resolved non-target raw contact before selecting a stable Screw1 contact",
        )
    flags = {
        "two_finger_contact": bool(approach.get("two_finger_contact_acquired", False)),
        "force_level_reasonable": bool(approach.get("force_level_reasonable", False)),
        "object_motion_stable": bool(approach.get("best_contact_object_motion_stable", False)),
        "finger3_contact": bool(approach.get("contact_refinement_best_finger3_contact", False)),
        "finger4_contact": bool(approach.get("contact_refinement_best_finger4_contact", False)),
    }
    if flags["two_finger_contact"] and flags["force_level_reasonable"] and flags["object_motion_stable"]:
        return (
            "STABLE_TWO_FINGER_CONTACT_ACQUIRED",
            "finger3 and finger4 both produced low-force Screw1 target-filtered contact without excessive object motion",
        )
    if flags["two_finger_contact"]:
        return (
            "TWO_FINGER_CONTACT_OBJECT_MOVED",
            "finger3 and finger4 both contacted Screw1, but object displacement exceeded the stability limit",
        )
    if flags["finger3_contact"] or flags["finger4_contact"]:
        return (
            "SINGLE_FINGER_CONTACT_ONLY",
            "two-seed refinement found Screw1 target-filtered contact on only one active finger",
        )
    return (
        "TWO_SEED_REFINEMENT_NO_TARGET_CONTACT",
        "two-seed refinement exhausted its bounded probes without Screw1 target-filtered contact",
    )


def _diagnostic_sweep_judgement(approach: dict[str, Any], threshold: float) -> str:
    raw_contact = float(approach.get("raw_any_screw1_peak_n", 0.0) or 0.0) > float(threshold)
    target_contact = bool(
        approach.get("contact_search_success")
        or approach.get("contact_search_target_contact_observed")
        or approach.get("contact_refinement_any_target_contact")
    )
    orientation_norm = float(approach.get("contact_search_contact_orientation_offset_norm_deg", 0.0) or 0.0)
    if target_contact or raw_contact:
        if orientation_norm > 1.0e-6:
            return "D_contact_only_with_orientation_offset"
        return "A_contact_found"
    attempted = max(1, int(approach.get("contact_search_steps", 0) or 0))
    reached = int(approach.get("contact_search_reached_pose_count", 0) or 0)
    if reached < max(3, int(math.ceil(0.25 * attempted))):
        return "B_pregrasp_generation_issue"
    return "C_no_contact_despite_broad_reachable_sweep"


def _raise_surface_lock_z(surface_lock: dict[str, Any], dz: float) -> dict[str, Any]:
    out = dict(surface_lock or {})
    for key in ("surface_point_local_xyz", "precontact_point_local_xyz", "press_target_local_xyz"):
        value = list(out.get(key, []) or [])
        if len(value) >= 3:
            value[2] = float(value[2]) + float(dz)
            out[key] = value
    out["contact_target_z_bias_m"] = float(out.get("contact_target_z_bias_m", 0.0) or 0.0) + float(dz)
    out["press_target_z_raised_for_press_preflight"] = True
    return out


def _move_ctrl_target_by_ik(
    base: Any,
    env_index: int,
    desired_palm: list[float],
    *,
    desired_quat: list[float] | None = None,
    max_time: float,
    pos_tol: float,
) -> dict[str, Any]:
    before_tip = _tensor_matrix(getattr(base, "dex_fingertip_pos", None), env_index)
    before_tip3 = before_tip[FINGER3_INDEX] if len(before_tip) > FINGER3_INDEX else [0.0, 0.0, 0.0]
    before_ctrl = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    before_quat = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4)
    ok = False
    error = ""
    try:
        target = torch.tensor(desired_palm, dtype=torch.float32, device=base.device)
        base.ctrl_target_fingertip_midpoint_pos[env_index, :] = target
        if desired_quat is not None and hasattr(base, "ctrl_target_fingertip_midpoint_quat"):
            quat = torch.tensor(_quat_normalize_wxyz(desired_quat), dtype=torch.float32, device=base.device)
            base.ctrl_target_fingertip_midpoint_quat[env_index, :] = quat
        env_ids = torch.tensor([env_index], dtype=torch.long, device=base.device)
        if hasattr(base, "set_pos_inverse_kinematics"):
            base.set_pos_inverse_kinematics(env_ids=env_ids, max_time=float(max_time), pos_tol=float(pos_tol))
        elif hasattr(base, "generate_ctrl_signals"):
            base.generate_ctrl_signals()
            base.step_sim_no_action()
        ok = True
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    after_tip = _tensor_matrix(getattr(base, "dex_fingertip_pos", None), env_index)
    after_tip3 = after_tip[FINGER3_INDEX] if len(after_tip) > FINGER3_INDEX else [0.0, 0.0, 0.0]
    after_ctrl = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    after_quat = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4)
    return {
        "ik_target_fallback_ok": ok,
        "ik_target_fallback_error": error,
        "ik_ctrl_target_before_x": _list_get(before_ctrl, 0),
        "ik_ctrl_target_before_y": _list_get(before_ctrl, 1),
        "ik_ctrl_target_before_z": _list_get(before_ctrl, 2),
        "ik_ctrl_target_after_x": _list_get(after_ctrl, 0),
        "ik_ctrl_target_after_y": _list_get(after_ctrl, 1),
        "ik_ctrl_target_after_z": _list_get(after_ctrl, 2),
        "ik_ctrl_target_delta_l2": _distance(before_ctrl, after_ctrl),
        "ik_ctrl_target_quat_before_w": _list_get(before_quat, 0, ""),
        "ik_ctrl_target_quat_before_x": _list_get(before_quat, 1, ""),
        "ik_ctrl_target_quat_before_y": _list_get(before_quat, 2, ""),
        "ik_ctrl_target_quat_before_z": _list_get(before_quat, 3, ""),
        "ik_ctrl_target_quat_after_w": _list_get(after_quat, 0, ""),
        "ik_ctrl_target_quat_after_x": _list_get(after_quat, 1, ""),
        "ik_ctrl_target_quat_after_y": _list_get(after_quat, 2, ""),
        "ik_ctrl_target_quat_after_z": _list_get(after_quat, 3, ""),
        "ik_ctrl_target_quat_delta_rad": _quat_angle_delta_wxyz(before_quat, after_quat),
        "ik_finger3_delta_l2": _distance(before_tip3, after_tip3),
    }


def _run_open_hand(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None = None,
    *,
    adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adjustment = dict(adjustment or {})
    steps = max(0, int(cfg.open_hand_steps))
    executed_steps = 0
    stop_reason = "not_executed"
    for step in range(steps):
        stop_reason = "max_steps_reached"
        state_before = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=adjustment,
        )
        if alignment is not None:
            alignment.update_markers(base, state_before, contact_target_local=state_before.get("contact_target_local_pos"))
        action = [0.0] * 16
        for col in range(6, 16):
            action[col] = float(cfg.open_hand_command)
        audit = _step_policy(env, mapper, env_index, action, alignment=alignment)
        state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=adjustment,
        )
        _append_log(
            logs,
            "open_hand",
            step,
            state,
            cfg,
            object_reset_pos,
            extra={"open_hand_command_value": cfg.open_hand_command, **audit},
            alignment=alignment,
            base=base,
        )
        executed_steps = step + 1
        if float(state.get("active_target_filtered_force_peak_n", 0.0) or 0.0) > float(cfg.force_threshold_n):
            stop_reason = "target_contact_during_open"
            break
        if _clamp_or_barrier(state, threshold_m=float(cfg.reanchor_z_clamp_threshold_m)):
            stop_reason = "workspace_or_table_clamp"
            break
    return {"executed": bool(executed_steps > 0), "steps": executed_steps, "stop_reason": stop_reason}


def _contact_search_side_dir(state: dict[str, Any]) -> list[float]:
    root = list(state.get("object_local_pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
    tip = list(state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
    side = [float(tip[0]) - float(root[0]), float(tip[1]) - float(root[1]), 0.0]
    side = _unit(side)
    return side if _norm(side) > 1.0e-6 else [1.0, 0.0, 0.0]


def _contact_search_entry_target(
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    state: dict[str, Any],
    *,
    radial_m: float | None = None,
    lateral_m: float = 0.0,
    z_offset_m: float = 0.0,
    side_dir: list[float] | None = None,
) -> dict[str, Any]:
    root = list(state.get("object_local_pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
    side = _unit(list(side_dir or _contact_search_side_dir(state)))
    if _norm(side) <= 1.0e-6:
        side = [1.0, 0.0, 0.0]
    tangent = [-side[1], side[0], 0.0]
    radius = float(cfg.contact_search_entry_radius_m if radial_m is None else radial_m)
    table_z = _table_z_estimate(base, env_index, root)
    base_z = max(float(root[2]), float(table_z) + float(cfg.surface_table_clearance_m))
    target = [
        float(root[0]) + side[0] * radius + tangent[0] * float(lateral_m),
        float(root[1]) + side[1] * radius + tangent[1] * float(lateral_m),
        float(base_z) + float(z_offset_m),
    ]
    return {
        "search_target_local_xyz": target,
        "side_dir_xy": [side[0], side[1], 0.0],
        "tangent_dir_xy": [tangent[0], tangent[1], 0.0],
        "base_z_m": base_z,
        "radial_m": radius,
        "lateral_m": float(lateral_m),
        "z_offset_m": float(z_offset_m),
    }


def _coarse_pregrasp_root_check(state: dict[str, Any], cfg: ScriptedBaselineConfig) -> dict[str, Any]:
    entry = list(state.get("contact_search_entry_local_xyz", []) or [])
    if len(entry) < 3:
        root = list(state.get("object_local_pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        side = _contact_search_side_dir(state)
        entry = [
            float(root[0]) + side[0] * float(cfg.contact_search_entry_radius_m),
            float(root[1]) + side[1] * float(cfg.contact_search_entry_radius_m),
            float(root[2]) + float(cfg.table_safe_clearance_m),
        ]
    f3 = _distance(list(state.get("finger3_tip_local_pos", [])), entry)
    f4 = _distance(list(state.get("finger4_tip_local_pos", [])), entry)
    near = bool(f3 <= cfg.pregrasp_fail_distance_m)
    goal = bool(f3 <= cfg.pregrasp_goal_distance_m)
    return {
        "pregrasp_finger3_distance": f3,
        "pregrasp_finger4_distance": f4,
        "pregrasp_near_enough": near,
        "pregrasp_goal_met": goal,
        "coarse_pregrasp_entry_local_xyz": entry,
        "coarse_pregrasp_finger3_distance_to_entry_m": f3,
        "coarse_pregrasp_finger4_distance_to_entry_m": f4,
    }


def _contact_search_lanes(cfg: ScriptedBaselineConfig) -> list[tuple[int, float, float]]:
    lanes: list[tuple[int, float, float]] = []
    lane_id = 0
    for z_offset in tuple(cfg.contact_search_z_offsets_m):
        for lateral in tuple(cfg.contact_search_lateral_offsets_m):
            lanes.append((lane_id, float(lateral), float(z_offset)))
            lane_id += 1
    return lanes or [(0, 0.0, 0.0)]


def _contact_search_object_contact_state(peaks: dict[str, Any], state: dict[str, Any], threshold: float) -> str:
    active_target = float(state.get("active_target_filtered_force_peak_n", 0.0) or 0.0)
    active_unfiltered = float(state.get("active_unfiltered_force_peak_n", 0.0) or 0.0)
    if active_target > threshold:
        return "target_filtered_screw1_contact"
    if float(peaks.get("raw_tip_screw1_peak_n", 0.0) or 0.0) > threshold:
        return "raw_tip_screw1_contact_no_target_filtered"
    if float(peaks.get("raw_any_screw1_peak_n", 0.0) or 0.0) > threshold:
        return "raw_screw1_contact_no_target_filtered"
    if float(peaks.get("raw_non_target_peak_n", 0.0) or 0.0) > threshold:
        return "raw_non_target_contact"
    if active_unfiltered > threshold and active_target <= threshold:
        return "unresolved_unfiltered_contact"
    return "none"


def _append_contact_search_log(
    logs: list[dict[str, Any]],
    step: int,
    state: dict[str, Any],
    cfg: ScriptedBaselineConfig,
    object_reset_pos: list[float],
    *,
    extra: dict[str, Any],
    alignment: AlignmentRecorder | None,
    base: Any,
    phase: str = "contact_search",
) -> dict[str, Any]:
    before = len(getattr(alignment, "contact_rows", []) or []) if alignment is not None else 0
    _append_log(
        logs,
        phase,
        step,
        state,
        cfg,
        object_reset_pos,
        extra=extra,
        alignment=alignment,
        base=base,
    )
    rows = list(getattr(alignment, "contact_rows", []) or [])[before:] if alignment is not None else []
    peaks = _contact_truth_peaks(rows, cfg.part, cfg.force_threshold_n)
    object_state = _contact_search_object_contact_state(peaks, state, float(cfg.force_threshold_n))
    update = {
        "raw_tip_screw1_peak_n": peaks.get("raw_tip_screw1_peak_n", 0.0),
        "raw_any_screw1_peak_n": peaks.get("raw_any_screw1_peak_n", 0.0),
        "raw_non_target_peak_n": peaks.get("raw_non_target_peak_n", 0.0),
        "raw_unknown_object_peak_n": peaks.get("raw_unknown_object_peak_n", 0.0),
        "raw_screw1_peak_body": peaks.get("raw_screw1_peak_body", ""),
        "raw_screw1_peak_object": peaks.get("raw_screw1_peak_object", ""),
        "raw_sensor_available": peaks.get("raw_sensor_available", False),
        "object_contact_state": object_state,
    }
    if logs:
        logs[-1].update(_plain(update))
    peaks["object_contact_state"] = object_state
    return peaks


def _run_contact_search_acquisition(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None = None,
    *,
    adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del mapper, adjustment
    if alignment is not None:
        alignment.setup_body_mapping_and_sensors(base)
    search_reanchored = _set_ctrl_target_to_measured_midpoint(base, env_index)
    _configure_action_control(base, env_index, mode="", allow_commanded_hand_actions=True)
    search_zero_audit = {"ctrl_target_delta_l2": 0.0}
    threshold = float(cfg.contact_refinement_force_threshold_n)
    hard_force_stop = float(cfg.contact_refinement_hard_force_stop_n)
    start_state = _state_with_contact_target(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.force_threshold_n),
    )
    search_center_f3 = list(start_state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
    search_center_f4 = list(start_state.get("finger4_tip_local_pos", [0.0, 0.0, 0.0]))
    search_center_palm = list(start_state.get("palm_local_pos", [0.0, 0.0, 0.0]))
    search_center_root = list(start_state.get("object_local_pos", [0.0, 0.0, 0.0]))
    anchor_quat = _quat_normalize_wxyz(
        list(start_state.get("palm_quat_wxyz", []) or start_state.get("ctrl_target_palm_quat_wxyz", []) or [])
    )
    seed_a_offset = [float(value) for value in tuple(cfg.contact_refinement_seed_a_position_offset_m)]
    seed_a_rpy = tuple(float(value) for value in tuple(cfg.contact_refinement_seed_a_orientation_rpy_deg))
    seed_b_offset = [float(value) for value in tuple(cfg.contact_refinement_seed_b_position_offset_m)]
    seed_b_rpy = tuple(float(value) for value in tuple(cfg.contact_refinement_seed_b_orientation_rpy_deg))
    midpoint_offset = [
        0.5 * (float(seed_a_offset[index]) + float(seed_b_offset[index]))
        for index in range(3)
    ]
    midpoint_rpy = tuple(0.5 * (float(seed_a_rpy[index]) + float(seed_b_rpy[index])) for index in range(3))
    refinement_centers = [
        {
            "center_id": "midpoint",
            "interpolation_factor": 0.5,
            "position_offset_xyz_m": midpoint_offset,
            "orientation_rpy_deg": list(midpoint_rpy),
        },
        {
            "center_id": "seed_a_finger4",
            "interpolation_factor": 0.0,
            "position_offset_xyz_m": seed_a_offset,
            "orientation_rpy_deg": list(seed_a_rpy),
        },
        {
            "center_id": "seed_b_finger3",
            "interpolation_factor": 1.0,
            "position_offset_xyz_m": seed_b_offset,
            "orientation_rpy_deg": list(seed_b_rpy),
        },
    ]
    coarse_positions = _grid_position_offsets(tuple(cfg.contact_refinement_position_offsets_m))
    coarse_orientations = _axis_orientation_offsets_deg(tuple(cfg.contact_refinement_orientation_offsets_deg))
    coarse_pose_count = len(coarse_positions) * len(coarse_orientations)
    planned_pose_count = coarse_pose_count * len(refinement_centers)
    max_steps = int(cfg.contact_search_max_steps)
    seed_target_f3 = _add_vec(search_center_f3, midpoint_offset)
    start_f3_distance = _distance(search_center_f3, seed_target_f3)
    start_f4_distance = _distance(search_center_f4, seed_target_f3)
    min_f3_distance = start_f3_distance
    min_f4_distance = start_f4_distance
    raw_tip_peak = 0.0
    raw_any_peak = 0.0
    raw_non_target_peak = 0.0
    raw_unknown_peak = 0.0
    unresolved_unfiltered_peak = 0.0
    unresolved_unfiltered_observed = False
    raw_peak_body = ""
    raw_peak_object = ""
    non_target = False
    clamp_seen = False
    target_contact_observed = bool(float(start_state.get("active_target_filtered_force_peak_n", 0.0) or 0.0) > threshold)
    termination_reason = "contact_refinement_completed"
    total_commanded = 0.0
    steps = 0
    no_motion_streak = 0
    last_object_contact_state = "none"
    executed = False
    reached_pose_count = 0
    clamped_probe_count = 0
    controller_no_motion_count = 0
    contact_pose: dict[str, Any] = {}
    best_candidate: dict[str, Any] | None = None
    best_two_finger_candidate: dict[str, Any] | None = None
    fine_pass_executed = False
    stop_requested = False
    refinement_start_object = list(start_state.get("object_local_pos", [0.0, 0.0, 0.0]))

    def _update_best(candidate: dict[str, Any]) -> None:
        nonlocal best_candidate, best_two_finger_candidate
        rank = _contact_refinement_rank(candidate, cfg)
        candidate["contact_refinement_rank"] = list(rank)
        if best_candidate is None or rank < tuple(best_candidate.get("contact_refinement_rank", [99.0, 99.0, 99.0, 99.0])):
            best_candidate = dict(candidate)
        if bool(_contact_refinement_force_flags(candidate, cfg).get("two_finger_contact", False)):
            if best_two_finger_candidate is None or rank < tuple(
                best_two_finger_candidate.get("contact_refinement_rank", [99.0, 99.0, 99.0, 99.0])
            ):
                best_two_finger_candidate = dict(candidate)

    def _run_refinement_pass(
        pass_name: str,
        positions: list[list[float]],
        orientations: list[tuple[float, float, float]],
        *,
        center: dict[str, Any],
    ) -> None:
        nonlocal steps, total_commanded, min_f3_distance, min_f4_distance
        nonlocal raw_tip_peak, raw_any_peak, raw_non_target_peak, raw_unknown_peak
        nonlocal raw_peak_body, raw_peak_object, last_object_contact_state
        nonlocal non_target, clamp_seen, target_contact_observed, termination_reason
        nonlocal executed, reached_pose_count, clamped_probe_count, controller_no_motion_count
        nonlocal no_motion_streak, unresolved_unfiltered_peak, unresolved_unfiltered_observed, contact_pose
        center_id = str(center.get("center_id", ""))
        center_interpolation = float(center.get("interpolation_factor", 0.0) or 0.0)
        center_position = [float(value) for value in list(center.get("position_offset_xyz_m", [0.0, 0.0, 0.0]))[:3]]
        center_rpy_values = list(center.get("orientation_rpy_deg", [0.0, 0.0, 0.0]))[:3]
        center_rpy = tuple(float(value) for value in center_rpy_values)
        for position_id, offset in enumerate(positions):
            for orientation_id, local_rpy in enumerate(orientations):
                if steps >= max_steps:
                    termination_reason = "max_search_steps"
                    return
                local_offset = [float(offset[0]), float(offset[1]), float(offset[2])]
                total_offset = _add_vec(center_position, local_offset)
                total_rpy = (
                    float(center_rpy[0]) + float(local_rpy[0]),
                    float(center_rpy[1]) + float(local_rpy[1]),
                    float(center_rpy[2]) + float(local_rpy[2]),
                )
                state = _state_with_contact_target(
                    base,
                    env_index,
                    cfg,
                    read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                )
                target = _add_vec(search_center_f3, total_offset)
                desired_palm = _add_vec(search_center_palm, total_offset)
                desired_quat = _quat_for_rpy_offset(anchor_quat, total_rpy)
                tip_before = list(state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
                commanded_delta = _sub_vec(target, tip_before)
                commanded_norm = _norm(commanded_delta)
                if alignment is not None:
                    alignment.update_markers(base, state, contact_target_local=target)
                ik_audit = _move_ctrl_target_by_ik(
                    base,
                    env_index,
                    desired_palm,
                    desired_quat=desired_quat,
                    max_time=float(cfg.contact_refinement_ik_max_time_s),
                    pos_tol=float(cfg.contact_refinement_pos_tol_m),
                )
                total_commanded += commanded_norm
                next_state = _state_with_contact_target(
                    base,
                    env_index,
                    cfg,
                    read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                )
                tip_after = list(next_state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
                actual_motion = _distance(tip_before, tip_after)
                if commanded_norm > 1.0e-6 and actual_motion < 1.0e-5:
                    no_motion_streak += 1
                else:
                    no_motion_streak = 0
                controller_no_motion_count += int(no_motion_streak >= int(cfg.contact_search_no_motion_steps))
                f3_to_target = _distance(tip_after, target)
                f4_to_target = _distance(next_state.get("finger4_tip_local_pos", []), target)
                probe_reached = bool(f3_to_target <= max(0.006, float(cfg.contact_refinement_pos_tol_m) * 2.0))
                reached_pose_count += int(probe_reached)
                min_f3_distance = min(min_f3_distance, f3_to_target)
                min_f4_distance = min(min_f4_distance, f4_to_target)
                f3_force = float(next_state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
                f4_force = float(next_state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
                total_target_force = f3_force + f4_force
                max_target_force = max(f3_force, f4_force)
                active_target = float(next_state.get("active_target_filtered_force_peak_n", 0.0) or 0.0)
                active_unfiltered = float(next_state.get("active_unfiltered_force_peak_n", 0.0) or 0.0)
                object_local = list(next_state.get("object_local_pos", [0.0, 0.0, 0.0]))
                object_motion = _distance(refinement_start_object, object_local)
                object_z_delta = _vec_delta_z(refinement_start_object, object_local)
                probe_clamped = _clamp_or_barrier(next_state, threshold_m=float(cfg.reanchor_z_clamp_threshold_m))
                clamp_seen = clamp_seen or probe_clamped
                clamped_probe_count += int(probe_clamped)
                candidate = {
                    "probe_id": steps,
                    "pass_name": pass_name,
                    "center_id": center_id,
                    "interpolation_factor": center_interpolation,
                    "center_position_offset_xyz_m": list(center_position),
                    "center_orientation_rpy_deg": [float(center_rpy[0]), float(center_rpy[1]), float(center_rpy[2])],
                    "local_position_offset_xyz_m": list(local_offset),
                    "total_position_offset_xyz_m": list(total_offset),
                    "local_orientation_offset_rpy_deg": [float(local_rpy[0]), float(local_rpy[1]), float(local_rpy[2])],
                    "total_orientation_rpy_deg": [float(total_rpy[0]), float(total_rpy[1]), float(total_rpy[2])],
                    "finger3_force_n": f3_force,
                    "finger4_force_n": f4_force,
                    "total_target_force_n": total_target_force,
                    "max_target_force_n": max_target_force,
                    "object_displacement_from_refinement_start_m": object_motion,
                    "object_z_delta_from_refinement_start_m": object_z_delta,
                    "probe_reached": probe_reached,
                    "probe_clamped": probe_clamped,
                }
                flags = _contact_refinement_force_flags(candidate, cfg)
                extra = {
                    "approach_stage": "two_seed_contact_refinement",
                    "search_step": steps,
                    "search_mode": "two_seed_contact_refinement",
                    "search_probe_id": steps,
                    "search_position_id": position_id,
                    "search_orientation_id": orientation_id,
                    "search_lane_id": f"{center_id}:{position_id}",
                    "contact_refinement_step": steps,
                    "contact_refinement_pass": pass_name,
                    "contact_refinement_center_id": center_id,
                    "contact_refinement_interpolation_factor": center_interpolation,
                    "contact_refinement_center_position_offset_xyz_m": list(center_position),
                    "contact_refinement_center_orientation_rpy_deg": [float(center_rpy[0]), float(center_rpy[1]), float(center_rpy[2])],
                    "contact_refinement_seed_a_position_offset_xyz_m": seed_a_offset,
                    "contact_refinement_seed_a_orientation_rpy_deg": list(seed_a_rpy),
                    "contact_refinement_seed_b_position_offset_xyz_m": seed_b_offset,
                    "contact_refinement_seed_b_orientation_rpy_deg": list(seed_b_rpy),
                    "contact_refinement_seed_position_offset_xyz_m": list(center_position),
                    "contact_refinement_seed_orientation_rpy_deg": [float(center_rpy[0]), float(center_rpy[1]), float(center_rpy[2])],
                    "contact_refinement_local_position_offset_xyz_m": list(local_offset),
                    "contact_refinement_total_position_offset_xyz_m": list(total_offset),
                    "contact_refinement_local_orientation_offset_rpy_deg": [float(local_rpy[0]), float(local_rpy[1]), float(local_rpy[2])],
                    "contact_refinement_total_orientation_rpy_deg": [float(total_rpy[0]), float(total_rpy[1]), float(total_rpy[2])],
                    "contact_refinement_target_palm_local_xyz": desired_palm,
                    "contact_refinement_target_finger3_local_xyz": target,
                    "contact_refinement_total_target_force_n": total_target_force,
                    "contact_refinement_max_target_force_n": max_target_force,
                    "contact_refinement_finger3_contact": bool(flags["finger3_contact"]),
                    "contact_refinement_finger4_contact": bool(flags["finger4_contact"]),
                    "contact_refinement_two_finger_contact": bool(flags["two_finger_contact"]),
                    "contact_refinement_force_level_reasonable": bool(flags["force_level_reasonable"]),
                    "contact_refinement_object_motion_stable": bool(flags["object_motion_stable"]),
                    "contact_refinement_stable_two_finger_contact": bool(flags["stable_two_finger_contact"]),
                    "two_seed_refinement_center_id": center_id,
                    "two_seed_refinement_interpolation_factor": center_interpolation,
                    "two_seed_refinement_center_position_offset_xyz_m": list(center_position),
                    "two_seed_refinement_center_orientation_rpy_deg": [float(center_rpy[0]), float(center_rpy[1]), float(center_rpy[2])],
                    "two_seed_refinement_local_position_offset_xyz_m": list(local_offset),
                    "two_seed_refinement_total_position_offset_xyz_m": list(total_offset),
                    "two_seed_refinement_local_orientation_offset_rpy_deg": [float(local_rpy[0]), float(local_rpy[1]), float(local_rpy[2])],
                    "two_seed_refinement_total_orientation_rpy_deg": [float(total_rpy[0]), float(total_rpy[1]), float(total_rpy[2])],
                    "object_displacement_from_refinement_start_m": object_motion,
                    "object_z_delta_from_refinement_start_m": object_z_delta,
                    "search_position_offset_x_m": _list_get(total_offset, 0),
                    "search_position_offset_y_m": _list_get(total_offset, 1),
                    "search_position_offset_z_m": _list_get(total_offset, 2),
                    "search_orientation_roll_deg": float(total_rpy[0]),
                    "search_orientation_pitch_deg": float(total_rpy[1]),
                    "search_orientation_yaw_deg": float(total_rpy[2]),
                    "search_orientation_offset_norm_deg": math.sqrt(sum(float(value) * float(value) for value in total_rpy)),
                    "search_center_finger3_local_xyz": search_center_f3,
                    "search_center_palm_local_xyz": search_center_palm,
                    "search_center_screw1_root_local_xyz": search_center_root,
                    "search_target_palm_local_xyz": desired_palm,
                    "search_target_quat_wxyz": desired_quat,
                    "search_target_local_xyz": target,
                    "search_target_x": _list_get(target, 0),
                    "search_target_y": _list_get(target, 1),
                    "search_target_z": _list_get(target, 2),
                    "commanded_delta_xyz": commanded_delta,
                    "commanded_delta_x": _list_get(commanded_delta, 0),
                    "commanded_delta_y": _list_get(commanded_delta, 1),
                    "commanded_delta_z": _list_get(commanded_delta, 2),
                    "commanded_delta_norm_m": commanded_norm,
                    "contact_search_total_commanded_m": total_commanded,
                    "finger3_tip_before_x": _list_get(tip_before, 0),
                    "finger3_tip_before_y": _list_get(tip_before, 1),
                    "finger3_tip_before_z": _list_get(tip_before, 2),
                    "finger3_tip_after_x": _list_get(tip_after, 0),
                    "finger3_tip_after_y": _list_get(tip_after, 1),
                    "finger3_tip_after_z": _list_get(tip_after, 2),
                    "finger3_actual_motion_m": actual_motion,
                    "finger3_distance_to_search_target_m": f3_to_target,
                    "finger4_distance_to_search_target_m": f4_to_target,
                    "search_probe_reached": probe_reached,
                    "search_probe_position_error_m": f3_to_target,
                    "search_probe_clamped": probe_clamped,
                    "finger3_target_filtered_force_n": f3_force,
                    "finger4_target_filtered_force_n": f4_force,
                    "active_target_filtered_force_peak_n": active_target,
                    "active_unfiltered_force_peak_n": active_unfiltered,
                    "termination_reason": "",
                    "geometry_distance_used_for_control": False,
                    "geometry_distance_used_for_success": False,
                    **ik_audit,
                }
                peaks = _append_contact_search_log(
                    logs,
                    steps,
                    next_state,
                    cfg,
                    object_reset_pos,
                    extra=extra,
                    alignment=alignment,
                    base=base,
                    phase="two_seed_contact_refinement",
                )
                executed = True
                steps += 1
                candidate["object_contact_state"] = str(peaks.get("object_contact_state", "none"))
                _update_best(candidate)
                raw_tip_peak = max(raw_tip_peak, float(peaks.get("raw_tip_screw1_peak_n", 0.0) or 0.0))
                raw_any_peak = max(raw_any_peak, float(peaks.get("raw_any_screw1_peak_n", 0.0) or 0.0))
                raw_non_target_peak = max(raw_non_target_peak, float(peaks.get("raw_non_target_peak_n", 0.0) or 0.0))
                raw_unknown_peak = max(raw_unknown_peak, float(peaks.get("raw_unknown_object_peak_n", 0.0) or 0.0))
                if float(peaks.get("raw_any_screw1_peak_n", 0.0) or 0.0) >= raw_any_peak:
                    raw_peak_body = str(peaks.get("raw_screw1_peak_body", ""))
                    raw_peak_object = str(peaks.get("raw_screw1_peak_object", ""))
                last_object_contact_state = str(peaks.get("object_contact_state", "none"))
                if active_unfiltered > threshold and active_target <= threshold and raw_non_target_peak <= threshold:
                    unresolved_unfiltered_peak = max(unresolved_unfiltered_peak, active_unfiltered)
                    unresolved_unfiltered_observed = True
                if max_target_force > threshold:
                    target_contact_observed = True
                    contact_pose = {
                        "contact_search_contact_probe_id": steps - 1,
                        "contact_search_contact_position_offset_xyz_m": list(total_offset),
                        "contact_search_contact_orientation_offset_rpy_deg": [float(total_rpy[0]), float(total_rpy[1]), float(total_rpy[2])],
                        "contact_search_contact_orientation_offset_norm_deg": math.sqrt(sum(float(value) * float(value) for value in total_rpy)),
                        "contact_search_contact_target_local_xyz": target,
                        "contact_search_contact_target_quat_wxyz": desired_quat,
                        "contact_search_contact_object_state": last_object_contact_state,
                    }
                if raw_non_target_peak > threshold:
                    non_target = True
                    termination_reason = "raw_non_target_contact"
                    return
                if max_target_force > hard_force_stop:
                    termination_reason = "two_seed_refinement_force_limit"
                    return
                if bool(flags.get("stable_two_finger_contact", False)):
                    termination_reason = "stable_two_finger_contact_acquired"
                    return

    for center in refinement_centers:
        _run_refinement_pass("two_seed", coarse_positions, coarse_orientations, center=center)
        if termination_reason != "contact_refinement_completed":
            break

    if termination_reason == "max_search_steps" and steps >= planned_pose_count:
        termination_reason = "contact_refinement_completed"
    if termination_reason == "contact_refinement_completed" and not target_contact_observed:
        termination_reason = "two_seed_refinement_no_target_contact"

    if logs and str(logs[-1].get("phase", "")) == "two_seed_contact_refinement":
        logs[-1]["termination_reason"] = termination_reason
    best = best_candidate or {}
    best_flags = _contact_refinement_force_flags(best, cfg)
    best_probe = int(best.get("probe_id", -1) if best else -1)
    best_f3 = float(best.get("finger3_force_n", 0.0) or 0.0)
    best_f4 = float(best.get("finger4_force_n", 0.0) or 0.0)
    best_total_force = best_f3 + best_f4
    best_max_force = max(best_f3, best_f4)
    best_object_motion = float(best.get("object_displacement_from_refinement_start_m", 0.0) or 0.0)
    best_pose_offset = list(best.get("total_position_offset_xyz_m", []) or [])
    best_local_offset = list(best.get("local_position_offset_xyz_m", []) or [])
    best_rpy = list(best.get("total_orientation_rpy_deg", []) or [])
    best_local_rpy = list(best.get("local_orientation_offset_rpy_deg", []) or [])
    best_center_id = str(best.get("center_id", ""))
    best_center_offset = list(best.get("center_position_offset_xyz_m", []) or [])
    best_center_rpy = list(best.get("center_orientation_rpy_deg", []) or [])
    best_interp = float(best.get("interpolation_factor", 0.0) or 0.0)
    best_two = best_two_finger_candidate or {}
    best_two_flags = _contact_refinement_force_flags(best_two, cfg)
    best_two_probe = int(best_two.get("probe_id", -1) if best_two else -1)
    best_two_f3 = float(best_two.get("finger3_force_n", 0.0) or 0.0)
    best_two_f4 = float(best_two.get("finger4_force_n", 0.0) or 0.0)
    best_two_total = best_two_f3 + best_two_f4
    best_two_object_motion = float(best_two.get("object_displacement_from_refinement_start_m", 0.0) or 0.0)
    best_two_pose_offset = list(best_two.get("total_position_offset_xyz_m", []) or [])
    best_two_rpy = list(best_two.get("total_orientation_rpy_deg", []) or [])
    best_two_center_id = str(best_two.get("center_id", ""))
    best_two_interp = float(best_two.get("interpolation_factor", 0.0) or 0.0)
    stable_two_finger = bool(best_two_flags.get("stable_two_finger_contact", False))
    if bool(best_flags.get("any_target_contact", False)):
        contact_pose = {
            **contact_pose,
            "contact_search_contact_probe_id": best_probe,
            "contact_search_contact_position_offset_xyz_m": best_pose_offset,
            "contact_search_contact_orientation_offset_rpy_deg": best_rpy,
            "contact_search_contact_orientation_offset_norm_deg": math.sqrt(sum(float(value) * float(value) for value in best_rpy)) if best_rpy else 0.0,
            "contact_search_contact_object_state": str(best.get("object_contact_state", "")),
        }
    max_total_pose_norm = max(
        [
            0.0,
            *[
                _norm(_add_vec(list(center.get("position_offset_xyz_m", [0.0, 0.0, 0.0])), offset))
                for center in refinement_centers
                for offset in coarse_positions
            ],
        ]
    )
    return {
        "executed": executed,
        "stop_reason": termination_reason,
        "steps": steps,
        "contact_search_executed": executed,
        "contact_search_success": stable_two_finger,
        "contact_search_target_contact_observed": target_contact_observed,
        "contact_search_reanchored_before_search": bool(search_reanchored),
        "contact_search_zero_action_ctrl_delta_l2": float(search_zero_audit.get("ctrl_target_delta_l2", 0.0) or 0.0),
        "contact_search_steps": steps,
        "contact_search_attempted_pose_count": steps,
        "contact_search_completed_planned_sweep": bool(
            steps >= planned_pose_count
            and termination_reason in {"contact_refinement_completed", "two_seed_refinement_no_target_contact"}
        ),
        "contact_search_termination_reason": termination_reason,
        "contact_search_total_commanded_m": total_commanded,
        "contact_search_max_step_m": max_total_pose_norm,
        "contact_search_mode": "two_seed_contact_refinement",
        "contact_search_planned_pose_count": planned_pose_count,
        "contact_search_max_steps": max_steps,
        "contact_search_reached_pose_count": reached_pose_count,
        "contact_search_clamped_probe_count": clamped_probe_count,
        "contact_search_controller_no_motion_count": controller_no_motion_count,
        "contact_search_position_offsets_m": [list(offset) for offset in coarse_positions],
        "contact_search_orientation_offsets_rpy_deg": [[float(a), float(b), float(c)] for a, b, c in coarse_orientations],
        "contact_search_position_extent_xyz_m": [0.005, 0.005, 0.005],
        "contact_search_orientation_extent_rpy_deg": [10.0, 10.0, 10.0],
        "contact_search_center_finger3_local_xyz": search_center_f3,
        "contact_search_center_finger4_local_xyz": search_center_f4,
        "contact_search_center_palm_local_xyz": search_center_palm,
        "contact_search_center_screw1_root_local_xyz": search_center_root,
        "contact_search_non_target_blocked": bool(non_target),
        "contact_search_raw_contact_found": bool(raw_any_peak > threshold),
        **contact_pose,
        "contact_refinement_executed": executed,
        "contact_refinement_probe_count": steps,
        "contact_refinement_coarse_probe_count": planned_pose_count,
        "contact_refinement_fine_pass_executed": fine_pass_executed,
        "contact_refinement_fine_probe_count": 0,
        "contact_refinement_termination_reason": termination_reason,
        "contact_refinement_seed_position_offset_xyz_m": midpoint_offset,
        "contact_refinement_seed_orientation_rpy_deg": list(midpoint_rpy),
        "contact_refinement_seed_a_position_offset_xyz_m": seed_a_offset,
        "contact_refinement_seed_a_orientation_rpy_deg": list(seed_a_rpy),
        "contact_refinement_seed_b_position_offset_xyz_m": seed_b_offset,
        "contact_refinement_seed_b_orientation_rpy_deg": list(seed_b_rpy),
        "contact_refinement_interpolation_centers": refinement_centers,
        "contact_refinement_best_probe_id": best_probe,
        "contact_refinement_best_pass": str(best.get("pass_name", "")),
        "contact_refinement_best_center_id": best_center_id,
        "contact_refinement_best_interpolation_factor": best_interp,
        "contact_refinement_best_center_position_offset_xyz_m": best_center_offset,
        "contact_refinement_best_center_orientation_rpy_deg": best_center_rpy,
        "contact_refinement_best_pose_offset_xyz_m": best_pose_offset,
        "contact_refinement_best_local_position_offset_xyz_m": best_local_offset,
        "contact_refinement_best_orientation_rpy_deg": best_rpy,
        "contact_refinement_best_local_orientation_offset_rpy_deg": best_local_rpy,
        "contact_refinement_best_finger3_force_n": best_f3,
        "contact_refinement_best_finger4_force_n": best_f4,
        "contact_refinement_best_total_target_force_n": best_total_force,
        "contact_refinement_best_max_target_force_n": best_max_force,
        "contact_refinement_best_finger3_contact": bool(best_flags.get("finger3_contact", False)),
        "contact_refinement_best_finger4_contact": bool(best_flags.get("finger4_contact", False)),
        "contact_refinement_any_target_contact": bool(best_flags.get("any_target_contact", False)),
        "two_finger_contact_acquired": bool(best_two_flags.get("two_finger_contact", False)),
        "force_level_reasonable": bool(best_flags.get("force_level_reasonable", False)),
        "best_contact_object_displacement_m": best_object_motion,
        "best_contact_object_motion_stable": bool(best_flags.get("object_motion_stable", False)),
        "contact_refinement_reasonable_force_min_n": threshold,
        "contact_refinement_reasonable_force_max_n": float(cfg.contact_refinement_reasonable_force_max_n),
        "contact_refinement_reasonable_total_force_min_n": float(cfg.contact_refinement_reasonable_total_force_min_n),
        "contact_refinement_reasonable_total_force_max_n": float(cfg.contact_refinement_reasonable_total_force_max_n),
        "contact_refinement_hard_force_stop_n": hard_force_stop,
        "contact_refinement_stable_object_motion_limit_m": float(cfg.contact_refinement_stable_object_motion_limit_m),
        "two_seed_contact_refinement_executed": executed,
        "two_seed_contact_refinement_probe_count": steps,
        "two_seed_contact_refinement_termination_reason": termination_reason,
        "two_seed_contact_refinement_seed_a_position_offset_xyz_m": seed_a_offset,
        "two_seed_contact_refinement_seed_a_orientation_rpy_deg": list(seed_a_rpy),
        "two_seed_contact_refinement_seed_b_position_offset_xyz_m": seed_b_offset,
        "two_seed_contact_refinement_seed_b_orientation_rpy_deg": list(seed_b_rpy),
        "two_seed_contact_refinement_interpolation_centers": refinement_centers,
        "two_seed_contact_refinement_position_offsets_m": [list(offset) for offset in coarse_positions],
        "two_seed_contact_refinement_orientation_offsets_rpy_deg": [
            [float(a), float(b), float(c)] for a, b, c in coarse_orientations
        ],
        "two_seed_contact_refinement_search_position_extent_xyz_m": [0.005, 0.005, 0.005],
        "two_seed_contact_refinement_search_orientation_extent_rpy_deg": [10.0, 10.0, 10.0],
        "two_seed_contact_refinement_best_two_finger_probe_id": best_two_probe,
        "two_seed_contact_refinement_best_two_finger_center_id": best_two_center_id,
        "two_seed_contact_refinement_best_two_finger_interpolation_factor": best_two_interp,
        "two_seed_contact_refinement_best_two_finger_pose_offset_xyz_m": best_two_pose_offset,
        "two_seed_contact_refinement_best_two_finger_orientation_rpy_deg": best_two_rpy,
        "two_seed_contact_refinement_best_two_finger_finger3_force_n": best_two_f3,
        "two_seed_contact_refinement_best_two_finger_finger4_force_n": best_two_f4,
        "two_seed_contact_refinement_best_two_finger_total_target_force_n": best_two_total,
        "two_seed_contact_refinement_best_two_finger_object_displacement_m": best_two_object_motion,
        "two_seed_contact_refinement_best_two_finger_stable": stable_two_finger,
        "two_seed_contact_refinement_best_overall_probe_id": best_probe,
        "two_seed_contact_refinement_best_overall_center_id": best_center_id,
        "two_seed_contact_refinement_best_overall_interpolation_factor": best_interp,
        "two_seed_contact_refinement_best_overall_pose_offset_xyz_m": best_pose_offset,
        "two_seed_contact_refinement_best_overall_orientation_rpy_deg": best_rpy,
        "two_seed_contact_refinement_best_overall_finger3_force_n": best_f3,
        "two_seed_contact_refinement_best_overall_finger4_force_n": best_f4,
        "two_seed_contact_refinement_best_overall_total_target_force_n": best_total_force,
        "two_seed_contact_refinement_best_overall_object_displacement_m": best_object_motion,
        "stable_two_finger_contact_acquired": stable_two_finger,
        "raw_tip_screw1_peak_n": raw_tip_peak,
        "raw_any_screw1_peak_n": raw_any_peak,
        "raw_non_target_peak_n": raw_non_target_peak,
        "raw_unknown_object_peak_n": raw_unknown_peak,
        "unresolved_unfiltered_contact_observed": unresolved_unfiltered_observed,
        "unresolved_unfiltered_force_peak_n": unresolved_unfiltered_peak,
        "raw_screw1_peak_body": raw_peak_body,
        "raw_screw1_peak_object": raw_peak_object,
        "object_contact_state": last_object_contact_state,
        "non_target_contact_before_target": non_target,
        "workspace_or_table_clamp_observed": clamp_seen,
        "contact_search_any_probe_clamped": clamp_seen,
        "start_finger3_distance_to_contact_target": start_f3_distance,
        "start_finger4_distance_to_contact_target": start_f4_distance,
        "min_finger3_distance_to_contact_target": min_f3_distance,
        "min_finger4_distance_to_contact_target": min_f4_distance,
        "min_finger3_distance_to_precontact": min_f3_distance,
        "min_finger3_distance_to_real_surface": "",
        "min_finger4_distance_to_real_surface": "",
        "finger3_tip_at_min_surface_distance": [],
        "finger4_tip_at_min_surface_distance": [],
        "press_depth_attempted": 0.0,
        "press_stage_entered": False,
        "press_start_reason": "not_used_contact_driven_refinement",
        "press_depth_commanded": 0.0,
        "press_depth_actual": 0.0,
        "press_direction_flipped": False,
        "press_surface_progress": False,
        "press_control_mode_used": "not_used_contact_driven_refinement",
        "press_policy_action_norm_peak": 0.0,
        "press_mapped_isaac_action_norm_peak": 0.0,
        "press_ctrl_target_delta_peak": 0.0,
        "press_ctrl_target_moved": False,
        "press_finger3_delta_dot_press_dir": 0.0,
        "press_finger3_delta_dot_press_dir_peak": 0.0,
        "press_z_clamp_before_press": 0.0,
        "press_z_clamp_after_reanchor": 0.0,
        "press_z_clamp_after_z_raise": 0.0,
        "workspace_clamp_max_during_press": 0.0,
        "press_blocked_by_z_clamp": False,
        "press_action_not_reaching_controller": False,
        "press_action_frame_wrong": False,
        "press_controller_target_moved_tip_static": False,
        "press_target_z_raised_after_preflight": False,
        "press_target_minus_tip_dot_press_dir": 0.0,
        "surface_point_minus_tip_dot_press_dir": 0.0,
        "fallback_triggered": False,
        "fallback_reason": "",
        "press_start_surface_distance": 0.0,
        "press_min_surface_distance": 0.0,
        "press_final_surface_distance": 0.0,
        "press_target_surface_distance": float(cfg.press_target_surface_distance_m),
        "total_press_limit_m": 0.0,
        "per_step_press_m": 0.0,
        "surface_distance_reduction_during_press": 0.0,
        "action_probe_surface_reduction": 0.0,
        "ik_surface_reduction": 0.0,
        "press_dir_surface_dir_dot_last": "",
        "press_dir_surface_dir_dot_min": "",
        "press_dir_surface_dir_dot_mean": "",
        "distance_reduced_during_approach": bool(total_commanded > 0.0),
        "distance_reduction_m": total_commanded,
        "gain_increased": False,
        "near_surface_reached": False,
        "precontact_reached": False,
        "target_contact_observed": target_contact_observed,
        "clamp_adjusted": False,
        "clamp_adjustment_type": "",
        "clamp_adjustment_step": -1,
    }


def _run_sanity_action_check(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None = None,
) -> dict[str, Any]:
    before = read_state(base, env_index, cfg.part, cfg.force_threshold_n)
    direction = _unit(_sub_vec(before["object_local_pos"], before["finger3_tip_local_pos"]))
    for step in range(max(1, int(cfg.sanity_steps))):
        action = [0.0] * 16
        for axis in range(3):
            action[axis] = direction[axis] * float(cfg.sanity_action_gain)
        if alignment is not None:
            alignment.update_markers(base, before)
        audit = _step_policy(env, mapper, env_index, action, alignment=alignment)
        state = read_state(base, env_index, cfg.part, cfg.force_threshold_n)
        _append_log(logs, "sanity", step, state, cfg, object_reset_pos, extra=audit, alignment=alignment, base=base)
    after = read_state(base, env_index, cfg.part, cfg.force_threshold_n)
    movement = _sub_vec(after["finger3_tip_local_pos"], before["finger3_tip_local_pos"])
    dot = _dot(movement, direction)
    norm = _norm(movement)
    return {
        "tip_movement_norm_m": norm,
        "movement_dot_direction_m": dot,
        "hand_moved_toward_object": bool(norm > 1.0e-5 and dot > 0.0),
    }


def _run_approach(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None = None,
    *,
    adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stop_reason = "max_steps_reached"
    non_target = False
    clamp_seen = False
    gain_increased = False
    clamp_adjusted = False
    clamp_adjustment_type = ""
    clamp_adjustment_step = -1
    blocked_axis = ""
    contact_adjustment: dict[str, Any] = dict(adjustment or {})
    contact_adjustment["target_mode"] = "precontact"
    current_gain = float(cfg.approach_action_gain)
    start_state = _state_with_contact_target(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        adjustment=contact_adjustment,
    )
    start_f3_distance = float(start_state.get("finger3_distance_to_contact_target_m", 0.0))
    start_f4_distance = float(start_state.get("finger4_distance_to_contact_target_m", 0.0))
    min_f3_distance = start_f3_distance
    min_f4_distance = start_f4_distance
    min_f3_precontact_distance = start_f3_distance
    min_f3_surface_distance = float(start_state.get("finger3_distance_to_real_surface_m", math.inf))
    min_f4_surface_distance = float(start_state.get("finger4_distance_to_real_surface_m", math.inf))
    finger3_tip_at_min_surface_distance = list(start_state.get("finger3_tip_local_pos", []))
    finger4_tip_at_min_surface_distance = list(start_state.get("finger4_tip_local_pos", []))
    near_surface_reached = bool(min_f3_surface_distance <= cfg.real_surface_micro_close_distance_m)
    target_contact_observed = False
    precontact_reached = False
    press_stage_entered = False
    press_start_reason = ""
    press_depth_attempted = 0.0
    press_depth_commanded = 0.0
    press_depth_actual = 0.0
    press_direction_flipped = False
    press_surface_progress = False
    press_control_mode_used = "not_entered"
    press_policy_action_norm_peak = 0.0
    press_mapped_isaac_action_norm_peak = 0.0
    press_ctrl_target_delta_peak = 0.0
    press_ctrl_target_moved = False
    press_cumulative_tip_progress = 0.0
    press_finger3_delta_dot_press_dir_peak = 0.0
    press_z_clamp_before_press = 0.0
    press_z_clamp_after_reanchor = 0.0
    press_z_clamp_after_z_raise = 0.0
    workspace_clamp_max_during_press = 0.0
    press_blocked_by_z_clamp = False
    press_action_not_reaching_controller = False
    press_action_frame_wrong = False
    press_controller_target_moved_tip_static = False
    press_target_z_raised_after_preflight = False
    press_target_minus_tip_dot_press_dir = 0.0
    surface_point_minus_tip_dot_press_dir = 0.0
    press_negative_motion_count = 0
    fallback_triggered = False
    fallback_reason = ""
    press_start_surface_distance = float(start_state.get("finger3_surface_distance_servo_m", min_f3_surface_distance))
    press_min_surface_distance = press_start_surface_distance
    press_final_surface_distance = press_start_surface_distance
    surface_distance_reduction_during_press = 0.0
    action_probe_surface_reduction = 0.0
    ik_surface_reduction = 0.0
    press_dir_surface_dir_dots: list[float] = []
    surface_lock: dict[str, Any] | None = None
    steps = 0

    def _record_metrics(next_state: dict[str, Any], next_f3_distance: float, next_f4_distance: float) -> None:
        nonlocal min_f3_distance, min_f4_distance, min_f3_surface_distance, min_f4_surface_distance
        nonlocal finger3_tip_at_min_surface_distance, finger4_tip_at_min_surface_distance
        nonlocal press_depth_actual, near_surface_reached
        min_f3_distance = min(min_f3_distance, next_f3_distance)
        min_f4_distance = min(min_f4_distance, next_f4_distance)
        f3_surface = float(next_state.get("finger3_distance_to_real_surface_m", math.inf))
        f4_surface = float(next_state.get("finger4_distance_to_real_surface_m", math.inf))
        if f3_surface < min_f3_surface_distance:
            min_f3_surface_distance = f3_surface
            finger3_tip_at_min_surface_distance = list(next_state.get("finger3_tip_local_pos", []))
        if f4_surface < min_f4_surface_distance:
            min_f4_surface_distance = f4_surface
            finger4_tip_at_min_surface_distance = list(next_state.get("finger4_tip_local_pos", []))
        press_depth_actual = max(press_depth_actual, float(next_state.get("finger3_surface_press_depth_m", 0.0) or 0.0))
        near_surface_reached = near_surface_reached or bool(f3_surface <= cfg.real_surface_micro_close_distance_m)

    precontact_budget = max(1, int(cfg.max_approach_steps) - max(0, int(cfg.max_micro_press_steps)))
    for step in range(precontact_budget):
        state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=contact_adjustment,
        )
        target = state["contact_target_local_pos"]
        delta = _sub_vec(target, state["finger3_tip_local_pos"])
        distance_before = _norm(delta)
        direction = _project_direction_for_clamp(_unit(delta), blocked_axis)
        action = [0.0] * 16
        for axis in range(3):
            action[axis] = direction[axis] * current_gain
        if alignment is not None:
            alignment.update_markers(base, state, contact_target_local=target)
        audit = _step_policy(env, mapper, env_index, action, alignment=alignment)
        next_state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=contact_adjustment,
        )
        next_f3_distance = float(next_state.get("finger3_distance_to_contact_target_m", distance_before))
        next_f4_distance = float(next_state.get("finger4_distance_to_contact_target_m", start_f4_distance))
        min_f3_precontact_distance = min(min_f3_precontact_distance, next_f3_distance)
        movement_toward_target = distance_before - next_f3_distance
        _record_metrics(next_state, next_f3_distance, next_f4_distance)
        clamp_seen = clamp_seen or _clamp_or_barrier(next_state)
        active_target = float(next_state["active_target_filtered_force_peak_n"])
        active_unfiltered = float(next_state["active_unfiltered_force_peak_n"])
        target_contact_observed = target_contact_observed or bool(active_target > cfg.force_threshold_n)
        non_target = non_target or bool(active_unfiltered > cfg.force_threshold_n and active_target <= cfg.force_threshold_n)
        _append_log(
            logs,
            "precontact",
            step,
            next_state,
            cfg,
            object_reset_pos,
            extra={
                "approach_target_local_pos": target,
                "tip3_to_contact_target_distance_m": next_f3_distance,
                "tip4_to_contact_target_distance_m": next_f4_distance,
                "movement_toward_target_m": movement_toward_target,
                "approach_gain_used": current_gain,
                "approach_gain_increased": gain_increased,
                "clamp_adjusted": clamp_adjusted,
                "clamp_adjustment_type": clamp_adjustment_type,
                "blocked_axis_after_clamp": blocked_axis,
                "approach_stage": "precontact",
                "approach_direction_xyz": direction,
                **audit,
            },
            alignment=alignment,
            base=base,
        )
        steps = step + 1
        if active_target > cfg.force_threshold_n:
            stop_reason = "target_filtered_contact_observed"
            break
        if non_target:
            stop_reason = "non_target_contact_before_target"
            break
        if next_f3_distance <= cfg.precontact_reached_distance_m:
            precontact_reached = True
            press_start_reason = f"finger3_precontact_distance<={float(cfg.precontact_reached_distance_m):.4f}m"
            surface_lock = _surface_lock_from_state(next_state)
            stop_reason = "precontact_reached"
            break
        current_clamped = _clamp_or_barrier(next_state)
        if current_clamped:
            if not clamp_adjusted:
                adjustment = _clamp_aware_target_adjustment(next_state, cfg)
                if adjustment:
                    contact_adjustment.update(adjustment.get("contact_adjustment", {}))
                    contact_adjustment["target_mode"] = "precontact"
                    blocked_axis = str(adjustment.get("blocked_axis", ""))
                    clamp_adjustment_type = str(adjustment.get("adjustment_type", ""))
                    clamp_adjusted = True
                    clamp_adjustment_step = step
                    continue
            elif clamp_adjustment_step >= 0 and step - clamp_adjustment_step >= 3:
                stop_reason = "workspace_or_table_clamp_observed_after_adjustment"
                break
        reduction = start_f3_distance - min_f3_distance
        window = max(1, int(cfg.approach_stall_window_steps))
        if not gain_increased and steps >= window and reduction < float(cfg.approach_gain_reduction_threshold_m):
            current_gain = float(cfg.approach_action_gain_after_stall)
            gain_increased = True
        elif gain_increased and steps >= 2 * window and reduction < float(cfg.approach_fail_reduction_threshold_m):
            stop_reason = "approach_not_reducing_distance"
            break
    if (
        not target_contact_observed
        and not non_target
        and precontact_reached
        and int(cfg.max_micro_press_steps) > 0
    ):
        press_stage_entered = True
        press_control_mode_used = "action_servo"
        if surface_lock is None:
            locked_state = _state_with_contact_target(
                base,
                env_index,
                cfg,
                read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                adjustment=contact_adjustment,
            )
            surface_lock = _surface_lock_from_state(locked_state)
        contact_adjustment["target_mode"] = "press"
        contact_adjustment["surface_lock"] = surface_lock
        press_start_step = steps
        locked_normal = _unit(list(surface_lock.get("contact_target_normal_xyz", [1.0, 0.0, 0.0]) or [1.0, 0.0, 0.0]))
        if _norm(locked_normal) <= 1.0e-6:
            locked_normal = [1.0, 0.0, 0.0]
        press_dir = _unit([-locked_normal[0], -locked_normal[1], -locked_normal[2]])
        if _norm(press_dir) <= 1.0e-6:
            press_dir = [-1.0, 0.0, 0.0]
        press_action_sign = 1.0
        state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=contact_adjustment,
        )
        press_target_minus_tip_dot_press_dir = _dot(
            _sub_vec(list(state.get("press_target_local_xyz", state["contact_target_local_pos"])), state["finger3_tip_local_pos"]),
            press_dir,
        )
        surface_point_minus_tip_dot_press_dir = _dot(
            _sub_vec(list(state.get("surface_point_local_xyz", state["contact_target_local_pos"])), state["finger3_tip_local_pos"]),
            press_dir,
        )

        press_z_clamp_before_press = _z_clamp_amount(state)
        _set_ctrl_target_to_measured_midpoint(base, env_index)
        _configure_action_control(base, env_index, mode="", allow_commanded_hand_actions=True)
        zero_audit = _step_policy(env, mapper, env_index, [0.0] * 16, alignment=alignment)
        state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=contact_adjustment,
        )
        press_z_clamp_after_reanchor = _z_clamp_amount(state)
        _append_log(
            logs,
            "micro_press_preflight",
            0,
            state,
            cfg,
            object_reset_pos,
            extra={
                "approach_stage": "micro_press_preflight",
                "press_dir_world_x": _list_get(press_dir, 0),
                "press_dir_world_y": _list_get(press_dir, 1),
                "press_dir_world_z": _list_get(press_dir, 2),
                "press_dir_local_x": _list_get(press_dir, 0),
                "press_dir_local_y": _list_get(press_dir, 1),
                "press_dir_local_z": _list_get(press_dir, 2),
                "press_target_minus_tip_dot_press_dir": press_target_minus_tip_dot_press_dir,
                "surface_point_minus_tip_dot_press_dir": surface_point_minus_tip_dot_press_dir,
                "z_clamp_before_press": press_z_clamp_before_press,
                "z_clamp_after_reanchor": press_z_clamp_after_reanchor,
                **zero_audit,
            },
            alignment=alignment,
            base=base,
        )
        if press_z_clamp_after_reanchor > float(cfg.press_z_clamp_block_threshold_m):
            press_target_z_raised_after_preflight = True
            surface_lock = _raise_surface_lock_z(surface_lock, float(cfg.press_z_raise_m))
            contact_adjustment["surface_lock"] = surface_lock
            desired_palm = list(state.get("palm_local_pos", [0.0, 0.0, 0.0]))
            if len(desired_palm) >= 3:
                desired_palm[2] += float(cfg.press_z_raise_m)
            ik_raise = _move_ctrl_target_by_ik(
                base,
                env_index,
                desired_palm,
                max_time=float(cfg.press_ik_fallback_max_time_s),
                pos_tol=float(cfg.press_ik_fallback_pos_tol_m),
            )
            _set_ctrl_target_to_measured_midpoint(base, env_index)
            _configure_action_control(base, env_index, mode="", allow_commanded_hand_actions=True)
            zero_raise_audit = _step_policy(env, mapper, env_index, [0.0] * 16, alignment=alignment)
            state = _state_with_contact_target(
                base,
                env_index,
                cfg,
                read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                adjustment=contact_adjustment,
            )
            press_z_clamp_after_z_raise = _z_clamp_amount(state)
            _append_log(
                logs,
                "micro_press_preflight",
                1,
                state,
                cfg,
                object_reset_pos,
                extra={
                    "approach_stage": "micro_press_preflight_z_raise",
                    "press_target_z_raised_after_preflight": press_target_z_raised_after_preflight,
                    "z_clamp_before_press": press_z_clamp_before_press,
                    "z_clamp_after_reanchor": press_z_clamp_after_reanchor,
                    "z_clamp_after_z_raise": press_z_clamp_after_z_raise,
                    **ik_raise,
                    **zero_raise_audit,
                },
                alignment=alignment,
                base=base,
            )
        else:
            press_z_clamp_after_z_raise = press_z_clamp_after_reanchor

        if press_z_clamp_after_z_raise > float(cfg.press_z_clamp_block_threshold_m):
            press_blocked_by_z_clamp = True
            clamp_seen = True
            stop_reason = "press_blocked_by_z_clamp"
        else:
            probe_steps = max(1, min(int(cfg.max_micro_press_steps), int(cfg.press_action_servo_probe_steps)))
            state = _state_with_contact_target(
                base,
                env_index,
                cfg,
                read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                adjustment=contact_adjustment,
            )
            press_start_surface_distance = float(
                state.get("finger3_surface_distance_servo_m", state.get("finger3_distance_to_real_surface_m", math.inf))
            )
            press_min_surface_distance = press_start_surface_distance
            press_final_surface_distance = press_start_surface_distance
            surface_distance_reduction_during_press = 0.0
            action_probe_surface_reduction = 0.0
            press_depth_actual = 0.0
            action_probe_start_surface_distance = press_start_surface_distance
            for press_step in range(probe_steps):
                state = _state_with_contact_target(
                    base,
                    env_index,
                    cfg,
                    read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                    adjustment=contact_adjustment,
                )
                target = list(state.get("finger3_closest_surface_point_local_xyz", state["contact_target_local_pos"]))
                direction = _unit(list(state.get("finger3_surface_direction_xyz", []) or []))
                if _norm(direction) <= 1.0e-6:
                    direction = _unit(press_dir)
                press_dir_surface_dir_dot = _dot(press_dir, direction)
                press_dir_surface_dir_dots.append(press_dir_surface_dir_dot)
                delta = _sub_vec(target, state["finger3_tip_local_pos"])
                distance_before = _norm(delta)
                surface_gap_before = float(
                    state.get("finger3_surface_distance_servo_m", state.get("finger3_distance_to_real_surface_m", math.inf))
                )
                tip_before = list(state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
                action = _policy_action_for_cartesian_step(
                    base,
                    [direction[axis] * float(cfg.micro_press_step_m) for axis in range(3)],
                )
                press_depth_commanded += float(cfg.micro_press_step_m)
                press_depth_attempted = press_depth_commanded
                if alignment is not None:
                    alignment.update_markers(base, state, contact_target_local=target)
                audit = _step_policy(env, mapper, env_index, action, alignment=alignment)
                press_policy_action_norm_peak = max(
                    press_policy_action_norm_peak,
                    float(audit.get("commanded_policy_xyz_norm", 0.0) or 0.0),
                )
                press_mapped_isaac_action_norm_peak = max(
                    press_mapped_isaac_action_norm_peak,
                    float(audit.get("mapped_isaac_action_xyz_norm", 0.0) or 0.0),
                )
                press_ctrl_target_delta_peak = max(
                    press_ctrl_target_delta_peak,
                    float(audit.get("ctrl_target_delta_l2", 0.0) or 0.0),
                )
                press_ctrl_target_moved = press_ctrl_target_moved or bool(audit.get("ctrl_target_moved"))
                next_state = _state_with_contact_target(
                    base,
                    env_index,
                    cfg,
                    read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                    adjustment=contact_adjustment,
                )
                next_f3_distance = float(next_state.get("finger3_distance_to_contact_target_m", distance_before))
                next_f4_distance = float(next_state.get("finger4_distance_to_contact_target_m", start_f4_distance))
                min_f3_precontact_distance = min(
                    min_f3_precontact_distance,
                    _distance(next_state["finger3_tip_local_pos"], next_state.get("precontact_point_local_xyz", target)),
                )
                movement_toward_target = distance_before - next_f3_distance
                _record_metrics(next_state, next_f3_distance, next_f4_distance)
                surface_gap_after = float(
                    next_state.get("finger3_surface_distance_servo_m", next_state.get("finger3_distance_to_real_surface_m", math.inf))
                )
                press_final_surface_distance = surface_gap_after
                if math.isfinite(surface_gap_after):
                    press_min_surface_distance = min(press_min_surface_distance, surface_gap_after)
                surface_distance_reduction_during_press = max(
                    0.0,
                    press_start_surface_distance - press_min_surface_distance,
                )
                action_probe_surface_reduction = max(
                    0.0,
                    action_probe_start_surface_distance - press_min_surface_distance,
                )
                tip_after = list(next_state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
                finger3_delta = _sub_vec(tip_after, tip_before)
                press_motion_actual = _dot(finger3_delta, direction)
                finger3_delta_dot_press_dir = _dot(finger3_delta, press_dir)
                press_finger3_delta_dot_press_dir_peak = max(press_finger3_delta_dot_press_dir_peak, finger3_delta_dot_press_dir)
                if (
                    press_motion_actual < -1.0e-5
                    and math.isfinite(surface_gap_before)
                    and math.isfinite(surface_gap_after)
                    and surface_gap_after > surface_gap_before + 1.0e-5
                ):
                    press_negative_motion_count += 1
                else:
                    press_negative_motion_count = 0
                press_cumulative_tip_progress += max(0.0, press_motion_actual)
                press_depth_actual = max(
                    press_depth_actual,
                    surface_distance_reduction_during_press,
                    float(next_state.get("finger3_surface_press_depth_m", 0.0) or 0.0),
                )
                press_surface_progress = bool(
                    press_surface_progress
                    or press_depth_actual >= 1.0e-5
                    or surface_distance_reduction_during_press >= 1.0e-5
                )
                clamp_seen = clamp_seen or _clamp_or_barrier(next_state)
                workspace_clamp_max_during_press = max(
                    workspace_clamp_max_during_press,
                    abs(float(next_state.get("workspace_clamp_delta_m", 0.0) or 0.0)),
                )
                if _z_clamp_amount(next_state) > float(cfg.press_z_clamp_block_threshold_m):
                    press_blocked_by_z_clamp = True
                active_target = float(next_state["finger3_target_filtered_force_n"])
                active_unfiltered = float(next_state["finger3_unfiltered_force_n"])
                target_contact_observed = target_contact_observed or bool(active_target > cfg.force_threshold_n)
                non_target = non_target or bool(active_unfiltered > cfg.force_threshold_n and active_target <= cfg.force_threshold_n)
                _append_log(
                    logs,
                    "micro_press",
                    press_step,
                    next_state,
                    cfg,
                    object_reset_pos,
                    extra={
                        "approach_target_local_pos": target,
                        "tip3_to_contact_target_distance_m": next_f3_distance,
                        "tip4_to_contact_target_distance_m": next_f4_distance,
                        "movement_toward_target_m": movement_toward_target,
                        "approach_gain_used": cfg.micro_press_action_gain,
                        "approach_gain_increased": gain_increased,
                        "clamp_adjusted": clamp_adjusted,
                        "clamp_adjustment_type": clamp_adjustment_type,
                        "blocked_axis_after_clamp": blocked_axis,
                        "approach_stage": "micro_press_action_servo",
                        "press_control_mode_used": press_control_mode_used,
                        "micro_press_step_index": press_step,
                        "micro_press_step_depth_cmd_m": float(cfg.micro_press_step_m),
                        "press_depth_attempted_m": press_depth_attempted,
                        "press_depth_commanded_m": press_depth_commanded,
                        "press_depth_actual_m": max(0.0, press_depth_actual),
                        "press_direction_flipped": press_direction_flipped,
                        "press_action_sign": press_action_sign,
                        "press_dir_world_x": _list_get(press_dir, 0),
                        "press_dir_world_y": _list_get(press_dir, 1),
                        "press_dir_world_z": _list_get(press_dir, 2),
                        "press_dir_local_x": _list_get(press_dir, 0),
                        "press_dir_local_y": _list_get(press_dir, 1),
                        "press_dir_local_z": _list_get(press_dir, 2),
                        "press_target_minus_tip_dot_press_dir": press_target_minus_tip_dot_press_dir,
                        "surface_point_minus_tip_dot_press_dir": surface_point_minus_tip_dot_press_dir,
                        "finger3_tip_before_x": _list_get(tip_before, 0),
                        "finger3_tip_before_y": _list_get(tip_before, 1),
                        "finger3_tip_before_z": _list_get(tip_before, 2),
                        "finger3_tip_after_x": _list_get(tip_after, 0),
                        "finger3_tip_after_y": _list_get(tip_after, 1),
                        "finger3_tip_after_z": _list_get(tip_after, 2),
                        "finger3_tip_delta_x": _list_get(finger3_delta, 0),
                        "finger3_tip_delta_y": _list_get(finger3_delta, 1),
                        "finger3_tip_delta_z": _list_get(finger3_delta, 2),
                        "finger3_delta_dot_press_dir": finger3_delta_dot_press_dir,
                        "finger3_delta_dot_surface_direction": press_motion_actual,
                        "finger3_delta_dot_press_dir_cumulative": press_cumulative_tip_progress,
                        "finger3_surface_distance_before_m": surface_gap_before,
                        "finger3_surface_distance_after_m": surface_gap_after,
                        "surface_distance_before_m": surface_gap_before,
                        "surface_distance_after_m": surface_gap_after,
                        "surface_distance_reduction_m": max(0.0, surface_gap_before - surface_gap_after),
                        "surface_distance_reduction_during_press": surface_distance_reduction_during_press,
                        "total_press_limit_m": float(cfg.press_ik_fallback_max_total_m),
                        "per_step_press_m": float(cfg.press_ik_fallback_step_m),
                        "target_surface_distance_m": float(cfg.press_target_surface_distance_m),
                        "closest_surface_point_x": _list_get(target, 0),
                        "closest_surface_point_y": _list_get(target, 1),
                        "closest_surface_point_z": _list_get(target, 2),
                        "surface_direction_x": _list_get(direction, 0),
                        "surface_direction_y": _list_get(direction, 1),
                        "surface_direction_z": _list_get(direction, 2),
                        "press_dir_surface_dir_dot": press_dir_surface_dir_dot,
                        "press_motion_actual_m": press_motion_actual,
                        "finger3_distance_to_surface_m": surface_gap_after,
                        "finger3_screw1_target_filtered_force_n": active_target,
                        "approach_direction_xyz": direction,
                        "z_clamp_before_press": press_z_clamp_before_press,
                        "z_clamp_after_reanchor": press_z_clamp_after_reanchor,
                        "z_clamp_after_z_raise": press_z_clamp_after_z_raise,
                        "press_blocked_by_z_clamp": press_blocked_by_z_clamp,
                        "press_action_frame_wrong": press_action_frame_wrong,
                        **audit,
                    },
                    alignment=alignment,
                    base=base,
                )
                steps = press_start_step + press_step + 1
                if active_target > cfg.force_threshold_n:
                    stop_reason = "target_filtered_contact_observed"
                    break
                if non_target:
                    stop_reason = "non_target_contact_before_target"
                    break
                if press_blocked_by_z_clamp:
                    stop_reason = "press_blocked_by_z_clamp"
                    break

            if not target_contact_observed and not non_target and not press_blocked_by_z_clamp:
                press_action_not_reaching_controller = bool(
                    press_policy_action_norm_peak <= 1.0e-8
                    or press_mapped_isaac_action_norm_peak <= 1.0e-8
                )
                commanded_for_efficiency = max(1.0e-9, press_depth_commanded)
                press_efficiency = action_probe_surface_reduction / commanded_for_efficiency
                fallback_reasons: list[str] = []
                if action_probe_surface_reduction < float(cfg.press_surface_reduction_before_ik_m):
                    fallback_reasons.append("surface_distance_reduction_below_threshold")
                if press_efficiency < float(cfg.press_efficiency_before_ik):
                    fallback_reasons.append("press_efficiency_below_threshold")
                if press_ctrl_target_delta_peak < float(cfg.press_ctrl_delta_before_ik_m):
                    fallback_reasons.append("ctrl_target_delta_below_threshold")
                if press_min_surface_distance > float(cfg.press_target_surface_distance_m):
                    fallback_reasons.append("surface_distance_above_target_after_probe")
                if fallback_reasons and not press_action_not_reaching_controller:
                    fallback_triggered = True
                    fallback_reason = ",".join(fallback_reasons)
                    press_control_mode_used = "mixed"
                    ik_start_surface_distance = float(press_min_surface_distance)
                    max_total_press = float(cfg.press_ik_fallback_max_total_m)
                    max_ik_steps = max(0, int(cfg.press_ik_fallback_max_steps))
                    for ik_step in range(max_ik_steps):
                        if press_depth_commanded >= max_total_press:
                            break
                        state = _state_with_contact_target(
                            base,
                            env_index,
                            cfg,
                            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                            adjustment=contact_adjustment,
                        )
                        current_surface_gap = float(
                            state.get(
                                "finger3_surface_distance_servo_m",
                                state.get("finger3_distance_to_real_surface_m", math.inf),
                            )
                        )
                        press_final_surface_distance = current_surface_gap
                        if current_surface_gap <= float(cfg.press_target_surface_distance_m):
                            stop_reason = "surface_distance_close_no_force"
                            break
                        tip_before = list(state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
                        palm = list(state.get("palm_local_pos", [0.0, 0.0, 0.0]))
                        target = list(state.get("finger3_closest_surface_point_local_xyz", state["contact_target_local_pos"]))
                        direction = _unit(list(state.get("finger3_surface_direction_xyz", []) or []))
                        if _norm(direction) <= 1.0e-6:
                            direction = _unit(press_dir)
                        press_dir_surface_dir_dot = _dot(press_dir, direction)
                        press_dir_surface_dir_dots.append(press_dir_surface_dir_dot)
                        surface_gap_before = float(
                            state.get(
                                "finger3_surface_distance_servo_m",
                                state.get("finger3_distance_to_real_surface_m", math.inf),
                            )
                        )
                        step_size = min(float(cfg.press_ik_fallback_step_m), max(0.0, max_total_press - press_depth_commanded))
                        if step_size <= 1.0e-9:
                            break
                        desired_palm = [palm[axis] + direction[axis] * step_size for axis in range(3)]
                        press_depth_commanded += step_size
                        press_depth_attempted = press_depth_commanded
                        ik_audit = _move_ctrl_target_by_ik(
                            base,
                            env_index,
                            desired_palm,
                            max_time=float(cfg.press_ik_fallback_max_time_s),
                            pos_tol=float(cfg.press_ik_fallback_pos_tol_m),
                        )
                        next_state = _state_with_contact_target(
                            base,
                            env_index,
                            cfg,
                            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
                            adjustment=contact_adjustment,
                        )
                        next_f3_distance = float(next_state.get("finger3_distance_to_contact_target_m", math.inf))
                        next_f4_distance = float(next_state.get("finger4_distance_to_contact_target_m", start_f4_distance))
                        min_f3_precontact_distance = min(
                            min_f3_precontact_distance,
                            _distance(next_state["finger3_tip_local_pos"], next_state.get("precontact_point_local_xyz", target)),
                        )
                        _record_metrics(next_state, next_f3_distance, next_f4_distance)
                        surface_gap_after = float(
                            next_state.get(
                                "finger3_surface_distance_servo_m",
                                next_state.get("finger3_distance_to_real_surface_m", math.inf),
                            )
                        )
                        press_final_surface_distance = surface_gap_after
                        if math.isfinite(surface_gap_after):
                            press_min_surface_distance = min(press_min_surface_distance, surface_gap_after)
                        surface_distance_reduction_during_press = max(
                            0.0,
                            press_start_surface_distance - press_min_surface_distance,
                        )
                        ik_surface_reduction = max(0.0, ik_start_surface_distance - press_min_surface_distance)
                        tip_after = list(next_state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
                        finger3_delta = _sub_vec(tip_after, tip_before)
                        press_motion_actual = _dot(finger3_delta, direction)
                        finger3_delta_dot_press_dir = _dot(finger3_delta, press_dir)
                        press_cumulative_tip_progress += max(0.0, press_motion_actual)
                        press_finger3_delta_dot_press_dir_peak = max(
                            press_finger3_delta_dot_press_dir_peak,
                            finger3_delta_dot_press_dir,
                        )
                        press_depth_actual = max(
                            press_depth_actual,
                            surface_distance_reduction_during_press,
                            float(next_state.get("finger3_surface_press_depth_m", 0.0) or 0.0),
                        )
                        press_surface_progress = bool(
                            press_surface_progress
                            or press_depth_actual >= 1.0e-5
                            or surface_distance_reduction_during_press >= 1.0e-5
                        )
                        press_ctrl_target_delta_peak = max(
                            press_ctrl_target_delta_peak,
                            float(ik_audit.get("ik_ctrl_target_delta_l2", 0.0) or 0.0),
                        )
                        press_ctrl_target_moved = press_ctrl_target_moved or bool(
                            float(ik_audit.get("ik_ctrl_target_delta_l2", 0.0) or 0.0) > 1.0e-6
                        )
                        workspace_clamp_max_during_press = max(
                            workspace_clamp_max_during_press,
                            abs(float(next_state.get("workspace_clamp_delta_m", 0.0) or 0.0)),
                        )
                        if _z_clamp_amount(next_state) > float(cfg.press_z_clamp_block_threshold_m):
                            press_blocked_by_z_clamp = True
                        active_target = float(next_state["finger3_target_filtered_force_n"])
                        active_unfiltered = float(next_state["finger3_unfiltered_force_n"])
                        target_contact_observed = target_contact_observed or bool(active_target > cfg.force_threshold_n)
                        non_target = non_target or bool(active_unfiltered > cfg.force_threshold_n and active_target <= cfg.force_threshold_n)
                        _append_log(
                            logs,
                            "micro_press_ik",
                            ik_step,
                            next_state,
                            cfg,
                            object_reset_pos,
                            extra={
                                "approach_stage": "micro_press_ik_target_fallback",
                                "press_control_mode_used": press_control_mode_used,
                                "fallback_triggered": fallback_triggered,
                                "fallback_reason": fallback_reason,
                                "press_depth_attempted_m": press_depth_attempted,
                                "press_depth_commanded_m": press_depth_commanded,
                                "press_depth_actual_m": max(0.0, press_depth_actual),
                                "press_dir_world_x": _list_get(press_dir, 0),
                                "press_dir_world_y": _list_get(press_dir, 1),
                                "press_dir_world_z": _list_get(press_dir, 2),
                                "press_dir_local_x": _list_get(press_dir, 0),
                                "press_dir_local_y": _list_get(press_dir, 1),
                                "press_dir_local_z": _list_get(press_dir, 2),
                                "finger3_tip_before_x": _list_get(tip_before, 0),
                                "finger3_tip_before_y": _list_get(tip_before, 1),
                                "finger3_tip_before_z": _list_get(tip_before, 2),
                                "finger3_tip_after_x": _list_get(tip_after, 0),
                                "finger3_tip_after_y": _list_get(tip_after, 1),
                                "finger3_tip_after_z": _list_get(tip_after, 2),
                                "finger3_tip_delta_x": _list_get(finger3_delta, 0),
                                "finger3_tip_delta_y": _list_get(finger3_delta, 1),
                                "finger3_tip_delta_z": _list_get(finger3_delta, 2),
                                "finger3_delta_dot_press_dir": finger3_delta_dot_press_dir,
                                "finger3_delta_dot_surface_direction": press_motion_actual,
                                "finger3_delta_dot_press_dir_cumulative": press_cumulative_tip_progress,
                                "finger3_surface_distance_before_m": surface_gap_before,
                                "finger3_surface_distance_after_m": surface_gap_after,
                                "surface_distance_before_m": surface_gap_before,
                                "surface_distance_after_m": surface_gap_after,
                                "surface_distance_reduction_m": max(0.0, surface_gap_before - surface_gap_after),
                                "surface_distance_reduction_during_press": surface_distance_reduction_during_press,
                                "total_press_limit_m": float(cfg.press_ik_fallback_max_total_m),
                                "per_step_press_m": step_size,
                                "target_surface_distance_m": float(cfg.press_target_surface_distance_m),
                                "ik_surface_reduction_m": ik_surface_reduction,
                                "closest_surface_point_x": _list_get(target, 0),
                                "closest_surface_point_y": _list_get(target, 1),
                                "closest_surface_point_z": _list_get(target, 2),
                                "surface_direction_x": _list_get(direction, 0),
                                "surface_direction_y": _list_get(direction, 1),
                                "surface_direction_z": _list_get(direction, 2),
                                "press_dir_surface_dir_dot": press_dir_surface_dir_dot,
                                "finger3_screw1_target_filtered_force_n": active_target,
                                "z_clamp_before_press": press_z_clamp_before_press,
                                "z_clamp_after_reanchor": press_z_clamp_after_reanchor,
                                "z_clamp_after_z_raise": press_z_clamp_after_z_raise,
                                "press_blocked_by_z_clamp": press_blocked_by_z_clamp,
                                "target_env_action_norm": 0.0,
                                "commanded_policy_x": 0.0,
                                "commanded_policy_y": 0.0,
                                "commanded_policy_z": 0.0,
                                "commanded_policy_xyz_norm": 0.0,
                                "mapped_isaac_action_x": 0.0,
                                "mapped_isaac_action_y": 0.0,
                                "mapped_isaac_action_z": 0.0,
                                "mapped_isaac_action_xyz_norm": 0.0,
                                "ctrl_target_before_x": ik_audit.get("ik_ctrl_target_before_x", ""),
                                "ctrl_target_before_y": ik_audit.get("ik_ctrl_target_before_y", ""),
                                "ctrl_target_before_z": ik_audit.get("ik_ctrl_target_before_z", ""),
                                "ctrl_target_after_x": ik_audit.get("ik_ctrl_target_after_x", ""),
                                "ctrl_target_after_y": ik_audit.get("ik_ctrl_target_after_y", ""),
                                "ctrl_target_after_z": ik_audit.get("ik_ctrl_target_after_z", ""),
                                "ctrl_target_delta_l2": ik_audit.get("ik_ctrl_target_delta_l2", 0.0),
                                "ctrl_target_moved": bool(float(ik_audit.get("ik_ctrl_target_delta_l2", 0.0) or 0.0) > 1.0e-6),
                                **ik_audit,
                            },
                            alignment=alignment,
                            base=base,
                        )
                        steps += 1
                        if active_target > cfg.force_threshold_n:
                            stop_reason = "target_filtered_contact_observed"
                            break
                        if non_target:
                            stop_reason = "non_target_contact_before_target"
                            break
                        if press_blocked_by_z_clamp:
                            stop_reason = "press_blocked_by_z_clamp"
                            break
                        if surface_gap_after <= float(cfg.press_target_surface_distance_m):
                            stop_reason = "surface_distance_close_no_force"
                            break
            if not target_contact_observed and not non_target:
                if press_blocked_by_z_clamp:
                    stop_reason = "press_blocked_by_z_clamp"
                elif press_action_not_reaching_controller:
                    stop_reason = "press_action_not_reaching_controller"
                elif (
                    fallback_triggered
                    and ik_surface_reduction < float(cfg.press_ik_min_surface_reduction_m)
                    and press_min_surface_distance > float(cfg.press_target_surface_distance_m)
                ):
                    stop_reason = "ik_press_no_surface_progress"
                elif press_min_surface_distance <= float(cfg.press_target_surface_distance_m):
                    stop_reason = "surface_distance_close_no_force"
                elif press_ctrl_target_moved and not press_surface_progress:
                    press_controller_target_moved_tip_static = True
                    stop_reason = "press_controller_target_moved_tip_static"
                elif press_surface_progress:
                    stop_reason = "press_progress_no_force"
                else:
                    stop_reason = "press_entered_no_surface_progress"
    elif precontact_reached and not target_contact_observed and not non_target:
        stop_reason = "precontact_reached_press_not_entered"
    elif not target_contact_observed and not non_target and stop_reason == "max_steps_reached":
        stop_reason = "precontact_not_reached"
    reduction = start_f3_distance - min_f3_distance
    return {
        "executed": True,
        "stop_reason": stop_reason,
        "steps": steps,
        "non_target_contact_before_target": non_target,
        "workspace_or_table_clamp_observed": clamp_seen,
        "start_finger3_distance_to_contact_target": start_f3_distance,
        "start_finger4_distance_to_contact_target": start_f4_distance,
        "min_finger3_distance_to_contact_target": min_f3_distance,
        "min_finger4_distance_to_contact_target": min_f4_distance,
        "min_finger3_distance_to_precontact": min_f3_precontact_distance,
        "min_finger3_distance_to_real_surface": min_f3_surface_distance,
        "min_finger4_distance_to_real_surface": min_f4_surface_distance,
        "finger3_tip_at_min_surface_distance": finger3_tip_at_min_surface_distance,
        "finger4_tip_at_min_surface_distance": finger4_tip_at_min_surface_distance,
        "press_depth_attempted": press_depth_attempted,
        "press_stage_entered": press_stage_entered,
        "press_start_reason": press_start_reason,
        "press_depth_commanded": press_depth_commanded,
        "press_depth_actual": press_depth_actual,
        "press_direction_flipped": press_direction_flipped,
        "press_surface_progress": press_surface_progress,
        "press_control_mode_used": press_control_mode_used,
        "press_policy_action_norm_peak": press_policy_action_norm_peak,
        "press_mapped_isaac_action_norm_peak": press_mapped_isaac_action_norm_peak,
        "press_ctrl_target_delta_peak": press_ctrl_target_delta_peak,
        "press_ctrl_target_moved": press_ctrl_target_moved,
        "press_finger3_delta_dot_press_dir": press_cumulative_tip_progress,
        "press_finger3_delta_dot_press_dir_peak": press_finger3_delta_dot_press_dir_peak,
        "press_z_clamp_before_press": press_z_clamp_before_press,
        "press_z_clamp_after_reanchor": press_z_clamp_after_reanchor,
        "press_z_clamp_after_z_raise": press_z_clamp_after_z_raise,
        "workspace_clamp_max_during_press": workspace_clamp_max_during_press,
        "press_blocked_by_z_clamp": press_blocked_by_z_clamp,
        "press_action_not_reaching_controller": press_action_not_reaching_controller,
        "press_action_frame_wrong": press_action_frame_wrong,
        "press_controller_target_moved_tip_static": press_controller_target_moved_tip_static,
        "press_target_z_raised_after_preflight": press_target_z_raised_after_preflight,
        "press_target_minus_tip_dot_press_dir": press_target_minus_tip_dot_press_dir,
        "surface_point_minus_tip_dot_press_dir": surface_point_minus_tip_dot_press_dir,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "press_start_surface_distance": press_start_surface_distance,
        "press_min_surface_distance": press_min_surface_distance,
        "press_final_surface_distance": press_final_surface_distance,
        "press_target_surface_distance": float(cfg.press_target_surface_distance_m),
        "total_press_limit_m": float(cfg.press_ik_fallback_max_total_m),
        "per_step_press_m": float(cfg.press_ik_fallback_step_m),
        "surface_distance_reduction_during_press": surface_distance_reduction_during_press,
        "action_probe_surface_reduction": action_probe_surface_reduction,
        "ik_surface_reduction": ik_surface_reduction,
        "press_dir_surface_dir_dot_last": press_dir_surface_dir_dots[-1] if press_dir_surface_dir_dots else "",
        "press_dir_surface_dir_dot_min": min(press_dir_surface_dir_dots) if press_dir_surface_dir_dots else "",
        "press_dir_surface_dir_dot_mean": (
            sum(press_dir_surface_dir_dots) / len(press_dir_surface_dir_dots)
            if press_dir_surface_dir_dots
            else ""
        ),
        "distance_reduced_during_approach": bool(reduction >= float(cfg.approach_fail_reduction_threshold_m)),
        "distance_reduction_m": reduction,
        "gain_increased": gain_increased,
        "near_surface_reached": near_surface_reached,
        "precontact_reached": precontact_reached,
        "target_contact_observed": target_contact_observed,
        "clamp_adjusted": clamp_adjusted,
        "clamp_adjustment_type": clamp_adjustment_type,
        "clamp_adjustment_step": clamp_adjustment_step,
    }


def _run_close(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    *,
    close_trigger: str,
    alignment: AlignmentRecorder | None = None,
) -> dict[str, Any]:
    limiter_applied = False
    limiter_bypassed = False
    for step in range(max(1, int(cfg.max_close_steps))):
        if close_trigger == "micro_close_debug_no_force":
            fraction = 0.05 + 0.15 * (step / max(1, int(cfg.max_close_steps) - 1))
        else:
            fraction = 0.5 + 0.5 * (step / max(1, int(cfg.max_close_steps) - 1))
        action = [0.0] * 16
        for col in range(6, 16):
            action[col] = fraction
        state_before = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        )
        if alignment is not None:
            alignment.update_markers(base, state_before, contact_target_local=state_before.get("contact_target_local_pos"))
        audit = _step_policy(env, mapper, env_index, action, alignment=alignment)
        state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        )
        limiter_applied = limiter_applied or bool(state.get("last_dex_hand_limiter_applied"))
        limiter_bypassed = limiter_bypassed or bool(state.get("last_dex_hand_limiter_bypassed_for_v95"))
        _append_log(
            logs,
            "close",
            step,
            state,
            cfg,
            object_reset_pos,
            extra={
                "close_command_value": fraction,
                "close_mode": "full_hand_distal_close_via_existing_mapper",
                "close_trigger": close_trigger,
                **audit,
            },
            alignment=alignment,
            base=base,
        )
    return {
        "executed": True,
        "close_trigger": close_trigger,
        "limiter_applied": limiter_applied,
        "limiter_bypassed_for_v95": limiter_bypassed,
        "debug_forced_close_without_contact": close_trigger in {"debug_forced_without_contact", "micro_close_debug_no_force"},
    }


def _run_lift(
    env: Any,
    base: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None = None,
) -> dict[str, Any]:
    for step in range(max(1, int(cfg.max_lift_steps))):
        action = [0.0] * 16
        action[2] = float(cfg.lift_action_gain)
        for col in range(6, 16):
            action[col] = 1.0
        state_before = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        )
        if alignment is not None:
            alignment.update_markers(base, state_before, contact_target_local=state_before.get("contact_target_local_pos"))
        audit = _step_policy(env, mapper, env_index, action, alignment=alignment)
        state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        )
        _append_log(
            logs,
            "lift",
            step,
            state,
            cfg,
            object_reset_pos,
            extra={"lift_command_z": cfg.lift_action_gain, **audit},
            alignment=alignment,
            base=base,
    )
    return {"executed": True, "steps": int(cfg.max_lift_steps), "skip_reason": ""}


def _run_contact_truth_pass(
    env: Any,
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    logs: list[dict[str, Any]],
    object_reset_pos: list[float],
    alignment: AlignmentRecorder | None,
    *,
    adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adjustment = dict(adjustment or {})
    if alignment is not None:
        alignment.setup_body_mapping_and_sensors(base)
    all_rows: list[dict[str, Any]] = []

    def _capture(phase: str, step: int, state: dict[str, Any], extra: dict[str, Any]) -> list[dict[str, Any]]:
        before = len(getattr(alignment, "contact_rows", []) or []) if alignment is not None else 0
        _append_log(
            logs,
            phase,
            step,
            state,
            cfg,
            object_reset_pos,
            extra=extra,
            alignment=alignment,
            base=base,
        )
        after_rows = list(getattr(alignment, "contact_rows", []) or [])[before:] if alignment is not None else []
        all_rows.extend(after_rows)
        return after_rows

    before_state = _state_with_contact_target(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        adjustment=adjustment,
    )
    before_distance = float(
        before_state.get("finger3_surface_distance_servo_m", before_state.get("finger3_distance_to_real_surface_m", math.inf))
    )
    before_tip = list(before_state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
    _capture(
        "contact_truth_before",
        0,
        before_state,
        {
            "contact_truth_event": "before_controlled_penetration",
            "contact_truth_probe_limit_m": float(cfg.contact_truth_extra_penetration_m),
            "contact_truth_step_m": float(cfg.contact_truth_step_m),
            "contact_truth_surface_distance_m": before_distance,
            "contact_truth_finger3_target_force_n": before_state.get("finger3_target_filtered_force_n", 0.0),
            "contact_truth_finger3_unfiltered_force_n": before_state.get("finger3_unfiltered_force_n", 0.0),
        },
    )

    commanded = 0.0
    actual_motion = 0.0
    stop_reason = "max_extra_penetration_reached"
    state = before_state
    max_steps = max(0, int(math.ceil(float(cfg.contact_truth_extra_penetration_m) / max(1.0e-9, float(cfg.contact_truth_step_m)))))
    for step in range(max_steps):
        peaks = _contact_truth_peaks(all_rows, cfg.part, cfg.force_threshold_n)
        if float(state.get("active_target_filtered_force_peak_n", 0.0) or 0.0) > cfg.force_threshold_n:
            stop_reason = "target_filtered_contact_before_extra_penetration"
            break
        if float(peaks.get("raw_tip_screw1_peak_n", 0.0) or 0.0) > cfg.force_threshold_n:
            stop_reason = "raw_tip_screw1_contact_before_extra_penetration"
            break
        if float(peaks.get("raw_non_target_peak_n", 0.0) or 0.0) > cfg.force_threshold_n:
            stop_reason = "raw_non_target_contact_before_extra_penetration"
            break
        if _clamp_or_barrier(state):
            stop_reason = "clamp_before_extra_penetration"
            break
        if actual_motion >= float(cfg.contact_truth_extra_penetration_m):
            stop_reason = "actual_extra_penetration_limit_reached"
            break
        limit = float(cfg.contact_truth_extra_penetration_m)
        remaining = min(limit - commanded, limit - actual_motion)
        step_size = min(float(cfg.contact_truth_step_m), max(0.0, remaining))
        if step_size <= 1.0e-9:
            break
        direction = _unit(list(state.get("finger3_surface_direction_xyz", []) or []))
        if _norm(direction) <= 1.0e-6:
            stop_reason = "surface_direction_unavailable"
            break
        palm = list(state.get("palm_local_pos", [0.0, 0.0, 0.0]))
        desired_palm = [palm[axis] + direction[axis] * step_size for axis in range(3)]
        tip_before = list(state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
        distance_before = float(
            state.get("finger3_surface_distance_servo_m", state.get("finger3_distance_to_real_surface_m", math.inf))
        )
        ik_audit = _move_ctrl_target_by_ik(
            base,
            env_index,
            desired_palm,
            max_time=float(cfg.press_ik_fallback_max_time_s),
            pos_tol=float(cfg.press_ik_fallback_pos_tol_m),
        )
        commanded += step_size
        next_state = _state_with_contact_target(
            base,
            env_index,
            cfg,
            read_state(base, env_index, cfg.part, cfg.force_threshold_n),
            adjustment=adjustment,
        )
        tip_after = list(next_state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]))
        motion = max(0.0, _dot(_sub_vec(tip_after, tip_before), direction))
        actual_motion += motion
        distance_after = float(
            next_state.get("finger3_surface_distance_servo_m", next_state.get("finger3_distance_to_real_surface_m", math.inf))
        )
        rows = _capture(
            "contact_truth_penetration",
            step,
            next_state,
            {
                "contact_truth_event": "controlled_penetration_step",
                "contact_truth_probe_limit_m": float(cfg.contact_truth_extra_penetration_m),
                "contact_truth_step_m": step_size,
                "contact_truth_commanded_m": commanded,
                "contact_truth_actual_motion_m": actual_motion,
                "contact_truth_surface_distance_before_m": distance_before,
                "contact_truth_surface_distance_after_m": distance_after,
                "contact_truth_surface_direction_x": _list_get(direction, 0),
                "contact_truth_surface_direction_y": _list_get(direction, 1),
                "contact_truth_surface_direction_z": _list_get(direction, 2),
                "contact_truth_finger3_target_force_n": next_state.get("finger3_target_filtered_force_n", 0.0),
                "contact_truth_finger3_unfiltered_force_n": next_state.get("finger3_unfiltered_force_n", 0.0),
                **ik_audit,
            },
        )
        peaks = _contact_truth_peaks(rows, cfg.part, cfg.force_threshold_n)
        state = next_state
        if float(next_state.get("active_target_filtered_force_peak_n", 0.0) or 0.0) > cfg.force_threshold_n:
            stop_reason = "target_filtered_contact_observed"
            break
        if float(peaks.get("raw_tip_screw1_peak_n", 0.0) or 0.0) > cfg.force_threshold_n:
            stop_reason = "raw_tip_screw1_contact_observed"
            break
        if float(peaks.get("raw_non_target_peak_n", 0.0) or 0.0) > cfg.force_threshold_n:
            stop_reason = "raw_non_target_contact_observed"
            break
        if _clamp_or_barrier(next_state):
            stop_reason = "clamp_observed"
            break

    after_state = _state_with_contact_target(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.force_threshold_n),
        adjustment=adjustment,
    )
    after_distance = float(
        after_state.get("finger3_surface_distance_servo_m", after_state.get("finger3_distance_to_real_surface_m", math.inf))
    )
    _capture(
        "contact_truth_after",
        0,
        after_state,
        {
            "contact_truth_event": "after_controlled_penetration",
            "contact_truth_stop_reason": stop_reason,
            "contact_truth_probe_limit_m": float(cfg.contact_truth_extra_penetration_m),
            "contact_truth_commanded_m": commanded,
            "contact_truth_actual_motion_m": actual_motion,
            "contact_truth_surface_distance_m": after_distance,
            "contact_truth_finger3_target_force_n": after_state.get("finger3_target_filtered_force_n", 0.0),
            "contact_truth_finger3_unfiltered_force_n": after_state.get("finger3_unfiltered_force_n", 0.0),
        },
    )
    peaks = _contact_truth_peaks(all_rows, cfg.part, cfg.force_threshold_n)
    body_chain = _finger3_body_chain_truth_metadata(base, env_index, after_state)
    screw_truth = _screw1_collision_truth_metadata(base, env_index, after_state)
    contact_class, reason, excluded = _classify_contact_truth_result(
        peaks=peaks,
        before_state=before_state,
        after_state=after_state,
        body_chain=body_chain,
        screw_truth=screw_truth,
        threshold=float(cfg.force_threshold_n),
    )
    return {
        "executed": True,
        "contact_truth_class": contact_class,
        "contact_truth_reason": reason,
        "excluded_classes": excluded,
        "stop_reason": stop_reason,
        "before_finger3_surface_distance_m": before_distance,
        "after_finger3_surface_distance_m": after_distance,
        "before_finger3_target_force_n": before_state.get("finger3_target_filtered_force_n", 0.0),
        "after_finger3_target_force_n": after_state.get("finger3_target_filtered_force_n", 0.0),
        "before_finger3_unfiltered_force_n": before_state.get("finger3_unfiltered_force_n", 0.0),
        "after_finger3_unfiltered_force_n": after_state.get("finger3_unfiltered_force_n", 0.0),
        "extra_penetration_commanded_m": commanded,
        "extra_penetration_actual_m": actual_motion,
        **peaks,
        **body_chain,
        **screw_truth,
    }


def _run_runtime_collision_debug(
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    state: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Query PhysX runtime collision distance without moving the hand or object."""

    probe_path = output_dir / "runtime_collision_probe.json"
    origin = _tensor_vec(getattr(getattr(base, "scene", None), "env_origins", None), env_index)
    root_path = f"/World/envs/env_{int(env_index)}/Screw1"
    tip_local = list(state.get("finger3_tip_local_pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
    tip_world = _add_vec(tip_local, origin)
    object_world = list(state.get("object_world_pos", []) or _add_vec(list(state.get("object_local_pos", [])), origin))
    aabb_distance = _finite_or_blank(
        state.get("finger3_surface_distance_servo_m", state.get("finger3_distance_to_real_surface_m", ""))
    )
    radius = _finger3_tip_probe_radius(state)
    query_points_world, query_labels = _finger3_tip_probe_points(tip_world, radius)
    metadata = _runtime_collision_usd_metadata(root_path, state)
    out: dict[str, Any] = {
        "runtime_collision_debug": True,
        "runtime_collision_executed": True,
        "runtime_collision_probe_json": str(probe_path),
        "runtime_collision_query_available": False,
        "runtime_collision_query_source": "",
        "runtime_collision_unavailable_reason": "",
        "runtime_collision_conclusion": "B_runtime_collision_unavailable_contact_driven_required",
        "target_root_path": root_path,
        "stage_units_meters_per_unit": metadata.get("stage_units_meters_per_unit", ""),
        "finger3_tip_local_xyz": tip_local,
        "finger3_tip_world_xyz": tip_world,
        "finger3_tip_collision_probe_radius_m": radius,
        "finger3_tip_probe_point_labels": query_labels,
        "finger3_tip_probe_points_world_xyz": query_points_world,
        "screw1_runtime_root_local_xyz": state.get("object_local_pos", []),
        "screw1_runtime_root_world_xyz": object_world,
        "aabb_distance_m": aabb_distance,
        "sdf_available": False,
        "sdf_error": "",
        "scene_query_available": False,
        "scene_query_error": "",
        **metadata,
    }
    sdf_result = _query_screw1_sdf_runtime_distance(
        root_path=root_path,
        collision_paths=list(metadata.get("screw1_collision_prim_paths", []) or []),
        query_points_world=query_points_world,
        query_labels=query_labels,
        radius=radius,
    )
    out.update(sdf_result)
    if bool(sdf_result.get("sdf_available")) and math.isfinite(
        float(sdf_result.get("distance_to_runtime_collision_m", math.inf) or math.inf)
    ):
        out["runtime_collision_query_available"] = True
        out["runtime_collision_query_source"] = "sdf_shape_view"
    else:
        directions = _runtime_collision_scene_query_directions(state, tip_world, object_world, metadata)
        scene_result = _query_screw1_scene_runtime_distance(root_path, tip_world, radius, directions)
        out.update(scene_result)
        if bool(scene_result.get("scene_query_available")) and math.isfinite(
            float(scene_result.get("scene_query_distance_m", math.inf) or math.inf)
        ):
            out["runtime_collision_query_available"] = True
            out["runtime_collision_query_source"] = str(scene_result.get("scene_query_source", "scene_query"))
            out["distance_to_runtime_collision_m"] = scene_result.get("distance_to_runtime_collision_m", "")
            out["finger3_collision_surface_to_runtime_collision_gap_m"] = scene_result.get(
                "finger3_collision_surface_to_runtime_collision_gap_m", ""
            )
            out["runtime_collision_normal_world_xyz"] = scene_result.get("scene_query_normal_world_xyz", [])
        elif not out.get("runtime_collision_unavailable_reason"):
            out["runtime_collision_unavailable_reason"] = ";".join(
                item
                for item in (
                    str(sdf_result.get("sdf_error", "")),
                    str(scene_result.get("scene_query_error", "")),
                )
                if item
            )
    distance = _finite_or_blank(out.get("distance_to_runtime_collision_m", ""))
    if isinstance(aabb_distance, float) and isinstance(distance, float):
        out["aabb_minus_runtime_distance_m"] = float(aabb_distance) - float(distance)
        out["runtime_minus_aabb_distance_m"] = float(distance) - float(aabb_distance)
    else:
        out["aabb_minus_runtime_distance_m"] = ""
        out["runtime_minus_aabb_distance_m"] = ""
    out["runtime_collision_conclusion"] = _classify_runtime_collision_probe(out)
    if not out.get("runtime_collision_query_available") and not out.get("runtime_collision_unavailable_reason"):
        out["runtime_collision_unavailable_reason"] = "no_target_specific_sdf_or_scene_query_distance"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(json.dumps(_plain(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _runtime_collision_usd_metadata(root_path: str, state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "screw1_root_prim_path": root_path,
        "screw1_visual_prim_paths": [],
        "screw1_collision_prim_paths": [],
        "screw1_collision_shape_count": 0,
        "screw1_collision_shape_types": [],
        "screw1_runtime_collision_representation": [],
        "screw1_sdf_collision_prim_paths": [],
        "stage_units_meters_per_unit": "",
        "screw1_runtime_collision_metadata_error": "",
    }
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        out["stage_units_meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))
        root = stage.GetPrimAtPath(root_path)
        if root is None or not root.IsValid():
            raise RuntimeError(f"screw1_root_prim_missing:{root_path}")
        root_matrix = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        out["screw1_root_world_translation_xyz"] = _gf_matrix_translation(root_matrix)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        collision_paths: list[str] = []
        collision_types: list[str] = []
        sdf_paths: list[str] = []
        visual_paths: list[str] = []
        shapes: list[dict[str, Any]] = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) and "collision" not in path.lower():
                visual_paths.append(path)
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if not collision_enabled:
                continue
            type_name = str(prim.GetTypeName() or prim.GetPrimTypeInfo().GetTypeName() or "")
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
            approximation = ""
            try:
                approximation = str(mesh_collision_api.GetApproximationAttr().Get() or "") if mesh_collision_api else ""
            except Exception:
                approximation = ""
            sdf_attrs = _usd_attrs_with_token(prim, "sdf")
            if approximation.lower() == "sdf" or bool(sdf_attrs):
                sdf_paths.append(path)
            box = _bbox_for_prims(cache, [prim])
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            runtime_root_world = list(state.get("object_world_pos", []) or [])
            shape = {
                "path": path,
                "type": f"{type_name}{':' + approximation if approximation else ''}",
                "type_name": type_name,
                "collision_enabled": collision_enabled,
                "mesh_collision_approximation": approximation,
                "physx_sdf_attrs": sdf_attrs,
                "world_translation_xyz": _gf_matrix_translation(matrix),
                "world_bbox_min_xyz": box.get("min", []),
                "world_bbox_max_xyz": box.get("max", []),
                "world_bbox_center_xyz": box.get("center", []),
                "world_bbox_extent_xyz": box.get("extent", []),
                "runtime_root_to_collision_prim_translation_m": _distance(
                    _gf_matrix_translation(root_matrix), _gf_matrix_translation(matrix)
                ),
                "runtime_root_to_collision_bbox_center_m": _distance(runtime_root_world, box.get("center", []))
                if len(runtime_root_world) >= 3
                else "",
            }
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get() or []
                face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
                shape["mesh_point_count"] = len(points)
                shape["mesh_face_count"] = len(face_counts)
            try:
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI(prim)
                shape["physx_sdf_margin"] = _usd_schema_attr(sdf_api, "GetSdfMarginAttr")
                shape["physx_sdf_resolution"] = _usd_schema_attr(sdf_api, "GetSdfResolutionAttr")
                shape["physx_sdf_subgrid_resolution"] = _usd_schema_attr(sdf_api, "GetSdfSubgridResolutionAttr")
            except Exception:
                pass
            shapes.append(shape)
            collision_paths.append(path)
            collision_types.append(shape["type"])
        out.update(
            {
                "screw1_visual_prim_paths": visual_paths,
                "screw1_collision_prim_paths": collision_paths,
                "screw1_collision_shape_count": len(collision_paths),
                "screw1_collision_shape_types": collision_types,
                "screw1_runtime_collision_representation": shapes,
                "screw1_sdf_collision_prim_paths": sdf_paths,
                "screw1_collision_bounds_source": state.get("surface_source", ""),
                "true_collision_surface_available": bool(sdf_paths),
                "true_collision_surface_unavailable_reason": ""
                if sdf_paths
                else "no_sdf_collision_prim_metadata_found_under_screw1",
            }
        )
    except Exception as exc:
        out["screw1_runtime_collision_metadata_error"] = f"{type(exc).__name__}:{exc}"
        out["true_collision_surface_available"] = False
        out["true_collision_surface_unavailable_reason"] = out["screw1_runtime_collision_metadata_error"]
    return out


def _query_screw1_sdf_runtime_distance(
    *,
    root_path: str,
    collision_paths: list[str],
    query_points_world: list[list[float]],
    query_labels: list[str],
    radius: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sdf_available": False,
        "sdf_error": "",
        "sdf_query_paths_attempted": [],
        "sdf_object_paths": [],
        "sdf_distance_component_index": 3,
        "sdf_raw_at_finger3_tip": [],
        "screw1_sdf_distance_at_finger3_tip_m": "",
        "screw1_sdf_gradient_at_finger3_tip_xyz": [],
        "distance_to_runtime_collision_m": "",
        "finger3_collision_surface_to_runtime_collision_gap_m": "",
    }
    if not query_points_world:
        out["sdf_error"] = "no_query_points"
        return out
    sdf_paths = [path for path in collision_paths if path.endswith("/geometry/mesh")]
    sdf_paths.extend(path for path in collision_paths if path not in sdf_paths)
    if not sdf_paths:
        sdf_paths = [f"{root_path}/geometry/mesh"]
    out["sdf_query_paths_attempted"] = sdf_paths
    errors: list[str] = []
    best: dict[str, Any] | None = None
    for path in sdf_paths:
        try:
            result = _query_single_sdf_path(path, query_points_world, query_labels, radius)
            if not result.get("sdf_available"):
                errors.append(f"{path}:{result.get('sdf_error', 'unavailable')}")
                continue
            value = float(result.get("finger3_collision_surface_to_runtime_collision_gap_m", math.inf) or math.inf)
            if best is None or value < float(best.get("finger3_collision_surface_to_runtime_collision_gap_m", math.inf)):
                best = result
        except Exception as exc:
            errors.append(f"{path}:{type(exc).__name__}:{exc}")
    if best is None:
        out["sdf_error"] = ";".join(errors) if errors else "sdf_shape_view_no_result"
        return out
    out.update(best)
    out["sdf_available"] = True
    return out


def _query_single_sdf_path(
    path: str,
    query_points_world: list[list[float]],
    query_labels: list[str],
    radius: float,
) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    import omni.physics.tensors as tensors  # noqa: PLC0415
    import omni.usd  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("stage_unavailable")
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"sdf_prim_missing:{path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inverse = matrix.GetInverse()
    query_points_local = [_gf_transform_point(inverse, point) for point in query_points_world]
    num_points = len(query_points_local)
    sim_view = tensors.create_simulation_view("warp")
    sdf_view = sim_view.create_sdf_shape_view(path, num_points)
    if int(getattr(sdf_view, "count", 0) or 0) <= 0:
        raise RuntimeError(f"sdf_shape_view_count_zero:{path}")
    if hasattr(sdf_view, "check") and not bool(sdf_view.check()):
        raise RuntimeError(f"sdf_shape_view_check_failed:{path}")
    points = np.asarray(query_points_local, dtype=np.float32).reshape(1, num_points, 3)
    points_wp = wp.from_numpy(points.flatten(), dtype=wp.float32, device=sim_view.device)
    raw = sdf_view.get_sdf_and_gradients(points_wp)
    raw_np = raw.numpy().reshape(int(sdf_view.count), num_points, 4)
    first = raw_np[0]
    # Installed tensor tests use the last channel as SDF and first three as gradient.
    distances = [float(value) for value in first[:, 3].tolist()]
    gradients = [[float(item) for item in row[:3].tolist()] for row in first]
    tip_distance = distances[0] if distances else math.inf
    sphere_sample_distances = distances[1:] if len(distances) > 1 else []
    surface_gap = min(sphere_sample_distances) if sphere_sample_distances else tip_distance - float(radius)
    object_paths = []
    try:
        object_paths = [str(item) for item in list(sdf_view.object_paths)]
    except Exception:
        object_paths = []
    return {
        "sdf_available": True,
        "sdf_error": "",
        "sdf_query_path": path,
        "sdf_object_paths": object_paths,
        "sdf_query_points_collision_local_xyz": query_points_local,
        "sdf_query_point_labels": query_labels,
        "sdf_raw_at_finger3_tip": [float(item) for item in first[0].tolist()],
        "sdf_raw_all_points": [[float(item) for item in row.tolist()] for row in first],
        "sdf_distances_m": distances,
        "sdf_gradients_xyz": gradients,
        "screw1_sdf_distance_at_finger3_tip_m": tip_distance,
        "screw1_sdf_gradient_at_finger3_tip_xyz": gradients[0] if gradients else [],
        "distance_to_runtime_collision_m": tip_distance,
        "finger3_collision_surface_to_runtime_collision_gap_m": surface_gap,
        "runtime_collision_normal_world_xyz": _unit(_gf_transform_vector(matrix, gradients[0] if gradients else [0.0, 0.0, 0.0])),
    }


def _query_screw1_scene_runtime_distance(
    root_path: str,
    tip_world: list[float],
    radius: float,
    directions: list[list[float]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scene_query_available": False,
        "scene_query_error": "",
        "scene_query_source": "",
        "scene_query_distance_m": "",
        "distance_to_runtime_collision_m": "",
        "finger3_collision_surface_to_runtime_collision_gap_m": "",
        "scene_query_collision_path": "",
        "scene_query_rigid_body_path": "",
        "scene_query_normal_world_xyz": [],
        "scene_query_hits": [],
        "scene_query_max_distance_m": 1.0,
    }
    try:
        from omni.physx import get_physx_scene_query_interface  # noqa: PLC0415
        from pxr import Gf  # noqa: PLC0415

        scene_query = get_physx_scene_query_interface()
        hits: list[dict[str, Any]] = []
        max_distance = float(out["scene_query_max_distance_m"])

        def _record_hit(hit: Any, source: str) -> dict[str, Any]:
            data = _scene_query_hit_to_dict(hit)
            data["source"] = source
            data["is_target_screw1"] = _scene_query_hit_is_target(data, root_path)
            hits.append(data)
            return data

        try:
            def _overlap_callback(hit: Any) -> bool:
                _record_hit(hit, "overlap_sphere")
                return True

            scene_query.overlap_sphere(float(radius), Gf.Vec3f(*tip_world[:3]), _overlap_callback)
            for hit in hits:
                if hit.get("source") == "overlap_sphere" and hit.get("is_target_screw1"):
                    out.update(_scene_query_result_from_hit(hit, "overlap_sphere", 0.0, radius))
                    out["scene_query_hits"] = hits
                    return out
        except Exception as exc:
            out["scene_query_error"] = f"overlap_sphere:{type(exc).__name__}:{exc}"

        for direction in directions:
            unit = _unit(direction)
            if _norm(unit) <= 1.0e-6:
                continue
            try:
                hit = scene_query.sweep_sphere_closest(
                    float(radius),
                    Gf.Vec3f(*tip_world[:3]),
                    Gf.Vec3f(*unit[:3]),
                    max_distance,
                )
                data = _record_hit(hit, "sweep_sphere_closest")
                if bool(data.get("hit")) and data.get("is_target_screw1"):
                    distance = float(data.get("distance", 0.0) or 0.0)
                    out.update(_scene_query_result_from_hit(data, "sweep_sphere_closest", distance, radius))
                    out["scene_query_hits"] = hits
                    return out
            except Exception as exc:
                out["scene_query_error"] = _join_errors(out.get("scene_query_error", ""), f"sweep:{type(exc).__name__}:{exc}")
            try:
                hit = scene_query.raycast_closest(
                    Gf.Vec3f(*tip_world[:3]),
                    Gf.Vec3f(*unit[:3]),
                    max_distance,
                )
                data = _record_hit(hit, "raycast_closest")
                if bool(data.get("hit")) and data.get("is_target_screw1"):
                    distance = float(data.get("distance", 0.0) or 0.0)
                    out.update(_scene_query_result_from_hit(data, "raycast_closest", distance, radius))
                    out["scene_query_hits"] = hits
                    return out
            except Exception as exc:
                out["scene_query_error"] = _join_errors(out.get("scene_query_error", ""), f"raycast:{type(exc).__name__}:{exc}")
        out["scene_query_hits"] = hits
        if not out["scene_query_error"]:
            out["scene_query_error"] = "no_target_screw1_scene_query_hit"
    except Exception as exc:
        out["scene_query_error"] = f"{type(exc).__name__}:{exc}"
    return out


def _scene_query_result_from_hit(hit: dict[str, Any], source: str, distance: float, radius: float) -> dict[str, Any]:
    if source == "sweep_sphere_closest":
        surface_gap = max(0.0, float(distance))
        origin_distance = float(distance) + float(radius)
    elif source == "overlap_sphere":
        surface_gap = 0.0
        origin_distance = max(0.0, float(radius))
    else:
        origin_distance = max(0.0, float(distance))
        surface_gap = max(0.0, float(distance) - float(radius))
    return {
        "scene_query_available": True,
        "scene_query_error": "",
        "scene_query_source": source,
        "scene_query_distance_m": surface_gap,
        "distance_to_runtime_collision_m": origin_distance,
        "finger3_collision_surface_to_runtime_collision_gap_m": surface_gap,
        "scene_query_collision_path": hit.get("collision", ""),
        "scene_query_rigid_body_path": hit.get("rigidBody", hit.get("rigid_body", "")),
        "scene_query_normal_world_xyz": hit.get("normal", []),
        "scene_query_position_world_xyz": hit.get("position", []),
    }


def _scene_query_hit_to_dict(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        return {
            "hit": bool(hit.get("hit", True)),
            "collision": str(hit.get("collision", "")),
            "rigidBody": str(hit.get("rigidBody", hit.get("rigid_body", ""))),
            "distance": _finite_or_blank(hit.get("distance", "")),
            "position": _vec_obj_to_list(hit.get("position", [])),
            "normal": _vec_obj_to_list(hit.get("normal", [])),
            "faceIndex": hit.get("faceIndex", hit.get("face_index", "")),
        }
    return {
        "hit": True,
        "collision": str(getattr(hit, "collision", "")),
        "rigidBody": str(getattr(hit, "rigid_body", getattr(hit, "rigidBody", ""))),
        "distance": _finite_or_blank(getattr(hit, "distance", "")),
        "position": _vec_obj_to_list(getattr(hit, "position", [])),
        "normal": _vec_obj_to_list(getattr(hit, "normal", [])),
        "faceIndex": getattr(hit, "face_index", getattr(hit, "faceIndex", "")),
    }


def _scene_query_hit_is_target(hit: dict[str, Any], root_path: str) -> bool:
    for key in ("collision", "rigidBody", "rigid_body"):
        value = str(hit.get(key, "") or "")
        if value == root_path or value.startswith(f"{root_path}/"):
            return True
    return False


def _runtime_collision_scene_query_directions(
    state: dict[str, Any],
    tip_world: list[float],
    object_world: list[float],
    metadata: dict[str, Any] | None = None,
) -> list[list[float]]:
    directions: list[list[float]] = []
    surface_dir = _unit(list(state.get("finger3_surface_direction_xyz", []) or []))
    if _norm(surface_dir) > 1.0e-6:
        directions.append(surface_dir)
    if len(object_world) >= 3 and len(tip_world) >= 3:
        directions.append(_unit(_sub_vec(object_world, tip_world)))
    normal = _unit(list(state.get("contact_target_normal_xyz", []) or []))
    if _norm(normal) > 1.0e-6:
        directions.append([-normal[0], -normal[1], -normal[2]])
        directions.append(normal)
    metadata = dict(metadata or {})
    for shape in list(metadata.get("screw1_runtime_collision_representation", []) or []):
        if not isinstance(shape, dict):
            continue
        center = list(shape.get("world_bbox_center_xyz", []) or [])
        if len(center) >= 3 and len(tip_world) >= 3:
            directions.append(_unit(_sub_vec(center, tip_world)))
    unique: list[list[float]] = []
    for direction in directions:
        unit = _unit(direction)
        if _norm(unit) <= 1.0e-6:
            continue
        if not any(abs(_dot(unit, other)) > 0.999 for other in unique):
            unique.append(unit)
    return unique


def _classify_runtime_collision_probe(probe: dict[str, Any]) -> str:
    if bool(probe.get("runtime_collision_query_available")):
        root_to_shape = [
            max(
                float(item.get("runtime_root_to_collision_prim_translation_m", 0.0) or 0.0),
                float(item.get("runtime_root_to_collision_bbox_center_m", 0.0) or 0.0),
            )
            for item in list(probe.get("screw1_runtime_collision_representation", []) or [])
            if isinstance(item, dict)
        ]
        max_root_to_shape = max([0.0, *root_to_shape])
        if max_root_to_shape > 0.25:
            return "C_asset_collision_wrong"
        return "A_runtime_collision_can_be_extracted"
    if int(probe.get("screw1_collision_shape_count", 0) or 0) <= 0:
        return "C_asset_collision_wrong"
    return "B_runtime_collision_unavailable_contact_driven_required"


def _finger3_tip_probe_radius(state: dict[str, Any]) -> float:
    extent = list(state.get("finger3_tip_collision_bbox_extent_xyz", []) or [])
    if len(extent) >= 3:
        radius = max(abs(float(value)) for value in extent[:3]) * 0.5
        if math.isfinite(radius) and radius > 1.0e-5:
            return float(radius)
    offset = _finite_or_blank(state.get("finger3_tip_origin_to_collision_surface_offset_m", ""))
    if isinstance(offset, float) and offset > 1.0e-5:
        return offset
    return 0.006


def _finger3_tip_probe_points(tip_world: list[float], radius: float) -> tuple[list[list[float]], list[str]]:
    axes = [
        ("origin", [0.0, 0.0, 0.0]),
        ("sphere_pos_x", [1.0, 0.0, 0.0]),
        ("sphere_neg_x", [-1.0, 0.0, 0.0]),
        ("sphere_pos_y", [0.0, 1.0, 0.0]),
        ("sphere_neg_y", [0.0, -1.0, 0.0]),
        ("sphere_pos_z", [0.0, 0.0, 1.0]),
        ("sphere_neg_z", [0.0, 0.0, -1.0]),
    ]
    points = [
        [float(tip_world[i]) + float(axis[i]) * float(radius) for i in range(3)]
        for _label, axis in axes
    ]
    labels = [label for label, _axis in axes]
    return points, labels


def _usd_attrs_with_token(prim: Any, token: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    lower = str(token).lower()
    try:
        for attr in prim.GetAttributes():
            name = str(attr.GetName())
            if lower not in name.lower():
                continue
            try:
                out[name] = _plain(attr.Get())
            except Exception as exc:
                out[name] = f"read_error:{type(exc).__name__}:{exc}"
    except Exception:
        pass
    return out


def _usd_schema_attr(schema: Any, getter_name: str) -> Any:
    try:
        getter = getattr(schema, getter_name, None)
        attr = getter() if callable(getter) else None
        return _plain(attr.Get()) if attr is not None and attr.IsValid() else ""
    except Exception:
        return ""


def _gf_transform_point(matrix: Any, point: list[float]) -> list[float]:
    from pxr import Gf  # noqa: PLC0415

    vec = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def _gf_transform_vector(matrix: Any, vec: list[float]) -> list[float]:
    origin = _gf_transform_point(matrix, [0.0, 0.0, 0.0])
    point = _gf_transform_point(matrix, vec)
    return _sub_vec(point, origin)


def _gf_matrix_translation(matrix: Any) -> list[float]:
    try:
        vec = matrix.ExtractTranslation()
        return [float(vec[0]), float(vec[1]), float(vec[2])]
    except Exception:
        try:
            return [float(matrix[3][0]), float(matrix[3][1]), float(matrix[3][2])]
        except Exception:
            return [0.0, 0.0, 0.0]


def _vec_obj_to_list(value: Any) -> list[float]:
    try:
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            return [float(value.x), float(value.y), float(value.z)]
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return []


def _finite_or_blank(value: Any) -> float | str:
    try:
        out = float(value)
        return out if math.isfinite(out) else ""
    except Exception:
        return ""


def _join_errors(existing: str, new: str) -> str:
    if not existing:
        return new
    if not new:
        return existing
    return f"{existing};{new}"


def _contact_truth_peaks(rows: list[dict[str, Any]], target_part: str, threshold: float) -> dict[str, Any]:
    raw_sensor_available = False
    raw_tip_screw1 = 0.0
    raw_any_screw1 = 0.0
    raw_palm_wrist_screw1 = 0.0
    raw_proximal_screw1 = 0.0
    raw_non_target = 0.0
    raw_unknown = 0.0
    peak_body = ""
    peak_object = ""
    for row in rows:
        source = str(row.get("source", ""))
        group = str(row.get("hand_group", ""))
        obj = str(row.get("object_name", ""))
        body = str(row.get("hand_body_name", ""))
        force = float(row.get("force_norm_or_contact_strength", 0.0) or 0.0)
        if source == "debug_contact_sensor" and obj in V83_PARTS:
            raw_sensor_available = True
        if source not in {"debug_contact_sensor", "rigid_body_net_contact_force"}:
            continue
        if obj == target_part:
            if force > raw_any_screw1:
                raw_any_screw1 = force
                peak_body = body
                peak_object = obj
            if group in {"finger3_tip", "finger4_tip"}:
                raw_tip_screw1 = max(raw_tip_screw1, force)
            elif group in {"palm", "wrist"}:
                raw_palm_wrist_screw1 = max(raw_palm_wrist_screw1, force)
            elif group in {"finger3_proximal", "finger4_proximal", "finger_proximal_links", "other_hand_links"}:
                raw_proximal_screw1 = max(raw_proximal_screw1, force)
        elif obj == "unknown_object":
            raw_unknown = max(raw_unknown, force)
        elif obj:
            raw_non_target = max(raw_non_target, force)
    return {
        "raw_sensor_available": raw_sensor_available,
        "raw_tip_screw1_peak_n": raw_tip_screw1,
        "raw_any_screw1_peak_n": raw_any_screw1,
        "raw_palm_wrist_screw1_peak_n": raw_palm_wrist_screw1,
        "raw_proximal_screw1_peak_n": raw_proximal_screw1,
        "raw_non_target_peak_n": raw_non_target,
        "raw_unknown_object_peak_n": raw_unknown,
        "raw_screw1_peak_body": peak_body,
        "raw_screw1_peak_object": peak_object,
        "raw_contact_threshold_n": float(threshold),
    }


def _classify_contact_truth_result(
    *,
    peaks: dict[str, Any],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    body_chain: dict[str, Any],
    screw_truth: dict[str, Any],
    threshold: float,
) -> tuple[str, str, list[str]]:
    excluded: list[str] = []
    max_target = max(
        float(before_state.get("active_target_filtered_force_peak_n", 0.0) or 0.0),
        float(after_state.get("active_target_filtered_force_peak_n", 0.0) or 0.0),
    )
    raw_tip = float(peaks.get("raw_tip_screw1_peak_n", 0.0) or 0.0)
    raw_any = float(peaks.get("raw_any_screw1_peak_n", 0.0) or 0.0)
    raw_palm_wrist = float(peaks.get("raw_palm_wrist_screw1_peak_n", 0.0) or 0.0)
    raw_proximal = float(peaks.get("raw_proximal_screw1_peak_n", 0.0) or 0.0)
    raw_non_target = float(peaks.get("raw_non_target_peak_n", 0.0) or 0.0)
    raw_available = bool(peaks.get("raw_sensor_available", False))
    body_map_ok = bool(body_chain.get("pose_body_equals_collision_body")) and bool(
        body_chain.get("collision_body_equals_sensor_body")
    )
    if not body_map_ok:
        return (
            "body_mapping_issue",
            "finger3 pose body, collision body, and contact sensor body do not map to the same runtime body",
            excluded,
        )
    excluded.append("body_mapping_issue")
    if max_target > threshold:
        excluded.extend(["target_filter_issue", "collision_geometry_issue", "hand_orientation_issue"])
        return "contact_acquired", "finger3/finger4 target-filtered Screw1 contact exceeded threshold", excluded
    if raw_tip > threshold:
        excluded.extend(["body_mapping_issue", "collision_geometry_issue", "hand_orientation_issue"])
        return (
            "target_filter_issue",
            "raw finger3/finger4 tip-to-Screw1 contact exists but dex fingertip target-filtered force remains zero",
            excluded,
        )
    excluded.append("contact_acquired")
    if raw_palm_wrist > threshold or raw_proximal > threshold:
        excluded.extend(["target_filter_issue", "collision_geometry_issue"])
        return (
            "hand_orientation_issue",
            "raw Screw1 contact is on palm/wrist/proximal/other hand body, not finger3/finger4 tip",
            excluded,
        )
    excluded.append("hand_orientation_issue")
    if not raw_available:
        excluded.extend(["target_filter_issue", "collision_geometry_issue"])
        return "contact_sensor_issue", "raw per-body object-resolved contact sensor rows are unavailable", excluded
    excluded.append("contact_sensor_issue")
    if raw_non_target > threshold:
        return "hand_orientation_issue", "raw contact is with a non-target object before Screw1 tip contact", excluded
    true_surface = bool(screw_truth.get("true_collision_surface_available", False))
    reason = "no raw hand-Screw1 contact after bbox-distance press and controlled penetration"
    if not true_surface:
        reason += "; Screw1 true PhysX collision surface is not verified by the USD bbox"
    excluded.append("target_filter_issue")
    return "collision_geometry_issue", reason, excluded


def _build_reset_pregrasp_plan(
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    first_state: dict[str, Any],
) -> list[dict[str, Any]]:
    object_pos = _reset_object_center_local(base, env_index, cfg.part, first_state)
    object_quat = list(first_state["object_quat_wxyz"])
    seeded_state = dict(first_state)
    seeded_state["object_local_pos"] = list(object_pos)
    seeded_state = _state_with_contact_target(base, env_index, cfg, seeded_state)
    search_target = _contact_search_entry_target(base, env_index, cfg, seeded_state)
    contact_target = list(search_target["search_target_local_xyz"])
    desired_tip = list(contact_target)
    palm = first_state["palm_local_pos"]
    tip3_offset = _sub_vec(first_state["finger3_tip_local_pos"], palm)
    palm_from_tip3 = _sub_vec(desired_tip, tip3_offset)
    desired_palm = list(palm_from_tip3)
    table_z = _tensor_scalar(getattr(base, "floating_table_z_est", None), env_index)
    if not math.isfinite(table_z) or table_z == 0.0:
        table_z = max(0.72, object_pos[2] - 0.02)
    return [
        {
            "part_name": cfg.part,
            "env_index": int(env_index),
            "local_env_index": int(env_index),
            "initial_condition_mode": "scripted_baseline_runtime_default_object_pose",
            "support_surface": "runtime_default_pose_preserved",
            "table_top_z_m": float(table_z),
            "object_support_half_height_m": 0.02,
            "object_center_local_xyz": object_pos,
            "object_quat_wxyz": object_quat,
            "hand_target_local_xyz": desired_palm,
            "hand_target_quat_wxyz": _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4),
            "hand_quat_source": "current_runtime_quat_preserved",
            "active_finger_group": "finger3_contact_search",
            "reset_only_object_write_allowed": True,
            "reset_only_hand_write_allowed": True,
            "object_write_after_reset_allowed": False,
            "scripted_contact_baseline_reset": True,
        }
    ]


def _reset_object_center_local(base: Any, env_index: int, part: str, first_state: dict[str, Any]) -> list[float]:
    """Choose a reset-only table-supported baseline start pose.

    Default asset poses for the unified five-object scene can leave Screw1 high
    enough to fall during the first live steps.  For this scripted baseline the
    object may be staged during reset, but never during live rollout.
    """

    if str(part) == "Screw1":
        table_z = _tensor_scalar(getattr(base, "floating_table_z_est", None), env_index)
        if not math.isfinite(table_z) or table_z < 0.70 or table_z > 0.78:
            table_z = 0.7399999965887228
        return [-0.2, 0.0, float(table_z) + 0.0072]
    return list(first_state["object_local_pos"])


def _target_env_index(base: Any, part: str) -> int:
    if part not in V83_PARTS:
        raise RuntimeError(f"unsupported_part_{part}")
    env_index = V83_PARTS.index(part)
    if env_index >= int(getattr(base, "num_envs", 0)):
        raise RuntimeError(f"target_env_{env_index}_outside_num_envs_{getattr(base, 'num_envs', None)}")
    return env_index


def _assert_runtime_interfaces(base: Any, part: str, env_index: int) -> None:
    registry = getattr(base, "v83_active_asset_registry", {})
    if part not in registry or dict(registry.get(part, {})).get("asset") is None:
        raise RuntimeError(f"v83_active_asset_registry_missing_{part}")
    if not hasattr(base, "dex_fingertip_pos"):
        raise RuntimeError("dex_fingertip_pos_missing")
    if str(getattr(base, "dex_fingertip_true_source", "")) != "tip_link_body":
        raise RuntimeError(f"dex_fingertip_true_source_not_tip_link_body:{getattr(base, 'dex_fingertip_true_source', '')}")
    names = list(getattr(base, "dex_fingertip_true_body_names", []) or [])
    if len(names) <= FINGER4_INDEX or names[FINGER3_INDEX] != "right_finger3_tip_link" or names[FINGER4_INDEX] != "right_finger4_tip_link":
        raise RuntimeError(f"unexpected_fingertip_true_body_names:{names}")
    filters = list(getattr(base, "dex_fingertip_target_filter_names", []) or [])
    if part not in filters:
        raise RuntimeError(f"target_filter_missing_{part}:{filters}")
    if env_index != 1 and part == "Screw1":
        raise RuntimeError(f"screw1_env_index_expected_1_got_{env_index}")


def _step_policy(
    env: Any,
    mapper: UnifiedActionMapper,
    env_index: int,
    policy_action: list[float],
    *,
    alignment: AlignmentRecorder | None = None,
) -> dict[str, Any]:
    base = _base_env(env)
    rows = [[0.0] * 16 for _ in range(int(getattr(base, "num_envs", 1)))]
    rows[int(env_index)] = [float(value) for value in policy_action[:16]]
    mapped, _audits = mapper.map_batch(env, rows, device=getattr(base, "device", None), env_indices=list(range(len(rows))))
    mapped_rows = _action_rows(mapped)
    policy_xyz = list(rows[int(env_index)][:3])
    mapped_xyz = list(_list_get_row(mapped_rows, int(env_index), [0.0] * 26)[:3])
    ctrl_before = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    action_audit = _local_action_audit(mapped_rows, int(env_index))
    action_audit.update(
        {
            "commanded_policy_x": _list_get(policy_xyz, 0),
            "commanded_policy_y": _list_get(policy_xyz, 1),
            "commanded_policy_z": _list_get(policy_xyz, 2),
            "commanded_policy_xyz_norm": _norm(policy_xyz),
            "mapped_isaac_action_x": _list_get(mapped_xyz, 0),
            "mapped_isaac_action_y": _list_get(mapped_xyz, 1),
            "mapped_isaac_action_z": _list_get(mapped_xyz, 2),
            "mapped_isaac_action_xyz_norm": _norm(mapped_xyz),
            "ctrl_target_before_x": _list_get(ctrl_before, 0),
            "ctrl_target_before_y": _list_get(ctrl_before, 1),
            "ctrl_target_before_z": _list_get(ctrl_before, 2),
        }
    )
    if alignment is not None:
        action_audit.update(alignment.audit_action(mapped))
    frame_before = alignment.frame_count() if alignment is not None else -1
    env.step(mapped)
    ctrl_after = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    ctrl_delta = _sub_vec(ctrl_after, ctrl_before)
    action_audit.update(
        {
            "ctrl_target_after_x": _list_get(ctrl_after, 0),
            "ctrl_target_after_y": _list_get(ctrl_after, 1),
            "ctrl_target_after_z": _list_get(ctrl_after, 2),
            "ctrl_target_delta_x": _list_get(ctrl_delta, 0),
            "ctrl_target_delta_y": _list_get(ctrl_delta, 1),
            "ctrl_target_delta_z": _list_get(ctrl_delta, 2),
            "ctrl_target_delta_l2": _norm(ctrl_delta),
            "ctrl_target_moved": bool(_norm(ctrl_delta) > 1.0e-6),
        }
    )
    frame_after = alignment.frame_count() if alignment is not None else -1
    if alignment is not None:
        action_audit["video_frame_index"] = int(frame_after - 1) if frame_after > frame_before else -1
    return action_audit


def _action_rows(action: Any) -> list[list[float]]:
    try:
        if hasattr(action, "detach"):
            return [[float(v) for v in row] for row in action.detach().cpu().tolist()]
        return [[float(v) for v in row] for row in action]
    except Exception:
        return []


def _list_get_row(rows: list[list[float]], index: int, default: list[float]) -> list[float]:
    try:
        if 0 <= int(index) < len(rows):
            return list(rows[int(index)])
    except Exception:
        pass
    return list(default)


def _local_action_audit(mapped_rows: list[list[float]], target_env_index: int) -> dict[str, Any]:
    norms = [_norm(row) for row in mapped_rows]
    target_norm = _list_get(norms, target_env_index, 0.0)
    non_target = [value for idx, value in enumerate(norms) if idx != int(target_env_index)]
    active = [idx for idx, value in enumerate(norms) if float(value) > 1.0e-8]
    return {
        "target_env_action_norm": target_norm,
        "non_target_env_action_norm_max": max([0.0, *non_target]),
        "non_target_env_action_norm_sum": sum(non_target),
        "active_action_env_ids": ";".join(str(idx) for idx in active),
        "action_sent_only_to_env_1": bool(active == [int(target_env_index)]),
    }


def _policy_action_for_cartesian_step(base: Any, desired_delta_xyz: list[float]) -> list[float]:
    scale = abs(_scalar_attr(base, "pos_threshold", 1.0)) * abs(_scalar_attr(base, "floating_action_pos_scale", 1.0))
    if scale <= 1.0e-9:
        scale = 1.0
    axis_scale = _axis_scale_attr(base, "floating_action_pos_axis_scale")
    action = [0.0] * 16
    for axis in range(3):
        denom = scale * max(1.0e-9, abs(float(axis_scale[axis])))
        action[axis] = max(-1.0, min(1.0, float(_list_get(desired_delta_xyz, axis, 0.0)) / denom))
    return action


def _scalar_attr(base: Any, name: str, default: float) -> float:
    value = getattr(base, name, default)
    try:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)
    except Exception:
        return float(default)


def _axis_scale_attr(base: Any, name: str) -> list[float]:
    value = getattr(base, name, None)
    try:
        if hasattr(value, "detach"):
            rows = [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
        elif isinstance(value, (list, tuple)):
            rows = [float(v) for v in value]
        elif value is not None:
            rows = [float(value)]
        else:
            rows = []
    except Exception:
        rows = []
    if not rows:
        rows = [1.0]
    while len(rows) < 3:
        rows.append(rows[-1])
    return [float(rows[0]), float(rows[1]), float(rows[2])]


def _append_log(
    logs: list[dict[str, Any]],
    phase: str,
    step: int,
    state: dict[str, Any],
    cfg: ScriptedBaselineConfig,
    object_reset_pos: list[float],
    *,
    extra: dict[str, Any] | None = None,
    alignment: AlignmentRecorder | None = None,
    base: Any | None = None,
) -> None:
    object_pos = list(state["object_local_pos"])
    contact_target = list(state.get("contact_target_local_pos", []))
    normal = list(state.get("contact_target_normal_xyz", []))
    surface_point = list(state.get("surface_point_local_xyz", []))
    precontact = list(state.get("precontact_point_local_xyz", []))
    press_target = list(state.get("press_target_local_xyz", []))
    bbox_min = list(state.get("surface_bbox_min_local_xyz", []))
    bbox_max = list(state.get("surface_bbox_max_local_xyz", []))
    bbox_extent = list(state.get("surface_bbox_extent_xyz", []))
    closest_f3 = list(state.get("finger3_closest_surface_point_local_xyz", []))
    surface_dir_f3 = list(state.get("finger3_surface_direction_xyz", []))
    row = {
        "phase": phase,
        "step": int(step),
        "part": cfg.part,
        "env_index": 1,
        "object_x": object_pos[0],
        "object_y": object_pos[1],
        "object_z": object_pos[2],
        "finger3_tip_x": state["finger3_tip_local_pos"][0],
        "finger3_tip_y": state["finger3_tip_local_pos"][1],
        "finger3_tip_z": state["finger3_tip_local_pos"][2],
        "finger4_tip_x": state["finger4_tip_local_pos"][0],
        "finger4_tip_y": state["finger4_tip_local_pos"][1],
        "finger4_tip_z": state["finger4_tip_local_pos"][2],
        "finger3_unfiltered_force_n": state["finger3_unfiltered_force_n"],
        "finger4_unfiltered_force_n": state["finger4_unfiltered_force_n"],
        "finger3_target_filtered_force_n": state["finger3_target_filtered_force_n"],
        "finger4_target_filtered_force_n": state["finger4_target_filtered_force_n"],
        "active_target_filtered_force_peak_n": state["active_target_filtered_force_peak_n"],
        "active_unfiltered_force_peak_n": state["active_unfiltered_force_peak_n"],
        "object_displacement_from_reset_m": _distance(object_reset_pos, object_pos),
        "object_z_delta_from_reset_m": _vec_delta_z(object_reset_pos, object_pos),
        "workspace_clamp_delta_m": state["workspace_clamp_delta_m"],
        "table_barrier_delta_z_m": state["table_barrier_delta_z_m"],
        "workspace_clamp_x": state.get("workspace_clamp_x", 0.0),
        "workspace_clamp_y": state.get("workspace_clamp_y", 0.0),
        "workspace_clamp_z": state.get("workspace_clamp_z", 0.0),
        "workspace_clamp_dominant_axis": state.get("workspace_clamp_dominant_axis", ""),
        "wrist_target_pre_clamp_x": _list_get(state.get("wrist_target_pre_clamp_xyz", []), 0, ""),
        "wrist_target_pre_clamp_y": _list_get(state.get("wrist_target_pre_clamp_xyz", []), 1, ""),
        "wrist_target_pre_clamp_z": _list_get(state.get("wrist_target_pre_clamp_xyz", []), 2, ""),
        "wrist_target_post_clamp_x": _list_get(state.get("wrist_target_post_clamp_xyz", []), 0, ""),
        "wrist_target_post_clamp_y": _list_get(state.get("wrist_target_post_clamp_xyz", []), 1, ""),
        "wrist_target_post_clamp_z": _list_get(state.get("wrist_target_post_clamp_xyz", []), 2, ""),
        "wrist_target_delta_x": _list_get(state["wrist_target_delta_xyz"], 0),
        "wrist_target_delta_y": _list_get(state["wrist_target_delta_xyz"], 1),
        "wrist_target_delta_z": _list_get(state["wrist_target_delta_xyz"], 2),
        "hand_target_delta_l2": state["hand_target_delta_l2"],
        "hand_limiter_delta_l2": state["hand_limiter_delta_l2"],
        "last_dex_hand_limiter_applied": state["last_dex_hand_limiter_applied"],
        "last_dex_hand_limiter_bypassed_for_v95": state["last_dex_hand_limiter_bypassed_for_v95"],
        "target_filter_index": state["target_filter_index"],
        "target_filtered_force_available": state["target_filtered_force_available"],
        "contact_target_x": _list_get(contact_target, 0, ""),
        "contact_target_y": _list_get(contact_target, 1, ""),
        "contact_target_z": _list_get(contact_target, 2, ""),
        "contact_target_mode": state.get("contact_target_mode", ""),
        "contact_target_source": state.get("contact_target_source", ""),
        "surface_source": state.get("surface_source", ""),
        "surface_source_error": state.get("surface_source_error", ""),
        "surface_bbox_rebased_to_runtime_root": state.get("surface_bbox_rebased_to_runtime_root", ""),
        "surface_collision_prim_count": state.get("surface_collision_prim_count", ""),
        "surface_visual_prim_count": state.get("surface_visual_prim_count", ""),
        "contact_target_normal_x": _list_get(normal, 0, ""),
        "contact_target_normal_y": _list_get(normal, 1, ""),
        "contact_target_normal_z": _list_get(normal, 2, ""),
        "surface_point_x": _list_get(surface_point, 0, ""),
        "surface_point_y": _list_get(surface_point, 1, ""),
        "surface_point_z": _list_get(surface_point, 2, ""),
        "precontact_x": _list_get(precontact, 0, ""),
        "precontact_y": _list_get(precontact, 1, ""),
        "precontact_z": _list_get(precontact, 2, ""),
        "press_target_x": _list_get(press_target, 0, ""),
        "press_target_y": _list_get(press_target, 1, ""),
        "press_target_z": _list_get(press_target, 2, ""),
        "surface_bbox_min_x": _list_get(bbox_min, 0, ""),
        "surface_bbox_min_y": _list_get(bbox_min, 1, ""),
        "surface_bbox_min_z": _list_get(bbox_min, 2, ""),
        "surface_bbox_max_x": _list_get(bbox_max, 0, ""),
        "surface_bbox_max_y": _list_get(bbox_max, 1, ""),
        "surface_bbox_max_z": _list_get(bbox_max, 2, ""),
        "surface_bbox_extent_x": _list_get(bbox_extent, 0, ""),
        "surface_bbox_extent_y": _list_get(bbox_extent, 1, ""),
        "surface_bbox_extent_z": _list_get(bbox_extent, 2, ""),
        "finger3_distance_to_real_surface_m": state.get("finger3_distance_to_real_surface_m", ""),
        "finger4_distance_to_real_surface_m": state.get("finger4_distance_to_real_surface_m", ""),
        "finger3_closest_surface_point_x": _list_get(closest_f3, 0, ""),
        "finger3_closest_surface_point_y": _list_get(closest_f3, 1, ""),
        "finger3_closest_surface_point_z": _list_get(closest_f3, 2, ""),
        "finger3_surface_direction_x": _list_get(surface_dir_f3, 0, ""),
        "finger3_surface_direction_y": _list_get(surface_dir_f3, 1, ""),
        "finger3_surface_direction_z": _list_get(surface_dir_f3, 2, ""),
        "finger3_surface_distance_servo_m": state.get("finger3_surface_distance_servo_m", ""),
        "finger3_signed_distance_to_screw1_surface_m": state.get("finger3_signed_distance_to_screw1_surface_m", ""),
        "finger4_signed_distance_to_screw1_surface_m": state.get("finger4_signed_distance_to_screw1_surface_m", ""),
        "finger3_surface_press_depth_m": state.get("finger3_surface_press_depth_m", ""),
        "finger4_surface_press_depth_m": state.get("finger4_surface_press_depth_m", ""),
        "press_depth_target_m": state.get("press_depth_target_m", ""),
        "precontact_gap_m": state.get("precontact_gap_m", ""),
        "contact_target_radius_m": state.get("contact_target_radius_m", ""),
        "contact_target_table_safe_z_m": state.get("contact_target_table_safe_z_m", ""),
        "contact_target_z_clamped_to_bbox": state.get("contact_target_z_clamped_to_bbox", ""),
        "contact_target_z_bias_m": state.get("contact_target_z_bias_m", ""),
        "contact_target_side_override_used": state.get("contact_target_side_override_used", ""),
        "finger3_distance_to_contact_target_m": state.get("finger3_distance_to_contact_target_m", ""),
        "finger4_distance_to_contact_target_m": state.get("finger4_distance_to_contact_target_m", ""),
        "palm_distance_to_screw1_m": state.get("palm_distance_to_screw1_m", ""),
    }
    if extra:
        row.update(extra)
    if alignment is not None and alignment.enabled:
        frame_index = int(float(row.get("video_frame_index", -1)))
        alignment_extra = alignment.make_log_extra(frame_index)
        for key, value in alignment_extra.items():
            row.setdefault(key, value)
    logs.append(_plain(row))
    if alignment is not None and alignment.enabled and base is not None:
        alignment.record_contacts(phase, step, state, base)


def _pregrasp_distance_check(state: dict[str, Any], cfg: ScriptedBaselineConfig) -> dict[str, Any]:
    f3 = float(state.get("finger3_distance_to_contact_target_m", math.inf))
    f4 = float(state.get("finger4_distance_to_contact_target_m", math.inf))
    near = bool(f3 <= cfg.pregrasp_fail_distance_m and f4 <= cfg.pregrasp_fail_distance_m)
    goal = bool(f3 <= cfg.pregrasp_goal_distance_m and f4 <= cfg.pregrasp_goal_distance_m)
    return {
        "pregrasp_finger3_distance": f3,
        "pregrasp_finger4_distance": f4,
        "pregrasp_near_enough": near,
        "pregrasp_goal_met": goal,
    }


def _close_decision(approach: dict[str, Any], cfg: ScriptedBaselineConfig) -> dict[str, Any]:
    if bool(approach.get("target_contact_observed")):
        return {"execute": True, "close_trigger": "target_filtered_contact"}
    if bool(approach.get("near_surface_reached")) and bool(cfg.force_close_without_contact_for_debug):
        return {"execute": True, "close_trigger": "micro_close_debug_no_force"}
    return {"execute": False, "close_trigger": "not_executed"}


def _max_target_force_from_logs(logs: list[dict[str, Any]]) -> float:
    values = []
    for row in logs:
        for key in ("finger3_target_filtered_force_n", "finger4_target_filtered_force_n"):
            try:
                values.append(float(row.get(key, 0.0) or 0.0))
            except Exception:
                pass
    return max([0.0, *values])


def _classify_result(
    *,
    pregrasp: dict[str, Any],
    approach: dict[str, Any],
    close: dict[str, Any],
    lift: dict[str, Any],
    max_target: float,
    threshold: float,
    cfg: ScriptedBaselineConfig,
) -> tuple[str, str]:
    target_surface_distance = float(approach.get("press_target_surface_distance", 0.001) or 0.001)
    if not bool(pregrasp.get("pregrasp_near_enough")):
        return "COARSE_PREGRASP_NOT_REACHED", "finger3 did not reach the root-based contact-search entry region"
    if bool(approach.get("contact_refinement_executed")):
        return _contact_refinement_result_class(approach, cfg)
    if max_target > threshold:
        return "TARGET_CONTACT_ACQUIRED", "Screw1 target-filtered contact observed before close/lift"
    if bool(approach.get("contact_search_executed")):
        termination = str(approach.get("contact_search_termination_reason", approach.get("stop_reason", "")))
        if termination in {"raw_non_target_contact", "unfiltered_non_target_contact"}:
            return (
                "CONTACT_SEARCH_STOPPED_BY_NON_TARGET_CONTACT",
                "bounded contact search stopped on non-target or unresolved unfiltered contact before Screw1 target-filtered contact",
            )
        if termination == "workspace_or_table_clamp" or bool(approach.get("workspace_or_table_clamp_observed")):
            return (
                "CONTACT_SEARCH_STOPPED_BY_CLAMP",
                "bounded contact search stopped because workspace/table clamp appeared before Screw1 target-filtered contact",
            )
        if termination == "controller_no_motion":
            return (
                "CONTACT_SEARCH_CONTROLLER_NO_MOTION",
                "bounded contact search commanded motion but the finger stopped making measurable controller progress",
            )
        if termination == "raw_screw1_contact_without_target_filtered":
            return (
                "CONTACT_SEARCH_NO_TARGET_CONTACT",
                "raw Screw1 contact was observed, but finger3/finger4 target-filtered Screw1 force did not exceed threshold",
            )
        return (
            "CONTACT_SEARCH_NO_TARGET_CONTACT",
            "bounded contact search exhausted its lanes or step budget without Screw1 target-filtered contact",
        )
    if bool(approach.get("press_blocked_by_z_clamp")):
        return "PRESS_BLOCKED_BY_Z_CLAMP", "z clamp persisted during press preflight or micro-press, so press was not allowed to fake progress"
    if bool(approach.get("press_action_not_reaching_controller")):
        return (
            "PRESS_ACTION_NOT_REACHING_CONTROLLER",
            "micro-press policy/mapped action or controller target delta was zero",
        )
    if bool(approach.get("press_action_frame_wrong")):
        return "PRESS_ACTION_FRAME_WRONG", "micro-press action moved finger3 opposite the frozen Screw1 surface normal after sign audit"
    if bool(approach.get("press_controller_target_moved_tip_static")):
        return (
            "PRESS_CONTROLLER_TARGET_MOVED_TIP_STATIC",
            "controller target moved during press but finger3 tip did not advance along the Screw1 surface normal",
        )
    if (
        bool(approach.get("fallback_triggered"))
        and float(approach.get("ik_surface_reduction", 0.0) or 0.0) < 0.002
        and float(approach.get("press_min_surface_distance", math.inf)) > target_surface_distance
    ):
        return (
            "IK_PRESS_NO_SURFACE_PROGRESS",
            "IK-assisted press ran but did not reduce finger3 distance to Screw1 surface by at least 2mm",
        )
    if bool(approach.get("press_stage_entered")) and float(approach.get("press_min_surface_distance", math.inf)) <= target_surface_distance:
        return (
            "SURFACE_DISTANCE_CLOSE_NO_FORCE",
            "finger3 reached within the target Screw1 collision surface distance but target-filtered force stayed zero",
        )
    if (
        bool(approach.get("press_stage_entered"))
        and float(approach.get("surface_distance_reduction_during_press", 0.0) or 0.0) >= 0.002
    ):
        return (
            "PRESS_PROGRESS_NO_FORCE",
            "finger3 reduced distance to the Screw1 surface but target-filtered force stayed zero",
        )
    if bool(approach.get("precontact_reached")) and not bool(approach.get("press_stage_entered")):
        return (
            "PRECONTACT_REACHED_PRESS_NOT_ENTERED",
            "finger3 reached the precontact point, but the micro-press stage did not execute",
        )
    if bool(approach.get("press_stage_entered")) and not bool(approach.get("press_surface_progress")):
        return (
            "PRESS_CONTROLLER_TARGET_MOVED_TIP_STATIC",
            "micro-press executed but finger3 did not make measurable progress along the Screw1 surface normal",
        )
    if bool(approach.get("press_stage_entered")):
        return (
            "PRESS_PROGRESS_NO_FORCE",
            "finger3 made limited micro-press progress toward the Screw1 surface but target-filtered force stayed zero",
        )
    if str(approach.get("stop_reason")) == "surface_aligned_no_force" or bool(approach.get("near_surface_reached")):
        return "SURFACE_ALIGNED_NO_FORCE", "finger3 reached Screw1 bbox surface/micro-press target but target-filtered force stayed zero"
    if bool(close.get("executed")) and str(close.get("close_trigger")) == "near_surface_no_force":
        return "NEAR_SURFACE_NO_FORCE", "finger reached near Screw1 surface target but target-filtered force stayed zero"
    if bool(approach.get("workspace_or_table_clamp_observed")):
        if bool(approach.get("distance_reduced_during_approach")):
            return (
                "APPROACH_REDUCED_BUT_CLAMPED_NO_CONTACT",
                "approach reduced distance but workspace/table clamp stopped before near-surface or target contact",
            )
        return (
            "APPROACH_STOPPED_BY_WORKSPACE_OR_TABLE_CLAMP",
            "workspace/table clamp stopped approach before meaningful distance reduction or target contact",
        )
    if bool(approach.get("distance_reduced_during_approach")):
        return (
            "APPROACH_NOT_REDUCING_DISTANCE",
            "approach reduced distance but did not reach near-surface or target contact",
        )
    return "APPROACH_NOT_REDUCING_DISTANCE", "approach did not reduce distance enough to reach Screw1 surface target"


def _next_baseline_fix(
    result_class: str,
    pregrasp: dict[str, Any],
    approach: dict[str, Any],
) -> str:
    f3 = float(pregrasp.get("pregrasp_finger3_distance", math.inf))
    f4 = float(pregrasp.get("pregrasp_finger4_distance", math.inf))
    if result_class == "STABLE_TWO_FINGER_CONTACT_ACQUIRED":
        return "close strategy later; keep contact-only baseline locked"
    if result_class == "TWO_FINGER_CONTACT_OBJECT_MOVED":
        return "force-limited approach/retraction; reduce object displacement"
    if result_class == "SINGLE_FINGER_CONTACT_ONLY":
        return "finger opposition geometry around the two contact seeds"
    if result_class == "TWO_SEED_REFINEMENT_FORCE_LIMIT":
        return "force-limited approach/retraction and finger opposition geometry"
    if result_class == "TWO_SEED_REFINEMENT_NO_TARGET_CONTACT":
        return "two-seed search ordering / hand orientation"
    if result_class == "TWO_SEED_REFINEMENT_STOPPED_BY_NON_TARGET_CONTACT":
        return "refinement bounds / non-target avoidance"
    if result_class in {"FINGER4_ONLY_LOW_FORCE_CONTACT", "FINGER3_ONLY_LOW_FORCE_CONTACT"}:
        return "opposing finger support / hand orientation"
    if result_class == "TARGET_CONTACT_HIGH_FORCE_ONLY":
        return "reduce penetration / refine contact pose"
    if result_class == "CONTACT_REFINEMENT_NO_TARGET_CONTACT":
        return "refinement seed / hand orientation"
    if result_class == "CONTACT_REFINEMENT_STOPPED_BY_FORCE_LIMIT":
        return "reduce penetration / force-limited approach"
    if result_class == "CONTACT_REFINEMENT_STOPPED_BY_NON_TARGET_CONTACT":
        return "refinement bounds / non-target avoidance"
    if result_class == "TARGET_CONTACT_ACQUIRED":
        return "finger4 support / close later"
    if result_class == "COARSE_PREGRASP_NOT_REACHED":
        return "pregrasp"
    if result_class == "CONTACT_SEARCH_NO_TARGET_CONTACT":
        return "contact-driven search volume / hand orientation"
    if result_class == "CONTACT_SEARCH_STOPPED_BY_NON_TARGET_CONTACT":
        return "search bounds / hand orientation"
    if result_class == "CONTACT_SEARCH_STOPPED_BY_CLAMP":
        return "search height / controller limits"
    if result_class == "CONTACT_SEARCH_CONTROLLER_NO_MOTION":
        return "IK/controller response"
    if result_class == "PREGRASP_NOT_NEAR_TARGET":
        if f3 <= 0.05 < f4:
            return "hand orientation"
        return "pregrasp"
    if result_class in {"APPROACH_NOT_REDUCING_DISTANCE", "APPROACH_REDUCED_BUT_CLAMPED_NO_CONTACT", "APPROACH_STOPPED_BY_WORKSPACE_OR_TABLE_CLAMP"}:
        min_f3 = float(approach.get("min_finger3_distance_to_contact_target", math.inf))
        if bool(approach.get("distance_reduced_during_approach")) and min_f3 <= 0.03:
            return "contact target surface"
        if bool(approach.get("workspace_or_table_clamp_observed")):
            return "contact target surface"
        return "approach gain"
    if result_class == "NEAR_SURFACE_NO_FORCE":
        return "contact target surface"
    if result_class == "SURFACE_ALIGNED_NO_FORCE":
        return "contact target surface"
    if result_class == "PRECONTACT_REACHED_PRESS_NOT_ENTERED":
        return "press trigger"
    if result_class == "PRESS_BLOCKED_BY_Z_CLAMP":
        return "z clamp / press target height"
    if result_class == "PRESS_ACTION_NOT_REACHING_CONTROLLER":
        return "action mapper/controller"
    if result_class == "PRESS_ACTION_FRAME_WRONG":
        return "action frame/sign"
    if result_class == "PRESS_CONTROLLER_TARGET_MOVED_TIP_STATIC":
        return "IK/controller response"
    if result_class == "IK_PRESS_NO_SURFACE_PROGRESS":
        return "IK/controller response"
    if result_class == "SURFACE_DISTANCE_CLOSE_NO_FORCE":
        return "hand orientation / fingertip collision geometry / sensor coverage"
    if result_class == "PRESS_PROGRESS_NO_FORCE":
        return "press depth / hand orientation"
    if result_class == "PRESS_ENTERED_NO_SURFACE_PROGRESS":
        return "micro-press action direction/scale"
    if result_class == "PRESS_ENTERED_NO_FORCE":
        return "contact target surface"
    if bool(approach.get("near_surface_reached")) and f4 > f3 + 0.03:
        return "hand orientation"
    return "hand orientation"


def _save_trajectory_plot(path: Path, logs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [float(row.get("finger3_tip_x", 0.0)) for row in logs]
        ys = [float(row.get("finger3_tip_y", 0.0)) for row in logs]
        zs = [float(row.get("finger3_tip_z", 0.0)) for row in logs]
        x4 = [float(row.get("finger4_tip_x", 0.0)) for row in logs]
        y4 = [float(row.get("finger4_tip_y", 0.0)) for row in logs]
        ox = [float(row.get("object_x", 0.0)) for row in logs]
        oy = [float(row.get("object_y", 0.0)) for row in logs]
        phases = [str(row.get("phase", "")) for row in logs]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(xs, ys, label="finger3 tip")
        axes[0].plot(x4, y4, label="finger4 tip")
        axes[0].plot(ox, oy, label="Screw1")
        axes[0].set_xlabel("local x (m)")
        axes[0].set_ylabel("local y (m)")
        axes[0].axis("equal")
        axes[0].legend(loc="best")
        axes[1].plot(zs, label="finger3 z")
        axes[1].plot([float(row.get("object_z", 0.0)) for row in logs], label="Screw1 z")
        axes[1].set_xlabel("log step")
        axes[1].set_ylabel("local z (m)")
        last_phase = ""
        for index, phase in enumerate(phases):
            if phase != last_phase:
                for axis in axes:
                    axis.axvline(index, color="0.8", linewidth=0.8)
                last_phase = phase
        axes[1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"trajectory_plot_failed:{type(exc).__name__}:{exc}\n", encoding="utf-8")


def _save_final_frame(env: Any, path: Path) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return {"final_frame_png": "", "final_frame_error": f"cv2_or_numpy_unavailable:{exc!r}"}
    try:
        frame = env.render()
        if isinstance(frame, tuple):
            frame = frame[0]
        if frame is None:
            return {"final_frame_png": "", "final_frame_error": "render_returned_none"}
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] < 3:
            return {"final_frame_png": "", "final_frame_error": f"unexpected_frame_shape:{frame.shape}"}
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return {"final_frame_png": str(path) if ok else "", "final_frame_error": "" if ok else "cv2_imwrite_failed"}
    except Exception as exc:
        return {"final_frame_png": "", "final_frame_error": f"{type(exc).__name__}:{exc}"}


def _finalize_video_artifact(video_recorder: Any | None, output_dir: Path) -> dict[str, Any]:
    if video_recorder is None:
        return {
            "video_available": False,
            "video_path": "",
            "video_unavailable_reason": "record_video_false",
        }
    diagnostics: dict[str, Any] = {}
    try:
        close_video = getattr(video_recorder, "close_video_recorder", None)
        if callable(close_video):
            close_video()
        diag_fn = getattr(video_recorder, "diagnostics", None)
        if callable(diag_fn):
            diagnostics = dict(diag_fn() or {})
    except Exception as exc:
        diagnostics = {"video_write_failed": True, "video_write_failure_reason": f"{type(exc).__name__}:{exc}"}
    raw_value = diagnostics.get("raw_video_path") or diagnostics.get("streaming_raw_video_path") or ""
    if not raw_value:
        recorder_raw = getattr(video_recorder, "raw_video_path", "")
        raw_value = str(recorder_raw) if recorder_raw else ""
    raw_path = Path(str(raw_value)) if raw_value else Path()
    if raw_path and not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    desired = output_dir / "videos" / "scripted_screw1_baseline.mp4"
    failure = str(diagnostics.get("video_write_failure_reason") or "")
    if raw_path.is_file():
        try:
            desired.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.resolve() != desired.resolve():
                shutil.copy2(raw_path, desired)
            if desired.is_file():
                transcode = _transcode_video_to_h264(desired)
                return {
                    "video_available": True,
                    "video_path": str(desired),
                    "video_raw_path": str(raw_path),
                    "video_unavailable_reason": "",
                    "video_diagnostics": diagnostics,
                    **transcode,
                }
        except Exception as exc:
            failure = f"video_copy_failed:{type(exc).__name__}:{exc}"
    if not failure:
        failure = "streaming_video_file_missing"
    return {
        "video_available": False,
        "video_path": "",
        "video_raw_path": str(raw_path) if raw_path else "",
        "video_unavailable_reason": failure,
        "video_diagnostics": diagnostics,
    }


def _transcode_video_to_h264(path: Path) -> dict[str, Any]:
    """Rewrite the public mp4 as browser-friendly H.264 when ffmpeg is available."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            return {"video_h264_transcoded": False, "video_h264_error": f"ffmpeg_unavailable:{type(exc).__name__}:{exc}"}
    tmp = path.with_name(f"{path.stem}_h264_tmp{path.suffix}")
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(path),
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
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["unknown_error"]
        return {"video_h264_transcoded": False, "video_h264_error": tail[0]}
    tmp.replace(path)
    return {"video_h264_transcoded": True, "video_h264_error": ""}


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


def _refresh_runtime(base: Any) -> None:
    try:
        base._compute_intermediate_values(dt=getattr(base, "physics_dt", 1.0 / 120.0))
    except Exception:
        pass
    try:
        if hasattr(base, "_refresh_v95_active_object_tensors"):
            base._refresh_v95_active_object_tensors()
    except Exception:
        pass


def _capture_video_frame(video_recorder: Any | None, alignment: AlignmentRecorder | None = None) -> int:
    if alignment is not None and alignment.enabled:
        return alignment.capture_frame()
    if video_recorder is None:
        return -1
    try:
        write_frame = getattr(video_recorder, "_write_frame", None)
        if callable(write_frame):
            write_frame()
    except Exception:
        pass
    return -1


def _hold_video_frames(video_recorder: Any | None, alignment: AlignmentRecorder | None, count: int) -> int:
    captured = 0
    for _ in range(max(0, int(count))):
        frame_index = _capture_video_frame(video_recorder, alignment)
        if frame_index >= 0:
            captured += 1
    return captured


def _action_dim(env: Any, base: Any) -> int:
    try:
        shape = getattr(getattr(env, "action_space", None), "shape", None)
        if shape:
            return int(shape[-1])
    except Exception:
        pass
    return int(getattr(base, "num_actions", 26) or 26)


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _lateral_approach_axis(state: dict[str, Any]) -> list[float]:
    raw = _sub_vec(state["object_local_pos"], state["finger3_tip_local_pos"])
    raw[2] = 0.0
    axis = _unit(raw)
    if _norm(axis) <= 1.0e-6:
        return [1.0, 0.0, 0.0]
    return axis


def _state_with_contact_target(
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    state: dict[str, Any],
    adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(state)
    target_info = _screw1_surface_contact_target(base, env_index, cfg, out, adjustment=adjustment)
    out.update(target_info)
    target = out["contact_target_local_pos"]
    out["finger3_distance_to_contact_target_m"] = _distance(out["finger3_tip_local_pos"], target)
    out["finger4_distance_to_contact_target_m"] = _distance(out["finger4_tip_local_pos"], target)
    bbox_min = list(out.get("surface_bbox_min_local_xyz", []) or [])
    bbox_max = list(out.get("surface_bbox_max_local_xyz", []) or [])
    normal = list(out.get("contact_target_normal_xyz", [1.0, 0.0, 0.0]) or [1.0, 0.0, 0.0])
    surface_point = list(out.get("surface_point_local_xyz", out["object_local_pos"]) or out["object_local_pos"])
    f3_signed = _signed_distance_to_aabb_bounds(out["finger3_tip_local_pos"], bbox_min, bbox_max)
    f4_signed = _signed_distance_to_aabb_bounds(out["finger4_tip_local_pos"], bbox_min, bbox_max)
    out["finger3_signed_distance_to_screw1_surface_m"] = f3_signed
    out["finger4_signed_distance_to_screw1_surface_m"] = f4_signed
    out["finger3_distance_to_real_surface_m"] = max(0.0, f3_signed)
    out["finger4_distance_to_real_surface_m"] = max(0.0, f4_signed)
    f3_surface_target = _surface_distance_servo_target(out, finger_key="finger3_tip_local_pos")
    f4_surface_target = _surface_distance_servo_target(out, finger_key="finger4_tip_local_pos")
    out["finger3_closest_surface_point_local_xyz"] = f3_surface_target["closest_surface_point_local_xyz"]
    out["finger3_surface_direction_xyz"] = f3_surface_target["surface_direction_xyz"]
    out["finger3_surface_distance_servo_m"] = f3_surface_target["surface_distance_m"]
    out["finger4_closest_surface_point_local_xyz"] = f4_surface_target["closest_surface_point_local_xyz"]
    out["finger4_surface_direction_xyz"] = f4_surface_target["surface_direction_xyz"]
    out["finger4_surface_distance_servo_m"] = f4_surface_target["surface_distance_m"]
    out["finger3_surface_press_depth_m"] = max(0.0, _dot(_sub_vec(surface_point, out["finger3_tip_local_pos"]), normal))
    out["finger4_surface_press_depth_m"] = max(0.0, _dot(_sub_vec(surface_point, out["finger4_tip_local_pos"]), normal))
    out["palm_distance_to_screw1_m"] = _distance(out["palm_local_pos"], out["object_local_pos"])
    return out


def _screw1_surface_contact_target(
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    state: dict[str, Any],
    *,
    adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adjustment = dict(adjustment or {})
    target_mode = str(adjustment.get("target_mode", "precontact") or "precontact")
    surface_lock = adjustment.get("surface_lock")
    if isinstance(surface_lock, dict) and surface_lock:
        normal = _unit(list(surface_lock.get("contact_target_normal_xyz", [1.0, 0.0, 0.0]) or [1.0, 0.0, 0.0]))
        if _norm(normal) <= 1.0e-6:
            normal = [1.0, 0.0, 0.0]
        precontact_point = list(surface_lock.get("precontact_point_local_xyz", []) or [])
        press_target = list(surface_lock.get("press_target_local_xyz", []) or [])
        contact_target = press_target if target_mode == "press" and press_target else precontact_point
        if not contact_target:
            contact_target = list(surface_lock.get("surface_point_local_xyz", state["object_local_pos"]) or state["object_local_pos"])
        return {
            "contact_target_local_pos": contact_target,
            "contact_target_mode": target_mode,
            "contact_target_side_dir_xy": [normal[0], normal[1], 0.0],
            "contact_target_normal_xyz": [normal[0], normal[1], normal[2]],
            "contact_target_radius_m": float(surface_lock.get("contact_target_radius_m", 0.0) or 0.0),
            "contact_target_table_safe_z_m": surface_lock.get("contact_target_table_safe_z_m", ""),
            "contact_target_z_bias_m": surface_lock.get("contact_target_z_bias_m", 0.0),
            "contact_target_side_override_used": surface_lock.get("contact_target_side_override_used", False),
            "contact_target_z_clamped_to_bbox": surface_lock.get("contact_target_z_clamped_to_bbox", False),
            "contact_target_source": f"screw1_{surface_lock.get('surface_source', 'unknown')}_{target_mode}_locked",
            "surface_source": surface_lock.get("surface_source", ""),
            "surface_source_error": surface_lock.get("surface_source_error", ""),
            "surface_bbox_rebased_to_runtime_root": surface_lock.get("surface_bbox_rebased_to_runtime_root", False),
            "surface_bbox_stage_center_local_xyz": surface_lock.get("surface_bbox_stage_center_local_xyz", []),
            "surface_collision_prim_count": surface_lock.get("surface_collision_prim_count", 0),
            "surface_visual_prim_count": surface_lock.get("surface_visual_prim_count", 0),
            "surface_bbox_min_local_xyz": surface_lock.get("surface_bbox_min_local_xyz", []),
            "surface_bbox_max_local_xyz": surface_lock.get("surface_bbox_max_local_xyz", []),
            "surface_bbox_center_local_xyz": surface_lock.get("surface_bbox_center_local_xyz", []),
            "surface_bbox_extent_xyz": surface_lock.get("surface_bbox_extent_xyz", []),
            "surface_bbox_min_world_xyz": surface_lock.get("surface_bbox_min_world_xyz", []),
            "surface_bbox_max_world_xyz": surface_lock.get("surface_bbox_max_world_xyz", []),
            "surface_bbox_center_world_xyz": surface_lock.get("surface_bbox_center_world_xyz", []),
            "surface_point_local_xyz": surface_lock.get("surface_point_local_xyz", []),
            "precontact_point_local_xyz": precontact_point,
            "press_target_local_xyz": press_target,
            "press_depth_target_m": float(surface_lock.get("press_depth_target_m", cfg.surface_press_depth_m) or cfg.surface_press_depth_m),
            "precontact_gap_m": float(surface_lock.get("precontact_gap_m", cfg.surface_precontact_gap_m) or cfg.surface_precontact_gap_m),
        }
    object_pos = list(state["object_local_pos"])
    tip3 = list(state["finger3_tip_local_pos"])
    geometry = _screw1_surface_geometry(base, env_index, cfg, state)
    bbox_min = list(geometry["surface_bbox_min_local_xyz"])
    bbox_max = list(geometry["surface_bbox_max_local_xyz"])
    bbox_center = list(geometry["surface_bbox_center_local_xyz"])
    bbox_extent = list(geometry["surface_bbox_extent_xyz"])
    side_raw = [tip3[0] - bbox_center[0], tip3[1] - bbox_center[1], 0.0]
    side_dir = _unit(side_raw)
    if _norm(side_dir) <= 1.0e-6:
        side_dir = [1.0, 0.0, 0.0]
    override = adjustment.get("side_dir_override")
    if isinstance(override, (list, tuple)) and len(override) >= 2:
        candidate = _unit([float(override[0]), float(override[1]), 0.0])
        if _norm(candidate) > 1.0e-6:
            side_dir = candidate
    table_z = _table_z_estimate(base, env_index, object_pos)
    table_safe_z = float(table_z) + float(cfg.surface_table_clearance_m)
    if float(adjustment.get("z_bias_m", 0.0) or 0.0) > 0.0:
        table_safe_z = max(table_safe_z, float(table_z) + float(cfg.clamp_adjust_min_table_clearance_m))
    target_z = max(float(bbox_center[2]), table_safe_z)
    target_z += float(adjustment.get("z_bias_m", 0.0) or 0.0)
    z_clamped_to_bbox = False
    if len(bbox_min) >= 3 and len(bbox_max) >= 3 and bbox_max[2] > bbox_min[2]:
        upper = float(bbox_max[2]) - min(0.001, max(0.0, (float(bbox_max[2]) - float(bbox_min[2])) * 0.15))
        lower = float(bbox_min[2]) + min(0.001, max(0.0, (float(bbox_max[2]) - float(bbox_min[2])) * 0.15))
        clamped_z = min(max(target_z, lower), upper)
        z_clamped_to_bbox = bool(abs(clamped_z - target_z) > 1.0e-6)
        target_z = clamped_z
    radius = _aabb_ray_radius_xy(bbox_min, bbox_max, bbox_center, side_dir)
    if radius <= 1.0e-6:
        radius = float(cfg.screw1_surface_radius_m)
    surface_point = [
        float(bbox_center[0]) + side_dir[0] * radius,
        float(bbox_center[1]) + side_dir[1] * radius,
        target_z,
    ]
    precontact_point = [
        surface_point[0] + side_dir[0] * float(cfg.surface_precontact_gap_m),
        surface_point[1] + side_dir[1] * float(cfg.surface_precontact_gap_m),
        surface_point[2],
    ]
    press_target = [
        surface_point[0] - side_dir[0] * float(cfg.surface_press_depth_m),
        surface_point[1] - side_dir[1] * float(cfg.surface_press_depth_m),
        surface_point[2],
    ]
    contact_target = press_target if target_mode == "press" else precontact_point
    return {
        "contact_target_local_pos": contact_target,
        "contact_target_mode": target_mode,
        "contact_target_side_dir_xy": [side_dir[0], side_dir[1], 0.0],
        "contact_target_normal_xyz": [side_dir[0], side_dir[1], 0.0],
        "contact_target_radius_m": radius,
        "contact_target_table_safe_z_m": table_safe_z,
        "contact_target_z_bias_m": float(adjustment.get("z_bias_m", 0.0) or 0.0),
        "contact_target_side_override_used": bool(isinstance(override, (list, tuple)) and len(override) >= 2),
        "contact_target_z_clamped_to_bbox": z_clamped_to_bbox,
        "contact_target_source": f"screw1_{geometry['surface_source']}_{target_mode}",
        "surface_source": geometry["surface_source"],
        "surface_source_error": geometry.get("surface_source_error", ""),
        "surface_bbox_rebased_to_runtime_root": geometry.get("surface_bbox_rebased_to_runtime_root", False),
        "surface_bbox_stage_center_local_xyz": geometry.get("surface_bbox_stage_center_local_xyz", []),
        "surface_collision_prim_count": geometry.get("surface_collision_prim_count", 0),
        "surface_visual_prim_count": geometry.get("surface_visual_prim_count", 0),
        "surface_bbox_min_local_xyz": bbox_min,
        "surface_bbox_max_local_xyz": bbox_max,
        "surface_bbox_center_local_xyz": bbox_center,
        "surface_bbox_extent_xyz": bbox_extent,
        "surface_bbox_min_world_xyz": geometry.get("surface_bbox_min_world_xyz", []),
        "surface_bbox_max_world_xyz": geometry.get("surface_bbox_max_world_xyz", []),
        "surface_bbox_center_world_xyz": geometry.get("surface_bbox_center_world_xyz", []),
        "surface_point_local_xyz": surface_point,
        "precontact_point_local_xyz": precontact_point,
        "press_target_local_xyz": press_target,
        "press_depth_target_m": float(cfg.surface_press_depth_m),
        "precontact_gap_m": float(cfg.surface_precontact_gap_m),
    }


def _surface_lock_from_state(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "contact_target_normal_xyz",
        "contact_target_radius_m",
        "contact_target_table_safe_z_m",
        "contact_target_z_bias_m",
        "contact_target_side_override_used",
        "contact_target_z_clamped_to_bbox",
        "surface_source",
        "surface_source_error",
        "surface_bbox_rebased_to_runtime_root",
        "surface_bbox_stage_center_local_xyz",
        "surface_collision_prim_count",
        "surface_visual_prim_count",
        "surface_bbox_min_local_xyz",
        "surface_bbox_max_local_xyz",
        "surface_bbox_center_local_xyz",
        "surface_bbox_extent_xyz",
        "surface_bbox_min_world_xyz",
        "surface_bbox_max_world_xyz",
        "surface_bbox_center_world_xyz",
        "surface_point_local_xyz",
        "precontact_point_local_xyz",
        "press_target_local_xyz",
        "press_depth_target_m",
        "precontact_gap_m",
    )
    return {key: _plain(state.get(key, "")) for key in keys}


def _screw1_surface_geometry(
    base: Any,
    env_index: int,
    cfg: ScriptedBaselineConfig,
    state: dict[str, Any],
) -> dict[str, Any]:
    object_pos = list(state.get("object_local_pos", [0.0, 0.0, 0.0]))
    origin = _tensor_vec(getattr(getattr(base, "scene", None), "env_origins", None), env_index)
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        root_path = f"/World/envs/env_{int(env_index)}/Screw1"
        root = stage.GetPrimAtPath(root_path)
        if root is None or not root.IsValid():
            raise RuntimeError(f"screw1_root_prim_missing:{root_path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        collision_prims = []
        visual_prims = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) and "collision" not in path.lower():
                visual_prims.append(prim)
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if collision_enabled:
                collision_prims.append(prim)
        collision_box = _bbox_for_prims(cache, collision_prims)
        visual_box = _bbox_for_prims(cache, visual_prims)
        if _bbox_has_extent(collision_box):
            return _surface_geometry_from_world_box(
                collision_box,
                origin,
                object_pos,
                "collision_bbox",
                collision_count=len(collision_prims),
                visual_count=len(visual_prims),
            )
        if _bbox_has_extent(visual_box):
            return _surface_geometry_from_world_box(
                visual_box,
                origin,
                object_pos,
                "visual_bbox",
                collision_count=len(collision_prims),
                visual_count=len(visual_prims),
            )
        return _fixed_surface_geometry(
            object_pos,
            origin,
            cfg,
            error="bbox_empty",
            collision_count=len(collision_prims),
            visual_count=len(visual_prims),
        )
    except Exception as exc:
        return _fixed_surface_geometry(
            object_pos,
            origin,
            cfg,
            error=f"{type(exc).__name__}:{exc}",
            collision_count=0,
            visual_count=0,
        )


def _finger3_tip_collision_geometry(base: Any, env_index: int, state: dict[str, Any]) -> dict[str, Any]:
    origin = _tensor_vec(getattr(getattr(base, "scene", None), "env_origins", None), env_index)
    tip = list(state.get("finger3_tip_local_pos", []) or [])
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        root_path = f"/World/envs/env_{int(env_index)}/Robot/right_finger3_tip_link"
        root = stage.GetPrimAtPath(root_path)
        if root is None or not root.IsValid():
            raise RuntimeError(f"finger3_tip_prim_missing:{root_path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        collision_prims = []
        for prim in Usd.PrimRange(root):
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if collision_enabled:
                collision_prims.append(prim)
        box = _bbox_for_prims(cache, collision_prims)
        if not _bbox_has_extent(box):
            raise RuntimeError(f"finger3_tip_collision_bbox_empty:{len(collision_prims)}")
        local_min = _sub_vec(list(box["min"]), origin)
        local_max = _sub_vec(list(box["max"]), origin)
        local_center = _sub_vec(list(box["center"]), origin)
        extent = list(box["extent"])
        stage_center = list(local_center)
        rebase = bool(len(tip) >= 3 and _distance(local_center, tip) > 0.08 and len(extent) >= 3)
        if rebase:
            half = [float(value) * 0.5 for value in extent[:3]]
            local_center = list(tip[:3])
            local_min = [float(local_center[index]) - half[index] for index in range(3)]
            local_max = [float(local_center[index]) + half[index] for index in range(3)]
        closest_tip_surface = _closest_point_on_aabb(tip, local_min, local_max)
        origin_offset = _distance(tip, closest_tip_surface)
        screw_surface_gap = float(
            state.get("finger3_surface_distance_servo_m", state.get("finger3_distance_to_real_surface_m", math.inf))
        )
        collision_surface_gap = max(0.0, screw_surface_gap - origin_offset) if math.isfinite(screw_surface_gap) else math.inf
        return {
            "finger3_tip_collision_geometry_available": True,
            "finger_tip_collision_geometry_unavailable": "",
            "finger3_tip_collision_geometry_source": "collision_bbox",
            "finger3_tip_collision_prim_count": int(len(collision_prims)),
            "finger3_tip_collision_bbox_rebased_to_runtime_tip": rebase,
            "finger3_tip_collision_bbox_stage_center_local_xyz": stage_center,
            "finger3_tip_collision_bbox_min_local_xyz": local_min,
            "finger3_tip_collision_bbox_max_local_xyz": local_max,
            "finger3_tip_collision_bbox_center_local_xyz": local_center,
            "finger3_tip_collision_bbox_extent_xyz": extent,
            "finger3_tip_origin_to_collision_surface_offset_m": origin_offset,
            "finger3_collision_surface_to_screw1_surface_gap_m": collision_surface_gap,
        }
    except Exception as exc:
        return {
            "finger3_tip_collision_geometry_available": False,
            "finger_tip_collision_geometry_unavailable": f"{type(exc).__name__}:{exc}",
            "finger3_tip_collision_geometry_source": "",
            "finger3_tip_collision_prim_count": 0,
            "finger3_tip_collision_bbox_rebased_to_runtime_tip": False,
            "finger3_tip_collision_bbox_stage_center_local_xyz": [],
            "finger3_tip_collision_bbox_min_local_xyz": [],
            "finger3_tip_collision_bbox_max_local_xyz": [],
            "finger3_tip_collision_bbox_center_local_xyz": [],
            "finger3_tip_collision_bbox_extent_xyz": [],
            "finger3_tip_origin_to_collision_surface_offset_m": "",
            "finger3_collision_surface_to_screw1_surface_gap_m": "",
        }


def _screw1_collision_truth_metadata(base: Any, env_index: int, state: dict[str, Any]) -> dict[str, Any]:
    object_pos = list(state.get("object_local_pos", [0.0, 0.0, 0.0]))
    out: dict[str, Any] = {
        "screw1_root_prim_path": f"/World/envs/env_{int(env_index)}/Screw1",
        "screw1_visual_prim_paths": [],
        "screw1_collision_prim_paths": [],
        "screw1_collision_shape_count": 0,
        "screw1_collision_shape_types": [],
        "screw1_collision_api_enabled": [],
        "screw1_collision_bounds_source": state.get("surface_source", ""),
        "screw1_runtime_root_local_xyz": object_pos,
        "screw1_bbox_rebase_delta_m": "",
        "true_collision_surface_available": False,
        "true_collision_surface_unavailable_reason": "not_evaluated",
    }
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        root = stage.GetPrimAtPath(out["screw1_root_prim_path"])
        if root is None or not root.IsValid():
            raise RuntimeError(f"screw1_root_prim_missing:{out['screw1_root_prim_path']}")
        collision_paths: list[str] = []
        collision_types: list[str] = []
        collision_enabled: list[bool] = []
        visual_paths: list[str] = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Mesh) and "collision" not in path.lower():
                visual_paths.append(path)
            collision_api = UsdPhysics.CollisionAPI(prim)
            enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if enabled:
                approx = ""
                for attr_name in ("physics:approximation", "physxCollision:approximation"):
                    try:
                        attr = prim.GetAttribute(attr_name)
                        value = attr.Get() if attr and attr.IsValid() else None
                        if value:
                            approx = str(value)
                            break
                    except Exception:
                        pass
                type_name = str(prim.GetTypeName() or prim.GetPrimTypeInfo().GetTypeName() or "")
                collision_paths.append(path)
                collision_types.append(f"{type_name}{':' + approx if approx else ''}")
                collision_enabled.append(enabled)
        stage_center = list(state.get("surface_bbox_stage_center_local_xyz", []) or [])
        rebase_delta = _distance(stage_center, object_pos) if len(stage_center) >= 3 else math.inf
        primitive_like = bool(
            collision_types
            and all(not item.lower().startswith("mesh") for item in collision_types)
            and rebase_delta <= 0.01
            and str(state.get("surface_source", "")) == "collision_bbox"
        )
        reason = ""
        if not collision_paths:
            reason = "no_enabled_collision_prims_found_under_screw1"
        elif any(item.lower().startswith("mesh") for item in collision_types):
            reason = "collision_bounds_are_mesh_bbox_not_verified_against_physx_runtime_convex_surface"
        elif rebase_delta > 0.01:
            reason = "collision_bbox_was_rebased_to_runtime_root_so_stage_bounds_do_not_directly_match_runtime_shape"
        elif str(state.get("surface_source", "")) != "collision_bbox":
            reason = f"surface_source_is_{state.get('surface_source', '')}"
        out.update(
            {
                "screw1_visual_prim_paths": visual_paths,
                "screw1_collision_prim_paths": collision_paths,
                "screw1_collision_shape_count": len(collision_paths),
                "screw1_collision_shape_types": collision_types,
                "screw1_collision_api_enabled": collision_enabled,
                "screw1_bbox_rebase_delta_m": rebase_delta,
                "true_collision_surface_available": primitive_like,
                "true_collision_surface_unavailable_reason": "" if primitive_like else reason,
            }
        )
    except Exception as exc:
        out["true_collision_surface_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
    return out


def _finger3_body_chain_truth_metadata(base: Any, env_index: int, state: dict[str, Any]) -> dict[str, Any]:
    pose_names = list(getattr(base, "dex_fingertip_true_body_names", []) or [])
    pose_body = pose_names[FINGER3_INDEX] if len(pose_names) > FINGER3_INDEX else ""
    sensor_body = "right_finger3_tip_link"
    robot = getattr(base, "_robot", None)
    body_names = list(getattr(robot, "body_names", []) or [])
    rigid_index = body_names.index(sensor_body) if sensor_body in body_names else -1
    root_path = f"/World/envs/env_{int(env_index)}/Robot/{sensor_body}"
    collision_paths: list[str] = []
    collision_types: list[str] = []
    collision_body = ""
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdPhysics  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        root = stage.GetPrimAtPath(root_path)
        if root is None or not root.IsValid():
            raise RuntimeError(f"finger3_tip_prim_missing:{root_path}")
        for prim in Usd.PrimRange(root):
            collision_api = UsdPhysics.CollisionAPI(prim)
            enabled = bool(collision_api.GetCollisionEnabledAttr().Get()) if collision_api else False
            if enabled:
                path = str(prim.GetPath())
                collision_paths.append(path)
                collision_types.append(str(prim.GetTypeName() or ""))
        if collision_paths:
            collision_body = sensor_body
    except Exception as exc:
        return {
            "finger3_pose_body": pose_body,
            "finger3_sensor_body": sensor_body,
            "finger3_collision_body": "",
            "finger3_logical_index": FINGER3_INDEX,
            "finger3_rigid_body_index": rigid_index,
            "finger3_collision_prim_paths": collision_paths,
            "finger3_collision_shape_count": 0,
            "finger3_collision_shape_types": [],
            "pose_body_equals_collision_body": False,
            "collision_body_equals_sensor_body": False,
            "finger3_body_chain_error": f"{type(exc).__name__}:{exc}",
        }
    return {
        "finger3_pose_body": pose_body,
        "finger3_sensor_body": sensor_body,
        "finger3_collision_body": collision_body,
        "finger3_logical_index": FINGER3_INDEX,
        "finger3_rigid_body_index": rigid_index,
        "finger3_collision_prim_paths": collision_paths,
        "finger3_collision_shape_count": len(collision_paths),
        "finger3_collision_shape_types": collision_types,
        "pose_body_equals_collision_body": bool(pose_body and collision_body and pose_body == collision_body),
        "collision_body_equals_sensor_body": bool(collision_body and collision_body == sensor_body),
        "finger3_body_chain_error": "",
    }


def _surface_geometry_from_world_box(
    box: dict[str, list[float]],
    origin: list[float],
    runtime_object_pos: list[float],
    source: str,
    *,
    collision_count: int,
    visual_count: int,
) -> dict[str, Any]:
    world_min = list(box.get("min", []))
    world_max = list(box.get("max", []))
    world_center = list(box.get("center", []))
    extent = list(box.get("extent", []))
    local_min = _sub_vec(world_min, origin)
    local_max = _sub_vec(world_max, origin)
    local_center = _sub_vec(world_center, origin)
    stage_center = list(local_center)
    rebase = bool(_distance(local_center, runtime_object_pos) > 0.08)
    if rebase and len(extent) >= 3:
        half = [float(value) * 0.5 for value in extent[:3]]
        local_center = list(runtime_object_pos)
        local_min = [float(local_center[index]) - half[index] for index in range(3)]
        local_max = [float(local_center[index]) + half[index] for index in range(3)]
        world_min = _add_vec(local_min, origin)
        world_max = _add_vec(local_max, origin)
        world_center = _add_vec(local_center, origin)
    return {
        "surface_source": source,
        "surface_source_error": "",
        "surface_bbox_rebased_to_runtime_root": rebase,
        "surface_bbox_stage_center_local_xyz": stage_center,
        "surface_collision_prim_count": int(collision_count),
        "surface_visual_prim_count": int(visual_count),
        "surface_bbox_min_world_xyz": world_min,
        "surface_bbox_max_world_xyz": world_max,
        "surface_bbox_center_world_xyz": world_center,
        "surface_bbox_min_local_xyz": local_min,
        "surface_bbox_max_local_xyz": local_max,
        "surface_bbox_center_local_xyz": local_center,
        "surface_bbox_extent_xyz": extent,
    }


def _fixed_surface_geometry(
    object_pos: list[float],
    origin: list[float],
    cfg: ScriptedBaselineConfig,
    *,
    error: str,
    collision_count: int,
    visual_count: int,
) -> dict[str, Any]:
    extent = [
        2.0 * float(cfg.screw1_surface_radius_m),
        2.0 * float(cfg.screw1_surface_radius_m),
        0.025,
    ]
    half = [value * 0.5 for value in extent]
    local_min = [float(object_pos[index]) - half[index] for index in range(3)]
    local_max = [float(object_pos[index]) + half[index] for index in range(3)]
    world_min = _add_vec(local_min, origin)
    world_max = _add_vec(local_max, origin)
    world_center = _add_vec(object_pos, origin)
    return {
        "surface_source": "fixed_estimate",
        "surface_source_error": error,
        "surface_bbox_rebased_to_runtime_root": False,
        "surface_bbox_stage_center_local_xyz": [],
        "surface_collision_prim_count": int(collision_count),
        "surface_visual_prim_count": int(visual_count),
        "surface_bbox_min_world_xyz": world_min,
        "surface_bbox_max_world_xyz": world_max,
        "surface_bbox_center_world_xyz": world_center,
        "surface_bbox_min_local_xyz": local_min,
        "surface_bbox_max_local_xyz": local_max,
        "surface_bbox_center_local_xyz": list(object_pos),
        "surface_bbox_extent_xyz": extent,
    }


def _bbox_for_prims(cache: Any, prims: list[Any]) -> dict[str, list[float]]:
    if not prims:
        return {"min": [], "max": [], "center": [], "extent": []}
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for prim in prims:
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn = box.GetMin()
            mx = box.GetMax()
            for axis in range(3):
                mins[axis] = min(mins[axis], float(mn[axis]))
                maxs[axis] = max(maxs[axis], float(mx[axis]))
        except Exception:
            continue
    if not all(math.isfinite(value) for value in [*mins, *maxs]):
        return {"min": [], "max": [], "center": [], "extent": []}
    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    extent = [maxs[axis] - mins[axis] for axis in range(3)]
    return {"min": mins, "max": maxs, "center": center, "extent": extent}


def _bbox_has_extent(box: dict[str, list[float]]) -> bool:
    extent = list(box.get("extent", []) or [])
    return bool(len(extent) >= 3 and all(math.isfinite(float(v)) for v in extent) and max(extent) > 1.0e-5)


def _aabb_ray_radius_xy(
    bbox_min: list[float],
    bbox_max: list[float],
    bbox_center: list[float],
    normal: list[float],
) -> float:
    if len(bbox_min) < 2 or len(bbox_max) < 2 or len(bbox_center) < 2:
        return 0.0
    candidates: list[float] = []
    for axis in (0, 1):
        n = float(normal[axis])
        if abs(n) <= 1.0e-8:
            continue
        face = float(bbox_max[axis]) if n > 0.0 else float(bbox_min[axis])
        distance = (face - float(bbox_center[axis])) / n
        if distance > 0.0 and math.isfinite(distance):
            candidates.append(distance)
    return min(candidates) if candidates else 0.0


def _closest_point_on_aabb(point: list[float], bbox_min: list[float], bbox_max: list[float]) -> list[float]:
    if len(point) < 3 or len(bbox_min) < 3 or len(bbox_max) < 3:
        return list(point[:3]) if len(point) >= 3 else [0.0, 0.0, 0.0]
    clamped = [
        min(max(float(point[axis]), float(bbox_min[axis])), float(bbox_max[axis]))
        for axis in range(3)
    ]
    inside = all(float(bbox_min[axis]) <= float(point[axis]) <= float(bbox_max[axis]) for axis in range(3))
    if not inside:
        return clamped
    distances = []
    for axis in range(3):
        distances.append((abs(float(point[axis]) - float(bbox_min[axis])), axis, float(bbox_min[axis])))
        distances.append((abs(float(bbox_max[axis]) - float(point[axis])), axis, float(bbox_max[axis])))
    _distance_to_face, axis, face_value = min(distances, key=lambda item: item[0])
    clamped[axis] = face_value
    return clamped


def _surface_distance_servo_target(state: dict[str, Any], *, finger_key: str = "finger3_tip_local_pos") -> dict[str, Any]:
    point = list(state.get(finger_key, []) or [])
    bbox_min = list(state.get("surface_bbox_min_local_xyz", []) or [])
    bbox_max = list(state.get("surface_bbox_max_local_xyz", []) or [])
    if len(point) < 3 or len(bbox_min) < 3 or len(bbox_max) < 3:
        return {
            "closest_surface_point_local_xyz": list(point[:3]) if len(point) >= 3 else [0.0, 0.0, 0.0],
            "surface_distance_m": math.inf,
            "surface_direction_xyz": [0.0, 0.0, 0.0],
        }
    closest = _closest_point_on_aabb(point, bbox_min, bbox_max)
    delta = _sub_vec(closest, point)
    distance = _norm(delta)
    direction = _unit(delta)
    return {
        "closest_surface_point_local_xyz": closest,
        "surface_distance_m": distance,
        "surface_direction_xyz": direction,
    }


def _signed_distance_to_aabb_bounds(point: list[float], bbox_min: list[float], bbox_max: list[float]) -> float:
    if len(point) < 3 or len(bbox_min) < 3 or len(bbox_max) < 3:
        return math.inf
    q = []
    for axis in range(3):
        center = (float(bbox_min[axis]) + float(bbox_max[axis])) * 0.5
        half = max(1.0e-6, (float(bbox_max[axis]) - float(bbox_min[axis])) * 0.5)
        q.append(abs(float(point[axis]) - center) - half)
    outside = [max(value, 0.0) for value in q]
    outside_norm = _norm(outside)
    if outside_norm > 0.0:
        return outside_norm
    return max(q)


def _table_z_estimate(base: Any, env_index: int, object_pos: list[float]) -> float:
    table_z = _tensor_scalar(getattr(base, "floating_table_z_est", None), env_index)
    if not math.isfinite(table_z) or table_z < 0.70 or table_z > 0.80:
        table_z = float(object_pos[2]) - 0.012
    return float(table_z)


def _clamp_or_barrier(state: dict[str, Any], *, threshold_m: float = 1.0e-4) -> bool:
    threshold = max(0.0, float(threshold_m))
    return bool(
        abs(float(state.get("workspace_clamp_delta_m", 0.0))) > threshold
        or abs(float(state.get("table_barrier_delta_z_m", 0.0))) > threshold
    )


def _dominant_clamp_axis(clamp_vec: list[float], table_barrier_delta_z: float = 0.0) -> str:
    values = [abs(_list_get(clamp_vec, index, 0.0)) for index in range(3)]
    table_abs = abs(float(table_barrier_delta_z or 0.0))
    if max([table_abs, *values]) <= 1.0e-4:
        return ""
    if table_abs >= max(values):
        return "z"
    return ("x", "y", "z")[int(max(range(3), key=lambda index: values[index]))]


def _project_direction_for_clamp(direction: list[float], blocked_axis: str) -> list[float]:
    if blocked_axis not in {"x", "y", "z"}:
        return direction
    projected = list(direction[:3])
    projected[{"x": 0, "y": 1, "z": 2}[blocked_axis]] = 0.0
    out = _unit(projected)
    return out if _norm(out) > 1.0e-6 else direction


def _clamp_aware_target_adjustment(state: dict[str, Any], cfg: ScriptedBaselineConfig) -> dict[str, Any]:
    axis = str(state.get("workspace_clamp_dominant_axis", ""))
    table_clamp = abs(float(state.get("table_barrier_delta_z_m", 0.0) or 0.0))
    if axis == "z" or table_clamp > 1.0e-4:
        return {
            "adjustment_type": "raise_contact_target_z",
            "blocked_axis": "z",
            "contact_adjustment": {"z_bias_m": float(cfg.clamp_adjust_z_raise_m)},
        }
    if axis in {"x", "y"}:
        post = list(state.get("wrist_target_post_clamp_xyz", []) or [])
        obj = list(state.get("object_local_pos", []) or [])
        side = [0.0, 0.0, 0.0]
        if len(post) >= 2 and len(obj) >= 2:
            side = [float(post[0]) - float(obj[0]), float(post[1]) - float(obj[1]), 0.0]
        side = _unit(side)
        adjustment: dict[str, Any] = {}
        if _norm(side) > 1.0e-6:
            adjustment["side_dir_override"] = side
        return {
            "adjustment_type": "reachable_side_xy",
            "blocked_axis": axis,
            "contact_adjustment": adjustment,
        }
    return {}


def _summarize_clamp_from_logs(logs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "workspace_clamp_max_m": 0.0,
        "table_clamp_max_m": 0.0,
        "first_clamp_phase": "",
        "first_clamp_step": "",
        "first_clamp_finger3_distance": "",
        "first_clamp_finger4_distance": "",
        "clamp_dominant_axis": "",
    }
    first: dict[str, Any] | None = None
    dominant_row: dict[str, Any] | None = None
    dominant_score = 0.0
    for row in logs:
        workspace = abs(_row_float(row, "workspace_clamp_delta_m"))
        table = abs(_row_float(row, "table_barrier_delta_z_m"))
        score = max(workspace, table)
        out["workspace_clamp_max_m"] = max(float(out["workspace_clamp_max_m"]), workspace)
        out["table_clamp_max_m"] = max(float(out["table_clamp_max_m"]), table)
        if score > 1.0e-4 and first is None:
            first = row
        if score > dominant_score:
            dominant_score = score
            dominant_row = row
    if first is not None:
        out.update(
            {
                "first_clamp_phase": str(first.get("phase", "")),
                "first_clamp_step": str(first.get("step", "")),
                "first_clamp_finger3_distance": _row_float(first, "finger3_distance_to_contact_target_m"),
                "first_clamp_finger4_distance": _row_float(first, "finger4_distance_to_contact_target_m"),
            }
        )
    if dominant_row is not None and dominant_score > 1.0e-4:
        axis = str(dominant_row.get("workspace_clamp_dominant_axis", ""))
        if not axis and abs(_row_float(dominant_row, "table_barrier_delta_z_m")) > 1.0e-4:
            axis = "z"
        out["clamp_dominant_axis"] = axis
    return out


def _summarize_surface_from_logs(logs: list[dict[str, Any]], final_state: dict[str, Any]) -> dict[str, Any]:
    def _point_from_row(row: dict[str, Any], prefix: str) -> list[float]:
        return [
            _row_float(row, f"{prefix}_x", 0.0),
            _row_float(row, f"{prefix}_y", 0.0),
            _row_float(row, f"{prefix}_z", 0.0),
        ]

    def _min_row(key: str) -> dict[str, Any]:
        best: dict[str, Any] = {}
        best_value = math.inf
        for row in logs:
            value = _row_float(row, key, math.inf)
            if math.isfinite(value) and value < best_value:
                best = row
                best_value = value
        return best

    f3_row = _min_row("finger3_distance_to_real_surface_m")
    f4_row = _min_row("finger4_distance_to_real_surface_m")
    press_rows = [row for row in logs if str(row.get("phase", "")) in {"micro_press", "micro_press_ik"}]
    approach_rows = [row for row in logs if str(row.get("phase", "")) in {"precontact", "micro_press", "micro_press_ik"}]
    press_commanded = max([0.0, *[_row_float(row, "press_depth_commanded_m", 0.0) for row in logs]])
    press_actual = max([0.0, *[_row_float(row, "press_depth_actual_m", 0.0) for row in logs]])
    if press_actual <= 0.0:
        press_actual = max([0.0, *[_row_float(row, "finger3_surface_press_depth_m", 0.0) for row in logs]])
    press_depth = max([press_commanded, *[_row_float(row, "press_depth_attempted_m", 0.0) for row in logs]])
    approach_z_clamp = max([0.0, *[abs(_row_float(row, "workspace_clamp_z", 0.0)) for row in approach_rows]])
    approach_table_clamp = max([0.0, *[abs(_row_float(row, "table_barrier_delta_z_m", 0.0)) for row in approach_rows]])
    press_z_clamp = max([0.0, *[abs(_row_float(row, "workspace_clamp_z", 0.0)) for row in press_rows]])
    press_table_clamp = max([0.0, *[abs(_row_float(row, "table_barrier_delta_z_m", 0.0)) for row in press_rows]])
    press_workspace_clamp = max([0.0, *[abs(_row_float(row, "workspace_clamp_delta_m", 0.0)) for row in press_rows]])
    press_policy_norm = max([0.0, *[_row_float(row, "commanded_policy_xyz_norm", 0.0) for row in press_rows]])
    press_mapped_norm = max([0.0, *[_row_float(row, "mapped_isaac_action_xyz_norm", 0.0) for row in press_rows]])
    press_ctrl_delta = max([0.0, *[_row_float(row, "ctrl_target_delta_l2", 0.0) for row in press_rows]])
    press_tip_dot = max([0.0, *[_row_float(row, "finger3_delta_dot_press_dir_cumulative", 0.0) for row in press_rows]])
    f3_min_surface = _row_float(f3_row, "finger3_distance_to_real_surface_m", math.inf)
    f3_tip_collision_offset = final_state.get("finger3_tip_origin_to_collision_surface_offset_m", "")
    try:
        f3_collision_gap_at_min = max(0.0, float(f3_min_surface) - float(f3_tip_collision_offset))
    except Exception:
        f3_collision_gap_at_min = final_state.get("finger3_collision_surface_to_screw1_surface_gap_m", "")
    return {
        "surface_source": final_state.get("surface_source", ""),
        "surface_source_error": final_state.get("surface_source_error", ""),
        "surface_bbox_rebased_to_runtime_root": final_state.get("surface_bbox_rebased_to_runtime_root", ""),
        "surface_bbox_stage_center_local_xyz": final_state.get("surface_bbox_stage_center_local_xyz", []),
        "screw1_root_local_position": final_state.get("object_local_pos", []),
        "screw1_runtime_root_local_xyz": final_state.get(
            "screw1_runtime_root_local_xyz", final_state.get("object_local_pos", [])
        ),
        "screw1_root_world_position": final_state.get("object_world_pos", []),
        "screw1_bbox_min_local_xyz": final_state.get("surface_bbox_min_local_xyz", []),
        "screw1_bbox_max_local_xyz": final_state.get("surface_bbox_max_local_xyz", []),
        "screw1_bbox_center_local_xyz": final_state.get("surface_bbox_center_local_xyz", []),
        "screw1_bbox_extent_xyz": final_state.get("surface_bbox_extent_xyz", []),
        "screw1_bbox_min_world_xyz": final_state.get("surface_bbox_min_world_xyz", []),
        "screw1_bbox_max_world_xyz": final_state.get("surface_bbox_max_world_xyz", []),
        "current_contact_target_position": final_state.get("contact_target_local_pos", []),
        "current_contact_target_z": _list_get(final_state.get("contact_target_local_pos", []), 2, ""),
        "contact_target_normal": final_state.get("contact_target_normal_xyz", []),
        "surface_point_local_xyz": final_state.get("surface_point_local_xyz", []),
        "precontact_point_local_xyz": final_state.get("precontact_point_local_xyz", []),
        "press_target_local_xyz": final_state.get("press_target_local_xyz", []),
        "table_safe_z": final_state.get("contact_target_table_safe_z_m", ""),
        "finger3_min_distance_to_real_surface_m": f3_min_surface,
        "finger4_min_distance_to_real_surface_m": _row_float(f4_row, "finger4_distance_to_real_surface_m", math.inf),
        "finger3_tip_at_min_surface_distance": _point_from_row(f3_row, "finger3_tip"),
        "finger4_tip_at_min_surface_distance": _point_from_row(f4_row, "finger4_tip"),
        "finger3_signed_distance_to_surface_at_min_m": _row_float(
            f3_row, "finger3_signed_distance_to_screw1_surface_m", math.inf
        ),
        "finger4_signed_distance_to_surface_at_min_m": _row_float(
            f4_row, "finger4_signed_distance_to_screw1_surface_m", math.inf
        ),
        "press_depth_attempted_m": press_depth,
        "press_depth_commanded_m": press_commanded,
        "press_depth_actual_m": press_actual,
        "workspace_z_clamp_max_during_approach_m": approach_z_clamp,
        "table_clamp_max_during_approach_m": approach_table_clamp,
        "workspace_z_clamp_max_during_press_m": press_z_clamp,
        "table_clamp_max_during_press_m": press_table_clamp,
        "workspace_clamp_max_during_press": press_workspace_clamp,
        "press_policy_action_norm_peak_from_log": press_policy_norm,
        "press_mapped_isaac_action_norm_peak_from_log": press_mapped_norm,
        "press_ctrl_target_delta_peak_from_log": press_ctrl_delta,
        "press_finger3_delta_dot_press_dir_from_log": press_tip_dot,
        "z_clamp_max_during_press": max(press_z_clamp, press_table_clamp),
        "finger3_tip_collision_geometry_available": bool(
            final_state.get("finger3_tip_collision_geometry_available", False)
        ),
        "finger_tip_collision_geometry_unavailable": final_state.get(
            "finger_tip_collision_geometry_unavailable", ""
        ),
        "finger3_tip_collision_geometry_source": final_state.get("finger3_tip_collision_geometry_source", ""),
        "finger3_tip_collision_prim_count": final_state.get("finger3_tip_collision_prim_count", 0),
        "finger3_tip_collision_bbox_rebased_to_runtime_tip": final_state.get(
            "finger3_tip_collision_bbox_rebased_to_runtime_tip", False
        ),
        "finger3_tip_collision_bbox_stage_center_local_xyz": final_state.get(
            "finger3_tip_collision_bbox_stage_center_local_xyz", []
        ),
        "finger3_tip_collision_bbox_min_local_xyz": final_state.get(
            "finger3_tip_collision_bbox_min_local_xyz", []
        ),
        "finger3_tip_collision_bbox_max_local_xyz": final_state.get(
            "finger3_tip_collision_bbox_max_local_xyz", []
        ),
        "finger3_tip_collision_bbox_center_local_xyz": final_state.get(
            "finger3_tip_collision_bbox_center_local_xyz", []
        ),
        "finger3_tip_collision_bbox_extent_xyz": final_state.get(
            "finger3_tip_collision_bbox_extent_xyz", []
        ),
        "finger3_tip_origin_to_collision_surface_offset_m": final_state.get(
            "finger3_tip_origin_to_collision_surface_offset_m", ""
        ),
        "finger3_collision_surface_to_screw1_surface_gap_m": f3_collision_gap_at_min,
        "true_collision_surface_available": bool(final_state.get("true_collision_surface_available", False)),
        "true_collision_surface_unavailable_reason": final_state.get(
            "true_collision_surface_unavailable_reason", ""
        ),
        "screw1_visual_prim_paths": final_state.get("screw1_visual_prim_paths", []),
        "screw1_collision_prim_paths": final_state.get("screw1_collision_prim_paths", []),
        "screw1_collision_shape_count": final_state.get("screw1_collision_shape_count", 0),
        "screw1_collision_shape_types": final_state.get("screw1_collision_shape_types", []),
        "screw1_collision_bounds_source": final_state.get("screw1_collision_bounds_source", ""),
        "screw1_bbox_rebase_delta_m": final_state.get("screw1_bbox_rebase_delta_m", ""),
        "finger3_pose_body": final_state.get("finger3_pose_body", ""),
        "finger3_sensor_body": final_state.get("finger3_sensor_body", ""),
        "finger3_collision_body": final_state.get("finger3_collision_body", ""),
        "finger3_logical_index": final_state.get("finger3_logical_index", FINGER3_INDEX),
        "finger3_rigid_body_index": final_state.get("finger3_rigid_body_index", -1),
        "finger3_collision_prim_paths": final_state.get("finger3_collision_prim_paths", []),
        "finger3_collision_shape_count": final_state.get("finger3_collision_shape_count", 0),
        "finger3_collision_shape_types": final_state.get("finger3_collision_shape_types", []),
        "pose_body_equals_collision_body": final_state.get("pose_body_equals_collision_body", False),
        "collision_body_equals_sensor_body": final_state.get("collision_body_equals_sensor_body", False),
    }


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _tensor_vec(value: Any, index: int, width: int = 3) -> list[float]:
    if value is None:
        return [0.0] * width
    try:
        if hasattr(value, "detach"):
            item = value[int(index)]
            return [float(v) for v in item.detach().cpu().reshape(-1).tolist()[:width]]
        item = value[int(index)]
        return [float(v) for v in list(item)[:width]]
    except Exception:
        return [0.0] * width


def _tensor_list(value: Any, index: int) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            return [float(v) for v in value[int(index)].detach().cpu().reshape(-1).tolist()]
        return [float(v) for v in list(value[int(index)])]
    except Exception:
        return []


def _tensor_matrix(value: Any, index: int) -> list[list[float]]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            rows = value[int(index)].detach().cpu().tolist()
        else:
            rows = value[int(index)]
        return [[float(v) for v in list(row)] for row in rows]
    except Exception:
        return []


def _tensor_scalar(value: Any, index: int) -> float:
    if value is None:
        return 0.0
    try:
        if hasattr(value, "detach"):
            item = value[int(index)] if value.ndim > 0 else value
            return float(item.detach().cpu().reshape(-1)[0].item())
        item = value[int(index)] if hasattr(value, "__getitem__") else value
        return float(item)
    except Exception:
        return 0.0


def _sub_vec(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(min(3, len(a), len(b)))]


def _add_vec(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(min(3, len(a), len(b)))]


def _distance(a: list[float], b: list[float]) -> float:
    return _norm(_sub_vec(a, b))


def _vec_delta_z(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return float(b[2]) - float(a[2])


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _unit(vec: list[float]) -> list[float]:
    norm = _norm(vec)
    if norm <= 1.0e-9:
        return [0.0, 0.0, 0.0]
    return [float(v) / norm for v in vec[:3]]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(min(len(a), len(b), 3)))


def _list_get(values: list[float], index: int, default: Any = 0.0) -> Any:
    try:
        return float(values[index]) if 0 <= int(index) < len(values) else default
    except Exception:
        return default


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
