"""Fresh-reset, force-guarded deterministic Screw1 grasp baseline v2.

This module is deliberately separate from ``scripted_contact_baseline``.  It
uses only the verified low-level runtime hooks: reset-time staging, direct 26D
Wuji floating-hand actions, target-filtered Screw1 force, and optional contact
alignment.  It never writes object poses after reset and never treats distance,
video, sticky/proxy behavior, or unfiltered-only force as success.
"""

from __future__ import annotations

import csv
import ast
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dual_contact_regulator import (
    ABORTED,
    CONTROLLED_CLOSE,
    DUAL_CONTACT_REGULATION,
    SLOW_LIFT,
    DualContactObservation,
    DualContactRegulator,
    DualContactRegulatorConfig,
)
from .scripted_contact_baseline import (
    FINGER3_INDEX,
    FINGER4_INDEX,
    _action_dim,
    _assert_runtime_interfaces,
    _base_env,
    _distance,
    _norm,
    _quat_for_rpy_offset,
    _quat_normalize_wxyz,
    _refresh_runtime,
    _target_env_index,
    _tensor_vec,
    read_state,
)
from .video_log_alignment import AlignmentRecorder, make_run_id, write_json

try:  # pragma: no cover - Isaac runtime dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


RESULT_CLASSES = (
    "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
    "STABLE_GRASP_ACQUIRED_LIFT_FAILED",
    "STABLE_CONTACT_ACQUIRED_CLOSE_FAILED",
    "ACTION_AUTHORITY_BLOCKER_PROVEN",
    "INITIAL_PHYSICS_BLOCKER_PROVEN",
    "ASSET_COLLISION_BLOCKER_PROVEN",
    "KINEMATIC_GRASP_BLOCKER_PROVEN",
    "DUAL_CONTACT_REGULATION_BLOCKER_PROVEN",
    "CONTACT_ACQUISITION_INCOMPLETE",
    "WRIST_CONTROL_BLOCKER_PROVEN",
    "WRIST_TRANSLATION_ISOLATION_PASS",
    "WRIST_TRANSLATION_VALIDATED",
    "TEST_METRIC_ERROR",
    "TARGET_BUFFER_OVERWRITE",
    "SCENE_COLLISION_OR_CONSTRAINT",
    "STALE_USD_DRIVE_CONFIG",
    "ARTICULATION_HIERARCHY_OR_DRIVE_FAILURE",
    "TASK_SPACE_WRIST_CONTROLLER_ISSUE",
    "SCENE_COLLISION_PATH_CONSTRAINT",
    "WRIST_ARTICULATION_DRIVE_ISSUE",
    "WRIST_FK_BODY_MAPPING_ISSUE",
    "SEED_PATH_GEOMETRY_ISSUE",
    "PHYSX_RUNTIME_BLOCKER_PROVEN",
    "HOST_GPU_BUSY",
    "HOST_RESOURCE_PREFLIGHT_BLOCKED",
    "HOST_ACTION_REQUIRED",
    "HEALTH_OK",
    "V2_IMPLEMENTATION_ERROR",
)
V2_PHASES = (
    "full",
    "health",
    "no_hand",
    "wrist_audit",
    "wrist_translation_isolation",
    "action_audit",
    "seed_replay",
    "grasp",
    "repeated_trials",
)


def _load_collision_offset_ab_summary_from_env() -> dict[str, Any]:
    path_text = os.environ.get("WUJI_SCREW1_V2_AB_SUMMARY_PATH", "").strip()
    if not path_text:
        return {}
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "collision_offset_ab_summary_path": str(path),
            "collision_offset_ab_summary_load_error": f"{type(exc).__name__}:{exc}",
        }
    if not isinstance(data, dict):
        return {
            "collision_offset_ab_summary_path": str(path),
            "collision_offset_ab_summary_load_error": "summary_not_json_object",
        }
    data["collision_offset_ab_summary_path"] = str(path)
    data["collision_offset_ab_summary_load_error"] = ""
    return data


@dataclass
class Screw1GraspBaselineV2Config:
    part: str = "Screw1"
    output_dir: str = "debug_runs/screw1_grasp_baseline_v2"
    phase: str = "full"
    run_id: str = ""
    alignment_debug: bool = False
    clean_output_info: dict[str, Any] = field(default_factory=dict)
    gpu_preflight_info: dict[str, Any] = field(default_factory=dict)
    accept_prior_wrist_action_gates: bool = True
    v2_collision_offset_fix: str = "auto"
    deterministic_seed: int = 20260710
    settle_steps: int = 30
    canonical_xy_m: tuple[float, float] = (-0.2, 0.0)
    support_drop_height_m: float = 0.015
    support_spawn_clearance_m: float = 0.0015
    canonical_hand_park_offset_m: tuple[float, float, float] = (0.16, 0.16, 0.12)
    canonical_hand_park_quat_wxyz: tuple[float, float, float, float] = (
        1.9524879917298676e-06,
        -0.9998642206192017,
        -0.016484227031469345,
        5.2112377488811035e-06,
    )
    support_calibration_max_steps: int = 600
    support_settle_required_steps: int = 60
    support_settle_linear_speed_mps: float = 0.010
    support_settle_angular_speed_radps: float = 0.20
    support_reset_pose_tolerance_m: float = 0.002
    no_hand_trials: int = 5
    no_hand_steps: int = 120
    no_hand_max_drift_m: float = 0.010
    no_hand_max_speed_mps: float = 0.05
    no_hand_max_angular_speed_radps: float = 0.30
    action_probe_steps: int = 5
    action_probe_value: float = 0.25
    action_authority_target_delta_min: float = 1.0e-4
    action_authority_actual_delta_min: float = 1.0e-4
    action_authority_tip_delta_min_m: float = 1.0e-5
    action_authority_crosstalk_max_ratio: float = 0.25
    fingerprint_tolerance_m: float = 0.002
    contact_threshold_n: float = 0.05
    soft_force_min_n: float = 0.05
    soft_force_max_n: float = 1.0
    hard_abort_force_n: float = 5.0
    seed_replay_repeats: int = 5
    seed_servo_max_steps: int = 600
    seed_servo_pos_step_m: float = 0.0005
    seed_servo_free_space_step_m: float = 0.002
    seed_servo_fine_radius_m: float = 0.040
    seed_servo_rot_step_deg: float = 1.0
    seed_servo_max_target_lead_m: float = 0.020
    seed_servo_max_rotation_target_lead_deg: float = 4.0
    seed_servo_tracking_timeout_steps: int = 240
    seed_servo_command_tol_m: float = 0.0010
    seed_servo_command_rot_tol_deg: float = 1.0
    seed_servo_tracking_stall_window_steps: int = 30
    seed_servo_tracking_stall_min_improvement_m: float = 0.00005
    seed_servo_tracking_stall_min_improvement_deg: float = 0.25
    seed_servo_standoff_lift_m: float = 0.350
    seed_servo_standoff_waypoint_spacing_m: float = 0.040
    seed_replay_contact_truth_transit_stride: int = 25
    seed_target_table_barrier_margin_m: float = 0.001
    seed_replay_use_verified_corridor_transit: bool = True
    seed_replay_skip_legacy_seeds: bool = False
    seed_replay_resume_progress: bool = False
    frozen_acquisition_plan_path: str = "debug_runs/screw1_grasp_baseline_v2_probe_handoff_fix/validated_acquisition_plan.json"
    trajectory_bank_candidate_limit: int = 9
    trajectory_bank_x_offsets_m: tuple[float, ...] = (0.075, 0.085, 0.065)
    trajectory_bank_y_offsets_m: tuple[float, ...] = (-0.045, -0.030, -0.060)
    trajectory_bank_z_offsets_m: tuple[float, ...] = (-0.010,)
    trajectory_bank_orientation_rpy_deg: tuple[tuple[float, float, float], ...] = (
        (0.0, 20.0, 0.0),
        (5.0, 20.0, 0.0),
        (10.0, 20.0, 0.0),
    )
    seed_reach_tol_m: float = 0.0025
    seed_reach_rot_tol_deg: float = 5.0
    hard_abort_retreat_steps: int = 8
    stable_contact_steps: int = 30
    stable_contact_duty_ratio: float = 0.8
    stable_preclose_object_motion_limit_m: float = 0.005
    stable_velocity_limit_mps: float = 0.03
    stable_angular_velocity_limit_radps: float = 0.8
    close_hold_steps: int = 30
    close_action_value: float = 0.005
    finger_acquisition_action_value: float = 0.006
    finger_acquisition_probe_action_value: float = 0.002
    finger_acquisition_probe_settle_steps: int = 5
    finger_acquisition_probe_restore_settle_steps: int = 5
    finger_acquisition_max_active_joints_per_finger: int = 1
    finger_acquisition_min_tip_effect_m: float = 1.0e-6
    finger_acquisition_reprobe_interval_steps: int = 60
    finger_acquisition_non_improving_reprobe_steps: int = 5
    finger_acquisition_max_steps: int = 600
    close_max_steps: int = 120
    lift_step_m: float = 0.0005
    lift_target_m: float = 0.012
    lift_max_steps: int = 80
    repeated_trials: int = 5
    repeated_success_required: int = 3
    repeated_success_consecutive: int = 3
    zero_action_hold_steps_after_trial: int = 10
    seed_a_position_offset_m: tuple[float, float, float] = (0.020, 0.010, -0.010)
    seed_a_orientation_rpy_deg: tuple[float, float, float] = (0.0, 20.0, 0.0)
    seed_b_position_offset_m: tuple[float, float, float] = (0.015, 0.015, -0.010)
    seed_b_orientation_rpy_deg: tuple[float, float, float] = (10.0, 20.0, 0.0)
    acquisition_plan_validation_repeats: int = 2
    acquisition_caging_perpendicular_max_m: float = 0.014
    acquisition_caging_tip_separation_min_m: float = 0.012
    acquisition_caging_tip_separation_max_m: float = 0.070
    acquisition_plan_geometry_tolerance_m: float = 0.008
    acquisition_endpoint_projection_margin: float = 0.25
    anchor_residual_approach_step_m: float = 0.0005
    anchor_residual_approach_max_steps: int = 80
    anchor_finger_guided_max_steps: int = 120
    anchor_finger_probe_action_value: float = 0.003
    anchor_finger_probe_restore_steps: int = 3
    anchor_finger_reprobe_interval_steps: int = 5
    anchor_finger_min_improvement_m: float = 2.5e-5
    opposing_finger_probe_action_value: float = 0.006
    opposing_finger_min_improvement_m: float = 2.5e-7
    opposing_finger_probe_settle_steps: int = 5
    opposing_finger_probe_restore_settle_steps: int = 5
    opposing_finger_reprobe_interval_steps: int = 30
    opposing_finger_max_steps: int = 1200
    opposing_contact_handoff_max: int = 2
    opposing_anchor_reacquire_wait_steps: int = 20
    dual_contact_keep_n: float = 0.035
    dual_contact_anchor_force_min_n: float = 0.08
    dual_contact_anchor_force_max_n: float = 0.30
    dual_contact_anchor_force_target_n: float = 0.15
    dual_contact_finger_step_rad: float = 0.001
    dual_contact_max_steps: int = 900
    dual_contact_stagnation_steps: int = 12
    dual_contact_stagnation_improvement_n: float = 0.005
    dual_contact_probe_object_motion_limit_m: float = 0.0005
    dual_contact_preferred_object_motion_limit_m: float = 0.001
    dual_contact_wrist_probe_settle_steps: int = 2
    dual_contact_wrist_probe_restore_steps: int = 3
    dual_contact_wrist_probe_enabled: bool = True
    dual_contact_heartbeat_stride: int = 25
    anchor_orientation_probe_deg: float = 1.0
    anchor_orientation_probe_hold_steps: int = 3
    anchor_orientation_probe_restore_steps: int = 4
    anchor_orientation_guided_max_steps: int = 40
    anchor_orientation_reprobe_interval_steps: int = 5
    anchor_orientation_min_improvement_m: float = 2.5e-5
    anchor_orientation_max_total_deg: float = 10.0
    acquisition_auto_correction_max_trials: int = 12
    acquisition_auto_correction_max_per_orientation: int = 2
    acquisition_offset_x_bounds_m: tuple[float, float] = (0.040, 0.110)
    acquisition_offset_y_bounds_m: tuple[float, float] = (-0.085, 0.025)
    acquisition_offset_z_bounds_m: tuple[float, float] = (-0.018, -0.004)
    active_finger_group: str = "34"
    wrist_audit_reset_hold_steps: int = 60
    wrist_audit_axis_settle_steps: int = 90
    wrist_audit_pos_delta_m: float = 0.010
    wrist_audit_rot_delta_deg: float = 5.0
    wrist_audit_reset_pos_tol_m: float = 0.002
    wrist_audit_reset_rot_tol_deg: float = 2.0
    wrist_audit_combined_pos_tol_m: float = 0.005
    wrist_audit_combined_rot_tol_deg: float = 5.0
    wrist_audit_axis_response_fraction: float = 0.50
    wrist_isolation_trials: int = 3
    wrist_isolation_total_x_m: float = -0.220
    wrist_isolation_waypoint_step_m: float = 0.055
    wrist_isolation_waypoint_max_steps: int = 120
    wrist_isolation_joint_pos_tol_m: float = 0.003
    wrist_isolation_joint_rot_tol_deg: float = 2.0
    wrist_isolation_contact_threshold_n: float = 0.05
    wrist_isolation_fk_body_pos_tol_m: float = 0.010
    wrist_isolation_fk_body_rot_tol_deg: float = 5.0
    wrist_isolation_scene_clearance_m: float = 0.180
    wrist_isolation_rot_limit_margin_deg: float = 10.0
    wrist_isolation_workspace_margin_m: float = 0.050
    canonical_support_pose: dict[str, Any] = field(default_factory=dict)


def run_screw1_grasp_baseline_v2(
    env: Any,
    cfg: Screw1GraspBaselineV2Config,
    *,
    video_recorder: Any | None = None,
) -> dict[str, Any]:
    """Run the requested v2 phase and write compact v2 artifacts."""

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = _initial_summary(cfg)
    summary.update(_load_collision_offset_ab_summary_from_env())
    episodes: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    alignment: AlignmentRecorder | None = None
    result_class = "HEALTH_OK"
    _write_v2_progress(output_dir, "initialized", summary)

    try:
        if torch is None:
            raise RuntimeError("torch_unavailable")
        if cfg.phase not in V2_PHASES:
            raise RuntimeError(f"unsupported_v2_phase_{cfg.phase}")

        base = _base_env(env)
        target_env_index = _target_env_index(base, cfg.part)
        _assert_runtime_interfaces(base, cfg.part, target_env_index)
        summary.update(
            {
                "target_env_index": int(target_env_index),
                "action_dim": int(_action_dim(env, base)),
                "direct_26d_action_used": True,
                "unified_action_mapper_used_for_control": False,
                "object_write_after_reset_used": False,
            }
        )
        summary.update(_screw1_env_repair_summary(base))
        _write_v2_progress(output_dir, "runtime_interfaces_ready", summary)

        alignment = AlignmentRecorder(
            enabled=bool(cfg.alignment_debug),
            output_dir=output_dir,
            run_id=str(cfg.run_id or make_run_id()),
            target_part=cfg.part,
            target_env_index=target_env_index,
            video_recorder=video_recorder,
        )

        health = _run_health_phase(output_dir)
        summary.update(health)
        if cfg.clean_output_info:
            summary.update(cfg.clean_output_info)
        if _health_resource_blocked(health):
            result_class = "HOST_RESOURCE_PREFLIGHT_BLOCKED"
            summary.update(
                {
                    "result_class": result_class,
                    "blocker": "host_resource_unhealthy_inside_v2",
                    "final_allowed_result_class": True,
                }
            )
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary
        if cfg.phase == "health":
            summary.update({"result_class": "HEALTH_OK", "blocker": "", "final_allowed_result_class": True})
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary

        _configure_v2_control(base, target_env_index)
        alignment.setup_body_mapping_and_sensors(base)
        _write_v2_progress(output_dir, "health_passed_control_configured", summary)

        support = _calibrate_canonical_support_pose(env, base, target_env_index, cfg, alignment, trace_rows)
        summary.update(support)
        _write_v2_progress(output_dir, "support_calibration_completed", summary)
        if not bool(support.get("canonical_support_pose_valid")):
            result_class = "INITIAL_PHYSICS_BLOCKER_PROVEN"
            summary.update(
                {
                    "result_class": result_class,
                    "blocker": "canonical_support_pose_calibration_failed_after_corrected_reset",
                    "final_allowed_result_class": True,
                }
            )
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary

        first_episode = _fresh_episode(env, base, target_env_index, cfg, alignment, trace_rows, "fresh_episode")
        episodes.append(first_episode["episode_row"])
        summary.update(_fingerprint_summary("fresh_episode", first_episode.get("fingerprint", {})))
        _write_v2_progress(output_dir, "fresh_episode_completed", summary)

        if cfg.phase in {"full", "no_hand", "wrist_translation_isolation"}:
            no_hand = _run_no_hand_stability(env, base, target_env_index, cfg, alignment, episodes, trace_rows)
            summary.update(no_hand)
            if not bool(no_hand.get("no_hand_stability_passed")):
                result_class = "INITIAL_PHYSICS_BLOCKER_PROVEN"
                summary.update(
                    {
                        "result_class": result_class,
                        "blocker": "Screw1 moved or contacted during no-hand settling",
                        "final_allowed_result_class": True,
                    }
                )
                _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
                return summary
            if cfg.phase == "no_hand":
                summary.update({"result_class": "HEALTH_OK", "blocker": "no_hand_phase_only", "final_allowed_result_class": True})
                _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
                return summary

        if cfg.phase == "wrist_translation_isolation":
            isolation = _run_wrist_translation_isolation(env, base, target_env_index, cfg, alignment, episodes, trace_rows)
            summary.update(isolation)
            result_class = str(isolation.get("result_class") or "WRIST_CONTROL_BLOCKER_PROVEN")
            summary.update(
                {
                    "result_class": result_class,
                    "blocker": str(isolation.get("blocker") or isolation.get("wrist_translation_isolation_subclass") or ""),
                    "final_allowed_result_class": result_class in RESULT_CLASSES,
                }
            )
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary

        use_prior_wrist_action = bool(
            cfg.accept_prior_wrist_action_gates and cfg.phase in {"seed_replay", "grasp", "repeated_trials"}
        )

        if cfg.phase in {"full", "wrist_audit"} or (
            cfg.phase in {"seed_replay", "grasp", "repeated_trials"} and not use_prior_wrist_action
        ):
            wrist_audit = _run_wrist_control_audit(env, base, target_env_index, cfg, alignment, episodes, trace_rows)
            summary.update(wrist_audit)
            _write_v2_progress(output_dir, "wrist_audit_completed", summary)
            if not bool(wrist_audit.get("wrist_audit_passed")):
                result_class = "WRIST_CONTROL_BLOCKER_PROVEN"
                summary.update(
                    {
                        "result_class": result_class,
                        "blocker": str(wrist_audit.get("wrist_audit_failure_reason") or "wrist_6d_control_audit_failed"),
                        "final_allowed_result_class": True,
                    }
                )
                _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
                return summary
            if cfg.phase == "wrist_audit":
                summary.update({"result_class": "HEALTH_OK", "blocker": "wrist_audit_phase_only", "final_allowed_result_class": True})
                _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
                return summary
        elif use_prior_wrist_action:
            summary.update(
                {
                    "wrist_audit_executed": False,
                    "wrist_audit_passed": True,
                    "wrist_audit_accepted_prior_evidence": True,
                    "wrist_audit_skip_reason": "accepted_prior_wrist_translation_and_control_validation",
                }
            )
            episodes.append(
                {
                    "phase": "wrist_audit",
                    "wrist_audit_passed": True,
                    "wrist_audit_executed": False,
                    "skip_reason": "accepted_prior_wrist_translation_and_control_validation",
                }
            )
            _write_v2_progress(output_dir, "wrist_audit_accepted_prior", summary)

        if use_prior_wrist_action:
            summary.update(
                {
                    "action_authority_executed": False,
                    "action_authority_required_joints_ok": True,
                    "action_authority_accepted_prior_evidence": True,
                    "action_authority_skip_reason": "accepted_prior_finger3_finger4_26d_action_authority",
                }
            )
            episodes.append(
                {
                    "phase": "action_authority",
                    "action_authority_required_joints_ok": True,
                    "action_authority_executed": False,
                    "skip_reason": "accepted_prior_finger3_finger4_26d_action_authority",
                }
            )
            _write_v2_progress(output_dir, "action_authority_accepted_prior", summary)
        else:
            authority = _run_action_authority_audit(env, base, target_env_index, cfg, alignment, episodes, trace_rows)
            authority_rows.extend(authority.pop("rows", []))
            summary.update(authority)
            _write_v2_progress(output_dir, "action_authority_completed", summary)
        if not bool(summary.get("action_authority_required_joints_ok")):
            result_class = "ACTION_AUTHORITY_BLOCKER_PROVEN"
            summary.update(
                {
                    "result_class": result_class,
                    "blocker": "finger3/finger4 single-column 26D action authority missing",
                    "final_allowed_result_class": True,
                }
            )
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary
        if cfg.phase == "action_audit":
            summary.update({"result_class": "HEALTH_OK", "blocker": "action_audit_phase_only", "final_allowed_result_class": True})
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary

        seed_replay: dict[str, Any] = {}
        if cfg.phase in {"full", "seed_replay", "grasp", "repeated_trials"}:
            frozen_plan_path = Path(cfg.frozen_acquisition_plan_path)
            use_frozen_plan = bool(cfg.phase in {"grasp", "repeated_trials"} and frozen_plan_path.exists())
            if use_frozen_plan:
                frozen_plan = json.loads(frozen_plan_path.read_text(encoding="utf-8"))
                frozen_offset = [float(value) for value in list(frozen_plan.get("seed_offset_xyz_m", []))[:3]]
                frozen_rpy = [float(value) for value in list(frozen_plan.get("final_orientation_rpy_deg", []))[:3]]
                if frozen_offset != [0.075, -0.045, -0.01] or frozen_rpy != [0.0, 20.0, 0.0]:
                    raise RuntimeError("frozen_acquisition_plan_pose_mismatch")
                seed_replay = {
                    "seed_replay_executed": False,
                    "seed_replay_validated": True,
                    "acquisition_plan_validated": True,
                    "validated_acquisition_plan": frozen_plan,
                    "validated_acquisition_plan_json": str(frozen_plan_path),
                    "frozen_acquisition_plan_reused": True,
                    "trajectory_bank_executed": False,
                    "legacy_seed_invalid_under_corrected_protocol": True,
                }
            else:
                seed_replay = _run_seed_replay(env, base, target_env_index, cfg, alignment, episodes, trace_rows)
            summary.update(seed_replay)
            _write_v2_progress(output_dir, "seed_replay_completed", summary)
            if cfg.phase == "seed_replay":
                summary.update({"result_class": "HEALTH_OK", "blocker": "seed_replay_phase_only", "final_allowed_result_class": True})
                _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
                return summary
            if not bool(seed_replay.get("seed_replay_validated")) and cfg.phase in {"full", "repeated_trials"}:
                result_class = "KINEMATIC_GRASP_BLOCKER_PROVEN"
                summary.update(
                    {
                        "result_class": result_class,
                        "blocker": "no_validated_seed_hint_after_fresh_replay",
                        "final_allowed_result_class": True,
                        "repeated_trials_executed": False,
                        "repeated_trial_count": 0,
                        "physical_lift_success_count": 0,
                        "physical_lift_success_best_consecutive": 0,
                    }
                )
                _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
                return summary

        if cfg.phase == "grasp":
            grasp = _run_bounded_dual_contact_development(
                env,
                base,
                target_env_index,
                cfg,
                alignment,
                episodes,
                trace_rows,
                authority_rows,
                seed_hint=seed_replay,
            )
            summary.update(grasp)
            result_class = str(grasp.get("result_class") or result_class)
            summary.update({"result_class": result_class, "final_allowed_result_class": result_class in RESULT_CLASSES})
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary

        if cfg.phase in {"full", "repeated_trials"}:
            repeated = _run_repeated_trials(
                env, base, target_env_index, cfg, alignment, episodes, trace_rows, authority_rows, seed_hint=seed_replay
            )
            summary.update(repeated)
            result_class = str(repeated.get("result_class") or result_class)
            summary.update({"result_class": result_class, "final_allowed_result_class": result_class in RESULT_CLASSES})
            _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
            return summary

        summary.update({"result_class": result_class, "blocker": f"phase_{cfg.phase}_completed_without_grasp"})
        _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
        return summary
    except Exception as exc:
        summary.update(
            {
                "result_class": "V2_IMPLEMENTATION_ERROR",
                "final_allowed_result_class": True,
                "blocker": f"{type(exc).__name__}:{exc}",
            }
        )
        _write_artifacts(output_dir, summary, episodes, trace_rows, authority_rows, alignment)
        return summary


def _initial_summary(cfg: Screw1GraspBaselineV2Config) -> dict[str, Any]:
    return {
        "baseline": "screw1_grasp_baseline_v2",
        "run_id": str(cfg.run_id or ""),
        "target_part": cfg.part,
        "v2_phase": cfg.phase,
        "v2_collision_offset_fix": cfg.v2_collision_offset_fix,
        "training_locked": True,
        "object_ready": False,
        "grasp_solved": False,
        "success_claimed": False,
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "physical_insert_success": False,
        "oracle_visual_only": False,
        "not_physical": True,
        "proxy_success_used": False,
        "forced_pose_success_used": False,
        "distance_only_success_used": False,
        "unfiltered_only_success_used": False,
        "root_pose_writes_after_reset_used": False,
        "close_executed": False,
        "lift_executed": False,
        "contact_threshold_n": float(cfg.contact_threshold_n),
        "soft_force_band_n": [float(cfg.soft_force_min_n), float(cfg.soft_force_max_n)],
        "hard_abort_force_n": float(cfg.hard_abort_force_n),
        "gpu_preflight_info": dict(cfg.gpu_preflight_info),
        "gpu_preflight_executed": bool(cfg.gpu_preflight_info.get("gpu_preflight_executed")),
        "gpu_preflight_result_class": cfg.gpu_preflight_info.get("gpu_preflight_result_class", ""),
        "result_class": "HEALTH_OK",
        "final_allowed_result_class": False,
    }


def _screw1_env_repair_summary(base: Any) -> dict[str, Any]:
    return {
        "screw1_dynamic_repair_before": _plain(getattr(base, "v92_screw1_dynamic_repair_before", {})),
        "screw1_dynamic_repair_after": _plain(getattr(base, "v92_screw1_dynamic_repair_after", {})),
        "screw1_v2_collision_offset_fix": _plain(getattr(base, "v2_screw1_collision_offset_fix", {})),
        "screw1_v2_wrist_pd_fix_requested": os.environ.get("WUJI_SCREW1_V2_WRIST_PD_FIX", "0"),
    }


def _write_v2_progress(output_dir: Path, stage: str, summary: dict[str, Any]) -> None:
    try:
        write_json(
            output_dir / "v2_progress.json",
            {
                "stage": str(stage),
                "result_class": summary.get("result_class", ""),
                "blocker": summary.get("blocker", ""),
                "target_env_index": summary.get("target_env_index", ""),
                "gpu_preflight_result_class": summary.get("gpu_preflight_result_class", ""),
                "support_calibration_valid": summary.get("canonical_support_pose_valid", ""),
                "wrist_audit_passed": summary.get("wrist_audit_passed", ""),
                "action_authority_required_joints_ok": summary.get("action_authority_required_joints_ok", ""),
                "seed_replay_probe_count": summary.get("seed_replay_probe_count", ""),
                "acquisition_plan_validated": summary.get("acquisition_plan_validated", ""),
                "validated_acquisition_plan_json": summary.get("validated_acquisition_plan_json", ""),
            },
        )
    except Exception:
        return


def _run_health_phase(output_dir: Path) -> dict[str, Any]:
    before = _health_snapshot(output_dir)
    after = _health_snapshot(output_dir)
    current_blocked = _health_resource_blocked({"health_before": before, "health_after": after, "host_resource_temp_write_ok": bool(before.get("temp_write_probe_ok") and after.get("temp_write_probe_ok"))})
    return {
        "health_executed": True,
        "health_cleanup_performed": False,
        "health_cleanup_reason": "no_automatic_cleanup_without_clean_v2_output; compact v2 artifacts by default",
        "health_before": before,
        "health_after": after,
        "errno28_seen_in_local_logs": bool(before.get("local_output_errno28_seen") or after.get("local_output_errno28_seen")),
        "historical_latest_kit_errno28_seen": bool(before.get("latest_kit_errno28_seen") or after.get("latest_kit_errno28_seen")),
        "health_historical_kit_errno28_blocks_launch": False,
        "health_current_resource_blocked": bool(current_blocked),
        "health_result_class": "HOST_RESOURCE_PREFLIGHT_BLOCKED" if current_blocked else "HEALTH_OK",
        "host_resource_temp_write_ok": bool(before.get("temp_write_probe_ok") and after.get("temp_write_probe_ok")),
    }


def _health_snapshot(output_dir: Path) -> dict[str, Any]:
    kit = _latest_kit_errno28()
    inotify = _inotify_watch_usage()
    return {
        "df_h": _run_cmd(["df", "-h"]),
        "du_debug_runs": _du("debug_runs"),
        "du_ov_cache": _du(os.path.expanduser("~/.cache/ov")),
        "du_tmp": _du("/tmp"),
        "shutil_disk_usage_repo": _disk_usage(Path.cwd()),
        **inotify,
        "inotify_max_user_instances": _read_text("/proc/sys/fs/inotify/max_user_instances"),
        "inotify_max_queued_events": _read_text("/proc/sys/fs/inotify/max_queued_events"),
        "local_output_errno28_seen": bool(_scan_errno28(output_dir)),
        "latest_kit_log_path": kit.get("latest_kit_log_path", ""),
        "latest_kit_errno28_seen": bool(kit.get("errno28_seen")),
        "latest_kit_errno28_count": int(kit.get("errno28_count", 0) or 0),
        "latest_kit_errno28_first_lines": kit.get("errno28_first_lines", []),
        **_runtime_temp_write_probe(output_dir),
    }


def _health_resource_blocked(health: dict[str, Any]) -> bool:
    if not bool(health.get("host_resource_temp_write_ok", True)):
        return True
    for key in ("health_before", "health_after"):
        snap = dict(health.get(key, {}) or {})
        disk_free = int(dict(snap.get("shutil_disk_usage_repo", {}) or {}).get("free", 0) or 0)
        if disk_free and disk_free < 2_000_000_000:
            return True
        max_watches = int(snap.get("inotify_max_user_watches", 0) or 0)
        used = int(snap.get("inotify_watch_count_total", 0) or 0)
        free_est = int(snap.get("inotify_watch_free_estimate", 0) or 0)
        if max_watches > 0 and (free_est < 50000 or used >= int(0.90 * max_watches)):
            return True
    for key in ("health_before", "health_after"):
        snap = dict(health.get(key, {}) or {})
        if bool(snap.get("local_output_errno28_seen")):
            return True
    return False


def _runtime_temp_write_probe(output_dir: Path) -> dict[str, Any]:
    path = output_dir / ".v2_runtime_temp_write_probe"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
        path.unlink(missing_ok=True)
        return {"temp_write_probe_ok": True, "temp_write_probe_error": ""}
    except Exception as exc:
        return {"temp_write_probe_ok": False, "temp_write_probe_error": f"{type(exc).__name__}:{exc}"}


def _run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def _du(path: str) -> str:
    return _run_cmd(["du", "-sh", path])


def _disk_usage(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {"total": usage.total, "used": usage.used, "free": usage.free}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}:{exc}"}


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def _scan_errno28(output_dir: Path) -> bool:
    needles = ("errno=28", "No space left on device")
    try:
        candidates = list(output_dir.glob("*.log")) + list((output_dir / "logs").glob("*.log"))
    except Exception:
        candidates = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(item in text for item in needles):
            return True
    return False


def _latest_kit_errno28() -> dict[str, Any]:
    log_root = Path.home() / "miniconda3/envs/isaac/lib/python3.10/site-packages/omni/logs/Kit/Isaac-Sim/4.5"
    try:
        logs = sorted(log_root.glob("kit_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    except Exception:
        logs = []
    if not logs:
        return {"latest_kit_log_path": "", "errno28_seen": False, "errno28_count": 0, "errno28_first_lines": []}
    path = logs[0]
    lines: list[str] = []
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if "errno=28" in line or "No space left on device" in line:
                    count += 1
                    if len(lines) < 10:
                        lines.append(line.strip())
    except Exception as exc:
        return {
            "latest_kit_log_path": str(path),
            "errno28_seen": False,
            "errno28_count": 0,
            "errno28_first_lines": [f"{type(exc).__name__}:{exc}"],
        }
    return {
        "latest_kit_log_path": str(path),
        "errno28_seen": count > 0,
        "errno28_count": count,
        "errno28_first_lines": lines,
    }


def _inotify_watch_consumers_top(limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        pid = proc.name
        count = 0
        try:
            for fdinfo in (proc / "fdinfo").glob("*"):
                try:
                    with fdinfo.open("r", encoding="utf-8", errors="ignore") as stream:
                        for line in stream:
                            if line.startswith("inotify"):
                                count += 1
                except Exception:
                    continue
        except Exception:
            continue
        if count <= 0:
            continue
        try:
            comm = (proc / "comm").read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            comm = ""
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        except Exception:
            cmdline = ""
        rows.append({"watch_count": count, "pid": int(pid), "comm": comm, "cmdline": cmdline[:240]})
    rows.sort(key=lambda row: int(row.get("watch_count", 0)), reverse=True)
    return rows[: int(limit)]


def _inotify_watch_usage() -> dict[str, Any]:
    rows = _inotify_watch_consumers_top(limit=1000000)
    try:
        max_watches = int(Path("/proc/sys/fs/inotify/max_user_watches").read_text(encoding="utf-8").strip())
    except Exception:
        max_watches = 0
    used = sum(int(row.get("watch_count", 0) or 0) for row in rows)
    free_est = max(0, max_watches - used) if max_watches > 0 else 0
    return {
        "inotify_max_user_watches": max_watches,
        "inotify_watch_count_total": used,
        "inotify_watch_count_top_sum": sum(int(row.get("watch_count", 0) or 0) for row in rows[:10]),
        "inotify_watch_free_estimate": free_est,
        "inotify_watch_consumers_top": rows[:10],
    }


def _list_get(values: Any, index: int, default: Any = "") -> Any:
    try:
        return list(values)[int(index)]
    except Exception:
        return default


def _canonical_hand_park_pos(cfg: Screw1GraspBaselineV2Config, object_pos: list[float]) -> list[float]:
    offset = list(cfg.canonical_hand_park_offset_m)
    return [float(object_pos[i]) + float(offset[i]) for i in range(3)]


def _canonical_pregrasp_base_pos(object_pos: list[float]) -> list[float]:
    return [float(object_pos[0]) - 0.110, float(object_pos[1]) + 0.018, float(object_pos[2]) + 0.018]


def _seed_target_pose(
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    seed_offset: tuple[float, float, float],
    seed_rpy: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    object_pos = list(state.get("object_local_pos", []))
    if len(object_pos) < 3:
        object_pos = list(cfg.canonical_support_pose.get("object_local_pos") or [cfg.canonical_xy_m[0], cfg.canonical_xy_m[1], 0.8])
    base_hand = _canonical_pregrasp_base_pos(object_pos)
    target = [float(base_hand[i]) + float(seed_offset[i]) for i in range(3)]
    target_quat = _quat_for_rpy_offset(_quat_normalize_wxyz(cfg.canonical_hand_park_quat_wxyz), tuple(seed_rpy))
    return target, target_quat


def _barrier_safe_seed_target(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    target: list[float],
    target_quat: list[float],
) -> tuple[list[float], dict[str, Any]]:
    """Lift seed targets just enough to satisfy the v2 palm/table barrier."""

    adjusted = [float(value) for value in list(target)[:3]]
    info: dict[str, Any] = {
        "seed_target_original_local_xyz": list(adjusted),
        "seed_target_barrier_adjustment_z_m": 0.0,
        "seed_target_table_top_z_m": "",
        "seed_target_barrier_min_target_z_m": "",
        "seed_target_grasp_frame_offset_z_m": "",
        "seed_target_table_barrier_margin_m": float(cfg.seed_target_table_barrier_margin_m),
    }
    if len(adjusted) < 3:
        return adjusted, info
    try:
        table_top = cfg.canonical_support_pose.get("table_top_z_m", "")
        table_top = float(table_top)
        if not math.isfinite(table_top):
            audit = _v2_table_override_audit(base, env_index)
            table_top = float(audit.get("v2_table_top_true_z_m", ""))
        if not math.isfinite(table_top):
            return adjusted, info
        grasp_offset = _flat_vec(getattr(base, "dex_grasp_frame_local_pos", None), 3)
        offset_world = _quat_apply_wxyz(target_quat, grasp_offset)
        clearance = float(getattr(base, "floating_table_clearance_min", 0.035))
        margin = float(cfg.seed_target_table_barrier_margin_m)
        min_target_z = table_top + clearance + float(offset_world[2]) + margin
        info.update(
            {
                "seed_target_table_top_z_m": table_top,
                "seed_target_barrier_min_target_z_m": min_target_z,
                "seed_target_grasp_frame_offset_z_m": float(offset_world[2]),
            }
        )
        if adjusted[2] < min_target_z:
            dz = min_target_z - adjusted[2]
            adjusted[2] = min_target_z
            info["seed_target_barrier_adjustment_z_m"] = dz
    except Exception as exc:
        info["seed_target_barrier_safe_error"] = f"{type(exc).__name__}:{exc}"
    return adjusted, info


def _floating_table_z_audit(base: Any, env_index: int) -> Any:
    value = getattr(base, "floating_table_z_est", None)
    if torch is not None and torch.is_tensor(value):
        try:
            return float(value[int(env_index)].detach().cpu().item())
        except Exception:
            return ""
    return ""


def _set_v2_table_top_override(base: Any, env_index: int, table_top_z: float | None) -> None:
    if torch is None:
        return
    override = getattr(base, "v2_table_top_z_override", None)
    active = getattr(base, "v2_table_top_override_env_ids", None)
    if not (torch.is_tensor(override) and torch.is_tensor(active)):
        return
    try:
        env_id = int(env_index)
        if table_top_z is None or not math.isfinite(float(table_top_z)):
            active[env_id] = False
            override[env_id] = float("nan")
        else:
            active[env_id] = True
            override[env_id] = float(table_top_z)
        if hasattr(base, "_refresh_floating_table_estimate"):
            ids = torch.tensor([env_id], dtype=torch.long, device=base.device)
            base._refresh_floating_table_estimate(ids)
    except Exception:
        return


def _v2_table_override_audit(base: Any, env_index: int) -> dict[str, Any]:
    out = {
        "v2_table_top_override_active": False,
        "v2_table_top_true_z_m": "",
        "v2_table_top_old_floating_estimate_m": "",
    }
    if torch is None:
        return out
    try:
        env_id = int(env_index)
        active = getattr(base, "v2_table_top_override_env_ids", None)
        override = getattr(base, "v2_table_top_z_override", None)
        old = getattr(base, "v2_table_top_old_floating_estimate", None)
        if torch.is_tensor(active):
            out["v2_table_top_override_active"] = bool(active[env_id].detach().cpu().item())
        if torch.is_tensor(override):
            value = float(override[env_id].detach().cpu().item())
            out["v2_table_top_true_z_m"] = value if math.isfinite(value) else ""
        if torch.is_tensor(old):
            value = float(old[env_id].detach().cpu().item())
            out["v2_table_top_old_floating_estimate_m"] = value if math.isfinite(value) else ""
    except Exception:
        pass
    return out


def _env_origin_xyz(base: Any, env_index: int) -> list[float]:
    origins = getattr(getattr(base, "scene", None), "env_origins", None)
    if torch is not None and torch.is_tensor(origins):
        try:
            return [float(value) for value in origins[int(env_index)].detach().cpu().tolist()[:3]]
        except Exception:
            pass
    return [0.0, 0.0, 0.0]


def _table_top_metadata(base: Any, env_index: int) -> dict[str, Any]:
    path = f"/World/envs/env_{int(env_index)}/Table"
    env_origin = _env_origin_xyz(base, env_index)
    out: dict[str, Any] = {
        "table_prim_path": path,
        "table_top_z_m": "",
        "table_top_source": "",
        "table_top_error": "",
        "env_origin_xyz": env_origin,
    }
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"table_prim_missing:{path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = box.GetMin()
        mx = box.GetMax()
        world_min = [float(mn[i]) for i in range(3)]
        world_max = [float(mx[i]) for i in range(3)]
        local_min = [world_min[i] - env_origin[i] for i in range(3)]
        local_max = [world_max[i] - env_origin[i] for i in range(3)]
        out.update(
            {
                "table_top_z_m": local_max[2],
                "table_top_world_z_m": world_max[2],
                "table_top_source": "runtime_usd_bbox_top_env_local",
                "table_bbox_min_xyz": local_min,
                "table_bbox_max_xyz": local_max,
                "table_bbox_world_min_xyz": world_min,
                "table_bbox_world_max_xyz": world_max,
            }
        )
    except Exception as exc:
        out["table_top_error"] = f"{type(exc).__name__}:{exc}"
    return out


def _part_bbox_bottom_z(part: str, env_index: int) -> Any:
    path = f"/World/envs/env_{int(env_index)}/{str(part)}"
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return ""
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            return ""
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        return float(cache.ComputeWorldBound(prim).ComputeAlignedBox().GetMin()[2])
    except Exception:
        return ""


def _screw1_bbox_bottom_z(env_index: int) -> Any:
    return _part_bbox_bottom_z("Screw1", env_index)


def _calibrate_canonical_support_pose(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    table = _table_top_metadata(base, env_index)
    table_top = float(table.get("table_top_z_m", math.nan) or math.nan)
    if not math.isfinite(table_top):
        return {
            "support_calibration_executed": True,
            "canonical_support_pose_valid": False,
            "canonical_support_pose_failure_reason": table.get("table_top_error", "table_top_unavailable"),
            **table,
        }
    _set_v2_table_top_override(base, env_index, table_top)
    object_pos = [float(cfg.canonical_xy_m[0]), float(cfg.canonical_xy_m[1]), table_top + float(cfg.support_drop_height_m)]
    object_quat = [1.0, 0.0, 0.0, 0.0]
    plan = _build_v2_reset_plan(
        base,
        env_index,
        cfg,
        read_state(base, env_index, cfg.part, cfg.contact_threshold_n),
        hand_mode="parked",
        seed_offset=None,
        seed_rpy=None,
        object_pos=object_pos,
        object_quat=object_quat,
        table=table,
        reset_reason="support_calibration_drop",
    )
    _reset_once_with_v2_plan(env, base, env_index, cfg, plan)
    _configure_v2_control(base, env_index, mode="")
    reset_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    reset_z = float(reset_state["object_local_pos"][2])
    settled_steps = 0
    final_state = reset_state
    peak_speed = 0.0
    peak_ang = 0.0
    for step in range(int(cfg.support_calibration_max_steps)):
        final_state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase="support_calibration",
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "planned_object_z": object_pos[2],
                "actual_reset_object_z": reset_z,
                "table_top_z_m": table_top,
                "table_top_source": table.get("table_top_source", ""),
            },
        )
        vel = _object_velocity(base, env_index, cfg.part)
        speed = float(vel.get("linear_velocity_norm", 0.0) or 0.0)
        ang = float(vel.get("angular_velocity_norm", 0.0) or 0.0)
        peak_speed = max(peak_speed, speed)
        peak_ang = max(peak_ang, ang)
        if speed < cfg.support_settle_linear_speed_mps and ang < cfg.support_settle_angular_speed_radps:
            settled_steps += 1
        else:
            settled_steps = 0
        if settled_steps >= int(cfg.support_settle_required_steps):
            break
    final_pos = list(final_state.get("object_local_pos", []))
    final_quat = _quat_normalize_wxyz(final_state.get("object_quat_wxyz", object_quat))
    cfg.canonical_support_pose = {
        "object_local_pos": [
            float(final_pos[0]) if len(final_pos) > 0 else float(cfg.canonical_xy_m[0]),
            float(final_pos[1]) if len(final_pos) > 1 else float(cfg.canonical_xy_m[1]),
            (float(final_pos[2]) if len(final_pos) > 2 else table_top) + float(cfg.support_spawn_clearance_m),
        ],
        "settled_object_local_pos": final_pos,
        "object_quat_wxyz": final_quat,
        "table_top_z_m": table_top,
        "table_top_source": table.get("table_top_source", ""),
        "table_prim_path": table.get("table_prim_path", ""),
        "support_calibration_steps": int(step + 1 if "step" in locals() else 0),
        "support_settled_window_steps": int(settled_steps),
    }
    cfg.canonical_support_pose.update(
        {
            "canonical_object_spawn_pos": list(cfg.canonical_support_pose["object_local_pos"]),
            "canonical_object_quat": list(final_quat),
            "canonical_hand_park_pos": _canonical_hand_park_pos(cfg, cfg.canonical_support_pose["object_local_pos"]),
            "canonical_hand_park_quat": list(_quat_normalize_wxyz(cfg.canonical_hand_park_quat_wxyz)),
            "canonical_hand_preshape": _hand_pose_list(base, "dex_hand_preshape_pose"),
            "canonical_table_top_local_z": table_top,
        }
    )
    bbox_bottom = _part_bbox_bottom_z(cfg.part, env_index)
    bottom_gap = ""
    bbox_bottom_usable = False
    try:
        bottom_gap = float(bbox_bottom) - table_top
        bbox_bottom_usable = bool(len(final_pos) >= 3 and float(bbox_bottom) <= float(final_pos[2]) + 0.010)
    except Exception:
        bottom_gap = ""
    bbox_min = table.get("table_bbox_min_xyz", [])
    bbox_max = table.get("table_bbox_max_xyz", [])
    try:
        xy_inside_table = bool(
            float(bbox_min[0]) <= float(final_pos[0]) <= float(bbox_max[0])
            and float(bbox_min[1]) <= float(final_pos[1]) <= float(bbox_max[1])
        )
    except Exception:
        xy_inside_table = True
    finite_pose = bool(
        len(final_pos) >= 3
        and all(math.isfinite(float(value)) for value in final_pos[:3])
        and all(math.isfinite(float(value)) for value in final_quat[:4])
    )
    hand_force_peak = max(
        float(final_state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
        float(final_state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
    )
    object_not_ground = bool(len(final_pos) >= 3 and float(final_pos[2]) > table_top - 0.005)
    bottom_near_table = True if not bbox_bottom_usable or bottom_gap == "" else bool(abs(float(bottom_gap)) <= 0.030)
    valid = bool(
        settled_steps >= int(cfg.support_settle_required_steps)
        and finite_pose
        and xy_inside_table
        and object_not_ground
        and bottom_near_table
        and hand_force_peak < cfg.contact_threshold_n
    )
    if not valid:
        cfg.canonical_support_pose = {}
    failure_reasons = []
    if settled_steps < int(cfg.support_settle_required_steps):
        failure_reasons.append("support_did_not_settle_within_max_steps")
    if not finite_pose:
        failure_reasons.append("nonfinite_final_pose")
    if not xy_inside_table:
        failure_reasons.append("object_xy_outside_table_footprint")
    if not object_not_ground:
        failure_reasons.append("object_not_supported_above_table")
    if bbox_bottom_usable and not bottom_near_table:
        failure_reasons.append("bbox_bottom_not_near_table_top")
    if hand_force_peak >= cfg.contact_threshold_n:
        failure_reasons.append("hand_contact_during_support_calibration")
    return {
        "support_calibration_executed": True,
        "canonical_support_pose_valid": valid,
        "canonical_support_pose_failure_reason": "" if valid else "|".join(failure_reasons),
        "canonical_support_pose": _plain(cfg.canonical_support_pose),
        "support_calibration_planned_object_z": object_pos[2],
        "support_calibration_actual_reset_object_z": reset_z,
        "support_calibration_final_object_z": final_state["object_local_pos"][2],
        "support_calibration_peak_speed_mps": peak_speed,
        "support_calibration_peak_angular_speed_radps": peak_ang,
        "support_calibration_screw1_bbox_bottom_z": bbox_bottom,
        "support_calibration_bbox_bottom_to_table_gap_m": bottom_gap,
        "support_calibration_bbox_bottom_usable": bbox_bottom_usable,
        "support_calibration_xy_inside_table_footprint": xy_inside_table,
        "support_calibration_finite_pose": finite_pose,
        "support_calibration_object_not_ground": object_not_ground,
        "support_calibration_bbox_bottom_near_table": bottom_near_table,
        "support_calibration_hand_force_peak_n": hand_force_peak,
        **_v2_table_override_audit(base, env_index),
        **table,
    }


def _fresh_episode(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    episode_tag: str,
    *,
    hand_mode: str = "parked",
    seed_offset: tuple[float, float, float] | None = None,
    seed_rpy: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    if torch is not None:
        try:
            torch.manual_seed(int(cfg.deterministic_seed))
        except Exception:
            pass
    _refresh_runtime(base)
    first_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    table = {
        "table_top_z_m": cfg.canonical_support_pose.get("table_top_z_m", ""),
        "table_top_source": cfg.canonical_support_pose.get("table_top_source", ""),
        "table_prim_path": cfg.canonical_support_pose.get("table_prim_path", ""),
    }
    try:
        _set_v2_table_top_override(base, env_index, float(table["table_top_z_m"]))
    except Exception:
        pass
    plan = _build_v2_reset_plan(
        base,
        env_index,
        cfg,
        first_state,
        hand_mode=hand_mode,
        seed_offset=seed_offset,
        seed_rpy=seed_rpy,
        table=table,
        reset_reason=episode_tag,
    )
    _reset_once_with_v2_plan(env, base, env_index, cfg, plan)
    reset_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    for step in range(max(0, int(cfg.settle_steps))):
        _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase=f"{episode_tag}_settle",
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
        )
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    if alignment is not None:
        try:
            alignment.configure_camera(env, base, state)
        except Exception:
            pass
    fingerprint = _capture_fingerprint(base, env_index, cfg, state)
    episode_row = {
        "phase": episode_tag,
        "episode_tag": episode_tag,
        "hand_mode": hand_mode,
        "seed": int(cfg.deterministic_seed),
        "object_x": state["object_local_pos"][0],
        "object_y": state["object_local_pos"][1],
        "object_z": state["object_local_pos"][2],
        "palm_x": state["palm_local_pos"][0],
        "palm_y": state["palm_local_pos"][1],
        "palm_z": state["palm_local_pos"][2],
        "finger3_force_n": state["finger3_target_filtered_force_n"],
        "finger4_force_n": state["finger4_target_filtered_force_n"],
        "target_contact": _target_contact_acquired(state, cfg),
        "fingerprint_json": json.dumps(_plain(fingerprint), sort_keys=True),
        "reset_plan_used": "v2_explicit_canonical_support_reset_once",
        "object_write_after_reset_used": False,
        "planned_object_z": _list_get(plan[0].get("object_center_local_xyz", []), 2, ""),
        "actual_reset_object_z": reset_state["object_local_pos"][2],
        "post_settle_object_z": state["object_local_pos"][2],
        "floating_table_z_est_for_audit_only": _floating_table_z_audit(base, env_index),
        **_v2_table_override_audit(base, env_index),
        "table_top_z_m": plan[0].get("table_top_z_m", ""),
        "table_top_source": plan[0].get("table_top_source", ""),
        "part_bbox_bottom_z": _part_bbox_bottom_z(cfg.part, env_index),
    }
    return {"state": state, "fingerprint": fingerprint, "episode_row": episode_row, "reset_plan": plan}


def _build_v2_reset_plan(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    first_state: dict[str, Any],
    *,
    hand_mode: str,
    seed_offset: tuple[float, float, float] | None,
    seed_rpy: tuple[float, float, float] | None,
    object_pos: list[float] | None = None,
    object_quat: list[float] | None = None,
    table: dict[str, Any] | None = None,
    reset_reason: str = "",
) -> list[dict[str, Any]]:
    table = dict(table or {})
    support_pose = dict(cfg.canonical_support_pose or {})
    if object_pos is None:
        object_pos = list(support_pose.get("object_local_pos") or first_state.get("object_local_pos") or [cfg.canonical_xy_m[0], cfg.canonical_xy_m[1], 0.82])
    if object_quat is None:
        object_quat = _quat_normalize_wxyz(support_pose.get("object_quat_wxyz") or first_state.get("object_quat_wxyz") or [1.0, 0.0, 0.0, 0.0])
    canonical_hand_quat = _quat_normalize_wxyz(
        support_pose.get("canonical_hand_park_quat") or cfg.canonical_hand_park_quat_wxyz
    )
    if hand_mode == "parked":
        hand_target = list(support_pose.get("canonical_hand_park_pos") or _canonical_hand_park_pos(cfg, object_pos))
        hand_quat = canonical_hand_quat
    else:
        base_hand = _canonical_pregrasp_base_pos(object_pos)
        if seed_offset is not None:
            hand_target = [float(base_hand[i]) + float(seed_offset[i]) for i in range(3)]
        else:
            hand_target = base_hand
        hand_quat = _quat_for_rpy_offset(canonical_hand_quat, tuple(seed_rpy or (0.0, 0.0, 0.0)))
    row = {
        "part_name": cfg.part,
        "env_index": int(env_index),
        "local_env_index": int(env_index),
        "initial_condition_mode": "v2_explicit_canonical_support" if support_pose else "v2_support_calibration_drop",
        "support_surface": "v2_table_runtime_bbox_or_raycast",
        "reset_reason": reset_reason,
        "table_top_z_m": table.get("table_top_z_m", support_pose.get("table_top_z_m", "")),
        "table_top_source": table.get("table_top_source", support_pose.get("table_top_source", "")),
        "table_prim_path": table.get("table_prim_path", support_pose.get("table_prim_path", "")),
        "object_support_half_height_m": "",
        "object_center_local_xyz": list(object_pos),
        "object_quat_wxyz": list(object_quat),
        "hand_target_local_xyz": hand_target,
        "hand_target_quat_wxyz": hand_quat,
        "hand_quat_source": "v2_fixed_canonical_park_quat_plus_seed_rpy" if seed_rpy else "v2_fixed_canonical_park_quat",
        "active_finger_group": cfg.active_finger_group,
        "v2_reset_protocol": True,
        "object_write_after_reset_allowed": False,
        "reset_only_object_write_allowed": True,
        "reset_only_hand_write_allowed": True,
    }
    return [row]


def _reset_once_with_v2_plan(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    plan: list[dict[str, Any]],
) -> None:
    if hasattr(base, "v95_clear_pregrasp"):
        base.v95_clear_pregrasp()
    if hasattr(base, "v95_configure_pregrasp"):
        base.v95_configure_pregrasp(plan)
    try:
        env.reset(seed=int(cfg.deterministic_seed))
    except TypeError:
        env.reset()
    _refresh_runtime(base)
    _configure_v2_control(base, env_index)


def _wait_for_object_settled(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    step_offset: int = 0,
) -> dict[str, Any]:
    settled_steps = 0
    final_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    peak_speed = 0.0
    peak_ang = 0.0
    for step in range(int(cfg.support_calibration_max_steps)):
        final_state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase=phase,
            step=int(step_offset) + step,
            trace_rows=trace_rows,
            alignment=alignment,
        )
        vel = _object_velocity(base, env_index, cfg.part)
        speed = float(vel.get("linear_velocity_norm", 0.0) or 0.0)
        ang = float(vel.get("angular_velocity_norm", 0.0) or 0.0)
        peak_speed = max(peak_speed, speed)
        peak_ang = max(peak_ang, ang)
        if speed < cfg.support_settle_linear_speed_mps and ang < cfg.support_settle_angular_speed_radps:
            settled_steps += 1
        else:
            settled_steps = 0
        if settled_steps >= int(cfg.support_settle_required_steps):
            break
    return {
        "settled": bool(settled_steps >= int(cfg.support_settle_required_steps)),
        "steps": int(step + 1 if "step" in locals() else 0),
        "settled_window_steps": int(settled_steps),
        "peak_speed_mps": peak_speed,
        "peak_angular_speed_radps": peak_ang,
        "final_state": final_state,
    }


def _configure_v2_control(base: Any, env_index: int, *, mode: str = "") -> None:
    if torch is None:
        return
    try:
        env_ids = torch.tensor([env_index], dtype=torch.long, device=base.device)
        if hasattr(base, "v95_configure_action_control"):
            base.v95_configure_action_control(mode=mode, allow_commanded_hand_actions=True, env_ids=env_ids)
    except Exception:
        pass


def _run_no_hand_stability(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    max_drift = 0.0
    max_speed = 0.0
    max_ang_speed = 0.0
    max_force = 0.0
    max_recursive_z_delta = 0.0
    reference_fp: dict[str, Any] | None = None
    max_fp_delta = 0.0
    prev_actual_z: float | None = None
    for trial in range(int(cfg.no_hand_trials)):
        fresh = _fresh_episode(
            env, base, env_index, cfg, alignment, trace_rows, f"no_hand_trial_{trial}", hand_mode="parked"
        )
        reset_state = fresh["state"]
        reset_pos = list(reset_state["object_local_pos"])
        immediate_reset_z = float(dict(fresh.get("episode_row", {}) or {}).get("actual_reset_object_z", reset_pos[2]))
        fp = dict(fresh.get("fingerprint", {}) or {})
        if reference_fp is None:
            reference_fp = fp
        else:
            max_fp_delta = max(max_fp_delta, _fingerprint_delta(reference_fp, fp))
        settle = _wait_for_object_settled(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase="no_hand_post_reset_settle",
            step_offset=trial * int(cfg.support_calibration_max_steps),
        )
        post_settle_state = settle["final_state"]
        start_eval_pos = list(post_settle_state["object_local_pos"])
        peak_force = 0.0
        speeds: list[float] = []
        angs: list[float] = []
        final_state = post_settle_state
        for step in range(int(cfg.no_hand_steps)):
            final_state = _step_direct_action(
                env,
                base,
                env_index,
                cfg,
                _zero_action(env, base),
                phase="no_hand_evaluation",
                step=trial * int(cfg.no_hand_steps) + step,
                trace_rows=trace_rows,
                alignment=alignment,
            )
            peak_force = max(peak_force, float(final_state.get("active_target_filtered_force_peak_n", 0.0) or 0.0))
            vel = _object_velocity(base, env_index, cfg.part)
            speeds.append(float(vel.get("linear_velocity_norm", 0.0) or 0.0))
            angs.append(float(vel.get("angular_velocity_norm", 0.0) or 0.0))
        steady_speed_max = max([0.0, *speeds[-60:]])
        steady_ang_max = max([0.0, *angs[-60:]])
        steady_speed_mean = sum(speeds[-60:]) / max(1, len(speeds[-60:]))
        steady_ang_mean = sum(angs[-60:]) / max(1, len(angs[-60:]))
        drift = _distance(start_eval_pos, final_state["object_local_pos"])
        actual_z = immediate_reset_z
        recursive_z_delta = 0.0 if prev_actual_z is None else abs(actual_z - prev_actual_z)
        prev_actual_z = actual_z
        max_recursive_z_delta = max(max_recursive_z_delta, recursive_z_delta)
        planned_z = _list_get(fresh.get("reset_plan", [{}])[0].get("object_center_local_xyz", []), 2, "")
        row = {
            "phase": "no_hand_corrected",
            "trial": trial,
            "planned_object_z": planned_z,
            "actual_reset_object_z": actual_z,
            "post_settle_object_z": post_settle_state["object_local_pos"][2],
            "evaluation_final_object_z": final_state["object_local_pos"][2],
            "floating_table_z_est_for_audit_only": _floating_table_z_audit(base, env_index),
            **_v2_table_override_audit(base, env_index),
            "table_top_z_m": cfg.canonical_support_pose.get("table_top_z_m", ""),
            "table_top_source": cfg.canonical_support_pose.get("table_top_source", ""),
            "part_bbox_bottom_z": _part_bbox_bottom_z(cfg.part, env_index),
            "reset_to_reset_z_delta_m": recursive_z_delta,
            "settle_steps": settle.get("steps", 0),
            "settled_before_evaluation": bool(settle.get("settled")),
            "object_drift_m": drift,
            "object_delta_z_m": float(final_state["object_local_pos"][2]) - float(start_eval_pos[2]),
            "peak_target_force_n": peak_force,
            "steady_linear_velocity_mean_mps": steady_speed_mean,
            "steady_linear_velocity_max_mps": steady_speed_max,
            "steady_angular_velocity_mean_radps": steady_ang_mean,
            "steady_angular_velocity_max_radps": steady_ang_max,
            "passed": bool(
                bool(settle.get("settled"))
                and drift <= cfg.no_hand_max_drift_m
                and steady_speed_max <= cfg.no_hand_max_speed_mps
                and steady_ang_max <= cfg.no_hand_max_angular_speed_radps
                and peak_force < cfg.contact_threshold_n
                and recursive_z_delta <= cfg.support_reset_pose_tolerance_m
            ),
        }
        rows.append(row)
        episodes.append(row)
        max_drift = max(max_drift, drift)
        max_speed = max(max_speed, steady_speed_max)
        max_ang_speed = max(max_ang_speed, steady_ang_max)
        max_force = max(max_force, peak_force)
    passed = all(bool(row.get("passed")) for row in rows)
    return {
        "no_hand_stability_executed": True,
        "no_hand_protocol": "canonical_reset_post_settle_then_steady_evaluation",
        "no_hand_trials": int(cfg.no_hand_trials),
        "no_hand_stability_passed": bool(passed),
        "no_hand_object_drift_peak_m": max_drift,
        "no_hand_steady_speed_peak_mps": max_speed,
        "no_hand_steady_angular_speed_peak_radps": max_ang_speed,
        "no_hand_target_force_peak_n": max_force,
        "no_hand_reset_recursive_z_delta_peak_m": max_recursive_z_delta,
        "fresh_reset_fingerprint_delta_peak": max_fp_delta,
    }


def _run_wrist_control_audit(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    output_dir = Path(cfg.output_dir)
    static = _run_wrist_quaternion_static_tests()
    reset = _run_wrist_reset_consistency(env, base, env_index, cfg, alignment, trace_rows, rows)
    axis_summaries: list[dict[str, Any]] = []
    for axis_name, delta_xyz, delta_rot, joint_axis, command_value in _wrist_axis_probe_specs(cfg):
        axis_summaries.append(
            _run_wrist_axis_probe(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                rows,
                axis_name=axis_name,
                delta_xyz=delta_xyz,
                delta_rot=delta_rot,
                joint_axis=joint_axis,
                command_value=command_value,
            )
        )
    run_seed_path_diagnostic = bool(cfg.phase == "wrist_audit")
    if run_seed_path_diagnostic:
        translation = _run_wrist_sequence_probe(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            rows,
            audit_test="translation_only_seed_standoff",
            sequence_builder=lambda state: _wrist_staged_seed_standoff_sequence(
                base,
                env_index,
                cfg,
                state,
                seed_offset=cfg.seed_a_position_offset_m,
                seed_rpy=cfg.seed_a_orientation_rpy_deg,
                use_seed_orientation=False,
                sequence_mode="translation_only",
            ),
        )
    else:
        translation = {
            "sequence_passed": True,
            "termination_reason": "skipped_outside_wrist_audit_phase",
            "first_failed_stage": "",
            "final_target_to_actual_position_error_m": "",
            "final_target_to_actual_angle_error_deg": "",
        }
        rows.append(
            {
                "phase": "wrist_audit",
                "audit_test": "translation_only_seed_standoff",
                "skipped": True,
                "skip_reason": "covered_by_wrist_translation_isolation_or_seed_replay_path",
            }
        )
    rotation = _run_wrist_sequence_probe(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        rows,
        audit_test="rotation_only_seed_orientation",
        sequence_builder=lambda state: [
            (
                "rotation_only",
                list(state.get("palm_local_pos", [])),
                _wrist_seed_a_quat(cfg),
                {"wrist_audit_safe_target_mode": "parked_position_seed_a_rotation"},
            )
        ],
    )
    core_wrist_pass = bool(
        static.get("quaternion_static_tests_passed")
        and reset.get("consistency_passed")
        and all(bool(row.get("axis_passed")) for row in axis_summaries)
        and rotation.get("sequence_passed")
    )
    seed_standoff_pass = bool(translation.get("sequence_passed"))
    preliminary_pass = bool(core_wrist_pass and seed_standoff_pass)
    combined_results: list[dict[str, Any]] = []
    if run_seed_path_diagnostic and preliminary_pass:
        combined_results.append(
            _run_wrist_sequence_probe(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                rows,
                audit_test="combined_orientation_then_translation",
                sequence_builder=lambda state: _wrist_combined_sequence(
                    base, env_index, cfg, state, orientation_first=True
                ),
            )
        )
        combined_results.append(
            _run_wrist_sequence_probe(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                rows,
                audit_test="combined_translation_then_orientation",
                sequence_builder=lambda state: _wrist_combined_sequence(
                    base, env_index, cfg, state, orientation_first=False
                ),
            )
        )
    else:
        rows.append(
            {
                "phase": "wrist_audit",
                "audit_test": "combined_pose",
                "skipped": True,
                "skip_reason": "reset_axis_or_single_mode_audit_failed",
            }
        )
    combined_pass = bool(
        (not run_seed_path_diagnostic)
        or (combined_results and any(bool(row.get("sequence_passed")) for row in combined_results))
    )
    failed_parts = []
    if not bool(static.get("quaternion_static_tests_passed")):
        failed_parts.append("quaternion_static_tests")
    if not bool(reset.get("consistency_passed")):
        failed_parts.append("reset_consistency")
    if not all(bool(row.get("axis_passed")) for row in axis_summaries):
        failed_parts.append("six_axis_authority")
    if not bool(rotation.get("sequence_passed")):
        failed_parts.append("rotation_only_seed_orientation")
    if run_seed_path_diagnostic and preliminary_pass and not combined_pass:
        failed_parts.append("combined_pose")
    passed = bool(not failed_parts)
    seed_path_geometry_issue = bool(run_seed_path_diagnostic and core_wrist_pass and not seed_standoff_pass)
    best_combined = _best_wrist_sequence_result(combined_results)
    summary = {
        "wrist_audit_executed": True,
        "wrist_audit_passed": passed,
        "wrist_audit_failure_reason": "" if passed else "|".join(failed_parts),
        "wrist_audit_core_passed": bool(core_wrist_pass),
        "wrist_audit_seed_standoff_path_passed": bool(seed_standoff_pass),
        "wrist_audit_seed_path_diagnostic_executed": bool(run_seed_path_diagnostic),
        "wrist_audit_seed_path_geometry_issue": seed_path_geometry_issue,
        "wrist_audit_seed_path_failure_reason": ""
        if seed_standoff_pass
        else str(translation.get("termination_reason") or translation.get("first_failed_stage") or "seed_standoff_path_failed"),
        "wrist_audit_seed_path_note": "diagnostic_only_not_wrist_control_blocker"
        if seed_path_geometry_issue
        else "",
        "wrist_audit_trace_csv": str(output_dir / "wrist_audit_trace.csv"),
        "wrist_audit_summary_json": str(output_dir / "wrist_audit_summary.json"),
        "wrist_audit_reset_pos_tol_m": float(cfg.wrist_audit_reset_pos_tol_m),
        "wrist_audit_reset_rot_tol_deg": float(cfg.wrist_audit_reset_rot_tol_deg),
        "wrist_audit_combined_pos_tol_m": float(cfg.wrist_audit_combined_pos_tol_m),
        "wrist_audit_combined_rot_tol_deg": float(cfg.wrist_audit_combined_rot_tol_deg),
        "wrist_audit_position_lead_limit_m": float(cfg.seed_servo_max_target_lead_m),
        "wrist_audit_rotation_lead_limit_deg": float(cfg.seed_servo_max_rotation_target_lead_deg),
        **static,
        **{f"wrist_audit_reset_{key}": value for key, value in reset.items()},
        "wrist_audit_axis_probe_count": len(axis_summaries),
        "wrist_audit_axis_pass_count": sum(1 for row in axis_summaries if bool(row.get("axis_passed"))),
        "wrist_audit_axis_failures": [
            str(row.get("axis_name")) for row in axis_summaries if not bool(row.get("axis_passed"))
        ],
        "wrist_audit_translation_only_passed": bool(translation.get("sequence_passed")),
        "wrist_audit_translation_only_first_failed_stage": translation.get("first_failed_stage", ""),
        "wrist_audit_translation_only_termination_reason": translation.get("termination_reason", ""),
        "wrist_audit_translation_only_final_target_to_actual_pos_error_m": translation.get(
            "final_target_to_actual_position_error_m", ""
        ),
        "wrist_audit_translation_only_final_target_to_actual_angle_error_deg": translation.get(
            "final_target_to_actual_angle_error_deg", ""
        ),
        "wrist_audit_rotation_only_passed": bool(rotation.get("sequence_passed")),
        "wrist_audit_rotation_only_final_target_to_actual_pos_error_m": rotation.get(
            "final_target_to_actual_position_error_m", ""
        ),
        "wrist_audit_rotation_only_final_target_to_actual_angle_error_deg": rotation.get(
            "final_target_to_actual_angle_error_deg", ""
        ),
        "wrist_audit_combined_pose_executed": bool(combined_results),
        "wrist_audit_combined_pose_passed": combined_pass,
        "wrist_audit_best_combined_test": best_combined.get("audit_test", ""),
        "wrist_audit_best_combined_order": best_combined.get("sequence_order", ""),
        "wrist_audit_best_combined_final_target_to_actual_pos_error_m": best_combined.get(
            "final_target_to_actual_position_error_m", ""
        ),
        "wrist_audit_best_combined_final_target_to_actual_angle_error_deg": best_combined.get(
            "final_target_to_actual_angle_error_deg", ""
        ),
    }
    episodes.append(
        {
            "phase": "wrist_audit",
            "wrist_audit_passed": passed,
            "wrist_audit_failure_reason": summary["wrist_audit_failure_reason"],
            "axis_pass_count": summary["wrist_audit_axis_pass_count"],
            "axis_probe_count": len(axis_summaries),
        }
    )
    _write_csv(output_dir / "wrist_audit_trace.csv", rows)
    write_json(output_dir / "wrist_audit_summary.json", summary)
    return summary


_V2_ISOLATION_OBJECT_NAMES = ("Table", "FixedAsset", "Screw1", "Plug2", "Backrest", "Rod", "Frame")


class _V2AllBodyContactTruth:
    def __init__(
        self,
        output_dir: Path,
        env_index: int,
        target_part: str,
        threshold_n: float,
        *,
        log_prefix: str = "wrist_translation",
    ):
        self.output_dir = Path(output_dir)
        self.env_index = int(env_index)
        self.target_part = str(target_part)
        self.threshold_n = float(threshold_n)
        self.log_prefix = str(log_prefix)
        self.rows: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._setup_done = False
        self._setup_error = ""

    def setup(self, base: Any) -> None:
        if self._setup_done:
            return
        self._setup_done = True
        try:
            import copy  # noqa: PLC0415
            import isaaclab.sim as sim_utils  # noqa: PLC0415
            from isaaclab.sensors import ContactSensor  # noqa: PLC0415
            from pxr import PhysxSchema  # noqa: PLC0415
        except Exception as exc:
            self._setup_error = f"import_error:{type(exc).__name__}:{exc}"
            return
        sensor_cfg_template = getattr(getattr(base, "cfg", None), "dex_fingertip_force_sensor", None)
        if sensor_cfg_template is None:
            self._setup_error = "dex_fingertip_force_sensor_cfg_missing"
            return
        body_names = list(getattr(getattr(base, "_robot", None), "body_names", []) or [])
        bodies = _v2_isolation_hand_bodies(body_names)
        errors: list[str] = []
        for body_name, group in bodies:
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
                cfg.filter_prim_paths_expr = [f"/World/envs/env_.*/{name}" for name in _V2_ISOLATION_OBJECT_NAMES]
                sensor = ContactSensor(cfg)
                status = "created"
                try:
                    sensor._initialize_impl()
                    sensor._is_initialized = True
                    status = "initialized"
                except Exception as exc:
                    status = f"init_error:{type(exc).__name__}:{exc}"
                self._records.append(
                    {
                        "body_name": body_name,
                        "hand_group": group,
                        "sensor": sensor,
                        "sensor_status": status,
                    }
                )
            except Exception as exc:
                errors.append(f"{body_name}:{type(exc).__name__}:{exc}")
        if errors:
            self._setup_error = "; ".join(errors[:10])

    def record(
        self,
        phase: str,
        step: int,
        state: dict[str, Any],
        base: Any,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.setup(base)
        extra = dict(extra or {})
        rows: list[dict[str, Any]] = []
        for record in self._records:
            rows.extend(self._sensor_rows(record, phase, step, state, extra))
        if not rows and self._setup_error:
            rows.append(
                {
                    "phase": phase,
                    "step": int(step),
                    "env_index": self.env_index,
                    "hand_group": "unknown_hand",
                    "hand_body_name": "",
                    "object_name": "unknown_scene_body",
                    "is_target_object": False,
                    "force_norm_or_contact_strength": 0.0,
                    "contact_source": "v2_all_body_contact_sensor",
                    "sensor_status": self._setup_error,
                    **extra,
                }
            )
        peak_row: dict[str, Any] = {}
        peak = 0.0
        target_peak = 0.0
        non_target_peak = 0.0
        non_target_peak_row: dict[str, Any] = {}
        active_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                force = float(row.get("force_norm_or_contact_strength", 0.0) or 0.0)
            except Exception:
                force = 0.0
            if force >= self.threshold_n:
                active_rows.append(row)
            if force > peak:
                peak = force
                peak_row = row
            if bool(row.get("is_target_object")):
                target_peak = max(target_peak, force)
            else:
                if force > non_target_peak:
                    non_target_peak = force
                    non_target_peak_row = row
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        summary_row = {
            "phase": phase,
            "step": int(step),
            "env_index": self.env_index,
            "hand_group": str(peak_row.get("hand_group", "all_body_peak") or "all_body_peak"),
            "hand_body_name": str(peak_row.get("hand_body_name", "")),
            "object_name": str(peak_row.get("object_name", "none") or "none"),
            "is_target_object": bool(peak_row.get("is_target_object", False)),
            "force_norm_or_contact_strength": peak,
            "contact_source": "v2_all_body_contact_peak_per_step",
            "sensor_status": self._setup_error,
            "finger3_target_filtered_force_n": f3,
            "finger4_target_filtered_force_n": f4,
            **extra,
        }
        self.rows.extend(_plain([*active_rows, summary_row]))
        return {
            "all_body_contact_peak_n": peak,
            "all_body_contact_active": bool(peak >= self.threshold_n),
            "all_body_contact_peak_body": str(peak_row.get("hand_body_name", "")),
            "all_body_contact_peak_group": str(peak_row.get("hand_group", "")),
            "all_body_contact_peak_object": str(peak_row.get("object_name", "")),
            "all_body_target_contact_peak_n": target_peak,
            "all_body_non_target_contact_peak_n": non_target_peak,
            "all_body_non_target_contact_peak_body": str(non_target_peak_row.get("hand_body_name", "")),
            "all_body_non_target_contact_peak_object": str(non_target_peak_row.get("object_name", "")),
            "all_body_non_target_contact_identified": bool(
                non_target_peak >= self.threshold_n
                and str(non_target_peak_row.get("object_name", "")) in _V2_ISOLATION_OBJECT_NAMES
                and str(non_target_peak_row.get("object_name", "")) != self.target_part
            ),
            "finger3_target_filtered_force_n": f3,
            "finger4_target_filtered_force_n": f4,
            "all_body_contact_setup_error": self._setup_error,
        }

    def _sensor_rows(
        self,
        record: dict[str, Any],
        phase: str,
        step: int,
        state: dict[str, Any],
        extra: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sensor = record.get("sensor")
        body = str(record.get("body_name", ""))
        group = str(record.get("hand_group", ""))
        status = str(record.get("sensor_status", ""))
        base_row = {
            "phase": phase,
            "step": int(step),
            "env_index": self.env_index,
            "hand_group": group,
            "hand_body_name": body,
            "finger3_target_filtered_force_n": float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
            "finger4_target_filtered_force_n": float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
            "contact_source": "v2_all_body_contact_sensor",
            "sensor_status": status,
            **extra,
        }
        rows: list[dict[str, Any]] = []
        try:
            if sensor is None or "error" in status:
                return [{**base_row, "object_name": "unknown_scene_body", "is_target_object": False, "force_norm_or_contact_strength": 0.0}]
            data = getattr(sensor, "data", None)
            matrix = getattr(data, "force_matrix_w", None)
            if torch is not None and torch.is_tensor(matrix) and matrix.ndim >= 4 and matrix.shape[1] > 0:
                cpu = matrix.detach().to("cpu")
                for obj_idx, obj_name in enumerate(_V2_ISOLATION_OBJECT_NAMES):
                    force = 0.0
                    if self.env_index < cpu.shape[0] and obj_idx < cpu.shape[2]:
                        force = float(torch.linalg.vector_norm(cpu[self.env_index, 0, obj_idx, :]).item())
                    rows.append(
                        {
                            **base_row,
                            "object_name": obj_name,
                            "is_target_object": obj_name == self.target_part,
                            "force_norm_or_contact_strength": force,
                        }
                    )
                return rows
            forces = getattr(data, "net_forces_w", None)
            force = 0.0
            if torch is not None and torch.is_tensor(forces) and forces.ndim >= 3 and self.env_index < forces.shape[0]:
                force = float(torch.linalg.vector_norm(forces.detach().to("cpu")[self.env_index, 0, :]).item())
            return [
                {
                    **base_row,
                    "object_name": "unknown_scene_body",
                    "is_target_object": False,
                    "force_norm_or_contact_strength": force,
                    "sensor_status": "force_matrix_unavailable",
                }
            ]
        except Exception as exc:
            return [
                {
                    **base_row,
                    "object_name": "unknown_scene_body",
                    "is_target_object": False,
                    "force_norm_or_contact_strength": 0.0,
                    "sensor_status": f"read_error:{type(exc).__name__}:{exc}",
                }
            ]

    def finalize(self) -> dict[str, Any]:
        log_path = self.output_dir / f"{self.log_prefix}_contact_log.csv"
        summary_path = self.output_dir / f"{self.log_prefix}_contact_summary.json"
        _write_csv(log_path, self.rows)
        first: dict[tuple[str, str], int] = {}
        duration: dict[tuple[str, str], int] = {}
        peaks: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        for row in self.rows:
            key = (str(row.get("hand_body_name", "")), str(row.get("object_name", "")))
            try:
                force = float(row.get("force_norm_or_contact_strength", 0.0) or 0.0)
            except Exception:
                force = 0.0
            if force >= self.threshold_n:
                step = int(row.get("step", -1) or -1)
                first[key] = min(first.get(key, step), step)
                duration[key] = duration.get(key, 0) + 1
            if force > peaks.get(key, (0.0, {}))[0]:
                peaks[key] = (force, row)
        peak_rows = [
            {
                "hand_body_name": key[0],
                "object_name": key[1],
                "peak_force_n": value[0],
                "first_step": first.get(key, ""),
                "duration_steps": duration.get(key, 0),
                "hand_group": value[1].get("hand_group", ""),
            }
            for key, value in peaks.items()
            if value[0] >= self.threshold_n
        ]
        peak_rows.sort(key=lambda row: float(row.get("peak_force_n", 0.0) or 0.0), reverse=True)
        summary = {
            "all_body_contact_log_csv": str(log_path),
            "all_body_contact_setup_error": self._setup_error,
            "all_body_contact_sensor_count": len(self._records),
            "all_body_contact_row_count": len(self.rows),
            "all_body_contact_peak_rows": peak_rows[:40],
            "all_body_contact_peak_n": max([0.0, *[float(row.get("peak_force_n", 0.0) or 0.0) for row in peak_rows]]),
            "all_body_contact_any_active": bool(peak_rows),
        }
        write_json(summary_path, summary)
        summary["all_body_contact_summary_json"] = str(summary_path)
        return summary


def _run_wrist_translation_isolation(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    rows: list[dict[str, Any]] = []
    write_json(
        output_dir / "wrist_translation_isolation_heartbeat.json",
        {
            "wrist_translation_isolation_started": True,
            "wrist_isolation_trials": int(cfg.wrist_isolation_trials),
            "wrist_isolation_waypoint_max_steps": int(cfg.wrist_isolation_waypoint_max_steps),
            "wrist_translation_isolation_pd_fix_env": os.environ.get("WUJI_SCREW1_V2_WRIST_PD_FIX", "0"),
        },
    )
    contact_truth = _V2AllBodyContactTruth(
        output_dir,
        env_index,
        cfg.part,
        max(float(cfg.wrist_isolation_contact_threshold_n), float(cfg.contact_threshold_n)),
    )
    contact_truth.setup(base)
    direct = _run_direct_wrist_x_joint_target_audit(env, base, env_index, cfg, alignment, contact_truth, episodes, trace_rows, rows)
    task = _run_task_space_wrist_x_audit(env, base, env_index, cfg, alignment, contact_truth, episodes, trace_rows, rows)
    seed_path = _run_wrist_sequence_probe(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        rows,
        audit_test="seed_standoff_path_audit",
        contact_truth=contact_truth,
        sequence_builder=lambda state: _wrist_staged_seed_standoff_sequence(
            base,
            env_index,
            cfg,
            state,
            seed_offset=cfg.seed_a_position_offset_m,
            seed_rpy=cfg.seed_a_orientation_rpy_deg,
            use_seed_orientation=False,
            sequence_mode="translation_only",
        ),
    )
    contact_summary = contact_truth.finalize()
    subclass, result_class, blocker = _classify_wrist_translation_isolation(direct, task, seed_path, contact_summary, cfg)
    summary = {
        "wrist_translation_isolation_executed": True,
        "wrist_translation_isolation_pd_fix_env": os.environ.get("WUJI_SCREW1_V2_WRIST_PD_FIX", "0"),
        "wrist_translation_isolation_subclass": subclass,
        "wrist_translation_isolation_result_class": result_class,
        "wrist_translation_isolation_blocker": blocker,
        "wrist_translation_isolation_trace_csv": str(output_dir / "wrist_translation_isolation_trace.csv"),
        "wrist_translation_isolation_summary_json": str(output_dir / "wrist_translation_isolation_summary.json"),
        "direct_wrist_x_joint_target_audit": direct,
        "task_space_wrist_x_audit": task,
        "seed_standoff_path_audit": seed_path,
        "direct_wrist_x_joint_target_passed": bool(direct.get("experiment_passed")),
        "task_space_wrist_x_passed": bool(task.get("experiment_passed")),
        "seed_standoff_path_passed": bool(seed_path.get("sequence_passed")),
        "result_class": result_class,
        "blocker": blocker,
        "final_allowed_result_class": result_class in RESULT_CLASSES,
        **contact_summary,
    }
    episodes.append(
        {
            "phase": "wrist_translation_isolation",
            "result_class": result_class,
            "subclass": subclass,
            "direct_passed": bool(direct.get("experiment_passed")),
            "task_space_passed": bool(task.get("experiment_passed")),
            "seed_path_passed": bool(seed_path.get("sequence_passed")),
        }
    )
    _write_csv(output_dir / "wrist_translation_isolation_trace.csv", rows)
    write_json(output_dir / "wrist_translation_isolation_summary.json", summary)
    return summary


def _run_direct_wrist_x_joint_target_audit(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    contact_truth: _V2AllBodyContactTruth,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_results: list[dict[str, Any]] = []
    for trial in range(int(cfg.wrist_isolation_trials)):
        fresh = _fresh_episode(env, base, env_index, cfg, alignment, trace_rows, f"wrist_isolation_direct_trial_{trial}", hand_mode="parked")
        episodes.append({**fresh["episode_row"], "phase": "wrist_translation_isolation_direct", "trial": trial})
        state = fresh["state"]
        corridor = _wrist_translation_corridor(base, env_index, cfg, state)
        _set_v2_workspace_override_for_corridor(base, env_index, corridor, cfg)
        trial_pass = True
        failure = ""
        last_result: dict[str, Any] = {}
        completed_waypoints = 0
        setup_joint = corridor.get("setup_joint")
        if bool(corridor.get("setup_required")) and isinstance(setup_joint, list):
            result = _run_direct_wrist_joint_waypoint(
                env,
                base,
                env_index,
                cfg,
                alignment,
                contact_truth,
                trace_rows,
                setup_joint,
                experiment="direct_wrist_x_joint_target_audit_corridor_setup",
                trial=trial,
                waypoint_index=-1,
                corridor=corridor,
            )
            rows.append(
                {
                    **result,
                    "phase": "wrist_translation_isolation",
                    "experiment": "direct_wrist_x_joint_target_audit_corridor_setup",
                }
            )
            last_result = result
            if not bool(result.get("waypoint_reached")):
                trial_pass = False
                failure = "corridor_setup_failed:" + str(result.get("termination_reason") or "direct_setup_unreached")
        if trial_pass:
            for waypoint_index, target_joint in enumerate(corridor["joint_waypoints"]):
                completed_waypoints = waypoint_index + 1
                result = _run_direct_wrist_joint_waypoint(
                    env,
                    base,
                    env_index,
                    cfg,
                    alignment,
                    contact_truth,
                    trace_rows,
                    target_joint,
                    experiment="direct_wrist_x_joint_target_audit",
                    trial=trial,
                    waypoint_index=waypoint_index,
                    corridor=corridor,
                )
                rows.append({**result, "phase": "wrist_translation_isolation", "experiment": "direct_wrist_x_joint_target_audit"})
                last_result = result
                if not bool(result.get("waypoint_reached")):
                    trial_pass = False
                    failure = str(result.get("termination_reason") or "direct_waypoint_unreached")
                    break
        _clear_v2_direct_wrist_override(base, env_index)
        _clear_v2_workspace_override(base, env_index)
        trial_results.append(
            {
                "trial": trial,
                "trial_passed": trial_pass,
                "failure_reason": failure,
                "completed_waypoints": completed_waypoints,
                "corridor": corridor,
                "last_result": last_result,
            }
        )
    return _summarize_isolation_experiment("direct_wrist_x_joint_target_audit", trial_results, cfg)


def _run_task_space_wrist_x_audit(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    contact_truth: _V2AllBodyContactTruth,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_results: list[dict[str, Any]] = []
    for trial in range(int(cfg.wrist_isolation_trials)):
        fresh = _fresh_episode(env, base, env_index, cfg, alignment, trace_rows, f"wrist_isolation_task_space_trial_{trial}", hand_mode="parked")
        episodes.append({**fresh["episode_row"], "phase": "wrist_translation_isolation_task_space", "trial": trial})
        state = fresh["state"]
        corridor = _wrist_translation_corridor(base, env_index, cfg, state)
        _set_v2_workspace_override_for_corridor(base, env_index, corridor, cfg)
        trial_pass = True
        failure = ""
        last_result: dict[str, Any] = {}
        completed_waypoints = 0
        setup_joint = corridor.get("setup_joint")
        if bool(corridor.get("setup_required")) and isinstance(setup_joint, list):
            setup = _run_direct_wrist_joint_waypoint(
                env,
                base,
                env_index,
                cfg,
                alignment,
                contact_truth,
                trace_rows,
                setup_joint,
                experiment="task_space_wrist_x_audit_corridor_setup",
                trial=trial,
                waypoint_index=-1,
                corridor=corridor,
            )
            rows.append(
                {
                    **setup,
                    "phase": "wrist_translation_isolation",
                    "experiment": "task_space_wrist_x_audit_corridor_setup",
                }
            )
            last_result = setup
            _clear_v2_direct_wrist_override(base, env_index)
            if not bool(setup.get("waypoint_reached")):
                trial_pass = False
                failure = "corridor_setup_failed:" + str(setup.get("termination_reason") or "task_setup_unreached")
        if trial_pass:
            for waypoint_index, target_joint in enumerate(corridor["joint_waypoints"]):
                completed_waypoints = waypoint_index + 1
                pose = _wrist_joint_values_to_dex_pose(base, target_joint)
                replay = _servo_to_pose_guarded(
                    env,
                    base,
                    env_index,
                    cfg,
                    alignment,
                    trace_rows,
                    phase="wrist_translation_isolation",
                    target_pos=list(pose.get("dex_pos", [])),
                    target_quat=list(pose.get("dex_quat", [])),
                    max_steps=max(1, min(int(cfg.seed_servo_max_steps), int(cfg.wrist_isolation_waypoint_max_steps))),
                    contact_truth=contact_truth,
                    extra={
                        "isolation_experiment": "task_space_wrist_x_audit",
                        "isolation_trial": trial,
                        "isolation_waypoint_index": waypoint_index,
                        "isolation_target_joint": _json(target_joint),
                        "isolation_corridor_total_x_m": corridor.get("total_x_m", ""),
                    },
                )
                snapshot = _wrist_pose_snapshot(
                    base,
                    env_index,
                    state=dict(replay.get("final_state") or {}),
                    desired_pos=list(pose.get("dex_pos", [])),
                    desired_quat=list(pose.get("dex_quat", [])),
                )
                result = {
                    "trial": trial,
                    "waypoint_index": waypoint_index,
                    "waypoint_reached": bool(replay.get("reached_pose")),
                    "termination_reason": replay.get("termination_reason", ""),
                    "tracking_timeout": bool(replay.get("tracking_timeout")),
                    "tracking_stall": bool(replay.get("tracking_stall")),
                    "all_body_contact_abort": bool(replay.get("all_body_contact_abort")),
                    "all_body_contact_peak_n": replay.get("all_body_contact_peak_n", 0.0),
                    "workspace_clamp_delta_m": replay.get("final_workspace_clamp_delta_m", 0.0),
                    "table_barrier_delta_z_m": replay.get("final_table_barrier_delta_z_m", 0.0),
                    **_wrist_row_fields(snapshot),
                }
                rows.append({**result, "phase": "wrist_translation_isolation", "experiment": "task_space_wrist_x_audit"})
                last_result = result
                if not bool(replay.get("reached_pose")):
                    trial_pass = False
                    failure = str(replay.get("termination_reason") or "task_space_waypoint_unreached")
                    break
        _clear_v2_workspace_override(base, env_index)
        _clear_v2_direct_wrist_override(base, env_index)
        trial_results.append(
            {
                "trial": trial,
                "trial_passed": trial_pass,
                "failure_reason": failure,
                "completed_waypoints": completed_waypoints,
                "corridor": corridor,
                "last_result": last_result,
            }
        )
    return _summarize_isolation_experiment("task_space_wrist_x_audit", trial_results, cfg)


def _run_direct_wrist_joint_waypoint(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    contact_truth: _V2AllBodyContactTruth,
    trace_rows: list[dict[str, Any]],
    target_joint: list[float],
    *,
    experiment: str,
    trial: int,
    waypoint_index: int,
    corridor: dict[str, Any],
) -> dict[str, Any]:
    target_joint = _flat_vec(target_joint, 6)
    _set_v2_direct_wrist_override(base, env_index, target_joint)
    termination = "max_steps_without_reach"
    reached = False
    tracking_stall = False
    contact_peak = 0.0
    max_fk_pos_error = 0.0
    max_fk_rot_error = 0.0
    min_x_error = float("inf")
    other_translation_error_peak = 0.0
    rotation_error_peak = 0.0
    final_axis_errors: dict[str, Any] = {}
    history: list[float] = []
    final_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    final_snapshot: dict[str, Any] = {}
    for step in range(max(1, min(int(cfg.seed_servo_max_steps), int(cfg.wrist_isolation_waypoint_max_steps)))):
        _configure_v2_control(base, env_index, mode="")
        final_state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase="wrist_translation_isolation",
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "isolation_experiment": experiment,
                "isolation_trial": trial,
                "isolation_waypoint_index": waypoint_index,
                "isolation_direct_joint_target": _json(target_joint),
                "isolation_corridor_total_x_m": corridor.get("total_x_m", ""),
            },
        )
        contact_info = contact_truth.record(
            "wrist_translation_isolation",
            step,
            final_state,
            base,
            extra={"isolation_experiment": experiment, "isolation_trial": trial, "isolation_waypoint_index": waypoint_index},
        )
        if trace_rows:
            trace_rows[-1].update(contact_info)
        contact_peak = max(contact_peak, float(contact_info.get("all_body_contact_peak_n", 0.0) or 0.0))
        final_snapshot = _wrist_pose_snapshot(base, env_index, state=final_state)
        actual_joint = _flat_vec(final_snapshot.get("wrist_joint_actual", []), 6)
        axis_errors = _wrist_axis_error_metrics(target_joint, actual_joint, cfg)
        final_axis_errors = axis_errors
        x_error = float(axis_errors.get("x_error_m", float("inf")) or float("inf"))
        min_x_error = min(min_x_error, x_error)
        other_translation_error_peak = max(
            other_translation_error_peak,
            float(axis_errors.get("other_translation_error_peak_m", 0.0) or 0.0),
        )
        rotation_error_peak = max(
            rotation_error_peak,
            float(axis_errors.get("rotation_error_peak_deg", 0.0) or 0.0),
        )
        progress_error = x_error
        if int(waypoint_index) < 0:
            signed = _flat_vec(axis_errors.get("wrist_axis_error_signed", []), 6)
            if len(signed) == 6:
                progress_error = max(abs(float(value)) for value in signed)
        history.append(progress_error)
        max_fk_pos_error = max(
            max_fk_pos_error,
            float(final_snapshot.get("actual_joint_fk_to_runtime_body_position_error_m", 0.0) or 0.0),
        )
        max_fk_rot_error = max(
            max_fk_rot_error,
            float(final_snapshot.get("actual_joint_fk_to_runtime_body_angle_error_rad", 0.0) or 0.0),
        )
        if bool(contact_info.get("all_body_contact_active")):
            termination = "all_body_contact_abort"
            break
        if (
            abs(float(final_state.get("workspace_clamp_delta_m", 0.0) or 0.0)) > 1.0e-6
            and not bool(corridor.get("workspace_override_applied"))
        ):
            termination = "workspace_clamp_abort"
            break
        if abs(float(final_state.get("table_barrier_delta_z_m", 0.0) or 0.0)) > 1.0e-6:
            termination = "table_barrier_abort"
            break
        if bool(axis_errors.get("direct_wrist_x_waypoint_reached")):
            reached = True
            termination = "waypoint_reached"
            break
        window = max(2, int(cfg.seed_servo_tracking_stall_window_steps))
        if len(history) > window:
            improvement = float(history[-window - 1]) - float(history[-1])
            if improvement < float(cfg.seed_servo_tracking_stall_min_improvement_m):
                tracking_stall = True
                termination = "direct_joint_tracking_stall"
                break
    return {
        "trial": int(trial),
        "waypoint_index": int(waypoint_index),
        "waypoint_reached": bool(reached),
        "termination_reason": termination,
        "steps": step + 1 if "step" in locals() else 0,
        "tracking_stall": bool(tracking_stall),
        "all_body_contact_peak_n": float(contact_peak),
        "all_body_contact_abort": bool(contact_peak >= float(cfg.wrist_isolation_contact_threshold_n)),
        "target_wrist_joint": _json(target_joint),
        "failing_axis": "" if reached else str(final_axis_errors.get("failing_axis", "wrist_x")),
        "x_error_m": final_axis_errors.get("x_error_m", ""),
        "target_x_error_m": final_axis_errors.get("target_x_error_m", ""),
        "hold_y_error_m": final_axis_errors.get("hold_y_error_m", ""),
        "hold_z_error_m": final_axis_errors.get("hold_z_error_m", ""),
        "other_translation_error_peak_m": other_translation_error_peak,
        "hold_roll_error_deg": final_axis_errors.get("hold_roll_error_deg", ""),
        "hold_pitch_error_deg": final_axis_errors.get("hold_pitch_error_deg", ""),
        "hold_yaw_error_deg": final_axis_errors.get("hold_yaw_error_deg", ""),
        "rotation_error_peak_deg": rotation_error_peak,
        "min_x_error_m": min_x_error if math.isfinite(min_x_error) else "",
        "direct_wrist_x_waypoint_reached_by_axis_tolerances": bool(reached),
        "actual_joint_fk_to_runtime_body_position_error_m": max_fk_pos_error,
        "actual_joint_fk_to_runtime_body_angle_error_deg": math.degrees(max_fk_rot_error),
        "workspace_clamp_delta_m": final_state.get("workspace_clamp_delta_m", 0.0),
        "table_barrier_delta_z_m": final_state.get("table_barrier_delta_z_m", 0.0),
        **_wrist_row_fields(final_snapshot),
    }


def _wrist_translation_corridor(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
) -> dict[str, Any]:
    wrist = _wrist_joint_snapshot(base, env_index)
    start = _flat_vec(wrist.get("wrist_joint_actual", wrist.get("wrist_joint_target", [])), 6)
    lower = _flat_vec(wrist.get("wrist_joint_lower_limits", []), 6)
    upper = _flat_vec(wrist.get("wrist_joint_upper_limits", []), 6)
    scene = _v2_scene_aabb_summary(env_index)
    table_top = cfg.canonical_support_pose.get("table_top_z_m", "")
    try:
        table_top_f = float(table_top)
    except Exception:
        table_top_f = float(_table_top_metadata(base, env_index).get("table_top_z_m") or 0.0)
    hand_z = float(state.get("palm_local_pos", [0.0, 0.0, 0.0])[2])
    scene_max_z = max([table_top_f, *[float(row.get("bbox_max_xyz", [0.0, 0.0, table_top_f])[2]) for row in scene]])
    setup = list(start)
    clearance = float(cfg.wrist_isolation_scene_clearance_m)
    if len(setup) >= 3:
        safe_z = max(float(setup[2]), float(scene_max_z) + clearance)
        if lower and upper:
            safe_z = max(float(lower[2]) + 0.010, min(float(upper[2]) - 0.010, safe_z))
        setup[2] = safe_z
    limit_margin = math.radians(float(cfg.wrist_isolation_rot_limit_margin_deg))
    if lower and upper:
        for axis in range(3, 6):
            value = float(setup[axis])
            lo = float(lower[axis])
            hi = float(upper[axis])
            if math.isfinite(lo) and value < lo + limit_margin:
                value = lo + limit_margin
            if math.isfinite(hi) and value > hi - limit_margin:
                value = hi - limit_margin
            setup[axis] = value
    setup_required = bool(
        _distance(start[:3], setup[:3]) > float(cfg.wrist_isolation_joint_pos_tol_m)
        or max(abs(float(start[i]) - float(setup[i])) for i in range(3, 6)) > math.radians(float(cfg.wrist_isolation_joint_rot_tol_deg))
    )
    requested_total = float(cfg.wrist_isolation_total_x_m)
    desired_end = setup[0] + requested_total
    clamped_end = max(float(lower[0]), min(float(upper[0]), desired_end)) if lower and upper else desired_end
    total = clamped_end - setup[0]
    step_size = max(float(cfg.wrist_isolation_waypoint_step_m), 1.0e-4)
    count = max(1, int(math.ceil(abs(total) / step_size)))
    waypoints = []
    for index in range(1, count + 1):
        alpha = index / count
        waypoint = list(setup)
        waypoint[0] = setup[0] + total * alpha
        waypoints.append(waypoint)
    dex_positions = []
    for joint in [setup, *waypoints]:
        pose = _wrist_joint_values_to_dex_pose(base, joint)
        dex = _flat_vec(pose.get("dex_pos", []), 3)
        if len(dex) == 3:
            dex_positions.append(dex)
    workspace_margin = float(cfg.wrist_isolation_workspace_margin_m)
    if dex_positions:
        lower_ws = [min(row[i] for row in dex_positions) - workspace_margin for i in range(3)]
        upper_ws = [max(row[i] for row in dex_positions) + workspace_margin for i in range(3)]
    else:
        lower_ws = []
        upper_ws = []
    return {
        "start_wrist_joint": start,
        "setup_joint": setup,
        "setup_required": setup_required,
        "setup_reason": "lift_above_scene_or_move_rotation_off_limit" if setup_required else "",
        "requested_total_x_m": requested_total,
        "total_x_m": total,
        "waypoint_count": len(waypoints),
        "waypoint_step_m": step_size,
        "joint_waypoints": waypoints,
        "scene_aabbs": scene,
        "table_top_z_m": table_top_f,
        "hand_start_z_m": hand_z,
        "scene_max_z_m": scene_max_z,
        "hand_to_scene_top_margin_m": hand_z - scene_max_z,
        "corridor_setup_palm_z_m": setup[2] if len(setup) >= 3 else "",
        "corridor_setup_to_scene_top_margin_m": (setup[2] - scene_max_z) if len(setup) >= 3 else "",
        "corridor_rot_limit_margin_deg": float(cfg.wrist_isolation_rot_limit_margin_deg),
        "workspace_override_active": bool(dex_positions),
        "workspace_override_lower_xyz": lower_ws,
        "workspace_override_upper_xyz": upper_ws,
    }


def _set_v2_workspace_override_for_corridor(
    base: Any,
    env_index: int,
    corridor: dict[str, Any],
    cfg: Screw1GraspBaselineV2Config,
) -> None:
    if torch is None or not bool(corridor.get("workspace_override_active")):
        return
    active = getattr(base, "v2_workspace_override_env_ids", None)
    lower_t = getattr(base, "v2_workspace_override_lower", None)
    upper_t = getattr(base, "v2_workspace_override_upper", None)
    if not (torch.is_tensor(active) and torch.is_tensor(lower_t) and torch.is_tensor(upper_t)):
        return
    lower = _flat_vec(corridor.get("workspace_override_lower_xyz", []), 3)
    upper = _flat_vec(corridor.get("workspace_override_upper_xyz", []), 3)
    if len(lower) != 3 or len(upper) != 3:
        return
    idx = int(env_index)
    active[idx] = True
    lower_t[idx, :] = torch.tensor(lower, dtype=torch.float32, device=lower_t.device)
    upper_t[idx, :] = torch.tensor(upper, dtype=torch.float32, device=upper_t.device)
    corridor["workspace_override_applied"] = True


def _clear_v2_workspace_override(base: Any, env_index: int) -> None:
    if torch is None:
        return
    active = getattr(base, "v2_workspace_override_env_ids", None)
    lower_t = getattr(base, "v2_workspace_override_lower", None)
    upper_t = getattr(base, "v2_workspace_override_upper", None)
    if torch.is_tensor(active):
        active[int(env_index)] = False
    if torch.is_tensor(lower_t):
        lower_t[int(env_index), :] = float("nan")
    if torch.is_tensor(upper_t):
        upper_t[int(env_index), :] = float("nan")


def _wrist_axis_error_metrics(
    target_joint: list[float],
    actual_joint: list[float],
    cfg: Screw1GraspBaselineV2Config,
) -> dict[str, Any]:
    target = _flat_vec(target_joint, 6)
    actual = _flat_vec(actual_joint, 6)
    errors = [float(target[i]) - float(actual[i]) for i in range(6)]
    abs_errors = [abs(value) for value in errors]
    pos_tol = float(cfg.wrist_isolation_joint_pos_tol_m)
    rot_tol_deg = float(cfg.wrist_isolation_joint_rot_tol_deg)
    rot_errors_deg = [math.degrees(abs_errors[i]) for i in range(3, 6)]
    checks = [
        ("wrist_x", abs_errors[0], pos_tol),
        ("wrist_y", abs_errors[1], pos_tol),
        ("wrist_z", abs_errors[2], pos_tol),
        ("wrist_roll", rot_errors_deg[0], rot_tol_deg),
        ("wrist_pitch", rot_errors_deg[1], rot_tol_deg),
        ("wrist_yaw", rot_errors_deg[2], rot_tol_deg),
    ]
    failing = [name for name, value, tol in checks if float(value) > float(tol)]
    return {
        "x_error_m": abs_errors[0],
        "target_x_error_m": abs_errors[0],
        "hold_y_error_m": abs_errors[1],
        "hold_z_error_m": abs_errors[2],
        "other_translation_error_peak_m": max(abs_errors[1], abs_errors[2]),
        "hold_roll_error_deg": rot_errors_deg[0],
        "hold_pitch_error_deg": rot_errors_deg[1],
        "hold_yaw_error_deg": rot_errors_deg[2],
        "rotation_error_peak_deg": max(rot_errors_deg),
        "failing_axis": ",".join(failing),
        "direct_wrist_x_waypoint_reached": not failing,
        "wrist_axis_error_signed": errors,
    }


def _summarize_isolation_experiment(
    name: str,
    trial_results: list[dict[str, Any]],
    cfg: Screw1GraspBaselineV2Config,
) -> dict[str, Any]:
    contact_peak = 0.0
    fk_pos_peak = 0.0
    fk_rot_peak = 0.0
    x_error_peak = 0.0
    other_translation_peak = 0.0
    rotation_peak = 0.0
    target_overwrite_peak = 0.0
    first_failure = ""
    first_failing_axis = ""
    completed = 0
    for trial in trial_results:
        last = dict(trial.get("last_result") or {})
        completed += int(trial.get("completed_waypoints", 0) or 0)
        contact_peak = max(contact_peak, float(last.get("all_body_contact_peak_n", 0.0) or 0.0))
        fk_pos_peak = max(fk_pos_peak, float(last.get("actual_joint_fk_to_runtime_body_position_error_m", 0.0) or 0.0))
        fk_rot_peak = max(fk_rot_peak, float(last.get("actual_joint_fk_to_runtime_body_angle_error_deg", 0.0) or 0.0))
        x_error_peak = max(x_error_peak, float(last.get("x_error_m", 0.0) or 0.0))
        other_translation_peak = max(
            other_translation_peak,
            float(last.get("other_translation_error_peak_m", 0.0) or 0.0),
        )
        rotation_peak = max(rotation_peak, float(last.get("rotation_error_peak_deg", 0.0) or 0.0))
        target_overwrite_peak = max(
            target_overwrite_peak,
            float(last.get("target_chain_application_to_articulation_error_peak", 0.0) or 0.0),
            float(last.get("target_chain_application_to_sim_error_peak", 0.0) or 0.0),
        )
        if not bool(trial.get("trial_passed")) and not first_failure:
            first_failure = str(trial.get("failure_reason") or "trial_failed")
            first_failing_axis = str(last.get("failing_axis", "") or "")
    passed = bool(trial_results and all(bool(row.get("trial_passed")) for row in trial_results))
    return {
        "experiment_name": name,
        "experiment_passed": passed,
        "trial_count": len(trial_results),
        "passed_trial_count": sum(1 for row in trial_results if bool(row.get("trial_passed"))),
        "completed_waypoints_total": completed,
        "first_failure_reason": first_failure,
        "all_body_contact_peak_n": contact_peak,
        "scene_contact_observed": bool(contact_peak >= float(cfg.wrist_isolation_contact_threshold_n)),
        "x_error_peak_m": x_error_peak,
        "other_translation_error_peak_m": other_translation_peak,
        "rotation_error_peak_deg": rotation_peak,
        "first_failing_axis": first_failing_axis,
        "target_chain_overwrite_error_peak": target_overwrite_peak,
        "target_overwrite_observed": bool(target_overwrite_peak > 1.0e-5),
        "actual_joint_fk_to_runtime_body_position_error_peak_m": fk_pos_peak,
        "actual_joint_fk_to_runtime_body_angle_error_peak_deg": fk_rot_peak,
        "trial_results": trial_results,
    }


def _classify_wrist_translation_isolation(
    direct: dict[str, Any],
    task: dict[str, Any],
    seed_path: dict[str, Any],
    contact_summary: dict[str, Any],
    cfg: Screw1GraspBaselineV2Config,
) -> tuple[str, str, str]:
    contact_peak = max(
        float(contact_summary.get("all_body_contact_peak_n", 0.0) or 0.0),
        float(direct.get("all_body_contact_peak_n", 0.0) or 0.0),
        float(task.get("all_body_contact_peak_n", 0.0) or 0.0),
        float(seed_path.get("all_body_contact_peak_n", 0.0) or 0.0),
    )
    direct_pass = bool(direct.get("experiment_passed"))
    task_pass = bool(task.get("experiment_passed"))
    seed_pass = bool(seed_path.get("sequence_passed"))
    overwrite_peak = max(
        float(direct.get("target_chain_overwrite_error_peak", 0.0) or 0.0),
        float(task.get("target_chain_overwrite_error_peak", 0.0) or 0.0),
    )
    if contact_peak >= float(cfg.wrist_isolation_contact_threshold_n):
        return (
            "SCENE_COLLISION_OR_CONSTRAINT",
            "SCENE_COLLISION_OR_CONSTRAINT",
            "raw_all_body_contact_seen_during_wrist_translation_or_seed_path",
        )
    if direct_pass and task_pass and seed_pass:
        return ("WRIST_TRANSLATION_VALIDATED", "WRIST_TRANSLATION_VALIDATED", "")
    if overwrite_peak > 1.0e-5:
        return (
            "TARGET_BUFFER_OVERWRITE",
            "TARGET_BUFFER_OVERWRITE",
            f"target_chain_overwrite_error_peak={overwrite_peak:.6g}",
        )
    if (
        not direct_pass
        and float(direct.get("x_error_peak_m", 1.0e9) or 1.0e9) <= float(cfg.wrist_isolation_joint_pos_tol_m)
        and "wrist_x" not in str(direct.get("first_failing_axis", ""))
    ):
        return (
            "TEST_METRIC_ERROR",
            "TEST_METRIC_ERROR",
            str(direct.get("first_failing_axis") or "non_x_axis_failed_after_x_reached"),
        )
    if direct_pass and not task_pass:
        return (
            "TASK_SPACE_WRIST_CONTROLLER_ISSUE",
            "TASK_SPACE_WRIST_CONTROLLER_ISSUE",
            str(task.get("first_failure_reason") or "direct_joint_passed_task_space_failed_without_contact"),
        )
    if direct_pass and task_pass and not seed_pass:
        return (
            "SEED_PATH_GEOMETRY_ISSUE",
            "SEED_PATH_GEOMETRY_ISSUE",
            str(seed_path.get("termination_reason") or "corridor_passed_seed_path_failed"),
        )
    fk_peak = max(
        float(direct.get("actual_joint_fk_to_runtime_body_position_error_peak_m", 0.0) or 0.0),
        float(task.get("actual_joint_fk_to_runtime_body_position_error_peak_m", 0.0) or 0.0),
    )
    if fk_peak > float(cfg.wrist_isolation_fk_body_pos_tol_m):
        return (
            "WRIST_FK_BODY_MAPPING_ISSUE",
            "WRIST_FK_BODY_MAPPING_ISSUE",
            f"fk_runtime_body_position_error_peak_m={fk_peak:.6f}",
        )
    if not direct_pass and not task_pass:
        return (
            "ARTICULATION_HIERARCHY_OR_DRIVE_FAILURE",
            "ARTICULATION_HIERARCHY_OR_DRIVE_FAILURE",
            str(
                direct.get("first_failure_reason")
                or task.get("first_failure_reason")
                or "direct_and_task_space_failed_without_contact_needs_standalone_proof"
            ),
        )
    return (
        "WRIST_CONTROL_BLOCKER_PROVEN",
        "WRIST_CONTROL_BLOCKER_PROVEN",
        "mixed_wrist_translation_isolation_failure_without_single_subclass",
    )


def _v2_isolation_hand_bodies(body_names: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in body_names:
        group = ""
        if name == "right_palm_link" or "palm" in name:
            group = "palm"
        elif "wrist" in name or "panda" in name:
            group = "wrist"
        elif name.startswith("right_finger") and "_tip" in name:
            finger = name.split("_tip", 1)[0].replace("right_", "")
            group = f"{finger}_tip"
        elif name.startswith("right_finger"):
            parts = name.split("_")
            finger = parts[1] if len(parts) > 1 else "finger_unknown"
            group = f"{finger}_proximal"
        if group:
            out.append((name, group))
    return list(dict.fromkeys(out))


def _set_v2_direct_wrist_override(base: Any, env_index: int, target_joint: list[float]) -> None:
    if torch is None:
        return
    active = getattr(base, "v2_direct_wrist_joint_target_override_active", None)
    targets = getattr(base, "v2_direct_wrist_joint_target_override", None)
    if not torch.is_tensor(active) or not torch.is_tensor(targets):
        return
    active[int(env_index)] = True
    targets[int(env_index), :] = torch.tensor(_flat_vec(target_joint, 6), dtype=torch.float32, device=targets.device)


def _clear_v2_direct_wrist_override(base: Any, env_index: int) -> None:
    active = getattr(base, "v2_direct_wrist_joint_target_override_active", None)
    targets = getattr(base, "v2_direct_wrist_joint_target_override", None)
    if torch.is_tensor(active):
        active[int(env_index)] = False
    if torch.is_tensor(targets):
        targets[int(env_index), :] = float("nan")


def _v2_scene_aabb_summary(env_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in _V2_ISOLATION_OBJECT_NAMES:
        row = _v2_usd_aabb_env_local(env_index, name)
        if row:
            rows.append(row)
    return rows


def _v2_usd_aabb_env_local(env_index: int, prim_name: str) -> dict[str, Any]:
    path = f"/World/envs/env_{int(env_index)}/{prim_name}"
    try:
        import omni.usd  # noqa: PLC0415
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return {"prim_name": prim_name, "prim_path": path, "bbox_error": "stage_unavailable"}
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            return {"prim_name": prim_name, "prim_path": path, "bbox_error": "prim_missing"}
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = box.GetMin()
        mx = box.GetMax()
        origin = [0.0, 0.0, 0.0]
        try:
            env_path = f"/World/envs/env_{int(env_index)}"
            env_prim = stage.GetPrimAtPath(env_path)
            if env_prim and env_prim.IsValid():
                env_xf = UsdGeom.Xformable(env_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                origin = [float(env_xf.ExtractTranslation()[i]) for i in range(3)]
        except Exception:
            origin = [0.0, 0.0, 0.0]
        world_min = [float(mn[i]) for i in range(3)]
        world_max = [float(mx[i]) for i in range(3)]
        return {
            "prim_name": prim_name,
            "prim_path": path,
            "bbox_min_xyz": [world_min[i] - origin[i] for i in range(3)],
            "bbox_max_xyz": [world_max[i] - origin[i] for i in range(3)],
            "bbox_world_min_xyz": world_min,
            "bbox_world_max_xyz": world_max,
            "bbox_error": "",
        }
    except Exception as exc:
        return {"prim_name": prim_name, "prim_path": path, "bbox_error": f"{type(exc).__name__}:{exc}"}


def _run_wrist_reset_consistency(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fresh = _fresh_episode(env, base, env_index, cfg, alignment, trace_rows, "wrist_audit_reset", hand_mode="parked")
    state = fresh["state"]
    max_pos = 0.0
    max_angle = 0.0
    max_force = 0.0
    final_snapshot = _wrist_pose_snapshot(base, env_index, state=state)
    for step in range(int(cfg.wrist_audit_reset_hold_steps)):
        _configure_v2_control(base, env_index, mode="anchored_delta")
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase="wrist_audit_reset_consistency",
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
        )
        final_snapshot = _wrist_pose_snapshot(base, env_index, state=state)
        max_pos = max(max_pos, float(final_snapshot.get("control_to_actual_position_error_m", 0.0) or 0.0))
        max_angle = max(max_angle, float(final_snapshot.get("control_to_actual_angle_error_rad", 0.0) or 0.0))
        max_force = max(max_force, float(state.get("active_target_filtered_force_peak_n", 0.0) or 0.0))
        rows.append(
            {
                "phase": "wrist_audit",
                "audit_test": "reset_consistency",
                "audit_step": step,
                "target_contact": _target_contact_acquired(state, cfg),
                **_wrist_row_fields(final_snapshot),
            }
        )
    final_pos = float(final_snapshot.get("control_to_actual_position_error_m", 0.0) or 0.0)
    final_angle = float(final_snapshot.get("control_to_actual_angle_error_rad", 0.0) or 0.0)
    passed = bool(
        max_pos <= float(cfg.wrist_audit_reset_pos_tol_m)
        and max_angle <= math.radians(float(cfg.wrist_audit_reset_rot_tol_deg))
        and max_force < float(cfg.contact_threshold_n)
    )
    return {
        "consistency_passed": passed,
        "final_control_to_actual_position_error_m": final_pos,
        "max_control_to_actual_position_error_m": max_pos,
        "final_control_to_actual_angle_error_deg": math.degrees(final_angle),
        "max_control_to_actual_angle_error_deg": math.degrees(max_angle),
        "max_target_force_n": max_force,
    }


def _wrist_axis_probe_specs(
    cfg: Screw1GraspBaselineV2Config,
) -> list[tuple[str, list[float], list[float], int, float]]:
    pos = float(cfg.wrist_audit_pos_delta_m)
    rot = math.radians(float(cfg.wrist_audit_rot_delta_deg))
    return [
        ("x_pos", [pos, 0.0, 0.0], [0.0, 0.0, 0.0], 0, pos),
        ("x_neg", [-pos, 0.0, 0.0], [0.0, 0.0, 0.0], 0, -pos),
        ("y_pos", [0.0, pos, 0.0], [0.0, 0.0, 0.0], 1, pos),
        ("y_neg", [0.0, -pos, 0.0], [0.0, 0.0, 0.0], 1, -pos),
        ("z_pos", [0.0, 0.0, pos], [0.0, 0.0, 0.0], 2, pos),
        ("z_neg", [0.0, 0.0, -pos], [0.0, 0.0, 0.0], 2, -pos),
        ("roll_pos", [0.0, 0.0, 0.0], [rot, 0.0, 0.0], 3, rot),
        ("roll_neg", [0.0, 0.0, 0.0], [-rot, 0.0, 0.0], 3, -rot),
        ("pitch_pos", [0.0, 0.0, 0.0], [0.0, rot, 0.0], 4, rot),
        ("pitch_neg", [0.0, 0.0, 0.0], [0.0, -rot, 0.0], 4, -rot),
        ("yaw_pos", [0.0, 0.0, 0.0], [0.0, 0.0, rot], 5, rot),
        ("yaw_neg", [0.0, 0.0, 0.0], [0.0, 0.0, -rot], 5, -rot),
    ]


def _run_wrist_axis_probe(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    axis_name: str,
    delta_xyz: list[float],
    delta_rot: list[float],
    joint_axis: int,
    command_value: float,
) -> dict[str, Any]:
    fresh = _fresh_episode(env, base, env_index, cfg, alignment, trace_rows, f"wrist_audit_axis_{axis_name}", hand_mode="parked")
    before_state = fresh["state"]
    before = _wrist_pose_snapshot(base, env_index, state=before_state)
    action = _zero_action(env, base)
    _set_wrist_delta_action(base, action, env_index, delta_xyz, delta_rot)
    _configure_v2_control(base, env_index, mode="anchored_delta")
    state = _step_direct_action(
        env,
        base,
        env_index,
        cfg,
        action,
        phase="wrist_audit_axis",
        step=0,
        trace_rows=trace_rows,
        alignment=alignment,
        extra={"wrist_audit_axis": axis_name},
    )
    for step in range(1, int(cfg.wrist_audit_axis_settle_steps) + 1):
        _configure_v2_control(base, env_index, mode="anchored_delta")
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase="wrist_audit_axis",
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={"wrist_audit_axis": axis_name},
        )
    after = _wrist_pose_snapshot(base, env_index, state=state)
    before_joints = list(before.get("wrist_joint_actual", []))
    after_joints = list(after.get("wrist_joint_actual", []))
    before_targets = list(before.get("wrist_joint_target", []))
    after_targets = list(after.get("wrist_joint_target", []))
    actual_joint_delta = _list_delta(after_joints, before_joints, joint_axis)
    target_joint_delta = _list_delta(after_targets, before_targets, joint_axis)
    other_peak = max(
        [
            0.0,
            *[
                abs(_list_delta(after_joints, before_joints, idx))
                for idx in range(len(after_joints))
                if idx != int(joint_axis)
            ],
        ]
    )
    response_min = max(abs(float(command_value)), abs(float(target_joint_delta))) * float(
        cfg.wrist_audit_axis_response_fraction
    )
    target_response_ok = bool(abs(target_joint_delta) >= abs(float(command_value)) * 0.25)
    actual_follows_target = bool(
        target_joint_delta * actual_joint_delta > 0.0
        and abs(actual_joint_delta) >= max(abs(target_joint_delta) * 0.50, 1.0e-5)
    )
    lower_margin = _list_get(after.get("wrist_joint_lower_margin", []), joint_axis, "")
    upper_margin = _list_get(after.get("wrist_joint_upper_margin", []), joint_axis, "")
    saturation_flags = list(after.get("wrist_joint_saturation_flags", []) or [])
    saturated = bool(bool(_list_get(saturation_flags, joint_axis, False)) or abs(float(target_joint_delta)) < 1.0e-5)
    command_points_out_of_limit = False
    try:
        if float(command_value) < 0.0 and float(lower_margin) <= 1.0e-4:
            command_points_out_of_limit = True
        if float(command_value) > 0.0 and float(upper_margin) <= 1.0e-4:
            command_points_out_of_limit = True
    except Exception:
        command_points_out_of_limit = False
    explained_limit_saturation = bool(saturated and command_points_out_of_limit)
    magnitude_ok = bool(actual_follows_target or explained_limit_saturation)
    force_peak = float(state.get("active_target_filtered_force_peak_n", 0.0) or 0.0)
    axis_ok = bool((target_response_ok and actual_follows_target) or explained_limit_saturation)
    row = {
        "phase": "wrist_audit",
        "audit_test": "six_axis_authority",
        "axis_name": axis_name,
        "commanded_delta_xyz": list(delta_xyz),
        "commanded_delta_rot_axis_angle": list(delta_rot),
        "commanded_joint_axis": int(joint_axis),
        "commanded_joint_delta": float(command_value),
        "target_joint_delta": target_joint_delta,
        "actual_joint_delta": actual_joint_delta,
        "actual_joint_response_fraction": abs(actual_joint_delta) / max(abs(float(command_value)), 1.0e-9),
        "target_response_ok": target_response_ok,
        "actual_follows_control_target": actual_follows_target,
        "explained_joint_limit_saturation": explained_limit_saturation,
        "command_points_out_of_limit": command_points_out_of_limit,
        "axis_saturation_detected": saturated,
        "other_wrist_joint_delta_peak": other_peak,
        "actual_dex_position_delta_xyz": _sub_vec(after.get("actual_dex_pos", []), before.get("actual_dex_pos", [])),
        "actual_dex_angle_delta_deg": math.degrees(
            _quat_angle_delta_wxyz(before.get("actual_dex_quat", []), after.get("actual_dex_quat", []))
        ),
        "target_contact": _target_contact_acquired(state, cfg),
        "force_peak_n": force_peak,
        "axis_passed": bool(axis_ok and magnitude_ok and force_peak < float(cfg.contact_threshold_n)),
        "axis_failure_reason": ""
        if bool(axis_ok and magnitude_ok and force_peak < float(cfg.contact_threshold_n))
        else "target_not_commanded_or_actual_not_following_or_contact",
        **_wrist_row_fields(after),
    }
    rows.append(row)
    return row


def _run_wrist_sequence_probe(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    audit_test: str,
    sequence_builder: Any,
    contact_truth: Any | None = None,
) -> dict[str, Any]:
    fresh = _fresh_episode(env, base, env_index, cfg, alignment, trace_rows, f"wrist_audit_{audit_test}", hand_mode="parked")
    state = fresh["state"]
    sequence = list(sequence_builder(state))
    result: dict[str, Any] = {}
    stage_results: list[dict[str, Any]] = []
    max_force = 0.0
    for stage_index, item in enumerate(sequence):
        stage_name, target_pos, target_quat, target_meta = item
        target_pos = [float(value) for value in list(target_pos)[:3]]
        target_quat = _quat_normalize_wxyz(list(target_quat))
        target_meta = dict(target_meta or {})
        replay = _servo_to_pose_guarded(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=f"wrist_audit_{audit_test}",
            target_pos=target_pos,
            target_quat=target_quat,
            max_steps=max(
                1,
                min(
                    int(cfg.seed_servo_max_steps),
                    int(cfg.wrist_isolation_waypoint_max_steps) if contact_truth is not None else 300,
                ),
            ),
            contact_truth=contact_truth,
            extra={
                "wrist_audit_test": audit_test,
                "wrist_audit_stage": stage_name,
                "wrist_audit_stage_index": stage_index,
                **target_meta,
            },
        )
        state = dict(replay.get("final_state") or state)
        snapshot = _wrist_pose_snapshot(base, env_index, state=state, desired_pos=target_pos, desired_quat=target_quat)
        max_force = max(
            max_force,
            float(replay.get("finger3_force_peak_n", 0.0) or 0.0),
            float(replay.get("finger4_force_peak_n", 0.0) or 0.0),
        )
        stage_row = {
            "phase": "wrist_audit",
            "audit_test": audit_test,
            "audit_stage": stage_name,
            "audit_stage_index": stage_index,
            "sequence_order": "->".join(str(row[0]) for row in sequence),
            "servo_steps": replay.get("steps", 0),
            "termination_reason": replay.get("termination_reason", ""),
            "reached_pose": bool(replay.get("reached_pose")),
            "tracking_timeout": bool(replay.get("tracking_timeout")),
            "hard_abort": bool(replay.get("hard_abort")),
            "tracking_stall": bool(replay.get("tracking_stall")),
            "all_body_contact_abort": bool(replay.get("all_body_contact_abort")),
            "all_body_contact_peak_n": replay.get("all_body_contact_peak_n", 0.0),
            "contact_ever_acquired": bool(replay.get("contact_ever_acquired")),
            "final_current_contact": bool(replay.get("final_current_contact")),
            "finger3_force_peak_n": replay.get("finger3_force_peak_n", 0.0),
            "finger4_force_peak_n": replay.get("finger4_force_peak_n", 0.0),
            **target_meta,
            **_wrist_row_fields(snapshot),
        }
        rows.append(stage_row)
        stage_results.append(stage_row)
        if (
            bool(replay.get("hard_abort"))
            or bool(replay.get("tracking_timeout"))
            or bool(replay.get("tracking_stall"))
            or bool(replay.get("all_body_contact_abort"))
            or not bool(replay.get("reached_pose"))
        ):
            break
    final = stage_results[-1] if stage_results else {}
    final_pos = float(final.get("target_to_actual_position_error_m", 1.0e9) or 1.0e9)
    final_angle_deg = float(final.get("target_to_actual_angle_error_deg", 1.0e9) or 1.0e9)
    passed = bool(
        stage_results
        and all(bool(row.get("reached_pose")) for row in stage_results)
        and final_pos <= float(cfg.wrist_audit_combined_pos_tol_m)
        and final_angle_deg <= float(cfg.wrist_audit_combined_rot_tol_deg)
        and max_force < float(cfg.contact_threshold_n)
        and not any(bool(row.get("tracking_timeout")) for row in stage_results)
    )
    result.update(
        {
            "audit_test": audit_test,
            "sequence_order": final.get("sequence_order", ""),
            "sequence_stage_count": len(stage_results),
            "sequence_passed": passed,
            "first_failed_stage": "" if passed else str(final.get("audit_stage", "")),
            "strict_serial_stop": bool(stage_results and not passed),
            "final_target_to_actual_position_error_m": final_pos if math.isfinite(final_pos) else "",
            "final_target_to_actual_angle_error_deg": final_angle_deg if math.isfinite(final_angle_deg) else "",
            "max_target_force_n": max_force,
            "termination_reason": final.get("termination_reason", "sequence_not_executed"),
            "all_body_contact_peak_n": max([0.0, *[float(row.get("all_body_contact_peak_n", 0.0) or 0.0) for row in stage_results]]),
        }
    )
    return result


def _wrist_safe_seed_standoff_target(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    *,
    use_seed_orientation: bool,
) -> tuple[list[float], list[float], dict[str, Any]]:
    return _wrist_seed_standoff_target(
        base,
        env_index,
        cfg,
        state,
        cfg.seed_a_position_offset_m,
        cfg.seed_a_orientation_rpy_deg,
        use_seed_orientation=use_seed_orientation,
    )


def _wrist_seed_standoff_target(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    seed_offset: tuple[float, float, float],
    seed_rpy: tuple[float, float, float],
    *,
    use_seed_orientation: bool,
) -> tuple[list[float], list[float], dict[str, Any]]:
    raw_target, seed_quat = _seed_target_pose(cfg, state, tuple(seed_offset), tuple(seed_rpy))
    requested_standoff_z = float(raw_target[2]) + float(cfg.seed_servo_standoff_lift_m)
    standoff_cap_info: dict[str, Any] = {
        "seed_standoff_requested_z_m": requested_standoff_z,
        "seed_standoff_workspace_ceiling_z_m": "",
        "seed_standoff_workspace_cap_margin_m": "",
        "seed_standoff_workspace_cap_applied": False,
    }
    try:
        table_top = cfg.canonical_support_pose.get("table_top_z_m", "")
        table_top = float(table_top)
        if not math.isfinite(table_top):
            table_top = float(_v2_table_override_audit(base, env_index).get("v2_table_top_true_z_m", ""))
        max_clearance = float(getattr(base, "floating_workspace_max_clearance", float("nan")))
        if math.isfinite(table_top) and math.isfinite(max_clearance):
            cap_margin = max(0.004, 2.0 * float(cfg.seed_servo_command_tol_m))
            workspace_ceiling = table_top + max_clearance
            capped_z = min(requested_standoff_z, workspace_ceiling - cap_margin)
            standoff_cap_info.update(
                {
                    "seed_standoff_workspace_ceiling_z_m": workspace_ceiling,
                    "seed_standoff_workspace_cap_margin_m": cap_margin,
                    "seed_standoff_workspace_cap_applied": bool(capped_z < requested_standoff_z),
                }
            )
            requested_standoff_z = capped_z
    except Exception as exc:
        standoff_cap_info["seed_standoff_workspace_cap_error"] = f"{type(exc).__name__}:{exc}"
    raw_target = [
        float(raw_target[0]),
        float(raw_target[1]),
        requested_standoff_z,
    ]
    target_quat = seed_quat if use_seed_orientation else _current_wrist_quat_from_state(state, cfg)
    target, safety = _barrier_safe_seed_target(base, env_index, cfg, raw_target, target_quat)
    safety.update(
        {
            **standoff_cap_info,
            "wrist_audit_safe_target_mode": "seed_a_standoff",
            "wrist_audit_safe_standoff_lift_m": float(cfg.seed_servo_standoff_lift_m),
            "wrist_audit_use_seed_orientation": bool(use_seed_orientation),
            "wrist_audit_standoff_orientation_source": "seed" if use_seed_orientation else "current_reset",
        }
    )
    return target, target_quat, safety


def _current_wrist_quat_from_state(state: dict[str, Any], cfg: Screw1GraspBaselineV2Config) -> list[float]:
    for key in ("ctrl_target_palm_quat_wxyz", "palm_quat_wxyz"):
        quat = list(state.get(key, []) or [])
        if len(quat) >= 4 and _norm(quat[:4]) > 1.0e-6:
            return _quat_normalize_wxyz(quat[:4])
    return _quat_normalize_wxyz(cfg.canonical_hand_park_quat_wxyz)


def _wrist_staged_seed_standoff_sequence(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    *,
    seed_offset: tuple[float, float, float],
    seed_rpy: tuple[float, float, float],
    use_seed_orientation: bool,
    sequence_mode: str,
) -> list[tuple[str, list[float], list[float], dict[str, Any]]]:
    target_pos, target_quat, safety = _wrist_seed_standoff_target(
        base,
        env_index,
        cfg,
        state,
        tuple(seed_offset),
        tuple(seed_rpy),
        use_seed_orientation=use_seed_orientation,
    )
    safety = {
        **safety,
        "wrist_audit_sequence_mode": sequence_mode,
        "wrist_audit_staged_translation": True,
        "wrist_audit_waypoint_spacing_m": float(cfg.seed_servo_standoff_waypoint_spacing_m),
    }
    return _wrist_staged_translation_sequence(
        cfg,
        state,
        target_pos,
        target_quat,
        safety,
        stage_prefix="translation",
    )


def _wrist_staged_translation_sequence(
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    target_pos: list[float],
    target_quat: list[float],
    meta: dict[str, Any],
    *,
    stage_prefix: str,
) -> list[tuple[str, list[float], list[float], dict[str, Any]]]:
    start = [float(value) for value in list(state.get("palm_local_pos", []))[:3]]
    if len(start) < 3:
        start = [float(value) for value in list(target_pos)[:3]]
    target = [float(value) for value in list(target_pos)[:3]]
    quat = _quat_normalize_wxyz(target_quat)
    spacing = max(float(cfg.seed_servo_standoff_waypoint_spacing_m), float(cfg.seed_servo_pos_step_m))
    sequence: list[tuple[str, list[float], list[float], dict[str, Any]]] = []
    cursor = list(start)
    safe_z = max(float(cursor[2]), float(target[2]))

    def append_axis_segments(axis: int, value: float, label: str) -> None:
        nonlocal cursor
        delta = float(value) - float(cursor[axis])
        count = max(1, int(math.ceil(abs(delta) / spacing))) if abs(delta) > 1.0e-6 else 0
        for index in range(count):
            alpha = float(index + 1) / float(count)
            pos = list(cursor)
            pos[axis] = float(cursor[axis]) + delta * alpha
            stage_meta = {
                **meta,
                "wrist_audit_axis_segment": label,
                "wrist_audit_axis_segment_index": index,
                "wrist_audit_axis_segment_count": count,
            }
            sequence.append((f"{stage_prefix}_{label}_{index:02d}", pos, quat, stage_meta))
        if count > 0:
            cursor[axis] = float(value)

    append_axis_segments(2, safe_z, "lift_z")
    append_axis_segments(0, target[0], "x")
    append_axis_segments(1, target[1], "y")
    append_axis_segments(2, target[2], "final_z")
    if not sequence:
        sequence.append((f"{stage_prefix}_already_at_target", target, quat, dict(meta)))
    return sequence


def _seed_replay_corridor_prelude_sequence(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
) -> tuple[list[tuple[str, list[float], list[float], dict[str, Any]]], dict[str, Any]]:
    wrist = _wrist_joint_snapshot(base, env_index)
    start = _flat_vec(wrist.get("wrist_joint_actual", wrist.get("wrist_joint_target", [])), 6)
    lower = _flat_vec(wrist.get("wrist_joint_lower_limits", []), 6)
    upper = _flat_vec(wrist.get("wrist_joint_upper_limits", []), 6)
    if len(start) != 6 or len(lower) != 6 or len(upper) != 6:
        return [], dict(state)
    setup_joint = list(start)
    limit_margin = math.radians(float(cfg.wrist_isolation_rot_limit_margin_deg))
    for axis in range(3, 6):
        value = float(setup_joint[axis])
        lo = float(lower[axis])
        hi = float(upper[axis])
        if math.isfinite(lo) and value < lo + limit_margin:
            value = lo + limit_margin
        if math.isfinite(hi) and value > hi - limit_margin:
            value = hi - limit_margin
        setup_joint[axis] = value
    setup_required = bool(
        max(abs(float(setup_joint[i]) - float(start[i])) for i in range(3, 6))
        > math.radians(float(cfg.wrist_isolation_joint_rot_tol_deg))
    )
    if not setup_required:
        return [], dict(state)
    pose = _wrist_joint_values_to_dex_pose(base, setup_joint)
    setup_pos = _flat_vec(pose.get("dex_pos", []), 3)
    setup_quat = _quat_normalize_wxyz(pose.get("dex_quat", []))
    if len(setup_pos) != 3 or _norm(setup_quat) <= 1.0e-6:
        return [], dict(state)
    meta = {
        "seed_replay_stage_kind": "corridor_prelude",
        "seed_replay_corridor_prelude": True,
        "seed_replay_corridor_setup_reason": "move_rotation_off_limit_only",
        "seed_replay_corridor_setup_joint": list(setup_joint),
        "seed_replay_corridor_scene_max_z_m": "",
        "seed_replay_corridor_setup_to_scene_top_margin_m": "",
        "seed_replay_corridor_rot_limit_margin_deg": float(cfg.wrist_isolation_rot_limit_margin_deg),
        "seed_replay_corridor_start_joint": list(start),
    }
    staged_state = dict(state)
    staged_state["palm_local_pos"] = list(setup_pos)
    staged_state["palm_quat_wxyz"] = list(setup_quat)
    staged_state["ctrl_target_palm_quat_wxyz"] = list(setup_quat)
    staged_state["ctrl_target_fingertip_midpoint_quat_wxyz"] = list(setup_quat)
    return [("seed_safe_corridor_prelude", setup_pos, setup_quat, meta)], staged_state


def _seed_replay_verified_corridor_sequence(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    *,
    seed_name: str,
    final_target: list[float],
    final_target_quat: list[float],
    final_target_safety: dict[str, Any],
) -> tuple[list[tuple[str, list[float], list[float], dict[str, Any]]], dict[str, Any]]:
    corridor = _wrist_translation_corridor(base, env_index, cfg, state)
    spacing = max(float(cfg.seed_servo_standoff_waypoint_spacing_m), float(cfg.seed_servo_pos_step_m))
    stages: list[tuple[str, list[float], list[float], dict[str, Any]]] = []
    planned_positions: list[list[float]] = []
    final_target = [float(value) for value in list(final_target)[:3]]
    final_target_quat = _quat_normalize_wxyz(final_target_quat)

    def add_stage(name: str, pos: list[float], quat: list[float], meta: dict[str, Any]) -> None:
        clean_pos = [float(value) for value in list(pos)[:3]]
        clean_quat = _quat_normalize_wxyz(quat)
        planned_positions.append(clean_pos)
        stages.append(
            (
                name,
                clean_pos,
                clean_quat,
                {
                    "seed_name": seed_name,
                    "seed_replay_verified_corridor_transit": True,
                    "seed_replay_corridor_setup_reason": corridor.get("setup_reason", ""),
                    "seed_replay_corridor_scene_max_z_m": corridor.get("scene_max_z_m", ""),
                    "seed_replay_corridor_setup_to_scene_top_margin_m": corridor.get(
                        "corridor_setup_to_scene_top_margin_m", ""
                    ),
                    "seed_replay_corridor_workspace_override_active": bool(
                        corridor.get("workspace_override_active")
                    ),
                    **meta,
                },
            )
        )

    def add_axis_segments(
        prefix: str,
        start_pos: list[float],
        end_pos: list[float],
        quat: list[float],
        meta: dict[str, Any],
    ) -> list[float]:
        cursor = [float(value) for value in list(start_pos)[:3]]
        target = [float(value) for value in list(end_pos)[:3]]
        delta = _sub_vec(target, cursor)
        count = max(1, int(math.ceil(_norm(delta) / spacing))) if _norm(delta) > 1.0e-6 else 0
        for index in range(count):
            alpha = float(index + 1) / float(count)
            pos = [float(cursor[i]) + float(delta[i]) * alpha for i in range(3)]
            add_stage(
                f"{prefix}_{index:02d}",
                pos,
                quat,
                {
                    **meta,
                    "wrist_audit_axis_segment_index": index,
                    "wrist_audit_axis_segment_count": count,
                },
            )
        return target

    setup_joint = _flat_vec(corridor.get("setup_joint", []), 6)
    setup_pose = _wrist_joint_values_to_dex_pose(base, setup_joint)
    setup_pos = _flat_vec(setup_pose.get("dex_pos", []), 3)
    setup_quat = _quat_normalize_wxyz(setup_pose.get("dex_quat", []))
    if len(setup_pos) == 3 and _norm(setup_quat) > 1.0e-6:
        add_stage(
            "verified_corridor_setup",
            setup_pos,
            setup_quat,
            {
                "seed_replay_stage_kind": "verified_corridor_setup",
                "seed_replay_corridor_setup_joint": setup_joint,
                "seed_replay_corridor_start_joint": corridor.get("start_wrist_joint", []),
            },
        )
    else:
        setup_pos = [float(value) for value in list(state.get("palm_local_pos", final_target))[:3]]
        setup_quat = _current_wrist_quat_from_state(state, cfg)

    cursor = list(setup_pos)
    corridor_quat = list(setup_quat)
    for index, joint in enumerate(list(corridor.get("joint_waypoints", []) or [])):
        pose = _wrist_joint_values_to_dex_pose(base, _flat_vec(joint, 6))
        pos = _flat_vec(pose.get("dex_pos", []), 3)
        quat = _quat_normalize_wxyz(pose.get("dex_quat", []))
        if len(pos) != 3 or _norm(quat) <= 1.0e-6:
            continue
        add_stage(
            f"verified_corridor_x_{index:02d}",
            pos,
            quat,
            {
                "seed_replay_stage_kind": "verified_corridor_x",
                "seed_replay_corridor_waypoint_index": index,
                "seed_replay_corridor_total_x_m": corridor.get("total_x_m", ""),
            },
        )
        cursor = list(pos)
        corridor_quat = list(quat)

    high_z = float(cursor[2]) if len(cursor) >= 3 else float(final_target[2])
    high_x = [float(final_target[0]), float(cursor[1]), high_z]
    cursor = add_axis_segments(
        "verified_corridor_extend_x",
        cursor,
        high_x,
        corridor_quat,
        {"seed_replay_stage_kind": "verified_corridor_extend_x"},
    )
    high_xy = [float(final_target[0]), float(final_target[1]), high_z]
    cursor = add_axis_segments(
        "verified_corridor_shift_y",
        cursor,
        high_xy,
        corridor_quat,
        {"seed_replay_stage_kind": "verified_corridor_shift_y"},
    )
    add_stage(
        "seed_orientation_at_verified_corridor",
        cursor,
        final_target_quat,
        {"seed_replay_stage_kind": "standoff_orientation"},
    )
    cursor = add_axis_segments(
        "force_guarded_final_seed_descent",
        cursor,
        final_target,
        final_target_quat,
        {
            **final_target_safety,
            "seed_replay_stage_kind": "final_contact_approach",
            "seed_target_local_xyz": list(final_target),
            "seed_target_quat_wxyz": list(final_target_quat),
        },
    )
    if not stages or _distance(cursor, final_target) > 1.0e-6:
        add_stage(
            "force_guarded_final_seed_approach",
            final_target,
            final_target_quat,
            {
                **final_target_safety,
                "seed_replay_stage_kind": "final_contact_approach",
                "seed_target_local_xyz": list(final_target),
                "seed_target_quat_wxyz": list(final_target_quat),
            },
        )

    if planned_positions:
        margin = max(float(cfg.wrist_isolation_workspace_margin_m), 0.020)
        lower = [min(row[i] for row in planned_positions) - margin for i in range(3)]
        upper = [max(row[i] for row in planned_positions) + margin for i in range(3)]
        corridor["workspace_override_active"] = True
        corridor["workspace_override_lower_xyz"] = lower
        corridor["workspace_override_upper_xyz"] = upper
        corridor["seed_replay_verified_corridor_stage_count"] = len(stages)
    return stages, corridor


def _wrist_seed_a_quat(cfg: Screw1GraspBaselineV2Config) -> list[float]:
    return _quat_for_rpy_offset(_quat_normalize_wxyz(cfg.canonical_hand_park_quat_wxyz), cfg.seed_a_orientation_rpy_deg)


def _wrist_combined_sequence(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    *,
    orientation_first: bool,
) -> list[tuple[str, list[float], list[float], dict[str, Any]]]:
    safe_pos, seed_quat, safety = _wrist_safe_seed_standoff_target(
        base, env_index, cfg, state, use_seed_orientation=True
    )
    current_quat = _current_wrist_quat_from_state(state, cfg)
    if orientation_first:
        return [
            (
                "safe_orientation",
                list(state.get("palm_local_pos", [])),
                seed_quat,
                {"wrist_audit_sequence_mode": "orientation_first"},
            ),
            *_wrist_staged_translation_sequence(
                cfg,
                state,
                safe_pos,
                seed_quat,
                {**safety, "wrist_audit_sequence_mode": "orientation_first", "wrist_audit_staged_translation": True},
                stage_prefix="translation_to_standoff",
            ),
        ]
    return [
        *_wrist_staged_translation_sequence(
            cfg,
            state,
            safe_pos,
            current_quat,
            {
                **safety,
                "wrist_audit_sequence_mode": "translation_first",
                "wrist_audit_staged_translation": True,
                "wrist_audit_standoff_orientation_source": "current_reset",
            },
            stage_prefix="translation_to_standoff",
        ),
        ("safe_orientation", safe_pos, seed_quat, {"wrist_audit_sequence_mode": "translation_first"}),
    ]


def _best_wrist_sequence_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def score(row: dict[str, Any]) -> tuple[int, float, float]:
        pos = float(row.get("final_target_to_actual_position_error_m", 1.0e9) or 1.0e9)
        angle = float(row.get("final_target_to_actual_angle_error_deg", 1.0e9) or 1.0e9)
        return (int(bool(row.get("sequence_passed"))), -pos, -angle)

    return dict(max(rows, key=score))


def _run_wrist_quaternion_static_tests() -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    base_cases = [
        ("identity", [0.0, 0.0, 0.0]),
        ("roll_pos", [15.0, 0.0, 0.0]),
        ("roll_neg", [-15.0, 0.0, 0.0]),
        ("pitch_pos", [0.0, 15.0, 0.0]),
        ("pitch_neg", [0.0, -15.0, 0.0]),
        ("yaw_pos", [0.0, 0.0, 15.0]),
        ("yaw_neg", [0.0, 0.0, -15.0]),
        ("combined", [10.0, 20.0, -30.0]),
        ("near_pi_pos", [179.0, -5.0, 3.0]),
        ("near_pi_neg", [-179.0, 5.0, -3.0]),
    ]
    identity = [1.0, 0.0, 0.0, 0.0]
    controls = [identity, _quat_from_serial_xyz_wxyz(math.radians(7.0), math.radians(-4.0), math.radians(9.0))]
    for control_idx, control in enumerate(controls):
        for name, rpy in base_cases:
            desired = _quat_mul_wxyz(
                _quat_from_serial_xyz_wxyz(
                    math.radians(float(rpy[0])),
                    math.radians(float(rpy[1])),
                    math.radians(float(rpy[2])),
                ),
                control,
            )
            command_axis = _quat_error_axis_angle_wxyz(control, desired)
            command_quat = _quat_from_axis_angle_vec_wxyz(command_axis)
            applied = _quat_mul_wxyz(command_quat, control)
            angle_error = _quat_angle_delta_wxyz(applied, desired)
            tests.append(
                {
                    "test_name": f"{name}_control{control_idx}",
                    "angle_error_rad": angle_error,
                    "passed": bool(angle_error <= 1.0e-6),
                }
            )
    max_error = max([0.0, *[float(row.get("angle_error_rad", 0.0) or 0.0) for row in tests]])
    return {
        "quaternion_static_tests_passed": all(bool(row.get("passed")) for row in tests),
        "quaternion_static_test_count": len(tests),
        "quaternion_static_max_apply_error_rad": max_error,
        "quaternion_static_tests": tests,
    }


def _run_action_authority_audit(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    joint_names = [str(item) for item in list(getattr(base, "dex_hand_joint_names", []) or [])]
    required = _required_finger_joint_specs(base, joint_names)
    finger_by_local: dict[int, str] = {
        int(spec.get("local_hand_index", -1)): str(spec.get("finger", ""))
        for spec in required
        if int(spec.get("local_hand_index", -1) or -1) >= 0
    }
    authority_by_joint: dict[str, bool] = {}
    for spec in required:
        if not bool(spec.get("joint_name_found")):
            row = {
                "phase": "action_authority",
                **spec,
                "probe_sign": "",
                "probe_action_value": "",
                "target_delta_rad": "",
                "actual_joint_delta_rad": "",
                "tip_delta_m": "",
                "crosstalk_actual_delta_peak_rad": "",
                "finger3_force_n": "",
                "finger4_force_n": "",
                "authority_ok": False,
                "authority_failure_reason": "required_joint_name_missing",
            }
            rows.append(row)
            authority_by_joint[str(spec["joint_name"])] = False
            continue
        joint_ok = False
        for sign in (1.0, -1.0):
            fresh = _fresh_episode(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                f"action_authority_{spec['finger']}_j{spec['joint_number']}_{'pos' if sign > 0 else 'neg'}",
                hand_mode="parked",
            )
            before = _hand_audit_snapshot(base, env_index)
            before_state = fresh["state"]
            action = _zero_action(env, base)
            action[env_index, int(spec["action_column"])] = float(sign * cfg.action_probe_value)
            after_state = before_state
            for step in range(int(cfg.action_probe_steps)):
                after_state = _step_direct_action(
                    env,
                    base,
                    env_index,
                    cfg,
                    action,
                    phase="action_authority",
                    step=len(rows) * int(cfg.action_probe_steps) + step,
                    trace_rows=trace_rows,
                    alignment=alignment,
                    extra={"probed_action_column": spec["action_column"], "probed_joint_name": spec["joint_name"]},
                )
            after = _hand_audit_snapshot(base, env_index)
            local = int(spec["local_hand_index"])
            target_delta = _list_delta(after.get("target", []), before.get("target", []), local)
            actual_delta = _list_delta(after.get("actual", []), before.get("actual", []), local)
            all_other: list[tuple[int, str, float]] = []
            same_finger_other: list[tuple[int, str, float]] = []
            cross_finger_other: list[tuple[int, str, float]] = []
            for idx in range(len(after.get("actual", []))):
                if idx == local:
                    continue
                delta = abs(_list_delta(after.get("actual", []), before.get("actual", []), idx))
                name = joint_names[idx] if 0 <= idx < len(joint_names) else f"hand_joint_{idx}"
                all_other.append((idx, name, delta))
                other_finger = finger_by_local.get(idx, "")
                if other_finger == str(spec["finger"]):
                    same_finger_other.append((idx, name, delta))
                elif other_finger:
                    cross_finger_other.append((idx, name, delta))
            other_idx, other_name, other_peak = max(all_other, key=lambda item: item[2], default=(-1, "", 0.0))
            same_idx, same_name, same_peak = max(
                same_finger_other, key=lambda item: item[2], default=(-1, "", 0.0)
            )
            cross_idx, cross_name, cross_peak = max(
                cross_finger_other, key=lambda item: item[2], default=(-1, "", 0.0)
            )
            crosstalk_ratio = other_peak / max(abs(actual_delta), 1.0e-9)
            same_finger_crosstalk_ratio = same_peak / max(abs(actual_delta), 1.0e-9)
            cross_finger_crosstalk_ratio = cross_peak / max(abs(actual_delta), 1.0e-9)
            tip_before = before_state["finger3_tip_local_pos"] if spec["finger"] == "finger3" else before_state["finger4_tip_local_pos"]
            tip_after = after_state["finger3_tip_local_pos"] if spec["finger"] == "finger3" else after_state["finger4_tip_local_pos"]
            tip_delta = _distance(tip_before, tip_after)
            ok = bool(
                abs(target_delta) >= cfg.action_authority_target_delta_min
                and abs(actual_delta) >= cfg.action_authority_actual_delta_min
                and cross_finger_crosstalk_ratio <= cfg.action_authority_crosstalk_max_ratio
                and max(
                    float(after_state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
                    float(after_state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
                )
                < cfg.contact_threshold_n
            )
            joint_ok = joint_ok or ok
            row = {
                "phase": "action_authority",
                **spec,
                "probe_sign": sign,
                "probe_action_value": sign * cfg.action_probe_value,
                "actual_robot_joint_index": spec.get("actual_robot_joint_index", ""),
                "target_delta_rad": target_delta,
                "actual_joint_delta_rad": actual_delta,
                "tip_delta_m": tip_delta,
                "tip_position_response_xyz": _sub_vec(tip_after, tip_before),
                "crosstalk_actual_delta_peak_rad": other_peak,
                "crosstalk_ratio": crosstalk_ratio,
                "crosstalk_peak_joint_index": other_idx,
                "crosstalk_peak_joint_name": other_name,
                "same_finger_crosstalk_peak_rad": same_peak,
                "same_finger_crosstalk_ratio": same_finger_crosstalk_ratio,
                "same_finger_crosstalk_peak_joint_index": same_idx,
                "same_finger_crosstalk_peak_joint_name": same_name,
                "cross_finger_crosstalk_peak_rad": cross_peak,
                "cross_finger_crosstalk_ratio": cross_finger_crosstalk_ratio,
                "cross_finger_crosstalk_peak_joint_index": cross_idx,
                "cross_finger_crosstalk_peak_joint_name": cross_name,
                "crosstalk_ratio_limit": cfg.action_authority_crosstalk_max_ratio,
                "finger3_force_n": after_state.get("finger3_target_filtered_force_n", 0.0),
                "finger4_force_n": after_state.get("finger4_target_filtered_force_n", 0.0),
                "authority_ok": ok,
                "authority_failure_reason": ""
                if ok
                else "target_or_actual_response_missing_or_cross_finger_crosstalk_or_contact",
                "authority_global_crosstalk_warning": bool(
                    crosstalk_ratio > cfg.action_authority_crosstalk_max_ratio
                ),
            }
            rows.append(row)
        authority_by_joint[str(spec["joint_name"])] = bool(joint_ok)
    required_ok = bool(required) and all(authority_by_joint.values())
    episodes.append(
        {
            "phase": "action_authority",
            "probe_count": len(rows),
            "required_joint_count": len(required),
            "required_joints_ok": required_ok,
        }
    )
    return {
        "action_authority_executed": True,
        "action_authority_probe_count": len(rows),
        "action_authority_required_joint_count": len(required),
        "action_authority_required_joints_ok": bool(required_ok),
        "action_authority_missing_joints": [
            name for name, ok in authority_by_joint.items() if not ok
        ],
        "action_authority_runtime_joint_map": required,
        "hand_joint_names": joint_names,
        "rows": rows,
    }


def _seed_replay_approach_sequence(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    *,
    seed_name: str,
    seed_offset: tuple[float, float, float],
    seed_rpy: tuple[float, float, float],
    final_target: list[float],
    final_target_quat: list[float],
    final_target_safety: dict[str, Any],
) -> list[tuple[str, list[float], list[float], dict[str, Any]]]:
    prelude, standoff_state = _seed_replay_corridor_prelude_sequence(base, env_index, cfg, state)
    standoff = _wrist_staged_seed_standoff_sequence(
        base,
        env_index,
        cfg,
        standoff_state,
        seed_offset=tuple(seed_offset),
        seed_rpy=tuple(seed_rpy),
        use_seed_orientation=False,
        sequence_mode="seed_replay_standoff",
    )
    standoff_pos = list(standoff[-1][1]) if standoff else list(final_target)
    return [
        *[
            (
                stage_name,
                pos,
                quat,
                {
                    **meta,
                    "seed_name": seed_name,
                },
            )
            for stage_name, pos, quat, meta in prelude
        ],
        *[
            (
                stage_name,
                pos,
                quat,
                {
                    **meta,
                    "seed_replay_stage_kind": "standoff_translation",
                    "seed_name": seed_name,
                },
            )
            for stage_name, pos, quat, meta in standoff
        ],
        (
            "seed_orientation_at_standoff",
            standoff_pos,
            _quat_normalize_wxyz(final_target_quat),
            {
                "seed_replay_stage_kind": "standoff_orientation",
                "seed_name": seed_name,
            },
        ),
        (
            "force_guarded_final_seed_approach",
            list(final_target),
            _quat_normalize_wxyz(final_target_quat),
            {
                **final_target_safety,
                "seed_replay_stage_kind": "final_contact_approach",
                "seed_name": seed_name,
                "seed_target_local_xyz": list(final_target),
                "seed_target_quat_wxyz": list(final_target_quat),
            },
        ),
    ]


def _run_servo_stage_sequence_guarded(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    stages: list[tuple[str, list[float], list[float], dict[str, Any]]],
    final_target_pos: list[float],
    final_target_quat: list[float],
    extra: dict[str, Any] | None = None,
    contact_truth: Any | None = None,
    all_body_abort_on_target: bool = True,
) -> dict[str, Any]:
    extra = dict(extra or {})
    stage_results: list[dict[str, Any]] = []
    final_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    final_target_pos = [float(value) for value in list(final_target_pos)[:3]]
    final_target_quat = _quat_normalize_wxyz(final_target_quat)
    initial_snapshot = _wrist_pose_snapshot(
        base,
        env_index,
        state=final_state,
        desired_pos=final_target_pos,
        desired_quat=final_target_quat,
    )
    min_final_pos_error = float(initial_snapshot.get("target_to_actual_position_error_m", float("inf")) or float("inf"))
    finger3_peak = 0.0
    finger4_peak = 0.0
    hard_abort = False
    tracking_timeout = False
    tracking_stall = False
    all_body_contact_abort = False
    all_body_contact_peak = 0.0
    all_body_target_contact_peak = 0.0
    all_body_non_target_contact_peak = 0.0
    reached_pose = False
    contact_ever = False
    termination = "sequence_not_executed"
    total_steps = 0
    free_space_steps = 0
    fine_steps = 0
    tracking_wait_steps = 0
    max_ctrl_to_palm = 0.0
    max_ctrl_to_actual_angle = 0.0
    max_target_to_ctrl = 0.0
    max_target_to_ctrl_angle = 0.0
    max_workspace_clamp = 0.0
    max_table_barrier = 0.0
    last_replay: dict[str, Any] = {}
    for stage_index, (stage_name, stage_pos, stage_quat, stage_meta) in enumerate(stages):
        stage_meta = dict(stage_meta or {})
        stage_kind = str(stage_meta.get("seed_replay_stage_kind", ""))
        stage_max_steps = int(cfg.seed_servo_max_steps)
        if stage_kind in {"standoff_translation", "standoff_orientation"}:
            stage_max_steps = max(1, min(stage_max_steps, 240))
        _write_seed_live_progress(
            cfg,
            {
                "phase": phase,
                "event": "servo_stage_started",
                "seed_name": extra.get("seed_name", stage_meta.get("seed_name", "")),
                "seed_repeat": extra.get("seed_repeat", ""),
                "servo_stage_name": stage_name,
                "servo_stage_index": int(stage_index),
                "servo_stage_count": int(len(stages)),
                "servo_stage_kind": stage_kind,
                "stage_max_steps": int(stage_max_steps),
                "target_pos_xyz": [float(value) for value in list(stage_pos)[:3]],
                "target_quat_wxyz": _quat_normalize_wxyz(stage_quat),
            },
        )
        replay = _servo_to_pose_guarded(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=phase,
            target_pos=[float(value) for value in list(stage_pos)[:3]],
            target_quat=_quat_normalize_wxyz(stage_quat),
            max_steps=stage_max_steps,
            contact_truth=contact_truth,
            all_body_abort_on_target=all_body_abort_on_target,
            extra={
                **extra,
                **stage_meta,
                "servo_stage_name": stage_name,
                "servo_stage_index": stage_index,
                "servo_stage_count": len(stages),
                "servo_sequence_order": "->".join(str(row[0]) for row in stages),
                "servo_final_target_pos_xyz": list(final_target_pos),
                "servo_final_target_quat_wxyz": list(final_target_quat),
            },
        )
        last_replay = replay
        _write_seed_live_progress(
            cfg,
            {
                "phase": phase,
                "event": "servo_stage_completed",
                "seed_name": extra.get("seed_name", stage_meta.get("seed_name", "")),
                "seed_repeat": extra.get("seed_repeat", ""),
                "servo_stage_name": stage_name,
                "servo_stage_index": int(stage_index),
                "servo_stage_count": int(len(stages)),
                "servo_stage_kind": stage_kind,
                "steps": int(replay.get("steps", 0) or 0),
                "termination_reason": replay.get("termination_reason", ""),
                "reached_pose": bool(replay.get("reached_pose")),
                "tracking_timeout": bool(replay.get("tracking_timeout")),
                "tracking_stall": bool(replay.get("tracking_stall")),
                "hard_abort": bool(replay.get("hard_abort")),
                "all_body_contact_abort": bool(replay.get("all_body_contact_abort")),
                "final_pos_error_m": replay.get("final_pos_error_m", ""),
                "final_finger3_force_n": replay.get("final_finger3_force_n", ""),
                "final_finger4_force_n": replay.get("final_finger4_force_n", ""),
            },
        )
        final_state = dict(replay.get("final_state") or final_state)
        total_steps += int(replay.get("steps", 0) or 0)
        free_space_steps += int(replay.get("free_space_step_count", 0) or 0)
        fine_steps += int(replay.get("fine_step_count", 0) or 0)
        tracking_wait_steps += int(replay.get("tracking_wait_step_count", 0) or 0)
        finger3_peak = max(finger3_peak, float(replay.get("finger3_force_peak_n", 0.0) or 0.0))
        finger4_peak = max(finger4_peak, float(replay.get("finger4_force_peak_n", 0.0) or 0.0))
        max_ctrl_to_palm = max(max_ctrl_to_palm, float(replay.get("max_ctrl_to_palm_error_m", 0.0) or 0.0))
        max_ctrl_to_actual_angle = max(
            max_ctrl_to_actual_angle,
            float(replay.get("max_ctrl_to_actual_angle_error_deg", 0.0) or 0.0),
        )
        max_target_to_ctrl = max(max_target_to_ctrl, float(replay.get("max_target_to_ctrl_error_m", 0.0) or 0.0))
        max_target_to_ctrl_angle = max(
            max_target_to_ctrl_angle,
            float(replay.get("max_target_to_ctrl_angle_error_deg", 0.0) or 0.0),
        )
        max_workspace_clamp = max(max_workspace_clamp, float(replay.get("max_workspace_clamp_delta_m", 0.0) or 0.0))
        max_table_barrier = max(max_table_barrier, float(replay.get("max_table_barrier_delta_z_m", 0.0) or 0.0))
        hard_abort = hard_abort or bool(replay.get("hard_abort"))
        tracking_timeout = tracking_timeout or bool(replay.get("tracking_timeout"))
        tracking_stall = tracking_stall or bool(replay.get("tracking_stall"))
        all_body_contact_abort = all_body_contact_abort or bool(replay.get("all_body_contact_abort"))
        all_body_contact_peak = max(
            all_body_contact_peak, float(replay.get("all_body_contact_peak_n", 0.0) or 0.0)
        )
        all_body_target_contact_peak = max(
            all_body_target_contact_peak, float(replay.get("all_body_target_contact_peak_n", 0.0) or 0.0)
        )
        all_body_non_target_contact_peak = max(
            all_body_non_target_contact_peak, float(replay.get("all_body_non_target_contact_peak_n", 0.0) or 0.0)
        )
        contact_ever = contact_ever or bool(replay.get("contact_ever_acquired"))
        reached_pose = bool(replay.get("reached_pose"))
        termination = str(replay.get("termination_reason", ""))
        snapshot = _wrist_pose_snapshot(
            base,
            env_index,
            state=final_state,
            desired_pos=final_target_pos,
            desired_quat=final_target_quat,
        )
        min_final_pos_error = min(
            min_final_pos_error,
            float(snapshot.get("target_to_actual_position_error_m", float("inf")) or float("inf")),
        )
        stage_results.append(
            {
                "stage_name": stage_name,
                "stage_kind": stage_kind,
                "termination_reason": termination,
                "steps": int(replay.get("steps", 0) or 0),
                "reached_pose": bool(replay.get("reached_pose")),
                "tracking_timeout": bool(replay.get("tracking_timeout")),
                "tracking_stall": bool(replay.get("tracking_stall")),
                "all_body_contact_abort": bool(replay.get("all_body_contact_abort")),
                "all_body_contact_peak_n": replay.get("all_body_contact_peak_n", 0.0),
                "all_body_target_contact_peak_n": replay.get("all_body_target_contact_peak_n", 0.0),
                "all_body_non_target_contact_peak_n": replay.get("all_body_non_target_contact_peak_n", 0.0),
                "hard_abort": bool(replay.get("hard_abort")),
                "final_target_to_actual_position_error_m": snapshot.get("target_to_actual_position_error_m", ""),
                "final_target_to_actual_angle_error_deg": _rad_to_deg_or_blank(
                    snapshot.get("target_to_actual_angle_error_rad", "")
                ),
            }
        )
        if (
            hard_abort
            or tracking_timeout
            or tracking_stall
            or all_body_contact_abort
            or bool(replay.get("final_current_contact"))
            or not bool(replay.get("reached_pose"))
        ):
            break
    final_snapshot = _wrist_pose_snapshot(
        base,
        env_index,
        state=final_state,
        desired_pos=final_target_pos,
        desired_quat=final_target_quat,
    )
    final_f3 = float(final_state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    final_f4 = float(final_state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    final_current_contact = bool(max(final_f3, final_f4) >= float(cfg.contact_threshold_n))
    final_pos_error = float(final_snapshot.get("target_to_actual_position_error_m", float("inf")) or float("inf"))
    final_angle_error = float(final_snapshot.get("target_to_actual_angle_error_rad", float("inf")) or float("inf"))
    return {
        "steps": total_steps,
        "termination_reason": termination,
        "final_state": final_state,
        "finger3_force_peak_n": finger3_peak,
        "finger4_force_peak_n": finger4_peak,
        "target_contact_acquired": bool(contact_ever or final_current_contact),
        "contact_ever_acquired": bool(contact_ever or final_current_contact),
        "final_current_contact": final_current_contact,
        "final_finger3_force_n": final_f3,
        "final_finger4_force_n": final_f4,
        "hard_abort": bool(hard_abort),
        "tracking_timeout": bool(tracking_timeout),
        "tracking_stall": bool(tracking_stall),
        "all_body_contact_abort": bool(all_body_contact_abort),
        "all_body_contact_peak_n": float(all_body_contact_peak),
        "all_body_target_contact_peak_n": float(all_body_target_contact_peak),
        "all_body_non_target_contact_peak_n": float(all_body_non_target_contact_peak),
        "reached_pose": bool(
            len(stage_results) == len(stages)
            and not bool(hard_abort or tracking_timeout or tracking_stall or all_body_contact_abort)
            and (
                reached_pose
                or (
                    final_pos_error <= float(cfg.seed_reach_tol_m)
                    and final_angle_error <= math.radians(float(cfg.seed_reach_rot_tol_deg))
                )
            )
        ),
        "final_pos_error_m": final_pos_error if math.isfinite(final_pos_error) else "",
        "min_pos_error_m": min_final_pos_error if math.isfinite(min_final_pos_error) else "",
        "free_space_step_count": int(free_space_steps),
        "fine_step_count": int(fine_steps),
        "tracking_wait_step_count": int(tracking_wait_steps),
        "max_ctrl_to_palm_error_m": max_ctrl_to_palm,
        "max_ctrl_to_actual_angle_error_deg": max_ctrl_to_actual_angle,
        "max_target_to_ctrl_error_m": max_target_to_ctrl,
        "max_target_to_ctrl_angle_error_deg": max_target_to_ctrl_angle,
        "final_ctrl_target_local_pos": final_snapshot.get("control_dex_pos", []),
        "final_target_to_ctrl_error_m": final_snapshot.get("target_to_control_position_error_m", ""),
        "final_ctrl_to_palm_error_m": final_snapshot.get("control_to_actual_position_error_m", ""),
        "final_ctrl_target_quat_wxyz": final_snapshot.get("control_dex_quat", []),
        "final_target_to_ctrl_angle_error_deg": _rad_to_deg_or_blank(
            final_snapshot.get("target_to_control_angle_error_rad", "")
        ),
        "final_ctrl_to_actual_angle_error_deg": _rad_to_deg_or_blank(
            final_snapshot.get("control_to_actual_angle_error_rad", "")
        ),
        "final_wrist_target_pre_clamp_xyz": last_replay.get("final_wrist_target_pre_clamp_xyz", []),
        "final_wrist_target_post_clamp_xyz": last_replay.get("final_wrist_target_post_clamp_xyz", []),
        "final_wrist_target_delta_xyz": last_replay.get("final_wrist_target_delta_xyz", []),
        "max_workspace_clamp_delta_m": max_workspace_clamp,
        "max_table_barrier_delta_z_m": max_table_barrier,
        "final_workspace_clamp_delta_m": final_state.get("workspace_clamp_delta_m", 0.0),
        "final_table_barrier_delta_z_m": final_state.get("table_barrier_delta_z_m", 0.0),
        "servo_stage_count": len(stages),
        "servo_completed_stage_count": len(stage_results),
        "servo_first_failed_stage": "" if len(stage_results) == len(stages) and bool(reached_pose) else str(stage_results[-1].get("stage_name", "") if stage_results else ""),
        "servo_strict_serial_stop": bool(len(stage_results) < len(stages) or not bool(reached_pose)),
        "servo_stage_results": stage_results,
        "servo_stage_sequence": [str(row[0]) for row in stages],
    }


def _run_seed_replay(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seeds: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
        ("SeedA_finger4_known_contact", cfg.seed_a_position_offset_m, cfg.seed_a_orientation_rpy_deg),
        ("SeedB_finger3_known_contact", cfg.seed_b_position_offset_m, cfg.seed_b_orientation_rpy_deg),
    ]
    rows = []
    best: dict[str, Any] = {}
    ref_fingerprint: dict[str, Any] | None = None
    max_fp_delta = 0.0
    contact_truth = _V2AllBodyContactTruth(
        Path(cfg.output_dir),
        env_index,
        cfg.part,
        max(float(getattr(cfg, "wrist_isolation_contact_threshold_n", cfg.contact_threshold_n)), float(cfg.contact_threshold_n)),
        log_prefix="seed_replay",
    )
    contact_truth.setup(base)
    resumed_seed_replay_progress = False
    if bool(getattr(cfg, "seed_replay_resume_progress", False)):
        loaded_rows = _load_seed_replay_progress_rows(cfg)
        if loaded_rows:
            for loaded_row in loaded_rows:
                loaded_row["acquisition_mode"] = _candidate_acquisition_mode(cfg, loaded_row)
                loaded_row["acquisition_plan_candidate"] = bool(loaded_row["acquisition_mode"])
            rows.extend(loaded_rows)
            resumed_seed_replay_progress = True
            for loaded_row in loaded_rows:
                best = _better_seed_replay(cfg, best, loaded_row)
            _write_seed_live_progress(
                cfg,
                {
                    "phase": "seed_replay",
                    "event": "resumed_seed_replay_progress",
                    "completed_candidate_count": int(len(rows)),
                    "resume_progress_csv": str(Path(cfg.output_dir) / "seed_replay_progress.csv"),
                },
            )

    def run_candidates(
        candidates: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
        repeats: int,
    ) -> None:
        nonlocal best, ref_fingerprint, max_fp_delta
        for seed_name, offset, rpy in candidates:
            for repeat in range(int(repeats)):
                _write_seed_live_progress(
                    cfg,
                    {
                        "phase": "seed_replay",
                        "event": "candidate_started",
                        "seed_name": seed_name,
                        "seed_repeat": int(repeat),
                        "seed_position_offset_xyz_m": list(offset),
                        "seed_orientation_rpy_deg": list(rpy),
                        "completed_candidate_count": int(len(rows)),
                    },
                )
                fresh = _fresh_episode(
                    env,
                    base,
                    env_index,
                    cfg,
                    alignment,
                    trace_rows,
                    f"seed_replay_{seed_name}_{repeat}",
                    hand_mode="parked",
                )
                if ref_fingerprint is None:
                    ref_fingerprint = dict(fresh.get("fingerprint", {}) or {})
                else:
                    max_fp_delta = max(
                        max_fp_delta,
                        _fingerprint_delta(ref_fingerprint, dict(fresh.get("fingerprint", {}) or {})),
                    )
                start_state = fresh["state"]
                raw_target, target_quat = _seed_target_pose(cfg, start_state, tuple(offset), tuple(rpy))
                target, target_safety = _barrier_safe_seed_target(
                    base, env_index, cfg, raw_target, target_quat
                )
                corridor: dict[str, Any] = {}
                workspace_override_applied = False
                if bool(cfg.seed_replay_use_verified_corridor_transit):
                    approach_stages, corridor = _seed_replay_verified_corridor_sequence(
                        base,
                        env_index,
                        cfg,
                        start_state,
                        seed_name=seed_name,
                        final_target=target,
                        final_target_quat=target_quat,
                        final_target_safety=target_safety,
                    )
                    _set_v2_workspace_override_for_corridor(base, env_index, corridor, cfg)
                    workspace_override_applied = bool(corridor.get("workspace_override_applied"))
                else:
                    approach_stages = _seed_replay_approach_sequence(
                        base,
                        env_index,
                        cfg,
                        start_state,
                        seed_name=seed_name,
                        seed_offset=tuple(offset),
                        seed_rpy=tuple(rpy),
                        final_target=target,
                        final_target_quat=target_quat,
                        final_target_safety=target_safety,
                    )
                try:
                    replay = _run_servo_stage_sequence_guarded(
                        env,
                        base,
                        env_index,
                        cfg,
                        alignment,
                        trace_rows,
                        phase="seed_replay",
                        stages=approach_stages,
                        final_target_pos=target,
                        final_target_quat=target_quat,
                        contact_truth=contact_truth,
                        all_body_abort_on_target=False,
                        extra={
                            "seed_name": seed_name,
                            "seed_repeat": repeat,
                            "seed_position_offset_xyz_m": list(offset),
                            "seed_orientation_rpy_deg": list(rpy),
                            "seed_replay_verified_corridor_transit": bool(
                                cfg.seed_replay_use_verified_corridor_transit
                            ),
                            "seed_replay_corridor_workspace_override_applied": bool(
                                workspace_override_applied
                            ),
                            "seed_replay_corridor_workspace_lower_xyz": corridor.get(
                                "workspace_override_lower_xyz", []
                            ),
                            "seed_replay_corridor_workspace_upper_xyz": corridor.get(
                                "workspace_override_upper_xyz", []
                            ),
                            **target_safety,
                            "seed_target_local_xyz": list(target),
                            "seed_target_quat_wxyz": list(target_quat),
                        },
                    )
                finally:
                    if workspace_override_applied:
                        _clear_v2_workspace_override(base, env_index)
                final = replay["final_state"]
                hold = _seed_contact_hold(
                    env,
                    base,
                    env_index,
                    cfg,
                    alignment,
                    trace_rows,
                    phase="seed_replay_contact_hold",
                    start_obj=start_state["object_local_pos"],
                    seed_name=seed_name,
                    repeat=repeat,
                ) if bool(replay.get("final_current_contact")) and not bool(replay.get("hard_abort")) else {}
                max_force = max(
                    float(replay.get("finger3_force_peak_n", 0.0) or 0.0),
                    float(replay.get("finger4_force_peak_n", 0.0) or 0.0),
                    float(hold.get("finger3_force_peak_n", 0.0) or 0.0),
                    float(hold.get("finger4_force_peak_n", 0.0) or 0.0),
                )
                object_displacement = max(
                    _distance(start_state["object_local_pos"], final["object_local_pos"]),
                    float(hold.get("object_displacement_m", 0.0) or 0.0),
                )
                soft_overshoot = bool(max_force > cfg.soft_force_max_n)
                valid_seed = bool(
                    bool(hold.get("seed_contact_sustained"))
                    and not bool(replay.get("hard_abort"))
                    and not bool(replay.get("tracking_timeout"))
                    and not soft_overshoot
                    and cfg.soft_force_min_n <= max_force <= cfg.soft_force_max_n
                    and object_displacement <= cfg.stable_preclose_object_motion_limit_m
                )
                geometry = _seed_candidate_geometry_metrics(final)
                row = {
                    "phase": "seed_replay",
                    "seed_name": seed_name,
                    "repeat": repeat,
                    "steps": replay["steps"],
                    "termination_reason": replay["termination_reason"],
                    "final_pos_error_m": replay.get("final_pos_error_m", ""),
                    "min_pos_error_m": replay.get("min_pos_error_m", ""),
                    "free_space_step_count": replay.get("free_space_step_count", 0),
                    "fine_step_count": replay.get("fine_step_count", 0),
                    "tracking_wait_step_count": replay.get("tracking_wait_step_count", 0),
                    "servo_stage_count": replay.get("servo_stage_count", 0),
                    "servo_completed_stage_count": replay.get("servo_completed_stage_count", 0),
                    "servo_stage_sequence": replay.get("servo_stage_sequence", []),
                    "servo_stage_results": replay.get("servo_stage_results", []),
                    "acquisition_stage_waypoints": _serialize_acquisition_stages(approach_stages),
                    "canonical_reset_fingerprint": fresh.get("fingerprint", {}),
                    "hand_preshape": _hand_pose_list(base, "dex_hand_preshape_pose"),
                    "max_ctrl_to_palm_error_m": replay.get("max_ctrl_to_palm_error_m", 0.0),
                    "final_palm_local_pos": final.get("palm_local_pos", []),
                    **geometry,
                    "final_ctrl_target_local_pos": replay.get("final_ctrl_target_local_pos", []),
                    "final_target_to_ctrl_error_m": replay.get("final_target_to_ctrl_error_m", ""),
                    "final_ctrl_to_palm_error_m": replay.get("final_ctrl_to_palm_error_m", ""),
                    "final_ctrl_target_quat_wxyz": replay.get("final_ctrl_target_quat_wxyz", []),
                    "final_target_to_ctrl_angle_error_deg": replay.get("final_target_to_ctrl_angle_error_deg", ""),
                    "final_ctrl_to_actual_angle_error_deg": replay.get("final_ctrl_to_actual_angle_error_deg", ""),
                    "max_ctrl_to_actual_angle_error_deg": replay.get("max_ctrl_to_actual_angle_error_deg", ""),
                    "max_target_to_ctrl_error_m": replay.get("max_target_to_ctrl_error_m", ""),
                    "max_target_to_ctrl_angle_error_deg": replay.get("max_target_to_ctrl_angle_error_deg", ""),
                    "tracking_timeout": bool(replay.get("tracking_timeout")),
                    "all_body_contact_abort": bool(replay.get("all_body_contact_abort")),
                    "all_body_contact_peak_n": replay.get("all_body_contact_peak_n", 0.0),
                    "all_body_target_contact_peak_n": replay.get("all_body_target_contact_peak_n", 0.0),
                    "all_body_non_target_contact_peak_n": replay.get("all_body_non_target_contact_peak_n", 0.0),
                    "final_wrist_target_pre_clamp_xyz": replay.get("final_wrist_target_pre_clamp_xyz", []),
                    "final_wrist_target_post_clamp_xyz": replay.get("final_wrist_target_post_clamp_xyz", []),
                    "final_wrist_target_delta_xyz": replay.get("final_wrist_target_delta_xyz", []),
                    "max_workspace_clamp_delta_m": replay.get("max_workspace_clamp_delta_m", 0.0),
                    "max_table_barrier_delta_z_m": replay.get("max_table_barrier_delta_z_m", 0.0),
                    "final_workspace_clamp_delta_m": replay.get("final_workspace_clamp_delta_m", 0.0),
                    "final_table_barrier_delta_z_m": replay.get("final_table_barrier_delta_z_m", 0.0),
                    "seed_replay_verified_corridor_transit": bool(cfg.seed_replay_use_verified_corridor_transit),
                    "seed_replay_corridor_workspace_override_applied": bool(workspace_override_applied),
                    "seed_replay_corridor_workspace_lower_xyz": corridor.get("workspace_override_lower_xyz", []),
                    "seed_replay_corridor_workspace_upper_xyz": corridor.get("workspace_override_upper_xyz", []),
                    "seed_replay_corridor_scene_max_z_m": corridor.get("scene_max_z_m", ""),
                    "seed_replay_corridor_setup_to_scene_top_margin_m": corridor.get(
                        "corridor_setup_to_scene_top_margin_m", ""
                    ),
                    "seed_target_original_local_xyz": target_safety.get("seed_target_original_local_xyz", []),
                    "seed_target_barrier_adjustment_z_m": target_safety.get(
                        "seed_target_barrier_adjustment_z_m", 0.0
                    ),
                    "seed_target_barrier_min_target_z_m": target_safety.get(
                        "seed_target_barrier_min_target_z_m", ""
                    ),
                    "seed_target_grasp_frame_offset_z_m": target_safety.get(
                        "seed_target_grasp_frame_offset_z_m", ""
                    ),
                    "target_pos_xyz": list(target),
                    "target_quat_wxyz": list(target_quat),
                    "finger3_force_peak_n": max(
                        float(replay.get("finger3_force_peak_n", 0.0) or 0.0),
                        float(hold.get("finger3_force_peak_n", 0.0) or 0.0),
                    ),
                    "finger4_force_peak_n": max(
                        float(replay.get("finger4_force_peak_n", 0.0) or 0.0),
                        float(hold.get("finger4_force_peak_n", 0.0) or 0.0),
                    ),
                    "target_contact_acquired": replay["target_contact_acquired"],
                    "contact_ever_acquired": bool(replay.get("contact_ever_acquired")),
                    "approach_final_current_contact": bool(replay.get("final_current_contact")),
                    "approach_final_finger3_force_n": replay.get("final_finger3_force_n", 0.0),
                    "approach_final_finger4_force_n": replay.get("final_finger4_force_n", 0.0),
                    "seed_contact_sustained": bool(hold.get("seed_contact_sustained")),
                    "seed_contact_duty_ratio": hold.get("target_contact_duty_ratio", 0.0),
                    "seed_final_current_contact": bool(hold.get("final_current_contact")),
                    "object_displacement_m": object_displacement,
                    "hard_abort": replay["hard_abort"],
                    "soft_overshoot": soft_overshoot,
                    "valid_seed": valid_seed,
                    "reached_pose": replay["reached_pose"],
                    "offset": list(offset),
                    "rpy": list(rpy),
                }
                row["acquisition_mode"] = _candidate_acquisition_mode(cfg, row)
                row["acquisition_plan_candidate"] = bool(row["acquisition_mode"])
                row["legacy_seed_invalid_under_corrected_protocol"] = bool(
                    str(seed_name).startswith("Seed") and not bool(row.get("target_contact_acquired"))
                )
                rows.append(row)
                episodes.append(row)
                best = _better_seed_replay(cfg, best, row)
                _write_seed_replay_progress(cfg, rows, best)

    if (not resumed_seed_replay_progress) and (not bool(getattr(cfg, "seed_replay_skip_legacy_seeds", False))):
        run_candidates(seeds, int(cfg.seed_replay_repeats))
    def candidate_already_tested(
        candidate: tuple[str, tuple[float, float, float], tuple[float, float, float]],
    ) -> bool:
        name, offset, rpy = candidate
        for row in rows:
            if str(row.get("seed_name", "")) == str(name):
                return True
            row_offset = list(row.get("offset", []) or [])
            row_rpy = list(row.get("rpy", []) or [])
            if len(row_offset) >= 3 and len(row_rpy) >= 3:
                if _distance(list(offset), row_offset[:3]) <= 5.0e-4 and _distance(list(rpy), row_rpy[:3]) <= 0.25:
                    return True
        return False

    legacy_seed_validated = any(bool(row.get("valid_seed")) for row in rows)
    trajectory_bank_executed = False
    if not legacy_seed_validated:
        missing_bank_candidates = [
            candidate for candidate in _trajectory_bank_seed_candidates(cfg) if not candidate_already_tested(candidate)
        ]
        if missing_bank_candidates:
            trajectory_bank_executed = True
            run_candidates(missing_bank_candidates, 1)
    automatic_correction_executed = False
    automatic_correction_candidate_count = 0
    acquisition_candidate = _best_acquisition_candidate(cfg, rows)
    if not acquisition_candidate:
        automatic = _auto_correction_seed_candidates(cfg, rows)
        automatic_correction_candidate_count = len(automatic)
        if automatic:
            automatic_correction_executed = True
            run_candidates(automatic, 1)
            acquisition_candidate = _best_acquisition_candidate(cfg, rows)

    validated_plan: dict[str, Any] = {}
    validated_plan_path = ""
    if acquisition_candidate:
        matches = _matching_plan_rows(cfg, rows, acquisition_candidate)
        needed = max(0, int(cfg.acquisition_plan_validation_repeats) - len(matches))
        if needed > 0:
            validation_name = f"{acquisition_candidate.get('seed_name', 'acquisition_plan')}_validation"
            run_candidates(
                [
                    (
                        validation_name,
                        tuple(float(value) for value in list(acquisition_candidate.get("offset", []) or [])[:3]),
                        tuple(float(value) for value in list(acquisition_candidate.get("rpy", []) or [])[:3]),
                    )
                ],
                needed,
            )
            acquisition_candidate = _best_acquisition_candidate(cfg, rows)
            matches = _matching_plan_rows(cfg, rows, acquisition_candidate) if acquisition_candidate else []
        if acquisition_candidate and len(matches) >= int(cfg.acquisition_plan_validation_repeats):
            validated_plan = _make_acquisition_plan_from_row(cfg, acquisition_candidate, validation_rows=matches)
            validated_plan_path = _write_validated_acquisition_plan(cfg, validated_plan)
    progress_best = _best_seed_progress_row(rows)
    best_for_summary = best if best else acquisition_candidate
    contact_summary = contact_truth.finalize()
    return {
        "seed_replay_executed": True,
        "seed_replay_skip_legacy_seeds": bool(getattr(cfg, "seed_replay_skip_legacy_seeds", False)),
        "seed_replay_legacy_seed_validated": bool(legacy_seed_validated),
        "legacy_seed_invalid_under_corrected_protocol": bool(
            rows
            and not bool(getattr(cfg, "seed_replay_skip_legacy_seeds", False))
            and not any(str(row.get("seed_name", "")).startswith("Seed") and bool(row.get("target_contact_acquired")) for row in rows)
        ),
        "trajectory_bank_executed": bool(trajectory_bank_executed),
        "trajectory_bank_candidate_limit": int(cfg.trajectory_bank_candidate_limit),
        "trajectory_bank_x_offsets_m": list(getattr(cfg, "trajectory_bank_x_offsets_m", ()) or ()),
        "trajectory_bank_y_offsets_m": list(getattr(cfg, "trajectory_bank_y_offsets_m", ()) or ()),
        "trajectory_bank_z_offsets_m": list(getattr(cfg, "trajectory_bank_z_offsets_m", ()) or ()),
        "trajectory_bank_orientation_rpy_deg": [
            list(rpy) for rpy in (getattr(cfg, "trajectory_bank_orientation_rpy_deg", ()) or ())
        ],
        "automatic_correction_executed": bool(automatic_correction_executed),
        "automatic_correction_candidate_count": int(automatic_correction_candidate_count),
        "acquisition_plan_candidate_found": bool(acquisition_candidate),
        "acquisition_plan_validated": bool(validated_plan),
        "validated_acquisition_plan_json": validated_plan_path,
        "validated_acquisition_plan": validated_plan,
        "acquisition_plan_mode": validated_plan.get("acquisition_mode", acquisition_candidate.get("acquisition_mode", "") if acquisition_candidate else ""),
        "seed_replay_probe_count": len(rows),
        "seed_replay_resumed_progress": bool(resumed_seed_replay_progress),
        "seed_replay_verified_corridor_transit_used": bool(cfg.seed_replay_use_verified_corridor_transit),
        "seed_replay_corridor_workspace_override_trial_count": sum(
            1 for row in rows if bool(row.get("seed_replay_corridor_workspace_override_applied"))
        ),
        "seed_replay_corridor_scene_max_z_m": progress_best.get("seed_replay_corridor_scene_max_z_m", ""),
        "seed_replay_corridor_setup_to_scene_top_margin_m": progress_best.get(
            "seed_replay_corridor_setup_to_scene_top_margin_m", ""
        ),
        "seed_replay_fingerprint_delta_peak": max_fp_delta,
        "seed_replay_contact_trial_count": sum(1 for row in rows if bool(row.get("target_contact_acquired"))),
        "seed_replay_contact_ever_trial_count": sum(1 for row in rows if bool(row.get("contact_ever_acquired"))),
        "seed_replay_approach_final_current_contact_trial_count": sum(
            1 for row in rows if bool(row.get("approach_final_current_contact"))
        ),
        "seed_replay_validated_trial_count": sum(1 for row in rows if bool(row.get("valid_seed"))),
        "seed_replay_hard_abort_count": sum(1 for row in rows if bool(row.get("hard_abort"))),
        "seed_replay_tracking_timeout_count": sum(1 for row in rows if bool(row.get("tracking_timeout"))),
        "seed_replay_all_body_contact_abort_count": sum(1 for row in rows if bool(row.get("all_body_contact_abort"))),
        "seed_replay_all_body_contact_peak_n": max(
            [0.0, *[float(row.get("all_body_contact_peak_n", 0.0) or 0.0) for row in rows]]
        ),
        "seed_replay_all_body_target_contact_peak_n": max(
            [0.0, *[float(row.get("all_body_target_contact_peak_n", 0.0) or 0.0) for row in rows]]
        ),
        "seed_replay_all_body_non_target_contact_peak_n": max(
            [0.0, *[float(row.get("all_body_non_target_contact_peak_n", 0.0) or 0.0) for row in rows]]
        ),
        "seed_replay_all_body_contact_log_csv": contact_summary.get("all_body_contact_log_csv", ""),
        "seed_replay_all_body_contact_summary_json": contact_summary.get("all_body_contact_summary_json", ""),
        "seed_replay_all_body_contact_setup_error": contact_summary.get("all_body_contact_setup_error", ""),
        "seed_replay_all_body_contact_any_active": bool(contact_summary.get("all_body_contact_any_active")),
        "seed_replay_soft_overshoot_count": sum(1 for row in rows if bool(row.get("soft_overshoot"))),
        "seed_replay_tracking_wait_step_count": sum(int(row.get("tracking_wait_step_count", 0) or 0) for row in rows),
        "seed_replay_free_space_step_count": sum(int(row.get("free_space_step_count", 0) or 0) for row in rows),
        "seed_replay_max_ctrl_to_palm_error_m": max(
            [0.0, *[float(row.get("max_ctrl_to_palm_error_m", 0.0) or 0.0) for row in rows]]
        ),
        "seed_replay_max_ctrl_to_actual_angle_error_deg": max(
            [0.0, *[float(row.get("max_ctrl_to_actual_angle_error_deg", 0.0) or 0.0) for row in rows]]
        ),
        "seed_replay_max_table_barrier_delta_z_m": max(
            [0.0, *[abs(float(row.get("max_table_barrier_delta_z_m", 0.0) or 0.0)) for row in rows]]
        ),
        "seed_replay_min_pos_error_m": progress_best.get("min_pos_error_m", ""),
        "seed_replay_best_progress_seed": progress_best.get("seed_name", ""),
        "seed_replay_best_progress_final_pos_error_m": progress_best.get("final_pos_error_m", ""),
        "seed_replay_best_progress_final_ctrl_to_palm_error_m": progress_best.get(
            "final_ctrl_to_palm_error_m", ""
        ),
        "seed_replay_best_progress_final_target_to_ctrl_error_m": progress_best.get(
            "final_target_to_ctrl_error_m", ""
        ),
        "seed_replay_best_progress_final_object_local_pos": progress_best.get("final_object_local_pos", []),
        "seed_replay_best_progress_final_palm_local_pos": progress_best.get("final_palm_local_pos", []),
        "seed_replay_best_progress_final_finger3_tip_local_pos": progress_best.get(
            "final_finger3_tip_local_pos", []
        ),
        "seed_replay_best_progress_final_finger4_tip_local_pos": progress_best.get(
            "final_finger4_tip_local_pos", []
        ),
        "seed_replay_best_progress_final_finger3_tip_to_object_xyz_m": progress_best.get(
            "final_finger3_tip_to_object_xyz_m", []
        ),
        "seed_replay_best_progress_final_finger4_tip_to_object_xyz_m": progress_best.get(
            "final_finger4_tip_to_object_xyz_m", []
        ),
        "seed_replay_best_progress_fingertip_midpoint_to_object_xyz_m": progress_best.get(
            "final_fingertip_midpoint_to_object_xyz_m", []
        ),
        "seed_replay_best_progress_tip_segment_xyz_m": progress_best.get("final_tip3_to_tip4_segment_xyz_m", []),
        "seed_replay_best_progress_segment_projection_t_raw": progress_best.get(
            "screw1_projection_parameter_on_tip_segment", ""
        ),
        "seed_replay_best_progress_segment_perpendicular_m": progress_best.get(
            "screw1_to_tip_segment_perpendicular_m", ""
        ),
        "seed_replay_best_progress_acquisition_mode": progress_best.get("acquisition_mode", ""),
        "seed_replay_validated": bool(best.get("valid_seed") or validated_plan),
        "seed_replay_best_seed": best_for_summary.get("seed_name", ""),
        "seed_replay_best_offset_xyz_m": best_for_summary.get("offset", []),
        "seed_replay_best_rpy_deg": best_for_summary.get("rpy", []),
        "seed_replay_best_finger3_force_peak_n": best_for_summary.get("finger3_force_peak_n", 0.0),
        "seed_replay_best_finger4_force_peak_n": best_for_summary.get("finger4_force_peak_n", 0.0),
        "seed_replay_best_object_displacement_m": best_for_summary.get("object_displacement_m", ""),
    }


def _best_seed_progress_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def score(row: dict[str, Any]) -> tuple[int, float, float]:
        contact = int(bool(row.get("target_contact_acquired")))
        try:
            min_err = float(row.get("min_pos_error_m", 1.0e9) or 1.0e9)
        except Exception:
            min_err = 1.0e9
        try:
            force = max(
                float(row.get("finger3_force_peak_n", 0.0) or 0.0),
                float(row.get("finger4_force_peak_n", 0.0) or 0.0),
            )
        except Exception:
            force = 0.0
        return (contact, -min_err, force)

    return dict(max(rows, key=score))


def _seed_candidate_geometry_metrics(state: dict[str, Any]) -> dict[str, Any]:
    obj = list(state.get("object_local_pos", []) or [])
    p3 = list(state.get("finger3_tip_local_pos", []) or [])
    p4 = list(state.get("finger4_tip_local_pos", []) or [])
    if len(obj) < 3 or len(p3) < 3 or len(p4) < 3:
        return {
            "candidate_geometry_available": False,
            "candidate_geometry_unavailable_reason": "missing_object_or_fingertip_pose",
        }
    midpoint = [(float(p3[i]) + float(p4[i])) * 0.5 for i in range(3)]
    segment = _sub_vec(p4, p3)
    seg_len = _norm(segment)
    tip3_to_obj = _sub_vec(p3, obj)
    tip4_to_obj = _sub_vec(p4, obj)
    midpoint_to_obj = _sub_vec(midpoint, obj)
    obj_from_p3 = _sub_vec(obj, p3)
    denom = max(1.0e-12, sum(float(segment[i]) * float(segment[i]) for i in range(3)))
    t_raw = sum(float(obj_from_p3[i]) * float(segment[i]) for i in range(3)) / denom
    t_clamped = max(0.0, min(1.0, float(t_raw)))
    projection = [float(p3[i]) + t_clamped * float(segment[i]) for i in range(3)]
    perpendicular = _distance(obj, projection)
    return {
        "candidate_geometry_available": True,
        "final_object_local_pos": obj,
        "final_finger3_tip_local_pos": p3,
        "final_finger4_tip_local_pos": p4,
        "final_fingertip_midpoint_local_pos": midpoint,
        "final_finger3_tip_to_object_xyz_m": tip3_to_obj,
        "final_finger4_tip_to_object_xyz_m": tip4_to_obj,
        "final_fingertip_midpoint_to_object_xyz_m": midpoint_to_obj,
        "final_tip3_to_tip4_segment_xyz_m": segment,
        "final_tip3_to_tip4_separation_m": seg_len,
        "screw1_projection_parameter_on_tip_segment": float(t_raw),
        "screw1_projection_parameter_clamped": float(t_clamped),
        "screw1_to_tip_segment_perpendicular_m": perpendicular,
        "screw1_projection_inside_tip_segment": bool(0.0 <= float(t_raw) <= 1.0),
    }


def _candidate_acquisition_mode(cfg: Screw1GraspBaselineV2Config, row: dict[str, Any]) -> str:
    f3 = max(float(row.get("finger3_force_peak_n", 0.0) or 0.0), float(row.get("approach_final_finger3_force_n", 0.0) or 0.0))
    f4 = max(float(row.get("finger4_force_peak_n", 0.0) or 0.0), float(row.get("approach_final_finger4_force_n", 0.0) or 0.0))
    max_force = max(f3, f4)
    object_motion = float(row.get("object_displacement_m", 1.0e9) or 1.0e9)
    t_raw = float(row.get("screw1_projection_parameter_on_tip_segment", 1.0e9) or 1.0e9)
    endpoint_margin = float(cfg.acquisition_endpoint_projection_margin)
    barrier_limited_pregrasp = bool(
        str(row.get("termination_reason", "")) == "table_barrier_abort"
        and float(row.get("max_table_barrier_delta_z_m", 1.0e9) or 1.0e9) <= 0.002
        and float(row.get("final_table_barrier_delta_z_m", 1.0e9) or 1.0e9) <= 0.002
    )
    if (
        bool(row.get("seed_contact_sustained"))
        and bool(row.get("seed_final_current_contact"))
        and float(cfg.soft_force_min_n) <= max_force <= float(cfg.soft_force_max_n)
        and object_motion <= float(cfg.stable_preclose_object_motion_limit_m)
        and not bool(row.get("hard_abort"))
        and not bool(row.get("soft_overshoot"))
    ):
        return "ANCHOR_CONTACT_PLAN"
    if (
        (bool(row.get("reached_pose")) or barrier_limited_pregrasp)
        and not bool(row.get("target_contact_acquired"))
        and not bool(row.get("contact_ever_acquired"))
        and not bool(row.get("all_body_contact_abort"))
        and object_motion <= float(cfg.stable_preclose_object_motion_limit_m)
        and math.isfinite(t_raw)
        and endpoint_margin <= t_raw <= 1.0 - endpoint_margin
        and float(row.get("screw1_to_tip_segment_perpendicular_m", 1.0e9) or 1.0e9)
        <= float(cfg.acquisition_caging_perpendicular_max_m)
        and float(cfg.acquisition_caging_tip_separation_min_m)
        <= float(row.get("final_tip3_to_tip4_separation_m", 0.0) or 0.0)
        <= float(cfg.acquisition_caging_tip_separation_max_m)
    ):
        return "CAGING_PREGRASP_PLAN"
    return ""


def _serialize_acquisition_stages(
    stages: list[tuple[str, list[float], list[float], dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "stage_name": str(name),
            "target_pos_xyz": [float(value) for value in list(pos)[:3]],
            "target_quat_wxyz": _quat_normalize_wxyz(quat),
            "metadata": dict(meta or {}),
        }
        for name, pos, quat, meta in stages
    ]


def _deserialize_acquisition_stages(plan: dict[str, Any]) -> list[tuple[str, list[float], list[float], dict[str, Any]]]:
    stage_rows = _coerce_seed_replay_csv_value(plan.get("staged_wrist_waypoints", []) or [])
    if not isinstance(stage_rows, list):
        stage_rows = []
    stages = []
    for row in stage_rows:
        if not isinstance(row, dict):
            continue
        pos = list(row.get("target_pos_xyz", []) or [])
        quat = list(row.get("target_quat_wxyz", []) or [])
        if len(pos) < 3 or len(quat) < 4:
            continue
        stages.append((str(row.get("stage_name", f"stage_{len(stages)}")), pos[:3], _quat_normalize_wxyz(quat), dict(row.get("metadata", {}) or {})))
    return stages


def _make_acquisition_plan_from_row(
    cfg: Screw1GraspBaselineV2Config,
    row: dict[str, Any],
    *,
    validation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mode = str(row.get("acquisition_mode") or "")
    plan_id = str(row.get("seed_name") or "acquisition_plan")
    validation_rows = list(validation_rows or [row])
    canonical_reset_fingerprint = _coerce_seed_replay_csv_value(row.get("canonical_reset_fingerprint", {}))
    if not isinstance(canonical_reset_fingerprint, dict):
        canonical_reset_fingerprint = {}
    hand_preshape = _coerce_seed_replay_csv_value(row.get("hand_preshape", []))
    if not isinstance(hand_preshape, list):
        hand_preshape = []
    staged_wrist_waypoints = _coerce_seed_replay_csv_value(row.get("acquisition_stage_waypoints", []))
    if not isinstance(staged_wrist_waypoints, list):
        staged_wrist_waypoints = []
    corridor_lower = _coerce_seed_replay_csv_value(row.get("seed_replay_corridor_workspace_lower_xyz", []))
    corridor_upper = _coerce_seed_replay_csv_value(row.get("seed_replay_corridor_workspace_upper_xyz", []))
    if not isinstance(corridor_lower, list):
        corridor_lower = []
    if not isinstance(corridor_upper, list):
        corridor_upper = []
    anchor_finger = ""
    if mode == "ANCHOR_CONTACT_PLAN":
        f3 = float(row.get("approach_final_finger3_force_n", row.get("finger3_force_peak_n", 0.0)) or 0.0)
        f4 = float(row.get("approach_final_finger4_force_n", row.get("finger4_force_peak_n", 0.0)) or 0.0)
        anchor_finger = "finger3" if f3 >= f4 else "finger4"
    correction = list(row.get("measured_correction_vector_xyz_m", []) or row.get("final_fingertip_midpoint_to_object_xyz_m", []) or [])
    return {
        "plan_id": plan_id,
        "acquisition_mode": mode,
        "canonical_reset_fingerprint": canonical_reset_fingerprint,
        "hand_preshape": hand_preshape,
        "seed_offset_xyz_m": row.get("offset", []),
        "final_orientation_rpy_deg": row.get("rpy", []),
        "final_target_pos_xyz": row.get("target_pos_xyz", []),
        "final_target_quat_wxyz": row.get("target_quat_wxyz", []),
        "verified_corridor_setup": {
            "workspace_override_applied": bool(row.get("seed_replay_corridor_workspace_override_applied")),
            "workspace_lower_xyz": corridor_lower,
            "workspace_upper_xyz": corridor_upper,
            "scene_max_z_m": row.get("seed_replay_corridor_scene_max_z_m", ""),
            "setup_to_scene_top_margin_m": row.get("seed_replay_corridor_setup_to_scene_top_margin_m", ""),
        },
        "staged_wrist_waypoints": staged_wrist_waypoints,
        "workspace_override_bounds": {
            "lower_xyz": corridor_lower,
            "upper_xyz": corridor_upper,
        },
        "measured_correction_vector_xyz_m": correction,
        "expected_tip3_geometry": {
            "tip_to_object_xyz_m": row.get("final_finger3_tip_to_object_xyz_m", []),
        },
        "expected_tip4_geometry": {
            "tip_to_object_xyz_m": row.get("final_finger4_tip_to_object_xyz_m", []),
        },
        "selected_anchor_finger": anchor_finger,
        "final_force_approach_direction_xyz": row.get("final_force_approach_direction_xyz", []),
        "final_force_approach_step_m": float(row.get("final_force_approach_step_m", 0.0005) or 0.0005),
        "force_thresholds": {
            "contact_threshold_n": float(cfg.contact_threshold_n),
            "soft_force_min_n": float(cfg.soft_force_min_n),
            "soft_force_max_n": float(cfg.soft_force_max_n),
            "hard_abort_force_n": float(cfg.hard_abort_force_n),
        },
        "valid_trial_evidence": validation_rows,
    }


def _acquisition_plan_score(cfg: Screw1GraspBaselineV2Config, row: dict[str, Any]) -> tuple[float, ...]:
    mode = str(row.get("acquisition_mode") or "")
    if mode == "ANCHOR_CONTACT_PLAN":
        priority = 3
    elif mode == "CAGING_PREGRASP_PLAN":
        priority = 2
    else:
        priority = 0
    object_motion = float(row.get("object_displacement_m", 1.0e9) or 1.0e9)
    perp = float(row.get("screw1_to_tip_segment_perpendicular_m", 1.0e9) or 1.0e9)
    t_raw = float(row.get("screw1_projection_parameter_on_tip_segment", 1.0e9) or 1.0e9)
    midpoint_error = abs(t_raw - 0.5) if math.isfinite(t_raw) else 1.0e9
    force = max(float(row.get("finger3_force_peak_n", 0.0) or 0.0), float(row.get("finger4_force_peak_n", 0.0) or 0.0))
    if mode == "CAGING_PREGRASP_PLAN":
        return (priority, -object_motion, -midpoint_error, -perp)
    return (priority, -object_motion, -abs(force - 0.3))


def _best_acquisition_candidate(cfg: Screw1GraspBaselineV2Config, rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if str(row.get("acquisition_mode") or "")]
    if not candidates:
        return {}
    return dict(max(candidates, key=lambda row: _acquisition_plan_score(cfg, row)))


def _matching_plan_rows(cfg: Screw1GraspBaselineV2Config, rows: list[dict[str, Any]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(candidate.get("acquisition_mode") or "")
    offset = list(candidate.get("offset", []) or [])
    rpy = list(candidate.get("rpy", []) or [])
    out = []
    for row in rows:
        if str(row.get("acquisition_mode") or "") != mode:
            continue
        if _distance(list(row.get("offset", []) or []), offset) > 1.0e-6:
            continue
        if _distance(list(row.get("rpy", []) or []), rpy) > 1.0e-6:
            continue
        if mode == "CAGING_PREGRASP_PLAN":
            if _distance(
                list(row.get("final_fingertip_midpoint_to_object_xyz_m", []) or []),
                list(candidate.get("final_fingertip_midpoint_to_object_xyz_m", []) or []),
            ) > float(cfg.acquisition_plan_geometry_tolerance_m):
                continue
        out.append(row)
    return out


def _write_validated_acquisition_plan(cfg: Screw1GraspBaselineV2Config, plan: dict[str, Any]) -> str:
    path = Path(cfg.output_dir) / "validated_acquisition_plan.json"
    write_json(path, plan)
    return str(path)


def _trajectory_bank_seed_candidates(
    cfg: Screw1GraspBaselineV2Config,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """Finite fresh-reset bank for corrected seed replay after legacy seeds fail.

    The legacy Seed A/B x offsets came from the pre-v2 reset protocol.  Under
    the corrected canonical reset plus verified corridor transit, both active
    tips landed roughly 5-6 cm negative of Screw1 in x.  This bank therefore
    starts with object-centered x offsets and the measured negative-y correction
    needed to bring the active tips around Screw1.
    """

    candidates: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
    x_offsets = tuple(float(value) for value in getattr(cfg, "trajectory_bank_x_offsets_m", ()) or ())
    y_offsets = tuple(float(value) for value in getattr(cfg, "trajectory_bank_y_offsets_m", ()) or ())
    z_offsets = tuple(float(value) for value in getattr(cfg, "trajectory_bank_z_offsets_m", ()) or ())
    rpy_offsets = tuple(
        tuple(float(value) for value in list(rpy)[:3])
        for rpy in (getattr(cfg, "trajectory_bank_orientation_rpy_deg", ()) or ())
        if len(list(rpy)) >= 3
    )

    if not x_offsets or not y_offsets or not z_offsets or not rpy_offsets:
        return []

    for z_index, z_offset in enumerate(z_offsets):
        x_groups = (x_offsets[:1], x_offsets[1:])
        for x_group in x_groups:
            for rpy_index, rpy in enumerate(rpy_offsets):
                for x_offset in x_group:
                    for y_offset in y_offsets:
                        offset = (float(x_offset), float(y_offset), float(z_offset))
                        candidates.append(
                            (
                                (
                                    f"bank_xcorr_rpy{rpy_index}_z{z_index}"
                                    f"_x{x_offset:+.4f}_y{y_offset:+.4f}_z{z_offset:+.4f}"
                                ),
                                offset,
                                rpy,
                            )
                        )
    limit = int(getattr(cfg, "trajectory_bank_candidate_limit", 0) or 0)
    if limit <= 0:
        return []
    return candidates[:limit]


def _clamp_offset(cfg: Screw1GraspBaselineV2Config, offset: list[float]) -> tuple[float, float, float]:
    bounds = (
        tuple(getattr(cfg, "acquisition_offset_x_bounds_m", (0.040, 0.110))),
        tuple(getattr(cfg, "acquisition_offset_y_bounds_m", (-0.085, 0.025))),
        tuple(getattr(cfg, "acquisition_offset_z_bounds_m", (-0.018, -0.004))),
    )
    out = []
    for index in range(3):
        lo, hi = float(bounds[index][0]), float(bounds[index][1])
        value = float(offset[index] if index < len(offset) else 0.0)
        out.append(max(lo, min(hi, value)))
    return (out[0], out[1], out[2])


def _auto_correction_seed_candidates(
    cfg: Screw1GraspBaselineV2Config,
    rows: list[dict[str, Any]],
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    def pose_key(offset: list[float] | tuple[float, ...], rpy: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        return tuple(round(float(value), 6) for value in [*list(offset)[:3], *list(rpy)[:3]])

    def pose_close(
        offset_a: list[float] | tuple[float, ...],
        rpy_a: list[float] | tuple[float, ...],
        offset_b: list[float] | tuple[float, ...],
        rpy_b: list[float] | tuple[float, ...],
    ) -> bool:
        if len(list(offset_a)[:3]) < 3 or len(list(offset_b)[:3]) < 3:
            return False
        if len(list(rpy_a)[:3]) < 3 or len(list(rpy_b)[:3]) < 3:
            return False
        return bool(
            _distance(list(offset_a)[:3], list(offset_b)[:3]) <= 5.0e-4
            and _distance(list(rpy_a)[:3], list(rpy_b)[:3]) <= 0.25
        )

    def vec3(row: dict[str, Any], key: str) -> list[float]:
        values = list(row.get(key, []) or [])
        if len(values) < 3:
            return [0.0, 0.0, 0.0]
        return [float(values[0]), float(values[1]), float(values[2])]

    completed_names = {str(row.get("seed_name", "")) for row in rows if str(row.get("seed_name", ""))}
    completed_poses = {
        pose_key(list(row.get("offset", []) or []), list(row.get("rpy", []) or []))
        for row in rows
        if len(list(row.get("offset", []) or [])) >= 3 and len(list(row.get("rpy", []) or [])) >= 3
    }
    source_rows = [
        row
        for row in rows
        if bool(row.get("candidate_geometry_available"))
        and not bool(row.get("hard_abort"))
        and not bool(row.get("all_body_contact_abort"))
        and not str(row.get("seed_name", "")).startswith("auto_")
        and list(row.get("offset", []) or [])
        and list(row.get("rpy", []) or [])
    ]
    if not source_rows:
        return []
    candidates: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
    limit = max(0, int(getattr(cfg, "acquisition_auto_correction_max_trials", 0) or 0))

    def add_candidate(kind: str, row: dict[str, Any], correction: list[float]) -> bool:
        rpy = tuple(float(value) for value in list(row.get("rpy", []) or [])[:3])
        if len(rpy) != 3:
            return False
        base_offset = [float(value) for value in list(row.get("offset", []) or [])[:3]]
        if len(base_offset) != 3:
            return False
        # Lateral correction is trusted most; z correction is intentionally bounded near the table.
        limited = [
            max(-0.020, min(0.020, float(correction[0]))),
            max(-0.030, min(0.030, float(correction[1]))),
            max(-0.010, min(0.004, float(correction[2]))),
        ]
        offset = _clamp_offset(cfg, [base_offset[i] + limited[i] for i in range(3)])
        name = (
            f"auto_{kind}_from_{row.get('seed_name', 'candidate')}"
            f"_dx{limited[0]:+.4f}_dy{limited[1]:+.4f}_dz{limited[2]:+.4f}"
        )
        if name in completed_names:
            return False
        key = pose_key(offset, rpy)
        if (
            key in completed_poses
            or any(
                pose_close(offset, rpy, list(row.get("offset", []) or []), list(row.get("rpy", []) or []))
                for row in rows
            )
            or any(pose_key(existing[1], existing[2]) == key or pose_close(offset, rpy, existing[1], existing[2]) for existing in candidates)
        ):
            return False
        candidates.append((name, offset, rpy))
        completed_names.add(name)
        completed_poses.add(key)
        return True

    ranked_by_kind = [
        (
            "cage_mid",
            sorted(
                source_rows,
                key=lambda row: (
                    not bool(row.get("screw1_projection_inside_tip_segment")),
                    float(row.get("screw1_to_tip_segment_perpendicular_m", 1.0e9) or 1.0e9),
                    abs(vec3(row, "final_fingertip_midpoint_to_object_xyz_m")[1]),
                    abs(vec3(row, "final_fingertip_midpoint_to_object_xyz_m")[0]),
                ),
            ),
            "final_fingertip_midpoint_to_object_xyz_m",
        ),
        (
            "finger3_anchor",
            sorted(source_rows, key=lambda row: _norm(vec3(row, "final_finger3_tip_to_object_xyz_m"))),
            "final_finger3_tip_to_object_xyz_m",
        ),
        (
            "finger4_anchor",
            sorted(source_rows, key=lambda row: _norm(vec3(row, "final_finger4_tip_to_object_xyz_m"))),
            "final_finger4_tip_to_object_xyz_m",
        ),
    ]
    for kind, ranked_rows, vector_key in ranked_by_kind:
        if len(candidates) >= limit:
            break
        if not ranked_rows:
            continue
        row = ranked_rows[0]
        correction = [-float(value) for value in vec3(row, vector_key)]
        add_candidate(kind, row, correction)
    return candidates


def _write_seed_replay_progress(
    cfg: Screw1GraspBaselineV2Config,
    rows: list[dict[str, Any]],
    best: dict[str, Any],
) -> None:
    output_dir = Path(cfg.output_dir)
    try:
        progress_best = _best_seed_progress_row(rows)
        _write_csv(output_dir / "seed_replay_progress.csv", rows)
        write_json(
            output_dir / "seed_replay_progress.json",
            {
                "seed_replay_probe_count": len(rows),
                "seed_replay_contact_trial_count": sum(
                    1 for row in rows if bool(row.get("target_contact_acquired"))
                ),
                "seed_replay_contact_ever_trial_count": sum(
                    1 for row in rows if bool(row.get("contact_ever_acquired"))
                ),
                "seed_replay_approach_final_current_contact_trial_count": sum(
                    1 for row in rows if bool(row.get("approach_final_current_contact"))
                ),
                "seed_replay_validated_trial_count": sum(1 for row in rows if bool(row.get("valid_seed"))),
                "seed_replay_tracking_timeout_count": sum(1 for row in rows if bool(row.get("tracking_timeout"))),
                "seed_replay_validated": bool(best.get("valid_seed")),
                "seed_replay_best_seed": best.get("seed_name", ""),
                "seed_replay_best_offset_xyz_m": best.get("offset", []),
                "seed_replay_best_rpy_deg": best.get("rpy", []),
                "seed_replay_best_final_pos_error_m": best.get("final_pos_error_m", ""),
                "seed_replay_tracking_wait_step_count": sum(
                    int(row.get("tracking_wait_step_count", 0) or 0) for row in rows
                ),
                "seed_replay_max_ctrl_to_palm_error_m": max(
                    [0.0, *[float(row.get("max_ctrl_to_palm_error_m", 0.0) or 0.0) for row in rows]]
                ),
                "seed_replay_best_progress_seed": progress_best.get("seed_name", ""),
                "seed_replay_best_progress_min_pos_error_m": progress_best.get("min_pos_error_m", ""),
                "seed_replay_best_progress_final_pos_error_m": progress_best.get("final_pos_error_m", ""),
                "seed_replay_best_progress_final_object_local_pos": progress_best.get("final_object_local_pos", []),
                "seed_replay_best_progress_final_palm_local_pos": progress_best.get("final_palm_local_pos", []),
                "seed_replay_best_progress_final_finger3_tip_local_pos": progress_best.get(
                    "final_finger3_tip_local_pos", []
                ),
                "seed_replay_best_progress_final_finger4_tip_local_pos": progress_best.get(
                    "final_finger4_tip_local_pos", []
                ),
                "seed_replay_best_progress_final_finger3_tip_to_object_xyz_m": progress_best.get(
                    "final_finger3_tip_to_object_xyz_m", []
                ),
                "seed_replay_best_progress_final_finger4_tip_to_object_xyz_m": progress_best.get(
                    "final_finger4_tip_to_object_xyz_m", []
                ),
                "seed_replay_best_progress_fingertip_midpoint_to_object_xyz_m": progress_best.get(
                    "final_fingertip_midpoint_to_object_xyz_m", []
                ),
                "seed_replay_best_progress_segment_projection_t_raw": progress_best.get(
                    "screw1_projection_parameter_on_tip_segment", ""
                ),
                "seed_replay_best_progress_segment_perpendicular_m": progress_best.get(
                    "screw1_to_tip_segment_perpendicular_m", ""
                ),
                "seed_replay_best_progress_acquisition_mode": progress_best.get("acquisition_mode", ""),
                "last_candidate": rows[-1] if rows else {},
            },
        )
    except Exception:
        return


def _write_seed_live_progress(cfg: Screw1GraspBaselineV2Config, payload: dict[str, Any]) -> None:
    try:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "seed_replay_live_progress.json", _plain(payload))
    except Exception:
        return


def _load_seed_replay_progress_rows(cfg: Screw1GraspBaselineV2Config) -> list[dict[str, Any]]:
    path = Path(cfg.output_dir) / "seed_replay_progress.csv"
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [
                {key: _coerce_seed_replay_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    except Exception:
        return []


def _coerce_seed_replay_csv_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "None":
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return value
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        return value


def _seed_contact_hold(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
    seed_name: str,
    repeat: int,
) -> dict[str, Any]:
    contacts = 0
    f3_peak = 0.0
    f4_peak = 0.0
    termination = "seed_contact_hold_completed"
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    for step in range(int(cfg.stable_contact_steps)):
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={"seed_name": seed_name, "seed_repeat": repeat},
        )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        contacts += 1 if max(f3, f4) >= cfg.contact_threshold_n else 0
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            termination = "hard_force_abort_during_seed_hold"
            break
        if guard == "retreat":
            termination = "soft_force_exceeded_during_seed_hold"
            break
        if _distance(start_obj, state["object_local_pos"]) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_during_seed_hold"
            break
    duty = contacts / max(1, int(cfg.stable_contact_steps))
    final_current_contact = _target_contact_acquired(state, cfg)
    sustained = bool(
        contacts >= int(math.ceil(cfg.stable_contact_steps * cfg.stable_contact_duty_ratio))
        and final_current_contact
        and termination == "seed_contact_hold_completed"
    )
    return {
        "seed_contact_sustained": sustained,
        "target_contact_duty_ratio": duty,
        "final_current_contact": final_current_contact,
        "termination_reason": termination,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
        "object_displacement_m": _distance(start_obj, state["object_local_pos"]),
    }


def _execute_acquisition_plan(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    phase: str,
    trial_id: int | str,
) -> dict[str, Any]:
    stages = _deserialize_acquisition_stages(plan)
    if not stages:
        return {
            "termination_reason": "acquisition_plan_has_no_staged_waypoints",
            "reached_pose": False,
            "final_state": read_state(base, env_index, cfg.part, cfg.contact_threshold_n),
        }
    final_target = list(plan.get("final_target_pos_xyz", []) or stages[-1][1])
    final_quat = _quat_normalize_wxyz(plan.get("final_target_quat_wxyz", []) or stages[-1][2])
    bounds = dict(plan.get("workspace_override_bounds", {}) or {})
    corridor = {
        "workspace_override_active": bool(bounds.get("lower_xyz") and bounds.get("upper_xyz")),
        "workspace_override_lower_xyz": bounds.get("lower_xyz", []),
        "workspace_override_upper_xyz": bounds.get("upper_xyz", []),
    }
    workspace_override_applied = False
    if corridor["workspace_override_active"]:
        _set_v2_workspace_override_for_corridor(base, env_index, corridor, cfg)
        workspace_override_applied = bool(corridor.get("workspace_override_applied"))
    try:
        return _run_servo_stage_sequence_guarded(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=phase,
            stages=stages,
            final_target_pos=final_target,
            final_target_quat=final_quat,
            extra={
                "trial_id": trial_id,
                "acquisition_plan_id": plan.get("plan_id", ""),
                "acquisition_mode": plan.get("acquisition_mode", ""),
                "acquisition_plan_replay": True,
                "acquisition_workspace_override_applied": workspace_override_applied,
            },
        )
    finally:
        if workspace_override_applied:
            _clear_v2_workspace_override(base, env_index)


def _caging_finger_contact_acquisition(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
) -> dict[str, Any]:
    initial_targets = _hand_audit_snapshot(base, env_index).get("target", [])
    if not initial_targets:
        return {
            "termination_reason": "hand_target_snapshot_unavailable",
            "stable_contact": False,
            "finger3_force_peak_n": 0.0,
            "finger4_force_peak_n": 0.0,
        }
    frozen: set[str] = set()
    finger_effects, probe_outcome = _measure_caging_finger_tip_effects(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        phase=f"{phase}_tip_effect_probe",
        initial_targets=list(initial_targets),
        start_obj=start_obj,
    )
    frozen.update(str(finger) for finger in list(probe_outcome.get("contact_fingers", []) or []))
    f3_peak = float(probe_outcome.get("finger3_force_peak_n", 0.0) or 0.0)
    f4_peak = float(probe_outcome.get("finger4_force_peak_n", 0.0) or 0.0)
    state = dict(probe_outcome.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
    probe_terminal = bool(probe_outcome.get("terminal"))
    termination = (
        str(probe_outcome.get("termination_reason") or "caging_probe_terminal")
        if probe_terminal
        else "caging_finger_acquisition_max_steps"
    )
    non_improving_steps = {"finger3": 0, "finger4": 0}
    blocked_updates: dict[str, set[tuple[int, int]]] = {"finger3": set(), "finger4": set()}
    last_updates_by_finger: dict[str, dict[str, Any]] = {}
    reprobe_required = False
    for step in range(0 if probe_terminal else int(cfg.finger_acquisition_max_steps)):
        if step > 0:
            interval = int(cfg.finger_acquisition_reprobe_interval_steps)
            if reprobe_required or (interval > 0 and step % interval == 0):
                current_targets = _hand_audit_snapshot(base, env_index).get("target", [])
                if current_targets:
                    finger_effects, probe_outcome = _measure_caging_finger_tip_effects(
                        env,
                        base,
                        env_index,
                        cfg,
                        alignment,
                        trace_rows,
                        phase=f"{phase}_tip_effect_reprobe",
                        initial_targets=list(current_targets),
                        start_obj=start_obj,
                    )
                    state = dict(
                        probe_outcome.get("final_state")
                        or read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
                    )
                    frozen.update(str(finger) for finger in list(probe_outcome.get("contact_fingers", []) or []))
                    f3_peak = max(f3_peak, float(probe_outcome.get("finger3_force_peak_n", 0.0) or 0.0))
                    f4_peak = max(f4_peak, float(probe_outcome.get("finger4_force_peak_n", 0.0) or 0.0))
                    if bool(probe_outcome.get("terminal")):
                        termination = str(probe_outcome.get("termination_reason") or "caging_probe_terminal")
                        break
                    reprobe_required = False
                    non_improving_steps = {"finger3": 0, "finger4": 0}
            if bool(probe_outcome.get("terminal")):
                break
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        if f3 >= cfg.contact_threshold_n:
            frozen.add("finger3")
        if f4 >= cfg.contact_threshold_n:
            frozen.add("finger4")
        if "finger3" in frozen and "finger4" in frozen:
            termination = "two_finger_contact_acquired_from_caging"
            break
        if _distance(start_obj, state["object_local_pos"]) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_during_caging_finger_acquisition"
            break
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            termination = "hard_force_abort_during_caging_finger_acquisition"
            _release_finger_step(env, base, env_index, cfg, alignment, trace_rows, f"{phase}_release", "finger3" if f3 >= f4 else "finger4")
            break
        if guard == "retreat":
            release = "finger3" if f3 >= f4 else "finger4"
            state = _release_finger_step(env, base, env_index, cfg, alignment, trace_rows, f"{phase}_soft_release", release)
            frozen.discard(release)
            termination = "soft_overforce_release_during_caging_finger_acquisition"
            continue
        active_fingers = [finger for finger in ("finger3", "finger4") if finger not in frozen]
        action_updates = _guided_fingertip_action_updates(
            state,
            finger_effects,
            active_fingers,
            cfg,
            blocked_updates=blocked_updates,
        )
        if not action_updates:
            termination = "no_guided_fingertip_action_reduces_residual"
            break
        action = _zero_action(env, base)
        for local, delta in action_updates:
            action[env_index, 6 + int(local)] = float(delta)
        before_distances = {
            finger: _distance(state["object_local_pos"], state[f"{finger}_tip_local_pos"])
            for finger in active_fingers
        }
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "caging_state": "BOTH_ADVANCE" if not frozen else "FIRST_CONTACT_HOLD",
                "frozen_fingers": sorted(frozen),
                "active_fingers": active_fingers,
                "guided_action_updates": list(action_updates),
                "finger3_tip_to_object_xyz_m": _sub_vec(state["object_local_pos"], state["finger3_tip_local_pos"]),
                "finger4_tip_to_object_xyz_m": _sub_vec(state["object_local_pos"], state["finger4_tip_local_pos"]),
                "finger3_tip_object_distance_m": _distance(state["object_local_pos"], state["finger3_tip_local_pos"]),
                "finger4_tip_object_distance_m": _distance(state["object_local_pos"], state["finger4_tip_local_pos"]),
            },
        )
        updated_locals = {int(local) for local, _ in action_updates}
        updates_by_finger: dict[str, list[tuple[int, float]]] = {"finger3": [], "finger4": []}
        for finger in active_fingers:
            finger_locals = {int(col) - 6 for col in _finger_action_columns(base, finger)}
            updates_by_finger[finger] = [
                (int(local), float(delta))
                for local, delta in action_updates
                if int(local) in finger_locals
            ]
            if updates_by_finger[finger]:
                local, delta = updates_by_finger[finger][0]
                last_updates_by_finger[finger] = {
                    "action_column": 6 + int(local),
                    "advance_sign": 1.0 if float(delta) >= 0.0 else -1.0,
                    "source": "caging_first_contact_action",
                }
        actual_improvements: dict[str, float] = {}
        for finger in active_fingers:
            finger_locals = {int(col) - 6 for col in _finger_action_columns(base, finger)}
            if not updated_locals.intersection(finger_locals):
                continue
            after_distance = _distance(state["object_local_pos"], state[f"{finger}_tip_local_pos"])
            improvement = float(before_distances[finger]) - float(after_distance)
            actual_improvements[finger] = improvement
            if improvement > max(1.0e-7, 0.1 * float(cfg.finger_acquisition_min_tip_effect_m)):
                non_improving_steps[finger] = 0
            else:
                non_improving_steps[finger] += 1
            if non_improving_steps[finger] >= max(1, int(cfg.finger_acquisition_non_improving_reprobe_steps)):
                for local, delta in updates_by_finger[finger]:
                    blocked_updates[finger].add((int(local), 1 if float(delta) > 0.0 else -1))
                reprobe_required = True
        if trace_rows:
            hand_snapshot = _hand_audit_snapshot(base, env_index)
            lower = _hand_pose_list(base, "dex_hand_joint_lower_limits")
            upper = _hand_pose_list(base, "dex_hand_joint_upper_limits")
            targets = list(hand_snapshot.get("target", []) or [])
            actual = list(hand_snapshot.get("actual", []) or [])
            trace_rows[-1].update(
                {
                    "caging_actual_distance_improvement_m": actual_improvements,
                    "caging_non_improving_steps": dict(non_improving_steps),
                    "caging_reprobe_required": bool(reprobe_required),
                    "caging_blocked_updates": {
                        finger: [[local, sign] for local, sign in sorted(values)]
                        for finger, values in blocked_updates.items()
                    },
                    "hand_joint_target": targets,
                    "hand_joint_actual": actual,
                    "hand_joint_target_lower_margin": [
                        float(targets[i]) - float(lower[i])
                        for i in range(min(len(targets), len(lower)))
                    ],
                    "hand_joint_target_upper_margin": [
                        float(upper[i]) - float(targets[i])
                        for i in range(min(len(targets), len(upper)))
                    ],
                }
            )
        if step % 25 == 0 or reprobe_required:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "caging_finger_acquisition_progress",
                    "step": step,
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "finger3_tip_object_distance_m": _distance(
                        state["object_local_pos"], state["finger3_tip_local_pos"]
                    ),
                    "finger4_tip_object_distance_m": _distance(
                        state["object_local_pos"], state["finger4_tip_local_pos"]
                    ),
                    "object_displacement_m": _distance(start_obj, state["object_local_pos"]),
                    "blocked_updates": {
                        finger: [[local, sign] for local, sign in sorted(values)]
                        for finger, values in blocked_updates.items()
                    },
                    "reprobe_required": bool(reprobe_required),
                },
            )
    return {
        "termination_reason": termination,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
        "final_state": state,
        "anchor_finger": "both" if "finger3" in frozen and "finger4" in frozen else next(iter(frozen), ""),
        "selected_finger_actions": last_updates_by_finger,
    }


def _measure_caging_finger_tip_effects(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    initial_targets: list[float],
    start_obj: list[float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    effects: dict[str, list[dict[str, Any]]] = {"finger3": [], "finger4": []}
    outcome: dict[str, Any] = {
        "termination_reason": "caging_probe_effects_measured",
        "terminal": False,
        "contact_fingers": [],
        "finger3_force_peak_n": 0.0,
        "finger4_force_peak_n": 0.0,
    }
    probe = float(cfg.finger_acquisition_probe_action_value)
    min_effect = float(cfg.finger_acquisition_min_tip_effect_m)
    settle_steps = max(1, int(cfg.finger_acquisition_probe_settle_steps))
    restore_settle_steps = max(1, int(cfg.finger_acquisition_probe_restore_settle_steps))
    for finger in ("finger3", "finger4"):
        active_columns = _finger_action_columns(base, finger)
        active_locals = [int(col) - 6 for col in active_columns]
        for col in active_columns:
            local = int(col) - 6
            if local < 0:
                continue
            before = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
            before_tip = list(before.get(f"{finger}_tip_local_pos", []) or [])
            if len(before_tip) < 3:
                continue
            action = _zero_action(env, base)
            for sign in (1.0, -1.0):
                before = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
                before_tip = list(before.get(f"{finger}_tip_local_pos", []) or [])
                if len(before_tip) < 3:
                    continue
                action = _zero_action(env, base)
                probe_delta = float(sign) * probe
                action[env_index, int(col)] = probe_delta
                after = _step_direct_action(
                    env,
                    base,
                    env_index,
                    cfg,
                    action,
                    phase=phase,
                    step=len(effects[finger]),
                    trace_rows=trace_rows,
                    alignment=alignment,
                    extra={
                        "probe_finger": finger,
                        "probe_action_column": int(col),
                        "probe_local_hand_index": int(local),
                        "probe_delta": probe_delta,
                    },
                )
                for settle_step in range(settle_steps):
                    after = _step_direct_action(
                        env,
                        base,
                        env_index,
                        cfg,
                        _zero_action(env, base),
                        phase=f"{phase}_settle",
                        step=settle_step,
                        trace_rows=trace_rows,
                        alignment=alignment,
                        extra={
                            "probe_finger": finger,
                            "probe_action_column": int(col),
                            "probe_local_hand_index": int(local),
                            "probe_delta": probe_delta,
                            "probe_settle": True,
                        },
                    )
                    f3 = float(after.get("finger3_target_filtered_force_n", 0.0) or 0.0)
                    f4 = float(after.get("finger4_target_filtered_force_n", 0.0) or 0.0)
                    outcome["finger3_force_peak_n"] = max(float(outcome["finger3_force_peak_n"]), f3)
                    outcome["finger4_force_peak_n"] = max(float(outcome["finger4_force_peak_n"]), f4)
                    guard = _force_guard_decision(f3, f4, cfg)
                    contact_fingers = [
                        name
                        for name, force in (("finger3", f3), ("finger4", f4))
                        if force >= float(cfg.contact_threshold_n)
                    ]
                    if contact_fingers and guard == "hold":
                        outcome.update(
                            {
                                "termination_reason": "safe_target_contact_acquired_during_caging_probe",
                                "contact_fingers": contact_fingers,
                                "final_state": after,
                            }
                        )
                        return effects, outcome
                    if guard in {"retreat", "abort"}:
                        release_finger = "finger3" if f3 >= f4 else "finger4"
                        released = _release_finger_step(
                            env,
                            base,
                            env_index,
                            cfg,
                            alignment,
                            trace_rows,
                            f"{phase}_{guard}_release",
                            release_finger,
                        )
                        outcome.update(
                            {
                                "termination_reason": f"{guard}_force_during_caging_probe",
                                "terminal": True,
                                "hard_abort": guard == "abort",
                                "final_state": released,
                            }
                        )
                        return effects, outcome
                    if _distance(start_obj, after["object_local_pos"]) > float(
                        cfg.stable_preclose_object_motion_limit_m
                    ):
                        outcome.update(
                            {
                                "termination_reason": "object_displacement_during_caging_probe",
                                "terminal": True,
                                "final_state": after,
                            }
                        )
                        return effects, outcome
                after_tip = list(after.get(f"{finger}_tip_local_pos", []) or [])
                delta = _sub_vec(after_tip, before_tip) if len(after_tip) >= 3 else [0.0, 0.0, 0.0]
                before_obj = list(before.get("object_local_pos", []) or [])
                after_obj = list(after.get("object_local_pos", []) or [])
                before_distance = _distance(before_obj, before_tip) if len(before_obj) >= 3 else math.inf
                after_distance = _distance(after_obj, after_tip) if len(after_obj) >= 3 and len(after_tip) >= 3 else math.inf
                distance_improvement = float(before_distance) - float(after_distance)
                if trace_rows:
                    trace_rows[-1].update(
                        {
                            "probe_before_distance_m": before_distance if math.isfinite(before_distance) else "",
                            "probe_after_distance_m": after_distance if math.isfinite(after_distance) else "",
                            "probe_distance_improvement_m": distance_improvement,
                            "probe_tip_delta_xyz_m": delta,
                            "probe_tip_delta_norm_m": _norm(delta),
                        }
                    )
                if (
                    _norm(delta) >= min_effect
                    and distance_improvement > max(1.0e-7, 0.1 * min_effect)
                ):
                    effects[finger].append(
                        {
                            "local": int(local),
                            "action_column": int(col),
                            "probe_action_delta": probe_delta,
                            "tip_delta_xyz_m": delta,
                            "tip_delta_norm_m": _norm(delta),
                            "probe_before_distance_m": before_distance,
                            "probe_after_distance_m": after_distance,
                            "probe_distance_improvement_m": distance_improvement,
                        }
                    )
                _restore_hand_targets(
                    env,
                    base,
                    env_index,
                    cfg,
                    alignment,
                    trace_rows,
                    phase=f"{phase}_restore",
                    desired_targets=initial_targets,
                    active_locals=active_locals,
                    max_step=probe,
                    extra={
                        "probe_finger": finger,
                        "probe_action_column": int(col),
                        "probe_local_hand_index": int(local),
                        "probe_restore": True,
                    },
                )
                for restore_step in range(restore_settle_steps):
                    _step_direct_action(
                        env,
                        base,
                        env_index,
                        cfg,
                        _zero_action(env, base),
                        phase=f"{phase}_restore_settle",
                        step=restore_step,
                        trace_rows=trace_rows,
                        alignment=alignment,
                        extra={
                            "probe_finger": finger,
                            "probe_action_column": int(col),
                            "probe_local_hand_index": int(local),
                            "probe_delta": probe_delta,
                            "guided_probe_restore_settle": True,
                        },
                    )
    outcome["final_state"] = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    return effects, outcome


def _plan_projection_parameter(plan: dict[str, Any]) -> float | None:
    sources: list[dict[str, Any]] = [plan]
    evidence = plan.get("valid_trial_evidence", [])
    if isinstance(evidence, list):
        sources.extend([item for item in evidence if isinstance(item, dict)])
    for source in sources:
        value = source.get("screw1_projection_parameter_on_tip_segment")
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _endpoint_caging_anchor_finger(
    plan: dict[str, Any],
    state: dict[str, Any],
    cfg: Screw1GraspBaselineV2Config,
) -> str:
    t_raw = _plan_projection_parameter(plan)
    margin = float(cfg.acquisition_endpoint_projection_margin)
    if t_raw is not None:
        if t_raw <= margin:
            return "finger3"
        if t_raw >= 1.0 - margin:
            return "finger4"
        return ""
    obj = list(state.get("object_local_pos", []) or [])
    p3 = list(state.get("finger3_tip_local_pos", []) or [])
    p4 = list(state.get("finger4_tip_local_pos", []) or [])
    if len(obj) >= 3 and len(p3) >= 3 and len(p4) >= 3:
        return "finger3" if _distance(obj, p3) <= _distance(obj, p4) else "finger4"
    return ""


def _endpoint_caging_anchor_wrist_approach(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
    anchor_finger: str,
) -> dict[str, Any]:
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    f3_peak = 0.0
    f4_peak = 0.0
    last_move = [0.0, 0.0, 0.0]
    termination = "endpoint_anchor_wrist_approach_max_steps"
    for step in range(int(cfg.anchor_residual_approach_max_steps)):
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        anchor_force = f3 if anchor_finger == "finger3" else f4
        if anchor_force >= float(cfg.contact_threshold_n):
            termination = "endpoint_anchor_contact_acquired"
            break
        if _distance(start_obj, state["object_local_pos"]) > float(cfg.stable_preclose_object_motion_limit_m):
            termination = "object_displacement_during_endpoint_anchor_wrist_approach"
            break
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            termination = "hard_force_abort_during_endpoint_anchor_wrist_approach"
            retreat = _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, last_move)
            state = dict(retreat.get("final_state") or state)
            break
        if guard == "retreat":
            termination = "soft_force_retreat_during_endpoint_anchor_wrist_approach"
            retreat = _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, last_move)
            state = dict(retreat.get("final_state") or state)
            break
        obj = list(state.get("object_local_pos", []) or [])
        tip = list(state.get(f"{anchor_finger}_tip_local_pos", []) or [])
        if len(obj) < 3 or len(tip) < 3:
            termination = "endpoint_anchor_missing_tip_or_object_pose"
            break
        residual = _sub_vec(obj, tip)
        residual_norm = _norm(residual)
        if int(step) == 0 or int(step) % 10 == 0:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "endpoint_anchor_wrist_step",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "anchor_residual_norm_m": residual_norm,
                    "anchor_residual_xyz_m": residual,
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "force_guard_decision": guard,
                    "object_displacement_m": _distance(start_obj, state["object_local_pos"]),
                    "table_barrier_delta_z_m": state.get("table_barrier_delta_z_m", 0.0),
                    "workspace_clamp_delta_m": state.get("workspace_clamp_delta_m", 0.0),
                },
            )
        if residual_norm <= 1.0e-5:
            termination = "endpoint_anchor_residual_exhausted_without_contact"
            break
        step_len = min(float(cfg.anchor_residual_approach_step_m), residual_norm)
        move = [float(value) / residual_norm * step_len for value in residual]
        last_move = list(move)
        action = _zero_action(env, base)
        _set_wrist_delta_action(base, action, env_index, move, [0.0, 0.0, 0.0])
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "anchor_finger": anchor_finger,
                "endpoint_anchor_residual_xyz_m": residual,
                "endpoint_anchor_residual_norm_m": residual_norm,
                "endpoint_anchor_wrist_move_xyz_m": move,
            },
        )
    final_f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    final_f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    final_anchor_force = final_f3 if anchor_finger == "finger3" else final_f4
    return {
        "termination_reason": termination,
        "finger3_force_peak_n": max(f3_peak, final_f3),
        "finger4_force_peak_n": max(f4_peak, final_f4),
        "final_state": state,
        "anchor_finger": anchor_finger if final_anchor_force >= cfg.contact_threshold_n else "",
    }


def _endpoint_anchor_tip_distance(state: dict[str, Any], anchor_finger: str) -> tuple[float, list[float]]:
    obj = list(state.get("object_local_pos", []) or [])
    tip = list(state.get(f"{anchor_finger}_tip_local_pos", []) or [])
    if len(obj) < 3 or len(tip) < 3:
        return float("inf"), []
    residual = _sub_vec(obj, tip)
    return _norm(residual), residual


def _restore_wrist_rotation_probe(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    delta_rot: list[float],
    anchor_finger: str,
    probe_index: int,
) -> dict[str, Any]:
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    restore_rot = [-float(value) for value in delta_rot]
    for step in range(max(1, int(cfg.anchor_orientation_probe_restore_steps))):
        action = _zero_action(env, base)
        rot = restore_rot if step == 0 else [0.0, 0.0, 0.0]
        _set_wrist_delta_action(base, action, env_index, [0.0, 0.0, 0.0], rot)
        _configure_v2_control(base, env_index, mode="anchored_delta")
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "anchor_finger": anchor_finger,
                "endpoint_orientation_probe_index": int(probe_index),
                "endpoint_orientation_probe_restore": True,
                "endpoint_orientation_restore_rot_axis_angle": restore_rot,
            },
        )
    return state


def _probe_best_endpoint_orientation_action(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
    anchor_finger: str,
    start_obj: list[float],
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    base_dist, base_residual = _endpoint_anchor_tip_distance(initial_state, anchor_finger)
    if not math.isfinite(base_dist):
        return {"termination_reason": "endpoint_anchor_orientation_probe_missing_inputs"}
    probe_rad = math.radians(float(cfg.anchor_orientation_probe_deg))
    min_improvement = float(cfg.anchor_orientation_min_improvement_m)
    best: dict[str, Any] = {}
    probe_index = 0
    axis_names = ("roll", "pitch", "yaw")
    for axis in range(3):
        for sign in (1.0, -1.0):
            before = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
            before_dist, before_residual = _endpoint_anchor_tip_distance(before, anchor_finger)
            if not math.isfinite(before_dist):
                continue
            delta_rot = [0.0, 0.0, 0.0]
            delta_rot[axis] = float(sign) * probe_rad
            action = _zero_action(env, base)
            _set_wrist_delta_action(base, action, env_index, [0.0, 0.0, 0.0], delta_rot)
            _configure_v2_control(base, env_index, mode="anchored_delta")
            after = _step_direct_action(
                env,
                base,
                env_index,
                cfg,
                action,
                phase=phase,
                step=probe_index,
                trace_rows=trace_rows,
                alignment=alignment,
                extra={
                    "anchor_finger": anchor_finger,
                    "endpoint_orientation_probe_parent_step": int(step),
                    "endpoint_orientation_probe_index": int(probe_index),
                    "endpoint_orientation_probe_axis": axis_names[axis],
                    "endpoint_orientation_probe_sign": float(sign),
                    "endpoint_orientation_probe_delta_deg": math.degrees(delta_rot[axis]),
                    "endpoint_orientation_probe_base_distance_m": base_dist,
                    "endpoint_orientation_probe_before_distance_m": before_dist,
                    "endpoint_orientation_probe_base_residual_xyz_m": base_residual,
                    "endpoint_orientation_probe_before_residual_xyz_m": before_residual,
                },
            )
            for hold_step in range(max(0, int(cfg.anchor_orientation_probe_hold_steps) - 1)):
                hold = _zero_action(env, base)
                _configure_v2_control(base, env_index, mode="anchored_delta")
                after = _step_direct_action(
                    env,
                    base,
                    env_index,
                    cfg,
                    hold,
                    phase=f"{phase}_hold",
                    step=probe_index * max(1, int(cfg.anchor_orientation_probe_hold_steps)) + hold_step,
                    trace_rows=trace_rows,
                    alignment=alignment,
                    extra={
                        "anchor_finger": anchor_finger,
                        "endpoint_orientation_probe_parent_step": int(step),
                        "endpoint_orientation_probe_index": int(probe_index),
                        "endpoint_orientation_probe_axis": axis_names[axis],
                        "endpoint_orientation_probe_sign": float(sign),
                        "endpoint_orientation_probe_hold": True,
                    },
                )
            f3 = float(after.get("finger3_target_filtered_force_n", 0.0) or 0.0)
            f4 = float(after.get("finger4_target_filtered_force_n", 0.0) or 0.0)
            guard = _force_guard_decision(f3, f4, cfg)
            contact = (f3 if anchor_finger == "finger3" else f4) >= float(cfg.contact_threshold_n)
            if contact:
                if guard == "abort":
                    retreat = _retreat(
                        env,
                        base,
                        env_index,
                        cfg,
                        alignment,
                        trace_rows,
                        phase,
                        [0.0, 0.0, 0.0],
                        delta_rot,
                    )
                    return {
                        "termination_reason": "hard_force_abort_during_endpoint_anchor_orientation_probe",
                        "hard_abort": True,
                        "final_state": dict(retreat.get("final_state") or after),
                    }
                if guard == "retreat":
                    retreat = _retreat(
                        env,
                        base,
                        env_index,
                        cfg,
                        alignment,
                        trace_rows,
                        phase,
                        [0.0, 0.0, 0.0],
                        delta_rot,
                    )
                    return {
                        "termination_reason": "soft_force_retreat_during_endpoint_anchor_orientation_probe",
                        "final_state": dict(retreat.get("final_state") or after),
                    }
                return {
                    "termination_reason": "endpoint_anchor_contact_acquired_by_orientation_probe",
                    "contact_acquired": True,
                    "final_state": after,
                    "chosen_rot_axis": axis_names[axis],
                    "chosen_rot_axis_index": int(axis),
                    "chosen_rot_delta_axis_angle": delta_rot,
                    "chosen_delta_deg": math.degrees(delta_rot[axis]),
                }
            if _distance(start_obj, after["object_local_pos"]) > float(cfg.stable_preclose_object_motion_limit_m):
                return {
                    "termination_reason": "object_displacement_during_endpoint_anchor_orientation_probe",
                    "final_state": after,
                }
            after_dist, after_residual = _endpoint_anchor_tip_distance(after, anchor_finger)
            improvement = float(before_dist) - float(after_dist) if math.isfinite(after_dist) else 0.0
            if improvement > min_improvement and improvement > float(best.get("improvement_m", 0.0) or 0.0):
                best = {
                    "rot_axis": axis_names[axis],
                    "rot_axis_index": int(axis),
                    "delta_axis_angle": delta_rot,
                    "delta_deg": math.degrees(delta_rot[axis]),
                    "improvement_m": float(improvement),
                    "before_distance_m": float(before_dist),
                    "after_distance_m": float(after_dist),
                    "after_residual_xyz_m": after_residual,
                }
            restored = _restore_wrist_rotation_probe(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase}_restore",
                delta_rot=delta_rot,
                anchor_finger=anchor_finger,
                probe_index=probe_index,
            )
            f3_restore = float(restored.get("finger3_target_filtered_force_n", 0.0) or 0.0)
            f4_restore = float(restored.get("finger4_target_filtered_force_n", 0.0) or 0.0)
            if _force_guard_decision(f3_restore, f4_restore, cfg) == "abort":
                return {
                    "termination_reason": "hard_force_abort_during_endpoint_anchor_orientation_probe_restore",
                    "hard_abort": True,
                    "final_state": restored,
                }
            probe_index += 1
    if not best:
        return {"termination_reason": "no_endpoint_orientation_probe_reduces_residual"}
    return {"termination_reason": "endpoint_orientation_probe_selected", **best}


def _endpoint_caging_anchor_orientation_refinement(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
    anchor_finger: str,
) -> dict[str, Any]:
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    f3_peak = 0.0
    f4_peak = 0.0
    termination = "endpoint_anchor_orientation_refinement_max_steps"
    selected_probe: dict[str, Any] = {}
    selected_probe_age = 0
    total_rot = [0.0, 0.0, 0.0]
    for step in range(int(cfg.anchor_orientation_guided_max_steps)):
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        residual_norm, residual = _endpoint_anchor_tip_distance(state, anchor_finger)
        if int(step) == 0 or int(step) % max(1, int(cfg.anchor_orientation_reprobe_interval_steps)) == 0:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "endpoint_anchor_orientation_step",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "anchor_residual_norm_m": residual_norm if math.isfinite(residual_norm) else "",
                    "anchor_residual_xyz_m": residual,
                    "selected_rot_axis": selected_probe.get("rot_axis", ""),
                    "selected_rot_delta_deg": selected_probe.get("delta_deg", ""),
                    "selected_probe_age": int(selected_probe_age),
                    "total_orientation_refinement_deg": math.degrees(_norm(total_rot)),
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "force_guard_decision": _force_guard_decision(f3, f4, cfg),
                    "object_displacement_m": _distance(start_obj, state["object_local_pos"]),
                    "table_barrier_delta_z_m": state.get("table_barrier_delta_z_m", 0.0),
                    "workspace_clamp_delta_m": state.get("workspace_clamp_delta_m", 0.0),
                },
            )
        anchor_force = f3 if anchor_finger == "finger3" else f4
        if anchor_force >= float(cfg.contact_threshold_n):
            termination = "endpoint_anchor_contact_acquired_by_orientation_refinement"
            break
        if _distance(start_obj, state["object_local_pos"]) > float(cfg.stable_preclose_object_motion_limit_m):
            termination = "object_displacement_during_endpoint_anchor_orientation_refinement"
            break
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            state = dict(
                _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, [0.0, 0.0, 0.0], total_rot).get(
                    "final_state"
                )
                or state
            )
            termination = "hard_force_abort_during_endpoint_anchor_orientation_refinement"
            break
        if guard == "retreat":
            state = dict(
                _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, [0.0, 0.0, 0.0], total_rot).get(
                    "final_state"
                )
                or state
            )
            termination = "soft_force_retreat_during_endpoint_anchor_orientation_refinement"
            break
        if math.degrees(_norm(total_rot)) >= float(cfg.anchor_orientation_max_total_deg):
            termination = "endpoint_anchor_orientation_refinement_rotation_limit"
            break
        reprobe_due = bool(
            not selected_probe
            or selected_probe_age >= max(1, int(cfg.anchor_orientation_reprobe_interval_steps))
        )
        if reprobe_due:
            probe = _probe_best_endpoint_orientation_action(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase}_probe",
                step=step,
                anchor_finger=anchor_finger,
                start_obj=start_obj,
                initial_state=state,
            )
            state = dict(probe.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
            f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
            f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
            f3_peak = max(f3_peak, f3)
            f4_peak = max(f4_peak, f4)
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "endpoint_anchor_orientation_probe_selected",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "termination_reason": probe.get("termination_reason", ""),
                    "chosen_rot_axis": probe.get("rot_axis", probe.get("chosen_rot_axis", "")),
                    "chosen_delta_deg": probe.get("delta_deg", probe.get("chosen_delta_deg", "")),
                    "chosen_improvement_m": probe.get("improvement_m", ""),
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                },
            )
            if bool(probe.get("contact_acquired")):
                termination = probe.get("termination_reason") or "endpoint_anchor_contact_acquired_by_orientation_probe"
                break
            if bool(probe.get("hard_abort")):
                termination = probe.get("termination_reason") or "hard_force_abort_during_endpoint_anchor_orientation_probe"
                break
            if "retreat" in str(probe.get("termination_reason") or ""):
                termination = probe.get("termination_reason")
                break
            if probe.get("rot_axis_index") is None:
                termination = probe.get("termination_reason") or "no_endpoint_orientation_probe_reduces_residual"
                break
            selected_probe = dict(probe)
            selected_probe_age = 0
        before_dist, _ = _endpoint_anchor_tip_distance(state, anchor_finger)
        delta_rot = list(selected_probe.get("delta_axis_angle", []) or [])
        if len(delta_rot) < 3:
            termination = "endpoint_anchor_orientation_probe_missing_delta"
            break
        action = _zero_action(env, base)
        _set_wrist_delta_action(base, action, env_index, [0.0, 0.0, 0.0], delta_rot)
        _configure_v2_control(base, env_index, mode="anchored_delta")
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "anchor_finger": anchor_finger,
                "endpoint_orientation_guided_axis": selected_probe.get("rot_axis", ""),
                "endpoint_orientation_guided_delta_deg": selected_probe.get("delta_deg", ""),
                "endpoint_orientation_guided_improvement_m": selected_probe.get("improvement_m", ""),
                "endpoint_orientation_before_distance_m": before_dist if math.isfinite(before_dist) else "",
                "endpoint_orientation_total_rot_axis_angle_before": list(total_rot),
            },
        )
        after_dist, _ = _endpoint_anchor_tip_distance(state, anchor_finger)
        actual_improvement = float(before_dist) - float(after_dist) if math.isfinite(before_dist) and math.isfinite(after_dist) else 0.0
        total_rot = [float(total_rot[i]) + float(delta_rot[i]) for i in range(3)]
        if trace_rows:
            trace_rows[-1].update(
                {
                    "endpoint_orientation_after_distance_m": after_dist if math.isfinite(after_dist) else "",
                    "endpoint_orientation_actual_improvement_m": actual_improvement,
                    "endpoint_orientation_total_rot_axis_angle_after": list(total_rot),
                    "endpoint_orientation_total_deg_after": math.degrees(_norm(total_rot)),
                }
            )
        if actual_improvement <= max(1.0e-7, float(cfg.anchor_orientation_min_improvement_m) * 0.25):
            selected_probe = {}
            selected_probe_age = 0
        else:
            selected_probe_age += 1
    final_f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    final_f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    return {
        "termination_reason": termination,
        "finger3_force_peak_n": max(f3_peak, final_f3),
        "finger4_force_peak_n": max(f4_peak, final_f4),
        "final_state": state,
        "anchor_finger": anchor_finger
        if (final_f3 if anchor_finger == "finger3" else final_f4) >= cfg.contact_threshold_n
        else "",
        "endpoint_orientation_total_rot_axis_angle": total_rot,
        "endpoint_orientation_total_deg": math.degrees(_norm(total_rot)),
    }


def _restore_hand_targets(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    desired_targets: list[float],
    active_locals: list[int],
    max_step: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    for step in range(max(1, int(cfg.anchor_finger_probe_restore_steps))):
        action = _zero_action(env, base)
        err_peak = _set_hand_target_servo_action(base, action, env_index, desired_targets, active_locals, max_step)
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={**dict(extra or {}), "restore_target_error_peak_rad": err_peak},
        )
        if err_peak <= max(1.0e-5, float(max_step) * 0.25):
            break
    return state


def _probe_best_anchor_finger_action(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
    anchor_finger: str,
    start_obj: list[float],
    initial_state: dict[str, Any],
    initial_targets: list[float],
    probe_action_value: float | None = None,
    min_improvement_m: float | None = None,
    probe_settle_steps: int = 0,
    probe_restore_settle_steps: int = 0,
) -> dict[str, Any]:
    active_columns = _finger_action_columns(base, anchor_finger)
    active_locals = [int(col) - 6 for col in active_columns]
    object_pos = list(initial_state.get("object_local_pos", []) or [])
    tip = list(initial_state.get(f"{anchor_finger}_tip_local_pos", []) or [])
    if len(object_pos) < 3 or len(tip) < 3 or not active_columns or not initial_targets:
        return {"termination_reason": "endpoint_anchor_finger_probe_missing_inputs"}
    base_residual = _sub_vec(object_pos, tip)
    base_dist = _norm(base_residual)
    best: dict[str, Any] = {}
    probe_delta_abs = float(probe_action_value if probe_action_value is not None else cfg.anchor_finger_probe_action_value)
    min_improvement = float(min_improvement_m if min_improvement_m is not None else cfg.anchor_finger_min_improvement_m)
    probe_index = 0
    for col in active_columns:
        local = int(col) - 6
        for sign in (1.0, -1.0):
            before = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
            before_tip = list(before.get(f"{anchor_finger}_tip_local_pos", []) or [])
            before_obj = list(before.get("object_local_pos", []) or [])
            if len(before_tip) < 3 or len(before_obj) < 3:
                continue
            before_dist = _distance(before_obj, before_tip)
            action = _zero_action(env, base)
            probe_delta = float(sign) * probe_delta_abs
            action[env_index, int(col)] = probe_delta
            after = _step_direct_action(
                env,
                base,
                env_index,
                cfg,
                action,
                phase=phase,
                step=probe_index,
                trace_rows=trace_rows,
                alignment=alignment,
                extra={
                    "anchor_finger": anchor_finger,
                    "anchor_probe_parent_step": int(step),
                    "anchor_probe_action_column": int(col),
                    "anchor_probe_local_hand_index": int(local),
                    "anchor_probe_delta": probe_delta,
                    "anchor_probe_base_residual_xyz_m": base_residual,
                    "anchor_probe_base_distance_m": base_dist,
                    "anchor_probe_before_distance_m": before_dist,
                },
            )
            probe_index += 1
            for settle_index in range(max(0, int(probe_settle_steps))):
                settle_f3 = float(after.get("finger3_target_filtered_force_n", 0.0) or 0.0)
                settle_f4 = float(after.get("finger4_target_filtered_force_n", 0.0) or 0.0)
                settle_target_force = settle_f3 if anchor_finger == "finger3" else settle_f4
                settle_guard = _force_guard_decision(settle_f3, settle_f4, cfg)
                if settle_target_force >= float(cfg.contact_threshold_n) or settle_guard in {"retreat", "abort"}:
                    break
                after = _step_direct_action(
                    env,
                    base,
                    env_index,
                    cfg,
                    _zero_action(env, base),
                    phase=f"{phase}_settle",
                    step=settle_index,
                    trace_rows=trace_rows,
                    alignment=alignment,
                    extra={
                        "anchor_finger": anchor_finger,
                        "anchor_probe_parent_step": int(step),
                        "anchor_probe_action_column": int(col),
                        "anchor_probe_local_hand_index": int(local),
                        "anchor_probe_delta": probe_delta,
                        "anchor_probe_settle": True,
                    },
                )
            f3 = float(after.get("finger3_target_filtered_force_n", 0.0) or 0.0)
            f4 = float(after.get("finger4_target_filtered_force_n", 0.0) or 0.0)
            guard = _force_guard_decision(f3, f4, cfg)
            target_force = f3 if anchor_finger == "finger3" else f4
            contact = target_force >= float(cfg.contact_threshold_n)
            if contact:
                if guard == "abort":
                    release_finger = "finger3" if f3 >= f4 else "finger4"
                    final_state = _release_finger_step(
                        env,
                        base,
                        env_index,
                        cfg,
                        alignment,
                        trace_rows,
                        f"{phase}_hard_release",
                        release_finger,
                    )
                    return {
                        "termination_reason": "hard_force_abort_during_endpoint_anchor_finger_probe",
                        "hard_abort": True,
                        "final_state": final_state,
                    }
                if guard == "retreat":
                    release_finger = "finger3" if f3 >= f4 else "finger4"
                    final_state = _release_finger_step(
                        env,
                        base,
                        env_index,
                        cfg,
                        alignment,
                        trace_rows,
                        f"{phase}_soft_release",
                        release_finger,
                    )
                    return {
                        "termination_reason": "soft_force_release_during_endpoint_anchor_finger_probe",
                        "final_state": final_state,
                    }
                return {
                    "termination_reason": "endpoint_anchor_contact_acquired_by_finger_probe",
                    "contact_acquired": True,
                    "final_state": after,
                    "action_column": int(col),
                    "local_hand_index": int(local),
                    "delta": probe_delta,
                    "improvement_m": max(0.0, before_dist - _distance(after.get("object_local_pos", []), after.get(f"{anchor_finger}_tip_local_pos", []))),
                    "chosen_action_column": int(col),
                    "chosen_local_hand_index": int(local),
                    "chosen_delta": probe_delta,
                    "chosen_improvement_m": max(0.0, before_dist - _distance(after.get("object_local_pos", []), after.get(f"{anchor_finger}_tip_local_pos", []))),
                }
            if _distance(start_obj, after["object_local_pos"]) > float(cfg.stable_preclose_object_motion_limit_m):
                return {
                    "termination_reason": "object_displacement_during_endpoint_anchor_finger_probe",
                    "final_state": after,
                }
            after_tip = list(after.get(f"{anchor_finger}_tip_local_pos", []) or [])
            after_obj = list(after.get("object_local_pos", []) or [])
            after_dist = _distance(after_obj, after_tip) if len(after_tip) >= 3 and len(after_obj) >= 3 else before_dist
            improvement = before_dist - after_dist
            if improvement > min_improvement and improvement > float(best.get("improvement_m", 0.0) or 0.0):
                best = {
                    "action_column": int(col),
                    "local_hand_index": int(local),
                    "delta": probe_delta,
                    "improvement_m": float(improvement),
                    "before_distance_m": float(before_dist),
                    "after_distance_m": float(after_dist),
                    "tip_delta_xyz_m": _sub_vec(after_tip, before_tip) if len(after_tip) >= 3 else [],
                }
            _restore_hand_targets(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase}_restore",
                desired_targets=initial_targets,
                active_locals=active_locals,
                max_step=probe_delta_abs,
                extra={
                    "anchor_finger": anchor_finger,
                    "anchor_probe_parent_step": int(step),
                    "anchor_probe_action_column": int(col),
                    "anchor_probe_local_hand_index": int(local),
                    "anchor_probe_delta": probe_delta,
                    "anchor_probe_improvement_m": float(improvement),
                },
            )
            for restore_settle_index in range(max(0, int(probe_restore_settle_steps))):
                after = _step_direct_action(
                    env,
                    base,
                    env_index,
                    cfg,
                    _zero_action(env, base),
                    phase=f"{phase}_restore_settle",
                    step=restore_settle_index,
                    trace_rows=trace_rows,
                    alignment=alignment,
                    extra={
                        "anchor_finger": anchor_finger,
                        "anchor_probe_parent_step": int(step),
                        "anchor_probe_action_column": int(col),
                        "anchor_probe_local_hand_index": int(local),
                        "anchor_probe_delta": probe_delta,
                        "anchor_probe_restore_settle": True,
                    },
                )
    if not best:
        return {"termination_reason": "no_single_finger_anchor_probe_reduces_residual"}
    return {"termination_reason": "single_finger_anchor_probe_selected", **best}


def _endpoint_caging_anchor_finger_acquisition(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
    anchor_finger: str,
) -> dict[str, Any]:
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    f3_peak = 0.0
    f4_peak = 0.0
    termination = "endpoint_anchor_finger_acquisition_max_steps"
    selected_probe: dict[str, Any] = {}
    selected_probe_age = 0
    for step in range(int(cfg.anchor_finger_guided_max_steps)):
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        object_pos = list(state.get("object_local_pos", []) or [])
        tip_pos = list(state.get(f"{anchor_finger}_tip_local_pos", []) or [])
        residual = _sub_vec(object_pos, tip_pos) if len(object_pos) >= 3 and len(tip_pos) >= 3 else []
        residual_norm = _norm(residual)
        if int(step) == 0 or int(step) % max(1, int(cfg.anchor_finger_reprobe_interval_steps)) == 0:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "endpoint_anchor_finger_step",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "anchor_residual_norm_m": residual_norm,
                    "anchor_residual_xyz_m": residual,
                    "selected_action_column": selected_probe.get("action_column", ""),
                    "selected_action_delta": selected_probe.get("delta", ""),
                    "selected_probe_age": int(selected_probe_age),
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "force_guard_decision": _force_guard_decision(f3, f4, cfg),
                    "object_displacement_m": _distance(start_obj, state["object_local_pos"]),
                    "table_barrier_delta_z_m": state.get("table_barrier_delta_z_m", 0.0),
                    "workspace_clamp_delta_m": state.get("workspace_clamp_delta_m", 0.0),
                },
            )
        anchor_force = f3 if anchor_finger == "finger3" else f4
        if anchor_force >= float(cfg.contact_threshold_n):
            termination = "endpoint_anchor_contact_acquired_by_guided_finger"
            break
        if _distance(start_obj, state["object_local_pos"]) > float(cfg.stable_preclose_object_motion_limit_m):
            termination = "object_displacement_during_endpoint_anchor_finger_acquisition"
            break
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            release_finger = "finger3" if f3 >= f4 else "finger4"
            state = _release_finger_step(
                env, base, env_index, cfg, alignment, trace_rows, f"{phase}_hard_release", release_finger
            )
            termination = "hard_force_abort_during_endpoint_anchor_finger_acquisition"
            break
        if guard == "retreat":
            release_finger = "finger3" if f3 >= f4 else "finger4"
            state = _release_finger_step(
                env, base, env_index, cfg, alignment, trace_rows, f"{phase}_soft_release", release_finger
            )
            termination = "soft_force_release_during_endpoint_anchor_finger_acquisition"
            break
        reprobe_due = bool(
            not selected_probe
            or selected_probe_age >= max(1, int(cfg.anchor_finger_reprobe_interval_steps))
        )
        if reprobe_due:
            initial_targets = _hand_audit_snapshot(base, env_index).get("target", [])
            probe = _probe_best_anchor_finger_action(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase}_single_joint_probe",
                step=step,
                anchor_finger=anchor_finger,
                start_obj=start_obj,
                initial_state=state,
                initial_targets=list(initial_targets),
            )
            state = dict(probe.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
            f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
            f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
            f3_peak = max(f3_peak, f3)
            f4_peak = max(f4_peak, f4)
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "endpoint_anchor_finger_probe_selected",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "termination_reason": probe.get("termination_reason", ""),
                    "chosen_action_column": probe.get("action_column", ""),
                    "chosen_local_hand_index": probe.get("local_hand_index", ""),
                    "chosen_delta": probe.get("delta", ""),
                    "chosen_improvement_m": probe.get("improvement_m", ""),
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                },
            )
            if bool(probe.get("contact_acquired")):
                termination = probe.get("termination_reason") or "endpoint_anchor_contact_acquired_by_finger_probe"
                break
            if bool(probe.get("hard_abort")):
                termination = probe.get("termination_reason") or "hard_force_abort_during_endpoint_anchor_finger_probe"
                break
            if "release" in str(probe.get("termination_reason") or ""):
                termination = probe.get("termination_reason")
                break
            if probe.get("action_column") is None:
                termination = probe.get("termination_reason") or "no_single_finger_anchor_probe_reduces_residual"
                break
            selected_probe = dict(probe)
            selected_probe_age = 0
        probe = selected_probe
        action = _zero_action(env, base)
        action[env_index, int(probe["action_column"])] = float(probe["delta"])
        before_dist = _distance(state["object_local_pos"], state[f"{anchor_finger}_tip_local_pos"])
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "anchor_finger": anchor_finger,
                "anchor_guided_action_column": int(probe["action_column"]),
                "anchor_guided_local_hand_index": int(probe["local_hand_index"]),
                "anchor_guided_delta": float(probe["delta"]),
                "anchor_guided_probe_improvement_m": float(probe["improvement_m"]),
                "anchor_guided_before_distance_m": before_dist,
                "anchor_guided_probe_before_distance_m": probe.get("before_distance_m", ""),
                "anchor_guided_probe_after_distance_m": probe.get("after_distance_m", ""),
                "anchor_guided_tip_delta_xyz_m": probe.get("tip_delta_xyz_m", []),
            },
        )
        after_dist = _distance(state["object_local_pos"], state[f"{anchor_finger}_tip_local_pos"])
        actual_improvement = float(before_dist) - float(after_dist)
        if actual_improvement <= max(1.0e-7, float(cfg.anchor_finger_min_improvement_m) * 0.25):
            selected_probe = {}
            selected_probe_age = 0
        else:
            selected_probe_age += 1
    final_f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    final_f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    return {
        "termination_reason": termination,
        "finger3_force_peak_n": max(f3_peak, final_f3),
        "finger4_force_peak_n": max(f4_peak, final_f4),
        "final_state": state,
        "anchor_finger": anchor_finger
        if (final_f3 if anchor_finger == "finger3" else final_f4) >= cfg.contact_threshold_n
        else "",
    }


def _guided_fingertip_action_updates(
    state: dict[str, Any],
    finger_effects: dict[str, list[dict[str, Any]]],
    active_fingers: list[str],
    cfg: Screw1GraspBaselineV2Config,
    *,
    blocked_updates: dict[str, set[tuple[int, int]]] | None = None,
) -> list[tuple[int, float]]:
    updates: list[tuple[int, float]] = []
    max_per_finger = max(1, int(cfg.finger_acquisition_max_active_joints_per_finger))
    step = float(cfg.finger_acquisition_action_value)
    object_pos = list(state.get("object_local_pos", []) or [])
    if len(object_pos) < 3:
        return updates
    for finger in active_fingers:
        tip = list(state.get(f"{finger}_tip_local_pos", []) or [])
        if len(tip) < 3:
            continue
        residual = _sub_vec(object_pos, tip)
        current_dist = _norm(residual)
        best_by_local: dict[int, tuple[float, int, float]] = {}
        for effect in list(finger_effects.get(finger, []) or []):
            local = int(effect.get("local", -1))
            delta = list(effect.get("tip_delta_xyz_m", []) or effect.get("positive_tip_delta_xyz_m", []) or [])
            probe_delta = float(effect.get("probe_action_delta", cfg.finger_acquisition_probe_action_value) or 0.0)
            if local < 0 or len(delta) < 3:
                continue
            if abs(probe_delta) <= 1.0e-9:
                continue
            action_sign = 1 if probe_delta > 0.0 else -1
            if (local, action_sign) in (blocked_updates or {}).get(finger, set()):
                continue
            scale = step / abs(probe_delta)
            measured_improvement = float(effect.get("probe_distance_improvement_m", 0.0) or 0.0)
            if measured_improvement <= max(1.0e-7, 0.1 * float(cfg.finger_acquisition_min_tip_effect_m)):
                continue
            predicted_tip_move = [float(delta[i]) * scale for i in range(3)]
            predicted_residual = [float(residual[i]) - predicted_tip_move[i] for i in range(3)]
            predicted_dist = _norm(predicted_residual)
            improvement = current_dist - predicted_dist
            if improvement <= max(1.0e-7, 0.1 * float(cfg.finger_acquisition_min_tip_effect_m)):
                continue
            action_delta = math.copysign(step, probe_delta)
            score = min(float(improvement), measured_improvement * scale)
            candidate = (score, local, float(action_delta))
            prev = best_by_local.get(local)
            if prev is None or candidate[0] > prev[0]:
                best_by_local[local] = candidate
        scored = list(best_by_local.values())
        scored.sort(reverse=True)
        for score, local, delta in scored[:max_per_finger]:
            if score <= 0.0:
                continue
            updates.append((int(local), float(delta)))
    return updates


def _run_force_guarded_grasp_trial(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    authority_rows: list[dict[str, Any]],
    *,
    trial_id: int,
    phase_name: str,
    seed_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict((seed_hint or {}).get("validated_acquisition_plan") or {})
    if not plan:
        plan_path = str((seed_hint or {}).get("validated_acquisition_plan_json") or "").strip()
        if plan_path:
            try:
                loaded = json.loads(Path(plan_path).read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    plan = loaded
            except Exception:
                plan = {}
    if not seed_hint or not bool(seed_hint.get("seed_replay_validated")) or not plan:
        row = {
            "phase": phase_name,
            "trial_id": trial_id,
            "result_class": "KINEMATIC_GRASP_BLOCKER_PROVEN",
            "termination_reason": "no_validated_acquisition_plan_after_fresh_replay",
            "stable_contact": False,
            "close_executed": False,
            "lift_executed": False,
        }
        episodes.append(row)
        return {**row, "success_claimed": False, "object_ready": False, "grasp_solved": False}
    fresh = _fresh_episode(
        env, base, env_index, cfg, alignment, trace_rows, f"{phase_name}_{trial_id}", hand_mode="parked"
    )
    start_state = fresh["state"]
    start_obj = list(start_state["object_local_pos"])
    start_rel = _sub_vec(start_state["object_local_pos"], start_state["palm_local_pos"])
    approach = _execute_acquisition_plan(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        plan=plan,
        phase=f"{phase_name}_slow_pregrasp",
        trial_id=trial_id,
    )
    after_approach = approach["final_state"]
    if bool(approach.get("hard_abort")) or "retreat" in str(approach.get("termination_reason") or ""):
        row = {
            "phase": phase_name,
            "trial_id": trial_id,
            "result_class": "KINEMATIC_GRASP_BLOCKER_PROVEN",
            "termination_reason": approach.get("termination_reason") or "approach_retreat_or_abort",
            "stable_contact": False,
            "finger3_force_peak_n": approach.get("finger3_force_peak_n", 0.0),
            "finger4_force_peak_n": approach.get("finger4_force_peak_n", 0.0),
            "object_displacement_m": _distance(start_obj, after_approach["object_local_pos"]),
            "close_executed": False,
            "lift_executed": False,
        }
        episodes.append(row)
        return {**row, "success_claimed": False, "object_ready": False, "grasp_solved": False}
    mode = str(plan.get("acquisition_mode") or "")
    f3_anchor = float(after_approach.get("finger3_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
    f4_anchor = float(after_approach.get("finger4_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
    opposing: dict[str, Any] = {}
    endpoint_anchor = _endpoint_caging_anchor_finger(plan, after_approach, cfg)
    endpoint_anchor_residual = ""
    if endpoint_anchor:
        obj_pos = list(after_approach.get("object_local_pos", []) or [])
        tip_pos = list(after_approach.get(f"{endpoint_anchor}_tip_local_pos", []) or [])
        if len(obj_pos) >= 3 and len(tip_pos) >= 3:
            endpoint_anchor_residual = _distance(obj_pos, tip_pos)
    pregrasp_replay_reached = bool(approach.get("reached_pose"))
    safe_standoff_residual_limit = max(0.035, float(cfg.acquisition_caging_perpendicular_max_m) * 3.0)
    if (
        mode == "CAGING_PREGRASP_PLAN"
        and endpoint_anchor
        and not (f3_anchor or f4_anchor)
        and not pregrasp_replay_reached
        and (
            endpoint_anchor_residual == ""
            or float(endpoint_anchor_residual) > safe_standoff_residual_limit
        )
    ):
        row = {
            "phase": phase_name,
            "trial_id": trial_id,
            "result_class": "KINEMATIC_GRASP_BLOCKER_PROVEN",
            "termination_reason": "acquisition_pregrasp_not_reached_for_endpoint_anchor",
            "stable_contact": False,
            "finger3_force_peak_n": approach.get("finger3_force_peak_n", 0.0),
            "finger4_force_peak_n": approach.get("finger4_force_peak_n", 0.0),
            "endpoint_anchor_finger": endpoint_anchor,
            "endpoint_anchor_residual_norm_m": endpoint_anchor_residual,
            "safe_standoff_residual_limit_m": safe_standoff_residual_limit,
            "approach_termination_reason": approach.get("termination_reason", ""),
            "approach_servo_first_failed_stage": approach.get("servo_first_failed_stage", ""),
            "approach_servo_completed_stage_count": approach.get("servo_completed_stage_count", ""),
            "approach_servo_stage_count": approach.get("servo_stage_count", ""),
            "object_displacement_m": _distance(start_obj, after_approach["object_local_pos"]),
            "close_executed": False,
            "lift_executed": False,
        }
        episodes.append(row)
        return {**row, "success_claimed": False, "object_ready": False, "grasp_solved": False}
    if mode == "CAGING_PREGRASP_PLAN" and endpoint_anchor and not (f3_anchor or f4_anchor):
        opposing = _endpoint_caging_anchor_wrist_approach(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=f"{phase_name}_endpoint_anchor_wrist_approach",
            start_obj=start_obj,
            anchor_finger=endpoint_anchor,
        )
        after_approach = dict(opposing.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
        f3_anchor = float(after_approach.get("finger3_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
        f4_anchor = float(after_approach.get("finger4_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
        endpoint_peak_f3 = float(opposing.get("finger3_force_peak_n", 0.0) or 0.0)
        endpoint_peak_f4 = float(opposing.get("finger4_force_peak_n", 0.0) or 0.0)
        endpoint_wrist_termination = str(opposing.get("termination_reason") or "")
        endpoint_orientation: dict[str, Any] = {}
        if not (f3_anchor or f4_anchor):
            endpoint_orientation = _endpoint_caging_anchor_orientation_refinement(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase_name}_endpoint_anchor_orientation_refinement",
                start_obj=start_obj,
                anchor_finger=endpoint_anchor,
            )
            after_approach = dict(
                endpoint_orientation.get("final_state")
                or read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
            )
            f3_anchor = float(after_approach.get("finger3_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
            f4_anchor = float(after_approach.get("finger4_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
            endpoint_peak_f3 = max(endpoint_peak_f3, float(endpoint_orientation.get("finger3_force_peak_n", 0.0) or 0.0))
            endpoint_peak_f4 = max(endpoint_peak_f4, float(endpoint_orientation.get("finger4_force_peak_n", 0.0) or 0.0))
            opposing = {
                **endpoint_orientation,
                "finger3_force_peak_n": endpoint_peak_f3,
                "finger4_force_peak_n": endpoint_peak_f4,
                "endpoint_wrist_termination_reason": endpoint_wrist_termination,
            }
        if not (f3_anchor or f4_anchor):
            endpoint_finger = _endpoint_caging_anchor_finger_acquisition(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase_name}_endpoint_anchor_finger_acquisition",
                start_obj=start_obj,
                anchor_finger=endpoint_anchor,
            )
            after_approach = dict(endpoint_finger.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
            f3_anchor = float(after_approach.get("finger3_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
            f4_anchor = float(after_approach.get("finger4_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
            endpoint_peak_f3 = max(endpoint_peak_f3, float(endpoint_finger.get("finger3_force_peak_n", 0.0) or 0.0))
            endpoint_peak_f4 = max(endpoint_peak_f4, float(endpoint_finger.get("finger4_force_peak_n", 0.0) or 0.0))
            opposing = {
                **endpoint_finger,
                "finger3_force_peak_n": endpoint_peak_f3,
                "finger4_force_peak_n": endpoint_peak_f4,
                "endpoint_wrist_termination_reason": endpoint_wrist_termination,
                "endpoint_orientation_termination_reason": endpoint_orientation.get("termination_reason", ""),
                "endpoint_orientation_total_deg": endpoint_orientation.get("endpoint_orientation_total_deg", ""),
            }
    if mode == "CAGING_PREGRASP_PLAN" and not (f3_anchor or f4_anchor):
        opposing = _caging_finger_contact_acquisition(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=f"{phase_name}_caging_finger_acquisition",
            start_obj=start_obj,
        )
        after_approach = dict(opposing.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
        f3_anchor = float(after_approach.get("finger3_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
        f4_anchor = float(after_approach.get("finger4_target_filtered_force_n", 0.0) or 0.0) >= cfg.contact_threshold_n
    if not (f3_anchor or f4_anchor):
        termination_reason = (
            opposing.get("termination_reason")
            or approach.get("termination_reason")
            or "first_anchor_contact_not_acquired"
        )
        result_class = (
            "CONTACT_ACQUISITION_INCOMPLETE"
            if termination_reason == "caging_finger_acquisition_max_steps"
            else "KINEMATIC_GRASP_BLOCKER_PROVEN"
        )
        row = {
            "phase": phase_name,
            "trial_id": trial_id,
            "result_class": result_class,
            "termination_reason": termination_reason,
            "stable_contact": False,
            "endpoint_anchor_finger": endpoint_anchor,
            "endpoint_wrist_termination_reason": opposing.get("endpoint_wrist_termination_reason", ""),
            "endpoint_orientation_termination_reason": opposing.get("endpoint_orientation_termination_reason", ""),
            "endpoint_orientation_total_deg": opposing.get("endpoint_orientation_total_deg", ""),
            "finger3_force_peak_n": max(approach.get("finger3_force_peak_n", 0.0), opposing.get("finger3_force_peak_n", 0.0)),
            "finger4_force_peak_n": max(approach.get("finger4_force_peak_n", 0.0), opposing.get("finger4_force_peak_n", 0.0)),
            "object_displacement_m": _distance(start_obj, after_approach["object_local_pos"]),
            "close_executed": False,
            "lift_executed": False,
        }
        episodes.append(row)
        return {**row, "success_claimed": False, "object_ready": False, "grasp_solved": False}
    anchor_finger = "finger3" if f3_anchor and not f4_anchor else "finger4" if f4_anchor and not f3_anchor else "both"
    opposing_executed = False
    if anchor_finger != "both":
        replay_profile = dict((seed_hint or {}).get("dual_contact_replay_wrist_profile") or {})
        if replay_profile:
            replay_action = _zero_action(env, base)
            replay_translation = [float(value) for value in list(replay_profile.get("translation_xyz_m", [0.0, 0.0, 0.0]))[:3]]
            replay_rotation = [math.radians(float(value)) for value in list(replay_profile.get("rotation_rpy_deg", [0.0, 0.0, 0.0]))[:3]]
            _set_wrist_delta_action(base, replay_action, env_index, replay_translation, replay_rotation)
            after_approach = _step_direct_action(
                env,
                base,
                env_index,
                cfg,
                replay_action,
                phase=f"{phase_name}_dual_profile_replay",
                step=0,
                trace_rows=trace_rows,
                alignment=alignment,
                extra={"dual_contact_replay_wrist_profile": replay_profile},
            )
        opposing_finger = "finger4" if anchor_finger == "finger3" else "finger3"
        after_approach = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase=f"{phase_name}_opposing_finger_approach_entry",
            step=0,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "anchor_finger": anchor_finger,
                "opposing_finger": opposing_finger,
                "opposing_finger_approach_entry": True,
            },
        )
        _write_seed_live_progress(
            cfg,
            {
                "phase": f"{phase_name}_opposing_finger_approach_entry",
                "event": "opposing_finger_approach_entry",
                "anchor_finger": anchor_finger,
                "opposing_finger": opposing_finger,
                "finger3_force_n": float(after_approach.get("finger3_target_filtered_force_n", 0.0) or 0.0),
                "finger4_force_n": float(after_approach.get("finger4_target_filtered_force_n", 0.0) or 0.0),
            },
        )
        opposing_executed = True
        opposing = _active_dual_contact_acquisition(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=f"{phase_name}_opposing_finger_approach",
            anchor_finger=anchor_finger,
            allow_wrist=bool((seed_hint or {}).get("dual_contact_allow_wrist", True)),
            initial_selected_actions=dict(opposing.get("selected_finger_actions") or {}),
        )
        after_approach = dict(opposing.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
    regulator = opposing.get("regulator")
    if not isinstance(regulator, DualContactRegulator):
        initial_anchor = "finger3" if f3_anchor else "finger4"
        regulator = DualContactRegulator(_dual_regulator_config(cfg), anchor_finger=initial_anchor)
    selected_actions = dict(opposing.get("selected_actions") or {})
    contact_hold = _active_dual_contact_hold(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        phase=f"{phase_name}_stable_contact_hold",
        start_obj=start_obj,
        regulator=regulator,
        selected_actions=selected_actions,
    )
    stable_contact = bool(contact_hold.get("stable_contact"))
    if not stable_contact:
        opposing_termination = str(opposing.get("termination_reason") or "")
        hold_termination = str(contact_hold.get("termination_reason") or "")
        termination_reason = (
            opposing_termination
            if opposing_executed and opposing_termination not in {"", "two_finger_contact_acquired"}
            else hold_termination or opposing_termination or str(approach.get("termination_reason") or "")
        )
        row = {
            "phase": phase_name,
            "trial_id": trial_id,
            "result_class": "DUAL_CONTACT_REGULATION_BLOCKER_PROVEN",
            "termination_reason": termination_reason,
            "stable_contact": False,
            "anchor_finger": anchor_finger,
            "opposing_finger_approach_executed": opposing_executed,
            "opposing_finger_approach_termination_reason": opposing.get("termination_reason", ""),
            "opposing_final_anchor_finger": opposing.get("final_anchor_finger", ""),
            "opposing_contact_handoff_count": opposing.get("contact_handoff_count", 0),
            "finger3_contact_duty_ratio": contact_hold.get("finger3_contact_duty_ratio", 0.0),
            "finger4_contact_duty_ratio": contact_hold.get("finger4_contact_duty_ratio", 0.0),
            "simultaneous_contact_duty_ratio": contact_hold.get("simultaneous_contact_duty_ratio", 0.0),
            "simultaneous_contact_steps": contact_hold.get("simultaneous_contact_steps", 0),
            "finger3_force_mean_n": contact_hold.get("finger3_force_mean_n", 0.0),
            "finger4_force_mean_n": contact_hold.get("finger4_force_mean_n", 0.0),
            "target_force_vector_dot_mean": contact_hold.get("target_force_vector_dot_mean", ""),
            "contact_position_source": contact_hold.get("contact_position_source", ""),
            "wrist_response_profile": opposing.get("wrist_response_profile", []),
            "selected_finger_actions": contact_hold.get("selected_actions", opposing.get("selected_actions", {})),
            "finger3_force_peak_n": max(approach.get("finger3_force_peak_n", 0.0), opposing.get("finger3_force_peak_n", 0.0), contact_hold.get("finger3_force_peak_n", 0.0)),
            "finger4_force_peak_n": max(approach.get("finger4_force_peak_n", 0.0), opposing.get("finger4_force_peak_n", 0.0), contact_hold.get("finger4_force_peak_n", 0.0)),
            "object_displacement_m": _distance(start_obj, after_approach["object_local_pos"]),
            "close_executed": False,
            "lift_executed": False,
        }
        episodes.append(row)
        return {**row, "success_claimed": False, "object_ready": False, "grasp_solved": False}

    regulator = contact_hold.get("regulator") if isinstance(contact_hold.get("regulator"), DualContactRegulator) else regulator
    selected_actions = dict(contact_hold.get("selected_actions") or selected_actions)
    close = _controlled_close_regulated(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        phase=f"{phase_name}_controlled_close",
        start_obj=start_obj,
        regulator=regulator,
        selected_actions=selected_actions,
    )
    if not bool(close.get("close_completed")):
        row = {
            "phase": phase_name,
            "trial_id": trial_id,
            "result_class": "STABLE_CONTACT_ACQUIRED_CLOSE_FAILED",
            "termination_reason": close.get("termination_reason", "close_failed_or_force_abort"),
            "stable_contact": True,
            "close_executed": True,
            "lift_executed": False,
            "anchor_finger": anchor_finger,
            "finger3_force_peak_n": max(approach.get("finger3_force_peak_n", 0.0), opposing.get("finger3_force_peak_n", 0.0), contact_hold.get("finger3_force_peak_n", 0.0), close.get("finger3_force_peak_n", 0.0)),
            "finger4_force_peak_n": max(approach.get("finger4_force_peak_n", 0.0), opposing.get("finger4_force_peak_n", 0.0), contact_hold.get("finger4_force_peak_n", 0.0), close.get("finger4_force_peak_n", 0.0)),
            "object_displacement_m": _distance(start_obj, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)["object_local_pos"]),
        }
        episodes.append(row)
        return {**row, "success_claimed": False, "object_ready": False, "grasp_solved": False}

    regulator = close.get("regulator") if isinstance(close.get("regulator"), DualContactRegulator) else regulator
    selected_actions = dict(close.get("selected_actions") or selected_actions)
    lift_start_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    lift_start_rel = _sub_vec(lift_start_state["object_local_pos"], lift_start_state["palm_local_pos"])
    lift = _slow_lift_regulated(
        env,
        base,
        env_index,
        cfg,
        alignment,
        trace_rows,
        start_obj=lift_start_state["object_local_pos"],
        start_rel=lift_start_rel,
        regulator=regulator,
        selected_actions=selected_actions,
    )
    result = "STABLE_GRASP_ACQUIRED_LIFT_FAILED"
    row = {
        "phase": phase_name,
        "trial_id": trial_id,
        "result_class": result,
        "termination_reason": lift.get("termination_reason", ""),
        "stable_contact": True,
        "anchor_finger": anchor_finger,
        "close_executed": True,
        "lift_executed": True,
        "lift_success": bool(lift.get("lift_success")),
        "development_outcome": "PHYSICAL_LIFT_ACQUIRED_PENDING_VALIDATION" if bool(lift.get("lift_success")) else "",
        "object_lift_delta_z_m": lift.get("object_lift_delta_z_m", 0.0),
        "relative_pose_drift_m": lift.get("relative_pose_drift_m", 0.0),
        "finger3_force_peak_n": max(approach.get("finger3_force_peak_n", 0.0), opposing.get("finger3_force_peak_n", 0.0), contact_hold.get("finger3_force_peak_n", 0.0), close.get("finger3_force_peak_n", 0.0), lift.get("finger3_force_peak_n", 0.0)),
        "finger4_force_peak_n": max(approach.get("finger4_force_peak_n", 0.0), opposing.get("finger4_force_peak_n", 0.0), contact_hold.get("finger4_force_peak_n", 0.0), close.get("finger4_force_peak_n", 0.0), lift.get("finger4_force_peak_n", 0.0)),
        "object_displacement_m": _distance(start_obj, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)["object_local_pos"]),
        "wrist_response_profile": opposing.get("wrist_response_profile", []),
        "selected_finger_actions": close.get("selected_actions", contact_hold.get("selected_actions", {})),
    }
    episodes.append(row)
    return {
        **row,
        "success_claimed": False,
        "object_ready": False,
        "grasp_solved": False,
        "physical_grasp_success": bool(close.get("close_completed")),
        "physical_lift_success": bool(lift.get("lift_success")),
        "not_physical": not bool(lift.get("lift_success")),
    }


def _write_dual_regulator_profile(
    cfg: Screw1GraspBaselineV2Config,
    *,
    plan: dict[str, Any],
    result: dict[str, Any],
    development_trial: int,
) -> str:
    path = Path(cfg.output_dir) / "dual_contact_regulator_profile.json"
    write_json(
        path,
        {
            "acquisition_plan_id": plan.get("plan_id", ""),
            "development_trial": int(development_trial),
            "contact_on_n": float(cfg.contact_threshold_n),
            "contact_keep_n": float(cfg.dual_contact_keep_n),
            "anchor_force_band_n": [
                float(cfg.dual_contact_anchor_force_min_n),
                float(cfg.dual_contact_anchor_force_max_n),
            ],
            "anchor_force_target_n": float(cfg.dual_contact_anchor_force_target_n),
            "soft_force_max_n": float(cfg.soft_force_max_n),
            "hard_abort_force_n": float(cfg.hard_abort_force_n),
            "finger_step_rad": float(cfg.dual_contact_finger_step_rad),
            "selected_finger_actions": result.get("selected_finger_actions", {}),
            "wrist_response_profile": result.get("wrist_response_profile", []),
            "stable_contact": bool(result.get("stable_contact")),
            "lift_success": bool(result.get("lift_success")),
            "object_write_after_reset_used": False,
            "sticky_used": False,
            "proxy_success_used": False,
        },
    )
    return str(path)


def _rank_local_plan_corrections(cfg: Screw1GraspBaselineV2Config) -> list[tuple[str, int]]:
    path = Path(cfg.output_dir) / "dual_contact_force_response.csv"
    if not path.exists():
        return []
    best: dict[str, tuple[tuple[float, ...], int]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("safe", "")).lower() not in {"true", "1"}:
                    continue
                axis = str(row.get("probe_axis", ""))
                if axis not in {"x", "y", "z", "roll", "pitch", "yaw"}:
                    continue
                try:
                    score_values = json.loads(str(row.get("response_score", "[]")))
                    score = tuple(float(value) for value in score_values)
                    sign = int(float(row.get("probe_sign", 0)))
                except Exception:
                    continue
                if sign not in {-1, 1}:
                    continue
                if axis not in best or score > best[axis][0]:
                    best[axis] = (score, sign)
    except Exception:
        return []
    ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
    return [(axis, value[1]) for axis, value in ranked[:6]]


def _local_corrected_acquisition_plan(plan: dict[str, Any], axis: str, sign: int) -> dict[str, Any]:
    corrected = json.loads(json.dumps(_plain(plan)))
    translation = [0.0, 0.0, 0.0]
    rotation = [0.0, 0.0, 0.0]
    if axis in {"x", "y", "z"}:
        translation[("x", "y", "z").index(axis)] = float(sign) * 0.001
    else:
        rotation[("roll", "pitch", "yaw").index(axis)] = float(sign) * 1.0
    target = [float(value) for value in list(corrected.get("final_target_pos_xyz", []))[:3]]
    if len(target) == 3:
        corrected["final_target_pos_xyz"] = [target[i] + translation[i] for i in range(3)]
    quat = _quat_normalize_wxyz(corrected.get("final_target_quat_wxyz", []))
    if len(quat) == 4 and any(rotation):
        corrected["final_target_quat_wxyz"] = _quat_for_rpy_offset(quat, tuple(rotation))
    offset = [float(value) for value in list(corrected.get("seed_offset_xyz_m", []))[:3]]
    if len(offset) == 3:
        corrected["seed_offset_xyz_m"] = [offset[i] + translation[i] for i in range(3)]
    rpy = [float(value) for value in list(corrected.get("final_orientation_rpy_deg", []))[:3]]
    if len(rpy) == 3:
        corrected["final_orientation_rpy_deg"] = [rpy[i] + rotation[i] for i in range(3)]
    stages = list(corrected.get("staged_wrist_waypoints", []) or [])
    if stages:
        final_stage = dict(stages[-1])
        if len(target) == 3:
            final_stage["target_pos_xyz"] = [target[i] + translation[i] for i in range(3)]
        if len(quat) == 4 and any(rotation):
            final_stage["target_quat_wxyz"] = corrected["final_target_quat_wxyz"]
        stages[-1] = final_stage
        corrected["staged_wrist_waypoints"] = stages
    corrected["parent_plan_id"] = plan.get("plan_id", "")
    corrected["plan_id"] = f"{plan.get('plan_id', 'acquisition_plan')}_dual_{axis}{int(sign):+d}"
    corrected["local_dual_contact_correction"] = {
        "axis": axis,
        "sign": int(sign),
        "translation_xyz_m": translation,
        "rotation_rpy_deg": rotation,
    }
    return corrected


def _run_bounded_dual_contact_development(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    authority_rows: list[dict[str, Any]],
    *,
    seed_hint: dict[str, Any],
) -> dict[str, Any]:
    plan = dict(seed_hint.get("validated_acquisition_plan") or {})
    trial_rows: list[dict[str, Any]] = []
    replay_profile: dict[str, Any] = {}
    for trial_id, mode in enumerate(("finger_only", "combined", "measured_replay")):
        if mode == "measured_replay" and not replay_profile:
            break
        hint = dict(seed_hint)
        hint["dual_contact_allow_wrist"] = mode == "combined"
        if mode == "measured_replay" and replay_profile:
            hint["dual_contact_replay_wrist_profile"] = replay_profile
        result = _run_force_guarded_grasp_trial(
            env,
            base,
            env_index,
            cfg,
            alignment,
            episodes,
            trace_rows,
            authority_rows,
            trial_id=trial_id,
            phase_name=f"force_guarded_grasp_{mode}",
            seed_hint=hint,
        )
        result["dual_contact_development_mode"] = mode
        trial_rows.append(result)
        profiles = list(result.get("wrist_response_profile", []) or [])
        if profiles:
            replay_profile = dict(profiles[-1])
        if bool(result.get("stable_contact")):
            profile_path = _write_dual_regulator_profile(
                cfg, plan=plan, result=result, development_trial=trial_id
            )
            return {
                **result,
                "dual_contact_development_trial_count": trial_id + 1,
                "dual_contact_development_trials": trial_rows,
                "dual_contact_regulator_profile_json": profile_path,
                "local_acquisition_correction_executed": False,
            }

    corrections = _rank_local_plan_corrections(cfg)
    for correction_index, (axis, sign) in enumerate(corrections[:6]):
        corrected_plan = _local_corrected_acquisition_plan(plan, axis, sign)
        hint = dict(seed_hint)
        hint["validated_acquisition_plan"] = corrected_plan
        hint["dual_contact_allow_wrist"] = True
        result = _run_force_guarded_grasp_trial(
            env,
            base,
            env_index,
            cfg,
            alignment,
            episodes,
            trace_rows,
            authority_rows,
            trial_id=3 + correction_index,
            phase_name=f"force_guarded_grasp_local_{axis}_{sign:+d}",
            seed_hint=hint,
        )
        result["local_acquisition_correction_axis"] = axis
        result["local_acquisition_correction_sign"] = sign
        trial_rows.append(result)
        if bool(result.get("stable_contact")):
            plan_path = Path(cfg.output_dir) / "validated_acquisition_plan_v2.json"
            write_json(plan_path, corrected_plan)
            profile_path = _write_dual_regulator_profile(
                cfg, plan=corrected_plan, result=result, development_trial=3 + correction_index
            )
            return {
                **result,
                "dual_contact_development_trial_count": len(trial_rows),
                "dual_contact_development_trials": trial_rows,
                "dual_contact_regulator_profile_json": profile_path,
                "validated_acquisition_plan_v2_json": str(plan_path),
                "local_acquisition_correction_executed": True,
                "local_acquisition_correction_trial_count": correction_index + 1,
            }
    return {
        "result_class": "DUAL_CONTACT_REGULATION_BLOCKER_PROVEN",
        "termination_reason": "bounded_active_regulation_and_local_corrections_exhausted",
        "stable_contact": False,
        "close_executed": False,
        "lift_executed": False,
        "success_claimed": False,
        "object_ready": False,
        "grasp_solved": False,
        "dual_contact_development_trial_count": len(trial_rows),
        "dual_contact_development_trials": trial_rows,
        "local_acquisition_correction_executed": bool(corrections),
        "local_acquisition_correction_trial_count": len(corrections[:6]),
    }


def _run_repeated_trials(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    authority_rows: list[dict[str, Any]],
    *,
    seed_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successes = 0
    best_consecutive = 0
    current_consecutive = 0
    final_result = "STABLE_GRASP_ACQUIRED_LIFT_FAILED"
    rows = []
    for trial in range(int(cfg.repeated_trials)):
        result = _run_force_guarded_grasp_trial(
            env,
            base,
            env_index,
            cfg,
            alignment,
            episodes,
            trace_rows,
            authority_rows,
            trial_id=trial,
            phase_name="repeated_trial",
            seed_hint=seed_hint,
        )
        ok = bool(result.get("lift_success"))
        successes += 1 if ok else 0
        current_consecutive = current_consecutive + 1 if ok else 0
        best_consecutive = max(best_consecutive, current_consecutive)
        rows.append(result)
    if successes >= int(cfg.repeated_success_required):
        final_result = "REPEATABLE_PHYSICAL_LIFT_ACQUIRED"
    elif any(bool(row.get("stable_contact")) for row in rows):
        final_result = "STABLE_GRASP_ACQUIRED_LIFT_FAILED"
    else:
        final_result = "DUAL_CONTACT_REGULATION_BLOCKER_PROVEN"
    return {
        "repeated_trials_executed": True,
        "repeated_trial_count": int(cfg.repeated_trials),
        "physical_lift_success_count": successes,
        "physical_lift_success_best_consecutive": best_consecutive,
        "result_class": final_result,
        "success_claimed": final_result == "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
        "object_ready": final_result == "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
        "grasp_solved": final_result == "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
        "physical_grasp_success": any(bool(row.get("stable_contact")) and bool(row.get("close_executed")) for row in rows),
        "physical_lift_success": final_result == "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
        "not_physical": final_result != "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
    }


def _servo_to_pose_guarded(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    target_pos: list[float],
    target_quat: list[float],
    max_steps: int,
    extra: dict[str, Any] | None = None,
    contact_truth: Any | None = None,
    all_body_abort_on_target: bool = True,
) -> dict[str, Any]:
    extra = dict(extra or {})
    finger3_peak = 0.0
    finger4_peak = 0.0
    hard_abort = False
    reached = False
    final_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    termination = "max_steps"
    last_move = [0.0, 0.0, 0.0]
    last_rot_move = [0.0, 0.0, 0.0]
    min_pos_error = float("inf")
    final_pos_error = float("inf")
    free_space_step_count = 0
    fine_step_count = 0
    max_workspace_clamp_delta = 0.0
    max_table_barrier_delta_z = 0.0
    tracking_wait_step_count = 0
    tracking_wait_consecutive_count = 0
    max_ctrl_to_palm_error = 0.0
    max_ctrl_to_actual_angle_error = 0.0
    max_target_to_ctrl_error = 0.0
    max_target_to_ctrl_angle_error = 0.0
    tracking_timeout = False
    tracking_stall = False
    tracking_stall_reason = ""
    command_complete_final = False
    actual_complete_final = False
    servo_state = "COMMAND_ADVANCE"
    actual_error_history: list[tuple[float, float]] = []
    tracking_wait_error_history: list[tuple[float, float]] = []
    all_body_contact_peak = 0.0
    all_body_target_contact_peak = 0.0
    all_body_non_target_contact_peak = 0.0
    all_body_contact_abort = False
    for step in range(int(max_steps)):
        state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
        final_state = state
        max_workspace_clamp_delta = max(
            max_workspace_clamp_delta,
            abs(float(state.get("workspace_clamp_delta_m", 0.0) or 0.0)),
        )
        max_table_barrier_delta_z = max(
            max_table_barrier_delta_z,
            abs(float(state.get("table_barrier_delta_z_m", 0.0) or 0.0)),
        )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        finger3_peak = max(finger3_peak, f3)
        finger4_peak = max(finger4_peak, f4)
        guard = _force_guard_decision(f3, f4, cfg)
        if int(step) == 0 or int(step) % 25 == 0:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "servo_step",
                    "step": int(step),
                    "max_steps": int(max_steps),
                    "seed_name": extra.get("seed_name", ""),
                    "seed_repeat": extra.get("seed_repeat", ""),
                    "servo_stage_name": extra.get("servo_stage_name", ""),
                    "servo_stage_index": extra.get("servo_stage_index", ""),
                    "servo_stage_count": extra.get("servo_stage_count", ""),
                    "servo_state": servo_state,
                    "termination_reason": termination,
                    "force_guard_decision": guard,
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "workspace_clamp_delta_m": state.get("workspace_clamp_delta_m", 0.0),
                    "table_barrier_delta_z_m": state.get("table_barrier_delta_z_m", 0.0),
                    "palm_local_pos": state.get("palm_local_pos", []),
                    "target_pos_xyz": list(target_pos),
                },
            )
        if guard == "abort":
            hard_abort = True
            termination = "hard_force_abort_before_step"
            retreat = _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, last_move, last_rot_move)
            final_state = dict(retreat.get("final_state") or final_state)
            break
        if guard == "retreat":
            termination = "soft_force_retreat_before_step"
            retreat = _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, last_move, last_rot_move)
            final_state = dict(retreat.get("final_state") or final_state)
            break
        if guard == "hold":
            termination = "target_contact_soft_band_acquired"
            break
        actual_pos = list(state["palm_local_pos"])
        actual_quat = _quat_normalize_wxyz(state["palm_quat_wxyz"])
        ctrl_pos = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
        if len(ctrl_pos) < 3:
            ctrl_pos = list(actual_pos)
        ctrl_quat = _quat_normalize_wxyz(
            state.get("ctrl_target_palm_quat_wxyz", [])
            or _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4)
        )
        pos_err = _sub_vec(target_pos, actual_pos)
        pos_dist = _norm(pos_err)
        min_pos_error = min(min_pos_error, pos_dist)
        final_pos_error = pos_dist
        target_quat = _quat_normalize_wxyz(target_quat)
        actual_rot_err = _quat_error_axis_angle_wxyz(actual_quat, target_quat)
        rot_dist = _norm(actual_rot_err)
        actual_complete = bool(
            pos_dist <= float(cfg.seed_reach_tol_m)
            and rot_dist <= math.radians(float(cfg.seed_reach_rot_tol_deg))
        )
        reached = actual_complete
        if actual_complete:
            termination = "target_pose_reached"
            command_complete_final = True
            actual_complete_final = True
            servo_state = "TARGET_REACHED"
            break
        command_origin = ctrl_pos
        ctrl_to_palm_error = _distance(command_origin, actual_pos)
        ctrl_to_actual_rot_err = _quat_error_axis_angle_wxyz(actual_quat, ctrl_quat)
        ctrl_to_actual_rot_dist = _norm(ctrl_to_actual_rot_err)
        max_ctrl_to_palm_error = max(max_ctrl_to_palm_error, ctrl_to_palm_error)
        max_ctrl_to_actual_angle_error = max(max_ctrl_to_actual_angle_error, ctrl_to_actual_rot_dist)
        command_err = _sub_vec(target_pos, command_origin)
        command_dist = _norm(command_err)
        command_rot_err = _quat_error_axis_angle_wxyz(ctrl_quat, target_quat)
        command_rot_dist = _norm(command_rot_err)
        max_target_to_ctrl_error = max(max_target_to_ctrl_error, command_dist)
        max_target_to_ctrl_angle_error = max(max_target_to_ctrl_angle_error, command_rot_dist)
        command_complete = bool(
            command_dist <= float(cfg.seed_servo_command_tol_m)
            and command_rot_dist <= math.radians(float(cfg.seed_servo_command_rot_tol_deg))
        )
        command_complete_final = command_complete
        actual_complete_final = actual_complete
        actual_error_history.append((float(pos_dist), float(rot_dist)))
        servo_mode = "free_space" if max(pos_dist, command_dist) > float(cfg.seed_servo_fine_radius_m) else "fine"
        servo_step_m = (
            float(cfg.seed_servo_free_space_step_m)
            if servo_mode == "free_space"
            else float(cfg.seed_servo_pos_step_m)
        )
        target_lead_limit = float(cfg.seed_servo_max_target_lead_m)
        rotation_lead_limit = math.radians(float(cfg.seed_servo_max_rotation_target_lead_deg))
        lead_too_high = bool(ctrl_to_palm_error > target_lead_limit or ctrl_to_actual_rot_dist > rotation_lead_limit)
        waiting_for_tracking = bool(command_complete or lead_too_high)
        if waiting_for_tracking:
            tracking_wait_step_count += 1
            tracking_wait_consecutive_count += 1
            servo_state = "WAIT_ACTUAL_TRACKING"
            servo_mode = "wait_actual_tracking" if command_complete else "hold_for_tracking_lead"
            delta = [0.0, 0.0, 0.0]
            rot_delta = [0.0, 0.0, 0.0]
            tracking_wait_error_history.append((float(pos_dist), float(rot_dist)))
            window = max(2, int(cfg.seed_servo_tracking_stall_window_steps))
            if len(tracking_wait_error_history) > window:
                old_pos, old_rot = tracking_wait_error_history[-window - 1]
                pos_improvement = float(old_pos) - float(pos_dist)
                rot_improvement_deg = math.degrees(float(old_rot) - float(rot_dist))
                if (
                    pos_improvement < float(cfg.seed_servo_tracking_stall_min_improvement_m)
                    and rot_improvement_deg < float(cfg.seed_servo_tracking_stall_min_improvement_deg)
                ):
                    tracking_stall = True
                    tracking_stall_reason = "actual_error_not_improving"
                    termination = "wrist_tracking_stall"
                    final_state = state
                    break
            if tracking_wait_consecutive_count >= int(cfg.seed_servo_tracking_timeout_steps):
                tracking_timeout = True
                termination = "wrist_tracking_timeout"
                final_state = state
                break
        else:
            tracking_wait_consecutive_count = 0
            tracking_wait_error_history.clear()
            servo_state = "COMMAND_ADVANCE"
            if servo_step_m > float(cfg.seed_servo_pos_step_m):
                free_space_step_count += 1
            else:
                fine_step_count += 1
            delta = _cap_vec(command_err, servo_step_m)
            rot_delta = _cap_vec(command_rot_err, math.radians(cfg.seed_servo_rot_step_deg))
        if _norm(delta) > 1.0e-9:
            last_move = list(delta)
        if _norm(rot_delta) > 1.0e-9:
            last_rot_move = list(rot_delta)
        action = _zero_action(env, base)
        _set_wrist_delta_action(base, action, env_index, delta, rot_delta)
        _configure_v2_control(base, env_index, mode="anchored_delta")
        final_state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                **extra,
                "target_pos_xyz": list(target_pos),
                "target_quat_wxyz": list(target_quat),
                "pos_error_m": pos_dist,
                "ctrl_target_error_m": command_dist,
                "ctrl_target_to_palm_error_m": ctrl_to_palm_error,
                "target_lead_limit_m": target_lead_limit,
                "rotation_target_lead_limit_deg": float(cfg.seed_servo_max_rotation_target_lead_deg),
                "tracking_wait_step": waiting_for_tracking,
                "tracking_wait_consecutive_count": tracking_wait_consecutive_count,
                "servo_state": servo_state,
                "command_complete": command_complete,
                "actual_complete": actual_complete,
                "lead_too_high": lead_too_high,
                "rot_error_rad": rot_dist,
                "target_to_ctrl_rot_error_rad": command_rot_dist,
                "ctrl_to_actual_rot_error_rad": ctrl_to_actual_rot_dist,
                "actual_tracking_rot_axis_angle": list(actual_rot_err),
                "command_rot_axis_angle_uncapped": list(command_rot_err),
                "tracking_rot_axis_angle": list(ctrl_to_actual_rot_err),
                "servo_step_limit_m": servo_step_m,
                "servo_mode": servo_mode,
                "commanded_delta_xyz": list(delta),
                "commanded_rot_axis_angle": list(rot_delta),
            },
        )
        contact_stage_kind = str(extra.get("seed_replay_stage_kind", ""))
        contact_stride = 1 if contact_stage_kind == "final_contact_approach" else max(
            1, int(getattr(cfg, "seed_replay_contact_truth_transit_stride", 25) or 25)
        )
        record_contact_truth = bool(
            contact_truth is not None
            and (contact_stride <= 1 or int(step) == 0 or int(step) % int(contact_stride) == 0)
        )
        if record_contact_truth:
            contact_info = contact_truth.record(phase, step, final_state, base, extra={**extra, "servo_state": servo_state})
            all_body_contact_peak = max(all_body_contact_peak, float(contact_info.get("all_body_contact_peak_n", 0.0) or 0.0))
            all_body_target_contact_peak = max(
                all_body_target_contact_peak,
                float(contact_info.get("all_body_target_contact_peak_n", 0.0) or 0.0),
            )
            all_body_non_target_contact_peak = max(
                all_body_non_target_contact_peak,
                float(contact_info.get("all_body_non_target_contact_peak_n", 0.0) or 0.0),
            )
            if trace_rows:
                trace_rows[-1].update(contact_info)
            abort_peak = (
                float(contact_info.get("all_body_contact_peak_n", 0.0) or 0.0)
                if bool(all_body_abort_on_target)
                else float(contact_info.get("all_body_non_target_contact_peak_n", 0.0) or 0.0)
            )
            if abort_peak >= float(contact_truth.threshold_n):
                all_body_contact_abort = True
                termination = "all_body_contact_abort" if bool(all_body_abort_on_target) else "all_body_non_target_contact_abort"
                break
        max_workspace_clamp_delta = max(
            max_workspace_clamp_delta,
            abs(float(final_state.get("workspace_clamp_delta_m", 0.0) or 0.0)),
        )
        max_table_barrier_delta_z = max(
            max_table_barrier_delta_z,
            abs(float(final_state.get("table_barrier_delta_z_m", 0.0) or 0.0)),
        )
        f3 = float(final_state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(final_state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        finger3_peak = max(finger3_peak, f3)
        finger4_peak = max(finger4_peak, f4)
        guard = _force_guard_decision(f3, f4, cfg)
        if int(step) == 0 or int(step) % 25 == 0:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "servo_step_after_action",
                    "step": int(step),
                    "max_steps": int(max_steps),
                    "seed_name": extra.get("seed_name", ""),
                    "seed_repeat": extra.get("seed_repeat", ""),
                    "servo_stage_name": extra.get("servo_stage_name", ""),
                    "servo_stage_index": extra.get("servo_stage_index", ""),
                    "servo_stage_count": extra.get("servo_stage_count", ""),
                    "servo_state": servo_state,
                    "termination_reason": termination,
                    "force_guard_decision": guard,
                    "pos_error_m": final_pos_error,
                    "min_pos_error_m": min_pos_error if math.isfinite(min_pos_error) else "",
                    "ctrl_target_error_m": command_dist,
                    "ctrl_target_to_palm_error_m": ctrl_to_palm_error,
                    "rot_error_rad": rot_dist,
                    "target_to_ctrl_rot_error_rad": command_rot_dist,
                    "ctrl_to_actual_rot_error_rad": ctrl_to_actual_rot_dist,
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "all_body_contact_peak_n": all_body_contact_peak,
                    "all_body_target_contact_peak_n": all_body_target_contact_peak,
                    "all_body_non_target_contact_peak_n": all_body_non_target_contact_peak,
                    "workspace_clamp_delta_m": final_state.get("workspace_clamp_delta_m", 0.0),
                    "table_barrier_delta_z_m": final_state.get("table_barrier_delta_z_m", 0.0),
                },
            )
        if guard == "abort":
            hard_abort = True
            termination = "hard_force_abort_after_step"
            retreat = _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, last_move, last_rot_move)
            final_state = dict(retreat.get("final_state") or final_state)
            break
        if guard == "retreat":
            termination = "soft_force_retreat_after_step"
            retreat = _retreat(env, base, env_index, cfg, alignment, trace_rows, phase, last_move, last_rot_move)
            final_state = dict(retreat.get("final_state") or final_state)
            break
        if guard == "hold":
            termination = "target_contact_soft_band_acquired"
            break
        if abs(float(final_state.get("workspace_clamp_delta_m", 0.0) or 0.0)) > 1.0e-6:
            termination = "workspace_clamp_abort"
            break
        if abs(float(final_state.get("table_barrier_delta_z_m", 0.0) or 0.0)) > 1.0e-6:
            termination = "table_barrier_abort"
            break
    ctrl_target = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    final_target_to_ctrl_error = _distance(target_pos, ctrl_target) if len(ctrl_target) >= 3 else ""
    final_ctrl_to_palm_error = (
        _distance(ctrl_target, final_state.get("palm_local_pos", [])) if len(ctrl_target) >= 3 else ""
    )
    final_ctrl_quat = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4)
    final_actual_quat = list(final_state.get("palm_quat_wxyz", []))
    final_target_to_ctrl_angle = _quat_angle_delta_wxyz(target_quat, final_ctrl_quat)
    final_ctrl_to_actual_angle = _quat_angle_delta_wxyz(final_ctrl_quat, final_actual_quat)
    final_f3 = float(final_state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    final_f4 = float(final_state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    final_current_contact = bool(max(final_f3, final_f4) >= cfg.contact_threshold_n)
    return {
        "steps": step + 1 if "step" in locals() else 0,
        "termination_reason": termination,
        "final_state": final_state,
        "finger3_force_peak_n": finger3_peak,
        "finger4_force_peak_n": finger4_peak,
        "target_contact_acquired": bool(max(finger3_peak, finger4_peak) >= cfg.contact_threshold_n),
        "contact_ever_acquired": bool(max(finger3_peak, finger4_peak) >= cfg.contact_threshold_n),
        "final_current_contact": final_current_contact,
        "final_finger3_force_n": final_f3,
        "final_finger4_force_n": final_f4,
        "hard_abort": bool(hard_abort),
        "tracking_timeout": bool(tracking_timeout),
        "tracking_stall": bool(tracking_stall),
        "tracking_stall_reason": tracking_stall_reason,
        "servo_state": servo_state,
        "command_complete": bool(command_complete_final),
        "actual_complete": bool(actual_complete_final),
        "all_body_contact_abort": bool(all_body_contact_abort),
        "all_body_contact_peak_n": float(all_body_contact_peak),
        "all_body_target_contact_peak_n": float(all_body_target_contact_peak),
        "all_body_non_target_contact_peak_n": float(all_body_non_target_contact_peak),
        "reached_pose": bool(reached),
        "final_pos_error_m": final_pos_error if math.isfinite(final_pos_error) else "",
        "min_pos_error_m": min_pos_error if math.isfinite(min_pos_error) else "",
        "free_space_step_count": int(free_space_step_count),
        "fine_step_count": int(fine_step_count),
        "tracking_wait_step_count": int(tracking_wait_step_count),
        "max_ctrl_to_palm_error_m": max_ctrl_to_palm_error,
        "max_ctrl_to_actual_angle_error_deg": math.degrees(max_ctrl_to_actual_angle_error),
        "max_target_to_ctrl_error_m": max_target_to_ctrl_error,
        "max_target_to_ctrl_angle_error_deg": math.degrees(max_target_to_ctrl_angle_error),
        "final_ctrl_target_local_pos": ctrl_target,
        "final_target_to_ctrl_error_m": final_target_to_ctrl_error,
        "final_ctrl_to_palm_error_m": final_ctrl_to_palm_error,
        "final_ctrl_target_quat_wxyz": final_ctrl_quat,
        "final_target_to_ctrl_angle_error_deg": math.degrees(final_target_to_ctrl_angle),
        "final_ctrl_to_actual_angle_error_deg": math.degrees(final_ctrl_to_actual_angle),
        "final_wrist_target_pre_clamp_xyz": final_state.get("wrist_target_pre_clamp_xyz", []),
        "final_wrist_target_post_clamp_xyz": final_state.get("wrist_target_post_clamp_xyz", []),
        "final_wrist_target_delta_xyz": final_state.get("wrist_target_delta_xyz", []),
        "max_workspace_clamp_delta_m": max_workspace_clamp_delta,
        "max_table_barrier_delta_z_m": max_table_barrier_delta_z,
        "final_workspace_clamp_delta_m": final_state.get("workspace_clamp_delta_m", 0.0),
        "final_table_barrier_delta_z_m": final_state.get("table_barrier_delta_z_m", 0.0),
    }


def _dual_regulator_config(cfg: Screw1GraspBaselineV2Config) -> DualContactRegulatorConfig:
    return DualContactRegulatorConfig(
        contact_on_n=float(cfg.contact_threshold_n),
        contact_keep_n=float(cfg.dual_contact_keep_n),
        anchor_force_min_n=float(cfg.dual_contact_anchor_force_min_n),
        anchor_force_max_n=float(cfg.dual_contact_anchor_force_max_n),
        anchor_force_target_n=float(cfg.dual_contact_anchor_force_target_n),
        soft_force_max_n=float(cfg.soft_force_max_n),
        hard_abort_force_n=float(cfg.hard_abort_force_n),
        finger_step_rad=float(cfg.dual_contact_finger_step_rad),
        stagnation_steps=int(cfg.dual_contact_stagnation_steps),
        stagnation_min_improvement_n=float(cfg.dual_contact_stagnation_improvement_n),
    )


def _augment_dual_contact_state(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
) -> dict[str, Any]:
    state = dict(state)
    names = [str(item) for item in list(getattr(base, "dex_fingertip_target_filter_names", []) or [])]
    target_index = names.index(cfg.part) if cfg.part in names else -1
    matrix = getattr(base, "dex_fingertip_target_force_xyz", None)
    vectors = {"finger3": [0.0, 0.0, 0.0], "finger4": [0.0, 0.0, 0.0]}
    if torch is not None and torch.is_tensor(matrix) and matrix.ndim >= 4 and target_index >= 0:
        try:
            vectors["finger3"] = [float(value) for value in matrix[env_index, FINGER3_INDEX, target_index, :3].detach().cpu().tolist()]
            vectors["finger4"] = [float(value) for value in matrix[env_index, FINGER4_INDEX, target_index, :3].detach().cpu().tolist()]
        except Exception:
            pass
    state["finger3_target_filtered_force_xyz_n"] = vectors["finger3"]
    state["finger4_target_filtered_force_xyz_n"] = vectors["finger4"]
    state["finger3_contact_position_proxy_xyz_m"] = list(state.get("finger3_tip_local_pos", []))
    state["finger4_contact_position_proxy_xyz_m"] = list(state.get("finger4_tip_local_pos", []))
    state["contact_position_source"] = "fingertip_body_position_proxy"
    n3 = _norm(vectors["finger3"])
    n4 = _norm(vectors["finger4"])
    state["target_force_vector_normalized_dot"] = (
        sum(vectors["finger3"][i] * vectors["finger4"][i] for i in range(3)) / (n3 * n4)
        if n3 > 1.0e-9 and n4 > 1.0e-9
        else ""
    )
    return state


def _dual_observation(
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    state: dict[str, Any],
    start_obj: list[float],
    trace_rows: list[dict[str, Any]],
) -> DualContactObservation:
    velocity = _object_velocity(base, env_index, cfg.part)
    raw_non_target = float(trace_rows[-1].get("raw_non_target_peak_n", 0.0) or 0.0) if trace_rows else 0.0
    return DualContactObservation(
        finger3_force_n=float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
        finger4_force_n=float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
        object_displacement_m=_distance(start_obj, state.get("object_local_pos", [])),
        object_speed_mps=float(velocity.get("linear_velocity_norm", 0.0) or 0.0),
        non_target_force_n=raw_non_target,
        workspace_clamp_m=float(state.get("workspace_clamp_delta_m", 0.0) or 0.0),
    )


def _regulator_trace_fields(state: dict[str, Any], command: Any) -> dict[str, Any]:
    return {
        "dual_regulator_state": command.state,
        "dual_regulator_action_kind": command.action_kind,
        "dual_regulator_anchor_finger": command.anchor_finger,
        "dual_regulator_opposing_finger": command.opposing_finger,
        "dual_regulator_handoff": command.handoff,
        "dual_regulator_termination_reason": command.termination_reason,
        "finger3_contact_latched": command.latched_finger3_contact,
        "finger4_contact_latched": command.latched_finger4_contact,
        "finger3_formal_contact": command.formal_finger3_contact,
        "finger4_formal_contact": command.formal_finger4_contact,
        "finger3_target_filtered_force_xyz_n": state.get("finger3_target_filtered_force_xyz_n", []),
        "finger4_target_filtered_force_xyz_n": state.get("finger4_target_filtered_force_xyz_n", []),
        "finger3_contact_position_proxy_xyz_m": state.get("finger3_contact_position_proxy_xyz_m", []),
        "finger4_contact_position_proxy_xyz_m": state.get("finger4_contact_position_proxy_xyz_m", []),
        "contact_position_source": state.get("contact_position_source", ""),
        "target_force_vector_normalized_dot": state.get("target_force_vector_normalized_dot", ""),
    }


def _append_dual_force_response(cfg: Screw1GraspBaselineV2Config, row: dict[str, Any]) -> None:
    path = Path(cfg.output_dir) / "dual_contact_force_response.csv"
    fields = [
        "phase", "step", "probe_axis", "probe_sign", "amplitude_level", "translation_xyz_m",
        "rotation_rpy_deg", "finger3_before_n", "finger4_before_n", "finger3_after_n", "finger4_after_n",
        "anchor_finger", "opposing_finger", "object_probe_displacement_m", "object_speed_mps",
        "raw_non_target_peak_n", "workspace_clamp_m", "safe", "response_score", "rollback_ok", "selected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: _json(value) if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})
        handle.flush()


def _write_dual_heartbeat(
    cfg: Screw1GraspBaselineV2Config,
    *,
    phase: str,
    step: int,
    state: dict[str, Any],
    command: Any,
    start_obj: list[float],
) -> None:
    write_json(
        Path(cfg.output_dir) / "dual_contact_heartbeat.json",
        {
            "phase": phase,
            "step": int(step),
            "controller_state": command.state,
            "action_kind": command.action_kind,
            "anchor_finger": command.anchor_finger,
            "opposing_finger": command.opposing_finger,
            "finger3_force_n": state.get("finger3_target_filtered_force_n", 0.0),
            "finger4_force_n": state.get("finger4_target_filtered_force_n", 0.0),
            "object_displacement_m": _distance(start_obj, state.get("object_local_pos", [])),
            "handoff_count": int(getattr(command, "handoff", False)),
        },
    )


def _execute_regulator_finger_action(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
    command: Any,
    selected_actions: dict[str, dict[str, Any]],
    state: dict[str, Any],
    start_obj: list[float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    requested = list(command.finger_deltas.items())
    if not requested:
        action = _zero_action(env, base)
        next_state = _step_direct_action(
            env, base, env_index, cfg, action, phase=phase, step=step, trace_rows=trace_rows, alignment=alignment,
            extra=_regulator_trace_fields(state, command),
        )
        return _augment_dual_contact_state(base, env_index, cfg, next_state), selected_actions
    finger, requested_delta = requested[0]
    selected = dict(selected_actions.get(finger, {}))
    if not selected:
        targets = _hand_audit_snapshot(base, env_index).get("target", [])
        probe = _probe_best_anchor_finger_action(
            env, base, env_index, cfg, alignment, trace_rows,
            phase=f"{phase}_select_action", step=step, anchor_finger=finger, start_obj=start_obj,
            initial_state=state, initial_targets=list(targets),
            probe_action_value=float(cfg.dual_contact_finger_step_rad),
            min_improvement_m=float(cfg.opposing_finger_min_improvement_m),
            probe_settle_steps=1, probe_restore_settle_steps=1,
        )
        state = _augment_dual_contact_state(
            base, env_index, cfg, dict(probe.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
        )
        if probe.get("action_column") is not None:
            selected = {
                "action_column": int(probe["action_column"]),
                "advance_sign": 1.0 if float(probe.get("delta", 1.0)) >= 0.0 else -1.0,
            }
            selected_actions[finger] = selected
        if bool(probe.get("contact_acquired")) or not selected:
            return state, selected_actions
    advance_sign = float(selected.get("advance_sign", 1.0))
    value = math.copysign(abs(float(requested_delta)), advance_sign)
    if float(requested_delta) < 0.0:
        value = -value
    action = _zero_action(env, base)
    action[env_index, int(selected["action_column"])] = value
    before_force = float(state.get(f"{finger}_target_filtered_force_n", 0.0) or 0.0)
    next_state = _step_direct_action(
        env, base, env_index, cfg, action, phase=phase, step=step, trace_rows=trace_rows, alignment=alignment,
        extra={
            **_regulator_trace_fields(state, command),
            "dual_regulator_finger": finger,
            "dual_regulator_action_column": int(selected["action_column"]),
            "dual_regulator_finger_delta_rad": value,
        },
    )
    next_state = _augment_dual_contact_state(base, env_index, cfg, next_state)
    after_force = float(next_state.get(f"{finger}_target_filtered_force_n", 0.0) or 0.0)
    force_response = after_force - before_force
    direction_reversed = False
    if (
        command.action_kind in {"INCREASE_ANCHOR_FORCE", "REACQUIRE_ANCHOR", "INCREASE_DUAL_FORCE", "ADVANCE_OPPOSING"}
        and force_response <= -float(cfg.dual_contact_stagnation_improvement_n)
    ):
        selected["advance_sign"] = -advance_sign
        selected["polarity_reversed_from_force_response"] = True
        selected["polarity_reversal_force_delta_n"] = force_response
        selected_actions[finger] = selected
        direction_reversed = True
    if trace_rows:
        trace_rows[-1].update(
            {
                "dual_regulator_finger_force_before_n": before_force,
                "dual_regulator_finger_force_after_n": after_force,
                "dual_regulator_finger_force_response_n": force_response,
                "dual_regulator_action_direction_reversed_after_response": direction_reversed,
            }
        )
    return next_state, selected_actions


def _run_bounded_wrist_probe_pair(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
    regulator: DualContactRegulator,
    state: dict[str, Any],
    start_obj: list[float],
    probe_pair_index: int,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    axis_count = len(regulator.config.wrist_axis_order)
    pair_index = max(0, min(axis_count * 2 - 1, int(probe_pair_index)))
    level = 0 if pair_index < axis_count else 1
    axis_index = pair_index % axis_count
    candidates = regulator.wrist_probe_candidates(level)[axis_index * 2 : axis_index * 2 + 2]
    baseline_state = _augment_dual_contact_state(base, env_index, cfg, state)
    baseline_obs = _dual_observation(base, env_index, cfg, baseline_state, start_obj, trace_rows)
    ranked: list[tuple[tuple[float, ...], Any, dict[str, Any]]] = []
    response_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        action = _zero_action(env, base)
        delta_rot = [math.radians(value) for value in candidate.rotation_rpy_deg]
        _set_wrist_delta_action(base, action, env_index, list(candidate.translation_xyz_m), delta_rot)
        candidate_state = _step_direct_action(
            env, base, env_index, cfg, action, phase=f"{phase}_probe", step=step,
            trace_rows=trace_rows, alignment=alignment,
            extra={"dual_wrist_probe_axis": candidate.axis, "dual_wrist_probe_sign": candidate.sign, "dual_wrist_probe_level": level},
        )
        for settle_step in range(int(cfg.dual_contact_wrist_probe_settle_steps)):
            candidate_state = _step_direct_action(
                env, base, env_index, cfg, _zero_action(env, base), phase=f"{phase}_probe_settle", step=settle_step,
                trace_rows=trace_rows, alignment=alignment,
                extra={"dual_wrist_probe_axis": candidate.axis, "dual_wrist_probe_sign": candidate.sign},
            )
        candidate_state = _augment_dual_contact_state(base, env_index, cfg, candidate_state)
        candidate_obs = _dual_observation(base, env_index, cfg, candidate_state, start_obj, trace_rows)
        probe_motion = _distance(baseline_state.get("object_local_pos", []), candidate_state.get("object_local_pos", []))
        anchor_force = candidate_obs.force(regulator.anchor_finger)
        safe = bool(
            max(candidate_obs.finger3_force_n, candidate_obs.finger4_force_n) <= cfg.soft_force_max_n
            and candidate_obs.non_target_force_n <= cfg.contact_threshold_n
            and candidate_obs.workspace_clamp_m <= 1.0e-9
            and probe_motion <= cfg.dual_contact_probe_object_motion_limit_m
            and candidate_obs.object_speed_mps <= cfg.stable_velocity_limit_mps
            and anchor_force >= cfg.dual_contact_keep_n
        )
        score = regulator.score_wrist_probe(baseline_obs, candidate_obs) if safe else (-math.inf,)
        if safe:
            ranked.append((score, candidate, candidate_state))
        inverse = _zero_action(env, base)
        _set_wrist_delta_action(
            base, inverse, env_index,
            [-value for value in candidate.translation_xyz_m],
            [-value for value in delta_rot],
        )
        restored = _step_direct_action(
            env, base, env_index, cfg, inverse, phase=f"{phase}_rollback", step=step,
            trace_rows=trace_rows, alignment=alignment,
            extra={"dual_wrist_probe_axis": candidate.axis, "dual_wrist_probe_sign": candidate.sign, "dual_wrist_probe_rollback": True},
        )
        for restore_step in range(int(cfg.dual_contact_wrist_probe_restore_steps)):
            restored = _step_direct_action(
                env, base, env_index, cfg, _zero_action(env, base), phase=f"{phase}_rollback_wait", step=restore_step,
                trace_rows=trace_rows, alignment=alignment,
                extra={"dual_wrist_probe_axis": candidate.axis, "dual_wrist_probe_sign": candidate.sign, "dual_wrist_probe_rollback": True},
            )
        restored = _augment_dual_contact_state(base, env_index, cfg, restored)
        rollback_ok = bool(
            float(restored.get("active_target_filtered_force_peak_n", 0.0) or 0.0) <= cfg.soft_force_max_n
            and float(restored.get("workspace_clamp_delta_m", 0.0) or 0.0) <= 1.0e-9
        )
        row = {
            "phase": phase, "step": int(step), "probe_axis": candidate.axis, "probe_sign": candidate.sign,
            "amplitude_level": level, "translation_xyz_m": candidate.translation_xyz_m,
            "rotation_rpy_deg": candidate.rotation_rpy_deg,
            "finger3_before_n": baseline_obs.finger3_force_n, "finger4_before_n": baseline_obs.finger4_force_n,
            "finger3_after_n": candidate_obs.finger3_force_n, "finger4_after_n": candidate_obs.finger4_force_n,
            "anchor_finger": regulator.anchor_finger, "opposing_finger": regulator.opposing_finger,
            "object_probe_displacement_m": probe_motion, "object_speed_mps": candidate_obs.object_speed_mps,
            "raw_non_target_peak_n": candidate_obs.non_target_force_n, "workspace_clamp_m": candidate_obs.workspace_clamp_m,
            "safe": safe, "response_score": score, "rollback_ok": rollback_ok, "selected": False,
        }
        response_rows.append(row)
        if not rollback_ok:
            _append_dual_force_response(cfg, row)
            return restored, pair_index + 1, {"termination_reason": "wrist_probe_rollback_failed"}
        state = restored
    selected_profile: dict[str, Any] = {}
    if ranked:
        best_score, best_candidate, _ = max(ranked, key=lambda item: item[0])
        if best_score[0] > 0.0 or best_score[1] >= cfg.dual_contact_stagnation_improvement_n:
            action = _zero_action(env, base)
            best_rot = [math.radians(value) for value in best_candidate.rotation_rpy_deg]
            _set_wrist_delta_action(base, action, env_index, list(best_candidate.translation_xyz_m), best_rot)
            state = _step_direct_action(
                env, base, env_index, cfg, action, phase=f"{phase}_apply", step=step,
                trace_rows=trace_rows, alignment=alignment,
                extra={"dual_wrist_probe_axis": best_candidate.axis, "dual_wrist_probe_sign": best_candidate.sign, "dual_wrist_probe_selected": True},
            )
            state = _augment_dual_contact_state(base, env_index, cfg, state)
            selected_profile = {
                "axis": best_candidate.axis,
                "sign": best_candidate.sign,
                "translation_xyz_m": list(best_candidate.translation_xyz_m),
                "rotation_rpy_deg": list(best_candidate.rotation_rpy_deg),
                "score": list(best_score),
            }
            for row in response_rows:
                row["selected"] = bool(row["probe_axis"] == best_candidate.axis and row["probe_sign"] == best_candidate.sign)
    for row in response_rows:
        _append_dual_force_response(cfg, row)
    return state, pair_index + 1, selected_profile


def _stable_hold(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
) -> dict[str, Any]:
    f3_contacts = 0
    f4_contacts = 0
    simultaneous_contacts = 0
    f3_peak = 0.0
    f4_peak = 0.0
    termination = "stable_contact_hold_completed"
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    for step in range(int(cfg.stable_contact_steps)):
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
        )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        f3_on = f3 >= cfg.contact_threshold_n
        f4_on = f4 >= cfg.contact_threshold_n
        f3_contacts += 1 if f3_on else 0
        f4_contacts += 1 if f4_on else 0
        simultaneous_contacts += 1 if f3_on and f4_on else 0
        vel = _object_velocity(base, env_index, cfg.part)
        if _force_guard_decision(f3, f4, cfg) == "abort":
            termination = "hard_force_abort_during_hold"
            break
        if _force_guard_decision(f3, f4, cfg) == "retreat":
            termination = "soft_force_exceeded_during_hold"
            break
        if _distance(start_obj, state["object_local_pos"]) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_before_close"
            break
        if float(vel.get("linear_velocity_norm", 0.0) or 0.0) > cfg.stable_velocity_limit_mps:
            termination = "object_velocity_not_settled"
            break
    f3_duty = f3_contacts / max(1, int(cfg.stable_contact_steps))
    f4_duty = f4_contacts / max(1, int(cfg.stable_contact_steps))
    simultaneous_duty = simultaneous_contacts / max(1, int(cfg.stable_contact_steps))
    stable = bool(
        simultaneous_contacts >= int(math.ceil(cfg.stable_contact_steps * cfg.stable_contact_duty_ratio))
        and max(f3_peak, f4_peak) < cfg.hard_abort_force_n
        and _distance(start_obj, state["object_local_pos"]) <= cfg.stable_preclose_object_motion_limit_m
    )
    return {
        "stable_contact": stable,
        "finger3_contact_duty_ratio": f3_duty,
        "finger4_contact_duty_ratio": f4_duty,
        "simultaneous_contact_duty_ratio": simultaneous_duty,
        "termination_reason": termination,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
    }


def _opposing_finger_approach(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    anchor_finger: str,
) -> dict[str, Any]:
    if anchor_finger == "both":
        state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
        return {
            "termination_reason": "both_fingers_already_contacting",
            "finger3_force_peak_n": state.get("finger3_target_filtered_force_n", 0.0),
            "finger4_force_peak_n": state.get("finger4_target_filtered_force_n", 0.0),
        }
    opposing = "finger4" if anchor_finger == "finger3" else "finger3"
    f3_peak = 0.0
    f4_peak = 0.0
    termination = "opposing_finger_max_steps"
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    start_obj = list(state.get("object_local_pos", []))
    selected_probe: dict[str, Any] = {}
    selected_probe_age = 0
    anchor_lost_wait = 0
    contact_handoff_count = 0
    for step in range(int(cfg.opposing_finger_max_steps)):
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        anchor_force = f3 if anchor_finger == "finger3" else f4
        opposing_force = f4 if opposing == "finger4" else f3
        if step % 25 == 0:
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "opposing_finger_progress",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "opposing_finger": opposing,
                    "contact_handoff_count": contact_handoff_count,
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                    "finger3_tip_object_distance_m": _endpoint_anchor_tip_distance(state, "finger3")[0],
                    "finger4_tip_object_distance_m": _endpoint_anchor_tip_distance(state, "finger4")[0],
                    "object_displacement_m": _distance(start_obj, state["object_local_pos"]),
                    "selected_action_column": selected_probe.get("action_column", ""),
                    "selected_action_delta": selected_probe.get("delta", ""),
                },
            )
        if f3 >= cfg.contact_threshold_n and f4 >= cfg.contact_threshold_n:
            termination = "two_finger_contact_acquired"
            break
        if anchor_force < cfg.contact_threshold_n:
            if (
                opposing_force >= cfg.contact_threshold_n
                and contact_handoff_count < max(0, int(cfg.opposing_contact_handoff_max))
            ):
                previous_anchor = anchor_finger
                anchor_finger, opposing = opposing, anchor_finger
                contact_handoff_count += 1
                selected_probe = {}
                selected_probe_age = 0
                anchor_lost_wait = 0
                _write_seed_live_progress(
                    cfg,
                    {
                        "phase": phase,
                        "event": "opposing_contact_handoff",
                        "step": int(step),
                        "previous_anchor_finger": previous_anchor,
                        "anchor_finger": anchor_finger,
                        "opposing_finger": opposing,
                        "contact_handoff_count": contact_handoff_count,
                        "finger3_force_n": f3,
                        "finger4_force_n": f4,
                    },
                )
                continue
            if anchor_lost_wait < max(0, int(cfg.opposing_anchor_reacquire_wait_steps)):
                state = _step_direct_action(
                    env,
                    base,
                    env_index,
                    cfg,
                    _zero_action(env, base),
                    phase=f"{phase}_anchor_reacquire_wait",
                    step=anchor_lost_wait,
                    trace_rows=trace_rows,
                    alignment=alignment,
                    extra={
                        "anchor_finger": anchor_finger,
                        "opposing_finger": opposing,
                        "anchor_reacquire_wait": True,
                        "anchor_reacquire_parent_step": int(step),
                        "anchor_reacquire_wait_step": int(anchor_lost_wait),
                        "anchor_force_n": anchor_force,
                        "opposing_force_n": opposing_force,
                    },
                )
                f3_peak = max(f3_peak, float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0))
                f4_peak = max(f4_peak, float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0))
                anchor_lost_wait += 1
                continue
            termination = "anchor_contact_lost_during_opposing_finger"
            break
        anchor_lost_wait = 0
        if _distance(start_obj, state["object_local_pos"]) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_during_opposing_finger"
            break
        vel = _object_velocity(base, env_index, cfg.part)
        if float(vel.get("linear_velocity_norm", 0.0) or 0.0) > cfg.stable_velocity_limit_mps:
            termination = "object_velocity_during_opposing_finger"
            break
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            termination = "hard_force_abort_during_opposing_finger"
            _release_finger_step(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                f"{phase}_release",
                "finger3" if f3 >= f4 else "finger4",
            )
            break
        if guard == "retreat":
            termination = "soft_force_exceeded_during_opposing_finger"
            _release_finger_step(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                f"{phase}_release",
                "finger3" if f3 >= f4 else "finger4",
            )
            break
        reprobe_due = bool(
            not selected_probe
            or selected_probe_age >= max(1, int(cfg.opposing_finger_reprobe_interval_steps))
            or opposing_force >= cfg.contact_threshold_n
        )
        if reprobe_due:
            initial_targets = _hand_audit_snapshot(base, env_index).get("target", [])
            probe = _probe_best_anchor_finger_action(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase}_single_joint_probe",
                step=step,
                anchor_finger=opposing,
                start_obj=start_obj,
                initial_state=state,
                initial_targets=list(initial_targets),
                probe_action_value=float(cfg.opposing_finger_probe_action_value),
                min_improvement_m=float(cfg.opposing_finger_min_improvement_m),
                probe_settle_steps=int(cfg.opposing_finger_probe_settle_steps),
                probe_restore_settle_steps=int(cfg.opposing_finger_probe_restore_settle_steps),
            )
            state = dict(probe.get("final_state") or read_state(base, env_index, cfg.part, cfg.contact_threshold_n))
            f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
            f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
            f3_peak = max(f3_peak, f3)
            f4_peak = max(f4_peak, f4)
            _write_seed_live_progress(
                cfg,
                {
                    "phase": phase,
                    "event": "opposing_finger_probe_selected",
                    "step": int(step),
                    "anchor_finger": anchor_finger,
                    "opposing_finger": opposing,
                    "termination_reason": probe.get("termination_reason", ""),
                    "chosen_action_column": probe.get("action_column", ""),
                    "chosen_local_hand_index": probe.get("local_hand_index", ""),
                    "chosen_delta": probe.get("delta", ""),
                    "chosen_improvement_m": probe.get("improvement_m", ""),
                    "finger3_force_n": f3,
                    "finger4_force_n": f4,
                },
            )
            if f3 >= cfg.contact_threshold_n and f4 >= cfg.contact_threshold_n:
                termination = probe.get("termination_reason") or "two_finger_contact_acquired_by_opposing_probe"
                break
            if bool(probe.get("contact_acquired")):
                termination = str(probe.get("termination_reason") or "opposing_contact_acquired_pending_handoff")
                selected_probe = {}
                selected_probe_age = 0
                continue
            if bool(probe.get("hard_abort")):
                termination = probe.get("termination_reason") or "hard_force_abort_during_opposing_finger_probe"
                break
            if "release" in str(probe.get("termination_reason") or ""):
                termination = probe.get("termination_reason")
                break
            if "object_displacement" in str(probe.get("termination_reason") or ""):
                termination = probe.get("termination_reason")
                break
            if probe.get("action_column") is None:
                termination = probe.get("termination_reason") or "no_guided_opposing_finger_action_reduces_residual"
                break
            selected_probe = dict(probe)
            selected_probe_age = 0
        before_dist, before_residual = _endpoint_anchor_tip_distance(state, opposing)
        action = _zero_action(env, base)
        action[env_index, int(selected_probe["action_column"])] = float(selected_probe["delta"])
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={
                "anchor_finger": anchor_finger,
                "opposing_finger": opposing,
                "opposing_guided_action_column": int(selected_probe["action_column"]),
                "opposing_guided_local_hand_index": int(selected_probe["local_hand_index"]),
                "opposing_guided_delta": float(selected_probe["delta"]),
                "opposing_guided_probe_improvement_m": selected_probe.get("improvement_m", ""),
                "opposing_before_distance_m": before_dist if math.isfinite(before_dist) else "",
                "opposing_before_residual_xyz_m": before_residual,
            },
        )
        after_dist, after_residual = _endpoint_anchor_tip_distance(state, opposing)
        actual_improvement = float(before_dist) - float(after_dist) if math.isfinite(before_dist) and math.isfinite(after_dist) else 0.0
        if trace_rows:
            trace_rows[-1].update(
                {
                    "opposing_after_distance_m": after_dist if math.isfinite(after_dist) else "",
                    "opposing_after_residual_xyz_m": after_residual,
                    "opposing_actual_improvement_m": actual_improvement,
                }
            )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        if f3 >= cfg.contact_threshold_n and f4 >= cfg.contact_threshold_n:
            termination = "two_finger_contact_acquired"
            break
        if actual_improvement <= max(1.0e-8, float(cfg.opposing_finger_min_improvement_m) * 0.25):
            selected_probe = {}
            selected_probe_age = 0
        else:
            selected_probe_age += 1
    return {
        "termination_reason": termination,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
        "final_state": state,
        "final_anchor_finger": anchor_finger,
        "contact_handoff_count": contact_handoff_count,
    }


def _active_dual_contact_acquisition(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    anchor_finger: str,
    allow_wrist: bool,
    initial_selected_actions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = _augment_dual_contact_state(
        base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    )
    start_obj = list(state.get("object_local_pos", []))
    regulator = DualContactRegulator(_dual_regulator_config(cfg), anchor_finger=anchor_finger)
    selected_actions: dict[str, dict[str, Any]] = dict(initial_selected_actions or {})
    wrist_profiles: list[dict[str, Any]] = []
    wrist_probe_cursor = 0
    f3_peak = 0.0
    f4_peak = 0.0
    termination = "dual_contact_regulation_max_steps"
    max_steps = min(int(cfg.opposing_finger_max_steps), int(cfg.dual_contact_max_steps))
    for step in range(max_steps):
        observation = _dual_observation(base, env_index, cfg, state, start_obj, trace_rows)
        command = regulator.update(observation, allow_wrist=bool(allow_wrist))
        f3_peak = max(f3_peak, observation.finger3_force_n)
        f4_peak = max(f4_peak, observation.finger4_force_n)
        if command.formal_finger3_contact and command.formal_finger4_contact:
            termination = "two_finger_contact_acquired"
            break
        if command.state == ABORTED:
            termination = command.termination_reason or "dual_contact_regulator_aborted"
            break
        if command.request_wrist_probe and allow_wrist:
            if wrist_probe_cursor >= len(regulator.config.wrist_axis_order) * 2:
                termination = "bounded_wrist_probe_budget_exhausted"
                break
            state, wrist_probe_cursor, profile = _run_bounded_wrist_probe_pair(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=f"{phase}_wrist_response",
                step=step,
                regulator=regulator,
                state=state,
                start_obj=start_obj,
                probe_pair_index=wrist_probe_cursor,
            )
            if profile.get("termination_reason"):
                termination = str(profile["termination_reason"])
                break
            if profile:
                wrist_profiles.append(profile)
        else:
            state, selected_actions = _execute_regulator_finger_action(
                env,
                base,
                env_index,
                cfg,
                alignment,
                trace_rows,
                phase=phase,
                step=step,
                command=command,
                selected_actions=selected_actions,
                state=state,
                start_obj=start_obj,
            )
        if step % max(1, int(cfg.dual_contact_heartbeat_stride)) == 0:
            _write_dual_heartbeat(cfg, phase=phase, step=step, state=state, command=command, start_obj=start_obj)
        if _distance(start_obj, state.get("object_local_pos", [])) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_during_dual_regulation"
            break
    final_f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    final_f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    return {
        "termination_reason": termination,
        "finger3_force_peak_n": max(f3_peak, final_f3),
        "finger4_force_peak_n": max(f4_peak, final_f4),
        "final_state": state,
        "final_anchor_finger": regulator.anchor_finger,
        "contact_handoff_count": regulator.handoff_count,
        "regulator": regulator,
        "selected_actions": selected_actions,
        "wrist_response_profile": wrist_profiles,
        "wrist_probe_pair_count": wrist_probe_cursor,
    }


def _active_dual_contact_hold(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
    regulator: DualContactRegulator,
    selected_actions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    regulator.set_state(DUAL_CONTACT_REGULATION)
    state = _augment_dual_contact_state(
        base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    )
    hold_start_obj = list(state.get("object_local_pos", []))
    f3_contacts = 0
    f4_contacts = 0
    simultaneous_contacts = 0
    f3_peak = 0.0
    f4_peak = 0.0
    f3_sum = 0.0
    f4_sum = 0.0
    vector_dots: list[float] = []
    termination = "active_dual_contact_hold_completed"
    executed_steps = 0
    for step in range(int(cfg.stable_contact_steps)):
        observation = _dual_observation(base, env_index, cfg, state, start_obj, trace_rows)
        command = regulator.update(observation, allow_wrist=False)
        if command.state == ABORTED:
            termination = command.termination_reason or "active_dual_hold_aborted"
            break
        state, selected_actions = _execute_regulator_finger_action(
            env,
            base,
            env_index,
            cfg,
            alignment,
            trace_rows,
            phase=phase,
            step=step,
            command=command,
            selected_actions=selected_actions,
            state=state,
            start_obj=start_obj,
        )
        executed_steps += 1
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        f3_sum += f3
        f4_sum += f4
        f3_on = f3 > cfg.contact_threshold_n
        f4_on = f4 > cfg.contact_threshold_n
        f3_contacts += int(f3_on)
        f4_contacts += int(f4_on)
        simultaneous_contacts += int(f3_on and f4_on)
        dot = state.get("target_force_vector_normalized_dot", "")
        if dot != "":
            vector_dots.append(float(dot))
        velocity = _object_velocity(base, env_index, cfg.part)
        if max(f3, f4) > cfg.soft_force_max_n:
            termination = "soft_force_exceeded_during_active_hold"
            break
        if _distance(start_obj, state.get("object_local_pos", [])) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_before_close"
            break
        if float(velocity.get("linear_velocity_norm", 0.0) or 0.0) > cfg.stable_velocity_limit_mps:
            termination = "object_velocity_not_settled"
            break
    required = int(math.ceil(cfg.stable_contact_steps * cfg.stable_contact_duty_ratio))
    stable = bool(
        executed_steps == int(cfg.stable_contact_steps)
        and simultaneous_contacts >= required
        and max(f3_peak, f4_peak) < cfg.soft_force_max_n
        and _distance(start_obj, state.get("object_local_pos", [])) <= cfg.stable_preclose_object_motion_limit_m
    )
    return {
        "stable_contact": stable,
        "finger3_contact_duty_ratio": f3_contacts / max(1, executed_steps),
        "finger4_contact_duty_ratio": f4_contacts / max(1, executed_steps),
        "simultaneous_contact_duty_ratio": simultaneous_contacts / max(1, executed_steps),
        "simultaneous_contact_steps": simultaneous_contacts,
        "termination_reason": termination,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
        "finger3_force_mean_n": f3_sum / max(1, executed_steps),
        "finger4_force_mean_n": f4_sum / max(1, executed_steps),
        "object_displacement_during_hold_m": _distance(hold_start_obj, state.get("object_local_pos", [])),
        "object_motion_preferred_limit_met": _distance(hold_start_obj, state.get("object_local_pos", [])) <= cfg.dual_contact_preferred_object_motion_limit_m,
        "target_force_vector_dot_mean": sum(vector_dots) / len(vector_dots) if vector_dots else "",
        "contact_position_source": state.get("contact_position_source", ""),
        "regulator": regulator,
        "selected_actions": selected_actions,
        "final_state": state,
    }


def _release_finger_step(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    phase: str,
    finger: str,
) -> dict[str, Any]:
    open_pose = _hand_pose_list(base, "dex_hand_preshape_pose") or _hand_pose_list(base, "dex_hand_open_pose")
    active_locals = [col - 6 for col in _finger_action_columns(base, finger)]
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    if not open_pose or not active_locals:
        return state
    action = _zero_action(env, base)
    err_peak = _set_hand_target_servo_action(base, action, env_index, open_pose, active_locals, cfg.close_action_value)
    return _step_direct_action(
        env,
        base,
        env_index,
        cfg,
        action,
        phase=phase,
        step=0,
        trace_rows=trace_rows,
        alignment=alignment,
        extra={"released_finger": finger, "release_target_error_peak_rad": err_peak},
    )


def _controlled_close_regulated(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    phase: str,
    start_obj: list[float],
    regulator: DualContactRegulator,
    selected_actions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    regulator.set_state(CONTROLLED_CLOSE)
    state = _augment_dual_contact_state(
        base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    )
    active_locals = sorted(
        set([column - 6 for column in _finger_action_columns(base, "finger3") + _finger_action_columns(base, "finger4")])
    )
    close_pose = _hand_pose_list(base, "dex_hand_close_pose")
    if not close_pose:
        return {"close_completed": False, "termination_reason": "dex_hand_close_pose_unavailable"}
    f3_peak = 0.0
    f4_peak = 0.0
    rolling_dual: list[int] = []
    termination = "regulated_close_max_steps"
    for step in range(int(cfg.close_max_steps)):
        observation = _dual_observation(base, env_index, cfg, state, start_obj, trace_rows)
        command = regulator.update(observation, allow_wrist=False)
        if command.state == ABORTED:
            termination = command.termination_reason or "regulated_close_aborted"
            break
        if command.finger_deltas:
            state, selected_actions = _execute_regulator_finger_action(
                env, base, env_index, cfg, alignment, trace_rows, phase=phase, step=step,
                command=command, selected_actions=selected_actions, state=state, start_obj=start_obj,
            )
        else:
            action = _zero_action(env, base)
            err_peak = _set_hand_target_servo_action(
                base, action, env_index, close_pose, active_locals, cfg.dual_contact_finger_step_rad
            )
            state = _step_direct_action(
                env, base, env_index, cfg, action, phase=phase, step=step,
                trace_rows=trace_rows, alignment=alignment,
                extra={**_regulator_trace_fields(state, command), "close_target_error_peak_rad": err_peak},
            )
            state = _augment_dual_contact_state(base, env_index, cfg, state)
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        rolling_dual.append(int(f3 > cfg.contact_threshold_n and f4 > cfg.contact_threshold_n))
        rolling_dual = rolling_dual[-10:]
        if max(f3, f4) > cfg.soft_force_max_n:
            termination = "soft_force_exceeded_during_regulated_close"
            break
        if _distance(start_obj, state.get("object_local_pos", [])) > cfg.stable_preclose_object_motion_limit_m:
            termination = "object_displacement_during_regulated_close"
            break
        if len(rolling_dual) == 10 and sum(rolling_dual) >= 8:
            termination = "stable_support_close_stopped_early"
            break
    hold = _active_dual_contact_hold(
        env, base, env_index, cfg, alignment, trace_rows,
        phase=f"{phase}_hold", start_obj=start_obj, regulator=regulator, selected_actions=selected_actions,
    )
    return {
        "close_completed": bool(hold.get("stable_contact")),
        "termination_reason": termination if hold.get("stable_contact") else hold.get("termination_reason", termination),
        "close_simultaneous_contact_duty_ratio": hold.get("simultaneous_contact_duty_ratio", 0.0),
        "close_simultaneous_contact_steps": hold.get("simultaneous_contact_steps", 0),
        "finger3_force_peak_n": max(f3_peak, float(hold.get("finger3_force_peak_n", 0.0) or 0.0)),
        "finger4_force_peak_n": max(f4_peak, float(hold.get("finger4_force_peak_n", 0.0) or 0.0)),
        "object_displacement_during_close_hold_m": hold.get("object_displacement_during_hold_m", 0.0),
        "regulator": regulator,
        "selected_actions": selected_actions,
        "final_state": hold.get("final_state", state),
    }


def _controlled_close(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    authority_rows: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    f3_peak = 0.0
    f4_peak = 0.0
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    termination = "close_completed"
    active_locals = sorted(set([col - 6 for col in _finger_action_columns(base, "finger3") + _finger_action_columns(base, "finger4")]))
    close_pose = _hand_pose_list(base, "dex_hand_close_pose")
    if not close_pose:
        return {
            "close_completed": False,
            "termination_reason": "dex_hand_close_pose_unavailable",
            "finger3_force_peak_n": 0.0,
            "finger4_force_peak_n": 0.0,
        }
    for step in range(int(cfg.close_max_steps)):
        action = _zero_action(env, base)
        err_peak = _set_hand_target_servo_action(base, action, env_index, close_pose, active_locals, cfg.close_action_value)
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=phase,
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={"close_target_error_peak_rad": err_peak},
        )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            termination = "hard_force_abort_during_close"
            return {"close_completed": False, "termination_reason": termination, "finger3_force_peak_n": f3_peak, "finger4_force_peak_n": f4_peak}
        if guard == "retreat":
            termination = "soft_force_exceeded_during_close"
            return {"close_completed": False, "termination_reason": termination, "finger3_force_peak_n": f3_peak, "finger4_force_peak_n": f4_peak}
        if err_peak <= 0.01 and f3 >= cfg.contact_threshold_n and f4 >= cfg.contact_threshold_n:
            break
    simultaneous = 0
    for hold_step in range(int(cfg.close_hold_steps)):
        state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            _zero_action(env, base),
            phase=f"{phase}_hold",
            step=hold_step,
            trace_rows=trace_rows,
            alignment=alignment,
        )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        simultaneous += 1 if f3 >= cfg.contact_threshold_n and f4 >= cfg.contact_threshold_n else 0
        if _force_guard_decision(f3, f4, cfg) in {"abort", "retreat"}:
            termination = "force_guard_failed_during_close_hold"
            break
    close_ok = bool(simultaneous >= int(math.ceil(cfg.close_hold_steps * cfg.stable_contact_duty_ratio)))
    return {
        "close_completed": close_ok,
        "termination_reason": termination,
        "close_simultaneous_contact_duty_ratio": simultaneous / max(1, int(cfg.close_hold_steps)),
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
    }


def _slow_lift_regulated(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    start_obj: list[float],
    start_rel: list[float],
    regulator: DualContactRegulator,
    selected_actions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    regulator.set_state(SLOW_LIFT)
    state = _augment_dual_contact_state(
        base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    )
    f3_peak = 0.0
    f4_peak = 0.0
    simultaneous = 0
    executed = 0
    regulation_pause_steps = 0
    termination = "regulated_lift_max_steps"
    for step in range(int(cfg.lift_max_steps)):
        observation = _dual_observation(base, env_index, cfg, state, start_obj, trace_rows)
        command = regulator.update(observation, allow_wrist=False)
        if command.state == ABORTED:
            termination = command.termination_reason or "regulated_lift_aborted"
            break
        if command.finger_deltas or not (
            observation.finger3_force_n > cfg.contact_threshold_n
            and observation.finger4_force_n > cfg.contact_threshold_n
        ):
            regulation_pause_steps += 1
            if regulation_pause_steps > 10:
                termination = "dual_contact_not_restored_during_lift"
                break
            state, selected_actions = _execute_regulator_finger_action(
                env, base, env_index, cfg, alignment, trace_rows, phase="slow_lift_contact_regulation", step=step,
                command=command, selected_actions=selected_actions, state=state, start_obj=start_obj,
            )
            continue
        regulation_pause_steps = 0
        action = _zero_action(env, base)
        delta = [0.0, 0.0, min(0.00025, float(cfg.lift_step_m))]
        _set_wrist_delta_action(base, action, env_index, delta, [0.0, 0.0, 0.0])
        state = _step_direct_action(
            env, base, env_index, cfg, action, phase="slow_lift", step=step,
            trace_rows=trace_rows, alignment=alignment,
            extra={**_regulator_trace_fields(state, command), "regulated_lift_step_m": delta[2]},
        )
        state = _augment_dual_contact_state(base, env_index, cfg, state)
        executed += 1
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        simultaneous += int(f3 > cfg.contact_threshold_n and f4 > cfg.contact_threshold_n)
        if max(f3, f4) > cfg.soft_force_max_n:
            termination = "soft_force_exceeded_during_lift"
            break
        if float(state.get("object_local_pos", [0.0, 0.0, 0.0])[2]) - float(start_obj[2]) >= cfg.lift_target_m:
            termination = "lift_target_reached"
            break
    lift_delta = float(state.get("object_local_pos", [0.0, 0.0, 0.0])[2]) - float(start_obj[2])
    relative = _sub_vec(state.get("object_local_pos", []), state.get("palm_local_pos", []))
    relative_drift = _distance(start_rel, relative)
    duty = simultaneous / max(1, executed)
    table_top = cfg.canonical_support_pose.get("table_top_z_m", "")
    try:
        above_table = float(state["object_local_pos"][2]) > float(table_top) + 0.010
    except Exception:
        above_table = False
    success = bool(
        termination == "lift_target_reached"
        and lift_delta >= 0.010
        and above_table
        and duty >= 0.8
        and relative_drift <= 0.010
    )
    return {
        "lift_success": success,
        "termination_reason": termination,
        "object_lift_delta_z_m": lift_delta,
        "object_above_table_after_lift": above_table,
        "relative_pose_drift_m": relative_drift,
        "lift_simultaneous_contact_duty_ratio": duty,
        "lift_simultaneous_contact_steps": simultaneous,
        "lift_motion_step_count": executed,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
    }


def _slow_lift(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    *,
    start_obj: list[float],
    start_rel: list[float],
) -> dict[str, Any]:
    f3_peak = 0.0
    f4_peak = 0.0
    simultaneous_contacts = 0
    contact_steps = 0
    state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    termination = "lift_max_steps"
    for step in range(int(cfg.lift_max_steps)):
        delta = [0.0, 0.0, float(cfg.lift_step_m)]
        action = _zero_action(env, base)
        _set_wrist_delta_action(base, action, env_index, delta, [0.0, 0.0, 0.0])
        state = _step_direct_action(
            env, base, env_index, cfg, action, phase="slow_lift", step=step, trace_rows=trace_rows, alignment=alignment
        )
        f3 = float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0)
        f4 = float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0)
        f3_peak = max(f3_peak, f3)
        f4_peak = max(f4_peak, f4)
        contact_steps += 1
        simultaneous_contacts += 1 if f3 >= cfg.contact_threshold_n and f4 >= cfg.contact_threshold_n else 0
        guard = _force_guard_decision(f3, f4, cfg)
        if guard == "abort":
            termination = "hard_force_abort_during_lift"
            break
        if guard == "retreat":
            termination = "soft_force_exceeded_during_lift"
            break
        if max(f3, f4) < cfg.contact_threshold_n:
            termination = "target_contact_lost_during_lift"
            break
        if float(state["object_local_pos"][2]) - float(start_obj[2]) >= cfg.lift_target_m:
            termination = "lift_target_reached"
            break
    rel = _sub_vec(state["object_local_pos"], state["palm_local_pos"])
    rel_drift = _distance(start_rel, rel)
    lift_delta = float(state["object_local_pos"][2]) - float(start_obj[2])
    simultaneous_duty = simultaneous_contacts / max(1, contact_steps)
    table_top = cfg.canonical_support_pose.get("table_top_z_m", "")
    try:
        above_table = float(state["object_local_pos"][2]) > float(table_top) + 0.010
    except Exception:
        above_table = True
    success = bool(
        lift_delta >= 0.010
        and above_table
        and rel_drift <= 0.010
        and simultaneous_duty >= 0.5
        and termination == "lift_target_reached"
    )
    return {
        "lift_success": success,
        "termination_reason": termination,
        "object_lift_delta_z_m": lift_delta,
        "object_above_table_after_lift": above_table,
        "relative_pose_drift_m": rel_drift,
        "lift_simultaneous_contact_duty_ratio": simultaneous_duty,
        "finger3_force_peak_n": f3_peak,
        "finger4_force_peak_n": f4_peak,
    }


def _step_direct_action(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    action: Any,
    *,
    phase: str,
    step: int,
    trace_rows: list[dict[str, Any]],
    alignment: AlignmentRecorder | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(extra or {})
    before = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    action_rows = _action_rows(action)
    target_action = action_rows[env_index] if 0 <= env_index < len(action_rows) else []
    frame_before = alignment.frame_count() if alignment is not None else -1
    env.step(action)
    _refresh_runtime(base)
    after = _augment_dual_contact_state(
        base, env_index, cfg, read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    )
    if alignment is not None:
        try:
            alignment.record_contacts(phase, step, after, base)
        except Exception:
            pass
    frame_after = alignment.frame_count() if alignment is not None else -1
    video_frame = int(frame_after - 1) if frame_after > frame_before else -1
    raw_peaks = _raw_contact_peaks(alignment, phase, step) if alignment is not None else {}
    f3_force = float(after.get("finger3_target_filtered_force_n", 0.0) or 0.0)
    f4_force = float(after.get("finger4_target_filtered_force_n", 0.0) or 0.0)
    ctrl_target = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    wrist_snapshot = _wrist_pose_snapshot(base, env_index, state=after)
    row = {
        "phase": phase,
        "step": int(step),
        "env_index": int(env_index),
        "action_dim": len(target_action),
        "direct_26d_action_used": True,
        "action_norm": _norm(target_action),
        "wrist_action_xyz": _json(target_action[:3]),
        "wrist_action_rpy_axis": _json(target_action[3:6]),
        "hand_action_nonzero_count": sum(1 for value in target_action[6:26] if abs(float(value)) > 1.0e-9),
        "finger3_target_filtered_force_n": f3_force,
        "finger4_target_filtered_force_n": f4_force,
        "finger3_target_filtered_force_xyz_n": _json(after.get("finger3_target_filtered_force_xyz_n", [])),
        "finger4_target_filtered_force_xyz_n": _json(after.get("finger4_target_filtered_force_xyz_n", [])),
        "finger3_contact_position_proxy_xyz_m": _json(after.get("finger3_contact_position_proxy_xyz_m", [])),
        "finger4_contact_position_proxy_xyz_m": _json(after.get("finger4_contact_position_proxy_xyz_m", [])),
        "contact_position_source": after.get("contact_position_source", ""),
        "target_force_vector_normalized_dot": after.get("target_force_vector_normalized_dot", ""),
        "force_guard_decision": _force_guard_decision(f3_force, f4_force, cfg),
        "active_target_filtered_force_peak_n": after.get("active_target_filtered_force_peak_n", 0.0),
        "finger3_unfiltered_force_n": after.get("finger3_unfiltered_force_n", 0.0),
        "finger4_unfiltered_force_n": after.get("finger4_unfiltered_force_n", 0.0),
        "object_x": after["object_local_pos"][0],
        "object_y": after["object_local_pos"][1],
        "object_z": after["object_local_pos"][2],
        "object_step_delta_m": _distance(before["object_local_pos"], after["object_local_pos"]),
        "palm_x": after["palm_local_pos"][0],
        "palm_y": after["palm_local_pos"][1],
        "palm_z": after["palm_local_pos"][2],
        "ctrl_target_x": ctrl_target[0] if len(ctrl_target) > 0 else "",
        "ctrl_target_y": ctrl_target[1] if len(ctrl_target) > 1 else "",
        "ctrl_target_z": ctrl_target[2] if len(ctrl_target) > 2 else "",
        "ctrl_target_to_palm_error_m": _distance(ctrl_target, after["palm_local_pos"]),
        "ctrl_target_quat_wxyz": _json(wrist_snapshot.get("control_dex_quat", [])),
        "actual_palm_quat_wxyz": _json(wrist_snapshot.get("actual_dex_quat", [])),
        "ctrl_target_to_palm_angle_error_deg": _rad_to_deg_or_blank(
            wrist_snapshot.get("control_to_actual_angle_error_rad", "")
        ),
        "wrist_joint_target": _json(wrist_snapshot.get("wrist_joint_target", [])),
        "wrist_joint_application_target": _json(wrist_snapshot.get("wrist_joint_application_target", [])),
        "wrist_joint_articulation_target": _json(wrist_snapshot.get("wrist_joint_articulation_target", [])),
        "wrist_joint_sim_target": _json(wrist_snapshot.get("wrist_joint_sim_target", [])),
        "wrist_joint_actual": _json(wrist_snapshot.get("wrist_joint_actual", [])),
        "wrist_joint_error": _json(wrist_snapshot.get("wrist_joint_error", [])),
        "wrist_joint_velocity": _json(wrist_snapshot.get("wrist_joint_velocity", [])),
        "wrist_joint_computed_torque": _json(wrist_snapshot.get("wrist_joint_computed_torque", [])),
        "wrist_joint_applied_torque": _json(wrist_snapshot.get("wrist_joint_applied_torque", [])),
        "wrist_joint_velocity_target": _json(wrist_snapshot.get("wrist_joint_velocity_target", [])),
        "wrist_joint_effort_target": _json(wrist_snapshot.get("wrist_joint_effort_target", [])),
        "wrist_joint_effort_limit_runtime": _json(wrist_snapshot.get("wrist_joint_effort_limit_runtime", [])),
        "wrist_joint_velocity_limit_runtime": _json(wrist_snapshot.get("wrist_joint_velocity_limit_runtime", [])),
        "wrist_joint_stiffness_runtime": _json(wrist_snapshot.get("wrist_joint_stiffness_runtime", [])),
        "wrist_joint_damping_runtime": _json(wrist_snapshot.get("wrist_joint_damping_runtime", [])),
        "target_chain_application_to_articulation_error_peak": wrist_snapshot.get(
            "target_chain_application_to_articulation_error_peak", ""
        ),
        "target_chain_application_to_sim_error_peak": wrist_snapshot.get("target_chain_application_to_sim_error_peak", ""),
        "target_chain_articulation_to_sim_error_peak": wrist_snapshot.get("target_chain_articulation_to_sim_error_peak", ""),
        "wrist_joint_saturation_flags": _json(wrist_snapshot.get("wrist_joint_saturation_flags", [])),
        "actual_joint_fk_to_runtime_body_position_error_m": wrist_snapshot.get(
            "actual_joint_fk_to_runtime_body_position_error_m", ""
        ),
        "actual_joint_fk_to_runtime_body_angle_error_deg": _rad_to_deg_or_blank(
            wrist_snapshot.get("actual_joint_fk_to_runtime_body_angle_error_rad", "")
        ),
        "control_actual_offset_contribution_m": wrist_snapshot.get("control_actual_offset_contribution_m", ""),
        "wrist_target_pre_clamp_xyz": _json(after.get("wrist_target_pre_clamp_xyz", [])),
        "wrist_target_post_clamp_xyz": _json(after.get("wrist_target_post_clamp_xyz", [])),
        "wrist_target_delta_xyz": _json(after.get("wrist_target_delta_xyz", [])),
        "workspace_clamp_vector_xyz": _json(after.get("workspace_clamp_vector_xyz", [])),
        "workspace_clamp_dominant_axis": after.get("workspace_clamp_dominant_axis", ""),
        "finger3_tip_x": after["finger3_tip_local_pos"][0],
        "finger3_tip_y": after["finger3_tip_local_pos"][1],
        "finger3_tip_z": after["finger3_tip_local_pos"][2],
        "finger4_tip_x": after["finger4_tip_local_pos"][0],
        "finger4_tip_y": after["finger4_tip_local_pos"][1],
        "finger4_tip_z": after["finger4_tip_local_pos"][2],
        "workspace_clamp_delta_m": after.get("workspace_clamp_delta_m", 0.0),
        "table_barrier_delta_z_m": after.get("table_barrier_delta_z_m", 0.0),
        "canonical_table_top_z_m": cfg.canonical_support_pose.get("table_top_z_m", ""),
        "floating_table_z_est_for_audit_only": _floating_table_z_audit(base, env_index),
        **_v2_table_override_audit(base, env_index),
        "target_contact_acquired": _target_contact_acquired(after, cfg),
        "hard_force_abort": float(after.get("active_target_filtered_force_peak_n", 0.0) or 0.0) >= cfg.hard_abort_force_n,
        "video_frame_index": video_frame,
        **raw_peaks,
        **extra,
    }
    trace_rows.append(_plain(row))
    return after


def _zero_action(env: Any, base: Any) -> Any:
    width = int(_action_dim(env, base))
    num_envs = int(getattr(base, "num_envs", 1) or 1)
    if torch is None:
        return [[0.0] * width for _ in range(num_envs)]
    return torch.zeros((num_envs, width), dtype=torch.float32, device=getattr(base, "device", None))


def _set_wrist_delta_action(base: Any, action: Any, env_index: int, delta_xyz: list[float], delta_rot: list[float]) -> None:
    pos_scale = abs(_scalar_attr(base, "pos_threshold", 1.0)) * abs(_scalar_attr(base, "floating_action_pos_scale", 1.0))
    rot_scale = abs(_scalar_attr(base, "rot_threshold", 1.0)) * abs(_scalar_attr(base, "floating_action_rot_scale", 1.0))
    pos_axis = _axis_scale_attr(base, "floating_action_pos_axis_scale")
    rot_axis = _axis_scale_attr(base, "floating_action_rot_axis_scale")
    for axis in range(3):
        denom = max(1.0e-9, pos_scale * abs(float(pos_axis[axis])))
        action[env_index, axis] = max(-1.0, min(1.0, float(delta_xyz[axis]) / denom))
    for axis in range(3):
        denom = max(1.0e-9, rot_scale * abs(float(rot_axis[axis])))
        action[env_index, 3 + axis] = max(-1.0, min(1.0, float(delta_rot[axis]) / denom))


def _retreat(
    env: Any,
    base: Any,
    env_index: int,
    cfg: Screw1GraspBaselineV2Config,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    phase: str,
    last_move: list[float],
    last_rot_move: list[float] | None = None,
) -> dict[str, Any]:
    retreat = [-float(value) for value in last_move]
    retreat_rot = [-float(value) for value in (last_rot_move or [0.0, 0.0, 0.0])]
    if _norm(retreat) <= 1.0e-9:
        retreat = [0.0, 0.0, float(cfg.seed_servo_pos_step_m)]
    final_state = read_state(base, env_index, cfg.part, cfg.contact_threshold_n)
    prev_force = max(
        float(final_state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
        float(final_state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
    )
    termination = "retreat_max_steps"
    for step in range(int(cfg.hard_abort_retreat_steps)):
        action = _zero_action(env, base)
        _set_wrist_delta_action(base, action, env_index, retreat, retreat_rot)
        final_state = _step_direct_action(
            env,
            base,
            env_index,
            cfg,
            action,
            phase=f"{phase}_retreat",
            step=step,
            trace_rows=trace_rows,
            alignment=alignment,
            extra={"retreat_after_hard_abort": True},
        )
        force = max(
            float(final_state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
            float(final_state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
        )
        if force < cfg.soft_force_min_n:
            termination = "retreat_force_safe"
            break
        if force > prev_force + 1.0e-6:
            termination = "retreat_force_increased_abort"
            break
        prev_force = force
    return {
        "final_state": final_state,
        "retreat_steps": int(step + 1 if "step" in locals() else 0),
        "retreat_termination_reason": termination,
        "retreat_final_force_n": prev_force,
    }


def _capture_fingerprint(base: Any, env_index: int, cfg: Screw1GraspBaselineV2Config, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_local_pos": list(state.get("object_local_pos", [])),
        "object_quat_wxyz": list(state.get("object_quat_wxyz", [])),
        "object_velocity": _object_velocity(base, env_index, cfg.part),
        "palm_local_pos": list(state.get("palm_local_pos", [])),
        "palm_quat_wxyz": list(state.get("palm_quat_wxyz", [])),
        "finger3_tip_local_pos": list(state.get("finger3_tip_local_pos", [])),
        "finger4_tip_local_pos": list(state.get("finger4_tip_local_pos", [])),
        "hand_joint_pos": _hand_audit_snapshot(base, env_index).get("actual", []),
        "hand_target_pos": _hand_audit_snapshot(base, env_index).get("target", []),
        "finger3_target_filtered_force_n": state.get("finger3_target_filtered_force_n", 0.0),
        "finger4_target_filtered_force_n": state.get("finger4_target_filtered_force_n", 0.0),
        "random_seed": int(cfg.deterministic_seed),
    }


def _fingerprint_summary(prefix: str, fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_object_pos": fingerprint.get("object_local_pos", []),
        f"{prefix}_palm_pos": fingerprint.get("palm_local_pos", []),
        f"{prefix}_finger3_tip_pos": fingerprint.get("finger3_tip_local_pos", []),
        f"{prefix}_finger4_tip_pos": fingerprint.get("finger4_tip_local_pos", []),
    }


def _fingerprint_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = ("object_local_pos", "palm_local_pos", "finger3_tip_local_pos", "finger4_tip_local_pos")
    return max([0.0, *[_distance(a.get(key, []), b.get(key, [])) for key in keys]])


def _hand_audit_snapshot(base: Any, env_index: int) -> dict[str, Any]:
    joint_pos = getattr(base, "joint_pos", None)
    target = getattr(base, "ctrl_target_hand_joint_pos", None)
    indices = getattr(base, "dex_hand_joint_indices", None)
    actual = []
    if torch is not None and torch.is_tensor(joint_pos) and torch.is_tensor(indices):
        try:
            actual = [float(v) for v in joint_pos[env_index, indices].detach().cpu().tolist()]
        except Exception:
            actual = []
    if torch is not None and torch.is_tensor(target):
        try:
            target_values = [float(v) for v in target[env_index].detach().cpu().tolist()]
        except Exception:
            target_values = []
    else:
        target_values = []
    return {"actual": actual, "target": target_values}


def _required_finger_joint_specs(base: Any, joint_names: list[str]) -> list[dict[str, Any]]:
    dex_indices = getattr(base, "dex_hand_joint_indices", None)
    dex_index_list: list[int] = []
    if torch is not None and torch.is_tensor(dex_indices):
        try:
            dex_index_list = [int(value) for value in dex_indices.detach().cpu().reshape(-1).tolist()]
        except Exception:
            dex_index_list = []
    elif isinstance(dex_indices, (list, tuple)):
        try:
            dex_index_list = [int(value) for value in dex_indices]
        except Exception:
            dex_index_list = []
    specs: list[dict[str, Any]] = []
    for finger in (3, 4):
        for joint in (1, 2, 3, 4):
            name = f"right_finger{finger}_joint{joint}"
            local = joint_names.index(name) if name in joint_names else -1
            actual_robot_index = dex_index_list[local] if 0 <= local < len(dex_index_list) else -1
            specs.append(
                {
                    "finger": f"finger{finger}",
                    "joint_number": joint,
                    "joint_name": name,
                    "local_hand_index": local,
                    "env_action_column": 6 + local if local >= 0 else -1,
                    "action_column": 6 + local if local >= 0 else -1,
                    "actual_robot_joint_index": actual_robot_index,
                    "joint_name_found": local >= 0,
                }
            )
    return specs


def _finger_action_columns(base: Any, finger: str) -> list[int]:
    names = [str(item) for item in list(getattr(base, "dex_hand_joint_names", []) or [])]
    try:
        finger_num = int(str(finger).replace("finger", ""))
    except Exception:
        return []
    cols = []
    for joint in (1, 2, 3, 4):
        name = f"right_finger{finger_num}_joint{joint}"
        if name in names:
            cols.append(6 + names.index(name))
    return cols


def _hand_pose_list(base: Any, attr_name: str) -> list[float]:
    value = getattr(base, attr_name, None)
    if torch is not None and torch.is_tensor(value):
        try:
            return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except Exception:
            return []
    return []


def _wrist_pose_snapshot(
    base: Any,
    env_index: int,
    *,
    state: dict[str, Any] | None = None,
    desired_pos: list[float] | None = None,
    desired_quat: list[float] | None = None,
) -> dict[str, Any]:
    if state is None:
        try:
            state = read_state(base, env_index, "Screw1", 0.05)
        except Exception:
            state = {}
    actual_pos = list(state.get("palm_local_pos", []))
    actual_quat = _quat_normalize_wxyz(state.get("palm_quat_wxyz", []))
    control_pos = _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_pos", None), env_index)
    control_quat = _quat_normalize_wxyz(
        state.get("ctrl_target_palm_quat_wxyz", [])
        or _tensor_vec(getattr(base, "ctrl_target_fingertip_midpoint_quat", None), env_index, width=4)
    )
    wrist = _wrist_joint_snapshot(base, env_index)
    offset = _flat_vec(getattr(base, "dex_grasp_frame_local_pos", None), 3)
    actual_origin = _sub_vec(actual_pos, _quat_apply_wxyz(actual_quat, offset))
    control_origin = _sub_vec(control_pos, _quat_apply_wxyz(control_quat, offset))
    out: dict[str, Any] = {
        "desired_dex_pos": list(desired_pos or []),
        "desired_dex_quat": list(_quat_normalize_wxyz(desired_quat) if desired_quat is not None else []),
        "control_dex_pos": control_pos,
        "control_dex_quat": control_quat,
        "actual_dex_pos": actual_pos,
        "actual_dex_quat": actual_quat,
        "control_palm_origin_pos": control_origin,
        "actual_palm_origin_pos": actual_origin,
        "dex_grasp_frame_local_offset_xyz": offset,
        "control_to_actual_position_error_m": _distance(control_pos, actual_pos),
        "control_to_actual_angle_error_rad": _quat_angle_delta_wxyz(control_quat, actual_quat),
        "control_to_actual_tracking_axis_angle": _quat_error_axis_angle_wxyz(actual_quat, control_quat),
        **wrist,
    }
    if desired_pos is not None and desired_quat is not None:
        desired_pos = [float(value) for value in list(desired_pos)[:3]]
        desired_quat = _quat_normalize_wxyz(desired_quat)
        desired_origin = _sub_vec(desired_pos, _quat_apply_wxyz(desired_quat, offset))
        out.update(
            {
                "desired_palm_origin_pos": desired_origin,
                "target_to_control_position_error_m": _distance(desired_pos, control_pos),
                "target_to_actual_position_error_m": _distance(desired_pos, actual_pos),
                "target_to_control_angle_error_rad": _quat_angle_delta_wxyz(desired_quat, control_quat),
                "target_to_actual_angle_error_rad": _quat_angle_delta_wxyz(desired_quat, actual_quat),
                "target_to_control_command_axis_angle": _quat_error_axis_angle_wxyz(control_quat, desired_quat),
                "target_to_actual_axis_angle": _quat_error_axis_angle_wxyz(actual_quat, desired_quat),
                "desired_actual_offset_contribution_m": _distance(
                    _quat_apply_wxyz(desired_quat, offset),
                    _quat_apply_wxyz(actual_quat, offset),
                ),
                "control_actual_offset_contribution_m": _distance(
                    _quat_apply_wxyz(control_quat, offset),
                    _quat_apply_wxyz(actual_quat, offset),
                ),
            }
        )
        target_joints = _wrist_ik_target_values(base, env_index, desired_pos, desired_quat)
        if target_joints:
            predicted = _wrist_joint_values_to_dex_pose(base, target_joints)
            out.update(
                {
                    "desired_wrist_joint_target_from_ik": target_joints,
                    "desired_to_joint_target_roundtrip_position_error_m": _distance(
                        desired_pos, predicted.get("dex_pos", [])
                    ),
                    "desired_to_joint_target_roundtrip_angle_error_rad": _quat_angle_delta_wxyz(
                        desired_quat, predicted.get("dex_quat", [])
                    ),
                    "desired_to_joint_target_roundtrip_predicted_dex_pos": predicted.get("dex_pos", []),
                    "desired_to_joint_target_roundtrip_predicted_dex_quat": predicted.get("dex_quat", []),
                    "desired_to_joint_target_roundtrip_palm_origin_pos": predicted.get("palm_origin_pos", []),
                }
            )
    actual_joints = list(wrist.get("wrist_joint_actual", []))
    if actual_joints:
        predicted_actual = _wrist_joint_values_to_dex_pose(base, actual_joints)
        out.update(
            {
                "actual_joint_fk_to_runtime_body_position_error_m": _distance(
                    predicted_actual.get("dex_pos", []), actual_pos
                ),
                "actual_joint_fk_to_runtime_body_angle_error_rad": _quat_angle_delta_wxyz(
                    predicted_actual.get("dex_quat", []), actual_quat
                ),
                "actual_joint_fk_predicted_dex_pos": predicted_actual.get("dex_pos", []),
                "actual_joint_fk_predicted_dex_quat": predicted_actual.get("dex_quat", []),
                "actual_joint_fk_palm_origin_pos": predicted_actual.get("palm_origin_pos", []),
            }
        )
    target_joints = list(wrist.get("wrist_joint_target", []))
    if target_joints:
        predicted_control = _wrist_joint_values_to_dex_pose(base, target_joints)
        out.update(
            {
                "control_joint_fk_to_control_target_position_error_m": _distance(
                    predicted_control.get("dex_pos", []), control_pos
                ),
                "control_joint_fk_to_control_target_angle_error_rad": _quat_angle_delta_wxyz(
                    predicted_control.get("dex_quat", []), control_quat
                ),
                "control_joint_fk_predicted_dex_pos": predicted_control.get("dex_pos", []),
                "control_joint_fk_predicted_dex_quat": predicted_control.get("dex_quat", []),
            }
        )
    return out


def _wrist_row_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "desired_dex_pos": _json(snapshot.get("desired_dex_pos", [])),
        "desired_dex_quat": _json(snapshot.get("desired_dex_quat", [])),
        "control_dex_pos": _json(snapshot.get("control_dex_pos", [])),
        "control_dex_quat": _json(snapshot.get("control_dex_quat", [])),
        "actual_dex_pos": _json(snapshot.get("actual_dex_pos", [])),
        "actual_dex_quat": _json(snapshot.get("actual_dex_quat", [])),
        "control_palm_origin_pos": _json(snapshot.get("control_palm_origin_pos", [])),
        "actual_palm_origin_pos": _json(snapshot.get("actual_palm_origin_pos", [])),
        "dex_grasp_frame_local_offset_xyz": _json(snapshot.get("dex_grasp_frame_local_offset_xyz", [])),
        "wrist_joint_names": _json(snapshot.get("wrist_joint_names", [])),
        "wrist_joint_indices": _json(snapshot.get("wrist_joint_indices", [])),
        "wrist_joint_application_target": _json(snapshot.get("wrist_joint_application_target", [])),
        "wrist_joint_articulation_target": _json(snapshot.get("wrist_joint_articulation_target", [])),
        "wrist_joint_sim_target": _json(snapshot.get("wrist_joint_sim_target", [])),
        "wrist_joint_target": _json(snapshot.get("wrist_joint_target", [])),
        "wrist_joint_actual": _json(snapshot.get("wrist_joint_actual", [])),
        "wrist_joint_velocity": _json(snapshot.get("wrist_joint_velocity", [])),
        "wrist_joint_computed_torque": _json(snapshot.get("wrist_joint_computed_torque", [])),
        "wrist_joint_applied_torque": _json(snapshot.get("wrist_joint_applied_torque", [])),
        "wrist_joint_velocity_target": _json(snapshot.get("wrist_joint_velocity_target", [])),
        "wrist_joint_effort_target": _json(snapshot.get("wrist_joint_effort_target", [])),
        "wrist_joint_effort_limit_runtime": _json(snapshot.get("wrist_joint_effort_limit_runtime", [])),
        "wrist_joint_velocity_limit_runtime": _json(snapshot.get("wrist_joint_velocity_limit_runtime", [])),
        "wrist_joint_stiffness_runtime": _json(snapshot.get("wrist_joint_stiffness_runtime", [])),
        "wrist_joint_damping_runtime": _json(snapshot.get("wrist_joint_damping_runtime", [])),
        "wrist_joint_error": _json(snapshot.get("wrist_joint_error", [])),
        "wrist_joint_lower_limits": _json(snapshot.get("wrist_joint_lower_limits", [])),
        "wrist_joint_upper_limits": _json(snapshot.get("wrist_joint_upper_limits", [])),
        "wrist_joint_lower_margin": _json(snapshot.get("wrist_joint_lower_margin", [])),
        "wrist_joint_upper_margin": _json(snapshot.get("wrist_joint_upper_margin", [])),
        "wrist_joint_saturation_flags": _json(snapshot.get("wrist_joint_saturation_flags", [])),
        "target_chain_application_to_articulation_error_peak": snapshot.get(
            "target_chain_application_to_articulation_error_peak", ""
        ),
        "target_chain_application_to_sim_error_peak": snapshot.get("target_chain_application_to_sim_error_peak", ""),
        "target_chain_articulation_to_sim_error_peak": snapshot.get("target_chain_articulation_to_sim_error_peak", ""),
        "articulation_target_available": snapshot.get("articulation_target_available", ""),
        "simulation_target_available": snapshot.get("simulation_target_available", ""),
        "computed_torque_available": snapshot.get("computed_torque_available", ""),
        "applied_torque_available": snapshot.get("applied_torque_available", ""),
        "target_to_control_position_error_m": snapshot.get("target_to_control_position_error_m", ""),
        "control_to_actual_position_error_m": snapshot.get("control_to_actual_position_error_m", ""),
        "target_to_actual_position_error_m": snapshot.get("target_to_actual_position_error_m", ""),
        "target_to_control_angle_error_deg": _rad_to_deg_or_blank(snapshot.get("target_to_control_angle_error_rad", "")),
        "control_to_actual_angle_error_deg": _rad_to_deg_or_blank(snapshot.get("control_to_actual_angle_error_rad", "")),
        "target_to_actual_angle_error_deg": _rad_to_deg_or_blank(snapshot.get("target_to_actual_angle_error_rad", "")),
        "target_to_control_command_axis_angle": _json(snapshot.get("target_to_control_command_axis_angle", [])),
        "control_to_actual_tracking_axis_angle": _json(snapshot.get("control_to_actual_tracking_axis_angle", [])),
        "desired_actual_offset_contribution_m": snapshot.get("desired_actual_offset_contribution_m", ""),
        "control_actual_offset_contribution_m": snapshot.get("control_actual_offset_contribution_m", ""),
        "desired_to_joint_target_roundtrip_position_error_m": snapshot.get(
            "desired_to_joint_target_roundtrip_position_error_m", ""
        ),
        "desired_to_joint_target_roundtrip_angle_error_deg": _rad_to_deg_or_blank(
            snapshot.get("desired_to_joint_target_roundtrip_angle_error_rad", "")
        ),
        "actual_joint_fk_to_runtime_body_position_error_m": snapshot.get(
            "actual_joint_fk_to_runtime_body_position_error_m", ""
        ),
        "actual_joint_fk_to_runtime_body_angle_error_deg": _rad_to_deg_or_blank(
            snapshot.get("actual_joint_fk_to_runtime_body_angle_error_rad", "")
        ),
        "control_joint_fk_to_control_target_position_error_m": snapshot.get(
            "control_joint_fk_to_control_target_position_error_m", ""
        ),
        "control_joint_fk_to_control_target_angle_error_deg": _rad_to_deg_or_blank(
            snapshot.get("control_joint_fk_to_control_target_angle_error_rad", "")
        ),
        "wrist_yaw_clamp_delta_rad": snapshot.get("wrist_yaw_clamp_delta_rad", ""),
    }


def _wrist_joint_snapshot(base: Any, env_index: int) -> dict[str, Any]:
    indices = _tensor_int_list(getattr(base, "floating_wrist_joint_indices", None))
    names = [str(value) for value in list(getattr(base, "floating_wrist_joint_names", []) or [])]
    application_target = _indexed_tensor_row(getattr(base, "ctrl_target_joint_pos", None), env_index, indices)
    target = list(application_target)
    robot = getattr(base, "_robot", None)
    data = getattr(robot, "data", None)
    articulation_target = _indexed_tensor_row(getattr(data, "joint_pos_target", None), env_index, indices)
    sim_target = _indexed_tensor_row(getattr(robot, "_joint_pos_target_sim", None), env_index, indices)
    computed_torque = _indexed_tensor_row(getattr(data, "computed_torque", None), env_index, indices)
    applied_torque = _indexed_tensor_row(getattr(data, "applied_torque", None), env_index, indices)
    velocity_target = _indexed_tensor_row(getattr(data, "joint_vel_target", None), env_index, indices)
    effort_target = _indexed_tensor_row(getattr(data, "joint_effort_target", None), env_index, indices)
    actual = _indexed_tensor_row(getattr(base, "joint_pos", None), env_index, indices)
    velocity = _indexed_tensor_row(getattr(base, "joint_vel", None), env_index, indices)
    lower = _tensor_flat_list(getattr(base, "floating_wrist_joint_lower_limits", None))
    upper = _tensor_flat_list(getattr(base, "floating_wrist_joint_upper_limits", None))
    error = [float(target[i]) - float(actual[i]) for i in range(min(len(target), len(actual)))]
    lower_margin = [
        float(actual[i]) - float(lower[i]) if i < len(actual) and i < len(lower) else ""
        for i in range(max(len(actual), len(lower)))
    ]
    upper_margin = [
        float(upper[i]) - float(actual[i]) if i < len(actual) and i < len(upper) else ""
        for i in range(max(len(actual), len(upper)))
    ]
    saturation = []
    for i, value in enumerate(target):
        lo = float(lower[i]) if i < len(lower) else -math.inf
        hi = float(upper[i]) if i < len(upper) else math.inf
        saturation.append(bool(value <= lo + 1.0e-5 or value >= hi - 1.0e-5))
    out = {
        "wrist_joint_names": names,
        "wrist_joint_indices": indices,
        "wrist_joint_application_target": application_target,
        "wrist_joint_articulation_target": articulation_target,
        "wrist_joint_sim_target": sim_target,
        "wrist_joint_target": target,
        "wrist_joint_actual": actual,
        "wrist_joint_velocity": velocity,
        "wrist_joint_computed_torque": computed_torque,
        "wrist_joint_applied_torque": applied_torque,
        "wrist_joint_velocity_target": velocity_target,
        "wrist_joint_effort_target": effort_target,
        "wrist_joint_effort_limit_runtime": _actuator_param_by_joint(robot, names, indices, "effort_limit"),
        "wrist_joint_velocity_limit_runtime": _actuator_param_by_joint(robot, names, indices, "velocity_limit"),
        "wrist_joint_stiffness_runtime": _actuator_param_by_joint(robot, names, indices, "stiffness"),
        "wrist_joint_damping_runtime": _actuator_param_by_joint(robot, names, indices, "damping"),
        "wrist_joint_error": error,
        "wrist_joint_lower_limits": lower,
        "wrist_joint_upper_limits": upper,
        "wrist_joint_lower_margin": lower_margin,
        "wrist_joint_upper_margin": upper_margin,
        "wrist_joint_saturation_flags": saturation,
        "target_chain_application_to_articulation_error_peak": _max_abs_delta(application_target, articulation_target),
        "target_chain_application_to_sim_error_peak": _max_abs_delta(application_target, sim_target),
        "target_chain_articulation_to_sim_error_peak": _max_abs_delta(articulation_target, sim_target),
        "articulation_target_available": bool(articulation_target),
        "simulation_target_available": bool(sim_target),
        "computed_torque_available": bool(computed_torque),
        "applied_torque_available": bool(applied_torque),
    }
    yaw = getattr(base, "floating_wrist_yaw_clamp_delta", None)
    if torch is not None and torch.is_tensor(yaw):
        try:
            out["wrist_yaw_clamp_delta_rad"] = float(yaw[int(env_index)].detach().cpu().item())
        except Exception:
            out["wrist_yaw_clamp_delta_rad"] = ""
    return out


def _wrist_ik_target_values(base: Any, env_index: int, target_pos: list[float], target_quat: list[float]) -> list[float]:
    if torch is None or not hasattr(base, "_target_pose_to_wrist_joint_pos"):
        return []
    try:
        pos = torch.tensor([_flat_vec(target_pos, 3)], dtype=torch.float32, device=base.device)
        quat = torch.tensor([_quat_normalize_wxyz(target_quat)], dtype=torch.float32, device=base.device)
        env_ids = torch.tensor([int(env_index)], dtype=torch.long, device=base.device)
        out = base._target_pose_to_wrist_joint_pos(pos, quat, env_ids=env_ids)
        return [float(value) for value in out.detach().cpu().reshape(-1).tolist()[:6]]
    except Exception:
        return []


def _wrist_joint_values_to_dex_pose(base: Any, wrist_joint_values: list[float]) -> dict[str, Any]:
    values = _flat_vec(wrist_joint_values, 6)
    palm_origin = [float(values[0]), float(values[1]), float(values[2])]
    quat = _quat_from_serial_xyz_wxyz(float(values[3]), float(values[4]), float(values[5]))
    offset = _flat_vec(getattr(base, "dex_grasp_frame_local_pos", None), 3)
    dex_pos = [palm_origin[i] + _quat_apply_wxyz(quat, offset)[i] for i in range(3)]
    return {"palm_origin_pos": palm_origin, "dex_pos": dex_pos, "dex_quat": quat}


def _indexed_tensor_row(value: Any, env_index: int, indices: list[int]) -> list[float]:
    if torch is None or not torch.is_tensor(value) or not indices:
        return []
    try:
        idx = torch.tensor(indices, dtype=torch.long, device=value.device)
        return [float(v) for v in value[int(env_index), idx].detach().cpu().tolist()]
    except Exception:
        return []


def _max_abs_delta(a: list[float], b: list[float]) -> Any:
    if not a or not b:
        return ""
    count = min(len(a), len(b))
    if count <= 0:
        return ""
    try:
        return max(abs(float(a[i]) - float(b[i])) for i in range(count))
    except Exception:
        return ""


def _actuator_param_by_joint(robot: Any, joint_names: list[str], joint_indices: list[int], attr: str) -> list[Any]:
    out: list[Any] = ["" for _ in joint_indices]
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, dict):
        return out
    for actuator in actuators.values():
        act_indices = _tensor_int_list(getattr(actuator, "joint_indices", None))
        act_names = [str(value) for value in list(getattr(actuator, "joint_names", []) or [])]
        value = getattr(actuator, attr, None)
        if value is None and attr == "effort_limit":
            value = getattr(actuator, "effort_limit_sim", None)
        if value is None and attr == "velocity_limit":
            value = getattr(actuator, "velocity_limit_sim", None)
        for row, global_index in enumerate(joint_indices):
            local = -1
            if global_index in act_indices:
                local = act_indices.index(global_index)
            elif row < len(joint_names) and joint_names[row] in act_names:
                local = act_names.index(joint_names[row])
            if local >= 0:
                param_value = _actuator_param_value(value, local)
                if param_value != "":
                    out[row] = param_value
    return out


def _actuator_param_value(value: Any, local_index: int) -> Any:
    try:
        if torch is not None and torch.is_tensor(value):
            flat = value.detach().cpu()
            if flat.ndim == 0:
                return float(flat.item())
            if flat.ndim == 1 and int(local_index) < flat.shape[0]:
                return float(flat[int(local_index)].item())
            if flat.ndim >= 2 and int(local_index) < flat.shape[-1]:
                return float(flat.reshape(-1, flat.shape[-1])[0, int(local_index)].item())
        if isinstance(value, (list, tuple)):
            if not value:
                return ""
            if int(local_index) < len(value):
                return float(value[int(local_index)])
            return float(value[0])
        if isinstance(value, dict):
            return ""
        if value is not None:
            return float(value)
    except Exception:
        return ""
    return ""


def _tensor_int_list(value: Any) -> list[int]:
    if torch is not None and torch.is_tensor(value):
        try:
            return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        try:
            return [int(v) for v in value]
        except Exception:
            return []
    return []


def _tensor_flat_list(value: Any) -> list[float]:
    if torch is not None and torch.is_tensor(value):
        try:
            return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except Exception:
            return []
    return []


def _rad_to_deg_or_blank(value: Any) -> Any:
    try:
        value = float(value)
        if math.isfinite(value):
            return math.degrees(value)
    except Exception:
        pass
    return ""


def _set_hand_target_servo_action(
    base: Any,
    action: Any,
    env_index: int,
    desired_pose: list[float],
    active_locals: list[int],
    max_step: float,
) -> float:
    current = _hand_audit_snapshot(base, env_index).get("target", [])
    peak = 0.0
    for local in active_locals:
        if local < 0 or local >= len(desired_pose):
            continue
        now = float(_list_get(current, local, 0.0) or 0.0)
        err = float(desired_pose[local]) - now
        peak = max(peak, abs(err))
        action[env_index, 6 + local] = max(-float(max_step), min(float(max_step), err))
    return peak


def _object_velocity(base: Any, env_index: int, part: str) -> dict[str, float]:
    registry = getattr(base, "v83_active_asset_registry", {}) or {}
    asset = dict(registry.get(part, {}) or {}).get("asset")
    data = getattr(asset, "data", None)
    lin = _tensor_vec(getattr(data, "root_lin_vel_w", None), env_index)
    ang = _tensor_vec(getattr(data, "root_ang_vel_w", None), env_index)
    return {
        "linear_velocity_norm": _norm(lin),
        "angular_velocity_norm": _norm(ang),
        "linear_velocity_xyz": lin,
        "angular_velocity_xyz": ang,
    }


def _target_contact_acquired(state: dict[str, Any], cfg: Screw1GraspBaselineV2Config) -> bool:
    return bool(
        max(
            float(state.get("finger3_target_filtered_force_n", 0.0) or 0.0),
            float(state.get("finger4_target_filtered_force_n", 0.0) or 0.0),
        )
        >= float(cfg.contact_threshold_n)
    )


def _force_guard_decision(f3: float, f4: float, cfg: Screw1GraspBaselineV2Config) -> str:
    peak = max(float(f3), float(f4))
    if peak >= float(cfg.hard_abort_force_n):
        return "abort"
    if peak > float(cfg.soft_force_max_n):
        return "retreat"
    if peak >= float(cfg.soft_force_min_n):
        return "hold"
    return "advance"


def _raw_contact_peaks(alignment: AlignmentRecorder | None, phase: str, step: int) -> dict[str, Any]:
    if alignment is None:
        return {}
    rows = [row for row in getattr(alignment, "contact_rows", []) if row.get("phase") == phase and int(row.get("step", -1)) == int(step)]
    target = [row for row in rows if bool(row.get("is_target_object"))]
    nontarget = [row for row in rows if not bool(row.get("is_target_object"))]
    tip_target = [
        row
        for row in target
        if str(row.get("hand_group") or "") in {"finger3_tip", "finger4_tip"}
    ]
    def peak(items: list[dict[str, Any]]) -> float:
        return max([0.0, *[float(row.get("force_norm_or_contact_strength", 0.0) or 0.0) for row in items]])
    return {
        "raw_tip_screw1_peak_n": peak(tip_target),
        "raw_any_screw1_peak_n": peak(target),
        "raw_non_target_peak_n": peak(nontarget),
    }


def _better_seed_replay(
    cfg: Screw1GraspBaselineV2Config, best: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    def rank(row: dict[str, Any]) -> tuple[int, float, float]:
        if not bool(row.get("valid_seed")) or bool(row.get("tracking_timeout")):
            return (-1, -1.0e9, -1.0e9)
        contact = int(bool(row.get("seed_contact_sustained")) and bool(row.get("seed_final_current_contact")))
        force = max(float(row.get("finger3_force_peak_n", 0.0) or 0.0), float(row.get("finger4_force_peak_n", 0.0) or 0.0))
        in_band = int(float(cfg.soft_force_min_n) <= force <= float(cfg.soft_force_max_n))
        return (contact + in_band, -float(row.get("object_displacement_m", 1e9) or 1e9), -abs(force - 0.5))
    if not best:
        return dict(candidate) if rank(candidate)[0] >= 0 else {}
    return dict(candidate) if rank(candidate) > rank(best) else best


def _quat_error_axis_angle_wxyz(current: list[float], target: list[float]) -> list[float]:
    current = _quat_normalize_wxyz(current)
    target = _quat_normalize_wxyz(target)
    inv = [current[0], -current[1], -current[2], -current[3]]
    err = _quat_mul_wxyz(target, inv)
    if err[0] < 0.0:
        err = [-value for value in err]
    w = max(-1.0, min(1.0, err[0]))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0e-12, 1.0 - w * w))
    axis = [err[1] / s, err[2] / s, err[3] / s]
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return [axis[i] * angle for i in range(3)]


def _quat_angle_delta_wxyz(a: list[float], b: list[float]) -> float:
    qa = _quat_normalize_wxyz(a)
    qb = _quat_normalize_wxyz(b)
    dot = abs(sum(qa[index] * qb[index] for index in range(4)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def _quat_from_axis_angle_vec_wxyz(axis_angle: list[float]) -> list[float]:
    vec = _flat_vec(axis_angle, 3)
    angle = _norm(vec)
    if angle <= 1.0e-12:
        return [1.0, 0.0, 0.0, 0.0]
    axis = [value / angle for value in vec]
    half = angle * 0.5
    s = math.sin(half)
    return _quat_normalize_wxyz([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_from_serial_xyz_wxyz(roll: float, pitch: float, yaw: float) -> list[float]:
    qx = [math.cos(float(roll) * 0.5), math.sin(float(roll) * 0.5), 0.0, 0.0]
    qy = [math.cos(float(pitch) * 0.5), 0.0, math.sin(float(pitch) * 0.5), 0.0]
    qz = [math.cos(float(yaw) * 0.5), 0.0, 0.0, math.sin(float(yaw) * 0.5)]
    return _quat_mul_wxyz(_quat_mul_wxyz(qx, qy), qz)


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


def _quat_apply_wxyz(quat: list[float], vec: list[float]) -> list[float]:
    q = _quat_normalize_wxyz(quat)
    v = _flat_vec(vec, 3)
    qv = [0.0, v[0], v[1], v[2]]
    inv = [q[0], -q[1], -q[2], -q[3]]
    out = _quat_mul_raw_wxyz(_quat_mul_raw_wxyz(q, qv), inv)
    return [float(out[1]), float(out[2]), float(out[3])]


def _quat_mul_raw_wxyz(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = _flat_vec(a, 4)
    bw, bx, by, bz = _flat_vec(b, 4)
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _sub_vec(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i] if i < len(a) else 0.0) - float(b[i] if i < len(b) else 0.0) for i in range(3)]


def _cap_vec(vec: list[float], max_norm: float) -> list[float]:
    norm = _norm(vec)
    if norm <= float(max_norm) or norm <= 1.0e-12:
        return [float(value) for value in vec[:3]]
    scale = float(max_norm) / norm
    return [float(value) * scale for value in vec[:3]]


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


def _flat_vec(value: Any, width: int = 3) -> list[float]:
    if value is None:
        return [0.0] * int(width)
    try:
        if hasattr(value, "detach"):
            rows = [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
        elif isinstance(value, (list, tuple)):
            rows = [float(v) for v in value]
        else:
            rows = [float(value)]
    except Exception:
        rows = []
    while len(rows) < int(width):
        rows.append(0.0)
    return [float(v) for v in rows[: int(width)]]


def _list_delta(after: list[float], before: list[float], index: int) -> float:
    if not (0 <= int(index) < len(after) and 0 <= int(index) < len(before)):
        return 0.0
    return float(after[int(index)]) - float(before[int(index)])


def _action_rows(action: Any) -> list[list[float]]:
    try:
        if hasattr(action, "detach"):
            return [[float(value) for value in row] for row in action.detach().cpu().tolist()]
        return [[float(value) for value in row] for row in action]
    except Exception:
        return []


def _json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True)


def _write_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    authority_rows: list[dict[str, Any]],
    alignment: AlignmentRecorder | None,
) -> None:
    summary["episode_summary_csv"] = str(output_dir / "episode_summary.csv")
    summary["active_step_trace_csv"] = str(output_dir / "active_step_trace.csv")
    summary["hand_action_authority_csv"] = str(output_dir / "hand_action_authority.csv")
    if alignment is not None:
        try:
            alignment_info = alignment.finalize(trace_rows, {**summary, "video_available": False})
            summary.update(alignment_info)
        except Exception as exc:
            summary["alignment_finalize_error"] = f"{type(exc).__name__}:{exc}"
    _write_csv(output_dir / "episode_summary.csv", episodes)
    _write_csv(output_dir / "active_step_trace.csv", trace_rows)
    _write_csv(output_dir / "hand_action_authority.csv", authority_rows)
    if bool(summary.get("collision_offset_ab_executed")):
        ab_summary = {
            key: value
            for key, value in summary.items()
            if key.startswith("collision_offset_ab_")
        }
        write_json(output_dir / "v2_collision_offset_ab_summary.json", ab_summary)
    write_json(output_dir / "v2_summary.json", summary)


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


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach"):
        try:
            data = value.detach().cpu()
            if data.ndim == 0:
                return data.item()
            return data.tolist()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    try:
        if isinstance(value, (bool, int, float, str)) or value is None:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return str(value)
            return value
    except Exception:
        pass
    return str(value)
