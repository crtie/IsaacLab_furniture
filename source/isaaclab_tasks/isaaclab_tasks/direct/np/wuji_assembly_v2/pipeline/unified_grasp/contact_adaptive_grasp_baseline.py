"""Generic fresh-reset baseline for morphology-aware contact-adaptive closure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ...env_adapter import WujiV2EnvAdapter
except ImportError:  # Runner exposes wuji_assembly_v2 as a top-level path.
    from env_adapter import WujiV2EnvAdapter
from .acquisition_plan_executor import execute_acquisition_plan, load_acquisition_plan, translated_plan
from .closure_prior import ClosurePrior, analyze_recording, write_prior
from .contact_adaptive_closure import (
    ABORT,
    COMPLETE,
    CONTROLLED_CLOSE,
    MULTI_CONTACT_BALANCE,
    POST_CLOSE_HOLD,
    SLOW_LIFT,
    ClosureObservation,
    ContactAdaptiveClosure,
)
from .contact_manager import ContactManager
from .hand_morphology import (
    JointEffectProbe,
    MorphologyCalibration,
    ObjectGraspSpec,
    build_wuji_morphology_spec,
    calibration_from_dict,
    default_object_grasp_specs,
    effect_matrix_from_probes,
)
from .scripted_contact_baseline import _base_env, _distance, _norm, _quat_for_rpy_offset, _target_env_index, read_state
from .screw1_grasp_baseline_v2 import (
    Screw1GraspBaselineV2Config,
    _calibrate_canonical_support_pose,
    _configure_v2_control,
    _fresh_episode,
    _hand_audit_snapshot,
    _part_bbox_bottom_z,
    _run_force_guarded_grasp_trial,
    _set_wrist_delta_action,
    _step_direct_action,
    _zero_action,
    _V2AllBodyContactTruth,
)
from .precontact_fingertip_controller import (
    ABORTED as PRECONTACT_ABORTED,
    LocalActiveJacobian,
    PreContactFingertipController,
    PreContactObservation,
    quat_wxyz_to_matrix,
)
from .video_log_alignment import AlignmentRecorder, make_run_id


GENERIC_RESULT_CLASSES = (
    "REPEATABLE_PHYSICAL_LIFT_ACQUIRED",
    "GENERIC_CLOSURE_VALIDATED_ON_TWO_OBJECTS",
    "SCREW1_LIFT_ACQUIRED_TRANSFER_PENDING",
    "PRECONTACT_JACOBIAN_CONTROLLABILITY_BLOCKER_PROVEN",
    "VERIFIED_NON_TARGET_SCENE_CONTACT_BLOCKER",
    "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP",
    "STABLE_CONTACT_ACQUIRED_CLOSE_FAILED",
    "STABLE_GRASP_ACQUIRED_LIFT_FAILED",
)
GENERIC_PHASES = (
    "prior_analysis",
    "morphology_calibration",
    "screw1_ab",
    "grasp",
    "repeated_trials",
    "transfer",
    "full",
)


@dataclass
class ContactAdaptiveGraspConfig:
    part: str = "Screw1"
    phase: str = "full"
    output_dir: str = "debug_runs/contact_adaptive_closure"
    run_id: str = ""
    alignment_debug: bool = False
    closure_prior_recording: str = "/mnt/data/recording_1.pkl"
    closure_prior_path: str = "debug_runs/closure_prior/other_hand_closure_prior.json"
    morphology_calibration_path: str = "debug_runs/closure_prior/wuji_hand_morphology_calibration.json"
    frozen_screw1_plan_path: str = (
        "debug_runs/screw1_grasp_baseline_v2_probe_handoff_fix/validated_acquisition_plan.json"
    )
    deterministic_seed: int = 20260710
    calibration_probe_delta_rad: float = 0.002
    calibration_settle_steps: int = 8
    calibration_crosstalk_limit: float = 0.25
    development_trials_per_variant: int = 3
    repeated_trials: int = 5
    max_closure_steps: int = 900
    local_probe_delta_rad: float = 0.001
    local_probe_settle_steps: int = 4
    target_lead_limit_rad: float = 0.003
    object_specs: dict[str, ObjectGraspSpec] = field(default_factory=default_object_grasp_specs)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tensor_rows(value: Any, env_index: int, width: int) -> list[list[float]]:
    if not torch.is_tensor(value) or value.ndim < 3 or value.shape[0] <= env_index:
        return []
    data = value[env_index].detach().cpu().reshape(-1, width)
    return [[float(item) for item in row] for row in data.tolist()]


def _quat_delta_rotvec(before: list[float], after: list[float]) -> tuple[float, float, float]:
    if len(before) < 4 or len(after) < 4:
        return (0.0, 0.0, 0.0)
    a = np.asarray(before[:4], dtype=np.float64)
    b = np.asarray(after[:4], dtype=np.float64)
    a /= max(np.linalg.norm(a), 1.0e-12)
    b /= max(np.linalg.norm(b), 1.0e-12)
    conj = np.array([a[0], -a[1], -a[2], -a[3]])
    q = np.array(
        [
            b[0] * conj[0] - np.dot(b[1:], conj[1:]),
            b[0] * conj[1] + conj[0] * b[1] + b[2] * conj[3] - b[3] * conj[2],
            b[0] * conj[2] + conj[0] * b[2] + b[3] * conj[1] - b[1] * conj[3],
            b[0] * conj[3] + conj[0] * b[3] + b[1] * conj[2] - b[2] * conj[1],
        ]
    )
    if q[0] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(float(q[0]), 1.0e-12))
    return tuple(float(item) for item in q[1:] * (angle / vector_norm))


def _calibrate_morphology(
    env: Any,
    base: Any,
    env_index: int,
    v2_cfg: Screw1GraspBaselineV2Config,
    cfg: ContactAdaptiveGraspConfig,
    spec: Any,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
) -> MorphologyCalibration:
    adapter = WujiV2EnvAdapter(base, env_id=env_index)
    manager = ContactManager()
    probes: list[JointEffectProbe] = []
    failures: list[str] = []
    action_scale = float(getattr(base, "dex_hand_action_scale", 1.0) or 1.0)
    for local, joint_name in enumerate(spec.joint_names):
        for sign in (1, -1):
            tag = f"morphology_calibration_j{local:02d}_{'pos' if sign > 0 else 'neg'}"
            _fresh_episode(env, base, env_index, v2_cfg, alignment, trace_rows, tag, hand_mode="parked")
            before_target = adapter.get_hand_control_target_pose().get("pose", [])
            before_actual = adapter.get_current_hand_joint_pose().get("pose", [])
            before_tip_pos = _tensor_rows(getattr(base, "dex_fingertip_pos", None), env_index, 3)
            before_tip_quat = _tensor_rows(getattr(base, "dex_fingertip_quat", None), env_index, 4)
            contact_before = manager.read_contact_state(
                base, env_index=env_index, part_name=v2_cfg.part, active_finger_group=v2_cfg.active_finger_group
            )
            action = _zero_action(env, base)
            action_value = float(sign) * cfg.calibration_probe_delta_rad / max(abs(action_scale), 1.0e-9)
            action[env_index, 6 + local] = action_value
            _step_direct_action(
                env,
                base,
                env_index,
                v2_cfg,
                action,
                phase=tag,
                step=0,
                trace_rows=trace_rows,
                alignment=alignment,
                extra={"morphology_probe_joint": joint_name, "morphology_probe_sign": sign},
            )
            settle_lag = cfg.calibration_settle_steps
            for settle in range(cfg.calibration_settle_steps):
                _step_direct_action(
                    env,
                    base,
                    env_index,
                    v2_cfg,
                    _zero_action(env, base),
                    phase=tag,
                    step=settle + 1,
                    trace_rows=trace_rows,
                    alignment=alignment,
                )
            after_target = adapter.get_hand_control_target_pose().get("pose", [])
            after_actual = adapter.get_current_hand_joint_pose().get("pose", [])
            after_tip_pos = _tensor_rows(getattr(base, "dex_fingertip_pos", None), env_index, 3)
            after_tip_quat = _tensor_rows(getattr(base, "dex_fingertip_quat", None), env_index, 4)
            app_delta = float(sign) * cfg.calibration_probe_delta_rad
            runtime_delta = float(after_target[local] - before_target[local])
            actual_delta = float(after_actual[local] - before_actual[local])
            other = max(
                [0.0]
                + [
                    abs(float(after_actual[index] - before_actual[index]))
                    for index in range(20)
                    if index != local
                ]
            )
            crosstalk = other / max(abs(actual_delta), 1.0e-9)
            pos_delta = tuple(
                tuple(float(after_tip_pos[tip][axis] - before_tip_pos[tip][axis]) for axis in range(3))
                for tip in range(5)
            )
            rot_delta = tuple(_quat_delta_rotvec(before_tip_quat[tip], after_tip_quat[tip]) for tip in range(5))
            contact_after = manager.read_contact_state(
                base, env_index=env_index, part_name=v2_cfg.part, active_finger_group=v2_cfg.active_finger_group
            )
            no_contact = bool(
                max(
                    [0.0]
                    + contact_before.per_finger_unfiltered_force_norm
                    + contact_after.per_finger_unfiltered_force_norm
                )
                < 0.05
            )
            target_ok = bool(
                abs(runtime_delta - app_delta) <= max(0.0005, 0.25 * abs(app_delta))
                and runtime_delta * app_delta > 0.0
            )
            response_ok = bool(
                no_contact
                and actual_delta * app_delta > 0.0
                and abs(actual_delta) >= 0.0005
                and crosstalk <= cfg.calibration_crosstalk_limit
                and np.isfinite(np.asarray(pos_delta)).all()
                and np.isfinite(np.asarray(rot_delta)).all()
            )
            if not target_ok or not response_ok:
                failures.append(f"{joint_name}:{sign}:target={target_ok}:response={response_ok}:crosstalk={crosstalk:.6f}")
            probes.append(
                JointEffectProbe(
                    joint_name=joint_name,
                    local_joint_index=local,
                    action_column=6 + local,
                    action_value=action_value,
                    sign=sign,
                    command_delta_rad=app_delta,
                    app_target_delta_rad=app_delta,
                    runtime_target_delta_rad=runtime_delta,
                    actual_delta_rad=actual_delta,
                    settle_lag_steps=settle_lag,
                    other_joint_actual_peak_rad=other,
                    crosstalk_ratio=crosstalk,
                    fingertip_position_delta_m=pos_delta,
                    fingertip_orientation_delta_rotvec=rot_delta,
                    target_chain_ok=target_ok,
                    response_ok=response_ok,
                )
            )
    matrix = effect_matrix_from_probes(probes)
    valid = bool(not failures and np.linalg.matrix_rank(matrix) >= 5)
    if np.linalg.matrix_rank(matrix) < 5:
        failures.append(f"effect_matrix_rank_below_5:{np.linalg.matrix_rank(matrix)}")
    return MorphologyCalibration(
        schema_version=1,
        hand_signature=spec.calibration_identity,
        probe_delta_rad=cfg.calibration_probe_delta_rad,
        joint_effect_probes=tuple(probes),
        effect_matrix_30x20=tuple(tuple(float(item) for item in row) for row in matrix),
        crosstalk_limit=cfg.calibration_crosstalk_limit,
        calibration_valid=valid,
        failure_reasons=tuple(failures),
    )


def _load_or_calibrate(
    env: Any,
    base: Any,
    env_index: int,
    v2_cfg: Screw1GraspBaselineV2Config,
    cfg: ContactAdaptiveGraspConfig,
    spec: Any,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
) -> tuple[MorphologyCalibration, bool]:
    path = Path(cfg.morphology_calibration_path)
    if path.is_file() and cfg.phase != "morphology_calibration":
        try:
            calibration = calibration_from_dict(json.loads(path.read_text(encoding="utf-8")))
            calibration.validate_for(spec)
            return calibration, True
        except Exception:
            pass
    calibration = _calibrate_morphology(env, base, env_index, v2_cfg, cfg, spec, alignment, trace_rows)
    _json_dump(path, calibration.to_dict())
    rows = [asdict(item) for item in calibration.joint_effect_probes]
    _write_csv(path.with_suffix(".csv"), rows)
    return calibration, False


def _generic_state(
    base: Any,
    env_index: int,
    v2_cfg: Screw1GraspBaselineV2Config,
    object_spec: ObjectGraspSpec,
    group: tuple[str, ...],
    *,
    start_object: list[float],
    lift_start_object: list[float] | None,
    lift_start_relative: list[float] | None,
) -> dict[str, Any]:
    adapter = WujiV2EnvAdapter(base, env_id=env_index)
    raw = read_state(base, env_index, object_spec.part_name, 0.05)
    contact = ContactManager().read_contact_state(
        base,
        env_index=env_index,
        part_name=object_spec.part_name,
        active_finger_group="".join(finger.replace("finger", "") for finger in group),
    )
    names = [f"finger{index}" for index in range(1, 6)]
    forces = {
        name: float(contact.per_finger_target_filtered_force_norm[index])
        if index < len(contact.per_finger_target_filtered_force_norm)
        else 0.0
        for index, name in enumerate(names)
    }
    force_xyz = {
        name: tuple(float(item) for item in contact.per_finger_target_filtered_force_xyz[index])
        if index < len(contact.per_finger_target_filtered_force_xyz)
        else (0.0, 0.0, 0.0)
        for index, name in enumerate(names)
    }
    unfiltered = contact.per_finger_unfiltered_force_norm
    unresolved_residual = max(
        [0.0]
        + [
            max(0.0, float(unfiltered[index]) - forces[names[index]])
            for index in range(min(len(unfiltered), len(names)))
        ]
    )
    registry = getattr(base, "v83_active_asset_registry", {}) or {}
    asset = dict(registry.get(object_spec.part_name, {}) or {}).get("asset")
    lin = getattr(getattr(asset, "data", None), "root_lin_vel_w", None)
    speed = float(torch.linalg.vector_norm(lin[env_index]).detach().cpu().item()) if torch.is_tensor(lin) else 0.0
    current_rel = [raw["object_local_pos"][index] - raw["palm_local_pos"][index] for index in range(3)]
    bbox_bottom = _part_bbox_bottom_z(object_spec.part_name, env_index)
    table_world = v2_cfg.canonical_support_pose.get("table_top_world_z_m", "")
    table_supported = True
    if bbox_bottom != "" and table_world != "":
        table_supported = float(bbox_bottom) <= float(table_world) + 0.0015
    workspace_clamp = _norm(raw.get("workspace_clamp_vector_xyz", []))
    hand = adapter.get_current_hand_joint_pose().get("pose", [])
    hand_target = adapter.get_hand_control_target_pose().get("pose", [])
    lift = 0.0 if lift_start_object is None else raw["object_local_pos"][2] - lift_start_object[2]
    drift = 0.0 if lift_start_relative is None else _distance(current_rel, lift_start_relative)
    return {
        "raw": raw,
        "contact": contact,
        "hand_q": hand,
        "hand_target_q": hand_target,
        "forces": forces,
        "force_xyz": force_xyz,
        "unfiltered_forces": {name: float(unfiltered[index]) if index < len(unfiltered) else 0.0 for index, name in enumerate(names)},
        "unresolved_unfiltered_residual_n": unresolved_residual,
        "identified_non_target_contact_force_n": 0.0,
        "identified_non_target_contact": False,
        "identified_contact_body": "",
        "identified_contact_hand_body": "",
        "contact_source_class": "unresolved_unfiltered_residual" if unresolved_residual > 0.05 else "no_scene_contact",
        "tip_positions": {
            name: tuple(float(item) for item in contact.per_finger_tip_positions[index])
            if index < len(contact.per_finger_tip_positions)
            else (0.0, 0.0, 0.0)
            for index, name in enumerate(names)
        },
        "object_quat_wxyz": tuple(float(item) for item in raw.get("object_quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
        "object_speed_mps": speed,
        "object_displacement_m": _distance(start_object, raw["object_local_pos"]),
        "workspace_clamp_m": workspace_clamp,
        "table_supported": table_supported,
        "object_lift_m": lift,
        "relative_drift_m": drift,
        "current_relative": current_rel,
    }


def _apply_contact_truth(
    state: dict[str, Any],
    contact_truth: _V2AllBodyContactTruth,
    base: Any,
    *,
    phase: str,
    step: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sensor_state = {
        "finger3_target_filtered_force_n": float(state["forces"].get("finger3", 0.0)),
        "finger4_target_filtered_force_n": float(state["forces"].get("finger4", 0.0)),
    }
    truth = contact_truth.record(phase, step, sensor_state, base, extra=extra)
    identified = bool(truth.get("all_body_non_target_contact_identified"))
    state.update(
        {
            "identified_non_target_contact": identified,
            "identified_non_target_contact_force_n": float(truth.get("all_body_non_target_contact_peak_n", 0.0) or 0.0)
            if identified
            else 0.0,
            "identified_contact_body": str(truth.get("all_body_non_target_contact_peak_object", "")),
            "identified_contact_hand_body": str(truth.get("all_body_non_target_contact_peak_body", "")),
            "contact_source_class": (
                "identified_scene_contact"
                if identified
                else "target_filtered_contact"
                if max(state["forces"].values(), default=0.0) > 0.05
                else "unresolved_unfiltered_residual"
                if state["unresolved_unfiltered_residual_n"] > 0.05
                else "no_scene_contact"
            ),
            "all_body_contact_truth": truth,
        }
    )
    return state


def _active_tip_vector(state: dict[str, Any], group: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(state["tip_positions"][finger], dtype=np.float64) for finger in group])


def _object_relative_desired_motion(
    state: dict[str, Any], object_spec: ObjectGraspSpec, group: tuple[str, ...]
) -> np.ndarray:
    rotation = quat_wxyz_to_matrix(state["object_quat_wxyz"])
    axis = rotation @ np.asarray(object_spec.pinch_axis_object, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    center = np.asarray(state["raw"]["object_local_pos"], dtype=np.float64) + rotation @ np.asarray(
        object_spec.contact_center_offset_object, dtype=np.float64
    )
    rows = []
    for finger in group:
        target = center + float(object_spec.finger_side_by_name[finger]) * object_spec.nominal_contact_offset_m * axis
        rows.append(target - np.asarray(state["tip_positions"][finger], dtype=np.float64))
    return np.concatenate(rows)


def _local_probe_violation(state: dict[str, Any], object_spec: ObjectGraspSpec) -> str:
    peak = max(state["forces"].values(), default=0.0)
    if peak >= object_spec.hard_abort_force_n:
        return "hard_force_during_local_jacobian_probe"
    if peak > 0.05:
        return "target_contact_during_local_jacobian_probe"
    if state["identified_non_target_contact"]:
        return "identified_non_target_contact_during_local_jacobian_probe"
    if state["object_displacement_m"] > 5.0e-4:
        return "object_motion_during_local_jacobian_probe"
    if state["workspace_clamp_m"] > 1.0e-9:
        return "workspace_clamp_during_local_jacobian_probe"
    return ""


def _local_jacobian_payload(local: LocalActiveJacobian, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_fingers": list(local.active_fingers),
        "active_joint_indices": list(local.active_joint_indices),
        "matrix_6x8": [list(row) for row in local.matrix_6x8],
        "positive_effects": [list(row) for row in local.positive_effects],
        "negative_effects": [list(row) for row in local.negative_effects],
        "parked_prior_matrix_6x8": [list(row) for row in local.parked_prior_matrix_6x8],
        "target_actual_lag_steps": list(local.target_actual_lag_steps),
        "crosstalk_ratio": list(local.crosstalk_ratio),
        **diagnostics,
    }


def _calibrate_local_active_jacobian(
    env: Any,
    base: Any,
    env_index: int,
    v2_cfg: Screw1GraspBaselineV2Config,
    cfg: ContactAdaptiveGraspConfig,
    morphology: Any,
    morphology_calibration: MorphologyCalibration,
    object_spec: ObjectGraspSpec,
    group: tuple[str, ...],
    start_object: list[float],
    contact_truth: _V2AllBodyContactTruth,
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    trial_id: int,
    calibration_id: int = 0,
) -> tuple[LocalActiveJacobian, dict[str, Any], list[dict[str, Any]]]:
    adapter = WujiV2EnvAdapter(base, env_id=env_index)
    active_joints = tuple(index for finger in group for index in morphology.finger_joint_groups[finger])
    if len(group) != 2 or len(active_joints) != 8:
        raise ValueError("local fingertip calibration requires two four-joint fingers")
    probe_rows: list[dict[str, Any]] = []
    signed_tip: dict[tuple[int, int], np.ndarray] = {}
    signed_actual: dict[tuple[int, int], float] = {}
    positive_effects: list[np.ndarray] = []
    negative_effects: list[np.ndarray] = []
    restore_joint_peak = 0.0
    restore_tip_peak = 0.0
    object_motion_peak = 0.0
    crosstalk_values: list[float] = []
    lag_values: list[int] = []
    failed_reason = ""
    action_scale = float(getattr(base, "dex_hand_action_scale", 1.0) or 1.0)
    phase = f"local_active_jacobian_t{trial_id}_r{calibration_id}"

    baseline = _generic_state(
        base, env_index, v2_cfg, object_spec, group, start_object=start_object,
        lift_start_object=None, lift_start_relative=None,
    )
    baseline_tip = _active_tip_vector(baseline, group)
    baseline_q = np.asarray(baseline["hand_q"], dtype=np.float64)
    for column, joint in enumerate(active_joints):
        per_sign_effect: dict[int, np.ndarray] = {}
        for sign in (1, -1):
            before = _generic_state(
                base, env_index, v2_cfg, object_spec, group, start_object=start_object,
                lift_start_object=None, lift_start_relative=None,
            )
            before_tip = _active_tip_vector(before, group)
            before_q = np.asarray(before["hand_q"], dtype=np.float64)
            before_target_q = np.asarray(before["hand_target_q"], dtype=np.float64)
            lower, upper = morphology.joint_limits[joint]
            requested_target = float(before_target_q[joint]) + sign * cfg.local_probe_delta_rad
            if requested_target <= float(lower) + 0.002 or requested_target >= float(upper) - 0.002:
                failed_reason = f"joint_limit_margin_probe_joint_{joint}"
                break
            action = _zero_action(env, base)
            action[env_index, 6 + joint] = sign * cfg.local_probe_delta_rad / max(abs(action_scale), 1.0e-9)
            _step_direct_action(
                env, base, env_index, v2_cfg, action, phase=phase, step=len(probe_rows),
                trace_rows=trace_rows, alignment=alignment,
                extra={"local_joint_index": joint, "probe_sign": sign, "probe_stage": "advance"},
            )
            probe_state = _generic_state(
                base, env_index, v2_cfg, object_spec, group, start_object=start_object,
                lift_start_object=None, lift_start_relative=None,
            )
            probe_state = _apply_contact_truth(
                probe_state, contact_truth, base, phase=phase, step=len(probe_rows),
                extra={"trial_id": trial_id, "local_joint_index": joint, "probe_sign": sign, "probe_stage": "advance"},
            )
            failed_reason = _local_probe_violation(probe_state, object_spec)
            lag = cfg.local_probe_settle_steps
            response_trace: list[float] = []
            for settle in range(cfg.local_probe_settle_steps):
                _step_direct_action(
                    env, base, env_index, v2_cfg, _zero_action(env, base), phase=phase,
                    step=len(probe_rows) + settle + 1, trace_rows=trace_rows, alignment=alignment,
                    extra={"local_joint_index": joint, "probe_sign": sign, "probe_stage": "settle"},
                )
                probe_state = _generic_state(
                    base, env_index, v2_cfg, object_spec, group, start_object=start_object,
                    lift_start_object=None, lift_start_relative=None,
                )
                probe_state = _apply_contact_truth(
                    probe_state, contact_truth, base, phase=phase, step=len(probe_rows) + settle + 1,
                    extra={"trial_id": trial_id, "local_joint_index": joint, "probe_sign": sign, "probe_stage": "settle"},
                )
                failed_reason = failed_reason or _local_probe_violation(probe_state, object_spec)
                response_trace.append(abs(float(probe_state["hand_q"][joint]) - float(before_q[joint])))
                if response_trace[-1] >= 0.5 * cfg.local_probe_delta_rad and lag == cfg.local_probe_settle_steps:
                    lag = settle + 1
            probe_state = _apply_contact_truth(
                probe_state, contact_truth, base, phase=phase, step=len(probe_rows),
                extra={"trial_id": trial_id, "local_joint_index": joint, "probe_sign": sign},
            )
            probe_tip = _active_tip_vector(probe_state, group)
            probe_q = np.asarray(probe_state["hand_q"], dtype=np.float64)
            probe_target_q = np.asarray(probe_state["hand_target_q"], dtype=np.float64)
            target_response = float(probe_q[joint] - before_q[joint])
            runtime_target_response = float(probe_target_q[joint] - before_target_q[joint])
            other_peak = max(
                [0.0] + [abs(float(probe_q[index] - before_q[index])) for index in active_joints if index != joint]
            )
            crosstalk = other_peak / max(abs(target_response), 1.0e-9)
            signed_tip[(column, sign)] = probe_tip
            signed_actual[(column, sign)] = float(probe_q[joint])
            per_sign_effect[sign] = (probe_tip - before_tip) / max(abs(target_response), 1.0e-9) * (1.0 if target_response >= 0.0 else -1.0)
            crosstalk_values.append(crosstalk)
            lag_values.append(lag)
            object_motion_peak = max(object_motion_peak, probe_state["object_displacement_m"])
            if (
                runtime_target_response * sign <= 0.0
                or abs(runtime_target_response - sign * cfg.local_probe_delta_rad) > 2.5e-4
            ):
                failed_reason = f"target_buffer_propagation_failed_joint_{joint}"

            restore = _zero_action(env, base)
            restore[env_index, 6 + joint] = -sign * cfg.local_probe_delta_rad / max(abs(action_scale), 1.0e-9)
            _step_direct_action(
                env, base, env_index, v2_cfg, restore, phase=phase, step=len(probe_rows) + 10,
                trace_rows=trace_rows, alignment=alignment,
                extra={"local_joint_index": joint, "probe_sign": sign, "probe_stage": "restore"},
            )
            restore_state = _generic_state(
                base, env_index, v2_cfg, object_spec, group, start_object=start_object,
                lift_start_object=None, lift_start_relative=None,
            )
            restore_state = _apply_contact_truth(
                restore_state, contact_truth, base, phase=phase, step=len(probe_rows) + 10,
                extra={"trial_id": trial_id, "local_joint_index": joint, "probe_sign": sign, "probe_stage": "restore"},
            )
            failed_reason = failed_reason or _local_probe_violation(restore_state, object_spec)
            for settle in range(cfg.local_probe_settle_steps):
                _step_direct_action(
                    env, base, env_index, v2_cfg, _zero_action(env, base), phase=phase,
                    step=len(probe_rows) + 11 + settle, trace_rows=trace_rows, alignment=alignment,
                    extra={"local_joint_index": joint, "probe_sign": sign, "probe_stage": "restore_settle"},
                )
                restore_state = _generic_state(
                    base, env_index, v2_cfg, object_spec, group, start_object=start_object,
                    lift_start_object=None, lift_start_relative=None,
                )
                restore_state = _apply_contact_truth(
                    restore_state, contact_truth, base, phase=phase, step=len(probe_rows) + 11 + settle,
                    extra={"trial_id": trial_id, "local_joint_index": joint, "probe_sign": sign,
                           "probe_stage": "restore_settle"},
                )
                failed_reason = failed_reason or _local_probe_violation(restore_state, object_spec)
            restored = _generic_state(
                base, env_index, v2_cfg, object_spec, group, start_object=start_object,
                lift_start_object=None, lift_start_relative=None,
            )
            restored_tip_error = float(np.max(np.linalg.norm((_active_tip_vector(restored, group) - before_tip).reshape(2, 3), axis=1)))
            restored_joint_error = float(np.max(np.abs(np.asarray(restored["hand_q"], dtype=np.float64) - before_q)))
            restore_joint_peak = max(restore_joint_peak, restored_joint_error)
            restore_tip_peak = max(restore_tip_peak, restored_tip_error)
            object_motion_peak = max(object_motion_peak, restored["object_displacement_m"])
            probe_rows.append(
                {
                    "trial_id": trial_id, "calibration_id": calibration_id, "joint_index": joint,
                    "joint_name": morphology.joint_names[joint], "matrix_column": column, "sign": sign,
                    "command_delta_rad": sign * cfg.local_probe_delta_rad, "actual_response_rad": target_response,
                    "runtime_target_response_rad": runtime_target_response,
                    "tip_delta_xyz_m": json.dumps((probe_tip - before_tip).tolist()), "crosstalk_ratio": crosstalk,
                    "target_actual_lag_steps": lag, "joint_restore_error_rad": restored_joint_error,
                    "tip_restore_error_m": restored_tip_error, "object_motion_m": restored["object_displacement_m"],
                    "target_filtered_contact": max(probe_state["forces"].values(), default=0.0) > 0.05,
                    "identified_scene_contact": probe_state["identified_non_target_contact"],
                    "unresolved_unfiltered_residual_n": probe_state["unresolved_unfiltered_residual_n"],
                }
            )
            if failed_reason:
                break
        positive_effects.append(per_sign_effect.get(1, np.zeros(6)))
        negative_effects.append(per_sign_effect.get(-1, np.zeros(6)))
        if failed_reason:
            break

    matrix = np.zeros((6, 8), dtype=np.float64)
    if not failed_reason and len(signed_tip) == 16:
        for column in range(8):
            denominator = signed_actual[(column, 1)] - signed_actual[(column, -1)]
            if abs(denominator) <= 1.0e-7:
                failed_reason = f"insufficient_actual_response_joint_{active_joints[column]}"
                break
            matrix[:, column] = (signed_tip[(column, 1)] - signed_tip[(column, -1)]) / denominator
    parked = np.asarray(morphology_calibration.effect_matrix_30x20, dtype=np.float64)
    parked_rows = [axis for finger in group for axis in range((int(finger.replace("finger", "")) - 1) * 6, (int(finger.replace("finger", "")) - 1) * 6 + 3)]
    parked_matrix = parked[np.ix_(parked_rows, active_joints)]
    local = LocalActiveJacobian(
        active_fingers=group,
        active_joint_indices=active_joints,
        matrix_6x8=tuple(tuple(float(value) for value in row) for row in matrix),
        positive_effects=tuple(tuple(float(value) for value in row) for row in positive_effects),
        negative_effects=tuple(tuple(float(value) for value in row) for row in negative_effects),
        joint_restore_error_rad=restore_joint_peak,
        tip_restore_error_m=restore_tip_peak,
        object_motion_m=object_motion_peak,
        target_actual_lag_steps=tuple(lag_values),
        crosstalk_ratio=tuple(crosstalk_values),
        parked_prior_matrix_6x8=tuple(tuple(float(value) for value in row) for row in parked_matrix),
    )
    final_state = _generic_state(
        base, env_index, v2_cfg, object_spec, group, start_object=start_object,
        lift_start_object=None, lift_start_relative=None,
    )
    desired = _object_relative_desired_motion(final_state, object_spec, group)
    diagnostics = local.diagnostics(desired)
    if failed_reason:
        diagnostics.update({"controllable": False, "failure_reason": failed_reason})
    artifact = _local_jacobian_payload(local, diagnostics)
    output = Path(cfg.output_dir)
    stem = f"local_active_jacobian_trial_{trial_id}_r{calibration_id}"
    _write_csv(output / f"{stem}.csv", probe_rows)
    _json_dump(output / f"{stem}.json", artifact)
    return local, diagnostics, probe_rows


def _run_synergy_trial(
    env: Any,
    base: Any,
    env_index: int,
    v2_cfg: Screw1GraspBaselineV2Config,
    cfg: ContactAdaptiveGraspConfig,
    spec: Any,
    calibration: MorphologyCalibration,
    prior: ClosurePrior,
    object_spec: ObjectGraspSpec,
    plan: dict[str, Any],
    group: tuple[str, ...],
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
    trial_id: int,
    contact_truth: _V2AllBodyContactTruth,
) -> dict[str, Any]:
    fresh = _fresh_episode(
        env, base, env_index, v2_cfg, alignment, trace_rows, f"synergy_trial_{trial_id}", hand_mode="parked"
    )
    start_object = list(fresh["state"]["object_local_pos"])
    approach = execute_acquisition_plan(
        env,
        base,
        env_index,
        v2_cfg,
        alignment,
        trace_rows,
        plan=plan,
        phase="generic_acquisition_plan_replay",
        trial_id=trial_id,
    )
    valid_evidence = list(plan.get("valid_trial_evidence") or [])
    validated_caging_endpoint = bool(
        str(plan.get("acquisition_mode") or "") == "CAGING_PREGRASP_PLAN"
        and len(valid_evidence) >= 2
        and all(
            str(row.get("termination_reason") or "") == "table_barrier_abort"
            and 0.0 <= float(row.get("screw1_projection_parameter_on_tip_segment", -1.0)) <= 1.0
            and float(row.get("screw1_to_tip_segment_perpendicular_m", 1.0)) <= 0.014
            and float(row.get("object_displacement_m", 1.0)) <= 0.005
            for row in valid_evidence
        )
        and str(approach.get("termination_reason") or "") == "table_barrier_abort"
        and not bool(approach.get("hard_abort"))
        and _distance(start_object, list((approach.get("final_state") or {}).get("object_local_pos") or start_object))
        <= 0.005
    )
    if bool(approach.get("hard_abort")) or not (bool(approach.get("reached_pose")) or validated_caging_endpoint):
        return {
            "variant": "B_SYNERGY",
            "trial_id": trial_id,
            "result_class": "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP",
            "termination_reason": str(approach.get("termination_reason") or "acquisition_plan_not_reached"),
            "physical_lift_success": False,
        }
    endpoint_state = _generic_state(
        base, env_index, v2_cfg, object_spec, group, start_object=start_object,
        lift_start_object=None, lift_start_relative=None,
    )
    local_calibration_start_object = list(endpoint_state["raw"]["object_local_pos"])
    local_jacobian, jacobian_diagnostics, probe_rows = _calibrate_local_active_jacobian(
        env, base, env_index, v2_cfg, cfg, spec, calibration, object_spec, group, local_calibration_start_object,
        contact_truth, alignment, trace_rows, trial_id,
    )
    if not bool(jacobian_diagnostics.get("controllable")):
        structural_controllability_failure = bool(
            int(jacobian_diagnostics.get("rank", 0) or 0) < 6
            or float(jacobian_diagnostics.get("minimum_singular_value_m_per_rad", 0.0) or 0.0) < 1.0e-4
            or float(jacobian_diagnostics.get("condition_number", float("inf")) or float("inf")) > 100.0
            or float(jacobian_diagnostics.get("desired_motion_projection_residual", 1.0) or 1.0) > 0.20
        )
        result_class = (
            "PRECONTACT_JACOBIAN_CONTROLLABILITY_BLOCKER_PROVEN"
            if structural_controllability_failure
            else "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP"
        )
        termination_reason = str(
            jacobian_diagnostics.get("failure_reason")
            or (
                "local_active_jacobian_controllability_gate_failed"
                if structural_controllability_failure
                else "local_active_jacobian_repeatability_gate_failed"
            )
        )
        return {
            "variant": "B_OBJECT_RELATIVE_PRECONTACT",
            "trial_id": trial_id,
            "part_name": object_spec.part_name,
            "active_finger_group": list(group),
            "result_class": result_class,
            "termination_reason": termination_reason,
            "local_jacobian_diagnostics": jacobian_diagnostics,
            "probe_count": len(probe_rows),
            "physical_lift_success": False,
        }
    precontact_controller = PreContactFingertipController(
        spec, object_spec, local_jacobian, prior, active_finger_group=group
    )
    closure_controller: ContactAdaptiveClosure | None = None
    hand_action_scale = float(getattr(base, "dex_hand_action_scale", 1.0) or 1.0)
    first_contact_step = -1
    second_contact_step = -1
    simultaneous_steps = 0
    simultaneous_history: list[bool] = []
    force_peaks = {finger: 0.0 for finger in group}
    min_force_peak = 0.0
    preclose_motion_peak = 0.0
    nonzero_action_steps = 0
    lift_start_object: list[float] | None = None
    lift_start_relative: list[float] | None = None
    lift_contact_steps = 0
    lift_steps = 0
    close_executed = False
    termination = "closure_step_limit"
    final_state: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    identified_non_target_seen = False
    local_relinearizations = 0
    precontact_ready_step = -1
    for step in range(cfg.max_closure_steps):
        state = _generic_state(
            base,
            env_index,
            v2_cfg,
            object_spec,
            group,
            start_object=start_object,
            lift_start_object=lift_start_object,
            lift_start_relative=lift_start_relative,
        )
        state = _apply_contact_truth(
            state, contact_truth, base, phase="contact_adaptive_precontact", step=step,
            extra={"trial_id": trial_id},
        )
        identified_non_target_seen = identified_non_target_seen or state["identified_non_target_contact"]
        prior_fields: dict[str, Any] = {}
        if closure_controller is None:
            pre_observation = PreContactObservation(
                hand_q=tuple(float(item) for item in state["hand_q"]),
                hand_target_q=tuple(float(item) for item in state["hand_target_q"]),
                fingertip_positions={finger: state["tip_positions"][finger] for finger in group},
                object_position=tuple(float(item) for item in state["raw"]["object_local_pos"]),
                object_quaternion_wxyz=state["object_quat_wxyz"],
                target_forces_n=state["forces"],
                object_displacement_m=state["object_displacement_m"],
                object_speed_mps=state["object_speed_mps"],
                workspace_clamp_m=state["workspace_clamp_m"],
                identified_non_target_contact=state["identified_non_target_contact"],
                unresolved_unfiltered_residual_n=state["unresolved_unfiltered_residual_n"],
            )
            command = precontact_controller.update(pre_observation)
            command_state = command.state
            command_kind = command.action_kind
            command_hand_delta = command.hand_delta_q
            command_wrist_delta = (0.0, 0.0, 0.0)
            command_wrist_rpy = (0.0, 0.0, 0.0)
            command_contacts = command.active_contacts
            command_termination = command.termination_reason
            prior_fields = {
                "prior_phase": command.prior_phase,
                "prior_progress": command.prior_progress,
                "latent_speed_scale": command.latent_speed_scale,
                "prior_hold_requested": command.hold_requested,
                "desired_tip_delta": json.dumps(command.desired_tip_delta),
                "predicted_tip_delta": json.dumps(command.predicted_tip_delta),
            }
            if command.state == PRECONTACT_ABORTED:
                termination = command.termination_reason
                break
            if command.ready_for_close:
                precontact_ready_step = step
                closure_controller = ContactAdaptiveClosure(
                    spec, calibration, object_spec, active_finger_group=group, initial_state=CONTROLLED_CLOSE
                )
        else:
            observation = ClosureObservation(
                hand_q=tuple(float(item) for item in state["hand_q"]),
                target_forces_n=state["forces"],
                target_force_xyz_n=state["force_xyz"],
                object_displacement_m=state["object_displacement_m"],
                object_speed_mps=state["object_speed_mps"],
                identified_non_target_contact_force_n=state["identified_non_target_contact_force_n"],
                unresolved_unfiltered_residual_n=state["unresolved_unfiltered_residual_n"],
                workspace_clamp_m=state["workspace_clamp_m"],
                table_supported=state["table_supported"],
                object_lift_m=state["object_lift_m"],
                object_to_hand_relative_drift_m=state["relative_drift_m"],
            )
            close_command = closure_controller.update(observation)
            command_state = close_command.state
            command_kind = close_command.action_kind
            command_hand_delta = close_command.hand_delta_q
            command_wrist_delta = close_command.wrist_delta_xyz_m
            command_wrist_rpy = close_command.wrist_delta_rpy_deg
            command_contacts = close_command.active_contacts
            command_termination = close_command.termination_reason
        contacts = [finger for finger in group if state["forces"].get(finger, 0.0) > 0.05]
        if contacts and first_contact_step < 0:
            first_contact_step = step
        if len(contacts) == len(group) and second_contact_step < 0:
            second_contact_step = step
        simultaneous = len(contacts) == len(group)
        simultaneous_history.append(simultaneous)
        simultaneous_steps += int(simultaneous)
        for finger in group:
            force_peaks[finger] = max(force_peaks[finger], float(state["forces"].get(finger, 0.0)))
        min_force_peak = max(min_force_peak, min(float(state["forces"].get(finger, 0.0)) for finger in group))
        if command_state not in {SLOW_LIFT, COMPLETE}:
            preclose_motion_peak = max(preclose_motion_peak, state["object_displacement_m"])
        close_executed = close_executed or command_state in {CONTROLLED_CLOSE, POST_CLOSE_HOLD, SLOW_LIFT, COMPLETE}
        if command_state == SLOW_LIFT:
            if lift_start_object is None:
                lift_start_object = list(state["raw"]["object_local_pos"])
                lift_start_relative = list(state["current_relative"])
            lift_steps += 1
            lift_contact_steps += int(simultaneous)
        action = _zero_action(env, base)
        target_lead = max(
            [0.0] + [abs(float(target) - float(actual)) for target, actual in zip(state["hand_target_q"], state["hand_q"])]
        )
        applied_hand_delta = command_hand_delta if target_lead <= cfg.target_lead_limit_rad else (0.0,) * 20
        for local, delta in enumerate(applied_hand_delta):
            action[env_index, 6 + local] = float(delta) / max(abs(hand_action_scale), 1.0e-9)
        if any(abs(float(item)) > 1.0e-12 for item in applied_hand_delta):
            nonzero_action_steps += 1
        _set_wrist_delta_action(
            base,
            action,
            env_index,
            list(command_wrist_delta),
            [math.radians(float(item)) for item in command_wrist_rpy],
        )
        before_tip = _active_tip_vector(state, group)
        error_before = float(
            np.linalg.norm(np.asarray(command.desired_tip_delta, dtype=np.float64))
            if closure_controller is None
            else np.linalg.norm(_object_relative_desired_motion(state, object_spec, group))
        )
        _step_direct_action(
            env, base, env_index, v2_cfg, action, phase="contact_adaptive_controller", step=step,
            trace_rows=trace_rows, alignment=alignment,
            extra={"trial_id": trial_id, "controller_state": command_state, "action_kind": command_kind},
        )
        final_state = _generic_state(
            base,
            env_index,
            v2_cfg,
            object_spec,
            group,
            start_object=start_object,
            lift_start_object=lift_start_object,
            lift_start_relative=lift_start_relative,
        )
        final_state = _apply_contact_truth(
            final_state, contact_truth, base, phase="contact_adaptive_controller_after", step=step,
            extra={"trial_id": trial_id},
        )
        feedback: dict[str, Any] = {}
        if closure_controller is None and any(abs(float(item)) > 1.0e-12 for item in applied_hand_delta):
            after_tip = _active_tip_vector(final_state, group)
            error_after = float(
                np.linalg.norm(np.asarray(command.desired_tip_delta, dtype=np.float64) - (after_tip - before_tip))
            )
            feedback = precontact_controller.evaluate_step(command, after_tip - before_tip, error_before, error_after)
            feedback.update({"step": step, "trial_id": trial_id})
            prediction_rows.append(feedback)
            if feedback["rollback_requested"]:
                rollback = _zero_action(env, base)
                for local, delta in enumerate(applied_hand_delta):
                    rollback[env_index, 6 + local] = -float(delta) / max(abs(hand_action_scale), 1.0e-9)
                _step_direct_action(
                    env, base, env_index, v2_cfg, rollback, phase="precontact_action_rollback", step=step,
                    trace_rows=trace_rows, alignment=alignment,
                    extra={"trial_id": trial_id, "rollback_reason": "three_non_improving_predictions"},
                )
                if feedback["relinearization_requested"]:
                    local_relinearizations += 1
                    local_jacobian, rel_diag, rel_rows = _calibrate_local_active_jacobian(
                        env, base, env_index, v2_cfg, cfg, spec, calibration, object_spec, group,
                        list(final_state["raw"]["object_local_pos"]),
                        contact_truth, alignment, trace_rows, trial_id, local_relinearizations,
                    )
                    probe_rows.extend(rel_rows)
                    if not bool(rel_diag.get("controllable")):
                        termination = "local_relinearization_gate_failed"
                        break
                    precontact_controller.replace_jacobian(local_jacobian)
                else:
                    termination = "local_effect_prediction_invalid_after_two_relinearizations"
                    break
        trace_rows.append(
            {
                "phase": "contact_adaptive_closure",
                "trial_id": trial_id,
                "step": step,
                "part_name": object_spec.part_name,
                "active_finger_group": ",".join(group),
                "controller_state": command_state,
                "action_kind": command_kind,
                "hand_delta_q": json.dumps(applied_hand_delta),
                "wrist_delta_xyz_m": json.dumps(command_wrist_delta),
                "target_forces_n": json.dumps(state["forces"], sort_keys=True),
                "target_force_xyz_n": json.dumps(state["force_xyz"], sort_keys=True),
                "unfiltered_forces_n": json.dumps(
                    state["contact"].per_finger_unfiltered_force_norm, sort_keys=True
                ),
                "unresolved_unfiltered_residual_n": state["unresolved_unfiltered_residual_n"],
                "identified_non_target_contact_force_n": state["identified_non_target_contact_force_n"],
                "identified_contact_body": state["identified_contact_body"],
                "identified_contact_hand_body": state["identified_contact_hand_body"],
                "contact_source_class": state["contact_source_class"],
                "active_contacts": json.dumps(command_contacts),
                "hand_actual_q": json.dumps(state["hand_q"]),
                "hand_target_q": json.dumps(state["hand_target_q"]),
                "fingertip_positions": json.dumps({finger: state["tip_positions"][finger] for finger in group}),
                "object_displacement_m": state["object_displacement_m"],
                "object_speed_mps": state["object_speed_mps"],
                "object_lift_m": state["object_lift_m"],
                "relative_drift_m": state["relative_drift_m"],
                "table_supported": state["table_supported"],
                "termination_reason": command_termination,
                "target_filtered_success_evidence": max(
                    [0.0] + [float(state["forces"].get(finger, 0.0)) for finger in group]
                ) > 0.05,
                "distance_only_success_used": False,
                "target_lead_rad": target_lead,
                "target_lead_hold": target_lead > cfg.target_lead_limit_rad,
                "prediction_feedback": json.dumps(feedback, sort_keys=True),
                **prior_fields,
            }
        )
        if step % 10 == 0:
            _write_csv(Path(cfg.output_dir) / "contact_adaptive_trace.csv", trace_rows)
            _write_csv(Path(cfg.output_dir) / f"precontact_prediction_trial_{trial_id}.csv", prediction_rows)
            _json_dump(
                Path(cfg.output_dir) / "contact_adaptive_controller_heartbeat.json",
                {"trial_id": trial_id, "step": step, "state": command_state, "forces": state["forces"],
                 "contact_source_class": state["contact_source_class"], "local_relinearizations": local_relinearizations},
            )
        if command_state == ABORT:
            termination = command_termination
            break
        if command_state == COMPLETE:
            termination = "physical_lift_complete"
            break
    lift_duty = float(lift_contact_steps / max(lift_steps, 1))
    physical_lift = bool(
        closure_controller is not None
        and closure_controller.state == COMPLETE
        and lift_duty >= 0.8
        and final_state.get("object_lift_m", 0.0) >= object_spec.lift_target_m
        and not bool(final_state.get("table_supported", True))
        and final_state.get("relative_drift_m", 1.0) <= 0.010
    )
    dual_window = simultaneous_history[-30:]
    stable_contact = bool(len(dual_window) == 30 and sum(dual_window) >= 24)
    trial_result_class = (
        "SCREW1_LIFT_ACQUIRED_TRANSFER_PENDING"
        if physical_lift and object_spec.part_name == "Screw1"
        else "STABLE_GRASP_ACQUIRED_LIFT_FAILED"
        if close_executed
        else "STABLE_CONTACT_ACQUIRED_CLOSE_FAILED"
        if stable_contact
        else "VERIFIED_NON_TARGET_SCENE_CONTACT_BLOCKER"
        if identified_non_target_seen
        else "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP"
    )
    return {
        "variant": "B_OBJECT_RELATIVE_PRECONTACT",
        "trial_id": trial_id,
        "part_name": object_spec.part_name,
        "active_finger_group": list(group),
        "result_class": trial_result_class,
        "termination_reason": termination,
        "first_contact_step": first_contact_step,
        "second_contact_step": second_contact_step,
        "simultaneous_contact_steps": simultaneous_steps,
        "simultaneous_contact_duty_ratio": float(sum(dual_window) / max(len(dual_window), 1)),
        "min_active_force_peak_n": min_force_peak,
        "force_peaks_n": force_peaks,
        "final_identified_non_target_force_n": final_state.get("identified_non_target_contact_force_n", 0.0),
        "final_unresolved_unfiltered_residual_n": final_state.get("unresolved_unfiltered_residual_n", 0.0),
        "preclose_object_displacement_peak_m": preclose_motion_peak,
        "physics_steps": step + 1,
        "probe_count": len(probe_rows),
        "nonzero_synergy_action_steps": nonzero_action_steps,
        "contact_handoff_count": 0,
        "precontact_ready_step": precontact_ready_step,
        "local_relinearization_count": local_relinearizations,
        "local_jacobian_diagnostics": jacobian_diagnostics,
        "stable_contact": stable_contact,
        "close_executed": close_executed,
        "lift_executed": lift_steps > 0,
        "lift_contact_duty_ratio": lift_duty,
        "object_lift_delta_z_m": final_state.get("object_lift_m", 0.0),
        "object_to_hand_relative_drift_m": final_state.get("relative_drift_m", 0.0),
        "object_off_table": not bool(final_state.get("table_supported", True)),
        "physical_grasp_success": stable_contact and close_executed,
        "physical_lift_success": physical_lift,
        "object_write_after_reset_used": False,
        "sticky_used": False,
        "proxy_success_used": False,
        "distance_only_success_used": False,
        "unfiltered_only_success_used": False,
    }


def _plug_candidate_plans(source: dict[str, Any], plug_object_pos: list[float]) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    source_object = list((source.get("canonical_reset_fingerprint") or {}).get("object_local_pos") or [])
    if len(source_object) < 3:
        raise ValueError("source Screw1 plan lacks canonical object position")
    base_delta = [float(plug_object_pos[index]) - float(source_object[index]) for index in range(3)]
    rows: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for group_text, lateral in (("23", -0.004), ("23", 0.0), ("23", 0.004), ("34", -0.004), ("34", 0.0), ("34", 0.004)):
        delta = (base_delta[0], base_delta[1] + lateral, base_delta[2])
        plan = translated_plan(
            source,
            plan_id=f"plug2_{'two_sided' if group_text == '23' else 'straddle'}_y{lateral:+.3f}",
            translation_xyz_m=delta,
            active_finger_group=group_text,
        )
        plan["plug2_prior_kind"] = "two_sided_pinch" if group_text == "23" else "straddle_pinch"
        rows.append((plan, tuple(f"finger{item}" for item in group_text)))
    return rows


def _select_plug_plan(
    env: Any,
    base: Any,
    env_index: int,
    v2_cfg: Screw1GraspBaselineV2Config,
    cfg: ContactAdaptiveGraspConfig,
    source_plan: dict[str, Any],
    alignment: AlignmentRecorder | None,
    trace_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    first = _fresh_episode(env, base, env_index, v2_cfg, alignment, trace_rows, "plug2_plan_reference")
    candidates = _plug_candidate_plans(source_plan, list(first["state"]["object_local_pos"]))
    rows: list[dict[str, Any]] = []
    selected: tuple[dict[str, Any], tuple[str, ...]] | None = None
    for index, (plan, group) in enumerate(candidates):
        fresh = _fresh_episode(env, base, env_index, v2_cfg, alignment, trace_rows, f"plug2_plan_{index}")
        start = list(fresh["state"]["object_local_pos"])
        result = execute_acquisition_plan(
            env,
            base,
            env_index,
            v2_cfg,
            alignment,
            trace_rows,
            plan=plan,
            phase="plug2_bounded_plan_validation",
            trial_id=index,
        )
        state = _generic_state(
            base,
            env_index,
            v2_cfg,
            cfg.object_specs["Plug2"],
            group,
            start_object=start,
            lift_start_object=None,
            lift_start_relative=None,
        )
        tips = state["contact"].per_finger_tip_positions
        indices = [int(finger.replace("finger", "")) - 1 for finger in group]
        projection = 2.0
        perpendicular = 1.0
        if len(indices) == 2 and max(indices) < len(tips):
            a = np.asarray(tips[indices[0]], dtype=np.float64)
            b = np.asarray(tips[indices[1]], dtype=np.float64)
            obj = np.asarray(state["raw"]["object_local_pos"], dtype=np.float64)
            segment = b - a
            denom = float(np.dot(segment, segment))
            if denom > 1.0e-12:
                projection = float(np.dot(obj - a, segment) / denom)
                perpendicular = float(np.linalg.norm(obj - (a + np.clip(projection, 0.0, 1.0) * segment)))
        valid = bool(
            result.get("reached_pose")
            and not result.get("hard_abort")
            and state["object_displacement_m"] <= 0.005
            and max(state["unfiltered_forces"].values(), default=0.0) <= 0.05
            and 0.0 <= projection <= 1.0
            and perpendicular <= 0.015
        )
        row = {
            "candidate_index": index,
            "plan_id": plan["plan_id"],
            "active_finger_group": list(group),
            "reached_pose": bool(result.get("reached_pose")),
            "termination_reason": result.get("termination_reason", ""),
            "object_displacement_m": state["object_displacement_m"],
            "tip_segment_projection": projection,
            "tip_segment_perpendicular_m": perpendicular,
            "valid_geometry_pregrasp": valid,
            "success_claimed": False,
        }
        rows.append(row)
        if valid:
            selected = (plan, group)
            break
    if selected is None:
        return {}, (), rows
    plan, group = selected
    for repeat in range(2):
        fresh = _fresh_episode(env, base, env_index, v2_cfg, alignment, trace_rows, f"plug2_plan_repeat_{repeat}")
        result = execute_acquisition_plan(
            env,
            base,
            env_index,
            v2_cfg,
            alignment,
            trace_rows,
            plan=plan,
            phase="plug2_selected_plan_repeat",
            trial_id=repeat,
        )
        if not bool(result.get("reached_pose")) or bool(result.get("hard_abort")):
            return {}, (), rows
    return plan, group, rows


def run_contact_adaptive_grasp_baseline(
    env: Any,
    cfg: ContactAdaptiveGraspConfig,
    *,
    video_recorder: Any | None = None,
) -> dict[str, Any]:
    if cfg.phase not in GENERIC_PHASES:
        raise ValueError(f"unsupported contact-adaptive phase: {cfg.phase}")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = analyze_recording(cfg.closure_prior_recording)
    write_prior(prior, cfg.closure_prior_path)
    summary: dict[str, Any] = {
        "baseline": "morphology_aware_contact_adaptive_closure",
        "part": cfg.part,
        "phase": cfg.phase,
        "run_id": cfg.run_id or make_run_id(),
        "closure_prior_path": cfg.closure_prior_path,
        "closure_prior_source_success": prior.source_success,
        "successful_grasp_demonstration": prior.successful_grasp_demonstration,
        "training_locked": True,
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": False,
        "object_write_after_reset_used": False,
        "oracle_visual_only": False,
        "not_physical": True,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "physical_insert_success": False,
        "success_claimed": False,
    }
    if cfg.phase == "prior_analysis":
        summary.update({"phase_result": "CLOSURE_PRIOR_ANALYZED", "final_classification_pending": True})
        _json_dump(output_dir / "contact_adaptive_summary.json", summary)
        return summary

    base = _base_env(env)
    env_index = _target_env_index(base, cfg.part)
    summary["target_env_index"] = int(env_index)
    v2_cfg = Screw1GraspBaselineV2Config(
        part=cfg.part,
        output_dir=cfg.output_dir,
        phase="grasp",
        run_id=summary["run_id"],
        deterministic_seed=cfg.deterministic_seed,
        active_finger_group="34" if cfg.part == "Screw1" else "23",
    )
    _configure_v2_control(base, env_index)
    trace_rows: list[dict[str, Any]] = []
    alignment = AlignmentRecorder(
        enabled=cfg.alignment_debug,
        output_dir=output_dir,
        run_id=summary["run_id"],
        target_part=cfg.part,
        target_env_index=env_index,
        video_recorder=video_recorder,
    )
    alignment.setup_body_mapping_and_sensors(base)
    support = _calibrate_canonical_support_pose(env, base, env_index, v2_cfg, alignment, trace_rows)
    summary.update(support)
    if not support.get("canonical_support_pose_valid"):
        summary.update(
            {
                "result_class": "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP",
                "termination_reason": "part_canonical_support_calibration_failed",
            }
        )
        _write_csv(output_dir / "contact_adaptive_trace.csv", trace_rows)
        _json_dump(output_dir / "contact_adaptive_summary.json", summary)
        return summary

    adapter = WujiV2EnvAdapter(base, env_id=env_index)
    spec = build_wuji_morphology_spec(adapter, calibration_source=cfg.morphology_calibration_path)
    _json_dump(output_dir / "hand_morphology_spec.json", spec.to_dict())
    calibration, reused = _load_or_calibrate(
        env, base, env_index, v2_cfg, cfg, spec, alignment, trace_rows
    )
    summary.update(
        {
            "morphology_calibration_path": cfg.morphology_calibration_path,
            "morphology_calibration_reused": reused,
            "morphology_calibration_valid": calibration.calibration_valid,
            "morphology_calibration_failure_reasons": list(calibration.failure_reasons),
            "morphology_calibration_probe_count": len(calibration.joint_effect_probes),
        }
    )
    if not calibration.calibration_valid:
        summary["result_class"] = "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP"
        _write_csv(output_dir / "contact_adaptive_trace.csv", trace_rows)
        _json_dump(output_dir / "contact_adaptive_summary.json", summary)
        return summary
    if cfg.phase == "morphology_calibration":
        summary.update({"phase_result": "MORPHOLOGY_CALIBRATION_VALIDATED", "final_classification_pending": True})
        _write_csv(output_dir / "contact_adaptive_trace.csv", trace_rows)
        _json_dump(output_dir / "contact_adaptive_summary.json", summary)
        return summary

    object_spec = cfg.object_specs[cfg.part]
    source_plan = load_acquisition_plan(cfg.frozen_screw1_plan_path)
    plan = source_plan
    group = object_spec.active_finger_groups[0]
    plug_plan_rows: list[dict[str, Any]] = []
    if cfg.part == "Plug2":
        plan, group, plug_plan_rows = _select_plug_plan(
            env, base, env_index, v2_cfg, cfg, source_plan, alignment, trace_rows
        )
        _write_csv(output_dir / "plug2_acquisition_plan_candidates.csv", plug_plan_rows)
        if not plan:
            summary.update(
                {
                    "result_class": "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP",
                    "termination_reason": "bounded_plug2_acquisition_plan_not_validated",
                    "plug2_candidate_count": len(plug_plan_rows),
                }
            )
            _write_csv(output_dir / "contact_adaptive_trace.csv", trace_rows)
            _json_dump(output_dir / "contact_adaptive_summary.json", summary)
            return summary
        _json_dump(output_dir / "validated_plug2_acquisition_plan.json", plan)

    trial_rows: list[dict[str, Any]] = []
    if cfg.part == "Screw1" and cfg.phase == "screw1_ab":
        hint = {
            "seed_replay_validated": True,
            "validated_acquisition_plan": source_plan,
            "validated_acquisition_plan_json": cfg.frozen_screw1_plan_path,
        }
        legacy_episodes: list[dict[str, Any]] = []
        for trial in range(cfg.development_trials_per_variant):
            result = _run_force_guarded_grasp_trial(
                env,
                base,
                env_index,
                v2_cfg,
                alignment,
                legacy_episodes,
                trace_rows,
                [],
                trial_id=trial,
                phase_name="legacy_regulator_a",
                seed_hint=hint,
            )
            trial_rows.append({"variant": "A_LEGACY", **result})
    trial_count = cfg.repeated_trials if cfg.phase == "repeated_trials" else cfg.development_trials_per_variant
    if cfg.part == "Plug2" and cfg.phase == "transfer":
        trial_count = 1
    contact_truth = _V2AllBodyContactTruth(
        output_dir, env_index, object_spec.part_name, 0.05, log_prefix="contact_adaptive_all_body"
    )
    contact_truth.setup(base)
    for trial in range(min(trial_count, 3) if cfg.phase != "repeated_trials" else trial_count):
        result = _run_synergy_trial(
            env,
            base,
            env_index,
            v2_cfg,
            cfg,
            spec,
            calibration,
            prior,
            object_spec,
            plan,
            group,
            alignment,
            trace_rows,
            trial,
            contact_truth,
        )
        trial_rows.append(result)
        _write_csv(output_dir / "contact_adaptive_trials.csv", trial_rows)
        _json_dump(
            output_dir / "contact_adaptive_heartbeat.json",
            {"completed_trial_count": len(trial_rows), "latest_trial": result},
        )
    contact_truth_summary = contact_truth.finalize()
    synergy_rows = [row for row in trial_rows if row.get("variant") == "B_OBJECT_RELATIVE_PRECONTACT"]
    success_count = sum(bool(row.get("physical_lift_success")) for row in synergy_rows)
    any_contact = any(float(row.get("min_active_force_peak_n", 0.0) or 0.0) > 0.05 for row in synergy_rows)
    if cfg.part == "Plug2" and success_count:
        result_class = "GENERIC_CLOSURE_VALIDATED_ON_TWO_OBJECTS"
    elif cfg.part == "Screw1" and cfg.phase == "repeated_trials" and success_count >= 3:
        result_class = "REPEATABLE_PHYSICAL_LIFT_ACQUIRED"
    elif cfg.part == "Screw1" and success_count:
        result_class = "SCREW1_LIFT_ACQUIRED_TRANSFER_PENDING"
    else:
        result_class = str(
            (synergy_rows[-1] if synergy_rows else {}).get(
                "result_class", "MORPHOLOGY_EFFECT_MODEL_INVALID_AT_PREGRASP"
            )
        )
    summary.update(
        {
            "result_class": result_class,
            "trial_count": len(synergy_rows),
            "physical_lift_success_count": success_count,
            "physical_grasp_success": any(bool(row.get("physical_grasp_success")) for row in synergy_rows),
            "physical_lift_success": success_count > 0,
            "not_physical": success_count == 0,
            "object_ready": bool(cfg.phase == "repeated_trials" and success_count >= 3),
            "grasp_solved": bool(cfg.phase == "repeated_trials" and success_count >= 3),
            "training_locked": not bool(cfg.phase == "repeated_trials" and success_count >= 3),
            "acquisition_plan_id": plan.get("plan_id", ""),
            "active_finger_group": list(group),
            "trials": trial_rows,
            "all_body_contact_truth": contact_truth_summary,
        }
    )
    _write_csv(output_dir / "contact_adaptive_trials.csv", trial_rows)
    _write_csv(output_dir / "contact_adaptive_trace.csv", trace_rows)
    _json_dump(output_dir / "contact_adaptive_summary.json", summary)
    return summary
