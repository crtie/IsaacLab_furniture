"""Wuji V2 active physical grasp launcher."""
from __future__ import annotations
import argparse
import importlib.machinery
import math
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from isaaclab.app import AppLauncher
TASK_NAME = "Isaac-WujiFloating-ChairTeacher-Direct-v0"
VIDEO_CAMERA_EYE = (0.65, -1.35, 1.25)
VIDEO_CAMERA_LOOKAT = (-0.10, -0.28, 0.82)
HEIGHT_VIDEO_CAMERA_EYE = (-0.02, -0.92, 0.88)
HEIGHT_VIDEO_CAMERA_LOOKAT = (-0.02, -0.28, 0.76)
VIDEO_RESOLUTION = (1280, 720)
TARGET_FPS = 30
def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the active Wuji V2 route.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chair_stage", type=int, default=2)
    parser.add_argument("--target_id", type=int, default=1)
    parser.add_argument("--active_part_name", type=str, default=None)
    parser.add_argument("--record_h264", type=_parse_bool, default=False)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--connection_idx", type=int, default=None)
    parser.add_argument("--insert_type", type=str, default=None)
    parser.add_argument("--assembly_target_name", type=str, default=None)
    parser.add_argument("--max_candidates", type=int, default=1)
    parser.add_argument("--live_video_numeric_run", action="store_true", default=False)
    parser.add_argument("--live_video_start_stage", choices=("before_reset", "before_physical_trial_start"), default="before_physical_trial_start")
    parser.add_argument("--fixture_mode", choices=("table", "raised", "upright_cradle", "upright_on_table"), default="upright_on_table")
    parser.add_argument("--parking_mode", choices=("none", "visible_nearby", "far_visible"), default="visible_nearby")
    parser.add_argument("--grasp_strategy", choices=("plug1_top_route_v1",), default="plug1_top_route_v1")
    parser.add_argument("--contact_model", choices=("thumb_tip_f2",), default="thumb_tip_f2")
    parser.add_argument("--plug_asset_variant", choices=("original", "box_grip"), default="original")
    parser.add_argument("--plug_contact_height_frac", type=float, default=0.80)
    parser.add_argument("--precontact_extra_gap_m", type=float, default=0.006)
    parser.add_argument("--wrist_angle_search", dest="wrist_angle_search", action="store_true", default=True)
    parser.add_argument("--no_wrist_angle_search", dest="wrist_angle_search", action="store_false")
    parser.add_argument("--wrist_pitch_degs", default="-45,-35,-25,-15,-10,-5,0,5,10,15,25,35,45")
    parser.add_argument("--wrist_roll_degs", default="-35,-25,-15,-10,-5,0,5,10,15,25,35")
    parser.add_argument("--wrist_yaw_degs", default="-45,-35,-25,-15,-5,0,5,15,25,35,45")
    parser.add_argument("--max_wrist_angle_candidates", type=int, default=400)
    parser.add_argument("--f2_first_bridge", action="store_true", default=False)
    parser.add_argument("--a3_skip_post_contact_stabilize_after_bridge", action="store_true", default=False)
    parser.add_argument(
        "--skip_post_contact_stabilize_after_bridge",
        dest="a3_skip_post_contact_stabilize_after_bridge",
        action="store_true",
        default=False,
        help="Alias for --a3_skip_post_contact_stabilize_after_bridge.",
    )
    parser.add_argument("--strict_lift_start_from_bridge_final", action="store_true", default=False)
    parser.add_argument("--post_bridge_hold_steps", type=int, default=0)
    parser.add_argument("--post_bridge_extra_close_thumb", type=float, default=0.0)
    parser.add_argument("--post_bridge_extra_close_f2", type=float, default=0.0)
    parser.add_argument("--post_bridge_extra_close_steps", type=int, default=0)
    parser.add_argument("--bridge_thumb_close_scale", type=float, default=1.0)
    parser.add_argument("--bridge_thumb_close_steps", type=int, default=0)
    parser.add_argument("--bridge_keep_f2_target_frozen", action="store_true", default=False)
    parser.add_argument("--bridge_f2_extra_hold_delta", type=float, default=0.0)
    parser.add_argument("--bridge_thumb_close_pause_after_f2_steps", type=int, default=0)
    parser.add_argument("--f2_first_bridge_thumb_step_delta", type=float, default=None)
    parser.add_argument("--f2_first_bridge_max_thumb_delta", type=float, default=None)
    parser.add_argument("--bridge_allow_f2_reacquire_during_thumb", action="store_true", default=False)
    parser.add_argument("--bridge_f2_loss_tolerance_steps", type=int, default=0)
    parser.add_argument("--bridge_root_nudge_x_m", type=float, default=0.0)
    parser.add_argument("--bridge_root_nudge_y_m", type=float, default=0.0)
    parser.add_argument("--plug_solver_velocity_iterations", type=int, default=None)
    parser.add_argument("--plug_static_friction", type=float, default=None)
    parser.add_argument("--plug_dynamic_friction", type=float, default=None)
    parser.add_argument("--plug_contact_offset", type=float, default=None)
    parser.add_argument("--oracle_sticky_after_valid_clamp_contact", action="store_true", default=False)
    parser.add_argument(
        "--oracle_sticky_activation_gate",
        choices=("thumb_finger2_contact", "thumb_finger2_force"),
        default="thumb_finger2_force",
    )
    parser.add_argument("--oracle_sticky_force_threshold_n", type=float, default=1.0)
    parser.add_argument("--plug_sticky_contact_consecutive_frames", type=int, default=3)
    parser.add_argument("--sticky_insert_after_lift", action="store_true", default=False)
    parser.add_argument("--sticky_insert_preinsert_offset_m", type=float, default=0.040)
    parser.add_argument("--sticky_insert_transport_steps", type=int, default=120)
    parser.add_argument("--sticky_insert_align_steps", type=int, default=40)
    parser.add_argument("--sticky_insert_push_steps", type=int, default=100)
    parser.add_argument("--sticky_insert_hold_steps", type=int, default=30)
    parser.add_argument("--sticky_insert_success_pos_m", type=float, default=0.015)
    parser.add_argument("--sticky_insert_success_rot_rad", type=float, default=0.35)
    parser.add_argument("--sticky_insert_trace", action="store_true", default=False)
    parser.add_argument("--pinch_line_align", dest="pinch_line_align", action="store_true", default=True)
    parser.add_argument("--no_pinch_line_align", dest="pinch_line_align", action="store_false")
    parser.add_argument("--pinch_line_align_max_total_m", type=float, default=0.014)
    parser.add_argument("--pinch_line_align_max_step_m", type=float, default=0.0015)
    parser.add_argument("--pinch_line_align_success_m", type=float, default=0.006)
    parser.add_argument("--pinch_line_align_target_t", type=float, default=0.50)
    parser.add_argument("--pinch_line_align_t_min", type=float, default=0.25)
    parser.add_argument("--pinch_line_align_t_max", type=float, default=0.75)
    parser.add_argument("--pinch_line_align_steps", type=int, default=20)
    parser.add_argument("--pinch_line_allow_safe_downward", dest="pinch_line_allow_safe_downward", action="store_true", default=True)
    parser.add_argument("--no_pinch_line_allow_safe_downward", dest="pinch_line_allow_safe_downward", action="store_false")
    parser.add_argument("--pinch_line_downward_clearance_margin_m", type=float, default=0.020)
    parser.add_argument("--pinch_line_max_downward_step_m", type=float, default=0.001)
    parser.add_argument("--pinch_line_max_total_downward_m", type=float, default=0.014)
    parser.add_argument("--preload_descent", dest="preload_descent", action="store_true", default=True)
    parser.add_argument("--no_preload_descent", dest="preload_descent", action="store_false")
    parser.add_argument("--preload_descent_step_m", type=float, default=0.0005)
    parser.add_argument("--preload_descent_max_m", type=float, default=0.008)
    parser.add_argument("--preload_descent_min_palm_clearance_m", type=float, default=0.008)
    parser.add_argument("--preload_descent_stop_active_tip_m", type=float, default=0.008)
    parser.add_argument("--preload_descent_stop_support_tip_m", type=float, default=0.012)
    parser.add_argument("--preload_descent_stop_on_contact", dest="preload_descent_stop_on_contact", action="store_true", default=True)
    parser.add_argument("--no_preload_descent_stop_on_contact", dest="preload_descent_stop_on_contact", action="store_false")
    parser.add_argument("--preload_descent_allow_safe_plug_motion_m", type=float, default=0.0015)
    parser.add_argument("--preload_descent_max_plug_rot_rad", type=float, default=0.08)
    parser.add_argument("--micro_lift_after_contact", dest="micro_lift_after_contact", action="store_true", default=True)
    parser.add_argument("--no_micro_lift_after_contact", dest="micro_lift_after_contact", action="store_false")
    parser.add_argument("--micro_lift_first_m", type=float, default=0.002)
    parser.add_argument("--micro_lift_second_m", type=float, default=0.002)
    parser.add_argument("--micro_lift_steps", type=int, default=40)
    parser.add_argument("--micro_lift_hold_steps", type=int, default=30)
    parser.add_argument("--lift_trace", action="store_true", default=False)
    parser.add_argument("--post_contact_stabilize", dest="post_contact_stabilize", action="store_true", default=True)
    parser.add_argument("--no_post_contact_stabilize", dest="post_contact_stabilize", action="store_false")
    parser.add_argument("--post_contact_hold_steps", type=int, default=80)
    parser.add_argument("--post_contact_extra_squeeze", type=float, default=0.02)
    parser.add_argument("--post_contact_max_plug_rot_rad", type=float, default=0.12)
    parser.add_argument("--post_contact_max_xy_drift_m", type=float, default=0.004)
    parser.add_argument("--settle_steps", type=int, default=0)
    parser.add_argument("--save_dir", default="records_wuji_v2_physical_grasp_search")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.run_cli_argv = list(sys.argv)
    # Internal defaults for legacy route helpers. These are intentionally no
    # longer public CLI knobs in the sticky-cleanup mainline.
    args.video_top_k = 1
    args.route_ready_q_source = "closure_margin_search"
    args.two_contact_yaw_range_deg = 45.0
    args.two_contact_yaw_step_deg = 5.0
    args.closure_margin_search_max = 120
    args.require_height_aligned_wrist_ik = True
    args.min_f2_j2_positive_margin = 0.06
    args.max_route_predicted_error_m = 0.025
    args.max_route_predicted_active_to_plug_m = 0.020
    args.max_route_predicted_support_to_plug_m = 0.035
    args.max_close_predicted_active_tip_m = 0.014
    args.approach_only = False
    args.close_probe_from_precontact = bool(args.f2_first_bridge)
    args.close_probe_no_lift = not bool(args.oracle_sticky_after_valid_clamp_contact)
    args.f2_first_bridge_variant = "close_thumb_probe_j2_neg_004"
    args.f2_first_bridge_steps_per_thumb_delta = 4
    if args.f2_first_bridge_thumb_step_delta is None:
        args.f2_first_bridge_thumb_step_delta = 0.005
    if args.f2_first_bridge_max_thumb_delta is None:
        args.f2_first_bridge_max_thumb_delta = 0.12
    args.f2_first_bridge_thumb_joint = "right_finger1_joint1"
    args.f2_first_bridge_allow_micro_lift = bool(args.oracle_sticky_after_valid_clamp_contact)
    args.require_load_bearing_candidate_for_micro_lift = True
    if args.chair_stage <= 0 or args.target_id <= 0:
        parser.error("--chair_stage and --target_id must be positive.")
    if args.connection_idx is not None and args.connection_idx <= 0:
        parser.error("--connection_idx must be positive when provided.")
    if args.max_candidates < 0 or args.video_top_k < 0 or args.settle_steps < 0:
        parser.error("--max_candidates, --video_top_k, and --settle_steps must be non-negative.")
    if bool(args.live_video_numeric_run) and int(args.max_candidates) != 1:
        parser.error("--live_video_numeric_run requires --max_candidates 1.")
    if not 0.0 <= args.plug_contact_height_frac <= 1.0:
        parser.error("--plug_contact_height_frac must be in [0, 1].")
    positive = (args.two_contact_yaw_step_deg,)
    non_negative = (
        args.precontact_extra_gap_m,
        args.two_contact_yaw_range_deg,
        args.min_f2_j2_positive_margin,
        args.max_route_predicted_error_m,
        args.max_route_predicted_active_to_plug_m,
        args.max_route_predicted_support_to_plug_m,
        args.max_close_predicted_active_tip_m,
        args.pinch_line_align_max_total_m,
        args.pinch_line_align_max_step_m,
        args.pinch_line_align_success_m,
        args.pinch_line_downward_clearance_margin_m,
        args.pinch_line_max_downward_step_m,
        args.pinch_line_max_total_downward_m,
        args.preload_descent_step_m,
        args.preload_descent_max_m,
        args.preload_descent_min_palm_clearance_m,
        args.preload_descent_stop_active_tip_m,
        args.preload_descent_stop_support_tip_m,
        args.preload_descent_allow_safe_plug_motion_m,
        args.preload_descent_max_plug_rot_rad,
        args.micro_lift_first_m,
        args.micro_lift_second_m,
        args.post_contact_extra_squeeze,
        args.post_contact_max_plug_rot_rad,
        args.post_contact_max_xy_drift_m,
        args.f2_first_bridge_thumb_step_delta,
        args.f2_first_bridge_max_thumb_delta,
        args.post_bridge_extra_close_thumb,
        args.post_bridge_extra_close_f2,
        args.bridge_thumb_close_scale,
        args.bridge_f2_extra_hold_delta,
        float(args.bridge_f2_loss_tolerance_steps),
        args.oracle_sticky_force_threshold_n,
        args.sticky_insert_preinsert_offset_m,
        args.sticky_insert_success_pos_m,
        args.sticky_insert_success_rot_rad,
    )
    if not all(math.isfinite(value) for value in non_negative):
        parser.error("distance/range/friction arguments must be finite.")
    signed_finite = (
        args.bridge_root_nudge_x_m,
        args.bridge_root_nudge_y_m,
        0.0 if args.plug_static_friction is None else args.plug_static_friction,
        0.0 if args.plug_dynamic_friction is None else args.plug_dynamic_friction,
        0.0 if args.plug_contact_offset is None else args.plug_contact_offset,
    )
    if not all(math.isfinite(value) for value in signed_finite):
        parser.error("--bridge_root_nudge and plug physics overrides must be finite.")
    if any(value < 0.0 for value in non_negative):
        parser.error("distance/range arguments must be non-negative.")
    if args.bridge_thumb_close_scale <= 0.0 or args.bridge_thumb_close_scale > 1.0:
        parser.error("--bridge_thumb_close_scale must be in (0, 1].")
    if args.bridge_f2_loss_tolerance_steps < 0:
        parser.error("--bridge_f2_loss_tolerance_steps must be non-negative.")
    if args.plug_static_friction is not None and args.plug_static_friction <= 0.0:
        parser.error("--plug_static_friction must be positive when provided.")
    if args.plug_dynamic_friction is not None and args.plug_dynamic_friction <= 0.0:
        parser.error("--plug_dynamic_friction must be positive when provided.")
    if args.plug_contact_offset is not None and args.plug_contact_offset < 0.0:
        parser.error("--plug_contact_offset must be non-negative when provided.")
    if args.plug_solver_velocity_iterations is not None and args.plug_solver_velocity_iterations <= 0:
        parser.error("--plug_solver_velocity_iterations must be positive when provided.")
    if any(value <= 0.0 for value in positive):
        parser.error("--two_contact_yaw_step_deg must be positive.")
    if args.closure_margin_search_max <= 0:
        parser.error("--closure_margin_search_max must be positive.")
    if args.max_wrist_angle_candidates <= 0:
        parser.error("--max_wrist_angle_candidates must be positive.")
    for name in ("wrist_pitch_degs", "wrist_roll_degs", "wrist_yaw_degs"):
        _parse_float_list(getattr(args, name), f"--{name}")
    if args.pinch_line_align_steps < 0:
        parser.error("--pinch_line_align_steps must be non-negative.")
    if args.f2_first_bridge_steps_per_thumb_delta < 0:
        parser.error("--f2_first_bridge_steps_per_thumb_delta must be non-negative.")
    if args.post_bridge_hold_steps < 0 or args.post_bridge_extra_close_steps < 0:
        parser.error("--post_bridge_hold_steps and --post_bridge_extra_close_steps must be non-negative.")
    if args.bridge_thumb_close_steps < 0 or args.bridge_thumb_close_pause_after_f2_steps < 0:
        parser.error("--bridge_thumb_close_steps and --bridge_thumb_close_pause_after_f2_steps must be non-negative.")
    if (
        args.sticky_insert_transport_steps < 0
        or args.sticky_insert_align_steps < 0
        or args.sticky_insert_push_steps < 0
        or args.sticky_insert_hold_steps < 0
    ):
        parser.error("--sticky_insert step counts must be non-negative.")
    if args.micro_lift_steps <= 0 or args.micro_lift_hold_steps < 0:
        parser.error("--micro_lift_steps must be positive and --micro_lift_hold_steps must be non-negative.")
    if args.post_contact_hold_steps < 0:
        parser.error("--post_contact_hold_steps must be non-negative.")
    if args.micro_lift_second_m < args.micro_lift_first_m:
        parser.error("--micro_lift_second_m must be greater than or equal to --micro_lift_first_m.")
    t_values = (args.pinch_line_align_target_t, args.pinch_line_align_t_min, args.pinch_line_align_t_max)
    if not all(math.isfinite(value) and value >= 0.0 for value in t_values) or not args.pinch_line_align_t_min <= args.pinch_line_align_target_t <= args.pinch_line_align_t_max:
        parser.error("--pinch_line_align t values must satisfy 0 <= t_min <= target_t <= t_max.")
    args.park_nonactive_parts = args.parking_mode != "none"
    args.min_park_distance_m = 0.16
    args.staging_pos = None
    args.table_z_override = None
    args.plug_contact_clearance_m = 0.003
    args.route_quat_source = "init_flat"
    args.enable_cameras = bool(args.video_top_k)
    return args
def _parse_float_list(value: str, label: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except Exception as exc:
        raise SystemExit(f"{label} must be a comma-separated finite float list.") from exc
    if not values or any(not math.isfinite(item) for item in values):
        raise SystemExit(f"{label} must be a non-empty comma-separated finite float list.")
    return values
args_cli = _parse_args()
if args_cli.seed is not None:
    random.seed(int(args_cli.seed))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from _repo_imports import prefer_repo_isaaclab_tasks
prefer_repo_isaaclab_tasks(__file__)
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.np.wuji_assembly_v2.physical_grasp import active_target, asset_variants, output, trial
from isaaclab_tasks.direct.np.wuji_assembly_v2.physical_grasp import video_live
from isaaclab_tasks.utils import parse_env_cfg
def _install_namespace_package(module_name: str, package_dir: Path) -> None:
    module = ModuleType(module_name)
    module.__file__ = str(package_dir / "__init__.py")
    module.__path__ = [str(package_dir)]
    module.__package__ = module_name
    spec = importlib.machinery.ModuleSpec(module_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules[module_name] = module
    parent, _, child = module_name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
def _chair_task_graph_module():
    repo_root = Path(__file__).resolve().parents[2]
    package = "isaaclab_tasks.direct.np.oracle_policies"
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]
    _install_namespace_package(package, repo_root / "source/isaaclab_tasks/isaaclab_tasks/direct/np/oracle_policies")
    from isaaclab_tasks.direct.np.oracle_policies import chair_task_graph
    return chair_task_graph
def _task_cfg_for_stage(stage_id: int):
    return _chair_task_graph_module().task_cfg_for_stage(int(stage_id))
def _resolve_active_target_args(args: argparse.Namespace) -> None:
    graph = _chair_task_graph_module()
    stage_id = int(args.chair_stage)
    target_id = int(args.target_id)
    try:
        spec = graph.build_chair_stage_spec(stage_id)
    except Exception as exc:
        raise SystemExit(f"failed to resolve chair_stage={stage_id}: {exc}") from exc
    target = next((item for item in spec.targets if int(item.target_id) == target_id), None)
    if target is None:
        valid = ", ".join(str(item.target_id) for item in spec.targets)
        raise SystemExit(f"target_id={target_id} is not available for chair_stage={stage_id}; valid target ids: {valid}")
    args.chair_stage = stage_id
    args.stage_id = stage_id
    args.target_id = target_id
    args.active_part_name = str(args.active_part_name or target.part_name)
    args.part_name = args.active_part_name
    args.connection_idx = int(args.connection_idx if args.connection_idx is not None else target.connection_idx)
    if args.connection_idx <= 0:
        raise SystemExit("--connection_idx must resolve to a positive integer.")
    args.insert_type = str(args.insert_type or target.insert_type)
    args.assembly_target_name = str(args.assembly_target_name or target.name)
    args.active_part_cfg_attr = str(getattr(target, "part_cfg_attr", ""))
    args.assembly_fixed_asset_name = str(getattr(target, "fixed_asset_name", ""))
    args.assembly_connection_attr = str(getattr(target, "connection_attr", ""))
    args.assembly_connector_path = str(getattr(target, "connector_path", ""))
    args.stage_run_dir_name = active_target.stage_run_dir_name(args)
def _make_env(run_dir: Path, render_rgb: bool = False):
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device or "cuda:0", num_envs=1, use_fabric=not args_cli.disable_fabric)
    if args_cli.seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = int(args_cli.seed)
    env_cfg.task = _task_cfg_for_stage(args_cli.chair_stage)
    args_cli.plug_asset_variant_metadata = asset_variants.apply_plug_asset_variant(
        env_cfg.task,
        args_cli.plug_asset_variant,
        solver_velocity_iterations=args_cli.plug_solver_velocity_iterations,
        static_friction=args_cli.plug_static_friction,
        dynamic_friction=args_cli.plug_dynamic_friction,
        contact_offset=args_cli.plug_contact_offset,
    )
    if render_rgb:
        env_cfg.viewer.eye = VIDEO_CAMERA_EYE
        env_cfg.viewer.lookat = VIDEO_CAMERA_LOOKAT
        env_cfg.viewer.resolution = VIDEO_RESOLUTION
    gym_env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array" if render_rgb else None)
    return gym_env, gym_env.unwrapped, gym_env
def _record_video_pass(gym_env, env, rows: list[dict[str, Any]], video_dir: Path, eye, lookat, prefix: str, field: str, role: str) -> str:
    if not rows or args_cli.video_top_k <= 0:
        return ""
    try:
        args_cli.video_replay_role = str(role)
        args_cli.video_camera_eye = tuple(float(item) for item in eye)
        args_cli.video_camera_lookat = tuple(float(item) for item in lookat)
        previous = {path for path in output.video_files(video_dir) if path.name.startswith(prefix)}
        video_env = gym.wrappers.RecordVideo(
            gym_env,
            str(video_dir),
            episode_trigger=lambda _episode_id: False,
            video_length=output.route_video_length_frames(args_cli),
            name_prefix=prefix,
            disable_logger=True,
        )
        for idx, row in enumerate(rows):
            if hasattr(env, "sim") and hasattr(env.sim, "set_camera_view"):
                env.sim.set_camera_view(eye=eye, target=lookat)
            setattr(video_env, "_wuji_deferred_recording_name", f"{prefix}-candidate-{idx}")
            setattr(video_env, "_wuji_deferred_recording_started", False)
            replay_row = trial.run_candidate(env, video_env, args_cli, row, args_cli.fixture_mode, lambda: simulation_app.is_running())
            _attach_replay_diagnostics(row, replay_row, field)
            if bool(getattr(video_env, "recording", False)):
                video_env.stop_recording()
            if not bool(getattr(video_env, "_wuji_deferred_recording_started", False)):
                setattr(video_env, "_wuji_deferred_recording_name", "")
        for _ in range(20):
            videos = sorted(output.video_files(video_dir) - previous, key=lambda path: path.stat().st_mtime)
            videos = [path for path in videos if path.name.startswith(prefix)]
            if len(videos) >= len(rows):
                break
            time.sleep(0.25)
        for row, path in zip(rows, videos[-len(rows) :]):
            row[field] = str(path)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return f"{type(exc).__name__}: {exc}"
    return ""
def _attach_replay_diagnostics(row: dict[str, Any], replay_row: dict[str, Any], field: str) -> None:
    prefix = "top_video" if field == "video_path" else "height_video"
    keys = (
        "video_replay_role",
        "fingerprint_after_reset",
        "fingerprint_after_plug1_staging",
        "fingerprint_after_nonactive_parking",
        "fingerprint_before_physical_trial_start",
        "fingerprint_after_route",
    )
    row[f"{prefix}_replay_row"] = {key: replay_row.get(key) for key in keys}
    row[f"{prefix}_replay_fingerprint_before_physical_trial_start"] = replay_row.get("fingerprint_before_physical_trial_start")
def _record_top_videos(gym_env, env, rows: list[dict[str, Any]]) -> str:
    video_dir = _run_dir(args_cli) / "top_candidate_videos/videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for path in output.video_files(video_dir):
        path.unlink()
    error = _record_video_pass(gym_env, env, rows, video_dir, VIDEO_CAMERA_EYE, VIDEO_CAMERA_LOOKAT, "wuji_v2_physical_top", "video_path", "video_top")
    return error or _record_video_pass(gym_env, env, rows, video_dir, HEIGHT_VIDEO_CAMERA_EYE, HEIGHT_VIDEO_CAMERA_LOOKAT, "wuji_v2_physical_height", "height_video_path", "video_height")
def _live_video_dir(run_dir: Path) -> Path: return run_dir / "videos"
def _clear_mp4s(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in output.video_files(path):
        item.unlink(missing_ok=True)
def _run_dir(args: argparse.Namespace) -> Path:
    return Path(str(args.output_dir)).expanduser() if str(getattr(args, "output_dir", "")).strip() else Path(args.save_dir).expanduser() / active_target.stage_run_dir_name(args)
def _candidate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [{
        "candidate_id": "plug1_top_route_v1",
        "grasp_strategy": "plug1_top_route_v1",
        "search_family": "plug1_top_route_v1",
        "close_mode": "plug1_top_route_v1",
        "close_schedule": "plug1_top_route_route_open_to_hold",
        "route_quat_source": str(args.route_quat_source),
    }][: max(0, int(args.max_candidates))]
def main() -> int:
    _resolve_active_target_args(args_cli)
    run_dir = _run_dir(args_cli)
    run_dir.mkdir(parents=True, exist_ok=True)
    args_cli.stage_run_dir = str(run_dir)
    candidate_rows = _candidate_rows(args_cli)
    gym_env, env, run_env = _make_env(run_dir, render_rgb=bool(args_cli.video_top_k))
    live_video_enabled = bool((args_cli.live_video_numeric_run or args_cli.record_h264) and args_cli.video_top_k > 0)
    live_video_meta: dict[str, Any] = {}
    live_video_previous: set[Path] = set()
    try:
        rows = []
        args_cli.video_replay_role = "numeric"
        if live_video_enabled:
            args_cli.video_camera_eye = VIDEO_CAMERA_EYE
            args_cli.video_camera_lookat = VIDEO_CAMERA_LOOKAT
            if hasattr(env, "sim") and hasattr(env.sim, "set_camera_view"):
                env.sim.set_camera_view(eye=VIDEO_CAMERA_EYE, target=VIDEO_CAMERA_LOOKAT)
            live_dir = _live_video_dir(run_dir)
            _clear_mp4s(live_dir)
            run_env, live_video_meta, live_video_previous = video_live.start_live_video(
                gym_env,
                run_dir,
                enabled=True,
                start_stage=args_cli.live_video_start_stage,
                video_length=output.route_video_length_frames(args_cli),
            )
        else:
            args_cli.video_camera_eye = None
            args_cli.video_camera_lookat = None
        for candidate in candidate_rows:
            print(f"[WujiV2PhysicalSearch] running {candidate['candidate_id']}", flush=True)
            rows.append(trial.run_candidate(env, run_env, args_cli, candidate, args_cli.fixture_mode, lambda: simulation_app.is_running()))
            print(f"[WujiV2PhysicalSearch] finished {candidate['candidate_id']}", flush=True)
        if live_video_enabled:
            live_video_meta = video_live.finish_live_video(run_env, run_dir, live_video_previous, live_video_meta)
            live_video_meta = output.finalize_live_video_evidence(args_cli, rows, live_video_meta)
            for row in rows:
                row.update(live_video_meta)
        top = output.rank_rows_for_video(args_cli, rows)[: min(int(args_cli.video_top_k), len(rows))]
        summary = output.build_summary(args_cli, run_dir, rows, candidate_rows, top)
        if live_video_enabled:
            summary.update(live_video_meta)
        output.write_outputs(run_dir, rows, summary)
        if not live_video_enabled:
            video_error = _record_top_videos(gym_env, env, top)
            summary = output.build_summary(args_cli, run_dir, rows, candidate_rows, top, video_error=video_error)
            output.write_outputs(run_dir, rows, summary)
        print(run_dir / "summary.json", flush=True)
    finally:
        run_env.close()
    return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
