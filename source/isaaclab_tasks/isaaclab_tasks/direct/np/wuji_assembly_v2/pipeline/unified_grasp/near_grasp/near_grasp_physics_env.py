"""Dedicated vectorized Wuji + Table + one-target physics environment."""

from __future__ import annotations

import math
import copy
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul
from isaaclab_assets.robots.wuji_hand import (
    DEX_GRASP_FRAME_LOCAL_POS,
    WUJI_FINGERTIP_BODY_NAMES,
    WUJI_FLOATING_HAND_CFG,
    WUJI_FLOATING_WRIST_JOINT_NAMES,
    WUJI_HAND_FINGER_JOINT_NAMES,
)

from .evaluator import EvaluationInput, PhysicalEvaluation, StrictPhysicalEvaluator
from .grasp_program import (
    GraspProgram,
    GraspTemplate,
    ProgramPhase,
    ProgramTermination,
    WUJI_HAND_JOINT_NAMES,
    resolve_preshape_profile,
)
from .grasp_synthesis.object_spec import ObjectGraspSpec
from .grasp_synthesis.forensic_trace import ForensicContactEvent
from .hand_prior_adapter import HandPriorAdapter
from .observation import NEAR_GRASP_OBSERVATION_SCHEMA
from .residual_rl import ResidualActionBounds


_NP_DIR = Path(__file__).resolve().parents[4]
_TABLE_USD = _NP_DIR / "asset" / "workdesk.usd"
_DEFAULT_TARGET_USD = _NP_DIR / "asset" / "chair" / "screw_ree3.usd"


@configclass
class NearGraspPhysicsEnvCfg(DirectRLEnvCfg):
    """All vector slots contain one dynamic target object and no furniture."""

    decimation = 1
    episode_length_s = 10.0
    action_space = 14
    observation_space = NEAR_GRASP_OBSERVATION_SCHEMA.width
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=8,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**22,
            gpu_collision_stack_size=2**29,
            gpu_heap_capacity=2**27,
            gpu_temp_buffer_capacity=2**25,
            gpu_max_num_partitions=1,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=1.5, replicate_physics=True)

    robot: ArticulationCfg = WUJI_FLOATING_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_DEFAULT_TARGET_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=2.0,
                max_angular_velocity=20.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=8,
                max_contact_impulse=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.010),
            scale=(0.6, 0.6, 0.7),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.18120920658111572, -0.002612590789794922, 0.7430),
            rot=(0.6045550107955933, 0.3139893710613251, 0.623823344707489, 0.3831036686897278),
        ),
    )
    target_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_finger.*_link4",
        filter_prim_paths_expr=["/World/envs/env_.*/TargetObject"],
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=0.05,
    )
    all_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_finger.*_link4",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=0.05,
    )
    replay_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/ReplayCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=32.0,
            focus_distance=0.5,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 5.0),
        ),
        width=960,
        height=540,
        update_period=0.0,
    )
    enable_replay_camera = False
    enable_contact_attribution = False
    defer_terminal_reset = False
    replay_camera_eye_local = (0.10, 0.32, 0.94)
    replay_camera_target_local = (-0.18, 0.0, 0.76)

    table_top_z_m = 0.7399999965887228
    canonical_object_pos = (-0.18120920658111572, -0.002612590789794922, 0.7415136694908142)
    canonical_object_quat = (0.6045550107955933, 0.3139893710613251, 0.623823344707489, 0.3831036686897278)
    object_xy_randomization_m = 0.015
    object_z_clearance_range_m = (0.001, 0.002)
    object_z_noise_m = 0.001
    object_rpy_randomization_rad = math.radians(15.0)
    object_mass_range_kg = (0.008, 0.012)
    friction_multiplier_range = (0.8, 1.2)
    coarse_acquisition_steps = 100
    multi_contact_hold_steps = 30
    controlled_close_steps = 60
    post_close_hold_steps = 30
    max_remaining_finger_steps = 180
    max_lift_steps = 64
    lift_increment_m = 0.00025
    formal_contact_force_n = 0.05
    soft_force_limit_n = 1.0
    hard_force_abort_n = 5.0
    target_part_name = "Screw1"
    target_asset_usd = str(_DEFAULT_TARGET_USD)
    target_asset_scale = (0.6, 0.6, 0.7)
    randomize_physics_on_reset = True


def apply_near_grasp_run_config(
    cfg: NearGraspPhysicsEnvCfg,
    values: Mapping[str, Any],
    *,
    num_envs: int,
    device: str,
    episode_steps: int,
    replay: bool = False,
    enable_replay_camera: bool = False,
    enable_contact_attribution: bool = False,
) -> NearGraspPhysicsEnvCfg:
    """Apply the frozen YAML values to an Isaac environment config."""

    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = float(values["scene"]["env_spacing"])
    cfg.sim.device = str(device)
    cfg.sim.dt = float(values["scene"]["dt"])
    cfg.table_top_z_m = float(values["scene"]["table_top_z_m"])
    cfg.canonical_object_pos = tuple(float(value) for value in values["object"]["canonical_pos"])
    cfg.canonical_object_quat = tuple(float(value) for value in values["object"]["canonical_quat_wxyz"])
    cfg.object_xy_randomization_m = float(values["object"]["xy_randomization_m"])
    cfg.object_z_clearance_range_m = tuple(float(value) for value in values["object"]["z_clearance_range_m"])
    cfg.object_z_noise_m = float(values["object"]["z_noise_m"])
    cfg.object_rpy_randomization_rad = math.radians(float(values["object"]["rpy_randomization_deg"]))
    cfg.object_mass_range_kg = tuple(float(value) for value in values["object"]["mass_range_kg"])
    cfg.friction_multiplier_range = tuple(float(value) for value in values["object"]["friction_multiplier_range"])
    for key, value in values["executor"].items():
        setattr(cfg, key, value)
    cfg.formal_contact_force_n = float(values["force"]["formal_contact_n"])
    cfg.soft_force_limit_n = float(values["force"]["soft_limit_n"])
    cfg.hard_force_abort_n = float(values["force"]["hard_abort_n"])
    cfg.episode_length_s = max(float(cfg.episode_length_s), float(episode_steps) * float(cfg.sim.dt))
    if replay:
        cfg.enable_replay_camera = bool(enable_replay_camera)
        cfg.enable_contact_attribution = bool(enable_contact_attribution)
        cfg.defer_terminal_reset = True
        cfg.scene.num_envs = 1
        cfg.replay_camera_eye_local = tuple(float(value) for value in values["replay"]["camera_eye_local"])
        cfg.replay_camera_target_local = tuple(float(value) for value in values["replay"]["camera_target_local"])
        cfg.replay_camera.width = int(values["replay"]["width"])
        cfg.replay_camera.height = int(values["replay"]["height"])
    return cfg


def apply_object_grasp_spec(
    cfg: NearGraspPhysicsEnvCfg,
    spec: ObjectGraspSpec,
    *,
    deterministic_gates: bool = True,
) -> NearGraspPhysicsEnvCfg:
    """Bind one object spec while keeping its physical USD unchanged."""

    cfg.target_part_name = spec.part_name
    cfg.target_asset_usd = spec.asset_usd
    cfg.target_asset_scale = spec.asset_scale
    cfg.target_object.prim_path = "/World/envs/env_.*/TargetObject"
    cfg.target_object.spawn.usd_path = spec.asset_usd
    cfg.target_object.spawn.scale = spec.asset_scale
    cfg.target_object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=spec.nominal_mass_kg)
    cfg.target_object.init_state.pos = spec.canonical_object_pos
    cfg.target_object.init_state.rot = spec.canonical_object_quat_wxyz
    cfg.target_contact_sensor.filter_prim_paths_expr = ["/World/envs/env_.*/TargetObject"]
    cfg.canonical_object_pos = spec.canonical_object_pos
    cfg.canonical_object_quat = spec.canonical_object_quat_wxyz
    cfg.object_mass_range_kg = (spec.nominal_mass_kg, spec.nominal_mass_kg)
    cfg.friction_multiplier_range = (spec.nominal_dynamic_friction, spec.nominal_dynamic_friction)
    cfg.randomize_physics_on_reset = not deterministic_gates
    if deterministic_gates:
        cfg.object_xy_randomization_m = 0.0
        cfg.object_z_clearance_range_m = (0.0, 0.0)
        cfg.object_z_noise_m = 0.0
        cfg.object_rpy_randomization_rad = 0.0
    return cfg


class NearGraspPhysicsEnv(DirectRLEnv):
    """Execute one GraspProgram per vector slot using only normal drive targets."""

    cfg: NearGraspPhysicsEnvCfg

    def __init__(self, cfg: NearGraspPhysicsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._wrist_ids = _ordered_joint_ids(self.robot, WUJI_FLOATING_WRIST_JOINT_NAMES, self.device)
        self._hand_ids = _ordered_joint_ids(self.robot, WUJI_HAND_FINGER_JOINT_NAMES, self.device)
        self._tip_body_ids = _ordered_body_ids(self.robot, WUJI_FINGERTIP_BODY_NAMES, self.device)
        palm_names = ("right_palm_link", "right_palm_base", "right_hand_base")
        palm_index = next((self.robot.body_names.index(name) for name in palm_names if name in self.robot.body_names), None)
        if palm_index is None:
            raise ValueError(f"Wuji palm body not found in {self.robot.body_names}")
        self._palm_body_id = int(palm_index)
        self._lower, self._upper = _joint_limits(self.robot, self.device)
        self.hand_preshape_q, self.hand_close_q = _runtime_hand_references(
            self.robot.data.default_joint_pos[0, self._hand_ids],
            self._lower[self._hand_ids],
            self._upper[self._hand_ids],
        )
        self._parked_wrist_q = torch.tensor(
            (-0.021207958459854126, 0.15737883746623993, 0.9691188335418701, -3.1415610313415527, -7.900382115622051e-06, -0.032967690378427505),
            dtype=torch.float32,
            device=self.device,
        )
        self._grasp_offset = torch.tensor(DEX_GRASP_FRAME_LOCAL_POS, dtype=torch.float32, device=self.device)
        self._canonical_object_quat = torch.tensor(self.cfg.canonical_object_quat, dtype=torch.float32, device=self.device)
        self._canonical_object_quat /= torch.clamp(torch.linalg.vector_norm(self._canonical_object_quat), min=1.0e-6)
        self._residual_bounds = ResidualActionBounds()
        self._residual_scale = torch.as_tensor(
            self._residual_bounds.upper,
            dtype=torch.float32,
            device=self.device,
        )
        self._evaluator = StrictPhysicalEvaluator()
        self._prior: HandPriorAdapter | None = None
        self._templates: dict[int, GraspTemplate] = {}
        self._allocate_program_buffers()
        self._contact_report_subscription = None
        self._contact_report_error = ""
        self._contact_pairs_by_env: list[set[tuple[str, str]]] = [set() for _ in range(self.num_envs)]
        self._forensic_contact_events_by_env: list[list[ForensicContactEvent]] = [
            [] for _ in range(self.num_envs)
        ]
        self._contact_report_available = self._subscribe_contact_reports() if self.cfg.enable_contact_attribution else False
        if self.replay_camera is not None:
            self._configure_replay_camera_view()
        self._completed: dict[int, PhysicalEvaluation] = {}
        self._completed_metadata: dict[int, dict[str, Any]] = {}
        self._completed_traces: dict[int, dict[str, np.ndarray]] = {}
        self.deterministic_stack_frozen = True
        self.near_grasp_search_allowed = True
        self.experimental_rl_allowed = True
        self.readiness_passed = False
        self.training_allowed = False

    def _subscribe_contact_reports(self) -> bool:
        try:
            import omni.physx
            from pxr import PhysicsSchemaTools

            interface = omni.physx.get_physx_simulation_interface()

            def callback(headers, contact_data) -> None:
                for header in headers:
                    first = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                    second = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                    match = re.search(r"/env_(\d+)/", first) or re.search(r"/env_(\d+)/", second)
                    if match is None:
                        continue
                    env_id = int(match.group(1))
                    if 0 <= env_id < self.num_envs:
                        self._contact_pairs_by_env[env_id].add((first, second))
                        offset = int(getattr(header, "contact_data_offset", 0))
                        count = int(getattr(header, "num_contact_data", 0))
                        rows = list(contact_data[offset : offset + count]) if count > 0 else []
                        self._forensic_contact_events_by_env[env_id].append(
                            ForensicContactEvent(
                                actor0=first,
                                actor1=second,
                                contact_count=count,
                                positions_world=tuple(_contact_vec3(row, "position") for row in rows),
                                normals_world=tuple(_contact_vec3(row, "normal") for row in rows),
                                separations_m=tuple(float(getattr(row, "separation", 0.0)) for row in rows),
                                impulses_ns=tuple(_contact_vec3(row, "impulse") for row in rows),
                            )
                        )

            self._contact_report_subscription = interface.subscribe_contact_report_events(callback)
            return self._contact_report_subscription is not None
        except Exception as exc:
            self._contact_report_subscription = None
            self._contact_report_error = f"{type(exc).__name__}:{exc}"
            return False

    def consume_contact_pairs(self) -> list[tuple[tuple[str, str], ...]]:
        rows = [tuple(sorted(pairs)) for pairs in self._contact_pairs_by_env]
        for pairs in self._contact_pairs_by_env:
            pairs.clear()
        return rows

    def consume_forensic_contact_events(self) -> list[tuple[ForensicContactEvent, ...]]:
        rows = [tuple(events) for events in self._forensic_contact_events_by_env]
        for events in self._forensic_contact_events_by_env:
            events.clear()
        return rows

    def robot_body_names_for_forensics(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.robot.body_names)

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, 0.0))
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=str(_TABLE_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        )
        table_cfg.scale = (1.0, 0.7, 1.0)
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        self.robot = Articulation(self.cfg.robot)
        self.target_object = RigidObject(self.cfg.target_object)
        self.target_contact_sensors = []
        self.table_contact_sensors = []
        self.ground_contact_sensors = []
        for tip_index, body_name in enumerate(WUJI_FINGERTIP_BODY_NAMES):
            sensor_cfg = copy.deepcopy(self.cfg.target_contact_sensor)
            sensor_cfg.prim_path = f"/World/envs/env_.*/Robot/{body_name}"
            sensor = ContactSensor(sensor_cfg)
            self.target_contact_sensors.append(sensor)
            self.scene.sensors[f"target_object_contact_{tip_index}"] = sensor
            if self.cfg.enable_contact_attribution:
                table_cfg = copy.deepcopy(sensor_cfg)
                table_cfg.filter_prim_paths_expr = ["/World/envs/env_.*/Table"]
                table_sensor = ContactSensor(table_cfg)
                self.table_contact_sensors.append(table_sensor)
                self.scene.sensors[f"table_contact_{tip_index}"] = table_sensor
                ground_cfg = copy.deepcopy(sensor_cfg)
                ground_cfg.filter_prim_paths_expr = ["/World/ground"]
                ground_sensor = ContactSensor(ground_cfg)
                self.ground_contact_sensors.append(ground_sensor)
                self.scene.sensors[f"ground_contact_{tip_index}"] = ground_sensor
        self.all_contact_sensor = ContactSensor(self.cfg.all_contact_sensor)
        self.replay_camera = TiledCamera(self.cfg.replay_camera) if self.cfg.enable_replay_camera else None
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target_object"] = self.target_object
        self.scene.sensors["fingertip_all_contact"] = self.all_contact_sensor
        if self.replay_camera is not None:
            self.scene.sensors["replay_camera"] = self.replay_camera
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.enable_contact_attribution:
            self._enable_contact_report_paths()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _enable_contact_report_paths(self) -> None:
        try:
            import omni.usd
            from pxr import PhysxSchema, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not any(marker in path for marker in ("/Robot/", "/TargetObject", "/Table", "/ground")):
                    continue
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report.CreateThresholdAttr().Set(0.0)
        except Exception:
            return

    def _allocate_program_buffers(self) -> None:
        n = self.num_envs
        d = self.device
        self._program_values = torch.zeros((n, 16), dtype=torch.float32, device=d)
        self._template_ids = torch.zeros(n, dtype=torch.long, device=d)
        self._template_relative_xyz = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self._template_quat = torch.zeros((n, 4), dtype=torch.float32, device=d)
        self._template_active_mask = torch.zeros((n, 5), dtype=torch.bool, device=d)
        self._template_preshape_target = torch.zeros((n, 20), dtype=torch.float32, device=d)
        self._candidate_close_target = torch.zeros((n, 20), dtype=torch.float32, device=d)
        self._env_ids = torch.arange(n, dtype=torch.long, device=d)
        self._hand_joint_finger_ids = torch.arange(20, dtype=torch.long, device=d) % 5
        self._reset_seeds = torch.arange(n, dtype=torch.long, device=d) + 20260713
        self._candidate_ids = torch.arange(n, dtype=torch.long, device=d)
        self._phase = torch.full((n,), int(ProgramPhase.COARSE_ACQUISITION), dtype=torch.long, device=d)
        self._phase_step = torch.zeros(n, dtype=torch.long, device=d)
        self._closure_progress = torch.zeros(n, dtype=torch.float32, device=d)
        self._first_contact_seen = torch.zeros(n, dtype=torch.bool, device=d)
        self._latched_contact = torch.zeros((n, 5), dtype=torch.bool, device=d)
        self._contact_below_keep = torch.zeros((n, 5), dtype=torch.long, device=d)
        self._latched_hand_target = torch.zeros((n, 20), dtype=torch.float32, device=d)
        self._current_target_force = torch.zeros((n, 5), dtype=torch.float32, device=d)
        self._current_target_force_xyz = torch.zeros((n, 5, 3), dtype=torch.float32, device=d)
        self._current_unfiltered_force = torch.zeros((n, 5), dtype=torch.float32, device=d)
        self._target_force_valid = torch.zeros(n, dtype=torch.bool, device=d)
        self._current_unfiltered_force_xyz = torch.zeros((n, 5, 3), dtype=torch.float32, device=d)
        self._current_table_force_xyz = torch.zeros((n, 5, 3), dtype=torch.float32, device=d)
        self._current_ground_force_xyz = torch.zeros((n, 5, 3), dtype=torch.float32, device=d)
        self._table_force_valid = torch.zeros(n, dtype=torch.bool, device=d)
        self._ground_force_valid = torch.zeros(n, dtype=torch.bool, device=d)
        self._unresolved_contact = torch.zeros(n, dtype=torch.bool, device=d)
        self._hard_abort = torch.zeros(n, dtype=torch.bool, device=d)
        self._flyout = torch.zeros(n, dtype=torch.bool, device=d)
        self._penetration = torch.zeros(n, dtype=torch.bool, device=d)
        self._controlled_close_completed = torch.zeros(n, dtype=torch.bool, device=d)
        self._close_start_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self._lift_start_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self._lift_progress_m = torch.zeros(n, dtype=torch.float32, device=d)
        self._weak_lift_pause = torch.zeros(n, dtype=torch.long, device=d)
        self._ctrl_target = self.robot.data.default_joint_pos.clone()
        self._previous_ctrl_target = self._ctrl_target.clone()
        self._previous_action = torch.zeros((n, 14), dtype=torch.float32, device=d)
        self._object_reset_pos_local = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self._object_reset_quat = torch.zeros((n, 4), dtype=torch.float32, device=d)
        self._object_mass_kg = torch.full((n,), 0.010, dtype=torch.float32, device=d)
        self._friction_multiplier = torch.ones(n, dtype=torch.float32, device=d)
        self._post_reset_object_writes = torch.zeros(n, dtype=torch.long, device=d)
        self._post_reset_wrist_state_writes = torch.zeros(n, dtype=torch.long, device=d)
        self._contact_counts = torch.zeros((n, 5), dtype=torch.long, device=d)
        self._history_length = int(math.ceil(self.cfg.episode_length_s / self.cfg.sim.dt)) + 8
        self._force_history = torch.zeros((self._history_length, n, 5), dtype=torch.float32, device=d)
        self._object_history = torch.zeros((self._history_length, n, 3), dtype=torch.float32, device=d)
        self._hand_history = torch.zeros((self._history_length, n, 3), dtype=torch.float32, device=d)
        self._support_history = torch.zeros((self._history_length, n), dtype=torch.bool, device=d)
        self._jerk_history = torch.zeros((self._history_length, n, 26), dtype=torch.float32, device=d)
        self._trace_length = torch.zeros(n, dtype=torch.long, device=d)
        self._termination = torch.full(
            (n,), int(ProgramTermination.NONE), dtype=torch.long, device=d
        )
        self._external_control_enabled = False
        self._external_reset_q26 = torch.zeros((n, 26), dtype=torch.float32, device=d)

    def configure_prior(self, prior: HandPriorAdapter) -> None:
        self._prior = prior

    def configure_privileged_control(
        self,
        reset_q26: Sequence[Sequence[float]],
        active_finger_masks: Sequence[Sequence[bool]],
        *,
        candidate_close_q20: Sequence[Sequence[float]] | None = None,
        reset_seeds: Sequence[int] | None = None,
    ) -> None:
        """Enable normal-drive external control with state writes confined to reset."""

        reset = np.asarray(reset_q26, dtype=np.float64)
        active = np.asarray(active_finger_masks, dtype=bool)
        if reset.shape != (self.num_envs, 26) or active.shape != (self.num_envs, 5):
            raise ValueError("privileged reset batch shapes are invalid")
        self._external_reset_q26[:] = torch.as_tensor(reset, dtype=torch.float32, device=self.device)
        self._template_active_mask[:] = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self._template_preshape_target[:] = self._external_reset_q26[:, 6:]
        if candidate_close_q20 is None:
            self._candidate_close_target[:] = self._external_reset_q26[:, 6:]
        else:
            close = np.asarray(candidate_close_q20, dtype=np.float64)
            if close.shape != (self.num_envs, 20):
                raise ValueError("candidate close target batch shape is invalid")
            self._candidate_close_target[:] = torch.as_tensor(close, dtype=torch.float32, device=self.device)
        self._external_control_enabled = True
        if reset_seeds is not None:
            seeds = np.asarray(reset_seeds, dtype=np.int64)
            if seeds.shape != (self.num_envs,):
                raise ValueError("privileged reset seed batch shape is invalid")
            self._reset_seeds[:] = torch.as_tensor(seeds, dtype=torch.long, device=self.device)

    def set_privileged_joint_targets(self, targets26: Sequence[Sequence[float]] | torch.Tensor) -> None:
        """Set articulation drive targets without writing simulation state."""

        target = torch.as_tensor(targets26, dtype=torch.float32, device=self.device)
        if target.shape != (self.num_envs, 26):
            raise ValueError("privileged joint targets must have shape [num_envs,26]")
        ordered = torch.cat((self._wrist_ids, self._hand_ids))
        bounded = torch.maximum(torch.minimum(target, self._upper[ordered]), self._lower[ordered])
        self._ctrl_target[:, ordered] = bounded

    def clear_reset_contact_cache_for_privileged_control(self) -> dict[str, Any]:
        """Clear stale pre-reset diagnostics before the first real physics step."""

        if not self._external_control_enabled or bool(torch.any(self.episode_length_buf != 0).item()):
            raise RuntimeError("reset contact cache can only be cleared at privileged episode step 0")
        audit = {
            "cleared_before_first_physics_step": True,
            "pair_counts": [len(pairs) for pairs in self._contact_pairs_by_env],
            "event_counts": [len(events) for events in self._forensic_contact_events_by_env],
            "target_force_peak_n": float(torch.max(self._current_target_force).item()),
            "unfiltered_force_peak_n": float(torch.max(self._current_unfiltered_force).item()),
            "control_target_modified": False,
            "physics_state_modified": False,
        }
        self._current_target_force.zero_()
        self._current_target_force_xyz.zero_()
        self._current_unfiltered_force.zero_()
        self._current_unfiltered_force_xyz.zero_()
        self._current_table_force_xyz.zero_()
        self._current_ground_force_xyz.zero_()
        self._unresolved_contact.zero_()
        for pairs in self._contact_pairs_by_env:
            pairs.clear()
        for events in self._forensic_contact_events_by_env:
            events.clear()
        return audit

    def privileged_snapshot(self) -> dict[str, np.ndarray]:
        """Return synchronized privileged state used by the contact controller."""

        self._refresh_contact_forces()
        jacobians = self.robot.root_physx_view.get_jacobians()
        tip_jacobians = []
        for body_id in self._tip_body_ids.tolist():
            tip_jacobians.append(jacobians[:, int(body_id) - 1, :3, self._hand_ids])
        object_local = self.target_object.data.root_pos_w - self.scene.env_origins
        tip_local = self.robot.data.body_pos_w[:, self._tip_body_ids] - self.scene.env_origins[:, None, :]
        tip_quat = self.robot.data.body_quat_w[:, self._tip_body_ids]
        palm_local = self.robot.data.body_pos_w[:, self._palm_body_id] - self.scene.env_origins
        return {
            "object_pos_local": object_local.detach().cpu().numpy().copy(),
            "object_quat_wxyz": self.target_object.data.root_quat_w.detach().cpu().numpy().copy(),
            "object_lin_vel": self.target_object.data.root_lin_vel_w.detach().cpu().numpy().copy(),
            "object_ang_vel": self.target_object.data.root_ang_vel_w.detach().cpu().numpy().copy(),
            "tip_pos_local": tip_local.detach().cpu().numpy().copy(),
            "tip_quat_wxyz": tip_quat.detach().cpu().numpy().copy(),
            "palm_pos_local": palm_local.detach().cpu().numpy().copy(),
            "tip_hand_jacobians": torch.stack(tip_jacobians, dim=1).detach().cpu().numpy().copy(),
            "joint_pos26": self.robot.data.joint_pos[:, torch.cat((self._wrist_ids, self._hand_ids))].detach().cpu().numpy().copy(),
            "joint_vel26": self.robot.data.joint_vel[:, torch.cat((self._wrist_ids, self._hand_ids))].detach().cpu().numpy().copy(),
            "joint_target26": self._ctrl_target[:, torch.cat((self._wrist_ids, self._hand_ids))].detach().cpu().numpy().copy(),
            "joint_sim_target26": self.robot.data.joint_pos_target[:, torch.cat((self._wrist_ids, self._hand_ids))].detach().cpu().numpy().copy(),
            "joint_lower26": self._lower[torch.cat((self._wrist_ids, self._hand_ids))].detach().cpu().numpy().copy(),
            "joint_upper26": self._upper[torch.cat((self._wrist_ids, self._hand_ids))].detach().cpu().numpy().copy(),
            "target_force_norms": self._current_target_force.detach().cpu().numpy().copy(),
            "target_force_xyz": self._current_target_force_xyz.detach().cpu().numpy().copy(),
            "all_force_xyz": self._current_unfiltered_force_xyz.detach().cpu().numpy().copy(),
            "table_force_xyz": self._current_table_force_xyz.detach().cpu().numpy().copy(),
            "ground_force_xyz": self._current_ground_force_xyz.detach().cpu().numpy().copy(),
            "target_filter_valid": self._target_force_valid.detach().cpu().numpy().copy(),
            "table_filter_valid": self._table_force_valid.detach().cpu().numpy().copy(),
            "ground_filter_valid": self._ground_force_valid.detach().cpu().numpy().copy(),
            "unresolved_contact": self._unresolved_contact.detach().cpu().numpy().copy(),
            "contact_report_available": np.full(self.num_envs, self._contact_report_available, dtype=bool),
            "post_reset_object_writes": self._post_reset_object_writes.detach().cpu().numpy().copy(),
            "post_reset_wrist_state_writes": self._post_reset_wrist_state_writes.detach().cpu().numpy().copy(),
        }

    def export_runtime_collision_geometry(self, output_dir: str | Path) -> dict[str, Any]:
        """Export TargetObject and Table collision meshes in metric planning frames."""

        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        env_origin = self.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
        object_pos = (self.target_object.data.root_pos_w[0] - self.scene.env_origins[0]).detach().cpu().numpy()
        object_quat = self.target_object.data.root_quat_w[0].detach().cpu().numpy()
        object_rotation = _quat_wxyz_matrix(object_quat)
        exports = {}
        for label, root_path, coordinate_frame in (
            ("target_object", "/World/envs/env_0/TargetObject", "object_local"),
            ("table", "/World/envs/env_0/Table", "env_local"),
        ):
            vertices = []
            faces = []
            prim_names = []
            inventory = []
            root_prim = stage.GetPrimAtPath(root_path)
            for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
                path = str(prim.GetPath())
                inventory.append(
                    {
                        "path": path,
                        "type": prim.GetTypeName(),
                        "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                        "mesh_collision_api": bool(prim.HasAPI(UsdPhysics.MeshCollisionAPI)),
                        "instance_proxy": bool(prim.IsInstanceProxy()),
                    }
                )
                if prim.GetTypeName() != "Mesh" or not _collision_enabled(prim, root_path, UsdPhysics):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                if not points or not counts or not indices:
                    continue
                matrix = cache.GetLocalToWorldTransform(prim)
                transformed = []
                for point in points:
                    world = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                    env_local = np.asarray((world[0], world[1], world[2]), dtype=np.float64) - env_origin
                    if coordinate_frame == "object_local":
                        transformed.append(object_rotation.T @ (env_local - object_pos))
                    else:
                        transformed.append(env_local)
                base = len(vertices)
                vertices.extend(transformed)
                cursor = 0
                for count in counts:
                    polygon = [base + int(value) for value in indices[cursor : cursor + int(count)]]
                    cursor += int(count)
                    for offset in range(1, len(polygon) - 1):
                        faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
                prim_names.append(path)
            vertex_array = np.asarray(vertices, dtype=np.float64)
            face_array = np.asarray(faces, dtype=np.int64)
            (destination / f"{label}_prim_inventory.json").write_text(
                json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or len(face_array) == 0:
                raise RuntimeError(f"no collision mesh was exported for {root_path}")
            digest = hashlib.sha256(vertex_array.tobytes() + face_array.tobytes()).hexdigest()
            npz_path = destination / f"{label}_collision_mesh.npz"
            np.savez_compressed(npz_path, vertices=vertex_array, faces=face_array)
            metadata = {
                "label": label,
                "root_path": root_path,
                "coordinate_frame": coordinate_frame,
                "source_prims": prim_names,
                "vertex_count": int(len(vertex_array)),
                "face_count": int(len(face_array)),
                "aabb_min": np.min(vertex_array, axis=0).tolist(),
                "aabb_max": np.max(vertex_array, axis=0).tolist(),
                "mesh_sha256": digest,
                "npz_path": str(npz_path),
            }
            metadata_path = destination / f"{label}_collision_mesh.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            exports[label] = metadata
        manifest = {
            "target_object": exports["target_object"],
            "table": exports["table"],
            "runtime_physics_fingerprint": self.physics_fingerprint(),
        }
        (destination / "runtime_collision_geometry_audit.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest

    def _configure_replay_camera_view(self) -> None:
        if self.replay_camera is None:
            return
        eye = torch.tensor(self.cfg.replay_camera_eye_local, dtype=torch.float32, device=self.device)
        target = torch.tensor(self.cfg.replay_camera_target_local, dtype=torch.float32, device=self.device)
        eyes = self.scene.env_origins + eye.reshape(1, 3)
        targets = self.scene.env_origins + target.reshape(1, 3)
        self.replay_camera.set_world_poses_from_view(eyes, targets)

    def set_program_batch(
        self,
        programs: Sequence[GraspProgram],
        templates: Sequence[GraspTemplate],
        *,
        reset_seeds: Sequence[int],
        candidate_ids: Sequence[int] | None = None,
        candidate_close_targets: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if len(programs) != self.num_envs or len(reset_seeds) != self.num_envs:
            raise ValueError(f"Expected exactly {self.num_envs} programs and reset seeds")
        self._templates = {int(template.template_id): template for template in templates}
        missing = sorted({program.template_id for program in programs} - set(self._templates))
        if missing:
            raise ValueError(f"Missing template definitions: {missing}")
        self._program_values[:] = torch.as_tensor(
            [program.values for program in programs], dtype=torch.float32, device=self.device
        )
        self._template_ids[:] = torch.as_tensor(
            [program.template_id for program in programs], dtype=torch.long, device=self.device
        )
        selected_templates = [self._templates[program.template_id] for program in programs]
        self._template_relative_xyz[:] = torch.as_tensor(
            [template.dex_pose_relative_object_xyz_m for template in selected_templates],
            dtype=torch.float32,
            device=self.device,
        )
        self._template_quat[:] = torch.as_tensor(
            [template.dex_quat_wxyz for template in selected_templates],
            dtype=torch.float32,
            device=self.device,
        )
        self._template_active_mask[:] = torch.as_tensor(
            [template.active_finger_mask for template in selected_templates],
            dtype=torch.bool,
            device=self.device,
        )
        resolved_preshapes = [
            resolve_preshape_profile(
                template.preshape_profile,
                self.hand_preshape_q.detach().cpu().numpy(),
                self.hand_close_q.detach().cpu().numpy(),
                template.active_finger_mask,
            )
            for template in selected_templates
        ]
        self._template_preshape_target[:] = torch.as_tensor(
            np.asarray(resolved_preshapes), dtype=torch.float32, device=self.device
        )
        if candidate_close_targets is None:
            self._candidate_close_target[:] = self.hand_close_q.reshape(1, -1)
        else:
            targets = np.asarray(candidate_close_targets, dtype=np.float64)
            if targets.shape != (self.num_envs, 20) or not np.all(np.isfinite(targets)):
                raise ValueError("candidate_close_targets must have shape [num_envs,20]")
            self._candidate_close_target[:] = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        self._reset_seeds[:] = torch.as_tensor(reset_seeds, dtype=torch.long, device=self.device)
        ids = list(range(self.num_envs)) if candidate_ids is None else list(candidate_ids)
        if len(ids) != self.num_envs:
            raise ValueError("candidate_ids length mismatch")
        self._candidate_ids[:] = torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions.to(device=self.device, dtype=torch.float32), -1.0, 1.0)
        self._previous_action[:] = actions
        if self._external_control_enabled:
            return
        residual = actions * self._residual_scale.reshape(1, -1)
        self._advance_program_vectorized(residual)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._ctrl_target)

    def _advance_program_vectorized(self, residual: torch.Tensor) -> None:
        """Advance every slot without per-environment CUDA scalar reads."""

        object_local = self.target_object.data.root_pos_w - self.scene.env_origins
        object_quat = self.target_object.data.root_quat_w
        final_wrist = self._program_wrist_targets_vectorized(object_local, object_quat, residual)
        high_wrist = final_wrist.clone()
        high_wrist[:, 2] = torch.maximum(high_wrist[:, 2], torch.full_like(high_wrist[:, 2], 1.1005))
        active = self._template_active_mask
        phase_start = self._phase.clone()

        terminal = (phase_start == int(ProgramPhase.COMPLETE)) | (phase_start == int(ProgramPhase.INVALID))
        hard_abort = torch.max(self._current_target_force, dim=1).values >= self.cfg.hard_force_abort_n
        self._hard_abort |= hard_abort
        invalid_contact = self._unresolved_contact
        newly_invalid = torch.logical_not(terminal) & (hard_abort | invalid_contact)
        self._set_termination(hard_abort & torch.logical_not(terminal), ProgramTermination.HARD_FORCE_ABORT)
        self._set_termination(
            invalid_contact & torch.logical_not(terminal | hard_abort),
            ProgramTermination.UNRESOLVED_CONTACT,
        )
        self._phase[newly_invalid] = int(ProgramPhase.INVALID)
        runnable = torch.logical_not(terminal | hard_abort | invalid_contact)

        formal = active & (self._current_target_force > self.cfg.formal_contact_force_n)
        newly_latched = formal & torch.logical_not(self._latched_contact) & runnable.reshape(-1, 1)
        self._latched_contact |= newly_latched
        below_keep = (
            active
            & self._latched_contact
            & (self._current_target_force < 0.035)
            & runnable.reshape(-1, 1)
        )
        self._contact_below_keep = torch.where(
            below_keep,
            self._contact_below_keep + 1,
            torch.zeros_like(self._contact_below_keep),
        )
        lost_contact = self._contact_below_keep >= 3
        self._latched_contact &= torch.logical_not(lost_contact)
        self._contact_below_keep[lost_contact] = 0
        newly_latched_joints = newly_latched[:, self._hand_joint_finger_ids]
        current_hand_target = self._ctrl_target[:, self._hand_ids]
        self._latched_hand_target[:] = torch.where(
            newly_latched_joints,
            current_hand_target,
            self._latched_hand_target,
        )

        coarse = runnable & (phase_start == int(ProgramPhase.COARSE_ACQUISITION))
        coarse_ratio = torch.clamp(
            (self._phase_step.to(dtype=torch.float32) + 1.0) / float(self.cfg.coarse_acquisition_steps),
            max=1.0,
        )
        coarse_wrist = torch.lerp(
            self._parked_wrist_q.reshape(1, -1).expand_as(high_wrist),
            high_wrist,
            coarse_ratio.reshape(-1, 1),
        )
        self._write_joint_target_rows(coarse, self._wrist_ids, coarse_wrist)
        self._write_joint_target_rows(
            coarse,
            self._hand_ids,
            self._template_preshape_target,
        )
        self._advance_or_transition_vectorized(
            coarse,
            coarse_ratio >= 1.0,
            ProgramPhase.WRIST_RESIDUAL_APPROACH,
        )

        approach = runnable & (phase_start == int(ProgramPhase.WRIST_RESIDUAL_APPROACH))
        approach_steps = torch.clamp(torch.round(self._program_values[:, 12]), min=1.0)
        approach_ratio = torch.clamp(
            (self._phase_step.to(dtype=torch.float32) + 1.0) / approach_steps,
            max=1.0,
        )
        approach_wrist = torch.lerp(high_wrist, final_wrist, approach_ratio.reshape(-1, 1))
        self._write_joint_target_rows(approach, self._wrist_ids, approach_wrist)
        self._advance_or_transition_vectorized(
            approach,
            approach_ratio >= 1.0,
            ProgramPhase.LATENT_CLOSURE,
        )

        latent = runnable & (phase_start == int(ProgramPhase.LATENT_CLOSURE))
        remaining = runnable & (phase_start == int(ProgramPhase.REMAINING_FINGER_PROGRESSION))
        latent_motion = latent | remaining
        prior_targets = None
        if bool(torch.any(latent_motion).item()):
            prior_targets = self._batch_prior_targets(residual[:, :6])
            self._set_latent_targets_vectorized(latent_motion, residual, prior_targets)
        self._write_joint_target_rows(latent, self._wrist_ids, final_wrist)
        first_contact = latent & torch.any(formal, dim=1)
        self._first_contact_seen |= first_contact
        self._transition_rows(first_contact, ProgramPhase.FIRST_CONTACT_HOLD)
        exhausted_latent = (
            latent
            & torch.logical_not(first_contact)
            & (self._closure_progress >= 1.0)
            & (self._phase_step > 30)
        )
        self._phase[exhausted_latent] = int(ProgramPhase.INVALID)
        self._set_termination(exhausted_latent, ProgramTermination.LATENT_CLOSURE_EXHAUSTED)
        self._phase_step[latent & torch.logical_not(first_contact | exhausted_latent)] += 1

        first_hold = runnable & (phase_start == int(ProgramPhase.FIRST_CONTACT_HOLD))
        hold_steps = torch.clamp(torch.round(self._program_values[:, 14] + residual[:, 13]), min=1.0)
        self._advance_or_transition_vectorized(
            first_hold,
            self._phase_step.to(dtype=torch.float32) >= hold_steps,
            ProgramPhase.REMAINING_FINGER_PROGRESSION,
        )

        all_active_contact = torch.all(torch.logical_or(torch.logical_not(active), formal), dim=1)
        remaining_contact = remaining & all_active_contact
        self._transition_rows(remaining_contact, ProgramPhase.MULTI_CONTACT_HOLD)
        remaining_timeout = remaining & torch.logical_not(remaining_contact) & (
            self._phase_step >= self.cfg.max_remaining_finger_steps
        )
        self._phase[remaining_timeout] = int(ProgramPhase.INVALID)
        self._set_termination(remaining_timeout, ProgramTermination.REMAINING_CONTACT_TIMEOUT)
        self._phase_step[remaining & torch.logical_not(remaining_contact | remaining_timeout)] += 1

        multi_hold = runnable & (phase_start == int(ProgramPhase.MULTI_CONTACT_HOLD))
        multi_done = multi_hold & (self._phase_step >= self.cfg.multi_contact_hold_steps)
        self._transition_rows(multi_done, ProgramPhase.CONTROLLED_CLOSE)
        self._close_start_step[multi_done] = self.episode_length_buf[multi_done]
        self._phase_step[multi_hold & torch.logical_not(multi_done)] += 1

        controlled_close = runnable & (phase_start == int(ProgramPhase.CONTROLLED_CLOSE))
        close_motion = controlled_close & (self._phase_step < self.cfg.controlled_close_steps)
        close_current = self._ctrl_target[:, self._hand_ids]
        close_delta = torch.clamp(self._candidate_close_target - close_current, -0.001, 0.001)
        active_joints = active[:, self._hand_joint_finger_ids]
        contacted_joints = self._latched_contact[:, self._hand_joint_finger_ids]
        close_delta *= (active_joints & torch.logical_not(contacted_joints)).to(dtype=close_delta.dtype)
        self._write_joint_target_rows(close_motion, self._hand_ids, close_current + close_delta)
        close_finished_now = controlled_close & (self._phase_step == self.cfg.controlled_close_steps)
        self._controlled_close_completed |= close_finished_now
        post_steps = self.cfg.post_close_hold_steps + torch.clamp(
            torch.round(self._program_values[:, 15] + residual[:, 13]),
            min=0.0,
        )
        close_done = controlled_close & (
            self._phase_step.to(dtype=torch.float32) >= float(self.cfg.controlled_close_steps) + post_steps
        )
        self._transition_rows(close_done, ProgramPhase.SLOW_LIFT)
        self._lift_start_step[close_done] = self.episode_length_buf[close_done]
        self._phase_step[controlled_close & torch.logical_not(close_done)] += 1

        slow_lift = runnable & (phase_start == int(ProgramPhase.SLOW_LIFT))
        lift_ready = slow_lift & all_active_contact
        lift_weak = slow_lift & torch.logical_not(all_active_contact)
        self._weak_lift_pause[lift_weak] += 1
        self._weak_lift_pause[lift_ready] = 0
        lift_timeout = lift_weak & (self._weak_lift_pause > 10)
        self._phase[lift_timeout] = int(ProgramPhase.INVALID)
        self._set_termination(lift_timeout, ProgramTermination.LIFT_CONTACT_LOSS)
        self._ctrl_target[lift_ready, self._wrist_ids[2]] += self.cfg.lift_increment_m
        self._lift_progress_m[lift_ready] += self.cfg.lift_increment_m
        self._phase_step[lift_ready] += 1
        lift_done = lift_ready & (self._phase_step >= self.cfg.max_lift_steps)
        self._phase[lift_done] = int(ProgramPhase.COMPLETE)
        self._set_termination(lift_done, ProgramTermination.COMPLETE)

        force_guard = first_hold | remaining | multi_hold | close_motion | (lift_weak & torch.logical_not(lift_timeout))
        self._apply_force_guard_vectorized(force_guard, active)

        self._ctrl_target[:] = torch.maximum(torch.minimum(self._ctrl_target, self._upper), self._lower)
        object_motion = torch.linalg.vector_norm(object_local - self._object_reset_pos_local, dim=1)
        self._flyout |= object_motion > 0.10
        self._penetration |= object_local[:, 2] < 0.60
        invalid = self._flyout | self._penetration
        self._set_termination(self._flyout, ProgramTermination.FLYOUT)
        self._set_termination(self._penetration, ProgramTermination.PENETRATION)
        self._phase[invalid] = int(ProgramPhase.INVALID)

    def _set_latent_targets_vectorized(
        self,
        rows: torch.Tensor,
        residual: torch.Tensor,
        prior_targets: torch.Tensor,
    ) -> None:
        speed = torch.clamp(self._program_values[:, 13] + residual[:, 12], min=0.25, max=1.5)
        next_progress = torch.clamp(self._closure_progress + 0.006 * speed, max=1.0)
        self._closure_progress[rows] = next_progress[rows]
        active_joints = self._template_active_mask[:, self._hand_joint_finger_ids]
        latched_joints = self._latched_contact[:, self._hand_joint_finger_ids]
        nominal = torch.where(
            active_joints,
            prior_targets,
            self._template_preshape_target,
        )
        nominal = torch.where(active_joints & latched_joints, self._latched_hand_target, nominal)
        actual = self.robot.data.joint_pos[:, self._hand_ids]
        target = actual + torch.clamp(nominal - actual, -0.01, 0.01)
        self._write_joint_target_rows(rows, self._hand_ids, target)

    def _apply_force_guard_vectorized(self, rows: torch.Tensor, active: torch.Tensor) -> None:
        guarded_fingers = rows.reshape(-1, 1) & active & (
            self._current_target_force > self.cfg.soft_force_limit_n
        )
        guarded_joints = guarded_fingers[:, self._hand_joint_finger_ids]
        actual = self.robot.data.joint_pos[:, self._hand_ids]
        target = self._ctrl_target[:, self._hand_ids]
        retreat = actual - 0.25 * (target - actual)
        self._ctrl_target[:, self._hand_ids] = torch.where(guarded_joints, retreat, target)

    def _write_joint_target_rows(
        self,
        rows: torch.Tensor,
        joint_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        current = self._ctrl_target[:, joint_ids]
        self._ctrl_target[:, joint_ids] = torch.where(rows.reshape(-1, 1), values, current)

    def _advance_or_transition_vectorized(
        self,
        rows: torch.Tensor,
        transition: torch.Tensor,
        next_phase: ProgramPhase,
    ) -> None:
        transition_rows = rows & transition
        self._transition_rows(transition_rows, next_phase)
        self._phase_step[rows & torch.logical_not(transition_rows)] += 1

    def _transition_rows(self, rows: torch.Tensor, phase: ProgramPhase) -> None:
        self._phase[rows] = int(phase)
        self._phase_step[rows] = 0

    def _set_termination(self, rows: torch.Tensor, reason: ProgramTermination) -> None:
        writable = rows & (self._termination == int(ProgramTermination.NONE))
        self._termination[writable] = int(reason)

    def _program_wrist_targets_vectorized(
        self,
        object_local: torch.Tensor,
        object_quat: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        program_xyz = self._program_values[:, :3] + residual[:, 6:9]
        dex_pos = object_local + quat_apply(object_quat, self._template_relative_xyz + program_xyz)
        canonical_inverse = quat_conjugate(self._canonical_object_quat).reshape(1, 4).expand_as(object_quat)
        object_delta = quat_mul(object_quat, canonical_inverse)
        base_quat = quat_mul(object_delta, self._template_quat)
        rpy = self._program_values[:, 3:6] + residual[:, 9:12]
        residual_quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        dex_quat = quat_mul(base_quat, residual_quat)
        dex_quat /= torch.clamp(torch.linalg.vector_norm(dex_quat, dim=1, keepdim=True), min=1.0e-6)
        grasp_offset = self._grasp_offset.reshape(1, 3).expand(self.num_envs, -1)
        palm_pos = dex_pos - quat_apply(dex_quat, grasp_offset)
        return torch.cat((palm_pos, _serial_xyz_euler_from_quat(dex_quat)), dim=1)

    def _advance_program(self, residual: torch.Tensor) -> None:
        object_local = self.target_object.data.root_pos_w - self.scene.env_origins
        object_quat = self.target_object.data.root_quat_w
        final_wrist = self._program_wrist_targets(object_local, object_quat, residual)
        high_wrist = final_wrist.clone()
        high_wrist[:, 2] = torch.maximum(high_wrist[:, 2], torch.full_like(high_wrist[:, 2], 1.1005))
        high_wrist[:, 3:6] = final_wrist[:, 3:6]
        active_masks = self._active_finger_masks()
        prior_targets = self._batch_prior_targets(residual[:, :6])

        for env_id in range(self.num_envs):
            phase = ProgramPhase(int(self._phase[env_id].item()))
            if phase in {ProgramPhase.COMPLETE, ProgramPhase.INVALID}:
                continue
            force = self._current_target_force[env_id]
            if float(torch.max(force).item()) >= self.cfg.hard_force_abort_n:
                self._hard_abort[env_id] = True
                self._phase[env_id] = int(ProgramPhase.INVALID)
                continue
            if self._unresolved_contact[env_id]:
                self._phase[env_id] = int(ProgramPhase.INVALID)
                continue
            active = active_masks[env_id]
            formal = active & (force > self.cfg.formal_contact_force_n)
            below_keep = active & self._latched_contact[env_id] & (force < 0.035)
            self._contact_below_keep[env_id] = torch.where(
                below_keep,
                self._contact_below_keep[env_id] + 1,
                torch.zeros_like(self._contact_below_keep[env_id]),
            )
            lost = self._contact_below_keep[env_id] >= 3
            self._latched_contact[env_id] &= torch.logical_not(lost)
            self._contact_below_keep[env_id, lost] = 0
            newly = formal & torch.logical_not(self._latched_contact[env_id])
            if bool(torch.any(newly)):
                self._latched_contact[env_id] |= newly
                for finger_id in torch.where(newly)[0].tolist():
                    joint_local = _finger_joint_local_indices(finger_id, self.device)
                    self._latched_hand_target[env_id, joint_local] = self._ctrl_target[env_id, self._hand_ids[joint_local]]

            if phase == ProgramPhase.COARSE_ACQUISITION:
                ratio = min(1.0, float(self._phase_step[env_id].item() + 1) / self.cfg.coarse_acquisition_steps)
                self._ctrl_target[env_id, self._wrist_ids] = torch.lerp(self._parked_wrist_q, high_wrist[env_id], ratio)
                self._ctrl_target[env_id, self._hand_ids] = self._template_preshape_target[env_id]
                if ratio >= 1.0:
                    self._transition(env_id, ProgramPhase.WRIST_RESIDUAL_APPROACH)
                else:
                    self._phase_step[env_id] += 1
            elif phase == ProgramPhase.WRIST_RESIDUAL_APPROACH:
                approach_steps = int(round(float(self._program_values[env_id, 12].item())))
                ratio = min(1.0, float(self._phase_step[env_id].item() + 1) / max(approach_steps, 1))
                self._ctrl_target[env_id, self._wrist_ids] = torch.lerp(high_wrist[env_id], final_wrist[env_id], ratio)
                if ratio >= 1.0:
                    self._transition(env_id, ProgramPhase.LATENT_CLOSURE)
                else:
                    self._phase_step[env_id] += 1
            elif phase == ProgramPhase.LATENT_CLOSURE:
                self._ctrl_target[env_id, self._wrist_ids] = final_wrist[env_id]
                self._advance_latent_target(env_id, active, residual, prior_targets[env_id])
                if bool(torch.any(formal)):
                    self._first_contact_seen[env_id] = True
                    self._transition(env_id, ProgramPhase.FIRST_CONTACT_HOLD)
                elif float(self._closure_progress[env_id].item()) >= 1.0 and int(self._phase_step[env_id]) > 30:
                    self._phase[env_id] = int(ProgramPhase.INVALID)
                else:
                    self._phase_step[env_id] += 1
            elif phase == ProgramPhase.FIRST_CONTACT_HOLD:
                self._apply_force_guard(env_id, active)
                hold_steps = int(round(float(self._program_values[env_id, 14].item() + residual[env_id, 13].item())))
                if int(self._phase_step[env_id]) >= max(1, hold_steps):
                    self._transition(env_id, ProgramPhase.REMAINING_FINGER_PROGRESSION)
                else:
                    self._phase_step[env_id] += 1
            elif phase == ProgramPhase.REMAINING_FINGER_PROGRESSION:
                self._advance_latent_target(env_id, active, residual, prior_targets[env_id])
                self._apply_force_guard(env_id, active)
                if bool(torch.all(torch.logical_or(torch.logical_not(active), formal))):
                    self._transition(env_id, ProgramPhase.MULTI_CONTACT_HOLD)
                elif int(self._phase_step[env_id]) >= self.cfg.max_remaining_finger_steps:
                    self._phase[env_id] = int(ProgramPhase.INVALID)
                else:
                    self._phase_step[env_id] += 1
            elif phase == ProgramPhase.MULTI_CONTACT_HOLD:
                self._apply_force_guard(env_id, active)
                if int(self._phase_step[env_id]) >= self.cfg.multi_contact_hold_steps:
                    self._transition(env_id, ProgramPhase.CONTROLLED_CLOSE)
                    self._close_start_step[env_id] = self.episode_length_buf[env_id]
                else:
                    self._phase_step[env_id] += 1
            elif phase == ProgramPhase.CONTROLLED_CLOSE:
                self._controlled_close_and_hold(env_id, active, residual)
            elif phase == ProgramPhase.SLOW_LIFT:
                self._slow_lift(env_id, active)

        self._ctrl_target[:] = torch.maximum(torch.minimum(self._ctrl_target, self._upper), self._lower)
        object_motion = torch.linalg.vector_norm(object_local - self._object_reset_pos_local, dim=1)
        self._flyout |= object_motion > 0.10
        self._penetration |= object_local[:, 2] < 0.60
        invalid = self._flyout | self._penetration
        self._phase[invalid] = int(ProgramPhase.INVALID)

    def _advance_latent_target(
        self,
        env_id: int,
        active: torch.Tensor,
        residual: torch.Tensor,
        prior_target: torch.Tensor,
    ) -> None:
        speed = torch.clamp(
            self._program_values[env_id, 13] + residual[env_id, 12],
            min=0.25,
            max=1.5,
        )
        progress = min(1.0, float(self._closure_progress[env_id].item()) + 0.006 * float(speed.item()))
        self._closure_progress[env_id] = progress
        nominal = prior_target.clone()
        for finger_id in range(5):
            local_ids = _finger_joint_local_indices(finger_id, self.device)
            if not bool(active[finger_id]):
                nominal[local_ids] = self._template_preshape_target[env_id, local_ids]
            elif bool(self._latched_contact[env_id, finger_id]):
                nominal[local_ids] = self._latched_hand_target[env_id, local_ids]
        lead = torch.clamp(nominal - self.robot.data.joint_pos[env_id, self._hand_ids], -0.01, 0.01)
        self._ctrl_target[env_id, self._hand_ids] = self.robot.data.joint_pos[env_id, self._hand_ids] + lead

    def _controlled_close_and_hold(self, env_id: int, active: torch.Tensor, residual: torch.Tensor) -> None:
        step = int(self._phase_step[env_id].item())
        close_steps = self.cfg.controlled_close_steps
        post_steps = self.cfg.post_close_hold_steps + max(
            0,
            int(round(float(self._program_values[env_id, 15].item() + residual[env_id, 13].item()))),
        )
        if step < close_steps:
            current = self._ctrl_target[env_id, self._hand_ids]
            error = self._candidate_close_target[env_id] - current
            delta = torch.clamp(error, -0.001, 0.001)
            for finger_id in range(5):
                if not bool(active[finger_id]) or bool(self._latched_contact[env_id, finger_id]):
                    delta[_finger_joint_local_indices(finger_id, self.device)] = 0.0
            self._ctrl_target[env_id, self._hand_ids] = current + delta
            self._apply_force_guard(env_id, active)
        elif step == close_steps:
            self._controlled_close_completed[env_id] = True
        if step >= close_steps + post_steps:
            self._transition(env_id, ProgramPhase.SLOW_LIFT)
            self._lift_start_step[env_id] = self.episode_length_buf[env_id]
        else:
            self._phase_step[env_id] += 1

    def _slow_lift(self, env_id: int, active: torch.Tensor) -> None:
        formal = active & (self._current_target_force[env_id] > self.cfg.formal_contact_force_n)
        if not bool(torch.all(torch.logical_or(torch.logical_not(active), formal))):
            self._weak_lift_pause[env_id] += 1
            self._apply_force_guard(env_id, active)
            if int(self._weak_lift_pause[env_id]) > 10:
                self._phase[env_id] = int(ProgramPhase.INVALID)
            return
        self._weak_lift_pause[env_id] = 0
        self._ctrl_target[env_id, self._wrist_ids[2]] += self.cfg.lift_increment_m
        self._lift_progress_m[env_id] += self.cfg.lift_increment_m
        self._phase_step[env_id] += 1
        if int(self._phase_step[env_id]) >= self.cfg.max_lift_steps:
            self._phase[env_id] = int(ProgramPhase.COMPLETE)

    def _apply_force_guard(self, env_id: int, active: torch.Tensor) -> None:
        for finger_id in range(5):
            if not bool(active[finger_id]):
                continue
            force = float(self._current_target_force[env_id, finger_id].item())
            if force > self.cfg.soft_force_limit_n:
                local = _finger_joint_local_indices(finger_id, self.device)
                actual = self.robot.data.joint_pos[env_id, self._hand_ids[local]]
                target = self._ctrl_target[env_id, self._hand_ids[local]]
                self._ctrl_target[env_id, self._hand_ids[local]] = actual - 0.25 * (target - actual)

    def _batch_prior_targets(self, residual_latent: torch.Tensor) -> torch.Tensor:
        if self._prior is None:
            progress = self._closure_progress.reshape(-1, 1)
            return self.hand_preshape_q.reshape(1, -1) + progress * (
                self.hand_close_q - self.hand_preshape_q
            ).reshape(1, -1)
        program_latent = self._program_values[:, 6:12] + residual_latent
        decoded = self._prior.decode_torch(
            self._coordex_proprio(),
            program_latent,
            self._closure_progress,
        )
        if torch.is_tensor(decoded):
            return decoded.to(device=self.device, dtype=torch.float32)
        proprio = self._coordex_proprio().detach().cpu().numpy()
        program_latent_np = program_latent.detach().cpu().numpy()
        progress = self._closure_progress.detach().cpu().numpy()
        decoded = self._prior.decode(proprio, program_latent_np, progress)
        return torch.as_tensor(decoded, dtype=torch.float32, device=self.device)

    def _program_wrist_targets(
        self,
        object_local: torch.Tensor,
        object_quat: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        target = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        for env_id in range(self.num_envs):
            template = self._templates.get(int(self._template_ids[env_id].item()))
            if template is None:
                continue
            relative = torch.tensor(template.dex_pose_relative_object_xyz_m, dtype=torch.float32, device=self.device)
            program_xyz = self._program_values[env_id, :3] + residual[env_id, 6:9]
            dex_pos = object_local[env_id] + quat_apply(object_quat[env_id], relative + program_xyz)
            template_quat = torch.tensor(template.dex_quat_wxyz, dtype=torch.float32, device=self.device)
            object_delta = quat_mul(object_quat[env_id], quat_conjugate(self._canonical_object_quat))
            base_quat = quat_mul(object_delta, template_quat)
            rpy = self._program_values[env_id, 3:6] + residual[env_id, 9:12]
            residual_quat = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3])[0]
            dex_quat = quat_mul(base_quat, residual_quat)
            dex_quat /= torch.clamp(torch.linalg.vector_norm(dex_quat), min=1.0e-6)
            palm_pos = dex_pos - quat_apply(dex_quat, self._grasp_offset)
            target[env_id, :3] = palm_pos
            target[env_id, 3:6] = _serial_xyz_euler_from_quat(dex_quat)
        return target

    def _active_finger_masks(self) -> torch.Tensor:
        return self._template_active_mask

    def _transition(self, env_id: int, phase: ProgramPhase) -> None:
        self._phase[env_id] = int(phase)
        self._phase_step[env_id] = 0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_contact_forces()
        self._record_step()
        object_pos = self.target_object.data.root_pos_w
        object_quat = self.target_object.data.root_quat_w
        object_vel = self.target_object.data.root_vel_w
        body_pos = self.robot.data.body_pos_w
        body_quat = self.robot.data.body_quat_w
        palm_pos = body_pos[:, self._palm_body_id]
        palm_quat = body_quat[:, self._palm_body_id]
        tip_pos = body_pos[:, self._tip_body_ids]
        relative_quat = quat_mul(object_quat, quat_conjugate(palm_quat))
        hand_q = self.robot.data.joint_pos[:, self._hand_ids]
        hand_qdot = self.robot.data.joint_vel[:, self._hand_ids]
        wrist_q = self.robot.data.joint_pos[:, self._wrist_ids]
        wrist_qdot = self.robot.data.joint_vel[:, self._wrist_ids]
        active = self._active_finger_masks()
        contact_flag = self._current_target_force > self.cfg.formal_contact_force_n
        duty = self._contact_counts.to(dtype=torch.float32) / torch.clamp(
            self.episode_length_buf.to(dtype=torch.float32).reshape(-1, 1) + 1.0,
            min=1.0,
        )
        object_local = object_pos - self.scene.env_origins
        table_clearance = object_local[:, 2] - self.cfg.table_top_z_m
        table_supported = (table_clearance < 0.004).to(dtype=torch.float32)
        embedding = torch.stack(
            (
                torch.full_like(table_clearance, 0.006),
                torch.full_like(table_clearance, 0.025),
                self._object_mass_kg,
                self._friction_multiplier,
                torch.ones_like(table_clearance),
                torch.zeros_like(table_clearance),
                active[:, 2].to(dtype=torch.float32),
                active[:, 3].to(dtype=torch.float32),
            ),
            dim=1,
        )
        fields = {
            "object_palm_relative_pose": torch.cat((object_pos - palm_pos, relative_quat), dim=1),
            "object_velocity": object_vel,
            "fingertip_relative_positions": (tip_pos - object_pos.unsqueeze(1)).reshape(self.num_envs, -1),
            "hand_q": hand_q,
            "hand_qdot": hand_qdot,
            "hand_target_error": self._ctrl_target[:, self._hand_ids] - hand_q,
            "wrist_q": wrist_q,
            "wrist_qdot": wrist_qdot,
            "wrist_target_error": self._ctrl_target[:, self._wrist_ids] - wrist_q,
            "target_force_norms": self._current_target_force,
            "target_force_vectors": self._current_target_force_xyz.reshape(self.num_envs, -1),
            "contact_history_duty": torch.cat((contact_flag.to(dtype=torch.float32), duty), dim=1),
            "table_state": torch.stack(
                (
                    torch.full_like(table_clearance, self.cfg.table_top_z_m),
                    table_clearance,
                    table_supported,
                    torch.zeros_like(table_clearance),
                ),
                dim=1,
            ),
            "object_hand_transform": torch.cat((object_pos - palm_pos, relative_quat), dim=1),
            "object_embedding": embedding,
            "previous_action": self._previous_action,
        }
        obs = torch.cat([fields[name] for name, _ in NEAR_GRASP_OBSERVATION_SCHEMA.names_and_widths], dim=1)
        return {"policy": obs}

    def _refresh_contact_forces(self) -> None:
        target_xyz, target_valid = self._filtered_force_xyz(self.target_contact_sensors)
        self._current_target_force_xyz[:] = target_xyz
        self._current_target_force[:] = torch.linalg.vector_norm(target_xyz, dim=-1)
        self._target_force_valid[:] = target_valid
        table_xyz, table_valid = self._filtered_force_xyz(self.table_contact_sensors)
        ground_xyz, ground_valid = self._filtered_force_xyz(self.ground_contact_sensors)
        self._current_table_force_xyz[:] = table_xyz
        self._current_ground_force_xyz[:] = ground_xyz
        self._table_force_valid[:] = table_valid
        self._ground_force_valid[:] = ground_valid
        all_forces = getattr(getattr(self.all_contact_sensor, "data", None), "net_forces_w", None)
        if torch.is_tensor(all_forces) and all_forces.ndim == 3 and all_forces.shape[1] >= 5:
            self._current_unfiltered_force_xyz[:] = all_forces[:, :5, :].to(
                device=self.device, dtype=torch.float32
            )
            self._current_unfiltered_force[:] = torch.linalg.vector_norm(self._current_unfiltered_force_xyz, dim=-1)
        else:
            self._current_unfiltered_force_xyz.zero_()
            self._current_unfiltered_force.zero_()
        explained_xyz = (
            self._current_target_force_xyz
            + self._current_table_force_xyz
            + self._current_ground_force_xyz
        )
        closure_residual = torch.linalg.vector_norm(
            self._current_unfiltered_force_xyz - explained_xyz,
            dim=-1,
        )
        all_nonzero = self._current_unfiltered_force > self.cfg.formal_contact_force_n
        unexplained_force = all_nonzero & (closure_residual > self.cfg.formal_contact_force_n)
        self._unresolved_contact |= torch.any(
            unexplained_force,
            dim=1,
        )
        self._contact_counts += (
            self._current_target_force > self.cfg.formal_contact_force_n
        ).to(dtype=torch.long)

    def _filtered_force_xyz(self, sensors: Sequence[ContactSensor]) -> tuple[torch.Tensor, bool]:
        force_xyz = torch.zeros_like(self._current_target_force_xyz)
        valid_count = 0
        for tip_index, sensor in enumerate(sensors):
            matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            if torch.is_tensor(matrix) and matrix.ndim == 4 and matrix.shape[1] >= 1 and matrix.shape[2] >= 1:
                force_xyz[:, tip_index, :] = matrix[:, 0, 0, :].to(device=self.device, dtype=torch.float32)
                valid_count += 1
        return force_xyz, bool(valid_count == 5)

    def _record_step(self) -> None:
        index = torch.clamp(self._trace_length, max=self._history_length - 1)
        object_local = self.target_object.data.root_pos_w - self.scene.env_origins
        palm_local = self.robot.data.body_pos_w[:, self._palm_body_id] - self.scene.env_origins
        support = (object_local[:, 2] - self.cfg.table_top_z_m) < 0.004
        jerk = self._ctrl_target - 2.0 * self._previous_ctrl_target + self.robot.data.joint_pos
        self._force_history[index, self._env_ids] = self._current_target_force
        self._object_history[index, self._env_ids] = object_local
        self._hand_history[index, self._env_ids] = palm_local
        self._support_history[index, self._env_ids] = support
        self._jerk_history[index, self._env_ids] = jerk
        self._previous_ctrl_target[:] = self._ctrl_target
        self._trace_length += 1

    def _get_rewards(self) -> torch.Tensor:
        active = self._active_finger_masks()
        contact = torch.any(
            active & (self._current_target_force > self.cfg.formal_contact_force_n), dim=1
        ).float()
        multi = torch.all(
            torch.logical_or(
                torch.logical_not(active),
                self._current_target_force > self.cfg.formal_contact_force_n,
            ),
            dim=1,
        ).float()
        object_local = self.target_object.data.root_pos_w - self.scene.env_origins
        object_motion = torch.linalg.vector_norm(object_local - self._object_reset_pos_local, dim=1)
        return (
            contact
            + 2.0 * multi
            + 5.0 * self._controlled_close_completed.to(dtype=torch.float32)
            + 100.0 * torch.clamp(self._lift_progress_m, 0.0, 0.02)
            - 3.0
            * torch.clamp(
                torch.max(self._current_target_force, dim=1).values - self.cfg.soft_force_limit_n,
                min=0.0,
            )
            - 4.0 * self._unresolved_contact.to(dtype=torch.float32)
            - 20.0 * object_motion
            - 0.5 * torch.linalg.vector_norm(self.target_object.data.root_vel_w, dim=1)
            - 2.0 * (
                self._first_contact_seen
                & ~torch.any(self._current_target_force > self.cfg.formal_contact_force_n, dim=1)
            ).to(dtype=torch.float32)
            - 0.05 * torch.linalg.vector_norm(self._ctrl_target - self._previous_ctrl_target, dim=1)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminal = (self._phase == int(ProgramPhase.COMPLETE)) | (self._phase == int(ProgramPhase.INVALID))
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self._set_termination(
            timeout & (self._termination == int(ProgramTermination.NONE)),
            ProgramTermination.EPISODE_TIMEOUT,
        )
        completed_ids = torch.where(terminal | timeout)[0]
        for env_id in completed_ids.tolist():
            candidate_id = int(self._candidate_ids[env_id].item())
            if candidate_id not in self._completed:
                self._finalize_evaluation(env_id, bool(timeout[env_id]))
        if self.cfg.defer_terminal_reset:
            return torch.zeros_like(terminal), torch.zeros_like(timeout)
        return terminal, timeout

    def _finalize_evaluation(self, env_id: int, timed_out: bool) -> None:
        length = int(min(self._trace_length[env_id].item(), self._history_length))
        if length <= 1:
            return
        active = self._active_finger_masks()[env_id].detach().cpu().tolist()
        close_start = int(self._close_start_step[env_id].item())
        lift_start = int(self._lift_start_step[env_id].item())
        evaluation = self._evaluator.evaluate(
            EvaluationInput(
                target_force_norms=self._force_history[:length, env_id].detach().cpu().numpy(),
                active_finger_mask=active,
                object_positions=self._object_history[:length, env_id].detach().cpu().numpy(),
                hand_positions=self._hand_history[:length, env_id].detach().cpu().numpy(),
                table_top_z_m=self.cfg.table_top_z_m,
                close_start_step=close_start if close_start >= 0 else max(0, length - 2),
                lift_start_step=lift_start if lift_start >= 0 else max(0, length - 1),
                controlled_close_completed=bool(self._controlled_close_completed[env_id]),
                unresolved_contact_truth=bool(self._unresolved_contact[env_id]),
                post_reset_object_writes=int(self._post_reset_object_writes[env_id].item()),
                post_reset_wrist_state_writes=int(self._post_reset_wrist_state_writes[env_id].item()),
                flyout=bool(self._flyout[env_id]),
                penetration=bool(self._penetration[env_id]),
                action_jerk=self._jerk_history[:length, env_id].detach().cpu().numpy(),
                object_supported_by_table=self._support_history[:length, env_id].detach().cpu().numpy(),
                metadata={
                    "candidate_id": int(self._candidate_ids[env_id].item()),
                    "env_id": env_id,
                    "timed_out": timed_out,
                    "phase": ProgramPhase(int(self._phase[env_id].item())).name,
                    "termination_reason": ProgramTermination(int(self._termination[env_id].item())).name,
                    "trace_length": length,
                    "reset_seed": int(self._reset_seeds[env_id].item()),
                    "mass_kg": float(self._object_mass_kg[env_id].item()),
                    "friction_multiplier": float(self._friction_multiplier[env_id].item()),
                    "target_force_sensor_valid": bool(self._target_force_valid[env_id]),
                    "reset_only_object_writes": 1,
                },
            )
        )
        candidate_id = int(self._candidate_ids[env_id].item())
        self._completed[candidate_id] = evaluation
        self._completed_metadata[candidate_id] = dict(evaluation.metadata)
        if evaluation.physical_lift_success:
            self._completed_traces[candidate_id] = {
                "target_force_norms": self._force_history[:length, env_id].detach().cpu().numpy().copy(),
                "object_positions": self._object_history[:length, env_id].detach().cpu().numpy().copy(),
                "hand_positions": self._hand_history[:length, env_id].detach().cpu().numpy().copy(),
                "table_support": self._support_history[:length, env_id].detach().cpu().numpy().copy(),
            }

    def consume_completed_evaluations(self) -> dict[int, PhysicalEvaluation]:
        completed = dict(self._completed)
        self._completed.clear()
        self._completed_metadata.clear()
        return completed

    def consume_completed_traces(self) -> dict[int, dict[str, np.ndarray]]:
        traces = dict(self._completed_traces)
        self._completed_traces.clear()
        return traces

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        super()._reset_idx(env_ids)
        count = env_ids.numel()
        random_rows = np.stack(
            [np.random.default_rng(int(self._reset_seeds[env_id].item())).uniform(size=8) for env_id in env_ids],
            axis=0,
        )
        random_values = torch.as_tensor(random_rows, dtype=torch.float32, device=self.device)
        canonical_pos = torch.tensor(self.cfg.canonical_object_pos, dtype=torch.float32, device=self.device)
        object_pos = canonical_pos.reshape(1, 3).repeat(count, 1)
        object_pos[:, :2] += (2.0 * random_values[:, :2] - 1.0) * self.cfg.object_xy_randomization_m
        clearance_min, clearance_max = self.cfg.object_z_clearance_range_m
        object_pos[:, 2] += clearance_min + (clearance_max - clearance_min) * random_values[:, 2]
        object_pos[:, 2] += (2.0 * random_values[:, 3] - 1.0) * self.cfg.object_z_noise_m
        rpy = (2.0 * random_values[:, 4:7] - 1.0) * self.cfg.object_rpy_randomization_rad
        random_quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        object_quat = quat_mul(self._canonical_object_quat.reshape(1, 4).repeat(count, 1), random_quat)
        root_pose = torch.cat((object_pos + self.scene.env_origins[env_ids], object_quat), dim=1)
        root_vel = torch.zeros((count, 6), dtype=torch.float32, device=self.device)
        self.target_object.write_root_pose_to_sim(root_pose, env_ids)
        self.target_object.write_root_velocity_to_sim(root_vel, env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        if self._external_control_enabled:
            joint_pos[:, self._wrist_ids] = self._external_reset_q26[env_ids, :6]
            joint_pos[:, self._hand_ids] = self._external_reset_q26[env_ids, 6:]
        else:
            joint_pos[:, self._wrist_ids] = self._parked_wrist_q
            joint_pos[:, self._hand_ids] = self._template_preshape_target[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._ctrl_target[env_ids] = joint_pos
        self._previous_ctrl_target[env_ids] = joint_pos

        mass_min, mass_max = self.cfg.object_mass_range_kg
        masses = self.target_object.root_physx_view.get_masses().clone()
        view_env_ids = env_ids.to(device=masses.device)
        sampled_masses = (
            mass_min + (mass_max - mass_min) * random_values[:, 7]
        ).reshape(-1, 1).to(device=masses.device)
        masses[view_env_ids, :] = sampled_masses
        self.target_object.root_physx_view.set_masses(masses, view_env_ids)
        materials = self.target_object.root_physx_view.get_material_properties().clone()
        friction_min, friction_max = self.cfg.friction_multiplier_range
        friction = friction_min + (friction_max - friction_min) * random_values[:, 0]
        material_env_ids = env_ids.to(device=materials.device)
        material_friction = friction.reshape(-1, 1).to(device=materials.device)
        materials[material_env_ids, :, 0] = material_friction
        materials[material_env_ids, :, 1] = material_friction
        materials[material_env_ids, :, 2] = 0.0
        self.target_object.root_physx_view.set_material_properties(materials, material_env_ids)

        self._object_reset_pos_local[env_ids] = object_pos
        self._object_reset_quat[env_ids] = object_quat
        self._object_mass_kg[env_ids] = masses[view_env_ids, 0].to(device=self.device)
        self._friction_multiplier[env_ids] = friction
        self._phase[env_ids] = int(ProgramPhase.COARSE_ACQUISITION)
        self._phase_step[env_ids] = 0
        self._closure_progress[env_ids] = 0.0
        self._first_contact_seen[env_ids] = False
        self._latched_contact[env_ids] = False
        self._contact_below_keep[env_ids] = 0
        self._latched_hand_target[env_ids] = self._template_preshape_target[env_ids]
        self._current_target_force[env_ids] = 0.0
        self._current_target_force_xyz[env_ids] = 0.0
        self._current_unfiltered_force_xyz[env_ids] = 0.0
        self._current_table_force_xyz[env_ids] = 0.0
        self._current_ground_force_xyz[env_ids] = 0.0
        self._current_unfiltered_force[env_ids] = 0.0
        self._target_force_valid[env_ids] = False
        self._table_force_valid[env_ids] = False
        self._ground_force_valid[env_ids] = False
        self._unresolved_contact[env_ids] = False
        self._hard_abort[env_ids] = False
        self._flyout[env_ids] = False
        self._penetration[env_ids] = False
        self._controlled_close_completed[env_ids] = False
        self._close_start_step[env_ids] = -1
        self._lift_start_step[env_ids] = -1
        self._lift_progress_m[env_ids] = 0.0
        self._weak_lift_pause[env_ids] = 0
        self._previous_action[env_ids] = 0.0
        self._post_reset_object_writes[env_ids] = 0
        self._post_reset_wrist_state_writes[env_ids] = 0
        self._contact_counts[env_ids] = 0
        self._trace_length[env_ids] = 0
        self._termination[env_ids] = int(ProgramTermination.NONE)
        self._force_history[:, env_ids] = 0.0
        self._object_history[:, env_ids] = 0.0
        self._hand_history[:, env_ids] = 0.0
        self._support_history[:, env_ids] = False
        self._jerk_history[:, env_ids] = 0.0
        for env_id in env_ids.detach().cpu().tolist():
            self._contact_pairs_by_env[int(env_id)].clear()
            self._forensic_contact_events_by_env[int(env_id)].clear()
        if self._prior is not None:
            proprio = self._coordex_proprio().detach().cpu().numpy()
            hand_q = self._ctrl_target[:, self._hand_ids].detach().cpu().numpy()
            self._prior.reset(proprio, hand_q)

    def _coordex_proprio(self) -> torch.Tensor:
        hand_q = self.robot.data.joint_pos[:, self._hand_ids]
        hand_qdot = self.robot.data.joint_vel[:, self._hand_ids]
        default_q = self.robot.data.default_joint_pos[:, self._hand_ids]
        default_qdot = self.robot.data.default_joint_vel[:, self._hand_ids]
        palm_quat = self.robot.data.body_quat_w[:, self._palm_body_id]
        inverse = quat_conjugate(palm_quat)
        palm_lin_body = quat_apply(inverse, self.robot.data.body_lin_vel_w[:, self._palm_body_id])
        palm_ang_body = quat_apply(inverse, self.robot.data.body_ang_vel_w[:, self._palm_body_id])
        previous_hand_action = (self._ctrl_target[:, self._hand_ids] - default_q) / 0.1
        return torch.cat(
            (
                palm_lin_body,
                palm_ang_body,
                hand_q - default_q,
                hand_qdot - default_qdot,
                previous_hand_action,
            ),
            dim=1,
        )

    def physics_fingerprint(self) -> dict[str, Any]:
        masses = self.target_object.root_physx_view.get_masses().detach().cpu().numpy()
        materials = self.target_object.root_physx_view.get_material_properties().detach().cpu().numpy()
        return {
            "scene": "WujiFloatingHand+Table+TargetObject+ground",
            "target_part_name": self.cfg.target_part_name,
            "num_envs": self.num_envs,
            "dt": self.cfg.sim.dt,
            "device": self.device,
            "table_top_z_m": self.cfg.table_top_z_m,
            "target_asset_usd": str(self.cfg.target_asset_usd),
            "target_asset_scale": list(self.cfg.target_asset_scale),
            "runtime_mass_kg_min": float(np.min(masses)),
            "runtime_mass_kg_max": float(np.max(masses)),
            "runtime_material_properties_first": materials[0].tolist() if materials.size else [],
            "randomize_physics_on_reset": bool(self.cfg.randomize_physics_on_reset),
            "table_usd": str(_TABLE_USD),
            "deterministic_stack_frozen": True,
            "near_grasp_search_allowed": True,
            "experimental_rl_allowed": True,
            "sticky_used": False,
            "snap_used": False,
            "teacher_motion_used": False,
            "post_reset_object_writes_allowed": False,
            "post_reset_wrist_state_writes_allowed": False,
            "replay_camera_enabled": bool(self.cfg.enable_replay_camera),
            "contact_attribution_enabled": bool(self.cfg.enable_contact_attribution),
            "contact_report_available": bool(self._contact_report_available),
            "contact_report_error": self._contact_report_error,
            "defer_terminal_reset": bool(self.cfg.defer_terminal_reset),
        }

    def rearm_replay_contact_guard_after_reset(self, env_id: int = 0) -> dict[str, Any]:
        """Discard only the stale reset-forward contact latch in replay mode.

        IsaacLab's public ``reset()`` calls ``sim.forward()`` and then reads the
        contact sensor without a physics/scene update.  The sensor can therefore
        still expose forces from the pre-reset articulation pose.  A persistent
        physical contact is sampled again after the first real physics step and
        remains subject to the unchanged force guard.
        """

        index = int(env_id)
        if not self.cfg.defer_terminal_reset:
            raise RuntimeError("contact-guard rearm is replay-only")
        if int(self.episode_length_buf[index].item()) != 0:
            raise RuntimeError("contact-guard rearm is only valid at reset step 0")
        target_peak = float(torch.max(self._current_target_force[index]).item())
        post_reset_writes = int(
            self._post_reset_object_writes[index].item()
            + self._post_reset_wrist_state_writes[index].item()
        )
        stale_latch = bool(self._unresolved_contact[index].item())
        eligible = bool(
            stale_latch
            and target_peak <= self.cfg.formal_contact_force_n
            and post_reset_writes == 0
        )
        if eligible:
            self._unresolved_contact[index] = False
        return {
            "step": 0,
            "rearmed": eligible,
            "reason": "STALE_RESET_FORWARD_CONTACT_CACHE" if eligible else "NOT_ELIGIBLE",
            "unresolved_latched_before": stale_latch,
            "unresolved_latched_after": bool(self._unresolved_contact[index].item()),
            "target_force_peak_n": target_peak,
            "unfiltered_force_peak_n": float(torch.max(self._current_unfiltered_force[index]).item()),
            "post_reset_state_writes": post_reset_writes,
            "control_target_modified": False,
            "physics_state_modified": False,
        }

    def replay_snapshot(self, env_id: int = 0) -> dict[str, Any]:
        """Return one synchronized diagnostic row for the single-env replay runner."""

        index = int(env_id)
        object_local = self.target_object.data.root_pos_w[index] - self.scene.env_origins[index]
        reset_pos = self._object_reset_pos_local[index]
        active = self._template_active_mask[index]
        return {
            "step": int(self.episode_length_buf[index].item()),
            "phase": ProgramPhase(int(self._phase[index].item())).name,
            "termination_reason": ProgramTermination(int(self._termination[index].item())).name,
            "candidate_id": int(self._candidate_ids[index].item()),
            "template_id": int(self._template_ids[index].item()),
            "target_force_n": self._current_target_force[index].detach().cpu().tolist(),
            "target_force_xyz": self._current_target_force_xyz[index].detach().cpu().tolist(),
            "all_force_xyz": self._current_unfiltered_force_xyz[index].detach().cpu().tolist(),
            "table_force_xyz": self._current_table_force_xyz[index].detach().cpu().tolist(),
            "ground_force_xyz": self._current_ground_force_xyz[index].detach().cpu().tolist(),
            "target_filter_valid": bool(self._target_force_valid[index].item()),
            "table_filter_valid": bool(self._table_force_valid[index].item()),
            "ground_filter_valid": bool(self._ground_force_valid[index].item()),
            "unresolved_contact_latched": bool(self._unresolved_contact[index].item()),
            "active_fingers": active.detach().cpu().tolist(),
            "object_pos_local": object_local.detach().cpu().tolist(),
            "object_delta_xyz": (object_local - reset_pos).detach().cpu().tolist(),
            "object_linear_velocity": self.target_object.data.root_lin_vel_w[index].detach().cpu().tolist(),
            "object_angular_velocity": self.target_object.data.root_ang_vel_w[index].detach().cpu().tolist(),
            "post_reset_object_writes": int(self._post_reset_object_writes[index].item()),
            "post_reset_wrist_state_writes": int(self._post_reset_wrist_state_writes[index].item()),
        }

    def replay_rgb_frame(self, env_id: int = 0) -> np.ndarray | None:
        if self.replay_camera is None:
            return None
        frame = self.replay_camera.data.output.get("rgb")
        if not torch.is_tensor(frame) or frame.ndim != 4 or frame.shape[0] <= int(env_id):
            return None
        return frame[int(env_id), :, :, :3].detach().cpu().numpy().astype(np.uint8, copy=True)


def _ordered_joint_ids(robot: Articulation, names: Sequence[str], device: str) -> torch.Tensor:
    ids, found = robot.find_joints(list(names), preserve_order=True)
    if list(found) != list(names):
        missing = sorted(set(names) - set(found))
        raise ValueError(f"Missing ordered Wuji joints: {missing}; found={found}")
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _contact_vec3(row: Any, field: str) -> tuple[float, float, float]:
    value = getattr(row, field, (0.0, 0.0, 0.0))
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return (0.0, 0.0, 0.0)


def _quat_wxyz_matrix(quat: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat((q[1], q[2], q[3], q[0])).as_matrix()


def _collision_enabled(prim: Any, root_path: str, usd_physics: Any) -> bool:
    current = prim
    while current and str(current.GetPath()).startswith(root_path):
        if current.HasAPI(usd_physics.MeshCollisionAPI):
            return True
        if current.HasAPI(usd_physics.CollisionAPI):
            enabled = usd_physics.CollisionAPI(current).GetCollisionEnabledAttr().Get()
            return enabled is not False
        current = current.GetParent()
    return False


def _ordered_body_ids(robot: Articulation, names: Sequence[str], device: str) -> torch.Tensor:
    indices = []
    for name in names:
        if name not in robot.body_names:
            raise ValueError(f"Missing Wuji fingertip body {name!r}")
        indices.append(robot.body_names.index(name))
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _joint_limits(robot: Articulation, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if not torch.is_tensor(limits):
        limits = robot.data.joint_pos_limits
    limits = limits[0].to(device=device, dtype=torch.float32)
    return limits[:, 0], limits[:, 1]


def _runtime_hand_references(
    default_q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    preshape = default_q.clone()
    metric = torch.zeros(20, dtype=torch.bool, device=default_q.device)
    metric[10:20] = True
    close_from_limits = lower + 0.75 * (upper - lower)
    preshape[~metric] = default_q[~metric] + 0.35 * (lower[~metric] - default_q[~metric])
    preshape[metric] = default_q[metric] + 0.35 * (close_from_limits[metric] - default_q[metric])
    close = preshape.clone()
    close[metric] = close_from_limits[metric]
    return torch.clamp(preshape, lower, upper), torch.clamp(close, lower, upper)


def _finger_joint_local_indices(finger_zero_based: int, device: str) -> torch.Tensor:
    # Runtime order is joint-major: joint1 finger1..5, then joint2, etc.
    return torch.as_tensor(
        [joint * 5 + int(finger_zero_based) for joint in range(4)], dtype=torch.long, device=device
    )


def _serial_xyz_euler_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r12 = 2.0 * (y * z - x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    pitch = torch.asin(torch.clamp(r02, -1.0 + 1.0e-6, 1.0 - 1.0e-6))
    roll = torch.atan2(-r12, r22)
    yaw = torch.atan2(-r01, r00)
    return torch.stack((roll, pitch, yaw), dim=-1)
