"""Independent single-environment physics scene for Frame Stage 4 delivery."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.wuji_hand import (
    WUJI_FLOATING_HAND_CFG,
    WUJI_FLOATING_WRIST_JOINT_NAMES,
    WUJI_HAND_FINGER_JOINT_NAMES,
)

from chair_tasks_cfg import ChairAssembly4
from ..pipeline.unified_grasp.near_grasp.grasp_synthesis.forensic_trace import ForensicContactEvent
from .contracts import ResetWriteGate


_NP_DIR = Path(__file__).resolve().parents[2]
_TABLE_USD = _NP_DIR / "asset" / "workdesk.usd"
_TASK = ChairAssembly4()
_SUPPORT_LINKS = tuple(
    f"right_finger{finger}_link{level}" for finger in (2, 3, 4) for level in (3, 4)
)


@configclass
class FrameStage4EnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 20.0
    action_space = 26
    observation_space = 1
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    robot = WUJI_FLOATING_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    fixed_asset = copy.deepcopy(_TASK.fixed_asset)
    frame = copy.deepcopy(_TASK.frame)
    replay_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/ReplayCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=31.0,
            focus_distance=0.65,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 5.0),
        ),
        width=960,
        height=540,
        update_period=0.0,
    )
    camera_eye_local = (0.56, 0.30, 1.30)
    camera_target_local = (0.05, -0.25, 0.87)


class FrameStage4Env(DirectRLEnv):
    cfg: FrameStage4EnvCfg

    def __init__(self, cfg: FrameStage4EnvCfg, render_mode: str | None = None, **kwargs):
        self._configured_reset_q26: np.ndarray | None = None
        self._write_gate = ResetWriteGate()
        self._post_reset_fixed_asset_root_writes = 0
        self._reset_fixed_asset_root_writes = 0
        self._contact_report_subscription = None
        self._contact_report_error = ""
        self._contact_events: list[list[ForensicContactEvent]] = [[]]
        super().__init__(cfg, render_mode, **kwargs)
        self._wrist_ids = _ordered_joint_ids(self.robot, WUJI_FLOATING_WRIST_JOINT_NAMES, self.device)
        self._hand_ids = _ordered_joint_ids(self.robot, WUJI_HAND_FINGER_JOINT_NAMES, self.device)
        self._ordered_ids = torch.cat((self._wrist_ids, self._hand_ids))
        self._lower = self.robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self._upper = self.robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self._palm_body_id = _body_id(self.robot, "right_palm_link")
        self._support_body_ids = torch.as_tensor(
            [_body_id(self.robot, name) for name in _SUPPORT_LINKS], dtype=torch.long, device=self.device
        )
        self.hand_preshape_q, self.hand_close_q = _runtime_hand_references(
            self.robot.data.default_joint_pos[0, self._hand_ids],
            self._lower[self._hand_ids],
            self._upper[self._hand_ids],
        )
        self._ctrl_target = self.robot.data.default_joint_pos.clone()
        self._contact_report_available = self._subscribe_contact_reports()
        self._configure_camera()

    def configure_reset(self, q26: Sequence[float]) -> None:
        values = np.asarray(q26, dtype=np.float64)
        if values.shape != (26,) or not np.all(np.isfinite(values)):
            raise ValueError("Stage4 reset q must be finite 26D")
        lower = self._lower[self._ordered_ids].detach().cpu().numpy()
        upper = self._upper[self._ordered_ids].detach().cpu().numpy()
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("Stage4 reset q violates runtime joint limits")
        self._configured_reset_q26 = values.copy()

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
        self.fixed_asset = Articulation(self.cfg.fixed_asset)
        self.frame = RigidObject(self.cfg.frame)
        self.support_contact_sensors: list[ContactSensor] = []
        for index, body_name in enumerate(_SUPPORT_LINKS):
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Frame"],
                update_period=0.0,
                history_length=3,
                track_air_time=True,
                force_threshold=0.0,
            )
            sensor = ContactSensor(sensor_cfg)
            self.support_contact_sensors.append(sensor)
            self.scene.sensors[f"frame_support_{index}"] = sensor
        self.frame_fixed_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Frame",
                filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/.*"],
                update_period=0.0,
                history_length=3,
                track_air_time=True,
                force_threshold=0.0,
            )
        )
        self.frame_table_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Frame",
                filter_prim_paths_expr=["/World/envs/env_.*/Table"],
                update_period=0.0,
                history_length=3,
                track_air_time=True,
                force_threshold=0.0,
            )
        )
        self.scene.sensors["frame_fixed_contact"] = self.frame_fixed_contact_sensor
        self.scene.sensors["frame_table_contact"] = self.frame_table_contact_sensor
        self.replay_camera = TiledCamera(self.cfg.replay_camera)
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["fixed_asset"] = self.fixed_asset
        self.scene.rigid_objects["frame"] = self.frame
        self.scene.sensors["replay_camera"] = self.replay_camera
        self.scene.clone_environments(copy_from_source=False)
        self._enable_contact_reports()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _enable_contact_reports(self) -> None:
        try:
            import omni.usd
            from pxr import PhysxSchema, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not any(marker in path for marker in ("/Robot/", "/Frame", "/FixedAsset", "/Table", "/ground")):
                    continue
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report.CreateThresholdAttr().Set(0.0)
        except Exception:
            return

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
                    if match is None or int(match.group(1)) != 0:
                        continue
                    offset = int(getattr(header, "contact_data_offset", 0))
                    count = int(getattr(header, "num_contact_data", 0))
                    rows = list(contact_data[offset : offset + count]) if count > 0 else []
                    self._contact_events[0].append(
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
            self._contact_report_error = f"{type(exc).__name__}:{exc}"
            return False

    def consume_contact_events(self) -> tuple[ForensicContactEvent, ...]:
        events = tuple(self._contact_events[0])
        self._contact_events[0].clear()
        return events

    def clear_reset_contact_cache(self) -> dict[str, Any]:
        if int(self.episode_length_buf[0].item()) != 0:
            raise RuntimeError("contact cache may only be cleared before the first physics step")
        count = len(self._contact_events[0])
        self._contact_events[0].clear()
        return {"cleared_before_first_physics_step": True, "event_count": count, "physics_state_modified": False}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        target = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if target.shape != (1, 26):
            raise ValueError("FrameStage4 action must have shape [1,26]")
        bounded = torch.maximum(
            torch.minimum(target, self._upper[self._ordered_ids].reshape(1, 26)),
            self._lower[self._ordered_ids].reshape(1, 26),
        )
        self._ctrl_target[:, self._ordered_ids] = bounded

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._ctrl_target)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": torch.zeros((1, 1), dtype=torch.float32, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(1, dtype=torch.bool, device=self.device)
        return zeros, zeros

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        super()._reset_idx(ids)
        self._write_gate = ResetWriteGate()
        self._post_reset_fixed_asset_root_writes = 0
        self._reset_fixed_asset_root_writes = 0

        frame_state = self.frame.data.default_root_state[ids].clone()
        frame_state[:, :3] += self.scene.env_origins[ids]
        self._write_gate.record("frame.write_root_pose_to_sim")
        self.frame.write_root_pose_to_sim(frame_state[:, :7], ids)
        self.frame.write_root_velocity_to_sim(frame_state[:, 7:], ids)

        fixed_state = self.fixed_asset.data.default_root_state[ids].clone()
        fixed_state[:, :3] += self.scene.env_origins[ids]
        self._reset_fixed_asset_root_writes += 1
        self.fixed_asset.write_root_pose_to_sim(fixed_state[:, :7], ids)
        self.fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], ids)
        if self.fixed_asset.num_joints:
            self.fixed_asset.write_joint_state_to_sim(
                self.fixed_asset.data.default_joint_pos[ids],
                self.fixed_asset.data.default_joint_vel[ids],
                env_ids=ids,
            )

        joint_pos = self.robot.data.default_joint_pos[ids].clone()
        if self._configured_reset_q26 is not None:
            reset = torch.as_tensor(self._configured_reset_q26, dtype=torch.float32, device=self.device)
            joint_pos[:, self._ordered_ids] = reset.reshape(1, 26)
        joint_vel = torch.zeros_like(joint_pos)
        self._write_gate.record("robot.write_joint_state_to_sim", wrist_state=True)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=ids)
        self._ctrl_target[ids] = joint_pos
        self._contact_events[0].clear()
        self._write_gate.lock()

    def snapshot(self) -> dict[str, Any]:
        frame_pos = self.frame.data.root_pos_w[0] - self.scene.env_origins[0]
        fixed_pos = self.fixed_asset.data.root_pos_w[0] - self.scene.env_origins[0]
        palm_pos = self.robot.data.body_pos_w[0, self._palm_body_id] - self.scene.env_origins[0]
        frame_quat = self.frame.data.root_quat_w[0]
        fixed_quat = self.fixed_asset.data.root_quat_w[0]
        palm_quat = self.robot.data.body_quat_w[0, self._palm_body_id]
        force_vectors = []
        force_valid = []
        for sensor in self.support_contact_sensors:
            matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            valid = bool(torch.is_tensor(matrix) and matrix.ndim == 4 and matrix.shape[1] and matrix.shape[2])
            vector = matrix[0, 0, 0, :] if valid else torch.zeros(3, device=self.device)
            force_vectors.append(vector.detach().cpu().numpy().copy())
            force_valid.append(valid)
        body_pos = self.robot.data.body_pos_w[0, self._support_body_ids] - self.scene.env_origins[0]
        body_quat = self.robot.data.body_quat_w[0, self._support_body_ids]
        audit = self.root_write_audit()
        frame_fixed_force, frame_fixed_valid = _single_filtered_force(self.frame_fixed_contact_sensor, self.device)
        frame_table_force, frame_table_valid = _single_filtered_force(self.frame_table_contact_sensor, self.device)
        return {
            "physics_frame_id": int(self.episode_length_buf[0].item()) - 1,
            "joint_pos26": self.robot.data.joint_pos[0, self._ordered_ids].detach().cpu().numpy().copy(),
            "joint_vel26": self.robot.data.joint_vel[0, self._ordered_ids].detach().cpu().numpy().copy(),
            "joint_target26": self._ctrl_target[0, self._ordered_ids].detach().cpu().numpy().copy(),
            "frame_position": frame_pos.detach().cpu().numpy().copy(),
            "frame_quat_wxyz": frame_quat.detach().cpu().numpy().copy(),
            "frame_linear_velocity": self.frame.data.root_lin_vel_w[0].detach().cpu().numpy().copy(),
            "frame_angular_velocity": self.frame.data.root_ang_vel_w[0].detach().cpu().numpy().copy(),
            "fixed_position": fixed_pos.detach().cpu().numpy().copy(),
            "fixed_quat_wxyz": fixed_quat.detach().cpu().numpy().copy(),
            "palm_position": palm_pos.detach().cpu().numpy().copy(),
            "palm_quat_wxyz": palm_quat.detach().cpu().numpy().copy(),
            "support_link_names": list(_SUPPORT_LINKS),
            "support_link_positions": body_pos.detach().cpu().numpy().copy(),
            "support_link_quat_wxyz": body_quat.detach().cpu().numpy().copy(),
            "support_force_xyz": np.asarray(force_vectors, dtype=np.float64),
            "support_force_n": np.linalg.norm(np.asarray(force_vectors, dtype=np.float64), axis=1),
            "support_force_filter_valid": bool(all(force_valid)),
            "frame_fixed_force_xyz": frame_fixed_force,
            "frame_fixed_force_n": float(np.linalg.norm(frame_fixed_force)),
            "frame_fixed_force_filter_valid": frame_fixed_valid,
            "frame_table_force_xyz": frame_table_force,
            "frame_table_force_n": float(np.linalg.norm(frame_table_force)),
            "frame_table_force_filter_valid": frame_table_valid,
            "contact_report_available": self._contact_report_available,
            "contact_report_error": self._contact_report_error,
            "frame_transform": _transform(frame_pos, frame_quat),
            "fixed_transform": _transform(fixed_pos, fixed_quat),
            "palm_transform": _transform(palm_pos, palm_quat),
            **audit,
        }

    def root_write_audit(self) -> dict[str, int | bool | list[str]]:
        audit = self._write_gate.audit()
        return {
            "reset_object_root_writes": audit.reset_object_root_writes,
            "reset_wrist_state_writes": audit.reset_wrist_state_writes,
            "reset_fixed_asset_root_writes": self._reset_fixed_asset_root_writes,
            "post_reset_object_root_writes": audit.post_reset_object_root_writes,
            "post_reset_wrist_state_writes": audit.post_reset_wrist_state_writes,
            "post_reset_fixed_asset_root_writes": self._post_reset_fixed_asset_root_writes,
            "locked": audit.locked,
            "rejected_operations": list(audit.rejected_operations),
        }

    def rgb_frame(self) -> np.ndarray | None:
        frame = self.replay_camera.data.output.get("rgb")
        if not torch.is_tensor(frame) or frame.ndim != 4 or not frame.shape[0]:
            return None
        return frame[0, :, :, :3].detach().cpu().numpy().astype(np.uint8, copy=True)

    def _configure_camera(self) -> None:
        eye = torch.tensor(self.cfg.camera_eye_local, dtype=torch.float32, device=self.device).reshape(1, 3)
        target = torch.tensor(self.cfg.camera_target_local, dtype=torch.float32, device=self.device).reshape(1, 3)
        self.replay_camera.set_world_poses_from_view(self.scene.env_origins + eye, self.scene.env_origins + target)

    def runtime_fingerprint(self) -> dict[str, Any]:
        frame_mass = self.frame.root_physx_view.get_masses().detach().cpu().numpy()
        frame_material = self.frame.root_physx_view.get_material_properties().detach().cpu().numpy()
        fixed_material = self.fixed_asset.root_physx_view.get_material_properties().detach().cpu().numpy()
        return {
            "frame_asset": str(self.cfg.frame.spawn.usd_path),
            "fixed_asset": str(self.cfg.fixed_asset.spawn.usd_path),
            "frame_mass_kg": float(frame_mass.reshape(-1)[0]),
            "frame_material": frame_material.tolist(),
            "fixed_material": fixed_material.tolist(),
            "dt": float(self.cfg.sim.dt),
            "contact_report_available": self._contact_report_available,
            "robot_self_collision_enabled": False,
        }


def _ordered_joint_ids(robot: Articulation, names: Sequence[str], device: str) -> torch.Tensor:
    ids, found = robot.find_joints(list(names), preserve_order=True)
    if list(found) != list(names):
        raise ValueError(f"missing Wuji joints: {sorted(set(names) - set(found))}")
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _runtime_hand_references(
    default_q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    preshape = default_q.clone()
    distal = torch.zeros(20, dtype=torch.bool, device=default_q.device)
    distal[10:20] = True
    close_from_limits = lower + 0.75 * (upper - lower)
    preshape[~distal] = default_q[~distal] + 0.35 * (lower[~distal] - default_q[~distal])
    preshape[distal] = default_q[distal] + 0.35 * (close_from_limits[distal] - default_q[distal])
    close = preshape.clone()
    close[distal] = close_from_limits[distal]
    return torch.clamp(preshape, lower, upper), torch.clamp(close, lower, upper)


def _body_id(robot: Articulation, name: str) -> int:
    if name not in robot.body_names:
        raise ValueError(f"missing Wuji body {name}")
    return int(robot.body_names.index(name))


def _contact_vec3(row: Any, field: str) -> tuple[float, float, float]:
    value = getattr(row, field, (0.0, 0.0, 0.0))
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except Exception:
        return 0.0, 0.0, 0.0


def _single_filtered_force(sensor: ContactSensor, device: str) -> tuple[np.ndarray, bool]:
    matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
    valid = bool(torch.is_tensor(matrix) and matrix.ndim == 4 and matrix.shape[1] and matrix.shape[2])
    if not valid:
        return np.zeros(3, dtype=np.float64), False
    return matrix[0, 0, 0, :].to(device=device, dtype=torch.float32).detach().cpu().numpy().copy(), True


def _transform(position: torch.Tensor, quat_wxyz: torch.Tensor) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pos = position.detach().cpu().numpy().astype(np.float64)
    quat = quat_wxyz.detach().cpu().numpy().astype(np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
    result[:3, 3] = pos
    return result
