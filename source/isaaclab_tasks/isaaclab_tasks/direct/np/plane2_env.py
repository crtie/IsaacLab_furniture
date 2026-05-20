# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import sys, os
sys.path.append(os.path.abspath(__file__))

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .np_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FrankaPlane2Cfg
from pdb import set_trace as bp
from .np_utils.group_utils import SE3dist
from .np_utils.viz_utils import define_markers
from scipy.spatial.transform import Rotation as R
import torch
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf, Gf, Tf
from omni.physx.scripts import utils
import omni.usd

class FrankaPlane2Env(DirectRLEnv):
    cfg: FrankaPlane2Cfg

    def __init__(self, cfg: FrankaPlane2Cfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task
        print(f"Using task: {self.cfg_task.name}")

        super().__init__(cfg, render_mode, **kwargs)

        self.joint_created = False
        self.fixed_joint_prim = None  # Will be set when the joint is created.

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)


    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        held_base_z_offset = 0.0

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()



        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)

        if self.cfg_task.name == "plane_assembly":
            self.fixed_success_pos_local[:, 2] = 0.0
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Scripted-policy state for plane2 idx=1 (tail_half2 ↦ tail_half_wider).
        # STRICTLY independent — all tensors carry `_p2_1` suffix so no
        # other idx can read/write these.
        self._xy_align_frames_p2_1     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p2_1  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p2_1   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p2_1  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Stall-detection + wobble state (phase B only).
        self._prev_along_p2_1          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p2_1        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p2_1         = 0   # global scalar — wobble phase accumulator

        # Scripted-policy state for plane2 idx=2 (body rod ↦ tail_half_wider).
        # STRICTLY independent of idx=1 — all tensors carry `_p2_2` suffix.
        self._xy_align_frames_p2_2     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p2_2  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p2_2   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p2_2  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_along_p2_2          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p2_2        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p2_2         = 0

        # Scripted-policy state for plane2 idx=3 (propeller hub ↦ tail rod).
        # STRICTLY independent of idx=1/2 — all tensors carry `_p2_3` suffix.
        self._xy_align_frames_p2_3     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p2_3  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p2_3   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p2_3  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_along_p2_3          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p2_3        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p2_3         = 0

        # Scripted-policy state for plane2 idx=4 (holder peg ↦ propeller hub hole).
        # STRICTLY independent of idx=1/2/3 — all tensors carry `_p2_4` suffix.
        self._xy_align_frames_p2_4     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p2_4  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p2_4   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p2_4  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_along_p2_4          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p2_4        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p2_4         = 0

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.0))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/workdesk.usd")
        cfg.scale = np.array([1.0, 0.7, 1.0])
        cfg.mass_props = sim_utils.MassPropertiesCfg(mass=1e7),
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0., 0.0, 0.0), orientation=(1, 0.0, 0.0, 0.0))
        print(f"robot cfg: {self.cfg.robot}")
        self._robot = Articulation(self.cfg.robot)
        if self.cfg_task.task_idx in [3, 4]:
            self.cfg_task.fixed_asset.init_state.pos += np.array([0.0, 0.0, 0.025])
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)

        if self.cfg_task.task_idx == 1:
            self._tailhalf = RigidObject(self.cfg_task.tailhalf)
            self._held_asset = self._tailhalf
            self._connection_cfg = self.cfg_task.connection_cfg1

        if self.cfg_task.task_idx ==2:
            self._tailhalf = RigidObject(self.cfg_task.tailhalf)
            self._body = RigidObject(self.cfg_task.body)
            self._held_asset = self._body
            self._connection_cfg = self.cfg_task.connection_cfg2

        if self.cfg_task.task_idx ==3:
            self._tailhalf = RigidObject(self.cfg_task.tailhalf)
            self._body = RigidObject(self.cfg_task.body)
            self._propeller = RigidObject(self.cfg_task.propeller)
            self._held_asset = self._propeller
            self._connection_cfg = self.cfg_task.connection_cfg3

        if self.cfg_task.task_idx ==4:
            self._tailhalf = RigidObject(self.cfg_task.tailhalf)
            self._body = RigidObject(self.cfg_task.body)
            self._propeller = RigidObject(self.cfg_task.propeller)
            self._holder = RigidObject(self.cfg_task.holder)
            self._held_asset = self._holder
            self._connection_cfg = self.cfg_task.connection_cfg1
        

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()
        self.marker_locations = torch.zeros((self.num_envs, 3)).cuda()
        self.marker_orientations = torch.zeros((self.num_envs, 4)).cuda()

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w
        self.fixed_linvel = self._fixed_asset.data.root_lin_vel_w
        self.fixed_angvel = self._fixed_asset.data.root_ang_vel_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w


        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp



    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0


    def _get_real_mat(self):
        from pxr import Gf


        held_pos = self.held_pos[0].cpu().numpy()
        held_quat = self.held_quat[0].cpu().numpy()
        fixed_pos = self.fixed_pos[0].cpu().numpy()
        fixed_quat = self.fixed_quat[0].cpu().numpy()

        held_mat = Gf.Matrix4d()
        held_mat.SetRotate(Gf.Rotation(Gf.Quatd(float(held_quat[0]), float(held_quat[1]), float(held_quat[2]), float(held_quat[3]))))
        held_mat.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in held_pos]))

        fixed_mat = Gf.Matrix4d()
        fixed_mat.SetRotate(Gf.Rotation(Gf.Quatd(float(fixed_quat[0]), float(fixed_quat[1]), float(fixed_quat[2]), float(fixed_quat[3]))))
        fixed_mat.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in fixed_pos]))

        from_pose = fixed_mat
        to_pose = held_mat
        relative_mat = to_pose * from_pose.GetInverse()
        rot = np.array(relative_mat.ExtractRotationMatrix())
        pos = np.array(relative_mat.ExtractTranslation())
        rel_mat_np = np.eye(4, dtype=np.float32)
        rel_mat_np[:3, :3] = rot
        rel_mat_np[:3, 3] = pos
        return rel_mat_np
    

    def _create_fixed_joint(self, connection_idx):
        """Create a fixed joint between the held asset and the fixed asset."""
        from pxr import Usd, UsdPhysics, PhysxSchema, Sdf, Gf
        from omni.physx.scripts import utils
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if connection_idx == 1:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/TailHalf")
            joint_path = "/World/envs/env_0/FixedJoint1"
            connection_cfg = self.cfg_task.connection_cfg1
        elif connection_idx == 2:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/MainBody")
            joint_path = "/World/envs/env_0/FixedJoint2"
            connection_cfg = self.cfg_task.connection_cfg2
        elif connection_idx == 3:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Propeller")
            joint_path = "/World/envs/env_0/FixedJoint3"
            connection_cfg = self.cfg_task.connection_cfg3

        fixed_prim = stage.GetPrimAtPath("/World/envs/env_0/FixedAsset")
        

        to_path = held_prim.GetPath()
        from_path = fixed_prim.GetPath()
        # rel_mat1 = self._get_real_mat()
        if not connection_cfg.connection_type == "euler":
            rel_mat = connection_cfg.pose_to_base
            pos1 = Gf.Vec3f([float(rel_mat[0, 3]), float(rel_mat[1, 3]), float(rel_mat[2, 3])])
            rot1q = torch_utils.rot_matrices_to_quats(torch.tensor(rel_mat[:3, :3]))
            rot1 = Gf.Quatf(float(rot1q[0]), float(rot1q[1]), float(rot1q[2]), float(rot1q[3]))
        else:
            rel_mat = connection_cfg.pose_to_base
            pos1 = Gf.Vec3f([float(rel_mat[0, 3]), float(rel_mat[1, 3]), float(rel_mat[2, 3])])
            rot_euler = torch.tensor(connection_cfg.pose_to_base_euler, device=self.device).unsqueeze(0)
            rot1q = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            ).squeeze()
            rot1 = Gf.Quatf(float(rot1q[0]), float(rot1q[1]), float(rot1q[2]), float(rot1q[3]))  
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])


        joint.CreateLocalPos0Attr().Set(pos1)
        joint.CreateLocalRot0Attr().Set(rot1)
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


        self.fixed_joint_prim = stage.GetPrimAtPath(joint_path)
        self.step_sim_no_action()


    def _check_attach_condition(self):
        rel_mat = self._get_real_mat()
        gt_real_mat = self._connection_cfg.pose_to_base

        R_dist, R_axis, t_tangent, t_normal = SE3dist(rel_mat, gt_real_mat, self._connection_cfg)
        print("rel_mat:", rel_mat)
        # print("gt_real_mat:", gt_real_mat)
        # bp()
        print("R_dist:", R_dist)
        print("t_tangent:", t_tangent)
        print("t_normal:", t_normal)
        # print("joint names of frame:",self._fixed_asset.joint_names)
        if not self.joint_created and R_dist < 0.1 and t_tangent < 0.005 and t_normal < 0.008:
            # self._create_fixed_joint(connection_idx=self.cfg_task.task_idx)
            # self._create_screw_joint(connection_idx=self.cfg_task.task_idx)
            self.joint_created = True
            rel_mat = self._get_real_mat()
            gt_real_mat = self._connection_cfg.pose_to_base
            R_dist, R_axis, t_tangent, t_normal = SE3dist(rel_mat, gt_real_mat, self._connection_cfg)
            self.R_axis = R_axis 
            print("Creating fixed joint.")


        elif self.joint_created :
            print("Joint already created.")
        else:
            print("Not creating fixed joint yet, waiting for conditions to be met.")

    def _visualize_markers(self):
        # offset markers so they are above the jetbot
        fixed_pos = self.fixed_pos[0].cpu().numpy()
        fixed_quat = self.fixed_quat[0].cpu().numpy()
        self.marker_locations[0] = torch.tensor(fixed_pos, device=self.device)
        self.marker_orientations[0] = torch.tensor(fixed_quat, device=self.device)
        loc = self.marker_locations
        rots = self.marker_orientations

        # render the markers
        all_envs = torch.arange(self.num_envs)
        indices = torch.zeros_like(all_envs)
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _visualize_centerlines_p2_1(self):
        """For PlaneAssembly2 idx=1: draw TWO key points.

          GREEN point = held (upper) part's OPENING midpoint
                        held_local (0, 0, -0.035) — center of the held's
                        −z face (the bottom of held in its own local
                        frame, where the slot/opening is on tail_half2).

          RED   point = fixed (lower) part's "lower-midpoint" anchor
                        fixed_local (0, 0, +0.035) — center of the
                        fixed's +z face (the protruding "lower-part"
                        end of tail_half_wider that the held's slot
                        drops onto).

        Mate target (per cfg.connection_cfg1): held_local (0, 0, -0.035)
        maps to fixed_local (0, 0, +0.029) — 6 mm past the fixed top
        face, i.e. the held drops by 6 mm onto the fixed protrusion.
        """
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p2_legend_printed", False):
            print("\n[plane2 idx=1] viewer legend (two anchor points):")
            print("  GREEN dot = held OPENING midpoint  held_local (0, 0, -0.035)")
            print("  RED   dot = fixed LOWER midpoint   fixed_local (0, 0, +0.035)")
            self._p2_legend_printed = True

        # ---- Held opening midpoint in world ----
        held_opening_local_p2_1 = torch.tensor(
            [0.0, 0.0, -0.035], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, held_opening_world_p2_1 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, held_opening_local_p2_1
        )

        # ---- Fixed lower midpoint in world ----
        fixed_lower_local_p2_1 = torch.tensor(
            [0.0, 0.0, +0.035], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, fixed_lower_world_p2_1 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_lower_local_p2_1
        )

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p2_1(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        points = [
            _to_tuple_p2_1(held_opening_world_p2_1[0]),
            _to_tuple_p2_1(fixed_lower_world_p2_1[0]),
        ]
        colors = [
            (0.2, 1.0, 0.2, 1.0),   # GREEN — held opening midpoint
            (1.0, 0.2, 0.2, 1.0),   # RED   — fixed lower midpoint
        ]
        sizes = [14.0, 14.0]

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)

    def _visualize_body_and_fixed_p2_2(self):
        """For PlaneAssembly2 idx=2: draw the body's CROSS-NOTCH and
        the fixed's TWO ELLIPSES (their long axes form a CROSS on the
        fixed). Body cross must align with the fixed cross before press.

        Geometry (offline-derived, body scale = 1.2):
          body -x cluster (398 verts) is the cross-shaped notch.
          4 arm tips ~20 mm out (unscaled) — arms ≈ along body +y & +z.

          cfg_R for cfg2: body +y → fixed +y,  body +z → fixed +x
          → cross arm-along-body-+z drops into ellipse-1 (long axis fixed +x)
          → cross arm-along-body-+y drops into ellipse-2 (long axis fixed +y)

          Both ellipses are centered at the mate point in fixed_local.

        Renders:
          GREEN dot   = body cross-notch center
          GREEN line  = body cross arm along held +y
          GREEN line  = body cross arm along held +z
          RED   dot   = fixed mate point (common center of both ellipses)
          RED   line  = ellipse-1 long axis along fixed +x
          RED   line  = ellipse-2 long axis along fixed +y
        """
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p2_2_legend_printed", False):
            print("\n[plane2 idx=2] viewer legend (body cross + fixed cross-of-2-ellipses):")
            print("  GREEN dot  = body cross-notch DEPTH MID  held_local (-0.094, 0, 0.0045) × 1.2")
            print("  GREEN line = cross arm along held +y    ~24 mm half-length each side")
            print("  GREEN line = cross arm along held +z    ~24 mm half-length each side")
            print("  RED   dot  = fixed mate point (CENTER)  fixed_local (0.0054, 0, 0) — at GEOMETRIC center, inside body")
            print("  RED   line = ellipse-1 long axis (fixed +x)  ~24 mm half-length each side")
            print("  RED   line = ellipse-2 long axis (fixed +y)  ~24 mm half-length each side")
            self._p2_2_legend_printed = True

        body_scale_p2_2 = 1.2
        # Cross arm unscaled half-length from mesh PCA = 20 mm. Apply body
        # scale so the drawn line matches the actual scaled cross arm.
        cross_arm_half_len_p2_2 = 0.020 * body_scale_p2_2          # 24 mm in held_local
        # Each ellipse long axis = full cross-arm span when mated.
        ellipse_long_half_len_p2_2 = cross_arm_half_len_p2_2        # 24 mm in fixed_local

        # ---- Cross-notch center in world (held) ----
        # x = -0.094 is the DEPTH MIDPOINT of the cross-notch (not the
        # outer face at -0.106). Histogram of body.obj verts in the
        # cross region:
        #   front-face mass at x ≈ -0.1065 (notch opening / surface)
        #   back-wall mass at x ≈ -0.0820 (notch bottom)
        #   → mid-depth ≈ -0.094
        notch_local_unscaled = [-0.094, 0.0, 0.0045]
        notch_local_p2_2 = torch.tensor(
            [c * body_scale_p2_2 for c in notch_local_unscaled],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, body_notch_world_p2_2 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, notch_local_p2_2
        )

        # Cross arm endpoints in held_local (arms along body +y and +z).
        arm_y_p_local = (notch_local_p2_2[0] + torch.tensor(
            [0.0, +cross_arm_half_len_p2_2, 0.0],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        arm_y_n_local = (notch_local_p2_2[0] + torch.tensor(
            [0.0, -cross_arm_half_len_p2_2, 0.0],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        arm_z_p_local = (notch_local_p2_2[0] + torch.tensor(
            [0.0, 0.0, +cross_arm_half_len_p2_2],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        arm_z_n_local = (notch_local_p2_2[0] + torch.tensor(
            [0.0, 0.0, -cross_arm_half_len_p2_2],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        _, arm_y_p_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, arm_y_p_local
        )
        _, arm_y_n_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, arm_y_n_local
        )
        _, arm_z_p_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, arm_z_p_local
        )
        _, arm_z_n_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, arm_z_n_local
        )

        # ---- Mate point + ellipse long-axis endpoints in world (fixed) ----
        # Force BOTH y=0 AND z=0 in fixed_local so the cross is exactly
        # at the geometric center of the fixed part — not near any surface.
        # (Earlier cfg-derived z = -0.0172 put the line close to fixed's
        # top z-face in world; z=0 puts it at world z=0.83 = exact mid.)
        mate_local_p2_2 = torch.tensor(
            [0.0045 * body_scale_p2_2,
             0.0,
             0.0],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, fixed_mate_world_p2_2 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, mate_local_p2_2
        )
        # User-tuned world-frame offset of the upper target: +5mm to the
        # right (world +y). Applied AFTER tf_combine so it's axis-aligned
        # in world regardless of fixed_quat. Keep in sync with policy.
        world_target_offset_p2_2 = torch.tensor(
            [0.0, 0.005, 0.0], dtype=torch.float32, device=self.device,
        )
        fixed_mate_world_p2_2 = fixed_mate_world_p2_2 + world_target_offset_p2_2
        # Ellipse-1 (long axis along fixed +x).
        ellipse1_p_local = (mate_local_p2_2[0] + torch.tensor(
            [+ellipse_long_half_len_p2_2, 0.0, 0.0],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        ellipse1_n_local = (mate_local_p2_2[0] + torch.tensor(
            [-ellipse_long_half_len_p2_2, 0.0, 0.0],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        _, ellipse1_p_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, ellipse1_p_local
        )
        _, ellipse1_n_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, ellipse1_n_local
        )
        # Ellipse-2 (long axis along fixed +y).
        ellipse2_p_local = (mate_local_p2_2[0] + torch.tensor(
            [0.0, +ellipse_long_half_len_p2_2, 0.0],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        ellipse2_n_local = (mate_local_p2_2[0] + torch.tensor(
            [0.0, -ellipse_long_half_len_p2_2, 0.0],
            dtype=torch.float32, device=self.device,
        )).unsqueeze(0).repeat(self.num_envs, 1)
        _, ellipse2_p_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, ellipse2_p_local
        )
        _, ellipse2_n_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, ellipse2_n_local
        )
        # Shift ellipse endpoints by the same world +y offset so the cross
        # is drawn centered on the (offset) mate point.
        ellipse1_p_world = ellipse1_p_world + world_target_offset_p2_2
        ellipse1_n_world = ellipse1_n_world + world_target_offset_p2_2
        ellipse2_p_world = ellipse2_p_world + world_target_offset_p2_2
        ellipse2_n_world = ellipse2_n_world + world_target_offset_p2_2

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p2_2(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        # Points (cross center + mate point).
        points = [
            _to_tuple_p2_2(body_notch_world_p2_2[0]),
            _to_tuple_p2_2(fixed_mate_world_p2_2[0]),
        ]
        colors = [
            (0.2, 1.0, 0.2, 1.0),   # GREEN — body cross-notch center
            (1.0, 0.2, 0.2, 1.0),   # RED   — fixed mate point
        ]
        sizes = [14.0, 14.0]

        # Lines (cross arms + the TWO ellipse long axes).
        line_starts = [
            _to_tuple_p2_2(arm_y_p_world[0]),
            _to_tuple_p2_2(arm_z_p_world[0]),
            _to_tuple_p2_2(ellipse1_p_world[0]),
            _to_tuple_p2_2(ellipse2_p_world[0]),
        ]
        line_ends = [
            _to_tuple_p2_2(arm_y_n_world[0]),
            _to_tuple_p2_2(arm_z_n_world[0]),
            _to_tuple_p2_2(ellipse1_n_world[0]),
            _to_tuple_p2_2(ellipse2_n_world[0]),
        ]
        line_colors = [
            (0.2, 1.0, 0.2, 1.0),   # GREEN — cross arm along held +y
            (0.0, 0.6, 0.0, 1.0),   # DARK GREEN — cross arm along held +z
            (1.0, 0.2, 0.2, 1.0),   # RED — ellipse-1 long axis along fixed +x
            (0.8, 0.1, 0.4, 1.0),   # DARK RED — ellipse-2 long axis along fixed +y
        ]
        line_sizes = [4.5, 4.5, 4.5, 4.5]

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)
        self._dbg_draw.draw_lines(line_starts, line_ends, line_colors, line_sizes)

    def _scripted_action_body_insert_p2_2(self):
        """PlaneAssembly2 idx=2 — drive body's CROSS-NOTCH (GREEN) toward
        the fixed mate point (RED). 3-phase latched policy with stall
        detection + perp-plane wobble (same shape as idx=1's policy,
        but written from scratch with `_p2_2` suffix everywhere — no
        shared state or helpers with idx=1).

        A1) xy align (no descent until perp < tol)
        A2) hover above mate point along -axis_t * hover_offset
        B ) press down slowly + wobble when stalled (find tiny cross hole)
        """
        # ---- Per-idx state ----
        xy_align_frames_p2_2    = self._xy_align_frames_p2_2
        xy_aligned_latched_p2_2 = self._xy_aligned_latched_p2_2
        near_hover_frames_p2_2  = self._near_hover_frames_p2_2
        press_down_latched_p2_2 = self._press_down_latched_p2_2

        # ---- Geometry constants (offline) ----
        body_scale_p2_2          = 1.2
        # GREEN ↔ cross-notch DEPTH MID-POINT in body local (not the
        # outer -x face). See viz function for the histogram analysis.
        notch_local_unscaled_p2_2 = [-0.094, 0.0, 0.0045]
        # Fixed mate point — force both y=0 (mid-thickness) AND z=0
        # (geometric center of the fixed body, INSIDE the part, not on
        # any surface). With fixed_quat (0,-0.707,0.707,0), R[2,2]=-1
        # so fixed_local z=0 → world z = fixed_pos.z (exact mid).
        mate_local_unscaled_p2_2  = [0.0045 * body_scale_p2_2,
                                     0.0,
                                     0.0]

        # ---- Tunables ----
        hover_offset_m_p2_2       = 0.05
        align_perp_tol_p2_2       = 0.003
        along_tol_p2_2            = 0.01
        dwell_xy_align_p2_2       = 5
        dwell_to_press_p2_2       = 5
        press_down_depth_p2_2     = 0.20
        descent_scale_p2_2        = 0.3
        press_scale_p2_2          = 0.2    # slow press to avoid bounce
        stall_threshold_m_p2_2    = 0.0003
        stall_dwell_frames_p2_2   = 8
        wobble_amplitude_p2_2     = 0.0015
        wobble_period_frames_p2_2 = 30

        # ---- Body cross-notch midpoint in world (GREEN drive point) ----
        notch_local_p2_2 = torch.tensor(
            [c * body_scale_p2_2 for c in notch_local_unscaled_p2_2],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, body_notch_world_p2_2 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, notch_local_p2_2
        )

        # ---- Fixed mate point in world (RED target) ----
        mate_local_p2_2 = torch.tensor(
            mate_local_unscaled_p2_2,
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, fixed_mate_world_p2_2 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, mate_local_p2_2
        )
        # User-tuned world-frame offset: shift upper target +5mm to the
        # right (world +y). Kept in sync with the viz function.
        world_target_offset_p2_2 = torch.tensor(
            [0.0, 0.005, 0.0], dtype=torch.float32, device=self.device,
        )
        fixed_mate_world_p2_2 = fixed_mate_world_p2_2 + world_target_offset_p2_2

        # ---- Insertion axis in world (from connection_cfg2.axis_t) ----
        axis_t_local_p2_2 = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t_p2_2 = torch.zeros_like(self.fixed_pos)
        _, axis_t_world_p2_2 = torch_utils.tf_combine(
            self.fixed_quat, zero_t_p2_2, self.identity_quat, axis_t_local_p2_2
        )

        # ---- Drive point = body cross-notch midpoint world ----
        drive_world_p2_2 = body_notch_world_p2_2

        # ---- Targets ----
        above_target_p2_2      = fixed_mate_world_p2_2 - axis_t_world_p2_2 * hover_offset_m_p2_2
        press_down_target_p2_2 = fixed_mate_world_p2_2 + axis_t_world_p2_2 * press_down_depth_p2_2

        # ---- Decompose (held notch - fixed mate) ----
        delta_hf_p2_2          = drive_world_p2_2 - fixed_mate_world_p2_2
        along_hf_p2_2          = (delta_hf_p2_2 * axis_t_world_p2_2).sum(-1, keepdim=True)
        perp_hf_p2_2           = delta_hf_p2_2 - along_hf_p2_2 * axis_t_world_p2_2
        held_to_mate_perp_p2_2 = torch.norm(perp_hf_p2_2, dim=-1)

        # ---- Phase A1 → A2 latch ----
        xy_aligned_p2_2 = held_to_mate_perp_p2_2 < align_perp_tol_p2_2
        xy_align_frames_p2_2 = torch.where(
            xy_aligned_p2_2,
            xy_align_frames_p2_2 + 1,
            torch.zeros_like(xy_align_frames_p2_2),
        )
        xy_aligned_latched_p2_2 = xy_aligned_latched_p2_2 | (
            xy_align_frames_p2_2 >= dwell_xy_align_p2_2
        )

        # ---- Phase A2 → B latch ----
        d_above_p2_2     = drive_world_p2_2 - above_target_p2_2
        along_above_p2_2 = torch.abs((d_above_p2_2 * axis_t_world_p2_2).sum(-1))
        near_hover_p2_2 = (
            xy_aligned_latched_p2_2
            & (held_to_mate_perp_p2_2 < align_perp_tol_p2_2)
            & (along_above_p2_2 < along_tol_p2_2)
        )
        near_hover_frames_p2_2 = torch.where(
            near_hover_p2_2,
            near_hover_frames_p2_2 + 1,
            torch.zeros_like(near_hover_frames_p2_2),
        )
        press_down_latched_p2_2 = press_down_latched_p2_2 | (
            near_hover_frames_p2_2 >= dwell_to_press_p2_2
        )

        # ---- Write phase-transition state back ----
        self._xy_align_frames_p2_2    = xy_align_frames_p2_2
        self._xy_aligned_latched_p2_2 = xy_aligned_latched_p2_2
        self._near_hover_frames_p2_2  = near_hover_frames_p2_2
        self._press_down_latched_p2_2 = press_down_latched_p2_2

        # ---- Per-phase target ----
        in_phase_A1_p2_2 = ~xy_aligned_latched_p2_2
        in_phase_A2_p2_2 = xy_aligned_latched_p2_2 & (~press_down_latched_p2_2)
        in_phase_B_p2_2  = press_down_latched_p2_2

        target_p2_2 = torch.where(
            in_phase_B_p2_2.unsqueeze(-1),
            press_down_target_p2_2,
            above_target_p2_2,
        )

        # ---- Drive notch → target via fingertip action ----
        delta_p2_2     = target_p2_2 - drive_world_p2_2
        pos_action_p2_2 = delta_p2_2 / self.pos_threshold
        pos_action_p2_2 = torch.clamp(pos_action_p2_2, -1.0, 1.0)

        # Phase A1: zero along-axis component (xy-only, no descent).
        pa_along_signed_p2_2 = (pos_action_p2_2 * axis_t_world_p2_2).sum(-1, keepdim=True)
        pa_perp_p2_2 = pos_action_p2_2 - pa_along_signed_p2_2 * axis_t_world_p2_2
        pos_action_p2_2 = torch.where(
            in_phase_A1_p2_2.unsqueeze(-1), pa_perp_p2_2, pos_action_p2_2
        )

        # Slower scale on phase B (press_scale_p2_2) to avoid bounce.
        scale_p2_2 = torch.where(
            in_phase_B_p2_2.unsqueeze(-1),
            torch.full_like(pos_action_p2_2, press_scale_p2_2),
            torch.full_like(pos_action_p2_2, descent_scale_p2_2),
        )
        pos_action_p2_2 = pos_action_p2_2 * scale_p2_2

        # ----------------------------------------------------------------
        # Stall detection + perp-plane wobble (phase B only).
        # ----------------------------------------------------------------
        along_now_p2_2 = (drive_world_p2_2 * axis_t_world_p2_2).sum(-1)
        progress_p2_2 = torch.abs(along_now_p2_2 - self._prev_along_p2_2)
        stalled_now_p2_2 = (progress_p2_2 < stall_threshold_m_p2_2) & in_phase_B_p2_2
        self._stall_frames_p2_2 = torch.where(
            stalled_now_p2_2,
            self._stall_frames_p2_2 + 1,
            torch.zeros_like(self._stall_frames_p2_2),
        )
        self._prev_along_p2_2 = along_now_p2_2.clone()
        wobble_active_p2_2 = self._stall_frames_p2_2 >= stall_dwell_frames_p2_2

        if bool(wobble_active_p2_2.any().item()):
            self._wobble_step_p2_2 += 1

        # Perp-to-axis_t orthonormal direction (use world x; fall back to z).
        ref_world_p2_2 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p2_2)
        parallel_x_p2_2 = (axis_t_world_p2_2 * ref_world_p2_2).sum(-1, keepdim=True).abs() > 0.9
        ref_world_p2_2 = torch.where(
            parallel_x_p2_2,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).expand_as(axis_t_world_p2_2),
            ref_world_p2_2,
        )
        ref_along_p2_2 = (ref_world_p2_2 * axis_t_world_p2_2).sum(-1, keepdim=True)
        e1_world_p2_2 = ref_world_p2_2 - ref_along_p2_2 * axis_t_world_p2_2
        e1_world_p2_2 = e1_world_p2_2 / e1_world_p2_2.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        import math as _math
        phase_rad_p2_2 = (
            2.0 * _math.pi * (self._wobble_step_p2_2 / wobble_period_frames_p2_2)
        )
        wobble_offset_m_p2_2 = wobble_amplitude_p2_2 * _math.sin(phase_rad_p2_2)
        wobble_norm_p2_2 = wobble_offset_m_p2_2 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p2_2 = e1_world_p2_2 * wobble_norm_p2_2
        pos_action_p2_2 = torch.where(
            wobble_active_p2_2.unsqueeze(-1),
            pos_action_p2_2 + wobble_vec_p2_2,
            pos_action_p2_2,
        )
        pos_action_p2_2 = torch.clamp(pos_action_p2_2, -1.0, 1.0)

        if self.joint_created:
            pos_action_p2_2 = torch.zeros_like(pos_action_p2_2)

        # ----------------------------------------------------------------
        # Yaw alignment around axis_t_world.
        # We want one body cross arm (along body +z) to point in the same
        # direction as ellipse-1 long axis (fixed +x) in world. The
        # signed rotation around axis_t_world that achieves this is
        # encoded as an axis-angle vector (axis_t_world * yaw_err); this
        # form auto-handles whether axis_t_world is +z or -z in world.
        # Active in phase A1/A2; frozen in B so descent doesn't shimmy
        # the cross out of orientation.
        # ----------------------------------------------------------------
        yaw_align_scale_p2_2 = 0.4

        # Body local +z direction expressed in world.
        body_z_local_p2_2 = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, body_z_world_p2_2 = torch_utils.tf_combine(
            self.held_quat, torch.zeros_like(self.held_pos),
            self.identity_quat, body_z_local_p2_2,
        )
        # Ellipse long axis (fixed +x) in world.
        ellipse_axis_local_p2_2 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, ellipse_axis_world_p2_2 = torch_utils.tf_combine(
            self.fixed_quat, torch.zeros_like(self.fixed_pos),
            self.identity_quat, ellipse_axis_local_p2_2,
        )
        # Project both onto plane perpendicular to axis_t_world.
        def _proj_perp_axist(vec, axt):
            a = (vec * axt).sum(-1, keepdim=True)
            p = vec - a * axt
            return p / p.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        body_z_perp_p2_2 = _proj_perp_axist(body_z_world_p2_2, axis_t_world_p2_2)
        ellipse_perp_p2_2 = _proj_perp_axist(ellipse_axis_world_p2_2, axis_t_world_p2_2)
        # Signed angle from body_z_perp to ellipse_perp, AROUND axis_t_world.
        cross_axt_p2_2 = torch.cross(body_z_perp_p2_2, ellipse_perp_p2_2, dim=-1)
        sin_yaw_p2_2 = (cross_axt_p2_2 * axis_t_world_p2_2).sum(-1)
        cos_yaw_p2_2 = (body_z_perp_p2_2 * ellipse_perp_p2_2).sum(-1)
        yaw_err_p2_2 = torch.atan2(sin_yaw_p2_2, cos_yaw_p2_2)
        # Map yaw_err to a 3-D axis-angle action: rotation around
        # axis_t_world by yaw_err. The controller divides by rot_threshold
        # to get axis-angle magnitude.
        rot_action_vec_p2_2 = axis_t_world_p2_2 * yaw_err_p2_2.unsqueeze(-1) * (
            yaw_align_scale_p2_2 / (self.rot_threshold + 1e-8)
        )
        rot_action_vec_p2_2 = torch.clamp(rot_action_vec_p2_2, -1.0, 1.0)
        # Freeze rotation during press phase.
        rot_action_vec_p2_2 = torch.where(
            in_phase_B_p2_2.unsqueeze(-1),
            torch.zeros_like(rot_action_vec_p2_2),
            rot_action_vec_p2_2,
        )
        rot_action_p2_2 = rot_action_vec_p2_2
        if self.joint_created:
            rot_action_p2_2 = torch.zeros_like(rot_action_p2_2)

        # ---- Diagnostic print ----
        green = drive_world_p2_2[0].tolist()
        red   = fixed_mate_world_p2_2[0].tolist()
        tgt   = target_p2_2[0].tolist()
        phase = "A1-xy" if bool(in_phase_A1_p2_2[0]) else (
            "A2-hover" if bool(in_phase_A2_p2_2[0]) else "B-press"
        )
        # Center-vs-center xy offset (axis_t-perp projection in world).
        gr_xy_diff_p2_2 = drive_world_p2_2[0] - fixed_mate_world_p2_2[0]
        gr_along_p2_2 = (gr_xy_diff_p2_2 * axis_t_world_p2_2[0]).sum()
        gr_perp_vec_p2_2 = gr_xy_diff_p2_2 - gr_along_p2_2 * axis_t_world_p2_2[0]
        center_dx_p2_2 = float(gr_perp_vec_p2_2[0])
        center_dy_p2_2 = float(gr_perp_vec_p2_2[1])
        print(
            f"[plane2_2 {phase}] "
            f"green=({green[0]:+.4f},{green[1]:+.4f},{green[2]:+.4f})  "
            f"red=({red[0]:+.4f},{red[1]:+.4f},{red[2]:+.4f})  "
            f"target=({tgt[0]:+.4f},{tgt[1]:+.4f},{tgt[2]:+.4f})  "
            f"dx={center_dx_p2_2*1000:+.2f}mm dy={center_dy_p2_2*1000:+.2f}mm  "
            f"perp={held_to_mate_perp_p2_2[0]*1000:.2f}mm "
            f"along_hover={along_above_p2_2[0]*1000:.2f}mm  "
            f"yaw={yaw_err_p2_2[0]*180/3.14159:+.2f}°  "
            f"xy_latch={int(xy_aligned_latched_p2_2[0])} "
            f"press_latch={int(press_down_latched_p2_2[0])}  "
            f"stall={int(self._stall_frames_p2_2[0])} "
            f"wobble={'YES' if bool(wobble_active_p2_2[0]) else 'no'}"
        )

        gripper_action_p2_2 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p2_2, rot_action_p2_2, gripper_action_p2_2], dim=-1)

    def _scripted_action_tail_insert_p2_1(self):
        """PlaneAssembly2 idx=1 — drive the held opening midpoint (GREEN)
        toward the fixed lower midpoint (RED) along the insertion axis.

        Anchor points (offline mesh-derived):
          held opening midpoint = held_local (0, 0, -0.035)
          fixed lower midpoint  = fixed_local (0, 0, +0.035)

        3-phase latched policy with `_p2_1` suffix on every state /
        local variable (no plane1 sharing).

        A1) xy align — drive only perp-to-axis_t (no descent)
        A2) hover    — drive 3D to hover_offset "above" fixed lower mid
        B ) press    — drive deep along -axis_t into the fixed half
        """
        # ---- Per-idx state ----
        xy_align_frames_p2_1    = self._xy_align_frames_p2_1
        xy_aligned_latched_p2_1 = self._xy_aligned_latched_p2_1
        near_hover_frames_p2_1  = self._near_hover_frames_p2_1
        press_down_latched_p2_1 = self._press_down_latched_p2_1

        # ---- Tunables (idx-specific copy) ----
        hover_offset_m_p2_1   = 0.05
        align_perp_tol_p2_1   = 0.003
        along_tol_p2_1        = 0.01
        dwell_xy_align_p2_1   = 5
        dwell_to_press_p2_1   = 5
        press_down_depth_p2_1 = 0.20
        descent_scale_p2_1    = 0.3
        # — slower press to avoid collision bounce —
        press_scale_p2_1      = 0.2   # phase B scale (was 1.0); slower descent
        # — stall detection + wobble (phase B only) —
        stall_threshold_m_p2_1   = 0.0003   # < 0.3 mm/frame along axis_t counts as stalled
        stall_dwell_frames_p2_1  = 8        # frames stalled before wobble engages
        wobble_amplitude_p2_1    = 0.0015   # 1.5 mm perp oscillation amplitude
        wobble_period_frames_p2_1 = 30      # full sine period in frames

        # ---- Held opening midpoint in world (DRIVE POINT, GREEN) ----
        held_opening_local_p2_1 = torch.tensor(
            [0.0, 0.0, -0.035], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, held_opening_world_p2_1 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, held_opening_local_p2_1
        )

        # ---- Fixed lower midpoint in world (TARGET ANCHOR, RED) ----
        fixed_lower_local_p2_1 = torch.tensor(
            [0.0, 0.0, +0.035], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, fixed_lower_world_p2_1 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_lower_local_p2_1
        )

        # ---- Insertion axis in world (from connection_cfg1.axis_t) ----
        axis_t_local_p2_1 = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t_p2_1 = torch.zeros_like(self.fixed_pos)
        _, axis_t_world_p2_1 = torch_utils.tf_combine(
            self.fixed_quat, zero_t_p2_1, self.identity_quat, axis_t_local_p2_1
        )

        # ---- Drive point = held opening midpoint world ----
        drive_world_p2_1 = held_opening_world_p2_1

        # ---- Targets relative to fixed lower midpoint ----
        # Sign convention: axis_t for plane2 idx=1 → world -z (down). So
        # "above" (toward gripper) needs -axis_t: above = fixed - axis_t * h.
        # "press" goes along +axis_t (deeper into the slot below).
        above_target_p2_1      = fixed_lower_world_p2_1 - axis_t_world_p2_1 * hover_offset_m_p2_1
        press_down_target_p2_1 = fixed_lower_world_p2_1 + axis_t_world_p2_1 * press_down_depth_p2_1

        # ---- Decompose (held opening - fixed lower) ----
        delta_hf_p2_1          = drive_world_p2_1 - fixed_lower_world_p2_1
        along_hf_p2_1          = (delta_hf_p2_1 * axis_t_world_p2_1).sum(-1, keepdim=True)
        perp_hf_p2_1           = delta_hf_p2_1 - along_hf_p2_1 * axis_t_world_p2_1
        held_to_mate_perp_p2_1 = torch.norm(perp_hf_p2_1, dim=-1)

        # ---- Phase A1 → A2 latch ----
        xy_aligned_p2_1 = held_to_mate_perp_p2_1 < align_perp_tol_p2_1
        xy_align_frames_p2_1 = torch.where(
            xy_aligned_p2_1,
            xy_align_frames_p2_1 + 1,
            torch.zeros_like(xy_align_frames_p2_1),
        )
        xy_aligned_latched_p2_1 = xy_aligned_latched_p2_1 | (
            xy_align_frames_p2_1 >= dwell_xy_align_p2_1
        )

        # ---- Phase A2 → B latch ----
        d_above_p2_1     = drive_world_p2_1 - above_target_p2_1
        along_above_p2_1 = torch.abs((d_above_p2_1 * axis_t_world_p2_1).sum(-1))
        near_hover_p2_1 = (
            xy_aligned_latched_p2_1
            & (held_to_mate_perp_p2_1 < align_perp_tol_p2_1)
            & (along_above_p2_1 < along_tol_p2_1)
        )
        near_hover_frames_p2_1 = torch.where(
            near_hover_p2_1,
            near_hover_frames_p2_1 + 1,
            torch.zeros_like(near_hover_frames_p2_1),
        )
        press_down_latched_p2_1 = press_down_latched_p2_1 | (
            near_hover_frames_p2_1 >= dwell_to_press_p2_1
        )

        # ---- Write state back ----
        self._xy_align_frames_p2_1    = xy_align_frames_p2_1
        self._xy_aligned_latched_p2_1 = xy_aligned_latched_p2_1
        self._near_hover_frames_p2_1  = near_hover_frames_p2_1
        self._press_down_latched_p2_1 = press_down_latched_p2_1

        # ---- Per-phase target ----
        in_phase_A1_p2_1 = ~xy_aligned_latched_p2_1
        in_phase_A2_p2_1 = xy_aligned_latched_p2_1 & (~press_down_latched_p2_1)
        in_phase_B_p2_1  = press_down_latched_p2_1

        target_p2_1 = torch.where(
            in_phase_B_p2_1.unsqueeze(-1),
            press_down_target_p2_1,
            above_target_p2_1,
        )

        # ---- Drive held_pos → target via fingertip action ----
        delta_p2_1     = target_p2_1 - drive_world_p2_1
        pos_action_p2_1 = delta_p2_1 / self.pos_threshold
        pos_action_p2_1 = torch.clamp(pos_action_p2_1, -1.0, 1.0)

        # Phase A1: zero along-axis component (xy-only, no descent).
        pa_along_signed_p2_1 = (pos_action_p2_1 * axis_t_world_p2_1).sum(-1, keepdim=True)
        pa_perp_p2_1 = pos_action_p2_1 - pa_along_signed_p2_1 * axis_t_world_p2_1
        pos_action_p2_1 = torch.where(
            in_phase_A1_p2_1.unsqueeze(-1), pa_perp_p2_1, pos_action_p2_1
        )

        # Phase B uses a SLOWER scale (press_scale_p2_1) to prevent the
        # held part from bouncing on the fixed surface during insertion.
        scale_p2_1 = torch.where(
            in_phase_B_p2_1.unsqueeze(-1),
            torch.full_like(pos_action_p2_1, press_scale_p2_1),
            torch.full_like(pos_action_p2_1, descent_scale_p2_1),
        )
        pos_action_p2_1 = pos_action_p2_1 * scale_p2_1

        # ----------------------------------------------------------------
        # Stall detection + perp-plane wobble (phase B only).
        # When the held part is pressing down but its along-axis_t
        # progress has stalled (likely caught on the lip of a tiny hole),
        # superimpose a 1.5 mm back-and-forth oscillation in the
        # axis_t-perpendicular plane so the tip can shimmy into the hole.
        # ----------------------------------------------------------------
        # Current along-axis_t position (signed scalar per env).
        along_now_p2_1 = (drive_world_p2_1 * axis_t_world_p2_1).sum(-1)
        progress_p2_1 = torch.abs(along_now_p2_1 - self._prev_along_p2_1)
        stalled_now_p2_1 = (progress_p2_1 < stall_threshold_m_p2_1) & in_phase_B_p2_1
        self._stall_frames_p2_1 = torch.where(
            stalled_now_p2_1,
            self._stall_frames_p2_1 + 1,
            torch.zeros_like(self._stall_frames_p2_1),
        )
        self._prev_along_p2_1 = along_now_p2_1.clone()
        wobble_active_p2_1 = self._stall_frames_p2_1 >= stall_dwell_frames_p2_1

        if bool(wobble_active_p2_1.any().item()):
            self._wobble_step_p2_1 += 1

        # Build a perp-to-axis_t orthonormal basis (e1) for the wobble
        # direction. Use world x as reference; flip to world z if axis_t
        # is nearly parallel to world x.
        ref_world_p2_1 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p2_1)
        parallel_x_p2_1 = (axis_t_world_p2_1 * ref_world_p2_1).sum(-1, keepdim=True).abs() > 0.9
        ref_world_p2_1 = torch.where(
            parallel_x_p2_1,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).expand_as(axis_t_world_p2_1),
            ref_world_p2_1,
        )
        # Project ref to perp of axis_t and normalize → e1.
        ref_along_p2_1 = (ref_world_p2_1 * axis_t_world_p2_1).sum(-1, keepdim=True)
        e1_world_p2_1 = ref_world_p2_1 - ref_along_p2_1 * axis_t_world_p2_1
        e1_world_p2_1 = e1_world_p2_1 / e1_world_p2_1.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 1-D sinusoidal wobble along e1.
        import math as _math
        phase_rad_p2_1 = (
            2.0 * _math.pi * (self._wobble_step_p2_1 / wobble_period_frames_p2_1)
        )
        wobble_offset_m_p2_1 = wobble_amplitude_p2_1 * _math.sin(phase_rad_p2_1)
        # Convert metres → normalized pos_action units (the controller
        # multiplies pos_action by pos_threshold inside _apply_action).
        wobble_norm_p2_1 = wobble_offset_m_p2_1 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p2_1 = e1_world_p2_1 * wobble_norm_p2_1
        pos_action_p2_1 = torch.where(
            wobble_active_p2_1.unsqueeze(-1),
            pos_action_p2_1 + wobble_vec_p2_1,
            pos_action_p2_1,
        )
        pos_action_p2_1 = torch.clamp(pos_action_p2_1, -1.0, 1.0)

        if self.joint_created:
            pos_action_p2_1 = torch.zeros_like(pos_action_p2_1)

        # ---- Diagnostic print ----
        green = drive_world_p2_1[0].tolist()
        red   = fixed_lower_world_p2_1[0].tolist()
        tgt   = target_p2_1[0].tolist()
        phase = "A1-xy" if bool(in_phase_A1_p2_1[0]) else (
            "A2-hover" if bool(in_phase_A2_p2_1[0]) else "B-press"
        )
        print(
            f"[plane2_1 {phase}] "
            f"green=({green[0]:+.4f},{green[1]:+.4f},{green[2]:+.4f})  "
            f"red=({red[0]:+.4f},{red[1]:+.4f},{red[2]:+.4f})  "
            f"target=({tgt[0]:+.4f},{tgt[1]:+.4f},{tgt[2]:+.4f})  "
            f"perp={held_to_mate_perp_p2_1[0]*1000:.2f}mm "
            f"along_hover={along_above_p2_1[0]*1000:.2f}mm  "
            f"xy_latch={int(xy_aligned_latched_p2_1[0])} "
            f"press_latch={int(press_down_latched_p2_1[0])}  "
            f"stall={int(self._stall_frames_p2_1[0])} "
            f"wobble={'YES' if bool(wobble_active_p2_1[0]) else 'no'}"
        )

        rot_action_p2_1     = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action_p2_1 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p2_1, rot_action_p2_1, gripper_action_p2_1], dim=-1)

    # ------------------------------------------------------------------
    # plane2 idx=3 — propeller hub circle (HELD) ↦ tail rod center (FIXED)
    # ------------------------------------------------------------------
    # STRICT independence: every name carries the `_p2_3` suffix so this
    # block shares NO state, helpers or constants with idx=1 or idx=2.
    # Offline mesh analysis (one-time):
    #   - propeller.obj bbox  [-0.07,0.07] × [-0.004,0.004] × [-0.012,0.012]
    #     → centroid ≈ (0, 0, 0): hub circle center is exactly at held_local
    #       origin. Apply held_scale=1.2 → still (0, 0, 0).
    #   - tail_half_wider1 connection_cfg3.pose_to_base translation
    #     (0.004, -0.006, -0.27) is where the held origin sits relative to
    #     the fixed origin when assembled; we use that as the rod-center
    #     target in fixed_local.
    def _visualize_propeller_and_tail_p2_3(self):
        """Draw two dots: GREEN = propeller hub center (held), RED = tail
        rod center (fixed). Used as visual confirmation that the script's
        anchor points are where we think they are."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p2_3_legend_printed", False):
            print("\n[plane2 idx=3] viewer legend (center-to-center align):")
            print("  GREEN dot = propeller hub center   held_local (0, 0, 0)")
            print("  RED   dot = BODY rod tip           body_local (0.1126, 0, 0.0044) × 1.2")
            self._p2_3_legend_printed = True

        # ---- Propeller hub center in world (held) ----
        # Centroid of propeller.obj is at the local origin; held_scale only
        # rescales non-origin points so (0,0,0) stays at (0,0,0).
        hub_local_p2_3 = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hub_world_p2_3 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hub_local_p2_3
        )

        # ---- Body rod tip in world (the "rod" is on the body, not the
        # tail). body.obj sparse-vertex band x∈[0.085,0.1126], y≈0, z≈0.0044
        # → free tip at body_local (0.1126, 0, 0.0044), scaled by 1.2.
        body_scale_p2_3 = 1.2
        body_pos_p2_3   = self._body.data.root_pos_w - self.scene.env_origins
        body_quat_p2_3  = self._body.data.root_quat_w
        rod_local_p2_3 = torch.tensor(
            [0.1126 * body_scale_p2_3,
             0.0    * body_scale_p2_3,
             0.0044 * body_scale_p2_3],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, rod_world_p2_3 = torch_utils.tf_combine(
            body_quat_p2_3, body_pos_p2_3, self.identity_quat, rod_local_p2_3
        )

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p2_3(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        points = [
            _to_tuple_p2_3(hub_world_p2_3[0]),
            _to_tuple_p2_3(rod_world_p2_3[0]),
        ]
        colors = [
            (0.0, 1.0, 0.0, 1.0),   # GREEN — held propeller hub
            (1.0, 0.0, 0.0, 1.0),   # RED   — fixed tail rod
        ]
        sizes = [22.0, 22.0]
        self._dbg_draw.clear_points()
        self._dbg_draw.draw_points(points, colors, sizes)

    def _scripted_action_propeller_insert_p2_3(self):
        """plane2 idx=3 — drive propeller hub (GREEN) onto tail rod center
        (RED). Pure center-to-center alignment as the user requested; no
        yaw constraint. 3-phase latched policy with stall + perp-plane
        wobble (in case the rod is a tight fit). Written from scratch
        with `_p2_3` suffix everywhere — no shared state or helpers."""
        # ---- Per-idx state ----
        xy_align_frames_p2_3    = self._xy_align_frames_p2_3
        xy_aligned_latched_p2_3 = self._xy_aligned_latched_p2_3
        near_hover_frames_p2_3  = self._near_hover_frames_p2_3
        press_down_latched_p2_3 = self._press_down_latched_p2_3

        # ---- Geometry constants (offline mesh analysis) ----
        # GREEN ↔ propeller hub center = held_local (0,0,0) (mesh centroid).
        # RED  ↔ body rod tip in body_local. body.obj has a sparse-vertex
        # band x∈[0.085,0.1126], y≈0, z≈0.0044 — the rod sticking out
        # from the body's +x face. Tip is at x=0.1126 (free end).
        # body_scale=1.2 must be applied to all body_local coordinates.
        hub_local_unscaled_p2_3 = [0.0, 0.0, 0.0]
        body_scale_p2_3         = 1.2
        rod_local_unscaled_p2_3 = [0.1126 * body_scale_p2_3,
                                   0.0    * body_scale_p2_3,
                                   0.0044 * body_scale_p2_3]

        # ---- Tunables ----
        hover_offset_m_p2_3       = 0.05
        align_perp_tol_p2_3       = 0.003
        along_tol_p2_3            = 0.01
        dwell_xy_align_p2_3       = 5
        dwell_to_press_p2_3       = 5
        press_down_depth_p2_3     = 0.20
        descent_scale_p2_3        = 0.3
        press_scale_p2_3          = 0.2
        stall_threshold_m_p2_3    = 0.0003
        stall_dwell_frames_p2_3   = 8
        wobble_amplitude_p2_3     = 0.0015
        wobble_period_frames_p2_3 = 30

        # ---- Propeller hub center in world (GREEN drive point) ----
        hub_local_p2_3 = torch.tensor(
            hub_local_unscaled_p2_3, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hub_world_p2_3 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hub_local_p2_3
        )

        # ---- Body rod tip in world (RED target) ----
        # The "rod" the propeller mounts onto is on the BODY (already
        # assembled to the tail in earlier idxs), not on the fixed
        # tail_half_wider itself. Read body pose live.
        body_pos_p2_3  = self._body.data.root_pos_w - self.scene.env_origins
        body_quat_p2_3 = self._body.data.root_quat_w
        rod_local_p2_3 = torch.tensor(
            rod_local_unscaled_p2_3, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, rod_world_p2_3 = torch_utils.tf_combine(
            body_quat_p2_3, body_pos_p2_3, self.identity_quat, rod_local_p2_3
        )

        # ---- Insertion axis in world ----
        # User-requested behavior: align GREEN xy onto RED in WORLD xy
        # plane, then descend in WORLD -z. Hardcode axis_t = world +z so
        # the perp plane is world xy and the press direction is straight
        # down — independent of body's orientation.
        axis_t_world_p2_3 = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        drive_world_p2_3 = hub_world_p2_3

        # ---- Targets ----
        # axis_t_world points body +x = OUTWARD from the rod's free tip.
        # Propeller approaches from that outward side, then slides onto
        # the rod toward the body interior (-axis_t direction).
        above_target_p2_3      = rod_world_p2_3 + axis_t_world_p2_3 * hover_offset_m_p2_3
        press_down_target_p2_3 = rod_world_p2_3 - axis_t_world_p2_3 * press_down_depth_p2_3

        # ---- Decompose (hub - rod) ----
        delta_hr_p2_3          = drive_world_p2_3 - rod_world_p2_3
        along_hr_p2_3          = (delta_hr_p2_3 * axis_t_world_p2_3).sum(-1, keepdim=True)
        perp_hr_p2_3           = delta_hr_p2_3 - along_hr_p2_3 * axis_t_world_p2_3
        hub_to_rod_perp_p2_3   = torch.norm(perp_hr_p2_3, dim=-1)

        # ---- A1: xy align latch ----
        xy_aligned_p2_3 = hub_to_rod_perp_p2_3 < align_perp_tol_p2_3
        xy_align_frames_p2_3 = torch.where(
            xy_aligned_p2_3,
            xy_align_frames_p2_3 + 1,
            torch.zeros_like(xy_align_frames_p2_3),
        )
        xy_aligned_latched_p2_3 = xy_aligned_latched_p2_3 | (
            xy_align_frames_p2_3 >= dwell_xy_align_p2_3
        )

        # ---- A2: near hover above mate ----
        d_above_p2_3     = drive_world_p2_3 - above_target_p2_3
        along_above_p2_3 = torch.abs((d_above_p2_3 * axis_t_world_p2_3).sum(-1))
        near_hover_p2_3 = (
            xy_aligned_latched_p2_3
            & (hub_to_rod_perp_p2_3 < align_perp_tol_p2_3)
            & (along_above_p2_3 < along_tol_p2_3)
        )
        near_hover_frames_p2_3 = torch.where(
            near_hover_p2_3,
            near_hover_frames_p2_3 + 1,
            torch.zeros_like(near_hover_frames_p2_3),
        )
        press_down_latched_p2_3 = press_down_latched_p2_3 | (
            near_hover_frames_p2_3 >= dwell_to_press_p2_3
        )

        # ---- Write back ----
        self._xy_align_frames_p2_3    = xy_align_frames_p2_3
        self._xy_aligned_latched_p2_3 = xy_aligned_latched_p2_3
        self._near_hover_frames_p2_3  = near_hover_frames_p2_3
        self._press_down_latched_p2_3 = press_down_latched_p2_3

        # ---- Phase split ----
        in_phase_A1_p2_3 = ~xy_aligned_latched_p2_3
        in_phase_A2_p2_3 = xy_aligned_latched_p2_3 & (~press_down_latched_p2_3)
        in_phase_B_p2_3  = press_down_latched_p2_3

        target_p2_3 = torch.where(
            in_phase_B_p2_3.unsqueeze(-1),
            press_down_target_p2_3,
            above_target_p2_3,
        )

        # ---- Raw pos action ----
        delta_p2_3      = target_p2_3 - drive_world_p2_3
        pos_action_p2_3 = delta_p2_3 / self.pos_threshold
        pos_action_p2_3 = torch.clamp(pos_action_p2_3, -1.0, 1.0)

        # A1: zero out along-axis component so we don't drop until xy locked.
        pa_along_signed_p2_3 = (pos_action_p2_3 * axis_t_world_p2_3).sum(-1, keepdim=True)
        pa_perp_p2_3 = pos_action_p2_3 - pa_along_signed_p2_3 * axis_t_world_p2_3
        pos_action_p2_3 = torch.where(
            in_phase_A1_p2_3.unsqueeze(-1), pa_perp_p2_3, pos_action_p2_3
        )

        # Slower phase B to avoid bounce.
        scale_p2_3 = torch.where(
            in_phase_B_p2_3.unsqueeze(-1),
            torch.full_like(pos_action_p2_3, press_scale_p2_3),
            torch.full_like(pos_action_p2_3, descent_scale_p2_3),
        )
        pos_action_p2_3 = pos_action_p2_3 * scale_p2_3

        # ---- Stall detection + perp-plane wobble during phase B ----
        along_now_p2_3 = (drive_world_p2_3 * axis_t_world_p2_3).sum(-1)
        progress_p2_3 = torch.abs(along_now_p2_3 - self._prev_along_p2_3)
        stalled_now_p2_3 = (progress_p2_3 < stall_threshold_m_p2_3) & in_phase_B_p2_3
        self._stall_frames_p2_3 = torch.where(
            stalled_now_p2_3,
            self._stall_frames_p2_3 + 1,
            torch.zeros_like(self._stall_frames_p2_3),
        )
        self._prev_along_p2_3 = along_now_p2_3.clone()
        wobble_active_p2_3 = self._stall_frames_p2_3 >= stall_dwell_frames_p2_3

        if bool(wobble_active_p2_3.any().item()):
            self._wobble_step_p2_3 += 1

        # Build a unit perp basis vector e1 ⟂ axis_t.
        ref_world_p2_3 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p2_3)
        parallel_x_p2_3 = (axis_t_world_p2_3 * ref_world_p2_3).sum(-1, keepdim=True).abs() > 0.9
        ref_world_p2_3 = torch.where(
            parallel_x_p2_3,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).expand_as(axis_t_world_p2_3),
            ref_world_p2_3,
        )
        ref_along_p2_3 = (ref_world_p2_3 * axis_t_world_p2_3).sum(-1, keepdim=True)
        e1_world_p2_3 = ref_world_p2_3 - ref_along_p2_3 * axis_t_world_p2_3
        e1_world_p2_3 = e1_world_p2_3 / e1_world_p2_3.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        import math as _math
        phase_rad_p2_3 = (
            2.0 * _math.pi * (self._wobble_step_p2_3 / wobble_period_frames_p2_3)
        )
        wobble_offset_m_p2_3 = wobble_amplitude_p2_3 * _math.sin(phase_rad_p2_3)
        wobble_norm_p2_3 = wobble_offset_m_p2_3 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p2_3 = e1_world_p2_3 * wobble_norm_p2_3
        pos_action_p2_3 = torch.where(
            wobble_active_p2_3.unsqueeze(-1),
            pos_action_p2_3 + wobble_vec_p2_3,
            pos_action_p2_3,
        )
        pos_action_p2_3 = torch.clamp(pos_action_p2_3, -1.0, 1.0)

        # ---- Diagnostic print ----
        green = drive_world_p2_3[0].tolist()
        red   = rod_world_p2_3[0].tolist()
        phase = "A1-xy" if bool(in_phase_A1_p2_3[0]) else (
            "A2-hover" if bool(in_phase_A2_p2_3[0]) else "B-press"
        )
        gr_diff_p2_3 = drive_world_p2_3[0] - rod_world_p2_3[0]
        gr_along_p2_3 = (gr_diff_p2_3 * axis_t_world_p2_3[0]).sum()
        gr_perp_vec_p2_3 = gr_diff_p2_3 - gr_along_p2_3 * axis_t_world_p2_3[0]
        center_dx_p2_3 = float(gr_perp_vec_p2_3[0])
        center_dy_p2_3 = float(gr_perp_vec_p2_3[1])
        print(
            f"[p2_3] phase={phase}  "
            f"hub=({green[0]:+.3f},{green[1]:+.3f},{green[2]:+.3f})  "
            f"rod=({red[0]:+.3f},{red[1]:+.3f},{red[2]:+.3f})  "
            f"dx={center_dx_p2_3*1000:+.2f}mm dy={center_dy_p2_3*1000:+.2f}mm  "
            f"perp={hub_to_rod_perp_p2_3[0]*1000:.2f}mm "
            f"along_hover={along_above_p2_3[0]*1000:.2f}mm  "
            f"xy_latch={int(xy_aligned_latched_p2_3[0])} "
            f"press_latch={int(press_down_latched_p2_3[0])}  "
            f"stall={int(self._stall_frames_p2_3[0])} "
            f"wobble={'YES' if bool(wobble_active_p2_3[0]) else 'no'}"
        )

        rot_action_p2_3     = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action_p2_3 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p2_3, rot_action_p2_3, gripper_action_p2_3], dim=-1)

    # ------------------------------------------------------------------
    # plane2 idx=4 — holder peg tip (HELD) ↦ propeller hub hole (FIXED-ish)
    # ------------------------------------------------------------------
    # STRICT independence: every name carries the `_p2_4` suffix. No
    # state, helpers or constants shared with idx=1/2/3.
    # Offline mesh analysis:
    #   - holder.obj: main body x∈[-0.0044,+0.0046], with a thin peg in
    #     x∈[0.0248,0.0256] (38 sparse verts) sticking out in +x. Peg tip
    #     ≈ holder_local (0.0256, 0, 0). Holder scale = (1.2, 1, 1) so
    #     scaled-local tip = (0.0307, 0, 0).
    #   - The "hole" is the propeller hub center (same point that was the
    #     GREEN dot in idx=3), at propeller_local (0, 0, 0) — read live
    #     from self._propeller.data.root_pos_w.
    def _visualize_holder_and_hole_p2_4(self):
        """Draw two dots: GREEN = holder peg tip (held drive point),
        RED = propeller hub hole (target)."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p2_4_legend_printed", False):
            print("\n[plane2 idx=4] viewer legend (peg-into-hole):")
            print("  GREEN dot = holder peg tip      held_local (0.0256, 0, 0) × scale(1.2,1,1)")
            print("  RED   dot = propeller hub hole  propeller_pos (live)")
            self._p2_4_legend_printed = True

        # ---- Holder peg tip in world (GREEN drive point) ----
        # Holder scale = (1.2, 1, 1) — apply per-axis before tf_combine.
        peg_tip_local_p2_4 = torch.tensor(
            [0.0256 * 1.2, 0.0 * 1.0, 0.0 * 1.0],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_tip_world_p2_4 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_tip_local_p2_4
        )

        # ---- Propeller hub center in world (RED target hole) ----
        prop_pos_p2_4  = self._propeller.data.root_pos_w - self.scene.env_origins
        prop_quat_p2_4 = self._propeller.data.root_quat_w
        hole_local_p2_4 = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_world_p2_4 = torch_utils.tf_combine(
            prop_quat_p2_4, prop_pos_p2_4, self.identity_quat, hole_local_p2_4
        )

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p2_4(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        points = [
            _to_tuple_p2_4(peg_tip_world_p2_4[0]),
            _to_tuple_p2_4(hole_world_p2_4[0]),
        ]
        colors = [
            (0.0, 1.0, 0.0, 1.0),   # GREEN — peg tip
            (1.0, 0.0, 0.0, 1.0),   # RED   — hole
        ]
        sizes = [22.0, 22.0]
        self._dbg_draw.clear_points()
        self._dbg_draw.draw_points(points, colors, sizes)

    def _scripted_action_holder_insert_p2_4(self):
        """plane2 idx=4 — drive holder peg tip (GREEN) into propeller hub
        hole (RED) along world -z. 3-phase latched policy with stall +
        perp-plane wobble. Written from scratch with `_p2_4` suffix
        everywhere — no shared state or helpers with idx=1/2/3."""
        # ---- Per-idx state ----
        xy_align_frames_p2_4    = self._xy_align_frames_p2_4
        xy_aligned_latched_p2_4 = self._xy_aligned_latched_p2_4
        near_hover_frames_p2_4  = self._near_hover_frames_p2_4
        press_down_latched_p2_4 = self._press_down_latched_p2_4

        # ---- Geometry constants (offline mesh analysis) ----
        peg_tip_local_unscaled_p2_4 = [0.0256 * 1.2, 0.0 * 1.0, 0.0 * 1.0]
        hole_local_unscaled_p2_4    = [0.0, 0.0, 0.0]  # propeller hub centroid

        # ---- Tunables ----
        hover_offset_m_p2_4       = 0.05
        align_perp_tol_p2_4       = 0.003
        along_tol_p2_4            = 0.01
        dwell_xy_align_p2_4       = 5
        dwell_to_press_p2_4       = 5
        press_down_depth_p2_4     = 0.20
        descent_scale_p2_4        = 0.3
        press_scale_p2_4          = 0.2
        stall_threshold_m_p2_4    = 0.0003
        stall_dwell_frames_p2_4   = 8
        wobble_amplitude_p2_4     = 0.0015
        wobble_period_frames_p2_4 = 30

        # ---- Holder peg tip in world (GREEN drive point) ----
        peg_tip_local_p2_4 = torch.tensor(
            peg_tip_local_unscaled_p2_4, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_tip_world_p2_4 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_tip_local_p2_4
        )

        # ---- Propeller hub hole in world (RED target) ----
        prop_pos_p2_4  = self._propeller.data.root_pos_w - self.scene.env_origins
        prop_quat_p2_4 = self._propeller.data.root_quat_w
        hole_local_p2_4 = torch.tensor(
            hole_local_unscaled_p2_4, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_world_p2_4 = torch_utils.tf_combine(
            prop_quat_p2_4, prop_pos_p2_4, self.identity_quat, hole_local_p2_4
        )

        # ---- Insertion axis = world +z (xy align then descend -z) ----
        axis_t_world_p2_4 = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        drive_world_p2_4 = peg_tip_world_p2_4

        # ---- Targets ----
        above_target_p2_4      = hole_world_p2_4 + axis_t_world_p2_4 * hover_offset_m_p2_4
        press_down_target_p2_4 = hole_world_p2_4 - axis_t_world_p2_4 * press_down_depth_p2_4

        # ---- Decompose (peg - hole) ----
        delta_ph_p2_4          = drive_world_p2_4 - hole_world_p2_4
        along_ph_p2_4          = (delta_ph_p2_4 * axis_t_world_p2_4).sum(-1, keepdim=True)
        perp_ph_p2_4           = delta_ph_p2_4 - along_ph_p2_4 * axis_t_world_p2_4
        peg_to_hole_perp_p2_4  = torch.norm(perp_ph_p2_4, dim=-1)

        # ---- A1: xy align latch ----
        xy_aligned_p2_4 = peg_to_hole_perp_p2_4 < align_perp_tol_p2_4
        xy_align_frames_p2_4 = torch.where(
            xy_aligned_p2_4,
            xy_align_frames_p2_4 + 1,
            torch.zeros_like(xy_align_frames_p2_4),
        )
        xy_aligned_latched_p2_4 = xy_aligned_latched_p2_4 | (
            xy_align_frames_p2_4 >= dwell_xy_align_p2_4
        )

        # ---- A2: near hover above hole ----
        d_above_p2_4     = drive_world_p2_4 - above_target_p2_4
        along_above_p2_4 = torch.abs((d_above_p2_4 * axis_t_world_p2_4).sum(-1))
        near_hover_p2_4 = (
            xy_aligned_latched_p2_4
            & (peg_to_hole_perp_p2_4 < align_perp_tol_p2_4)
            & (along_above_p2_4 < along_tol_p2_4)
        )
        near_hover_frames_p2_4 = torch.where(
            near_hover_p2_4,
            near_hover_frames_p2_4 + 1,
            torch.zeros_like(near_hover_frames_p2_4),
        )
        press_down_latched_p2_4 = press_down_latched_p2_4 | (
            near_hover_frames_p2_4 >= dwell_to_press_p2_4
        )

        # ---- Write back ----
        self._xy_align_frames_p2_4    = xy_align_frames_p2_4
        self._xy_aligned_latched_p2_4 = xy_aligned_latched_p2_4
        self._near_hover_frames_p2_4  = near_hover_frames_p2_4
        self._press_down_latched_p2_4 = press_down_latched_p2_4

        # ---- Phase split ----
        in_phase_A1_p2_4 = ~xy_aligned_latched_p2_4
        in_phase_A2_p2_4 = xy_aligned_latched_p2_4 & (~press_down_latched_p2_4)
        in_phase_B_p2_4  = press_down_latched_p2_4

        target_p2_4 = torch.where(
            in_phase_B_p2_4.unsqueeze(-1),
            press_down_target_p2_4,
            above_target_p2_4,
        )

        # ---- Raw pos action ----
        delta_p2_4      = target_p2_4 - drive_world_p2_4
        pos_action_p2_4 = delta_p2_4 / self.pos_threshold
        pos_action_p2_4 = torch.clamp(pos_action_p2_4, -1.0, 1.0)

        # A1: zero out along-axis component so we don't drop until xy locked.
        pa_along_signed_p2_4 = (pos_action_p2_4 * axis_t_world_p2_4).sum(-1, keepdim=True)
        pa_perp_p2_4 = pos_action_p2_4 - pa_along_signed_p2_4 * axis_t_world_p2_4
        pos_action_p2_4 = torch.where(
            in_phase_A1_p2_4.unsqueeze(-1), pa_perp_p2_4, pos_action_p2_4
        )

        # Slower phase B to avoid bounce.
        scale_p2_4 = torch.where(
            in_phase_B_p2_4.unsqueeze(-1),
            torch.full_like(pos_action_p2_4, press_scale_p2_4),
            torch.full_like(pos_action_p2_4, descent_scale_p2_4),
        )
        pos_action_p2_4 = pos_action_p2_4 * scale_p2_4

        # ---- Stall detection + perp-plane wobble during phase B ----
        along_now_p2_4 = (drive_world_p2_4 * axis_t_world_p2_4).sum(-1)
        progress_p2_4 = torch.abs(along_now_p2_4 - self._prev_along_p2_4)
        stalled_now_p2_4 = (progress_p2_4 < stall_threshold_m_p2_4) & in_phase_B_p2_4
        self._stall_frames_p2_4 = torch.where(
            stalled_now_p2_4,
            self._stall_frames_p2_4 + 1,
            torch.zeros_like(self._stall_frames_p2_4),
        )
        self._prev_along_p2_4 = along_now_p2_4.clone()
        wobble_active_p2_4 = self._stall_frames_p2_4 >= stall_dwell_frames_p2_4

        if bool(wobble_active_p2_4.any().item()):
            self._wobble_step_p2_4 += 1

        # Build unit perp basis vector e1 ⟂ axis_t (axis_t = world +z, so
        # any horizontal vector works; pick world +x).
        e1_world_p2_4 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p2_4)

        import math as _math
        phase_rad_p2_4 = (
            2.0 * _math.pi * (self._wobble_step_p2_4 / wobble_period_frames_p2_4)
        )
        wobble_offset_m_p2_4 = wobble_amplitude_p2_4 * _math.sin(phase_rad_p2_4)
        wobble_norm_p2_4 = wobble_offset_m_p2_4 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p2_4 = e1_world_p2_4 * wobble_norm_p2_4
        pos_action_p2_4 = torch.where(
            wobble_active_p2_4.unsqueeze(-1),
            pos_action_p2_4 + wobble_vec_p2_4,
            pos_action_p2_4,
        )
        pos_action_p2_4 = torch.clamp(pos_action_p2_4, -1.0, 1.0)

        # ---- Diagnostic print ----
        green = drive_world_p2_4[0].tolist()
        red   = hole_world_p2_4[0].tolist()
        phase = "A1-xy" if bool(in_phase_A1_p2_4[0]) else (
            "A2-hover" if bool(in_phase_A2_p2_4[0]) else "B-press"
        )
        gr_diff_p2_4 = drive_world_p2_4[0] - hole_world_p2_4[0]
        gr_along_p2_4 = (gr_diff_p2_4 * axis_t_world_p2_4[0]).sum()
        gr_perp_vec_p2_4 = gr_diff_p2_4 - gr_along_p2_4 * axis_t_world_p2_4[0]
        center_dx_p2_4 = float(gr_perp_vec_p2_4[0])
        center_dy_p2_4 = float(gr_perp_vec_p2_4[1])
        print(
            f"[p2_4] phase={phase}  "
            f"peg=({green[0]:+.3f},{green[1]:+.3f},{green[2]:+.3f})  "
            f"hole=({red[0]:+.3f},{red[1]:+.3f},{red[2]:+.3f})  "
            f"dx={center_dx_p2_4*1000:+.2f}mm dy={center_dy_p2_4*1000:+.2f}mm  "
            f"perp={peg_to_hole_perp_p2_4[0]*1000:.2f}mm "
            f"along_hover={along_above_p2_4[0]*1000:.2f}mm  "
            f"xy_latch={int(xy_aligned_latched_p2_4[0])} "
            f"press_latch={int(press_down_latched_p2_4[0])}  "
            f"stall={int(self._stall_frames_p2_4[0])} "
            f"wobble={'YES' if bool(wobble_active_p2_4[0]) else 'no'}"
        )

        rot_action_p2_4     = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action_p2_4 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p2_4, rot_action_p2_4, gripper_action_p2_4], dim=-1)

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        # self._visualize_markers()
        if self.cfg_task.task_idx == 1:
            self._visualize_centerlines_p2_1()
        elif self.cfg_task.task_idx == 2:
            self._visualize_body_and_fixed_p2_2()
        elif self.cfg_task.task_idx == 3:
            self._visualize_propeller_and_tail_p2_3()
        elif self.cfg_task.task_idx == 4:
            self._visualize_holder_and_hole_p2_4()
        self._check_attach_condition()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
        # Each idx has its own scripted policy; remaining idx fall back to RL.
        if self.cfg_task.task_idx == 1:
            self.actions = self._scripted_action_tail_insert_p2_1()
        elif self.cfg_task.task_idx == 2:
            self.actions = self._scripted_action_body_insert_p2_2()
        elif self.cfg_task.task_idx == 3:
            self.actions = self._scripted_action_propeller_insert_p2_3()
        elif self.cfg_task.task_idx == 4:
            self.actions = self._scripted_action_holder_insert_p2_4()
        else:
            self.actions = action.clone()

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 7), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # print("current actions:", self.actions)
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]

        # Interpret actions as target gripper DOF velocity
        gripper_actions = self.actions[:, 6] #

        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.04 if gripper_actions < 0.0 else 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        # self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        z_disp = self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "plane_assembly":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            is_rotated = self.curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = self._update_rew_buf(curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = {}

        # Keypoint rewards.
        def squashing_fn(x, a, b):
            return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        rew_dict["kp_baseline"] = squashing_fn(self.keypoint_dist, a0, b0)
        # a1, b1 = 25, 2
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        rew_dict["kp_coarse"] = squashing_fn(self.keypoint_dist, a1, b1)
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # a2, b2 = 300, 0
        rew_dict["kp_fine"] = squashing_fn(self.keypoint_dist, a2, b2)

        # Action penalties.
        rew_dict["action_penalty"] = torch.norm(self.actions, p=2)
        rew_dict["action_grad_penalty"] = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        rew_dict["curr_engaged"] = (
            self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False).clone().float()
        )
        rew_dict["curr_successes"] = curr_successes.clone().float()

        rew_buf = (
            rew_dict["kp_coarse"]
            + rew_dict["kp_baseline"]
            + rew_dict["kp_fine"]
            - rew_dict["action_penalty"] * self.cfg_task.action_penalty_scale
            - rew_dict["action_grad_penalty"] * self.cfg_task.action_grad_penalty_scale
            + rew_dict["curr_engaged"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _remove_fixed_joint(self):
        if self.fixed_joint_prim and self.fixed_joint_prim.IsValid():
            stage = self.fixed_joint_prim.GetStage()
            stage.RemovePrim(self.fixed_joint_prim.GetPath())
            self.joint_created = False
            self.fixed_joint_prim = None
            print("Removed fixed joint.")

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)
        print("Resetting envs:", env_ids)
        self._remove_fixed_joint()
        # Reset plane2 idx=1 scripted-policy state.
        self._xy_align_frames_p2_1.zero_()
        self._xy_aligned_latched_p2_1.zero_()
        self._near_hover_frames_p2_1.zero_()
        self._press_down_latched_p2_1.zero_()
        self._prev_along_p2_1.zero_()
        self._stall_frames_p2_1.zero_()
        self._wobble_step_p2_1 = 0
        # Reset plane2 idx=2 scripted-policy state (independent of idx=1).
        self._xy_align_frames_p2_2.zero_()
        self._xy_aligned_latched_p2_2.zero_()
        self._near_hover_frames_p2_2.zero_()
        self._press_down_latched_p2_2.zero_()
        self._prev_along_p2_2.zero_()
        self._stall_frames_p2_2.zero_()
        self._wobble_step_p2_2 = 0

        self._xy_align_frames_p2_3.zero_()
        self._xy_aligned_latched_p2_3.zero_()
        self._near_hover_frames_p2_3.zero_()
        self._press_down_latched_p2_3.zero_()
        self._prev_along_p2_3.zero_()
        self._stall_frames_p2_3.zero_()
        self._wobble_step_p2_3 = 0

        self._xy_align_frames_p2_4.zero_()
        self._xy_aligned_latched_p2_4.zero_()
        self._near_hover_frames_p2_4.zero_()
        self._press_down_latched_p2_4.zero_()
        self._prev_along_p2_4.zero_()
        self._stall_frames_p2_4.zero_()
        self._wobble_step_p2_4 = 0
        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()
        self.randomize_initial_state(env_ids)


    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name == "plane_assembly" and self.cfg_task.task_idx in [1]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.
            held_asset_relative_pos[:, 1] += 0.0
            held_asset_relative_pos[:, 0] += 0.02

        elif self.cfg_task.name == "plane_assembly" and self.cfg_task.task_idx in [2]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.
            held_asset_relative_pos[:, 1] += 0.0
            held_asset_relative_pos[:, 0] -= 0.01

        elif self.cfg_task.name == "plane_assembly" and self.cfg_task.task_idx in [3]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.
            held_asset_relative_pos[:, 1] += 0.0
            held_asset_relative_pos[:, 0] += 0.0

        elif self.cfg_task.name == "plane_assembly" and self.cfg_task.task_idx in [4]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.
            held_asset_relative_pos[:, 1] += 0.0
            held_asset_relative_pos[:, 0] -= 0.01
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = self.identity_quat

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)


    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""

        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 0.0 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids] 
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        # self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        # self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()


        if self.cfg_task.task_idx in [1]:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] -= 0.2
            rela_trans[:, 1] += 0.
            rela_trans[:, 0] -= 0.

        if self.cfg_task.task_idx in [2]:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] -= 0.25
            rela_trans[:, 1] += 0.
            rela_trans[:, 0] -= 0.

        if self.cfg_task.task_idx in [3]:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] -= 0.25
            rela_trans[:, 1] += 0.
            rela_trans[:, 0] -= 0.

        if self.cfg_task.task_idx in [4]:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] -= 0.25
            rela_trans[:, 1] += 0.
            rela_trans[:, 0] -= 0.



        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5) + 0.5  # [-1, 1] # [-0.5, 1.5]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            rela_trans[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 0.01 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            self.hand_down_euler[bad_envs, ...] = hand_down_euler
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )
            self.marker_locations[bad_envs, ...] = rela_trans[bad_envs, ...]
            self.marker_orientations[bad_envs, ...] = hand_down_quat[bad_envs, :]
            # (c) iterative IK Method
            self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = rela_trans[bad_envs, ...]
            self.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

            pos_error, aa_error = self.set_pos_inverse_kinematics(env_ids=bad_envs)
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1
            if ik_attempt >= 10:
                print("IK failed to find solution, resetting to default pose.")
                break

        self.step_sim_no_action()

        if self.cfg_task.task_idx == 2:
            self._create_fixed_joint(connection_idx=1)
        elif self.cfg_task.task_idx == 3:
            self._create_fixed_joint(connection_idx=1)
            self._create_fixed_joint(connection_idx=2)
        elif self.cfg_task.task_idx == 4:
            self._create_fixed_joint(connection_idx=1)
            self._create_fixed_joint(connection_idx=2)
            self._create_fixed_joint(connection_idx=3)


        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros_like(self.fingertip_midpoint_pos),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.held_asset_pos_noise = 0.00 * (rand_sample - 0.5)  # [-1, 1]

        held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=self.identity_quat,
            t2=self.held_asset_pos_noise,
        )

        if self.cfg_task.task_idx == 2:
            rot_euler = torch.tensor([0.0, -1.5707, 1.5707], device=self.device).repeat(
            self.num_envs, 1
            )
            translated_held_asset_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        if self.cfg_task.task_idx == 3:
            rot_euler = torch.tensor([1.5707, 0, 0], device=self.device).repeat(
            self.num_envs, 1
            )
            translated_held_asset_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        if self.cfg_task.task_idx == 4:
            rot_euler = torch.tensor([0, 1.5707, 0], device=self.device).repeat(
            self.num_envs, 1
            )
            translated_held_asset_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 1:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        # Back out what actions should be for initial state.
        # Relative position to bolt tip.
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        pos_actions = self.fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        self.step_sim_no_action()