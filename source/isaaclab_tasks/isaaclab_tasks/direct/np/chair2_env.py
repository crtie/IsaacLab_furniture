# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import sys, os
sys.path.append(os.path.abspath(__file__))

import math

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .np_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FrankaChair2Cfg
from .chair_tasks_cfg import ChairAssembly1, ConnectionCfg
from pdb import set_trace as bp
from .np_utils.group_utils import SE3dist
from scipy.spatial.transform import Rotation as R
import torch
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf, Gf, Tf
from omni.physx.scripts import utils
import omni.usd

class FrankaChair2Env(DirectRLEnv):
    cfg: FrankaChair2Cfg

    def __init__(self, cfg: FrankaChair2Cfg, render_mode: str | None = None, **kwargs):
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

        # Make the chair-frame surface frictionless so the plug can slide
        # freely while the spiral searches for the hole.
        if self.cfg_task.task_idx in (1, 2):
            self._set_friction(self._fixed_asset, 0.0)

        # idx=3 (rod → 2 seated plugs): make the 2 already-seated plugs
        # frictionless so the rod can slide onto them without snagging.
        if self.cfg_task.task_idx == 3:
            for plug_attr in ("_plug1", "_plug2"):
                plug = getattr(self, plug_attr, None)
                if plug is not None and plug is not self._held_asset:
                    self._set_friction(plug, 0.0)

        # idx=4/5 (plug → rod top hole): the rod is a scene fixture that the
        # held plug needs to slide into — set its friction to 0 so the peg
        # doesn't snag on the hole rim during descent.
        if (
            self.cfg_task.task_idx in (4, 5)
            and getattr(self, "_rod_asset", None) is not None
            and self._rod_asset is not self._held_asset
        ):
            self._set_friction(self._rod_asset, 0.0)

        # Any other seated plug (idx 2, 5) in the scene is also frictionless.
        for plug_attr in ("_plug1", "_plug2"):
            plug = getattr(self, plug_attr, None)
            if plug is not None and plug is not self._held_asset:
                self._set_friction(plug, 0.0)

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

        if self.cfg_task.name == "chair_assembly":
            self.fixed_success_pos_local[:, 2] = 0.0
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Scripted-policy state machine (idx=1/2 plug→hole insertion).
        self._near_hole_frames = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._search_active_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._search_step = 0
        self._press_down_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Extra state for idx=3 (rod → frame): 4-phase machine with descent-stall recovery.
        self._descend_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_held_z = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._stuck_frames = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._search_center_perp = torch.zeros((self.num_envs, 3), device=self.device)

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

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)

        if self.cfg_task.task_idx == 1:
            self._plug1 = RigidObject(self.cfg_task.plug1)
            self._held_asset = self._plug1
            self._connection_cfg = self.cfg_task.connection_cfg1

        if self.cfg_task.task_idx ==2:
            self._plug1 = RigidObject(self.cfg_task.plug1)
            self._plug2 = RigidObject(self.cfg_task.plug2)
            self._held_asset = self._plug2
            self._connection_cfg = self.cfg_task.connection_cfg2
        
        if self.cfg_task.task_idx == 3:
            self._plug1 = RigidObject(self.cfg_task.plug1)
            self._plug2 = RigidObject(self.cfg_task.plug2)
            self._rod_asset = RigidObject(self.cfg_task.rod)
            self._held_asset = self._rod_asset
            self._connection_cfg = self.cfg_task.connection_cfg3
            
        if self.cfg_task.task_idx == 4:
            self._plug1 = RigidObject(self.cfg_task.plug1)
            self._held_asset = self._plug1
            self._rod_asset = RigidObject(self.cfg_task.rod)
            self._connection_cfg = self.cfg_task.connection_cfg4

        if self.cfg_task.task_idx == 5:
            self._plug1 = RigidObject(self.cfg_task.plug1)
            self._plug2 = RigidObject(self.cfg_task.plug2)
            self._held_asset = self._plug2
            self._rod_asset = RigidObject(self.cfg_task.rod)
            self._connection_cfg = self.cfg_task.connection_cfg5
        

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
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Plug1")
            joint_path = "/World/envs/env_0/FixedJoint1"
            connection_cfg = self.cfg_task.connection_cfg1
        elif connection_idx == 2:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Plug2")
            joint_path = "/World/envs/env_0/FixedJoint2"
            connection_cfg = self.cfg_task.connection_cfg2
        elif connection_idx == 3:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Rod")
            joint_path = "/World/envs/env_0/FixedJoint3"
            connection_cfg = self.cfg_task.connection_cfg3
        elif connection_idx == 4:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Plug1")
            joint_path = "/World/envs/env_0/FixedJoint4"
            connection_cfg = self.cfg_task.connection_cfg4
        elif connection_idx == 5:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Plug2")
            joint_path = "/World/envs/env_0/FixedJoint5"
            connection_cfg = self.cfg_task.connection_cfg5


        fixed_prim = stage.GetPrimAtPath("/World/envs/env_0/FixedAsset")
        

        to_path = held_prim.GetPath()
        from_path = fixed_prim.GetPath()
        # rel_mat1 = self._get_real_mat()
        rel_mat = connection_cfg.pose_to_base
        # rel_mat = np.eye(4, dtype=np.float32)
        # rel_mat[:3, :3] = rel_mat1[:3, :3]
        # rel_mat[:3, 3] = rel_mat2[:3, 3]
        pos1 = Gf.Vec3f([float(rel_mat[0, 3]), float(rel_mat[1, 3]), float(rel_mat[2, 3])])
        rot1q = torch_utils.rot_matrices_to_quats(torch.tensor(rel_mat[:3, :3]))
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
        if not self.joint_created and R_dist < 0.1 and t_tangent < 0.003 and t_normal < 0.01:
            self._create_fixed_joint(connection_idx=self.cfg_task.task_idx)
            self.joint_created = True
            rel_mat = self._get_real_mat()
            gt_real_mat = self._connection_cfg.pose_to_base
            R_dist, R_axis, t_tangent, t_normal = SE3dist(rel_mat, gt_real_mat, self._connection_cfg)
            self.R_axis = R_axis 
            print("Creating fixed joint.")


        elif self.joint_created :
            print("Fixed joint already created.")
        else:
            print("Not creating fixed joint yet, waiting for conditions to be met.")

    def _sync_held_asset(self):
        # 1. 获取当前相对位姿和目标相对位姿
        rel_mat = self._get_real_mat()  # 当前 held 相对 fixed 的4x4矩阵
        gt_real_mat = self._connection_cfg.pose_to_base  # 目标相对位姿

        R_dist, R_axis, t_tangent, t_normal = SE3dist(rel_mat, gt_real_mat, self._connection_cfg)
        delta_theta = R_axis - self.R_axis  # 计算旋转轴的变化量

        # 5. 根据螺距 pitch 计算z方向的位移
        pitch = getattr(self._connection_cfg, "pitch", 0.5)  # 螺距，单位：米/弧度
        dz = float(delta_theta * pitch)  # 螺旋升降量
        print("dz:", dz)
        if abs(dz) >0.1:
            print("triggering joint limit1")
            prim = self.fixed_joint_prim
            limit_api = UsdPhysics.LimitAPI.Apply(prim, "transZ")
            limit_api.CreateLowAttr(-0.005)
            limit_api.CreateHighAttr(0.005)
        if abs(dz) > 0.15:
            print("triggering joint limit2")
            prim = self.fixed_joint_prim
            limit_api = UsdPhysics.LimitAPI.Apply(prim, "transZ")
            limit_api.CreateLowAttr(-0.01)
            limit_api.CreateHighAttr(0.01)

    def _visualize_tip_and_hole(self, peg_tip_world, hole_pos_world):
        """Visualize peg tip (red), plug origin (blue), hole (green), delta (yellow)."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()

        env_origin = self.scene.env_origins[0]

        def to_tuple(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        tip_t = to_tuple(peg_tip_world[0])
        hole_t = to_tuple(hole_pos_world[0])
        held_t = to_tuple(self.held_pos[0])

        self._dbg_draw.draw_points(
            [tip_t, hole_t, held_t],
            [
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.2, 0.5, 1.0, 1.0),
            ],
            [25.0, 25.0, 14.0],
        )
        self._dbg_draw.draw_lines(
            [tip_t],
            [hole_t],
            [(1.0, 1.0, 0.0, 1.0)],
            [3.0],
        )

    def _compute_scripted_action(self):
        """Router — dispatch per task_idx. Returns None for task_idx with no
        scripted policy so the caller falls back to the RL action."""
        if self.cfg_task.task_idx in (1, 2):
            return self._scripted_action_plug_to_hole()
        if self.cfg_task.task_idx == 3:
            return self._scripted_action_rod_to_frame()
        if self.cfg_task.task_idx in (4, 5):
            return self._scripted_action_plug_to_rod()
        return None

    def _scripted_action_plug_to_hole(self):
        """Plug → chair-frame hole (idx 1/2) — offset + spiral search (mirrors
        chair1_env.py's idx=4/5 plug→backrest policy structure).

        Flow: approach an offset 5 mm along the two-hole line → dwell →
        spiral outward → PEG TIP (red) passes over real hole → press_down →
        joint_created.
        """
        # ==== Tunables ====
        head_protrusion = 0.0094
        plug_usd_length = 0.029
        plug_scale = float(self.cfg_task.plug1.spawn.scale[2])
        plug_axis_center_local = 0.004 * plug_scale  # plug bbox corner → cylinder axis
        guess_offset_magnitude = 0.003     # 5 mm along the two-hole line
        spiral_angle_step = math.pi / 12   # 15°/step (24 per turn)
        spiral_radial_step = 0.0001        # 0.1 mm/step (2.4 mm/turn)
        press_detect_radius = 0.0015       # 1.5 mm trigger threshold (perp plane)
        spiral_idx_max = 300
        descent_scale = 0.3                # slow descent factor (entire trajectory)
        search_press_magnitude = 0.1       # small downward action during spiral to keep peg on surface

        # ==== Peg lower tip in world (cylinder center) ====
        tip_offset_local = torch.zeros((self.num_envs, 3), device=self.device)
        tip_offset_local[:, 0] = plug_axis_center_local
        tip_offset_local[:, 1] = plug_axis_center_local
        tip_offset_local[:, 2] = -plug_usd_length * plug_scale
        _, peg_tip_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, tip_offset_local
        )

        # ==== Hole center in world (corrected for plug bbox-corner origin) ====
        pose_to_base = torch.as_tensor(
            self._connection_cfg.pose_to_base, dtype=torch.float32, device=self.device
        )
        cfg_R = pose_to_base[:3, :3]
        cfg_t = pose_to_base[:3, 3]
        axis_center_chair = cfg_R @ torch.tensor(
            [plug_axis_center_local, plug_axis_center_local, 0.0],
            dtype=torch.float32, device=self.device,
        )
        hole_pos_local = (cfg_t + axis_center_chair).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_pos_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_pos_local
        )
        axis_t_local = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t = torch.zeros_like(self.fixed_pos)
        _, axis_t_world = torch_utils.tf_combine(
            self.fixed_quat, zero_t, self.identity_quat, axis_t_local
        )
        hole_opening_world = hole_pos_world - axis_t_world * head_protrusion
        press_down_target = hole_pos_world - axis_t_world * (plug_usd_length * plug_scale)

        # ==== Perp-plane basis for the spiral ====
        ref = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis_t_local)
        parallel = (axis_t_local * ref).sum(-1, keepdim=True).abs() > 0.9
        ref = torch.where(
            parallel,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(axis_t_local),
            ref,
        )
        e1_local = torch.cross(axis_t_local, ref, dim=-1)
        e1_local = e1_local / torch.norm(e1_local, dim=-1, keepdim=True).clamp(min=1e-6)
        e2_local = torch.cross(axis_t_local, e1_local, dim=-1)
        e2_local = e2_local / torch.norm(e2_local, dim=-1, keepdim=True).clamp(min=1e-6)
        _, e1_world = torch_utils.tf_combine(self.fixed_quat, zero_t, self.identity_quat, e1_local)
        _, e2_world = torch_utils.tf_combine(self.fixed_quat, zero_t, self.identity_quat, e2_local)

        # ==== Offset start point — 5 mm along the line to the OTHER hole ====
        if self.cfg_task.task_idx == 1:
            other_pose_np = self.cfg_task.connection_cfg2.pose_to_base
        else:
            other_pose_np = self.cfg_task.connection_cfg1.pose_to_base
        other_hole_local = torch.as_tensor(
            other_pose_np[:3, 3], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, other_hole_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, other_hole_local
        )
        other_hole_opening_world = other_hole_world - axis_t_world * head_protrusion
        line_dir = other_hole_opening_world - hole_opening_world
        line_dir = line_dir / torch.norm(line_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        offset_center_world = hole_opening_world + line_dir * guess_offset_magnitude

        # ==== Phase transitions ====
        # Phase 1 (approach):  ~_xy_aligned                          → xy motion to offset_center
        # Phase 2 (descent):    _xy_aligned & ~_search_active        → z descent to surface
        # Phase 3 (search):     _search_active & ~_press_down        → xy spiral
        # Phase 4 (press_down): _press_down                          → z descent into hole

        # Peg position relative to offset_center, split into perp (xy) and along (z) components.
        delta_to_offset = offset_center_world - peg_tip_world
        along_to_offset = (delta_to_offset * axis_t_world).sum(-1, keepdim=True)
        perp_to_offset = delta_to_offset - along_to_offset * axis_t_world
        perp_dist_to_offset = torch.norm(perp_to_offset, dim=-1)
        along_dist_to_offset = along_to_offset.squeeze(-1).abs()

        # Phase 1 -> 2: rough xy alignment.
        rough_xy_tol = 0.01   # 1 cm "差不多" tolerance
        z_tol = 0.003         # 3 mm to declare "at surface"
        self._xy_aligned_latched = self._xy_aligned_latched | (
            perp_dist_to_offset < rough_xy_tol
        )

        # Phase 2 -> 3: peg has descended to the surface (z near offset_center.z).
        self._search_active_latched = self._search_active_latched | (
            self._xy_aligned_latched & (along_dist_to_offset < z_tol)
        )

        # Spiral step counter (Phase 3).
        in_search_phase = self._search_active_latched & (~self._press_down_latched)
        if bool(in_search_phase.any().item()):
            self._search_step += 1

        k = min(self._search_step, spiral_idx_max)
        angle = k * spiral_angle_step
        radius = k * spiral_radial_step
        sx = radius * math.cos(angle)
        sy = radius * math.sin(angle)
        search_target = offset_center_world + sx * e1_world + sy * e2_world

        # Phase 3 -> 4: PEG TIP (red) actually over the hole center.
        delta_pt = peg_tip_world - hole_opening_world
        along_pt = (delta_pt * axis_t_world).sum(-1, keepdim=True)
        perp_pt = delta_pt - along_pt * axis_t_world
        peg_to_hole_perp = torch.norm(perp_pt, dim=-1)
        self._press_down_latched = self._press_down_latched | (
            in_search_phase & (peg_to_hole_perp < press_detect_radius)
        )

        # ==== Target select ====
        # Phase 1 + 2 share the offset_center target (filter will pick xy or z).
        # Phase 3 uses spiral target. Phase 4 uses press_down target.
        target = torch.where(
            self._press_down_latched.unsqueeze(-1),
            press_down_target,
            torch.where(
                in_search_phase.unsqueeze(-1),
                search_target,
                offset_center_world,
            ),
        )

        # ==== Visualization (red = peg tip, green = real hole center) ====
        self._visualize_tip_and_hole(peg_tip_world, hole_opening_world)

        # ==== Compute action ====
        delta = target - peg_tip_world
        pos_action = delta / self.pos_threshold
        pos_action = torch.clamp(pos_action, -1.0, 1.0)
        pos_action = pos_action * descent_scale

        # Per-phase axis filter:
        # Phase 1 (approach)  → perp only (xy align)
        # Phase 2 (descent)   → along only (drop to surface)
        # Phase 3 (search)    → perp + small constant press to keep on surface
        # Phase 4 (press)     → along only (descend into hole)
        along_action = (pos_action * axis_t_world).sum(-1, keepdim=True) * axis_t_world
        perp_action = pos_action - along_action

        in_descent_phase = self._xy_aligned_latched & (~self._search_active_latched)
        in_search_phase_for_press = self._search_active_latched & (~self._press_down_latched)
        use_along = in_descent_phase | self._press_down_latched

        # Constant small downward press during spiral to keep the peg pressed
        # against the surface (prevents "lift-off" from contact dynamics).
        search_press_action = (-axis_t_world) * search_press_magnitude
        search_press_action = torch.where(
            in_search_phase_for_press.unsqueeze(-1),
            search_press_action,
            torch.zeros_like(search_press_action),
        )

        pos_action = torch.where(
            use_along.unsqueeze(-1),
            along_action,
            perp_action + search_press_action,
        )

        if self.joint_created:
            pos_action = torch.zeros_like(pos_action)

        rot_action = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action, rot_action, gripper_action], dim=-1)

    def _scripted_action_rod_to_frame(self):
        """Insert the rod's 2 lower holes onto the 2 already-seated plugs
        (task_idx == 3). Mirror of chair1_env.py's _scripted_action_backrest_to_frame,
        but with the rod's geometry (instead of the backrest).

        Rod local hole positions are derived by transforming
        `connection_cfg1.translation` and `connection_cfg2.translation` (plug
        seated positions in chair-frame local) through `connection_cfg3.pose_to_base`
        (rod seated pose in chair-frame local). With the plug-bbox-corner
        correction:
          plug1 axis → rod local (0.1916, 0.00955, -0.009)
          plug2 axis → rod local (0.1916, 0.00955, -0.032)
        Following backrest convention, x is flipped to 0 (the visible end of
        the rod — the user-confirmed orientation). Flip back to 0.1916 if the
        markers show up on the wrong end of the rod.
        """
        # ==== Rod's 2 hole openings in rod local frame ====
        # x=0.1916 is the rod end where the seated plugs poke in (per cfg3
        # projection math). Flip to 0.0 if the markers appear on the wrong end.
        hole1_local = torch.tensor([0.1916, 0.00955, -0.009], device=self.device)
        hole2_local = torch.tensor([0.1916, 0.00955, -0.032], device=self.device)
        hole1_local = hole1_local.unsqueeze(0).repeat(self.num_envs, 1)
        hole2_local = hole2_local.unsqueeze(0).repeat(self.num_envs, 1)

        _, hole1_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hole1_local
        )
        _, hole2_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hole2_local
        )
        hole_mid_world = (hole1_world + hole2_world) / 2.0

        # ==== The 2 plug-top centers in world ====
        # Apply cfg.pose_to_base to plug_local_center (cylinder axis on the
        # plug's top face). connection_cfg.translation is plug's USD origin
        # (= bbox corner), not the cylinder axis.
        plug_scale = float(self.cfg_task.plug1.spawn.scale[0])
        plug_top_center_local = 0.004 * plug_scale  # 3 mm in plug local

        def _peg_top_center_chair_local(cfg_pose_to_base_np):
            R = cfg_pose_to_base_np[:3, :3]
            t = cfg_pose_to_base_np[:3, 3]
            offset_chair = R @ np.array([plug_top_center_local, plug_top_center_local, 0.0])
            return t + offset_chair

        peg1_chair_np = _peg_top_center_chair_local(self.cfg_task.connection_cfg1.pose_to_base)
        peg2_chair_np = _peg_top_center_chair_local(self.cfg_task.connection_cfg2.pose_to_base)

        peg1_chair_local = torch.as_tensor(
            peg1_chair_np, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg2_chair_local = torch.as_tensor(
            peg2_chair_np, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg1_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg1_chair_local
        )
        _, peg2_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg2_chair_local
        )
        peg_mid_world = (peg1_world + peg2_world) / 2.0

        # ==== Insertion axis in world ====
        axis_t_local = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t = torch.zeros_like(self.fixed_pos)
        _, axis_t_world = torch_utils.tf_combine(
            self.fixed_quat, zero_t, self.identity_quat, axis_t_local
        )

        # ==== Orthonormal perp-plane basis (for spiral) ====
        ref = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis_t_local)
        parallel = (axis_t_local * ref).sum(-1, keepdim=True).abs() > 0.9
        ref = torch.where(
            parallel,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(axis_t_local),
            ref,
        )
        e1_local = torch.cross(axis_t_local, ref, dim=-1)
        e1_local = e1_local / torch.norm(e1_local, dim=-1, keepdim=True).clamp(min=1e-6)
        e2_local = torch.cross(axis_t_local, e1_local, dim=-1)
        e2_local = e2_local / torch.norm(e2_local, dim=-1, keepdim=True).clamp(min=1e-6)
        _, e1_world = torch_utils.tf_combine(self.fixed_quat, zero_t, self.identity_quat, e1_local)
        _, e2_world = torch_utils.tf_combine(self.fixed_quat, zero_t, self.identity_quat, e2_local)

        # ==== Four-phase targets ====
        # Phase 1 hover:   peg_mid + axis_t * descent_clearance + perp_offset
        # Phase 2 descend: peg_mid + perp_offset (at peg level)
        # Phase 3 search:  peg_mid + stalled_perp + spiral_offset
        # Phase 4 press:   peg_mid - axis_t * seating_depth
        descent_clearance = 0.025
        rod_seating_depth = 0.0084          # plug head protrudes 8.4 mm into rod (tune if rod thickness differs)
        guess_offset_magnitude = 0.003      # 5 mm perp offset (forces a stall→search)
        offset_perp_world = e1_world * guess_offset_magnitude

        hover_target = peg_mid_world + axis_t_world * descent_clearance + offset_perp_world
        descend_target = peg_mid_world + offset_perp_world
        press_down_target = peg_mid_world - axis_t_world * rod_seating_depth

        # ==== Trigger parameters ====
        near_radius = 0.005
        dwell_to_descend = 5
        z_progress_eps = 1e-4
        stall_frames_thresh = 10
        spiral_angle_step = math.pi / 12
        spiral_radial_step = 0.0001
        press_detect_radius = 0.0015
        spiral_idx_max = 300

        # Phase 1 -> 2: hole_mid near hover_target for dwell frames.
        mid_to_hover = torch.norm(hover_target - hole_mid_world, dim=-1)
        near_hover = mid_to_hover < near_radius
        self._near_hole_frames = torch.where(
            near_hover,
            self._near_hole_frames + 1,
            torch.zeros_like(self._near_hole_frames),
        )
        self._descend_latched = self._descend_latched | (
            self._near_hole_frames >= dwell_to_descend
        )

        # Stall detection (during descend only).
        curr_along = (hole_mid_world * axis_t_world).sum(-1)
        descending = (
            self._descend_latched
            & (~self._search_active_latched)
            & (~self._press_down_latched)
        )
        along_drop = self._prev_held_z - curr_along
        prev_valid = ~torch.isnan(self._prev_held_z)
        stalled = descending & prev_valid & (along_drop < z_progress_eps)
        self._stuck_frames = torch.where(
            stalled,
            self._stuck_frames + 1,
            torch.zeros_like(self._stuck_frames),
        )
        self._prev_held_z = torch.where(
            descending, curr_along, torch.full_like(curr_along, float("nan"))
        )

        # Phase 2 -> 3: stall threshold → latch search, snapshot spiral center.
        just_search_latched = (~self._search_active_latched) & (
            self._stuck_frames >= stall_frames_thresh
        )
        if bool(just_search_latched.any().item()):
            delta_now = hole_mid_world - peg_mid_world
            along_now = (delta_now * axis_t_world).sum(-1, keepdim=True)
            perp_now = delta_now - along_now * axis_t_world
            self._search_center_perp = torch.where(
                just_search_latched.unsqueeze(-1),
                perp_now,
                self._search_center_perp,
            )
        self._search_active_latched = self._search_active_latched | (
            self._stuck_frames >= stall_frames_thresh
        )

        # Spiral step counter.
        in_search_phase = self._search_active_latched & (~self._press_down_latched)
        if bool(in_search_phase.any().item()):
            self._search_step += 1

        # Angle keeps growing without bound; radius caps at `spiral_idx_max`
        # so once the spiral reaches max radius it just orbits at that radius
        # indefinitely until red/green markers align.
        radius_k = min(self._search_step, spiral_idx_max)
        angle = self._search_step * spiral_angle_step
        radius = radius_k * spiral_radial_step
        sx = radius * math.cos(angle)
        sy = radius * math.sin(angle)
        spiral_offset_world = sx * e1_world + sy * e2_world

        # Tap-tap rhythm during search: every `tap_period` frames, alternate
        # between LIFTED (a few mm above the surface) and DOWN (on the
        # surface). Helps the long rod avoid dragging while xy spirals.
        tap_period = 30           # frames for a full lift-down cycle
        tap_half = tap_period // 2
        tap_lift_amount = 0.005   # 5 mm lift during the up half-cycle
        tap_phase = self._search_step % tap_period
        tap_lift = tap_lift_amount if tap_phase < tap_half else 0.0
        search_target = (
            peg_mid_world
            + self._search_center_perp
            + spiral_offset_world
            + axis_t_world * tap_lift
        )

        # Phase 3 -> 4: trigger press_down only when the ACTUAL hole_mid (red
        # midpoint, in physics) lines up with peg_mid (green midpoint) in the
        # perp plane. Using the commanded spiral target instead would fire too
        # early (the rod lags the commanded target due to PD tracking).
        delta_pm = hole_mid_world - peg_mid_world
        along_pm = (delta_pm * axis_t_world).sum(-1, keepdim=True)
        perp_pm = delta_pm - along_pm * axis_t_world
        rod_to_peg_perp = torch.norm(perp_pm, dim=-1)
        self._press_down_latched = self._press_down_latched | (
            in_search_phase & (rod_to_peg_perp < press_detect_radius)
        )

        # Four-way target select.
        target = torch.where(
            self._press_down_latched.unsqueeze(-1),
            press_down_target,
            torch.where(
                self._search_active_latched.unsqueeze(-1),
                search_target,
                torch.where(
                    self._descend_latched.unsqueeze(-1),
                    descend_target,
                    hover_target,
                ),
            ),
        )

        # ==== Visualization ====
        self._visualize_rod_alignment(
            hole1_world, hole2_world, peg1_world, peg2_world, hole_mid_world, target
        )

        # ==== Apply delta ====
        delta = target - hole_mid_world
        pos_action = delta / self.pos_threshold
        pos_action = torch.clamp(pos_action, -1.0, 1.0)

        # Slow the initial descent (×0.3 between hover-dwell and stall/search).
        in_descend_phase = (
            self._descend_latched
            & (~self._search_active_latched)
            & (~self._press_down_latched)
        )
        descend_scale = torch.where(
            in_descend_phase.unsqueeze(-1),
            torch.full_like(pos_action, 0.3),
            torch.ones_like(pos_action),
        )
        pos_action = pos_action * descend_scale

        if self.joint_created:
            pos_action = torch.zeros_like(pos_action)

        rot_action = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action, rot_action, gripper_action], dim=-1)

    def _visualize_rod_alignment(self, hole1, hole2, peg1, peg2,
                                  hole_mid, target_mid):
        """Draw the 2 hole-peg pairs (red↔green) plus the mid-to-target arrow."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()

        env_origin = self.scene.env_origins[0]

        def to_tuple(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        h1 = to_tuple(hole1[0])
        h2 = to_tuple(hole2[0])
        g1 = to_tuple(peg1[0])
        g2 = to_tuple(peg2[0])
        m_red = to_tuple(hole_mid[0])
        m_green = to_tuple(target_mid[0])

        self._dbg_draw.draw_points(
            [h1, h2, g1, g2, m_red, m_green],
            [
                (1.0, 0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (1.0, 0.5, 0.0, 1.0),
                (0.0, 1.0, 1.0, 1.0),
            ],
            [22.0, 22.0, 22.0, 22.0, 14.0, 14.0],
        )
        self._dbg_draw.draw_lines(
            [h1, h2, m_red],
            [g1, g2, m_green],
            [
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 0.5, 0.0, 1.0),
            ],
            [3.0, 3.0, 2.0],
        )

    def _scripted_action_plug_to_rod(self):
        """Plug → rod top hole (chair2 idx 4/5) — mirror of chair1_env.py's
        _scripted_action_plug_to_backrest, but with chair2's rod geometry.

        Flow: approach an offset 5 mm along the two-hole line → dwell →
        spiral outward → trigger press_down when the REAL peg tip passes
        over the hole center → plug descends → joint_created.

        Geometry differences:
          - cfg4/5.pose_to_base.y = 0.236, rod top in chair y = 0.229
            → head_protrusion = 7 mm (vs 9.06 mm for chair1 backrest)
          - Same plug + same axis_t direction, so all other math identical.
        """
        # ==== Tunables ====
        head_protrusion = 0.007            # plug origin overshoot above rod top
        plug_usd_length = 0.029
        plug_scale = float(self.cfg_task.plug1.spawn.scale[2])
        plug_axis_center_local = 0.004 * plug_scale  # plug bbox corner → cylinder axis
        guess_offset_magnitude = 0.003     # 5 mm along the two-hole line
        near_radius = 0.005                # dwell radius for offset_center
        dwell_to_search = 5                # frames near offset before search latches
        spiral_angle_step = math.pi / 12   # 15°/step (24 per turn)
        spiral_radial_step = 0.0001        # 0.1 mm/step (2.4 mm/turn)
        press_detect_radius = 0.0015       # 1.5 mm trigger threshold (perp plane)
        spiral_idx_max = 300
        descent_scale = 0.3                # slow descent factor

        # ==== Peg lower tip in world (cylinder center) ====
        tip_offset_local = torch.zeros((self.num_envs, 3), device=self.device)
        tip_offset_local[:, 0] = plug_axis_center_local
        tip_offset_local[:, 1] = plug_axis_center_local
        tip_offset_local[:, 2] = -plug_usd_length * plug_scale
        _, peg_tip_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, tip_offset_local
        )

        # ==== Hole center in world (corrected for plug bbox-corner origin) ====
        pose_to_base = torch.as_tensor(
            self._connection_cfg.pose_to_base, dtype=torch.float32, device=self.device
        )
        cfg_R = pose_to_base[:3, :3]
        cfg_t = pose_to_base[:3, 3]
        axis_center_chair = cfg_R @ torch.tensor(
            [plug_axis_center_local, plug_axis_center_local, 0.0],
            dtype=torch.float32, device=self.device,
        )
        hole_pos_local = (cfg_t + axis_center_chair).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_pos_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_pos_local
        )
        axis_t_local = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t = torch.zeros_like(self.fixed_pos)
        _, axis_t_world = torch_utils.tf_combine(
            self.fixed_quat, zero_t, self.identity_quat, axis_t_local
        )
        hole_opening_world = hole_pos_world - axis_t_world * head_protrusion
        press_down_target = hole_pos_world - axis_t_world * (plug_usd_length * plug_scale)

        # ==== Perp-plane basis for the spiral ====
        ref = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis_t_local)
        parallel = (axis_t_local * ref).sum(-1, keepdim=True).abs() > 0.9
        ref = torch.where(
            parallel,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(axis_t_local),
            ref,
        )
        e1_local = torch.cross(axis_t_local, ref, dim=-1)
        e1_local = e1_local / torch.norm(e1_local, dim=-1, keepdim=True).clamp(min=1e-6)
        e2_local = torch.cross(axis_t_local, e1_local, dim=-1)
        e2_local = e2_local / torch.norm(e2_local, dim=-1, keepdim=True).clamp(min=1e-6)
        _, e1_world = torch_utils.tf_combine(self.fixed_quat, zero_t, self.identity_quat, e1_local)
        _, e2_world = torch_utils.tf_combine(self.fixed_quat, zero_t, self.identity_quat, e2_local)

        # ==== Offset start point — 5 mm along the line to the OTHER rod hole ====
        if self.cfg_task.task_idx == 4:
            other_pose_np = self.cfg_task.connection_cfg5.pose_to_base
        else:
            other_pose_np = self.cfg_task.connection_cfg4.pose_to_base
        other_hole_local = torch.as_tensor(
            other_pose_np[:3, 3], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, other_hole_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, other_hole_local
        )
        other_hole_opening_world = other_hole_world - axis_t_world * head_protrusion
        line_dir = other_hole_opening_world - hole_opening_world
        line_dir = line_dir / torch.norm(line_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        offset_center_world = hole_opening_world + line_dir * guess_offset_magnitude

        # ==== Phase 1 -> 2: tip near offset_center for dwell_to_search frames ====
        tip_to_offset = torch.norm(offset_center_world - peg_tip_world, dim=-1)
        near_offset = tip_to_offset < near_radius
        self._near_hole_frames = torch.where(
            near_offset,
            self._near_hole_frames + 1,
            torch.zeros_like(self._near_hole_frames),
        )
        self._search_active_latched = self._search_active_latched | (
            self._near_hole_frames >= dwell_to_search
        )

        # ==== Spiral step ====
        in_search_phase = self._search_active_latched & (~self._press_down_latched)
        if bool(in_search_phase.any().item()):
            self._search_step += 1

        k = min(self._search_step, spiral_idx_max)
        angle = k * spiral_angle_step
        radius = k * spiral_radial_step
        sx = radius * math.cos(angle)
        sy = radius * math.sin(angle)
        search_target = offset_center_world + sx * e1_world + sy * e2_world

        # ==== Phase 2 -> 3: REAL peg tip (red) over the hole center ====
        # Use real peg position (not commanded target) so press_down fires
        # only after the peg has actually caught up.
        delta_pt = peg_tip_world - hole_opening_world
        along_pt = (delta_pt * axis_t_world).sum(-1, keepdim=True)
        perp_pt = delta_pt - along_pt * axis_t_world
        peg_to_hole_perp = torch.norm(perp_pt, dim=-1)
        self._press_down_latched = self._press_down_latched | (
            in_search_phase & (peg_to_hole_perp < press_detect_radius)
        )

        # ==== Target select ====
        target = torch.where(
            self._press_down_latched.unsqueeze(-1),
            press_down_target,
            torch.where(
                in_search_phase.unsqueeze(-1),
                search_target,
                offset_center_world,
            ),
        )

        # ==== Visualization (red = peg tip, green = real hole center) ====
        self._visualize_tip_and_hole(peg_tip_world, hole_opening_world)

        # ==== Compute action ====
        delta = target - peg_tip_world
        pos_action = delta / self.pos_threshold
        pos_action = torch.clamp(pos_action, -1.0, 1.0)
        pos_action = pos_action * descent_scale

        if self.joint_created:
            pos_action = torch.zeros_like(pos_action)

        rot_action = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action, rot_action, gripper_action], dim=-1)

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        self._check_attach_condition()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
        # idx 1/2 use scripted plug-to-hole policy; other task_idx keep RL action.
        scripted = self._compute_scripted_action()
        if scripted is not None:
            self.actions = scripted
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

        self.ctrl_target_gripper_dof_pos = 0.015 if gripper_actions < 0.0 else 0.0
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
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "chair_assembly":
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
        # Reset scripted-policy state.
        self._near_hole_frames.zero_()
        self._xy_aligned_latched.zero_()
        self._search_active_latched.zero_()
        self._search_step = 0
        self._press_down_latched.zero_()
        self._descend_latched.zero_()
        self._prev_held_z.fill_(float("nan"))
        self._stuck_frames.zero_()
        self._search_center_perp.zero_()
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
        if self.cfg_task.name == "chair_assembly" and self.cfg_task.task_idx in [1, 2, 4, 5]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "chair_assembly" and self.cfg_task.task_idx == 3:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.rod_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 0] = 0.025
            held_asset_relative_pos[:, 1] = -0.01
            # held_asset_relative_pos[:, 2] = 0.05
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
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
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


        if self.cfg_task.task_idx in [1,2,4,5]:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 1] += 0.22
            rela_trans[:, 0] += 0.22


        elif self.cfg_task.task_idx == 3:
            rod_tip_pos_local = torch.zeros_like(self.held_pos)
            rod_tip_pos_local[:, 2] += self.cfg_task.rod_asset_cfg.base_height
            rod_tip_quat_local = (
            torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
            _, rod_tip_pos = torch_utils.tf_combine(
                self.held_quat, self.held_pos, rod_tip_quat_local, rod_tip_pos_local
            )
            rela_trans = rod_tip_pos.clone()
            rela_trans[:, 0] += 0.1
            rela_trans[:, 1] -= 0.25


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
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            self.hand_down_euler[bad_envs, ...] = hand_down_euler
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

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

        self.step_sim_no_action()

        if self.cfg_task.task_idx == 2:
            self._create_fixed_joint(connection_idx=1)

        elif self.cfg_task.task_idx == 3:
            self._create_fixed_joint(connection_idx=1)
            self._create_fixed_joint(connection_idx=2)

        elif self.cfg_task.task_idx == 4:
            self._create_fixed_joint(connection_idx=3)

        if self.cfg_task.task_idx == 5:
            self._create_fixed_joint(connection_idx=3)
            self._create_fixed_joint(connection_idx=4)

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
        self.held_asset_pos_noise = 0.001 * (rand_sample - 0.5)  # [-1, 1]

        held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=self.identity_quat,
            t2=self.held_asset_pos_noise,
        )

        if self.cfg_task.task_idx == 3:
            rot_euler = torch.tensor([0.0, 1.5707, 1.5707], device=self.device).repeat(
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
        while grasp_time < 0.25:
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