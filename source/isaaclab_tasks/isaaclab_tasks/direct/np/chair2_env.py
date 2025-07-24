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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control as fc
from .np_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FrankaChair2Cfg
from .np_tasks_cfg import ChairAssembly1, ConnectionCfg
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
            self._rod_asset = RigidObject(self.cfg_task.rod_asset)
            self._held_asset = self._rod_asset
            self._connection_cfg = self.cfg_task.connection_cfg3
            
        if self.cfg_task.task_idx == 4:
            self._plug1 = RigidObject(self.cfg_task.screw)
            self._held_asset = self._plug1
            self._connection_cfg = self.cfg_task.connection_cfg5
        # self._backrest_asset = RigidObject(self.cfg_task.backrest_asset)
        # self._rod_asset = RigidObject(self.cfg_task.rod_asset)
        

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
        fixed_prim = stage.GetPrimAtPath("/World/envs/env_0/FixedAsset")
        

        # 获取 held/fixed asset 的物理 pose（世界坐标）

        fixed_pos = self.fixed_pos[0].cpu().numpy()
        fixed_quat = self.fixed_quat[0].cpu().numpy()

        if connection_idx == 1:
            held_pos = self._plug1.data.root_pos_w - self.scene.env_origins
            held_pos = held_pos[0].cpu().numpy()
            held_quat = self._plug1.data.root_quat_w
            held_quat = held_quat[0].cpu().numpy()
        elif connection_idx == 2:
            held_pos = self._plug2.data.root_pos_w - self.scene.env_origins
            held_pos = held_pos[0].cpu().numpy()
            held_quat = self._plug2.data.root_quat_w
            held_quat = held_quat[0].cpu().numpy()
        elif connection_idx == 3:
            held_pos = self._rod_asset.data.root_pos_w - self.scene.env_origins
            held_pos = held_pos[0].cpu().numpy()
            held_quat = self._rod_asset.data.root_quat_w
            held_quat = held_quat[0].cpu().numpy()


        held_mat = Gf.Matrix4d()
        held_mat.SetRotate(Gf.Rotation(Gf.Quatd(float(held_quat[0]), float(held_quat[1]), float(held_quat[2]), float(held_quat[3]))))
        held_mat.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in held_pos]))

        fixed_mat = Gf.Matrix4d()
        fixed_mat.SetRotate(Gf.Rotation(Gf.Quatd(float(fixed_quat[0]), float(fixed_quat[1]), float(fixed_quat[2]), float(fixed_quat[3]))))
        fixed_mat.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in fixed_pos]))


        to_pose = held_mat
        from_pose = fixed_mat
        to_path = held_prim.GetPath()
        from_path = fixed_prim.GetPath()

        rel_pose = to_pose * from_pose.GetInverse()
        rel_pose = rel_pose.RemoveScaleShear()
        # pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
        # rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())

        # rel_mat = self._get_real_mat()
        rel_mat = connection_cfg.pose_to_base
        pos1 = Gf.Vec3f([float(rel_mat[0, 3]), float(rel_mat[1, 3]), float(rel_mat[2, 3])])
        rot1q = torch_utils.rot_matrices_to_quats(torch.tensor(rel_mat[:3, :3]))
        rot1 = Gf.Quatf(float(rot1q[0]), float(rot1q[1]), float(rot1q[2]), float(rot1q[3]))


        # set the velocity of the held and fixed assets to zero before creating the joint
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()
        fixed_state = self._fixed_asset.data.default_root_state.clone()
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:])
        self._fixed_asset.reset()


        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])


        joint.CreateLocalPos0Attr().Set(pos1)
        joint.CreateLocalRot0Attr().Set(rot1)
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


        self.fixed_joint_prim = stage.GetPrimAtPath(joint_path)
        self.step_sim_no_action()

    def _create_screw_joint(self):
        stage = omni.usd.get_context().get_stage()
        held_prim = stage.GetPrimAtPath("/World/envs/env_0/Plug1")
        fixed_prim = stage.GetPrimAtPath("/World/envs/env_0/FixedAsset")
        connection_cfg = self._connection_cfg

        to_path = held_prim.GetPath()
        from_path = fixed_prim.GetPath()

        # 计算关节的相对位姿
        rel_mat = connection_cfg.pose_to_base
        pos1 = Gf.Vec3f([float(rel_mat[0, 3]), float(rel_mat[1, 3]), float(rel_mat[2, 3])])
        rot1q = torch_utils.rot_matrices_to_quats(torch.tensor(rel_mat[:3, :3]))
        rot1 = Gf.Quatf(float(rot1q[0]), float(rot1q[1]), float(rot1q[2]), float(rot1q[3]))

        # 归零 held 和 fixed asset 的速度
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()
        fixed_state = self._fixed_asset.data.default_root_state.clone()
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:])
        self._fixed_asset.reset()
        self.step_sim_no_action()

        # 创建 D6 Joint，允许 Z 轴旋转和平移
        joint_path = "/World/envs/env_0/ScrewJoint"
        d6_joint = UsdPhysics.Joint.Define(stage, joint_path)
        
        # 设置 body 关系
        d6_joint.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        d6_joint.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        
        # 设置位置和旋转
        d6_joint.CreateLocalPos0Attr().Set(pos1)
        d6_joint.CreateLocalRot0Attr().Set(rot1)
        d6_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        d6_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        prim = d6_joint.GetPrim()
        for limit_name in ["transX", "transY", "transZ", "rotX", "rotY"]:
            limit_api = UsdPhysics.LimitAPI.Apply(prim, limit_name)
            limit_api.CreateLowAttr(0.0)
            limit_api.CreateHighAttr(0.0)

        for limit_name in ["rotZ"]:
            limit_api = UsdPhysics.LimitAPI.Apply(prim, limit_name)
            limit_api.CreateLowAttr(-3.14)
            limit_api.CreateHighAttr(3.14)  
        # 保存 joint prim
        self.fixed_joint_prim = stage.GetPrimAtPath(joint_path)
        for i in range(3):
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
        if not self.joint_created and R_dist < 0.1 and t_tangent < 0.003 and t_normal < 0.005:
            if self.cfg_task.task_idx == 4:
                self._create_screw_joint()
            else:
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




    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        self._check_attach_condition()
        if self.joint_created and self.cfg_task.task_idx == 4:
            self._sync_held_asset()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
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

        # rod_state = self._rod_asset.data.default_root_state.clone()[env_ids]
        # rod_state[:, 0:3] += self.scene.env_origins[env_ids]
        # rod_state[:, 7:] = 0.0
        # self._rod_asset.write_root_pose_to_sim(rod_state[:, 0:7], env_ids=env_ids)
        # self._rod_asset.write_root_velocity_to_sim(rod_state[:, 7:], env_ids=env_ids)
        # self._rod_asset.reset()   

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
        if self.cfg_task.name == "chair_assembly" and (self.cfg_task.task_idx == 1 or self.cfg_task.task_idx == 2 or self.cfg_task.task_idx == 4):
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "chair_assembly" and self.cfg_task.task_idx == 3:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.rod_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 0] = -0.02
            held_asset_relative_pos[:, 1] = 0.01
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

        # insert the first plug into the frame
        if self.cfg_task.task_idx ==1 or self.cfg_task.task_idx == 2 or self.cfg_task.task_idx == 4: 
            # Disable gravity.
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

            # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
            # For example, the tip of the bolt can be used as the observation frame
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos

            # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
            # (a) get position vector to target
            bad_envs = env_ids.clone()
            ik_attempt = 0

            hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            while True:
                n_bad = bad_envs.shape[0]

                above_fixed_pos = fixed_tip_pos.clone()
                above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]
                above_fixed_pos[:, 1] += 0.2

                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_fixed_pos_rand = 2 * (rand_sample - 0.5) + 0.5  # [-1, 1] # [-0.5, 1.5]
                hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
                above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
                above_fixed_pos[bad_envs] += above_fixed_pos_rand

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
                self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
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

                fixed_pos = self.fixed_pos[0]
                fixed_quat = self.fixed_quat[0]
                rel_SE3 = self.cfg_task.connection_cfg1.pose_to_base
                r = R.from_matrix(rel_SE3[:3, :3])
                quat_xyzw = r.as_quat()
                quat_wxyz = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=self.device, dtype=torch.float32)
                rel_t = torch.tensor(rel_SE3[:3, 3], device=self.device, dtype=torch.float32)

                translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                    q1=fixed_quat, t1=fixed_pos, q2=quat_wxyz, t2=rel_t
                )


                held_state = self._plug1.data.default_root_state.clone()
                held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
                held_state[:, 3:7] = translated_held_asset_quat
                held_state[:, 7:] = 0.0
                self._plug1.write_root_pose_to_sim(held_state[:, 0:7])
                self._plug1.write_root_velocity_to_sim(held_state[:, 7:])
                self._plug1.reset()
                self._create_fixed_joint(connection_idx=1)


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
            self.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]

            held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
            self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=translated_held_asset_quat,
                t1=translated_held_asset_pos,
                q2=self.identity_quat,
                t2=self.held_asset_pos_noise,
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

        # insert the rod1 into the frame via the plugin
        elif self.cfg_task.task_idx == 3:
            # Disable gravity.
            physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
            physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))


            # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
            # For example, the tip of the bolt can be used as the observation frame
            rod_tip_pos_local = torch.zeros_like(self.held_pos)
            # rod_tip_pos_local[:, 2] += self.cfg_task.rod_asset_cfg.height
            rod_tip_pos_local[:, 2] += self.cfg_task.rod_asset_cfg.base_height
            rod_tip_quat_local = (
            torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
            _, rod_tip_pos = torch_utils.tf_combine(
                self.held_quat, self.held_pos, rod_tip_quat_local, rod_tip_pos_local
            )

            # (2) Move gripper to randomizes location above rod asset. Keep trying until IK succeeds.
            # (a) get position vector to target
            bad_envs = env_ids.clone()
            ik_attempt = 0

            hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            while True:
                n_bad = bad_envs.shape[0]

                above_rod_pos = rod_tip_pos.clone()
                # above_rod_pos[:, 2] += self.cfg_task.hand_init_pos[2]

                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_rod_pos_rand = 0.01 * (rand_sample - 0.5) 
                hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
                above_rod_pos_rand = above_rod_pos_rand @ torch.diag(hand_init_pos_rand)
                above_rod_pos[bad_envs] += above_rod_pos_rand

                # (b) get random orientation facing down
                hand_down_euler = (
                    torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
                )

                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_rod_orn_noise = .01 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
                above_rod_orn_noise = above_rod_orn_noise @ torch.diag(hand_init_orn_rand)
                hand_down_euler += above_rod_orn_noise
                self.hand_down_euler[bad_envs, ...] = hand_down_euler
                hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                    roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
                )

                # (c) iterative IK Method
                self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_rod_pos[bad_envs, ...]
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
                    joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.04], env_ids=bad_envs
                )

                ik_attempt += 1

            self.step_sim_no_action()


            fixed_pos = self.fixed_pos[0]
            fixed_quat = self.fixed_quat[0]

            rel1_SE3 = self.cfg_task.connection_cfg1.pose_to_base
            r = R.from_matrix(rel1_SE3[:3, :3])
            quat_xyzw = r.as_quat()
            quat_wxyz = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=self.device, dtype=torch.float32)
            rel_t = torch.tensor(rel1_SE3[:3, 3], device=self.device, dtype=torch.float32)
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=fixed_quat, t1=fixed_pos, q2=quat_wxyz, t2=rel_t
            )
            plug1_state = self._plug1.data.default_root_state.clone()
            plug1_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
            plug1_state[:, 3:7] = translated_held_asset_quat
            plug1_state[:, 7:] = 0.0
            self._plug1.write_root_pose_to_sim(plug1_state[:, 0:7])
            self._plug1.write_root_velocity_to_sim(plug1_state[:, 7:])
            self._plug1.reset()
            self._create_fixed_joint(connection_idx=1)

            rel2_SE3 = self.cfg_task.connection_cfg2.pose_to_base
            r = R.from_matrix(rel2_SE3[:3, :3])
            quat_xyzw = r.as_quat()
            quat_wxyz = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=self.device, dtype=torch.float32)
            rel_t = torch.tensor(rel2_SE3[:3, 3], device=self.device, dtype=torch.float32)
            translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
                q1=fixed_quat, t1=fixed_pos, q2=quat_wxyz, t2=rel_t
            )
            held_state = self._plug2.data.default_root_state.clone()
            held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
            held_state[:, 3:7] = translated_held_asset_quat
            held_state[:, 7:] = 0.0
            self._plug2.write_root_pose_to_sim(held_state[:, 0:7])
            self._plug2.write_root_velocity_to_sim(held_state[:, 7:])
            self._plug2.reset()
            self._create_fixed_joint(connection_idx=2)



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

            rot_euler = torch.tensor([0.0, 1.5707, -1.5707], device=self.device).repeat(
                self.num_envs, 1
            )
            translated_held_asset_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )
            held_state = self._held_asset.data.default_root_state.clone()
            held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
            held_state[:, 3:7] = translated_held_asset_quat 
            held_state[:, 7:] = 0.0
            self._rod_asset.write_root_pose_to_sim(held_state[:, 0:7])
            self._rod_asset.write_root_velocity_to_sim(held_state[:, 7:])
            self._rod_asset.reset()



            #  Close hand
            # Set gains to use for quick resets.
            reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
                (self.num_envs, 1)
            )
            reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
            self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

            self.step_sim_no_action()

            # grasp_time = 0.0
            # while grasp_time < 0.25:
            #     self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            #     self.ctrl_target_gripper_dof_pos = 0.0
            #     self.close_gripper_in_place()
            #     self.step_sim_no_action()
            #     grasp_time += self.sim.get_physics_dt()

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
            # self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

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
            # self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action

            # Zero initial velocity.
            self.ee_angvel_fd[:, :] = 0.0
            self.ee_linvel_fd[:, :] = 0.0

            # Set initial gains for the episode.
            self._set_gains(self.default_gains)
            physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
            self.step_sim_no_action()

