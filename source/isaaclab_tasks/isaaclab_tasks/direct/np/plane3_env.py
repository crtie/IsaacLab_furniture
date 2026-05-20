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
from .np_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FrankaPlane3Cfg
from pdb import set_trace as bp
from .np_utils.group_utils import SE3dist
from .np_utils.viz_utils import define_markers
from scipy.spatial.transform import Rotation as R
import torch
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf, Gf, Tf
from omni.physx.scripts import utils
import omni.usd

class FrankaPlane3Env(DirectRLEnv):
    cfg: FrankaPlane3Cfg

    def __init__(self, cfg: FrankaPlane3Cfg, render_mode: str | None = None, **kwargs):
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

        # Scripted-policy state for plane3 idx=1 (body3 two-hole ↦ wheel_all two-peg).
        # STRICTLY independent — every tensor carries `_p3_1` suffix so no
        # variable bleeds across idx boundaries.
        self._xy_align_frames_p3_1     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p3_1  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p3_1   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p3_1  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_along_p3_1          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p3_1        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p3_1         = 0

        # Scripted-policy state for plane3 idx=2 (crossbar 2-peg ↦ wheel_all 2-hole).
        # STRICTLY independent of idx=1 — every tensor carries `_p3_2`.
        self._xy_align_frames_p3_2     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p3_2  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p3_2   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p3_2  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_along_p3_2          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p3_2        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p3_2         = 0

        # Scripted-policy state for plane3 idx=3 (crossbar2 2-peg ↦ wing RIGHT 2-hole).
        # STRICTLY independent of idx=1/2 — every tensor carries `_p3_3`.
        self._xy_align_frames_p3_3     = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._xy_aligned_latched_p3_3  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._near_hover_frames_p3_3   = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._press_down_latched_p3_3  = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_along_p3_3          = torch.zeros((self.num_envs,), device=self.device)
        self._stall_frames_p3_3        = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._wobble_step_p3_3         = 0

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
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)

        if self.cfg_task.task_idx == 1:
            self._body = RigidObject(self.cfg_task.body)
            self._held_asset = self._body
            self._connection_cfg = self.cfg_task.connection_cfg1

        if self.cfg_task.task_idx ==2:
            self._body = RigidObject(self.cfg_task.body)
            self._crossbar = RigidObject(self.cfg_task.crossbar1)
            self._held_asset = self._crossbar
            self._connection_cfg = self.cfg_task.connection_cfg2

        if self.cfg_task.task_idx ==3:
            self._body = RigidObject(self.cfg_task.body)
            self._crossbar1 = RigidObject(self.cfg_task.crossbar1)
            self._crossbar2 = RigidObject(self.cfg_task.crossbar2)
            self._held_asset = self._crossbar2
            self._connection_cfg = self.cfg_task.connection_cfg3

        if self.cfg_task.task_idx ==4:
            self._body = RigidObject(self.cfg_task.tailhalf)
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
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Body")
            joint_path = "/World/envs/env_0/FixedJoint1"
            connection_cfg = self.cfg_task.connection_cfg1
        elif connection_idx == 2:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/Crossbar1")
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
        
        if connection_idx == 2:
            self._body.reset()

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
        held_state = self._body.data.default_root_state.clone()[0]
        fixed_pos = held_state[:3].cpu().numpy()
        fixed_quat = held_state[3:7].cpu().numpy()
        self.marker_locations[0] = torch.tensor(fixed_pos, device=self.device)
        self.marker_orientations[0] = torch.tensor(fixed_quat, device=self.device)
        loc = self.marker_locations
        rots = self.marker_orientations

        # render the markers
        all_envs = torch.arange(self.num_envs)
        indices = torch.zeros_like(all_envs)
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    # ------------------------------------------------------------------
    # plane3 idx=1 — body3 two holes (HELD) ↦ wheel_all two pegs (FIXED)
    # ------------------------------------------------------------------
    # STRICT independence: every name carries the `_p3_1` suffix. No
    # state or helpers shared with any other idx in any other file.
    # Offline mesh analysis (one-time):
    #   - wheel_all.obj transformed by fixed_quat (0,-0.707,0.707,0) and
    #     fixed_pos (0.11,-0,0.79): the 2 WORLD-HIGHEST points sit at
    #     world z≈0.8376 = the 2 peg tips. Back-solving into fixed_local
    #     (unscaled): (±0.0121, +0.001, -0.0397). fixed_scale=1.2.
    #   - body3.obj: user visually verified in pick_hole_p3.py that
    #     the peg fixed_local UNSCALED coords (±0.0121, +0.001, -0.0397)
    #     drawn into body3's mesh frame land exactly in the 2 body3 hole
    #     centers. Reuse those exact values as the hole positions in
    #     held_local UNSCALED (held_scale=1.2 applied at draw time).
    P3_1_PEG_LOCAL_LEFT   = (-0.0121, +0.0009, -0.0397)   # fixed_local, UNSCALED
    P3_1_PEG_LOCAL_RIGHT  = (+0.0121, +0.0012, -0.0397)
    P3_1_PEG_SCALE        = 1.2
    # Precise DBSCAN ring detection in the (±0.012, ~0, -0.04) region
    # confirmed 2 small circular holes (r≈3mm each) in body3:
    #   R1 left  (x<0): held_local (-0.0123, +0.0027, -0.0460)
    #   R2 right (x>0): held_local (+0.0117, +0.0027, -0.0460)
    # Both detected across z=[-0.052, -0.040] (through-holes).
    P3_1_HOLE_LOCAL_LEFT  = (-0.0123, +0.0027, -0.0460)   # held_local, UNSCALED — body3 left through-hole
    P3_1_HOLE_LOCAL_RIGHT = (+0.0117, +0.0027, -0.0460)   # held_local, UNSCALED — body3 right through-hole
    P3_1_HOLE_SCALE       = 1.2

    # All 66 circular-feature candidates detected by offline DBSCAN ring
    # scanning of body3.obj (z-axis xy-plane slabs + y-axis xz-plane
    # slabs, eps=6mm, ratio<0.4, angle coverage>0.45). Format:
    # (held_local_x, held_local_y, held_local_z, radius_m, n_detections)
    # User picks 2 of these by candidate # in the viewer; we then set
    # P3_1_HOLE_LOCAL_LEFT/RIGHT to those constants and remove this list.
    P3_1_ALL_HOLE_CANDIDATES_UNSCALED = [
        (+0.0002, -0.1660, -0.1185, 0.0053, 6),  # 1
        (+0.0001, -0.1821, -0.1115, 0.0038, 8),  # 2
        (+0.0004, -0.1499, -0.1118, 0.0038, 8),  # 3
        (+0.0004, -0.1417, -0.1013, 0.0038, 4),  # 4
        (+0.0001, -0.1901, -0.1010, 0.0037, 4),  # 5
        (+0.0003, -0.1381, -0.0929, 0.0042, 3),  # 6
        (+0.0001, -0.1939, -0.0924, 0.0041, 4),  # 7
        (+0.0002, -0.1356, -0.0860, 0.0034, 1),  # 8
        (+0.0003, -0.1972, -0.0800, 0.0035, 3),  # 9
        (-0.0105, -0.1556, -0.0761, 0.0067, 3),  # 10
        (+0.0002, -0.1988, -0.0710, 0.0034, 3),  # 11
        (-0.0007, +0.0691, -0.0710, 0.0060, 3),  # 12
        (+0.0020, -0.1333, -0.0710, 0.0036, 1),  # 13
        (-0.0002, -0.0695, -0.0710, 0.0120, 1),  # 14
        (+0.0641, +0.0670, -0.0680, 0.0035, 3),  # 15
        (-0.0654, +0.0670, -0.0680, 0.0035, 3),  # 16
        (-0.0005, +0.0494, -0.0621, 0.0132, 8),  # 17
        (-0.0007, +0.0712, -0.0617, 0.0115, 7),  # 18
        (-0.0657, +0.0667, -0.0592, 0.0056, 7),  # 19
        (+0.0644, +0.0667, -0.0592, 0.0056, 7),  # 20
        (-0.0007, +0.0721, -0.0552, 0.0063, 4),  # 21
        (-0.0002, -0.0695, -0.0560, 0.0120, 1),  # 22
        (+0.0002, -0.1988, -0.0530, 0.0034, 1),  # 23
        (+0.0049, -0.1421, -0.0485, 0.0119, 2),  # 24
        (+0.0006, -0.1344, -0.0410, 0.0035, 2),  # 25
        (+0.0002, -0.1974, -0.0425, 0.0034, 2),  # 26
        (+0.0118, +0.0027, -0.0410, 0.0030, 1),  # 27
        (-0.0000, -0.1375, -0.0308, 0.0039, 4),  # 28
        (+0.0003, -0.1942, -0.0312, 0.0039, 4),  # 29
        (+0.0001, -0.1411, -0.0223, 0.0038, 4),  # 30
        (+0.0004, -0.1906, -0.0227, 0.0039, 4),  # 31
        (+0.0004, -0.1846, -0.0134, 0.0039, 6),  # 32
        (-0.0001, -0.1466, -0.0140, 0.0038, 5),  # 33
        (+0.0001, -0.1551, -0.0065, 0.0037, 5),  # 34
        (+0.0002, -0.1760, -0.0061, 0.0037, 4),  # 35
        (+0.0001, -0.1670, -0.0037, 0.0051, 3),  # 36
        (+0.0116, -0.1991, -0.0550, 0.0154, 1),  # 37
        (-0.0108, -0.1991, -0.0674, 0.0151, 1),  # 38
        (-0.0385, -0.1946, -0.0607, 0.0048, 2),  # 39
        (+0.0403, -0.1946, -0.0614, 0.0047, 2),  # 40
        (+0.0508, -0.1871, -0.0613, 0.0041, 3),  # 41
        (-0.0491, -0.1871, -0.0608, 0.0041, 3),  # 42
        (+0.0574, -0.1781, -0.0612, 0.0039, 3),  # 43
        (-0.0557, -0.1781, -0.0609, 0.0039, 3),  # 44
        (+0.0591, -0.1691, -0.0610, 0.0038, 3),  # 45
        (-0.0574, -0.1691, -0.0611, 0.0037, 3),  # 46
        (+0.0565, -0.1601, -0.0610, 0.0038, 3),  # 47
        (-0.0548, -0.1601, -0.0610, 0.0039, 3),  # 48
        (-0.0125, -0.1556, -0.0496, 0.0084, 2),  # 49
        (+0.0112, -0.1541, -0.0486, 0.0079, 3),  # 50
        (+0.0124, -0.1556, -0.0738, 0.0080, 2),  # 51
        (+0.0484, -0.1511, -0.0609, 0.0042, 3),  # 52
        (-0.0468, -0.1511, -0.0612, 0.0042, 3),  # 53
        (+0.0393, -0.1451, -0.0607, 0.0049, 1),  # 54
        (-0.0337, -0.1436, -0.0612, 0.0055, 2),  # 55
        (-0.0174, -0.1391, -0.0608, 0.0085, 1),  # 56
        (+0.0177, -0.1391, -0.0607, 0.0093, 1),  # 57
        (+0.0002, -0.0986, -0.0769, 0.0156, 2),  # 58
        (-0.0005, -0.0461, -0.0789, 0.0171, 1),  # 59
        (-0.0002, -0.0281, -0.0546, 0.0251, 1),  # 60
        (-0.0002, +0.0139, -0.0616, 0.0060, 1),  # 61
        (-0.0011, +0.0229, -0.0616, 0.0058, 1),  # 62
        (-0.0002, +0.0319, -0.0381, 0.0114, 1),  # 63
        (-0.0002, +0.0319, -0.0616, 0.0105, 2),  # 64
        (-0.0011, +0.0409, -0.0616, 0.0094, 1),  # 65
        (-0.0008, +0.0619, -0.0616, 0.0107, 1),  # 66
    ]

    # Head-recess focused candidate list — these are the unique features
    # detected by an aggressive rescan of body3's head area
    # (x∈[-0.04,+0.04], y∈[+0.02,+0.09], z∈[-0.10,-0.04]).
    # Format: (held_local_x, held_local_y, held_local_z, radius_m, n_detections)
    P3_1_HEAD_HOLE_CANDIDATES_UNSCALED = [
        (-0.0002, +0.0463, -0.0840, 0.0100, 3),   # H1
        (+0.0188, +0.0463, -0.0770, 0.0033, 2),   # H2
        (-0.0186, +0.0461, -0.0780, 0.0033, 1),   # H3
        (-0.0001, +0.0457, -0.0730, 0.0173, 2),   # H4
        (-0.0007, +0.0690, -0.0710, 0.0059, 4),   # H5
        (-0.0004, +0.0453, -0.0680, 0.0179, 1),   # H6
        (-0.0008, +0.0509, -0.0640, 0.0038, 3),   # H7
        (-0.0007, +0.0722, -0.0650, 0.0118, 2),   # H8
        (-0.0000, +0.0307, -0.0660, 0.0037, 1),   # H9
        (-0.0012, +0.0218, -0.0640, 0.0042, 2),   # H10
        (-0.0199, +0.0452, -0.0620, 0.0074, 3),   # H11
        (+0.0188, +0.0450, -0.0620, 0.0081, 3),   # H12
        (-0.0007, +0.0722, -0.0570, 0.0100, 4),   # H13
        (-0.0010, +0.0508, -0.0580, 0.0029, 1),   # H14
        (-0.0002, +0.0451, -0.0580, 0.0191, 1),   # H15
        (-0.0004, +0.0307, -0.0580, 0.0044, 1),   # H16
        (-0.0017, +0.0218, -0.0580, 0.0045, 1),   # H17
        (-0.0204, +0.0458, -0.0540, 0.0059, 1),   # H18
        (-0.0007, +0.0674, -0.0510, 0.0050, 2),   # H19
        (-0.0006, +0.0471, -0.0480, 0.0097, 2),   # H20
        (-0.0198, +0.0465, -0.0470, 0.0032, 2),   # H21
        (+0.0182, +0.0466, -0.0460, 0.0032, 1),   # H22
        (-0.0009, +0.0307, -0.0460, 0.0025, 1),   # H23
        (-0.0002, +0.0465, -0.0420, 0.0135, 1),   # H24
    ]

    # body3 belly FLAT PANEL — the small region where the otherwise
    # curved cylindrical fuselage has been flattened. From the mesh
    # analysis: the cylinder's curved bottom dips to z≈-0.072, but a
    # rectangular panel of width x∈[-0.024,+0.024] and y∈[-0.100,-0.045]
    # all sits at z≈-0.072 (no further down-curve). This panel is the
    # flat patch the user is pointing at.
    P3_1_BELLY_PLANE_Z          = -0.072
    P3_1_BELLY_PLANE_X_BOUNDS   = (-0.024, +0.024)
    P3_1_BELLY_PLANE_Y_BOUNDS   = (-0.100, -0.045)
    P3_1_BELLY_PLANE_GRID_N     = 10

    def _draw_belly_plane_p3_1(self, points, colors, sizes, lines_a, lines_b, line_colors, line_widths):
        """Append belly-plane geometry to the draw buffers: 4 yellow
        corner dots + 4 boundary lines + a sparse grid of small yellow
        dots filling the plane interior."""
        s = self.P3_1_HOLE_SCALE
        env_origin = self.scene.env_origins[0]
        z = self.P3_1_BELLY_PLANE_Z * s
        xs = (self.P3_1_BELLY_PLANE_X_BOUNDS[0] * s,
              self.P3_1_BELLY_PLANE_X_BOUNDS[1] * s)
        ys = (self.P3_1_BELLY_PLANE_Y_BOUNDS[0] * s,
              self.P3_1_BELLY_PLANE_Y_BOUNDS[1] * s)

        def _to_world(lx, ly, lz):
            local_p = torch.tensor(
                [lx, ly, lz], dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        corner_worlds = [
            _to_world(xs[0], ys[0], z),
            _to_world(xs[1], ys[0], z),
            _to_world(xs[1], ys[1], z),
            _to_world(xs[0], ys[1], z),
        ]
        # Big yellow corner dots.
        for cw in corner_worlds:
            points.append(cw)
            colors.append((1.0, 1.0, 0.0, 1.0))
            sizes.append(24.0)
        # 4 boundary lines (yellow).
        for k in range(4):
            lines_a.append(corner_worlds[k])
            lines_b.append(corner_worlds[(k + 1) % 4])
            line_colors.append((1.0, 1.0, 0.0, 1.0))
            line_widths.append(3.0)
        # Sparse grid of small yellow dots across the plane.
        n_grid = self.P3_1_BELLY_PLANE_GRID_N
        for gi in range(n_grid):
            for gj in range(n_grid):
                u = gi / (n_grid - 1)
                v = gj / (n_grid - 1)
                lx = xs[0] + (xs[1] - xs[0]) * u
                ly = ys[0] + (ys[1] - ys[0]) * v
                points.append(_to_world(lx, ly, z))
                colors.append((1.0, 1.0, 0.0, 0.7))
                sizes.append(4.0)

    # Candidate pairs whose held_local separation is within ±15% of the
    # peg-pair separation (peg sep = 2×0.0121 = 0.0242 m unscaled in
    # fixed_local). Both endpoints scaled by held_scale=1.2 at draw
    # time. (pair_id, idx_A, idx_B)
    # Generated offline by filtering all 66×65/2 pair distances; only
    # pairs in held_local distance ∈ [0.020, 0.028] m are kept.
    P3_1_PEG_SEP_PAIRS = [
        ( 1, 13, 24),    # d=0.0243 (best match) — middle/tail belly
        ( 2, 17, 21),    # d=0.0237 — head recess
        ( 3,  8, 10),    # d=0.0247 — tail wheels
        ( 4, 33, 36),    # d=0.0229 — tail x-ribs
        ( 5,  6, 13),    # d=0.0225 — middle/tail
        ( 6,  2,  7),    # d=0.0225 — back wheel
        ( 7,  3,  6),    # d=0.0223 — front wheel
        ( 8, 23, 29),    # d=0.0223 — tail ribs
        ( 9,  5,  9),    # d=0.0222 — back wheel
        (10, 31, 35),    # d=0.0221 — tail x-ribs
        (11,  7, 11),    # d=0.0220 — back wheel
        (12, 17, 18),    # d=0.0218 — head recess
        (13, 12, 17),    # d=0.0216 — head
        (14, 30, 34),    # d=0.0211 — front ribs
        (15, 26, 31),    # d=0.0209 — back ribs
        (16, 34, 35),    # d=0.0209 — tail
        (17, 29, 32),    # d=0.0202 — back ribs
        (18, 32, 36),    # d=0.0201 — back ribs
    ]

    # Live held-frame pose dump — print held_pos and held_quat once a
    # second so user can copy them along with world coords picked from
    # the viewer. World→held_local conversion (UNSCALED) is then:
    #   held_world_offset = world_xyz - (held_pos + env_origin)
    #   held_local_scaled = quat_apply_inverse(held_quat, held_world_offset)
    #   held_local_unscaled = held_local_scaled / held_scale (=1.2)
    def _dump_held_frame_p3_1(self):
        c = getattr(self, "_p3_1_pose_dump_counter", 0) + 1
        self._p3_1_pose_dump_counter = c
        if c % 60 != 1:
            return
        env_origin = self.scene.env_origins[0]
        hp = (self.held_pos[0] + env_origin).detach().cpu().numpy()
        hq = self.held_quat[0].detach().cpu().numpy()
        print(
            f"\n[plane3 idx=1] held_pose (env 0)  "
            f"held_pos_world=({hp[0]:+.4f},{hp[1]:+.4f},{hp[2]:+.4f})  "
            f"held_quat_wxyz=({hq[0]:+.4f},{hq[1]:+.4f},{hq[2]:+.4f},{hq[3]:+.4f})  "
            f"held_scale={self.P3_1_HOLE_SCALE}"
        )
        print(
            "  To convert a world coord (x,y,z) to held_local UNSCALED:\n"
            "    offset = world - held_pos_world\n"
            "    held_local_scaled = quat_apply_inverse(held_quat, offset)\n"
            "    held_local_unscaled = held_local_scaled / 1.2"
        )

    def _visualize_pegs_only_p3_1(self):
        """Minimal viz: only the 2 fixed-asset peg tips as big red dots.
        No held-body markers, no belly plane, no candidate spheres."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        env_origin = self.scene.env_origins[0]
        s_peg = self.P3_1_PEG_SCALE
        points, colors, sizes = [], [], []
        for peg_local in [self.P3_1_PEG_LOCAL_LEFT, self.P3_1_PEG_LOCAL_RIGHT]:
            local_p = torch.tensor(
                [c * s_peg for c in peg_local],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            points.append((float(v[0]), float(v[1]), float(v[2])))
            colors.append((1.0, 0.0, 0.0, 1.0))
            sizes.append(28.0)
        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)

    def _visualize_peg_sep_pairs_p3_1(self):
        """Render ONLY the candidate pairs whose held_local separation
        matches the peg-pair separation (±15%). Each pair gets a unique
        HSV color and shares both endpoint dots + a line connecting them.
        A number stack (pair_id small white dots) sits at the midpoint of
        each line. User counts the dots to read pair_id."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        import colorsys
        s = self.P3_1_HOLE_SCALE
        env_origin = self.scene.env_origins[0]
        all_cands = self.P3_1_ALL_HOLE_CANDIDATES_UNSCALED
        pairs = self.P3_1_PEG_SEP_PAIRS
        n_pairs = len(pairs)

        if not getattr(self, "_p3_1_pairs_legend_printed", False):
            print(f"\n[plane3 idx=1] peg-sep matched pairs (#1..#{n_pairs})")
            print("  Each pair = colored line + 2 endpoint dots + N white dots at midpoint.")
            print(f"  Peg separation in held_local = {2*0.0121:.4f} m unscaled.")
            for pid, a, b in pairs:
                ca, cb = all_cands[a-1], all_cands[b-1]
                dx, dy, dz = ca[0]-cb[0], ca[1]-cb[1], ca[2]-cb[2]
                d = (dx*dx + dy*dy + dz*dz) ** 0.5
                print(f"  pair#{pid:2d}  ({a:2d}+{b:2d}) d={d:.4f}  "
                      f"A=({ca[0]:+.4f},{ca[1]:+.4f},{ca[2]:+.4f})  "
                      f"B=({cb[0]:+.4f},{cb[1]:+.4f},{cb[2]:+.4f})")
            self._p3_1_pairs_legend_printed = True

        def _to_world(lx, ly, lz):
            local_p = torch.tensor(
                [lx * s, ly * s, lz * s], dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        points = []
        colors = []
        sizes  = []
        lines_a, lines_b, line_colors, line_widths = [], [], [], []

        for i, (pid, a_idx, b_idx) in enumerate(pairs):
            ca = all_cands[a_idx - 1]
            cb = all_cands[b_idx - 1]
            wa = _to_world(ca[0], ca[1], ca[2])
            wb = _to_world(cb[0], cb[1], cb[2])
            h = (i / n_pairs) % 1.0
            rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            color = (rgb[0], rgb[1], rgb[2], 1.0)
            # 2 endpoint dots
            points.append(wa); colors.append(color); sizes.append(16.0)
            points.append(wb); colors.append(color); sizes.append(16.0)
            # connecting line
            lines_a.append(wa); lines_b.append(wb)
            line_colors.append(color); line_widths.append(3.0)
            # pair_id white dots stacked at midpoint along world +z
            mid = (
                (wa[0] + wb[0]) * 0.5,
                (wa[1] + wb[1]) * 0.5,
                (wa[2] + wb[2]) * 0.5,
            )
            for j in range(pid):
                points.append((mid[0], mid[1], mid[2] + 0.005 + 0.005 * j))
                colors.append((1.0, 1.0, 1.0, 1.0))
                sizes.append(5.0)

        # Draw pegs (red) for reference.
        s_peg = self.P3_1_PEG_SCALE
        for peg_local in [self.P3_1_PEG_LOCAL_LEFT, self.P3_1_PEG_LOCAL_RIGHT]:
            local_p = torch.tensor(
                [c * s_peg for c in peg_local],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            points.append((float(v[0]), float(v[1]), float(v[2])))
            colors.append((1.0, 0.0, 0.0, 1.0))
            sizes.append(28.0)

        # belly flat panel for reference
        self._draw_belly_plane_p3_1(
            points, colors, sizes, lines_a, lines_b, line_colors, line_widths,
        )

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)
        if lines_a:
            self._dbg_draw.draw_lines(lines_a, lines_b, line_colors, line_widths)

    def _visualize_head_candidates_numbered_p3_1(self):
        """Draw each head-recess candidate H1..H24 with its number rendered
        as a vertical stack of N small white dots ABOVE the candidate's
        colored marker. User counts the stacked dots to identify the
        candidate # of any specific hole they see in the viewer."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        import colorsys
        s = self.P3_1_HOLE_SCALE
        env_origin = self.scene.env_origins[0]
        candidates = self.P3_1_HEAD_HOLE_CANDIDATES_UNSCALED
        n_total = len(candidates)

        if not getattr(self, "_p3_1_head_legend_printed", False):
            print(f"\n[plane3 idx=1] head-recess candidates H1..H{n_total}")
            print("  Each candidate = colored main dot + N small white dots stacked ABOVE.")
            print("  Count the stacked dots to read candidate # (H1..H24).")
            print("  Use the console printout below to look up exact held_local coords.")
            self._p3_1_head_legend_printed = True

        points = []
        colors = []
        sizes  = []
        for i, (lx, ly, lz, r, _nd) in enumerate(candidates):
            cand_id = i + 1
            local_p = torch.tensor(
                [lx * s, ly * s, lz * s], dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            base_x, base_y, base_z = float(v[0]), float(v[1]), float(v[2])

            # Main colored dot — HSV hue cycles across the head candidate list.
            h = (i / n_total) % 1.0
            rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            points.append((base_x, base_y, base_z))
            colors.append((rgb[0], rgb[1], rgb[2], 1.0))
            sizes.append(18.0)

            # Stacked small white dots above the main dot.
            # Use 4 mm vertical spacing in WORLD z to keep the column visible.
            for j in range(cand_id):
                points.append((base_x, base_y, base_z + 0.004 + 0.004 * j))
                colors.append((1.0, 1.0, 1.0, 1.0))
                sizes.append(5.0)

        # Throttle console dump of the head-area coordinates.
        self._p3_1_head_dump_counter = getattr(self, "_p3_1_head_dump_counter", 0) + 1
        if self._p3_1_head_dump_counter % 60 == 1:
            print(f"\n[plane3 idx=1] head-recess candidate world positions (env 0):")
            for i, (lx, ly, lz, r, _nd) in enumerate(candidates):
                local_p = torch.tensor(
                    [lx * s, ly * s, lz * s], dtype=torch.float32, device=self.device,
                ).unsqueeze(0).repeat(self.num_envs, 1)
                _, world_p = torch_utils.tf_combine(
                    self.held_quat, self.held_pos, self.identity_quat, local_p
                )
                v = (world_p[0] + env_origin).detach().cpu().numpy()
                h = (i / n_total) % 1.0
                rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                print(f"  H{i+1:2d}  hue={h*360:5.1f}°  rgb=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})  "
                      f"world=({float(v[0]):+.4f},{float(v[1]):+.4f},{float(v[2]):+.4f})  "
                      f"held_local=({lx:+.4f},{ly:+.4f},{lz:+.4f})  r={r:.4f}")

        # Also draw RED dots for pegs so user can reference them.
        s_peg = self.P3_1_PEG_SCALE
        for peg_local in [self.P3_1_PEG_LOCAL_LEFT, self.P3_1_PEG_LOCAL_RIGHT]:
            local_p = torch.tensor(
                [c * s_peg for c in peg_local],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            points.append((float(v[0]), float(v[1]), float(v[2])))
            colors.append((1.0, 0.0, 0.0, 1.0))
            sizes.append(28.0)

        # Draw the belly flat plane (yellow outline + corners + grid)
        # so the user can see where the underside of body3 is and pick
        # the correct hole candidate by reference to it.
        lines_a, lines_b, line_colors, line_widths = [], [], [], []
        self._draw_belly_plane_p3_1(
            points, colors, sizes, lines_a, lines_b, line_colors, line_widths,
        )

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)
        if lines_a:
            self._dbg_draw.draw_lines(lines_a, lines_b, line_colors, line_widths)

    def _visualize_all_body3_hole_candidates_p3_1(self):
        """Draw ALL 66 detected circular candidates on body3, each with a
        unique HSV color. Console prints the mapping #→color/world-pos so
        user can identify the actual 2 hole indices from the viewer."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        import colorsys
        s = self.P3_1_HOLE_SCALE
        env_origin = self.scene.env_origins[0]

        candidates = self.P3_1_ALL_HOLE_CANDIDATES_UNSCALED
        n = len(candidates)

        if not getattr(self, "_p3_1_all_candidates_printed", False):
            print(f"\n[plane3 idx=1] dumping ALL {n} body3 circular-feature candidates")
            print("  Each rendered as a dot with a unique HSV color (#→hue 0…360°).")
            print("  Hover over a dot in the viewer or use its world position to find its #.")
            self._p3_1_all_candidates_printed = True

        points = []
        colors = []
        sizes  = []
        # Also compute world positions to print every N frames so user can match.
        world_positions = []
        for i, (lx, ly, lz, r, nd) in enumerate(candidates):
            local_p = torch.tensor(
                [lx * s, ly * s, lz * s], dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            points.append((float(v[0]), float(v[1]), float(v[2])))
            world_positions.append((float(v[0]), float(v[1]), float(v[2])))
            # HSV color: hue cycles every 360°, saturation+value full
            h = (i / n) % 1.0
            rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            colors.append((rgb[0], rgb[1], rgb[2], 1.0))
            # Bigger size for higher detection count (more confident)
            sizes.append(8.0 + min(2.0 * nd, 16.0))

        # Throttle the world-position dump
        self._p3_1_dump_counter = getattr(self, "_p3_1_dump_counter", 0) + 1
        if self._p3_1_dump_counter % 60 == 1:  # ~once per second at 60Hz
            print(f"\n[plane3 idx=1] candidate world positions (env 0):")
            for i, ((x,y,z), (lx,ly,lz,r,nd)) in enumerate(zip(world_positions, candidates)):
                h = (i / n) % 1.0
                rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                print(f"  #{i+1:2d}  hue={h*360:5.1f}°  rgb=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})  "
                      f"world=({x:+.4f},{y:+.4f},{z:+.4f})  held_local=({lx:+.4f},{ly:+.4f},{lz:+.4f})  "
                      f"r={r:.4f}  det={nd}")

        # Also draw RED dots for pegs so user can see relative position.
        s_peg = self.P3_1_PEG_SCALE
        peg_L_local = torch.tensor(
            [c * s_peg for c in self.P3_1_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local = torch.tensor(
            [c * s_peg for c in self.P3_1_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg_L_local
        )
        _, peg_R_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg_R_local
        )
        v = (peg_L_world[0] + env_origin).detach().cpu().numpy()
        points.append((float(v[0]), float(v[1]), float(v[2])))
        colors.append((1.0, 1.0, 1.0, 1.0))   # WHITE for pegs
        sizes.append(30.0)
        v = (peg_R_world[0] + env_origin).detach().cpu().numpy()
        points.append((float(v[0]), float(v[1]), float(v[2])))
        colors.append((1.0, 1.0, 1.0, 1.0))
        sizes.append(30.0)

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)

    def _visualize_two_pegs_two_holes_p3_1(self):
        """Draw GREEN dots = 2 body3 hole centers (held); RED dots = 2
        wheel_all peg tips (fixed). Lines connect the 4 points so the
        viewer can see alignment instantly."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p3_1_legend_printed", False):
            print("\n[plane3 idx=1] viewer legend (2 holes ↦ 2 pegs):")
            print(f"  GREEN dots = body3 hole centers (held)")
            print(f"     L: held_local {self.P3_1_HOLE_LOCAL_LEFT}  × {self.P3_1_HOLE_SCALE}")
            print(f"     R: held_local {self.P3_1_HOLE_LOCAL_RIGHT} × {self.P3_1_HOLE_SCALE}")
            print(f"  RED   dots = wheel_all peg tips (fixed)")
            print(f"     L: fixed_local {self.P3_1_PEG_LOCAL_LEFT}  × {self.P3_1_PEG_SCALE}")
            print(f"     R: fixed_local {self.P3_1_PEG_LOCAL_RIGHT} × {self.P3_1_PEG_SCALE}")
            self._p3_1_legend_printed = True

        s_hole = self.P3_1_HOLE_SCALE
        s_peg  = self.P3_1_PEG_SCALE
        # ---- 2 hole centers in world (held) ----
        hole_L_local_p3_1 = torch.tensor(
            [c * s_hole for c in self.P3_1_HOLE_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        hole_R_local_p3_1 = torch.tensor(
            [c * s_hole for c in self.P3_1_HOLE_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_L_world_p3_1 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hole_L_local_p3_1
        )
        _, hole_R_world_p3_1 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hole_R_local_p3_1
        )

        # ---- 2 peg tips in world (fixed) ----
        peg_L_local_p3_1 = torch.tensor(
            [c * s_peg for c in self.P3_1_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local_p3_1 = torch.tensor(
            [c * s_peg for c in self.P3_1_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world_p3_1 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg_L_local_p3_1
        )
        _, peg_R_world_p3_1 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg_R_local_p3_1
        )

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p3_1(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        gL = _to_tuple_p3_1(hole_L_world_p3_1[0])
        gR = _to_tuple_p3_1(hole_R_world_p3_1[0])
        rL = _to_tuple_p3_1(peg_L_world_p3_1[0])
        rR = _to_tuple_p3_1(peg_R_world_p3_1[0])
        points = [gL, gR, rL, rR]
        colors = [
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        sizes = [22.0, 22.0, 22.0, 22.0]
        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)
        # Connect the two GREEN with a green line; same for RED.
        self._dbg_draw.draw_lines(
            [gL, rL],
            [gR, rR],
            [(0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)],
            [3.0, 3.0],
        )

    def _scripted_action_two_peg_insert_p3_1(self):
        """plane3 idx=1 — drive body3's two-hole midpoint (GREEN-mid) onto
        wheel_all's two-peg midpoint (RED-mid) in world xy, then descend
        along world -z. Also align the hole-line yaw to the peg-line yaw
        so the holes match the pegs individually (2-point fit needs both
        translation and rotation). 3-phase latched policy with stall +
        wobble. Written from scratch with `_p3_1` suffix — no shared
        state or helpers."""
        # ---- Per-idx state ----
        xy_align_frames_p3_1    = self._xy_align_frames_p3_1
        xy_aligned_latched_p3_1 = self._xy_aligned_latched_p3_1
        near_hover_frames_p3_1  = self._near_hover_frames_p3_1
        press_down_latched_p3_1 = self._press_down_latched_p3_1

        # ---- Tunables ----
        hover_offset_m_p3_1       = 0.05
        align_perp_tol_p3_1       = 0.003
        along_tol_p3_1            = 0.01
        yaw_align_tol_rad_p3_1    = 0.05      # ~3°
        dwell_xy_align_p3_1       = 5
        dwell_to_press_p3_1       = 5
        press_down_depth_p3_1     = 0.20
        descent_scale_p3_1        = 0.3
        press_scale_p3_1          = 0.2
        yaw_align_scale_p3_1      = 0.4
        stall_threshold_m_p3_1    = 0.0003
        stall_dwell_frames_p3_1   = 8
        wobble_amplitude_p3_1     = 0.0015
        wobble_period_frames_p3_1 = 30

        s_hole = self.P3_1_HOLE_SCALE
        s_peg  = self.P3_1_PEG_SCALE

        # ---- 2 hole centers in world (held) ----
        hole_L_local_p3_1 = torch.tensor(
            [c * s_hole for c in self.P3_1_HOLE_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        hole_R_local_p3_1 = torch.tensor(
            [c * s_hole for c in self.P3_1_HOLE_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_L_world_p3_1 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hole_L_local_p3_1
        )
        _, hole_R_world_p3_1 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, hole_R_local_p3_1
        )
        hole_mid_world_p3_1 = 0.5 * (hole_L_world_p3_1 + hole_R_world_p3_1)

        # ---- 2 peg tips in world (fixed) ----
        peg_L_local_p3_1 = torch.tensor(
            [c * s_peg for c in self.P3_1_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local_p3_1 = torch.tensor(
            [c * s_peg for c in self.P3_1_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world_p3_1 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg_L_local_p3_1
        )
        _, peg_R_world_p3_1 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, peg_R_local_p3_1
        )
        peg_mid_world_p3_1 = 0.5 * (peg_L_world_p3_1 + peg_R_world_p3_1)

        # ---- Insertion axis = world +z (xy align then descend -z) ----
        axis_t_world_p3_1 = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        drive_world_p3_1  = hole_mid_world_p3_1
        target_world_p3_1 = peg_mid_world_p3_1

        # ---- Targets along axis ----
        above_target_p3_1      = target_world_p3_1 + axis_t_world_p3_1 * hover_offset_m_p3_1
        press_down_target_p3_1 = target_world_p3_1 - axis_t_world_p3_1 * press_down_depth_p3_1

        # ---- xy perp norm (mid-mid) ----
        delta_mid_p3_1     = drive_world_p3_1 - target_world_p3_1
        along_mid_p3_1     = (delta_mid_p3_1 * axis_t_world_p3_1).sum(-1, keepdim=True)
        perp_mid_p3_1      = delta_mid_p3_1 - along_mid_p3_1 * axis_t_world_p3_1
        mid_perp_norm_p3_1 = torch.norm(perp_mid_p3_1, dim=-1)

        # ---- Yaw err: angle from hole-line to peg-line, projected on
        # the perp plane (world xy) using cross-product trick ----
        # Body3 grasped 180° rotated → flip hole_vec direction (L-R
        # instead of R-L) so its orientation matches peg_vec without
        # touching the HOLE_LOCAL_* constants.
        hole_vec_world_p3_1 = hole_L_world_p3_1 - hole_R_world_p3_1
        peg_vec_world_p3_1  = peg_R_world_p3_1  - peg_L_world_p3_1
        # Project both onto perp plane (zero out z component).
        hole_vec_perp_p3_1 = hole_vec_world_p3_1.clone()
        hole_vec_perp_p3_1[:, 2] = 0.0
        peg_vec_perp_p3_1  = peg_vec_world_p3_1.clone()
        peg_vec_perp_p3_1[:, 2] = 0.0
        cross_p3_1 = torch.cross(hole_vec_perp_p3_1, peg_vec_perp_p3_1, dim=-1)
        sin_yaw_p3_1 = (cross_p3_1 * axis_t_world_p3_1).sum(-1)
        cos_yaw_p3_1 = (hole_vec_perp_p3_1 * peg_vec_perp_p3_1).sum(-1)
        yaw_err_p3_1 = torch.atan2(sin_yaw_p3_1, cos_yaw_p3_1)

        # ---- A1: xy mid + yaw align latch ----
        xy_aligned_p3_1 = (mid_perp_norm_p3_1 < align_perp_tol_p3_1) & (
            torch.abs(yaw_err_p3_1) < yaw_align_tol_rad_p3_1
        )
        xy_align_frames_p3_1 = torch.where(
            xy_aligned_p3_1,
            xy_align_frames_p3_1 + 1,
            torch.zeros_like(xy_align_frames_p3_1),
        )
        xy_aligned_latched_p3_1 = xy_aligned_latched_p3_1 | (
            xy_align_frames_p3_1 >= dwell_xy_align_p3_1
        )

        # ---- A2: near hover above mid ----
        d_above_p3_1     = drive_world_p3_1 - above_target_p3_1
        along_above_p3_1 = torch.abs((d_above_p3_1 * axis_t_world_p3_1).sum(-1))
        near_hover_p3_1 = (
            xy_aligned_latched_p3_1
            & (mid_perp_norm_p3_1 < align_perp_tol_p3_1)
            & (along_above_p3_1 < along_tol_p3_1)
        )
        near_hover_frames_p3_1 = torch.where(
            near_hover_p3_1,
            near_hover_frames_p3_1 + 1,
            torch.zeros_like(near_hover_frames_p3_1),
        )
        press_down_latched_p3_1 = press_down_latched_p3_1 | (
            near_hover_frames_p3_1 >= dwell_to_press_p3_1
        )

        # ---- Write back ----
        self._xy_align_frames_p3_1    = xy_align_frames_p3_1
        self._xy_aligned_latched_p3_1 = xy_aligned_latched_p3_1
        self._near_hover_frames_p3_1  = near_hover_frames_p3_1
        self._press_down_latched_p3_1 = press_down_latched_p3_1

        # ---- Phase split ----
        in_phase_A1_p3_1 = ~xy_aligned_latched_p3_1
        in_phase_A2_p3_1 = xy_aligned_latched_p3_1 & (~press_down_latched_p3_1)
        in_phase_B_p3_1  = press_down_latched_p3_1

        target_p3_1 = torch.where(
            in_phase_B_p3_1.unsqueeze(-1),
            press_down_target_p3_1,
            above_target_p3_1,
        )

        # ---- Raw pos action ----
        delta_p3_1      = target_p3_1 - drive_world_p3_1
        pos_action_p3_1 = delta_p3_1 / self.pos_threshold
        pos_action_p3_1 = torch.clamp(pos_action_p3_1, -1.0, 1.0)

        # A1: zero out along-axis so we don't descend until xy+yaw locked.
        pa_along_signed_p3_1 = (pos_action_p3_1 * axis_t_world_p3_1).sum(-1, keepdim=True)
        pa_perp_p3_1 = pos_action_p3_1 - pa_along_signed_p3_1 * axis_t_world_p3_1
        pos_action_p3_1 = torch.where(
            in_phase_A1_p3_1.unsqueeze(-1), pa_perp_p3_1, pos_action_p3_1
        )

        # Slower phase B.
        scale_p3_1 = torch.where(
            in_phase_B_p3_1.unsqueeze(-1),
            torch.full_like(pos_action_p3_1, press_scale_p3_1),
            torch.full_like(pos_action_p3_1, descent_scale_p3_1),
        )
        pos_action_p3_1 = pos_action_p3_1 * scale_p3_1

        # ---- Stall detection + perp-plane wobble (axis_t = world +z, so
        # e1 = world +x is a valid perp basis vector) ----
        along_now_p3_1 = (drive_world_p3_1 * axis_t_world_p3_1).sum(-1)
        progress_p3_1 = torch.abs(along_now_p3_1 - self._prev_along_p3_1)
        stalled_now_p3_1 = (progress_p3_1 < stall_threshold_m_p3_1) & in_phase_B_p3_1
        self._stall_frames_p3_1 = torch.where(
            stalled_now_p3_1,
            self._stall_frames_p3_1 + 1,
            torch.zeros_like(self._stall_frames_p3_1),
        )
        self._prev_along_p3_1 = along_now_p3_1.clone()
        wobble_active_p3_1 = self._stall_frames_p3_1 >= stall_dwell_frames_p3_1

        if bool(wobble_active_p3_1.any().item()):
            self._wobble_step_p3_1 += 1

        e1_world_p3_1 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p3_1)

        import math as _math
        phase_rad_p3_1 = (
            2.0 * _math.pi * (self._wobble_step_p3_1 / wobble_period_frames_p3_1)
        )
        wobble_offset_m_p3_1 = wobble_amplitude_p3_1 * _math.sin(phase_rad_p3_1)
        wobble_norm_p3_1 = wobble_offset_m_p3_1 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p3_1 = e1_world_p3_1 * wobble_norm_p3_1
        pos_action_p3_1 = torch.where(
            wobble_active_p3_1.unsqueeze(-1),
            pos_action_p3_1 + wobble_vec_p3_1,
            pos_action_p3_1,
        )
        pos_action_p3_1 = torch.clamp(pos_action_p3_1, -1.0, 1.0)

        # ---- Yaw P-control around world +z (axis-angle vector form) ----
        rot_action_vec_p3_1 = axis_t_world_p3_1 * yaw_err_p3_1.unsqueeze(-1) * (
            yaw_align_scale_p3_1 / (self.rot_threshold + 1e-8)
        )
        rot_action_vec_p3_1 = torch.clamp(rot_action_vec_p3_1, -1.0, 1.0)
        # Freeze yaw during phase B.
        rot_action_vec_p3_1 = torch.where(
            in_phase_B_p3_1.unsqueeze(-1),
            torch.zeros_like(rot_action_vec_p3_1),
            rot_action_vec_p3_1,
        )

        # ---- Diagnostic print ----
        gL = hole_L_world_p3_1[0].tolist()
        gR = hole_R_world_p3_1[0].tolist()
        rL = peg_L_world_p3_1[0].tolist()
        rR = peg_R_world_p3_1[0].tolist()
        phase = "A1-xy/yaw" if bool(in_phase_A1_p3_1[0]) else (
            "A2-hover" if bool(in_phase_A2_p3_1[0]) else "B-press"
        )
        center_dx_p3_1 = float(perp_mid_p3_1[0, 0])
        center_dy_p3_1 = float(perp_mid_p3_1[0, 1])
        print(
            f"[p3_1] phase={phase}  "
            f"holeL=({gL[0]:+.3f},{gL[1]:+.3f},{gL[2]:+.3f}) "
            f"holeR=({gR[0]:+.3f},{gR[1]:+.3f},{gR[2]:+.3f})  "
            f"pegL=({rL[0]:+.3f},{rL[1]:+.3f},{rL[2]:+.3f}) "
            f"pegR=({rR[0]:+.3f},{rR[1]:+.3f},{rR[2]:+.3f})  "
            f"dx={center_dx_p3_1*1000:+.2f}mm dy={center_dy_p3_1*1000:+.2f}mm  "
            f"perp_mid={mid_perp_norm_p3_1[0]*1000:.2f}mm "
            f"yaw={yaw_err_p3_1[0]*180/3.14159:+.2f}°  "
            f"xy_latch={int(xy_aligned_latched_p3_1[0])} "
            f"press_latch={int(press_down_latched_p3_1[0])}  "
            f"stall={int(self._stall_frames_p3_1[0])} "
            f"wobble={'YES' if bool(wobble_active_p3_1[0]) else 'no'}"
        )

        rot_action_p3_1     = rot_action_vec_p3_1
        gripper_action_p3_1 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p3_1, rot_action_p3_1, gripper_action_p3_1], dim=-1)

    # ------------------------------------------------------------------
    # plane3 idx=2 — crossbar 2-peg (HELD) ↦ wheel_all 2-hole (FIXED)
    # ------------------------------------------------------------------
    # STRICT independence: every name carries the `_p3_2` suffix. No
    # state or helpers shared with idx=1.
    # Offline mesh analysis:
    #   - crossbar.obj: 2 horizontal cylindrical pegs along held_local x,
    #     tips at x = ±0.038, y = 0, z slightly offset due to a small tilt.
    #     LEFT  peg tip: held_local (-0.038, 0.000, -0.005)
    #     RIGHT peg tip: held_local (+0.038, 0.000, +0.005)
    #     When grasped the bar is rotated so the pegs point DOWN
    #     (held_local x → world -z), so the peg tips are world-bottom.
    #   - wheel_all.obj: 2 vertical through-cylinders along fixed_local z
    #     at x = ±0.0423, y = +0.0006, spanning z ∈ [-0.013, +0.020].
    #     These are the wheel hubs. The TOP rim (z=+0.020) is the insertion
    #     entry face.
    #     LEFT  hole entry: fixed_local (-0.0423, +0.0006, +0.0200)
    #     RIGHT hole entry: fixed_local (+0.0423, +0.0006, +0.0200)
    # User picked #H1 + #H2 (LEFT side of wing). Using the EXACT circle
    # centers from DBSCAN ring detection (no rounding):
    #   #H1 L-back  : fixed_local (-0.0700, -0.0069, -0.0300)  r=0.0020
    #   #H2 L-front : fixed_local (-0.0700, +0.0091, -0.0300)  r=0.0020
    # Δy = 0.0160 m between hole centers.
    # Crossbar's LEFT END has 2 sub-pegs at the SAME x=-0.038, also
    # using circle centers from the yz cross-section DBSCAN:
    #   peg back  (lower z) : held_local (-0.038, 0.000, -0.0128)
    #   peg front (upper z) : held_local (-0.038, 0.000, +0.0028)
    # Δz = 0.0156 m between peg centers (matches Δy=0.016).
    P3_2_PEG_LOCAL_LEFT    = (-0.038, 0.000, -0.0128)     # crossbar L-end back peg CENTER
    P3_2_PEG_LOCAL_RIGHT   = (-0.038, 0.000, +0.0028)     # crossbar L-end front peg CENTER
    P3_2_PEG_SCALE         = 1.2          # crossbar.scale
    P3_2_HOLE_LOCAL_LEFT   = (-0.0700, -0.0069, -0.0300)  # #H1 L-back CENTER
    P3_2_HOLE_LOCAL_RIGHT  = (-0.0700, +0.0091, -0.0300)  # #H2 L-front CENTER
    P3_2_HOLE_SCALE        = 1.2          # fixed_asset.scale
    # All 4 detected wing-top hole CIRCLE CENTERS (DBSCAN raw output):
    P3_2_ALL_WING_HOLES_UNSCALED = [
        (-0.0700, -0.0069, -0.0300, "H1-Lback"),
        (-0.0700, +0.0091, -0.0300, "H2-Lfront"),
        (+0.0700, -0.0069, -0.0300, "H3-Rback"),
        (+0.0700, +0.0091, -0.0300, "H4-Rfront"),
    ]

    def _visualize_two_pegs_two_holes_p3_2(self):
        """Draw GREEN = 2 crossbar peg tips (held drive points),
        RED = 2 wheel_all hole entries (target). Lines connect L↔L and
        R↔R to show alignment direction."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p3_2_legend_printed", False):
            print("\n[plane3 idx=2] viewer legend (crossbar 2 pegs ↦ wheel_all 2 holes):")
            print(f"  GREEN dots = crossbar peg tips (held drive points)")
            print(f"     L: held_local {self.P3_2_PEG_LOCAL_LEFT}  × {self.P3_2_PEG_SCALE}")
            print(f"     R: held_local {self.P3_2_PEG_LOCAL_RIGHT} × {self.P3_2_PEG_SCALE}")
            print(f"  RED   dots = wheel_all hole top rims (fixed targets)")
            print(f"     L: fixed_local {self.P3_2_HOLE_LOCAL_LEFT}  × {self.P3_2_HOLE_SCALE}")
            print(f"     R: fixed_local {self.P3_2_HOLE_LOCAL_RIGHT} × {self.P3_2_HOLE_SCALE}")
            self._p3_2_legend_printed = True

        s_peg = self.P3_2_PEG_SCALE
        s_hole = self.P3_2_HOLE_SCALE
        # 2 crossbar peg tips in world (GREEN drive points).
        peg_L_local = torch.tensor(
            [c * s_peg for c in self.P3_2_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local = torch.tensor(
            [c * s_peg for c in self.P3_2_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_L_local
        )
        _, peg_R_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_R_local
        )
        # 2 wheel_all hole entries in world (RED targets).
        hole_L_local = torch.tensor(
            [c * s_hole for c in self.P3_2_HOLE_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        hole_R_local = torch.tensor(
            [c * s_hole for c in self.P3_2_HOLE_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_L_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_L_local
        )
        _, hole_R_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_R_local
        )

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p3_2(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        gL = _to_tuple_p3_2(peg_L_world[0])
        gR = _to_tuple_p3_2(peg_R_world[0])
        rL = _to_tuple_p3_2(hole_L_world[0])
        rR = _to_tuple_p3_2(hole_R_world[0])
        points = [gL, gR, rL, rR]
        colors = [
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        sizes = [22.0, 22.0, 22.0, 22.0]

        # Also draw all 4 wing holes (#H1..#H4) as small ORANGE dots with
        # N stacked tiny white dots above for numbering. User can compare
        # the 2 chosen RED targets against the 4 candidates.
        for i, cand in enumerate(self.P3_2_ALL_WING_HOLES_UNSCALED):
            lx, ly, lz = cand[0], cand[1], cand[2]
            local_p = torch.tensor(
                [lx * s_hole, ly * s_hole, lz * s_hole],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, world_p = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, local_p
            )
            v = (world_p[0] + env_origin).detach().cpu().numpy()
            wx, wy, wz = float(v[0]), float(v[1]), float(v[2])
            points.append((wx, wy, wz))
            colors.append((1.0, 0.5, 0.0, 1.0))  # ORANGE
            sizes.append(10.0)
            # stacked white dots for number
            for j in range(i + 1):
                points.append((wx, wy, wz + 0.004 + 0.004 * j))
                colors.append((1.0, 1.0, 1.0, 1.0))
                sizes.append(4.0)

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)
        self._dbg_draw.draw_lines(
            [gL, rL],
            [gR, rR],
            [(0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)],
            [3.0, 3.0],
        )

    def _scripted_action_crossbar_insert_p3_2(self):
        """plane3 idx=2 — drive crossbar's two-peg midpoint (GREEN-mid)
        onto wheel_all's two-hole midpoint (RED-mid) in world xy, then
        descend along world -z. Yaw aligns the peg-line direction with
        the hole-line direction. 3-phase latched policy with stall +
        wobble. Written from scratch with `_p3_2` suffix everywhere."""
        # ---- Per-idx state ----
        xy_align_frames_p3_2    = self._xy_align_frames_p3_2
        xy_aligned_latched_p3_2 = self._xy_aligned_latched_p3_2
        near_hover_frames_p3_2  = self._near_hover_frames_p3_2
        press_down_latched_p3_2 = self._press_down_latched_p3_2

        # ---- Tunables ----
        hover_offset_m_p3_2       = 0.05
        align_perp_tol_p3_2       = 0.003
        along_tol_p3_2            = 0.01
        yaw_align_tol_rad_p3_2    = 0.05
        dwell_xy_align_p3_2       = 5
        dwell_to_press_p3_2       = 5
        press_down_depth_p3_2     = 0.20
        descent_scale_p3_2        = 0.3
        press_scale_p3_2          = 0.2
        yaw_align_scale_p3_2      = 0.4
        stall_threshold_m_p3_2    = 0.0003
        stall_dwell_frames_p3_2   = 8
        wobble_amplitude_p3_2     = 0.0015
        wobble_period_frames_p3_2 = 30

        s_peg = self.P3_2_PEG_SCALE
        s_hole = self.P3_2_HOLE_SCALE

        # ---- 2 crossbar peg tips in world ----
        peg_L_local_p3_2 = torch.tensor(
            [c * s_peg for c in self.P3_2_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local_p3_2 = torch.tensor(
            [c * s_peg for c in self.P3_2_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world_p3_2 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_L_local_p3_2
        )
        _, peg_R_world_p3_2 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_R_local_p3_2
        )
        peg_mid_world_p3_2 = 0.5 * (peg_L_world_p3_2 + peg_R_world_p3_2)

        # ---- 2 wheel_all hole entries in world ----
        hole_L_local_p3_2 = torch.tensor(
            [c * s_hole for c in self.P3_2_HOLE_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        hole_R_local_p3_2 = torch.tensor(
            [c * s_hole for c in self.P3_2_HOLE_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_L_world_p3_2 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_L_local_p3_2
        )
        _, hole_R_world_p3_2 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_R_local_p3_2
        )
        hole_mid_world_p3_2 = 0.5 * (hole_L_world_p3_2 + hole_R_world_p3_2)

        # ---- Insertion axis = world +z (descend -z) ----
        axis_t_world_p3_2 = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        drive_world_p3_2  = peg_mid_world_p3_2
        target_world_p3_2 = hole_mid_world_p3_2

        # ---- Targets along axis ----
        above_target_p3_2      = target_world_p3_2 + axis_t_world_p3_2 * hover_offset_m_p3_2
        press_down_target_p3_2 = target_world_p3_2 - axis_t_world_p3_2 * press_down_depth_p3_2

        # ---- xy perp norm ----
        delta_mid_p3_2     = drive_world_p3_2 - target_world_p3_2
        along_mid_p3_2     = (delta_mid_p3_2 * axis_t_world_p3_2).sum(-1, keepdim=True)
        perp_mid_p3_2      = delta_mid_p3_2 - along_mid_p3_2 * axis_t_world_p3_2
        mid_perp_norm_p3_2 = torch.norm(perp_mid_p3_2, dim=-1)

        # ---- Yaw err (cross-product trick on world xy projection) ----
        peg_vec_world_p3_2  = peg_R_world_p3_2 - peg_L_world_p3_2
        hole_vec_world_p3_2 = hole_R_world_p3_2 - hole_L_world_p3_2
        peg_vec_perp_p3_2  = peg_vec_world_p3_2.clone();  peg_vec_perp_p3_2[:, 2] = 0.0
        hole_vec_perp_p3_2 = hole_vec_world_p3_2.clone(); hole_vec_perp_p3_2[:, 2] = 0.0
        cross_p3_2 = torch.cross(peg_vec_perp_p3_2, hole_vec_perp_p3_2, dim=-1)
        sin_yaw_p3_2 = (cross_p3_2 * axis_t_world_p3_2).sum(-1)
        cos_yaw_p3_2 = (peg_vec_perp_p3_2 * hole_vec_perp_p3_2).sum(-1)
        yaw_err_p3_2 = torch.atan2(sin_yaw_p3_2, cos_yaw_p3_2)

        # ---- A1: xy mid + yaw align latch ----
        xy_aligned_p3_2 = (mid_perp_norm_p3_2 < align_perp_tol_p3_2) & (
            torch.abs(yaw_err_p3_2) < yaw_align_tol_rad_p3_2
        )
        xy_align_frames_p3_2 = torch.where(
            xy_aligned_p3_2,
            xy_align_frames_p3_2 + 1,
            torch.zeros_like(xy_align_frames_p3_2),
        )
        xy_aligned_latched_p3_2 = xy_aligned_latched_p3_2 | (
            xy_align_frames_p3_2 >= dwell_xy_align_p3_2
        )

        # ---- A2: near hover above mid ----
        d_above_p3_2     = drive_world_p3_2 - above_target_p3_2
        along_above_p3_2 = torch.abs((d_above_p3_2 * axis_t_world_p3_2).sum(-1))
        near_hover_p3_2 = (
            xy_aligned_latched_p3_2
            & (mid_perp_norm_p3_2 < align_perp_tol_p3_2)
            & (along_above_p3_2 < along_tol_p3_2)
        )
        near_hover_frames_p3_2 = torch.where(
            near_hover_p3_2,
            near_hover_frames_p3_2 + 1,
            torch.zeros_like(near_hover_frames_p3_2),
        )
        press_down_latched_p3_2 = press_down_latched_p3_2 | (
            near_hover_frames_p3_2 >= dwell_to_press_p3_2
        )

        # ---- Write back ----
        self._xy_align_frames_p3_2    = xy_align_frames_p3_2
        self._xy_aligned_latched_p3_2 = xy_aligned_latched_p3_2
        self._near_hover_frames_p3_2  = near_hover_frames_p3_2
        self._press_down_latched_p3_2 = press_down_latched_p3_2

        # ---- Phase split ----
        in_phase_A1_p3_2 = ~xy_aligned_latched_p3_2
        in_phase_A2_p3_2 = xy_aligned_latched_p3_2 & (~press_down_latched_p3_2)
        in_phase_B_p3_2  = press_down_latched_p3_2

        target_p3_2 = torch.where(
            in_phase_B_p3_2.unsqueeze(-1),
            press_down_target_p3_2,
            above_target_p3_2,
        )

        # ---- Raw pos action ----
        delta_p3_2      = target_p3_2 - drive_world_p3_2
        pos_action_p3_2 = delta_p3_2 / self.pos_threshold
        pos_action_p3_2 = torch.clamp(pos_action_p3_2, -1.0, 1.0)

        # A1: zero out along-axis component.
        pa_along_signed_p3_2 = (pos_action_p3_2 * axis_t_world_p3_2).sum(-1, keepdim=True)
        pa_perp_p3_2 = pos_action_p3_2 - pa_along_signed_p3_2 * axis_t_world_p3_2
        pos_action_p3_2 = torch.where(
            in_phase_A1_p3_2.unsqueeze(-1), pa_perp_p3_2, pos_action_p3_2
        )

        # Slower phase B to avoid bounce.
        scale_p3_2 = torch.where(
            in_phase_B_p3_2.unsqueeze(-1),
            torch.full_like(pos_action_p3_2, press_scale_p3_2),
            torch.full_like(pos_action_p3_2, descent_scale_p3_2),
        )
        pos_action_p3_2 = pos_action_p3_2 * scale_p3_2

        # ---- Stall detection + perp-plane wobble ----
        along_now_p3_2 = (drive_world_p3_2 * axis_t_world_p3_2).sum(-1)
        progress_p3_2 = torch.abs(along_now_p3_2 - self._prev_along_p3_2)
        stalled_now_p3_2 = (progress_p3_2 < stall_threshold_m_p3_2) & in_phase_B_p3_2
        self._stall_frames_p3_2 = torch.where(
            stalled_now_p3_2,
            self._stall_frames_p3_2 + 1,
            torch.zeros_like(self._stall_frames_p3_2),
        )
        self._prev_along_p3_2 = along_now_p3_2.clone()
        wobble_active_p3_2 = self._stall_frames_p3_2 >= stall_dwell_frames_p3_2

        if bool(wobble_active_p3_2.any().item()):
            self._wobble_step_p3_2 += 1

        e1_world_p3_2 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p3_2)

        import math as _math
        phase_rad_p3_2 = (
            2.0 * _math.pi * (self._wobble_step_p3_2 / wobble_period_frames_p3_2)
        )
        wobble_offset_m_p3_2 = wobble_amplitude_p3_2 * _math.sin(phase_rad_p3_2)
        wobble_norm_p3_2 = wobble_offset_m_p3_2 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p3_2 = e1_world_p3_2 * wobble_norm_p3_2
        pos_action_p3_2 = torch.where(
            wobble_active_p3_2.unsqueeze(-1),
            pos_action_p3_2 + wobble_vec_p3_2,
            pos_action_p3_2,
        )
        pos_action_p3_2 = torch.clamp(pos_action_p3_2, -1.0, 1.0)

        # ---- Yaw P-control (axis-angle vector form) ----
        rot_action_vec_p3_2 = axis_t_world_p3_2 * yaw_err_p3_2.unsqueeze(-1) * (
            yaw_align_scale_p3_2 / (self.rot_threshold + 1e-8)
        )
        rot_action_vec_p3_2 = torch.clamp(rot_action_vec_p3_2, -1.0, 1.0)
        rot_action_vec_p3_2 = torch.where(
            in_phase_B_p3_2.unsqueeze(-1),
            torch.zeros_like(rot_action_vec_p3_2),
            rot_action_vec_p3_2,
        )

        # ---- Diagnostic print ----
        gL = peg_L_world_p3_2[0].tolist()
        gR = peg_R_world_p3_2[0].tolist()
        rL = hole_L_world_p3_2[0].tolist()
        rR = hole_R_world_p3_2[0].tolist()
        phase = "A1-xy/yaw" if bool(in_phase_A1_p3_2[0]) else (
            "A2-hover" if bool(in_phase_A2_p3_2[0]) else "B-press"
        )
        center_dx_p3_2 = float(perp_mid_p3_2[0, 0])
        center_dy_p3_2 = float(perp_mid_p3_2[0, 1])
        print(
            f"[p3_2] phase={phase}  "
            f"pegL=({gL[0]:+.3f},{gL[1]:+.3f},{gL[2]:+.3f}) "
            f"pegR=({gR[0]:+.3f},{gR[1]:+.3f},{gR[2]:+.3f})  "
            f"holeL=({rL[0]:+.3f},{rL[1]:+.3f},{rL[2]:+.3f}) "
            f"holeR=({rR[0]:+.3f},{rR[1]:+.3f},{rR[2]:+.3f})  "
            f"dx={center_dx_p3_2*1000:+.2f}mm dy={center_dy_p3_2*1000:+.2f}mm  "
            f"perp_mid={mid_perp_norm_p3_2[0]*1000:.2f}mm "
            f"yaw={yaw_err_p3_2[0]*180/3.14159:+.2f}°  "
            f"xy_latch={int(xy_aligned_latched_p3_2[0])} "
            f"press_latch={int(press_down_latched_p3_2[0])}  "
            f"stall={int(self._stall_frames_p3_2[0])} "
            f"wobble={'YES' if bool(wobble_active_p3_2[0]) else 'no'}"
        )

        rot_action_p3_2     = rot_action_vec_p3_2
        gripper_action_p3_2 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p3_2, rot_action_p3_2, gripper_action_p3_2], dim=-1)

    # ------------------------------------------------------------------
    # plane3 idx=3 — crossbar2 2-peg (HELD) ↦ wing RIGHT 2-hole (FIXED)
    # ------------------------------------------------------------------
    # STRICT independence: every name carries the `_p3_3` suffix. No
    # state or helpers shared with idx=1 or idx=2.
    # Offline mesh analysis (same crossbar.obj as idx=2, but using the
    # RIGHT end of the bar):
    #   peg back  (lower z): held_local (+0.038, 0.000, -0.0033)
    #   peg front (upper z): held_local (+0.038, 0.000, +0.0127)
    # Δz = 0.0160 m between peg centers.
    # Wing RIGHT side holes (from DBSCAN ring detection on wheel_all):
    #   #H3 R-back  : fixed_local (+0.0700, -0.0069, -0.0300)
    #   #H4 R-front : fixed_local (+0.0700, +0.0091, -0.0300)
    # Δy = 0.0160 m matches Δz.
    # User confirmed: crossbar2's LEFT end (mesh x<0) is the end with
    # the protruding pegs, same as crossbar1 in idx=2. crossbar2 is just
    # placed at a different scene location pointing toward the wing's
    # right-side holes. So the peg constants here match idx=2's pegs
    # (P1, P2), but the target holes are on the wing's right side.
    P3_3_PEG_LOCAL_LEFT    = (-0.038, +0.0001, -0.0128)   # #P1 L-end back peg CENTER
    P3_3_PEG_LOCAL_RIGHT   = (-0.038, +0.0002, +0.0028)   # #P2 L-end front peg CENTER
    P3_3_PEG_SCALE         = 1.2
    P3_3_HOLE_LOCAL_LEFT   = (+0.0700, -0.0069, -0.0300)  # #H3 R-back CENTER
    P3_3_HOLE_LOCAL_RIGHT  = (+0.0700, +0.0091, -0.0300)  # #H4 R-front CENTER
    P3_3_HOLE_SCALE        = 1.2
    # All 4 detected crossbar peg CENTERS (held_local UNSCALED, r=0.0012):
    #   #P1 L-back  : (-0.038, +0.0001, -0.0128)
    #   #P2 L-front : (-0.038, +0.0002, +0.0028)
    #   #P3 R-back  : (+0.038, +0.0002, -0.0033)
    #   #P4 R-front : (+0.038, -0.0000, +0.0127)
    P3_3_ALL_CROSSBAR_PEGS_UNSCALED = [
        (-0.038, +0.0001, -0.0128, "P1-Lback"),
        (-0.038, +0.0002, +0.0028, "P2-Lfront"),
        (+0.038, +0.0002, -0.0033, "P3-Rback"),
        (+0.038, -0.0000, +0.0127, "P4-Rfront"),
    ]

    def _visualize_two_pegs_two_holes_p3_3(self):
        """Draw GREEN = crossbar2 RIGHT-end 2 sub-peg centers,
        RED = wing RIGHT 2 hole circle centers. Lines connect L↔L, R↔R."""
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_p3_3_legend_printed", False):
            print("\n[plane3 idx=3] viewer legend (crossbar R-end 2 sub-pegs ↦ wing RIGHT 2 holes):")
            print(f"  GREEN dots = crossbar2 right-end peg CENTERS")
            print(f"     L (back) : held_local {self.P3_3_PEG_LOCAL_LEFT}")
            print(f"     R (front): held_local {self.P3_3_PEG_LOCAL_RIGHT}")
            print(f"  RED dots = wing RIGHT hole CENTERS (#H3, #H4)")
            print(f"     L (back) : fixed_local {self.P3_3_HOLE_LOCAL_LEFT}")
            print(f"     R (front): fixed_local {self.P3_3_HOLE_LOCAL_RIGHT}")
            self._p3_3_legend_printed = True

        s_peg = self.P3_3_PEG_SCALE
        s_hole = self.P3_3_HOLE_SCALE
        peg_L_local = torch.tensor(
            [c * s_peg for c in self.P3_3_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local = torch.tensor(
            [c * s_peg for c in self.P3_3_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_L_local
        )
        _, peg_R_world = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_R_local
        )
        hole_L_local = torch.tensor(
            [c * s_hole for c in self.P3_3_HOLE_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        hole_R_local = torch.tensor(
            [c * s_hole for c in self.P3_3_HOLE_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_L_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_L_local
        )
        _, hole_R_world = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_R_local
        )

        env_origin = self.scene.env_origins[0]

        def _to_tuple_p3_3(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        gL = _to_tuple_p3_3(peg_L_world[0])
        gR = _to_tuple_p3_3(peg_R_world[0])
        rL = _to_tuple_p3_3(hole_L_world[0])
        rR = _to_tuple_p3_3(hole_R_world[0])
        points = [gL, gR, rL, rR]
        colors = [
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        sizes = [22.0, 22.0, 22.0, 22.0]

        # Also draw all 4 detected crossbar peg CENTERS (#P1..#P4) as
        # LARGE CYAN dots + a YELLOW line along each peg's local x axis
        # from the bar-shoulder (x=∓0.027) out to the detected tip
        # (x=∓0.038). Stacked white dots above each CYAN dot encode the
        # peg #P (count = 1..4).
        cand_line_a, cand_line_b, cand_line_c, cand_line_w = [], [], [], []
        for i, cand in enumerate(self.P3_3_ALL_CROSSBAR_PEGS_UNSCALED):
            tx, ty, tz = cand[0], cand[1], cand[2]
            # base point at the bar shoulder (same y/z, x retracted to ±0.025)
            bx = -0.025 if tx < 0 else +0.025
            by, bz = ty, tz
            tip_local = torch.tensor(
                [tx * s_peg, ty * s_peg, tz * s_peg],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            base_local = torch.tensor(
                [bx * s_peg, by * s_peg, bz * s_peg],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, tip_world = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, tip_local
            )
            _, base_world = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, base_local
            )
            tip_v = (tip_world[0] + env_origin).detach().cpu().numpy()
            base_v = (base_world[0] + env_origin).detach().cpu().numpy()
            tw = (float(tip_v[0]), float(tip_v[1]), float(tip_v[2]))
            bw = (float(base_v[0]), float(base_v[1]), float(base_v[2]))
            # CYAN tip dot
            points.append(tw)
            colors.append((0.0, 1.0, 1.0, 1.0))
            sizes.append(18.0)
            # number = stacked white dots above tip
            for j in range(i + 1):
                points.append((tw[0], tw[1], tw[2] + 0.004 + 0.004 * j))
                colors.append((1.0, 1.0, 1.0, 1.0))
                sizes.append(5.0)
            # YELLOW line from base→tip showing peg axis
            cand_line_a.append(bw)
            cand_line_b.append(tw)
            cand_line_c.append((1.0, 1.0, 0.0, 1.0))
            cand_line_w.append(2.5)

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()
        self._dbg_draw.draw_points(points, colors, sizes)
        line_a = [gL, gR] + cand_line_a
        line_b = [rL, rR] + cand_line_b
        line_c = [(0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)] + cand_line_c
        line_w = [3.0, 3.0] + cand_line_w
        self._dbg_draw.draw_lines(line_a, line_b, line_c, line_w)

    def _scripted_action_crossbar_insert_p3_3(self):
        """plane3 idx=3 — drive crossbar2's RIGHT-end 2-peg midpoint
        onto wing's RIGHT 2-hole midpoint in world xy, yaw-align the
        peg-line direction to the hole-line direction, then descend
        along world -z. Latched 3-phase policy with stall + wobble.
        Written from scratch with `_p3_3` suffix — no shared state."""
        xy_align_frames_p3_3    = self._xy_align_frames_p3_3
        xy_aligned_latched_p3_3 = self._xy_aligned_latched_p3_3
        near_hover_frames_p3_3  = self._near_hover_frames_p3_3
        press_down_latched_p3_3 = self._press_down_latched_p3_3

        hover_offset_m_p3_3       = 0.05
        align_perp_tol_p3_3       = 0.003
        along_tol_p3_3            = 0.01
        yaw_align_tol_rad_p3_3    = 0.05
        dwell_xy_align_p3_3       = 5
        dwell_to_press_p3_3       = 5
        press_down_depth_p3_3     = 0.20
        descent_scale_p3_3        = 0.3
        press_scale_p3_3          = 0.2
        yaw_align_scale_p3_3      = 0.4
        stall_threshold_m_p3_3    = 0.0003
        stall_dwell_frames_p3_3   = 8
        wobble_amplitude_p3_3     = 0.0015
        wobble_period_frames_p3_3 = 30

        s_peg = self.P3_3_PEG_SCALE
        s_hole = self.P3_3_HOLE_SCALE

        peg_L_local_p3_3 = torch.tensor(
            [c * s_peg for c in self.P3_3_PEG_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        peg_R_local_p3_3 = torch.tensor(
            [c * s_peg for c in self.P3_3_PEG_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, peg_L_world_p3_3 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_L_local_p3_3
        )
        _, peg_R_world_p3_3 = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.identity_quat, peg_R_local_p3_3
        )
        peg_mid_world_p3_3 = 0.5 * (peg_L_world_p3_3 + peg_R_world_p3_3)

        hole_L_local_p3_3 = torch.tensor(
            [c * s_hole for c in self.P3_3_HOLE_LOCAL_LEFT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        hole_R_local_p3_3 = torch.tensor(
            [c * s_hole for c in self.P3_3_HOLE_LOCAL_RIGHT],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        _, hole_L_world_p3_3 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_L_local_p3_3
        )
        _, hole_R_world_p3_3 = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, hole_R_local_p3_3
        )
        hole_mid_world_p3_3 = 0.5 * (hole_L_world_p3_3 + hole_R_world_p3_3)

        axis_t_world_p3_3 = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        drive_world_p3_3  = peg_mid_world_p3_3
        target_world_p3_3 = hole_mid_world_p3_3

        above_target_p3_3      = target_world_p3_3 + axis_t_world_p3_3 * hover_offset_m_p3_3
        press_down_target_p3_3 = target_world_p3_3 - axis_t_world_p3_3 * press_down_depth_p3_3

        delta_mid_p3_3     = drive_world_p3_3 - target_world_p3_3
        along_mid_p3_3     = (delta_mid_p3_3 * axis_t_world_p3_3).sum(-1, keepdim=True)
        perp_mid_p3_3      = delta_mid_p3_3 - along_mid_p3_3 * axis_t_world_p3_3
        mid_perp_norm_p3_3 = torch.norm(perp_mid_p3_3, dim=-1)

        peg_vec_world_p3_3  = peg_R_world_p3_3 - peg_L_world_p3_3
        hole_vec_world_p3_3 = hole_R_world_p3_3 - hole_L_world_p3_3
        peg_vec_perp_p3_3  = peg_vec_world_p3_3.clone();  peg_vec_perp_p3_3[:, 2] = 0.0
        hole_vec_perp_p3_3 = hole_vec_world_p3_3.clone(); hole_vec_perp_p3_3[:, 2] = 0.0
        cross_p3_3 = torch.cross(peg_vec_perp_p3_3, hole_vec_perp_p3_3, dim=-1)
        sin_yaw_p3_3 = (cross_p3_3 * axis_t_world_p3_3).sum(-1)
        cos_yaw_p3_3 = (peg_vec_perp_p3_3 * hole_vec_perp_p3_3).sum(-1)
        yaw_err_p3_3 = torch.atan2(sin_yaw_p3_3, cos_yaw_p3_3)

        xy_aligned_p3_3 = (mid_perp_norm_p3_3 < align_perp_tol_p3_3) & (
            torch.abs(yaw_err_p3_3) < yaw_align_tol_rad_p3_3
        )
        xy_align_frames_p3_3 = torch.where(
            xy_aligned_p3_3,
            xy_align_frames_p3_3 + 1,
            torch.zeros_like(xy_align_frames_p3_3),
        )
        xy_aligned_latched_p3_3 = xy_aligned_latched_p3_3 | (
            xy_align_frames_p3_3 >= dwell_xy_align_p3_3
        )

        d_above_p3_3     = drive_world_p3_3 - above_target_p3_3
        along_above_p3_3 = torch.abs((d_above_p3_3 * axis_t_world_p3_3).sum(-1))
        near_hover_p3_3 = (
            xy_aligned_latched_p3_3
            & (mid_perp_norm_p3_3 < align_perp_tol_p3_3)
            & (along_above_p3_3 < along_tol_p3_3)
        )
        near_hover_frames_p3_3 = torch.where(
            near_hover_p3_3,
            near_hover_frames_p3_3 + 1,
            torch.zeros_like(near_hover_frames_p3_3),
        )
        press_down_latched_p3_3 = press_down_latched_p3_3 | (
            near_hover_frames_p3_3 >= dwell_to_press_p3_3
        )

        self._xy_align_frames_p3_3    = xy_align_frames_p3_3
        self._xy_aligned_latched_p3_3 = xy_aligned_latched_p3_3
        self._near_hover_frames_p3_3  = near_hover_frames_p3_3
        self._press_down_latched_p3_3 = press_down_latched_p3_3

        in_phase_A1_p3_3 = ~xy_aligned_latched_p3_3
        in_phase_A2_p3_3 = xy_aligned_latched_p3_3 & (~press_down_latched_p3_3)
        in_phase_B_p3_3  = press_down_latched_p3_3

        target_p3_3 = torch.where(
            in_phase_B_p3_3.unsqueeze(-1),
            press_down_target_p3_3,
            above_target_p3_3,
        )

        delta_p3_3      = target_p3_3 - drive_world_p3_3
        pos_action_p3_3 = delta_p3_3 / self.pos_threshold
        pos_action_p3_3 = torch.clamp(pos_action_p3_3, -1.0, 1.0)

        pa_along_signed_p3_3 = (pos_action_p3_3 * axis_t_world_p3_3).sum(-1, keepdim=True)
        pa_perp_p3_3 = pos_action_p3_3 - pa_along_signed_p3_3 * axis_t_world_p3_3
        pos_action_p3_3 = torch.where(
            in_phase_A1_p3_3.unsqueeze(-1), pa_perp_p3_3, pos_action_p3_3
        )

        scale_p3_3 = torch.where(
            in_phase_B_p3_3.unsqueeze(-1),
            torch.full_like(pos_action_p3_3, press_scale_p3_3),
            torch.full_like(pos_action_p3_3, descent_scale_p3_3),
        )
        pos_action_p3_3 = pos_action_p3_3 * scale_p3_3

        along_now_p3_3 = (drive_world_p3_3 * axis_t_world_p3_3).sum(-1)
        progress_p3_3 = torch.abs(along_now_p3_3 - self._prev_along_p3_3)
        stalled_now_p3_3 = (progress_p3_3 < stall_threshold_m_p3_3) & in_phase_B_p3_3
        self._stall_frames_p3_3 = torch.where(
            stalled_now_p3_3,
            self._stall_frames_p3_3 + 1,
            torch.zeros_like(self._stall_frames_p3_3),
        )
        self._prev_along_p3_3 = along_now_p3_3.clone()
        wobble_active_p3_3 = self._stall_frames_p3_3 >= stall_dwell_frames_p3_3

        if bool(wobble_active_p3_3.any().item()):
            self._wobble_step_p3_3 += 1

        e1_world_p3_3 = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device,
        ).expand_as(axis_t_world_p3_3)

        import math as _math
        phase_rad_p3_3 = (
            2.0 * _math.pi * (self._wobble_step_p3_3 / wobble_period_frames_p3_3)
        )
        wobble_offset_m_p3_3 = wobble_amplitude_p3_3 * _math.sin(phase_rad_p3_3)
        wobble_norm_p3_3 = wobble_offset_m_p3_3 / float(self.pos_threshold[0, 0].item())
        wobble_vec_p3_3 = e1_world_p3_3 * wobble_norm_p3_3
        pos_action_p3_3 = torch.where(
            wobble_active_p3_3.unsqueeze(-1),
            pos_action_p3_3 + wobble_vec_p3_3,
            pos_action_p3_3,
        )
        pos_action_p3_3 = torch.clamp(pos_action_p3_3, -1.0, 1.0)

        rot_action_vec_p3_3 = axis_t_world_p3_3 * yaw_err_p3_3.unsqueeze(-1) * (
            yaw_align_scale_p3_3 / (self.rot_threshold + 1e-8)
        )
        rot_action_vec_p3_3 = torch.clamp(rot_action_vec_p3_3, -1.0, 1.0)
        rot_action_vec_p3_3 = torch.where(
            in_phase_B_p3_3.unsqueeze(-1),
            torch.zeros_like(rot_action_vec_p3_3),
            rot_action_vec_p3_3,
        )

        gL = peg_L_world_p3_3[0].tolist()
        gR = peg_R_world_p3_3[0].tolist()
        rL = hole_L_world_p3_3[0].tolist()
        rR = hole_R_world_p3_3[0].tolist()
        phase = "A1-xy/yaw" if bool(in_phase_A1_p3_3[0]) else (
            "A2-hover" if bool(in_phase_A2_p3_3[0]) else "B-press"
        )
        center_dx_p3_3 = float(perp_mid_p3_3[0, 0])
        center_dy_p3_3 = float(perp_mid_p3_3[0, 1])
        print(
            f"[p3_3] phase={phase}  "
            f"pegL=({gL[0]:+.3f},{gL[1]:+.3f},{gL[2]:+.3f}) "
            f"pegR=({gR[0]:+.3f},{gR[1]:+.3f},{gR[2]:+.3f})  "
            f"holeL=({rL[0]:+.3f},{rL[1]:+.3f},{rL[2]:+.3f}) "
            f"holeR=({rR[0]:+.3f},{rR[1]:+.3f},{rR[2]:+.3f})  "
            f"dx={center_dx_p3_3*1000:+.2f}mm dy={center_dy_p3_3*1000:+.2f}mm  "
            f"perp_mid={mid_perp_norm_p3_3[0]*1000:.2f}mm "
            f"yaw={yaw_err_p3_3[0]*180/3.14159:+.2f}°  "
            f"xy_latch={int(xy_aligned_latched_p3_3[0])} "
            f"press_latch={int(press_down_latched_p3_3[0])}  "
            f"stall={int(self._stall_frames_p3_3[0])} "
            f"wobble={'YES' if bool(wobble_active_p3_3[0]) else 'no'}"
        )

        rot_action_p3_3     = rot_action_vec_p3_3
        gripper_action_p3_3 = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action_p3_3, rot_action_p3_3, gripper_action_p3_3], dim=-1)

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        self._visualize_markers()
        if self.cfg_task.task_idx == 1:
            # Normal 2-hole + 2-peg viz (4 dots + 2 lines).
            self._visualize_two_pegs_two_holes_p3_1()
        elif self.cfg_task.task_idx == 2:
            self._visualize_two_pegs_two_holes_p3_2()
        elif self.cfg_task.task_idx == 3:
            self._visualize_two_pegs_two_holes_p3_3()
        self._check_attach_condition()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
        if self.cfg_task.task_idx == 1:
            self.actions = self._scripted_action_two_peg_insert_p3_1()
        elif self.cfg_task.task_idx == 2:
            self.actions = self._scripted_action_crossbar_insert_p3_2()
        elif self.cfg_task.task_idx == 3:
            self.actions = self._scripted_action_crossbar_insert_p3_3()
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

        # Reset scripted-policy state for plane3 idx=1 (independent of any
        # other idx — only zeroes the `_p3_1` tensors).
        self._xy_align_frames_p3_1.zero_()
        self._xy_aligned_latched_p3_1.zero_()
        self._near_hover_frames_p3_1.zero_()
        self._press_down_latched_p3_1.zero_()
        self._prev_along_p3_1.zero_()
        self._stall_frames_p3_1.zero_()
        self._wobble_step_p3_1 = 0

        # Reset scripted-policy state for plane3 idx=2 (independent —
        # only zeroes the `_p3_2` tensors, no overlap with `_p3_1`).
        self._xy_align_frames_p3_2.zero_()
        self._xy_aligned_latched_p3_2.zero_()
        self._near_hover_frames_p3_2.zero_()
        self._press_down_latched_p3_2.zero_()
        self._prev_along_p3_2.zero_()
        self._stall_frames_p3_2.zero_()
        self._wobble_step_p3_2 = 0

        # Reset scripted-policy state for plane3 idx=3 (independent —
        # only zeroes the `_p3_3` tensors, no overlap with `_p3_1/_p3_2`).
        self._xy_align_frames_p3_3.zero_()
        self._xy_aligned_latched_p3_3.zero_()
        self._near_hover_frames_p3_3.zero_()
        self._press_down_latched_p3_3.zero_()
        self._prev_along_p3_3.zero_()
        self._stall_frames_p3_3.zero_()
        self._wobble_step_p3_3 = 0

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()
        self.randomize_initial_state(env_ids)

        # ---- Calibration: at reset the user reports the 2 peg-tip world
        # positions coincide with the 2 body3 hole world positions. So
        # back-solve the hole positions in held_local from peg world pos
        # using the inverse held_quat. Print once per reset so user can
        # paste the values directly into HOLE_LOCAL_LEFT/RIGHT.
        if self.cfg_task.task_idx == 1:
            self._calibrate_holes_from_pegs_p3_1()


    def _calibrate_holes_from_pegs_p3_1(self):
        """At reset time, assume peg-tip world positions = hole-center
        world positions, then back-solve hole positions in held_local
        (UNSCALED). Write the result to /tmp/p3_1_calibration.txt AND
        print to console so the value can't be lost in scroll-back."""
        # Refresh tensors so fixed_pos / held_pos reflect post-reset state.
        self._compute_intermediate_values(dt=self.physics_dt)

        s_peg = self.P3_1_PEG_SCALE
        s_hole = self.P3_1_HOLE_SCALE
        results = {}

        for label, peg_local in [("LEFT",  self.P3_1_PEG_LOCAL_LEFT),
                                  ("RIGHT", self.P3_1_PEG_LOCAL_RIGHT)]:
            peg_local_scaled = torch.tensor(
                [c * s_peg for c in peg_local],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, peg_world = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, peg_local_scaled
            )

            # Inverse transform peg world position into held_local frame.
            offset = peg_world - self.held_pos
            held_quat_inv = torch_utils.quat_conjugate(self.held_quat)
            held_local_scaled = torch_utils.quat_apply(held_quat_inv, offset)
            held_local_unscaled = held_local_scaled / s_hole

            x, y, z = (
                float(held_local_unscaled[0, 0]),
                float(held_local_unscaled[0, 1]),
                float(held_local_unscaled[0, 2]),
            )
            results[label] = (x, y, z)

        banner = "=" * 78
        msg_lines = [
            banner,
            "[plane3 idx=1 CALIBRATION] peg world → held_local UNSCALED",
            f"  P3_1_HOLE_LOCAL_LEFT  = ({results['LEFT'][0]:+.5f}, "
            f"{results['LEFT'][1]:+.5f}, {results['LEFT'][2]:+.5f})",
            f"  P3_1_HOLE_LOCAL_RIGHT = ({results['RIGHT'][0]:+.5f}, "
            f"{results['RIGHT'][1]:+.5f}, {results['RIGHT'][2]:+.5f})",
            banner,
        ]
        for line in msg_lines:
            print(line, flush=True)

        # Persist to a file so user can find it after the simulation has
        # spammed output.
        try:
            with open("/tmp/p3_1_calibration.txt", "w") as f:
                for line in msg_lines:
                    f.write(line + "\n")
        except Exception as e:
            print(f"[CALIBRATION write error] {e!r}", flush=True)


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
            held_asset_relative_pos[:, 2] += 0.07
            held_asset_relative_pos[:, 1] += 0.0
            held_asset_relative_pos[:, 0] += 0.04

        elif self.cfg_task.name == "plane_assembly" and self.cfg_task.task_idx in [2]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.
            held_asset_relative_pos[:, 1] += 0.0
            held_asset_relative_pos[:, 0] -= 0.0

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
            rela_trans[:, 2] -= 0.45
            rela_trans[:, 1] += 0.1
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
            rela_trans[:, 2] -= 0.45
            rela_trans[:, 1] -= 0.1
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

        if self.cfg_task.task_idx == 1:
            rot_euler = torch.tensor([0, 3.1415, -1.5707], device=self.device).repeat(
            self.num_envs, 1
            )
            translated_held_asset_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        if self.cfg_task.task_idx in [2, 3]:
            rot_euler = torch.tensor([0.0, -1.5707, 0], device=self.device).repeat(
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