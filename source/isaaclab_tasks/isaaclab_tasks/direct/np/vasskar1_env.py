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
from .np_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FrankaVasskar1Cfg
from pdb import set_trace as bp
from .np_utils.group_utils import SE3dist
from .np_utils.viz_utils import define_markers
from scipy.spatial.transform import Rotation as R
import torch
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf, Gf, Tf
from omni.physx.scripts import utils
import omni.usd

class FrankaVasskar1Env(DirectRLEnv):
    cfg: FrankaVasskar1Cfg

    def __init__(self, cfg: FrankaVasskar1Cfg, render_mode: str | None = None, **kwargs):
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

        # All scene surfaces frictionless for the spiral search to slide
        # without binding (gripper still holds via robot 5.0 friction).
        if self.cfg_task.task_idx == 1:
            self._set_friction(self._fixed_asset, 0.0)
            self._set_friction(self._held_asset, 0.0)
        if self.cfg_task.task_idx == 2:
            self._set_friction(self._fixed_asset, 0.0)
            self._set_friction(self._held_asset, 0.0)
        if self.cfg_task.task_idx == 3:
            # idx=3 contact happens between the held lid's slots and the
            # top1/top2 leg tops — zero ALL of them so the search/overshoot
            # phase can slide laterally without binding.
            self._set_friction(self._fixed_asset, 0.0)
            self._set_friction(self._held_asset, 0.0)
            self._set_friction(self._top1, 0.0)
            self._set_friction(self._top2, 0.0)

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

        if self.cfg_task.name == "vasskar_assembly":
            self.fixed_success_pos_local[:, 2] = 0.0
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Scripted-policy state machine (idx=1 top-frame insertion only).
        self._near_hole_frames = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._search_active_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._search_step = 0
        self._press_down_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Contact detection during deliberate-offset press-down: track peg z and
        # count frames where descent stalls — once it stalls long enough we
        # latch _search_active_latched and start sliding back toward the hole.
        self._prev_peg_z = torch.zeros((self.num_envs,), device=self.device)
        self._contact_frames = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # -y overshoot during slide-back: once peg reaches the overshoot
        # position past the true hole, wait N frames, then switch to a deeper
        # final press-down target.
        self._overshoot_reached_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._overshoot_wait_frames = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._final_press_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Scripted-policy state machine (idx=2 second top-frame insertion).
        # Mirrored from idx=1 but kept fully independent — no shared tensors.
        self._near_hole_frames_2 = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._search_active_latched_2 = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._search_step_2 = 0
        self._press_down_latched_2 = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._prev_peg_z_2 = torch.zeros((self.num_envs,), device=self.device)
        self._contact_frames_2 = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._overshoot_reached_latched_2 = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._overshoot_wait_frames_2 = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._final_press_latched_2 = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

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
            self._top1 = RigidObject(self.cfg_task.top1)
            self._held_asset = self._top1
            self._connection_cfg = self.cfg_task.connection_cfg1

        if self.cfg_task.task_idx ==2:
            self._top1 = RigidObject(self.cfg_task.top1)
            self._top2 = RigidObject(self.cfg_task.top2)
            self._held_asset = self._top2
            self._connection_cfg = self.cfg_task.connection_cfg2

        if self.cfg_task.task_idx ==3:
            self._top1 = RigidObject(self.cfg_task.top1)
            self._top2 = RigidObject(self.cfg_task.top2)
            self._frame2 = RigidObject(self.cfg_task.frame2)
            self._held_asset = self._frame2
            self._connection_cfg = self.cfg_task.connection_cfg3
        

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
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/TopFrame1")
            joint_path = "/World/envs/env_0/FixedJoint1"
            connection_cfg = self.cfg_task.connection_cfg1_fix
        elif connection_idx == 2:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/TopFrame2")
            joint_path = "/World/envs/env_0/FixedJoint2"
            connection_cfg = self.cfg_task.connection_cfg2_fix

        elif connection_idx == 3:
            held_prim = stage.GetPrimAtPath("/World/envs/env_0/SideFrame")
            joint_path = "/World/envs/env_0/FixedJoint3"
            connection_cfg = self.cfg_task.connection_cfg3

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
        # idx=3: skip auto-attach entirely. The cfg target pose snaps the
        # lid to z=1.172 (world) which is ~12 cm above the leg tops — that
        # locks the lid floating, not seated. Instead let physics + the
        # policy's continuous downward force settle the lid naturally on
        # the legs without ever creating a USD FixedJoint.
        if self.cfg_task.task_idx == 3:
            return

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
        if not self.joint_created and R_dist < 0.1 and t_tangent < 0.005 and t_normal < 0.002:
            if self.cfg_task.task_idx == 3:
                self._create_fixed_joint(connection_idx=self.cfg_task.task_idx)
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
        loc = self.marker_locations
        rots = self.marker_orientations

        # render the markers
        all_envs = torch.arange(self.num_envs)
        indices = torch.zeros_like(all_envs)
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _visualize_candidates(
        self,
        all_holes_world,      # list of 4 tensors (1 per hole), each (num_envs, 3)
        all_pegs_world,       # list of 4 tensors (1 per top corner / peg candidate)
        chosen_pegs_world,    # list of 2 tensors — currently selected pegs
        chosen_holes_world,   # list of 2 tensors — currently selected holes
        peg_mid_world,
        target_mid_world,
        all_slots_world=None, # optional 3rd group (e.g. held-asset slots for idx=3)
    ):
        """Render every candidate hole/peg labeled by a vertical stack of
        small "tick" dots above its main marker. Plus the currently chosen
        2 pegs / 2 holes drawn larger for visibility. Console legend printed
        once at first call.

        Labels (per main marker, stacked vertically above by 0.025+0.012*j):
          1 tick  = #0     2 ticks = #1     3 ticks = #2     4 ticks = #3
        """
        if not hasattr(self, "_dbg_draw"):
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw
            self._dbg_draw = _debug_draw.acquire_debug_draw_interface()

        if not getattr(self, "_legend_printed", False):
            print("\n[vasskar1] viewer legend (each label = N ticks stacked above main dot)")
            print("  Base holes — frame local coords:")
            print("    #0 (1 tick)  RED     at (0.010, 0.040, 0.000)")
            print("    #1 (2 ticks) ORANGE  at (0.010, 0.220, 0.000)")
            print("    #2 (3 ticks) YELLOW  at (0.290, 0.040, 0.000)")
            print("    #3 (4 ticks) PINK    at (0.290, 0.220, 0.000)")
            print("  Top1 corners — corner-cluster 3D centroids (unscaled):")
            print("    A (1 tick)  MAGENTA at (0.0025, 0.010, -0.010)")
            print("    B (2 ticks) CYAN    at (0.5625, 0.010, -0.010)")
            print("    C (3 ticks) LIME    at (0.0025, 0.290, -0.010)")
            print("    D (4 ticks) PURPLE  at (0.5625, 0.290, -0.010)")
            print("  CURRENTLY CHOSEN: large RED = active pegs, large GREEN = active holes")
            self._legend_printed = True

        self._dbg_draw.clear_points()
        self._dbg_draw.clear_lines()

        env_origin = self.scene.env_origins[0]

        def to_tuple(t):
            v = (t + env_origin).detach().cpu().numpy()
            return (float(v[0]), float(v[1]), float(v[2]))

        points, colors, sizes = [], [], []

        # ---- chosen pegs (large red) + chosen holes (large green) ----
        for p in chosen_pegs_world:
            points.append(to_tuple(p[0]))
            colors.append((1.0, 0.0, 0.0, 1.0))
            sizes.append(24.0)
        for h in chosen_holes_world:
            points.append(to_tuple(h[0]))
            colors.append((0.0, 1.0, 0.0, 1.0))
            sizes.append(24.0)
        # mid points
        points.append(to_tuple(peg_mid_world[0]))
        colors.append((1.0, 0.5, 0.0, 1.0))
        sizes.append(14.0)
        points.append(to_tuple(target_mid_world[0]))
        colors.append((0.0, 1.0, 1.0, 1.0))
        sizes.append(14.0)

        HOLE_COLORS = [
            (1.0, 0.2, 0.2, 1.0),  # #0 red
            (1.0, 0.6, 0.1, 1.0),  # #1 orange
            (1.0, 1.0, 0.2, 1.0),  # #2 yellow
            (1.0, 0.4, 0.7, 1.0),  # #3 pink
        ]
        PEG_COLORS = [
            (1.0, 0.0, 1.0, 1.0),  # A magenta
            (0.0, 1.0, 1.0, 1.0),  # B cyan
            (0.5, 1.0, 0.0, 1.0),  # C lime
            (0.6, 0.0, 1.0, 1.0),  # D purple
        ]

        # ---- 4 hole candidates with tick labels ----
        for i, hw in enumerate(all_holes_world):
            base = hw[0]
            points.append(to_tuple(base))
            colors.append(HOLE_COLORS[i])
            sizes.append(16.0)
            for j in range(i + 1):
                tick = base.clone()
                tick[2] = tick[2] + 0.025 + j * 0.012
                points.append(to_tuple(tick))
                colors.append(HOLE_COLORS[i])
                sizes.append(8.0)

        # ---- 4 top-corner candidates with tick labels ----
        for i, pw in enumerate(all_pegs_world):
            base = pw[0]
            points.append(to_tuple(base))
            colors.append(PEG_COLORS[i])
            sizes.append(16.0)
            for j in range(i + 1):
                tick = base.clone()
                tick[2] = tick[2] + 0.025 + j * 0.012
                points.append(to_tuple(tick))
                colors.append(PEG_COLORS[i])
                sizes.append(8.0)

        # ---- optional 3rd group: held-asset slots with tick labels ----
        if all_slots_world is not None:
            SLOT_COLORS = [
                (0.2, 0.4, 1.0, 1.0),   # blue
                (0.0, 0.8, 0.8, 1.0),   # teal
                (0.7, 0.7, 1.0, 1.0),   # light blue
                (0.4, 0.2, 0.8, 1.0),   # indigo
            ]
            for i, sw in enumerate(all_slots_world):
                base = sw[0]
                points.append(to_tuple(base))
                colors.append(SLOT_COLORS[i])
                sizes.append(16.0)
                for j in range(i + 1):
                    tick = base.clone()
                    tick[2] = tick[2] + 0.025 + j * 0.012
                    points.append(to_tuple(tick))
                    colors.append(SLOT_COLORS[i])
                    sizes.append(8.0)

        self._dbg_draw.draw_points(points, colors, sizes)

        # ---- alignment lines: chosen pegs ↔ chosen holes (yellow) + mid line (orange) ----
        if len(chosen_pegs_world) == 2 and len(chosen_holes_world) == 2:
            line_starts = [
                to_tuple(chosen_pegs_world[0][0]),
                to_tuple(chosen_pegs_world[1][0]),
                to_tuple(peg_mid_world[0]),
            ]
            line_ends = [
                to_tuple(chosen_holes_world[0][0]),
                to_tuple(chosen_holes_world[1][0]),
                to_tuple(target_mid_world[0]),
            ]
            self._dbg_draw.draw_lines(
                line_starts,
                line_ends,
                [
                    (1.0, 1.0, 0.0, 1.0),
                    (1.0, 1.0, 0.0, 1.0),
                    (1.0, 0.5, 0.0, 1.0),
                ],
                [3.0, 3.0, 2.0],
            )

    def _compute_scripted_action(self):
        """Router — each task_idx has its own INDEPENDENT scripted policy."""
        if self.cfg_task.task_idx == 1:
            return self._scripted_action_top1_to_frame()
        if self.cfg_task.task_idx == 2:
            return self._scripted_action_top2_to_frame()
        if self.cfg_task.task_idx == 3:
            return self._scripted_action_side_to_frame()
        return None

    def _scripted_action_top1_to_frame(self):
        """Top-frame 1 → vasskar base (idx 1) — 2-peg / 2-hole alignment,
        same state machine as chair3 rod_to_frame. Visualizes ALL 4 base
        holes and ALL 4 top1 corners with tick labels so the user can pick
        which 2 of each to use.

        Configure via ACTIVE_HOLE_INDICES and ACTIVE_PEG_INDICES below.
        """
        # =============================================================
        # USER-SELECTABLE INDICES — change these once you see the labels.
        # ACTIVE_HOLE_INDICES picks 2 entries from all_hole_locals (#0..#3)
        # ACTIVE_PEG_INDICES  picks 2 entries from all_peg_top_locals (A..D)
        # =============================================================
        ACTIVE_HOLE_INDICES = (0, 2)   # red(#0, 1 tick) + yellow(#2, 3 ticks) — user-confirmed
        ACTIVE_PEG_INDICES  = (0, 2)   # magenta(A) → red(#0), lime(C) → yellow(#2)

        # =============================================================
        # 1) ALL 4 base-hole candidates in frame local
        # =============================================================
        all_hole_locals_np = [
            [0.010, 0.040, 0.000],   # #0
            [0.010, 0.220, 0.000],   # #1
            [0.290, 0.040, 0.000],   # #2
            [0.290, 0.220, 0.000],   # #3
        ]
        all_holes_world = []
        for hl in all_hole_locals_np:
            hl_t = torch.tensor(hl, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            _, hw = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, hl_t
            )
            all_holes_world.append(hw)

        # =============================================================
        # 2) ALL 4 top1-corner candidates in held local (apply scale)
        # =============================================================
        top_scale = torch.as_tensor(
            self.cfg_task.top1.spawn.scale, dtype=torch.float32, device=self.device
        )
        sx_, sy_, sz_ = float(top_scale[0]), float(top_scale[1]), float(top_scale[2])
        # Corner-cluster 3D bbox centers from top_re.obj inspection (not edge
        # points). Each corner spans roughly 5–15 mm in x, 20 mm in y, 20 mm
        # in z — these are the geometric centers of those clusters.
        all_peg_top_locals_unscaled = [
            [0.0025, 0.010, -0.010],   # A — corner near (0, 0)
            [0.5625, 0.010, -0.010],   # B — corner near (0.57, 0)
            [0.0025, 0.290, -0.010],   # C — corner near (0, 0.3)
            [0.5625, 0.290, -0.010],   # D — corner near (0.57, 0.3)
        ]
        all_pegs_world = []
        for pu in all_peg_top_locals_unscaled:
            pl = torch.tensor(
                [pu[0] * sx_, pu[1] * sy_, pu[2] * sz_], device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, pw = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, pl
            )
            all_pegs_world.append(pw)

        # =============================================================
        # 3) Pick the ACTIVE pair for the policy
        # =============================================================
        hole1_world = all_holes_world[ACTIVE_HOLE_INDICES[0]]
        hole2_world = all_holes_world[ACTIVE_HOLE_INDICES[1]]
        peg1_world  = all_pegs_world [ACTIVE_PEG_INDICES[0]]
        peg2_world  = all_pegs_world [ACTIVE_PEG_INDICES[1]]
        hole_mid_world = (hole1_world + hole2_world) / 2.0
        peg_mid_world  = (peg1_world  + peg2_world ) / 2.0

        # =============================================================
        # 4) Insertion axis in world
        # =============================================================
        axis_t_local = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t = torch.zeros_like(self.fixed_pos)
        _, axis_t_world = torch_utils.tf_combine(
            self.fixed_quat, zero_t, self.identity_quat, axis_t_local
        )

        # =============================================================
        # 5) Tunables — 2-phase direct insertion: approach above hole_mid,
        # then press straight down. No deliberate miss, no overshoot,
        # no slide-back search.
        # =============================================================
        approach_height   = 0.10   # 100 mm above hole_mid along -axis_t
        align_perp_tol    = 0.005  # 5 mm xy residual to latch press
        align_along_tol   = 0.02   # 20 mm z residual to latch press
        dwell_to_press    = 3      # frames near above_target before press
        descent_scale     = 0.3
        press_down_scale  = 0.2
        final_press_depth = 0.10   # 100 mm into base
        # NOTE on sign: vasskar1 fixed init rot = 180° about y, so
        # axis_t = (0,0,1)_local → (0,0,-1)_world (points DOWN). Hence
        # "above" = hole_mid - axis_t * h, "press" = hole_mid + axis_t * d.

        # =============================================================
        # 6) Targets
        # =============================================================
        above_target       = hole_mid_world - axis_t_world * approach_height
        final_press_target = hole_mid_world + axis_t_world * final_press_depth

        # =============================================================
        # 7) State machine: phase A approach → phase B straight press.
        # =============================================================
        delta_pm = peg_mid_world - above_target
        along_pm = (delta_pm * axis_t_world).sum(-1, keepdim=True)
        perp_pm  = delta_pm - along_pm * axis_t_world
        perp_dist  = torch.norm(perp_pm, dim=-1)
        along_dist = torch.abs(along_pm.squeeze(-1))

        near_above = (perp_dist < align_perp_tol) & (along_dist < align_along_tol)
        self._near_hole_frames = torch.where(
            near_above,
            self._near_hole_frames + 1,
            torch.zeros_like(self._near_hole_frames),
        )
        self._press_down_latched = self._press_down_latched | (
            self._near_hole_frames >= dwell_to_press
        )

        target = torch.where(
            self._press_down_latched.unsqueeze(-1),
            final_press_target,
            above_target,
        )

        # =============================================================
        # 8) Visualization
        # =============================================================
        self._visualize_candidates(
            all_holes_world=all_holes_world,
            all_pegs_world=all_pegs_world,
            chosen_pegs_world=[peg1_world, peg2_world],
            chosen_holes_world=[hole1_world, hole2_world],
            peg_mid_world=peg_mid_world,
            target_mid_world=target,
        )

        delta = target - peg_mid_world
        pos_action = delta / self.pos_threshold
        pos_action = torch.clamp(pos_action, -1.0, 1.0)
        scale_per_env = torch.where(
            self._press_down_latched.unsqueeze(-1),
            torch.full_like(pos_action, press_down_scale),
            torch.full_like(pos_action, descent_scale),
        )
        pos_action = pos_action * scale_per_env

        if self.joint_created:
            pos_action = torch.zeros_like(pos_action)

        rot_action = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action, rot_action, gripper_action], dim=-1)

    def _scripted_action_top2_to_frame(self):
        """Top-frame 2 → vasskar base (idx 2). Fully independent duplicate of
        the idx-1 scripted policy. Same logic and tunables, only the target
        insertion point differs: idx=2 targets the hole pair at y=0.220
        (indices #1 and #3) instead of y=0.040 (indices #0 and #2). No state
        is shared with idx-1 — all latches live on `_*_2` tensors.
        """
        # =============================================================
        # USER-SELECTABLE INDICES — change these once you see the labels.
        # ACTIVE_HOLE_INDICES picks 2 entries from all_hole_locals (#0..#3)
        # ACTIVE_PEG_INDICES  picks 2 entries from all_peg_top_locals (A..D)
        # =============================================================
        ACTIVE_HOLE_INDICES = (1, 3)   # orange(#1) + pink(#3) — y=0.220 pair
        ACTIVE_PEG_INDICES  = (0, 2)   # magenta(A) → orange(#1), lime(C) → pink(#3)

        # =============================================================
        # 1) ALL 4 base-hole candidates in frame local
        # =============================================================
        all_hole_locals_np = [
            [0.010, 0.040, 0.000],   # #0
            [0.010, 0.220, 0.000],   # #1
            [0.290, 0.040, 0.000],   # #2
            [0.290, 0.220, 0.000],   # #3
        ]
        all_holes_world = []
        for hl in all_hole_locals_np:
            hl_t = torch.tensor(hl, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            _, hw = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, hl_t
            )
            all_holes_world.append(hw)

        # =============================================================
        # 2) ALL 4 top2-corner candidates in held local (apply scale)
        # =============================================================
        top_scale = torch.as_tensor(
            self.cfg_task.top2.spawn.scale, dtype=torch.float32, device=self.device
        )
        sx_, sy_, sz_ = float(top_scale[0]), float(top_scale[1]), float(top_scale[2])
        all_peg_top_locals_unscaled = [
            [0.0025, 0.010, -0.010],   # A — corner near (0, 0)
            [0.5625, 0.010, -0.010],   # B — corner near (0.57, 0)
            [0.0025, 0.290, -0.010],   # C — corner near (0, 0.3)
            [0.5625, 0.290, -0.010],   # D — corner near (0.57, 0.3)
        ]
        all_pegs_world = []
        for pu in all_peg_top_locals_unscaled:
            pl = torch.tensor(
                [pu[0] * sx_, pu[1] * sy_, pu[2] * sz_], device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, pw = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, pl
            )
            all_pegs_world.append(pw)

        # =============================================================
        # 3) Pick the ACTIVE pair for the policy
        # =============================================================
        hole1_world = all_holes_world[ACTIVE_HOLE_INDICES[0]]
        hole2_world = all_holes_world[ACTIVE_HOLE_INDICES[1]]
        peg1_world  = all_pegs_world [ACTIVE_PEG_INDICES[0]]
        peg2_world  = all_pegs_world [ACTIVE_PEG_INDICES[1]]
        hole_mid_world = (hole1_world + hole2_world) / 2.0
        peg_mid_world  = (peg1_world  + peg2_world ) / 2.0

        # =============================================================
        # 4) Insertion axis in world
        # =============================================================
        axis_t_local = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t = torch.zeros_like(self.fixed_pos)
        _, axis_t_world = torch_utils.tf_combine(
            self.fixed_quat, zero_t, self.identity_quat, axis_t_local
        )

        # =============================================================
        # 5) Tunables — 2-phase direct insertion (same shape as idx=1).
        # No deliberate miss, no overshoot. State lives on _2 latches.
        # =============================================================
        approach_height   = 0.10
        align_perp_tol    = 0.005
        align_along_tol   = 0.02
        dwell_to_press    = 3
        descent_scale     = 0.3
        press_down_scale  = 0.2
        final_press_depth = 0.10

        # =============================================================
        # 6) Targets
        # =============================================================
        above_target       = hole_mid_world - axis_t_world * approach_height
        final_press_target = hole_mid_world + axis_t_world * final_press_depth

        # =============================================================
        # 7) State machine — approach → press. Only _2 latches are touched.
        # =============================================================
        delta_pm = peg_mid_world - above_target
        along_pm = (delta_pm * axis_t_world).sum(-1, keepdim=True)
        perp_pm  = delta_pm - along_pm * axis_t_world
        perp_dist  = torch.norm(perp_pm, dim=-1)
        along_dist = torch.abs(along_pm.squeeze(-1))

        near_above = (perp_dist < align_perp_tol) & (along_dist < align_along_tol)
        self._near_hole_frames_2 = torch.where(
            near_above,
            self._near_hole_frames_2 + 1,
            torch.zeros_like(self._near_hole_frames_2),
        )
        self._press_down_latched_2 = self._press_down_latched_2 | (
            self._near_hole_frames_2 >= dwell_to_press
        )

        target = torch.where(
            self._press_down_latched_2.unsqueeze(-1),
            final_press_target,
            above_target,
        )

        # =============================================================
        # 8) Visualization
        # =============================================================
        self._visualize_candidates(
            all_holes_world=all_holes_world,
            all_pegs_world=all_pegs_world,
            chosen_pegs_world=[peg1_world, peg2_world],
            chosen_holes_world=[hole1_world, hole2_world],
            peg_mid_world=peg_mid_world,
            target_mid_world=target,
        )

        delta = target - peg_mid_world
        pos_action = delta / self.pos_threshold
        pos_action = torch.clamp(pos_action, -1.0, 1.0)
        scale_per_env = torch.where(
            self._press_down_latched_2.unsqueeze(-1),
            torch.full_like(pos_action, press_down_scale),
            torch.full_like(pos_action, descent_scale),
        )
        pos_action = pos_action * scale_per_env

        if self.joint_created:
            pos_action = torch.zeros_like(pos_action)

        rot_action = torch.zeros((self.num_envs, 3), device=self.device)
        gripper_action = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action, rot_action, gripper_action], dim=-1)

    def _scripted_action_side_to_frame(self):
        """Side-frame → assembled base+top1+top2 (idx=3) — INDEPENDENT policy.

        Alignment pairs (user-confirmed via visualization colors):
          • slot #0 (BLUE on held)  ↔ top1 leg #0 (RED)
          • slot #1 (TEAL on held)  ↔ top1 leg #2 (YELLOW)

        3-phase flow with residual feedback at every transition:
          A) AIR ALIGN: in the air, drive peg_mid → above hole_mid AND
             align yaw (peg-line ‖ hole-line). Latch only when xy/z/yaw
             residuals are all within tight tol for sustained frames.
          B) DESCEND: straight drop, peg_mid → press_down_target. Yaw
             frozen so descent doesn't twist the lid.
          C) POST-LANDING FINE-TUNE: target = deeper press, continuously
             correct xy + yaw residuals while maintaining DOWNWARD force
             so the lid never lifts off the legs. Latches done when the
             attach-joint condition fires.
        """
        # =============================================================
        # 4 leg-cluster centroids per top plate (already-verified from
        # idx=1: each leg spans ~5–15 mm × 20 mm × 20 mm; these are the
        # 3D geometric centers of those clusters in top_local, unscaled).
        # =============================================================
        top_scale = torch.as_tensor(
            self.cfg_task.top1.spawn.scale, dtype=torch.float32, device=self.device
        )
        sx_, sy_, sz_ = float(top_scale[0]), float(top_scale[1]), float(top_scale[2])
        top_corner_locals_unscaled = [
            [0.0025, 0.010, -0.010],   # leg #0 (1 tick)  x-y- corner
            [0.5625, 0.010, -0.010],   # leg #1 (2 ticks) x+y-
            [0.0025, 0.290, -0.010],   # leg #2 (3 ticks) x-y+
            [0.5625, 0.290, -0.010],   # leg #3 (4 ticks) x+y+
        ]

        # top1 world pose
        top1_pos = self._top1.data.root_pos_w - self.scene.env_origins
        top1_quat = self._top1.data.root_quat_w
        top1_corners_world = []
        for cu in top_corner_locals_unscaled:
            cl = torch.tensor(
                [cu[0] * sx_, cu[1] * sy_, cu[2] * sz_], device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, cw = torch_utils.tf_combine(
                top1_quat, top1_pos, self.identity_quat, cl
            )
            top1_corners_world.append(cw)

        # top2 world pose
        top2_pos = self._top2.data.root_pos_w - self.scene.env_origins
        top2_quat = self._top2.data.root_quat_w
        top2_corners_world = []
        for cu in top_corner_locals_unscaled:
            cl = torch.tensor(
                [cu[0] * sx_, cu[1] * sy_, cu[2] * sz_], device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            _, cw = torch_utils.tf_combine(
                top2_quat, top2_pos, self.identity_quat, cl
            )
            top2_corners_world.append(cw)

        # =============================================================
        # Held asset (frame3.usd) has 4 LARGE ELLIPTICAL slots on its
        # BOTTOM face (z=-0.020). Each ellipse spans ~8 mm in x and
        # ~20 mm in y (verified from mesh-vertex clusters). These are
        # the 4 凹槽 that drop onto the 4 leg tops of top1+top2.
        # =============================================================
        held_slot_locals_unscaled = [
            [0.010, 0.040, -0.020],   # slot #0 (1 tick)  blue
            [0.290, 0.040, -0.020],   # slot #1 (2 ticks) teal
            [0.010, 0.220, -0.020],   # slot #2 (3 ticks) light blue
            [0.290, 0.220, -0.020],   # slot #3 (4 ticks) indigo
        ]
        held_slots_world = []
        for sl in held_slot_locals_unscaled:
            sl_t = torch.tensor(sl, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            _, sw = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.identity_quat, sl_t
            )
            held_slots_world.append(sw)

        # =============================================================
        # 1) Active pair selection (user-confirmed correspondence).
        #    Pegs (movable) = held slots #0, #1.
        #    Holes (stationary) = top1 leg #0, leg #2.
        # =============================================================
        peg1_world = held_slots_world[0]
        peg2_world = held_slots_world[1]
        hole1_world = top1_corners_world[0]
        hole2_world = top1_corners_world[2]
        peg_mid_world  = (peg1_world  + peg2_world)  / 2.0
        hole_mid_world = (hole1_world + hole2_world) / 2.0

        # =============================================================
        # 2) Insertion axis in world (from connection_cfg3.axis_t).
        # =============================================================
        axis_t_local = torch.as_tensor(
            self._connection_cfg.axis_t, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        zero_t = torch.zeros_like(self.fixed_pos)
        _, axis_t_world = torch_utils.tf_combine(
            self.fixed_quat, zero_t, self.identity_quat, axis_t_local
        )

        # =============================================================
        # 3) Tunables — residual-gated, tight tolerances.
        # =============================================================
        approach_height   = 0.10
        align_perp_tol    = 0.002   # 2 mm xy residual
        align_along_tol   = 0.005   # 5 mm z residual
        dwell_to_press    = 10      # frames the air-aligned state must hold
        descent_scale     = 0.3     # phase A approach scale
        press_down_depth  = 0.05
        press_down_scale  = 0.2     # phase B descent scale
        # — yaw alignment (active in phase A air-align and phase C fine-
        #   tune; frozen during phase B descent) —
        yaw_align_tol     = 0.02    # ~1.15° yaw residual
        yaw_align_scale   = 0.4     # P-gain on yaw_err
        # — phase B→C contact detection on peg_mid —
        contact_dz_thresh = 0.0003
        contact_dwell     = 5
        # — phase C continuous fine-tune (xy + yaw while keeping DOWNWARD
        #   force so the lid never lifts off the legs) —
        finetune_scale    = 0.15
        final_press_depth = 0.10

        # =============================================================
        # 4) Targets.
        # Phase A: peg_mid + yaw align in the air above hole_mid.
        # Phase B: straight descent (peg_mid → press_down_target).
        # Phase C: post-landing fine-tune — target = deeper press so
        #          pos_action always has +axis_t component (lid stays
        #          seated on the legs); xy + yaw residuals get corrected.
        # =============================================================
        above_target       = hole_mid_world - axis_t_world * approach_height
        press_down_target  = hole_mid_world + axis_t_world * press_down_depth
        final_press_target = hole_mid_world + axis_t_world * final_press_depth

        # =============================================================
        # 5) Yaw error (signed) between peg-line and hole-line in xy.
        # =============================================================
        peg_dir_world  = peg2_world  - peg1_world
        hole_dir_world = hole2_world - hole1_world
        peg_xy  = peg_dir_world[:,  :2]
        hole_xy = hole_dir_world[:, :2]
        peg_xy_n  = peg_xy  / (torch.norm(peg_xy,  dim=-1, keepdim=True) + 1e-8)
        hole_xy_n = hole_xy / (torch.norm(hole_xy, dim=-1, keepdim=True) + 1e-8)
        cross_z = peg_xy_n[:, 0] * hole_xy_n[:, 1] - peg_xy_n[:, 1] * hole_xy_n[:, 0]
        dot_xy  = (peg_xy_n * hole_xy_n).sum(-1)
        yaw_err = torch.atan2(cross_z, dot_xy)
        yaw_aligned = torch.abs(yaw_err) < yaw_align_tol

        # =============================================================
        # 6) Phase transitions (residual-gated).
        # =============================================================
        # Phase A → B: peg_mid AND yaw both within tol, sustained.
        d_A = peg_mid_world - above_target
        along_A = (d_A * axis_t_world).sum(-1, keepdim=True)
        perp_A  = d_A - along_A * axis_t_world
        perp_dist_A  = torch.norm(perp_A, dim=-1)
        along_dist_A = torch.abs(along_A.squeeze(-1))
        air_aligned = (
            (perp_dist_A < align_perp_tol)
            & (along_dist_A < align_along_tol)
            & yaw_aligned
        )
        self._near_hole_frames = torch.where(
            air_aligned,
            self._near_hole_frames + 1,
            torch.zeros_like(self._near_hole_frames),
        )
        self._press_down_latched = self._press_down_latched | (
            self._near_hole_frames >= dwell_to_press
        )
        # print(f"near_hole_frames: {self._near_hole_frames[0]}, perp: {perp_dist_A[0]:.4f}, along: {along_dist_A[0]:.4f}, yaw: {yaw_err[0]:.4f}")

        # Phase B → C: peg_mid.z stalls (lid has landed on the legs).
        peg_mid_z = peg_mid_world[:, 2]
        dz_descent = self._prev_peg_z - peg_mid_z
        stalled = (dz_descent < contact_dz_thresh) & self._press_down_latched
        self._contact_frames = torch.where(
            stalled,
            self._contact_frames + 1,
            torch.zeros_like(self._contact_frames),
        )
        self._search_active_latched = self._search_active_latched | (
            self._contact_frames >= contact_dwell
        )
        self._prev_peg_z = peg_mid_z.clone()

        # _final_press_latched / _overshoot_wait_frames unused now — fine-
        # tune happens continuously inside phase C until joint_created.

        # =============================================================
        # 7) Per-phase target. ALL phases drive peg_mid.
        # =============================================================
        in_phase_A = ~self._press_down_latched
        in_phase_B = self._press_down_latched & (~self._search_active_latched)
        in_phase_C = self._search_active_latched

        drive_point = peg_mid_world
        target = torch.where(
            in_phase_C.unsqueeze(-1),
            final_press_target,
            torch.where(
                in_phase_B.unsqueeze(-1),
                press_down_target,
                above_target,
            ),
        )

        # =============================================================
        # 6) Visualization (3 groups + chosen pair + alignment lines).
        # =============================================================
        self._visualize_candidates(
            all_holes_world=top1_corners_world,
            all_pegs_world=top2_corners_world,
            chosen_pegs_world=[peg1_world, peg2_world],
            chosen_holes_world=[hole1_world, hole2_world],
            peg_mid_world=peg_mid_world,
            target_mid_world=target,
            all_slots_world=held_slots_world,
        )

        # Drive peg_mid toward the per-phase target.
        delta = target - drive_point
        pos_action = delta / self.pos_threshold
        pos_action = torch.clamp(pos_action, -1.0, 1.0)
        # Phase A → descent_scale; B → press_down_scale; C → finetune_scale.
        scale_per_env = torch.where(
            in_phase_C.unsqueeze(-1),
            torch.full_like(pos_action, finetune_scale),
            torch.where(
                in_phase_B.unsqueeze(-1),
                torch.full_like(pos_action, press_down_scale),
                torch.full_like(pos_action, descent_scale),
            ),
        )
        pos_action = pos_action * scale_per_env

        # Phase C: PRESS-DOWN has the highest priority. Force the
        # along-axis_t component to FULL SATURATION (1.0) so the lid
        # keeps being pressed onto the legs at the maximum allowed
        # action magnitude. This overrides whatever the P-control or
        # scale logic would have produced for the along-axis component.
        # The in-plane (xy / perp to axis_t) component is preserved so
        # P-control still closes xy residuals.
        pos_along_signed = (pos_action * axis_t_world).sum(-1, keepdim=True)
        pos_perp = pos_action - pos_along_signed * axis_t_world
        pos_action_phaseC = pos_perp + axis_t_world * 1.0   # saturated max push
        pos_action = torch.where(
            in_phase_C.unsqueeze(-1),
            pos_action_phaseC,
            pos_action,
        )

        # NOTE idx=3: NO joint_created short-circuit. Auto-attach is
        # disabled in _check_attach_condition for this task, and we want
        # the phase C fine-tune to keep pushing down + correcting yaw
        # indefinitely (lid settles physically on the legs).

        # Yaw P-control active in phase A (air align) AND phase C (fine-
        # tune). Frozen in phase B so the descent doesn't twist the lid.
        rot_threshold_z = self.rot_threshold[:, 2]
        yaw_action_z = (yaw_err / (rot_threshold_z + 1e-8)) * yaw_align_scale
        yaw_action_z = torch.clamp(yaw_action_z, -1.0, 1.0)
        yaw_active = in_phase_A | in_phase_C
        yaw_action_z = torch.where(
            yaw_active,
            yaw_action_z,
            torch.zeros_like(yaw_action_z),
        )
        rot_action = torch.zeros((self.num_envs, 3), device=self.device)
        rot_action[:, 2] = yaw_action_z

        gripper_action = torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat([pos_action, rot_action, gripper_action], dim=-1)

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        self._visualize_markers()
        self._check_attach_condition()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
        # Scripted policy for idx=1 only — idx 2/3 keep the RL action.
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

        self.ctrl_target_gripper_dof_pos = 0.03 if gripper_actions < 0.0 else 0.0
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
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "vasskar_assembly":
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
        # Reset scripted-policy state (idx=1).
        self._near_hole_frames.zero_()
        self._search_active_latched.zero_()
        self._search_step = 0
        self._press_down_latched.zero_()
        self._prev_peg_z.zero_()
        self._contact_frames.zero_()
        self._overshoot_reached_latched.zero_()
        self._overshoot_wait_frames.zero_()
        self._final_press_latched.zero_()
        # Reset scripted-policy state (idx=2) — kept fully independent.
        self._near_hole_frames_2.zero_()
        self._search_active_latched_2.zero_()
        self._search_step_2 = 0
        self._press_down_latched_2.zero_()
        self._prev_peg_z_2.zero_()
        self._contact_frames_2.zero_()
        self._overshoot_reached_latched_2.zero_()
        self._overshoot_wait_frames_2.zero_()
        self._final_press_latched_2.zero_()
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
        if self.cfg_task.name == "vasskar_assembly" and self.cfg_task.task_idx in [1, 2]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.0
            held_asset_relative_pos[:, 1] -= 0.01
            held_asset_relative_pos[:, 0] = 0.12

        elif self.cfg_task.name == "vasskar_assembly" and self.cfg_task.task_idx in [3]:
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.fixed_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
            held_asset_relative_pos[:, 2] += 0.0
            held_asset_relative_pos[:, 1] -= 0.22
            held_asset_relative_pos[:, 0] -= 0.159
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


        if self.cfg_task.task_idx ==1:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] += 0.11
            rela_trans[:, 1] += 0.04
            rela_trans[:, 0] -= 0.12

        if self.cfg_task.task_idx ==2:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] += 0.11
            rela_trans[:, 1] += 0.21
            rela_trans[:, 0] -= 0.12


        if self.cfg_task.task_idx ==3:
            fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

            _, fixed_tip_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
            )
            self.fixed_pos_obs_frame[:] = fixed_tip_pos
            rela_trans = fixed_tip_pos.clone()
            rela_trans[:, 2] += self.cfg_task.hand_init_pos[2]
            rela_trans[:, 2] += 0.16
            rela_trans[:, 1] += 0.22
            rela_trans[:, 0] -= 0.16


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
            # abs_t = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            # abs_R = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            # abs_t[:] = torch.tensor(([-0.1408,-0.10, 0.755]))
            # abs_R[:] = torch.tensor(([0.5, -0.5, -0.5, -0.5]))
            # top1_state = torch.concat((abs_t, abs_R),dim=1)  # [N, 7]
            # self._top1.write_root_pose_to_sim(top1_state)
            # self._top1.reset()
            # abs_t[:] = torch.tensor(([-0.1408,0.08, 0.755]))
            # abs_R[:] = torch.tensor(([0.5, -0.5, -0.5, -0.5]))
            # top2_state = torch.concat((abs_t, abs_R),dim=1)  # [N, 7]
            # self._top2.write_root_pose_to_sim(top2_state)
            # self._top2.reset()
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

        if self.cfg_task.task_idx in [1, 2]:
            rot_euler = torch.tensor([1.5707, -1.5707, 0.0], device=self.device).repeat(
            self.num_envs, 1
            )
            translated_held_asset_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        if self.cfg_task.task_idx in [3]:
            rot_euler = torch.tensor([0, 0.0, 0.0], device=self.device).repeat(
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