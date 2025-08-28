# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os, sys
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import numpy as np

dir_path = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
PLANE_ASSET_DIR = os.path.join(dir_path, "asset", "plane")

@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 5.0
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 10.0


@configclass
class ConnectionCfg:
    connection_type: str = "plug_connection"  
    base_path: str = ""
    connector_path: str = ""
    pose_to_base: np.ndarray = np.eye(4)  # 4x4 transformation matrix from connector to base.
    pose_to_base1: np.ndarray = np.eye(4)  # 4x4 transformation matrix from connector to base, used for the first connection.
    axis_t: np.ndarray = np.array([0.0, 0.0, 1.0])  # Axis of the connection, describe the connection direction.
    axis_r: np.ndarray = np.array([0.0, 0.0, 1.0])  # Axis of the connection, describe the rotation symmetry of the connection, i.e. the axis around which the connector can rotate relative to the base.

@configclass
class FactoryTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0


    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_thread task only.
    action_penalty_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]  # General movement towards fixed object.
    keypoint_coef_coarse: list = [50, 2]  # Movement to align the assets.
    keypoint_coef_fine: list = [100, 0]  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9



@configclass
class WheelAxis(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/wheel_axis1.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Wheel(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/wheel.usd"
    diameter = 0.06
    height = 0.0
    mass = 0.01
    base_height = 0.0

@configclass
class WheelDowel(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/wheel_dowel.usd"
    diameter = 0.02
    height = 0.1
    mass = 0.01
    base_height = 0.0

@configclass
class WheelAxisHalf(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/wheel_axis_half1.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0



@configclass
class PlaneAssembly1(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "the first top frame",
    #! crtie: task 2 is "the second top frame",
    #! crtie: task 3 is "the side frame".,

    task_idx = 4


    name = "plane_assembly"

    wheel_axis_cfg = WheelAxis()
    wheel_axis_half_cfg = WheelAxisHalf()
    wheel_cfg = Wheel()
    wheel_dowel_cfg = WheelDowel()
    if task_idx in [1, 3]:
        held_asset_cfg = Wheel()
    elif task_idx in [2, 4]:
        held_asset_cfg = WheelDowel()
    if task_idx in [1, 2]:
        fixed_asset_cfg = WheelAxis()
    elif task_idx in [3, 4]:
        fixed_asset_cfg = WheelAxisHalf()
    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    # For the first two tasks, the hand is oriented towards the fixed asset.
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]


    # Fixed Asset (applies to all tasks)
    # fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_pos_noise: list = [0.00, 0.00, 0.00]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            scale = np.array([2.0, 2.0, 2.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, -0.15, 0.65), rot=(0.707, 0.0, -0.707, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    wheel: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Wheel1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=wheel_cfg.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1, 
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass = 0.01),
            scale = np.array([1.8, 1.8, 1.8]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= True),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    wheel_dowel: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WheelDowel1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=wheel_dowel_cfg.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1, 
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass = 0.01),
            scale = np.array([1.2, 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled=True),
                                                            # collision_enabled = False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame1",
        pose_to_base = np.array(
        [[1.0, 0.0, 0.0, 0.0369899 ],
        [ 0.0, 0.0, -1.0, 0.24668184],
        [ 0.0, 1.0, 0.0,-0.02043285],
        [ 0.,  0.,  0.,  1. ]]),
        axis_r = np.array([1.0, 0.0, 0.0]),
        axis_t = np.array([1.0, 0.0, 0.0]),
    )

    connection_cfg1_fix: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame1",
        pose_to_base = np.array(
        [[1.0, 0.0, 0.0, 0.018], #z
        [ 0.0, 0.0, -1.0, 0.122],
        [ 0.0, 1.0, 0.0,  -0.012],#x
        [ 0.,  0.,  0.,  1. ]]),
        # axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[1.0, 0.0, 0.0, 0.053],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, -1.0, 0.0, 0.035],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([1.0, 0.0, 0.0]),
        axis_t = np.array([1.0, 0.0, 0.0]),
    )

    connection_cfg2_fix: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, -1.0, 0.0, 0.3],
            [0.0, 0.0, -1.0, 0.21],
            [1.0, 0.0, 0.0, -0.416],
            [0.0, 0.0, 0.0, 1.0]]),
        # axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg3_fix: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame1",
        pose_to_base = np.array(
        [[1.0, 0.0, 0.0, 0.018], #z
        [ 0.0, 0.0, -1.0, 0.122],
        [ 0.0, 1.0, 0.0,  -0.012],#x
        [ 0.,  0.,  0.,  1. ]]),
        # axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )