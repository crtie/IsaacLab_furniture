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
    pose_to_base_euler: np.ndarray = np.array([0.0, 0.0, 0.0])  # Euler angles (in radians) representing rotation from connector to base.
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
class TailHalf1(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/tail_half_wider1.usd"
    diameter = 0.06
    height = 0.0
    mass = 0.01
    base_height = 0.0

@configclass
class TailHalf2(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/tail_half2.usd"
    diameter = 0.06
    height = 0.02
    mass = 0.01
    base_height = 0.0

@configclass
class Body(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/body.usd"
    diameter = 0.06
    height = 0.12
    mass = 0.01
    base_height = 0.0

@configclass
class Propeller(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/propeller.usd"
    diameter = 0.06
    height = 0.01
    mass = 0.01
    base_height = 0.0

@configclass
class Holder(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/holder.usd"
    diameter = 0.06
    height = 0.01
    mass = 0.01
    base_height = 0.0

@configclass
class WheelAll(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/wheel_all1.usd"
    diameter = 0.06
    height = 0.01
    mass = 0.01
    base_height = 0.0

@configclass
class BodyAll(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/body3.usd"
    diameter = 0.06
    height = 0.01
    mass = 0.01
    base_height = 0.0

@configclass
class Crossbar(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/crossbar.usd"
    diameter = 0.02
    height = 0.04
    mass = 0.01
    base_height = 0.0

@configclass
class PlaneWoUpper(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/plane_wo_upper1.usd"
    diameter = 0.06
    height = 0.01
    mass = 0.01
    base_height = 0.0

@configclass
class UpperWing(FixedAssetCfg):
    usd_path = f"{PLANE_ASSET_DIR}/upper_wing.usd"
    diameter = 0.08
    height = 0.0
    mass = 0.01
    base_height = 0.0


@configclass
class PlaneAssembly1(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1, 2, 3, 4]

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

@configclass
class PlaneAssembly2(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1, 2, 3, 4]

    task_idx = 4


    name = "plane_assembly"
    tailhalf_cfg = TailHalf2()
    body_cfg = Body()
    propeller_cfg = Propeller()
    holder_cfg = Holder()
    if task_idx == 1:
        fixed_asset_cfg = TailHalf1()
        held_asset_cfg = TailHalf2()
    elif task_idx == 2:
        fixed_asset_cfg = TailHalf1()
        held_asset_cfg = Body()
    elif task_idx == 3:
        fixed_asset_cfg = TailHalf1()
        held_asset_cfg = Propeller()
    elif task_idx == 4:
        fixed_asset_cfg = TailHalf1()
        held_asset_cfg = Holder()
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
            scale = np.array([1.0, 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.11, -0., 0.83), rot=(0., -0.707, 0.707, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    tailhalf: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TailHalf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tailhalf_cfg.usd_path,
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
            scale = np.array([1.0, 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= (task_idx in [1, 2, 3])),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    body: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/MainBody",
        spawn=sim_utils.UsdFileCfg(
            usd_path=body_cfg.usd_path,
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= (task_idx in [2, 3])),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    propeller: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Propeller",
        spawn=sim_utils.UsdFileCfg(
            usd_path=propeller_cfg.usd_path,
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= True),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    holder: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Holder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=holder_cfg.usd_path,
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
            scale = np.array([1.2, 1., 1.]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= True),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame1",
        pose_to_base = np.array(
        [[0.0, -1.0, 0.0, 0.0 ],
        [ -1.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, -1.0, -0.006],
        [ 0.,  0.,  0.,  1. ]]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, 0.0, 1.0, 0.],
            [0.0, 1.0, 0.0, -0.006],
            [-1.0, 0.0, 0.0, -0.13],
            [0.0, 0.0, 0.0, 1.0]]),
        # axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        #! crtie: something fking wrong with transformation matrix here, euler angle works fine
        connection_type = "euler",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, 0.0, 1.0, 0.004], #y
            [1.0, 0.0, 0.0, -0.006],
            [0.0, 1.0, 0.0, -0.27],
            [0.0, 0.0, 0.0, 1.0]]),
        pose_to_base_euler = np.array([1.5708, 0.0, 1.5708]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg4: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[1.0, 0.0, 0.0, 0.007],
            [0.0, 0.0, -1.0, -0.008],
            [0.0, 1.0, 0.0, -0.0275],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )


@configclass
class PlaneAssembly3(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1, 2, 3]


    task_idx = 3


    name = "plane_assembly"
    wheel_all_cfg = WheelAll()
    body_cfg = BodyAll()
    crossbar_cfg = Crossbar()
    if task_idx == 1:
        fixed_asset_cfg = WheelAll()
        held_asset_cfg = BodyAll()
    elif task_idx == 2:
        fixed_asset_cfg = WheelAll()
        held_asset_cfg = Crossbar()
    elif task_idx == 3:
        fixed_asset_cfg = WheelAll()
        held_asset_cfg = Crossbar()
    # elif task_idx == 3:
    #     fixed_asset_cfg = TailHalf1()
    #     held_asset_cfg = Propeller()
    # elif task_idx == 4:
    #     fixed_asset_cfg = TailHalf1()
    #     held_asset_cfg = Holder()
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.11, -0., 0.79), rot=(0., -0.707, 0.707, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    body: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Body",
        spawn=sim_utils.UsdFileCfg(
            usd_path= body_cfg.usd_path,
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= (task_idx in [1])),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    crossbar1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Crossbar1",
        spawn=sim_utils.UsdFileCfg(
            usd_path= crossbar_cfg.usd_path,
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= (task_idx in [2])),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    crossbar2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Crossbar2",
        spawn=sim_utils.UsdFileCfg(
            usd_path= crossbar_cfg.usd_path,
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled= True),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    connection_cfg1: ConnectionCfg = ConnectionCfg(
        #! crtie: something fking wrong with transformation matrix here, euler angle works fine
        #! crtie: all the representions are fking strange, need to investigate later
        connection_type = "euler",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, 0.0, -1.0, 0.00], #y
            [0.0, 1.0, 0.0, 0.004],
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 0.0, 0.0, 1.0]]),
        pose_to_base_euler = np.array([3.1415, 3.14, 0]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "euler",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, 0.0, 1.0, -0.068],
            [0.0, 1.0, 0.0, 0.007],
            [-1.0, 0.0, 0.0, -0.062],
            [0.0, 0.0, 0.0, 1.0]]),
        pose_to_base_euler = np.array([3.1415, 1.5707, -1.5707]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        #! crtie: something fking wrong with transformation matrix here, euler angle works fine
        connection_type = "euler",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, 0.0, 1.0, 0.004], #y
            [1.0, 0.0, 0.0, -0.006],
            [0.0, 1.0, 0.0, -0.27],
            [0.0, 0.0, 0.0, 1.0]]),
        pose_to_base_euler = np.array([1.5708, 0.0, 1.5708]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

    connection_cfg4: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[1.0, 0.0, 0.0, 0.007],
            [0.0, 0.0, -1.0, -0.008],
            [0.0, 1.0, 0.0, -0.0275],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )


@configclass
class PlaneAssembly4(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    plane_wo_upper_cfg = PlaneWoUpper()
    upper_wing_cfg = UpperWing()
    fixed_asset_cfg = PlaneWoUpper()
    held_asset_cfg = UpperWing()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    hand_init_orn: list = [3.1416, 0.0, 1.57]
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.07, -0., 0.82), rot=(0., 0.707, 0.707, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    upperwing: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/UpperWing",
        spawn=sim_utils.UsdFileCfg(
            usd_path= upper_wing_cfg.usd_path,
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
            scale = np.array([1.2, 1.2, 1.2]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled = True),
                                                            # collision_enabled = False),
                                                            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    connection_cfg1: ConnectionCfg = ConnectionCfg(
        #! crtie: something fking wrong with transformation matrix here, euler angle works fine
        #! crtie: all the representions are fking strange, need to investigate later
        connection_type = "euler",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/TopFrame2",
        pose_to_base = np.array(
            [[0.0, 0.0, -1.0, 0.00], #y
            [0.0, 1.0, 0.0, 0.004],
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 0.0, 0.0, 1.0]]),
        pose_to_base_euler = np.array([3.1415, 3.14, 0]),
        axis_r = np.array([0.0, 0.0, 1.0]),
        axis_t = np.array([0.0, 0.0, 1.0]),
    )

