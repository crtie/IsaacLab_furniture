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
LEGO_ASSET_DIR = os.path.join(dir_path, "asset", "lego")

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
class Arm(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/arm1.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Hand(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/hand.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Hips(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/hips1.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Hips1leg(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/hips_1leg1.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Hips2leg(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/hips_2leg1.usd"
    diameter = 0.04
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Leg(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/leg_largehole.usd"
    diameter = 0.08
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class LegMirror(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/leg_mirror.usd"
    diameter = 0.08
    height = -0.005
    mass = 0.01
    base_height = 0.0

@configclass
class Torso(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/torso.usd"
    diameter = 0.08
    height = 0.18
    mass = 0.01
    base_height = 0.0
    
@configclass
class TorsoLeg(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/torso_leg1.usd"
    diameter = 0.08
    height = 0.18
    mass = 0.01
    base_height = 0.0

@configclass
class LeftArm(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/leftarm.usd"
    diameter = 0.08
    height = 0.015
    mass = 0.01
    base_height = 0.0

@configclass
class RightArm(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/rightarm.usd"
    diameter = 0.04
    height = 0.015
    mass = 0.01
    base_height = 0.0

@configclass
class TorsoLegLeftArm(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/torso_leg_leftarm1.usd"
    diameter = 0.08
    height = 0.18
    mass = 0.01
    base_height = 0.0

@configclass
class Headless(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/headless1.usd"
    diameter = 0.08
    height = 0.18
    mass = 0.01
    base_height = 0.0
    
@configclass
class Head(FixedAssetCfg):
    usd_path = f"{LEGO_ASSET_DIR}/head.usd"
    diameter = 0.05
    height = 0.065
    mass = 0.01
    base_height = 0.0

@configclass
class LegoAssembly1(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    arm_cfg = Arm()
    hand_cfg = Hand()
    fixed_asset_cfg = Arm()
    held_asset_cfg = Hand()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

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
            pos=(0.07, -0., 0.77), rot=(0., 0.707, 0.707, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    hand: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path= hand_cfg.usd_path,
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
            scale = np.array([0.85, 0.85, 1.0]), 
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


@configclass
class LegoAssembly2(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    hips_cfg = Hips()
    leg_cfg = Leg()
    fixed_asset_cfg = Hips()
    held_asset_cfg = Leg()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

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
            pos=(0.07, -0., 0.82), rot=(0.5, 0.5, 0.5, -0.5), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path= held_asset_cfg.usd_path,
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


@configclass
class LegoAssembly3(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    hips_cfg = Hips1leg()
    leg_cfg = LegMirror()
    fixed_asset_cfg = Hips1leg()
    held_asset_cfg = LegMirror()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

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
            pos=(0.07, -0., 0.82), rot=(0.5, 0.5, 0.5, -0.5), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path= held_asset_cfg.usd_path,
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

@configclass
class LegoAssembly4(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    hips_cfg = Hips2leg()
    torso_cfg = Torso()
    fixed_asset_cfg = Hips2leg()
    held_asset_cfg = Torso()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    hand_init_orn: list = [3.1416, 0.0, 1.5707]
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
            scale = np.array([1., 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.07, -0., 0.87), rot=(0.5, 0.5, 0.5, 0.5), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path= held_asset_cfg.usd_path,
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
            scale = np.array([1.0, 1.0, 1.0]), 
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


@configclass
class LegoAssembly5(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    fixed_asset_cfg = TorsoLeg()
    held_asset_cfg = LeftArm()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    hand_init_orn: list = [3.1416, 0.0, 0]
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
            scale = np.array([1., 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.07, -0., 0.85), rot=(0.5, -0.5, -0.5, -0.5), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path= held_asset_cfg.usd_path,
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
            scale = np.array([1.0, 1.0, 1.0]), 
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


@configclass
class LegoAssembly6(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    fixed_asset_cfg = TorsoLegLeftArm()
    held_asset_cfg = RightArm()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    hand_init_orn: list = [3.1416, 0.0, 0]
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
            scale = np.array([1., 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.07, -0., 0.87), rot=(0.5, 0.5, 0.5, -0.5), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path= held_asset_cfg.usd_path,
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
            scale = np.array([1.0, 1.0, 1.0]), 
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

@configclass
class LegoAssembly7(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment
    #! crtie: task_idx can be [1]
    task_idx = 1


    name = "plane_assembly"
    fixed_asset_cfg = Headless()
    held_asset_cfg = Head()

    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.30]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    hand_init_orn: list = [3.1416, 0.0, 0]
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
            scale = np.array([1., 1.0, 1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3,
                                                             ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.18, -0., 0.98), rot=(0.5, 0.5, 0.5, 0.5), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path= held_asset_cfg.usd_path,
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
            scale = np.array([1.15, 1.15, 1.15]), 
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