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
CHAIR_ASSET_DIR = os.path.join(dir_path, "asset")

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
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 5.0


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
class ChairFrame(FixedAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/frame_re2.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0

@configclass
class ChairFrameBack(FixedAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/mesh/frame_back.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0

@configclass
class ChairFrameBackRod(FixedAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/mesh/frame_back_rod.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0

@configclass
class ChairFrameBackRodRod(FixedAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/mesh/frame_back_rod_rod.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0

@configclass
class ChairwoSeat(FixedAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/mesh/frame_wo_seat.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0

@configclass
class ChairwSeat(FixedAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/mesh/chair_all.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0

@configclass
class Frame(HeldAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/frame_mirror.usd"
    diameter = 0.03
    height = -0.01
    mass = 0.01
    base_height = 0.0

@configclass
class Plug(HeldAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/plug.usd"
    diameter = 0.007986
    height = 0.015
    mass = 0.001  # Mass is set to 0.001 to avoid large forces during insertion.

@configclass
class Screw(HeldAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/screw_ree3.usd"
    diameter = 0.007986
    height = 0.005
    mass = 0.001  # Mass is set to 0.001 to avoid large forces during insertion.

@configclass
class backrest_asset_config():
    usd_path = f"{CHAIR_ASSET_DIR}/backrest3.usd"
    mass = 0.1
    height = 0.18
    base_height = 0.25

@configclass
class rod_asset_config(HeldAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/rod.usd"
    mass = 0.05
    height = -0.04
    base_height = 0.25


@configclass
class ChairAssembly1(FactoryTask):
    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "insert the first plug into the first hole",
    #! crtie: task 2 is "insert the second plug into the second hole",
    #! crtie: task 3 is "insert the backrest into the frame via the plug",
    #! crtie: task 4 is "insert the plug1 into the backrest".
    #! crtie: task 5 is "insert the plug2 into the backrest".
    task_idx = 3


    name = "chair_assembly"
    fixed_asset_cfg = ChairFrame()
    held_asset_cfg = Plug()
    backrest_asset_cfg = backrest_asset_config()
    rod_asset_cfg = rod_asset_config()
    plug_config = Plug()
    asset_size = 8.0
    duration_s = 10.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.06]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    if task_idx == 1 or task_idx == 2:
        # For the first two tasks, the hand is oriented towards the fixed asset.
        hand_init_orn: list = [3.1416, 0.0, 0.0]
        hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    elif task_idx == 3:
        hand_init_pos: list = [0.0, 0.0, 0.12]  # Relative to fixed asset tip.
        hand_init_orn: list = [3.1416, 0.0, 1.5708]  # For the rod insertion task, the hand is oriented towards the rod.
        hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    elif task_idx == 4 or task_idx == 5:
        hand_init_pos: list = [0.0, 0.0, 0.28]

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
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.16, -0.3, 0.74), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    plug1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.75,0.75,0.75]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plug2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.75,0.75,0.75]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    backrest: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Backrest",
        spawn=sim_utils.UsdFileCfg(
            usd_path=backrest_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass = 0.1),
            scale=(1., 1., 1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, 0.09, 0.97), rot=(0.5, -0.5, 0.5, 0.5)),
    )



    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.01824],
            [0.24, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.45],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.0179],
            [0.24, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.418],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Backrest",
        pose_to_base = np.array(
            [[0.0, 1.0, 0.0,  4.8889988e-03],
            [ -1.0, 0.0,  0.0, 2.1893580e-01],
            [0.0, -0.0,  1.0, -3.7405658e-01],
            [ 0.0,  0.0,  0.0,  1.0]]),
        # pose_to_base = np.array(
        #     [[0.0, 1.0, 0.0,  2.3256524e-02],
        #     [ -1.0, 0.0,  0.0, 2.8873468e-02],
        #     [0.0, -0.0,  1.0, -3.7360814e-01],
        #     [ 0.0,  0.0,  0.0,  1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg4: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.018],
            [0.0, 0.0, 1.0, 0.228],
            [0.0, 1.0, 0.0, -0.45],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg5: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.018],
            [0.0, 0.0, 1.0, 0.228],
            [0.0, 1.0, 0.0, -0.417],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


@configclass
class ChairAssembly2(FactoryTask):
    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "insert the first plug into the first hole",
    #! crtie: task 2 is "insert the second plug into the second hole",
    #! crtie: task 3 is "insert the rod into the frame via the plug",
    #! crtie: task 4 is "insert the plug1 into the rod".
    #! crtie: task 5 is "insert the plug2 into the rod".
    task_idx = 5


    name = "chair_assembly"
    fixed_asset_cfg = ChairFrameBack()
    held_asset_cfg = Plug()
    backrest_asset_cfg = backrest_asset_config()
    rod_asset_cfg = rod_asset_config()
    plug_config = Plug()
    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.06]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    if task_idx == 1 or task_idx == 2:
        # For the first two tasks, the hand is oriented towards the fixed asset.
        hand_init_orn: list = [3.1416, 0.0, 0.0]
        hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    elif task_idx == 3:
        hand_init_pos: list = [0.0, 0.0, 0.12]  # Relative to fixed asset tip.
        hand_init_orn: list = [3.1416, 0.0, 1.5708]  # For the rod insertion task, the hand is oriented towards the rod.
        hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    elif task_idx == 4 or task_idx == 5:
        hand_init_pos: list = [0.0, 0.0, 0.28]


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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.16, -0.3, 0.74), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    plug1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.75,0.75,0.75]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plug2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.75,0.75,0.75]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    rod: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rod",
        spawn=sim_utils.UsdFileCfg(
            usd_path=rod_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass = 0.1),
            scale=(1., 1., 1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, 0.09, 0.97), rot=(0.5, -0.5, 0.5, 0.5)),
    )

    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.27545],
            [0.0, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.269],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.27545],
            [0.0, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.246],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Rod",
        pose_to_base = np.array(
            [[0.0, -1.0, 0.0,  2.8157920e-01],
            [ -1.0, 0.0,  0.0,  2.2909781e-01],
            [0.0, -0.0,  -1.0, -2.7506271e-01],
            [ 0.0,  0.0,  0.0,  1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg4: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.27545],
            [0.0, 0.0, 1.0, 0.236],
            [0.0, 1.0, 0.0, -0.268],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg5: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.27545],
            [0.0, 0.0, 1.0, 0.236],
            [0.0, 1.0, 0.0, -0.245],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


@configclass
class ChairAssembly3(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "insert the first plug into the first hole",
    #! crtie: task 2 is "insert the second plug into the second hole",
    #! crtie: task 3 is "insert the rod into the frame via the plug",
    #! crtie: task 4 is "insert the plug1 into the rod".
    #! crtie: task 5 is "insert the plug2 into the rod".
    task_idx = 3


    name = "chair_assembly"
    fixed_asset_cfg = ChairFrameBackRod()
    held_asset_cfg = Plug()
    backrest_asset_cfg = backrest_asset_config()
    rod_asset_cfg = rod_asset_config()
    plug_config = Plug()
    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.06]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]

    if task_idx == 1 or task_idx == 2:
        # For the first two tasks, the hand is oriented towards the fixed asset.
        hand_init_orn: list = [3.1416, 0.0, 0.0]
        hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    elif task_idx == 3:
        hand_init_pos: list = [0.0, 0.0, 0.12]  # Relative to fixed asset tip.
        hand_init_orn: list = [3.1416, 0.0, 1.5708]  # For the rod insertion task, the hand is oriented towards the rod.
        hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    elif task_idx == 4 or task_idx == 5:
        hand_init_pos: list = [0.0, 0.0, 0.28]


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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.16, -0.3, 0.74), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    plug1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.75,0.75,0.75]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plug2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.75,0.75,0.75]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    rod: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rod",
        spawn=sim_utils.UsdFileCfg(
            usd_path=rod_asset_cfg.usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass = 0.1),
            scale=(1., 1., 1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, 0.09, 0.97), rot=(0.5, -0.5, 0.5, 0.5)),
    )

    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.0185],
            [0.24, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.271],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.0185],
            [0.24, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.249],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Rod",
        pose_to_base = np.array(
            [[0.0, -1.0, 0.0,  2.4958238e-02],
            [ -1.0, 0.0,  0.0,  2.2895001e-01],
            [0.0, -0.0,  -1.0, -2.7744323e-01],
            [ 0.0,  0.0,  0.0,  1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


    connection_cfg4: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.0182],
            [0.24, 0.0, 1.0, 0.236],
            [0.0, 1.0, 0.0, -0.27],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg5: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.0186],
            [0.24, 0.0, 1.0, 0.236],
            [0.0, 1.0, 0.0, -0.247],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


@configclass
class ChairAssembly4(FactoryTask):
    #! crtie: task 1 is "connect the other frame to the subassembly",
    task_idx = 1


    name = "chair_assembly"
    fixed_asset_cfg = ChairFrameBackRodRod()
    held_asset_cfg = Frame()
    frame_config = Frame()
    asset_size = 8.0
    duration_s = 10.0


    # Robot
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_pos: list = [0.0, 0.0, 0.3]  # Relative to fixed asset tip.
    hand_init_orn: list = [3.1416, 0.0, 0.0]  # For the rod insertion task, the hand is oriented towards the rod.
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]



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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.16, -0.3, 0.74), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    frame: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Frame",
        spawn=sim_utils.UsdFileCfg(
            usd_path=frame_config.usd_path,
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
            scale = np.array([1.0,1.0,1.0]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, -0.30, 1.02), rot=(0.0, 0.0, -0.707, 0.707)),
    )

    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Frame",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 2.8771558e-01],
            [0.0, -1.0, 0.0, 2.5098464e-01],
            [0.0, 0.0, 1.0, -5.1539927e-04],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = None,
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


@configclass
class ChairAssembly5(FactoryTask):

    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "the first screw
    #! crtie: task 2 is "the second screw",
    #! crtie: task 3 is "the third screw"

    task_idx = 3


    name = "chair_assembly"
    fixed_asset_cfg = ChairwoSeat()
    held_asset_cfg = Screw()
    plug_config = Screw()
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.16, -0.3, 0.74), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    screw1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.6,0.6,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,)
                                                             #collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    screw2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.6,0.6,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    screw3: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.6,0.6,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.016],
            [0.0, 0.0, 1.0, 0.277],
            [0.0, 1.0, 0.0, -0.2625],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg1_fix: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.0124],
            [0.0, 0.0, 1.0, 0.26],
            [0.0, 1.0, 0.0, -0.252],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw2",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.273],
            [0.0, 0.0, 1.0, 0.277],
            [0.0, 1.0, 0.0, -0.2595],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw3",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0,  0.020],
            [0.0, 0.0, 1.0,  0.277],
            [0.0, 1.0, 0.0, -0.498],
            [ 0.0,  0.0,  0.0,  1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


@configclass
class ChairAssembly6(FactoryTask):


    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "the first screw
    #! crtie: task 2 is "the second screw",
    #! crtie: task 3 is "the third screw"
    #! crtie: task 4 is "the fourth screw"

    task_idx = 1


    name = "chair_assembly"
    fixed_asset_cfg = ChairwSeat()
    held_asset_cfg = Screw()
    plug_config = Screw()
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
                                                             collision_enabled=False),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, -0.1, 1.3), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            # pos=(-0.16, -0.14, 3.20), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    screw1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.6,0.6,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    screw2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.6,0.6,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3,
                                                             collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    screw3: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Screw3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_config.usd_path,
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
            scale = np.array([0.6,0.6,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.020],
            [0.0, 0.0, 1.0, 0.074],
            [0.0, 1.0, 0.0, -0.220],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg1_fix: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw1",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.0124],
            [0.0, 0.0, 1.0, 0.26],
            [0.0, 1.0, 0.0, -0.252],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw2",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0, 0.273],
            [0.0, 0.0, 1.0, 0.277],
            [0.0, 1.0, 0.0, -0.2595],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Screw3",
        pose_to_base = np.array(
            [[-1.0, 0.0, 0.0,  0.0117],
            [0.0, 0.0, 1.0,  0.277],
            [0.0, 1.0, 0.0, -0.4895],
            [ 0.0,  0.0,  0.0,  1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )