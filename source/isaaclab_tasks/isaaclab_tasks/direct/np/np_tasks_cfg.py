# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import numpy as np

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
CHAIR_ASSET_DIR = "/home/crtie/crtie/Manual2Skill2/chair_real"

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
    friction: float = 0.75


@configclass
class ConnectionCfg:
    connection_type: str = "plug_connection"  
    base_path: str = ""
    connector_path: str = ""
    pose_to_base: np.ndarray = np.eye(4)  # 4x4 transformation matrix from connector to base.
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
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0



@configclass
class ChairFrame(FixedAssetCfg):
    # usd_path = f"{CHAIR_ASSET_DIR}/frame_test2.usd"
    usd_path = f"{CHAIR_ASSET_DIR}/frame_re2.usd"
    diameter = 0.0081
    height = 0.025
    mass = 0.05
    base_height = 0.0


@configclass
class Plug(HeldAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/plug.usd"
    # usd_path = f"{CHAIR_ASSET_DIR}/screw_ree2.usd"
    diameter = 0.007986
    height = 0.01
    mass = 0.001  # Mass is set to 0.001 to avoid large forces during insertion.

@configclass
class Screw(HeldAssetCfg):
    usd_path = f"{CHAIR_ASSET_DIR}/screw_ree3.usd"
    diameter = 0.007986
    height = 0.01
    mass = 0.001  # Mass is set to 0.001 to avoid large forces during insertion.

@configclass
class backrest_asset_config():
    usd_path = f"{CHAIR_ASSET_DIR}/backrest2.usd"
    mass = 0.1

@configclass
class rod_asset_config():
    usd_path = f"{CHAIR_ASSET_DIR}/rod2.usd"
    mass = 0.05
    height = -0.02
    base_height = 0.25

@configclass
class ChairAssembly(FactoryTask):
    #! crtie: task_idx is used to identify the task in the environment.
    #! crtie: task 1 is "insert the first plug into the first hole",
    #! crtie: task 2 is "insert the second plug into the second hole",
    #! crtie: task 3 is "insert the rod into the frame via the plug",
    task_idx = 1


    name = "chair_assembly"
    # fixed_asset_cfg = Hole8mm()
    fixed_asset_cfg = ChairFrame()
    # held_asset_cfg = Peg8mm()
    # held_asset_cfg = Plug()
    held_asset_cfg = Screw()
    backrest_asset_cfg = backrest_asset_config()
    rod_asset_cfg = rod_asset_config()
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
        hand_init_orn: list = [3.1416, 0.0, 1.5708]  # For the rod insertion task, the hand is oriented towards the rod.
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
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=5e-3),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.25, 0.74), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


    plug1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
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
            usd_path=held_asset_cfg.usd_path,
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
            scale = np.array([0.6,0.6,0.6]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-3, rest_offset=1e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    screw: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
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
            scale = np.array([0.7,0.7,0.7]), 
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Set to False for RigidObject
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=5e-3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0-0.55, 0.4, 0.1+0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    rod_asset: RigidObjectCfg = RigidObjectCfg(
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=1e-4, rest_offset=1e-4),
        ),
        #this param is place the rod at the edge
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.21, -0.17, 0.76), rot=(0.5, -0.5, 0.5, -0.5)),
        #this param is place the rod at the frame(not connected)
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.01, 0.09, 0.97), rot=(0.5, -0.5, 0.5, 0.5)),
    )


    connection_cfg1: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.01717465],
            [0.24, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.24663083],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg2: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug2",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.01717465],
            [0.24, 0.0, 1.0, 0.0373803],
            [0.0, 1.0, 0.0, -0.27053083],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg3: ConnectionCfg = ConnectionCfg(
        connection_type = "plug_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Rod",
        pose_to_base = np.array(
            [[0.0, -1.0, 0.0,  3.9181127e-03],
            [ 1.0, 0.0,  0.0,  2.2797813e-01],
            [0.0, -0.0,  1.0, -2.3246072e-01],
            [ 0.0,  0.0,  0.0,  1.0]]),
        axis_r = None,
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


    connection_cfg4: ConnectionCfg = ConnectionCfg(
        connection_type = "screw_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-0.97, 0.0, 0.0, 0.0188],
            [0.24, 0.0, 1.0, 0.05],
            [0.0, 1.0, 0.0, -0.2464],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )

    connection_cfg5: ConnectionCfg = ConnectionCfg(
        connection_type = "screw_connection",
        base_path = "/World/envs/env_.*/FixedAsset",
        connector_path = "/World/envs/env_.*/Plug1",
        pose_to_base = np.array(
            [[-0.97,  0.0,  0.0,  0.0188],
            [0.24,  0.0,  1.0,  0.06],
            [0.0,  1.0, 0.0, -0.2484],
            [0.0, 0.0, 0.0, 1.0]]),
        axis_r = np.array([0.0, 1.0, 0.0]),
        axis_t = np.array([0.0, 1.0, 0.0]),
    )


