# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the local Franka arm + Wuji dexterous hand asset."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


FRANKA_ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
WUJI_HAND_FINGER_JOINT_NAMES = [f"right_finger{i}_joint{j}" for j in range(1, 5) for i in range(1, 6)]
FRANKA_WUJI_JOINT_NAMES = FRANKA_ARM_JOINT_NAMES + WUJI_HAND_FINGER_JOINT_NAMES
FRANKA_WUJI_ACTION_DIM = 26
FRANKA_WUJI_JOINT_DIM = 27
WUJI_FLOATING_WRIST_JOINT_NAMES = ["wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw"]
WUJI_FLOATING_JOINT_NAMES = WUJI_FLOATING_WRIST_JOINT_NAMES + WUJI_HAND_FINGER_JOINT_NAMES
WUJI_FLOATING_ACTION_DIM = 6 + len(WUJI_HAND_FINGER_JOINT_NAMES)
WUJI_FLOATING_JOINT_DIM = len(WUJI_FLOATING_JOINT_NAMES)

FRANKA_WUJI_HAND_MOUNT_XYZ = (0.0, 0.0, 0.0)
FRANKA_WUJI_HAND_MOUNT_RPY = (0.0, 1.5708, -0.7854)
DEX_GRASP_FRAME_LOCAL_POS = (0.0, 0.0, 0.10)
DEX_GRASP_FRAME_LOCAL_RPY = (0.0, 0.0, 0.0)
DEX_GRASP_FRAME_NAME = "dex_grasp_frame"
WUJI_FINGERTIP_BODY_NAMES = [f"right_finger{i}_link4" for i in range(1, 6)]

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRANKA_WUJI_HAND_URDF_PATH = os.path.join(
    _CURRENT_DIR, "wuji-hand-description/urdf/franka_wuji_hand.urdf"
)
WUJI_FLOATING_HAND_URDF_PATH = os.path.join(
    _CURRENT_DIR, "wuji-hand-description/urdf/right_block_palm_floating_drone.urdf"
)
FRANKA_WUJI_HAND_USD_DIR = os.path.join(_CURRENT_DIR, "usd")
FRANKA_WUJI_HAND_USD_PATH = os.path.join(FRANKA_WUJI_HAND_USD_DIR, "franka_wuji_hand.usd")
WUJI_FLOATING_HAND_USD_DIR = os.path.join(FRANKA_WUJI_HAND_USD_DIR, "wuji_right_floating_hand")
WUJI_FLOATING_HAND_USD_PATH = os.environ.get(
    "WUJI_FLOATING_HAND_USD_PATH_OVERRIDE",
    os.path.join(WUJI_FLOATING_HAND_USD_DIR, "wuji_right_floating_hand.usd"),
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


_SCREW1_V2_WRIST_PD_FIX = _env_flag("WUJI_SCREW1_V2_WRIST_PD_FIX")
_FLOATING_WRIST_TRANSLATION_EFFORT = 5000.0 if _SCREW1_V2_WRIST_PD_FIX else 120.0
_FLOATING_WRIST_ROTATION_EFFORT = 1000.0 if _SCREW1_V2_WRIST_PD_FIX else 35.0
_FLOATING_WRIST_TRANSLATION_VELOCITY = 2.0 if _SCREW1_V2_WRIST_PD_FIX else 0.45
_FLOATING_WRIST_ROTATION_VELOCITY = 3.0 if _SCREW1_V2_WRIST_PD_FIX else 1.2
_FLOATING_WRIST_TRANSLATION_STIFFNESS = 20000.0 if _SCREW1_V2_WRIST_PD_FIX else 800.0
_FLOATING_WRIST_ROTATION_STIFFNESS = 2000.0 if _SCREW1_V2_WRIST_PD_FIX else 120.0
_FLOATING_WRIST_TRANSLATION_DAMPING = 500.0 if _SCREW1_V2_WRIST_PD_FIX else 80.0
_FLOATING_WRIST_ROTATION_DAMPING = 80.0 if _SCREW1_V2_WRIST_PD_FIX else 12.0


FRANKA_WUJI_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_WUJI_HAND_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=2.0,
            max_angular_velocity=8.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=250.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=10,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.00871,
            "panda_joint2": -0.10368,
            "panda_joint3": -0.00794,
            "panda_joint4": -1.49139,
            "panda_joint5": -0.00083,
            "panda_joint6": 1.38774,
            "panda_joint7": 0.0,
            "right_finger.*_joint1": 0.04,
            "right_finger.*_joint(2|3|4)": 0.0,
        },
        pos=(-0.55, 0.0, 0.75),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87.0,
            velocity_limit_sim=124.6,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=12.0,
            velocity_limit_sim=149.5,
        ),
        "wuji_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_finger.*_joint.*"],
            effort_limit_sim={
                "right_finger(1|2|3|4|5)_joint(1|2)": 28.0,
                "right_finger(1|2|3|4|5)_joint3": 18.0,
                "right_finger(1|2|3|4|5)_joint4": 10.0,
            },
            velocity_limit_sim={
                "right_finger(1|2|3|4|5)_joint(1|2)": 12.0,
                "right_finger(1|2|3|4|5)_joint3": 14.0,
                "right_finger(1|2|3|4|5)_joint4": 14.0,
            },
            stiffness={
                "right_finger(1|2|3|4|5)_joint(1|2)": 80.0,
                "right_finger(1|2|3|4|5)_joint3": 45.0,
                "right_finger(1|2|3|4|5)_joint4": 24.0,
            },
            damping={
                "right_finger.*_joint(1|2)": 2.4,
                "right_finger.*_joint(3|4)": 1.4,
            },
            friction=0.01,
            armature=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Single articulation: Franka 7-DoF arm plus Wuji 20-DoF right dexterous hand."""


WUJI_FLOATING_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=WUJI_FLOATING_HAND_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=10,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "wrist_x": 0.0,
            "wrist_y": 0.0,
            "wrist_z": 0.8,
            "wrist_roll": 0.0,
            "wrist_pitch": 0.0,
            "wrist_yaw": 0.0,
            "right_finger.*_joint1": 0.04,
            "right_finger.*_joint(2|3|4)": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "floating_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_(x|y|z|roll|pitch|yaw)"],
            effort_limit_sim={
                "wrist_(x|y|z)": _FLOATING_WRIST_TRANSLATION_EFFORT,
                "wrist_(roll|pitch|yaw)": _FLOATING_WRIST_ROTATION_EFFORT,
            },
            velocity_limit_sim={
                "wrist_(x|y|z)": _FLOATING_WRIST_TRANSLATION_VELOCITY,
                "wrist_(roll|pitch|yaw)": _FLOATING_WRIST_ROTATION_VELOCITY,
            },
            stiffness={
                "wrist_(x|y|z)": _FLOATING_WRIST_TRANSLATION_STIFFNESS,
                "wrist_(roll|pitch|yaw)": _FLOATING_WRIST_ROTATION_STIFFNESS,
            },
            damping={
                "wrist_(x|y|z)": _FLOATING_WRIST_TRANSLATION_DAMPING,
                "wrist_(roll|pitch|yaw)": _FLOATING_WRIST_ROTATION_DAMPING,
            },
            friction=0.0,
            armature=0.0,
        ),
        "wuji_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_finger.*_joint.*"],
            effort_limit_sim={
                "right_finger(1|2|3|4|5)_joint(1|2)": 28.0,
                "right_finger(1|2|3|4|5)_joint3": 18.0,
                "right_finger(1|2|3|4|5)_joint4": 10.0,
            },
            velocity_limit_sim={
                "right_finger(1|2|3|4|5)_joint(1|2)": 12.0,
                "right_finger(1|2|3|4|5)_joint3": 14.0,
                "right_finger(1|2|3|4|5)_joint4": 14.0,
            },
            stiffness={
                "right_finger(1|2|3|4|5)_joint(1|2)": 80.0,
                "right_finger(1|2|3|4|5)_joint3": 45.0,
                "right_finger(1|2|3|4|5)_joint4": 24.0,
            },
            damping={
                "right_finger.*_joint(1|2)": 2.4,
                "right_finger.*_joint(3|4)": 1.4,
            },
            friction=0.01,
            armature=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Single articulation: fixed-base 6-DoF virtual wrist plus Wuji 20-DoF right dexterous hand."""
