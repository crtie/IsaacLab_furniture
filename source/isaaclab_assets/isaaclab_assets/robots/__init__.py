# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Configuration for different assets.
##

import importlib


_ROBOT_MODULES = (
    "allegro",
    "ant",
    "anymal",
    "cart_double_pendulum",
    "cartpole",
    "fourier",
    "franka",
    "humanoid",
    "humanoid_28",
    "kinova",
    "quadcopter",
    "ridgeback_franka",
    "sawyer",
    "shadow_hand",
    "spot",
    "unitree",
    "universal_robots",
    "sharpawave",
    "sharpawave_isaac",
)
# Wuji is a legacy, opt-in hand family. Keep its package-level names
# compatible through ``__getattr__`` below, but do not import its Isaac/Kit
# configuration while a different robot (notably SharpaWave) is preflighted.
# Callers that need Wuji can continue to use the explicit
# ``isaaclab_assets.robots.wuji_hand`` module path.
_LAZY_ROBOT_EXPORTS = frozenset(
    {
        "FRANKA_ARM_JOINT_NAMES",
        "WUJI_HAND_FINGER_JOINT_NAMES",
        "FRANKA_WUJI_JOINT_NAMES",
        "FRANKA_WUJI_ACTION_DIM",
        "FRANKA_WUJI_JOINT_DIM",
        "WUJI_FLOATING_WRIST_JOINT_NAMES",
        "WUJI_FLOATING_JOINT_NAMES",
        "WUJI_FLOATING_ACTION_DIM",
        "WUJI_FLOATING_JOINT_DIM",
        "FRANKA_WUJI_HAND_MOUNT_XYZ",
        "FRANKA_WUJI_HAND_MOUNT_RPY",
        "DEX_GRASP_FRAME_LOCAL_POS",
        "DEX_GRASP_FRAME_LOCAL_RPY",
        "DEX_GRASP_FRAME_NAME",
        "WUJI_FINGERTIP_BODY_NAMES",
        "FRANKA_WUJI_HAND_URDF_PATH",
        "WUJI_FLOATING_HAND_URDF_PATH",
        "FRANKA_WUJI_HAND_USD_DIR",
        "FRANKA_WUJI_HAND_USD_PATH",
        "WUJI_FLOATING_HAND_USD_DIR",
        "WUJI_FLOATING_HAND_USD_PATH",
        "FRANKA_WUJI_HAND_CFG",
        "WUJI_FLOATING_HAND_CFG",
    }
)
_OPTIONAL_RUNTIME_PREFIXES = ("carb", "isaaclab", "omni", "pxr")
_OPTIONAL_RUNTIME_NAMES = frozenset({"numpy", "torch", "warp"})


def _is_optional_runtime_failure(error: ModuleNotFoundError) -> bool:
    missing = str(getattr(error, "name", "") or "")
    return missing in _OPTIONAL_RUNTIME_NAMES or any(
        missing == prefix or missing.startswith(f"{prefix}.")
        for prefix in _OPTIONAL_RUNTIME_PREFIXES
    )


def _load_robot_family(module_name: str) -> None:
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as error:
        if _is_optional_runtime_failure(error):
            return
        raise
    exported = getattr(module, "__all__", None)
    names = exported if exported is not None else (
        name for name in vars(module) if not name.startswith("_")
    )
    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)


for _module_name in _ROBOT_MODULES:
    _load_robot_family(_module_name)


def __getattr__(name: str):
    """Load legacy Wuji exports only when a caller explicitly asks for one."""

    if name not in _LAZY_ROBOT_EXPORTS:
        raise AttributeError(name)
    module = importlib.import_module(".wuji_hand", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ROBOT_EXPORTS))

__all__ = tuple(
    name for name in globals() if not name.startswith("_") and name != "importlib"
) + tuple(sorted(_LAZY_ROBOT_EXPORTS))
