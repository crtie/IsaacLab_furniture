# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import importlib.util
import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

# The blacklist is used to prevent importing configs from sub-packages
# TODO(@ashwinvk): Remove pick_place from the blacklist once pinocchio from Isaac Sim is compatibility
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place"]
_DEFER_DISCOVERY_ENV = "ISAACLAB_TASKS_DEFER_DISCOVERY"
# Preserve Isaac Lab's upstream task discovery when the Isaac/Omniverse
# runtime is available. Lightweight tools (unit tests and CLI argument help)
# run before Kit is loaded and intentionally defer discovery until AppLauncher
# starts; they still use explicit mainline registration.
def _kit_log_available() -> bool:
    try:
        return importlib.util.find_spec("omni.log") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def register_upstream_tasks() -> None:
    """Discover IsaacLab's upstream task packages after Kit is initialized."""

    if not _kit_log_available():
        return
    from .utils import import_packages

    import_packages(__name__, _BLACKLIST_PKGS)


def _discovery_deferred() -> bool:
    return os.environ.get(_DEFER_DISCOVERY_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if _kit_log_available() and not _discovery_deferred():
    register_upstream_tasks()
