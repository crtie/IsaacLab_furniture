# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os
import importlib
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]


_OPTIONAL_RUNTIME_MODULES = frozenset(
    {
        "carb",
        "isaaclab",
        "numpy",
        "omni",
        "pxr",
        "torch",
        "warp",
    }
)


def _is_optional_runtime_failure(error: ModuleNotFoundError) -> bool:
    missing = str(getattr(error, "name", "") or "")
    return missing in _OPTIONAL_RUNTIME_MODULES or any(
        missing.startswith(prefix) for prefix in ("carb.", "isaaclab.", "omni.", "pxr.")
    )


def _wildcard_import_optional(module_name: str) -> None:
    """Load an asset family while keeping dependency-light schemas importable."""

    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if _is_optional_runtime_failure(error):
            return
        raise
    exported = getattr(module, "__all__", None)
    names = exported if exported is not None else (
        name for name in vars(module) if not name.startswith("_")
    )
    lazy_names = frozenset(getattr(module, "_LAZY_ROBOT_EXPORTS", ()))
    for name in names:
        if name in lazy_names:
            continue
        if hasattr(module, name):
            globals()[name] = getattr(module, name)


# The upstream asset families remain available under Kit.  If Kit is absent,
# only families that require it are skipped; the dependency-light SharpaWave
# declarative schema still imports normally for static contracts and manifests.
_wildcard_import_optional(".robots")
_wildcard_import_optional(".sensors")


def __getattr__(name: str):
    """Preserve explicit access to robot-family exports that are intentionally lazy."""

    robots = importlib.import_module(".robots", __name__)
    if name in getattr(robots, "_LAZY_ROBOT_EXPORTS", ()):
        value = getattr(robots, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
