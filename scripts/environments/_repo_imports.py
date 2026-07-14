"""Helpers for preferring this checkout's task extension in runnable scripts."""

from __future__ import annotations

import sys
import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType


def prefer_repo_isaaclab_tasks(anchor_file: str) -> None:
    """Ensure local Isaac Lab task/assets extensions resolve to this repository.

    ``AppLauncher`` may preload an Isaac Lab checkout from another workspace.
    When that happens, Python keeps the first ``isaaclab_tasks`` package in
    ``sys.modules`` and later imports can miss local task packages such as
    ``isaaclab_tasks.direct.np``. Runnable scripts call this immediately after
    launching the app, before importing task modules.
    """

    repo_root = Path(anchor_file).resolve().parents[2]
    local_tasks_source = repo_root / "source" / "isaaclab_tasks"
    local_tasks_package = local_tasks_source / "isaaclab_tasks"
    local_assets_source = repo_root / "source" / "isaaclab_assets"
    local_assets_package = local_assets_source / "isaaclab_assets"
    if not local_tasks_package.is_dir():
        return

    for local_source in (local_assets_source, local_tasks_source):
        if not local_source.is_dir():
            continue
        local_source_str = str(local_source)
        if local_source_str in sys.path:
            sys.path.remove(local_source_str)
        sys.path.insert(0, local_source_str)

    if local_assets_package.is_dir():
        _clear_module_tree("isaaclab_assets")
        # Keep asset loading narrow.  Executing isaaclab_assets.__init__ pulls
        # every robot module; a different Isaac Lab checkout may then import
        # tasks that expect incompatible Wuji symbols.
        _install_namespace_package("isaaclab_assets", local_assets_package)
        _install_namespace_package("isaaclab_assets.robots", local_assets_package / "robots")
        _load_local_module("isaaclab_assets.robots.wuji_hand", local_assets_package / "robots" / "wuji_hand.py")

    _clear_module_tree("isaaclab_tasks")
    # Do not execute isaaclab_tasks.__init__.py here.  Its recursive task
    # discovery can cross into the Isaac Lab checkout that launched the app.
    # The runnable scripts only need the local utils and direct.np package.
    _install_namespace_package("isaaclab_tasks", local_tasks_package)
    _load_local_package("isaaclab_tasks.utils", local_tasks_package / "utils")
    _install_namespace_package("isaaclab_tasks.direct", local_tasks_package / "direct")
    _load_local_package("isaaclab_tasks.direct.np", local_tasks_package / "direct" / "np")


def _clear_module_tree(module_name: str) -> None:
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            del sys.modules[loaded_name]


def _install_namespace_package(module_name: str, package_dir: Path) -> None:
    module = ModuleType(module_name)
    module.__file__ = str(package_dir / "__init__.py")
    module.__path__ = [str(package_dir)]
    module.__package__ = module_name
    spec = importlib.machinery.ModuleSpec(module_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules[module_name] = module
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)


def _load_local_package(module_name: str, package_dir: Path) -> None:
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            del sys.modules[loaded_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    spec.loader.exec_module(module)


def _load_local_module(module_name: str, module_file: Path) -> None:
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            del sys.modules[loaded_name]

    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    spec.loader.exec_module(module)
