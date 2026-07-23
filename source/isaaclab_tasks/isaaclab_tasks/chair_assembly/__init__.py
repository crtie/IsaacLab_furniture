"""Formal public API with lazy imports that keep robot runtime morphology-only."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AssemblyBackend": (".backend", "AssemblyBackend"),
    "AssemblyRuntime": (".backend", "AssemblyRuntime"),
    "AssemblyReport": (".models", "AssemblyReport"),
    "AssemblyTarget": (".models", "AssemblyTarget"),
    "ResultCode": (".models", "ResultCode"),
    "TargetResult": (".models", "TargetResult"),
    "ChairAssemblyRunner": (".runner", "ChairAssemblyRunner"),
    "PolicyAssemblyBackend": (".policy_backend", "PolicyAssemblyBackend"),
    "SystemValidationBackend": (".system_validation", "SystemValidationBackend"),
    "SharpawaveIsaacRuntime": (".sharpawave_runtime", "SharpawaveIsaacRuntime"),
    "load_runtime_calibration": (".sharpawave_runtime", "load_runtime_calibration"),
    "STAGE_TARGET_COUNTS": (".catalog", "STAGE_TARGET_COUNTS"),
    "get_task_catalog": (".catalog", "get_task_catalog"),
    "validate_task_catalog": (".catalog", "validate_task_catalog"),
    "PolicyAction": ("isaaclab_tasks.robot_adapters.policy", "PolicyAction"),
    "PolicyContext": ("isaaclab_tasks.robot_adapters.policy", "PolicyContext"),
    "SkillObservation": ("isaaclab_tasks.robot_adapters.policy", "SkillObservation"),
    "SkillPolicy": ("isaaclab_tasks.robot_adapters.policy", "SkillPolicy"),
    "SharpaWaveAdapter": ("isaaclab_tasks.robot_adapters.sharpa_wave", "SharpaWaveAdapter"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, __name__) if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
