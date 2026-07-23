"""Formal Sharpawave registration and CLI isolation tests."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from isaaclab_tasks.direct.np.sharpawave_registration import SHARPAWAVE_ENV_SPECS, register_sharpawave_environments
from isaaclab_tasks.robot_adapters import MISSING_CALIBRATION, MissingCapabilityError


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts/environments/run_chair_assembly.py"


def test_sharpawave_registration_is_explicit_and_idempotent():
    ids = {item[0] for item in SHARPAWAVE_ENV_SPECS}
    previous = {env_id: gym.registry.pop(env_id, None) for env_id in ids}
    try:
        assert set(register_sharpawave_environments()) == ids
        assert register_sharpawave_environments() == ()
    finally:
        for env_id in ids:
            gym.registry.pop(env_id, None)
            if previous[env_id] is not None:
                gym.registry[env_id] = previous[env_id]


def test_schema_env_exposes_variant_dimensions_and_calibration_gate():
    ids = {item[0] for item in SHARPAWAVE_ENV_SPECS[:2]}
    previous = {env_id: gym.registry.pop(env_id, None) for env_id in ids}
    try:
        register_sharpawave_environments(include_chair=False)
        floating = gym.make("Isaac-SharpaWave-Floating-Schema-v0")
        fixedrot = gym.make("Isaac-SharpaWave-PegFixedRot-Schema-v0")
        assert floating.action_space.shape == (28,)
        assert fixedrot.action_space.shape == (25,)
        observation, info = floating.reset()
        assert observation["joint_pos"].shape == (28,)
        assert info["not_physical"] is True
        floating.step(np.zeros(28, dtype=np.float32))
        with pytest.raises(MissingCapabilityError) as error:
            floating.step(np.ones(28, dtype=np.float32))
        assert error.value.code == MISSING_CALIBRATION
        floating.close()
        fixedrot.close()
    finally:
        for env_id in ids:
            gym.registry.pop(env_id, None)
            if previous[env_id] is not None:
                gym.registry[env_id] = previous[env_id]


def test_formal_runner_has_only_explicit_policy_and_system_validation_backends():
    spec = importlib.util.spec_from_file_location("formal_chair_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    choices = next(action.choices for action in module._parser()._actions if action.dest == "backend")
    assert tuple(choices) == ("policy", "system-validation")


def test_formal_runner_does_not_import_assistance_or_legacy_pipeline():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(token in name for name in imports for token in ("wuji", "oracle", "sticky", "director"))
