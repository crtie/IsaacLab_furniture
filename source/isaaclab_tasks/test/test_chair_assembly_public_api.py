from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from isaaclab_tasks.chair_assembly import ChairAssemblyRunner, STAGE_TARGET_COUNTS, get_task_catalog
from isaaclab_tasks.chair_assembly.policy_backend import PolicyAssemblyBackend
from isaaclab_tasks.chair_assembly.system_validation import SystemValidationBackend
from isaaclab_tasks.chair_assembly.sharpawave_runtime import load_runtime_calibration
from isaaclab_tasks.chair_assembly.asset_install import validate_asset_install
from isaaclab_tasks.robot_adapters.policy import PolicyManifestError, load_policy_manifest


class FakeRuntime:
    robot_name = "sharpawave"
    variant = "floating"
    action_schema_id = "sharpawave.robot_schema.v1.floating"
    action_dim = 28

    def __init__(self):
        self.calls = []
        self.closed = False

    def reset(self): self.calls.append("reset")
    def initialize_target(self, target): self.calls.append(("initialize", target.target_name))
    def observe(self, target, phase): return {"joint_pos": np.zeros(28), "phase": phase}
    def apply_action(self, action, phase):
        assert action.shape == (1, 28)
        self.calls.append(("action", phase))
    def verify_grasp(self, target): return True
    def transport_held_part(self, target): self.calls.append(("transport", target.target_name))
    def verify_insert(self, target): return True
    def release(self, target): self.calls.append(("release", target.target_name))
    def state_token(self, target): return target.target_name
    def close(self): self.closed = True


def _manifest(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "mock.ckpt"
    checkpoint.write_bytes(b"interface-only")
    payload = {
        "schema_version": 1,
        "policies": {
            skill: {
                "skill": skill,
                "entrypoint": "isaaclab_tasks.chair_assembly.mock_policy:ZeroMockPolicy",
                "checkpoint": str(checkpoint),
                "action_schema": "sharpawave.robot_schema.v1.floating",
                "observation_schema": "chair.skill_observation.v1",
                "normalization": {"type": "identity"},
                "control_frequency_hz": 20.0,
            }
            for skill in ("pick", "insert")
        },
    }
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_policy_manifest(path, expected_action_schema="sharpawave.robot_schema.v1.floating")


def test_catalog_has_six_stages_and_22_targets():
    targets = get_task_catalog()
    assert STAGE_TARGET_COUNTS == (5, 5, 5, 1, 3, 3)
    assert len(targets) == 22
    assert tuple(sum(target.stage_id == stage for target in targets) for stage in range(1, 7)) == STAGE_TARGET_COUNTS


def test_system_validation_is_complete_but_never_physical_or_bc():
    report = ChairAssemblyRunner(SystemValidationBackend()).run()
    assert report.targets_completed == 22
    assert report.not_physical is True
    assert report.oracle_visual_only is True
    assert report.bc_training_eligible is False
    assert not report.physical_grasp_success
    assert all(not row.physical_insert_success for row in report.target_results)


def test_formal_policy_backend_calls_pick_transport_insert_without_reset(tmp_path: Path):
    runtime = FakeRuntime()
    report = ChairAssemblyRunner(PolicyAssemblyBackend(runtime, _manifest(tmp_path))).run((4,))
    assert runtime.closed
    assert runtime.calls.count("reset") == 1
    assert [call for call in runtime.calls if isinstance(call, tuple) and call[0] == "action"] == [
        ("action", "pick"), ("action", "insert")
    ]
    assert report.targets_completed == 1
    assert report.not_physical is True
    assert report.bc_training_eligible is False
    assert report.physical_grasp_success is False


def test_policy_manifest_rejects_missing_insert(tmp_path: Path):
    checkpoint = tmp_path / "p.ckpt"
    checkpoint.write_bytes(b"x")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "policies": {"pick": {
        "skill": "pick", "entrypoint": "x:y", "checkpoint": str(checkpoint),
        "action_schema": "a", "observation_schema": "o", "normalization": {}, "control_frequency_hz": 20,
    }}}), encoding="utf-8")
    with pytest.raises(PolicyManifestError) as exc:
        load_policy_manifest(path)
    assert exc.value.code == "POLICY_UNAVAILABLE"


def test_formal_backend_has_no_assistance_imports():
    path = Path(__file__).parents[1] / "isaaclab_tasks/chair_assembly/policy_backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("system_validation", "oracle", "sticky", "wuji_assembly_v2.pipeline")
    assert not any(any(token in name for token in forbidden) for name in imports)


def test_asset_derived_runtime_calibration_is_complete():
    path = Path(__file__).parents[3] / "configs/chair_assembly/sharpawave_runtime_v1.json"
    payload = load_runtime_calibration(path)
    assert payload["action_schema"] == "sharpawave.robot_schema.v1.floating"
    assert len(payload["action_scale"]) == 28
    assert all(value > 0 for value in payload["action_scale"])
    assert payload["physical_grasp_profile"] is None
    assert payload["physical_insert_profile"] is None


def test_sharpawave_asset_install_hash_and_missing_error(tmp_path: Path):
    requirement = Path(__file__).parents[3] / "configs/chair_assembly/sharpawave_asset_requirement.json"
    current = validate_asset_install(requirement)
    assert current["ok"] is True
    assert current["tree_sha256"] == current["expected_tree_sha256"]
    missing = validate_asset_install(requirement, asset_root=tmp_path / "missing")
    assert missing["ok"] is False
    assert missing["code"] == "MISSING_ASSET"
    assert "Place or symlink" in missing["install_hint"]
