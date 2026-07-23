"""Pure contract tests for the independent SharpaWave adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_tasks.robot_adapters import (
    CONTACT_UNAVAILABLE,
    MissingCapabilityError,
    SharpaWaveAdapter,
    create_robot_adapter,
)
from isaaclab_tasks.robot_adapters.policy import PolicyManifestError, load_policy_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/robot_adapters/sharpa_wave.py"


@pytest.mark.parametrize(
    ("variant", "expected_base", "expected_dim"),
    (("floating", 6, 28), ("peg_fixedrot", 3, 25)),
)
def test_sharpawave_variant_dimensions_and_asset_schema(variant: str, expected_base: int, expected_dim: int):
    adapter = SharpaWaveAdapter(variant)
    assert adapter.wrist_action_dim == expected_base
    assert adapter.finger_action_dim == 22
    assert adapter.action_dim == expected_dim
    report = adapter.validate_asset_schema()
    assert report.ok, report.as_dict()
    assert len(adapter.model.fixed_joint_names) >= 10
    assert len(adapter.model.fingertips) == 5


def test_runtime_joint_order_is_resolved_by_name_and_not_traversal_order():
    adapter = SharpaWaveAdapter("floating")
    runtime_names = list(reversed(adapter.canonical_joint_names)) + ["root_joint"]
    mapped = adapter.map_canonical_action(
        np.zeros((2, adapter.action_dim), dtype=np.float32),
        runtime_joint_names=runtime_names,
    )
    assert mapped.shape == (2, len(runtime_names))
    assert np.allclose(mapped, 0.0)
    with pytest.raises(Exception, match="missing|schema"):
        adapter.resolve_runtime_indices(runtime_names[1:])


def test_action_validation_rejects_wrong_width_nan_and_out_of_range():
    adapter = SharpaWaveAdapter("peg_fixedrot")
    with pytest.raises(Exception):
        adapter.map_canonical_action(np.zeros((1, adapter.action_dim - 1)))
    with pytest.raises(Exception):
        adapter.map_canonical_action(np.full((1, adapter.action_dim), np.nan))
    with pytest.raises(Exception):
        adapter.map_canonical_action(np.full((1, adapter.action_dim), 2.0))


def test_pose_frames_and_contact_bodies_are_distinct():
    adapter = SharpaWaveAdapter("floating")
    body_names = list(adapter.model.fingertips[i].contact_body for i in range(5))
    body_names += list(adapter.model.fingertips[i].pose_frame for i in range(5))
    fake = SimpleNamespace(
        joint_names=list(adapter.canonical_joint_names),
        body_names=body_names,
        data=SimpleNamespace(
            joint_pos=np.zeros((1, adapter.action_dim)),
            joint_vel=np.zeros((1, adapter.action_dim)),
            body_pos_w=np.zeros((1, len(body_names), 3)),
            body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, len(body_names), 1)),
        ),
    )
    frames = adapter.get_fingertip_frames(fake)
    assert frames["thumb"]["pose_frame"] == "right_thumb_fingertip"
    assert frames["thumb"]["contact_body"] == "right_thumb_elastomer"
    assert frames["thumb"]["pose_index"] != frames["thumb"]["contact_index"]
    contact = adapter.get_contact_measurements(fake)
    assert not contact.available
    with pytest.raises(MissingCapabilityError) as exc_info:
        adapter.get_contact_measurements(fake, strict=True)
    assert exc_info.value.code == CONTACT_UNAVAILABLE


def test_runtime_observation_and_contact_mapping_when_provider_exists():
    adapter = SharpaWaveAdapter("peg_fixedrot")
    body_names = [item.contact_body for item in adapter.model.fingertips]
    body_names += [item.pose_frame for item in adapter.model.fingertips]
    forces = {name: np.ones((2,)) * index for index, name in enumerate(body_names[:5], start=1)}
    fake = SimpleNamespace(
        joint_names=list(reversed(adapter.canonical_joint_names)),
        body_names=body_names,
        data=SimpleNamespace(
            joint_pos=np.zeros((2, adapter.action_dim)),
            joint_vel=np.zeros((2, adapter.action_dim)),
            body_pos_w=np.zeros((2, len(body_names), 3)),
            body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, len(body_names), 1)),
            net_contact_forces=forces,
        ),
    )
    report = adapter.validate_runtime_articulation(fake)
    assert report.ok, report.as_dict()
    observation = adapter.get_robot_observation(fake)
    assert np.asarray(observation.joint_position).shape == (2, adapter.action_dim)
    assert observation.contact_measurements is not None
    assert observation.contact_measurements.available


def test_missing_calibration_never_synthesizes_open_or_close_pose():
    adapter = SharpaWaveAdapter("floating")
    with pytest.raises(MissingCapabilityError) as exc_info:
        adapter.open_hand_action()
    assert exc_info.value.code == "MISSING_CALIBRATION"


def test_sharpawave_module_has_no_legacy_robot_import_or_fixed_action_slice():
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    assert not any("wuji" in name.lower() for name in imported)
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "action[:, 6:" not in source
    assert "right_finger" not in source


def test_factory_never_substitutes_wuji_for_sharpawave():
    adapter = create_robot_adapter("sharpawave", variant="floating")
    assert isinstance(adapter, SharpaWaveAdapter)
    assert adapter.action_dim == 28


def test_policy_manifest_missing_checkpoint_is_explicit(tmp_path: Path):
    manifest = tmp_path / "policies.json"
    manifest.write_text(
        '{"schema_version": 1, "policies": {"pick": {"skill": "pick", '
        '"entrypoint": "demo:Policy", "checkpoint": "missing.ckpt", '
        '"action_schema": "sharpawave/floating/action-v1"}}}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyManifestError) as exc_info:
        load_policy_manifest(manifest)
    assert exc_info.value.code == "POLICY_UNAVAILABLE"

