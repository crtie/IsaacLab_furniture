"""Static Sharpawave asset contract and external-install checks."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab_assets.robots.sharpawave import get_sharpawave_variant
from isaaclab_tasks.chair_assembly.asset_install import validate_asset_install
from isaaclab_tasks.robot_adapters import SharpaWaveAdapter


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIREMENT = REPO_ROOT / "configs/chair_assembly/sharpawave_asset_requirement.json"


def test_asset_requirement_uses_relative_install_contract_and_full_hash():
    payload = json.loads(REQUIREMENT.read_text(encoding="utf-8"))
    assert payload["install_relative_to_isaaclab_assets_robots"] == "sharpa-wave-description"
    assert payload["file_count"] == 93
    assert len(payload["tree_sha256"]) == 64
    assert all(not Path(item).is_absolute() for item in payload["required_files"])


def test_installed_asset_matches_required_tree_and_mesh_closure():
    report = validate_asset_install(REQUIREMENT)
    assert report["ok"] is True
    for variant in ("floating", "peg_fixedrot"):
        adapter = SharpaWaveAdapter(variant)
        assert adapter.validate_asset_schema().ok


def test_dependency_light_variants_have_expected_dimensions():
    assert get_sharpawave_variant("floating").action_dim == 28
    assert get_sharpawave_variant("peg_fixedrot").action_dim == 25


def test_fixed_joints_are_not_actions():
    for variant in ("floating", "peg_fixedrot"):
        adapter = SharpaWaveAdapter(variant)
        assert set(adapter.model.fixed_joint_names).isdisjoint(adapter.canonical_joint_names)
