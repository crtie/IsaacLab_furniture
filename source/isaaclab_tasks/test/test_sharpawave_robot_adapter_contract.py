"""Pure-Python contracts for the independent SharpaWave robot adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_tasks.robot_adapters import (
    CONTACT_UNAVAILABLE,
    MISSING_CALIBRATION,
    MissingCapabilityError,
    SchemaValidationError,
    SHARPAWAVE_FINGERTIP_SPECS,
    SharpaWaveAdapter,
    SUPPORTED_VARIANTS,
    get_sharpawave_spec,
    run_sharpawave_preflight,
)


def _runtime(adapter: SharpaWaveAdapter, *, reverse: bool = True, include_contact: bool = True):
    names = list(adapter.canonical_joint_names)
    if reverse:
        names.reverse()
    bodies = [item.pose_frame for item in SHARPAWAVE_FINGERTIP_SPECS]
    if include_contact:
        bodies += [item.contact_body for item in SHARPAWAVE_FINGERTIP_SPECS]
    return SimpleNamespace(
        joint_names=names,
        body_names=bodies,
        joint_pos=np.zeros((2, len(names)), dtype=np.float64),
        joint_vel=np.zeros((2, len(names)), dtype=np.float64),
        body_pos_w=np.zeros((2, len(bodies), 3), dtype=np.float64),
        body_quat_w=np.zeros((2, len(bodies), 4), dtype=np.float64),
        contact_forces=np.ones((2, len(bodies), 3), dtype=np.float64) if include_contact else None,
    )


@pytest.mark.parametrize(("variant", "expected_dim", "expected_wrist"), (("floating", 28, 6), ("peg_fixedrot", 25, 3)))
def test_variant_dimensions_and_semantic_hand_layout(variant: str, expected_dim: int, expected_wrist: int):
    adapter = SharpaWaveAdapter(variant)
    assert adapter.action_dim == expected_dim
    assert adapter.wrist_action_dim == expected_wrist
    assert adapter.finger_action_dim == 22
    assert len(set(adapter.canonical_joint_names)) == expected_dim
    assert tuple(adapter.hand_joint_names) == tuple(adapter.schema.finger_joint_names)
    assert adapter.validate_asset_schema().ok


def test_name_based_action_mapping_survives_runtime_reordering():
    adapter = SharpaWaveAdapter(
        "floating", calibration={"action_scale": np.ones(28, dtype=np.float64)}
    )
    runtime_names = tuple(reversed(adapter.canonical_joint_names)) + ("fixed_helper",)
    action = np.zeros((2, adapter.action_dim), dtype=np.float64)
    action[:, 0] = 0.5
    mapped = adapter.map_canonical_action(action, runtime_joint_names=runtime_names)
    assert mapped.shape == (2, len(runtime_names))
    assert np.allclose(
        mapped[:, runtime_names.index(adapter.canonical_joint_names[0])],
        0.5 * adapter.schema.action_scale[0],
    )
    assert np.allclose(mapped[:, -1], 0.0)


def test_action_contract_rejects_non_batch_nan_and_wrong_schema():
    adapter = SharpaWaveAdapter()
    with pytest.raises(SchemaValidationError):
        adapter.map_canonical_action(np.zeros(adapter.action_dim))
    with pytest.raises(SchemaValidationError):
        adapter.map_canonical_action(np.full((1, adapter.action_dim), np.nan))
    with pytest.raises(SchemaValidationError):
        adapter.map_canonical_action(np.zeros((1, adapter.action_dim + 1)))
    with pytest.raises(SchemaValidationError):
        adapter.map_canonical_action(np.full((1, adapter.action_dim), 2.0))
    with pytest.raises(SchemaValidationError):
        adapter.map_canonical_action(
            np.zeros((1, adapter.action_dim)),
            action_schema_id="sharpawave/wrong/action-v1",
        )


def test_uncalibrated_mapping_only_allows_zero_delta_noop():
    adapter = SharpaWaveAdapter("floating")
    zero = np.zeros((2, adapter.action_dim), dtype=np.float64)
    assert np.allclose(adapter.map_canonical_action(zero), 0.0)
    configuration = adapter.default_configuration()
    assert configuration["action_scale"] is None
    assert configuration["action_scale_status"] == "missing_calibration"
    assert configuration["action_scale_calibrated"] is False
    with pytest.raises(MissingCapabilityError) as delta_error:
        adapter.map_canonical_action(np.full_like(zero, 0.1))
    assert delta_error.value.code == MISSING_CALIBRATION
    with pytest.raises(MissingCapabilityError) as target_error:
        adapter.map_canonical_action(zero, profile="joint_target")
    assert target_error.value.code == MISSING_CALIBRATION


def test_uncalibrated_tiny_nonzero_delta_is_not_treated_as_zero():
    """Do not let a tolerance silently turn a real command into a no-op."""

    adapter = SharpaWaveAdapter("floating")
    action = np.zeros((1, adapter.action_dim), dtype=np.float64)
    action[0, 0] = 1.0e-13
    with pytest.raises(MissingCapabilityError) as error:
        adapter.map_canonical_action(action)
    assert error.value.code == MISSING_CALIBRATION


@pytest.mark.parametrize(
    "scale",
    (
        np.ones(27, dtype=np.float64),
        np.full(28, np.nan, dtype=np.float64),
        np.zeros(28, dtype=np.float64),
    ),
)
def test_invalid_action_scale_is_a_structured_calibration_error(scale):
    adapter = SharpaWaveAdapter("floating", calibration={"action_scale": scale})
    with pytest.raises(MissingCapabilityError) as error:
        adapter.map_canonical_action(np.zeros((1, adapter.action_dim)))
    assert error.value.code == MISSING_CALIBRATION


def test_explicit_action_scale_enables_delta_and_target_profiles():
    scale = np.linspace(0.01, 0.28, 28, dtype=np.float64)
    adapter = SharpaWaveAdapter("floating", calibration={"action_scale": scale})
    action = np.ones((1, adapter.action_dim), dtype=np.float64)
    delta = adapter.map_canonical_action(action, profile="joint_delta")
    assert np.allclose(delta[0], scale)
    target = adapter.map_canonical_action(np.zeros_like(action), profile="joint_target")
    assert np.allclose(target[0], (adapter.schema.lower + adapter.schema.upper) / 2.0)
    configuration = adapter.default_configuration()
    assert configuration["action_scale_status"] == "calibrated"
    assert configuration["action_scale_calibrated"] is True


def test_runtime_observation_and_contact_mapping_are_semantic():
    adapter = SharpaWaveAdapter()
    runtime = _runtime(adapter)
    report = adapter.validate_runtime_articulation(runtime, require_contact=True)
    assert report.ok
    observation = adapter.get_robot_observation(runtime)
    assert observation.joint_position.shape == (2, adapter.action_dim)
    assert observation.contact_measurements is not None
    assert observation.contact_measurements.available
    assert set(observation.contact_measurements.values) == {item.role for item in SHARPAWAVE_FINGERTIP_SPECS}


def test_mapping_valued_runtime_state_is_not_reordered_twice():
    adapter = SharpaWaveAdapter("peg_fixedrot")
    runtime = _runtime(adapter)
    runtime.joint_pos = {
        name: np.full(2, index, dtype=np.float64)
        for index, name in enumerate(adapter.canonical_joint_names)
    }
    runtime.joint_vel = {
        name: np.full(2, -index, dtype=np.float64)
        for index, name in enumerate(adapter.canonical_joint_names)
    }

    observation = adapter.get_robot_observation(runtime)

    expected = np.arange(adapter.action_dim, dtype=np.float64)
    assert np.allclose(observation.joint_position[0], expected)
    assert np.allclose(observation.joint_velocity[0], -expected)


def test_contact_is_unavailable_instead_of_fabricated_zero_values():
    adapter = SharpaWaveAdapter()
    runtime = _runtime(adapter, include_contact=False)
    result = adapter.get_contact_measurements(runtime)
    assert not result.available
    assert result.values == {}
    with pytest.raises(MissingCapabilityError) as exc_info:
        adapter.get_contact_measurements(runtime, strict=True)
    assert exc_info.value.code == CONTACT_UNAVAILABLE


def test_contact_history_reduction_preserves_body_axis():
    adapter = SharpaWaveAdapter()
    runtime = _runtime(adapter)
    runtime.contact_forces = np.ones((2, 3, len(runtime.body_names), 3), dtype=np.float64)
    result = adapter.get_contact_measurements(runtime)
    assert result.available
    assert result.values["thumb"].shape == (2,)
    assert np.allclose(result.values["thumb"], 3.0 * np.sqrt(3.0))


def test_calibration_modes_fail_explicitly_until_profiles_are_supplied():
    adapter = SharpaWaveAdapter()
    with pytest.raises(MissingCapabilityError) as exc_info:
        adapter.hand_action_for_mode("open")
    assert exc_info.value.code == MISSING_CALIBRATION


def test_asset_package_absolute_paths_are_normalized_in_public_schema():
    adapter = SharpaWaveAdapter.from_spec(get_sharpawave_spec("floating"))
    configuration = adapter.default_configuration()
    assert not str(configuration["urdf_path"]).startswith("/")
    assert not str(configuration["usd_path"]).startswith("/")


def test_sharpawave_preflight_writes_only_to_explicit_output(tmp_path):
    output = tmp_path / "preflight"
    assert run_sharpawave_preflight(variant="floating", output_dir=output) == 0
    assert (output / "sharpawave_floating_preflight.json").is_file()


def test_explicit_finger_only_calibration_expands_without_inventing_wrist_values():
    adapter = SharpaWaveAdapter(
        calibration={"open": np.zeros((1, 22), dtype=np.float64)}
    )
    action = adapter.open_hand_action(batch_size=3)
    assert action.shape == (3, adapter.action_dim)
    assert np.allclose(action[:, : adapter.wrist_action_dim], 0.0)


def test_nonzero_named_hand_profile_cannot_bypass_action_scale_gate():
    adapter = SharpaWaveAdapter(
        calibration={"open": np.full((1, 22), 0.1, dtype=np.float64)}
    )
    with pytest.raises(MissingCapabilityError) as error:
        adapter.open_hand_action()
    assert error.value.code == MISSING_CALIBRATION


def test_runtime_schema_missing_joint_is_reported_without_index_fallback():
    adapter = SharpaWaveAdapter("peg_fixedrot")
    runtime = _runtime(adapter)
    runtime.joint_names = runtime.joint_names[:-1]
    report = adapter.validate_runtime_articulation(runtime)
    assert not report.ok
    assert report.missing
