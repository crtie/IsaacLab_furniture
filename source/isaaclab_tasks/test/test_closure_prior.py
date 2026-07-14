from pathlib import Path
import importlib.util
import io
import pickle
import sys

import pytest


MODULE = (
    Path(__file__).parents[1]
    / "isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/closure_prior.py"
)
SPEC = importlib.util.spec_from_file_location("closure_prior_test_module", MODULE)
prior_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = prior_module
SPEC.loader.exec_module(prior_module)


def test_restricted_unpickler_rejects_non_numpy_global():
    with pytest.raises(pickle.UnpicklingError, match="forbidden pickle global"):
        prior_module.RestrictedNumpyUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_known_recording_is_prior_only_with_expected_pca_and_phases():
    prior = prior_module.analyze_recording(prior_module.FALLBACK_RECORDING)
    assert prior.frame_count == 262
    assert prior.source_joint_count == 8
    assert not prior.source_success
    assert not prior.successful_grasp_demonstration
    assert prior.discontinuity_frames == (60, 133)
    assert prior.cumulative_explained_variance[1] == pytest.approx(0.92416190599)
    assert prior.cumulative_explained_variance[3] == pytest.approx(0.99188575058)
    assert prior.tracking_best_lag_frames == 1
    assert prior.recording_audit["object_motion_max_m"] == 0.0
    assert [(item.kind, item.start_frame, item.end_frame) for item in prior.phases] == [
        ("motion", 0, 12),
        ("hold", 13, 59),
        ("reset_discontinuity", 60, 60),
        ("motion", 61, 80),
        ("hold", 81, 132),
        ("reset_discontinuity", 133, 133),
        ("motion", 134, 247),
        ("secondary_adjustment", 248, 256),
        ("hold", 257, 259),
        ("secondary_adjustment", 260, 261),
    ]
