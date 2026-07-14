from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
NP_DIR = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np"
if str(NP_DIR) not in sys.path:
    sys.path.insert(0, str(NP_DIR))

from wuji_assembly_v2.physical_delivery.contracts import (  # noqa: E402
    DeliveryLevel,
    FrameStage4Failure,
    ResetWriteGate,
    stage4_delivery_video_allowed,
    video_evidence_matches_numeric_rollout,
)
from wuji_assembly_v2.physical_delivery.frame_fork_support import (  # noqa: E402
    partial_hook_q20,
    stage4_goal_transform,
)
from wuji_assembly_v2.physical_delivery.frame_stage4_evaluator import (  # noqa: E402
    FrameStage4Evaluator,
    attribute_frame_contacts,
)


@dataclass
class _Event:
    actor0: str
    actor1: str


def test_partial_hook_changes_only_active_distal_joints():
    preshape = np.arange(20, dtype=np.float64) * 0.01
    close = preshape + 1.0
    result = partial_hook_q20(preshape, close, 0.25)
    changed = set(np.flatnonzero(np.abs(result - preshape) > 1.0e-12).tolist())
    expected = {(finger - 1) + 5 * joint for finger in (2, 3, 4) for joint in (2, 3)}
    assert changed == expected
    assert np.allclose(result[list(expected)], preshape[list(expected)] + 0.25)


def test_stage4_goal_transform_matches_catalog_position():
    transform = stage4_goal_transform(
        (-0.16, -0.3, 0.74),
        (0.70710678, 0.70710678, 0.0, 0.0),
        [
            [-1.0, 0.0, 0.0, 0.28771558],
            [0.0, -1.0, 0.0, 0.25098464],
            [0.0, 0.0, 1.0, -0.00051539927],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert np.allclose(transform[:3, 3], (0.12771558, -0.29948460073, 0.99098464), atol=1.0e-8)


def test_link3_and_link4_count_as_real_fork_support():
    events = [
        _Event("/World/envs/env_0/Robot/right_finger2_link3", "/World/envs/env_0/Frame"),
        _Event("/World/envs/env_0/Frame", "/World/envs/env_0/Robot/right_finger4_link4"),
    ]
    summary = attribute_frame_contacts(events)
    assert summary.support_fingers == (2, 4)
    assert not summary.illegal_contact
    assert summary.hand_frame_contact


def test_palm_frame_contact_is_illegal():
    summary = attribute_frame_contacts(
        [_Event("/World/envs/env_0/Robot/right_palm_link", "/World/envs/env_0/Frame")]
    )
    assert summary.illegal_contact
    assert summary.hand_frame_contact


def test_delivery_level_evaluation_is_milestone_based():
    evaluator = FrameStage4Evaluator()
    audit = {
        "post_reset_object_root_writes": 0,
        "post_reset_wrist_state_writes": 0,
        "post_reset_fixed_asset_root_writes": 0,
    }
    level_c = evaluator.finalize(
        physical_support_acquired=True,
        physical_lift_success=True,
        preinsert_success=True,
        physical_insert_success=False,
        release_stable=False,
        failure=FrameStage4Failure.INSERT_FAILED,
        write_audit=audit,
        trace_path="trace.csv",
        video_path="debug.mp4",
    )
    assert level_c.delivery_level is DeliveryLevel.C
    level_b = evaluator.finalize(
        physical_support_acquired=True,
        physical_lift_success=True,
        preinsert_success=True,
        physical_insert_success=True,
        release_stable=False,
        failure=FrameStage4Failure.RELEASE_UNSTABLE,
        write_audit=audit,
        trace_path="trace.csv",
        video_path="delivery.mp4",
    )
    assert level_b.delivery_level is DeliveryLevel.B


def test_reset_write_gate_rejects_post_reset_root_state():
    gate = ResetWriteGate()
    gate.record("frame reset")
    gate.record("robot reset", wrist_state=True)
    gate.lock()
    try:
        gate.record("frame teleport")
    except RuntimeError:
        pass
    else:
        raise AssertionError("post-reset write should fail closed")
    assert gate.audit().post_reset_object_root_writes == 1


def test_only_a_or_b_can_publish_formal_delivery_video():
    common = dict(
        continuous_rollout=True,
        trace_matched=True,
        sticky_used=False,
        snap_used=False,
        fixed_joint_used=False,
        proxy_used=False,
        object_follow_used=False,
        post_reset_root_writes=0,
    )
    assert stage4_delivery_video_allowed(DeliveryLevel.A, **common)
    assert stage4_delivery_video_allowed(DeliveryLevel.B, **common)
    assert not stage4_delivery_video_allowed(DeliveryLevel.C, **common)


def test_video_evidence_must_match_the_numeric_attempt():
    assert video_evidence_matches_numeric_rollout("attempt_0", "attempt_0")
    assert not video_evidence_matches_numeric_rollout("attempt_0", "correction_1")
    assert not video_evidence_matches_numeric_rollout("", "correction_1")


def test_rc2_delivery_rebuild_is_explicitly_opt_in():
    source = (ROOT / "scripts/environments/run_frame_stage4_delivery_rc2.py").read_text(encoding="utf-8")
    assert '"--build-delivery"' in source
    assert "if args.build_delivery:\n            _build_delivery(final, selected)" in source


def test_rc2_ast_has_no_legacy_controller_or_search_imports():
    paths = [
        ROOT / "scripts/environments/run_frame_stage4_delivery_rc2.py",
        NP_DIR / "wuji_assembly_v2/physical_delivery/frame_stage4_env.py",
        NP_DIR / "wuji_assembly_v2/physical_delivery/frame_stage4_controller.py",
    ]
    forbidden = (
        "frame_cage_planner",
        "chair4_env",
        "near_grasp.cem",
        "residual_rl",
        "AttachedPartController",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
        joined = " ".join(imports + calls)
        assert not any(value in joined for value in forbidden)
        assert "_create_fixed_joint" not in calls
        assert "_sync_held_asset" not in calls
