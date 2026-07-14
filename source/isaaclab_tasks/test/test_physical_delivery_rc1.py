from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
NP_DIR = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np"
sys.path.insert(0, str(NP_DIR))

from wuji_assembly_v2.physical_delivery.contracts import (  # noqa: E402
    CageCandidate,
    CageTopology,
    ControlFeature,
    DeliveryLevel,
    ResetWriteGate,
    video_allowed,
)
from wuji_assembly_v2.physical_delivery.frame_cage_planner import (  # noqa: E402
    _candidate_templates,
    extract_frame_bars,
    ray_through_central_hole,
)
from wuji_assembly_v2.physical_delivery.grasp_executor import CageContactPolicy  # noqa: E402


def _frame_mesh() -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    for center, extents in (
        ((0.018, 0.015, -0.275), (0.036, 0.028, 0.55)),
        ((0.269, 0.015, -0.275), (0.036, 0.028, 0.55)),
        ((0.1435, 0.015, -0.018), (0.287, 0.028, 0.036)),
        ((0.1435, 0.015, -0.532), (0.287, 0.028, 0.036)),
    ):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        boxes.append(mesh)
    mesh = trimesh.util.concatenate(boxes)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def test_frame_bar_extraction_excludes_hole_and_raycast_crosses_it() -> None:
    vertices, faces = _frame_mesh()
    bars, hole = extract_frame_bars(vertices, faces)
    assert set(bars) == {"left_vertical", "right_vertical", "top_horizontal", "bottom_horizontal"}
    center = 0.5 * (hole[0] + hole[1])
    for bar in bars.values():
        lower, upper = map(np.asarray, bar.surface_bounds_object)
        assert not np.all((center >= lower) & (center <= upper))
    assert ray_through_central_hole(vertices, faces, hole)


def test_exactly_twelve_topology_templates_in_priority_order() -> None:
    rows = _candidate_templates()
    assert len(rows) == 12
    assert [row.topology for row in rows[:4]] == [CageTopology.HOOK_THROUGH_FRAME] * 4
    assert [row.topology for row in rows[4:8]] == [CageTopology.THREE_FINGER_WRAP] * 4
    assert [row.topology for row in rows[8:]] == [CageTopology.TWO_FINGER_BRACKET] * 4


def test_candidate_gate_requires_positive_escape_margin() -> None:
    q = tuple(0.0 for _ in range(26))
    feature = ControlFeature(2, "right_finger2_link4", (0, 0, 0), (0, 0, 0), (0, 1, 0), "bar", "side")
    values = dict(
        candidate_id="c",
        topology=CageTopology.TWO_FINGER_BRACKET,
        bar_id="left_vertical",
        finger_group="2",
        preclose_q26=q,
        closed_q26=q,
        standoff_q26=q,
        intended_contact_links=(feature.link_name,),
        intended_contact_surfaces=("bar",),
        control_features=(feature,),
        approach_direction_object=(0, 1, 0),
        cage_margin_m=0.001,
        gravity_escape_gap_m=0.005,
        bar_local_thickness_m=0.010,
        forbidden_penetration_m=0.0,
        self_collision_free=True,
        table_collision_free=True,
        preclose_path_free=True,
        solver_status="VALID",
        failure_reason="",
        source_seed_id="seed",
        runtime_mesh_sha256="mesh",
        max_contact_residual_m=0.0,
        joint_limit_margin_rad=0.01,
    )
    assert CageCandidate(**values).geometry_valid
    assert not CageCandidate(**{**values, "gravity_escape_gap_m": 0.0095}).geometry_valid


def test_contact_policy_releases_and_reacquires() -> None:
    config = {
        "cage": {"dls_damping": 0.01, "runtime_joint_step_limit_rad": 0.002},
        "force": {
            "formal_contact_n": 0.05,
            "keep_contact_n": 0.035,
            "loss_samples": 3,
            "target_n": 0.15,
        },
    }
    policy = CageContactPolicy(config, (False, True, False, False, False))
    jacobians = np.zeros((5, 3, 20))
    jacobians[1, :, (1, 6, 11)] = np.eye(3)
    kwargs = dict(
        tip_positions=np.zeros((5, 3)),
        targets=np.zeros((5, 3)),
        normals=np.tile((1.0, 0.0, 0.0), (5, 1)),
        jacobians=jacobians,
        hand_q=np.zeros(20),
        lower=np.full(20, -1.0),
        upper=np.full(20, 1.0),
    )
    force = np.zeros(5)
    force[1] = 0.06
    pair = np.zeros(5, dtype=bool)
    pair[1] = True
    policy.command(forces=force, pair_contact=pair, **kwargs)
    assert policy.latched[1]
    for _ in range(3):
        policy.command(forces=np.zeros(5), pair_contact=np.zeros(5, dtype=bool), **kwargs)
    assert not policy.latched[1]
    policy.command(forces=force, pair_contact=pair, **kwargs)
    assert policy.latched[1]


def test_post_reset_state_write_fails_closed() -> None:
    gate = ResetWriteGate()
    gate.record("reset_frame_pose")
    gate.record("reset_wrist_joint_state", wrist_state=True)
    gate.lock()
    with pytest.raises(RuntimeError, match="forbidden post-reset"):
        gate.record("late_frame_pose")
    audit = gate.audit()
    assert audit.post_reset_object_root_writes == 1
    assert audit.post_reset_wrist_state_writes == 0


def test_video_requires_real_continuous_delivery_and_no_assistance() -> None:
    assert not video_allowed(
        DeliveryLevel.NONE,
        continuous_rollout=True,
        sticky_used=False,
        snap_used=False,
        proxy_used=False,
        post_reset_root_writes=0,
    )
    assert video_allowed(
        DeliveryLevel.C,
        continuous_rollout=True,
        sticky_used=False,
        snap_used=False,
        proxy_used=False,
        post_reset_root_writes=0,
    )
    assert not video_allowed(
        DeliveryLevel.A,
        continuous_rollout=True,
        sticky_used=True,
        snap_used=False,
        proxy_used=False,
        post_reset_root_writes=0,
    )


def test_probe_import_boundary_excludes_forbidden_legacy_modules() -> None:
    path = REPO_ROOT / "scripts/environments/probe_frame_physical_cage.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    joined = "\n".join(imports).lower()
    for forbidden in ("pipeline.attachment", "pipeline.director", "sticky", "snap", "plug_sticky"):
        assert forbidden not in joined
