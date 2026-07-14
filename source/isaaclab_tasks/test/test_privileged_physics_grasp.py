from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).parents[1] / "isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
sys.path.insert(0, str(MODULE_DIR))

from near_grasp.grasp_program import resolve_preshape_profile  # noqa: E402
from near_grasp.grasp_synthesis.candidate_cache import CandidateCache, is_formal_gate_candidate  # noqa: E402
from near_grasp.grasp_synthesis.contact_controller import (  # noqa: E402
    ContactControllerObservation,
    ObjectSpaceContactController,
)
from near_grasp.grasp_synthesis.contact_sampler import (  # noqa: E402
    CollisionMesh,
    fallback_variants,
    sample_contact_sets,
)
from near_grasp.grasp_synthesis.forensic_trace import (  # noqa: E402
    ForensicContactEvent,
    ForensicFrame,
    link_contact_source_map,
    sequential_gate_schedule,
)
from near_grasp.grasp_synthesis.grasp_energy import (  # noqa: E402
    SearchMetrics,
    gravity_wrench_feasibility,
)
from near_grasp.grasp_synthesis.object_spec import ObjectGraspSpec, SUPPORTED_PARTS  # noqa: E402
from near_grasp.grasp_synthesis.physics_validator import (  # noqa: E402
    ContactSource,
    GateKind,
    GateResult,
    attribute_contact,
    gate_passed,
)
from near_grasp.grasp_synthesis.wuji_ik import (  # noqa: E402
    WujiKinematicModel,
    closed_pose_collision_is_valid,
)
from near_grasp.hand_prior_adapter import RetargetedPcaPriorAdapter, build_coordex_proprio  # noqa: E402
from near_grasp.residual_rl import DirectContactResidualBounds, ResidualRlGate  # noqa: E402


def _spec(part: str) -> ObjectGraspSpec:
    return ObjectGraspSpec.load(
        REPO_ROOT / f"configs/grasp_synthesis/objects/{part.lower()}.yaml",
        repo_root=REPO_ROOT,
    )


def test_all_object_specs_are_valid_and_hash_stable() -> None:
    parts = {part.lower(): _spec(part).part_name for part in SUPPORTED_PARTS}
    assert set(parts.values()) == set(SUPPORTED_PARTS)
    assert _spec("Plug2").content_hash == _spec("Plug2").content_hash
    assert all(_spec(part).schema_version == 3 for part in SUPPORTED_PARTS)
    audit = _spec("Plug2").audit_static_assets()
    assert audit["static_asset_audit_ok"]
    assert audit["visual_extent_m"] == pytest.approx([0.006, 0.006, 0.02175])
    assert audit["analytic_geometry_matches_visual"]


def test_contact_sampling_is_deterministic_and_opposed() -> None:
    spec = _spec("Plug2")
    mesh = _plug_collision_mesh(spec)
    first = sample_contact_sets(spec, count=32, seed=20260713, collision_mesh=mesh)
    second = sample_contact_sets(spec, count=32, seed=20260713, collision_mesh=mesh)
    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert all(row.opposition_quality >= 0.75 for row in first)
    assert {row.finger_group for row in first} == {"23"}
    assert all(row.raycast_hit and row.axis_delta_m <= 0.0005 for row in first)
    assert all(
        row.gravity_wrench["feasible"]
        or bool((row.metadata.get("third_contact") or {}).get("gravity_wrench", {}).get("feasible", False))
        for row in first
    )
    assert all(abs(contact.normal_object[2]) <= 0.35 for row in first for contact in row.contacts)
    unique_geometry = {
        tuple(round(value, 8) for contact in row.contacts for value in contact.position_object)
        for row in first
    }
    assert len(unique_geometry) >= 8
    variants = fallback_variants(first[0], spec)
    assert [row.fallback_tier for row in variants[:3]] == [
        "antipodal_pair",
        "three_point",
        "alternate_finger_group",
    ]
    assert variants[1].finger_group == "234"
    assert variants[2].finger_group == "34"


def _plug_collision_mesh(spec: ObjectGraspSpec) -> CollisionMesh:
    import trimesh

    geometry = spec.geometry
    mesh = trimesh.creation.cylinder(radius=geometry.radius_m, height=2.0 * geometry.half_length_m, sections=48)
    mesh.apply_translation(np.asarray(geometry.center_object))
    return CollisionMesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        sha256="test-cylinder",
        coordinate_frame="object_local",
        source_prims=("test",),
    )


def test_gravity_wrench_uses_friction_cone_not_two_column_svd() -> None:
    points = np.asarray(((0.003, 0.0, 0.0), (-0.003, 0.0, 0.0)))
    normals = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    feasible = gravity_wrench_feasibility(
        points,
        normals,
        center_of_mass=(0.0, 0.0, 0.0),
        mass_kg=0.01,
        friction=1.0,
        normal_force_range_n=(0.08, 0.30),
    )
    blocked = gravity_wrench_feasibility(
        points,
        normals,
        center_of_mass=(0.0, 0.0, 0.0),
        mass_kg=0.01,
        friction=0.0,
        normal_force_range_n=(0.08, 0.30),
    )
    assert feasible.feasible
    assert feasible.force_residual_n <= 0.001
    assert feasible.torque_residual_nm <= 0.00001
    assert not blocked.feasible
    assert "force_closure_proxy" not in inspect.getsource(WujiKinematicModel.optimize_contact_set)


def test_safe_single_contact_strictly_outranks_no_contact() -> None:
    no_contact = SearchMetrics(
        safe_target_contact_count=0,
        mean_surface_distance_m=0.0,
        opposition_quality=1.0,
        force_closure_quality=1.0,
        force_balance_error_n=0.0,
        penetration_m=0.0,
        table_collision_force_n=0.0,
        relative_drift_m=0.0,
        jerk_metric=0.0,
    )
    safe_single = SearchMetrics(
        safe_target_contact_count=1,
        min_active_finger_contact_duty=0.1,
        mean_surface_distance_m=0.01,
        opposition_quality=0.0,
        force_closure_quality=0.0,
        force_balance_error_n=0.2,
        penetration_m=0.001,
        table_collision_force_n=0.0,
        relative_drift_m=0.01,
        jerk_metric=1.0,
    )
    assert safe_single.ranking_key() > no_contact.ranking_key()


def _controller_observation(force: np.ndarray, target_x: float = 0.001) -> ContactControllerObservation:
    jacobians = np.zeros((5, 3, 20))
    for finger in range(5):
        base = finger
        jacobians[finger, :, [base, base + 5, base + 10]] = np.eye(3)
    targets = np.zeros((5, 3))
    targets[:, 0] = target_x
    normals = np.zeros((5, 3))
    normals[:, 0] = 1.0
    return ContactControllerObservation(
        active_finger_mask=np.asarray([False, True, True, False, False]),
        current_tip_positions=np.zeros((5, 3)),
        target_contact_positions=targets,
        surface_normals_world=normals,
        target_force_norms=force,
        target_force_n=0.0,
        fingertip_jacobians=jacobians,
        current_hand_q=np.zeros(20),
        joint_lower=np.full(20, -1.0),
        joint_upper=np.full(20, 1.0),
    )


def test_contact_controller_reduces_error_and_reacquires_after_three_losses() -> None:
    controller = ObjectSpaceContactController()
    command = controller.step(_controller_observation(np.zeros(5)))
    assert command.hand_delta20[1] > 0.0
    force = np.zeros(5)
    force[1] = 0.06
    command = controller.step(_controller_observation(force, target_x=0.0))
    assert command.latched_contact[1]
    for _ in range(2):
        command = controller.step(_controller_observation(np.zeros(5), target_x=0.0))
        assert command.latched_contact[1]
    command = controller.step(_controller_observation(np.zeros(5), target_x=0.0))
    assert not command.latched_contact[1]
    assert command.reacquiring[1]


def test_contact_attribution_never_invents_a_body_from_residual() -> None:
    target = attribute_contact(
        all_force_xyz=(0.2, 0.0, 0.0),
        filtered_forces={"TARGET_OBJECT": (0.2, 0.0, 0.0), "TABLE": (0, 0, 0), "GROUND": (0, 0, 0)},
        filter_valid={"TARGET_OBJECT": True, "TABLE": True, "GROUND": True},
    )
    assert target.source == ContactSource.TARGET_OBJECT
    unresolved = attribute_contact(
        all_force_xyz=(0.2, 0.0, 0.0),
        filtered_forces={"TARGET_OBJECT": (0, 0, 0), "TABLE": (0, 0, 0), "GROUND": (0, 0, 0)},
        filter_valid={"TARGET_OBJECT": True, "TABLE": False, "GROUND": False},
    )
    assert unresolved.source == ContactSource.UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT


def test_gate_a_cannot_claim_grasp_or_full_route() -> None:
    with pytest.raises(ValueError):
        GateResult(
            gate=GateKind.RESET_CLOSED_HOLD_LIFT,
            candidate_id="c0",
            trial_id=0,
            passed=True,
            termination_layer="PASS",
            hold_contact_steps=30,
            hold_total_steps=30,
            peak_target_force_n=0.2,
            lift_contact_duty=1.0,
            object_z_gain_m=0.012,
            lost_table_support=True,
            relative_drift_m=0.001,
            physical_grasp_success=True,
            physical_lift_success=True,
            full_route_success=False,
            oracle_reset_grasp=True,
        )
    rows = [
        GateResult(
            gate=GateKind.STANDOFF_APPROACH_CLOSE_LIFT,
            candidate_id="c1",
            trial_id=index,
            passed=index < 3,
            termination_layer="PASS" if index < 3 else "GATE_C_STANDOFF_APPROACH_CLOSE_LIFT_FAILED",
            hold_contact_steps=24,
            hold_total_steps=30,
            peak_target_force_n=0.2,
            lift_contact_duty=0.8,
            object_z_gain_m=0.010,
            lost_table_support=True,
            relative_drift_m=0.005,
            physical_grasp_success=index < 3,
            physical_lift_success=index < 3,
            full_route_success=False,
            oracle_reset_grasp=False,
            approach_close_lift_success=index < 3,
        )
        for index in range(5)
    ]
    assert gate_passed(rows)
    assert sum(row.approach_close_lift_success for row in rows) == 3
    assert not any(row.full_route_success for row in rows)
    with pytest.raises(ValueError, match="stop at lift"):
        GateResult(
            **{
                **rows[0].__dict__,
                "full_route_success": True,
            }
        )


def test_v3_delivery_aggregator_never_promotes_gate_c_to_full_route() -> None:
    source = (REPO_ROOT / "scripts/environments/build_multi_object_privileged_grasp_v3_delivery.py").read_text(
        encoding="utf-8"
    )
    assert 'gate_c.get("approach_close_lift_success", False)' in source
    assert '"full_route_success": False' in source
    assert 'gate_c.get("full_route_success", False)' not in source


def test_candidate_cache_is_immutable(tmp_path: Path) -> None:
    cache = CandidateCache(tmp_path)
    key = cache.input_hash(spec={"part": "Plug2"}, fingerprints={"urdf": "a"})
    cache.save(part_name="Plug2", input_hash=key, candidates=[{"id": 1}], metadata={})
    cache.save(part_name="Plug2", input_hash=key, candidates=[{"id": 1}], metadata={})
    with pytest.raises(FileExistsError):
        cache.save(part_name="Plug2", input_hash=key, candidates=[{"id": 2}], metadata={})


def test_formal_gate_rejects_residual_failure_optimization_failure_and_legacy_cache(tmp_path: Path) -> None:
    valid = {
        "schema_version": 3,
        "optimization_success": True,
        "contact_residual_success": True,
        "physics_gate_a_eligible": True,
        "gate_eligibility": {"gate_a": True, "gate_b": False, "gate_c": False},
    }
    assert is_formal_gate_candidate(valid, "gate_a")
    assert not is_formal_gate_candidate({**valid, "optimization_success": False}, "gate_a")
    assert not is_formal_gate_candidate({**valid, "contact_residual_success": False}, "gate_a")
    cache = CandidateCache(tmp_path)
    legacy = cache.path_for("Rod", "old")
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"schema_version":1}\n', encoding="utf-8")
    legacy.with_name("manifest.json").write_text(
        '{"schema_version":1,"candidate_schema_version":1,"part_name":"Rod","input_hash":"old"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy candidate cache"):
        cache.load_validated(legacy, expected_part_name="Rod")


def test_closed_collision_validity_is_independent_from_contact_residual() -> None:
    assert closed_pose_collision_is_valid({"valid": True}, 0.0)
    candidate = {
        "schema_version": 3,
        "optimization_success": True,
        "contact_residual_success": False,
        "closed_pose_collision_success": True,
        "physics_gate_a_eligible": True,
        "gate_eligibility": {"gate_a": True},
        "tip_error_m": 0.004,
    }
    assert not is_formal_gate_candidate(candidate, "gate_a")


def test_frame_regions_exclude_central_hole_and_big_objects_offer_two_and_three_finger_groups() -> None:
    frame = _spec("Frame")
    hole = np.asarray((0.1435, 0.015, -0.275))
    assert not any(
        np.all(np.abs(hole - np.asarray(region.center_object)) <= np.asarray(region.half_extents_m))
        for region in frame.allowed_contact_regions
    )
    for part in ("Rod", "Backrest", "Frame"):
        groups = _spec(part).allowed_finger_groups
        assert any(len(group) == 2 for group in groups)
        assert any(len(group) == 3 for group in groups)


def test_coordex_proprio_uses_official_field_order() -> None:
    proprio = build_coordex_proprio(
        palm_linear_velocity_body=np.asarray((1, 2, 3)),
        palm_angular_velocity_body=np.asarray((4, 5, 6)),
        joint_position=np.arange(20) + 10,
        default_joint_position=np.arange(20),
        joint_velocity=np.arange(20) + 20,
        default_joint_velocity=np.arange(20),
        previous_joint_action=np.arange(20) + 30,
    )
    assert proprio.shape == (66,)
    assert proprio[:6].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.all(proprio[6:26] == 10)
    assert np.all(proprio[26:46] == 20)
    assert np.all(proprio[46:66] == np.arange(20) + 30)


def test_preshape_and_pca_endpoint_are_executable() -> None:
    preshape = np.zeros(20)
    close = np.ones(20)
    pinch = resolve_preshape_profile("retargeted_pinch", preshape, close, (False, True, True, False, False))
    straddle = resolve_preshape_profile("retargeted_straddle", preshape, close, (False, True, True, False, False))
    assert np.max(pinch) == pytest.approx(0.12)
    assert np.max(straddle) == pytest.approx(0.04)
    artifact = RetargetedPcaPriorAdapter.build_runtime_fallback(preshape, close, steps=16)
    adapter = RetargetedPcaPriorAdapter(artifact)
    adapter.reset(np.zeros(1), preshape)
    nominal = adapter.decode(np.zeros(1), np.zeros(6), 1.0)
    latent = adapter.decode(np.zeros(1), np.ones(6) * 0.1, 1.0)
    assert not np.allclose(latent, nominal)


def test_direct_residual_gate_and_mask() -> None:
    bounds = DirectContactResidualBounds()
    decoded = bounds.decode(np.ones(21), (False, True, True, False, False))
    assert np.all(decoded[:3] == 0.0)
    assert np.all(decoded[3:9] != 0.0)
    assert np.all(decoded[9:15] == 0.0)
    blocked = ResidualRlGate(
        optimized_pregrasp_available=True,
        safe_contact_trials=3,
        unstable_hold_trials=3,
        eligibility_packet_present=False,
    )
    assert not blocked.training_allowed


def test_wuji_pinocchio_compatibility_audit() -> None:
    urdf = REPO_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/wuji-hand-description/urdf/right_block_palm_floating_drone.urdf"
    model = WujiKinematicModel(urdf)
    assert model.model.nq == 26
    assert len(model.tip_frame_ids) == 5
    import trimesh

    mesh_root = urdf.parent.parent / "meshes/right"
    for index, values in enumerate(model.tip_collision_vertices, 1):
        link4 = trimesh.load_mesh(mesh_root / f"right_finger{index}_link4.STL", process=False)
        assert len(values) > len(link4.vertices)


def test_stage_a_has_one_bounded_nonlinear_fallback_per_target() -> None:
    source = inspect.getsource(WujiKinematicModel.optimize_contact_set_staged)
    assert source.count("least_squares(") == 1
    assert "seed_lower[3:6]" in source
    assert "stage_b_seed_evaluations" in source


def test_generic_controller_has_no_part_name_branch() -> None:
    source = inspect.getsource(ObjectSpaceContactController)
    for part in SUPPORTED_PARTS:
        assert part not in source


def test_forensic_schedule_trace_and_all_link_attribution() -> None:
    schedule = sequential_gate_schedule([f"c{index}" for index in range(16)], repeats=5)
    assert len(schedule) == 80
    assert schedule[:5] == tuple(("c0", index) for index in range(5))
    event = ForensicContactEvent(
        actor0="/World/envs/env_0/Robot/right_finger3_link4",
        actor1="/World/envs/env_0/TargetObject",
        contact_count=1,
    )
    sources = link_contact_source_map(("right_finger3_link4", "right_palm_link"), (event,))
    assert sources["right_finger3_link4"] == ("TARGET_OBJECT",)
    assert sources["right_palm_link"] == ("NO_CONTACT",)
    zeros26 = tuple(0.0 for _ in range(26))
    tips = tuple((0.0, 0.0, 0.0) for _ in range(5))
    quats = tuple((1.0, 0.0, 0.0, 0.0) for _ in range(5))
    frame = ForensicFrame(
        physics_frame_id=0,
        candidate_id="c0",
        reset_index=0,
        stage="SETTLE",
        applied_target26=zeros26,
        next_target26=zeros26,
        actual_joint26=zeros26,
        joint_velocity26=zeros26,
        joint_error26=zeros26,
        isaac_tip_positions=tips,
        isaac_tip_quat_wxyz=quats,
        target_contact_positions=tips,
        object_position=(0.0, 0.0, 0.0),
        object_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        object_linear_velocity=(0.0, 0.0, 0.0),
        object_angular_velocity=(0.0, 0.0, 0.0),
        target_force_norms=(0.0,) * 5,
        target_force_xyz=tips,
        all_force_xyz=tips,
        table_force_xyz=tips,
        ground_force_xyz=tips,
        contact_events=(event,),
        link_contact_sources=sources,
        terminal=True,
        termination_layer="HARD_FORCE_ABORT",
    )
    assert frame.to_dict()["physics_frame_id"] == 0
    assert frame.to_dict()["terminal"]
