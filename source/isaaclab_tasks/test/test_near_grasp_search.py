from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MODULE_DIR = Path(__file__).parents[1] / "isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
sys.path.insert(0, str(MODULE_DIR))

from near_grasp.cem import (  # noqa: E402
    CandidateResult,
    MixedCem,
    MixedCemConfig,
)
from near_grasp.configuration import NearGraspRunConfig  # noqa: E402
from near_grasp.evaluator import (  # noqa: E402
    EvaluationInput,
    PhysicalEvaluation,
    StrictPhysicalEvaluator,
)
from near_grasp.grasp_program import (  # noqa: E402
    PROGRAM_DIM,
    FirstContactFreezer,
    GraspProgram,
    GraspProgramBounds,
    default_screw1_templates,
    ProgramTermination,
)
from near_grasp.hand_prior_adapter import (  # noqa: E402
    CoorDexWujiPriorAdapter,
    RetargetedPcaPriorAdapter,
)
from near_grasp.observation import (  # noqa: E402
    NEAR_GRASP_OBSERVATION_SCHEMA,
)
from near_grasp.residual_rl import (  # noqa: E402
    CurriculumStage,
    EvaluationRates,
    ResidualActionBounds,
    ResidualRlGate,
    residual_reward,
)
from near_grasp.replay import (  # noqa: E402
    ReplayRequest,
    attribute_contact_step,
    evaluation_class,
    load_candidate_rows,
    select_candidate,
)


def _evaluation(*, lift=False, close=False, duty=0.0, peak=0.0, invalid=False) -> PhysicalEvaluation:
    return PhysicalEvaluation(
        valid_candidate=not invalid,
        hard_invalid=invalid,
        invalid_reasons=("invalid",) if invalid else (),
        physical_lift_success=lift,
        stable_close=close,
        simultaneous_contact_duty=duty,
        lift_contact_duty=duty,
        peak_target_force_n=peak,
        safe_force=peak < 1.0,
        object_z_gain_m=0.011 if lift else 0.0,
        table_clearance_m=0.011 if lift else 0.0,
        lost_table_support=lift,
        lateral_displacement_m=0.0,
        relative_drift_m=0.0,
        jerk_metric=0.0,
        target_filtered_success_evidence=duty > 0.0,
        honesty={},
        metadata={},
    )


def test_grasp_program_decoding_and_bounds() -> None:
    lower, upper = GraspProgramBounds().arrays()
    program = GraspProgram.from_vector(np.linspace(-100.0, 100.0, PROGRAM_DIM), 2)
    values = np.asarray(program.values)
    assert np.all(values >= lower)
    assert np.all(values <= upper)
    assert isinstance(program.approach_steps, int)
    assert isinstance(program.first_contact_hold_steps, int)
    assert len(program.hand_latent6) == 6


def test_templates_are_bounded_frozen_seed_variants() -> None:
    templates = default_screw1_templates()
    assert len(templates) == 6
    assert {template.template_id for template in templates} == set(range(6))
    assert {template.preshape_profile for template in templates} == {"retargeted_pinch", "retargeted_straddle"}
    assert all(template.active_finger_mask[2] and template.active_finger_mask[3] for template in templates)


def test_first_contact_freezes_only_contacted_finger() -> None:
    freezer = FirstContactFreezer()
    delta = np.ones(20)
    forces = np.zeros(5)
    forces[2] = 0.051
    out = freezer.apply(delta, forces, (False, False, True, True, False))
    names = freezer.joint_names
    assert all(out[index] == 0.0 for index, name in enumerate(names) if "finger3_" in name)
    assert all(out[index] == 1.0 for index, name in enumerate(names) if "finger4_" in name)
    forces[:] = 0.0
    for _ in range(2):
        out = freezer.apply(delta, forces, (False, False, True, True, False))
        assert all(out[index] == 0.0 for index, name in enumerate(names) if "finger3_" in name)
    out = freezer.apply(delta, forces, (False, False, True, True, False))
    assert all(out[index] == 1.0 for index, name in enumerate(names) if "finger3_" in name)


def test_pca_fit_is_deterministic_and_decodes_20d() -> None:
    rng = np.random.default_rng(19)
    trajectories = rng.normal(size=(8, 24, 20)) * 0.01
    first = RetargetedPcaPriorAdapter.fit(trajectories, source="test")
    second = RetargetedPcaPriorAdapter.fit(trajectories, source="test")
    assert first.to_dict() == second.to_dict()
    adapter = RetargetedPcaPriorAdapter(first)
    adapter.reset(np.zeros(3), np.zeros(20))
    target = adapter.decode(np.zeros(3), np.ones(6) * 0.1, 0.5)
    assert target.shape == (20,)
    tensor_target = adapter.decode_torch(
        torch.zeros((1, 3)),
        torch.ones((1, 6)) * 0.1,
        torch.tensor([0.5]),
    )
    assert tensor_target.shape == (1, 20)
    assert np.allclose(tensor_target.numpy()[0], target, atol=1.0e-6)
    assert adapter.signature["program_latent_dim"] == 6


def test_coordex_checkpoint_adapter_signature_when_checkpoint_present() -> None:
    checkpoint = Path("third_party/external_grasp_baselines/coordex/ckpts/hand_prior/kinematic_wrist_16k.pt")
    if not checkpoint.is_file():
        pytest.skip("isolated CoorDex checkpoint not cloned")
    adapter = CoorDexWujiPriorAdapter(checkpoint)
    adapter.reset(np.zeros(66), np.zeros(20))
    target = adapter.decode(np.zeros(66), np.zeros(6), 0.5)
    assert target.shape == (20,)
    assert adapter.signature["native_latent_dim"] == 12
    assert adapter.signature["redistribution_allowed"] is False


def test_named_observation_slices_round_trip() -> None:
    fields = {
        name: np.full((3, width), index, dtype=np.float32)
        for index, (name, width) in enumerate(NEAR_GRASP_OBSERVATION_SCHEMA.names_and_widths)
    }
    packed = NEAR_GRASP_OBSERVATION_SCHEMA.pack(fields)
    assert packed.shape == (3, 169)
    unpacked = NEAR_GRASP_OBSERVATION_SCHEMA.unpack(packed)
    assert set(unpacked) == set(fields)
    assert NEAR_GRASP_OBSERVATION_SCHEMA.slices["previous_action"] == slice(155, 169)


def test_strict_evaluator_ignores_reward_and_requires_physics() -> None:
    steps = 80
    force = np.zeros((steps, 5))
    force[20:, 2:4] = 0.15
    object_pos = np.zeros((steps, 3))
    hand_pos = np.zeros((steps, 3))
    object_pos[:, 2] = 0.742
    hand_pos[:, 2] = 0.80
    object_pos[60:, 2] += np.linspace(0.0, 0.012, steps - 60)
    hand_pos[60:, 2] += np.linspace(0.0, 0.012, steps - 60)
    evidence = EvaluationInput(
        target_force_norms=force,
        active_finger_mask=(False, False, True, True, False),
        object_positions=object_pos,
        hand_positions=hand_pos,
        table_top_z_m=0.740,
        close_start_step=20,
        lift_start_step=60,
        controlled_close_completed=True,
        object_supported_by_table=np.asarray([True] * 61 + [False] * 19),
        metadata={"reward_total": 1.0e9},
    )
    result = StrictPhysicalEvaluator().evaluate(evidence)
    assert result.physical_lift_success
    invalid = StrictPhysicalEvaluator().evaluate(
        EvaluationInput(**{**evidence.__dict__, "target_force_norms": np.zeros_like(force)})
    )
    assert not invalid.physical_lift_success
    assert not invalid.target_filtered_success_evidence


def test_force_guards_hard_invalidate_candidate() -> None:
    steps = 40
    force = np.zeros((steps, 5))
    force[:, 2:4] = 5.0
    result = StrictPhysicalEvaluator().evaluate(
        EvaluationInput(
            target_force_norms=force,
            active_finger_mask=(False, False, True, True, False),
            object_positions=np.zeros((steps, 3)),
            hand_positions=np.zeros((steps, 3)),
            table_top_z_m=0.0,
            close_start_step=0,
            lift_start_step=30,
            controlled_close_completed=True,
        )
    )
    assert result.hard_invalid
    assert "hard_force_abort" in result.invalid_reasons


def test_cem_elite_order_and_update() -> None:
    config = MixedCemConfig(population=8, physical_batch=4, elite_count=2, min_generations=5, max_generations=5)
    cem = MixedCem(template_count=3, config=config)
    programs = cem.sample()
    results = []
    for index, program in enumerate(programs):
        results.append(
            CandidateResult(
                candidate_id=index,
                program=program,
                evaluation=_evaluation(lift=index == 5, close=index in {2, 5}, duty=index / 8.0),
                reset_seed=index,
            )
        )
    elite = cem.elites(results, 2)
    assert elite[0].candidate_id == 5
    row = cem.update(results)
    assert row["generation"] == 1
    assert np.isclose(np.sum(cem.probabilities), 1.0)
    assert np.all(cem.probabilities > 0.0)


def test_residual_bounds_and_curriculum_checkpoint_does_not_advance() -> None:
    decoded = ResidualActionBounds().decode(np.ones(14) * 2.0)
    assert decoded.shape == (14,)
    gate = ResidualRlGate(
        optimized_pregrasp_available=True,
        safe_contact_trials=3,
        unstable_hold_trials=3,
        eligibility_packet_present=True,
    )
    assert not gate.consider(EvaluationRates(511, 1.0, 1.0, 1.0, 1.0), checkpoint_created=True)
    assert gate.stage == CurriculumStage.CONTACT
    assert gate.consider(EvaluationRates(512, 0.7, 0.0, 0.0, 0.0), checkpoint_created=False)
    assert gate.stage == CurriculumStage.MULTI_CONTACT


def test_reward_is_separate_from_evaluator() -> None:
    source = inspect.getsource(StrictPhysicalEvaluator.evaluate)
    assert "reward_total" not in source
    reward = residual_reward(
        {
            "target_contact": 1.0,
            "contact_duty": 1.0,
            "stable_close": 1.0,
            "lift_height_m": 0.01,
            "force_excess_n": 0.0,
            "collision": 0.0,
            "object_displacement_m": 0.0,
            "object_velocity": 0.0,
            "contact_loss": 0.0,
            "jerk": 0.0,
        }
    )
    assert float(reward) > 0.0


def test_dedicated_env_has_no_part_name_controller_branches_and_reset_only_writes() -> None:
    env_source = (
        MODULE_DIR / "near_grasp/near_grasp_physics_env.py"
    ).read_text(encoding="utf-8")
    assert "Plug2" not in env_source
    assert "V83_PARTS" not in env_source
    assert "v86" not in env_source.lower()
    reset_start = env_source.index("    def _reset_idx(")
    reset_end = env_source.index("    def _coordex_proprio(", reset_start)
    reset_source = env_source[reset_start:reset_end]
    outside_reset = env_source[:reset_start] + env_source[reset_end:]
    assert "write_root_pose_to_sim" in reset_source
    assert "write_joint_state_to_sim" in reset_source
    assert "write_root_pose_to_sim" not in outside_reset
    assert "write_joint_state_to_sim" not in outside_reset


def test_frozen_config_hash_and_cli_override_are_deterministic() -> None:
    path = Path("configs/near_grasp/screw1_cem_v1.yaml")
    first = NearGraspRunConfig.load(path)
    second = NearGraspRunConfig.load(path)
    assert first.sha256 == second.sha256
    assert first.values["classification"] == "NEAR_GRASP_SEARCH_INCOMPLETE"
    overridden = first.with_overrides({"scene": {"num_envs": 16}})
    assert overridden.values["scene"]["num_envs"] == 16
    assert overridden.values["cem"]["population"] == 256
    assert overridden.sha256 != first.sha256


def test_candidate_selection_uses_frozen_failure_evidence() -> None:
    rows = load_candidate_rows(
        "artifacts/physical_delivery/screw1_near_grasp_v0_1/cem_candidate_results.jsonl"
    )
    best = select_candidate(rows, mode="best-valid")
    hard = select_candidate(rows, mode="target-contact-abort")
    exact = select_candidate(rows, mode="exact", candidate_id=707)
    assert best["candidate_id"] == 373
    assert hard["candidate_id"] == 33
    assert exact["candidate_id"] == 707
    assert evaluation_class(best["evaluation"]) == "NO_GRASP_CONTACT"
    assert evaluation_class(hard["evaluation"]) == "TARGET_CONTACT_HARD_ABORT"
    assert not ReplayRequest("rows.jsonl", mode="best-valid").record_video


def test_contact_attribution_does_not_invent_a_limit_for_zero_force() -> None:
    result = attribute_contact_step(
        step=1,
        all_force_xyz=(0.0, 0.0, 0.0),
        screw1_force_xyz=(0.0, 0.0, 0.0),
        table_force_xyz=(0.0, 0.0, 0.0),
        ground_force_xyz=(0.0, 0.0, 0.0),
        filter_valid={"Screw1": True, "Table": False, "ground": False},
    )
    assert result.classification == "NO_CONTACT"
    assert not result.instrumentation_limit


def test_contact_attribution_marks_only_unexplained_nonzero_force() -> None:
    explained = attribute_contact_step(
        step=2,
        all_force_xyz=(0.0, 0.0, 0.2),
        screw1_force_xyz=(0.0, 0.0, 0.2),
        table_force_xyz=(0.0, 0.0, 0.0),
        ground_force_xyz=(0.0, 0.0, 0.0),
        filter_valid={"Screw1": True, "Table": False, "ground": False},
    )
    unresolved = attribute_contact_step(
        step=3,
        all_force_xyz=(0.0, 0.0, 0.2),
        screw1_force_xyz=(0.0, 0.0, 0.0),
        table_force_xyz=(0.0, 0.0, 0.0),
        ground_force_xyz=(0.0, 0.0, 0.0),
        filter_valid={"Screw1": True, "Table": False, "ground": False},
    )
    assert explained.classification == "IDENTIFIED_SCENE_CONTACT"
    assert not explained.instrumentation_limit
    assert unresolved.classification == "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT"
    assert unresolved.instrumentation_limit


def test_contact_attribution_is_vector_based_and_honest() -> None:
    result = attribute_contact_step(
        step=4,
        all_force_xyz=(0.0, 0.0, 0.2),
        screw1_force_xyz=(0.0, 0.0, 0.15),
        table_force_xyz=(0.0, 0.0, 0.05),
        ground_force_xyz=(0.0, 0.0, 0.0),
        filter_valid={"Screw1": True, "Table": True, "ground": True},
    )
    assert result.classification == "IDENTIFIED_SCENE_CONTACT"
    assert result.identified_contacts == ("Screw1",)
    assert result.residual_force_n < 1.0e-9
    unresolved = attribute_contact_step(
        step=5,
        all_force_xyz=(0.2, 0.0, 0.0),
        screw1_force_xyz=(0.0, 0.0, 0.0),
        table_force_xyz=(0.0, 0.0, 0.0),
        ground_force_xyz=(0.0, 0.0, 0.0),
        filter_valid={"Screw1": True, "Table": False, "ground": False},
    )
    assert unresolved.classification == "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT"
    assert unresolved.instrumentation_limit


def test_termination_reasons_cover_replay_failures() -> None:
    assert ProgramTermination.HARD_FORCE_ABORT.name == "HARD_FORCE_ABORT"
    assert ProgramTermination.LATENT_CLOSURE_EXHAUSTED.name == "LATENT_CLOSURE_EXHAUSTED"
    assert ProgramTermination.FLYOUT.name == "FLYOUT"


def test_active_runner_uses_lightweight_near_grasp_imports() -> None:
    source = Path("scripts/environments/run_near_grasp_cem.py").read_text(encoding="utf-8")
    forbidden = (
        "pipeline.unified_grasp.v8",
        "scripted_contact_baseline",
        "screw1_grasp_baseline_v2",
        "contact_adaptive",
        "unified_grasp_env",
    )
    assert "from near_grasp." in source
    assert all(token not in source for token in forbidden)


def test_active_runtime_import_closure_has_no_legacy_grasp_modules() -> None:
    paths = [
        Path("scripts/environments/run_near_grasp_cem.py"),
        Path("scripts/environments/replay_near_grasp_candidate.py"),
        *sorted((MODULE_DIR / "near_grasp").glob("*.py")),
    ]
    imported_modules = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
    forbidden = (
        "v80",
        "v81",
        "v88",
        "v89",
        "v90",
        "v91",
        "v92",
        "v93",
        "v94",
        "v95",
        "scripted_contact_baseline",
        "screw1_grasp_baseline_v2",
        "contact_adaptive",
        "unified_grasp_env",
        "train_unified_rl",
        "eval_unified_rl",
    )
    assert all(not any(token in module for token in forbidden) for module in imported_modules)
