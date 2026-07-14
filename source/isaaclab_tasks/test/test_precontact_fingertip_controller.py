from pathlib import Path
import inspect
import sys

import numpy as np


MODULE_DIR = Path(__file__).parents[1] / "isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
sys.path.insert(0, str(MODULE_DIR))
import closure_prior  # noqa: E402
import hand_morphology as morphology  # noqa: E402
import precontact_fingertip_controller as precontact  # noqa: E402


def make_morphology():
    names = tuple(f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
    groups = {f"finger{finger}": tuple(range((finger - 1) * 4, finger * 4)) for finger in range(1, 6)}
    return morphology.HandMorphologySpec(
        hand_name="test",
        joint_names=names,
        joint_limits=((-1.0, 1.0),) * 20,
        finger_names=tuple(f"finger{x}" for x in range(1, 6)),
        finger_joint_groups=groups,
        fingertip_body_names=tuple(f"tip{x}" for x in range(1, 6)),
        preshape_q=(0.0,) * 20,
        close_reference_q=(0.5,) * 20,
        contact_capable_fingers=tuple(f"finger{x}" for x in range(1, 6)),
        palm_support_body="palm",
        action_columns=tuple(range(6, 26)),
        morphology_calibration_source="test",
        calibration_identity="test",
    )


def make_object(part="arbitrary"):
    return morphology.ObjectGraspSpec(
        part_name=part,
        grasp_family="small",
        active_finger_groups=(("finger3", "finger4"),),
        approach_frame="object_local",
        acquisition_plan_path="plan.json",
        preshape_profile="preshape",
        closure_prior_name="prior",
        closure_axes_by_finger={"finger3": (0.0, -1.0, 0.0), "finger4": (0.0, 1.0, 0.0)},
        force_band_n=(0.05, 1.0),
        hard_abort_force_n=5.0,
        preclose_object_motion_limit_m=0.005,
        palm_support_allowed=False,
        lift_direction_xyz=(0.0, 0.0, 1.0),
        lift_target_m=0.010,
        pinch_axis_object=(0.0, 1.0, 0.0),
        finger_side_by_name={"finger3": -1.0, "finger4": 1.0},
        finger_approach_gain={"finger3": 1.0, "finger4": 1.0},
        allowed_contact_bodies=(part,),
    )


def make_prior():
    latent = np.zeros((262, 4), dtype=float)
    latent[134:, 0] = np.linspace(0.0, 1.0, 128)
    latent[134:, 1] = np.sin(np.linspace(0.0, np.pi, 128))
    return closure_prior.ClosurePrior(
        schema_version=1,
        source_path="test",
        source_sha256="test",
        source_morphology_unknown=True,
        source_joint_count=8,
        frame_count=262,
        source_success=False,
        successful_grasp_demonstration=False,
        source_role="timing_only",
        recording_audit={},
        phases=(
            closure_prior.ClosurePhase("motion", 134, 247, 114, ()),
            closure_prior.ClosurePhase("secondary_adjustment", 248, 256, 9, ()),
            closure_prior.ClosurePhase("hold", 257, 259, 3, ()),
            closure_prior.ClosurePhase("secondary_adjustment", 260, 261, 2, ()),
        ),
        discontinuity_frames=(60, 133),
        explained_variance_ratio=(1.0,),
        cumulative_explained_variance=(1.0,),
        synergy_basis=((1.0,),),
        normalized_latent_trajectory=tuple(tuple(float(x) for x in row) for row in latent),
        reconstruction_rmse_by_rank={"1": 0.0},
        tracking_best_lag_frames=1,
        tracking_rmse=0.0,
        tracking_rmse_by_joint=(0.0,) * 8,
        hold_fraction=0.0,
    )


def make_jacobian(matrix=None, **kwargs):
    if matrix is None:
        matrix = np.zeros((6, 8), dtype=float)
        matrix[:, :6] = np.eye(6) * 0.02
        matrix[:, 6:] = 0.002
    return precontact.LocalActiveJacobian(
        active_fingers=("finger3", "finger4"),
        active_joint_indices=tuple(range(8, 16)),
        matrix_6x8=tuple(tuple(float(x) for x in row) for row in matrix),
        **kwargs,
    )


def observation(**forces):
    return precontact.PreContactObservation(
        hand_q=(0.0,) * 20,
        hand_target_q=(0.0,) * 20,
        fingertip_positions={"finger3": (-0.01, 0.0, 0.0), "finger4": (0.01, 0.0, 0.0)},
        object_position=(0.0, 0.0, 0.0),
        object_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        target_forces_n=forces,
    )


def test_quaternion_frame_transform_and_opposite_pinch_sides():
    controller = precontact.PreContactFingertipController(make_morphology(), make_object(), make_jacobian(), make_prior())
    obs = observation()
    obs = precontact.PreContactObservation(**{**obs.__dict__, "object_quaternion_wxyz": (2**-0.5, 0.0, 0.0, 2**-0.5)})
    targets, axis = controller.object_relative_targets(obs)
    assert np.allclose(axis, (-1.0, 0.0, 0.0), atol=1e-7)
    assert np.allclose(targets["finger3"], -targets["finger4"])


def test_prior_drives_runtime_hold_and_latent_scale():
    scheduler = precontact.PriorScheduler(make_prior())
    first = scheduler.peek()
    for _ in range(174):
        scheduler.accept()
    late = scheduler.peek()
    assert first.phase == "motion"
    assert 0.5 <= first.latent_speed_scale <= 1.5
    assert late.phase == "hold"
    assert late.hold_requested


def test_precontact_dls_uses_all_active_joint_columns_and_reduces_error():
    matrix = make_jacobian().matrix
    desired = np.array([0.0001, 0.0, 0.0, -0.0001, 0.0, 0.0])
    delta = precontact.damped_least_squares(matrix, desired, 1e-5)
    assert np.linalg.norm(desired - matrix @ delta) < np.linalg.norm(desired)
    controller = precontact.PreContactFingertipController(make_morphology(), make_object(), make_jacobian(), make_prior())
    controller.update(observation())
    command = controller.update(observation())
    assert command.action_kind in {"OBJECT_RELATIVE_TIP_ALIGNMENT", "OBJECT_RELATIVE_PINCH_APPROACH"}
    assert any(abs(command.hand_delta_q[index]) > 0.0 for index in range(8, 16))


def test_first_contact_freezes_that_fingers_joint_columns():
    controller = precontact.PreContactFingertipController(make_morphology(), make_object(), make_jacobian(), make_prior())
    controller.update(observation())
    hold = controller.update(observation(finger3=0.06, finger4=0.0))
    assert hold.state == precontact.FIRST_CONTACT_FORCE_HOLD
    assert not any(hold.hand_delta_q)
    command = controller.update(observation(finger3=0.06, finger4=0.0))
    assert command.state == precontact.SECOND_FINGER_APPROACH
    assert all(command.hand_delta_q[index] == 0.0 for index in range(8, 12))
    assert any(command.hand_delta_q[index] != 0.0 for index in range(12, 16))


def test_jacobian_gate_reports_rank_and_restore_blockers():
    rank_bad = make_jacobian(np.zeros((6, 8))).diagnostics(np.ones(6) * 1e-4)
    assert not rank_bad["controllable"]
    assert rank_bad["rank"] == 0
    restore_bad = make_jacobian(joint_restore_error_rad=3e-4).diagnostics(np.ones(6) * 1e-4)
    assert not restore_bad["controllable"]


def test_controller_has_no_part_name_branch():
    source = inspect.getsource(precontact.PreContactFingertipController)
    assert "Screw1" not in source
    assert "Plug2" not in source
