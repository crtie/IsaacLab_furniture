from pathlib import Path
import inspect
import sys

import numpy as np


MODULE_DIR = (
    Path(__file__).parents[1]
    / "isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
)
sys.path.insert(0, str(MODULE_DIR))
import contact_adaptive_closure as closure  # noqa: E402
import hand_morphology as morphology  # noqa: E402


def make_morphology():
    names = tuple(f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
    groups = {f"finger{finger}": tuple(range((finger - 1) * 4, finger * 4)) for finger in range(1, 6)}
    payload = {
        "hand_name": "test",
        "joint_names": names,
        "joint_limits": [(-1.0, 1.0)] * 20,
        "tip_names": [f"tip{finger}" for finger in range(1, 6)],
        "preshape_q": [0.0] * 20,
        "action_columns": list(range(6, 26)),
        "action_scale": 1.0,
        "asset": "test",
    }
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
        calibration_identity=morphology.morphology_signature(payload),
    )


def make_calibration(spec):
    matrix = np.zeros((30, 20))
    for finger in range(5):
        for joint in range(4):
            matrix[finger * 6 + 1, finger * 4 + joint] = 0.01 * (-1.0 if finger % 2 else 1.0)
    return morphology.MorphologyCalibration(
        schema_version=1,
        hand_signature=spec.calibration_identity,
        probe_delta_rad=0.002,
        joint_effect_probes=(),
        effect_matrix_30x20=tuple(tuple(float(x) for x in row) for row in matrix),
        crosstalk_limit=0.25,
        calibration_valid=True,
    )


def make_object(group=("finger3", "finger4")):
    return morphology.ObjectGraspSpec(
        part_name="arbitrary_part",
        grasp_family="small_object",
        active_finger_groups=(group,),
        approach_frame="object_local",
        acquisition_plan_path="plan.json",
        preshape_profile="preshape",
        closure_prior_name="prior",
        closure_axes_by_finger={finger: (0.0, 1.0, 0.0) for finger in group},
        force_band_n=(0.05, 1.0),
        hard_abort_force_n=5.0,
        preclose_object_motion_limit_m=0.005,
        palm_support_allowed=False,
        lift_direction_xyz=(0.0, 0.0, 1.0),
        lift_target_m=0.010,
        finger_side_by_name={finger: (-1.0 if index == 0 else 1.0) for index, finger in enumerate(group)},
        finger_approach_gain={finger: 1.0 for finger in group},
        allowed_contact_bodies=("arbitrary_part",),
    )


def obs(**forces):
    return closure.ClosureObservation(hand_q=(0.0,) * 20, target_forces_n=forces)


def test_arbitrary_finger_group_masks_contacted_finger():
    spec = make_morphology()
    controller = closure.ContactAdaptiveClosure(spec, make_calibration(spec), make_object(("finger2", "finger5")))
    command = controller.update(obs(finger2=0.06, finger5=0.0))
    assert command.termination_reason == "precontact_executor_required"
    assert not any(command.hand_delta_q)


def test_force_guards_precede_nominal_motion():
    spec = make_morphology()
    controller = closure.ContactAdaptiveClosure(spec, make_calibration(spec), make_object())
    assert controller.update(obs(finger3=5.0, finger4=0.0)).termination_reason == "hard_force_abort"


def test_dual_contact_duty_advances_to_close_without_part_branch():
    spec = make_morphology()
    controller = closure.ContactAdaptiveClosure(spec, make_calibration(spec), make_object())
    command = controller.update(obs(finger3=0.10, finger4=0.10))
    assert command.state == closure.CONTROLLED_CLOSE
    assert command.action_kind == "PRECONTACT_HANDOFF_ACCEPTED"


def test_contact_hysteresis_suppresses_chatter():
    spec = make_morphology()
    controller = closure.ContactAdaptiveClosure(spec, make_calibration(spec), make_object())
    controller.update(obs(finger3=0.06, finger4=0.0))
    for value in (0.049, 0.040, 0.036):
        command = controller.update(obs(finger3=value, finger4=0.0))
        assert "finger3" in command.active_contacts
    for _ in range(3):
        command = controller.update(obs(finger3=0.0, finger4=0.0))
    assert "finger3" not in command.active_contacts


def test_unresolved_residual_is_not_identified_non_target_contact():
    spec = make_morphology()
    controller = closure.ContactAdaptiveClosure(
        spec, make_calibration(spec), make_object(), initial_state=closure.CONTROLLED_CLOSE
    )
    observation = closure.ClosureObservation(
        hand_q=(0.0,) * 20,
        target_forces_n={"finger3": 0.1, "finger4": 0.1},
        unresolved_unfiltered_residual_n=3.0,
    )
    assert controller.update(observation).termination_reason == ""
    identified = closure.ClosureObservation(
        hand_q=(0.0,) * 20,
        target_forces_n={"finger3": 0.1, "finger4": 0.1},
        identified_non_target_contact_force_n=0.1,
    )
    assert controller.update(identified).termination_reason == "non_target_contact"


def test_controlled_close_stops_on_remaining_target_error():
    spec = make_morphology()
    controller = closure.ContactAdaptiveClosure(
        spec, make_calibration(spec), make_object(), initial_state=closure.CONTROLLED_CLOSE
    )
    command = controller.update(
        closure.ClosureObservation(
            hand_q=spec.close_reference_q,
            target_forces_n={"finger3": 0.06, "finger4": 0.06},
        )
    )
    assert command.state == closure.POST_CLOSE_HOLD
    assert not any(command.hand_delta_q)


def test_controller_has_no_object_name_branch():
    source = inspect.getsource(closure.ContactAdaptiveClosure)
    assert "Screw1" not in source
    assert "Plug2" not in source


def test_effect_matrix_averages_signed_joint_responses():
    position = ((0.001, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 4
    rotation = ((0.0, 0.0, 0.0),) * 5
    probes = []
    for sign in (1, -1):
        probes.append(
            morphology.JointEffectProbe(
                joint_name="right_finger1_joint1",
                local_joint_index=0,
                action_column=6,
                action_value=sign * 0.002,
                sign=sign,
                command_delta_rad=sign * 0.002,
                app_target_delta_rad=sign * 0.002,
                runtime_target_delta_rad=sign * 0.002,
                actual_delta_rad=sign * 0.002,
                settle_lag_steps=2,
                other_joint_actual_peak_rad=0.0,
                crosstalk_ratio=0.0,
                fingertip_position_delta_m=tuple(
                    tuple(sign * value for value in row) for row in position
                ),
                fingertip_orientation_delta_rotvec=rotation,
                target_chain_ok=True,
                response_ok=True,
            )
        )
    matrix = morphology.effect_matrix_from_probes(probes)
    assert matrix.shape == (30, 20)
    assert matrix[0, 0] == 0.5


def test_calibration_signature_mismatch_is_rejected():
    spec = make_morphology()
    calibration = make_calibration(spec)
    bad = morphology.MorphologyCalibration(
        schema_version=calibration.schema_version,
        hand_signature="different",
        probe_delta_rad=calibration.probe_delta_rad,
        joint_effect_probes=(),
        effect_matrix_30x20=calibration.effect_matrix_30x20,
        crosstalk_limit=0.25,
        calibration_valid=True,
    )
    try:
        bad.validate_for(spec)
    except ValueError as exc:
        assert "signature mismatch" in str(exc)
    else:
        raise AssertionError("signature mismatch was accepted")
