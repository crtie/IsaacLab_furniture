from pathlib import Path
import importlib.util
import sys


MODULE = (
    Path(__file__).parents[1]
    / "isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/dual_contact_regulator.py"
)
SPEC = importlib.util.spec_from_file_location("dual_contact_regulator", MODULE)
reg = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = reg
SPEC.loader.exec_module(reg)


def obs(f3, f4, **kwargs):
    return reg.DualContactObservation(f3, f4, **kwargs)


def test_hysteresis_does_not_lower_formal_contact_threshold():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    first = controller.update(obs(0.06, 0.0), allow_wrist=False)
    assert first.latched_finger3_contact
    assert first.formal_finger3_contact
    for value in (0.049, 0.040, 0.036):
        command = controller.update(obs(value, 0.0), allow_wrist=False)
        assert command.latched_finger3_contact
        assert not command.formal_finger3_contact


def test_latch_requires_three_below_keep_samples_to_drop():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    controller.update(obs(0.06, 0.0), allow_wrist=False)
    assert controller.update(obs(0.02, 0.0), allow_wrist=False).latched_finger3_contact
    assert controller.update(obs(0.02, 0.0), allow_wrist=False).latched_finger3_contact
    assert not controller.update(obs(0.02, 0.0), allow_wrist=False).latched_finger3_contact


def test_handoff_is_bounded_and_requires_sustained_other_contact():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    controller.update(obs(0.06, 0.0), allow_wrist=False)
    commands = [controller.update(obs(0.0, 0.07), allow_wrist=False) for _ in range(12)]
    assert sum(command.handoff for command in commands) == 1
    assert controller.handoff_count == 1


def test_failed_anchor_reacquisition_terminates_instead_of_zero_action_spin():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    controller.update(obs(0.06, 0.0), allow_wrist=False)
    command = None
    for _ in range(8):
        command = controller.update(obs(0.0, 0.0), allow_wrist=False)
        if command.state == reg.ABORTED:
            break
    assert command is not None
    assert command.state == reg.ABORTED
    assert command.termination_reason == "anchor_reacquisition_failed"


def test_force_guards_precede_regulation():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    assert controller.update(obs(5.0, 0.0), allow_wrist=True).action_kind == "ABORT"
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    assert controller.update(obs(1.1, 0.0), allow_wrist=True).action_kind == "RELEASE_OVERFORCE"


def test_wrist_probe_order_and_scoring_are_deterministic():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    candidates = controller.wrist_probe_candidates(0)
    assert [(item.axis, item.sign) for item in candidates[:4]] == [("y", 1), ("y", -1), ("z", 1), ("z", -1)]
    assert candidates[0].translation_xyz_m == (0.0, 0.0001, 0.0)
    before = obs(0.10, 0.01)
    weak = obs(0.10, 0.03)
    dual = obs(0.10, 0.06)
    assert controller.score_wrist_probe(before, dual) > controller.score_wrist_probe(before, weak)


def test_stagnation_requests_transactional_wrist_probe():
    controller = reg.DualContactRegulator(anchor_finger="finger3")
    command = None
    for _ in range(controller.config.stagnation_steps + 1):
        command = controller.update(obs(0.15, 0.0), allow_wrist=True)
    assert command is not None
    assert any(
        controller.update(obs(0.15, 0.0), allow_wrist=True).request_wrist_probe
        for _ in range(controller.config.stagnation_steps + 1)
    )
