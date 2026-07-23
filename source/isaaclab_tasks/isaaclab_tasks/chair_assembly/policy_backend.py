"""Formal same-process Pick/Insert policy backend with no assisted fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab_tasks.robot_adapters.policy import (
    PolicyContext,
    SkillObservation,
    instantiate_policy,
    validate_policy_action,
)

from .backend import AssemblyBackend, AssemblyRuntime
from .models import AssemblyReport, AssemblyTarget, ResultCode, TargetResult


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    policy: Any
    spec: dict[str, Any]


class PolicyAssemblyBackend(AssemblyBackend):
    """Execute the formal continuous-state Pick/transport/Insert state machine."""

    name = "policy"

    def __init__(self, runtime: AssemblyRuntime, manifest: dict[str, Any]):
        self.runtime = runtime
        policies = manifest["policies"]
        self.pick = LoadedSkill("pick", instantiate_policy(policies["pick"]), policies["pick"])
        self.insert = LoadedSkill("insert", instantiate_policy(policies["insert"]), policies["insert"])

    def _policy_action(self, skill: LoadedSkill, target: AssemblyTarget, phase: str):
        observation_schema = str(skill.spec["observation_schema"])
        context = PolicyContext(
            robot_name=self.runtime.robot_name,
            variant=self.runtime.variant,
            action_schema_id=self.runtime.action_schema_id,
            observation_schema=observation_schema,
            metadata={
                "action_dim": self.runtime.action_dim,
                "stage_id": target.stage_id,
                "target_id": target.target_id,
                "target_name": target.target_name,
                "part_name": target.part_name,
                "phase": phase,
                "control_frequency_hz": float(skill.spec["control_frequency_hz"]),
                "normalization": skill.spec["normalization"],
            },
        )
        skill.policy.reset(1, context)
        action = skill.policy.act(
            SkillObservation(schema_id=observation_schema, values=self.runtime.observe(target, phase))
        )
        return validate_policy_action(
            action,
            expected_schema=self.runtime.action_schema_id,
            batch_size=1,
            expected_action_dim=self.runtime.action_dim,
        )

    def run(self, targets: tuple[AssemblyTarget, ...]) -> AssemblyReport:
        results: list[TargetResult] = []
        code = ResultCode.OK
        try:
            self.runtime.reset()
            for target in targets:
                self.runtime.initialize_target(target)
                token_before_pick = self.runtime.state_token(target)
                self.runtime.apply_action(self._policy_action(self.pick, target, "pick"), "pick")
                if not self.runtime.verify_grasp(target):
                    code = ResultCode.GRASP_NOT_VERIFIED
                    results.append(TargetResult(target.stage_id, target.target_id, target.target_name, code.value))
                    break
                token_after_pick = self.runtime.state_token(target)
                self.runtime.transport_held_part(target)
                if token_after_pick != self.runtime.state_token(target) or token_before_pick == "":
                    raise RuntimeError("physical state continuity violation between Pick and Insert")
                self.runtime.apply_action(self._policy_action(self.insert, target, "insert"), "insert")
                if not self.runtime.verify_insert(target):
                    code = ResultCode.INSERT_NOT_VERIFIED
                    results.append(TargetResult(target.stage_id, target.target_id, target.target_name, code.value))
                    break
                self.runtime.release(target)
                mock_used = getattr(self.pick.policy, "is_mock_policy", False) or getattr(
                    self.insert.policy, "is_mock_policy", False
                )
                results.append(
                    TargetResult(
                        target.stage_id,
                        target.target_id,
                        target.target_name,
                        ResultCode.OK.value,
                        policy_ids={"pick": self.pick.name, "insert": self.insert.name},
                        physical_grasp_success=not mock_used,
                        physical_lift_success=not mock_used,
                        physical_insert_success=not mock_used,
                        not_physical=mock_used,
                        bc_training_eligible=not mock_used,
                    )
                )
        finally:
            self.pick.policy.close()
            self.insert.policy.close()
            self.runtime.close()
        completed = sum(row.status == ResultCode.OK.value for row in results)
        mock_used = getattr(self.pick.policy, "is_mock_policy", False) or getattr(self.insert.policy, "is_mock_policy", False)
        return AssemblyReport(
            backend=self.name,
            robot=self.runtime.robot_name,
            variant=self.runtime.variant,
            status="complete" if completed == len(targets) and not mock_used else "incomplete",
            result_code=(ResultCode.OK if completed == len(targets) and not mock_used else code).value,
            targets_planned=len(targets),
            targets_completed=completed,
            target_results=results,
            not_physical=mock_used,
            bc_training_eligible=bool(results) and all(row.bc_training_eligible for row in results),
            physical_grasp_success=bool(results) and all(row.physical_grasp_success for row in results),
            physical_lift_success=bool(results) and all(row.physical_lift_success for row in results),
            physical_insert_success=bool(results) and all(row.physical_insert_success for row in results),
            metadata={"mock_policy_used": mock_used},
        )
