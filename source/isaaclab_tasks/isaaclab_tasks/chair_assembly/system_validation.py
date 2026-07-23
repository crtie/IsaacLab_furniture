"""Non-physical orchestration backend, isolated from the formal policy backend."""

from __future__ import annotations

from typing import Callable

from .backend import AssemblyBackend
from .models import AssemblyReport, AssemblyTarget, ResultCode, TargetResult


ValidationExecutor = Callable[[AssemblyTarget], list[str]]


class SystemValidationBackend(AssemblyBackend):
    name = "system-validation"

    def __init__(self, executor: ValidationExecutor | None = None):
        self.executor = executor or self._orchestration_stub

    @staticmethod
    def _orchestration_stub(target: AssemblyTarget) -> list[str]:
        # Deliberately validates catalog/state progression without pretending to
        # manipulate physics. Runtime visual/oracle reproduction remains a
        # separately explicit historical command.
        return ["non_physical_state_substitution"]

    def run(self, targets: tuple[AssemblyTarget, ...]) -> AssemblyReport:
        rows: list[TargetResult] = []
        try:
            for target in targets:
                operations = list(self.executor(target))
                rows.append(
                    TargetResult(
                        target.stage_id,
                        target.target_id,
                        target.target_name,
                        ResultCode.SYSTEM_VALIDATION_COMPLETE.value,
                        assistance_operations=operations,
                        sticky_used="sticky" in operations,
                        snap_used="snap" in operations,
                        teacher_motion_used="teacher_motion" in operations,
                        root_pose_writes_used="root_pose_write" in operations,
                        oracle_visual_only=True,
                        not_physical=True,
                        bc_training_eligible=False,
                    )
                )
            code = ResultCode.SYSTEM_VALIDATION_COMPLETE
        except Exception as exc:
            code = ResultCode.SYSTEM_VALIDATION_FAILED
            if targets:
                target = targets[min(len(rows), len(targets) - 1)]
                rows.append(TargetResult(target.stage_id, target.target_id, target.target_name, code.value, reason=str(exc)))
        return AssemblyReport(
            backend=self.name,
            robot="sharpawave",
            variant="floating",
            status="complete" if len(rows) == len(targets) and code is ResultCode.SYSTEM_VALIDATION_COMPLETE else "failed",
            result_code=code.value,
            targets_planned=len(targets),
            targets_completed=sum(row.status == ResultCode.SYSTEM_VALIDATION_COMPLETE.value for row in rows),
            target_results=rows,
            sticky_used=any(row.sticky_used for row in rows),
            snap_used=any(row.snap_used for row in rows),
            teacher_motion_used=any(row.teacher_motion_used for row in rows),
            root_pose_writes_used=any(row.root_pose_writes_used for row in rows),
            oracle_visual_only=True,
            not_physical=True,
            bc_training_eligible=False,
            metadata={"data_type": "system_validation_only", "physical_success_claim_allowed": False},
        )

