"""Interface-only zero policy. It is never evidence of BC or physical success."""

from __future__ import annotations

import numpy as np

from isaaclab_tasks.robot_adapters.policy import PolicyAction, PolicyContext, SkillObservation


class ZeroMockPolicy:
    is_mock_policy = True

    def __init__(self, checkpoint: str, metadata: dict | None = None):
        self.checkpoint = checkpoint
        self.metadata = dict(metadata or {})
        self._context: PolicyContext | None = None
        self._batch_size = 0

    def reset(self, batch_size: int, context: PolicyContext) -> None:
        self._batch_size = int(batch_size)
        self._context = context

    def act(self, observation: SkillObservation) -> PolicyAction:
        if self._context is None:
            raise RuntimeError("mock policy must be reset before act")
        width = int(self._context.metadata["action_dim"])
        return PolicyAction(
            schema_id=self._context.action_schema_id,
            values=np.zeros((self._batch_size, width), dtype=np.float64),
        )

    def close(self) -> None:
        self._context = None

