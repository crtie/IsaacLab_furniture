"""Backend contracts shared by formal policy and system-validation execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from .models import AssemblyReport, AssemblyTarget


class AssemblyRuntime(Protocol):
    """Minimal state-continuous runtime required by the formal policy backend."""

    robot_name: str
    variant: str
    action_schema_id: str
    action_dim: int

    def reset(self) -> None: ...
    def initialize_target(self, target: AssemblyTarget) -> None: ...
    def observe(self, target: AssemblyTarget, phase: str) -> dict: ...
    def apply_action(self, action, phase: str) -> None: ...
    def verify_grasp(self, target: AssemblyTarget) -> bool: ...
    def transport_held_part(self, target: AssemblyTarget) -> None: ...
    def verify_insert(self, target: AssemblyTarget) -> bool: ...
    def release(self, target: AssemblyTarget) -> None: ...
    def state_token(self, target: AssemblyTarget) -> str: ...
    def close(self) -> None: ...


class AssemblyBackend(ABC):
    name: str

    @abstractmethod
    def run(self, targets: tuple[AssemblyTarget, ...]) -> AssemblyReport:
        raise NotImplementedError

