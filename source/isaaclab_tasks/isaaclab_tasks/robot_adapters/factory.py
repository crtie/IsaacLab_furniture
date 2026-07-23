"""Explicit robot adapter selection without eager legacy imports."""

from __future__ import annotations

from typing import Any

from .protocol import RobotAdapter, RobotAdapterError, INVALID_SCHEMA
from .sharpa_wave import SharpaWaveAdapter


class RobotAdapterSelectionError(RobotAdapterError):
    """Raised when a CLI/config robot name has no registered adapter."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message, code=INVALID_SCHEMA, details=details)


def create_robot_adapter(
    robot_name: str,
    *,
    variant: str = "floating",
    env: Any | None = None,
    env_id: int = 0,
    debug_privileged_allowed: bool = False,
    **kwargs: Any,
) -> RobotAdapter:
    """Create an adapter by explicit robot identity.

    Only the packaged Sharpawave adapter is supported by the formal interface.
    """

    key = str(robot_name).strip().lower().replace("-", "_")
    if key in {"sharpawave", "sharpa_wave", "sharpa"}:
        return SharpaWaveAdapter(
            variant=str(variant),
            env=env,
            env_id=int(env_id),
            debug_privileged_allowed=bool(debug_privileged_allowed),
            **kwargs,
        )
    raise RobotAdapterSelectionError(
        f"unknown robot adapter {robot_name!r}; supported adapter: sharpawave",
        details={"robot_name": str(robot_name), "variant": str(variant)},
    )


__all__ = ["RobotAdapterSelectionError", "create_robot_adapter"]
