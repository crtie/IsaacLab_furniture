"""Opt-in Gym registration for the independent SharpaWave adapter."""

from __future__ import annotations

import gymnasium as gym


_SCHEMA_SPECS = (
    (
        "Isaac-SharpaWave-Floating-Schema-v0",
        "isaaclab_tasks.direct.np.sharpawave_env:SharpaWaveSchemaEnv",
        "floating",
    ),
    (
        "Isaac-SharpaWave-PegFixedRot-Schema-v0",
        "isaaclab_tasks.direct.np.sharpawave_env:SharpaWaveSchemaEnv",
        "peg_fixedrot",
    ),
)

_CHAIR_SPECS = (
    (
        "Isaac-SharpaWaveFloating-ChairTeacher-Direct-v0",
        "isaaclab_tasks.direct.np.sharpawave_env:SharpaWaveChairTeacherEnv",
        "floating",
    ),
    (
        "Isaac-SharpaWavePegFixedRot-ChairTeacher-Direct-v0",
        "isaaclab_tasks.direct.np.sharpawave_env:SharpaWaveChairTeacherEnv",
        "peg_fixedrot",
    ),
)

SHARPAWAVE_ENV_SPECS = _SCHEMA_SPECS + _CHAIR_SPECS


def register_sharpawave_environments(*, include_schema: bool = True, include_chair: bool = True) -> tuple[str, ...]:
    """Register SharpaWave IDs only when explicitly requested.

    No function in ``direct.np`` calls this during package import, so the
    default registration surface remains unchanged.
    """

    specs = ()
    if include_schema:
        specs += _SCHEMA_SPECS
    if include_chair:
        specs += _CHAIR_SPECS
    registered: list[str] = []
    for env_id, entry_point, variant in specs:
        existing = gym.registry.get(env_id)
        kwargs = {"variant": variant, "robot_name": "sharpawave"}
        if existing is not None:
            if existing.entry_point != entry_point or existing.kwargs.get("variant") != variant:
                raise RuntimeError(
                    f"Gym ID {env_id!r} is already registered with a conflicting SharpaWave entry point."
                )
            continue
        gym.register(id=env_id, entry_point=entry_point, disable_env_checker=True, kwargs=kwargs)
        registered.append(env_id)
    return tuple(registered)


__all__ = ["SHARPAWAVE_ENV_SPECS", "register_sharpawave_environments"]
