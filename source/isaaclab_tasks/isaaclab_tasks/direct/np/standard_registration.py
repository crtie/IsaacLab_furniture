"""Gym registrations owned by the existing Franka NP task family."""

from __future__ import annotations

import gymnasium as gym


_AGENT_CFG = "isaaclab_tasks.direct.np.agents:rl_games_ppo_cfg.yaml"

# These tasks are outside the Wuji v2 cleanup boundary. String entry points
# preserve their registration without importing every environment at package load.
STANDARD_ENV_SPECS = (
    ("Isaac-Franka-Chair1-Direct-v0", "chair1_env:FrankaChair1Env", "np_env_cfg:FrankaChair1Cfg"),
    ("Isaac-Franka-Chair2-Direct-v0", "franka_chair2_env:FrankaChair2Env", "np_env_cfg:FrankaChair2Cfg"),
    ("Isaac-Franka-ChairTeacher-Direct-v0", "franka_teacher_compat_env:FrankaChairTeacherEnv", "np_env_cfg:FrankaChairTeacherCfg"),
    ("Isaac-Franka-Chair3-Direct-v0", "chair3_env:FrankaChair3Env", "np_env_cfg:FrankaChair3Cfg"),
    ("Isaac-Franka-Chair4-Direct-v0", "chair4_env:FrankaChair4Env", "np_env_cfg:FrankaChair4Cfg"),
    ("Isaac-Franka-Chair5-Direct-v0", "chair5_env:FrankaChair5Env", "np_env_cfg:FrankaChair5Cfg"),
    ("Isaac-Franka-Chair6-Direct-v0", "chair6_env:FrankaChair6Env", "np_env_cfg:FrankaChair6Cfg"),
    ("Isaac-Franka-Vasskar1-Direct-v0", "vasskar1_env:FrankaVasskar1Env", "np_env_cfg:FrankaVasskar1Cfg"),
    ("Isaac-Franka-Vasskar2-Direct-v0", "vasskar2_env:FrankaVasskar2Env", "np_env_cfg:FrankaVasskar2Cfg"),
    ("Isaac-Franka-Plane1-Direct-v0", "plane1_env:FrankaPlane1Env", "np_env_cfg:FrankaPlane1Cfg"),
    ("Isaac-Franka-Plane2-Direct-v0", "plane2_env:FrankaPlane2Env", "np_env_cfg:FrankaPlane2Cfg"),
    ("Isaac-Franka-Plane3-Direct-v0", "plane3_env:FrankaPlane3Env", "np_env_cfg:FrankaPlane3Cfg"),
    ("Isaac-Franka-Plane4-Direct-v0", "plane4_env:FrankaPlane4Env", "np_env_cfg:FrankaPlane4Cfg"),
    ("Isaac-Franka-Lego1-Direct-v0", "lego1_env:FrankaLego1Env", "np_env_cfg:FrankaLego1Cfg"),
    ("Isaac-Franka-Lego2-Direct-v0", "lego2_env:FrankaLego2Env", "np_env_cfg:FrankaLego2Cfg"),
    ("Isaac-Franka-Lego3-Direct-v0", "lego3_env:FrankaLego3Env", "np_env_cfg:FrankaLego3Cfg"),
    ("Isaac-Franka-Lego4-Direct-v0", "lego4_env:FrankaLego4Env", "np_env_cfg:FrankaLego4Cfg"),
    ("Isaac-Franka-Lego5-Direct-v0", "lego5_env:FrankaLego5Env", "np_env_cfg:FrankaLego5Cfg"),
    ("Isaac-Franka-Lego6-Direct-v0", "lego6_env:FrankaLego6Env", "np_env_cfg:FrankaLego6Cfg"),
    ("Isaac-Franka-Lego7-Direct-v0", "lego7_env:FrankaLego7Env", "np_env_cfg:FrankaLego7Cfg"),
)


def register_standard_environments() -> tuple[str, ...]:
    """Register the unrelated NP tasks without importing implementations."""

    registered: list[str] = []
    module_root = "isaaclab_tasks.direct.np"
    for env_id, env_suffix, cfg_suffix in STANDARD_ENV_SPECS:
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point=f"{module_root}.{env_suffix}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{module_root}.{cfg_suffix}",
                "rl_games_cfg_entry_point": _AGENT_CFG,
            },
        )
        registered.append(env_id)
    return tuple(registered)


__all__ = ["STANDARD_ENV_SPECS", "register_standard_environments"]
