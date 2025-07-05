# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .np_env import FrankaChairEnv
from .np_env_cfg import FrankaChairCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Chair-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChairCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Factory-GearMesh-Direct-v0",
#     entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskGearMeshCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Factory-NutThread-Direct-v0",
#     entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskNutThreadCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )
