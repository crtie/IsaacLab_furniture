# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .chair1_env import FrankaChair1Env
from .chair2_env import FrankaChair2Env
from .chair3_env import FrankaChair3Env
from .chair4_env import FrankaChair4Env
from .chair5_env import FrankaChair5Env
from .chair6_env import FrankaChair6Env
from .np_env_cfg import FrankaChair1Cfg, FrankaChair2Cfg, FrankaChair3Cfg, FrankaChair4Cfg, FrankaChair5Cfg, FrankaChair6Cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Chair1-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChair1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChair1Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Chair2-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChair2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChair2Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Chair3-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChair3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChair3Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Chair4-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChair4Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChair4Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Chair5-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChair5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChair5Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Chair6-Direct-v0",
    entry_point="isaaclab_tasks.direct.np:FrankaChair6Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaChair6Cfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)