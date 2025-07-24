# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .chair1_env import FrankaChair1Env
from .chair2_env import FrankaChair2Env
from .np_env_cfg import FrankaChair1Cfg, FrankaChair2Cfg

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
