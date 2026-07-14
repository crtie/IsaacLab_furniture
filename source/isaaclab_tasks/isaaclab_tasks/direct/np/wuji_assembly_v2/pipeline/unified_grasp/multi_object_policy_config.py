"""Policy configuration helpers for v80 unified grasp training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .v80_reports import V80_PARTS, write_json


@dataclass
class UnifiedPolicyConfig:
    policy_mode: str = "shared_multi_object_policy"
    framework: str = "rsl_rl"
    num_envs_requested: int = 128
    env_count_fallbacks: tuple[int, ...] = (64, 32, 16)
    horizon: int = 48
    parts: tuple[str, ...] = tuple(V80_PARTS)
    object_type_embedding: bool = True
    physics_profile_embedding: bool = True
    grasp_kind_embedding: bool = True
    optional_per_object_action_head: bool = True
    sticky_disabled_during_training: bool = True
    domain_randomization_after_smoke: bool = True
    allow_diagnostic_dynamics: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_unified_rl_config(run_dir: str | Path, config: UnifiedPolicyConfig) -> Path:
    return write_json(Path(run_dir) / "unified_rl_config.json", config.to_dict())
