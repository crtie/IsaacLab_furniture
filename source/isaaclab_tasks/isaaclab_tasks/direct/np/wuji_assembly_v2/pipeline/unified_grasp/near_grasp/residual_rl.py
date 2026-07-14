"""Residual PPO action contract, reward terms, and evidence gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


RESIDUAL_ACTION_DIM = 14
DIRECT_CONTACT_RESIDUAL_ACTION_DIM = 21


class CurriculumStage(IntEnum):
    CONTACT = 0
    MULTI_CONTACT = 1
    STABLE_CLOSE = 2
    LIFT = 3
    COMPLETE = 4


@dataclass(frozen=True)
class ResidualActionBounds:
    """Deprecated 14D latent contract retained for artifact decoding only."""
    latent_delta: float = 0.25
    wrist_translation_m: float = 0.001
    wrist_rotation_rad: float = np.deg2rad(1.0)
    closure_speed_adjustment: float = 0.25
    hold_adjustment: float = 8.0

    @property
    def lower(self) -> np.ndarray:
        return -self.upper

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(
            [self.latent_delta] * 6
            + [self.wrist_translation_m] * 3
            + [self.wrist_rotation_rad] * 3
            + [self.closure_speed_adjustment, self.hold_adjustment],
            dtype=np.float64,
        )

    def decode(self, normalized_action: Sequence[float]) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape[-1] != RESIDUAL_ACTION_DIM:
            raise ValueError(f"Residual action must have width {RESIDUAL_ACTION_DIM}")
        return np.clip(action, -1.0, 1.0) * self.upper


@dataclass(frozen=True)
class DirectContactResidualBounds:
    """Five masked fingertip Cartesian residuals plus bounded wrist SE(3)."""

    fingertip_translation_m: float = 0.00015
    wrist_translation_m: float = 0.00025
    wrist_rotation_rad: float = np.deg2rad(0.25)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(
            [self.fingertip_translation_m] * 15
            + [self.wrist_translation_m] * 3
            + [self.wrist_rotation_rad] * 3,
            dtype=np.float64,
        )

    def decode(self, normalized_action: Sequence[float], active_finger_mask: Sequence[bool]) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float64)
        active = np.asarray(active_finger_mask, dtype=bool)
        if action.shape != (DIRECT_CONTACT_RESIDUAL_ACTION_DIM,) or active.shape != (5,):
            raise ValueError("direct contact residual shapes are invalid")
        decoded = np.clip(action, -1.0, 1.0) * self.upper
        decoded[:15] *= np.repeat(active, 3)
        return decoded


@dataclass(frozen=True)
class EvaluationRates:
    episodes: int
    contact: float
    multi_contact: float
    stable_close: float
    lift: float


class ResidualRlGate:
    """Allow training only around verified contact-capable pregrasps."""

    thresholds = {
        CurriculumStage.CONTACT: ("contact", 0.70),
        CurriculumStage.MULTI_CONTACT: ("multi_contact", 0.50),
        CurriculumStage.STABLE_CLOSE: ("stable_close", 0.35),
        CurriculumStage.LIFT: ("lift", 0.20),
    }

    def __init__(
        self,
        *,
        optimized_pregrasp_available: bool,
        safe_contact_trials: int,
        unstable_hold_trials: int,
        eligibility_packet_present: bool,
    ):
        self.optimized_pregrasp_available = bool(optimized_pregrasp_available)
        self.safe_contact_trials = int(safe_contact_trials)
        self.unstable_hold_trials = int(unstable_hold_trials)
        self.eligibility_packet_present = bool(eligibility_packet_present)
        self.stage = CurriculumStage.CONTACT
        self.history: list[dict[str, Any]] = []

    @property
    def training_allowed(self) -> bool:
        return bool(
            self.optimized_pregrasp_available
            and self.safe_contact_trials >= 3
            and self.unstable_hold_trials >= 3
            and self.eligibility_packet_present
        )

    def consider(self, rates: EvaluationRates, *, checkpoint_created: bool = False) -> bool:
        del checkpoint_created
        if not self.training_allowed or rates.episodes < 512 or self.stage == CurriculumStage.COMPLETE:
            return False
        field, threshold = self.thresholds[self.stage]
        rate = float(getattr(rates, field))
        advanced = rate >= threshold
        self.history.append(
            {
                "stage": self.stage.name,
                "episodes": rates.episodes,
                "metric": field,
                "rate": rate,
                "threshold": threshold,
                "advanced": advanced,
                "checkpoint_irrelevant_to_gate": True,
            }
        )
        if advanced:
            self.stage = CurriculumStage(int(self.stage) + 1)
        return advanced


def residual_reward(terms: Mapping[str, np.ndarray | float]) -> np.ndarray:
    """Training reward only; strict evaluation never calls this function."""

    required = {
        "target_contact",
        "contact_duty",
        "stable_close",
        "lift_height_m",
        "force_excess_n",
        "collision",
        "object_displacement_m",
        "object_velocity",
        "contact_loss",
        "jerk",
    }
    missing = required - set(terms)
    if missing:
        raise ValueError(f"Missing residual reward terms: {sorted(missing)}")
    values = {name: np.asarray(terms[name], dtype=np.float64) for name in required}
    return (
        1.0 * values["target_contact"]
        + 2.0 * values["contact_duty"]
        + 5.0 * values["stable_close"]
        + 100.0 * np.clip(values["lift_height_m"], 0.0, 0.02)
        - 3.0 * values["force_excess_n"]
        - 4.0 * values["collision"]
        - 20.0 * values["object_displacement_m"]
        - 0.5 * values["object_velocity"]
        - 2.0 * values["contact_loss"]
        - 0.05 * values["jerk"]
    )


def make_rsl_rl_ppo_config(*, max_iterations: int = 1000) -> dict[str, Any]:
    """Future direct-contact PPO config; no runner starts it automatically."""

    return {
        "num_steps_per_env": 32,
        "max_iterations": int(max_iterations),
        "save_interval": 50,
        "experiment_name": "privileged_physics_direct_contact_residual_ppo",
        "empirical_normalization": True,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.35,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 8,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
