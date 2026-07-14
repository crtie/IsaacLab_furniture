"""Mixed continuous/categorical cross-entropy search for GraspProgram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .evaluator import PhysicalEvaluation
from .grasp_program import PROGRAM_DIM, GraspProgram, GraspProgramBounds
from .grasp_synthesis.grasp_energy import SearchMetrics


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: int
    program: GraspProgram
    evaluation: PhysicalEvaluation
    reset_seed: int
    physics_fingerprint: str = ""
    search_metrics: SearchMetrics | None = None

    @property
    def ranking_key(self) -> tuple[float, ...]:
        metrics = self.search_metrics or SearchMetrics.from_physical_evaluation(self.evaluation)
        return metrics.ranking_key()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "program": self.program.to_dict(),
            "reset_seed": self.reset_seed,
            "physics_fingerprint": self.physics_fingerprint,
            "evaluation": self.evaluation.to_dict(),
            "search_metrics": (
                self.search_metrics or SearchMetrics.from_physical_evaluation(self.evaluation)
            ).to_dict(),
        }


@dataclass(frozen=True)
class MixedCemConfig:
    population: int = 256
    physical_batch: int = 128
    elite_count: int = 32
    categorical_floor: float = 0.05
    min_generations: int = 5
    max_generations: int = 20
    seed: int = 20260713
    initial_std_fraction: float = 0.25
    min_std_fraction: float = 0.01
    smoothing: float = 0.2
    stagnant_generations: int = 5

    def __post_init__(self) -> None:
        if self.population <= 0 or self.physical_batch <= 0 or self.elite_count <= 0:
            raise ValueError("CEM sizes must be positive")
        if self.elite_count > self.population:
            raise ValueError("elite_count cannot exceed population")
        if self.min_generations < 5 or self.max_generations < self.min_generations:
            raise ValueError("CEM requires at least five generations")
        if not 0.0 <= self.categorical_floor < 1.0:
            raise ValueError("categorical_floor must be in [0, 1)")


class MixedCem:
    def __init__(
        self,
        template_count: int,
        config: MixedCemConfig | None = None,
        bounds: GraspProgramBounds | None = None,
    ):
        if int(template_count) <= 0:
            raise ValueError("template_count must be positive")
        self.template_count = int(template_count)
        self.config = config or MixedCemConfig()
        self.bounds = bounds or GraspProgramBounds()
        self.lower, self.upper = self.bounds.arrays()
        self.mean = 0.5 * (self.lower + self.upper)
        self.std = self.config.initial_std_fraction * (self.upper - self.lower)
        self.probabilities = np.full(self.template_count, 1.0 / self.template_count, dtype=np.float64)
        self.rng = np.random.default_rng(self.config.seed)
        self.generation = 0
        self.best_key: tuple[float, ...] | None = None
        self.stagnant_count = 0
        self.history: list[dict[str, Any]] = []

    def sample(self) -> list[GraspProgram]:
        raw = self.rng.normal(self.mean, self.std, size=(self.config.population, PROGRAM_DIM))
        raw = np.clip(raw, self.lower, self.upper)
        templates = self.rng.choice(self.template_count, size=self.config.population, p=self.probabilities)
        return [GraspProgram.from_vector(raw[index], int(templates[index]), bounds=self.bounds) for index in range(raw.shape[0])]

    @staticmethod
    def elites(results: Sequence[CandidateResult], elite_count: int) -> list[CandidateResult]:
        if not results:
            raise ValueError("Cannot select elites from an empty result set")
        return sorted(results, key=lambda item: item.ranking_key, reverse=True)[: int(elite_count)]

    def update(self, results: Sequence[CandidateResult]) -> dict[str, Any]:
        if len(results) != self.config.population:
            raise ValueError(f"Expected {self.config.population} CEM results, got {len(results)}")
        elite = self.elites(results, self.config.elite_count)
        values = np.asarray([item.program.values for item in elite], dtype=np.float64)
        new_mean = np.mean(values, axis=0)
        new_std = np.std(values, axis=0)
        min_std = self.config.min_std_fraction * (self.upper - self.lower)
        alpha = float(self.config.smoothing)
        self.mean = np.clip(alpha * self.mean + (1.0 - alpha) * new_mean, self.lower, self.upper)
        self.std = np.maximum(alpha * self.std + (1.0 - alpha) * new_std, min_std)

        counts = np.bincount([item.program.template_id for item in elite], minlength=self.template_count).astype(np.float64)
        probabilities = counts / max(float(np.sum(counts)), 1.0)
        floor = min(float(self.config.categorical_floor), 1.0 / self.template_count)
        probabilities = np.maximum(probabilities, floor)
        probabilities /= np.sum(probabilities)
        self.probabilities = alpha * self.probabilities + (1.0 - alpha) * probabilities
        self.probabilities /= np.sum(self.probabilities)

        current_best = elite[0].ranking_key
        if self.best_key is None or current_best > self.best_key:
            self.best_key = current_best
            self.stagnant_count = 0
        else:
            self.stagnant_count += 1
        self.generation += 1
        row = {
            "generation": self.generation,
            "best_candidate_id": elite[0].candidate_id,
            "best_ranking_key": list(current_best),
            "physical_lift_count": sum(int(item.evaluation.physical_lift_success) for item in results),
            "stable_close_count": sum(int(item.evaluation.stable_close) for item in results),
            "valid_candidate_count": sum(int(item.evaluation.valid_candidate) for item in results),
            "elite_valid_count": sum(int(item.evaluation.valid_candidate) for item in elite),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "categorical_probabilities": self.probabilities.tolist(),
            "stagnant_generations": self.stagnant_count,
            "covariance_collapsed": self.covariance_collapsed,
        }
        self.history.append(row)
        return row

    @property
    def covariance_collapsed(self) -> bool:
        span = self.upper - self.lower
        return bool(np.all(self.std <= 1.05 * self.config.min_std_fraction * span))

    @property
    def parameterization_insufficient(self) -> bool:
        return bool(
            self.generation >= self.config.min_generations
            and self.stagnant_count >= self.config.stagnant_generations
            and self.covariance_collapsed
        )

    @property
    def should_stop(self) -> bool:
        return bool(self.generation >= self.config.max_generations or self.parameterization_insufficient)
