"""Pure configuration contract for reproducible near-grasp runs."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_SECTIONS = (
    "scene",
    "object",
    "program",
    "templates",
    "executor",
    "force",
    "cem",
    "prior",
    "assets",
    "artifacts",
    "replay",
)


@dataclass(frozen=True)
class NearGraspRunConfig:
    """Validated, serializable configuration with a stable content hash."""

    source_path: str
    values: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "NearGraspRunConfig":
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("near-grasp config root must be a mapping")
        _validate(payload)
        return cls(str(source), copy.deepcopy(payload))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "NearGraspRunConfig":
        values = copy.deepcopy(dict(self.values))
        _deep_merge(values, overrides)
        _validate(values)
        return NearGraspRunConfig(self.source_path, values)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.values))

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return output


def _validate(payload: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_SECTIONS if name not in payload]
    if missing:
        raise ValueError(f"near-grasp config missing sections: {missing}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported near-grasp config schema")
    scene = payload["scene"]
    cem = payload["cem"]
    force = payload["force"]
    if int(scene["num_envs"]) <= 0 or float(scene["dt"]) <= 0.0:
        raise ValueError("scene num_envs and dt must be positive")
    if int(cem["population"]) <= 0 or int(cem["physical_batch"]) <= 0 or int(cem["elite_count"]) <= 0:
        raise ValueError("CEM sizes must be positive")
    if int(cem["elite_count"]) > int(cem["population"]):
        raise ValueError("CEM elite_count cannot exceed population")
    if int(cem["min_generations"]) <= 0 or int(cem["max_generations"]) < int(cem["min_generations"]):
        raise ValueError("CEM generation bounds are invalid")
    if not (0.0 < float(force["formal_contact_n"]) < float(force["soft_limit_n"]) < float(force["hard_abort_n"])):
        raise ValueError("force thresholds must be strictly ordered")
    templates = payload["templates"]
    if not isinstance(templates, list) or [int(row["id"]) for row in templates] != list(range(6)):
        raise ValueError("the frozen Screw1 config requires template IDs 0..5")


def _deep_merge(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
