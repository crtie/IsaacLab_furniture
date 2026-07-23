"""Small same-process BC policy contract for robot-schema-aware callers."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .protocol import POLICY_UNAVAILABLE, RobotAdapterError


class PolicyManifestError(RobotAdapterError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message, code=POLICY_UNAVAILABLE, details=details)


@dataclass(frozen=True)
class PolicyContext:
    robot_name: str
    variant: str
    action_schema_id: str
    observation_schema: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillObservation:
    schema_id: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyAction:
    schema_id: str
    values: Any


@runtime_checkable
class SkillPolicy(Protocol):
    def reset(self, batch_size: int, context: PolicyContext) -> None: ...

    def act(self, observation: SkillObservation) -> PolicyAction: ...

    def close(self) -> None: ...


def load_policy_manifest(path: str | Path, *, expected_action_schema: str | None = None) -> dict[str, Any]:
    """Load and validate a JSON/YAML policy manifest without importing policy code."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise PolicyManifestError(
            f"policy manifest does not exist: {manifest_path}",
            details={"path": str(manifest_path)},
        )
    try:
        if manifest_path.suffix.lower() == ".json":
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise PolicyManifestError("YAML manifest requires PyYAML", details={"path": str(manifest_path)}) from exc
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except PolicyManifestError:
        raise
    except Exception as exc:
        raise PolicyManifestError(f"unable to parse policy manifest: {exc}", details={"path": str(manifest_path)}) from exc
    if not isinstance(payload, dict):
        raise PolicyManifestError("policy manifest root must be a mapping", details={"path": str(manifest_path)})
    if int(payload.get("schema_version", 0) or 0) < 1:
        raise PolicyManifestError("policy manifest schema_version is missing or invalid")
    policies = payload.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise PolicyManifestError("policy manifest must contain a non-empty policies mapping")
    for name, spec in policies.items():
        if not isinstance(spec, dict):
            raise PolicyManifestError(f"policy {name!r} must be a mapping")
        for key in ("skill", "entrypoint", "checkpoint", "action_schema"):
            if not str(spec.get(key, "")).strip():
                raise PolicyManifestError(f"policy {name!r} is missing {key}")
        checkpoint = Path(str(spec["checkpoint"])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (manifest_path.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            raise PolicyManifestError(
                f"policy checkpoint is unavailable for {name!r}: {checkpoint}",
                details={"policy": str(name), "checkpoint": str(checkpoint)},
            )
        if expected_action_schema and str(spec["action_schema"]) != str(expected_action_schema):
            raise PolicyManifestError(
                f"policy {name!r} action schema does not match the selected robot",
                details={"expected": expected_action_schema, "actual": spec["action_schema"]},
            )
        for key in ("observation_schema", "normalization", "control_frequency_hz"):
            if key not in spec:
                raise PolicyManifestError(f"policy {name!r} is missing {key}")
        if not str(spec["observation_schema"]).strip():
            raise PolicyManifestError(f"policy {name!r} observation_schema is empty")
        if not isinstance(spec["normalization"], dict):
            raise PolicyManifestError(f"policy {name!r} normalization must be a mapping")
        try:
            frequency = float(spec["control_frequency_hz"])
        except (TypeError, ValueError) as exc:
            raise PolicyManifestError(f"policy {name!r} control_frequency_hz must be numeric") from exc
        if not np.isfinite(frequency) or frequency <= 0:
            raise PolicyManifestError(f"policy {name!r} control_frequency_hz must be finite and positive")
        spec["control_frequency_hz"] = frequency
        spec["checkpoint"] = str(checkpoint)
    for required_skill in ("pick", "insert"):
        if required_skill not in policies:
            raise PolicyManifestError(f"policy manifest is missing required {required_skill!r} policy")
    return payload


def instantiate_policy(spec: Mapping[str, Any]) -> SkillPolicy:
    """Instantiate one same-process policy from ``module:attribute`` metadata."""

    entrypoint = str(spec.get("entrypoint", ""))
    module_name, separator, attribute_name = entrypoint.partition(":")
    if not separator or not module_name or not attribute_name:
        raise PolicyManifestError("policy entrypoint must use module:attribute syntax", details={"entrypoint": entrypoint})
    try:
        factory = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise PolicyManifestError(
            f"unable to import policy entrypoint {entrypoint!r}: {exc}", details={"entrypoint": entrypoint}
        ) from exc
    kwargs = {"checkpoint": str(spec["checkpoint"]), "metadata": dict(spec.get("metadata", {}))}
    try:
        policy = factory(**kwargs) if inspect.isclass(factory) or callable(factory) else None
    except Exception as exc:
        raise PolicyManifestError(f"unable to construct policy {entrypoint!r}: {exc}") from exc
    if not isinstance(policy, SkillPolicy):
        raise PolicyManifestError(
            f"policy {entrypoint!r} does not implement reset/act/close", details={"entrypoint": entrypoint}
        )
    return policy


def validate_policy_action(
    action: PolicyAction,
    *,
    expected_schema: str,
    batch_size: int | None = None,
    action_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> np.ndarray:
    """Validate a policy action against schema identity and tensor shape.

    ``action_dim`` is optional for callers that only have a manifest schema ID;
    when supplied, it prevents a policy from silently emitting a different
    morphology width under a matching string ID.  ``expected_action_dim`` is a
    descriptive alias used by report/CLI callers.
    """

    if str(action.schema_id) != str(expected_schema):
        raise PolicyManifestError(
            "policy action schema mismatch",
            details={"expected": expected_schema, "actual": action.schema_id},
        )
    if action_dim is not None and expected_action_dim is not None and int(action_dim) != int(expected_action_dim):
        raise PolicyManifestError(
            "conflicting expected action dimensions",
            details={"action_dim": int(action_dim), "expected_action_dim": int(expected_action_dim)},
        )
    width = expected_action_dim if expected_action_dim is not None else action_dim
    values = np.asarray(action.values, dtype=np.float64)
    if values.ndim != 2 or (batch_size is not None and values.shape[0] != int(batch_size)):
        raise PolicyManifestError("policy action must be batch-first", details={"shape": tuple(values.shape)})
    if width is not None and values.shape[1] != int(width):
        raise PolicyManifestError(
            "policy action width does not match the selected schema",
            details={"expected_width": int(width), "actual_width": int(values.shape[1]), "schema_id": str(expected_schema)},
        )
    if not np.isfinite(values).all():
        raise PolicyManifestError("policy action contains NaN or infinite values")
    return values


__all__ = [
    "PolicyContext",
    "PolicyAction",
    "PolicyManifestError",
    "SkillObservation",
    "SkillPolicy",
    "load_policy_manifest",
    "instantiate_policy",
    "validate_policy_action",
]
