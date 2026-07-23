"""Robot-independent contracts used by task adapters.

The contracts in this module deliberately use plain Python and NumPy types.  Isaac
Lab/Sim objects are accepted at the boundary, but importing this module does not
start Kit and does not require a simulator process.  This keeps schema and action
tests useful on a workstation without Isaac Sim installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np


ActionMode = Literal["joint_delta", "joint_target"]


def _numpy_like(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    """Convert NumPy/Torch-like values without requiring a tensor package."""

    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
    return np.asarray(value, dtype=dtype)


class CapabilityCode:
    """Stable machine-readable capability/error codes.

    String constants are used instead of an enum at call sites so reports can be
    serialized directly to JSON and remain compatible with older report readers.
    """

    INVALID_ACTION = "INVALID_ACTION"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    MISSING_ASSET = "MISSING_ASSET"
    MISSING_RUNTIME_ARTICULATION = "MISSING_RUNTIME_ARTICULATION"
    MISSING_CALIBRATION = "MISSING_CALIBRATION"
    CONTACT_UNAVAILABLE = "CONTACT_UNAVAILABLE"
    POLICY_UNAVAILABLE = "POLICY_UNAVAILABLE"
    MISSING_LIMITS = "MISSING_LIMITS"
    RUNTIME_SCHEMA_MISMATCH = "RUNTIME_SCHEMA_MISMATCH"


CapabilityErrorCode = CapabilityCode


# Public aliases are convenient in CLI/report code and make the contract easy to
# discover without requiring callers to import CapabilityCode.
INVALID_ACTION = CapabilityCode.INVALID_ACTION
INVALID_SCHEMA = CapabilityCode.INVALID_SCHEMA
MISSING_ASSET = CapabilityCode.MISSING_ASSET
MISSING_RUNTIME_ARTICULATION = CapabilityCode.MISSING_RUNTIME_ARTICULATION
MISSING_CALIBRATION = CapabilityCode.MISSING_CALIBRATION
CONTACT_UNAVAILABLE = CapabilityCode.CONTACT_UNAVAILABLE
POLICY_UNAVAILABLE = CapabilityCode.POLICY_UNAVAILABLE
MISSING_LIMITS = CapabilityCode.MISSING_LIMITS
RUNTIME_SCHEMA_MISMATCH = CapabilityCode.RUNTIME_SCHEMA_MISMATCH


class RobotAdapterError(RuntimeError):
    """Base exception carrying a serializable capability code."""

    def __init__(self, message: str, *, code: str = INVALID_SCHEMA, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.reason = str(message)
        self.code = str(code)
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "message": str(self), "details": dict(self.details)}


class SchemaValidationError(RobotAdapterError):
    """Raised when a schema or action cannot be interpreted safely."""


class MissingCapabilityError(RobotAdapterError):
    """Raised when an optional runtime capability is not available.

    In particular, this is used for missing calibration, contact sensors, and BC
    checkpoints.  Callers can catch one type while preserving a precise ``code``.
    """

    def __init__(
        self,
        capability: str,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        **extra_details: Any,
    ):
        self.capability = str(capability)
        merged_details = dict(details or {})
        merged_details.update(extra_details)
        super().__init__(message or f"Required capability is unavailable: {self.capability}", code=self.capability, details=merged_details)


@dataclass(frozen=True)
class JointSpec:
    """A single named controllable degree of freedom."""

    name: str
    role: str
    unit: str = "rad"
    lower: float | None = None
    upper: float | None = None
    action_scale: float = 1.0
    continuous: bool = False
    effort: float | None = None
    velocity: float | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("JointSpec.name must be non-empty")
        if self.unit not in {"m", "rad", "dimensionless"}:
            raise ValueError(f"Unsupported joint unit: {self.unit!r}")
        for label, value in (
            ("lower", self.lower),
            ("upper", self.upper),
            ("action_scale", self.action_scale),
            ("effort", self.effort),
            ("velocity", self.velocity),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"JointSpec.{label} must be finite")
            if label in {"effort", "velocity"} and value is not None and float(value) < 0.0:
                raise ValueError(f"JointSpec.{label} must be non-negative")
        if self.lower is not None and self.upper is not None and float(self.lower) > float(self.upper):
            raise ValueError(f"JointSpec lower bound exceeds upper bound for {self.name!r}")
        if float(self.action_scale) <= 0:
            raise ValueError(f"JointSpec.action_scale must be positive for {self.name!r}")

    @property
    def kind(self) -> str:
        if self.role == "wrist" and self.unit == "m":
            return "prismatic"
        if self.continuous:
            return "continuous"
        return "revolute"

    @property
    def effort_limit(self) -> float | None:
        return self.effort

    @property
    def velocity_limit(self) -> float | None:
        return self.velocity

    def validate_value(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{self.name}: value must be finite")
        if self.lower is not None and numeric < float(self.lower):
            raise ValueError(f"{self.name}: value {numeric} is below {self.lower}")
        if self.upper is not None and numeric > float(self.upper):
            raise ValueError(f"{self.name}: value {numeric} is above {self.upper}")


@dataclass(frozen=True)
class FingertipSpec:
    """Semantic mapping for one finger's pose and contact frames."""

    role: str
    pose_frame: str
    contact_body: str
    contact_sensor: str | None = None

    def __post_init__(self) -> None:
        for label in ("role", "pose_frame", "contact_body"):
            if not str(getattr(self, label)).strip():
                raise ValueError(f"FingertipSpec.{label} must be non-empty")

    @property
    def contact_sensor_required(self) -> bool:
        return True

    @property
    def contact_status(self) -> str:
        return "asset_declared_runtime_unverified"


@dataclass(frozen=True)
class ActionSchema:
    """Versioned canonical action layout.

    Canonical actions are batch-first arrays of normalized values.  ``joint_delta``
    profiles interpret each value as a signed delta scaled by ``action_scale``;
    ``joint_target`` profiles map values into the configured joint limits.
    """

    schema_id: str
    version: int
    joints: tuple[JointSpec, ...]
    wrist_joint_names: tuple[str, ...] = ()
    finger_joint_names: tuple[str, ...] = ()
    mode: ActionMode = "joint_delta"
    normalized_low: float = -1.0
    normalized_high: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.schema_id).strip():
            raise ValueError("ActionSchema.schema_id must be non-empty")
        if int(self.version) < 1:
            raise ValueError("ActionSchema.version must be positive")
        if self.mode not in ("joint_delta", "joint_target"):
            raise ValueError(f"Unsupported action mode: {self.mode!r}")
        joints = tuple(self.joints)
        object.__setattr__(self, "joints", joints)
        wrist = tuple(str(item) for item in self.wrist_joint_names)
        fingers = tuple(str(item) for item in self.finger_joint_names)
        object.__setattr__(self, "wrist_joint_names", wrist)
        object.__setattr__(self, "finger_joint_names", fingers)
        names = tuple(joint.name for joint in joints)
        if len(set(names)) != len(names):
            raise ValueError("ActionSchema joint names must be unique")
        if set(wrist) | set(fingers) != set(names) or set(wrist) & set(fingers):
            raise ValueError("wrist_joint_names and finger_joint_names must partition joints")
        if not math.isfinite(float(self.normalized_low)) or not math.isfinite(float(self.normalized_high)):
            raise ValueError("ActionSchema normalized bounds must be finite")
        if float(self.normalized_low) >= float(self.normalized_high):
            raise ValueError("ActionSchema normalized_low must be below normalized_high")

    @property
    def canonical_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Alias retained for schema consumers that use the shorter spelling."""

        return self.canonical_joint_names

    @property
    def action_dim(self) -> int:
        return len(self.joints)

    @property
    def dimension(self) -> int:
        return self.action_dim

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("joint_delta", "joint_target")

    @property
    def default_profile(self) -> str:
        return self.mode

    @property
    def schema_version(self) -> str:
        return f"v{int(self.version)}"

    @property
    def scale_status(self) -> str:
        return "missing_calibration"

    @property
    def wrist_action_dim(self) -> int:
        return len(self.wrist_joint_names)

    @property
    def finger_action_dim(self) -> int:
        return len(self.finger_joint_names)

    @property
    def action_scale(self) -> np.ndarray:
        return np.asarray([float(joint.action_scale) for joint in self.joints], dtype=np.float64)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(joint.unit for joint in self.joints)

    @property
    def lower(self) -> np.ndarray:
        values = [(-math.inf if joint.lower is None else float(joint.lower)) for joint in self.joints]
        return np.asarray(values, dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        values = [(math.inf if joint.upper is None else float(joint.upper)) for joint in self.joints]
        return np.asarray(values, dtype=np.float64)

    def validate_batch(self, values: Any, *, name: str = "action") -> np.ndarray:
        """Return a float array and enforce the batch-first contract."""

        array = _numpy_like(values)
        if array.ndim != 2:
            raise SchemaValidationError(
                f"{name} must be a batch-first rank-2 array, got shape {array.shape}",
                code=INVALID_ACTION,
                details={"expected_rank": 2, "shape": tuple(array.shape), "schema_id": self.schema_id},
            )
        if array.shape[1] != self.action_dim:
            raise SchemaValidationError(
                f"{name} width {array.shape[1]} does not match schema width {self.action_dim}",
                code=INVALID_ACTION,
                details={"expected_width": self.action_dim, "actual_width": int(array.shape[1]), "schema_id": self.schema_id},
            )
        if not np.all(np.isfinite(array)):
            raise SchemaValidationError(
                f"{name} contains NaN or infinite values",
                code=INVALID_ACTION,
                details={"schema_id": self.schema_id},
            )
        if np.any(array < float(self.normalized_low)) or np.any(array > float(self.normalized_high)):
            raise SchemaValidationError(
                f"{name} contains values outside [{self.normalized_low}, {self.normalized_high}]",
                code=INVALID_ACTION,
                details={"schema_id": self.schema_id, "low": self.normalized_low, "high": self.normalized_high},
            )
        return array

    def validate_action_shape(self, values: Any) -> tuple[int, ...]:
        """Validate only shape, accepting NumPy/Torch-like objects."""

        shape = getattr(values, "shape", None)
        if shape is None:
            raise TypeError("action must expose a shape")
        normalized = tuple(int(item) for item in shape)
        if len(normalized) != 2 or normalized[-1] != self.action_dim:
            raise ValueError(f"expected batch-first shape (N, {self.action_dim}), got {normalized}")
        return normalized

    def validate_profile(self, profile: str) -> None:
        if str(profile) not in self.supported_profiles:
            raise ValueError(f"unsupported action profile {profile!r}; supported={self.supported_profiles}")


@dataclass(frozen=True)
class HandModelSpec:
    """Robot morphology and asset metadata independent of a task environment."""

    robot_name: str
    variant: str
    schema: ActionSchema
    fingertips: tuple[FingertipSpec, ...]
    urdf_path: str
    usd_path: str | None = None
    fixed_joint_names: tuple[str, ...] = ()
    schema_version: str = "v1"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    root_link_name: str = "base_root"
    vendor_fix_base: bool = True
    root_semantics_status: str = "runtime_unverified"
    contact_status: str = "asset_declared_runtime_unverified"
    actuator_control_mode: str = "position_target"
    actuator_limits_source: str = "urdf"
    actuator_gains_status: str = "missing_calibration"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingertips", tuple(self.fingertips))
        object.__setattr__(self, "fixed_joint_names", tuple(str(item) for item in self.fixed_joint_names))
        if len({finger.role for finger in self.fingertips}) != len(self.fingertips):
            raise ValueError("Fingertip roles must be unique")
        schema_namespace = self.schema.schema_id.split("/")[0]
        if schema_namespace != self.robot_name and not self.schema.schema_id.startswith(f"{self.robot_name}."):
            raise ValueError("schema_id must be namespaced by robot_name")

    @property
    def action_dim(self) -> int:
        return self.schema.action_dim

    @property
    def action_schema(self) -> ActionSchema:
        """Compatibility name used by environment configuration code."""

        return self.schema

    @property
    def schema_id(self) -> str:
        return self.schema.schema_id

    @property
    def name(self) -> str:
        return self.variant

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.schema.canonical_joint_names

    @property
    def canonical_joint_names(self) -> tuple[str, ...]:
        return self.schema.canonical_joint_names

    @property
    def dimension(self) -> int:
        return self.action_dim

    @property
    def fingertip_frames(self) -> tuple[str, ...]:
        return tuple(item.pose_frame for item in self.fingertips)

    @property
    def contact_body_names(self) -> tuple[str, ...]:
        return tuple(item.contact_body for item in self.fingertips)

    @property
    def base_joint_names(self) -> tuple[str, ...]:
        return self.schema.wrist_joint_names

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return self.schema.finger_joint_names

    @property
    def joint_specs(self) -> tuple[JointSpec, ...]:
        return self.schema.joints

    @property
    def contact_layer_paths(self) -> tuple[str, ...]:
        return (self.usd_path,) if self.usd_path else ()

    def resolve_joint_indices(self, runtime_joint_names: Sequence[str]) -> tuple[int, ...]:
        names = tuple(str(item) for item in runtime_joint_names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate runtime joint names: {duplicates}")
        missing = [name for name in self.joint_names if name not in names]
        if missing:
            raise ValueError(f"runtime schema is missing joints: {missing}")
        return tuple(names.index(name) for name in self.joint_names)

    def validate_runtime_joint_names(self, runtime_joint_names: Sequence[str]) -> "ValidationReport":
        names = tuple(str(item) for item in runtime_joint_names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        missing = [name for name in self.joint_names if name not in names]
        if duplicates or missing:
            errors = [f"duplicate runtime joints: {', '.join(duplicates)}"] if duplicates else []
            if missing:
                errors.append(f"missing runtime joints: {', '.join(missing)}")
            return ValidationReport.failure("RUNTIME_SCHEMA_MISMATCH", errors, missing=missing)
        extras = [name for name in names if name not in self.joint_names]
        return ValidationReport.success(warnings=(f"extra runtime joints: {', '.join(extras)}",) if extras else ())

    def validate_action_shape(self, values: Any) -> tuple[int, ...]:
        return self.schema.validate_action_shape(values)

    def default_configuration(self) -> dict[str, Any]:
        return {
            "robot": self.robot_name,
            "variant": self.variant,
            "schema_id": self.schema.schema_id,
            "schema_version": self.schema.schema_version,
            "urdf_path": self.urdf_path,
            "usd_path": self.usd_path,
            "action_dim": self.action_dim,
            "wrist_action_dim": self.wrist_action_dim,
            "finger_action_dim": self.finger_action_dim,
            "joint_names": self.joint_names,
            "fixed_joint_names": self.fixed_joint_names,
            "action_profile": self.schema.default_profile,
            "action_profiles": self.schema.supported_profiles,
            "action_scale_status": self.schema.scale_status,
            "root_link_name": self.root_link_name,
            "vendor_fix_base": self.vendor_fix_base,
            "root_semantics_status": self.root_semantics_status,
            "contact_status": self.contact_status,
            "actuator_control_mode": self.actuator_control_mode,
            "actuator_limits_source": self.actuator_limits_source,
            "actuator_gains_status": self.actuator_gains_status,
            "fingertips": tuple(
                {
                    "role": item.role,
                    "pose_frame": item.pose_frame,
                    "contact_body": item.contact_body,
                }
                for item in self.fingertips
            ),
        }

    def validate_files(self) -> ValidationReport:
        """Lightweight path check for callers using a declarative spec directly."""

        paths = [Path(self.urdf_path)]
        if self.usd_path:
            paths.append(Path(self.usd_path))
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            return ValidationReport.failure("MISSING_ASSET", [f"missing asset: {item}" for item in missing], missing=missing)
        return ValidationReport.success(details={"paths": [str(path) for path in paths]})

    @property
    def wrist_action_dim(self) -> int:
        return self.schema.wrist_action_dim

    @property
    def finger_action_dim(self) -> int:
        return self.schema.finger_action_dim


@dataclass(frozen=True)
class ContactMeasurements(Mapping[str, Any]):
    """Contact result that can represent unavailable sensors without fake zeros."""

    available: bool
    values: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    reason: str = ""
    body_names: tuple[str, ...] = ()

    def __getitem__(self, key: str) -> Any:
        if key == "available":
            return self.available
        if key == "values":
            return self.values
        if key == "source":
            return self.source
        if key == "reason":
            return self.reason
        if key == "body_names":
            return self.body_names
        return self.values[key]

    def __iter__(self):
        return iter(("available", "values", "source", "reason", "body_names"))

    def __len__(self) -> int:
        return 5

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "values": dict(self.values),
            "source": self.source,
            "reason": self.reason,
            "body_names": list(self.body_names),
        }


@dataclass(frozen=True)
class RobotObservation:
    """Batch-first observation returned by a robot adapter."""

    schema_id: str
    joint_names: tuple[str, ...]
    joint_position: Any
    joint_velocity: Any | None = None
    fingertip_poses: Mapping[str, Any] = field(default_factory=dict)
    contact_measurements: ContactMeasurements | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.schema_id).strip():
            raise ValueError("RobotObservation.schema_id must be non-empty")
        names = tuple(str(name) for name in self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError("RobotObservation joint_names must be unique")
        object.__setattr__(self, "joint_names", names)
        position = _numpy_like(self.joint_position)
        if position.ndim != 2:
            raise ValueError(f"joint_position must be batch-first rank-2, got {position.shape}")
        if position.shape[1] != len(names):
            raise ValueError("joint_position width does not match joint_names")
        if not np.all(np.isfinite(position)):
            raise ValueError("joint_position contains NaN or infinite values")
        if self.joint_velocity is not None:
            velocity = _numpy_like(self.joint_velocity)
            if velocity.ndim != 2 or velocity.shape != position.shape:
                raise ValueError("joint_velocity must have the same batch-first shape as joint_position")
            if not np.all(np.isfinite(velocity)):
                raise ValueError("joint_velocity contains NaN or infinite values")

    @property
    def batch_size(self) -> int:
        return int(_numpy_like(self.joint_position).shape[0])

    @property
    def schema_version(self) -> str:
        """Expose the version encoded by the schema identifier."""

        marker = self.schema_id.rsplit("-v", 1)
        return f"v{marker[-1]}" if len(marker) == 2 and marker[-1].isdigit() else "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "joint_names": list(self.joint_names),
            "joint_position": self.joint_position,
            "joint_velocity": self.joint_velocity,
            "fingertip_poses": dict(self.fingertip_poses),
            "contact_measurements": None
            if self.contact_measurements is None
            else self.contact_measurements.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Serializable result of asset or runtime schema validation."""

    ok: bool
    code: str = "ok"
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """Alias used by filesystem/runtime report consumers."""

        return bool(self.ok)

    @property
    def message(self) -> str:
        return "; ".join(self.errors)

    @property
    def missing_paths(self) -> tuple[str, ...]:
        return self.missing

    @property
    def present_paths(self) -> tuple[str, ...]:
        raw = self.details.get("paths", ())
        return tuple(str(item) for item in raw)

    @classmethod
    def success(cls, *, details: Mapping[str, Any] | None = None, warnings: Sequence[str] = ()) -> "ValidationReport":
        return cls(True, "ok", (), tuple(str(item) for item in warnings), (), dict(details or {}))

    @classmethod
    def failure(
        cls,
        code: str,
        errors: Sequence[str],
        *,
        missing: Sequence[str] = (),
        warnings: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> "ValidationReport":
        return cls(False, str(code), tuple(str(item) for item in errors), tuple(str(item) for item in warnings), tuple(str(item) for item in missing), dict(details or {}))

    def require_ok(self) -> "ValidationReport":
        if not self.ok:
            raise SchemaValidationError(
                "; ".join(self.errors) or f"validation failed: {self.code}", code=self.code, details=self.as_dict()
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "code": self.code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "missing": list(self.missing),
            "details": dict(self.details),
        }


class RobotAdapter(ABC):
    """Minimal adapter protocol for morphology-neutral task code."""

    robot_name: ClassVar[str]

    @property
    @abstractmethod
    def model(self) -> HandModelSpec:
        raise NotImplementedError

    @property
    def schema(self) -> ActionSchema:
        return self.model.schema

    @property
    def action_dim(self) -> int:
        return self.schema.action_dim

    @property
    def wrist_action_dim(self) -> int:
        return self.schema.wrist_action_dim

    @property
    def finger_action_dim(self) -> int:
        return self.schema.finger_action_dim

    @abstractmethod
    def validate_asset_schema(self) -> ValidationReport:
        raise NotImplementedError

    @abstractmethod
    def validate_runtime_articulation(self, runtime: Any | None = None, *, require_contact: bool = False) -> ValidationReport:
        raise NotImplementedError

    @abstractmethod
    def get_robot_observation(self, runtime: Any | None = None) -> RobotObservation:
        raise NotImplementedError

    @abstractmethod
    def map_canonical_action(
        self,
        action: Any,
        *,
        runtime_joint_names: Sequence[str] | None = None,
        current_position: Any | None = None,
        profile: ActionMode | None = None,
        action_schema_id: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_fingertip_frames(self, runtime: Any | None = None) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_contact_measurements(self, runtime: Any | None = None, *, strict: bool = False) -> ContactMeasurements:
        raise NotImplementedError

    @abstractmethod
    def default_configuration(self) -> Mapping[str, Any]:
        raise NotImplementedError


__all__ = [
    "ActionMode",
    "ActionSchema",
    "CapabilityCode",
    "CapabilityErrorCode",
    "ContactMeasurements",
    "FingertipSpec",
    "HandModelSpec",
    "JointSpec",
    "MissingCapabilityError",
    "RobotAdapter",
    "RobotAdapterError",
    "RobotObservation",
    "SchemaValidationError",
    "ValidationReport",
    "CONTACT_UNAVAILABLE",
    "INVALID_ACTION",
    "INVALID_SCHEMA",
    "MISSING_ASSET",
    "MISSING_CALIBRATION",
    "MISSING_LIMITS",
    "MISSING_RUNTIME_ARTICULATION",
    "POLICY_UNAVAILABLE",
    "RUNTIME_SCHEMA_MISMATCH",
]
