"""Independent adapter for the right-hand SharpaWave assets.

This module contains no simulator imports.  It describes the two supplied URDF
variants and provides a small, name-addressed bridge for an Isaac Lab articulation
when one is supplied at runtime.  The bridge intentionally does not infer
calibration, contact, or policy behavior from morphology: unavailable capabilities
are reported explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .protocol import (
    ActionMode,
    ActionSchema,
    CONTACT_UNAVAILABLE,
    ContactMeasurements,
    FingertipSpec,
    HandModelSpec,
    INVALID_ACTION,
    INVALID_SCHEMA,
    MISSING_ASSET,
    MISSING_CALIBRATION,
    MISSING_LIMITS,
    MISSING_RUNTIME_ARTICULATION,
    MissingCapabilityError,
    POLICY_UNAVAILABLE,
    RobotAdapter,
    RobotObservation,
    RUNTIME_SCHEMA_MISMATCH,
    SchemaValidationError,
    JointSpec,
    ValidationReport,
)


class MissingCalibrationError(MissingCapabilityError):
    """Compatibility error for an unavailable calibrated profile or mapper."""

    def __init__(self, code: str = MISSING_CALIBRATION, message: str | None = None, **details: Any):
        super().__init__(code, message, details=details)


ROBOT_NAME = "sharpawave"
SHARPAWAVE = ROBOT_NAME
SCHEMA_VERSION = 1
VARIANT_FLOATING = "floating"
VARIANT_PEG_FIXEDROT = "peg_fixedrot"
SUPPORTED_VARIANTS = (VARIANT_FLOATING, VARIANT_PEG_FIXEDROT)

ASSET_ROOT_RELATIVE = Path(
    "source/isaaclab_assets/isaaclab_assets/robots/sharpa-wave-description"
)
URDF_ROOT_RELATIVE = ASSET_ROOT_RELATIVE / "right_sharpa_wave_urdf"

# This is an explicit semantic order, not the order returned by USD/URDF
# traversal.  The order follows the supplied asset's named joints and is part of
# each action schema's public identity.
SHARPAWAVE_HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

SHARPAWAVE_FLOATING_BASE_JOINT_NAMES = (
    "right_x_joint",
    "right_y_joint",
    "right_z_joint",
    "right_roll_joint",
    "right_pitch_joint",
    "right_yaw_joint",
)
SHARPAWAVE_PEG_FIXEDROT_BASE_JOINT_NAMES = (
    "right_x_joint",
    "right_y_joint",
    "right_z_joint",
)

SHARPAWAVE_FINGERTIP_SPECS = (
    FingertipSpec("thumb", "right_thumb_fingertip", "right_thumb_elastomer"),
    FingertipSpec("index", "right_index_fingertip", "right_index_elastomer"),
    FingertipSpec("middle", "right_middle_fingertip", "right_middle_elastomer"),
    FingertipSpec("ring", "right_ring_fingertip", "right_ring_elastomer"),
    FingertipSpec("pinky", "right_pinky_fingertip", "right_pinky_elastomer"),
)

def _asset_file(relative_path: str | Path) -> Path:
    """Resolve a tracked asset before the repository-root helper is defined."""

    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    for parent in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    return relative


def _load_joint_specs(
    urdf_relative_path: str | Path,
    base_names: Sequence[str],
    finger_names: Sequence[str],
) -> tuple[JointSpec, ...]:
    """Read limits/units/actuator facts from the tracked URDF by name.

    The canonical order still comes from the explicit variant manifest.  The
    numerical joint facts come from the asset itself, avoiding a second limit
    table in the task package.  Missing assets retain the names with unknown
    limits so schema-only tooling can report a dependency failure honestly.
    """

    path = _asset_file(urdf_relative_path)
    joints: dict[str, ET.Element] = {}
    try:
        root = ET.parse(path).getroot()
        joints = {
            str(element.attrib.get("name")): element
            for element in root.findall("joint")
            if str(element.attrib.get("type", "")) != "fixed"
        }
    except (ET.ParseError, OSError):
        joints = {}
    base_set = set(str(name) for name in base_names)
    specs: list[JointSpec] = []
    for name in tuple(base_names) + tuple(finger_names):
        element = joints.get(str(name))
        joint_type = str(element.attrib.get("type", "")) if element is not None else ""
        is_translation = str(name) in {
            "right_x_joint",
            "right_y_joint",
            "right_z_joint",
        }
        is_continuous_wrist = str(name) in {
            "right_roll_joint",
            "right_pitch_joint",
            "right_yaw_joint",
        }
        limit = element.find("limit") if element is not None else None
        try:
            lower = None if limit is None else float(limit.attrib["lower"])
            upper = None if limit is None else float(limit.attrib["upper"])
        except (KeyError, TypeError, ValueError):
            lower = upper = None
        try:
            effort = None if limit is None else float(limit.attrib["effort"])
        except (KeyError, TypeError, ValueError):
            effort = None
        try:
            velocity = None if limit is None else float(limit.attrib["velocity"])
        except (KeyError, TypeError, ValueError):
            velocity = None
        specs.append(
            JointSpec(
                name=str(name),
                role="wrist" if str(name) in base_set else "finger",
                unit="m" if joint_type == "prismatic" or is_translation else "rad",
                lower=lower,
                upper=upper,
                action_scale=1.0,
                continuous=joint_type == "continuous" or is_continuous_wrist,
                effort=effort,
                velocity=velocity,
            )
        )
    return tuple(specs)


def _make_model(variant: str) -> HandModelSpec:
    if variant == VARIANT_FLOATING:
        base = SHARPAWAVE_FLOATING_BASE_JOINT_NAMES
        urdf = URDF_ROOT_RELATIVE / "right_sharpa_wave_floating.urdf"
        usd = URDF_ROOT_RELATIVE / "usd" / "right_sharpa_wave_floating_runtime_contact_deinstanced.usd"
        fixed = (
            "right_thumb_elastomer_fix_joint",
            "right_thumb_fingertip_fix_joint",
            "right_index_elastomer_fix_joint",
            "right_index_fingertip_fix_joint",
            "right_middle_elastomer_fix_joint",
            "right_middle_fingertip_fix_joint",
            "right_ring_elastomer_fix_joint",
            "right_ring_fingertip_fix_joint",
            "right_pinky_elastomer_fix_joint",
            "right_pinky_fingertip_fix_joint",
        )
    elif variant == VARIANT_PEG_FIXEDROT:
        base = SHARPAWAVE_PEG_FIXEDROT_BASE_JOINT_NAMES
        urdf = URDF_ROOT_RELATIVE / "right_sharpa_wave_floating_peg_fixedrot.urdf"
        usd = URDF_ROOT_RELATIVE / "usd" / "right_sharpa_wave_floating_3t_fixedrot.usd"
        fixed = ("right_wrist_fixed_orientation",) + tuple(
            item.replace("_elastomer_fix_joint", "_elastomer_fix_joint")
            for item in (
                "right_thumb_elastomer_fix_joint",
                "right_thumb_fingertip_fix_joint",
                "right_index_elastomer_fix_joint",
                "right_index_fingertip_fix_joint",
                "right_middle_elastomer_fix_joint",
                "right_middle_fingertip_fix_joint",
                "right_ring_elastomer_fix_joint",
                "right_ring_fingertip_fix_joint",
                "right_pinky_elastomer_fix_joint",
                "right_pinky_fingertip_fix_joint",
            )
        )
    else:
        raise ValueError(f"Unsupported SharpaWave variant: {variant!r}")
    joints = _load_joint_specs(urdf, base, SHARPAWAVE_HAND_JOINT_NAMES)
    schema = ActionSchema(
        schema_id=f"{ROBOT_NAME}/{variant}/action-v{SCHEMA_VERSION}",
        version=SCHEMA_VERSION,
        joints=joints,
        wrist_joint_names=tuple(base),
        finger_joint_names=SHARPAWAVE_HAND_JOINT_NAMES,
        mode="joint_delta",
    )
    return HandModelSpec(
        robot_name=ROBOT_NAME,
        variant=variant,
        schema=schema,
        fingertips=SHARPAWAVE_FINGERTIP_SPECS,
        urdf_path=urdf.as_posix(),
        usd_path=usd.as_posix(),
        fixed_joint_names=fixed,
        schema_version=f"v{SCHEMA_VERSION}",
        provenance={"source": "right_sharpa_wave_urdf", "calibrated": False},
    )


# Public immutable manifests.  A few descriptive aliases make discovery from
# notebooks and report tooling straightforward.
SHARPAWAVE_VARIANTS = {variant: _make_model(variant) for variant in SUPPORTED_VARIANTS}
VARIANT_MANIFESTS = SHARPAWAVE_VARIANTS
SHARPAWAVE_MANIFEST = SHARPAWAVE_VARIANTS


def _repository_root() -> Path | None:
    """Find the checkout containing the tracked asset tree without machine paths."""

    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ASSET_ROOT_RELATIVE).is_dir():
            return parent
    return None


def resolve_asset_path(relative_path: str | Path, *, repository_root: str | Path | None = None) -> Path:
    """Resolve a manifest path from a checkout or explicit asset root."""

    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    root = Path(repository_root).expanduser().resolve() if repository_root is not None else _repository_root()
    if root is None:
        # Keep the result deterministic for an installed package; validation will
        # report MISSING_ASSET rather than silently selecting another checkout.
        root = Path(__file__).resolve().parents[3]
    return root / candidate


def _public_manifest_path(
    value: Any,
    *,
    label: str,
    repository_root: str | Path | None = None,
) -> str:
    """Normalize asset-package paths without leaking a checkout prefix."""

    candidate = Path(str(value))
    if not candidate.is_absolute():
        return candidate.as_posix()
    roots: list[Path] = []
    if repository_root is not None:
        roots.append(Path(repository_root).expanduser().resolve())
    discovered_root = _repository_root()
    if discovered_root is not None and discovered_root not in roots:
        roots.append(discovered_root)
    for root in roots:
        # Preserve the lexical package path when the externally distributed
        # asset directory is installed as a symlink. Resolving the candidate
        # first would incorrectly make an authorized external target look like
        # an unrelated machine path.
        try:
            return candidate.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            pass
        try:
            return candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    raise ValueError(
        f"{label} must be repository-relative or belong to the active checkout; "
        f"refusing external absolute path {candidate}"
    )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_name_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return ()


def _first_attr(objects: Sequence[Any], names: Sequence[str]) -> Any:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            try:
                value = getattr(obj, name)
            except (AttributeError, RuntimeError, TypeError):
                continue
            if value is not None:
                return value
    return None


def _to_numpy(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _batch_matrix(value: Any, *, label: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise SchemaValidationError(f"{label} must be rank-2 after batching, got {array.shape}", code=INVALID_SCHEMA)
    if not np.all(np.isfinite(array)):
        raise SchemaValidationError(
            f"{label} contains NaN or infinite values",
            code=INVALID_ACTION if label == "action" else INVALID_SCHEMA,
        )
    return array


@dataclass(frozen=True)
class SharpaWaveConfiguration:
    """Explicit runtime configuration with unverified capabilities visible."""

    robot_name: str
    variant: str
    schema_id: str
    action_mode: ActionMode
    action_scale: tuple[float, ...] | None
    calibration_available: bool = False
    base_pose_mapper_available: bool = False
    contact_sensor_required: bool = True
    policy_available: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "variant": self.variant,
            "schema_id": self.schema_id,
            "action_mode": self.action_mode,
            "action_scale": None if self.action_scale is None else list(self.action_scale),
            "calibration_available": self.calibration_available,
            "base_pose_mapper_available": self.base_pose_mapper_available,
            "contact_sensor_required": self.contact_sensor_required,
            "policy_available": self.policy_available,
        }


class SharpaWaveAdapter(RobotAdapter):
    """Name-based adapter for either supplied SharpaWave morphology variant."""

    robot_name = ROBOT_NAME

    @classmethod
    def from_spec(cls, specification: Any, env: Any | None = None, **kwargs: Any) -> "SharpaWaveAdapter":
        """Construct an adapter from either this package's model or an asset cfg.

        The asset package intentionally has its own declarative config type.  A
        small duck-typed conversion here keeps the runtime bridge independent of
        that package's optional imports while preserving its canonical names.
        """

        if isinstance(specification, HandModelSpec):
            normalized_urdf = _public_manifest_path(
                specification.urdf_path,
                label="SharpaWave URDF path",
                repository_root=kwargs.get("repository_root"),
            )
            normalized_usd = (
                None
                if specification.usd_path is None
                else _public_manifest_path(
                    specification.usd_path,
                    label="SharpaWave USD path",
                    repository_root=kwargs.get("repository_root"),
                )
            )
            if normalized_urdf != specification.urdf_path or normalized_usd != specification.usd_path:
                specification = replace(
                    specification,
                    urdf_path=normalized_urdf,
                    usd_path=normalized_usd,
                )
            return cls(specification.variant, env=env, manifest=specification, **kwargs)
        variant = str(getattr(specification, "name", getattr(specification, "variant", VARIANT_FLOATING)))
        base_names = tuple(str(item) for item in getattr(specification, "base_joint_names", ()))
        finger_names = tuple(str(item) for item in getattr(specification, "finger_joint_names", SHARPAWAVE_HAND_JOINT_NAMES))
        source_specs = tuple(getattr(specification, "joint_specs", ()))
        source_by_name = {str(getattr(item, "name", "")): item for item in source_specs}
        joints: list[JointSpec] = []
        for name in base_names + finger_names:
            source = source_by_name.get(name)
            lower = getattr(source, "lower", None)
            upper = getattr(source, "upper", None)
            unit = str(getattr(source, "unit", "rad"))
            joints.append(
                JointSpec(
                    name=name,
                    role="wrist" if name in base_names else "finger",
                    unit=unit if unit in {"m", "rad", "dimensionless"} else "rad",
                    lower=lower,
                    upper=upper,
                    # The asset manifest has no calibrated policy scale. Keep
                    # identity for schema conversion and expose missing
                    # calibration in adapter metadata.
                    action_scale=1.0,
                    continuous=bool(getattr(source, "continuous", False)),
                    effort=getattr(source, "effort_limit", None),
                    velocity=getattr(source, "velocity_limit", None),
                )
            )
        source_fingertips = tuple(getattr(specification, "fingertip_specs", SHARPAWAVE_FINGERTIP_SPECS))
        fingertips = tuple(
            FingertipSpec(
                str(getattr(item, "role")),
                str(getattr(item, "pose_frame")),
                str(getattr(item, "contact_body")),
                getattr(item, "contact_sensor", None),
            )
            for item in source_fingertips
        )
        mode = str(getattr(getattr(specification, "action_schema", None), "default_profile", "joint_delta"))
        if mode not in ("joint_delta", "joint_target"):
            mode = "joint_delta"
        model = HandModelSpec(
            robot_name=ROBOT_NAME,
            variant=variant,
            schema=ActionSchema(
                schema_id=str(getattr(specification, "schema_id", f"{ROBOT_NAME}/{variant}/action-v{SCHEMA_VERSION}")),
                version=SCHEMA_VERSION,
                joints=tuple(joints),
                wrist_joint_names=base_names,
                finger_joint_names=finger_names,
                mode=mode,
            ),
            fingertips=fingertips,
            urdf_path=_public_manifest_path(
                getattr(specification, "urdf_path", SHARPAWAVE_VARIANTS[variant].urdf_path),
                label="SharpaWave URDF path",
                repository_root=kwargs.get("repository_root"),
            ),
            usd_path=_public_manifest_path(
                getattr(specification, "usd_path", SHARPAWAVE_VARIANTS[variant].usd_path),
                label="SharpaWave USD path",
                repository_root=kwargs.get("repository_root"),
            ),
            fixed_joint_names=tuple(str(item) for item in getattr(specification, "fixed_joint_names", ())),
            schema_version=f"v{SCHEMA_VERSION}",
            provenance={"source": "asset_config", "calibrated": False},
            root_link_name=str(getattr(specification, "root_link_name", "base_root")),
            vendor_fix_base=bool(getattr(specification, "vendor_fix_base", True)),
            root_semantics_status=str(getattr(specification, "root_semantics_status", "runtime_unverified")),
            contact_status=str(getattr(specification, "contact_status", "asset_declared_runtime_unverified")),
            actuator_control_mode=str(getattr(specification, "actuator_control_mode", "position_target")),
            actuator_limits_source=str(getattr(specification, "actuator_limits_source", "urdf")),
            actuator_gains_status=str(getattr(specification, "actuator_gains_status", "missing_calibration")),
        )
        return cls(variant, env=env, manifest=model, **kwargs)

    def __init__(
        self,
        variant: str = VARIANT_FLOATING,
        env: Any | None = None,
        *,
        env_id: int = 0,
        debug_privileged_allowed: bool = False,
        repository_root: str | Path | None = None,
        calibration: Mapping[str, Any] | None = None,
        contact_sensor: Any | None = None,
        manifest: HandModelSpec | None = None,
    ) -> None:
        if not isinstance(variant, str):
            # Permit the natural ``SharpaWaveAdapter(env, variant="...")`` style
            # without making the normal string-first constructor ambiguous.
            if env is None:
                env, variant = variant, VARIANT_FLOATING
            else:
                raise TypeError("variant must be a string when env is supplied positionally")
        variant = str(variant)
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported SharpaWave variant {variant!r}; expected one of {SUPPORTED_VARIANTS}")
        self.variant = variant
        self.env = env
        self.env_id = int(env_id)
        self.debug_privileged_allowed = bool(debug_privileged_allowed)
        self.repository_root = Path(repository_root).expanduser().resolve() if repository_root is not None else _repository_root()
        if manifest is None:
            # Prefer the dependency-light asset configuration whenever the
            # local assets extension is available.  The in-module model is a
            # deterministic fallback for installations that intentionally ship
            # task contracts without the 93-file asset tree.
            try:
                asset_spec = get_sharpawave_spec(variant)
            except (ImportError, ModuleNotFoundError, AttributeError):
                asset_spec = None
            if asset_spec is not None and not isinstance(asset_spec, HandModelSpec):
                manifest = type(self).from_spec(
                    asset_spec,
                    env=None,
                    repository_root=self.repository_root,
                ).model
            elif isinstance(asset_spec, HandModelSpec):
                manifest = asset_spec
        self._model = manifest or SHARPAWAVE_VARIANTS[variant]
        if self._model.variant != variant:
            raise ValueError("manifest variant does not match adapter variant")
        self.calibration = dict(calibration or {})
        self.contact_sensor = contact_sensor
        self._runtime_indices: dict[str, int] = {}
        self._body_indices: dict[str, int] = {}

    @classmethod
    def for_variant(cls, variant: str = VARIANT_FLOATING, env: Any | None = None, **kwargs: Any) -> "SharpaWaveAdapter":
        return cls(variant, env=env, **kwargs)

    @property
    def model(self) -> HandModelSpec:
        return self._model

    @property
    def schema(self) -> ActionSchema:
        return self._model.schema

    def get_action_schema(self) -> ActionSchema:
        return self.schema

    def get_fingertip_specs(self) -> tuple[FingertipSpec, ...]:
        return tuple(self.model.fingertips)

    @property
    def canonical_joint_names(self) -> tuple[str, ...]:
        return self.schema.canonical_joint_names

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return SHARPAWAVE_HAND_JOINT_NAMES

    @property
    def wrist_joint_names(self) -> tuple[str, ...]:
        return self.schema.wrist_joint_names

    def get_action_dim(self) -> int:
        """Compatibility method for task code that uses method-style dims."""

        return int(self.action_dim)

    def get_hand_joint_names(self) -> list[str]:
        return list(self.hand_joint_names)

    def get_finger_joint_names_for_role(self, finger_role: str | int) -> list[str]:
        """Resolve a semantic finger role using the SharpaWave manifest."""

        role = str(finger_role).strip().lower()
        role_to_name = {
            "1": "thumb",
            "2": "index",
            "3": "middle",
            "4": "ring",
            "5": "pinky",
        }
        role = role_to_name.get(role, role)
        if role not in role_to_name.values():
            return []
        prefix = f"right_{role}_"
        return [name for name in self.hand_joint_names if name.startswith(prefix)]

    def get_named_close_profile(self, profile_name: str) -> None:
        """SharpaWave profiles are unavailable until calibrated evidence exists."""

        raise MissingCapabilityError(
            MISSING_CALIBRATION,
            f"no SharpaWave close profile is calibrated for {profile_name!r}",
            details={"profile_name": str(profile_name), "variant": self.variant},
        )

    def get_base_joint_names(self) -> list[str]:
        return list(self.wrist_joint_names)

    def get_floating_wrist_joint_names(self) -> list[str]:
        return self.get_base_joint_names()

    def zero_action(self, batch_size: int = 1) -> np.ndarray:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        return np.zeros((int(batch_size), self.action_dim), dtype=np.float32)

    def get_finger_metric_joint_names(self, finger_group: str) -> dict[str, Any]:
        """Resolve semantic finger groups without assuming another hand's suffixes.

        The returned profile is suitable for diagnostics; physical metric
        calibration is deliberately marked unavailable until a grasp profile
        is supplied.
        """

        roles = list(dict.fromkeys(str(item).strip().lower() for item in str(finger_group).replace(",", " ").split() if item.strip()))
        if not roles:
            return {"ok": False, "names": [], "missing_fingers": [], "calibrated": False}
        names: list[str] = []
        missing: list[str] = []
        for role in roles:
            prefix = f"right_{role}_"
            matched = [name for name in self.hand_joint_names if name.startswith(prefix)]
            if not matched:
                missing.append(role)
            names.extend(matched)
        return {
            "ok": bool(names) and not missing,
            "names": names,
            "metric_joint_names": names,
            "nonmetric_joint_names": [],
            "missing_fingers": missing,
            "calibrated": False,
            "source": "sharpawave_semantic_manifest",
        }

    def open_hand_action(self, batch_size: int = 1) -> np.ndarray:
        return self.hand_action_for_mode("open", batch_size=batch_size)

    def close_hand_action(self, batch_size: int = 1) -> np.ndarray:
        return self.hand_action_for_mode("close_all", batch_size=batch_size)

    def default_configuration(self) -> Mapping[str, Any]:
        configured_scale = self._configured_action_scale()
        configuration = SharpaWaveConfiguration(
            robot_name=ROBOT_NAME,
            variant=self.variant,
            schema_id=self.schema.schema_id,
            action_mode=self.schema.mode,
            action_scale=None
            if configured_scale is None
            else tuple(float(item) for item in configured_scale),
            calibration_available=bool(self.calibration),
            base_pose_mapper_available=callable(self.calibration.get("base_pose_mapper")),
            contact_sensor_required=True,
            policy_available=False,
        ).as_dict() | {
            "robot": ROBOT_NAME,
            "action_dim": self.action_dim,
            "wrist_action_dim": self.wrist_action_dim,
            "finger_action_dim": self.finger_action_dim,
            "joint_names": list(self.canonical_joint_names),
            "action_units": list(self.schema.units),
            "joint_limits": [
                {
                    "name": joint.name,
                    "unit": joint.unit,
                    "lower": joint.lower,
                    "upper": joint.upper,
                    "continuous": joint.continuous,
                }
                for joint in self.schema.joints
            ],
            "fixed_joint_names": list(self.model.fixed_joint_names),
            "urdf_path": self.model.urdf_path,
            "usd_path": self.model.usd_path,
            "action_profile": self.schema.default_profile,
            "action_profiles": self.schema.supported_profiles,
            "action_scale_status": "calibrated" if configured_scale is not None else self.schema.scale_status,
            "action_scale_calibrated": configured_scale is not None,
            "root_link_name": self.model.root_link_name,
            "vendor_fix_base": self.model.vendor_fix_base,
            "root_semantics_status": self.model.root_semantics_status,
            "contact_status": self.model.contact_status,
            "actuator_control_mode": self.model.actuator_control_mode,
            "actuator_limits_source": self.model.actuator_limits_source,
            "actuator_gains_status": self.model.actuator_gains_status,
            "base_pose_mapper_status": "available" if callable(self.calibration.get("base_pose_mapper")) else "missing_calibration",
        }
        return configuration

    def _asset_path(self, relative: str | Path) -> Path:
        if self.repository_root is not None:
            return self.repository_root / relative
        return resolve_asset_path(relative)

    def validate_asset_schema(self) -> ValidationReport:
        """Validate URDF names, fixed joints, limits, and referenced meshes."""

        urdf_path = self._asset_path(self.model.urdf_path)
        usd_path = self._asset_path(self.model.usd_path) if self.model.usd_path else None
        if not urdf_path.is_file():
            return ValidationReport.failure(
                MISSING_ASSET,
                [f"URDF does not exist: {self.model.urdf_path}"],
                missing=[self.model.urdf_path],
                details={"urdf_path": self.model.urdf_path, "schema_id": self.schema.schema_id},
            )
        try:
            root = ET.parse(urdf_path).getroot()
        except (ET.ParseError, OSError) as exc:
            return ValidationReport.failure(
                INVALID_SCHEMA,
                [f"Unable to parse URDF {self.model.urdf_path}: {exc}"],
                details={"urdf_path": self.model.urdf_path, "schema_id": self.schema.schema_id},
            )
        joints = {str(item.attrib.get("name")): item for item in root.findall("joint")}
        expected = set(self.canonical_joint_names)
        missing = sorted(expected - set(joints))
        errors: list[str] = []
        warnings: list[str] = []
        if missing:
            errors.append(f"missing canonical joints: {', '.join(missing)}")
        fixed_missing = sorted(set(self.model.fixed_joint_names) - set(joints))
        if fixed_missing:
            errors.append(f"missing fixed joints: {', '.join(fixed_missing)}")
        for spec in self.schema.joints:
            joint = joints.get(spec.name)
            if joint is None:
                continue
            joint_type = str(joint.attrib.get("type", ""))
            if joint_type == "fixed":
                errors.append(f"canonical joint is fixed: {spec.name}")
            limit = joint.find("limit")
            if limit is None:
                warnings.append(f"joint has no explicit limit: {spec.name}")
                continue
            try:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            except (KeyError, ValueError):
                errors.append(f"invalid limit for joint: {spec.name}")
                continue
            if spec.lower is not None and abs(lower - float(spec.lower)) > 2.0e-3:
                errors.append(f"lower limit mismatch for {spec.name}: manifest={spec.lower} urdf={lower}")
            if spec.upper is not None and abs(upper - float(spec.upper)) > 2.0e-3:
                errors.append(f"upper limit mismatch for {spec.name}: manifest={spec.upper} urdf={upper}")
        link_names = {str(item.attrib.get("name")) for item in root.findall("link")}
        expected_links = {finger.pose_frame for finger in self.model.fingertips} | {
            finger.contact_body for finger in self.model.fingertips
        }
        missing_links = sorted(expected_links - link_names)
        if missing_links:
            errors.append(f"missing fingertip/contact links: {', '.join(missing_links)}")
        mesh_missing: list[str] = []
        for mesh in root.findall(".//mesh"):
            filename = str(mesh.attrib.get("filename", ""))
            if not filename or filename.startswith("package://"):
                warnings.append(f"unresolved mesh URI: {filename or '<empty>'}")
                continue
            mesh_path = (urdf_path.parent / filename).resolve() if not Path(filename).is_absolute() else Path(filename)
            if not mesh_path.is_file():
                mesh_missing.append(filename)
        if mesh_missing:
            errors.append(f"missing mesh references: {', '.join(sorted(set(mesh_missing)))}")
        if usd_path is not None and not usd_path.is_file():
            errors.append(f"USD asset does not exist: {self.model.usd_path}")
        details = {
            "robot_name": ROBOT_NAME,
            "variant": self.variant,
            "schema_id": self.schema.schema_id,
            "schema_version": self.schema.schema_version,
            "urdf_path": self.model.urdf_path,
            "urdf_sha256": _sha256(urdf_path),
            "usd_path": self.model.usd_path,
            "usd_sha256": None if usd_path is None else _sha256(usd_path),
            "joint_count": len(joints),
            "movable_joint_count": sum(str(item.attrib.get("type")) != "fixed" for item in joints.values()),
            "canonical_action_dim": self.action_dim,
            "expected_action_dim": self.action_dim,
            "mesh_count": len(root.findall(".//mesh")),
        }
        if errors:
            return ValidationReport.failure(INVALID_SCHEMA, errors, missing=missing + fixed_missing, warnings=warnings, details=details)
        return ValidationReport.success(details=details, warnings=warnings)

    def _articulation(self, runtime: Any | None = None) -> Any | None:
        value = runtime if runtime is not None else self.env
        if value is None:
            return None
        # Common environment wrappers; the first object with names wins.
        candidates = [value]
        for attr in ("robot", "_robot", "articulation", "_articulation"):
            try:
                candidate = getattr(value, attr, None)
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
        scene = getattr(value, "scene", None)
        if scene is not None:
            candidates.append(scene)
            try:
                candidate = scene["robot"]
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
        for candidate in candidates:
            if _first_attr([candidate, getattr(candidate, "data", None)], ("joint_names", "body_names")) is not None:
                return candidate
        return candidates[-1] if candidates else None

    def _runtime_names(self, runtime: Any | None = None) -> tuple[tuple[str, ...], tuple[str, ...], Any | None]:
        articulation = self._articulation(runtime)
        if articulation is None:
            return (), (), None
        data = getattr(articulation, "data", None)
        joint_names = _as_name_tuple(_first_attr([articulation, data], ("joint_names", "dof_names")))
        body_names = _as_name_tuple(_first_attr([articulation, data], ("body_names", "link_names")))
        return joint_names, body_names, articulation

    def resolve_runtime_indices(self, runtime_joint_names: Sequence[str] | Mapping[str, int]) -> dict[str, int]:
        if isinstance(runtime_joint_names, Mapping):
            try:
                supplied = {str(name): int(index) for name, index in runtime_joint_names.items()}
            except (TypeError, ValueError) as exc:
                raise SchemaValidationError("runtime joint index mapping is not numeric", code=RUNTIME_SCHEMA_MISMATCH) from exc
            names = tuple(supplied)
            duplicates = []
            missing = [name for name in self.canonical_joint_names if name not in supplied]
            if missing:
                raise SchemaValidationError(
                    f"runtime articulation is missing canonical joints: {missing}",
                    code=RUNTIME_SCHEMA_MISMATCH,
                    details={"missing": missing, "schema_id": self.schema.schema_id},
                )
            mapping = {name: supplied[name] for name in self.canonical_joint_names}
            if len(set(mapping.values())) != len(mapping):
                raise SchemaValidationError("runtime joint index mapping contains duplicates", code=RUNTIME_SCHEMA_MISMATCH)
            if any(index < 0 for index in mapping.values()):
                raise SchemaValidationError("runtime joint index mapping contains negative indices", code=RUNTIME_SCHEMA_MISMATCH)
            self._runtime_indices = mapping
            return dict(mapping)
        names = tuple(str(item) for item in runtime_joint_names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise SchemaValidationError(f"runtime joint names contain duplicates: {duplicates}", code=RUNTIME_SCHEMA_MISMATCH)
        missing = [name for name in self.canonical_joint_names if name not in names]
        if missing:
            raise SchemaValidationError(
                f"runtime articulation is missing canonical joints: {missing}",
                code=RUNTIME_SCHEMA_MISMATCH,
                details={"missing": missing, "schema_id": self.schema.schema_id},
            )
        mapping = {name: names.index(name) for name in self.canonical_joint_names}
        self._runtime_indices = mapping
        return dict(mapping)

    def resolve_joint_indices(self, runtime_joint_names: Sequence[str] | Mapping[str, int]) -> tuple[int, ...]:
        """Return canonical indices in a compact asset-config-compatible form."""

        mapping = self.resolve_runtime_indices(runtime_joint_names)
        return tuple(mapping[name] for name in self.canonical_joint_names)

    def validate_runtime_joint_names(self, runtime_joint_names: Sequence[str]) -> ValidationReport:
        """Validate names without requiring a live articulation object."""

        names = tuple(str(item) for item in runtime_joint_names)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        missing = [name for name in self.canonical_joint_names if name not in names]
        extras = [name for name in names if name not in self.canonical_joint_names]
        details = {"runtime_joint_names": list(names), "extra_runtime_names": extras}
        if duplicates or missing:
            errors = []
            if duplicates:
                errors.append(f"duplicate runtime joints: {', '.join(duplicates)}")
            if missing:
                errors.append(f"missing runtime joints: {', '.join(missing)}")
            return ValidationReport.failure(RUNTIME_SCHEMA_MISMATCH, errors, missing=missing, details=details)
        return ValidationReport.success(details=details, warnings=(f"extra runtime joints: {', '.join(extras)}",) if extras else ())

    def validate_runtime_articulation(self, runtime: Any | None = None, *, require_contact: bool = False) -> ValidationReport:
        joint_names, body_names, articulation = self._runtime_names(runtime)
        if articulation is None or not joint_names:
            return ValidationReport.failure(
                MISSING_RUNTIME_ARTICULATION,
                ["runtime articulation does not expose joint_names"],
                missing=["joint_names"],
                details={"schema_id": self.schema.schema_id},
            )
        errors: list[str] = []
        warnings: list[str] = []
        missing = [name for name in self.canonical_joint_names if name not in joint_names]
        if missing:
            errors.append(f"missing runtime joints: {', '.join(missing)}")
        if len({name for name in joint_names}) != len(joint_names):
            errors.append("runtime joint names contain duplicates")
        if not missing:
            self._runtime_indices = {name: joint_names.index(name) for name in self.canonical_joint_names}
        missing_pose = [finger.pose_frame for finger in self.model.fingertips if finger.pose_frame not in body_names]
        missing_contact = [finger.contact_body for finger in self.model.fingertips if finger.contact_body not in body_names]
        if missing_pose:
            errors.append(f"missing runtime pose frames: {', '.join(missing_pose)}")
        if missing_contact:
            message = f"missing runtime contact bodies: {', '.join(missing_contact)}"
            (errors if require_contact else warnings).append(message)
        dof_count = _first_attr([articulation, getattr(articulation, "data", None)], ("num_dof", "num_joints"))
        if dof_count is not None:
            try:
                if int(dof_count) < self.action_dim:
                    errors.append(f"runtime DoF count {dof_count} is below schema width {self.action_dim}")
            except (TypeError, ValueError):
                warnings.append("runtime DoF count is not numeric")
        details = {
            "schema_id": self.schema.schema_id,
            "runtime_joint_names": list(joint_names),
            "runtime_body_names": list(body_names),
            "runtime_joint_indices": dict(self._runtime_indices),
            "pose_frames_available": not missing_pose,
            "contact_bodies_available": not missing_contact,
            "contact_required": bool(require_contact),
        }
        if errors:
            return ValidationReport.failure(RUNTIME_SCHEMA_MISMATCH, errors, missing=missing + missing_pose + missing_contact, warnings=warnings, details=details)
        return ValidationReport.success(details=details, warnings=warnings)

    def _read_joint_value(self, articulation: Any, names: Sequence[str]) -> Any:
        data = getattr(articulation, "data", None)
        value = _first_attr([articulation, data], names)
        if isinstance(value, Mapping):
            columns = []
            for name in self.canonical_joint_names:
                if name not in value:
                    return None
                column = _to_numpy(value[name]).reshape(-1)
                columns.append(column)
            batch = max(len(column) for column in columns)
            normalized = [np.full(batch, float(column[0])) if len(column) == 1 and batch > 1 else column for column in columns]
            return np.stack(normalized, axis=1)
        return value

    def _select_runtime_columns(self, value: Any, indices: Sequence[int], *, label: str) -> np.ndarray:
        matrix = _batch_matrix(value, label=label)
        if not indices or max(indices) >= matrix.shape[1]:
            raise SchemaValidationError(f"{label} does not contain all runtime joint columns", code=RUNTIME_SCHEMA_MISMATCH)
        return matrix[:, list(indices)]

    def get_robot_observation(self, runtime: Any | None = None) -> RobotObservation:
        report = self.validate_runtime_articulation(runtime)
        report.require_ok()
        _, body_names, articulation = self._runtime_names(runtime)
        assert articulation is not None
        data = getattr(articulation, "data", None)
        raw_positions = _first_attr(
            [articulation, data], ("joint_pos", "joint_position", "positions", "q")
        )
        positions = self._read_joint_value(articulation, ("joint_pos", "joint_position", "positions", "q"))
        if positions is None:
            raise MissingCapabilityError(MISSING_RUNTIME_ARTICULATION, "runtime articulation has no joint positions")
        # Mapping-valued articulation state is already assembled in canonical
        # name order by ``_read_joint_value``.  Applying runtime indices again
        # would silently permute those columns a second time when USD/URDF
        # traversal order differs from the manifest.  Tensor-valued state still
        # follows the explicit runtime name-to-index mapping below.
        if isinstance(raw_positions, Mapping):
            position = _batch_matrix(positions, label="joint_position")
        else:
            position = self._select_runtime_columns(
                positions,
                [self._runtime_indices[name] for name in self.canonical_joint_names],
                label="joint_position",
            )
        raw_velocities = _first_attr(
            [articulation, data], ("joint_vel", "joint_velocity", "velocities", "qd")
        )
        velocities_raw = self._read_joint_value(articulation, ("joint_vel", "joint_velocity", "velocities", "qd"))
        if velocities_raw is None:
            velocity = None
        elif isinstance(raw_velocities, Mapping):
            velocity = _batch_matrix(velocities_raw, label="joint_velocity")
        else:
            velocity = self._select_runtime_columns(
                velocities_raw,
                [self._runtime_indices[name] for name in self.canonical_joint_names],
                label="joint_velocity",
            )
        frames = self.get_fingertip_frames(articulation)
        contacts = self.get_contact_measurements(articulation, strict=False)
        metadata = {
            "runtime_joint_indices": dict(self._runtime_indices),
            "runtime_body_names": list(body_names),
            "validation": report.as_dict(),
        }
        return RobotObservation(
            schema_id=self.schema.schema_id,
            joint_names=self.canonical_joint_names,
            joint_position=position,
            joint_velocity=velocity,
            fingertip_poses=frames,
            contact_measurements=contacts,
            metadata=metadata,
        )

    def _mapping_to_batch(self, action: Mapping[str, Any]) -> np.ndarray:
        missing = [name for name in self.canonical_joint_names if name not in action]
        if missing:
            raise SchemaValidationError(f"canonical action mapping is missing joints: {missing}", code=INVALID_ACTION)
        columns = [_to_numpy(action[name]).reshape(-1) for name in self.canonical_joint_names]
        batch = max(len(column) for column in columns)
        normalized: list[np.ndarray] = []
        for name, column in zip(self.canonical_joint_names, columns):
            if len(column) == 1 and batch > 1:
                column = np.full(batch, float(column[0]), dtype=np.float64)
            if len(column) != batch:
                raise SchemaValidationError(f"action column {name!r} has inconsistent batch length", code=INVALID_ACTION)
            normalized.append(column)
        return np.stack(normalized, axis=1)

    def _configured_action_scale(self) -> np.ndarray | None:
        """Return a validated policy scale, or ``None`` when it is absent.

        The identity values carried by the morphology schema are placeholders for
        shape/type tests, never a physical calibration.  Runtime action mapping
        must use an explicit finite positive vector supplied under
        ``calibration['action_scale']``.
        """

        if "action_scale" not in self.calibration:
            return None
        raw = self.calibration.get("action_scale")
        try:
            scale = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MissingCalibrationError(
                MISSING_CALIBRATION,
                "calibration['action_scale'] must be a numeric vector",
                field="action_scale",
            ) from exc
        if scale.ndim != 1 or scale.shape[0] != self.action_dim:
            raise MissingCalibrationError(
                MISSING_CALIBRATION,
                f"calibration['action_scale'] must have width {self.action_dim}",
                field="action_scale",
                expected_width=self.action_dim,
                actual_shape=tuple(scale.shape),
            )
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise MissingCalibrationError(
                MISSING_CALIBRATION,
                "calibration['action_scale'] must contain finite positive values",
                field="action_scale",
            )
        return scale

    def _require_action_scale(self, canonical: np.ndarray, profile: ActionMode) -> np.ndarray | None:
        scale = self._configured_action_scale()
        # The calibration gate is intentionally exact: every non-zero command
        # needs an explicitly supplied scale.  A tolerance here would turn
        # tiny-but-real policy outputs into an uncalibrated motion silently,
        # which is especially dangerous for prismatic wrist joints.  The
        # all-zero matrix remains the one documented schema no-op exemption.
        if profile == "joint_target" or np.any(canonical != 0.0):
            if scale is None:
                raise MissingCalibrationError(
                    MISSING_CALIBRATION,
                    f"action profile {profile!r} requires calibrated action_scale",
                    profile=profile,
                    action_dim=self.action_dim,
                )
        return scale

    def map_canonical_action(
        self,
        action: Any,
        *,
        runtime_joint_names: Sequence[str] | None = None,
        current_position: Any | None = None,
        profile: ActionMode | None = None,
        clip_to_limits: bool = True,
        action_schema_id: str | None = None,
    ) -> np.ndarray:
        """Map a canonical batch action into runtime name order.

        A mapping input is accepted for policy code that emits semantic names;
        array input always follows ``schema.canonical_joint_names``.  Runtime
        ordering is resolved by exact names and never by a fixed slice/index.
        """

        if action_schema_id is not None and str(action_schema_id) != self.schema.schema_id:
            raise SchemaValidationError(
                f"action schema {action_schema_id!r} does not match {self.schema.schema_id!r}",
                code=INVALID_SCHEMA,
                details={"expected": self.schema.schema_id, "actual": str(action_schema_id)},
            )
        canonical = self._mapping_to_batch(action) if isinstance(action, Mapping) else action
        canonical = self.schema.validate_batch(canonical)
        selected_profile = profile or self.schema.mode
        if selected_profile not in ("joint_delta", "joint_target"):
            raise SchemaValidationError(f"unknown action profile {selected_profile!r}", code=INVALID_ACTION)
        configured_scale = self._require_action_scale(canonical, selected_profile)
        if selected_profile == "joint_delta":
            # A zero delta is a safe schema no-op even before calibration.  Any
            # non-zero delta must use the explicit calibrated vector above.
            scale = np.zeros(self.action_dim, dtype=np.float64) if configured_scale is None else configured_scale
            physical = canonical * scale.reshape(1, -1)
            if current_position is not None:
                current = _batch_matrix(current_position, label="current_position")
                if runtime_joint_names is not None and current.shape[1] == len(tuple(runtime_joint_names)):
                    runtime_map = self.resolve_runtime_indices(runtime_joint_names)
                    current = current[:, [runtime_map[name] for name in self.canonical_joint_names]]
                elif current.shape[1] != self.action_dim:
                    raise SchemaValidationError("current_position width does not match canonical or runtime schema", code=INVALID_ACTION)
                if current.shape[0] == 1 and physical.shape[0] > 1:
                    current = np.repeat(current, physical.shape[0], axis=0)
                if current.shape != physical.shape:
                    raise SchemaValidationError("current_position batch does not match action batch", code=INVALID_ACTION)
                physical = current + physical
        else:
            # The scale is a required calibration gate for target policies even
            # though the normalized target is resolved against asset limits.
            lower = self.schema.lower
            upper = self.schema.upper
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise MissingCapabilityError(MISSING_LIMITS, "joint_target profile requires finite limits")
            physical = lower.reshape(1, -1) + (canonical - self.schema.normalized_low) / (self.schema.normalized_high - self.schema.normalized_low) * (upper - lower).reshape(1, -1)
        continuous = np.asarray(
            [bool(joint.continuous) for joint in self.schema.joints], dtype=bool
        )
        if np.any(continuous):
            # Continuous wrist rotations use the configured radians convention;
            # wrapping is applied after deltas/targets are resolved and before
            # soft-limit clipping so a crossing at +/-pi remains continuous.
            indices = np.flatnonzero(continuous)
            raw_continuous = physical[:, indices]
            wrapped = (raw_continuous + math.pi) % (2.0 * math.pi) - math.pi
            # Preserve the positive limit for an exact +pi target instead of
            # representing the same pose as -pi.
            wrapped = np.where(
                np.isclose(wrapped, -math.pi, atol=1.0e-12) & (raw_continuous > 0.0),
                math.pi,
                wrapped,
            )
            physical[:, indices] = wrapped
        # A delta without a reference position is still a delta; clipping it to
        # absolute joint limits would incorrectly erase signed commands for joints
        # whose home range starts at zero.  Clip absolute targets, or deltas once
        # they have been added to a supplied current position.
        if clip_to_limits and (selected_profile == "joint_target" or current_position is not None):
            lower = self.schema.lower
            upper = self.schema.upper
            finite_lower = np.isfinite(lower)
            finite_upper = np.isfinite(upper)
            if np.any(finite_lower) or np.any(finite_upper):
                physical = np.minimum(physical, np.where(finite_upper, upper, np.inf).reshape(1, -1))
                physical = np.maximum(physical, np.where(finite_lower, lower, -np.inf).reshape(1, -1))
        if runtime_joint_names is None:
            return physical
        runtime_names = tuple(str(name) for name in runtime_joint_names)
        runtime_map = self.resolve_runtime_indices(runtime_names)
        output = np.zeros((physical.shape[0], len(runtime_names)), dtype=np.float64)
        for canonical_index, name in enumerate(self.canonical_joint_names):
            output[:, runtime_map[name]] = physical[:, canonical_index]
        return output

    def get_fingertip_frames(self, runtime: Any | None = None) -> Mapping[str, Any]:
        _, body_names, _ = self._runtime_names(runtime)
        self._body_indices = {name: body_names.index(name) for name in body_names}
        position_raw = _first_attr(
            [self._articulation(runtime), getattr(self._articulation(runtime), "data", None) if self._articulation(runtime) is not None else None],
            ("body_pos_w", "body_position", "body_pos", "link_pos"),
        )
        quaternion_raw = _first_attr(
            [self._articulation(runtime), getattr(self._articulation(runtime), "data", None) if self._articulation(runtime) is not None else None],
            ("body_quat_w", "body_quaternion", "body_quat", "link_quat"),
        )
        result: dict[str, Any] = {}
        for finger in self.model.fingertips:
            pose_index = self._body_indices.get(finger.pose_frame)
            contact_index = self._body_indices.get(finger.contact_body)
            item: dict[str, Any] = {
                "role": finger.role,
                "pose_frame": finger.pose_frame,
                "contact_body": finger.contact_body,
                "pose_index": pose_index,
                "contact_index": contact_index,
                "pose_available": pose_index is not None,
                "contact_available": contact_index is not None,
            }
            if pose_index is not None and position_raw is not None:
                # A flattened body position array is common in simple test doubles;
                # retain the body axis when present and avoid fabricating values.
                raw = _to_numpy(position_raw)
                if raw.ndim >= 3 and raw.shape[1] > pose_index:
                    item["position"] = raw[:, pose_index, :]
                elif raw.ndim == 2 and raw.shape[1] >= (pose_index + 1) * 3:
                    item["position"] = raw[:, pose_index * 3 : pose_index * 3 + 3]
            if pose_index is not None and quaternion_raw is not None:
                raw = _to_numpy(quaternion_raw)
                if raw.ndim >= 3 and raw.shape[1] > pose_index:
                    item["orientation"] = raw[:, pose_index, :]
                elif raw.ndim == 2 and raw.shape[1] >= (pose_index + 1) * 4:
                    item["orientation"] = raw[:, pose_index * 4 : pose_index * 4 + 4]
            result[finger.role] = item
        return result

    def _contact_array(self, articulation: Any, body_names: tuple[str, ...]) -> tuple[np.ndarray | None, str]:
        data = getattr(articulation, "data", None)
        sensor_data = getattr(self.contact_sensor, "data", None)
        raw = _first_attr(
            [self.contact_sensor, sensor_data, articulation, data],
            ("net_forces_w", "net_contact_forces", "contact_forces", "force_matrix_w", "forces"),
        )
        source = "runtime_contact_forces"
        if raw is None and self.contact_sensor is not None and callable(self.contact_sensor):
            try:
                raw = self.contact_sensor(body_names)
                source = "contact_sensor_callback"
            except TypeError:
                raw = self.contact_sensor()
                source = "contact_sensor_callback"
        if raw is None:
            return None, "no contact force tensor or callback"
        if isinstance(raw, Mapping):
            values: dict[str, np.ndarray] = {}
            for name in body_names:
                if name in raw:
                    value = _to_numpy(raw[name])
                    if value.ndim >= 2 and value.shape[-1] in (2, 3, 4):
                        value = np.linalg.norm(value, axis=-1)
                    values[name] = np.asarray(value, dtype=np.float64).reshape(-1)
            if not values:
                return None, "contact mapping has no known body names"
            max_batch = max(np.asarray(value).reshape(-1).shape[0] for value in values.values())
            array = np.zeros((max_batch, len(body_names)), dtype=np.float64)
            for index, name in enumerate(body_names):
                if name in values:
                    column = np.abs(values[name].reshape(-1))
                    if len(column) == 1:
                        array[:, index] = column[0]
                    elif len(column) == max_batch:
                        array[:, index] = column
                    else:
                        return None, f"contact mapping batch mismatch for {name}"
            return array, source
        array = _to_numpy(raw)
        if array.ndim < 2:
            return None, "contact force tensor is not batch-first"
        if array.ndim >= 3 and array.shape[-1] in (2, 3, 4):
            # Reduce vector components, then reduce history/pair dimensions while
            # preserving the axis whose width matches the known body count.
            magnitudes = np.linalg.norm(array, axis=-1)
            if magnitudes.ndim > 2:
                body_axis = next(
                    (axis for axis in range(1, magnitudes.ndim) if magnitudes.shape[axis] == len(body_names)),
                    None,
                )
                if body_axis is None:
                    return None, "contact force tensor has no body axis"
                magnitudes = np.moveaxis(magnitudes, body_axis, -1)
                reduce_axes = tuple(range(1, magnitudes.ndim - 1))
                array = magnitudes.sum(axis=reduce_axes) if reduce_axes else magnitudes
            else:
                array = magnitudes
        elif array.ndim > 2:
            array = np.linalg.norm(array, axis=-1)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            return None, "contact force tensor cannot be reduced to batch x body"
        return np.abs(array), source

    def get_contact_measurements(self, runtime: Any | None = None, *, strict: bool = False) -> ContactMeasurements:
        _, body_names, articulation = self._runtime_names(runtime)
        known = [finger.contact_body for finger in self.model.fingertips]
        missing = [name for name in known if name not in body_names]
        if articulation is None or not body_names:
            result = ContactMeasurements(False, reason="runtime articulation has no body_names", body_names=tuple(known))
            if strict:
                raise MissingCapabilityError(CONTACT_UNAVAILABLE, result.reason, details=result.as_dict())
            return result
        if missing:
            result = ContactMeasurements(False, reason=f"missing contact bodies: {', '.join(missing)}", body_names=tuple(body_names))
            if strict:
                raise MissingCapabilityError(CONTACT_UNAVAILABLE, result.reason, details=result.as_dict())
            return result
        array, source = self._contact_array(articulation, body_names)
        if array is None:
            result = ContactMeasurements(False, reason=source, body_names=tuple(known))
            if strict:
                raise MissingCapabilityError(CONTACT_UNAVAILABLE, source, details=result.as_dict())
            return result
        values: dict[str, Any] = {}
        for finger in self.model.fingertips:
            index = body_names.index(finger.contact_body)
            if index >= array.shape[1]:
                result = ContactMeasurements(False, reason="contact tensor does not contain all body columns", source=source, body_names=tuple(body_names))
                if strict:
                    raise MissingCapabilityError(CONTACT_UNAVAILABLE, result.reason, details=result.as_dict())
                return result
            values[finger.role] = array[:, index]
        return ContactMeasurements(True, values=values, source=source, body_names=tuple(known))

    def hand_action_for_mode(self, mode: str, *, batch_size: int | None = None) -> np.ndarray:
        """Return a calibrated hand action or raise instead of guessing one."""

        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if mode not in self.calibration:
            raise MissingCalibrationError(
                MISSING_CALIBRATION,
                f"no calibrated hand profile for mode {mode!r}",
                mode=str(mode),
                available_modes=sorted(self.calibration),
            )
        raw = self.calibration[mode]
        # Calibration files commonly store only finger commands.  Expand that
        # explicit 22-DoF profile into the full variant schema with zero wrist
        # commands; never infer a wrist profile from finger values.
        if isinstance(raw, Mapping):
            missing = [name for name in self.schema.finger_joint_names if name not in raw]
            if missing:
                raise MissingCalibrationError(
                    MISSING_CALIBRATION,
                    f"calibrated hand profile is missing joints: {missing}",
                    mode=str(mode),
                )
            columns = [_to_numpy(raw[name]).reshape(-1) for name in self.schema.finger_joint_names]
            batch = max(len(column) for column in columns)
            finger = np.stack(
                [np.full(batch, float(column[0])) if len(column) == 1 and batch > 1 else column for column in columns],
                axis=1,
            )
            full = np.zeros((batch, self.action_dim), dtype=np.float64)
            full[:, self.wrist_action_dim :] = finger
            result = self.schema.validate_batch(full, name=f"calibration[{mode}]")
            self._require_action_scale(result, self.schema.mode)
            if batch_size is not None and result.shape[0] == 1 and int(batch_size) > 1:
                result = np.repeat(result, int(batch_size), axis=0)
            return result
        array = _to_numpy(raw)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim == 2 and array.shape[1] == self.finger_action_dim:
            full = np.zeros((array.shape[0], self.action_dim), dtype=np.float64)
            full[:, self.wrist_action_dim :] = array
            array = full
        result = self.schema.validate_batch(array, name=f"calibration[{mode}]")
        # A named pose is still a canonical action.  Keep the same calibration
        # gate as ``map_canonical_action`` so it cannot bypass the missing-scale
        # check when a generic task calls ``open_hand_action``/``close_hand_action``
        # directly.  An all-zero profile remains a safe schema-only no-op.
        self._require_action_scale(result, self.schema.mode)
        if batch_size is not None and result.shape[0] == 1 and int(batch_size) > 1:
            result = np.repeat(result, int(batch_size), axis=0)
        return result

    def asset_manifest(self) -> dict[str, Any]:
        """Return a JSON-ready manifest snapshot with current file hashes."""

        urdf = self._asset_path(self.model.urdf_path)
        usd = self._asset_path(self.model.usd_path) if self.model.usd_path else None
        return {
            "robot_name": ROBOT_NAME,
            "variant": self.variant,
            "schema_id": self.schema.schema_id,
            "schema_version": self.model.schema_version,
            "urdf_path": self.model.urdf_path,
            "urdf_sha256": _sha256(urdf),
            "usd_path": self.model.usd_path,
            "usd_sha256": None if usd is None else _sha256(usd),
            "canonical_joint_names": list(self.canonical_joint_names),
            "fixed_joint_names": list(self.model.fixed_joint_names),
            "fingertips": [
                {
                    "role": finger.role,
                    "pose_frame": finger.pose_frame,
                    "contact_body": finger.contact_body,
                }
                for finger in self.model.fingertips
            ],
        }


# Friendly aliases used by external scripts.
SharpaWaveRobotAdapter = SharpaWaveAdapter
SharpaWaveVariantManifest = HandModelSpec
# Report aliases mirror the declarative asset module while retaining one
# serializable result type in the adapter layer.
AssetValidationReport = ValidationReport
RuntimeSchemaReport = ValidationReport


def get_sharpawave_spec(variant: str = VARIANT_FLOATING) -> Any:
    """Return the declarative asset specification for a supported variant.

    When the assets extension is importable, its manifest is returned directly so
    limits and provenance have one source of truth.  The local immutable model is
    a dependency-light fallback for environments that only install task code.
    """

    key = str(variant).strip().lower()
    if key not in SUPPORTED_VARIANTS:
        choices = ", ".join(SUPPORTED_VARIANTS)
        raise ValueError(f"unknown SharpaWave variant {variant!r}; choose one of: {choices}")
    try:
        from isaaclab_assets.robots.sharpawave import get_sharpawave_variant

        return get_sharpawave_variant(key)  # type: ignore[return-value]
    except (ImportError, ModuleNotFoundError, AttributeError):
        return SHARPAWAVE_VARIANTS[key]


def run_sharpawave_preflight(
    *,
    variant: str = VARIANT_FLOATING,
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    require_runtime: bool = False,
) -> int:
    """Run a dependency-only preflight for scripts and CI.

    A non-zero result means the local asset schema is not available.  Runtime
    calibration/contact/policy gates are reported in the returned JSON when an
    output directory is supplied, but they are not silently treated as physical
    success.
    """

    destination: Path | None = None
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()

    adapter = SharpaWaveAdapter.from_spec(
        get_sharpawave_spec(variant), repository_root=repo_root
    )
    asset_report = adapter.validate_asset_schema()
    payload = {
        "robot_name": ROBOT_NAME,
        "variant": adapter.variant,
        "schema_id": adapter.schema.schema_id,
        "schema_version": adapter.schema.schema_version,
        "status": "system_only" if asset_report.ok else "missing_dependency",
        "asset_validation": asset_report.as_dict(),
        "configuration": adapter.default_configuration(),
        "action_dim": adapter.action_dim,
        "wrist_action_dim": adapter.wrist_action_dim,
        "finger_action_dim": adapter.finger_action_dim,
        # This is a schema/preflight result, not a manipulation rollout. Keep
        # every honesty field explicit so downstream report readers never infer
        # physical or privileged behavior from an omitted key.
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": False,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "physical_insert_success": False,
        "physical_success": False,
        "oracle_visual_only": False,
        "not_physical": True,
        "runtime_validation": {
            "status": "not_run",
            "reason": "Isaac articulation/contact/calibration gates are not supplied",
        },
    }
    if destination is not None:
        import json

        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"sharpawave_{adapter.variant}_preflight.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    if not asset_report.ok or bool(require_runtime):
        return 2
    return 0


__all__ = [
    "ASSET_ROOT_RELATIVE",
    "AssetValidationReport",
    "CONTACT_UNAVAILABLE",
    "INVALID_ACTION",
    "INVALID_SCHEMA",
    "MISSING_ASSET",
    "MISSING_CALIBRATION",
    "MISSING_LIMITS",
    "MISSING_RUNTIME_ARTICULATION",
    "POLICY_UNAVAILABLE",
    "RUNTIME_SCHEMA_MISMATCH",
    "ROBOT_NAME",
    "SCHEMA_VERSION",
    "SHARPAWAVE",
    "SHARPAWAVE_FLOATING_BASE_JOINT_NAMES",
    "SHARPAWAVE_FINGERTIP_SPECS",
    "SHARPAWAVE_HAND_JOINT_NAMES",
    "SHARPAWAVE_MANIFEST",
    "SHARPAWAVE_PEG_FIXEDROT_BASE_JOINT_NAMES",
    "SHARPAWAVE_VARIANTS",
    "SUPPORTED_VARIANTS",
    "VARIANT_FLOATING",
    "VARIANT_MANIFESTS",
    "VARIANT_PEG_FIXEDROT",
    "SharpaWaveAdapter",
    "SharpaWaveConfiguration",
    "SharpaWaveRobotAdapter",
    "SharpaWaveVariantManifest",
    "RuntimeSchemaReport",
    "MissingCalibrationError",
    "get_sharpawave_spec",
    "run_sharpawave_preflight",
    "resolve_asset_path",
]
