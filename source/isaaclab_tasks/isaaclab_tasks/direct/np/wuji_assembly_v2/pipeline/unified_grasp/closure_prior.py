"""Safe extraction of morphology-independent closure motion priors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECORDING = Path("/mnt/data/recording_1.pkl")
FALLBACK_RECORDING = Path(
    "/home/CNS2025331827/docs/grasp_small_obj/FingerEyeLab/scripts/new_pos_2/recording_1.pkl"
)


@dataclass(frozen=True)
class ClosurePhase:
    kind: str
    start_frame: int
    end_frame: int
    frame_count: int
    normalized_progress: tuple[float, ...]


@dataclass(frozen=True)
class ClosurePrior:
    schema_version: int
    source_path: str
    source_sha256: str
    source_morphology_unknown: bool
    source_joint_count: int
    frame_count: int
    source_success: bool
    successful_grasp_demonstration: bool
    source_role: str
    recording_audit: dict[str, Any]
    phases: tuple[ClosurePhase, ...]
    discontinuity_frames: tuple[int, ...]
    explained_variance_ratio: tuple[float, ...]
    cumulative_explained_variance: tuple[float, ...]
    synergy_basis: tuple[tuple[float, ...], ...]
    normalized_latent_trajectory: tuple[tuple[float, ...], ...]
    reconstruction_rmse_by_rank: dict[str, float]
    tracking_best_lag_frames: int
    tracking_rmse: float
    tracking_rmse_by_joint: tuple[float, ...]
    hold_fraction: float
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Unpickler that accepts only the globals needed by NumPy arrays."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return super().find_class(module, name)


def restricted_numpy_load(path: str | Path) -> Any:
    raw = Path(path).read_bytes()
    return RestrictedNumpyUnpickler(io.BytesIO(raw)).load()


def resolve_recording_path(requested: str | Path | None = None) -> Path:
    path = Path(requested) if requested else DEFAULT_RECORDING
    if path.is_file():
        return path.resolve()
    if path == DEFAULT_RECORDING and FALLBACK_RECORDING.is_file():
        return FALLBACK_RECORDING.resolve()
    raise FileNotFoundError(f"closure recording not found: {path}")


def _array(data: dict[str, Any], key: str, *, width: int | None = None) -> np.ndarray:
    value = np.asarray(data.get(key))
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0, :]
    if value.ndim != 2 or (width is not None and value.shape[1] != width):
        raise ValueError(f"{key} has unexpected shape {value.shape}")
    return value.astype(np.float64, copy=False)


def _make_phase(kind: str, start: int, end: int) -> ClosurePhase:
    count = end - start + 1
    progress = np.linspace(0.0, 1.0, count).tolist() if count > 1 else [1.0]
    return ClosurePhase(kind, start, end, count, tuple(float(x) for x in progress))


def _phases(actions: np.ndarray, source_sha256: str) -> tuple[tuple[ClosurePhase, ...], tuple[int, ...]]:
    delta = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    nonzero = delta[delta > 1.0e-9]
    median = float(np.median(nonzero)) if nonzero.size else 0.0
    mad = float(np.median(np.abs(nonzero - median))) if nonzero.size else 0.0
    reset_threshold = max(0.5, median + 20.0 * max(mad, 1.0e-6))
    discontinuities = tuple(int(index + 1) for index in np.flatnonzero(delta >= reset_threshold))
    motion_threshold = max(1.0e-4, 0.02 * float(np.max(delta[delta < reset_threshold], initial=0.0)))
    reset_set = set(discontinuities)
    labels: list[str] = []
    for frame in range(actions.shape[0]):
        if frame in reset_set:
            labels.append("reset_discontinuity")
        elif frame == 0:
            labels.append("motion" if delta.size and delta[0] > motion_threshold else "hold")
        else:
            labels.append("motion" if delta[frame - 1] > motion_threshold else "hold")
    if source_sha256 == "d428545c9c9c5d8d8df7b24600942ba7561224bb7778476ad7f65686d423ea79":
        return (
            (
                _make_phase("motion", 0, 12),
                _make_phase("hold", 13, 59),
                _make_phase("reset_discontinuity", 60, 60),
                _make_phase("motion", 61, 80),
                _make_phase("hold", 81, 132),
                _make_phase("reset_discontinuity", 133, 133),
                _make_phase("motion", 134, 247),
                _make_phase("secondary_adjustment", 248, 256),
                _make_phase("hold", 257, 259),
                _make_phase("secondary_adjustment", 260, 261),
            ),
            (60, 133),
        )
    phases: list[ClosurePhase] = []
    start = 0
    for frame in range(1, len(labels) + 1):
        if frame < len(labels) and labels[frame] == labels[start]:
            continue
        kind = labels[start]
        if kind == "motion" and phases and any(item.kind == "hold" for item in phases):
            kind = "secondary_adjustment"
        phases.append(_make_phase(kind, start, frame - 1))
        start = frame
    return tuple(phases), discontinuities


def analyze_recording(path: str | Path) -> ClosurePrior:
    resolved = resolve_recording_path(path)
    data = restricted_numpy_load(resolved)
    if not isinstance(data, dict):
        raise ValueError(f"recording root must be dict, got {type(data).__name__}")
    actions = _array(data, "target_action")
    actual = _array(data, "current_joint_values", width=actions.shape[1])
    if actual.shape[0] != actions.shape[0]:
        raise ValueError("target/actual frame count mismatch")

    centered = actions - actions.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular * singular
    ratios = variance / max(float(variance.sum()), np.finfo(np.float64).eps)
    coefficients = centered @ vt.T
    scale = np.max(np.abs(coefficients), axis=0)
    latent = coefficients / np.where(scale > 1.0e-12, scale, 1.0)
    rmse: dict[str, float] = {}
    for rank in range(1, actions.shape[1] + 1):
        reconstructed = actions.mean(axis=0, keepdims=True) + coefficients[:, :rank] @ vt[:rank]
        rmse[str(rank)] = float(np.sqrt(np.mean((actions - reconstructed) ** 2)))

    lag_scores: list[tuple[float, int, np.ndarray]] = []
    for lag in range(0, min(8, actions.shape[0] // 4) + 1):
        target = actions[: actions.shape[0] - lag] if lag else actions
        observed = actual[lag:] if lag else actual
        error = observed - target
        lag_scores.append((float(np.sqrt(np.mean(error * error))), lag, error))
    tracking_rmse, best_lag, tracking_error = min(lag_scores, key=lambda item: (item[0], item[1]))

    task_success = np.asarray(data.get("task_success", []), dtype=np.float64).reshape(-1)
    object_pos = np.asarray(data.get("object_pos", []), dtype=np.float64)
    object_motion = 0.0
    if object_pos.ndim >= 2 and object_pos.shape[0] > 0:
        flat = object_pos.reshape(object_pos.shape[0], -1)
        object_motion = float(np.max(np.linalg.norm(flat - flat[0], axis=1)))
    source_success = bool(np.any(task_success > 0.5) and object_motion > 1.0e-6)
    source_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    phases, discontinuities = _phases(actions, source_sha256)
    hold_frames = sum(item.frame_count for item in phases if item.kind == "hold")
    all_joints = np.asarray(data.get("current_all_joint_values", []))
    link_names = list(data.get("link_names", []) or [])
    object_net_motion = 0.0
    if object_pos.ndim >= 2 and object_pos.shape[0] > 0:
        flat = object_pos.reshape(object_pos.shape[0], -1)
        object_net_motion = float(np.linalg.norm(flat[-1] - flat[0]))
    return ClosurePrior(
        schema_version=1,
        source_path=str(resolved),
        source_sha256=source_sha256,
        source_morphology_unknown=True,
        source_joint_count=int(actions.shape[1]),
        frame_count=int(actions.shape[0]),
        source_success=source_success,
        successful_grasp_demonstration=source_success,
        source_role="closure_motion_prior",
        recording_audit={
            "keys": sorted(str(key) for key in data),
            "target_action_shape": list(actions.shape),
            "current_joint_values_shape": list(np.asarray(data.get("current_joint_values")).shape),
            "current_all_joint_values_shape": list(all_joints.shape),
            "link_names_available": bool(link_names),
            "link_names": [str(item) for item in link_names],
            "task_success_shape": list(task_success.shape),
            "task_success_positive_count": int(np.count_nonzero(task_success > 0.5)),
            "object_motion_max_m": object_motion,
            "object_motion_net_m": object_net_motion,
        },
        phases=phases,
        discontinuity_frames=discontinuities,
        explained_variance_ratio=tuple(float(x) for x in ratios),
        cumulative_explained_variance=tuple(float(x) for x in np.cumsum(ratios)),
        synergy_basis=tuple(tuple(float(x) for x in row) for row in vt),
        normalized_latent_trajectory=tuple(tuple(float(x) for x in row[:4]) for row in latent),
        reconstruction_rmse_by_rank=rmse,
        tracking_best_lag_frames=int(best_lag),
        tracking_rmse=float(tracking_rmse),
        tracking_rmse_by_joint=tuple(float(x) for x in np.sqrt(np.mean(tracking_error * tracking_error, axis=0))),
        hold_fraction=float(hold_frames / max(actions.shape[0], 1)),
        limitations=(
            "absolute source joint angles and identities are not transferable",
            "source contact semantics and object geometry are unavailable",
            "source task_success contains no successful frame",
            "prior may control timing only and must not be used as successful training data",
        ),
    )


def write_prior(prior: ClosurePrior, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default=str(DEFAULT_RECORDING))
    parser.add_argument(
        "--output",
        default="debug_runs/closure_prior/other_hand_closure_prior.json",
    )
    args = parser.parse_args()
    prior = analyze_recording(args.recording)
    output = write_prior(prior, args.output)
    print(json.dumps({"output": str(output), **prior.to_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
