"""Morphology-correct Wuji hand priors behind a common 6D program interface."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .grasp_program import WUJI_HAND_JOINT_NAMES


PROGRAM_LATENT_DIM = 6
NATIVE_COORDEX_LATENT_DIM = 12
COORDEX_PROPRIO_DIM = 66
HAND_DIM = 20
COORDEX_PROPRIO_FIELDS = (
    ("palm_linear_velocity_body", 3),
    ("palm_angular_velocity_body", 3),
    ("joint_position_default_relative", 20),
    ("joint_velocity_default_relative", 20),
    ("previous_joint_action", 20),
)


class HandPriorAdapter(ABC):
    """Decode a 6D program latent into 20 Wuji joint position targets."""

    native_latent_dim: int
    program_latent_dim: int = PROGRAM_LATENT_DIM

    @abstractmethod
    def reset(self, proprio: np.ndarray, q: np.ndarray) -> None:
        """Reset state from current proprioception and Wuji joint positions."""

    @abstractmethod
    def decode(self, proprio: np.ndarray, latent6: np.ndarray, progress: float | np.ndarray) -> np.ndarray:
        """Return 20D joint targets for one or more observations."""

    def decode_torch(self, proprio, latent6, progress):
        """Optionally decode tensors without a device round-trip."""

        return None

    @property
    @abstractmethod
    def signature(self) -> dict[str, Any]:
        """Return an immutable compatibility signature."""


@dataclass(frozen=True)
class RetargetedPcaArtifact:
    joint_names: tuple[str, ...]
    progress_grid: tuple[float, ...]
    mean_trajectory: tuple[tuple[float, ...], ...]
    components6x20: tuple[tuple[float, ...], ...]
    coefficient_scale6: tuple[float, ...]
    explained_variance_ratio6: tuple[float, ...]
    source: str
    source_hash: str

    def __post_init__(self) -> None:
        if len(self.joint_names) != HAND_DIM or len(self.components6x20) != PROGRAM_LATENT_DIM:
            raise ValueError("PCA artifact has the wrong Wuji dimensions")
        if len(self.progress_grid) != len(self.mean_trajectory) or len(self.progress_grid) < 2:
            raise ValueError("PCA artifact requires a multi-step mean trajectory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "joint_names": list(self.joint_names),
            "progress_grid": list(self.progress_grid),
            "mean_trajectory": [list(row) for row in self.mean_trajectory],
            "components6x20": [list(row) for row in self.components6x20],
            "coefficient_scale6": list(self.coefficient_scale6),
            "explained_variance_ratio6": list(self.explained_variance_ratio6),
            "source": self.source,
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetargetedPcaArtifact":
        return cls(
            joint_names=tuple(str(value) for value in payload["joint_names"]),
            progress_grid=tuple(float(value) for value in payload["progress_grid"]),
            mean_trajectory=tuple(tuple(float(value) for value in row) for row in payload["mean_trajectory"]),
            components6x20=tuple(tuple(float(value) for value in row) for row in payload["components6x20"]),
            coefficient_scale6=tuple(float(value) for value in payload["coefficient_scale6"]),
            explained_variance_ratio6=tuple(float(value) for value in payload["explained_variance_ratio6"]),
            source=str(payload["source"]),
            source_hash=str(payload["source_hash"]),
        )


class RetargetedPcaPriorAdapter(HandPriorAdapter):
    """Mandatory deterministic PCA fallback from official Wuji trajectories."""

    native_latent_dim = PROGRAM_LATENT_DIM

    def __init__(self, artifact: RetargetedPcaArtifact, *, joint_lower=None, joint_upper=None):
        if tuple(artifact.joint_names) != WUJI_HAND_JOINT_NAMES:
            raise ValueError("Retargeted PCA joint order does not match the project Wuji order")
        self.artifact = artifact
        self._grid = np.asarray(artifact.progress_grid, dtype=np.float64)
        self._mean = np.asarray(artifact.mean_trajectory, dtype=np.float64)
        self._components = np.asarray(artifact.components6x20, dtype=np.float64)
        self._scale = np.asarray(artifact.coefficient_scale6, dtype=np.float64)
        self._lower = _limit_array(joint_lower, -np.inf)
        self._upper = _limit_array(joint_upper, np.inf)
        self._q0: np.ndarray | None = None
        self._torch_cache: dict[tuple[str, str], tuple[Any, ...]] = {}

    def reset(self, proprio: np.ndarray, q: np.ndarray) -> None:
        _validate_last_dim(proprio, None, "proprio")
        q_array = _validate_last_dim(q, HAND_DIM, "q")
        self._q0 = q_array.copy()
        self._torch_cache.clear()

    def decode(self, proprio: np.ndarray, latent6: np.ndarray, progress: float | np.ndarray) -> np.ndarray:
        del proprio
        if self._q0 is None:
            raise RuntimeError("RetargetedPcaPriorAdapter.reset must be called before decode")
        latent = _validate_last_dim(latent6, PROGRAM_LATENT_DIM, "latent6")
        single = latent.ndim == 1
        latent = latent.reshape(-1, PROGRAM_LATENT_DIM)
        progress_array = np.asarray(progress, dtype=np.float64)
        if progress_array.ndim == 0:
            progress_array = np.full((latent.shape[0],), float(progress_array))
        progress_array = np.broadcast_to(progress_array.reshape(-1), (latent.shape[0],))
        progress_array = np.clip(progress_array, 0.0, 1.0)
        nominal = np.stack(
            [np.asarray([np.interp(p, self._grid, self._mean[:, joint]) for joint in range(HAND_DIM)]) for p in progress_array]
        )
        residual = (latent * self._scale.reshape(1, -1)) @ self._components
        smooth = progress_array * progress_array * (3.0 - 2.0 * progress_array)
        residual *= smooth.reshape(-1, 1)
        target = np.clip(nominal + residual, self._lower, self._upper)
        return target[0] if single else target

    def decode_torch(self, proprio, latent6, progress):
        del proprio
        if self._q0 is None:
            raise RuntimeError("RetargetedPcaPriorAdapter.reset must be called before decode")
        import torch

        if not torch.is_tensor(latent6) or latent6.ndim != 2 or latent6.shape[1] != PROGRAM_LATENT_DIM:
            raise ValueError("latent6 must be a [N,6] tensor")
        key = (str(latent6.device), str(latent6.dtype))
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = tuple(
                torch.as_tensor(value, dtype=latent6.dtype, device=latent6.device)
                for value in (self._grid, self._mean, self._components, self._scale, self._lower, self._upper)
            )
            self._torch_cache[key] = cached
        grid, mean, components, scale, lower, upper = cached
        progress_tensor = torch.as_tensor(progress, dtype=latent6.dtype, device=latent6.device).reshape(-1)
        progress_tensor = torch.clamp(progress_tensor, 0.0, 1.0)
        upper_index = torch.searchsorted(grid, progress_tensor, right=True).clamp(1, grid.numel() - 1)
        lower_index = upper_index - 1
        alpha = (progress_tensor - grid[lower_index]) / torch.clamp(
            grid[upper_index] - grid[lower_index], min=torch.finfo(latent6.dtype).eps
        )
        nominal = torch.lerp(mean[lower_index], mean[upper_index], alpha.reshape(-1, 1))
        residual = (latent6 * scale.reshape(1, -1)) @ components
        smooth = progress_tensor * progress_tensor * (3.0 - 2.0 * progress_tensor)
        residual *= smooth.reshape(-1, 1)
        return torch.maximum(torch.minimum(nominal + residual, upper), lower)

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "native_latent_dim": self.native_latent_dim,
            "program_latent_dim": self.program_latent_dim,
            "joint_names": list(WUJI_HAND_JOINT_NAMES),
            "artifact_source": self.artifact.source,
            "artifact_hash": self.artifact.source_hash,
        }

    @classmethod
    def fit(
        cls,
        trajectories: np.ndarray,
        *,
        source: str,
        joint_names: Sequence[str] = WUJI_HAND_JOINT_NAMES,
    ) -> RetargetedPcaArtifact:
        """Fit a sign-stable PCA6 artifact from N x T x 20 Wuji trajectories."""

        values = np.asarray(trajectories, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 2 or values.shape[2] != HAND_DIM:
            raise ValueError("trajectories must have shape [N>=2, T>=2, 20]")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectories contain non-finite values")
        mean_trajectory = np.mean(values, axis=0)
        centered = (values - mean_trajectory.reshape(1, values.shape[1], HAND_DIM)).reshape(-1, HAND_DIM)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:PROGRAM_LATENT_DIM].copy()
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                row *= -1.0
        variance = singular_values * singular_values
        ratio = variance[:PROGRAM_LATENT_DIM] / max(float(np.sum(variance)), np.finfo(np.float64).eps)
        coefficient_scale = np.maximum(
            singular_values[:PROGRAM_LATENT_DIM] / np.sqrt(max(centered.shape[0] - 1, 1)),
            1.0e-4,
        )
        source_hash = hashlib.sha256(values.tobytes() + str(source).encode("utf-8")).hexdigest()
        return RetargetedPcaArtifact(
            joint_names=tuple(str(name) for name in joint_names),
            progress_grid=tuple(float(value) for value in np.linspace(0.0, 1.0, values.shape[1])),
            mean_trajectory=tuple(tuple(float(value) for value in row) for row in mean_trajectory),
            components6x20=tuple(tuple(float(value) for value in row) for row in components),
            coefficient_scale6=tuple(float(value) for value in coefficient_scale),
            explained_variance_ratio6=tuple(float(value) for value in ratio),
            source=str(source),
            source_hash=source_hash,
        )

    @classmethod
    def build_runtime_fallback(
        cls,
        preshape_q: Sequence[float],
        close_q: Sequence[float],
        *,
        steps: int = 96,
    ) -> RetargetedPcaArtifact:
        """Build deterministic morphology-correct pinch/straddle trajectories.

        This fallback consumes only Wuji runtime references.  The external
        wuji-retargeting smoke is recorded separately; other-hand recording
        angles and success labels are never consumed.
        """

        preshape = _validate_last_dim(np.asarray(preshape_q), HAND_DIM, "preshape_q")
        close = _validate_last_dim(np.asarray(close_q), HAND_DIM, "close_q")
        progress = np.linspace(0.0, 1.0, int(steps))
        smooth = progress * progress * (3.0 - 2.0 * progress)
        masks = []
        for active_fingers in ((3, 4), (2, 3), (2, 3, 4), (1, 2, 3, 4, 5), (3,), (4,), (2, 4), (1, 3, 5)):
            mask = np.zeros(HAND_DIM, dtype=np.float64)
            for joint_index, name in enumerate(WUJI_HAND_JOINT_NAMES):
                finger = int(name.split("finger", 1)[1].split("_", 1)[0])
                if finger in active_fingers:
                    mask[joint_index] = 1.0
            masks.append(mask)
        trajectories = []
        delta = close - preshape
        for index, mask in enumerate(masks):
            phase = np.clip(smooth + 0.04 * np.sin((index + 1) * np.pi * progress) * progress * (1.0 - progress), 0.0, 1.0)
            trajectories.append(preshape.reshape(1, -1) + phase.reshape(-1, 1) * delta.reshape(1, -1) * mask)
        return cls.fit(
            np.asarray(trajectories),
            source="official_wuji_runtime_references_pca6_fallback",
        )


class CoorDexWujiPriorAdapter(HandPriorAdapter):
    """Read the isolated CoorDex checkpoint without importing its package."""

    native_latent_dim = NATIVE_COORDEX_LATENT_DIM

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        joint_names: Sequence[str] = WUJI_HAND_JOINT_NAMES,
        action_scale: float = 0.1,
        device: str = "cpu",
        joint_lower=None,
        joint_upper=None,
        default_q=None,
    ):
        if tuple(str(name) for name in joint_names) != WUJI_HAND_JOINT_NAMES:
            raise ValueError("CoorDex joint order mismatch")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.action_scale = float(action_scale)
        self.device_name = str(device)
        self._lower = _limit_array(joint_lower, -np.inf)
        self._upper = _limit_array(joint_upper, np.inf)
        self._default_q = None if default_q is None else _validate_last_dim(np.asarray(default_q), HAND_DIM, "default_q")
        self._torch, self._prior, self._decoder, self._obs_mean, self._obs_var = self._load_checkpoint()
        self._basis12x6 = self._build_response_basis()
        self._q0: np.ndarray | None = None
        self._prior_mean: np.ndarray | None = None
        self._torch_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
        self._checkpoint_sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()

    def _load_checkpoint(self):
        import torch

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise ValueError("CoorDex checkpoint is missing model_state_dict")
        prior = _sequential_from_state(torch, state, "prior_net", COORDEX_PROPRIO_DIM, 24, "elu")
        decoder = _sequential_from_state(
            torch,
            state,
            "decoder",
            COORDEX_PROPRIO_DIM + NATIVE_COORDEX_LATENT_DIM,
            HAND_DIM,
            "elu",
        )
        norm = payload.get("obs_norm_state_dict", {})
        mean = norm.get("_mean")
        var = norm.get("_var")
        if not torch.is_tensor(mean) or not torch.is_tensor(var):
            raise ValueError("CoorDex checkpoint is missing observation normalization")
        mean = mean.reshape(-1).to(dtype=torch.float32)
        var = var.reshape(-1).to(dtype=torch.float32)
        if mean.numel() < COORDEX_PROPRIO_DIM or var.numel() < COORDEX_PROPRIO_DIM:
            raise ValueError("CoorDex checkpoint normalization dimension mismatch")
        # The kinematic-wrist checkpoint stores full-policy normalization
        # (currently 201D); CoorDex's own loader selects the trailing 66D hand
        # proprio block for this prior variant.
        mean = mean[-COORDEX_PROPRIO_DIM:]
        var = var[-COORDEX_PROPRIO_DIM:]
        prior.to(self.device_name).eval()
        decoder.to(self.device_name).eval()
        return torch, prior, decoder, mean.to(self.device_name), var.to(self.device_name)

    def _normalize(self, proprio: np.ndarray):
        tensor = self._torch.as_tensor(proprio, dtype=self._torch.float32, device=self.device_name)
        return self._torch.clamp((tensor - self._obs_mean) / self._torch.sqrt(self._obs_var + 1.0e-8), -10.0, 10.0)

    def _build_response_basis(self) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            proprio = self._normalize(np.zeros((1, COORDEX_PROPRIO_DIM), dtype=np.float32))
            native = torch.zeros((1, NATIVE_COORDEX_LATENT_DIM), dtype=torch.float32, device=self.device_name)
            columns = []
            epsilon = 0.25
            for axis in range(NATIVE_COORDEX_LATENT_DIM):
                plus = native.clone()
                minus = native.clone()
                plus[:, axis] += epsilon
                minus[:, axis] -= epsilon
                response = (self._decoder(torch.cat((proprio, plus), dim=1)) - self._decoder(torch.cat((proprio, minus), dim=1))) / (2.0 * epsilon)
                columns.append(response[0].cpu().numpy())
        jacobian = np.stack(columns, axis=1)
        if jacobian.shape != (HAND_DIM, NATIVE_COORDEX_LATENT_DIM) or not np.all(np.isfinite(jacobian)):
            raise ValueError("CoorDex decoder response basis is invalid")
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        if singular.shape[0] < PROGRAM_LATENT_DIM or singular[PROGRAM_LATENT_DIM - 1] <= 1.0e-8:
            raise ValueError("CoorDex decoder response basis is rank deficient")
        basis = vt[:PROGRAM_LATENT_DIM].T
        for column in range(basis.shape[1]):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0.0:
                basis[:, column] *= -1.0
        return basis

    def reset(self, proprio: np.ndarray, q: np.ndarray) -> None:
        proprio_array = _validate_last_dim(proprio, COORDEX_PROPRIO_DIM, "proprio")
        q_array = _validate_last_dim(q, HAND_DIM, "q")
        single = proprio_array.ndim == 1
        normalized = self._normalize(proprio_array.reshape(-1, COORDEX_PROPRIO_DIM))
        with self._torch.no_grad():
            prior_output = self._prior(normalized)
        if tuple(prior_output.shape[1:]) != (24,):
            raise ValueError("CoorDex prior output dimension mismatch")
        self._prior_mean = prior_output[:, :NATIVE_COORDEX_LATENT_DIM].cpu().numpy()
        self._q0 = q_array.reshape(-1, HAND_DIM).copy()
        if self._default_q is None:
            self._default_q = self._q0[0].copy()
        self._torch_cache.clear()
        if self._q0.shape[0] not in {1, self._prior_mean.shape[0]}:
            raise ValueError("CoorDex q/proprio batch mismatch")
        if single:
            self._prior_mean = self._prior_mean[:1]

    def decode(self, proprio: np.ndarray, latent6: np.ndarray, progress: float | np.ndarray) -> np.ndarray:
        if self._q0 is None or self._prior_mean is None:
            raise RuntimeError("CoorDexWujiPriorAdapter.reset must be called before decode")
        proprio_array = _validate_last_dim(proprio, COORDEX_PROPRIO_DIM, "proprio")
        latent = _validate_last_dim(latent6, PROGRAM_LATENT_DIM, "latent6")
        single = latent.ndim == 1
        latent = latent.reshape(-1, PROGRAM_LATENT_DIM)
        normalized = self._normalize(proprio_array.reshape(-1, COORDEX_PROPRIO_DIM))
        if normalized.shape[0] != latent.shape[0]:
            raise ValueError("CoorDex proprio/latent batch mismatch")
        progress_array = np.asarray(progress, dtype=np.float64)
        if progress_array.ndim == 0:
            progress_array = np.full((latent.shape[0],), float(progress_array))
        progress_array = np.broadcast_to(progress_array.reshape(-1), (latent.shape[0],))
        progress_array = np.clip(progress_array, 0.0, 1.0)
        prior_mean = np.broadcast_to(self._prior_mean, (latent.shape[0], NATIVE_COORDEX_LATENT_DIM))
        native = prior_mean + (latent @ self._basis12x6.T) * progress_array.reshape(-1, 1)
        with self._torch.no_grad():
            decoded = self._decoder(
                self._torch.cat(
                    (normalized, self._torch.as_tensor(native, dtype=self._torch.float32, device=self.device_name)),
                    dim=1,
                )
            ).cpu().numpy()
        q0 = np.broadcast_to(self._q0, (latent.shape[0], HAND_DIM))
        endpoint = np.asarray(self._default_q).reshape(1, -1) + self.action_scale * decoded
        target = np.clip(q0 + progress_array.reshape(-1, 1) * (endpoint - q0), self._lower, self._upper)
        return target[0] if single else target

    def decode_torch(self, proprio, latent6, progress):
        if self._q0 is None or self._prior_mean is None:
            raise RuntimeError("CoorDexWujiPriorAdapter.reset must be called before decode")
        torch = self._torch
        if not torch.is_tensor(proprio) or proprio.ndim != 2 or proprio.shape[1] != COORDEX_PROPRIO_DIM:
            raise ValueError("proprio must be a [N,66] tensor")
        if not torch.is_tensor(latent6) or latent6.ndim != 2 or latent6.shape[1] != PROGRAM_LATENT_DIM:
            raise ValueError("latent6 must be a [N,6] tensor")
        proprio = proprio.to(device=self.device_name, dtype=torch.float32)
        latent6 = latent6.to(device=self.device_name, dtype=torch.float32)
        key = (str(latent6.device), str(latent6.dtype))
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = tuple(
                torch.as_tensor(value, dtype=latent6.dtype, device=latent6.device)
                for value in (self._basis12x6, self._prior_mean, self._q0, self._default_q, self._lower, self._upper)
            )
            self._torch_cache[key] = cached
        basis, prior_mean, q0, default_q, lower, upper = cached
        normalized = torch.clamp(
            (proprio - self._obs_mean) / torch.sqrt(self._obs_var + 1.0e-8),
            -10.0,
            10.0,
        )
        progress_tensor = torch.as_tensor(progress, dtype=latent6.dtype, device=latent6.device).reshape(-1)
        progress_tensor = torch.clamp(progress_tensor, 0.0, 1.0)
        prior_mean = prior_mean.expand(latent6.shape[0], -1)
        native = prior_mean + (latent6 @ basis.T) * progress_tensor.reshape(-1, 1)
        with torch.no_grad():
            decoded = self._decoder(torch.cat((normalized, native), dim=1))
        q0 = q0.expand(latent6.shape[0], -1)
        endpoint = default_q.reshape(1, -1) + self.action_scale * decoded
        target = q0 + progress_tensor.reshape(-1, 1) * (endpoint - q0)
        return torch.maximum(torch.minimum(target, upper), lower)

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "native_latent_dim": self.native_latent_dim,
            "program_latent_dim": self.program_latent_dim,
            "proprio_dim": COORDEX_PROPRIO_DIM,
            "output_dim": HAND_DIM,
            "joint_names": list(WUJI_HAND_JOINT_NAMES),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "redistribution_allowed": False,
            "basis_sha256": hashlib.sha256(self._basis12x6.tobytes()).hexdigest(),
            "proprio_fields": [{"name": name, "width": width} for name, width in COORDEX_PROPRIO_FIELDS],
            "target_semantics": "default_joint_position_plus_0.1_times_decoder_action",
        }


def build_coordex_proprio(
    *,
    palm_linear_velocity_body: np.ndarray,
    palm_angular_velocity_body: np.ndarray,
    joint_position: np.ndarray,
    default_joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    default_joint_velocity: np.ndarray,
    previous_joint_action: np.ndarray,
) -> np.ndarray:
    """Build the official CoorDex floating-kinematic-wrist 66D observation."""

    values = [
        np.asarray(palm_linear_velocity_body, dtype=np.float64),
        np.asarray(palm_angular_velocity_body, dtype=np.float64),
        np.asarray(joint_position, dtype=np.float64) - np.asarray(default_joint_position, dtype=np.float64),
        np.asarray(joint_velocity, dtype=np.float64) - np.asarray(default_joint_velocity, dtype=np.float64),
        np.asarray(previous_joint_action, dtype=np.float64),
    ]
    result = np.concatenate(values, axis=-1)
    if result.shape[-1] != COORDEX_PROPRIO_DIM or not np.all(np.isfinite(result)):
        raise ValueError("CoorDex proprio fields do not form a finite 66D vector")
    return result


def save_pca_artifact(artifact: RetargetedPcaArtifact, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_pca_artifact(path: str | Path) -> RetargetedPcaArtifact:
    return RetargetedPcaArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sequential_from_state(torch, state: dict[str, Any], prefix: str, input_dim: int, output_dim: int, activation: str):
    rows = []
    index = 0
    current_dim = int(input_dim)
    while f"{prefix}.{index}.weight" in state:
        weight = state[f"{prefix}.{index}.weight"]
        bias = state.get(f"{prefix}.{index}.bias")
        if weight.ndim != 2 or weight.shape[1] != current_dim or bias is None or bias.shape[0] != weight.shape[0]:
            raise ValueError(f"CoorDex {prefix} layer {index} has an incompatible shape")
        layer = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        rows.append(layer)
        current_dim = int(weight.shape[0])
        next_index = index + 2
        if f"{prefix}.{next_index}.weight" in state:
            rows.append(torch.nn.ELU() if activation == "elu" else torch.nn.Tanh())
        index = next_index
    if not rows or current_dim != int(output_dim):
        raise ValueError(f"CoorDex {prefix} output dimension mismatch: {current_dim} != {output_dim}")
    return torch.nn.Sequential(*rows)


def _validate_last_dim(value: np.ndarray, width: int | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in {1, 2} or (width is not None and array.shape[-1] != width):
        expected = "any" if width is None else str(width)
        raise ValueError(f"{name} must be one- or two-dimensional with last dimension {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _limit_array(value, default: float) -> np.ndarray:
    if value is None:
        return np.full((HAND_DIM,), default, dtype=np.float64)
    return _validate_last_dim(np.asarray(value), HAND_DIM, "joint limit").reshape(HAND_DIM)
