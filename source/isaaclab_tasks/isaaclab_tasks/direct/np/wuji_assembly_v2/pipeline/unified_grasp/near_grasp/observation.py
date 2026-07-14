"""Named observation contract for the dedicated near-grasp environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NamedObservationSchema:
    names_and_widths: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.names_and_widths]
        if len(names) != len(set(names)):
            raise ValueError("Observation names must be unique")
        if any(int(width) <= 0 for _, width in self.names_and_widths):
            raise ValueError("Observation slice widths must be positive")

    @property
    def slices(self) -> dict[str, slice]:
        cursor = 0
        out: dict[str, slice] = {}
        for name, width in self.names_and_widths:
            out[name] = slice(cursor, cursor + int(width))
            cursor += int(width)
        return out

    @property
    def width(self) -> int:
        return sum(int(width) for _, width in self.names_and_widths)

    def pack(self, fields: Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
        unknown = set(fields) - set(self.slices)
        missing = set(self.slices) - set(fields)
        if unknown or missing:
            raise ValueError(f"Observation fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        arrays: list[np.ndarray] = []
        batch_size: int | None = None
        for name, width in self.names_and_widths:
            array = np.asarray(fields[name], dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != width:
                raise ValueError(f"Observation field {name!r} expected width {width}, got {array.shape}")
            batch_size = array.shape[0] if batch_size is None else batch_size
            if array.shape[0] != batch_size:
                raise ValueError("All observation fields must have the same batch dimension")
            arrays.append(array)
        return np.concatenate(arrays, axis=1)

    def unpack(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        array = np.asarray(observation)
        if array.shape[-1] != self.width:
            raise ValueError(f"Expected observation width {self.width}, got {array.shape[-1]}")
        return {name: array[..., value] for name, value in self.slices.items()}


NEAR_GRASP_OBSERVATION_SCHEMA = NamedObservationSchema(
    (
        ("object_palm_relative_pose", 7),
        ("object_velocity", 6),
        ("fingertip_relative_positions", 15),
        ("hand_q", 20),
        ("hand_qdot", 20),
        ("hand_target_error", 20),
        ("wrist_q", 6),
        ("wrist_qdot", 6),
        ("wrist_target_error", 6),
        ("target_force_norms", 5),
        ("target_force_vectors", 15),
        ("contact_history_duty", 10),
        ("table_state", 4),
        ("object_hand_transform", 7),
        ("object_embedding", 8),
        ("previous_action", 14),
    )
)
