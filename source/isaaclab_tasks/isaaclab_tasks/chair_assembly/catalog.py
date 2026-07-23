"""Pure-Python six-stage, 22-operation chair task catalog."""

from __future__ import annotations

from typing import Iterable

from .models import AssemblyTarget


STAGE_TARGET_COUNTS = (5, 5, 5, 1, 3, 3)

_ROWS = (
    (1, 1, "plug1_to_frame", "Plug1", "linear_insert"),
    (1, 2, "plug2_to_frame", "Plug2", "linear_insert"),
    (1, 3, "backrest_to_frame", "Backrest", "frame"),
    (1, 4, "plug1_to_backrest", "Plug1", "linear_insert"),
    (1, 5, "plug2_to_backrest", "Plug2", "linear_insert"),
    (2, 1, "plug1_to_frame_back", "Plug1", "linear_insert"),
    (2, 2, "plug2_to_frame_back", "Plug2", "linear_insert"),
    (2, 3, "rod_to_frame_back", "Rod", "rod"),
    (2, 4, "plug1_to_rod", "Plug1", "linear_insert"),
    (2, 5, "plug2_to_rod", "Plug2", "linear_insert"),
    (3, 1, "plug1_to_second_rod_side", "Plug1", "linear_insert"),
    (3, 2, "plug2_to_second_rod_side", "Plug2", "linear_insert"),
    (3, 3, "second_rod_to_frame_back", "Rod", "rod"),
    (3, 4, "plug1_to_second_rod", "Plug1", "linear_insert"),
    (3, 5, "plug2_to_second_rod", "Plug2", "linear_insert"),
    (4, 1, "mirror_frame_to_subassembly", "Frame", "frame"),
    (5, 1, "screw1_to_frame_wo_seat", "Screw1", "screw"),
    (5, 2, "screw2_to_frame_wo_seat", "Screw2", "screw"),
    (5, 3, "screw3_to_frame_wo_seat", "Screw3", "screw"),
    (6, 1, "screw1_to_chair_all", "Screw1", "screw"),
    (6, 2, "screw2_to_chair_all", "Screw2", "screw"),
    (6, 3, "screw3_to_chair_all", "Screw3", "screw"),
)


def get_task_catalog(stages: Iterable[int] | None = None) -> tuple[AssemblyTarget, ...]:
    selected = set(range(1, 7) if stages is None else (int(stage) for stage in stages))
    invalid = sorted(selected.difference(range(1, 7)))
    if invalid:
        raise ValueError(f"invalid chair stages {invalid}; expected values in 1..6")
    return tuple(
        AssemblyTarget(stage, target, name, part, insert_type)
        for stage, target, name, part, insert_type in _ROWS
        if stage in selected
    )


def validate_task_catalog() -> None:
    counts = tuple(sum(row[0] == stage for row in _ROWS) for stage in range(1, 7))
    if counts != STAGE_TARGET_COUNTS or len(_ROWS) != 22:
        raise RuntimeError(f"chair catalog mismatch: counts={counts}, total={len(_ROWS)}")
