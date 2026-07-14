"""Pure synchronized trace contracts for Plug2 Gate A forensic runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ForensicContactEvent:
    actor0: str
    actor1: str
    contact_count: int = 0
    positions_world: tuple[tuple[float, float, float], ...] = ()
    normals_world: tuple[tuple[float, float, float], ...] = ()
    separations_m: tuple[float, ...] = ()
    impulses_ns: tuple[tuple[float, float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForensicFrame:
    physics_frame_id: int
    candidate_id: str
    reset_index: int
    stage: str
    applied_target26: tuple[float, ...]
    next_target26: tuple[float, ...]
    actual_joint26: tuple[float, ...]
    joint_velocity26: tuple[float, ...]
    joint_error26: tuple[float, ...]
    isaac_tip_positions: tuple[tuple[float, float, float], ...]
    isaac_tip_quat_wxyz: tuple[tuple[float, float, float, float], ...]
    target_contact_positions: tuple[tuple[float, float, float], ...]
    object_position: tuple[float, float, float]
    object_quat_wxyz: tuple[float, float, float, float]
    object_linear_velocity: tuple[float, float, float]
    object_angular_velocity: tuple[float, float, float]
    target_force_norms: tuple[float, ...]
    target_force_xyz: tuple[tuple[float, float, float], ...]
    all_force_xyz: tuple[tuple[float, float, float], ...]
    table_force_xyz: tuple[tuple[float, float, float], ...]
    ground_force_xyz: tuple[tuple[float, float, float], ...]
    contact_events: tuple[ForensicContactEvent, ...]
    link_contact_sources: Mapping[str, tuple[str, ...]]
    terminal: bool
    termination_layer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.applied_target26) != 26 or len(self.next_target26) != 26:
            raise ValueError("forensic targets must be 26D")
        if len(self.actual_joint26) != 26 or len(self.joint_error26) != 26:
            raise ValueError("forensic joint state must be 26D")
        if len(self.isaac_tip_positions) != 5 or len(self.isaac_tip_quat_wxyz) != 5:
            raise ValueError("forensic frame requires five Isaac fingertip poses")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["contact_events"] = [event.to_dict() for event in self.contact_events]
        row["link_contact_sources"] = {name: list(values) for name, values in self.link_contact_sources.items()}
        return row


def classify_actor_pair(actor0: str, actor1: str) -> str:
    joined = f"{actor0} {actor1}".lower()
    if "robot" in actor0.lower() and "robot" in actor1.lower():
        return "SELF"
    if "targetobject" in joined:
        return "TARGET_OBJECT"
    if "table" in joined:
        return "TABLE"
    if "ground" in joined:
        return "GROUND"
    return "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT"


def link_contact_source_map(
    robot_body_paths: Sequence[str],
    events: Sequence[ForensicContactEvent],
) -> dict[str, tuple[str, ...]]:
    sources: dict[str, set[str]] = {str(path): set() for path in robot_body_paths}
    for event in events:
        source = classify_actor_pair(event.actor0, event.actor1)
        for actor in (event.actor0, event.actor1):
            for path in sources:
                if actor == path or actor.endswith("/" + path.split("/")[-1]):
                    sources[path].add(source)
    return {path: tuple(sorted(values)) if values else ("NO_CONTACT",) for path, values in sources.items()}


def merge_filtered_fingertip_sources(
    link_sources: Mapping[str, Sequence[str]],
    attributions: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Merge same-frame filtered fingertip truth into actor-pair link sources."""

    merged = {str(name): set(str(value) for value in values if value != "NO_CONTACT") for name, values in link_sources.items()}
    for attribution in attributions:
        finger = int(attribution["finger"])
        source = str(attribution["source"])
        if source == "NO_CONTACT":
            continue
        marker = f"right_finger{finger}_link4"
        for name in merged:
            if name == marker or name.endswith("/" + marker):
                merged[name].add(source)
    return {name: tuple(sorted(values)) if values else ("NO_CONTACT",) for name, values in merged.items()}


def sequential_gate_schedule(candidate_ids: Sequence[str], repeats: int = 5) -> tuple[tuple[str, int], ...]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return tuple((str(candidate_id), reset_index) for candidate_id in candidate_ids for reset_index in range(repeats))
