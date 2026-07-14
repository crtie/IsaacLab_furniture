"""Contact attribution and strict milestone evaluation for Frame Stage 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import DeliveryLevel, FrameStage4Failure, FrameStage4Result


@dataclass(frozen=True)
class FrameContactSummary:
    support_fingers: tuple[int, ...]
    support_links: tuple[str, ...]
    hand_frame_contact: bool
    frame_fixed_contact: bool
    frame_table_contact: bool
    frame_ground_contact: bool
    illegal_contact: bool
    illegal_pairs: tuple[tuple[str, str], ...]
    self_contact: bool
    unresolved: bool


def attribute_frame_contacts(events: Sequence[Any], *, report_available: bool = True) -> FrameContactSummary:
    fingers: set[int] = set()
    links: set[str] = set()
    illegal: list[tuple[str, str]] = []
    hand_frame = False
    frame_fixed = False
    frame_table = False
    frame_ground = False
    self_contact = False
    for event in events:
        first = str(getattr(event, "actor0", ""))
        second = str(getattr(event, "actor1", ""))
        pair = (first, second)
        robot_first = "/Robot/" in first
        robot_second = "/Robot/" in second
        frame_first = "/Frame" in first
        frame_second = "/Frame" in second
        fixed_first = "/FixedAsset" in first
        fixed_second = "/FixedAsset" in second
        table_first = "/Table" in first
        table_second = "/Table" in second
        ground_first = "/ground" in first
        ground_second = "/ground" in second

        if robot_first and robot_second:
            self_contact = True
            illegal.append(pair)
        if (frame_first and fixed_second) or (frame_second and fixed_first):
            frame_fixed = True
        if (frame_first and table_second) or (frame_second and table_first):
            frame_table = True
        if (frame_first and ground_second) or (frame_second and ground_first):
            frame_ground = True
        if (robot_first or robot_second) and (fixed_first or fixed_second or table_first or table_second or ground_first or ground_second):
            illegal.append(pair)
        if (robot_first and frame_second) or (robot_second and frame_first):
            hand_frame = True
            robot_path = first if robot_first else second
            accepted = False
            for finger in (2, 3, 4):
                for marker in (
                    f"right_finger{finger}_link3",
                    f"right_finger{finger}_link4",
                    f"right_finger{finger}_tip_link",
                ):
                    if marker in robot_path:
                        fingers.add(finger)
                        links.add(marker)
                        accepted = True
            if not accepted:
                illegal.append(pair)
    return FrameContactSummary(
        support_fingers=tuple(sorted(fingers)),
        support_links=tuple(sorted(links)),
        hand_frame_contact=hand_frame,
        frame_fixed_contact=frame_fixed,
        frame_table_contact=frame_table,
        frame_ground_contact=frame_ground,
        illegal_contact=bool(illegal),
        illegal_pairs=tuple(illegal),
        self_contact=self_contact,
        unresolved=not bool(report_available),
    )


class FrameStage4Evaluator:
    def __init__(self) -> None:
        self.initial_frame_z: float | None = None
        self.max_lift_m = 0.0
        self.max_relative_position_drift_m = 0.0
        self.max_relative_rotation_drift_deg = 0.0
        self.peak_force_n = 0.0
        self.physics_frames = 0

    def update(
        self,
        snapshot: Mapping[str, Any],
        *,
        relative_reference: np.ndarray | None,
    ) -> None:
        frame_position = np.asarray(snapshot["frame_position"], dtype=np.float64)
        if self.initial_frame_z is None:
            self.initial_frame_z = float(frame_position[2])
        self.max_lift_m = max(self.max_lift_m, float(frame_position[2]) - self.initial_frame_z)
        forces = np.asarray(snapshot.get("support_force_n", ()), dtype=np.float64)
        self.peak_force_n = max(self.peak_force_n, float(np.max(forces, initial=0.0)))
        self.physics_frames += 1
        if relative_reference is not None:
            current = inverse_transform(snapshot["palm_transform"]) @ np.asarray(snapshot["frame_transform"])
            self.max_relative_position_drift_m = max(
                self.max_relative_position_drift_m,
                float(np.linalg.norm(current[:3, 3] - relative_reference[:3, 3])),
            )
            self.max_relative_rotation_drift_deg = max(
                self.max_relative_rotation_drift_deg,
                rotation_error_deg(current[:3, :3], relative_reference[:3, :3]),
            )

    def finalize(
        self,
        *,
        physical_support_acquired: bool,
        physical_lift_success: bool,
        preinsert_success: bool,
        physical_insert_success: bool,
        release_stable: bool,
        failure: FrameStage4Failure,
        write_audit: Mapping[str, int],
        trace_path: str,
        video_path: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FrameStage4Result:
        if release_stable:
            level = DeliveryLevel.A
        elif physical_insert_success:
            level = DeliveryLevel.B
        elif physical_support_acquired and physical_lift_success and preinsert_success:
            level = DeliveryLevel.C
        else:
            level = DeliveryLevel.NONE
        return FrameStage4Result(
            delivery_level=level,
            physical_support_acquired=bool(physical_support_acquired),
            physical_lift_success=bool(physical_lift_success),
            preinsert_success=bool(preinsert_success),
            physical_insert_success=bool(physical_insert_success),
            release_stable=bool(release_stable),
            failure=failure,
            physics_frames=self.physics_frames,
            frame_lift_m=self.max_lift_m,
            max_relative_position_drift_m=self.max_relative_position_drift_m,
            max_relative_rotation_drift_deg=self.max_relative_rotation_drift_deg,
            peak_force_n=self.peak_force_n,
            post_reset_object_root_writes=int(write_audit.get("post_reset_object_root_writes", 0)),
            post_reset_wrist_state_writes=int(write_audit.get("post_reset_wrist_state_writes", 0)),
            post_reset_fixed_asset_root_writes=int(write_audit.get("post_reset_fixed_asset_root_writes", 0)),
            video_path=str(video_path),
            trace_path=str(trace_path),
            metadata=dict(metadata or {}),
        )


def pose_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    first = np.asarray(current, dtype=np.float64).reshape(4, 4)
    second = np.asarray(target, dtype=np.float64).reshape(4, 4)
    return (
        float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
        rotation_error_deg(first[:3, :3], second[:3, :3]),
    )


def rotation_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    cosine = np.clip((np.trace(np.asarray(current).T @ np.asarray(target)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result
