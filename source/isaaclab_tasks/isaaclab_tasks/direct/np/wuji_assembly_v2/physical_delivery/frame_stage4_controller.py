"""One fixed action-driven trajectory for Frame Stage 4 delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ForkSupportPose,
    FrameStage4Command,
    FrameStage4Failure,
    FrameStage4State,
)
from .frame_fork_support import stage4_goal_transform
from .frame_stage4_evaluator import FrameContactSummary, inverse_transform, pose_error, rotation_error_deg


@dataclass
class ControllerMilestones:
    physical_support_acquired: bool = False
    physical_lift_success: bool = False
    preinsert_success: bool = False
    physical_insert_success: bool = False
    release_stable: bool = False


class FrameStage4Controller:
    def __init__(
        self,
        config: Mapping[str, Any],
        fork_pose: ForkSupportPose,
        *,
        pose_to_base: Sequence[Sequence[float]],
        insertion_axis_fixed: Sequence[float],
    ):
        self.config = dict(config)
        self.control = dict(config["control"])
        self.runtime = dict(config["runtime"])
        self.fork = fork_pose
        self.pose_to_base = np.asarray(pose_to_base, dtype=np.float64).reshape(4, 4)
        self.axis_fixed = _unit(np.asarray(insertion_axis_fixed, dtype=np.float64))
        self.state = FrameStage4State.RESET_PRELOAD
        self.failure = FrameStage4Failure.NONE
        self.milestones = ControllerMilestones()
        self.target_q26 = np.asarray(fork_pose.q26, dtype=np.float64).copy()
        self.state_step = 0
        self.total_steps = 0
        self.support_streak = 0
        self.support_free_streak = 0
        self.contact_loss_steps = 0
        self.insert_contact_loss_steps = 0
        self.insert_hold_steps = 0
        self.preinsert_hold_steps = 0
        self.release_hold_steps = 0
        self.initial_frame_position: np.ndarray | None = None
        self.relative_reference: np.ndarray | None = None
        self._trajectory: _PoseTrajectory | None = None
        self._frame_goal: np.ndarray | None = None
        self._frame_preinsert: np.ndarray | None = None
        self._axis_world: np.ndarray | None = None
        self._lift_commanded = 0.0
        self._release_phase = ""
        self._release_step = 0
        self._release_anchor_q = self.target_q26.copy()
        self._retreat_anchor_q = self.target_q26.copy()

    def command(
        self,
        snapshot: Mapping[str, Any],
        contacts: FrameContactSummary,
    ) -> FrameStage4Command:
        self.total_steps += 1
        self.state_step += 1
        if self.initial_frame_position is None:
            self.initial_frame_position = np.asarray(snapshot["frame_position"], dtype=np.float64).copy()
        self._refresh_goal(snapshot)

        global_failure = self._global_failure(snapshot, contacts)
        if global_failure != FrameStage4Failure.NONE:
            return self._fail(global_failure)

        support_now = len(contacts.support_fingers) >= 2
        self.support_streak = self.support_streak + 1 if support_now else 0
        if self.state == FrameStage4State.RESET_PRELOAD:
            self._transition(FrameStage4State.SETTLE_ON_FORK)
        elif self.state == FrameStage4State.SETTLE_ON_FORK:
            if self.state_step >= int(self.control["settle_steps"]):
                if self.support_streak >= int(self.control["support_streak_steps"]):
                    self._confirm_support(snapshot)
                else:
                    self._transition(FrameStage4State.SUPPORT_CONFIRM)
        elif self.state == FrameStage4State.SUPPORT_CONFIRM:
            if self.support_streak >= int(self.control["support_streak_steps"]):
                self._confirm_support(snapshot)
            elif self.state_step >= int(self.runtime["support_confirm_timeout_steps"]):
                return self._fail(FrameStage4Failure.SUPPORT_NOT_ACQUIRED)
        elif self.state == FrameStage4State.LIFT:
            terminal = self._lift(snapshot, contacts)
            if terminal is not None:
                return terminal
        elif self.state == FrameStage4State.PREINSERT:
            terminal = self._preinsert(snapshot, contacts)
            if terminal is not None:
                return terminal
        elif self.state == FrameStage4State.INSERT:
            terminal = self._insert(snapshot)
            if terminal is not None:
                return terminal
        elif self.state == FrameStage4State.INSERT_HOLD:
            terminal = self._insert_hold(snapshot, contacts)
            if terminal is not None:
                return terminal
        elif self.state == FrameStage4State.RELEASE_TRANSFER:
            terminal = self._release(snapshot, contacts)
            if terminal is not None:
                return terminal
        elif self.state == FrameStage4State.RETREAT:
            terminal = self._retreat(snapshot, contacts)
            if terminal is not None:
                return terminal

        if self.total_steps >= int(self.runtime["max_rollout_steps"]):
            return self._fail(FrameStage4Failure.TIMEOUT)
        return FrameStage4Command(self.state, tuple(float(value) for value in self.target_q26))

    def _global_failure(
        self, snapshot: Mapping[str, Any], contacts: FrameContactSummary
    ) -> FrameStage4Failure:
        if int(snapshot.get("post_reset_object_root_writes", 0)) or int(
            snapshot.get("post_reset_wrist_state_writes", 0)
        ) or int(snapshot.get("post_reset_fixed_asset_root_writes", 0)):
            return FrameStage4Failure.FORBIDDEN_STATE_WRITE
        if not bool(snapshot.get("contact_report_available", False)):
            return FrameStage4Failure.CONTACT_INSTRUMENTATION_UNAVAILABLE
        if contacts.illegal_contact or contacts.frame_table_contact or contacts.frame_ground_contact:
            return FrameStage4Failure.ILLEGAL_CONTACT
        peak = float(np.max(np.asarray(snapshot.get("support_force_n", (0.0,))), initial=0.0))
        if peak >= float(self.control["hard_force_abort_n"]):
            return FrameStage4Failure.HARD_FORCE_ABORT
        if self.state in {
            FrameStage4State.RESET_PRELOAD,
            FrameStage4State.SETTLE_ON_FORK,
            FrameStage4State.SUPPORT_CONFIRM,
        }:
            displacement = float(
                np.linalg.norm(np.asarray(snapshot["frame_position"]) - self.initial_frame_position)
            )
            if displacement > float(self.control["frame_jump_abort_m"]):
                return FrameStage4Failure.FRAME_JUMP
        return FrameStage4Failure.NONE

    def _confirm_support(self, snapshot: Mapping[str, Any]) -> None:
        self.milestones.physical_support_acquired = True
        self.relative_reference = inverse_transform(snapshot["palm_transform"]) @ np.asarray(
            snapshot["frame_transform"]
        )
        self._lift_commanded = 0.0
        self._transition(FrameStage4State.LIFT)

    def _lift(
        self, snapshot: Mapping[str, Any], contacts: FrameContactSummary
    ) -> FrameStage4Command | None:
        if contacts.hand_frame_contact:
            self.contact_loss_steps = 0
        else:
            self.contact_loss_steps += 1
        if self.contact_loss_steps > int(self.control["contact_loss_grace_steps"]):
            return self._fail(FrameStage4Failure.LIFT_SUPPORT_LOST)
        if contacts.frame_fixed_contact or contacts.frame_table_contact or contacts.frame_ground_contact:
            self.support_free_streak = 0
        else:
            self.support_free_streak += 1
        position_drift, rotation_drift = self.relative_drift(snapshot)
        if position_drift > float(self.control["relative_position_limit_m"]) or rotation_drift > float(
            self.control["relative_rotation_limit_deg"]
        ):
            return self._fail(FrameStage4Failure.LIFT_SUPPORT_LOST)
        remaining = float(self.control["lift_distance_m"]) - self._lift_commanded
        if remaining > 1.0e-9:
            increment = min(float(self.control["lift_increment_m"]), remaining)
            self.target_q26[2] += increment
            self._lift_commanded += increment
        frame_lift = float(np.asarray(snapshot["frame_position"])[2] - self.initial_frame_position[2])
        if (
            self._lift_commanded >= float(self.control["lift_distance_m"]) - 1.0e-9
            and frame_lift >= float(self.control["lift_success_m"])
            and self.support_free_streak >= int(self.control["support_free_steps"])
        ):
            self.milestones.physical_lift_success = True
            self._begin_preinsert(snapshot)
        elif self.state_step > int(np.ceil(float(self.control["lift_distance_m"]) / float(self.control["lift_increment_m"]))) + 120:
            return self._fail(FrameStage4Failure.LIFT_SUPPORT_LOST)
        return None

    def _begin_preinsert(self, snapshot: Mapping[str, Any]) -> None:
        assert self.relative_reference is not None and self._frame_preinsert is not None
        hand_goal = self._frame_preinsert @ inverse_transform(self.relative_reference)
        self._trajectory = _PoseTrajectory(
            np.asarray(snapshot["palm_transform"]),
            hand_goal,
            translation_step=float(self.control["wrist_translation_step_limit_m"]),
            rotation_step_deg=float(self.control["wrist_rotation_step_limit_deg"]),
        )
        self.preinsert_hold_steps = 0
        self._transition(FrameStage4State.PREINSERT)

    def _preinsert(
        self, snapshot: Mapping[str, Any], contacts: FrameContactSummary
    ) -> FrameStage4Command | None:
        if not contacts.hand_frame_contact:
            self.contact_loss_steps += 1
        else:
            self.contact_loss_steps = 0
        if self.contact_loss_steps > int(self.control["contact_loss_grace_steps"]):
            return self._fail(FrameStage4Failure.PREINSERT_FAILED)
        if self._trajectory is not None and not self._trajectory.complete:
            self.target_q26[:6] = _wrist_q6_from_transform(self._trajectory.advance(), self.target_q26[:6])
            return None
        error_m, error_deg = pose_error(snapshot["frame_transform"], self._frame_preinsert)
        stable = (
            error_m <= float(self.control["pose_position_tolerance_m"])
            and error_deg <= float(self.control["pose_rotation_tolerance_deg"])
            and not contacts.frame_fixed_contact
        )
        self.preinsert_hold_steps = self.preinsert_hold_steps + 1 if stable else 0
        if self.preinsert_hold_steps >= int(self.control["preinsert_hold_steps"]):
            self.milestones.preinsert_success = True
            self._begin_insert(snapshot)
        elif self.state_step > (self._trajectory.steps if self._trajectory else 0) + 100:
            return self._fail(FrameStage4Failure.PREINSERT_FAILED)
        return None

    def _begin_insert(self, snapshot: Mapping[str, Any]) -> None:
        assert self.relative_reference is not None and self._frame_goal is not None
        hand_goal = self._frame_goal @ inverse_transform(self.relative_reference)
        self._trajectory = _PoseTrajectory(
            np.asarray(snapshot["palm_transform"]),
            hand_goal,
            translation_step=float(self.control["insert_increment_m"]),
            rotation_step_deg=float(self.control["wrist_rotation_step_limit_deg"]),
        )
        self._transition(FrameStage4State.INSERT)

    def _insert(self, snapshot: Mapping[str, Any]) -> FrameStage4Command | None:
        if self._trajectory is not None and not self._trajectory.complete:
            self.target_q26[:6] = _wrist_q6_from_transform(self._trajectory.advance(), self.target_q26[:6])
            return None
        self.insert_hold_steps = 0
        self.insert_contact_loss_steps = 0
        self._transition(FrameStage4State.INSERT_HOLD)
        return None

    def _insert_hold(
        self, snapshot: Mapping[str, Any], contacts: FrameContactSummary
    ) -> FrameStage4Command | None:
        error_m, error_deg = pose_error(snapshot["frame_transform"], self._frame_goal)
        if contacts.frame_fixed_contact:
            self.insert_contact_loss_steps = 0
        else:
            self.insert_contact_loss_steps += 1
        valid = (
            error_m <= float(self.control["pose_position_tolerance_m"])
            and error_deg <= float(self.control["pose_rotation_tolerance_deg"])
            and self.insert_contact_loss_steps <= int(self.control["insert_contact_loss_grace_steps"])
        )
        self.insert_hold_steps = self.insert_hold_steps + 1 if valid else 0
        if self.insert_hold_steps >= int(self.control["inserted_hold_steps"]):
            self.milestones.physical_insert_success = True
            self._release_phase = "lower"
            self._release_step = 0
            self._release_anchor_q = self.target_q26.copy()
            self._transition(FrameStage4State.RELEASE_TRANSFER)
        elif self.state_step > int(self.control["inserted_hold_steps"]) + 120:
            return self._fail(FrameStage4Failure.INSERT_FAILED)
        return None

    def _release(
        self, snapshot: Mapping[str, Any], contacts: FrameContactSummary
    ) -> FrameStage4Command | None:
        error_m, error_deg = pose_error(snapshot["frame_transform"], self._frame_goal)
        if (
            error_m > float(self.control["release_position_warning_m"])
            or error_deg > float(self.control["release_rotation_warning_deg"])
            or not contacts.frame_fixed_contact
        ):
            return self._finish_b()
        self._release_step += 1
        translation_step = float(self.control["wrist_translation_step_limit_m"])
        if self._release_phase == "lower":
            steps = int(np.ceil(float(self.control["release_lower_m"]) / translation_step))
            self.target_q26[2] = self._release_anchor_q[2] - min(
                self._release_step * translation_step, float(self.control["release_lower_m"])
            )
            if self._release_step >= steps:
                self._release_phase = "unhook"
                self._release_step = 0
                self._release_anchor_q = self.target_q26.copy()
        elif self._release_phase == "unhook":
            steps = int(self.config["fork"]["partial_hook_steps"])
            ratio = min(1.0, self._release_step / max(steps, 1))
            self.target_q26[6:] = (1.0 - ratio) * np.asarray(self.fork.hook_q20) + ratio * np.asarray(
                self.fork.fork_q20
            )
            if self._release_step >= steps:
                self._release_phase = "withdraw"
                self._release_step = 0
                self._release_anchor_q = self.target_q26.copy()
        elif self._release_phase == "withdraw":
            assert self._axis_world is not None
            steps = int(np.ceil(float(self.control["release_withdraw_m"]) / translation_step))
            distance = min(self._release_step * translation_step, float(self.control["release_withdraw_m"]))
            self.target_q26[:3] = self._release_anchor_q[:3] + self._axis_world * distance
            if self._release_step >= steps:
                self._retreat_anchor_q = self.target_q26.copy()
                self._release_step = 0
                self.release_hold_steps = 0
                self._transition(FrameStage4State.RETREAT)
        return None

    def _retreat(
        self, snapshot: Mapping[str, Any], contacts: FrameContactSummary
    ) -> FrameStage4Command | None:
        assert self._axis_world is not None
        step = float(self.control["wrist_translation_step_limit_m"])
        move_steps = int(np.ceil(float(self.control["retreat_m"]) / step))
        if self.state_step <= move_steps:
            distance = min(self.state_step * step, float(self.control["retreat_m"]))
            self.target_q26[:3] = self._retreat_anchor_q[:3] + self._axis_world * distance
            return None
        error_m, error_deg = pose_error(snapshot["frame_transform"], self._frame_goal)
        stable = (
            not contacts.hand_frame_contact
            and contacts.frame_fixed_contact
            and error_m <= float(self.control["release_position_limit_m"])
            and error_deg <= float(self.control["release_rotation_limit_deg"])
        )
        self.release_hold_steps = self.release_hold_steps + 1 if stable else 0
        if self.release_hold_steps >= int(self.control["release_hold_steps"]):
            self.milestones.release_stable = True
            self.state = FrameStage4State.DONE
            return FrameStage4Command(
                self.state, tuple(float(value) for value in self.target_q26), terminal=True
            )
        if self.state_step > move_steps + 2 * int(self.control["release_hold_steps"]):
            return self._finish_b()
        return None

    def _finish_b(self) -> FrameStage4Command:
        self.failure = FrameStage4Failure.RELEASE_UNSTABLE
        self.state = FrameStage4State.DONE
        return FrameStage4Command(
            self.state,
            tuple(float(value) for value in self.target_q26),
            terminal=True,
            failure=self.failure,
        )

    def _refresh_goal(self, snapshot: Mapping[str, Any]) -> None:
        self._frame_goal = stage4_goal_transform(
            snapshot["fixed_position"], snapshot["fixed_quat_wxyz"], self.pose_to_base
        )
        fixed_rotation = np.asarray(snapshot["fixed_transform"], dtype=np.float64)[:3, :3]
        self._axis_world = _unit(fixed_rotation @ self.axis_fixed)
        self._frame_preinsert = self._frame_goal.copy()
        self._frame_preinsert[:3, 3] += self._axis_world * float(self.control["preinsert_offset_m"])

    def relative_drift(self, snapshot: Mapping[str, Any]) -> tuple[float, float]:
        if self.relative_reference is None:
            return 0.0, 0.0
        current = inverse_transform(snapshot["palm_transform"]) @ np.asarray(snapshot["frame_transform"])
        return (
            float(np.linalg.norm(current[:3, 3] - self.relative_reference[:3, 3])),
            rotation_error_deg(current[:3, :3], self.relative_reference[:3, :3]),
        )

    def pose_errors(self, snapshot: Mapping[str, Any]) -> tuple[float, float, float, float]:
        pre = pose_error(snapshot["frame_transform"], self._frame_preinsert)
        goal = pose_error(snapshot["frame_transform"], self._frame_goal)
        return pre[0], pre[1], goal[0], goal[1]

    def _transition(self, state: FrameStage4State) -> None:
        self.state = state
        self.state_step = 0

    def _fail(self, failure: FrameStage4Failure) -> FrameStage4Command:
        self.failure = failure
        self.state = FrameStage4State.FAILED
        return FrameStage4Command(
            self.state,
            tuple(float(value) for value in self.target_q26),
            terminal=True,
            failure=failure,
        )


class _PoseTrajectory:
    def __init__(self, start: np.ndarray, end: np.ndarray, *, translation_step: float, rotation_step_deg: float):
        self.start = np.asarray(start, dtype=np.float64).reshape(4, 4)
        self.end = np.asarray(end, dtype=np.float64).reshape(4, 4)
        translation = float(np.linalg.norm(self.end[:3, 3] - self.start[:3, 3]))
        rotation = rotation_error_deg(self.start[:3, :3], self.end[:3, :3])
        self.steps = max(1, int(np.ceil(translation / translation_step)), int(np.ceil(rotation / rotation_step_deg)))
        self.index = 0

    @property
    def complete(self) -> bool:
        return self.index >= self.steps

    def advance(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        self.index = min(self.index + 1, self.steps)
        ratio = self.index / self.steps
        alpha = ratio**3 * (10.0 - 15.0 * ratio + 6.0 * ratio**2)
        result = np.eye(4, dtype=np.float64)
        result[:3, 3] = (1.0 - alpha) * self.start[:3, 3] + alpha * self.end[:3, 3]
        delta = Rotation.from_matrix(self.start[:3, :3].T @ self.end[:3, :3]).as_rotvec()
        result[:3, :3] = self.start[:3, :3] @ Rotation.from_rotvec(alpha * delta).as_matrix()
        return result


def _wrist_q6_from_transform(transform: np.ndarray, reference_q6: Sequence[float]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    pitch = float(np.arcsin(np.clip(rotation[0, 2], -1.0, 1.0)))
    roll = float(np.arctan2(-rotation[1, 2], rotation[2, 2]))
    yaw = float(np.arctan2(-rotation[0, 1], rotation[0, 0]))
    rpy = np.asarray((roll, pitch, yaw), dtype=np.float64)
    reference = np.asarray(reference_q6, dtype=np.float64)
    rpy += 2.0 * np.pi * np.round((reference[3:6] - rpy) / (2.0 * np.pi))
    return np.concatenate((matrix[:3, 3], rpy))


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("zero axis")
    return value / norm
