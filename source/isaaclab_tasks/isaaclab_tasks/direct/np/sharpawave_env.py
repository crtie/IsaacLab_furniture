"""Small, honest SharpaWave schema environment and runtime capability gate.

This module deliberately does not inherit from the Wuji chair environments.
The schema environment is useful for adapter/policy contract tests; the chair
entry point fails explicitly until an Isaac articulation, calibrated poses and
runtime contact reporting are supplied.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from isaaclab_tasks.robot_adapters.sharpa_wave import (
    MISSING_CALIBRATION,
    MissingCalibrationError,
    SharpaWaveAdapter,
    get_sharpawave_spec,
)


def _honesty_info() -> dict[str, Any]:
    """Return explicit non-physical metadata for schema-only transitions."""

    return {
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
    }


class SharpaWaveSchemaEnv(gym.Env):
    """Deterministic schema-only Gym facade for a selected Sharpa variant.

    It exercises action/observation shape contracts, not Isaac physics. The
    metadata explicitly marks the environment as ``system_only``.
    """

    metadata = {"render_modes": (), "robot_name": "sharpawave", "status": "system_only"}

    def __init__(self, *, variant: str = "floating", **_: Any):
        self.variant = str(variant)
        self.specification = get_sharpawave_spec(self.variant)
        self.adapter = SharpaWaveAdapter.from_spec(self.specification, env=None)
        # Normalize the optional asset-extension config to the dependency-light
        # adapter model before reading action dimensions.
        self.specification = self.adapter.model
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.adapter.action_dim,),
            dtype=np.float32,
        )
        # The schema facade exposes q/dq for every actuated joint. Real
        # fingertip/contact values are intentionally unavailable here.
        self.observation_space = gym.spaces.Dict(
            {
                "joint_pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.adapter.action_dim,),
                    dtype=np.float32,
                ),
                "joint_vel": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.adapter.action_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self._joint_pos = np.zeros(self.action_space.shape, dtype=np.float32)
        self._joint_vel = np.zeros_like(self._joint_pos)

    def _observation(self) -> dict[str, np.ndarray]:
        return {"joint_pos": self._joint_pos.copy(), "joint_vel": self._joint_vel.copy()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._joint_pos.fill(0.0)
        self._joint_vel.fill(0.0)
        info = {"robot_name": "sharpawave", "variant": self.variant, "status": "system_only"}
        info.update(_honesty_info())
        return self._observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"SHARPAWAVE_ACTION_SHAPE:{action.shape}!={self.action_space.shape}")
        if not np.isfinite(action).all():
            raise ValueError("SHARPAWAVE_ACTION_NONFINITE")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("SHARPAWAVE_ACTION_OUT_OF_RANGE")
        # The facade has no calibrated actuator scale.  Route even its schema
        # actions through the adapter so a non-zero command cannot acquire an
        # accidental implicit scale here.  The only accepted transition is the
        # zero-delta no-op used by schema smoke tests.
        command = self.adapter.map_canonical_action(
            action.reshape(1, -1),
            profile="joint_delta",
        )
        self._joint_vel = np.asarray(command[0], dtype=np.float32)
        info = {
            "robot_name": "sharpawave",
            "variant": self.variant,
            "status": "system_only",
        }
        info.update(_honesty_info())
        return self._observation(), 0.0, False, False, info


class SharpaWaveChairTeacherEnv(gym.Env):
    """Explicit gate for the not-yet-verified Isaac chair integration."""

    metadata = {"render_modes": (), "robot_name": "sharpawave", "status": "missing_dependency"}

    def __init__(self, *, variant: str = "floating", **_: Any):
        self.variant = str(variant)
        raise MissingCalibrationError(
            MISSING_CALIBRATION,
            "chair runtime requires an Isaac articulation, base-pose mapper, calibrated hand profiles, "
            "and verified elastomer contact sensors; use the schema environment for contract tests",
            variant=self.variant,
            capability="SHARPAWAVE_RUNTIME_NOT_READY",
        )

    def reset(self, **kwargs):  # pragma: no cover - constructor is the gate
        raise RuntimeError("SHARPAWAVE_RUNTIME_NOT_READY")

    def step(self, action):  # pragma: no cover - constructor is the gate
        raise RuntimeError("SHARPAWAVE_RUNTIME_NOT_READY")
