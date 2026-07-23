"""Live Isaac Lab runtime bridge for the independent Sharpawave articulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab_tasks.robot_adapters.sharpa_wave import MissingCalibrationError, SharpaWaveAdapter
from isaaclab_tasks.robot_adapters.protocol import MISSING_ASSET, MissingCapabilityError

from .asset_install import validate_asset_install


def load_runtime_calibration(path: str | Path, *, variant: str = "floating") -> dict[str, Any]:
    calibration_path = Path(path).expanduser().resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_schema = f"sharpawave.robot_schema.v1.{variant}"
    if payload.get("robot") != "sharpawave" or payload.get("variant") != variant:
        raise MissingCalibrationError(message="runtime calibration robot/variant mismatch", path=str(calibration_path))
    if payload.get("action_schema") != expected_schema:
        raise MissingCalibrationError(message="runtime calibration action schema mismatch", expected=expected_schema)
    adapter = SharpaWaveAdapter(variant, calibration={"action_scale": payload.get("action_scale")})
    adapter._configured_action_scale()
    return payload


class SharpawaveIsaacRuntime:
    """Own the Sharpawave articulation/sensors inside an existing SimulationContext.

    The class handles spawning, reset, canonical action mapping, readback, and
    five elastomer force sensors. It does not implement grasp or insertion
    success and has no Wuji/oracle/system-validation fallback.
    """

    robot_name = "sharpawave"

    def __init__(self, sim: Any, robot: Any, sensors: list[Any], adapter: SharpaWaveAdapter, calibration: dict[str, Any]):
        self.sim = sim
        self.robot = robot
        self.sensors = list(sensors)
        self.adapter = adapter
        self.calibration = dict(calibration)
        self.variant = adapter.variant
        self.action_schema_id = f"sharpawave.robot_schema.v1.{self.variant}"
        self.action_dim = adapter.action_dim
        self.closed = False

    @classmethod
    def spawn(
        cls,
        sim: Any,
        *,
        calibration_path: str | Path,
        variant: str = "floating",
        prim_path: str = "/World/Robot",
        init_pos: tuple[float, float, float] = (0.38, -0.48, 0.92),
    ) -> "SharpawaveIsaacRuntime":
        repo_root = Path(__file__).resolve().parents[4]
        requirement_path = repo_root / "configs/chair_assembly/sharpawave_asset_requirement.json"
        asset_report = validate_asset_install(requirement_path)
        if not asset_report["ok"]:
            raise MissingCapabilityError(
                MISSING_ASSET,
                f"Sharpawave assets are not installed or do not match the required hash: {asset_report}",
                details=asset_report,
            )
        from isaaclab.assets import Articulation
        from isaaclab.sensors import ContactSensor
        from isaaclab_assets.robots.sharpawave import get_sharpawave_variant
        from isaaclab_assets.robots.sharpawave_isaac import (
            build_sharpawave_articulation_cfg,
            build_sharpawave_contact_sensor_cfgs,
        )

        calibration = load_runtime_calibration(calibration_path, variant=variant)
        cfg = build_sharpawave_articulation_cfg(
            variant,
            prim_path=prim_path,
            translation_stiffness=float(calibration["translation_stiffness"]),
            translation_damping=float(calibration["translation_damping"]),
            rotation_stiffness=float(calibration["rotation_stiffness"]),
            rotation_damping=float(calibration["rotation_damping"]),
            finger_stiffness=float(calibration["finger_stiffness"]),
            finger_damping=float(calibration["finger_damping"]),
        )
        cfg.init_state.pos = init_pos
        robot = Articulation(cfg)
        sensors = [ContactSensor(sensor_cfg) for sensor_cfg in build_sharpawave_contact_sensor_cfgs(variant, prim_prefix=prim_path)]
        adapter = SharpaWaveAdapter.from_spec(
            get_sharpawave_variant(variant),
            env=robot,
            calibration={"action_scale": calibration["action_scale"]},
        )
        return cls(sim, robot, sensors, adapter, calibration)

    def reset(self, *, warmup_steps: int = 12) -> None:
        self.sim.reset()
        dt = self.sim.get_physics_dt()
        self.robot.update(dt)
        target = self.robot.data.default_joint_pos.clone()
        for _ in range(int(warmup_steps)):
            self.robot.set_joint_position_target(target)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(dt)
            for sensor in self.sensors:
                sensor.update(dt)
        self.adapter.validate_runtime_articulation(self.robot, require_contact=True).require_ok()

    def apply_action(self, action: Any, *, profile: str = "joint_delta", steps: int = 1) -> None:
        import torch

        current = self.robot.data.joint_pos.detach().cpu().numpy()
        runtime_target = self.adapter.map_canonical_action(
            action,
            runtime_joint_names=self.robot.joint_names,
            current_position=current,
            profile=profile,
            action_schema_id=self.adapter.schema.schema_id,
        )
        target = torch.as_tensor(runtime_target, dtype=self.robot.data.joint_pos.dtype, device=self.robot.device)
        dt = self.sim.get_physics_dt()
        for _ in range(int(steps)):
            self.robot.set_joint_position_target(target)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(dt)
            for sensor in self.sensors:
                sensor.update(dt)

    def observation(self) -> dict[str, Any]:
        obs = self.adapter.get_robot_observation(self.robot)
        contact = {}
        for spec, sensor in zip(self.adapter.get_fingertip_specs(), self.sensors):
            forces = sensor.data.net_forces_w.detach().cpu().numpy()
            contact[spec.role] = np.linalg.norm(forces, axis=-1).reshape(forces.shape[0], -1).max(axis=1)
        return {
            "schema_id": obs.schema_id,
            "joint_names": list(obs.joint_names),
            "joint_position": obs.joint_position,
            "joint_velocity": obs.joint_velocity,
            "fingertip_poses": obs.fingertip_poses,
            "contact_force_norm": contact,
        }

    def close(self) -> None:
        self.closed = True
