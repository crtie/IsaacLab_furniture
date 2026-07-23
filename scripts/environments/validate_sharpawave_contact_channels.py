"""Validate five Sharpawave elastomer contact channels with one known probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading

os.environ.setdefault("ISAACLAB_TASKS_DEFER_DISCOVERY", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--calibration", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--contact-steps", type=int, default=12)
parser.add_argument("--recovery-steps", type=int, default=24)
parser.add_argument("--probe-radius", type=float, default=0.015)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402
from isaaclab_tasks.chair_assembly.sharpawave_runtime import SharpawaveIsaacRuntime  # noqa: E402


ROLES = ("thumb", "index", "middle", "ring", "pinky")


def _probe_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/ContactProbe",
        spawn=sim_utils.SphereCfg(
            radius=float(args.probe_radius),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=0.25,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 2.0)),
    )


def _step(runtime: SharpawaveIsaacRuntime, probe: RigidObject, steps: int) -> None:
    dt = runtime.sim.get_physics_dt()
    target = runtime.robot.data.joint_pos.clone()
    for _ in range(int(steps)):
        runtime.robot.set_joint_position_target(target)
        runtime.robot.write_data_to_sim()
        probe.write_data_to_sim()
        runtime.sim.step(render=False)
        runtime.robot.update(dt)
        probe.update(dt)
        for sensor in runtime.sensors:
            sensor.update(dt, force_recompute=True)


def _forces(runtime: SharpawaveIsaacRuntime) -> dict[str, float]:
    values = {}
    for role, sensor in zip(ROLES, runtime.sensors):
        array = sensor.data.net_forces_w.detach().cpu().numpy()
        values[role] = float(np.linalg.norm(array, axis=-1).max())
    return values


def _move_probe(probe: RigidObject, position: torch.Tensor) -> None:
    pose = probe.data.root_pose_w.clone()
    pose[:, :3] = position.reshape(1, 3)
    pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=pose.dtype, device=pose.device)
    probe.write_root_pose_to_sim(pose)
    probe.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=pose.dtype, device=pose.device))


def main() -> int:
    payload = {}
    try:
        sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
        spawn_ground_plane("/World/Ground", GroundPlaneCfg())
        runtime = SharpawaveIsaacRuntime.spawn(sim, calibration_path=args.calibration)
        probe = RigidObject(_probe_cfg())
        runtime.reset(warmup_steps=18)
        probe.update(sim.get_physics_dt())
        _move_probe(probe, torch.tensor((2.0, 2.0, 2.0), device=runtime.robot.device))
        _step(runtime, probe, int(args.recovery_steps))
        baseline = _forces(runtime)
        body_index = {name: index for index, name in enumerate(runtime.robot.body_names)}
        rows = []
        for role, spec in zip(ROLES, runtime.adapter.get_fingertip_specs()):
            contact_position = runtime.robot.data.body_pos_w[0, body_index[spec.contact_body]].clone()
            _move_probe(probe, contact_position)
            peaks = {name: 0.0 for name in ROLES}
            for _ in range(int(args.contact_steps)):
                _step(runtime, probe, 1)
                sample = _forces(runtime)
                for name in ROLES:
                    peaks[name] = max(peaks[name], sample[name])
            active = peaks[role]
            other_peak = max(peaks[name] for name in ROLES if name != role)
            _move_probe(probe, torch.tensor((2.0, 2.0, 2.0), device=runtime.robot.device))
            _step(runtime, probe, int(args.recovery_steps))
            recovery = _forces(runtime)
            recovery_max = max(recovery.values())
            nonzero = active > max(0.01, baseline[role] + 0.01)
            isolated = active > 0.02 and other_peak <= 0.01
            recovered = recovery_max < max(0.01, active * 0.05)
            rows.append({
                "role": role,
                "contact_body": spec.contact_body,
                "baseline_n": baseline[role],
                "active_peak_n": active,
                "other_channel_peak_n": other_peak,
                "all_contact_peaks_n": peaks,
                "recovery_max_n": recovery_max,
                "nonzero": nonzero,
                "isolated": isolated,
                "recovered": recovered,
                "passed": nonzero and isolated and recovered,
            })
        wuji = sorted(name for name in sys.modules if "wuji" in name.lower())
        passed = all(row["passed"] for row in rows) and not wuji
        payload = {
            "status": "SHARPAWAVE_CONTACT_CHANNELS_OK" if passed else "SHARPAWAVE_CONTACT_CHANNELS_FAILED",
            "passed": passed,
            "probe": {"shape": "sphere", "radius_m": float(args.probe_radius), "kinematic": True},
            "baseline_all_n": baseline,
            "channels": rows,
            "wuji_modules_loaded": wuji,
            "contact_response_probed": True,
            "sticky_used": False,
            "snap_used": False,
            "teacher_motion_used": False,
            "root_pose_writes_used": True,
            "physical_grasp_success": False,
            "physical_lift_success": False,
            "physical_insert_success": False,
            "oracle_visual_only": False,
            "not_physical": True,
        }
        runtime.close()
    except Exception as exc:
        payload = {"status": "SHARPAWAVE_CONTACT_CHANNELS_FAILED", "passed": False, "error_type": type(exc).__name__, "error": str(exc), "not_physical": True}
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=5.0)
    return 0 if payload.get("passed") else 2


if __name__ == "__main__":
    status = main()
    os._exit(int(status))
