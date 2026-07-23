"""Run the packaged SharpawaveIsaacRuntime without chair policy or Wuji imports."""

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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402
from isaaclab_tasks.chair_assembly import SharpawaveIsaacRuntime  # noqa: E402


def main() -> int:
    payload = {}
    try:
        sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
        spawn_ground_plane("/World/Ground", GroundPlaneCfg())
        runtime = SharpawaveIsaacRuntime.spawn(sim, calibration_path=args.calibration)
        runtime.reset()
        before = runtime.observation()
        action = np.zeros((1, runtime.action_dim), dtype=np.float64)
        index = runtime.adapter.canonical_joint_names.index("right_index_MCP_FE")
        action[0, index] = 0.5
        runtime.apply_action(action, steps=60)
        after = runtime.observation()
        delta = float(after["joint_position"][0, index] - before["joint_position"][0, index])
        wuji_modules = sorted(name for name in sys.modules if "wuji" in name.lower())
        contacts = after["contact_force_norm"]
        passed = bool(
            delta > 0.005
            and len(before["joint_names"]) == 28
            and len(contacts) == 5
            and all(np.isfinite(value).all() for value in contacts.values())
            and not wuji_modules
        )
        payload = {
            "status": "SHARPAWAVE_RUNTIME_ADAPTER_OK" if passed else "SHARPAWAVE_RUNTIME_ADAPTER_FAILED",
            "passed": passed,
            "action_dim": runtime.action_dim,
            "schema_id": before["schema_id"],
            "index_motion_delta_rad": delta,
            "contact_roles": sorted(contacts),
            "contact_force_api_finite": all(np.isfinite(value).all() for value in contacts.values()),
            "contact_response_probed": False,
            "wuji_modules_loaded": wuji_modules,
            "sticky_used": False,
            "snap_used": False,
            "teacher_motion_used": False,
            "root_pose_writes_used": False,
            "physical_grasp_success": False,
            "physical_lift_success": False,
            "physical_insert_success": False,
            "oracle_visual_only": False,
            "not_physical": True
        }
        runtime.close()
    except Exception as exc:
        payload = {"status": "SHARPAWAVE_RUNTIME_ADAPTER_FAILED", "passed": False, "error_type": type(exc).__name__, "error": str(exc), "not_physical": True}
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
