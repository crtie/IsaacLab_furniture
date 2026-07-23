"""Independent SharpaWave runtime canary and chair stage-initialization recorder.

This delivery-only runner deliberately bypasses the Wuji environments, the
22-target director, sticky/snap/oracle helpers, and policy code.  It validates
the selected SharpaWave articulation by exact joint name and records short
initialization evidence only.  It does not claim grasp or assembly success.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


os.environ.setdefault("ISAACLAB_TASKS_DEFER_DISCOVERY", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", type=int, choices=range(1, 7), default=None)
parser.add_argument("--canary-only", action="store_true")
parser.add_argument("--robot-variant", choices=("floating", "peg_fixedrot"), default="floating")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration", type=float, default=3.2)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--warmup-steps", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.stage is None and not args_cli.canary_only:
    parser.error("select --canary-only or --stage 1..6")
if args_cli.stage is not None and args_cli.record_video and args_cli.output is None:
    parser.error("--record-video requires --output")
if args_cli.stage is not None and args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensor  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402,F401
from isaacsim.core.utils import prims as prim_utils  # noqa: E402
from isaaclab_assets.robots.sharpawave import get_sharpawave_variant  # noqa: E402
from isaaclab_assets.robots.sharpawave_isaac import (  # noqa: E402
    build_sharpawave_articulation_cfg,
    build_sharpawave_contact_sensor_cfgs,
)
from isaaclab_tasks.robot_adapters.sharpa_wave import SharpaWaveAdapter  # noqa: E402
from isaaclab_tasks.direct.np.chair_tasks_cfg import (  # noqa: E402
    ChairAssembly1,
    ChairAssembly2,
    ChairAssembly3,
    ChairAssembly4,
    ChairAssembly5,
    ChairAssembly6,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FFMPEG = (
    Path(sys.executable).resolve().parent
    / "lib"
    / "python3.10"
    / "site-packages"
    / "imageio_ffmpeg"
    / "binaries"
    / "ffmpeg-linux-x86_64-v7.0.2"
)
if not FFMPEG.is_file():
    try:
        import imageio_ffmpeg

        FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

STAGE_CLASSES = {
    1: ChairAssembly1,
    2: ChairAssembly2,
    3: ChairAssembly3,
    4: ChairAssembly4,
    5: ChairAssembly5,
    6: ChairAssembly6,
}
STAGE_PARTS = {
    1: (("Plug1", "plug1"), ("Plug2", "plug2"), ("Backrest", "backrest")),
    2: (("Plug1", "plug1"), ("Plug2", "plug2"), ("Rod", "rod")),
    3: (("Plug1", "plug1"), ("Plug2", "plug2"), ("Rod", "rod")),
    4: (("Frame", "frame"),),
    5: (("Screw1", "screw1"), ("Screw2", "screw2"), ("Screw3", "screw3")),
    6: (("Screw1", "screw1"), ("Screw2", "screw2"), ("Screw3", "screw3")),
}
STAGE_TARGET_COUNTS = {1: 5, 2: 5, 3: 5, 4: 1, 5: 3, 6: 3}
STAGING_POS = {
    "Plug1": (0.245, -0.090, 0.865),
    "Plug2": (0.315, -0.090, 0.865),
    "Rod": (0.285, -0.170, 0.900),
    "Backrest": (0.300, -0.180, 0.920),
    "Frame": (0.300, -0.180, 0.920),
    "Screw1": (0.245, -0.090, 0.865),
    "Screw2": (0.300, -0.090, 0.865),
    "Screw3": (0.355, -0.090, 0.865),
}
STAGING_QUAT = {
    "Plug1": (1.0, 0.0, 0.0, 0.0),
    "Plug2": (1.0, 0.0, 0.0, 0.0),
    "Rod": (0.5, -0.5, 0.5, 0.5),
    "Backrest": (0.5, -0.5, 0.5, 0.5),
    "Frame": (1.0, 0.0, 0.0, 0.0),
    "Screw1": (1.0, 0.0, 0.0, 0.0),
    "Screw2": (1.0, 0.0, 0.0, 0.0),
    "Screw3": (1.0, 0.0, 0.0, 0.0),
}


def _honesty(*, root_writes: bool) -> dict[str, Any]:
    return {
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": bool(root_writes),
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "physical_insert_success": False,
        "oracle_visual_only": False,
        "not_physical": True,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_report(payload: dict[str, Any]) -> None:
    args_cli.report.parent.mkdir(parents=True, exist_ok=True)
    args_cli.report.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quat_matrix_wxyz(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    n = max(w * w + x * x + y * y + z * z, 1.0e-12)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _target_cfgs(task: Any, stage: int) -> list[Any]:
    if stage in (5, 6):
        names = ("connection_cfg1_fix", "connection_cfg2", "connection_cfg3")
    else:
        names = tuple(f"connection_cfg{i}" for i in range(1, STAGE_TARGET_COUNTS[stage] + 1))
    return [getattr(task, name) for name in names]


def _spawn_target_markers(task: Any, stage: int) -> list[list[float]]:
    fixed_pos = np.asarray(task.fixed_asset.init_state.pos, dtype=np.float64)
    fixed_rot = _quat_matrix_wxyz(tuple(task.fixed_asset.init_state.rot))
    out: list[list[float]] = []
    for index, cfg in enumerate(_target_cfgs(task, stage), start=1):
        pose = np.asarray(cfg.pose_to_base, dtype=np.float64)
        pos = fixed_pos + fixed_rot @ pose[:3, 3]
        marker = sim_utils.SphereCfg(
            radius=0.008,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.25), emissive_color=(0.02, 0.2, 0.04)),
            collision_props=None,
        )
        marker.func(f"/World/Targets/Target{index}", marker, translation=tuple(float(v) for v in pos))
        out.append([float(v) for v in pos])
    return out


def _make_robot() -> tuple[Articulation, SharpaWaveAdapter, Any]:
    variant_cfg = get_sharpawave_variant(args_cli.robot_variant)
    robot_cfg = build_sharpawave_articulation_cfg(
        args_cli.robot_variant,
        prim_path="/World/Robot",
        activate_contact_sensors=True,
    )
    robot_cfg.init_state.pos = (0.38, -0.48, 0.92)
    robot = Articulation(robot_cfg)
    adapter = SharpaWaveAdapter.from_spec(variant_cfg, env=robot)
    return robot, adapter, variant_cfg


def _make_camera() -> Camera:
    cfg = CameraCfg(
        height=720,
        width=1280,
        prim_path="/World/DeliveryCamera",
        update_period=0.0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 100.0),
        ),
    )
    return Camera(cfg)


def _step(sim: SimulationContext, robot: Articulation, target: torch.Tensor, *, render: bool) -> None:
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step(render=render)
    robot.update(sim.get_physics_dt())


def _finite_robot(robot: Articulation) -> bool:
    return bool(torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all())


def _run_motion_canary(sim: SimulationContext, robot: Articulation) -> dict[str, Any]:
    names = list(robot.joint_names)
    wrist_name = "right_x_joint"
    finger_name = "right_index_MCP_FE"
    indices = {name: names.index(name) for name in (wrist_name, finger_name)}
    start = robot.data.joint_pos.clone()
    target = start.clone()
    checks: dict[str, Any] = {}
    for name, delta, minimum in ((wrist_name, 0.002, 0.0002), (finger_name, 0.02, 0.002)):
        index = indices[name]
        before = float(robot.data.joint_pos[0, index].item())
        target.copy_(start)
        target[0, index] = before + float(delta)
        for _ in range(60):
            _step(sim, robot, target, render=False)
        after = float(robot.data.joint_pos[0, index].item())
        readback = after - before
        checks[name] = {
            "command_delta": float(delta),
            "readback_delta": float(readback),
            "minimum_same_direction_readback": float(minimum),
            "passed": bool(np.isfinite(readback) and readback * delta > 0.0 and abs(readback) >= minimum),
        }
        target.copy_(start)
        for _ in range(60):
            _step(sim, robot, target, render=False)
    return checks


def _contact_canary(sensors: list[ContactSensor], dt: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sensor in sensors:
        sensor.update(dt, force_recompute=True)
        forces = sensor.data.net_forces_w
        array = forces.detach().cpu().numpy()
        rows.append(
            {
                "prim_path": sensor.cfg.prim_path,
                "shape": list(array.shape),
                "finite": bool(np.isfinite(array).all()),
                "force_norm_max": float(np.linalg.norm(array, axis=-1).max()) if array.size else 0.0,
            }
        )
    passed = len(rows) == 5 and all(row["finite"] and row["shape"][-1:] == [3] for row in rows)
    return {
        "passed": passed,
        "sensor_count": len(rows),
        "force_api_available": passed,
        "contact_response_probed": False,
        "sensors": rows,
    }


def _runtime_contract(robot: Articulation, adapter: SharpaWaveAdapter, variant_cfg: Any) -> dict[str, Any]:
    actual = list(robot.joint_names)
    expected = list(variant_cfg.canonical_joint_names)
    report = adapter.validate_runtime_articulation(robot, require_contact=False)
    wuji_modules = sorted(name for name in sys.modules if "wuji" in name.lower())
    name_set_match = len(actual) == len(expected) and set(actual) == set(expected) and len(set(actual)) == len(actual)
    return {
        "expected_joint_names": expected,
        "actual_joint_names": actual,
        "expected_dof": len(expected),
        "actual_dof": len(actual),
        "exact_joint_order_match": actual == expected,
        "runtime_joint_schema_match": name_set_match,
        "runtime_reordering_required": actual != expected,
        "runtime_name_to_index": {name: actual.index(name) for name in expected if name in actual},
        "adapter_validation": report.as_dict(),
        "finite_after_reset": _finite_robot(robot),
        "wuji_modules_loaded": wuji_modules,
        "no_wuji_fallback": not wuji_modules,
    }


def _encode_frames(frame_dir: Path, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(FFMPEG),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-tag:v",
        "avc1",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(int(fps)),
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    subprocess.run(command, check=True)


def _run_canary() -> dict[str, Any]:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    spawn_ground_plane("/World/Ground", GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.82, 0.82))
    light.func("/World/Light", light)
    robot, adapter, variant_cfg = _make_robot()
    contact_sensors = [ContactSensor(cfg) for cfg in build_sharpawave_contact_sensor_cfgs(args_cli.robot_variant, prim_prefix="/World/Robot")]
    reset_started = time.monotonic()
    sim.reset()
    reset_seconds = time.monotonic() - reset_started
    robot.update(sim.get_physics_dt())
    target = robot.data.default_joint_pos.clone()
    for _ in range(int(args_cli.warmup_steps)):
        _step(sim, robot, target, render=False)
        for sensor in contact_sensors:
            sensor.update(sim.get_physics_dt())
    contract = _runtime_contract(robot, adapter, variant_cfg)
    motions = _run_motion_canary(sim, robot) if all(
        (
            contract["runtime_joint_schema_match"],
            contract["finite_after_reset"],
            contract["no_wuji_fallback"],
            bool(contract["adapter_validation"].get("ok")),
        )
    ) else {}
    contacts = _contact_canary(contact_sensors, sim.get_physics_dt())
    passed = bool(
        contract["runtime_joint_schema_match"]
        and contract["finite_after_reset"]
        and contract["no_wuji_fallback"]
        and contract["adapter_validation"].get("ok")
        and motions
        and all(item.get("passed") for item in motions.values())
        and contacts.get("passed")
        and _finite_robot(robot)
    )
    return {
        "kind": "sharpawave_runtime_canary",
        "status": "SHARPAWAVE_RUNTIME_VERIFIED" if passed else "SHARPAWAVE_RUNTIME_BLOCKED",
        "passed": passed,
        "variant": args_cli.robot_variant,
        "asset_path": variant_cfg.usd_path,
        "reset_seconds": reset_seconds,
        "runtime_contract": contract,
        "motion_canary": motions,
        "contact_sensor_canary": contacts,
        "canonical_action_mapping_used": False,
        "diagnostic_joint_targets_in_physical_units": True,
        **_honesty(root_writes=False),
    }


def _stage_assets(stage: int, task: Any) -> tuple[Articulation, dict[str, RigidObject]]:
    fixed = Articulation(task.fixed_asset.replace(prim_path="/World/FixedAsset"))
    parts: dict[str, RigidObject] = {}
    for public_name, attr in STAGE_PARTS[stage]:
        parts[public_name] = RigidObject(getattr(task, attr).replace(prim_path=f"/World/{public_name}"))
    return fixed, parts


def _declared_part_states(parts: dict[str, RigidObject]) -> dict[str, list[float]]:
    """Record task-declared initial states without teleporting scene objects."""

    staged: dict[str, list[float]] = {}
    for name, part in parts.items():
        state = part.data.default_root_state.clone()
        staged[name] = [float(v) for v in state[0, :7].detach().cpu().tolist()]
    return staged


def _run_stage(stage: int) -> dict[str, Any]:
    task = STAGE_CLASSES[stage]()
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    spawn_ground_plane("/World/Ground", GroundPlaneCfg())
    table_path = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/workdesk.usd"
    table = sim_utils.UsdFileCfg(
        usd_path=str(table_path),
        scale=np.array([1.0, 0.7, 1.0]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0e7),
    )
    table.func("/World/Table", table)
    light = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.82, 0.82))
    light.func("/World/Light", light)
    robot, adapter, variant_cfg = _make_robot()
    fixed, parts = _stage_assets(stage, task)
    targets = _spawn_target_markers(task, stage)
    camera = _make_camera() if args_cli.record_video else None
    reset_started = time.monotonic()
    sim.reset()
    reset_seconds = time.monotonic() - reset_started
    dt = sim.get_physics_dt()
    robot.update(dt)
    fixed.update(dt)
    for part in parts.values():
        part.update(dt)
    staged = _declared_part_states(parts)
    target = robot.data.default_joint_pos.clone()
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[0.70, -1.32, 1.28]], dtype=torch.float32, device=camera.device),
            torch.tensor([[-0.03, -0.18, 0.90]], dtype=torch.float32, device=camera.device),
        )
    for _ in range(int(args_cli.warmup_steps)):
        _step(sim, robot, target, render=bool(args_cli.record_video))
        fixed.update(dt)
        for part in parts.values():
            part.update(dt)
        if camera is not None:
            camera.update(dt)
    contract = _runtime_contract(robot, adapter, variant_cfg)
    initial_positions = {name: part.data.root_pos_w[0].detach().cpu().numpy().copy() for name, part in parts.items()}
    frame_dir = None
    if args_cli.record_video:
        assert args_cli.output is not None
        frame_dir = args_cli.output.parent / f".{args_cli.output.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(float(args_cli.duration) * int(args_cli.fps))))
    sim_steps_per_frame = max(1, int(round((1.0 / int(args_cli.fps)) / dt)))
    finite = True
    for frame_index in range(total_frames):
        for _ in range(sim_steps_per_frame):
            _step(sim, robot, target, render=bool(args_cli.record_video))
            fixed.update(dt)
            for part in parts.values():
                part.update(dt)
            if camera is not None:
                camera.update(dt)
        if camera is not None and frame_dir is not None:
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(frame_dir / f"frame_{frame_index:05d}.png"), frame):
                raise RuntimeError("unable to write camera frame")
        finite = bool(finite and _finite_robot(robot))
        for part in parts.values():
            finite = bool(finite and torch.isfinite(part.data.root_state_w).all())
    if args_cli.record_video:
        assert frame_dir is not None and args_cli.output is not None
        _encode_frames(frame_dir, args_cli.output, int(args_cli.fps))
    final_positions = {name: part.data.root_pos_w[0].detach().cpu().numpy().copy() for name, part in parts.items()}
    displacement = {
        name: float(np.linalg.norm(final_positions[name] - initial_positions[name])) for name in parts
    }
    prims = ["/World/Robot", "/World/FixedAsset", *[f"/World/{name}" for name in parts]]
    prim_presence = {path: bool(prim_utils.is_prim_path_valid(path)) for path in prims}
    stable = bool(all(value < 0.50 for value in displacement.values()))
    passed = bool(
        contract["runtime_joint_schema_match"]
        and contract["finite_after_reset"]
        and contract["no_wuji_fallback"]
        and contract["adapter_validation"].get("ok")
        and finite
        and stable
        and all(prim_presence.values())
        and len(targets) == STAGE_TARGET_COUNTS[stage]
        and (not args_cli.record_video or (args_cli.output is not None and args_cli.output.is_file()))
    )
    return {
        "kind": "sharpawave_stage_initialization",
        "stage": stage,
        "status": "STAGE_INIT_OK" if passed else "SYSTEM_ONLY_PREVIEW",
        "passed": passed,
        "variant": args_cli.robot_variant,
        "asset_path": variant_cfg.usd_path,
        "reset_seconds": reset_seconds,
        "runtime_contract": contract,
        "required_parts": list(parts),
        "part_staging_root_states": staged,
        "target_count_expected": STAGE_TARGET_COUNTS[stage],
        "target_positions": targets,
        "prim_presence": prim_presence,
        "finite_state": finite,
        "part_displacement_m": displacement,
        "no_first_frame_explosion": stable,
        "video": args_cli.output if args_cli.record_video else None,
        "video_recorded": bool(args_cli.record_video),
        "duration_s": float(args_cli.duration),
        "fps": int(args_cli.fps),
        "contact_sensor_canary": "not_required_not_run",
        "initialization_only": True,
        **_honesty(root_writes=False),
    }


def main() -> int:
    started = time.monotonic()
    payload: dict[str, Any]
    try:
        torch.manual_seed(int(args_cli.seed))
        np.random.seed(int(args_cli.seed))
        payload = _run_canary() if args_cli.canary_only else _run_stage(int(args_cli.stage))
        payload["elapsed_seconds"] = time.monotonic() - started
        payload["seed"] = int(args_cli.seed)
        payload["command"] = [Path(sys.argv[0]).name, *sys.argv[1:]]
        _write_report(payload)
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True), flush=True)
        return 0 if bool(payload.get("passed")) else 2
    except Exception as exc:
        payload = {
            "kind": "sharpawave_runtime_delivery",
            "stage": args_cli.stage,
            "variant": args_cli.robot_variant,
            "status": "SHARPAWAVE_RUNTIME_BLOCKED" if args_cli.canary_only else "SYSTEM_ONLY_PREVIEW",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "seed": int(args_cli.seed),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            **_honesty(root_writes=bool(args_cli.stage)),
        }
        _write_report(payload)
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    status = main()
    # Isaac Kit can occasionally block indefinitely during shutdown after a
    # short headless stage check. Give it a bounded cleanup window, then exit
    # the isolated validation process so six-stage orchestration is reliable.
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=5.0)
    os._exit(int(status))
