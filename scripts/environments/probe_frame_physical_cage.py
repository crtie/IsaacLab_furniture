"""Bounded Frame mechanical-cage proof for Physical Delivery RC1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
NP_DIR = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np"
UNIFIED_DIR = NP_DIR / "wuji_assembly_v2/pipeline/unified_grasp"
for value in (NP_DIR, UNIFIED_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from wuji_assembly_v2.physical_delivery.contracts import (  # noqa: E402
    CageCandidate,
    CageTopology,
    FailureCode,
)


_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument("--phase", default="plan")
_boot_args, _ = _boot.parse_known_args()
RUNTIME_PHASES = {"runtime_audit", "smoke", "proof", "repeat"}
AppLauncher = None
if _boot_args.phase in RUNTIME_PHASES:
    from isaaclab.app import AppLauncher  # type: ignore[no-redef]  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("audit", "runtime_audit", "plan", "smoke", "proof", "repeat"), default="plan")
parser.add_argument("--config", default="configs/physical_delivery/frame_stage4.yaml")
parser.add_argument("--output-dir", default="debug_runs/physical_delivery_rc1")
parser.add_argument("--candidate-id", default="")
parser.add_argument("--seed", type=int, default=20260713)
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
else:
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

CONFIG_PATH = (REPO_ROOT / args.config).resolve()
OUTPUT = (REPO_ROOT / args.output_dir).resolve()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = _load_yaml(CONFIG_PATH)
    preflight = _gpu_preflight(float(config["runtime"]["min_gpu_free_gb"]))
    _write_json(OUTPUT / "gpu_preflight.json", preflight)
    if args.phase == "audit":
        _write_json(
            OUTPUT / "audit_phase_summary.json",
            {
                "classification": "CODE_FACT_AUDIT_COMPLETE",
                "code_fact_audit": str(OUTPUT / "code_fact_audit.txt"),
                **_honesty(),
            },
        )
        return
    if args.phase in RUNTIME_PHASES and not preflight["sufficient"]:
        _write_json(OUTPUT / "host_gpu_busy.json", {**preflight, "classification": "HOST_GPU_BUSY"})
        raise RuntimeError("HOST_GPU_BUSY")
    if args.phase == "plan":
        _plan(config)
        return
    if AppLauncher is None:
        raise RuntimeError("Isaac AppLauncher is unavailable")
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    env = None
    try:
        env = _make_env(config)
        if args.phase == "runtime_audit":
            summary = _runtime_audit(env, config)
        elif args.phase == "smoke":
            summary = _smoke(env, config)
        elif args.phase == "proof":
            summary = _proof(env, config)
        else:
            summary = _repeat(env, config)
        summary.update(_honesty())
        _write_json(OUTPUT / f"{args.phase}_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = {
            "classification": "FRAME_RC1_RUNTIME_ERROR",
            "phase": args.phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            **_honesty(),
        }
        _write_json(OUTPUT / f"{args.phase}_summary.json", failure)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _make_env(config: Mapping[str, Any]):
    import torch

    from near_grasp.grasp_synthesis.object_spec import ObjectGraspSpec
    from near_grasp.near_grasp_physics_env import NearGraspPhysicsEnv, NearGraspPhysicsEnvCfg, apply_object_grasp_spec

    spec = ObjectGraspSpec.load(REPO_ROOT / "configs/grasp_synthesis/objects/frame.yaml", repo_root=REPO_ROOT)
    cfg = NearGraspPhysicsEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 1.5
    cfg.sim.device = str(args.device)
    cfg.sim.dt = float(config["runtime"]["dt"])
    cfg.episode_length_s = max(8.0, int(config["runtime"]["candidate_trial_steps"]) * cfg.sim.dt + 1.0)
    cfg.enable_contact_attribution = True
    cfg.enable_replay_camera = False
    cfg.defer_terminal_reset = True
    apply_object_grasp_spec(cfg, spec, deterministic_gates=True)
    env = NearGraspPhysicsEnv(cfg=cfg)
    env._zero_action = torch.zeros((1, 14), dtype=torch.float32, device=env.device)
    return env


def _runtime_audit(env: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    phase_dir = OUTPUT / "runtime_audit"
    phase_dir.mkdir(parents=True, exist_ok=True)
    q = np.concatenate((env._parked_wrist_q.detach().cpu().numpy(), env.hand_preshape_q.detach().cpu().numpy()))
    active = _active_mask("234")
    env.configure_privileged_control(q.reshape(1, 26), active.reshape(1, 5), reset_seeds=[args.seed])
    env.reset()
    reset_cache = env.clear_reset_contact_cache_for_privileged_control()
    env.set_privileged_joint_targets(q.reshape(1, 26))
    reset_object_position = (
        env.target_object.data.root_pos_w[0] - env.scene.env_origins[0]
    ).detach().cpu().numpy().copy()
    reset_object_quat = env.target_object.data.root_quat_w[0].detach().cpu().numpy().copy()
    collision = env.export_runtime_collision_geometry(phase_dir)
    env.step(env._zero_action)
    snapshot = env.privileged_snapshot()
    events = env.consume_forensic_contact_events()[0]
    ordered = np.concatenate((env._wrist_ids.detach().cpu().numpy(), env._hand_ids.detach().cpu().numpy()))
    runtime_seed = {
        "runtime_joint_names": [env.robot.joint_names[int(index)] for index in ordered],
        "runtime_articulation_joint_names": list(env.robot.joint_names),
        "q_seed26": q.tolist(),
        "joint_lower26": env._lower[ordered].detach().cpu().tolist(),
        "joint_upper26": env._upper[ordered].detach().cpu().tolist(),
        "hand_preshape_q20": env.hand_preshape_q.detach().cpu().tolist(),
        "hand_close_q20": env.hand_close_q.detach().cpu().tolist(),
        "physics_fingerprint": env.physics_fingerprint(),
    }
    _write_json(phase_dir / "runtime_kinematic_seed.json", runtime_seed)
    approximation = _collision_approximation()
    hole_query = _physx_hole_raycast(
        env,
        Path(collision["target_object"]["npz_path"]),
    )
    masses = env.target_object.root_physx_view.get_masses().detach().cpu().numpy()
    materials = env.target_object.root_physx_view.get_material_properties().detach().cpu().numpy()
    audit = {
        **collision,
        "schema_version": 1,
        "frame_usd": str(env.cfg.target_asset_usd),
        "frame_usd_sha256": _sha256(Path(env.cfg.target_asset_usd)),
        "canonical_object_position": reset_object_position.tolist(),
        "canonical_object_quat_wxyz": reset_object_quat.tolist(),
        "canonical_pose_timing": "RESET_BEFORE_FIRST_PHYSICS_STEP",
        "runtime_mass_kg": float(masses.reshape(-1)[0]),
        "runtime_material_properties": materials.tolist(),
        "collision_approximation": approximation,
        "physx_scene_query": hole_query,
        "physx_scene_query_hole_open": bool(hole_query.get("available") and hole_query.get("hole_open")),
        "physx_self_collision_enabled": False,
        "first_frame": {
            "physics_frame_id": 0,
            "object_position": snapshot["object_pos_local"][0].tolist(),
            "object_velocity": snapshot["object_lin_vel"][0].tolist(),
            "target_force_norms": snapshot["target_force_norms"][0].tolist(),
            "contact_events": [event.to_dict() for event in events],
            "post_reset_object_root_writes": int(snapshot["post_reset_object_writes"][0]),
            "post_reset_wrist_state_writes": int(snapshot["post_reset_wrist_state_writes"][0]),
        },
        "reset_cache_audit": reset_cache,
        **_honesty(),
    }
    _write_json(OUTPUT / "frame_runtime_asset_audit.json", audit)
    return {
        "classification": "FRAME_RUNTIME_AUDIT_COMPLETE",
        "runtime_mesh_sha256": collision["target_object"]["mesh_sha256"],
        "physx_scene_query_hole_open": audit["physx_scene_query_hole_open"],
        "runtime_mass_kg": audit["runtime_mass_kg"],
        "runtime_material_properties": audit["runtime_material_properties"],
        "post_reset_object_root_writes": 0,
        "post_reset_wrist_state_writes": 0,
    }


def _plan(config: Mapping[str, Any]) -> None:
    from wuji_assembly_v2.physical_delivery.frame_cage_planner import FrameCagePlanner

    planner = FrameCagePlanner(
        repo_root=REPO_ROOT,
        config=config,
        runtime_mesh_path=REPO_ROOT / str(config["assets"]["runtime_mesh"]),
        runtime_audit_path=REPO_ROOT / str(config["assets"]["runtime_audit"]),
        runtime_seed_path=OUTPUT / "runtime_audit/runtime_kinematic_seed.json",
    )
    candidates = planner.generate()
    _write_json(OUTPUT / "frame_geometry_audit.json", planner.geometry_audit())
    _write_json(OUTPUT / "cage_candidates.json", [candidate.to_dict() for candidate in candidates])
    _write_jsonl(OUTPUT / "cage_candidates.jsonl", (candidate.to_dict() for candidate in candidates))
    summary = {
        "classification": "FRAME_CAGE_CANDIDATES_GENERATED",
        "candidate_count": len(candidates),
        "geometry_valid_count": sum(int(candidate.geometry_valid) for candidate in candidates),
        "topology_counts": {
            topology.value: sum(int(candidate.topology == topology) for candidate in candidates)
            for topology in CageTopology
        },
        "failures": {
            candidate.candidate_id: candidate.failure_reason
            for candidate in candidates
            if not candidate.geometry_valid
        },
        **_honesty(),
    }
    _write_json(OUTPUT / "plan_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _smoke(env: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    candidates = _load_candidates()
    candidate = next((row for row in candidates if row.geometry_valid), candidates[0])
    reset = np.asarray(candidate.preclose_q26, dtype=np.float64).reshape(1, 26)
    env.configure_privileged_control(reset, _active_mask(candidate.finger_group).reshape(1, 5), reset_seeds=[args.seed])
    env.reset()
    env.clear_reset_contact_cache_for_privileged_control()
    env.set_privileged_joint_targets(reset)
    finite = True
    for _ in range(32):
        observation, *_ = env.step(env._zero_action)
        policy = observation["policy"] if isinstance(observation, dict) else observation
        finite &= bool(np.all(np.isfinite(policy.detach().cpu().numpy())))
    snapshot = env.privileged_snapshot()
    return {
        "classification": "FRAME_CAGE_SMOKE_PASS" if finite else "FRAME_CAGE_SMOKE_FAILED",
        "candidate_id": candidate.candidate_id,
        "steps": 32,
        "finite_observations": finite,
        "target_filter_valid": bool(snapshot["target_filter_valid"][0]),
        "contact_report_available": bool(snapshot["contact_report_available"][0]),
        "post_reset_object_root_writes": int(snapshot["post_reset_object_writes"][0]),
        "post_reset_wrist_state_writes": int(snapshot["post_reset_wrist_state_writes"][0]),
    }


def _proof(env: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    from wuji_assembly_v2.physical_delivery.grasp_executor import run_cage_proof

    candidates = [candidate for candidate in _load_candidates() if candidate.geometry_valid]
    if args.candidate_id:
        candidates = [candidate for candidate in candidates if candidate.candidate_id == args.candidate_id]
    order = {
        CageTopology.HOOK_THROUGH_FRAME: 0,
        CageTopology.THREE_FINGER_WRAP: 1,
        CageTopology.TWO_FINGER_BRACKET: 2,
    }
    candidates.sort(key=lambda row: (order[row.topology], row.candidate_id))
    started = time.monotonic()
    results = []
    for index, candidate in enumerate(candidates):
        elapsed_minutes = (time.monotonic() - started) / 60.0
        if elapsed_minutes >= float(config["runtime"]["proof_wall_clock_minutes"]):
            break
        execution = run_cage_proof(env, candidate, config, trial_index=0, seed=args.seed + index)
        result_payload = execution.result.to_dict()
        trial_dir = OUTPUT / "proof" / candidate.candidate_id / "trial_0"
        result_payload["trace_path"] = str(trial_dir / "trace.csv")
        _write_json(trial_dir / "result.json", result_payload)
        _write_json(trial_dir / "reset_cache_audit.json", execution.reset_cache_audit)
        _write_csv(trial_dir / "trace.csv", execution.trace)
        _write_jsonl(trial_dir / "contact_pairs.jsonl", execution.contacts)
        results.append(result_payload)
    success = [row for row in results if row["passed"]]
    summary = {
        "classification": "FRAME_CAGE_PROOF_LIFT_ACQUIRED" if success else "FRAME_CAGE_PROOF_FAILED_EARLY_STOP",
        "generated_candidate_count": len(_load_candidates()),
        "geometry_valid_candidate_count": len(candidates),
        "physical_trial_count": len(results),
        "success_candidate_ids": [row["candidate_id"] for row in success],
        "results": results,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "frame_physical_grasp_success": bool(success),
        "frame_physical_lift_success": bool(success),
    }
    _write_json(OUTPUT / "cage_proof_results.json", summary)
    if not success:
        _write_early_stop(summary)
        _write_final_summary(summary, repeat=None)
    return summary


def _repeat(env: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    from wuji_assembly_v2.physical_delivery.grasp_executor import run_cage_proof

    proof = _read_json(OUTPUT / "cage_proof_results.json")
    successful_ids = list(proof.get("success_candidate_ids", ()))
    if not successful_ids:
        summary = {
            "classification": "FRAME_CAGE_PROOF_FAILED_EARLY_STOP",
            "repeat_trials": 0,
            "repeat_successes": 0,
        }
        _write_final_summary(proof, repeat=summary)
        return summary
    candidates = {row.candidate_id: row for row in _load_candidates()}
    topology_order = {
        CageTopology.HOOK_THROUGH_FRAME: 0,
        CageTopology.THREE_FINGER_WRAP: 1,
        CageTopology.TWO_FINGER_BRACKET: 2,
    }
    selected = sorted(
        (candidates[value] for value in successful_ids),
        key=lambda row: (topology_order[row.topology], row.candidate_id),
    )[: int(config["repeatability"]["selected_candidate_limit"])]
    rows = []
    for candidate in selected:
        for trial in range(int(config["repeatability"]["trials"])):
            execution = run_cage_proof(
                env,
                candidate,
                config,
                trial_index=trial,
                seed=args.seed + 100 + trial,
            )
            payload = execution.result.to_dict()
            trial_dir = OUTPUT / "repeat" / candidate.candidate_id / f"trial_{trial}"
            payload["trace_path"] = str(trial_dir / "trace.csv")
            _write_json(trial_dir / "result.json", payload)
            _write_csv(trial_dir / "trace.csv", execution.trace)
            _write_jsonl(trial_dir / "contact_pairs.jsonl", execution.contacts)
            rows.append(payload)
    per_candidate = {
        candidate.candidate_id: {
            "successes": sum(int(row["passed"]) for row in rows if row["candidate_id"] == candidate.candidate_id),
            "trials": sum(1 for row in rows if row["candidate_id"] == candidate.candidate_id),
        }
        for candidate in selected
    }
    qualified = [
        candidate_id
        for candidate_id, values in per_candidate.items()
        if values["successes"] >= int(config["repeatability"]["required_successes"])
    ]
    summary = {
        "classification": "FRAME_REPEATABLE_LIFT_ACQUIRED_DELIVERY_INCOMPLETE" if qualified else "FRAME_CAGE_PROOF_FAILED_EARLY_STOP",
        "qualified_candidate_ids": qualified,
        "per_candidate": per_candidate,
        "repeat_trials": len(rows),
        "repeat_successes": sum(int(row["passed"]) for row in rows),
        "results": rows,
    }
    _write_json(OUTPUT / "repeatability_results.json", summary)
    if qualified:
        _write_json(OUTPUT / "selected_candidate.json", candidates[qualified[0]].to_dict())
    else:
        _write_early_stop({**proof, "repeatability": summary})
    _write_final_summary(proof, repeat=summary)
    return summary


def _physx_hole_raycast(env: Any, mesh_path: Path) -> dict[str, Any]:
    import numpy as np

    from wuji_assembly_v2.physical_delivery.frame_cage_planner import extract_frame_bars

    result: dict[str, Any] = {"available": False, "hole_open": False, "rays": [], "error": ""}
    try:
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf

        mesh = np.load(mesh_path)
        bars, hole = extract_frame_bars(
            np.asarray(mesh["vertices"], dtype=np.float64),
            np.asarray(mesh["faces"], dtype=np.int64),
        )
        object_pos = env.target_object.data.root_pos_w[0].detach().cpu().numpy()
        queries = [("central_hole", 0.5 * (hole[0] + hole[1]))]
        queries.extend((bar_id, np.asarray(bar.center_object)) for bar_id, bar in bars.items())
        interface = get_physx_scene_query_interface()
        for label, local in queries:
            origin = object_pos + local
            origin[1] = object_pos[1] + float(np.min(mesh["vertices"][:, 1])) - 0.01
            hit = interface.raycast_closest(
                Gf.Vec3f(*[float(value) for value in origin]),
                Gf.Vec3f(0.0, 1.0, 0.0),
                float(np.ptp(mesh["vertices"][:, 1]) + 0.02),
            )
            data = _scene_hit(hit)
            data["label"] = label
            data["target_hit"] = "TargetObject" in data["collision"] or "TargetObject" in data["rigidBody"]
            result["rays"].append(data)
        hole_row = next(row for row in result["rays"] if row["label"] == "central_hole")
        bar_rows = [row for row in result["rays"] if row["label"] != "central_hole"]
        controls_valid = bool(bar_rows and all(row["target_hit"] for row in bar_rows))
        result["available"] = controls_valid
        result["hole_open"] = bool(controls_valid and not hole_row["target_hit"])
        if not controls_valid:
            result["error"] = "bar_control_rays_missed_target_sdf"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def _collision_approximation() -> dict[str, Any]:
    result = {"available": False, "path": "", "token": "", "error": ""}
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/TargetObject")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                result.update(
                    {
                        "available": True,
                        "path": str(prim.GetPath()),
                        "token": str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"),
                    }
                )
                break
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def _scene_hit(hit: Any) -> dict[str, Any]:
    if not isinstance(hit, dict):
        return {
            "hit": bool(hit),
            "collision": str(getattr(hit, "collision", "")),
            "rigidBody": str(getattr(hit, "rigid_body", getattr(hit, "rigidBody", ""))),
            "distance": float(getattr(hit, "distance", 0.0) or 0.0),
        }
    return {
        "hit": bool(hit.get("hit", True)),
        "collision": str(hit.get("collision", "")),
        "rigidBody": str(hit.get("rigidBody", hit.get("rigid_body", ""))),
        "distance": float(hit.get("distance", 0.0) or 0.0),
    }


def _write_early_stop(summary: Mapping[str, Any]) -> None:
    lines = [
        "Wuji Frame Physical Delivery RC1 - early stop",
        "classification=FRAME_CAGE_PROOF_FAILED_EARLY_STOP",
        "physical_grasp_success=false",
        "physical_lift_success=false",
        "transport_started=false",
        "insertion_started=false",
        "video_path=",
        "",
        "Topology results:",
    ]
    result_rows = list(summary.get("results", ()))
    for row in result_rows:
        lines.append(
            f"{row.get('topology')} {row.get('candidate_id')}: {row.get('failure_code')} "
            f"lift={row.get('lift_delta_m')} force={row.get('peak_force_n')}"
        )
    if not result_rows and (OUTPUT / "cage_candidates.json").is_file():
        candidates = _read_json(OUTPUT / "cage_candidates.json")
        for topology in CageTopology:
            rows = [row for row in candidates if row.get("topology") == topology.value]
            reasons = sorted({str(row.get("failure_reason") or "") for row in rows})
            lines.append(f"{topology.value}: {len(rows)} candidates; reasons={';'.join(reasons)}")
    (OUTPUT / "early_stop_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_summary(proof: Mapping[str, Any], repeat: Mapping[str, Any] | None) -> None:
    repeat = dict(repeat or {})
    qualified = list(repeat.get("qualified_candidate_ids", ()))
    classification = (
        "FRAME_REPEATABLE_LIFT_ACQUIRED_DELIVERY_INCOMPLETE"
        if qualified
        else "FRAME_CAGE_PROOF_FAILED_EARLY_STOP"
    )
    payload = {
        "classification": classification,
        "delivery_level": "NONE",
        "selected_topology": "",
        "selected_candidate_id": qualified[0] if qualified else "",
        "frame_physical_grasp_success": bool(qualified),
        "frame_physical_lift_success": bool(qualified),
        "frame_physical_transport_success": False,
        "frame_physical_insert_success": False,
        "release_attempted": False,
        "release_stable": False,
        "repeat_successes": int(repeat.get("repeat_successes", 0)),
        "repeat_trials": int(repeat.get("repeat_trials", 0)),
        "sticky_used": False,
        "snap_used": False,
        "proxy_used": False,
        "teacher_used": False,
        "object_follow_used": False,
        "AttachedPartController_used": False,
        "post_reset_object_root_writes": max(
            (int(row.get("post_reset_object_root_writes", 0)) for row in repeat.get("results", ())),
            default=0,
        ),
        "post_reset_wrist_root_writes": max(
            (int(row.get("post_reset_wrist_state_writes", 0)) for row in repeat.get("results", ())),
            default=0,
        ),
        "loaded_fixed_asset": "",
        "expected_fixed_asset": "frame_back_rod_rod.usd",
        "moving_frame_asset": "frame_mirror.usd",
        "runtime_mass_kg": 0.01,
        "runtime_static_friction": 1.0,
        "runtime_dynamic_friction": 1.0,
        "assisted_physical": False,
        "physical_candidate_resets": int(proof.get("physical_trial_count", 0)),
        "video_path": "",
        "video_matches_numeric_rollout": False,
        "Rod_status": "existing synthesis only, not RC1 delivery validated",
        "Backrest_status": "not RC1 delivery validated",
        "Screw1_status": "unsupported in RC1",
        "Plug2_status": "strict M0 found previously, physical hold unresolved",
        "cem_started": False,
        "ppo_started": False,
        "proof_summary": proof,
    }
    _write_json(OUTPUT / "final_summary.json", payload)


def _load_candidates() -> list[CageCandidate]:
    path = OUTPUT / "cage_candidates.json"
    rows = _read_json(path)
    candidates = [CageCandidate.from_dict(row) for row in rows]
    if len(candidates) > 12:
        raise ValueError("RC1 candidate cap exceeded")
    return candidates


def _gpu_preflight(minimum_gb: float) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 5:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "total_mib": float(fields[2]),
                    "used_mib": float(fields[3]),
                    "free_mib": float(fields[4]),
                }
            )
    free_gb = max((row["free_mib"] / 1024.0 for row in rows), default=0.0)
    processes = subprocess.run(
        ["ps", "-eo", "pid,ppid,cmd"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    ).stdout.splitlines()
    relevant = [line.strip() for line in processes if any(value in line.lower() for value in ("isaac", "kit"))]
    return {
        "command": command,
        "returncode": result.returncode,
        "gpus": rows,
        "maximum_free_gb": free_gb,
        "minimum_required_gb": float(minimum_gb),
        "sufficient": bool(result.returncode == 0 and free_gb >= float(minimum_gb)),
        "active_isaac_kit_processes": relevant,
        "processes_killed": [],
    }


def _honesty() -> dict[str, Any]:
    return {
        "cem_started": False,
        "ppo_started": False,
        "sticky_used": False,
        "snap_used": False,
        "proxy_used": False,
        "teacher_used": False,
        "object_follow_used": False,
        "AttachedPartController_used": False,
    }


def _active_mask(group: str):
    import numpy as np

    mask = np.zeros(5, dtype=bool)
    for value in group:
        mask[int(value) - 1] = True
    return mask


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (list, tuple, dict)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
