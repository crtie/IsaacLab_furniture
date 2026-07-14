"""Run the dedicated Screw1 near-grasp vector smoke, mixed CEM, replay, or PPO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_GRASP_PARENT = (
    REPO_ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
)
if str(NEAR_GRASP_PARENT) not in sys.path:
    sys.path.insert(0, str(NEAR_GRASP_PARENT))

from near_grasp.configuration import NearGraspRunConfig  # noqa: E402


_config_parser = argparse.ArgumentParser(add_help=False)
_config_parser.add_argument(
    "--config",
    default=str(REPO_ROOT / "configs/near_grasp/screw1_cem_v1.yaml"),
)
_config_args, _ = _config_parser.parse_known_args()
BASE_CONFIG = NearGraspRunConfig.load(_config_args.config)
_base = BASE_CONFIG.values


from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=_config_args.config)
parser.add_argument("--mode", choices=("vector_smoke", "cem", "replay", "ppo", "full"), default="vector_smoke")
parser.add_argument("--output_dir", default="debug_runs/screw1_near_grasp_cem")
parser.add_argument("--num_envs", type=int, default=int(_base["scene"]["num_envs"]))
parser.add_argument("--population", type=int, default=int(_base["cem"]["population"]))
parser.add_argument("--elite_count", type=int, default=int(_base["cem"]["elite_count"]))
parser.add_argument("--min_generations", type=int, default=int(_base["cem"]["min_generations"]))
parser.add_argument("--max_generations", type=int, default=int(_base["cem"]["max_generations"]))
parser.add_argument("--seed", type=int, default=int(_base["cem"]["seed"]))
parser.add_argument("--prior", choices=("auto", "coordex", "pca"), default=str(_base["prior"]["requested"]))
parser.add_argument("--min_gpu_free_gb", type=float, default=6.0)
parser.add_argument("--smoke_steps", type=int, default=10)
parser.add_argument("--max_episode_steps", type=int, default=int(_base["scene"]["episode_steps"]))
parser.add_argument("--record_first_lift", action="store_true")
parser.add_argument("--top_programs", default="")
parser.add_argument("--ppo_iterations", type=int, default=1000)
parser.add_argument("--skip_gpu_preflight", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

RUN_CONFIG = BASE_CONFIG.with_overrides(
    {
        "scene": {"num_envs": int(args_cli.num_envs), "episode_steps": int(args_cli.max_episode_steps)},
        "cem": {
            "population": int(args_cli.population),
            "physical_batch": int(args_cli.num_envs),
            "elite_count": int(args_cli.elite_count),
            "min_generations": int(args_cli.min_generations),
            "max_generations": int(args_cli.max_generations),
            "seed": int(args_cli.seed),
        },
        "prior": {"requested": str(args_cli.prior)},
    }
)


OUTPUT_DIR = (REPO_ROOT / args_cli.output_dir).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GPU_PREFLIGHT = _gpu_preflight = None


def _run_gpu_preflight() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    gpus = []
    for line in query.stdout.splitlines() if query.returncode == 0 else []:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 5:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": float(fields[2]),
                    "memory_used_mib": float(fields[3]),
                    "memory_free_mib": float(fields[4]),
                }
            )
    processes = []
    for line in process_query.stdout.splitlines() if process_query.returncode == 0 else []:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            processes.append(
                {"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": float(fields[2])}
            )
    free_mib = max([0.0, *[row["memory_free_mib"] for row in gpus]])
    blocked = bool(query.returncode != 0 or free_mib < float(args_cli.min_gpu_free_gb) * 1024.0)
    return {
        "classification": "HOST_GPU_BUSY" if blocked else "HEALTH_OK",
        "blocked": blocked,
        "minimum_free_memory_gb": float(args_cli.min_gpu_free_gb),
        "gpus": gpus,
        "active_compute_processes": processes,
        "nvidia_smi_return_code": query.returncode,
        "nvidia_smi_output": query.stdout[-4000:],
        "process_query_output": process_query.stdout[-4000:],
        "processes_killed": False,
    }


GPU_PREFLIGHT = _run_gpu_preflight()
(OUTPUT_DIR / "gpu_preflight.json").write_text(
    json.dumps(GPU_PREFLIGHT, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if GPU_PREFLIGHT["blocked"] and not args_cli.skip_gpu_preflight:
    print(json.dumps(GPU_PREFLIGHT, indent=2, sort_keys=True))
    raise SystemExit(2)


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402

from near_grasp.cem import (  # noqa: E402
    CandidateResult,
    MixedCem,
    MixedCemConfig,
)
from near_grasp.evaluator import (  # noqa: E402
    PhysicalEvaluation,
)
from near_grasp.grasp_program import (  # noqa: E402
    GraspProgram,
    GraspTemplate,
    grasp_templates_from_config,
)
from near_grasp.hand_prior_adapter import (  # noqa: E402
    CoorDexWujiPriorAdapter,
    RetargetedPcaPriorAdapter,
    save_pca_artifact,
)
from near_grasp.near_grasp_physics_env import (  # noqa: E402
    NearGraspPhysicsEnv,
    NearGraspPhysicsEnvCfg,
    apply_near_grasp_run_config,
)
from near_grasp.residual_rl import (  # noqa: E402
    EvaluationRates,
    ResidualRlGate,
    make_rsl_rl_ppo_config,
)
from near_grasp.run_manifest import RunManifest  # noqa: E402


def main() -> None:
    env = None
    _write_json(OUTPUT_DIR / "runner_heartbeat.json", {"state": "MAIN_ENTERED"})
    try:
        _write_json(OUTPUT_DIR / "runner_heartbeat.json", {"state": "CREATING_ENV"})
        env = _make_env()
        _write_json(OUTPUT_DIR / "runner_heartbeat.json", {"state": "ENV_CREATED", "num_envs": env.num_envs})
        prior, prior_status = _make_prior(env)
        env.configure_prior(prior)
        _write_json(OUTPUT_DIR / "prior_status.json", prior_status)
        _write_json(OUTPUT_DIR / "physics_fingerprint.json", env.physics_fingerprint())
        resolved_path = RUN_CONFIG.write(OUTPUT_DIR / "resolved_config.yaml")
        _write_run_manifest(resolved_path, prior_status)
        if args_cli.mode == "vector_smoke":
            result = _run_vector_smoke(env)
        elif args_cli.mode in {"cem", "full"}:
            result = _run_cem(env)
            if args_cli.mode == "full" and result.get("replayable_lift"):
                result["ppo"] = _run_ppo(env, result)
        elif args_cli.mode == "replay":
            result = _run_replay_from_artifact(env)
        else:
            result = _run_ppo_from_artifacts(env)
        result.update(_status_fields(result))
        _write_json(OUTPUT_DIR / "near_grasp_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = {
            "classification": "NEAR_GRASP_ENV_VECTORIZATION_BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "implementation_error": True,
        }
        _write_json(OUTPUT_DIR / "runner_implementation_error.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr, flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _write_run_manifest(resolved_path: Path, prior_status: dict[str, Any]) -> None:
    import isaaclab

    source_paths = [Path(__file__).resolve(), *sorted(NEAR_GRASP_PARENT.joinpath("near_grasp").glob("*.py"))]
    assets = [REPO_ROOT / value for value in RUN_CONFIG.values["assets"].values()]
    priors = [OUTPUT_DIR / "retargeted_wuji_pca6.json"]
    coordex = REPO_ROOT / str(RUN_CONFIG.values["prior"]["coordex_checkpoint"])
    if coordex.is_file():
        priors.append(coordex)
    manifest = RunManifest.capture(
        repo_root=REPO_ROOT,
        run_id=OUTPUT_DIR.name,
        command=[sys.executable, *sys.argv],
        config_path=resolved_path,
        config_sha256=RUN_CONFIG.sha256,
        source_paths=source_paths,
        asset_paths=assets,
        prior_paths=priors,
        runtime={
            "isaaclab_module": str(Path(isaaclab.__file__).resolve()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "gpu_preflight": GPU_PREFLIGHT,
            "prior_selected": prior_status.get("selected", ""),
        },
    )
    manifest.write(OUTPUT_DIR / "run_manifest.json")


def _make_env() -> NearGraspPhysicsEnv:
    cfg = NearGraspPhysicsEnvCfg()
    cfg.seed = int(args_cli.seed)
    apply_near_grasp_run_config(
        cfg,
        RUN_CONFIG.values,
        num_envs=int(args_cli.num_envs),
        device=str(args_cli.device),
        episode_steps=int(args_cli.max_episode_steps),
    )
    return NearGraspPhysicsEnv(cfg=cfg, render_mode="rgb_array" if args_cli.record_first_lift else None)


def _configured_templates() -> tuple[GraspTemplate, ...]:
    return grasp_templates_from_config(RUN_CONFIG.values["templates"])


def _make_prior(env: NearGraspPhysicsEnv):
    artifact = RetargetedPcaPriorAdapter.build_runtime_fallback(
        env.hand_preshape_q.detach().cpu().numpy(),
        env.hand_close_q.detach().cpu().numpy(),
    )
    artifact_path = save_pca_artifact(artifact, OUTPUT_DIR / "retargeted_wuji_pca6.json")
    isolated_smoke_path = REPO_ROOT / "debug_runs" / "external_baseline_audit" / "coordex_isolated_runtime_smoke.json"
    isolated_smoke = _read_json(isolated_smoke_path) if isolated_smoke_path.is_file() else {}
    isolated_pass = isolated_smoke.get("status") in {"PASS", "ISOLATED_RUNTIME_SMOKE_PASS"}
    checkpoint = REPO_ROOT / str(RUN_CONFIG.values["prior"]["coordex_checkpoint"])
    coordex_error = ""
    if args_cli.prior in {"auto", "coordex"} and isolated_pass:
        try:
            adapter = CoorDexWujiPriorAdapter(
                checkpoint,
                joint_lower=env._lower[env._hand_ids].detach().cpu().numpy(),
                joint_upper=env._upper[env._hand_ids].detach().cpu().numpy(),
                default_q=env.robot.data.default_joint_pos[0, env._hand_ids].detach().cpu().numpy(),
                device=str(env.device),
            )
            return adapter, {
                "selected": "coordex",
                "signature": adapter.signature,
                "isolated_smoke": isolated_smoke,
                "fallback_artifact": str(artifact_path),
            }
        except Exception as exc:
            coordex_error = f"{type(exc).__name__}:{exc}"
    elif args_cli.prior == "coordex":
        coordex_error = "isolated Isaac Sim 5.0 / IsaacLab 2.2 smoke has not passed"
    adapter = RetargetedPcaPriorAdapter(
        artifact,
        joint_lower=env._lower[env._hand_ids].detach().cpu().numpy(),
        joint_upper=env._upper[env._hand_ids].detach().cpu().numpy(),
    )
    return adapter, {
        "selected": "retargeted_pca6",
        "signature": adapter.signature,
        "artifact": str(artifact_path),
        "coordex_requested": args_cli.prior in {"auto", "coordex"},
        "coordex_isolated_smoke_passed": isolated_pass,
        "coordex_fallback_reason": coordex_error or "isolated smoke not available",
        "other_hand_recording_angles_used": False,
    }


def _run_vector_smoke(env: NearGraspPhysicsEnv) -> dict[str, Any]:
    templates = _configured_templates()
    values = np.zeros(16, dtype=np.float64)
    values[12:] = (80.0, 0.75, 24.0, 24.0)
    programs = [GraspProgram.from_vector(values, index % len(templates)) for index in range(env.num_envs)]
    seeds = [args_cli.seed + index for index in range(env.num_envs)]
    env.set_program_batch(programs, templates, reset_seeds=seeds)
    obs, _ = env.reset()
    reset_shape = list(obs["policy"].shape)
    finite = bool(torch.all(torch.isfinite(obs["policy"])).item())
    force_valid_seen = False
    for _ in range(int(args_cli.smoke_steps)):
        obs, _, _, _, _ = env.step(torch.zeros((env.num_envs, 14), device=env.device))
        finite = finite and bool(torch.all(torch.isfinite(obs["policy"])).item())
        force_valid_seen = force_valid_seen or bool(torch.any(env._target_force_valid).item())
    passed = bool(reset_shape == [env.num_envs, 169] and finite and env.num_envs >= 16)
    result = {
        "mode": "vector_smoke",
        "classification": "NEAR_GRASP_ENV_VECTORIZATION_PASS" if passed else "NEAR_GRASP_ENV_VECTORIZATION_BLOCKED",
        "passed": passed,
        "num_envs": env.num_envs,
        "required_min_envs": 16,
        "steps": int(args_cli.smoke_steps),
        "observation_shape": reset_shape,
        "observations_finite": finite,
        "target_force_sensor_valid_seen": force_valid_seen,
        "reset_only_object_writes": True,
        "post_reset_object_writes": 0,
        "post_reset_wrist_state_writes": 0,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "not_physical": True,
    }
    _write_json(OUTPUT_DIR / "vector_smoke.json", result)
    return result


def _run_cem(env: NearGraspPhysicsEnv) -> dict[str, Any]:
    templates = _configured_templates()
    config = MixedCemConfig(
        population=int(args_cli.population),
        physical_batch=int(args_cli.num_envs),
        elite_count=int(args_cli.elite_count),
        categorical_floor=float(RUN_CONFIG.values["cem"]["categorical_floor"]),
        min_generations=int(args_cli.min_generations),
        max_generations=int(args_cli.max_generations),
        seed=int(args_cli.seed),
    )
    cem = MixedCem(len(templates), config=config)
    all_results: list[CandidateResult] = []
    metrics_path = OUTPUT_DIR / "cem_generation_metrics.csv"
    candidates_path = OUTPUT_DIR / "cem_candidate_results.jsonl"
    candidates_path.write_text("", encoding="utf-8")
    first_lift: CandidateResult | None = None
    first_lift_trace: dict[str, np.ndarray] | None = None
    for generation in range(config.max_generations):
        programs = cem.sample()
        generation_results: list[CandidateResult] = []
        for start in range(0, config.population, env.num_envs):
            chunk = programs[start : start + env.num_envs]
            if len(chunk) < env.num_envs:
                chunk.extend(programs[: env.num_envs - len(chunk)])
            candidate_ids = [generation * config.population + start + index for index in range(env.num_envs)]
            seeds = [args_cli.seed + 1_000_003 * generation + start + index for index in range(env.num_envs)]
            evaluations, traces = _run_physics_batch(env, chunk, templates, seeds, candidate_ids)
            for local_index, program in enumerate(chunk[: min(env.num_envs, config.population - start)]):
                candidate_id = candidate_ids[local_index]
                evaluation = evaluations.get(candidate_id, _runtime_failure_evaluation(candidate_id))
                result = CandidateResult(candidate_id, program, evaluation, seeds[local_index], _fingerprint_hash(env))
                generation_results.append(result)
                all_results.append(result)
                _append_jsonl(candidates_path, result.to_dict())
                if first_lift is None and evaluation.physical_lift_success:
                    first_lift = result
                    first_lift_trace = traces.get(candidate_id)
                    _freeze_first_lift(result, env, first_lift_trace)
        row = cem.update(generation_results)
        _append_csv(metrics_path, row)
        _write_top_programs(all_results)
        if first_lift is not None and cem.generation >= config.min_generations:
            break
        if cem.should_stop:
            break
    replay = _replay_first_lift(env, first_lift, templates) if first_lift is not None else {}
    stable_close_count = sum(int(item.evaluation.stable_close) for item in all_results)
    target_contact_count = sum(int(item.evaluation.target_filtered_success_evidence) for item in all_results)
    hard_invalid_count = sum(int(item.evaluation.hard_invalid) for item in all_results)
    invalid_reason_counts: dict[str, int] = {}
    for item in all_results:
        for reason in item.evaluation.invalid_reasons:
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
    classification = (
        "FIRST_PHYSICAL_SCREW1_LIFT_ACQUIRED"
        if first_lift is not None
        else "GRASP_PROGRAM_PARAMETERIZATION_INSUFFICIENT"
        if cem.parameterization_insufficient
        else "NEAR_GRASP_SEARCH_INCOMPLETE"
    )
    return {
        "mode": "cem",
        "classification": classification,
        "generations_completed": cem.generation,
        "episodes_completed": len(all_results),
        "population": config.population,
        "physical_batch": config.physical_batch,
        "elite_count": config.elite_count,
        "first_physical_lift": first_lift.to_dict() if first_lift else None,
        "replay": replay,
        "replayable_lift": bool(replay.get("successful_replay_count", 0) > 0),
        "safe_elite_count": sum(
            int(item.evaluation.valid_candidate and item.evaluation.peak_target_force_n < 1.0)
            for item in MixedCem.elites(all_results, min(config.elite_count, len(all_results)))
        ),
        "parameterization_insufficient": cem.parameterization_insufficient,
        "stable_close_count": stable_close_count,
        "target_filtered_contact_candidate_count": target_contact_count,
        "hard_invalid_count": hard_invalid_count,
        "invalid_reason_counts": invalid_reason_counts,
        "physical_grasp_success": bool(stable_close_count > 0 or first_lift is not None),
        "physical_lift_success": bool(first_lift is not None),
        "post_reset_object_writes": 0,
        "post_reset_wrist_state_writes": 0,
        "not_physical": bool(first_lift is None),
        "ppo_started": False,
    }


def _run_physics_batch(
    env: NearGraspPhysicsEnv,
    programs: Sequence[GraspProgram],
    templates,
    seeds: Sequence[int],
    candidate_ids: Sequence[int],
    *,
    capture_video: Path | None = None,
):
    env.set_program_batch(programs, templates, reset_seeds=seeds, candidate_ids=candidate_ids)
    env.reset()
    frames = []
    expected = set(int(value) for value in candidate_ids)
    for step in range(int(args_cli.max_episode_steps)):
        env.step(torch.zeros((env.num_envs, 14), dtype=torch.float32, device=env.device))
        if step % 25 == 0:
            _write_json(
                OUTPUT_DIR / "physics_batch_heartbeat.json",
                {
                    "state": "PHYSICS_BATCH_RUNNING",
                    "step": step,
                    "max_episode_steps": int(args_cli.max_episode_steps),
                    "candidate_id_min": min(expected),
                    "candidate_id_max": max(expected),
                    "completed_candidates": len(expected.intersection(env._completed)),
                    "expected_candidates": len(expected),
                },
            )
        if capture_video is not None and step % 2 == 0:
            frame = env.render()
            if isinstance(frame, np.ndarray) and frame.size:
                frames.append(frame.copy())
        if expected.issubset(env._completed):
            break
    _write_json(
        OUTPUT_DIR / "physics_batch_heartbeat.json",
        {
            "state": "PHYSICS_BATCH_FINISHED",
            "step": step,
            "max_episode_steps": int(args_cli.max_episode_steps),
            "candidate_id_min": min(expected),
            "candidate_id_max": max(expected),
            "completed_candidates": len(expected.intersection(env._completed)),
            "expected_candidates": len(expected),
        },
    )
    evaluations = env.consume_completed_evaluations()
    traces = env.consume_completed_traces()
    if capture_video is not None and frames:
        _write_video(capture_video, frames, fps=60)
    return evaluations, traces


def _freeze_first_lift(
    result: CandidateResult,
    env: NearGraspPhysicsEnv,
    trace: dict[str, np.ndarray] | None,
) -> None:
    payload = {
        "run_id": f"cem_candidate_{result.candidate_id}",
        "program": result.program.to_dict(),
        "reset_seed": result.reset_seed,
        "physics_fingerprint": env.physics_fingerprint(),
        "physics_fingerprint_sha256": _fingerprint_hash(env),
        "evaluation": result.evaluation.to_dict(),
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": True,
        "root_pose_writes_reset_only": True,
        "physical_grasp_success": result.evaluation.honesty.get("physical_grasp_success", False),
        "physical_lift_success": True,
        "physical_insert_success": False,
        "oracle_visual_only": False,
        "not_physical": False,
    }
    _write_json(OUTPUT_DIR / "first_physical_lift_summary.json", payload)
    if trace:
        _write_trace_csv(OUTPUT_DIR / "first_physical_lift_trace.csv", trace)


def _replay_first_lift(env: NearGraspPhysicsEnv, first_lift: CandidateResult, templates) -> dict[str, Any]:
    programs = [first_lift.program for _ in range(env.num_envs)]
    seeds = [first_lift.reset_seed for _ in range(env.num_envs)]
    candidate_ids = list(range(10_000_000, 10_000_000 + env.num_envs))
    video_path = OUTPUT_DIR / "first_successful_screw1_grasp.mp4" if args_cli.record_first_lift else None
    evaluations, traces = _run_physics_batch(
        env,
        programs,
        templates,
        seeds,
        candidate_ids,
        capture_video=video_path,
    )
    first_three = [evaluations.get(candidate_id) for candidate_id in candidate_ids[:3]]
    success_count = sum(int(value is not None and value.physical_lift_success) for value in first_three)
    successful_id = next(
        (candidate_ids[index] for index, value in enumerate(first_three) if value and value.physical_lift_success),
        None,
    )
    if successful_id is not None and successful_id in traces:
        _write_trace_csv(OUTPUT_DIR / "first_successful_screw1_grasp_aligned_trace.csv", traces[successful_id])
    replay = {
        "attempted_replays": 3,
        "successful_replay_count": success_count,
        "exact_program_reused": True,
        "exact_reset_seed_reused": True,
        "physics_fingerprint_sha256": _fingerprint_hash(env),
        "video_path": str(video_path) if video_path and video_path.is_file() else "",
        "aligned_trace_path": str(OUTPUT_DIR / "first_successful_screw1_grasp_aligned_trace.csv") if successful_id else "",
    }
    _write_json(OUTPUT_DIR / "first_lift_replay_summary.json", replay)
    return replay


def _run_replay_from_artifact(env: NearGraspPhysicsEnv) -> dict[str, Any]:
    top_path = _top_program_path()
    payload = _read_json(top_path)
    rows = payload.get("programs", [])
    if not rows:
        return {"mode": "replay", "classification": "GRASP_PROGRAM_PARAMETERIZATION_INSUFFICIENT", "reason": "no top program"}
    program = GraspProgram.from_vector(rows[0]["program"]["continuous16"], rows[0]["program"]["template_id"])
    result = CandidateResult(0, program, _runtime_failure_evaluation(0), int(rows[0].get("reset_seed", args_cli.seed)))
    replay = _replay_first_lift(env, result, _configured_templates())
    return {"mode": "replay", "classification": "FIRST_PHYSICAL_SCREW1_LIFT_ACQUIRED" if replay["successful_replay_count"] else "GRASP_PROGRAM_PARAMETERIZATION_INSUFFICIENT", "replay": replay}


def _run_ppo_from_artifacts(env: NearGraspPhysicsEnv) -> dict[str, Any]:
    summary_path = OUTPUT_DIR / "near_grasp_summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    return _run_ppo(env, summary)


def _run_ppo(env: NearGraspPhysicsEnv, cem_summary: dict[str, Any]) -> dict[str, Any]:
    del env, cem_summary
    status = {
        "framework": "RSL-RL PPO",
        "legacy_action_dim": 14,
        "required_action_dim": 21,
        "training_allowed": False,
        "checkpoint_advances_curriculum": False,
        "independent_evaluation_episodes": 512,
        "started": False,
        "reason": "LEGACY_14D_RESIDUAL_RL_DISABLED_USE_PRIVILEGED_ELIGIBILITY_PACKET",
    }
    _write_json(OUTPUT_DIR / "residual_rl_status.json", status)
    return status


def _write_top_programs(results: Sequence[CandidateResult]) -> None:
    top = sorted(results, key=lambda item: item.ranking_key, reverse=True)[:32]
    payload = {
        "schema_version": 1,
        "programs": [item.to_dict() for item in top],
        "strict_evaluator_used": True,
        "reward_total_used_for_ordering": False,
    }
    _write_json(OUTPUT_DIR / "top_grasp_programs.json", payload)


def _runtime_failure_evaluation(candidate_id: int) -> PhysicalEvaluation:
    return PhysicalEvaluation(
        valid_candidate=False,
        hard_invalid=True,
        invalid_reasons=("runtime_or_timeout_without_evaluation",),
        physical_lift_success=False,
        stable_close=False,
        simultaneous_contact_duty=0.0,
        lift_contact_duty=0.0,
        peak_target_force_n=0.0,
        safe_force=False,
        object_z_gain_m=0.0,
        table_clearance_m=0.0,
        lost_table_support=False,
        lateral_displacement_m=0.0,
        relative_drift_m=0.0,
        jerk_metric=0.0,
        target_filtered_success_evidence=False,
        honesty={
            "sticky_used": False,
            "snap_used": False,
            "teacher_motion_used": False,
            "root_pose_writes_used": True,
            "root_pose_writes_reset_only": True,
            "physical_grasp_success": False,
            "physical_lift_success": False,
            "physical_insert_success": False,
            "oracle_visual_only": False,
            "not_physical": True,
        },
        metadata={"candidate_id": candidate_id},
    )


def _status_fields(result: dict[str, Any]) -> dict[str, Any]:
    physical_lift = bool(result.get("physical_lift_success", False))
    physical_grasp = bool(result.get("physical_grasp_success", physical_lift))
    training_allowed = bool(result.get("replayable_lift", False) and int(result.get("safe_elite_count", 0)) > 0)
    return {
        "deterministic_stack_frozen": True,
        "near_grasp_search_allowed": True,
        "experimental_rl_allowed": True,
        "readiness_passed": False,
        "training_allowed": training_allowed,
        "legacy_local_jacobian_mainline": False,
        "legacy_local_jacobian_diagnostic_only": True,
        "five_object_modulo_assignment_used": False,
        "legacy_v86_v89_backend_used": False,
        "sticky_used": False,
        "snap_used": False,
        "teacher_motion_used": False,
        "root_pose_writes_used": True,
        "root_pose_writes_reset_only": True,
        "physical_grasp_success": physical_grasp,
        "physical_lift_success": physical_lift,
        "physical_insert_success": False,
        "oracle_visual_only": False,
        "not_physical": not physical_lift,
    }


def _fingerprint_hash(env: NearGraspPhysicsEnv) -> str:
    payload = json.dumps(env.physics_fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _top_program_path() -> Path:
    return Path(args_cli.top_programs).resolve() if args_cli.top_programs else OUTPUT_DIR / "top_grasp_programs.json"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_csv(path: Path, payload: dict[str, Any]) -> None:
    flat = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value
        for key, value in payload.items()
    }
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def _write_trace_csv(path: Path, trace: dict[str, np.ndarray]) -> None:
    forces = trace["target_force_norms"]
    object_pos = trace["object_positions"]
    hand_pos = trace["hand_positions"]
    support = trace["table_support"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = ["step", "object_x", "object_y", "object_z", "hand_x", "hand_y", "hand_z", "table_support"] + [f"finger{index}_target_force_n" for index in range(1, 6)]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for step in range(forces.shape[0]):
            writer.writerow(
                {
                    "step": step,
                    "object_x": object_pos[step, 0],
                    "object_y": object_pos[step, 1],
                    "object_z": object_pos[step, 2],
                    "hand_x": hand_pos[step, 0],
                    "hand_y": hand_pos[step, 1],
                    "hand_z": hand_pos[step, 2],
                    "table_support": bool(support[step]),
                    **{f"finger{index + 1}_target_force_n": forces[step, index] for index in range(5)},
                }
            )


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=int(fps), codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
