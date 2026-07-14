"""Synthesize and validate privileged-physics Wuji grasps without CEM or PPO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_GRASP_PARENT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
if str(NEAR_GRASP_PARENT) not in sys.path:
    sys.path.insert(0, str(NEAR_GRASP_PARENT))

from near_grasp.grasp_synthesis.candidate_cache import CandidateCache, is_formal_gate_candidate  # noqa: E402
from near_grasp.grasp_synthesis.contact_controller import (  # noqa: E402
    ContactControllerObservation,
    ObjectSpaceContactController,
)
from near_grasp.grasp_synthesis.contact_sampler import retain_geometric_candidates, sample_contact_sets  # noqa: E402
from near_grasp.grasp_synthesis.grasp_energy import SearchMetrics  # noqa: E402
from near_grasp.grasp_synthesis.object_spec import ObjectGraspSpec  # noqa: E402
from near_grasp.grasp_synthesis.physics_validator import (  # noqa: E402
    ContactSource,
    GateKind,
    attribute_contact,
    evaluate_gate_trace,
    gate_passed,
)
from near_grasp.grasp_synthesis.wuji_ik import GraspCandidate, WujiKinematicModel  # noqa: E402


_boot_parser = argparse.ArgumentParser(add_help=False)
_boot_parser.add_argument("--phase", default="audit")
_boot_args, _ = _boot_parser.parse_known_args()
AppLauncher = None
if _boot_args.phase not in {"audit", "audit_before", "external_priors", "synthesize", "m0_screen", "m0_full", "synthesize_forensic_v2"}:
    from isaaclab.app import AppLauncher  # type: ignore[no-redef]  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--phase",
    choices=(
        "audit",
        "audit_before",
        "external_priors",
        "synthesize",
        "m0_screen",
        "m0_full",
        "synthesize_forensic_v2",
        "vector_smoke",
        "runtime_audit",
        "forensic_audit",
        "gate_a_forensic_v2",
        "gate_a",
        "gate_b",
        "gate_c",
    ),
    default="audit",
)
parser.add_argument("--part", choices=("Plug2", "Rod", "Backrest", "Screw1", "Frame"), default="Plug2")
parser.add_argument("--spec", default="")
parser.add_argument("--candidate-json", default="")
parser.add_argument("--output-dir", default="debug_runs/multi_object_privileged_grasp_v3")
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--sample-count", type=int, default=2048)
parser.add_argument("--geometric-count", type=int, default=128)
parser.add_argument("--optimize-count", type=int, default=64)
parser.add_argument("--physics-count", type=int, default=16)
parser.add_argument("--smoke-steps", type=int, default=32)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--alignment-debug", action="store_true")
parser.add_argument("--min-gpu-free-gb", type=float, default=5.0)
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
else:
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--device", default="cuda:0")
args_cli = parser.parse_args()


SPEC_PATH = Path(args_cli.spec).resolve() if args_cli.spec else REPO_ROOT / f"configs/grasp_synthesis/objects/{args_cli.part.lower()}.yaml"
SPEC = ObjectGraspSpec.load(SPEC_PATH, repo_root=REPO_ROOT)
if SPEC.part_name != args_cli.part:
    parser.error("--part does not match ObjectGraspSpec")
RUN_ROOT = (REPO_ROOT / args_cli.output_dir).resolve()
OUTPUT_ROOT = (RUN_ROOT / args_cli.part).resolve()
PHASE_DIR = OUTPUT_ROOT / args_cli.phase
CACHE = CandidateCache(REPO_ROOT / "state_banks/multi_object_privileged_grasp_v3")


def main() -> None:
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(PHASE_DIR / "gpu_preflight.json", _gpu_preflight())
    _write_json(PHASE_DIR / "resolved_object_spec.json", {**SPEC.to_dict(), "content_hash": SPEC.content_hash})
    static_audit = SPEC.audit_static_assets()
    _write_json(PHASE_DIR / "static_asset_audit.json", static_audit)
    if args_cli.phase in {"audit", "audit_before"}:
        if args_cli.phase == "audit_before":
            _write_json(
                RUN_ROOT / "code_fact_audit_before.json",
                {
                    "schema_version": 3,
                    "head": _git_head(),
                    "facts": {
                        "safe_single_contact_outranks_no_contact": True,
                        "pca_endpoint_effect_preserved": True,
                        "contact_latch_releases_after_three_below_keep_samples": True,
                        "direct_contact_residual_rl_gate_present": True,
                        "preshape_profile_resolved_but_injected_at_reset": True,
                        "formal_gate_preshape_was_not_action_driven": True,
                        "legacy_gate_a_eligibility_depended_only_on_closed_collision": True,
                        "legacy_forensic_loader_required_optimization_success": False,
                        "legacy_sampler_hardcoded_plug2_transverse_z_and_group_23": True,
                        "legacy_general_synthesis_did_not_bind_runtime_collision_mesh": True,
                        "legacy_coordex_proprio_order_mismatched_official": True,
                        "legacy_coordex_target_used_reset_q_instead_of_official_default_offset": True,
                        "near_grasp_target_asset_parameterized_but_defaults_remain_screw1": True,
                        "isaac_pinocchio_version": "2.7.0",
                        "wuji_retargeting_required_pinocchio_version": "3.8.0",
                    },
                    "sticky_snap_proxy_teacher_used": False,
                },
            )
        _write_json(
            PHASE_DIR / "summary.json",
            {
                "classification": "STATIC_ASSET_AUDIT_COMPLETE" if static_audit["static_asset_audit_ok"] else "ASSET_GEOMETRY_AUDIT_FAILED",
                "physical_grasp_success": False,
                "physical_lift_success": False,
                "full_route_success": False,
                "approach_close_lift_success": False,
                "cem_started": False,
                "ppo_started": False,
            },
        )
        return
    if args_cli.phase == "external_priors":
        summary = _external_prior_audit()
        _write_json(PHASE_DIR / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    if args_cli.phase in {"synthesize", "m0_screen", "m0_full", "synthesize_forensic_v2"}:
        try:
            summary = _synthesize_v3() if args_cli.phase != "synthesize_forensic_v2" else _synthesize_forensic_v2()
            summary.update(
                {
                    "part_name": SPEC.part_name,
                    "spec_hash": SPEC.content_hash,
                    "cem_started": False,
                    "ppo_started": False,
                    "sticky_used": False,
                    "snap_used": False,
                    "teacher_motion_used": False,
                    "proxy_used": False,
                }
            )
            _write_json(PHASE_DIR / "summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        except BaseException as exc:
            failure = {
                "classification": _failure_classification(args_cli.phase),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "physical_grasp_success": False,
                "physical_lift_success": False,
                "full_route_success": False,
                "approach_close_lift_success": False,
                "cem_started": False,
                "ppo_started": False,
            }
            _write_json(PHASE_DIR / "summary.json", failure)
            raise
        return

    if AppLauncher is None:
        raise RuntimeError("Isaac AppLauncher is unavailable for a runtime phase")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    env = None
    try:
        if args_cli.phase in {"vector_smoke", "forensic_audit", "runtime_audit"}:
            env = _make_env(16 if args_cli.phase == "vector_smoke" else 1, camera=False)
            summary = _vector_smoke(env) if args_cli.phase == "vector_smoke" else _forensic_audit(env)
        elif args_cli.phase == "gate_a_forensic_v2":
            candidates = _load_forensic_candidates()
            if len(candidates) != 16:
                raise RuntimeError(f"forensic Gate A requires 16 eligible candidates, got {len(candidates)}")
            env = _make_env(1, camera=bool(args_cli.record_video))
            summary = _run_forensic_gate_a(env, candidates)
        else:
            candidates = _load_candidates()
            selected = _eligible_candidates(candidates, _gate_kind())
            if not selected:
                summary = {
                    "classification": "NO_VALID_M0_CANDIDATE",
                    "requested_gate": args_cli.phase,
                    "physical_reset_count": 0,
                    "physical_grasp_success": False,
                    "physical_lift_success": False,
                    "full_route_success": False,
                    "approach_close_lift_success": False,
                }
                summary.update(
                    {
                        "part_name": SPEC.part_name,
                        "spec_hash": SPEC.content_hash,
                        "cem_started": False,
                        "ppo_started": False,
                        "sticky_used": False,
                        "snap_used": False,
                        "teacher_motion_used": False,
                        "proxy_used": False,
                    }
                )
                _write_json(PHASE_DIR / "summary.json", summary)
                print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
                return
            env_count = 1
            env = _make_env(env_count, camera=bool(args_cli.record_video))
            summary = _run_gate(env, selected, _gate_kind())
        summary.update(
            {
                "part_name": SPEC.part_name,
                "spec_hash": SPEC.content_hash,
                "cem_started": False,
                "ppo_started": False,
                "sticky_used": False,
                "snap_used": False,
                "teacher_motion_used": False,
                "proxy_used": False,
            }
        )
        _write_json(PHASE_DIR / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as exc:
        failure = {
            "classification": _failure_classification(args_cli.phase),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "physical_grasp_success": False,
            "physical_lift_success": False,
            "full_route_success": False,
            "approach_close_lift_success": False,
            "cem_started": False,
            "ppo_started": False,
        }
        _write_json(PHASE_DIR / "summary.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _make_env(num_envs: int, *, camera: bool):
    import torch
    from near_grasp.near_grasp_physics_env import NearGraspPhysicsEnv, NearGraspPhysicsEnvCfg, apply_object_grasp_spec

    cfg = NearGraspPhysicsEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 1.5
    cfg.sim.device = str(args_cli.device)
    cfg.episode_length_s = 8.0
    cfg.enable_contact_attribution = True
    cfg.enable_replay_camera = bool(camera)
    cfg.defer_terminal_reset = True
    cfg.replay_camera.width = 960
    cfg.replay_camera.height = 540
    apply_object_grasp_spec(cfg, SPEC, deterministic_gates=True)
    env = NearGraspPhysicsEnv(cfg=cfg, render_mode="rgb_array" if camera else None)
    env._zero_action = torch.zeros((num_envs, 14), dtype=torch.float32, device=env.device)
    return env


def _external_prior_audit() -> dict[str, Any]:
    import numpy as np

    from near_grasp.hand_prior_adapter import CoorDexWujiPriorAdapter, build_coordex_proprio

    output = RUN_ROOT / "external_priors"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = REPO_ROOT / "third_party/external_grasp_baselines/coordex/ckpts/hand_prior/kinematic_wrist_16k.pt"
    proprio = build_coordex_proprio(
        palm_linear_velocity_body=np.zeros(3),
        palm_angular_velocity_body=np.zeros(3),
        joint_position=np.zeros(20),
        default_joint_position=np.zeros(20),
        joint_velocity=np.zeros(20),
        default_joint_velocity=np.zeros(20),
        previous_joint_action=np.zeros(20),
    )
    adapter = CoorDexWujiPriorAdapter(checkpoint, default_q=np.zeros(20))
    adapter.reset(proprio, np.zeros(20))
    latent_rows = [np.zeros(6)]
    for axis in range(6):
        for sign in (-1.0, 1.0):
            row = np.zeros(6)
            row[axis] = sign
            latent_rows.append(row)
    decoded = np.asarray([adapter.decode(proprio, row, 1.0) for row in latent_rows])
    repositories = []
    for name, relative, expected in (
        ("coordex", "third_party/external_grasp_baselines/coordex", "9a5dfe0f52efd2624507f8cc9ed117aef6ef7472"),
        ("wuji-retargeting", "third_party/external_grasp_baselines/wuji-retargeting", "6eafdb22085f0e29c1d58c62f88f77ae1e971d8c"),
        ("gendexgrasp", "/tmp/multi_object_privileged_grasp_v3_external/GenDexGrasp", "29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7"),
        ("dro-grasp", "/tmp/multi_object_privileged_grasp_v3_external/DRO-Grasp", "07590bd8aeb074671e0d133fb372027f46fbd5f3"),
        ("dexgraspnet", "/tmp/multi_object_privileged_grasp_v3_external/DexGraspNet", "bd1b13d7248729af117e1d46aaa6266b147a3c7b"),
    ):
        path = Path(relative) if relative.startswith("/") else REPO_ROOT / relative
        commit = ""
        if (path / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            commit = result.stdout.strip() if result.returncode == 0 else ""
        license_files = sorted(str(value.relative_to(path)) for value in path.glob("LICEN[CS]E*") if value.is_file()) if path.is_dir() else []
        repositories.append(
            {
                "name": name,
                "path": str(path),
                "present": path.is_dir(),
                "expected_commit": expected,
                "actual_commit": commit,
                "commit_matches": commit == expected,
                "license_files": license_files,
            }
        )
    retarget_error = ""
    retarget_status = "NOT_RUN"
    retarget_seed_bank = output / "wuji_retargeting_seed_bank.json"
    retarget_seed_payload = _read_json(retarget_seed_bank) if retarget_seed_bank.is_file() else {}
    try:
        import pinocchio

        retarget_status = "ISOLATED_PIN_3_8_REQUIRED" if str(pinocchio.__version__) != "3.8.0" else "PIN_VERSION_READY"
    except Exception as exc:
        retarget_error = f"{type(exc).__name__}:{exc}"
    audit = {
        "schema_version": 3,
        "coordex": {
            "checkpoint": str(checkpoint),
            "signature": adapter.signature,
            "finite_decode": bool(np.all(np.isfinite(decoded))),
            "decoded_shape": list(decoded.shape),
            "decoded_min": float(np.min(decoded)),
            "decoded_max": float(np.max(decoded)),
            "proprio_order_verified_from_official_source": True,
            "target_semantics_verified_from_official_source": True,
            "used_as_seed_only": True,
            "redistribution_allowed": False,
        },
        "wuji_retargeting": {
            "current_runtime_status": "OFFICIAL_REPLAY_SEED_BANK_PASS" if retarget_seed_payload else retarget_status,
            "current_runtime_error": retarget_error,
            "required_pin_version": "3.8.0",
            "isolated_numpy_version": "2.2.6" if retarget_seed_payload else "",
            "seed_bank": str(retarget_seed_bank),
            "seed_count": len(retarget_seed_payload.get("seeds", ())),
            "source_commit": retarget_seed_payload.get("source_commit", ""),
            "source_pkl_sha256": retarget_seed_payload.get("source_pkl_sha256", ""),
            "used_as_seed_only": True,
        },
        "repositories": repositories,
        "external_model_output_used_as_physical_success": False,
    }
    _write_json(output / "external_model_audit.json", audit)
    return {
        "classification": "EXTERNAL_PRIOR_AUDIT_COMPLETE",
        "external_model_audit": str(output / "external_model_audit.json"),
        "coordex_decode_finite": audit["coordex"]["finite_decode"],
        "cem_started": False,
        "ppo_started": False,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
        "approach_close_lift_success": False,
    }


def _synthesize_v3() -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    from near_grasp.grasp_synthesis.contact_sampler import (
        crop_collision_mesh,
        fallback_variants,
        load_collision_mesh,
    )
    from near_grasp.hand_prior_adapter import CoorDexWujiPriorAdapter, build_coordex_proprio

    fixed = None
    if args_cli.phase == "m0_screen":
        fixed = {"sample_count": 512, "geometric_count": 64, "optimize_count": 32, "physics_count": 8}
    elif args_cli.phase == "m0_full":
        fixed = {"sample_count": 2048, "geometric_count": 128, "optimize_count": 64, "physics_count": 16}
    budget = fixed or {
        "sample_count": int(args_cli.sample_count),
        "geometric_count": int(args_cli.geometric_count),
        "optimize_count": int(args_cli.optimize_count),
        "physics_count": int(args_cli.physics_count),
    }
    if fixed:
        actual = {
            "sample_count": int(args_cli.sample_count),
            "geometric_count": int(args_cli.geometric_count),
            "optimize_count": int(args_cli.optimize_count),
            "physics_count": int(args_cli.physics_count),
        }
        defaults = {"sample_count": 2048, "geometric_count": 128, "optimize_count": 64, "physics_count": 16}
        if actual != defaults and actual != fixed:
            raise ValueError(f"{args_cli.phase} uses fixed budget {fixed}; got explicit {actual}")

    audit_dir = OUTPUT_ROOT / "runtime_audit"
    if not audit_dir.is_dir() and (OUTPUT_ROOT / "forensic_audit").is_dir():
        audit_dir = OUTPUT_ROOT / "forensic_audit"
    runtime_seed_path = audit_dir / "runtime_kinematic_seed.json"
    required = (
        runtime_seed_path,
        audit_dir / "target_object_collision_mesh.npz",
        audit_dir / "target_object_collision_mesh.json",
        audit_dir / "table_collision_mesh.npz",
        audit_dir / "table_collision_mesh.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"run runtime_audit for {SPEC.part_name} first: {missing}")
    runtime_seed = json.loads(runtime_seed_path.read_text(encoding="utf-8"))
    target_mesh = load_collision_mesh(
        audit_dir / "target_object_collision_mesh.npz",
        metadata_path=audit_dir / "target_object_collision_mesh.json",
    )
    table_full = load_collision_mesh(
        audit_dir / "table_collision_mesh.npz",
        metadata_path=audit_dir / "table_collision_mesh.json",
    )
    object_pos = np.asarray(SPEC.canonical_object_pos, dtype=np.float64)
    table_mesh = crop_collision_mesh(
        table_full,
        lower=(float(object_pos[0] - 0.60), float(object_pos[1] - 0.50), 0.68),
        upper=(float(object_pos[0] + 0.60), float(object_pos[1] + 0.50), 0.80),
    )
    runtime_names = tuple(runtime_seed["runtime_joint_names"])
    urdf = REPO_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/wuji-hand-description/urdf/right_block_palm_floating_drone.urdf"
    model = WujiKinematicModel(urdf, runtime_joint_names=runtime_names, package_dirs=[urdf.parent])
    model.set_scene_collision_meshes(target_mesh, table_mesh)
    compatibility = model.validate_runtime_joint_order(runtime_names)
    compatibility.update(
        {
            "target_collision_mesh_sha256": target_mesh.sha256,
            "table_collision_mesh_sha256": table_mesh.sha256,
            "scene_collision_available": model.scene_collision_available,
        }
    )
    _write_json(PHASE_DIR / "wuji_joint_asset_compatibility.json", compatibility)
    if not compatibility["valid"] or not model.scene_collision_available:
        raise RuntimeError("v3 runtime/Pinocchio/collision compatibility failed")
    provider_commits = {
        "coordex": "9a5dfe0f52efd2624507f8cc9ed117aef6ef7472",
        "wuji_retargeting": "6eafdb22085f0e29c1d58c62f88f77ae1e971d8c",
    }
    fingerprints = {
        "cache_schema_version": 3,
        "candidate_schema_version": 3,
        "solver_name": "staged_pad_dls_collision_continuation",
        "solver_version": "3.1",
        "spec_hash": SPEC.content_hash,
        "urdf_sha256": model.urdf_sha256,
        "runtime_joint_names": list(runtime_names),
        "target_collision_mesh_sha256": target_mesh.sha256,
        "table_collision_mesh_sha256": table_mesh.sha256,
        "seed_provider_commits": provider_commits,
        "thresholds": {
            "contact_residual_m": 0.001,
            "active_pad_penetration_m": 0.0001,
            "other_penetration_m": 0.00005,
            "other_clearance_m": 0.0005,
        },
        "budget": budget,
        "physics": runtime_seed["physics_fingerprint"],
    }
    input_hash = CACHE.input_hash(spec=SPEC.to_dict(), fingerprints=fingerprints)
    mass_kg = float(runtime_seed["physics_fingerprint"].get("runtime_mass_kg_min", SPEC.nominal_mass_kg))
    samples = sample_contact_sets(
        SPEC,
        count=int(budget["sample_count"]),
        seed=int(args_cli.seed),
        collision_mesh=target_mesh,
        axis_delta_limit_m=0.0005,
        mass_kg=mass_kg,
        friction=SPEC.nominal_dynamic_friction,
    )
    retained = retain_geometric_candidates(samples, count=int(budget["geometric_count"]))
    _write_jsonl(PHASE_DIR / "sampled_contact_sets.jsonl", (row.to_dict() for row in samples))
    _write_jsonl(PHASE_DIR / "retained_contact_sets.jsonl", (row.to_dict() for row in retained))
    q_seed = np.asarray(runtime_seed["q_seed26"], dtype=np.float64)
    default_q = np.asarray(runtime_seed.get("default_q26", runtime_seed["q_seed26"]), dtype=np.float64)
    preshape = np.asarray(runtime_seed.get("hand_preshape_q20", q_seed[6:]), dtype=np.float64)
    close = np.asarray(runtime_seed.get("hand_close_q20", preshape), dtype=np.float64)
    lower = np.asarray(runtime_seed["joint_lower26"], dtype=np.float64)
    upper = np.asarray(runtime_seed["joint_upper26"], dtype=np.float64)
    quat = np.asarray(SPEC.canonical_object_quat_wxyz, dtype=np.float64)
    object_rotation = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
    checkpoint = REPO_ROOT / "third_party/external_grasp_baselines/coordex/ckpts/hand_prior/kinematic_wrist_16k.pt"
    coordex_seed = None
    coordex_error = ""
    try:
        proprio = build_coordex_proprio(
            palm_linear_velocity_body=np.zeros(3),
            palm_angular_velocity_body=np.zeros(3),
            joint_position=default_q[6:],
            default_joint_position=default_q[6:],
            joint_velocity=np.zeros(20),
            default_joint_velocity=np.zeros(20),
            previous_joint_action=np.zeros(20),
        )
        adapter = CoorDexWujiPriorAdapter(
            checkpoint,
            joint_lower=lower[6:],
            joint_upper=upper[6:],
            default_q=default_q[6:],
        )
        adapter.reset(proprio, default_q[6:])
        coordex_seed = adapter.decode(proprio, np.zeros(6), 1.0)
    except Exception as exc:
        coordex_error = f"{type(exc).__name__}:{exc}"
    retarget_bank_path = RUN_ROOT / "external_priors/wuji_retargeting_seed_bank.json"
    retarget_rows = []
    if retarget_bank_path.is_file():
        retarget_rows = list(json.loads(retarget_bank_path.read_text(encoding="utf-8")).get("seeds", ()))

    variants = []
    for base in retained:
        variants.extend(fallback_variants(base, SPEC))
        if len(variants) >= int(budget["optimize_count"]):
            break
    variants = variants[: int(budget["optimize_count"])]
    candidates = []
    provider_counts: dict[str, int] = {}
    incremental = PHASE_DIR / "optimized_candidates.jsonl"
    incremental.unlink(missing_ok=True)
    seed_filter_log = PHASE_DIR / "seed_filter_diagnostics.jsonl"
    seed_filter_log.unlink(missing_ok=True)
    for index, contact_set in enumerate(variants):
        finger_mask = _active_mask(contact_set.finger_group)
        joint_mask = np.repeat(finger_mask.reshape(1, 5), 4, axis=0).reshape(-1)
        active_joint_mask = joint_mask.astype(bool)
        safe_preshape = preshape.copy()
        for finger_index in range(5):
            if finger_mask[finger_index] > 0.0:
                continue
            tucked_joints = (0, 2, 3) if finger_index == 0 else (2, 3)
            for joint_index in tucked_joints:
                hand_index = joint_index * 5 + finger_index
                safe_preshape[hand_index] = upper[6 + hand_index]

        def masked_external_seed(values: Any) -> np.ndarray:
            external = np.asarray(values, dtype=np.float64)
            if external.shape != (20,) or not np.all(np.isfinite(external)):
                raise ValueError("external Wuji seed must be finite 20D")
            merged = safe_preshape.copy()
            merged[active_joint_mask] = external[active_joint_mask]
            return np.clip(merged, lower[6:], upper[6:])

        hand_seeds: list[tuple[str, np.ndarray]] = [
            ("runtime_default", masked_external_seed(default_q[6:])),
            ("authored_pinch", safe_preshape + 0.25 * (close - preshape) * joint_mask),
            ("authored_wrap", safe_preshape + 0.55 * (close - preshape) * joint_mask),
        ]
        matching = [row for row in retarget_rows if str(row.get("finger_group", "")) == contact_set.finger_group]
        for row in matching[:2]:
            hand_seeds.append(
                (
                    f"wuji_retargeting:{row.get('seed_id', len(hand_seeds))}",
                    masked_external_seed(row["q20"]),
                )
            )
        if coordex_seed is not None:
            hand_seeds.append(("coordex_prior_mean", masked_external_seed(coordex_seed)))
        filtered_hand_seeds = []
        seed_filter_rows = []
        for provider, hand_q in hand_seeds[:6]:
            hand_q = np.clip(np.asarray(hand_q, dtype=np.float64), lower[6:], upper[6:])
            check_q = q_seed.copy()
            check_q[6:] = hand_q
            self_penetration, self_pairs = model.self_collision_penetration(check_q)
            accepted = bool(np.all(np.isfinite(hand_q)) and self_penetration <= (0.00005**2))
            seed_filter_rows.append(
                {
                    "contact_sample_id": contact_set.sample_id,
                    "finger_group": contact_set.finger_group,
                    "provider": provider,
                    "accepted": accepted,
                    "self_penetration_squared_m2": self_penetration,
                    "self_collision_pairs": [list(pair) for pair in self_pairs],
                }
            )
            if accepted:
                filtered_hand_seeds.append((provider, hand_q))
        with seed_filter_log.open("a", encoding="utf-8") as stream:
            for row in seed_filter_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if not filtered_hand_seeds:
            raise RuntimeError(f"all hand seeds failed the collision filter for group {contact_set.finger_group}")
        hand_seeds = filtered_hand_seeds
        contact_world = object_pos + np.asarray([row.position_object for row in contact_set.contacts]) @ object_rotation.T
        normal_world = np.asarray([row.normal_object for row in contact_set.contacts]) @ object_rotation.T
        fingers = [int(value) - 1 for value in contact_set.finger_group]
        approach_world = object_rotation @ np.asarray(contact_set.approach_direction_object)
        seed_candidates = []
        for provider, hand_q in hand_seeds:
            base_q = q_seed.copy()
            base_q[6:] = np.clip(hand_q, lower[6:], upper[6:])
            pad_points = model.pad_points_and_jacobians(base_q, fingers, normal_world)[0]
            base_q[:3] += np.mean(contact_world - pad_points, axis=0)
            base_q = np.clip(base_q, lower, upper)
            seed_candidates.append((f"{provider}:centered", base_q.copy()))
            approach_q = base_q.copy()
            approach_q[:3] -= 0.010 * approach_world
            dominant_axis = int(np.argmax(np.abs(approach_world)))
            dominant_sign = float(np.sign(approach_world[dominant_axis]) or 1.0)
            if dominant_axis == 0:
                approach_q[4] -= dominant_sign * 0.5 * np.pi
            elif dominant_axis == 1:
                approach_q[3] += dominant_sign * 0.5 * np.pi
            elif dominant_sign > 0.0:
                approach_q[3] += np.pi
            approach_q[3:6] = (approach_q[3:6] + np.pi) % (2.0 * np.pi) - np.pi
            seed_candidates.append(
                (f"{provider}:approach_aligned10mm", np.clip(approach_q, lower, upper))
            )
        candidate = model.optimize_contact_set_staged(
            part_name=SPEC.part_name,
            contact_set=contact_set,
            object_position_world=object_pos,
            object_rotation_world=object_rotation,
            seed_candidates=seed_candidates,
            joint_lower26=lower,
            joint_upper26=upper,
            target_force_n=SPEC.target_force_n,
            input_hash=input_hash,
        )
        candidates.append(candidate)
        provider_counts[candidate.seed_provider] = provider_counts.get(candidate.seed_provider, 0) + int(candidate.contact_residual_success)
        with incremental.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")
        _write_json(
            PHASE_DIR / "synthesis_heartbeat.json",
            {
                "completed": index + 1,
                "requested": len(variants),
                "strict_m0_count": sum(int(row.physics_gate_a_eligible) for row in candidates),
                "last_candidate": candidate.candidate_id,
            },
        )
    candidates.sort(
        key=lambda row: (
            row.physics_gate_a_eligible,
            row.gate_eligibility.get("gate_c", False),
            row.contact_residual_success,
            -row.tip_error_m,
            -float(row.energy["total"]),
        ),
        reverse=True,
    )
    cache_path = CACHE.save(
        part_name=SPEC.part_name,
        input_hash=input_hash,
        candidates=(row.to_dict() for row in candidates),
        metadata={"fingerprints": fingerprints, "budget": budget, "coordex_error": coordex_error},
    )
    pointer = cache_path.parents[1] / "latest_m0.json"
    _write_json(pointer, {"input_hash": input_hash, "candidate_json": str(cache_path), "spec_hash": SPEC.content_hash})
    strict = [row for row in candidates if row.physics_gate_a_eligible]
    top = strict[: int(budget["physics_count"])]
    _write_json(
        PHASE_DIR / "top_candidates.json",
        {"input_hash": input_hash, "candidate_json": str(cache_path), "candidates": [row.to_dict() for row in top]},
    )
    summary = {
        "classification": "STRICT_M0_CANDIDATES_SYNTHESIZED" if top else "NO_VALID_M0_CANDIDATE",
        "sample_count": len(samples),
        "retained_count": len(retained),
        "optimized_count": len(candidates),
        "reachability_success_count": sum(int(row.reachability_success) for row in candidates),
        "contact_residual_success_count": sum(int(row.contact_residual_success) for row in candidates),
        "closed_pose_collision_success_count": sum(int(row.closed_pose_collision_success) for row in candidates),
        "strict_m0_candidate_count": len(strict),
        "physics_candidate_count": len(top),
        "provider_contact_residual_success": provider_counts,
        "candidate_json": str(cache_path),
        "input_hash": input_hash,
        "physical_reset_count": 0,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
        "approach_close_lift_success": False,
    }
    _write_json(PHASE_DIR / "m0_summary.json", summary)
    return summary


def _synthesize() -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    runtime_seed_path = OUTPUT_ROOT / "vector_smoke" / "runtime_kinematic_seed.json"
    if not runtime_seed_path.is_file():
        raise FileNotFoundError(f"run vector_smoke first: {runtime_seed_path}")
    runtime_seed = json.loads(runtime_seed_path.read_text(encoding="utf-8"))
    runtime_names = tuple(runtime_seed["runtime_joint_names"])
    urdf = REPO_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/wuji-hand-description/urdf/right_block_palm_floating_drone.urdf"
    model = WujiKinematicModel(urdf, runtime_joint_names=runtime_names, package_dirs=[urdf.parent])
    compatibility = model.validate_runtime_joint_order(runtime_names)
    compatibility["runtime_articulation_joint_names"] = list(runtime_seed["runtime_articulation_joint_names"])
    compatibility["selected_runtime_order"] = list(runtime_names)
    _write_json(PHASE_DIR / "wuji_joint_asset_compatibility.json", compatibility)
    if not compatibility["valid"]:
        raise RuntimeError("Wuji runtime/Pinocchio joint compatibility failed")

    fingerprints = {
        "synthesis_schema_version": 6,
        "spec_hash": SPEC.content_hash,
        "urdf_sha256": model.urdf_sha256,
        "runtime_joint_names": list(runtime_names),
        "collision_model_available": model.collision_model_available,
        "collision_pair_count": model.collision_pair_count,
        "physics": runtime_seed["physics_fingerprint"],
    }
    input_hash = CACHE.input_hash(spec=SPEC.to_dict(), fingerprints=fingerprints)
    samples = sample_contact_sets(SPEC, count=int(args_cli.sample_count), seed=int(args_cli.seed))
    retained = retain_geometric_candidates(samples, count=int(args_cli.geometric_count))
    _write_jsonl(PHASE_DIR / "sampled_contact_sets.jsonl", (row.to_dict() for row in samples))
    _write_jsonl(PHASE_DIR / "retained_contact_sets.jsonl", (row.to_dict() for row in retained))

    q_seed = np.asarray(runtime_seed["q_seed26"], dtype=np.float64)
    lower = np.asarray(runtime_seed["joint_lower26"], dtype=np.float64)
    upper = np.asarray(runtime_seed["joint_upper26"], dtype=np.float64)
    quat = np.asarray(SPEC.canonical_object_quat_wxyz)
    rotation = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
    candidates = []
    incremental = PHASE_DIR / "optimized_candidates.jsonl"
    if incremental.exists():
        incremental.unlink()
    for index, contact_set in enumerate(retained[: int(args_cli.optimize_count)]):
        candidate = model.optimize_contact_set(
            part_name=SPEC.part_name,
            contact_set=contact_set,
            object_position_world=SPEC.canonical_object_pos,
            object_rotation_world=rotation,
            q_seed26=q_seed,
            joint_lower26=lower,
            joint_upper26=upper,
            table_top_z_m=float(runtime_seed["physics_fingerprint"]["table_top_z_m"]),
            target_force_n=SPEC.target_force_n,
            input_hash=input_hash,
        )
        candidates.append(candidate)
        with incremental.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")
        _write_json(
            PHASE_DIR / "synthesis_heartbeat.json",
            {"completed": index + 1, "requested": int(args_cli.optimize_count), "last_candidate": candidate.candidate_id},
        )
    candidates.sort(
        key=lambda row: (
            row.optimization_success,
            row.collision_free_path,
            -float(row.energy["total"]),
            -row.tip_error_m,
        ),
        reverse=True,
    )
    cache_path = CACHE.save(
        part_name=SPEC.part_name,
        input_hash=input_hash,
        candidates=(row.to_dict() for row in candidates),
        metadata={"fingerprints": fingerprints, "sample_budget": int(args_cli.sample_count)},
    )
    latest = cache_path.parents[1] / "latest.json"
    _write_json(latest, {"input_hash": input_hash, "candidate_json": str(cache_path), "spec_hash": SPEC.content_hash})
    top = candidates[: int(args_cli.physics_count)]
    _write_json(
        PHASE_DIR / "top_candidates.json",
        {"input_hash": input_hash, "candidate_json": str(cache_path), "candidates": [row.to_dict() for row in top]},
    )
    return {
        "classification": "GRASP_CANDIDATES_SYNTHESIZED" if any(row.optimization_success for row in top) else "IK_COLLISION_PATH_INFEASIBLE",
        "sample_count": len(samples),
        "retained_count": len(retained),
        "optimized_count": len(candidates),
        "physics_candidate_count": len(top),
        "optimization_success_count": sum(int(row.optimization_success) for row in candidates),
        "candidate_json": str(cache_path),
        "input_hash": input_hash,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
        "approach_close_lift_success": False,
    }


def _synthesize_forensic_v2() -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    from near_grasp.grasp_synthesis.contact_sampler import (
        crop_collision_mesh,
        fallback_variants,
        load_collision_mesh,
    )

    fixed = {
        "sample_count": 2048,
        "geometric_count": 128,
        "optimize_count": 64,
        "physics_count": 16,
        "seed": 20260713,
    }
    actual = {
        "sample_count": int(args_cli.sample_count),
        "geometric_count": int(args_cli.geometric_count),
        "optimize_count": int(args_cli.optimize_count),
        "physics_count": int(args_cli.physics_count),
        "seed": int(args_cli.seed),
    }
    if actual != fixed:
        raise ValueError(f"forensic v2 budget is frozen: expected {fixed}, got {actual}")
    audit_dir = OUTPUT_ROOT / "forensic_audit"
    runtime_seed_path = audit_dir / "runtime_kinematic_seed.json"
    target_mesh_path = audit_dir / "target_object_collision_mesh.npz"
    table_mesh_path = audit_dir / "table_collision_mesh.npz"
    required = (
        runtime_seed_path,
        target_mesh_path,
        audit_dir / "target_object_collision_mesh.json",
        table_mesh_path,
        audit_dir / "table_collision_mesh.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"forensic runtime audit artifacts are missing: {missing}")
    runtime_seed = json.loads(runtime_seed_path.read_text(encoding="utf-8"))
    target_mesh = load_collision_mesh(target_mesh_path, metadata_path=audit_dir / "target_object_collision_mesh.json")
    table_mesh_full = load_collision_mesh(table_mesh_path, metadata_path=audit_dir / "table_collision_mesh.json")
    table_crop_bounds = {"lower": (-0.40, -0.20, 0.70), "upper": (0.10, 0.20, 0.76)}
    table_mesh = crop_collision_mesh(
        table_mesh_full,
        lower=table_crop_bounds["lower"],
        upper=table_crop_bounds["upper"],
    )
    cropped_table_path = PHASE_DIR / "table_collision_mesh_workspace_crop.npz"
    np.savez_compressed(cropped_table_path, vertices=table_mesh.vertices, faces=table_mesh.faces)
    _write_json(
        PHASE_DIR / "table_collision_mesh_workspace_crop.json",
        {
            "source_full_mesh_sha256": table_mesh_full.sha256,
            "cropped_mesh_sha256": table_mesh.sha256,
            "crop_bounds_env_local": table_crop_bounds,
            "vertex_count": len(table_mesh.vertices),
            "face_count": len(table_mesh.faces),
            "coordinate_frame": table_mesh.coordinate_frame,
            "purpose": "actual audited Table triangles intersecting the complete Plug2 hand workspace",
        },
    )
    runtime_names = tuple(runtime_seed["runtime_joint_names"])
    urdf = REPO_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/wuji-hand-description/urdf/right_block_palm_floating_drone.urdf"
    model = WujiKinematicModel(urdf, runtime_joint_names=runtime_names, package_dirs=[urdf.parent])
    model.set_scene_collision_meshes(target_mesh, table_mesh)
    compatibility = model.validate_runtime_joint_order(runtime_names)
    compatibility.update(
        {
            "scene_collision_available": model.scene_collision_available,
            "target_collision_mesh_sha256": target_mesh.sha256,
            "table_collision_mesh_sha256": table_mesh.sha256,
        }
    )
    _write_json(PHASE_DIR / "wuji_joint_asset_compatibility.json", compatibility)
    if not compatibility["valid"] or not model.scene_collision_available:
        raise RuntimeError("forensic Wuji/scene collision compatibility failed")
    fingerprints = {
        "synthesis_schema_version": 7,
        "forensic_version": 2,
        "spec_hash": SPEC.content_hash,
        "urdf_sha256": model.urdf_sha256,
        "runtime_joint_names": list(runtime_names),
        "target_collision_mesh_sha256": target_mesh.sha256,
        "table_collision_mesh_sha256": table_mesh.sha256,
        "table_full_collision_mesh_sha256": table_mesh_full.sha256,
        "table_crop_bounds_env_local": table_crop_bounds,
        "fallback_order": list(SPEC.candidate_fallback_order),
        "axis_delta_limit_m": 0.0005,
        "friction_cone_edges": 8,
        "force_residual_limit_n": 0.001,
        "torque_residual_limit_nm": 0.00001,
        "active_tip_penetration_limit_m": 0.0001,
        "other_clearance_limit_m": 0.0005,
        "slsqp_max_iterations": 32,
        "slsqp_variable_scope": "wrist6_plus_active_finger_joints",
        "physics": runtime_seed["physics_fingerprint"],
    }
    forensic_cache = CandidateCache(REPO_ROOT / "state_banks/grasp_synthesis/Plug2")
    input_hash = forensic_cache.input_hash(spec=SPEC.to_dict(), fingerprints=fingerprints)
    samples = sample_contact_sets(
        SPEC,
        count=fixed["sample_count"],
        seed=fixed["seed"],
        collision_mesh=target_mesh,
        axis_delta_limit_m=0.0005,
    )
    retained = retain_geometric_candidates(samples, count=fixed["geometric_count"])
    _write_jsonl(PHASE_DIR / "sampled_contact_sets.jsonl", (row.to_dict() for row in samples))
    _write_jsonl(PHASE_DIR / "retained_contact_sets.jsonl", (row.to_dict() for row in retained))
    q_seed = np.asarray(runtime_seed["q_seed26"], dtype=np.float64)
    lower = np.asarray(runtime_seed["joint_lower26"], dtype=np.float64)
    upper = np.asarray(runtime_seed["joint_upper26"], dtype=np.float64)
    quat = np.asarray(SPEC.canonical_object_quat_wxyz)
    rotation = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
    candidates = []
    attempt_rows = []
    calls = 0
    stop = False
    for base in retained:
        for variant_index, variant in enumerate(fallback_variants(base, SPEC)):
            if calls >= fixed["optimize_count"]:
                stop = True
                break
            candidate = model.optimize_contact_set(
                part_name=SPEC.part_name,
                contact_set=variant,
                object_position_world=SPEC.canonical_object_pos,
                object_rotation_world=rotation,
                q_seed26=q_seed,
                joint_lower26=lower,
                joint_upper26=upper,
                table_top_z_m=float(runtime_seed["physics_fingerprint"]["table_top_z_m"]),
                target_force_n=SPEC.target_force_n,
                input_hash=input_hash,
            )
            calls += 1
            candidates.append(candidate)
            attempt_rows.append(
                {
                    "optimization_call": calls,
                    "base_sample_id": base.sample_id,
                    "fallback_variant_index": variant_index,
                    "fallback_tier": variant.fallback_tier,
                    "finger_group": variant.finger_group,
                    "approach_direction_object": list(variant.approach_direction_object),
                    "candidate_id": candidate.candidate_id,
                    "optimization_success": candidate.optimization_success,
                    "gate_a_eligible": bool(candidate.gate_eligibility.get("gate_a", False)),
                    "optimizer_message": candidate.metadata.get("optimizer_message", ""),
                }
            )
            _write_json(
                PHASE_DIR / "synthesis_heartbeat.json",
                {"optimization_calls": calls, "fixed_limit": 64, "last_attempt": attempt_rows[-1]},
            )
            if candidate.optimization_success:
                break
        if stop:
            break
    if calls != fixed["optimize_count"]:
        raise RuntimeError(f"forensic optimizer executed {calls}/64 calls")
    _write_json(PHASE_DIR / "fallback_attempts.json", attempt_rows)
    candidates.sort(
        key=lambda row: (
            bool(row.optimization_success),
            bool(row.gate_eligibility.get("gate_a", False)),
            -float(row.energy["total"]),
            -row.tip_error_m,
        ),
        reverse=True,
    )
    cache_path = forensic_cache.save(
        part_name="forensic_v2",
        input_hash=input_hash,
        candidates=(row.to_dict() for row in candidates),
        metadata={"fingerprints": fingerprints, "fixed_budget": fixed},
    )
    pointer = forensic_cache.root / "forensic_v2_latest.json"
    _write_json(
        pointer,
        {
            "input_hash": input_hash,
            "candidate_json": str(cache_path),
            "spec_hash": SPEC.content_hash,
            "legacy_latest_untouched": True,
        },
    )
    top = []
    seen_physical_ids = set()
    for candidate in candidates:
        if candidate.candidate_id in seen_physical_ids:
            continue
        seen_physical_ids.add(candidate.candidate_id)
        top.append(candidate)
        if len(top) >= fixed["physics_count"]:
            break
    _write_json(
        PHASE_DIR / "top_candidates.json",
        {"input_hash": input_hash, "candidate_json": str(cache_path), "candidates": [row.to_dict() for row in top]},
    )
    _write_legacy_forensic_comparison(top)
    return {
        "classification": "FORENSIC_GRASP_CANDIDATES_SYNTHESIZED",
        "sample_count": len(samples),
        "retained_count": len(retained),
        "optimization_call_count": calls,
        "optimization_success_count": sum(int(row.optimization_success) for row in candidates),
        "gate_a_eligible_count": sum(int(row.gate_eligibility.get("gate_a", False)) for row in top),
        "physics_candidate_count": len(top),
        "candidate_json": str(cache_path),
        "forensic_pointer": str(pointer),
        "input_hash": input_hash,
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
        "approach_close_lift_success": False,
    }


def _write_legacy_forensic_comparison(new_top: Sequence[GraspCandidate]) -> None:
    legacy_path = (
        REPO_ROOT
        / "state_banks/grasp_synthesis/Plug2/d92179a09d69c7150262035dd99895b3d9570c868cde11b4ff44d0c1c37b7882/candidates.jsonl"
    )
    legacy = []
    if legacy_path.is_file():
        legacy = [
            GraspCandidate.from_dict(json.loads(line))
            for line in legacy_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:16]
    rows = []
    for rank in range(max(len(legacy), len(new_top))):
        old = legacy[rank] if rank < len(legacy) else None
        new = new_top[rank] if rank < len(new_top) else None
        old_metadata = dict(old.metadata) if old is not None else {}
        new_metadata = dict(new.metadata) if new is not None else {}
        rows.append(
            {
                "rank": rank,
                "legacy_candidate_id": old.candidate_id if old else "not_available",
                "forensic_candidate_id": new.candidate_id if new else "not_available",
                "legacy_finger_group": old.finger_group if old else "not_available",
                "forensic_finger_group": new.finger_group if new else "not_available",
                "legacy_axis_delta_m": old_metadata.get("axis_delta_m", "not_recorded"),
                "forensic_axis_delta_m": new_metadata.get("axis_delta_m", "not_available"),
                "legacy_raycast_hit": old_metadata.get("raycast_hit", "not_recorded"),
                "forensic_raycast_hit": new_metadata.get("raycast_hit", "not_available"),
                "legacy_gravity_wrench": old_metadata.get("gravity_wrench", "not_recorded"),
                "forensic_gravity_wrench": new_metadata.get("gravity_wrench", "not_available"),
                "legacy_support_vertex": old_metadata.get("pad_support_vertex_local", "not_recorded"),
                "forensic_support_vertex": new_metadata.get("pad_support_vertex_local", "not_available"),
                "legacy_hand_object_penetration": old.energy.get("hand_object_penetration", "not_recorded") if old else "not_available",
                "forensic_hand_object_penetration": new.energy.get("hand_object_penetration", "not_available") if new else "not_available",
                "legacy_scene_clearance": old_metadata.get("scene_clearance", "not_recorded"),
                "forensic_scene_clearance": new_metadata.get("scene_clearance", "not_available"),
                "legacy_tip_error_m": old.tip_error_m if old else "not_available",
                "forensic_tip_error_m": new.tip_error_m if new else "not_available",
                "legacy_gate_a_eligible": old.gate_eligibility.get("gate_a", False) if old else "not_available",
                "forensic_gate_a_eligible": new.gate_eligibility.get("gate_a", False) if new else "not_available",
            }
        )
    _write_json(PHASE_DIR / "legacy_vs_forensic_candidate_geometry.json", rows)
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
        )
    _write_csv(PHASE_DIR / "legacy_vs_forensic_candidate_geometry.csv", csv_rows)


def _vector_smoke(env) -> dict[str, Any]:
    import numpy as np

    q = np.concatenate((env._parked_wrist_q.detach().cpu().numpy(), env.hand_preshape_q.detach().cpu().numpy()))
    active = _active_mask(SPEC.allowed_finger_groups[0])
    env.configure_privileged_control(
        np.repeat(q.reshape(1, -1), env.num_envs, axis=0),
        np.repeat(active.reshape(1, -1), env.num_envs, axis=0),
        reset_seeds=[args_cli.seed + index for index in range(env.num_envs)],
    )
    env.reset()
    finite = True
    for _ in range(int(args_cli.smoke_steps)):
        observation, *_ = env.step(env._zero_action)
        policy = observation["policy"] if isinstance(observation, dict) else observation
        finite &= bool(np.all(np.isfinite(policy.detach().cpu().numpy())))
    snapshot = env.privileged_snapshot()
    ordered = np.concatenate((env._wrist_ids.detach().cpu().numpy(), env._hand_ids.detach().cpu().numpy()))
    runtime_seed = {
        "runtime_joint_names": [env.robot.joint_names[int(index)] for index in ordered],
        "runtime_articulation_joint_names": list(env.robot.joint_names),
        "q_seed26": q.tolist(),
        "joint_lower26": env._lower[ordered].detach().cpu().tolist(),
        "joint_upper26": env._upper[ordered].detach().cpu().tolist(),
        "physics_fingerprint": env.physics_fingerprint(),
    }
    _write_json(PHASE_DIR / "runtime_kinematic_seed.json", runtime_seed)
    summary = {
        "classification": "PRIVILEGED_VECTOR_SMOKE_PASS" if finite else "NEAR_GRASP_ENV_VECTORIZATION_BLOCKED",
        "num_envs": env.num_envs,
        "steps": int(args_cli.smoke_steps),
        "observation_shape": list(policy.shape),
        "finite_observations": finite,
        "target_filter_valid": bool(np.all(snapshot["target_filter_valid"])),
        "contact_report_available": bool(np.all(snapshot["contact_report_available"])),
        "post_reset_object_writes": int(np.max(snapshot["post_reset_object_writes"])),
        "post_reset_wrist_state_writes": int(np.max(snapshot["post_reset_wrist_state_writes"])),
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
        "approach_close_lift_success": False,
    }
    _write_json(PHASE_DIR / "runtime_physics_fingerprint.json", env.physics_fingerprint())
    return summary


def _forensic_audit(env) -> dict[str, Any]:
    import numpy as np

    from near_grasp.grasp_synthesis.forensic_trace import link_contact_source_map

    q = np.concatenate((env._parked_wrist_q.detach().cpu().numpy(), env.hand_preshape_q.detach().cpu().numpy()))
    active = _active_mask(SPEC.allowed_finger_groups[0])
    env.configure_privileged_control(q.reshape(1, -1), active.reshape(1, -1), reset_seeds=[args_cli.seed])
    env.reset()
    env.clear_reset_contact_cache_for_privileged_control()
    env.set_privileged_joint_targets(q.reshape(1, -1))
    collision_audit = env.export_runtime_collision_geometry(PHASE_DIR)
    env.step(env._zero_action)
    snapshot = env.privileged_snapshot()
    events = env.consume_forensic_contact_events()[0]
    ordered = np.concatenate((env._wrist_ids.detach().cpu().numpy(), env._hand_ids.detach().cpu().numpy()))
    runtime_seed = {
        "runtime_joint_names": [env.robot.joint_names[int(index)] for index in ordered],
        "runtime_articulation_joint_names": list(env.robot.joint_names),
        "q_seed26": q.tolist(),
        "default_q26": np.concatenate(
            (
                env.robot.data.default_joint_pos[0, env._wrist_ids].detach().cpu().numpy(),
                env.robot.data.default_joint_pos[0, env._hand_ids].detach().cpu().numpy(),
            )
        ).tolist(),
        "hand_preshape_q20": env.hand_preshape_q.detach().cpu().tolist(),
        "hand_close_q20": env.hand_close_q.detach().cpu().tolist(),
        "joint_lower26": env._lower[ordered].detach().cpu().tolist(),
        "joint_upper26": env._upper[ordered].detach().cpu().tolist(),
        "physics_fingerprint": env.physics_fingerprint(),
    }
    _write_json(PHASE_DIR / "runtime_kinematic_seed.json", runtime_seed)
    first_frame = {
        "physics_frame_id": 0,
        "joint_pos26": snapshot["joint_pos26"][0].tolist(),
        "joint_target26": snapshot["joint_target26"][0].tolist(),
        "joint_sim_target26": snapshot["joint_sim_target26"][0].tolist(),
        "joint_error26": (snapshot["joint_target26"][0] - snapshot["joint_pos26"][0]).tolist(),
        "isaac_tip_positions": snapshot["tip_pos_local"][0].tolist(),
        "isaac_tip_quat_wxyz": snapshot["tip_quat_wxyz"][0].tolist(),
        "object_position": snapshot["object_pos_local"][0].tolist(),
        "object_quat_wxyz": snapshot["object_quat_wxyz"][0].tolist(),
        "object_linear_velocity": snapshot["object_lin_vel"][0].tolist(),
        "object_angular_velocity": snapshot["object_ang_vel"][0].tolist(),
        "target_force_norms": snapshot["target_force_norms"][0].tolist(),
        "target_force_xyz": snapshot["target_force_xyz"][0].tolist(),
        "all_force_xyz": snapshot["all_force_xyz"][0].tolist(),
        "table_force_xyz": snapshot["table_force_xyz"][0].tolist(),
        "ground_force_xyz": snapshot["ground_force_xyz"][0].tolist(),
        "contact_events": [event.to_dict() for event in events],
        "link_contact_sources": link_contact_source_map(env.robot_body_names_for_forensics(), events),
        "post_reset_object_writes": int(snapshot["post_reset_object_writes"][0]),
        "post_reset_wrist_state_writes": int(snapshot["post_reset_wrist_state_writes"][0]),
    }
    _write_json(PHASE_DIR / "first_physics_frame.json", first_frame)
    return {
        "classification": "RUNTIME_COLLISION_AND_KINEMATIC_AUDIT_PASS",
        "physics_frame_zero_recorded": True,
        "collision_geometry": collision_audit,
        "post_reset_object_writes": first_frame["post_reset_object_writes"],
        "post_reset_wrist_state_writes": first_frame["post_reset_wrist_state_writes"],
        "physical_grasp_success": False,
        "physical_lift_success": False,
        "full_route_success": False,
        "approach_close_lift_success": False,
    }


def _load_forensic_candidates() -> list[GraspCandidate]:
    pointer = REPO_ROOT / "state_banks/grasp_synthesis/Plug2/forensic_v2_latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"forensic candidate pointer missing: {pointer}")
    path = Path(json.loads(pointer.read_text(encoding="utf-8"))["candidate_json"])
    rows = [
        GraspCandidate.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = []
    seen = set()
    for row in rows:
        if row.candidate_id in seen or not row.gate_eligibility.get("gate_a", False):
            continue
        if not bool(row.metadata.get("gravity_wrench", {}).get("feasible", False)):
            continue
        seen.add(row.candidate_id)
        eligible.append(row)
        if len(eligible) >= 16:
            break
    return eligible


def _run_forensic_gate_a(env, candidates: list[GraspCandidate]) -> dict[str, Any]:
    from near_grasp.grasp_synthesis.forensic_trace import sequential_gate_schedule

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    schedule = sequential_gate_schedule([candidate.candidate_id for candidate in candidates], repeats=5)
    if len(schedule) != 80 or env.num_envs != 1:
        raise RuntimeError("forensic Gate A must execute 80 sequential single-env resets")
    results = []
    first_frames = []
    _write_json(
        PHASE_DIR / "gate_a_physics_shortlist.json",
        {
            "selection": "first 16 unique scene-clearance and gravity-wrench eligible physical poses",
            "optimization_success_required": False,
            "reason": "all 64 bounded SLSQP calls retained 20-41mm tip residual; Gate A is forensic evidence, not candidate success",
            "candidates": [candidate.to_dict() for candidate in candidates],
        },
    )
    for schedule_index, (candidate_id, reset_index) in enumerate(schedule):
        candidate = by_id[candidate_id]
        result, trace = _run_forensic_gate_a_reset(env, candidate, reset_index)
        results.append(result)
        first_frames.append(trace[0])
        _write_json(
            PHASE_DIR / "gate_heartbeat.json",
            {
                "completed_resets": schedule_index + 1,
                "fixed_total_resets": 80,
                "candidate_id": candidate_id,
                "reset_index": reset_index,
                "termination_layer": result.termination_layer,
                "passed": result.passed,
            },
        )
    grouped = {}
    for candidate in candidates:
        rows = [row for row in results if row.candidate_id == candidate.candidate_id]
        grouped[candidate.candidate_id] = {
            "successes": sum(int(row.passed) for row in rows),
            "trials": len(rows),
            "passed": gate_passed(rows),
            "termination_layers": [row.termination_layer for row in rows],
            "peak_target_force_n": [row.peak_target_force_n for row in rows],
            "trace_lengths": [row.metadata.get("trace_length", 0) for row in rows],
        }
    qualified = [candidate_id for candidate_id, values in grouped.items() if values["passed"]]
    first_frame_rows = [
        {
            "candidate_id": row["candidate_id"],
            "reset_index": row["reset_index"],
            "physics_frame_id": row["physics_frame_id"],
            "stage": row["stage"],
            "target_force_norms": row["target_force_norms"],
            "all_force_xyz": row["all_force_xyz"],
            "contact_events": row["contact_events"],
            "link_contact_sources": row["link_contact_sources"],
            "object_linear_velocity": row["object_linear_velocity"],
            "object_angular_velocity": row["object_angular_velocity"],
            "terminal": row["terminal"],
            "termination_layer": row["termination_layer"],
        }
        for row in first_frames
    ]
    _write_json(PHASE_DIR / "first_frame_collision_diagnostics.json", first_frame_rows)
    _write_csv(
        PHASE_DIR / "first_frame_collision_diagnostics.csv",
        [_jsonify_nested(row) for row in first_frame_rows],
    )
    summary = {
        "classification": "GATE_A_FORENSIC_V2_PASS" if qualified else "GATE_A_FORENSIC_V2_FAILED",
        "gate": GateKind.RESET_CLOSED_HOLD_LIFT.name,
        "candidate_results": grouped,
        "qualified_candidate_ids": qualified,
        "candidate_count": len(candidates),
        "resets_per_candidate": 5,
        "sequential_fresh_reset_count": len(schedule),
        "num_envs": 1,
        "matched_video": "",
        "video_replay_matched": False,
        "oracle_reset_grasp": True,
        "physical_grasp_success": False,
        "physical_lift_success": bool(qualified),
        "full_route_success": False,
        "approach_close_lift_success": False,
        "runtime_physics_fingerprint": env.physics_fingerprint(),
    }
    _write_json(PHASE_DIR / "gate_a_forensic_summary.json", summary)
    return summary


def _run_forensic_gate_a_reset(env, candidate: GraspCandidate, reset_index: int):
    import numpy as np

    from near_grasp.grasp_synthesis.forensic_trace import (
        ForensicFrame,
        link_contact_source_map,
        merge_filtered_fingertip_sources,
    )

    reset_q = np.asarray(candidate.closed_joint_q26, dtype=np.float64).reshape(1, 26)
    active_mask = _active_mask(candidate.finger_group)
    env.configure_privileged_control(
        reset_q,
        active_mask.reshape(1, 5),
        candidate_close_q20=reset_q[:, 6:],
        reset_seeds=[int(args_cli.seed) + int(reset_index)],
    )
    env.reset()
    reset_cache_audit = env.clear_reset_contact_cache_for_privileged_control()
    env.set_privileged_joint_targets(reset_q)
    controller = ObjectSpaceContactController()
    stage = "SETTLE"
    stage_steps = 0
    hold_start = -1
    lift_start = -1
    force_history = []
    object_history = []
    palm_history = []
    support_history = []
    trace_rows = []
    contact_rows = []
    verified_non_target = False
    unresolved_contact = False
    applied_target = reset_q[0].copy()
    for physics_frame_id in range(160):
        env.step(env._zero_action)
        snapshot = env.privileged_snapshot()
        events = env.consume_forensic_contact_events()[0]
        current_stage = stage
        target_positions, normals = _forensic_contact_targets(snapshot, candidate)
        pairs = tuple(
            (event.actor0, event.actor1)
            for event in events
            if "Robot" in event.actor0 or "Robot" in event.actor1
        )
        attributions = []
        for finger in np.flatnonzero(active_mask):
            attribution = attribute_contact(
                all_force_xyz=snapshot["all_force_xyz"][0, finger],
                filtered_forces={
                    "TARGET_OBJECT": snapshot["target_force_xyz"][0, finger],
                    "TABLE": snapshot["table_force_xyz"][0, finger],
                    "GROUND": snapshot["ground_force_xyz"][0, finger],
                },
                identified_pairs=pairs,
                filter_valid={
                    "TARGET_OBJECT": bool(snapshot["target_filter_valid"][0]),
                    "TABLE": bool(snapshot["table_filter_valid"][0]),
                    "GROUND": bool(snapshot["ground_filter_valid"][0]),
                },
            )
            attributions.append({"finger": int(finger + 1), **attribution.to_dict()})
            unresolved_contact |= attribution.instrumentation_limit
            verified_non_target |= attribution.source in {
                ContactSource.TABLE,
                ContactSource.GROUND,
                ContactSource.SELF,
                ContactSource.MULTIPLE_IDENTIFIED,
            }
        link_sources = link_contact_source_map(env.robot_body_names_for_forensics(), events)
        link_sources = merge_filtered_fingertip_sources(link_sources, attributions)
        verified_non_target |= any(
            source in {"TABLE", "GROUND", "SELF"}
            for sources in link_sources.values()
            for source in sources
        )
        unresolved_contact |= any(
            "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT" in sources for sources in link_sources.values()
        )
        active_force = snapshot["target_force_norms"][0, active_mask]
        hard_abort = float(np.max(active_force, initial=0.0)) >= 5.0
        terminal = bool(hard_abort or verified_non_target or unresolved_contact)
        termination = (
            "HARD_FORCE_ABORT"
            if hard_abort
            else "VERIFIED_NON_TARGET_SCENE_CONTACT"
            if verified_non_target
            else "CONTACT_ATTRIBUTION_INSTRUMENTATION_LIMIT"
            if unresolved_contact
            else ""
        )
        next_target = snapshot["joint_target26"][0].copy()
        if not terminal:
            command = controller.step(
                ContactControllerObservation(
                    active_finger_mask=active_mask,
                    current_tip_positions=snapshot["tip_pos_local"][0],
                    target_contact_positions=target_positions,
                    surface_normals_world=normals,
                    target_force_norms=snapshot["target_force_norms"][0],
                    target_force_n=candidate.target_force_n,
                    fingertip_jacobians=snapshot["tip_hand_jacobians"][0],
                    current_hand_q=snapshot["joint_pos26"][0, 6:],
                    joint_lower=snapshot["joint_lower26"][6:],
                    joint_upper=snapshot["joint_upper26"][6:],
                    wrist_error6=None,
                    non_target_contact=verified_non_target,
                    unresolved_contact=unresolved_contact,
                )
            )
            if current_stage != "SETTLE":
                next_target[6:] += command.hand_delta20
            stage_steps += 1
            if current_stage == "SETTLE" and stage_steps >= 30:
                stage = "HOLD"
                stage_steps = 0
                hold_start = physics_frame_id + 1
            elif current_stage == "HOLD" and stage_steps >= 30:
                stage = "LIFT"
                stage_steps = 0
                lift_start = physics_frame_id + 1
                next_target[2] += 0.00025
            elif current_stage == "LIFT":
                if stage_steps >= 64:
                    terminal = True
                    termination = "STRICT_EVALUATION_PENDING"
                else:
                    next_target[2] += 0.00025
        frame = ForensicFrame(
            physics_frame_id=physics_frame_id,
            candidate_id=candidate.candidate_id,
            reset_index=int(reset_index),
            stage=current_stage,
            applied_target26=tuple(float(value) for value in snapshot["joint_target26"][0]),
            next_target26=tuple(float(value) for value in next_target),
            actual_joint26=tuple(float(value) for value in snapshot["joint_pos26"][0]),
            joint_velocity26=tuple(float(value) for value in snapshot["joint_vel26"][0]),
            joint_error26=tuple(float(value) for value in snapshot["joint_target26"][0] - snapshot["joint_pos26"][0]),
            isaac_tip_positions=tuple(tuple(float(value) for value in row) for row in snapshot["tip_pos_local"][0]),
            isaac_tip_quat_wxyz=tuple(tuple(float(value) for value in row) for row in snapshot["tip_quat_wxyz"][0]),
            target_contact_positions=tuple(tuple(float(value) for value in row) for row in target_positions),
            object_position=tuple(float(value) for value in snapshot["object_pos_local"][0]),
            object_quat_wxyz=tuple(float(value) for value in snapshot["object_quat_wxyz"][0]),
            object_linear_velocity=tuple(float(value) for value in snapshot["object_lin_vel"][0]),
            object_angular_velocity=tuple(float(value) for value in snapshot["object_ang_vel"][0]),
            target_force_norms=tuple(float(value) for value in snapshot["target_force_norms"][0]),
            target_force_xyz=tuple(tuple(float(value) for value in row) for row in snapshot["target_force_xyz"][0]),
            all_force_xyz=tuple(tuple(float(value) for value in row) for row in snapshot["all_force_xyz"][0]),
            table_force_xyz=tuple(tuple(float(value) for value in row) for row in snapshot["table_force_xyz"][0]),
            ground_force_xyz=tuple(tuple(float(value) for value in row) for row in snapshot["ground_force_xyz"][0]),
            contact_events=tuple(events),
            link_contact_sources=link_sources,
            terminal=terminal,
            termination_layer=termination,
            metadata={
                "attributions": attributions,
                "joint_sim_target26": snapshot["joint_sim_target26"][0].tolist(),
                "target_filter_valid": bool(snapshot["target_filter_valid"][0]),
                "table_filter_valid": bool(snapshot["table_filter_valid"][0]),
                "ground_filter_valid": bool(snapshot["ground_filter_valid"][0]),
            },
        ).to_dict()
        trace_rows.append(frame)
        contact_rows.append(
            {
                "physics_frame_id": physics_frame_id,
                "events": [event.to_dict() for event in events],
                "link_contact_sources": link_sources,
                "attributions": attributions,
            }
        )
        force_history.append(snapshot["target_force_norms"][0].copy())
        object_history.append(snapshot["object_pos_local"][0].copy())
        palm_history.append(snapshot["palm_pos_local"][0].copy())
        support_history.append(
            bool(_table_support(snapshot["object_pos_local"], snapshot["object_quat_wxyz"], float(env.cfg.table_top_z_m))[0])
        )
        applied_target = next_target.copy()
        if terminal:
            break
        env.set_privileged_joint_targets(next_target.reshape(1, 26))
    final = env.privileged_snapshot()
    forces = np.asarray(force_history)
    objects = np.asarray(object_history)
    palms = np.asarray(palm_history)
    supports = np.asarray(support_history, dtype=bool)
    hold_index = hold_start if hold_start >= 0 else max(0, len(forces) - 31)
    lift_index = lift_start if lift_start >= 0 else max(hold_index + 1, len(forces) - 1)
    result = evaluate_gate_trace(
        gate=GateKind.RESET_CLOSED_HOLD_LIFT,
        candidate_id=candidate.candidate_id,
        trial_id=int(reset_index),
        target_force_norms=forces,
        active_finger_mask=active_mask,
        object_positions=objects,
        hand_positions=palms,
        table_supported=supports,
        hold_start_step=hold_index,
        lift_start_step=lift_index,
        controlled_close_completed=False,
        verified_non_target_contact=verified_non_target,
        unresolved_contact=unresolved_contact,
        post_reset_object_writes=int(final["post_reset_object_writes"][0]),
        post_reset_wrist_writes=int(final["post_reset_wrist_state_writes"][0]),
    )
    result = type(result)(
        **{
            **result.__dict__,
            "metadata": {
                **result.metadata,
                "trace_length": len(trace_rows),
                "first_physics_frame_recorded": bool(trace_rows and trace_rows[0]["physics_frame_id"] == 0),
                "reset_cache_audit": reset_cache_audit,
            },
        }
    )
    trace_rows[-1]["termination_layer"] = result.termination_layer
    trace_rows[-1]["terminal"] = True
    reset_dir = PHASE_DIR / "candidates" / candidate.candidate_id / f"reset_{reset_index}"
    _write_json(reset_dir / "reset_cache_audit.json", reset_cache_audit)
    _write_json(reset_dir / "first_frame.json", trace_rows[0])
    _write_json(reset_dir / "result.json", result.to_dict())
    _write_jsonl(reset_dir / "contact_events.jsonl", contact_rows)
    _write_csv(reset_dir / "trace.csv", [_jsonify_nested(row) for row in trace_rows])
    return result, trace_rows


def _forensic_contact_targets(snapshot: dict[str, Any], candidate: GraspCandidate):
    import numpy as np

    target_positions = snapshot["tip_pos_local"][0].copy()
    normals = np.zeros((5, 3), dtype=np.float64)
    object_rotation = _quat_wxyz_to_matrix(snapshot["object_quat_wxyz"][0])
    support_vertices = candidate.metadata.get("pad_support_vertex_local")
    if support_vertices is None or len(support_vertices) != len(candidate.finger_group):
        raise RuntimeError(f"candidate {candidate.candidate_id} lacks full 3D pad support vertices")
    for contact_index, finger_digit in enumerate(candidate.finger_group):
        finger = int(finger_digit) - 1
        contact = np.asarray(candidate.contact_positions_object[contact_index], dtype=np.float64)
        normal = object_rotation @ np.asarray(candidate.contact_normals_object[contact_index], dtype=np.float64)
        tip_rotation = _quat_wxyz_to_matrix(snapshot["tip_quat_wxyz"][0, finger])
        support_vertex = np.asarray(support_vertices[contact_index], dtype=np.float64)
        target_positions[finger] = (
            snapshot["object_pos_local"][0] + object_rotation @ contact - tip_rotation @ support_vertex
        )
        normals[finger] = normal
    return target_positions, normals


def _jsonify_nested(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        for key, value in row.items()
    }


def _run_gate(env, candidates: list[GraspCandidate], gate: GateKind) -> dict[str, Any]:
    if env.num_envs != 1:
        raise RuntimeError("formal v3 gates require sequential single-environment fresh resets")
    if gate == GateKind.RESET_CLOSED_HOLD_LIFT:
        screen = candidates[: int(args_cli.physics_count)]
        screen_rows = []
        for candidate in screen:
            rows, _trace = _run_gate_batch(env, [candidate], gate, trial_ids=[0])
            screen_rows.extend(rows)
        _write_json(PHASE_DIR / "screen_results.json", [row.to_dict() for row in screen_rows])
        ranked = sorted(
            zip(screen, screen_rows),
            key=lambda pair: (
                pair[1].passed,
                pair[1].hold_contact_steps,
                pair[1].lift_contact_duty,
                -pair[1].peak_target_force_n,
            ),
            reverse=True,
        )
        candidates = [pair[0] for pair in ranked[:4]]
    else:
        candidates = candidates[:4]
    rows = []
    traces = []
    for candidate in candidates:
        for trial_id in range(5):
            trial_rows, trial_trace = _run_gate_batch(env, [candidate], gate, trial_ids=[trial_id])
            rows.extend(trial_rows)
            traces.extend(trial_trace)
    _write_json(PHASE_DIR / "trial_results.json", [row.to_dict() for row in rows])
    _write_trace_csv(PHASE_DIR / "matched_trace.csv", traces)
    grouped = {}
    for candidate in candidates:
        candidate_rows = [row for row in rows if row.candidate_id == candidate.candidate_id]
        grouped[candidate.candidate_id] = {
            "successes": sum(int(row.passed) for row in candidate_rows),
            "trials": len(candidate_rows),
            "passed": gate_passed(candidate_rows),
            "termination_layers": [row.termination_layer for row in candidate_rows],
        }
    qualified_ids = [candidate_id for candidate_id, row in grouped.items() if row["passed"]]
    _write_json(PHASE_DIR / "gate_summary.json", {"gate": gate.name, "candidates": grouped, "qualified_candidate_ids": qualified_ids})
    video_path = ""
    replay_match = False
    if qualified_ids and args_cli.record_video:
        winner = next(candidate for candidate in candidates if candidate.candidate_id == qualified_ids[0])
        video_path, replay_match = _record_exact_replay(env, winner, gate)
    classification = f"{gate.name}_PASS" if qualified_ids else _failure_classification(args_cli.phase)
    return {
        "classification": classification,
        "gate": gate.name,
        "qualified_candidate_ids": qualified_ids,
        "candidate_results": grouped,
        "matched_video": video_path,
        "video_replay_matched": replay_match,
        "physical_grasp_success": bool(qualified_ids and gate != GateKind.RESET_CLOSED_HOLD_LIFT),
        "physical_lift_success": bool(qualified_ids),
        "full_route_success": False,
        "approach_close_lift_success": bool(
            qualified_ids and gate == GateKind.STANDOFF_APPROACH_CLOSE_LIFT
        ),
        "oracle_reset_grasp": gate == GateKind.RESET_CLOSED_HOLD_LIFT,
        "runtime_physics_fingerprint": env.physics_fingerprint(),
    }


def _run_gate_batch(env, candidates: list[GraspCandidate], gate: GateKind, *, trial_ids: Sequence[int]):
    import numpy as np
    from scipy.spatial.transform import Rotation

    n = env.num_envs
    closed = np.asarray([candidate.closed_joint_q26 for candidate in candidates])
    pre = np.asarray([candidate.pregrasp_joint_q26_by_opening["5mm"] for candidate in candidates])
    standoff = np.asarray([candidate.standoff_joint_q26_by_distance["30mm"] for candidate in candidates])
    standoff[:, 6:] = pre[:, 6:]
    transit = np.asarray(
        [candidate.metadata.get("transit_open_q26", candidate.pregrasp_joint_q26_by_opening["8mm"]) for candidate in candidates]
    )
    transit[:, :6] = standoff[:, :6]
    if gate == GateKind.RESET_CLOSED_HOLD_LIFT:
        reset_q = closed.copy()
        stages = np.asarray(["SETTLE"] * n, dtype=object)
        settle_steps = 30
    elif gate == GateKind.PREGRASP_CLOSE_LIFT:
        reset_q = pre.copy()
        stages = np.asarray(["SETTLE"] * n, dtype=object)
        settle_steps = 10
    else:
        reset_q = transit.copy()
        stages = np.asarray(["SETTLE"] * n, dtype=object)
        settle_steps = 10
    active_masks = np.asarray([_active_mask(candidate.finger_group) for candidate in candidates])
    env.configure_privileged_control(
        reset_q,
        active_masks,
        candidate_close_q20=closed[:, 6:],
        reset_seeds=[args_cli.seed + int(trial_ids[index]) for index in range(n)],
    )
    env.reset()
    env.clear_reset_contact_cache_for_privileged_control()
    env.set_privileged_joint_targets(reset_q)
    env.step(env._zero_action)
    controllers = [ObjectSpaceContactController() for _ in range(n)]
    stage_steps = np.zeros(n, dtype=np.int64)
    contact_consecutive = np.zeros(n, dtype=np.int64)
    tracking_consecutive = np.zeros(n, dtype=np.int64)
    hold_start = np.full(n, -1, dtype=np.int64)
    lift_start = np.full(n, -1, dtype=np.int64)
    close_completed = np.zeros(n, dtype=bool)
    verified_non_target = np.zeros(n, dtype=bool)
    unresolved = np.zeros(n, dtype=bool)
    force_history = []
    object_history = []
    palm_history = []
    support_history = []
    trace_rows = []
    max_steps = 460
    for step in range(max_steps):
        snapshot = env.privileged_snapshot()
        pairs = env.consume_contact_pairs()
        target_q = snapshot["joint_target26"].copy()
        attribution_debug = [None] * n
        for env_id in range(n):
            if stages[env_id] in {"DONE", "FAILED"}:
                continue
            rotation = _quat_wxyz_to_matrix(snapshot["object_quat_wxyz"][env_id])
            target_positions = snapshot["tip_pos_local"][env_id].copy()
            normals = np.zeros((5, 3), dtype=np.float64)
            for contact_index, finger_digit in enumerate(candidates[env_id].finger_group):
                finger = int(finger_digit) - 1
                local_point = np.asarray(candidates[env_id].contact_positions_object[contact_index])
                local_normal = np.asarray(candidates[env_id].contact_normals_object[contact_index])
                normal_world = rotation @ local_normal
                support_vertices = candidates[env_id].metadata.get("pad_support_vertex_local", ())
                if len(support_vertices) != len(candidates[env_id].finger_group):
                    raise RuntimeError(f"candidate {candidates[env_id].candidate_id} lacks full pad support vertices")
                tip_rotation = _quat_wxyz_to_matrix(snapshot["tip_quat_wxyz"][env_id, finger])
                support_vertex = np.asarray(support_vertices[contact_index], dtype=np.float64)
                target_positions[finger] = (
                    snapshot["object_pos_local"][env_id]
                    + rotation @ local_point
                    - tip_rotation @ support_vertex
                )
                normals[finger] = normal_world
            robot_pairs = tuple(pair for pair in pairs[env_id] if "Robot" in pair[0] or "Robot" in pair[1])
            finger_attributions = []
            for finger in np.flatnonzero(active_masks[env_id]):
                attribution = attribute_contact(
                    all_force_xyz=snapshot["all_force_xyz"][env_id, finger],
                    filtered_forces={
                        "TARGET_OBJECT": snapshot["target_force_xyz"][env_id, finger],
                        "TABLE": snapshot["table_force_xyz"][env_id, finger],
                        "GROUND": snapshot["ground_force_xyz"][env_id, finger],
                    },
                    identified_pairs=robot_pairs,
                    filter_valid={
                        "TARGET_OBJECT": bool(snapshot["target_filter_valid"][env_id]),
                        "TABLE": bool(snapshot["table_filter_valid"][env_id]),
                        "GROUND": bool(snapshot["ground_filter_valid"][env_id]),
                    },
                )
                finger_attributions.append({"finger": int(finger + 1), **attribution.to_dict()})
                unresolved[env_id] |= attribution.instrumentation_limit
                verified_non_target[env_id] |= attribution.source in {
                    ContactSource.TABLE,
                    ContactSource.GROUND,
                    ContactSource.SELF,
                    ContactSource.MULTIPLE_IDENTIFIED,
                }
            attribution_debug[env_id] = {
                "actor_pairs": [list(pair) for pair in robot_pairs],
                "fingers": finger_attributions,
            }
            active_force = snapshot["target_force_norms"][env_id, active_masks[env_id]]
            if float(np.max(active_force, initial=0.0)) >= 5.0 or unresolved[env_id] or verified_non_target[env_id]:
                stages[env_id] = "FAILED"
                continue
            if stages[env_id] == "SETTLE":
                target_q[env_id] = reset_q[env_id]
                if stage_steps[env_id] + 1 >= settle_steps:
                    if gate == GateKind.RESET_CLOSED_HOLD_LIFT:
                        stages[env_id] = "CLOSE"
                    elif gate == GateKind.PREGRASP_CLOSE_LIFT:
                        stages[env_id] = "CLOSE"
                    else:
                        stages[env_id] = "PRESHAPE"
                    stage_steps[env_id] = 0
                else:
                    stage_steps[env_id] += 1
                continue
            if stages[env_id] == "PRESHAPE":
                delta = pre[env_id, 6:] - target_q[env_id, 6:]
                target_q[env_id, :6] = standoff[env_id, :6]
                target_q[env_id, 6:] += np.clip(delta, -0.002, 0.002)
                actual_error = float(np.max(np.abs(pre[env_id, 6:] - snapshot["joint_pos26"][env_id, 6:])))
                tracking_consecutive[env_id] = tracking_consecutive[env_id] + 1 if actual_error <= 0.003 else 0
                stage_steps[env_id] += 1
                if tracking_consecutive[env_id] >= 3:
                    stages[env_id] = "APPROACH"
                    stage_steps[env_id] = 0
                elif stage_steps[env_id] >= 180:
                    stages[env_id] = "FAILED"
                continue
            if stages[env_id] == "APPROACH":
                ratio = min(1.0, float(stage_steps[env_id] + 1) / 120.0)
                target_q[env_id, :6] = standoff[env_id, :6] + ratio * (pre[env_id, :6] - standoff[env_id, :6])
                target_q[env_id, 6:] = pre[env_id, 6:]
                stage_steps[env_id] += 1
                if ratio >= 1.0:
                    stages[env_id] = "CLOSE"
                    stage_steps[env_id] = 0
                continue
            wrist_error = closed[env_id, :6] - snapshot["joint_target26"][env_id, :6]
            command = controllers[env_id].step(
                ContactControllerObservation(
                    active_finger_mask=active_masks[env_id],
                    current_tip_positions=snapshot["tip_pos_local"][env_id],
                    target_contact_positions=target_positions,
                    surface_normals_world=normals,
                    target_force_norms=snapshot["target_force_norms"][env_id],
                    target_force_n=candidates[env_id].target_force_n,
                    fingertip_jacobians=snapshot["tip_hand_jacobians"][env_id],
                    current_hand_q=snapshot["joint_pos26"][env_id, 6:],
                    joint_lower=snapshot["joint_lower26"][6:],
                    joint_upper=snapshot["joint_upper26"][6:],
                    wrist_error6=wrist_error if stages[env_id] != "LIFT" else None,
                    non_target_contact=bool(verified_non_target[env_id]),
                    unresolved_contact=bool(unresolved[env_id]),
                )
            )
            target_q[env_id, 6:] += command.hand_delta20
            if stages[env_id] != "LIFT":
                target_q[env_id, :6] += command.wrist_delta6
            if stages[env_id] == "CLOSE":
                all_contact = bool(np.all(active_force > 0.05))
                contact_consecutive[env_id] = contact_consecutive[env_id] + 1 if all_contact else 0
                stage_steps[env_id] += 1
                if contact_consecutive[env_id] >= 3:
                    stages[env_id] = "HOLD"
                    hold_start[env_id] = step + 1
                    close_completed[env_id] = True
                    stage_steps[env_id] = 0
                elif stage_steps[env_id] >= 180:
                    stages[env_id] = "FAILED"
            elif stages[env_id] == "HOLD":
                stage_steps[env_id] += 1
                if stage_steps[env_id] >= 30:
                    stages[env_id] = "LIFT"
                    lift_start[env_id] = step + 1
                    stage_steps[env_id] = 0
            elif stages[env_id] == "LIFT":
                target_q[env_id, 2] += 0.00025
                stage_steps[env_id] += 1
                if stage_steps[env_id] >= 64:
                    stages[env_id] = "DONE"
        force_history.append(snapshot["target_force_norms"])
        object_history.append(snapshot["object_pos_local"])
        palm_history.append(snapshot["palm_pos_local"])
        support = _table_support(snapshot["object_pos_local"], snapshot["object_quat_wxyz"], float(env.cfg.table_top_z_m))
        support_history.append(support)
        for env_id in range(n):
            trace_rows.append(
                {
                    "step": step,
                    "env_id": env_id,
                    "candidate_id": candidates[env_id].candidate_id,
                    "trial_id": int(trial_ids[env_id]),
                    "stage": str(stages[env_id]),
                    "physics_frame_id": step,
                    "applied_target26": json.dumps(snapshot["joint_target26"][env_id].tolist()),
                    "next_target26": json.dumps(target_q[env_id].tolist()),
                    "actual_joint26": json.dumps(snapshot["joint_pos26"][env_id].tolist()),
                    "joint_error26": json.dumps((snapshot["joint_target26"][env_id] - snapshot["joint_pos26"][env_id]).tolist()),
                    "tip_pos_local": json.dumps(snapshot["tip_pos_local"][env_id].tolist()),
                    "target_force_n": json.dumps(snapshot["target_force_norms"][env_id].tolist()),
                    "target_force_xyz": json.dumps(snapshot["target_force_xyz"][env_id].tolist()),
                    "all_force_xyz": json.dumps(snapshot["all_force_xyz"][env_id].tolist()),
                    "table_force_xyz": json.dumps(snapshot["table_force_xyz"][env_id].tolist()),
                    "ground_force_xyz": json.dumps(snapshot["ground_force_xyz"][env_id].tolist()),
                    "contact_attribution": json.dumps(attribution_debug[env_id], sort_keys=True),
                    "object_pos_local": json.dumps(snapshot["object_pos_local"][env_id].tolist()),
                    "object_linear_velocity": json.dumps(snapshot["object_lin_vel"][env_id].tolist()),
                    "object_angular_velocity": json.dumps(snapshot["object_ang_vel"][env_id].tolist()),
                    "table_supported": bool(support[env_id]),
                    "unresolved_contact": bool(unresolved[env_id]),
                    "verified_non_target_contact": bool(verified_non_target[env_id]),
                    "terminal": bool(stages[env_id] in {"DONE", "FAILED"}),
                }
            )
        if np.all(np.isin(stages, ("DONE", "FAILED"))):
            break
        env.set_privileged_joint_targets(target_q)
        env.step(env._zero_action)
    force_array = np.asarray(force_history)
    object_array = np.asarray(object_history)
    palm_array = np.asarray(palm_history)
    support_array = np.asarray(support_history)
    final = env.privileged_snapshot()
    results = []
    for env_id, candidate in enumerate(candidates):
        hold_index = int(hold_start[env_id] if hold_start[env_id] >= 0 else max(0, len(force_array) - 31))
        lift_index = int(lift_start[env_id] if lift_start[env_id] >= 0 else max(hold_index + 1, len(force_array) - 1))
        result = evaluate_gate_trace(
            gate=gate,
            candidate_id=candidate.candidate_id,
            trial_id=int(trial_ids[env_id]),
            target_force_norms=force_array[:, env_id],
            active_finger_mask=active_masks[env_id],
            object_positions=object_array[:, env_id],
            hand_positions=palm_array[:, env_id],
            table_supported=support_array[:, env_id],
            hold_start_step=hold_index,
            lift_start_step=lift_index,
            controlled_close_completed=bool(close_completed[env_id]),
            verified_non_target_contact=bool(verified_non_target[env_id]),
            unresolved_contact=bool(unresolved[env_id]),
            post_reset_object_writes=int(final["post_reset_object_writes"][env_id]),
            post_reset_wrist_writes=int(final["post_reset_wrist_state_writes"][env_id]),
        )
        results.append(result)
    return results, trace_rows


def _record_exact_replay(env, candidate: GraspCandidate, gate: GateKind) -> tuple[str, bool]:
    candidates = [candidate] * env.num_envs
    frames = []
    original_step = env.step

    def recording_step(action):
        output = original_step(action)
        frame = env.replay_rgb_frame(0)
        if frame is not None:
            snapshot = env.replay_snapshot(0)
            frames.append(_overlay_frame(frame, candidate, gate, snapshot))
        return output

    env.step = recording_step
    try:
        rows, trace = _run_gate_batch(env, candidates, gate, trial_ids=[0] * env.num_envs)
    finally:
        env.step = original_step
    matched = bool(rows[0].passed)
    if not matched or not frames:
        _write_json(PHASE_DIR / "video_replay_mismatch.json", rows[0].to_dict())
        return "", False
    slug = SPEC.part_name.lower()
    video_name = {
        GateKind.RESET_CLOSED_HOLD_LIFT: f"multi_object_privileged_grasp_v3_{slug}_gate_a_oracle_lift.mp4",
        GateKind.PREGRASP_CLOSE_LIFT: f"multi_object_privileged_grasp_v3_{slug}_gate_b_close_and_lift.mp4",
        GateKind.STANDOFF_APPROACH_CLOSE_LIFT: f"multi_object_privileged_grasp_v3_{slug}_gate_c_approach_close_lift.mp4",
    }[gate]
    path = OUTPUT_ROOT / "videos" / video_name
    _write_h264(path, frames, fps=60)
    alignment = [
        {"run_id": f"{SPEC.part_name}_{gate.name}_{candidate.candidate_id}", "frame_index": index, "trace_row": min(index, len(trace) - 1)}
        for index in range(len(frames))
    ]
    _write_csv(path.with_name(path.stem + "_alignment.csv"), alignment)
    return str(path), True


def _table_support(object_pos, object_quat, table_top_z: float):
    import numpy as np

    local_points = _geometry_support_points()
    supported = []
    for position, quat in zip(object_pos, object_quat):
        rotation = _quat_wxyz_to_matrix(quat)
        bottom = float(np.min((local_points @ rotation.T + position.reshape(1, 3))[:, 2]))
        supported.append(bottom <= table_top_z + 0.002)
    return np.asarray(supported, dtype=bool)


def _geometry_support_points():
    import numpy as np

    geometry = SPEC.geometry
    if geometry.kind in {"cylinder_z", "capsule_z"}:
        center = np.asarray(geometry.center_object)
        points = []
        for z in (-geometry.half_length_m, geometry.half_length_m):
            for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                points.append(center + (geometry.radius_m * np.cos(angle), geometry.radius_m * np.sin(angle), z))
        return np.asarray(points)
    if geometry.kind == "box":
        return _box_corners(np.asarray(geometry.center_object), np.asarray(geometry.half_extents_m))
    if geometry.kind == "box_union":
        return np.concatenate([_box_corners(np.asarray(center), np.asarray(half)) for center, half in geometry.boxes], axis=0)
    return np.zeros((1, 3))


def _box_corners(center, half):
    import numpy as np

    return np.asarray([center + np.asarray((sx, sy, sz)) * half for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _quat_wxyz_to_matrix(quat):
    import numpy as np
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat((q[1], q[2], q[3], q[0])).as_matrix()


def _active_mask(group: str):
    import numpy as np

    mask = np.zeros(5, dtype=bool)
    for digit in group:
        mask[int(digit) - 1] = True
    return mask


def _pad_batch(candidates: Sequence[GraspCandidate], count: int) -> list[GraspCandidate]:
    if not candidates:
        raise ValueError("cannot pad an empty candidate batch")
    rows = list(candidates)
    while len(rows) < count:
        rows.append(rows[len(rows) % len(candidates)])
    return rows[:count]


def _load_candidates() -> list[GraspCandidate]:
    if args_cli.candidate_json:
        path = Path(args_cli.candidate_json).resolve()
    else:
        latest = CACHE.root / SPEC.part_name / "latest_m0.json"
        if not latest.is_file():
            raise FileNotFoundError(f"candidate cache pointer missing: {latest}")
        path = Path(json.loads(latest.read_text(encoding="utf-8"))["candidate_json"])
    pointer_hash = None
    pointer = CACHE.root / SPEC.part_name / "latest_m0.json"
    if pointer.is_file() and not args_cli.candidate_json:
        pointer_hash = str(json.loads(pointer.read_text(encoding="utf-8"))["input_hash"])
    payloads, _manifest = CACHE.load_validated(
        path,
        expected_part_name=SPEC.part_name,
        expected_input_hash=pointer_hash,
    )
    rows = [GraspCandidate.from_dict(payload) for payload in payloads]
    return [
        row
        for row, payload in zip(rows, payloads)
        if is_formal_gate_candidate(payload, "gate_a") and row.input_hash == str(_manifest["input_hash"])
    ]


def _eligible_candidates(candidates: list[GraspCandidate], gate: GateKind) -> list[GraspCandidate]:
    gate_key = {
        GateKind.RESET_CLOSED_HOLD_LIFT: "gate_a",
        GateKind.PREGRASP_CLOSE_LIFT: "gate_b",
        GateKind.STANDOFF_APPROACH_CLOSE_LIFT: "gate_c",
    }[gate]
    candidates = [
        candidate
        for candidate in candidates
        if candidate.optimization_success
        and candidate.contact_residual_success
        and candidate.physics_gate_a_eligible
        and candidate.gate_eligibility.get(gate_key, False)
    ]
    if gate == GateKind.RESET_CLOSED_HOLD_LIFT:
        return candidates[: int(args_cli.physics_count)]
    previous = OUTPUT_ROOT / ("gate_a" if gate == GateKind.PREGRASP_CLOSE_LIFT else "gate_b") / "gate_summary.json"
    if not previous.is_file():
        return []
    ids = set(json.loads(previous.read_text(encoding="utf-8"))["qualified_candidate_ids"])
    return [candidate for candidate in candidates if candidate.candidate_id in ids]


def _gate_kind() -> GateKind:
    return {
        "gate_a": GateKind.RESET_CLOSED_HOLD_LIFT,
        "gate_b": GateKind.PREGRASP_CLOSE_LIFT,
        "gate_c": GateKind.STANDOFF_APPROACH_CLOSE_LIFT,
    }[args_cli.phase]


def _overlay_frame(frame, candidate: GraspCandidate, gate: GateKind, snapshot: dict[str, Any]):
    import cv2

    output = frame.copy()
    lines = [
        f"{SPEC.part_name} {gate.name}",
        f"candidate: {candidate.candidate_id}",
        f"phase: {snapshot['phase']}",
        f"force N: {[round(value, 3) for value in snapshot['target_force_n']]}",
        "CEM: OFF   PPO: OFF",
        "NO STICKY / NO PROXY / RESET-ONLY WRITES",
    ]
    for index, line in enumerate(lines):
        cv2.putText(output, line, (20, 34 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _write_h264(path: Path, frames: Sequence[Any], *, fps: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "_temporary.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    temporary.unlink(missing_ok=True)
    if result.returncode != 0 or not path.is_file():
        raise RuntimeError(f"H.264 encoding failed: {result.stdout[-1000:]}")


def _gpu_preflight() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines() if result.returncode == 0 else []:
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append({"index": int(values[0]), "name": values[1], "total_mib": float(values[2]), "used_mib": float(values[3]), "free_mib": float(values[4])})
    blocked = result.returncode != 0 or max([0.0, *[row["free_mib"] for row in rows]]) < float(args_cli.min_gpu_free_gb) * 1024.0
    if blocked and args_cli.phase != "audit":
        raise RuntimeError("HOST_GPU_BUSY")
    return {"classification": "HOST_GPU_BUSY" if blocked else "HEALTH_OK", "gpus": rows, "processes_killed": False, "raw_output": result.stdout[-2000:]}


def _failure_classification(phase: str) -> str:
    return {
        "audit": "ASSET_GEOMETRY_AUDIT_FAILED",
        "audit_before": "CODE_FACT_AUDIT_FAILED",
        "external_priors": "EXTERNAL_PRIOR_AUDIT_FAILED",
        "synthesize": "IK_COLLISION_PATH_INFEASIBLE",
        "m0_screen": "NO_VALID_M0_CANDIDATE",
        "m0_full": "NO_VALID_M0_CANDIDATE",
        "synthesize_forensic_v2": "FORENSIC_IK_COLLISION_PATH_INFEASIBLE",
        "vector_smoke": "NEAR_GRASP_ENV_VECTORIZATION_BLOCKED",
        "forensic_audit": "FORENSIC_RUNTIME_AUDIT_FAILED",
        "runtime_audit": "RUNTIME_COLLISION_MESH_EXPORT_BLOCKED",
        "gate_a_forensic_v2": "GATE_A_FORENSIC_V2_FAILED",
        "gate_a": "GATE_A_RESET_HOLD_LIFT_FAILED",
        "gate_b": "GATE_B_PREGRASP_CLOSE_LIFT_FAILED",
        "gate_c": "GATE_C_STANDOFF_APPROACH_CLOSE_LIFT_FAILED",
    }[phase]


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    main()
