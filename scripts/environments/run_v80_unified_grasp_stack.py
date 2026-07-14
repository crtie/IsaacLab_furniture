"""Run v80 unified multi-object dexterous grasp diagnostics and RL gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
for rel in ("source/isaaclab", "source/isaaclab_tasks", "source/isaaclab_assets"):
    path = str(REPO_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
PACKAGE_ROOT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipeline.unified_grasp.asset_audit import audit_assets  # noqa: E402
from pipeline.unified_grasp.contact_manager import probe_contact_signals  # noqa: E402
from pipeline.unified_grasp.curriculum import write_curriculum_artifacts  # noqa: E402
from pipeline.unified_grasp.eval_unified_rl import evaluate_unified_policy  # noqa: E402
from pipeline.unified_grasp.grasp_prior_generator import write_grasp_prior_bank  # noqa: E402
from pipeline.unified_grasp.object_randomization import write_object_randomization_report  # noqa: E402
from pipeline.unified_grasp.seed_bank import load_unified_seed_bank  # noqa: E402
from pipeline.unified_grasp.serial_replay_validator import validate_single_final_replay  # noqa: E402
from pipeline.unified_grasp.train_unified_rl import train_unified_policy  # noqa: E402
from pipeline.unified_grasp.unified_grasp_env import run_env_smoke  # noqa: E402
from pipeline.unified_grasp.v80_reports import (  # noqa: E402
    V80_PARTS,
    default_progress_row,
    ensure_run_dir,
    merge_progress_rows,
    write_failure_taxonomy,
    write_json,
    write_progress_matrix,
    write_source_archive_check,
)


RUN_DIR_BY_MODE = {
    "asset_contact_foundation_audit": "debug_runs/v80_asset_contact_foundation_audit",
    "unified_env_vectorized_smoke": "debug_runs/v80_unified_env_vectorized_smoke",
    "unified_residual_rl_multitask_smoke": "debug_runs/v80_unified_residual_rl_multitask_smoke",
    "per_object_policy_smoke": "debug_runs/v80_per_object_policy_smoke",
    "single_final_replay_validation": "debug_runs/v80_single_final_replay_validation",
}


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_mode", choices=tuple(RUN_DIR_BY_MODE), default="asset_contact_foundation_audit")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--parts", default=",".join(V80_PARTS))
    parser.add_argument("--physics_profile", default="canonical")
    parser.add_argument("--policy_mode", choices=("shared_multi_object_policy", "per_object_policy"), default="shared_multi_object_policy")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=48)
    parser.add_argument("--top_k_priors_per_part", type=int, default=32)
    parser.add_argument("--program_prior_search_roots", default="debug_runs")
    parser.add_argument("--allow_diagnostic_dynamics", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--sticky_after_support_validation", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--checkpoint_path", default="")
    return parser.parse_args()


def _parts(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    return [part for part in V80_PARTS if part in requested] or list(V80_PARTS)


def _run_dir(args: argparse.Namespace) -> Path:
    return ensure_run_dir(args.output_dir or RUN_DIR_BY_MODE[args.run_mode])


def _progress_from_foundation(
    *,
    parts: list[str],
    physics_profile: str,
    asset_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    priors: list[dict[str, Any]],
    curriculum_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        asset = next((row for row in asset_rows if row.get("part_name") == part), {})
        contact = next((row for row in contact_rows if row.get("part_name") == part), {})
        contact_available = bool(contact.get("contact_sensor_available"))
        seed_count = sum(1 for row in seeds if row.get("part_name") == part)
        prior_count = sum(1 for row in priors if row.get("part_name") == part and bool(row.get("reachable_prior")))
        curriculum_count = sum(1 for row in curriculum_rows if row.get("part_name") == part)
        failure = "" if contact_available else "CONTACT_SENSOR_UNAVAILABLE_FALLBACK_ONLY"
        blocker = "" if contact_available else "contact_sensor_unavailable_or_not_probed"
        rows.append(
            default_progress_row(
                part,
                asset_audit_ok=bool(asset.get("asset_audit_ok")),
                contact_sensor_available=contact_available,
                contact_evidence_source=contact.get("contact_evidence_source", "not_probed"),
                physics_profile=physics_profile,
                seed_count=seed_count,
                reachable_prior_count=prior_count,
                curriculum_state_count=curriculum_count,
                failure_category=failure or "TRAINING_NOT_STARTED",
                blocker=blocker or "ready_for_unified_env_smoke",
                next_action="run_v80_unified_env_vectorized_smoke",
            )
        )
    return rows


def _progress_from_env(parts: list[str], env_rows: list[dict[str, Any]], policy_mode: str) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        env = next((row for row in env_rows if row.get("part_name") == part), {})
        ok = bool(env.get("rl_env_vectorized_ok"))
        rows.append(
            {
                "part_name": part,
                "rl_env_vectorized_ok": ok,
                "actual_env_count": int(env.get("actual_env_count") or 0),
                "policy_mode": policy_mode,
                "failure_category": "" if ok else "ENV_NOT_VECTORIZEABLE",
                "blocker": "" if ok else str(env.get("blocker") or "physical_isaac_backend_not_configured"),
                "next_action": "run_v80_unified_residual_rl_multitask_smoke" if ok else "attach_real_isaac_contact_vector_backend",
            }
        )
    return rows


def _progress_from_training(parts: list[str], train_rows: list[dict[str, Any]], train_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        train = next((row for row in train_rows if row.get("part_name") == part), {})
        trained = bool(train.get("rl_trained"))
        rows.append(
            {
                "part_name": part,
                "policy_mode": str(train.get("policy_mode") or ""),
                "rl_trained": trained,
                "training_curve_nonempty": bool(train.get("training_curve_nonempty")),
                "checkpoint_path": str(train_result.get("checkpoint_path") or ""),
                "failure_category": str(train.get("failure_category") or ("CONTACT_SENSOR_UNAVAILABLE_FALLBACK_ONLY" if trained else "TRAINING_NOT_STARTED")),
                "blocker": str(train.get("blocker") or train_result.get("failure_reason") or ""),
                "next_action": "run_v80_eval" if trained else "fix_training_blocker",
            }
        )
    return rows


def _progress_from_eval(parts: list[str], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for part in parts:
        row = next((item for item in eval_rows if item.get("part_name") == part), {})
        eval_episode_count = int(row.get("eval_episode_count") or 0)
        update = {
            "part_name": part,
            "eval_episode_count": eval_episode_count,
            "success_no_sticky_rate": float(row.get("success_no_sticky_rate") or 0.0),
            "success_with_sticky_after_support_rate": float(row.get("success_with_sticky_after_support_rate") or 0.0),
            "support_gate_rate": float(row.get("support_gate_rate") or 0.0),
            "force_contact_count_mean": float(row.get("average_effective_contact_count_force") or 0.0),
            "distance_contact_count_mean": float(row.get("average_effective_contact_count_distance") or 0.0),
            "object_motion_before_contact_rate": float(row.get("object_motion_before_contact_rate") or 0.0),
            "penetration_rate": float(row.get("penetration_rate") or 0.0),
            "table_collision_rate": float(row.get("table_collision_rate") or 0.0),
            "lift_success_rate": float(row.get("lift_success_rate") or 0.0),
            "sticky_used_as_stabilizer": bool(row.get("sticky_used_as_stabilizer")),
            "usable_training_row_count": int(row.get("usable_training_row_count") or 0),
        }
        if eval_episode_count > 0 or bool(row.get("support_gate_ok")):
            update.update(
                {
                    "failure_category": str(row.get("failure_category") or "TRAINING_NOT_STARTED"),
                    "blocker": str(row.get("blocker") or ""),
                    "next_action": "single_final_replay_validation" if bool(row.get("support_gate_ok")) else "improve_policy_contact_support",
                }
            )
        rows.append(update)
    return rows


def main() -> int:
    args = _parse_args()
    parts = _parts(args.parts)
    run_dir = _run_dir(args)
    metadata: dict[str, Any] = {"run_mode": args.run_mode, "parts": parts, "repo_root": str(REPO_ROOT)}

    asset = audit_assets(run_dir, parts=parts, physics_profile=args.physics_profile)
    randomization = write_object_randomization_report(run_dir, parts=parts)
    contact = probe_contact_signals(run_dir, parts=parts)
    seeds = load_unified_seed_bank(
        run_dir,
        search_roots=[item.strip() for item in args.program_prior_search_roots.split(",") if item.strip()],
        parts=parts,
    )
    priors = write_grasp_prior_bank(run_dir, parts=parts, seeds=seeds["rows"], top_k_per_part=args.top_k_priors_per_part)
    curriculum = write_curriculum_artifacts(run_dir, seeds=seeds["rows"], priors=priors["rows"], parts=parts, physics_profile=args.physics_profile)
    progress_rows = _progress_from_foundation(
        parts=parts,
        physics_profile=args.physics_profile,
        asset_rows=asset["rows"],
        contact_rows=contact["rows"],
        seeds=seeds["rows"],
        priors=priors["rows"],
        curriculum_rows=curriculum["rows"],
    )

    env_result: dict[str, Any] = {}
    train_result: dict[str, Any] = {}
    eval_result: dict[str, Any] = {}
    replay_result: dict[str, Any] = {}

    if args.run_mode in {
        "unified_env_vectorized_smoke",
        "unified_residual_rl_multitask_smoke",
        "per_object_policy_smoke",
    }:
        env_result = run_env_smoke(
            run_dir,
            num_envs=args.num_envs,
            steps=min(args.max_steps, 12),
            parts=parts,
            physics_profile=args.physics_profile,
            allow_diagnostic_dynamics=args.allow_diagnostic_dynamics,
        )
        progress_rows = merge_progress_rows(progress_rows, _progress_from_env(parts, env_result["rows"], args.policy_mode))

    if args.run_mode in {"unified_residual_rl_multitask_smoke", "per_object_policy_smoke"}:
        policy_mode = "per_object_policy" if args.run_mode == "per_object_policy_smoke" else args.policy_mode
        train_result = train_unified_policy(
            run_dir,
            policy_mode=policy_mode,
            parts=parts,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            physics_profile=args.physics_profile,
            allow_diagnostic_dynamics=args.allow_diagnostic_dynamics,
        )
        progress_rows = merge_progress_rows(progress_rows, _progress_from_training(parts, train_result.get("rows", []), train_result))
        eval_result = evaluate_unified_policy(
            run_dir,
            checkpoint_path=str(train_result.get("checkpoint_path") or ""),
            parts=parts,
            sticky_after_support=args.sticky_after_support_validation,
            eval_episode_count=8 if train_result.get("rl_trained") else 0,
        )
        progress_rows = merge_progress_rows(progress_rows, _progress_from_eval(parts, eval_result["rows"]))

    if args.run_mode == "single_final_replay_validation":
        eval_result = evaluate_unified_policy(
            run_dir,
            checkpoint_path=args.checkpoint_path,
            parts=parts,
            sticky_after_support=args.sticky_after_support_validation,
            eval_episode_count=8 if args.checkpoint_path else 0,
        )
        candidate = next((row for row in eval_result["rows"] if row.get("support_gate_ok")), None)
        replay_result = validate_single_final_replay(
            run_dir,
            candidate=candidate,
            allow_sticky_after_support=args.sticky_after_support_validation,
        )
        progress_rows = merge_progress_rows(progress_rows, _progress_from_eval(parts, eval_result["rows"]))

    progress = write_progress_matrix(run_dir, progress_rows, also_root=True)
    failure_rows = [
        {"part_name": row.get("part_name"), "failure_category": row.get("failure_category"), "blocker": row.get("blocker")}
        for row in progress_rows
    ]
    failure = write_failure_taxonomy(run_dir, failure_rows)
    source = write_source_archive_check(REPO_ROOT, run_dir)
    metadata.update(
        {
            "asset": {k: v for k, v in asset.items() if k != "rows"},
            "randomization": randomization,
            "contact": {k: v for k, v in contact.items() if k != "rows"},
            "seeds": {k: v for k, v in seeds.items() if k != "rows"},
            "priors": {k: v for k, v in priors.items() if k != "rows"},
            "curriculum": {k: v for k, v in curriculum.items() if k != "rows"},
            "env": {k: v for k, v in env_result.items() if k != "rows"},
            "training": {k: v for k, v in train_result.items() if k != "rows"},
            "eval": {k: v for k, v in eval_result.items() if k not in {"rows", "failure_rows"}},
            "replay": {k: v for k, v in replay_result.items() if k != "rows"},
            "progress": progress,
            "failure": failure,
            "source_archive": source,
            "sticky_disabled_during_training": True,
            "sticky_after_support_validation": bool(args.sticky_after_support_validation),
            "allow_diagnostic_dynamics": bool(args.allow_diagnostic_dynamics),
        }
    )
    metadata_path = write_json(run_dir / "v80_run_metadata.json", metadata)
    print(f"v80 run complete: {run_dir}")
    print(f"metadata: {metadata_path}")
    print(f"progress: {progress.get('v80_progress_csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
