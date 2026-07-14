"""Orchestrate the bounded five-object privileged-physics grasp v3 pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PARTS = ("Rod", "Backrest", "Frame", "Screw1", "Plug2")
BIG_PARTS = ("Rod", "Backrest", "Frame")
RUN_ROOT = REPO_ROOT / "debug_runs/multi_object_privileged_grasp_v3"
CHILD = REPO_ROOT / "scripts/environments/run_privileged_physics_grasp_v1.py"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--phase",
    choices=("audit_before", "external_priors", "runtime_audit", "m0_screen", "m0_full", "difficulty", "gates", "full"),
    default="full",
)
parser.add_argument("--parts", nargs="*", choices=PARTS, default=list(PARTS))
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--record-video", action="store_true")
args = parser.parse_args()


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    requested = tuple(args.parts)
    phases = {
        "audit_before": _audit_before,
        "external_priors": _external_priors,
        "runtime_audit": lambda: _runtime_audits(requested),
        "m0_screen": lambda: _m0(requested, "m0_screen"),
        "m0_full": lambda: _m0(tuple(part for part in BIG_PARTS if part in requested), "m0_full"),
        "difficulty": _difficulty,
        "gates": _gates,
    }
    if args.phase == "full":
        _audit_before()
        _external_priors()
        _runtime_audits(requested)
        _m0(requested, "m0_screen")
        _m0(tuple(part for part in BIG_PARTS if part in requested), "m0_full")
        _difficulty()
        _gates()
    else:
        phases[args.phase]()


def _base(part: str, phase: str) -> list[str]:
    return [
        str(CHILD),
        "--phase",
        phase,
        "--part",
        part,
        "--output-dir",
        "debug_runs/multi_object_privileged_grasp_v3",
        "--seed",
        str(args.seed),
        "--min-gpu-free-gb",
        "12",
    ]


def _audit_before() -> None:
    _run([sys.executable, *_base("Rod", "audit_before")])


def _external_priors() -> None:
    _run([sys.executable, *_base("Rod", "external_priors")])


def _runtime_audits(parts: tuple[str, ...]) -> None:
    for part in parts:
        command = [
            str(REPO_ROOT / "isaaclab.sh"),
            "-p",
            *_base(part, "runtime_audit"),
            "--headless",
            "--device",
            args.device,
        ]
        _run(command, isaac=True)


def _m0(parts: tuple[str, ...], phase: str) -> None:
    budget = (512, 64, 32, 8) if phase == "m0_screen" else (2048, 128, 64, 16)
    for part in parts:
        command = [
            sys.executable,
            *_base(part, phase),
            "--sample-count",
            str(budget[0]),
            "--geometric-count",
            str(budget[1]),
            "--optimize-count",
            str(budget[2]),
            "--physics-count",
            str(budget[3]),
        ]
        _run(command)


def _difficulty() -> None:
    rows = []
    tie = {part: len(BIG_PARTS) - index for index, part in enumerate(BIG_PARTS)}
    for part in PARTS:
        screen = _read(RUN_ROOT / part / "m0_screen/summary.json")
        full = _read(RUN_ROOT / part / "m0_full/summary.json")
        summary = full or screen
        spec = _read(RUN_ROOT / part / "runtime_audit/resolved_object_spec.json")
        regions = spec.get("allowed_contact_regions", []) if spec else []
        thickness = min(
            (float(region.get("local_thickness_range_m", [0.0, 0.0])[1]) for region in regions),
            default=0.0,
        )
        optimized = max(int(summary.get("optimized_count", 0)), 1)
        rows.append(
            {
                "part_name": part,
                "local_thickness_m": thickness,
                "reachability_rate": float(summary.get("reachability_success_count", 0)) / optimized,
                "contact_residual_rate": float(summary.get("contact_residual_success_count", 0)) / optimized,
                "collision_feasible_rate": float(summary.get("closed_pose_collision_success_count", 0)) / optimized,
                "strict_m0_candidate_count": int(summary.get("strict_m0_candidate_count", 0)),
                "tie_priority": tie.get(part, 0),
            }
        )
    rows.sort(
        key=lambda row: (
            row["strict_m0_candidate_count"],
            row["collision_feasible_rate"],
            row["contact_residual_rate"],
            row["reachability_rate"],
            row["local_thickness_m"],
            row["tie_priority"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    _write(RUN_ROOT / "object_difficulty_ranking.json", {"schema_version": 3, "ranking": rows})
    with (RUN_ROOT / "m0_reachability_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["part_name"])
        writer.writeheader()
        writer.writerows(rows)


def _gates() -> None:
    ranking = _read(RUN_ROOT / "object_difficulty_ranking.json").get("ranking", [])
    eligible_big = [row["part_name"] for row in ranking if row["part_name"] in BIG_PARTS and row["strict_m0_candidate_count"] > 0]
    eligible_fallback = [
        row["part_name"]
        for row in ranking
        if row["part_name"] not in BIG_PARTS and row["strict_m0_candidate_count"] > 0
    ]
    eligible = eligible_big + eligible_fallback
    if not eligible:
        _write(
            RUN_ROOT / "gate_orchestration_summary.json",
            {"classification": "NO_VALID_M0_CANDIDATE", "physical_reset_count": 0, "cem_started": False, "ppo_started": False},
        )
        return
    part = eligible[0]
    for phase in ("gate_a", "gate_b", "gate_c"):
        command = [
            str(REPO_ROOT / "isaaclab.sh"),
            "-p",
            *_base(part, phase),
            "--headless",
            "--device",
            args.device,
        ]
        if args.record_video and phase == "gate_c":
            command.extend(("--record-video", "--enable_cameras", "--alignment-debug"))
        _run(command, isaac=True)
        summary = _read(RUN_ROOT / part / phase / "summary.json")
        if not summary.get("qualified_candidate_ids"):
            break


def _run(command: list[str], *, isaac: bool = False) -> None:
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    if not isaac:
        # Stage A/B solve only small dense systems.  The isaac environment's
        # 24-thread OpenBLAS default adds substantial launch overhead and makes
        # identical offline candidates needlessly slow.
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = ":".join(
        (
            str(REPO_ROOT / "source/isaaclab"),
            str(REPO_ROOT / "source/isaaclab_tasks"),
            str(REPO_ROOT / "source/isaaclab_assets"),
            env.get("PYTHONPATH", ""),
        )
    )
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        kind = "Isaac" if isaac else "offline"
        raise RuntimeError(f"{kind} child failed ({result.returncode}): {' '.join(command)}")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
