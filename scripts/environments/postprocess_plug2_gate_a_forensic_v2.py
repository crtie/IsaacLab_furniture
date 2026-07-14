"""Add Pinocchio fingertip poses to Plug2 Gate A forensic traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_GRASP_PARENT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
if str(NEAR_GRASP_PARENT) not in sys.path:
    sys.path.insert(0, str(NEAR_GRASP_PARENT))

from near_grasp.grasp_synthesis.wuji_ik import WujiKinematicModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="debug_runs/plug2_gate_a_forensic_v2")
    args = parser.parse_args()
    root = (REPO_ROOT / args.output_dir / "Plug2").resolve()
    runtime_seed = json.loads((root / "forensic_audit/runtime_kinematic_seed.json").read_text(encoding="utf-8"))
    urdf = REPO_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/wuji-hand-description/urdf/right_block_palm_floating_drone.urdf"
    model = WujiKinematicModel(
        urdf,
        runtime_joint_names=runtime_seed["runtime_joint_names"],
        package_dirs=[urdf.parent],
    )
    summaries = []
    for trace_path in sorted((root / "gate_a_forensic_v2/candidates").glob("*/reset_*/trace.csv")):
        rows = _read_csv(trace_path)
        enriched, metrics = enrich_trace_rows(rows, model)
        _write_csv(trace_path.with_name("trace_enriched.csv"), enriched)
        candidate_id = trace_path.parents[1].name
        reset_index = int(trace_path.parent.name.split("_")[-1])
        for finger_index, values in enumerate(metrics):
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "reset_index": reset_index,
                    "finger": finger_index + 1,
                    **values,
                }
            )
    if not summaries:
        raise RuntimeError("no forensic Gate A traces were found")
    phase_dir = root / "gate_a_forensic_v2"
    _write_csv(phase_dir / "fk_cross_engine_summary.csv", summaries)
    aggregate = {
        "trace_count": len(list((phase_dir / "candidates").glob("*/reset_*/trace.csv"))),
        "rows": summaries,
        "global_position_error_mm": _aggregate([row["position_error_mean_mm"] for row in summaries]),
        "global_orientation_error_deg": _aggregate([row["orientation_error_mean_deg"] for row in summaries]),
    }
    (phase_dir / "fk_cross_engine_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def enrich_trace_rows(rows: list[dict[str, str]], model: WujiKinematicModel):
    position_errors: list[list[float]] = [[] for _ in range(5)]
    orientation_errors: list[list[float]] = [[] for _ in range(5)]
    enriched = []
    for row in rows:
        q = np.asarray(json.loads(row["actual_joint26"]), dtype=np.float64)
        isaac_positions = np.asarray(json.loads(row["isaac_tip_positions"]), dtype=np.float64)
        isaac_quats = np.asarray(json.loads(row["isaac_tip_quat_wxyz"]), dtype=np.float64)
        pin_positions, pin_rotations, _ = model.tip_poses_and_jacobians(q)
        pin_quats = []
        frame_position_errors = []
        frame_orientation_errors = []
        for finger in range(5):
            xyzw = Rotation.from_matrix(pin_rotations[finger]).as_quat()
            pin_quat = np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
            pin_quats.append(pin_quat)
            position_error = float(np.linalg.norm(pin_positions[finger] - isaac_positions[finger]) * 1000.0)
            orientation_error = _quat_error_deg(pin_quat, isaac_quats[finger])
            position_errors[finger].append(position_error)
            orientation_errors[finger].append(orientation_error)
            frame_position_errors.append(position_error)
            frame_orientation_errors.append(orientation_error)
        enriched.append(
            {
                **row,
                "pinocchio_tip_positions": json.dumps(np.asarray(pin_positions).tolist()),
                "pinocchio_tip_quat_wxyz": json.dumps(np.asarray(pin_quats).tolist()),
                "fk_position_error_mm": json.dumps(frame_position_errors),
                "fk_orientation_error_deg": json.dumps(frame_orientation_errors),
            }
        )
    summaries = []
    for position_values, orientation_values in zip(position_errors, orientation_errors):
        summaries.append(
            {
                "position_error_mean_mm": float(np.mean(position_values)),
                "position_error_p95_mm": float(np.percentile(position_values, 95)),
                "position_error_max_mm": float(np.max(position_values)),
                "orientation_error_mean_deg": float(np.mean(orientation_values)),
                "orientation_error_p95_deg": float(np.percentile(orientation_values, 95)),
                "orientation_error_max_deg": float(np.max(orientation_values)),
                "frame_count": len(position_values),
            }
        )
    return enriched, summaries


def _quat_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _aggregate(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
