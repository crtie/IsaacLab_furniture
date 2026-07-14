"""Run the official Wuji retargeter on its trusted replay and emit v3 hand seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = REPO_ROOT / "third_party/external_grasp_baselines/wuji-retargeting"
PROJECT_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}" for joint in range(1, 5) for finger in range(1, 6)
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="debug_runs/multi_object_privileged_grasp_v3/external_priors/wuji_retargeting_seed_bank.json")
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--trust-official-pkl", action="store_true")
args = parser.parse_args()


def main() -> None:
    if not args.trust_official_pkl:
        raise RuntimeError("official pickle execution requires --trust-official-pkl")
    sys.path.insert(0, str(EXTERNAL))
    from wuji_retargeting import Retargeter

    import pinocchio

    if str(pinocchio.__version__) != "3.8.0":
        raise RuntimeError(f"wuji-retargeting requires pinocchio 3.8.0, got {pinocchio.__version__}")
    replay = EXTERNAL / "example/data/avp1.pkl"
    config = EXTERNAL / "example/config/adaptive_analytical_avp.yaml"
    with replay.open("rb") as stream:
        rows = pickle.load(stream)
    retargeter = Retargeter.from_yaml(str(config), hand_side="right")
    source_names = tuple(str(name) for name in retargeter.optimizer.robot.dof_joint_names)
    if set(source_names) != set(PROJECT_JOINT_NAMES):
        raise RuntimeError(f"retargeted/project joint sets differ: {source_names}")
    reorder = np.asarray([source_names.index(name) for name in PROJECT_JOINT_NAMES], dtype=np.int64)
    records = []
    for frame_index in range(0, len(rows), max(1, int(args.stride))):
        landmarks = np.asarray(rows[frame_index]["right_fingers"], dtype=np.float64)
        q_source = np.asarray(retargeter.retarget(landmarks, apply_filter=False), dtype=np.float64)
        q20 = q_source[reorder]
        if q20.shape != (20,) or not np.all(np.isfinite(q20)):
            continue
        tips = landmarks[[8, 12, 16]]
        records.append(
            {
                "frame_index": frame_index,
                "q20": q20,
                "distance_23": float(np.linalg.norm(tips[0] - tips[1])),
                "distance_34": float(np.linalg.norm(tips[1] - tips[2])),
                "wrap_span_234": float(np.max(np.linalg.norm(tips[:, None] - tips[None, :], axis=-1))),
            }
        )
    seeds = []
    selections = (
        ("23", "pinch", "distance_23", False),
        ("34", "pinch", "distance_34", False),
        ("234", "wrap", "wrap_span_234", False),
    )
    for group, family, field, reverse in selections:
        ordered = sorted(records, key=lambda row: row[field], reverse=reverse)
        chosen = []
        for row in ordered:
            q = np.asarray(row["q20"])
            if any(np.linalg.norm(q - np.asarray(existing["q20"])) < 0.05 for existing in chosen):
                continue
            chosen.append(row)
            if len(chosen) == 2:
                break
        for rank, row in enumerate(chosen):
            seeds.append(
                {
                    "seed_id": f"official_avp_{group}_{family}_{rank}",
                    "finger_group": group,
                    "family": family,
                    "frame_index": row["frame_index"],
                    "q20": np.asarray(row["q20"]).tolist(),
                    "selection_metric": field,
                    "selection_value": row[field],
                }
            )
    commit = subprocess.check_output(["git", "-C", str(EXTERNAL), "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema_version": 3,
        "source": "official_wuji_retargeting_avp1_right_hand",
        "source_commit": commit,
        "source_pkl": str(replay),
        "source_pkl_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "pinocchio_version": str(pinocchio.__version__),
        "joint_names": list(PROJECT_JOINT_NAMES),
        "sampled_frame_count": len(records),
        "seeds": seeds,
        "physical_success_evidence": False,
    }
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seed_count": len(seeds), "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
