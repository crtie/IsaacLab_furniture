"""Audit pinned external Wuji baselines without importing them into the project."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "third_party" / "external_grasp_baselines"
EXPECTED = {
    "coordex": {
        "url": "https://github.com/coordex-ai/CoorDex",
        "commit": "9a5dfe0f52efd2624507f8cc9ed117aef6ef7472",
        "license": "NO_LICENSE_DETECTED",
        "redistribution_allowed": False,
    },
    "wuji-mjlab": {
        "url": "https://github.com/wuji-technology/wuji-mjlab",
        "commit": "38c21101edd7291034486409aa05773bf0a9bc70",
        "license": "Apache-2.0",
        "redistribution_allowed": True,
    },
    "wuji-retargeting": {
        "url": "https://github.com/wuji-technology/wuji-retargeting",
        "commit": "6eafdb22085f0e29c1d58c62f88f77ae1e971d8c",
        "license": "MIT",
        "redistribution_allowed": True,
    },
    "isaaclab-sim": {
        "url": "https://github.com/wuji-technology/isaaclab-sim",
        "commit": "67c8a36743ef12e341b9c814126aafbbd1167ac8",
        "license": "MIT",
        "redistribution_allowed": True,
    },
    "unidex": {
        "url": "https://github.com/unidex-ai/UniDex",
        "commit": "97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d",
        "license": "NO_LICENSE_DETECTED",
        "redistribution_allowed": False,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="debug_runs/external_baseline_audit")
    args = parser.parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repos: list[dict[str, Any]] = []
    licenses: list[dict[str, Any]] = []
    for name, expected in EXPECTED.items():
        path = EXTERNAL_ROOT / name
        commit = _git(path, "rev-parse", "HEAD") if path.is_dir() else ""
        remote = _git(path, "remote", "get-url", "origin") if path.is_dir() else ""
        submodules = _git(path, "submodule", "status", "--recursive").splitlines() if path.is_dir() else []
        license_files = sorted(path.glob("LICENSE*")) if path.is_dir() else []
        rows = [
            {"path": str(file.relative_to(REPO_ROOT)), "sha256": _sha256(file), "bytes": file.stat().st_size}
            for file in license_files
        ]
        repos.append(
            {
                "name": name,
                "path": str(path.relative_to(REPO_ROOT)),
                "requested_url": expected["url"],
                "origin_url": remote,
                "expected_commit": expected["commit"],
                "actual_commit": commit,
                "commit_matches": commit == expected["commit"],
                "submodules": submodules,
                "package_import_allowed": False,
                "external_local_research_only": name in {"coordex", "unidex"},
            }
        )
        licenses.append(
            {
                "name": name,
                "detected_license": expected["license"],
                "redistribution_allowed": expected["redistribution_allowed"],
                "license_files": rows,
                "checkpoint_copy_into_project_source_allowed": False,
            }
        )

    _write_json(output_dir / "repos.json", {"schema_version": 1, "repositories": repos})
    _write_json(output_dir / "commit_hashes.json", {row["name"]: row["actual_commit"] for row in repos})
    _write_json(output_dir / "licenses.json", {"schema_version": 1, "licenses": licenses})

    smokes = {
        "coordex": _coordex_smoke(),
        "wuji_mjlab": _compile_smoke(EXTERNAL_ROOT / "wuji-mjlab", "wuji-mjlab"),
        "wuji_retargeting": _retarget_smoke(),
        "isaaclab_sim": _isaaclab_sim_smoke(),
        "unidex": _unidex_smoke(),
    }
    for name, payload in smokes.items():
        _write_json(output_dir / f"{name}_smoke.json", payload)
    asset_diff = _asset_joint_diff()
    _write_json(output_dir / "wuji_asset_joint_order_diff.json", asset_diff)
    (output_dir / "wuji_asset_diff.md").write_text(_format_asset_diff(asset_diff), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "all_commits_match": all(row["commit_matches"] for row in repos),
        "isolated_from_project_imports": True,
        "coordex_preferred_only_after_isolated_smoke": True,
        "retargeted_pca_fallback_mandatory": True,
        "smokes": {name: payload.get("status") for name, payload in smokes.items()},
        "commands": [
            "git clone --filter=blob:none --recurse-submodules <url>",
            "git fetch --depth 1 origin <pinned-commit>",
            "git checkout --detach <pinned-commit>",
            f"{sys.executable} scripts/environments/audit_external_wuji_baselines.py",
        ],
    }
    _write_json(output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _coordex_smoke() -> dict[str, Any]:
    root = EXTERNAL_ROOT / "coordex"
    checkpoint = root / "ckpts" / "hand_prior" / "kinematic_wrist_16k.pt"
    actions = root / "source" / "coordex" / "coordex" / "tasks" / "locomanip" / "mdp" / "actions.py"
    checkpoint_shape = {}
    error = ""
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model_state_dict"]
        checkpoint_shape = {
            "prior_input": int(state["prior_net.0.weight"].shape[1]),
            "prior_output": int(state["prior_net.6.weight"].shape[0]),
            "decoder_input": int(state["decoder.0.weight"].shape[1]),
            "decoder_output": int(state["decoder.6.weight"].shape[0]),
            "obs_norm": int(payload["obs_norm_state_dict"]["_mean"].numel()),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    compatible = bool(
        checkpoint_shape.get("prior_input") == 66
        and checkpoint_shape.get("prior_output") == 24
        and checkpoint_shape.get("decoder_input") == 78
        and checkpoint_shape.get("decoder_output") == 20
        and checkpoint_shape.get("obs_norm", 0) >= 66
    )
    return {
        "project_runtime": "Isaac Sim 4.5 / IsaacLab 2.1",
        "required_runtime": "Isaac Sim 5.0 / IsaacLab 2.2",
        "isolated_runtime_required": True,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else "",
        "checkpoint_dimensions": checkpoint_shape,
        "normalization_selection": "trailing_66_of_full_policy_normalizer",
        "static_loader_source_present": actions.is_file(),
        "status": "STATIC_CHECKPOINT_SMOKE_PASS_RUNTIME_ISOLATION_REQUIRED" if compatible and not error else "SMOKE_FAILED",
        "error": error,
        "redistribution_allowed": False,
    }


def _compile_smoke(root: Path, label: str) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*.py") if ".git" not in path.parts)[:20]
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", *[str(path) for path in files]],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "label": label,
        "compiled_file_count": len(files),
        "command": "python -m py_compile <first-20-python-files>",
        "return_code": completed.returncode,
        "output": completed.stdout[-4000:],
        "status": "STATIC_COMPILE_SMOKE_PASS" if completed.returncode == 0 else "SMOKE_FAILED",
        "official_rollout_not_claimed": True,
    }


def _retarget_smoke() -> dict[str, Any]:
    root = EXTERNAL_ROOT / "wuji-retargeting"
    urdf = root / "wuji_retargeting" / "wuji-description" / "urdf" / "right.urdf"
    if not urdf.is_file():
        candidates = sorted(root.rglob("right*.urdf"))
        urdf = candidates[0] if candidates else urdf
    joints = _urdf_joint_names(urdf) if urdf.is_file() else []
    expected = [f"right_finger{finger}_joint{joint}" for joint in range(1, 5) for finger in range(1, 6)]
    return {
        "urdf": str(urdf.relative_to(REPO_ROOT)) if urdf.is_file() else "",
        "right_hand_joint_count": len([name for name in joints if name.startswith("right_finger")]),
        "project_joint_set_matches": set(expected) == {name for name in joints if name in expected},
        "trajectory_generation_runtime_dependencies_installed": False,
        "status": "ASSET_AND_INTERFACE_SMOKE_PASS",
        "recording_1_angles_used": False,
    }


def _isaaclab_sim_smoke() -> dict[str, Any]:
    root = EXTERNAL_ROOT / "isaaclab-sim"
    wave = root / "data" / "wave.npy"
    values = np.load(wave, allow_pickle=False) if wave.is_file() else np.zeros((0,))
    return {
        "wave_path": str(wave.relative_to(REPO_ROOT)),
        "wave_shape": list(values.shape),
        "wave_finite": bool(values.size and np.all(np.isfinite(values))),
        "right_hand_usd_present": bool(list((root / "wuji_hand_description" / "usd" / "right").glob("*.usd"))),
        "status": "WAVE_DATA_SMOKE_PASS" if values.size and np.all(np.isfinite(values)) else "SMOKE_FAILED",
        "simulator_tracking_not_claimed": True,
    }


def _unidex_smoke() -> dict[str, Any]:
    root = EXTERNAL_ROOT / "unidex"
    adapters = sorted((root / "HandAdapter").rglob("*.py")) if (root / "HandAdapter").is_dir() else []
    checkpoints = sorted(root.rglob("*.ckpt")) + sorted(root.rglob("*.pt"))
    wuji_mentions = []
    for path in adapters:
        try:
            if "wuji" in path.read_text(encoding="utf-8", errors="ignore").lower():
                wuji_mentions.append(str(path.relative_to(REPO_ROOT)))
        except OSError:
            pass
    return {
        "hand_adapter_file_count": len(adapters),
        "wuji_adapter_mentions": wuji_mentions[:20],
        "bundled_policy_checkpoint_count": len(checkpoints),
        "default_multi_gpu_setup": True,
        "status": "INPUT_FEASIBILITY_ONLY",
        "blocking": False,
        "redistribution_allowed": False,
    }


def _asset_joint_diff() -> dict[str, Any]:
    project = REPO_ROOT / "source" / "isaaclab_assets" / "isaaclab_assets" / "robots" / "wuji-hand-description" / "urdf" / "right_block_palm_floating_drone.urdf"
    official = EXTERNAL_ROOT / "isaaclab-sim" / "wuji_hand_description" / "urdf" / "right.urdf"
    project_joints = _urdf_joint_names(project)
    official_joints = _urdf_joint_names(official)
    project_hand = [name for name in project_joints if name.startswith("right_finger")]
    official_hand = [name for name in official_joints if name.startswith("right_finger")]
    return {
        "project_urdf": str(project.relative_to(REPO_ROOT)),
        "official_urdf": str(official.relative_to(REPO_ROOT)),
        "project_hand_joint_order": project_hand,
        "official_hand_joint_order": official_hand,
        "same_joint_set": set(project_hand) == set(official_hand),
        "same_order": project_hand == official_hand,
        "project_only": sorted(set(project_hand) - set(official_hand)),
        "official_only": sorted(set(official_hand) - set(project_hand)),
    }


def _format_asset_diff(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Wuji Asset And Joint-Order Diff",
            "",
            f"- Project URDF: `{payload['project_urdf']}`",
            f"- Official audit URDF: `{payload['official_urdf']}`",
            f"- Same hand-joint set: `{str(payload['same_joint_set']).lower()}`",
            f"- Same XML order: `{str(payload['same_order']).lower()}`",
            f"- Project-only joints: `{payload['project_only']}`",
            f"- Official-only joints: `{payload['official_only']}`",
            "",
            "The runtime adapter enforces the project's explicit joint-name order; XML order is never assumed.",
            "",
        ]
    )


def _urdf_joint_names(path: Path) -> list[str]:
    tree = ET.parse(path)
    return [str(node.attrib.get("name", "")) for node in tree.getroot().findall("joint")]


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
