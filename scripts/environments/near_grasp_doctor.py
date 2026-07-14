"""Fast host, asset, prior, and artifact health check for near-grasp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/near_grasp/screw1_cem_v1.yaml")
    parser.add_argument("--output", default="debug_runs/handoff_release/doctor_report.json")
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report = _report(config, config_path)
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["classification"] == "HEALTH_OK" else 2)


def _report(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["python"] = {
        "executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "is_isaac_env": os.environ.get("CONDA_DEFAULT_ENV") == "isaac" or "/envs/isaac/" in sys.executable,
    }
    try:
        import isaaclab
        import torch

        checks["isaaclab"] = {"import_ok": True, "module": str(Path(isaaclab.__file__).resolve())}
        checks["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    except Exception as exc:
        checks["isaaclab"] = {"import_ok": False, "error": f"{type(exc).__name__}:{exc}"}
        checks["cuda"] = {"available": False}
    checks["nvidia_smi"] = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    assets = {}
    for name, relative in config["assets"].items():
        path = REPO_ROOT / str(relative)
        assets[name] = _file_check(path)
    checks["assets"] = assets
    prior_path = REPO_ROOT / str(config["prior"]["artifact"])
    coordex_path = REPO_ROOT / str(config["prior"]["coordex_checkpoint"])
    checks["prior"] = {
        "resolved": config["prior"]["resolved"],
        "artifact": _file_check(prior_path),
        "coordex_checkpoint": _file_check(coordex_path),
        "coordex_redistribution_allowed": False,
    }
    artifacts = {}
    for name, relative in config["artifacts"].items():
        artifacts[name] = _file_check(REPO_ROOT / str(relative))
    checks["artifacts"] = artifacts
    candidate_path = REPO_ROOT / str(config["artifacts"]["candidate_results"])
    candidate_count = sum(1 for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()) if candidate_path.is_file() else 0
    checks["candidate_count"] = {"actual": candidate_count, "expected": 1280, "matches": candidate_count == 1280}
    usage = shutil.disk_usage(REPO_ROOT)
    checks["disk"] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    checks["temp_write"] = _temp_write()
    checks["inotify"] = _inotify()
    checks["config"] = {"path": str(config_path), "sha256": _sha(config_path)}
    canonical = REPO_ROOT / "debug_runs/full_oracle_sticky_assembly_pipeline/report.md"
    checks["canonical_baseline"] = {
        "present": canonical.is_file(),
        "path": str(canonical),
        "action": "read_only_do_not_restore_or_overwrite",
    }
    blockers = []
    if not checks["python"]["is_isaac_env"]:
        blockers.append("python_not_in_isaac_conda_env")
    if not checks["isaaclab"].get("import_ok"):
        blockers.append("isaaclab_import_failed")
    if not checks["cuda"].get("available"):
        blockers.append("cuda_unavailable")
    if any(not row["exists"] for row in assets.values()):
        blockers.append("required_asset_missing")
    if not checks["prior"]["artifact"]["exists"]:
        blockers.append("resolved_prior_missing")
    if not all(row["exists"] for row in artifacts.values()):
        blockers.append("required_cem_artifact_missing")
    if not checks["candidate_count"]["matches"]:
        blockers.append("candidate_count_mismatch")
    if usage.free < 2 * 1024**3 or not checks["temp_write"]["ok"]:
        blockers.append("disk_or_temp_write_blocked")
    if checks["inotify"]["free_estimate"] < 1024:
        blockers.append("inotify_headroom_low")
    return {
        "schema_version": 1,
        "classification": "HEALTH_OK" if not blockers else "HOST_RESOURCE_PREFLIGHT_BLOCKED",
        "blockers": blockers,
        "checks": checks,
        "algorithm_experiment_started": False,
    }


def _file_check(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha(path) if path.is_file() else "",
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": command, "returncode": process.returncode, "output": process.stdout.strip()}


def _temp_write() -> dict[str, Any]:
    try:
        with tempfile.NamedTemporaryFile(prefix="near_grasp_doctor_", dir=REPO_ROOT / "debug_runs", delete=True) as stream:
            stream.write(b"ok")
            stream.flush()
        return {"ok": True, "error": ""}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def _inotify() -> dict[str, Any]:
    max_path = Path("/proc/sys/fs/inotify/max_user_watches")
    maximum = int(max_path.read_text(encoding="utf-8").strip()) if max_path.is_file() else 0
    total = 0
    consumers = []
    for proc in Path("/proc").glob("[0-9]*"):
        count = 0
        try:
            for info in (proc / "fdinfo").glob("*"):
                try:
                    count += sum(1 for line in info.read_text(encoding="utf-8", errors="ignore").splitlines() if line.startswith("inotify"))
                except OSError:
                    continue
        except OSError:
            continue
        if count:
            total += count
            try:
                command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:240]
            except OSError:
                command = ""
            consumers.append({"pid": int(proc.name), "watch_count": count, "command": command})
    consumers.sort(key=lambda row: row["watch_count"], reverse=True)
    return {
        "maximum": maximum,
        "used_estimate": total,
        "free_estimate": max(0, maximum - total),
        "top_consumers": consumers[:10],
    }


if __name__ == "__main__":
    main()
