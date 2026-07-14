"""Run provenance and hash manifests without Isaac runtime dependencies."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    timestamp_utc: str
    git_commit: str
    git_dirty: bool
    command: tuple[str, ...]
    config_path: str
    config_sha256: str
    source_hashes: Mapping[str, str]
    asset_hashes: Mapping[str, str]
    prior_hashes: Mapping[str, str]
    runtime: Mapping[str, Any]
    honesty: Mapping[str, Any]

    @classmethod
    def capture(
        cls,
        *,
        repo_root: str | Path,
        run_id: str,
        command: Sequence[str],
        config_path: str | Path,
        config_sha256: str,
        source_paths: Iterable[str | Path],
        asset_paths: Iterable[str | Path],
        prior_paths: Iterable[str | Path],
        runtime: Mapping[str, Any] | None = None,
    ) -> "RunManifest":
        root = Path(repo_root).resolve()
        commit = _git(root, "rev-parse", "HEAD")
        dirty = bool(_git(root, "status", "--porcelain"))
        return cls(
            run_id=str(run_id),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=commit,
            git_dirty=dirty,
            command=tuple(str(value) for value in command),
            config_path=_relative(root, Path(config_path)),
            config_sha256=str(config_sha256),
            source_hashes=_hash_paths(root, source_paths),
            asset_hashes=_hash_paths(root, asset_paths),
            prior_hashes=_hash_paths(root, prior_paths),
            runtime={
                "python_executable": sys.executable,
                "python_version": sys.version,
                "platform": platform.platform(),
                **dict(runtime or {}),
            },
            honesty={
                "sticky_used": False,
                "snap_used": False,
                "teacher_motion_used": False,
                "root_pose_writes_used": True,
                "root_pose_writes_reset_only": True,
                "post_reset_object_writes": 0,
                "post_reset_wrist_state_writes": 0,
                "physical_grasp_success": False,
                "physical_lift_success": False,
                "physical_insert_success": False,
                "oracle_visual_only": False,
                "not_physical": True,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output


def _hash_paths(root: Path, paths: Iterable[str | Path]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for value in paths:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            rows[_relative(root, path)] = sha256_file(path)
    return dict(sorted(rows.items()))


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip()
