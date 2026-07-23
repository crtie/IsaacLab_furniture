"""Installation-path and hash validation for externally distributed Sharpawave assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def expected_asset_root() -> Path:
    import isaaclab_assets.robots.sharpawave as contract

    return Path(contract.__file__).resolve().parent / "sharpa-wave-description"


def load_asset_requirement(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("robot") != "sharpawave" or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("invalid Sharpawave asset requirement manifest")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_tree_identity(root: str | Path) -> dict[str, Any]:
    asset_root = Path(root).expanduser().resolve()
    if not asset_root.is_dir():
        return {"ok": False, "code": "MISSING_ASSET", "asset_root": str(asset_root), "reason": "asset directory does not exist"}
    records = []
    total = 0
    for path in sorted(item for item in asset_root.rglob("*") if item.is_file()):
        relative = path.relative_to(asset_root).as_posix()
        digest = _sha256(path)
        records.append((relative, digest))
        total += path.stat().st_size
    tree = hashlib.sha256()
    for relative, digest in records:
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
    return {
        "ok": True,
        "code": "OK",
        "asset_root": str(asset_root),
        "file_count": len(records),
        "total_size_bytes": total,
        "tree_sha256": tree.hexdigest(),
    }


def validate_asset_install(requirement_path: str | Path, *, asset_root: str | Path | None = None) -> dict[str, Any]:
    requirement = load_asset_requirement(requirement_path)
    root = expected_asset_root() if asset_root is None else Path(asset_root).expanduser().resolve()
    identity = asset_tree_identity(root)
    identity["expected_asset_root"] = str(expected_asset_root())
    identity["expected_tree_sha256"] = requirement["tree_sha256"]
    if not identity["ok"]:
        identity["install_hint"] = (
            "Place or symlink the authorized sharpa-wave-description directory at expected_asset_root, "
            "then run scripts/environments/install_sharpawave_assets.py --verify-only."
        )
        return identity
    missing = [name for name in requirement["required_files"] if not (root / name).is_file()]
    matches = (
        identity["file_count"] == int(requirement["file_count"])
        and identity["total_size_bytes"] == int(requirement["total_size_bytes"])
        and identity["tree_sha256"] == str(requirement["tree_sha256"])
        and not missing
    )
    identity.update({
        "ok": matches,
        "code": "OK" if matches else "ASSET_VERSION_MISMATCH",
        "missing_required_files": missing,
        "reason": "" if matches else "installed Sharpawave asset tree does not match the required version/hash",
    })
    return identity

