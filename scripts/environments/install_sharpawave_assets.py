"""Verify or install an authorized Sharpawave asset tree at the package path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from isaaclab_tasks.chair_assembly.asset_install import expected_asset_root, validate_asset_install


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENT = REPO_ROOT / "configs/chair_assembly/sharpawave_asset_requirement.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    destination = expected_asset_root()
    if not args.verify_only:
        if args.source is None:
            parser.error("--source is required unless --verify-only is used")
        source = args.source.expanduser().resolve()
        source_report = validate_asset_install(REQUIREMENT, asset_root=source)
        if not source_report["ok"]:
            payload = {"status": "SOURCE_ASSET_INVALID", **source_report}
            if args.report:
                args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 2
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing asset path: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "symlink":
            destination.symlink_to(source, target_is_directory=True)
        else:
            shutil.copytree(source, destination)
    result = validate_asset_install(REQUIREMENT)
    payload = {"status": "SHARPAWAVE_ASSET_READY" if result["ok"] else "SHARPAWAVE_ASSET_MISSING", **result}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
