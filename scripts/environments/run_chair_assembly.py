"""Single supported CLI for the packaged Sharpawave chair assembly workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab_tasks.chair_assembly import ChairAssemblyRunner, ResultCode
from isaaclab_tasks.chair_assembly.system_validation import SystemValidationBackend
from isaaclab_tasks.chair_assembly.sharpawave_runtime import load_runtime_calibration
from isaaclab_tasks.robot_adapters.sharpa_wave import MissingCalibrationError
from isaaclab_tasks.robot_adapters.policy import PolicyManifestError, load_policy_manifest


def _stages(value: str) -> tuple[int, ...] | None:
    if value.strip().lower() == "all":
        return None
    stages = tuple(int(item) for item in value.split(",") if item.strip())
    if not stages or any(stage < 1 or stage > 6 for stage in stages):
        raise argparse.ArgumentTypeError("--stages must be 'all' or comma-separated values in 1..6")
    return stages


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("policy", "system-validation"), required=True)
    parser.add_argument("--robot", choices=("sharpawave",), default="sharpawave")
    parser.add_argument("--variant", choices=("floating", "peg_fixedrot"), default="floating")
    parser.add_argument("--stages", type=_stages, default=None)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected_schema = f"sharpawave.robot_schema.v1.{args.variant}"
    if args.backend == "system-validation":
        report = ChairAssemblyRunner(SystemValidationBackend()).run(args.stages)
        report.robot = args.robot
        report.variant = args.variant
        _write_report(args.report, report.to_dict())
        return 0 if report.status == "complete" else 1

    if args.policy_manifest is None:
        payload = {
            "backend": "policy", "robot": args.robot, "variant": args.variant,
            "status": "blocked", "result_code": ResultCode.POLICY_UNAVAILABLE.value,
            "reason": "--policy-manifest is required; no assisted fallback is allowed",
            "not_physical": True, "bc_training_eligible": False,
        }
        _write_report(args.report, payload)
        return 2
    try:
        load_policy_manifest(args.policy_manifest, expected_action_schema=expected_schema)
    except PolicyManifestError as exc:
        _write_report(args.report, {
            "backend": "policy", "robot": args.robot, "variant": args.variant,
            "status": "blocked", "result_code": exc.code, "reason": str(exc),
            "details": exc.details, "not_physical": True, "bc_training_eligible": False,
        })
        return 2
    if args.calibration is None or not args.calibration.is_file():
        _write_report(args.report, {
            "backend": "policy", "robot": args.robot, "variant": args.variant,
            "status": "blocked", "result_code": ResultCode.MISSING_CALIBRATION.value,
            "reason": "a versioned Sharpawave calibration file is required for non-zero policy actions",
            "not_physical": True, "bc_training_eligible": False,
        })
        return 2
    try:
        load_runtime_calibration(args.calibration, variant=args.variant)
    except (OSError, ValueError, json.JSONDecodeError, MissingCalibrationError) as exc:
        _write_report(args.report, {
            "backend": "policy", "robot": args.robot, "variant": args.variant,
            "status": "blocked", "result_code": ResultCode.MISSING_CALIBRATION.value,
            "reason": str(exc), "not_physical": True, "bc_training_eligible": False,
        })
        return 2
    _write_report(args.report, {
        "backend": "policy", "robot": args.robot, "variant": args.variant,
        "status": "blocked", "result_code": ResultCode.CONTACT_UNAVAILABLE.value,
        "reason": "formal runtime contact/grasp/insert gates are not calibrated; policy was not executed",
        "not_physical": True, "bc_training_eligible": False,
    })
    return 2


if __name__ == "__main__":
    sys.exit(main())
