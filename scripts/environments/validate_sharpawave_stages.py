"""Run no-video Sharpawave initialization checks for chair stages 1 through 6."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for stage in range(1, 7):
        report = args.output_dir / f"stage_{stage}.json"
        command = [
            str(REPO_ROOT / "isaaclab.sh"), "-p",
            "scripts/environments/record_sharpawave_stage_init.py",
            "--stage", str(stage), "--duration", "0.2", "--fps", "10",
            "--warmup-steps", "12", "--report", str(report), "--headless",
        ]
        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=float(args.timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            return_code = 124
        if not report.is_file():
            raise RuntimeError(f"stage {stage} produced no report; process return code {return_code}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        passed = bool(
            payload.get("passed")
            and payload.get("status") == "STAGE_INIT_OK"
            and not payload.get("video_recorded")
            and not payload.get("root_pose_writes_used")
            and payload.get("runtime_contract", {}).get("no_wuji_fallback")
        )
        rows.append({"stage": stage, "passed": passed, "return_code": return_code})
        if not passed:
            print(json.dumps(rows, indent=2))
            return 2
    print(json.dumps({"status": "SHARPAWAVE_STAGES_OK", "stages": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
