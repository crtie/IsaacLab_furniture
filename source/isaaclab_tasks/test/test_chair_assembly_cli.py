from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts/environments/run_chair_assembly.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_chair_assembly", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_system_validation_writes_honest_report(tmp_path: Path):
    report = tmp_path / "report.json"
    assert _module().main(["--backend", "system-validation", "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["targets_completed"] == 22
    assert payload["not_physical"] is True
    assert payload["bc_training_eligible"] is False


def test_cli_policy_without_manifest_is_explicit(tmp_path: Path):
    report = tmp_path / "report.json"
    assert _module().main(["--backend", "policy", "--report", str(report)]) == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["result_code"] == "POLICY_UNAVAILABLE"
    assert "fallback" in payload["reason"]

