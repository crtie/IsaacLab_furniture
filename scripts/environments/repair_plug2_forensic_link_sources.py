"""Merge recorded filtered fingertip attribution into forensic link-source maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
NEAR_GRASP_PARENT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp"
if str(NEAR_GRASP_PARENT) not in sys.path:
    sys.path.insert(0, str(NEAR_GRASP_PARENT))

from near_grasp.grasp_synthesis.forensic_trace import merge_filtered_fingertip_sources  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="debug_runs/plug2_gate_a_forensic_v2")
    args = parser.parse_args()
    gate = (REPO_ROOT / args.output_dir / "Plug2/gate_a_forensic_v2").resolve()
    repaired = 0
    for reset_dir in sorted(gate.glob("candidates/*/reset_*")):
        trace_path = reset_dir / "trace.csv"
        rows = _read_csv(trace_path)
        for row in rows:
            metadata = json.loads(row["metadata"])
            sources = json.loads(row["link_contact_sources"])
            row["link_contact_sources"] = json.dumps(
                merge_filtered_fingertip_sources(sources, metadata["attributions"]), sort_keys=True
            )
            metadata["filtered_fingertip_sources_merged"] = True
            row["metadata"] = json.dumps(metadata, sort_keys=True)
        _write_csv(trace_path, rows)
        enriched_path = reset_dir / "trace_enriched.csv"
        if enriched_path.is_file():
            enriched = _read_csv(enriched_path)
            for row, raw in zip(enriched, rows):
                row["link_contact_sources"] = raw["link_contact_sources"]
                row["metadata"] = raw["metadata"]
            _write_csv(enriched_path, enriched)
        first = json.loads((reset_dir / "first_frame.json").read_text(encoding="utf-8"))
        first["link_contact_sources"] = json.loads(rows[0]["link_contact_sources"])
        first["metadata"] = json.loads(rows[0]["metadata"])
        (reset_dir / "first_frame.json").write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        contacts_path = reset_dir / "contact_events.jsonl"
        contact_rows = [json.loads(line) for line in contacts_path.read_text(encoding="utf-8").splitlines()]
        for contact_row in contact_rows:
            contact_row["link_contact_sources"] = {
                name: list(values)
                for name, values in merge_filtered_fingertip_sources(
                    contact_row["link_contact_sources"], contact_row["attributions"]
                ).items()
            }
            contact_row["filtered_fingertip_sources_merged"] = True
        contacts_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in contact_rows), encoding="utf-8"
        )
        repaired += 1
    if repaired != 80:
        raise RuntimeError(f"expected 80 reset traces, repaired {repaired}")
    print(json.dumps({"repaired_reset_traces": repaired, "physics_rerun": False}, indent=2))


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
