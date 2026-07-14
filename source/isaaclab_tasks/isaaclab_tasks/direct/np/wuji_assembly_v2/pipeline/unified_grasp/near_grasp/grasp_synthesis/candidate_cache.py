"""Hash-addressed immutable candidate storage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


CACHE_SCHEMA_VERSION = 3
CANDIDATE_SCHEMA_VERSION = 3


def is_formal_gate_candidate(row: Mapping[str, Any], gate_key: str) -> bool:
    if gate_key not in {"gate_a", "gate_b", "gate_c"}:
        raise ValueError(f"unsupported formal gate {gate_key!r}")
    eligibility = row.get("gate_eligibility", {})
    return bool(
        int(row.get("schema_version", -1)) == CANDIDATE_SCHEMA_VERSION
        and row.get("optimization_success") is True
        and row.get("contact_residual_success") is True
        and row.get("physics_gate_a_eligible") is True
        and isinstance(eligibility, Mapping)
        and eligibility.get(gate_key) is True
    )


class CandidateCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def input_hash(*, spec: Mapping[str, Any], fingerprints: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            {"spec": spec, "fingerprints": fingerprints},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, part_name: str, input_hash: str) -> Path:
        return self.root / part_name / input_hash / "candidates.jsonl"

    def save(
        self,
        *,
        part_name: str,
        input_hash: str,
        candidates: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> Path:
        output = self.path_for(part_name, input_hash)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(row) for row in candidates]
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if output.exists():
            if output.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"immutable candidate cache mismatch: {output}")
            return output
        temporary = output.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "part_name": part_name,
            "input_hash": input_hash,
            "candidate_count": len(rows),
            "candidates_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": dict(metadata),
        }
        output.with_name("manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    def load(self, *, part_name: str, input_hash: str) -> list[dict[str, Any]]:
        path = self.path_for(part_name, input_hash)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_validated(
        self,
        path: str | Path,
        *,
        expected_part_name: str,
        expected_input_hash: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidate_path = Path(path).resolve()
        manifest_path = candidate_path.with_name("manifest.json")
        if not candidate_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"candidate cache or manifest is missing: {candidate_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValueError("legacy candidate cache schema is not valid for formal v3 gates")
        if int(manifest.get("candidate_schema_version", -1)) != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("candidate schema does not match formal v3 gates")
        if str(manifest.get("part_name", "")) != str(expected_part_name):
            raise ValueError("candidate cache part does not match the requested object")
        if expected_input_hash is not None and str(manifest.get("input_hash", "")) != str(expected_input_hash):
            raise ValueError("candidate cache input hash mismatch")
        content = candidate_path.read_text(encoding="utf-8")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != str(manifest.get("candidates_sha256", "")):
            raise ValueError("candidate cache content hash mismatch")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if len(rows) != int(manifest.get("candidate_count", -1)):
            raise ValueError("candidate cache row count mismatch")
        if any(int(row.get("schema_version", -1)) != CANDIDATE_SCHEMA_VERSION for row in rows):
            raise ValueError("legacy or incomplete candidate row rejected")
        return rows, manifest
