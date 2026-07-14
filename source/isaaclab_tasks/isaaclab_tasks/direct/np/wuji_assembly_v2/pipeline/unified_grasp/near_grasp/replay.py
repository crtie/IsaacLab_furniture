"""Pure candidate selection and replay evidence contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReplayRequest:
    candidate_json: str
    mode: str = "best-valid"
    candidate_id: int | None = None
    record_video: bool = False
    alignment_debug: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"best-valid", "target-contact-abort", "exact"}:
            raise ValueError(f"unsupported replay mode: {self.mode}")
        if self.mode == "exact" and self.candidate_id is None:
            raise ValueError("exact replay requires candidate_id")


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    candidate_id: int
    original_evaluation: Mapping[str, Any]
    replay_evaluation: Mapping[str, Any]
    classification: str
    matched_original_class: bool
    trace_path: str
    video_path: str
    contact_attribution_path: str
    physical_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactAttribution:
    step: int
    identified_contacts: tuple[str, ...]
    all_force_xyz: tuple[float, float, float]
    screw1_force_xyz: tuple[float, float, float]
    table_force_xyz: tuple[float, float, float]
    ground_force_xyz: tuple[float, float, float]
    residual_force_xyz: tuple[float, float, float]
    residual_force_n: float
    classification: str
    instrumentation_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("programs", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"candidate artifact is empty: {source}")
    for row in rows:
        if not isinstance(row, dict) or "candidate_id" not in row or "program" not in row or "evaluation" not in row:
            raise ValueError("candidate artifact has an incompatible row")
    return rows


def select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    candidate_id: int | None = None,
) -> dict[str, Any]:
    if candidate_id is not None:
        selected = next((row for row in rows if int(row["candidate_id"]) == int(candidate_id)), None)
        if selected is None:
            raise KeyError(f"candidate_id {candidate_id} not found")
        return dict(selected)
    if mode == "exact":
        raise ValueError("exact replay requires candidate_id")
    if mode == "best-valid":
        eligible = [row for row in rows if bool(row["evaluation"].get("valid_candidate"))]
        if not eligible:
            raise ValueError("candidate artifact has no valid candidates")
        return dict(max(eligible, key=_candidate_ranking_key))
    if mode == "target-contact-abort":
        eligible = [
            row
            for row in rows
            if bool(row["evaluation"].get("target_filtered_success_evidence"))
            and "hard_force_abort" in row["evaluation"].get("invalid_reasons", [])
        ]
        if not eligible:
            raise ValueError("candidate artifact has no target-contact hard abort")
        return dict(min(eligible, key=lambda row: float(row["evaluation"].get("peak_target_force_n", np.inf))))
    raise ValueError(f"unsupported replay mode: {mode}")


def attribute_contact_step(
    *,
    step: int,
    all_force_xyz: Sequence[float],
    screw1_force_xyz: Sequence[float],
    table_force_xyz: Sequence[float],
    ground_force_xyz: Sequence[float],
    filter_valid: Mapping[str, bool],
    threshold_n: float = 0.05,
) -> ContactAttribution:
    vectors = {
        "Screw1": _vector(screw1_force_xyz),
        "Table": _vector(table_force_xyz),
        "ground": _vector(ground_force_xyz),
    }
    all_force = _vector(all_force_xyz)
    valid_vectors = {
        name: vector for name, vector in vectors.items() if bool(filter_valid.get(name, False))
    }
    identified = tuple(
        name for name, vector in valid_vectors.items() if np.linalg.norm(vector) > threshold_n
    )
    residual = all_force - sum(valid_vectors.values(), np.zeros(3, dtype=np.float64))
    residual_n = float(np.linalg.norm(residual))
    all_force_n = float(np.linalg.norm(all_force))
    instrumentation_limit = bool(all_force_n > threshold_n and residual_n > threshold_n)
    if all_force_n <= threshold_n:
        classification = "NO_CONTACT"
    elif instrumentation_limit:
        classification = "UNRESOLVED_CONTACT_INSTRUMENTATION_LIMIT"
    elif identified:
        classification = "IDENTIFIED_SCENE_CONTACT"
    else:
        classification = "NO_CONTACT"
    return ContactAttribution(
        step=int(step),
        identified_contacts=identified,
        all_force_xyz=tuple(float(value) for value in all_force),
        screw1_force_xyz=tuple(float(value) for value in vectors["Screw1"]),
        table_force_xyz=tuple(float(value) for value in vectors["Table"]),
        ground_force_xyz=tuple(float(value) for value in vectors["ground"]),
        residual_force_xyz=tuple(float(value) for value in residual),
        residual_force_n=residual_n,
        classification=classification,
        instrumentation_limit=instrumentation_limit,
    )


def evaluation_class(evaluation: Mapping[str, Any]) -> str:
    if bool(evaluation.get("physical_lift_success")):
        return "PHYSICAL_LIFT"
    reasons = set(evaluation.get("invalid_reasons", []))
    if "hard_force_abort" in reasons:
        return "TARGET_CONTACT_HARD_ABORT" if evaluation.get("target_filtered_success_evidence") else "HARD_FORCE_ABORT"
    if "flyout" in reasons:
        return "FLYOUT"
    if "unresolved_contact_truth" in reasons:
        return "UNRESOLVED_CONTACT"
    if bool(evaluation.get("stable_close")):
        return "STABLE_CLOSE_NO_LIFT"
    return "NO_GRASP_CONTACT"


def _candidate_ranking_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    evaluation = row["evaluation"]
    hard_invalid = bool(evaluation.get("hard_invalid"))
    return (
        float(not hard_invalid),
        float(bool(evaluation.get("physical_lift_success"))),
        float(bool(evaluation.get("stable_close"))),
        float(evaluation.get("simultaneous_contact_duty", 0.0)),
        float(bool(evaluation.get("safe_force"))),
        -float(evaluation.get("lateral_displacement_m", np.inf)),
        -float(evaluation.get("relative_drift_m", np.inf)),
        -float(evaluation.get("jerk_metric", np.inf)),
    )


def _vector(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("contact force vectors must be finite xyz triples")
    return vector
