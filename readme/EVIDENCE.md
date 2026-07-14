# Evidence Guide

## Curated Files

All curated files are under `artifacts/physical_delivery/`. They are small enough for Git and preserve only the evidence needed to inspect current failure modes.

### Screw1

| File | Meaning |
|---|---|
| `screw1_near_grasp_v0_1/cem_candidate_results.jsonl` | 1,280 frozen CEM candidate records |
| `retargeted_wuji_pca6.json` | Redistributable PCA6 fallback prior |
| `top_grasp_programs.json` | Search ranking, not success evidence |
| `cem_generation_metrics.csv` | Five generation summaries |
| `candidate_373_no_contact_failure.mp4` | Matched no-contact failure replay |
| `candidate_33_target_contact_hard_abort.mp4` | Matched target-contact hard-force abort |

### Frame RC2

| File | Meaning |
|---|---|
| `frame_stage4_rc2/attempt_0_hard_force_failure.mp4` | Baseline preload hard-force failure |
| `attempt_0_trace.csv` / alignment | Matching numeric and video evidence |
| `correction_1_frame_jump.mp4` | Height correction followed by early Frame displacement |
| `correction_1_trace.csv` / alignment | Matching correction evidence |
| `final_summary.json` | Historical summary preserved without rewriting provenance |

## RC2 Semantic Correction

The historical summary records:

```text
video_evidence_attempt = attempt_0
numeric_final_attempt = correction_1
```

The old `video_matches_numeric_rollout=true` value is incorrect because the attempt IDs differ. Current code compares IDs directly and returns false for this case.

## Evidence Levels

- `CONFIRMED`: present in a synchronized trace, summary, or unambiguous code path.
- `INFERRED`: consistent with measurements but lacks direct actor-pair or equivalent proof.
- `UNKNOWN`: absent, contradictory, or not measured.

Frame baseline penetration/impact and the correction displacement are confirmed. The exact collider causing correction_1 is inferred because actor-pair evidence was unavailable.

## Video Inspection

```bash
xdg-open artifacts/physical_delivery/screw1_near_grasp_v0_1/candidate_373_no_contact_failure.mp4

conda run -n isaac python -c \
  "import cv2; p='artifacts/physical_delivery/frame_stage4_rc2/attempt_0_hard_force_failure.mp4'; v=cv2.VideoCapture(p); print(v.isOpened(), int(v.get(cv2.CAP_PROP_FRAME_COUNT)), int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))); v.release()"
```

Current curated videos are 960x540 H.264-compatible MP4 files with nonzero frame counts.

## Success Boundary

Do not infer physical success from visual appearance, reward, candidate rank, reset-in-grasp, sticky, snap, fixed joint, proxy, teacher motion, or object-follow. Physical milestones require the strict evaluator and matching trace/video evidence from one rollout.
