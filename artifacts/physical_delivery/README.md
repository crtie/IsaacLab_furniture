# Curated Physical Delivery Evidence

This directory is the small, Git-friendly evidence packet. Every MP4 here is failure evidence; none shows a successful physical grasp, lift, insertion, or release.

## Screw1 Near-Grasp

| File | Meaning |
|---|---|
| `screw1_near_grasp_v0_1/cem_candidate_results.jsonl` | All 1,280 frozen five-generation CEM candidate records |
| `retargeted_wuji_pca6.json` | Redistributable PCA6 Wuji fallback prior used by the run |
| `top_grasp_programs.json` | Ranked frozen programs; ranking is search evidence, not success |
| `cem_generation_metrics.csv` | Five generation summaries |
| `honesty_summary.json` | Zero physical grasp/lift and prohibited-assistance fields |
| `candidate_373_no_contact_failure.mp4` | Best-valid candidate replay with no target grasp contact |
| `candidate_33_target_contact_hard_abort.mp4` | Target contact followed by the hard-force abort |

`configs/near_grasp/screw1_cem_v1.yaml` points to these files so `make demo-current` works without the ignored original `debug_runs/` tree.

## Frame Stage 4 RC2

| File | Meaning |
|---|---|
| `frame_stage4_rc2/attempt_0_hard_force_failure.mp4` | Baseline preload; about 119.54 N on the second numeric frame |
| `attempt_0_trace.csv` | Synchronized baseline state/force trace |
| `attempt_0_video_alignment.json` | Video frame to physics-frame mapping for attempt 0 |
| `correction_1_frame_jump.mp4` | The `-5 mm` correction; hand-Frame force is zero while Frame moves |
| `correction_1_trace.csv` | Synchronized correction trace |
| `correction_1_video_alignment.json` | Video frame to physics-frame mapping for correction 1 |
| `final_summary.json` | Immutable copy of the historical RC2 summary |

The historical summary contains `video_evidence_attempt="attempt_0"`, `numeric_final_attempt="correction_1"`, and the old incorrect field `video_matches_numeric_rollout=true`. Because the attempt IDs differ, the correct interpretation is `false`. The source helper and tests now enforce exact attempt-ID equality; this evidence copy remains unchanged so its provenance is visible.

## Honesty Boundary

- No sticky, snap, proxy, fixed joint, teacher motion, or object-follow result is represented as physical success here.
- A Gate A reset-in-grasp lift would be oracle evidence, not autonomous grasp acquisition.
- Generic Gate C means approach/close/hold/lift only and cannot set `full_route_success=true`.
- Use `readme/experiment_ledger.json` for machine-readable history and `readme/README.md` as the documentation entry point.
