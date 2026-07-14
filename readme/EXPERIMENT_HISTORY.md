# Physical Grasp And Delivery Experiment History

## Current conclusion

The audit found **9 unique experiment series** plus one replay-only handoff. Delivery copies with identical evidence hashes are not counted twice. There is currently no synchronized evidence of a physical grasp, physical lift, preinsert, insert, or release.

The path documented in `AGENTS.md` as `debug_runs/full_oracle_sticky_assembly_pipeline` was absent during this audit. Its contract is visual/oracle/sticky and not physical success. This task did not delete, move, restore, or replace it.

## Experiment ledger

| ID | Experiment | Budget / resets | Deepest stage | Result |
|---|---|---:|---|---|
| EXP-01 | Screw1 v2 baseline family | 45 variant directories; aggregate resets unknown | pre-contact/single-contact diagnostics | no repeatable dual contact or lift |
| EXP-02 | Contact-adaptive Screw1 closure | 40 calibration probes + 7 A/B trials | pre-contact closure | non-target abort before dual contact |
| EXP-03 | Object-relative local Jacobian | 1 fresh reset, 16 signed probes | Jacobian calibration | restore-repeatability gate failed |
| EXP-04 | Screw1 near-grasp CEM | 5 x 256 = 1,280 resets | two unsafe single contacts | no simultaneous contact, close, or lift |
| EXP-05 | Privileged Physics Grasp v1 | 2048/128/64/16; 4 x 5 = 20 resets | Plug2 Gate A hold | every candidate 0/5 |
| EXP-06 | Plug2 forensic v2 | 2048/128/64/16; 16 x 5 = 80 resets | Plug2 Gate A hold | 75 hold failures, 5 hard aborts |
| EXP-07 | Five-object M0 v3 | screen all; full Rod/Backrest/Frame; 12 Plug2 resets | Plug2 Gate A hold | strict M0 0 for four objects; Plug2 failed |
| EXP-08 | Frame RC1 | 12 cage candidates, 0 physical resets | offline geometry gate | 0 geometry-valid candidates |
| EXP-09 | Frame RC2 | fixed pose, 2 resets | reset preload | 119.54 N baseline; corrected Frame jump |

Full machine-readable fields, evidence paths and confidence levels are in [`experiment_ledger.json`](experiment_ledger.json).

## Mandatory result reconciliation

### Privileged Physics Grasp v1

Plug2 used the requested `2048 sampled / 128 retained / 64 optimized / 16 physics shortlist` budget. Four optimized candidates entered formal Gate A and each ran five fresh resets. All were `0/5`; failures included hard force, multi-contact hold and unresolved instrumentation. Gate B/C and all other objects were correctly not started. This excludes “no candidate was ever physically tested,” but it does not exclude other Plug2 topologies or establish an autonomous grasp.

### Plug2 forensic v2

All 16 shortlisted candidates ran five sequential fresh resets, for 80 resets. Seventy-five ended in multi-contact hold failure and five in hard-force abort. Runtime tip residuals were about `20.459-40.984 mm`, while the cross-engine FK comparison was small. The evidence therefore points away from joint-order/FK disagreement as the primary cause and toward candidate/support-target geometry at runtime. It does not identify a successful alternative target.

### Five-object M0

- Rod: `31/32` screen and `63/64` full candidates were Stage-A reachable; strict M0 was 0.
- Backrest: `32/32` screen and `64/64` full were reachable; strict M0 was 0.
- Frame: `24/32` screen and `50/64` full were reachable; strict M0 was 0.
- Screw1: `32/32` screen were reachable; strict M0 was 0.
- Plug2: `30/32` screen were reachable; one strict candidate reached physics, then failed 12 resets across screen/retry/formal runs.

The large-object result is an M0 acceptance failure, not a physical grasp failure: those objects received zero physical resets. Stage-A reachability also shows that raw kinematic reach was not the only blocker.

### Frame RC1

RC1 generated 12 deterministic cage candidates and rejected all before physics. Scene-query control rays missed both the opening and bars. That confirms an instrumentation/verification failure, but does **not** confirm that cooked collision filled the opening or that a cage was mechanically impossible. Formal physical reset count was zero.

### Frame RC2

The baseline's second numeric frame reported `119.5378 N` on the Frame-filtered `right_finger2_link3` sensor and hard-aborted. This is confirmed impact/penetration evidence. The sole `-5 mm` correction removed measured hand-Frame force, but Frame moved `19.1769 mm` in the first three frames and failed `FRAME_JUMP`. The actor-pair callback yielded no events and the Frame-fixed filter pattern was invalid, so a specific Frame-FixedAsset collision is only **inferred**, not confirmed.

The delivered debug video is from `attempt_0`; the final numeric result is `correction_1`. The old summary's `video_matches_numeric_rollout=true` is semantically wrong. No correction_1 delivery video or complete matched delivery trace was preserved in `5/`; the source fix affects future runs only.

## Evidence-level rules

- `CONFIRMED`: directly present in synchronized summary/trace or unambiguous code path.
- `INFERRED`: best explanation consistent with evidence, but missing actor-pair or equivalent direct proof.
- `UNKNOWN`: unavailable, contradictory, or not measured. Unknown residuals are never named as a collider.

## What earlier choices taught us

| Judgment checked | Evidence | Impact | Avoidance |
|---|---|---|---|
| Small Plug2/Screw1 objects were prioritized before Rod/Backrest | CONFIRMED by EXP-01 through EXP-06 ordering | High geometric sensitivity consumed most physical resets before easier large-object behavior was tested | Start with morphology-compatible objects and preserve small objects as transfer tests |
| One antipodal family was used across dissimilar shapes | PARTIALLY CONFIRMED; later YAML adds regions/modes, but early samplers shared one dominant pattern | Object-specific enclosure and support topology entered late | Make contact mode and region semantics object-owned inputs |
| Strict M0 became the development entrance | CONFIRMED in EXP-07 | Rod, Backrest, Frame and Screw1 had zero physical resets despite high reachability | Keep strict M0 for acceptance while allowing bounded diagnostic physics with explicit non-success labels |
| Contact attribution and forensic tracing preceded useful contact | CONFIRMED in EXP-04 to EXP-06 | Instrumentation improved honesty but did not reduce centimeter-scale target error | First validate coarse geometry/contact acquisition, then increase forensic detail |
| Scene-query limits were treated as mechanical blockers | PARTIALLY CONFIRMED in RC1 early stop | Zero physics trials could be misread as cage infeasibility | Report query capability separately from mechanical feasibility |
| Offline FK/support targets created runtime mismatch | CONFIRMED for large runtime tip residual; exact source contribution is INFERRED | Offline-valid candidates missed runtime targets by 20-41 mm | Compare first-frame Isaac support geometry before repeated Gate trials |
| A colliding frame was used to derive correction | CONFIRMED in RC2 | The one correction removed one force signal but produced a different reset jump | Derive corrections only from collision-free, attributable measurements |
| Generic Gate C was called full route | CONFIRMED in code | Approach-close-hold-lift could be mistaken for assembly | Use `approach_close_lift_success`; reserve full route for transport+insert+release |
| RC2 final numeric and video attempts differed | CONFIRMED | The failure video was synchronized to baseline, not correction_1 | Compare attempt IDs explicitly and report mismatch |
| CoorDex/Wuji models were seed providers | CONFIRMED in v3/CEM code and reports | External prior use did not choose or validate the final grasp | Keep provider provenance, but assign success only through current physics evaluator |

## What remains unproved

No experiment proves that any of the five objects is mechanically ungraspable. No Gate C run completed transport, preinsert, insert or release. No sticky, snap, fixed joint, proxy, teacher motion or object-follow output may be relabeled as physical success. The next implementation should begin from the code ownership map, not from historical success labels.
