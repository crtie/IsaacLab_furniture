# Wuji Physical Grasp Manual

## Status

- No validated physical grasp, lift, preinsert, insertion, or release is currently available.
- Existing MP4 files are synchronized failure evidence, not success demonstrations.
- Generic Gate C ends at approach, close, hold, and lift. It is not an assembly route.
- Sticky, snap, fixed joint, proxy, teacher motion, and object-follow results are not physical success.

## Scope

The repository provides a floating Wuji hand with 6 wrist and 20 finger joints, five object specifications, a vectorized single-object environment, grasp synthesis, Pinocchio kinematics, collision/path checks, contact control, strict physical evaluation, trace/video alignment, and a Frame insertion state machine.

Audited reusable code: approximately 16,060 lines in the current stack and four core test files. Current validation result: 56 core tests and 24 historical-controller tests pass.

## Documents

| File | Contents |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | Environment setup, exact commands, outputs, and replay procedures |
| [`CODE_MAP.md`](CODE_MAP.md) | Entrypoints, module ownership, data flow, and success fields |
| [`EXPERIMENT_HISTORY.md`](EXPERIMENT_HISTORY.md) | Nine deduplicated experiment series and measured results |
| [`EVIDENCE.md`](EVIDENCE.md) | Curated videos, traces, artifact semantics, and limitations |
| [`DEBUG_REFERENCE.txt`](DEBUG_REFERENCE.txt) | Detailed appendix covering 74 historical debug directories |
| [`experiment_ledger.json`](experiment_ledger.json) | Machine-readable experiment ledger |

## Main Entrypoints

| Entrypoint | Purpose |
|---|---|
| `scripts/environments/replay_near_grasp_candidate.py` | Replay one frozen Screw1 failure candidate |
| `scripts/environments/run_privileged_physics_grasp_v1.py` | Five-object audit, synthesis, smoke, and Gate A/B/C |
| `scripts/environments/run_multi_object_privileged_grasp_v3.py` | Bounded five-object orchestration |
| `scripts/environments/probe_frame_physical_cage.py` | Frame RC1 geometry/cage proof |
| `scripts/environments/run_frame_stage4_delivery_rc2.py` | Fixed Frame support/lift/insert/release route |

Historical CEM, scripted, contact-adaptive, and old RL runners remain research references. They are not default execution paths.

## Quick Validation

```bash
cd /path/to/IsaacLab_furniture0
source /home/CNS2025331827/miniconda3/etc/profile.d/conda.sh
conda activate isaac
export PYTHONPATH="$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:${PYTHONPATH:-}"
export TERM=xterm-256color

make test
make reproduce-current
```

`make reproduce-current` only verifies the frozen 1,280-candidate artifact. It does not rerun CEM.

## Current Evidence

- Screw1 CEM: 1,280 candidates, 2 unsafe target contacts, 0 simultaneous contacts, 0 stable closes, 0 lifts.
- Plug2: 112 auditable physical resets across three experiment series, 0 lifts.
- Rod/Backrest: high Stage-A reachability, but strict M0 produced no physical reset.
- Frame RC1: 12 candidates, 0 geometry-valid, 0 physical reset.
- Frame RC2: baseline hard-aborted at about 119.54 N; one correction produced a 19.18 mm early Frame displacement; no lift or insertion stage was reached.

## Recommended Development Boundary

Start with Rod or Backrest and validate runtime contact geometry before strict M0. Use low-risk diagnostic physics for reachable candidates, then establish multi-finger contact and hold before considering residual learning. Do not restart the old Screw1 CEM or old 14D PPO unchanged.
