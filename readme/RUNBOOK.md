# Runbook

## Requirements

- Isaac Sim 4.5 / IsaacLab 2.1 project environment
- Conda environment: `isaac`
- NVIDIA GPU for physical replay
- Git LFS assets present

## Setup

```bash
cd /path/to/IsaacLab_furniture0
git lfs pull
source /home/CNS2025331827/miniconda3/etc/profile.d/conda.sh
conda activate isaac
export PYTHONPATH="$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:${PYTHONPATH:-}"
export TERM=xterm-256color
which python
python -c "import isaaclab; print('isaaclab import ok', isaaclab.__file__)"
```

## Pure Validation

```bash
conda run -n isaac pytest -q \
  source/isaaclab_tasks/test/test_near_grasp_search.py \
  source/isaaclab_tasks/test/test_privileged_physics_grasp.py \
  source/isaaclab_tasks/test/test_physical_delivery_rc1.py \
  source/isaaclab_tasks/test/test_frame_stage4_delivery_rc2.py

make reproduce-current
python -m json.tool readme/experiment_ledger.json >/dev/null
```

Expected current results: `56 passed`; frozen verification reports 1,280 candidates and zero physical lifts.

## CLI Inspection

```bash
python scripts/environments/run_privileged_physics_grasp_v1.py --help
python scripts/environments/probe_frame_physical_cage.py --help
TERM=xterm-256color ./isaaclab.sh -p scripts/environments/run_frame_stage4_delivery_rc2.py --help
TERM=xterm-256color ./isaaclab.sh -p scripts/environments/replay_near_grasp_candidate.py --help
```

## Frozen Screw1 Failure Replay

```bash
make demo-current
```

This replays candidate `373` (no target grasp contact) and candidate `33` (target-contact hard-force abort). It does not start CEM or PPO. New output is written under:

```text
debug_runs/handoff_release/replays/replay_candidate_<id>_<timestamp>/
```

Direct single-candidate command:

```bash
TERM=xterm-256color ./isaaclab.sh -p scripts/environments/replay_near_grasp_candidate.py \
  --mode exact --candidate-id 373 --record-video --alignment-debug \
  --output-dir debug_runs/manual_video_replays --headless --device cuda:0
```

## Frame RC2 Debug Replay

Use a new output directory. Do not pass `--build-delivery` during ordinary replay.

```bash
RUN="debug_runs/frame_stage4_rc2_manual_$(date +%Y%m%d_%H%M%S)"
TERM=xterm-256color ./isaaclab.sh -p scripts/environments/run_frame_stage4_delivery_rc2.py \
  --output-dir "$RUN" --no-directed-corrections \
  --headless --enable_cameras --device cuda:0
```

Expected files:

```text
<RUN>/videos/debug_first.mp4
<RUN>/attempt_0/full_route_trace.csv
<RUN>/attempt_0/contact_pairs.jsonl
<RUN>/attempt_0/video_alignment.json
<RUN>/final_summary.json
```

These outputs remain failure evidence unless all strict milestones pass in the same rollout.

## Generic Five-Object Phases

```bash
python scripts/environments/run_privileged_physics_grasp_v1.py --phase audit --part Rod
```

Physical phases require `isaaclab.sh -p`. Gate order is mandatory: Gate A, then Gate B, then Gate C. A failed or empty earlier gate blocks later gates. Gate C can only set `approach_close_lift_success`; `full_route_success` remains false.

## Output Rules

- Generated runs belong under `debug_runs/` and are not committed.
- Curated Git evidence is under `artifacts/physical_delivery/`.
- A video is evidence only when its attempt ID, trace, alignment, and numeric summary match.
- No actor-pair evidence means contact attribution remains unresolved.
