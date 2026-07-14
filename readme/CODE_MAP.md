# Code Map

## Generic Data Flow

```text
Object YAML
  -> ObjectGraspSpec
  -> runtime collision-mesh audit
  -> contact_sampler / grasp_energy
  -> WujiKinematicModel staged optimization
  -> GraspCandidate / CandidateCache
  -> NearGraspPhysicsEnv
  -> ObjectSpaceContactController
  -> StrictPhysicalEvaluator / GateResult
  -> trace, report, alignment, optional video
```

## Configuration

`configs/grasp_synthesis/objects/*.yaml` owns object differences:

- asset and audited scale
- allowed contact regions
- finger groups (`23`, `34`, `234`)
- pinch/wrap/contact modes
- section axis and local thickness
- approach directions
- mass, friction, force range, and lift direction
- candidate fallback order

The runtime controller does not branch on object name.

## Generic Modules

| Module | Responsibility |
|---|---|
| `object_spec.py` | Load and validate object YAML; compute content hash |
| `contact_sampler.py` | Mesh sampling, opposite raycast, three-point and fallback expansion |
| `grasp_energy.py` | Search metrics and friction-cone gravity-wrench feasibility |
| `wuji_ik.py` | 26D FK/Jacobian, staged reachability, collision and path checks |
| `candidate_cache.py` | Immutable hash-addressed candidate storage and eligibility |
| `contact_controller.py` | Per-finger DLS, force control, latch release and reacquisition |
| `physics_validator.py` | Contact attribution, Gate thresholds, and result semantics |
| `near_grasp_physics_env.py` | Wuji/Table/ground/one-target scene, sensors and observations |
| `forensic_trace.py` | Same-frame force, pose, contact and terminal trace contracts |

Source root:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/
  pipeline/unified_grasp/near_grasp/
  physical_delivery/
```

## Frame Modules

| Module | Responsibility |
|---|---|
| `contracts.py` | Frame states, results, contact evidence, write gate, video matching |
| `frame_cage_planner.py` | RC1 HOOK/WRAP/BRACKET geometry candidates |
| `grasp_executor.py` | RC1 contact/lift/hold proof |
| `frame_fork_support.py` | RC2 fixed three-finger fork pose |
| `frame_stage4_env.py` | Independent scene, sensors, snapshots and reset-only writes |
| `frame_stage4_controller.py` | Support, lift, preinsert, insert, release and retreat states |
| `frame_stage4_evaluator.py` | A/B/C/NONE milestone evaluation |

RC1 implements lift proof only. RC2 contains insertion/release states, but current execution stops at reset preload.

## Important Data Contracts

- `GraspCandidate`: closed pose, pregrasp/standoff maps, contact points/normals, support vertices, solver and cache hashes.
- `SearchMetrics`: candidate ordering only; never physical success.
- `GateResult`: measured contact/hold/lift fields.
- `approach_close_lift_success`: generic Gate C completion.
- `full_route_success`: reserved for transport, preinsert, insert, release; always false in the generic runner.
- `FrameStage4Result`: delivery level A/B/C/NONE and synchronized milestone fields.
- `video_matches_numeric_rollout`: true only when attempt IDs are identical.

## Historical Reference Modules

The top level of `pipeline/unified_grasp/` contains acquisition replay, hand morphology, closure prior, local Jacobian, dual-contact regulation, scripted baselines, old physical backend, and old RL interfaces. These modules preserve tested diagnostics and experiment provenance; they are not default policy code.

## Read Order

1. `scripts/environments/run_privileged_physics_grasp_v1.py`
2. `grasp_synthesis/object_spec.py`
3. `contact_sampler.py` and `grasp_energy.py`
4. `wuji_ik.py` and `candidate_cache.py`
5. `near_grasp_physics_env.py` and `contact_controller.py`
6. `physics_validator.py`

For Frame RC2, read the runner, `frame_fork_support.py`, environment, controller, then evaluator.
