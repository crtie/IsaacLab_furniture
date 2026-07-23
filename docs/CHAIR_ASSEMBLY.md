# Chair assembly API

Run the public CLI from the repository root:

```bash
python scripts/environments/run_chair_assembly.py --backend system-validation \
  --robot sharpawave --variant floating --stages all \
  --report artifacts/chair_assembly/system_validation.json
```

The catalog has six stages and 22 targets with counts `5/5/5/1/3/3`.

`policy` requires a policy manifest and calibration. It validates entry points,
checkpoint paths, schema, batch shape, width, finite values, normalization and
frequency. Any failure returns `POLICY_UNAVAILABLE`; it never selects another
backend and never uses sticky, Oracle, air-grasp, snap, teleport or root-pose
substitution.

`system-validation` checks task ordering and state progression only. Its result
is always non-physical and has `bc_training_eligible=false` and all physical
success fields false.

Python imports are provided by `isaaclab_tasks.chair_assembly`, including
`ChairAssemblyRunner`, `AssemblyBackend`, `SkillPolicy`, `PolicyContext`,
`SkillObservation`, `PolicyAction`, the task catalog and structured result
codes.

The Sharpawave `floating` runtime provides 28-DoF name mapping, reset, step,
action dispatch, joint/fingertip observations and five contact-force channels.
Runtime control and contact-channel wiring are validated; physical grasp,
lift, insertion and BC checkpoint success are not claimed.

Run the formal suite with:

```bash
make test
```
