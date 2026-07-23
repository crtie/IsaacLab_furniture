# Same-process BC Pick/Insert Integration

Copy `configs/chair_assembly/policy_manifest.example.json`, then provide two
independently replaceable policies named `pick` and `insert`. Each entry must
declare a `module:attribute` Python entry point, checkpoint, action and
observation schema, normalization metadata, and positive control frequency.

The entry point is constructed as:

```python
Policy(checkpoint="/resolved/checkpoint/path", metadata={...})
```

It must implement:

```python
reset(batch_size: int, context: PolicyContext) -> None
act(observation: SkillObservation) -> PolicyAction
close() -> None
```

Sharpawave floating actions use schema
`sharpawave.robot_schema.v1.floating` and width 28. Actions are batch-first,
finite arrays. Schema, batch, width, checkpoint, entrypoint, normalization, or
frequency errors are rejected as `POLICY_UNAVAILABLE`; they never select the
system-validation backend.

The formal state sequence is target initialization, Pick, physical grasp gate,
transport of the already-held part, Insert, physical insertion gate, release,
and target advance. There is one runtime reset before the run. No reset,
teleport, root-state write, Oracle, sticky, or snap is permitted between Pick
and Insert.

`isaaclab_tasks.chair_assembly.mock_policy.ZeroMockPolicy` is supplied only for
interface tests. Its output can never be labeled BC or physical success and is
never eligible for training data.
