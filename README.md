# Sharpawave Chair Assembly

Run every command from the repository root.

## 1. Activate the environment

```bash
eval "$(conda shell.bash hook)"
conda activate isaac
export PYTHONPATH="$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:${PYTHONPATH:-}"
export TERM=xterm-256color
which python
python -c "import isaaclab; print('isaaclab import ok', isaaclab.__file__)"
```

The Python executable must belong to the `isaac` environment and the import
command must print `isaaclab import ok`.

## 2. Install the project

```bash
python -m pip install --no-deps -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks
make doctor
```

Expected marker:

```text
chair assembly import ok: 22 targets
```

## 3. Install the external Sharpawave assets

The Git repository does not contain the Sharpawave binary asset tree. Before
installation, verify that the missing-asset error is explicit:

```bash
set +e
make asset-verify
rc=$?
set -e
test "$rc" -eq 2
```

The output must contain `MISSING_ASSET`, the expected installation path, and an
installation hint.

Set the source to an authorized copy of the exact asset version, then install
it as a symlink:

```bash
export SHARPAWAVE_ASSET_SOURCE=/path/to/authorized/sharpa-wave-description
test -d "$SHARPAWAVE_ASSET_SOURCE"
python scripts/environments/install_sharpawave_assets.py \
  --source "$SHARPAWAVE_ASSET_SOURCE" --mode symlink
make asset-verify
```

Expected marker: `SHARPAWAVE_ASSET_READY`. Validation checks 93 files, required
URDF/USD/config members, total bytes, and the recorded tree SHA256. Use
`--mode copy` instead of `--mode symlink` when a self-contained checkout is
required.

## 4. Minimal system validation

```bash
make system-validation
python -c "import json; p=json.load(open('artifacts/chair_assembly/system_validation.json')); assert p['result_code']=='SYSTEM_VALIDATION_COMPLETE' and p['targets_completed']==22 and p['not_physical'] and not p['bc_training_eligible']; print('system validation ok: 22 targets')"
```

This backend validates the six-stage `5/5/5/1/3/3` task sequence. It is
non-physical and cannot produce BC training data.

## 5. Sharpawave runtime validation

```bash
make sharpawave-runtime
```

Expected marker: `SHARPAWAVE_RUNTIME_ADAPTER_OK`. The report is written to
`artifacts/chair_assembly/sharpawave_runtime.json`.

## 6. Five-finger contact-channel validation

```bash
make sharpawave-contact-channels
```

Expected marker: `SHARPAWAVE_CONTACT_CHANNELS_OK`. Each elastomer channel must
respond to its own probe, other channels must remain zero, and all channels
must return to zero after separation.

## 7. Six-stage initialization validation

```bash
make sharpawave-stages
```

Expected marker: `SHARPAWAVE_STAGES_OK`. Stages 1 through 6 must report
`STAGE_INIT_OK`, no video, no object root-pose writes, and no Wuji fallback.

## 8. Pick/Insert BC policy interface

Copy and edit the example manifest. Pick and Insert policies may use different
Python entry points and checkpoints:

```bash
cp configs/chair_assembly/policy_manifest.example.json /tmp/sharpawave_policy_manifest.json
```

The policy class named by each `module:attribute` entry point must implement
`reset(batch_size, context)`, `act(observation)`, and `close()`. Actions use the
28D schema `sharpawave.robot_schema.v1.floating`.

The repository has no BC checkpoint. Verify the required failure behavior with
the unchanged example manifest:

```bash
set +e
python scripts/environments/run_chair_assembly.py \
  --backend policy \
  --policy-manifest configs/chair_assembly/policy_manifest.example.json \
  --calibration configs/chair_assembly/sharpawave_runtime_v1.json \
  --report artifacts/chair_assembly/policy_unavailable.json
rc=$?
set -e
test "$rc" -eq 2
python -c "import json; p=json.load(open('artifacts/chair_assembly/policy_unavailable.json')); assert p['result_code']=='POLICY_UNAVAILABLE'; print('POLICY_UNAVAILABLE')"
```

After replacing the manifest entry points and checkpoint paths with compatible
files, run the same command. Schema, width, normalization, entry-point, or
checkpoint errors stop the formal policy path; they never select
system-validation or another robot.

## 9. Tests

```bash
make test
```

Expected result: all formal chair-assembly and Sharpawave tests pass. The test
target also compiles the public package and every supported command-line tool.
