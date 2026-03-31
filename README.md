# Furniture Assembly Environments

Robotic furniture assembly environments built on Isaac Lab, featuring force-guided insertion algorithms for autonomous part assembly using a Franka Panda robot.

## Environments

| Environment | Task ID | Description |
|---|---|---|
| Chair1 | `Isaac-Franka-Chair1-Direct-v0` | Plug insertion into chair frame (peg-in-hole with grid search algorithm) |
| Chair2 | `Isaac-Franka-Chair2-Direct-v0` | Rod insertion into chair frame (grid scan + Z-threshold detection) |
| Chair4 | `Isaac-Franka-Chair4-Direct-v0` | Board insertion into chair frame (grid scan + slope detection) |
| Vasskar1 | `Isaac-Franka-Vasskar1-Direct-v0` | Shelf frame assembly (grid scan + XY-stuck detection + hard press) |

## Quick Start

All commands should be run from the IsaacLab root directory:

```bash
cd /path/to/IsaacLab
```

### Chair1 - Plug Insertion

```bash
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Chair1-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

### Chair2 - Rod Insertion

```bash
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Chair2-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

### Chair4 - Board Insertion

```bash
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Chair4-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

### Vasskar1 - Shelf Assembly

```bash
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Vasskar1-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

## Insertion Algorithm

Each environment includes a `process_force_for_dev_experiment()` method that implements an autonomous insertion algorithm:

1. **Descend** - Gripper moves downward until Z-position stabilizes (contact detected)
2. **Grid Scan** - Snake-pattern grid search around contact point while pressing down
3. **Detect & Insert** - Hole detected via Z-threshold, slope analysis, or XY-stuck condition, then fast insertion

The algorithm overrides keyboard/RL control and runs automatically on environment start.

## File Structure

```
np/
  chair1_env.py          # Chair plug insertion environment
  chair2_env.py          # Chair rod insertion environment
  chair4_env.py          # Chair board insertion environment
  vasskar1_env.py        # Shelf frame assembly environment
  chair_tasks_cfg.py     # Chair task configurations
  vasskar_tasks_cfg.py   # Shelf task configurations
  np_env_cfg.py          # Environment base configurations
  factory_control.py     # Operational space controller
  asset/                 # USD/OBJ model files
    chair/               # Chair assembly assets
    vasskar/             # Shelf assembly assets
```
