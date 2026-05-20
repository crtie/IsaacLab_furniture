# Assembly Teleoperation Environments

This README lists all available teleoperation tasks (Chair / Vasskar / Plane / Lego).

For each task family, the `--task` argument selects the **sub-environment** (the number after the task name, e.g. `Chair1`, `Chair2`, ...). When a single sub-environment contains multiple internal sub-tasks, you must **manually edit the `task_idx` field** in the corresponding `*_tasks_cfg.py` file before launching.

---

## Chair

Config file: [chair_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/np/chair_tasks_cfg.py)

Sub-envs `Chair1` … `Chair6`. Edit `task_idx` inside the matching `ChairAssemblyN` class to pick the internal sub-task.

```
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Chair1-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

For each specific task within a sub-env, manually edit the `task_idx` in [chair_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/np/chair_tasks_cfg.py). Example:

```python
@configclass
class ChairAssembly1(FactoryTask):
    task_idx = 2
```

| Sub-env (`--task`)        | Class            | `task_idx` range | Sub-task description                                                                                                                                                                                |
| ------------------------- | ---------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Isaac-Franka-Chair1-Direct-v0` | `ChairAssembly1` | 1–5              | 1: insert first plug into first hole · 2: insert second plug into second hole · 3: insert backrest into frame via plug · 4: insert plug1 into backrest · 5: insert plug2 into backrest             |
| `Isaac-Franka-Chair2-Direct-v0` | `ChairAssembly2` | 1–5              | 1: insert first plug into first hole · 2: insert second plug into second hole · 3: insert rod into frame via plug · 4: insert plug1 into rod · 5: insert plug2 into rod                            |
| `Isaac-Franka-Chair3-Direct-v0` | `ChairAssembly3` | 1–5              | 1: insert first plug into first hole · 2: insert second plug into second hole · 3: insert rod into frame via plug · 4: insert plug1 into rod · 5: insert plug2 into rod                            |
| `Isaac-Franka-Chair4-Direct-v0` | `ChairAssembly4` | 1                | 1: connect the other frame to the subassembly                                                                                                                                                       |
| `Isaac-Franka-Chair5-Direct-v0` | `ChairAssembly5` | 1–3              | 1: the first screw · 2: the second screw · 3: the third screw                                                                                                                                       |
| `Isaac-Franka-Chair6-Direct-v0` | `ChairAssembly6` | 1–4              | 1: the first screw · 2: the second screw · 3: the third screw · 4: the fourth screw                                                                                                                 |

---

## Vasskar

Config file: [vasskar_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/np/vasskar_tasks_cfg.py)

Sub-envs `Vasskar1` and `Vasskar2`. Edit `task_idx` inside the matching `VasskarAssemblyN` class.

```
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Vasskar2-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

| Sub-env (`--task`)            | Class              | `task_idx` range | Sub-task description                                                                                  |
| ----------------------------- | ------------------ | ---------------- | ----------------------------------------------------------------------------------------------------- |
| `Isaac-Franka-Vasskar1-Direct-v0` | `VasskarAssembly1` | 1–3              | 1: the first top frame · 2: the second top frame · 3: the side frame                                  |
| `Isaac-Franka-Vasskar2-Direct-v0` | `VasskarAssembly2` | 1–4              | 1: the first screw · 2: the second screw · 3: the third screw · 4: the fourth screw                  |

---

## Plane

Config file: [plane_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/np/plane_tasks_cfg.py)

Sub-envs `Plane1` … `Plane4`. Edit `task_idx` inside the matching `PlaneAssemblyN` class.

```
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Plane4-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

| Sub-env (`--task`)          | Class            | `task_idx` range |
| --------------------------- | ---------------- | ---------------- |
| `Isaac-Franka-Plane1-Direct-v0` | `PlaneAssembly1` | 1–4              |
| `Isaac-Franka-Plane2-Direct-v0` | `PlaneAssembly2` | 1–4              |
| `Isaac-Franka-Plane3-Direct-v0` | `PlaneAssembly3` | 1–3              |
| `Isaac-Franka-Plane4-Direct-v0` | `PlaneAssembly4` | 1                |

---

## Lego

Config file: [lego_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/np/lego_tasks_cfg.py)

Sub-envs `Lego1` … `Lego7`. Each sub-env contains a single sub-task (`task_idx = 1`), so no `task_idx` editing is required — just pick the right sub-env number.

```
python scripts/environments/teleoperation/teleop_se3_agent_custom.py \
  --task Isaac-Franka-Lego7-Direct-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --sensitivity 10
```

| Sub-env (`--task`)         | Class           | `task_idx` range |
| -------------------------- | --------------- | ---------------- |
| `Isaac-Franka-Lego1-Direct-v0` | `LegoAssembly1` | 1                |
| `Isaac-Franka-Lego2-Direct-v0` | `LegoAssembly2` | 1                |
| `Isaac-Franka-Lego3-Direct-v0` | `LegoAssembly3` | 1                |
| `Isaac-Franka-Lego4-Direct-v0` | `LegoAssembly4` | 1                |
| `Isaac-Franka-Lego5-Direct-v0` | `LegoAssembly5` | 1                |
| `Isaac-Franka-Lego6-Direct-v0` | `LegoAssembly6` | 1                |
| `Isaac-Franka-Lego7-Direct-v0` | `LegoAssembly7` | 1                |

---

## Workflow summary

1. Pick the sub-env number from the table above and put it in `--task Isaac-Franka-<Family><N>-Direct-v0`.
2. If the sub-env supports multiple `task_idx` values, open the matching `*_tasks_cfg.py` file and edit `task_idx` to the desired sub-task before launching.
3. Run the `teleop_se3_agent_custom.py` command with your chosen `--teleop_device` and `--sensitivity`.
