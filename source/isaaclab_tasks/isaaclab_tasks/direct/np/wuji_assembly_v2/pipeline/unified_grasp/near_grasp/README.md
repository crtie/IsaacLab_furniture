# Near-Grasp Mainline

This package is the active Screw1 physics-search implementation. It does not import v80-v95, scripted baselines, contact-adaptive controllers, sticky helpers, or the old unified RL environment.

## Data Flow

1. `configuration.py` loads the frozen YAML and produces a stable hash.
2. `grasp_program.py` defines the 16D program, templates, phases, and termination reasons.
3. `hand_prior_adapter.py` maps a six-dimensional latent to 20 Wuji joint targets.
4. `near_grasp_physics_env.py` executes programs in the dedicated Wuji + Table + Screw1 scene.
5. `evaluator.py` computes strict physical outcomes independently of reward.
6. `cem.py` ranks and updates mixed continuous/categorical candidates.
7. `replay.py` selects frozen failures and attributes diagnostic contact vectors.
8. `run_manifest.py` records source, asset, prior, command, and runtime hashes.

`observation.py` owns the named 169D schema. `residual_rl.py` is optional and remains gated behind a replayable strict lift; training is not currently allowed.

The normal entrypoint is `scripts/environments/run_near_grasp_cem.py`. The demonstration entrypoint is `scripts/environments/replay_near_grasp_candidate.py`.

Current classification: `NEAR_GRASP_SEARCH_INCOMPLETE`.
