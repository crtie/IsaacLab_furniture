SHELL := /bin/bash
REPO := $(CURDIR)
ISAAC_SETUP := source /home/CNS2025331827/miniconda3/etc/profile.d/conda.sh && conda activate isaac && export PYTHONPATH="$(REPO)/source/isaaclab:$(REPO)/source/isaaclab_tasks:$(REPO)/source/isaaclab_assets:$${PYTHONPATH:-}" && export TERM=xterm-256color

.PHONY: doctor test demo-current cem-smoke reproduce-current docs-check handoff

doctor:
	@bash -lc '$(ISAAC_SETUP) && python scripts/environments/near_grasp_doctor.py'

test:
	@bash -lc '$(ISAAC_SETUP) && python -m py_compile scripts/environments/run_near_grasp_cem.py scripts/environments/replay_near_grasp_candidate.py scripts/environments/audit_near_grasp_code_paths.py scripts/environments/near_grasp_doctor.py scripts/environments/build_near_grasp_handoff.py source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp/near_grasp/*.py'
	@bash -lc '$(ISAAC_SETUP) && pytest -q source/isaaclab_tasks/test/test_near_grasp_search.py'
	@bash -lc '$(ISAAC_SETUP) && python -c "import sys; sys.path.insert(0, \"source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2/pipeline/unified_grasp\"); import near_grasp; print(\"near_grasp import ok\", near_grasp.__file__)"'
	@bash -lc '$(ISAAC_SETUP) && TERM=xterm-256color ./isaaclab.sh -p scripts/environments/run_near_grasp_cem.py --help >/dev/null'

demo-current:
	@bash -lc '$(ISAAC_SETUP) && ./isaaclab.sh -p scripts/environments/replay_near_grasp_candidate.py --mode exact --candidate-id 373 --record-video --alignment-debug --headless'
	@bash -lc '$(ISAAC_SETUP) && ./isaaclab.sh -p scripts/environments/replay_near_grasp_candidate.py --mode exact --candidate-id 33 --record-video --alignment-debug --headless'

cem-smoke:
	@bash -lc '$(ISAAC_SETUP) && ./isaaclab.sh -p scripts/environments/run_near_grasp_cem.py --mode vector_smoke --num_envs 16 --smoke_steps 32 --max_episode_steps 64 --output_dir debug_runs/handoff_release/vector_smoke --headless'

reproduce-current:
	@bash -lc '$(ISAAC_SETUP) && python scripts/environments/build_near_grasp_handoff.py --verify-only'

docs-check:
	@bash -lc '$(ISAAC_SETUP) && python scripts/environments/build_near_grasp_handoff.py --docs-check'

handoff:
	@bash -lc '$(ISAAC_SETUP) && python scripts/environments/build_near_grasp_handoff.py --build'
