SHELL := /bin/bash
.DEFAULT_GOAL := help

REPO := $(CURDIR)
PYTHONPATH_ENV := PYTHONPATH="$(REPO)/source/isaaclab:$(REPO)/source/isaaclab_tasks:$(REPO)/source/isaaclab_assets:$${PYTHONPATH:-}"
CHAIR_RUNNER := scripts/environments/run_chair_assembly.py
CALIBRATION := configs/chair_assembly/sharpawave_runtime_v1.json
ARTIFACTS := artifacts/chair_assembly
FORMAL_TESTS := \
	source/isaaclab_tasks/test/test_chair_assembly_public_api.py \
	source/isaaclab_tasks/test/test_chair_assembly_cli.py \
	source/isaaclab_tasks/test/test_sharpawave_robot_adapter.py \
	source/isaaclab_tasks/test/test_sharpawave_robot_adapter_contract.py \
	source/isaaclab_tasks/test/test_sharpawave_registration.py \
	source/isaaclab_tasks/test/test_sharpawave_asset_static.py
FORMAL_PYTHON := \
	$(shell find source/isaaclab_tasks/isaaclab_tasks/chair_assembly -type f -name '*.py' | sort) \
	$(shell find source/isaaclab_tasks/isaaclab_tasks/robot_adapters -type f -name '*.py' | sort) \
	source/isaaclab_assets/isaaclab_assets/robots/sharpawave.py \
	source/isaaclab_assets/isaaclab_assets/robots/sharpawave_isaac.py \
	scripts/environments/run_chair_assembly.py \
	scripts/environments/install_sharpawave_assets.py \
	scripts/environments/audit_sharpawave_runtime_adapter.py \
	scripts/environments/validate_sharpawave_contact_channels.py \
	scripts/environments/record_sharpawave_stage_init.py \
	scripts/environments/validate_sharpawave_stages.py

.PHONY: help doctor test asset-verify system-validation policy-unavailable sharpawave-runtime sharpawave-contact-channels sharpawave-stages

help:
	@$(PYTHONPATH_ENV) python $(CHAIR_RUNNER) --help

doctor:
	@$(PYTHONPATH_ENV) python -c "import isaaclab, isaaclab_assets, isaaclab_tasks; from isaaclab_tasks.chair_assembly import get_task_catalog; assert len(get_task_catalog()) == 22; print('chair assembly import ok: 22 targets')"

test:
	@$(PYTHONPATH_ENV) python -m py_compile $(FORMAL_PYTHON)
	@$(PYTHONPATH_ENV) python -m pytest -q $(FORMAL_TESTS)

asset-verify:
	@$(PYTHONPATH_ENV) python scripts/environments/install_sharpawave_assets.py --verify-only

system-validation:
	@mkdir -p $(ARTIFACTS)
	@$(PYTHONPATH_ENV) python $(CHAIR_RUNNER) --backend system-validation --report $(ARTIFACTS)/system_validation.json

policy-unavailable:
	@mkdir -p $(ARTIFACTS)
	@$(PYTHONPATH_ENV) python $(CHAIR_RUNNER) --backend policy --report $(ARTIFACTS)/policy_unavailable.json; rc=$$?; \
		test $$rc -eq 2; \
		$(PYTHONPATH_ENV) python -c "import json; p=json.load(open('$(ARTIFACTS)/policy_unavailable.json')); assert p['result_code']=='POLICY_UNAVAILABLE'; print('POLICY_UNAVAILABLE')"

sharpawave-runtime:
	@mkdir -p $(ARTIFACTS)
	@$(PYTHONPATH_ENV) TERM=xterm-256color ./isaaclab.sh -p scripts/environments/audit_sharpawave_runtime_adapter.py \
		--calibration $(CALIBRATION) --report $(ARTIFACTS)/sharpawave_runtime.json --headless

sharpawave-contact-channels:
	@mkdir -p $(ARTIFACTS)
	@$(PYTHONPATH_ENV) TERM=xterm-256color ./isaaclab.sh -p scripts/environments/validate_sharpawave_contact_channels.py \
		--calibration $(CALIBRATION) --contact-steps 24 --recovery-steps 36 \
		--report $(ARTIFACTS)/sharpawave_contact_channels.json --headless

sharpawave-stages:
	@mkdir -p $(ARTIFACTS)
	@$(PYTHONPATH_ENV) python scripts/environments/validate_sharpawave_stages.py --output-dir $(ARTIFACTS)/stages
	@echo SHARPAWAVE_STAGES_OK
