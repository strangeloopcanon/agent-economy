.PHONY: setup bootstrap format check test all

VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
# Use venv python when it exists; otherwise fall back to python from PATH.
# This makes check/test/all work in sandboxes where .venv/ is not present.
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python)
UV ?= uv

setup:
	@test -x "$(VENV_PYTHON)" || $(UV) venv $(VENV)
	$(UV) pip install -e ".[dev]" --python $(VENV_PYTHON)

bootstrap: setup

format:
	$(PYTHON) -m ruff format .

check:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .
	$(PYTHON) -m compileall agent_economy tests

test:
	$(PYTHON) -m pytest -q

all: check test
