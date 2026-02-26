.PHONY: venv lint test run-local run-adk

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e ".[dev]"

lint:
	. .venv/bin/activate && ruff check .

test:
	. .venv/bin/activate && pytest -q

run-local:
	. .venv/bin/activate && ncaa2026 run-local

run-adk:
	. .venv/bin/activate && ncaa2026 run-adk
