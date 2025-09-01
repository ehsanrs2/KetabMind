PY=poetry run

.PHONY: setup lint test run up down index-sample qa

setup:
	poetry install
	pre-commit install || true

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy .

test:
	$(PY) pytest -q

run:
	$(PY) uvicorn apps.api.main:app --reload

up:
	docker compose up -d

down:
	docker compose down

index-sample:
	$(PY) python -m scripts.index_sample

qa:
	$(PY) python -m core.eval.offline_eval --ds data/eval.jsonl || true

