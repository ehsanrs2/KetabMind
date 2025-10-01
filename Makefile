PY=poetry run

.PHONY: setup lint test run up down index-sample qa logs restart seed

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

logs:
	docker compose logs -f

restart:
	docker compose restart

seed:
	docker compose exec api poetry run python -m scripts.index_sample || true

index-sample:
	$(PY) python -m scripts.index_sample

qa:
	$(PY) python -m core.eval.offline_eval --ds data/eval.jsonl || true
