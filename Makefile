PYTHON=python3.11

setup:
poetry install

lint:
poetry run pre-commit run --all-files

test:
poetry run pytest

run:
poetry run uvicorn apps.api.main:app --reload

up:
docker-compose up -d

down:
docker-compose down

index-sample:
poetry run ${PYTHON} scripts/index_sample.py

qa:
curl -X POST "http://localhost:8000/query?q=test"
