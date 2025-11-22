export PATH := $(HOME)/.local/bin:$(PATH)
export POETRY_VIRTUALENVS_IN_PROJECT ?= true

POETRY ?= $(shell command -v poetry 2>/dev/null || echo $(HOME)/.local/bin/poetry)
VENV_DIR ?= $(CURDIR)/.venv
VENV_BIN ?= $(VENV_DIR)/bin
PY=$(POETRY) run

.PHONY: setup lint test run up down index-sample qa logs restart seed install-poetry ensure-ollama ensure-venv

setup: ensure-venv
        pre-commit install || true

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy .

test:
        $(PY) pytest -q

run: ensure-ollama ensure-venv
        @echo "Starting API and UI (Ctrl+C to stop)..."
        @$(VENV_BIN)/uvicorn apps.api.main:app --reload & \
        API_PID=$$!; \
        cd apps/ui && npm ci && npm run dev & \
        UI_PID=$$!; \
        trap "kill $$API_PID $$UI_PID" INT TERM; \
        wait $$API_PID $$UI_PID

install-poetry:
        @if ! command -v poetry >/dev/null 2>&1; then \
                echo "Poetry not found. Installing..."; \
                curl -sSL https://install.python-poetry.org | python3 -; \
        else \
                echo "Poetry already installed."; \
        fi

ensure-venv: install-poetry
        @if [ ! -d "$(VENV_DIR)" ]; then \
                echo "Creating virtual environment in $(VENV_DIR) and installing dependencies..."; \
                $(POETRY) install; \
        else \
                echo "Using existing virtual environment in $(VENV_DIR)."; \
        fi

ensure-ollama:
        @if pgrep -x "ollama" >/dev/null 2>&1; then \
                echo "Ollama is already running."; \
        else \
                if command -v ollama >/dev/null 2>&1; then \
                        echo "Starting ollama serve..."; \
                        nohup ollama serve >/tmp/ollama.log 2>&1 & \
                        sleep 2; \
                else \
                        echo "Warning: ollama is not installed; skipping startup."; \
                fi; \
        fi

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
