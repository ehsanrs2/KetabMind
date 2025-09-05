FROM python:3.11-slim
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN pip install --no-cache-dir poetry && poetry install --only main --no-root
COPY . .
CMD ["poetry", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]