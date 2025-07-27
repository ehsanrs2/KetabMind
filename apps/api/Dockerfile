FROM python:3.11-slim
WORKDIR /app
COPY ../../poetry.lock ../../pyproject.toml ./
RUN pip install poetry && poetry install --no-dev
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
