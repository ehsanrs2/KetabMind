# Documentation

Set `EMBED_MODEL` in `.env` to choose the embedding backend:

- `small` (default) uses BGE small (384 dims)
- `base` uses BGE base (768 dims)
- `mock` uses a lightweight mock embedder for tests

The first run with a real model downloads roughly 400 MB of weights.

Set the LLM environment variables in `.env` to control answer generation:

- `LLM_BACKEND`: `mock`, `ollama`, or `transformers`.
- `LLM_MODEL`: backend-specific model identifier.
- `LLM_MAX_INPUT_TOKENS`: prompt token budget (default `4096`).
- `LLM_MAX_NEW_TOKENS`: maximum tokens to sample (default `256`).
- `LLM_TEMPERATURE`: sampling temperature (default `0.2`).
- `LLM_TOP_P`: nucleus sampling cutoff (default `0.95`).

Example configuration for a local Ollama server:

```
OLLAMA_HOST=http://localhost:11434
LLM_BACKEND=ollama
LLM_MODEL=mistral:7b-instruct-q4_K_M
```

Note: for GPU backend see next steps.

## Logging

Structured logs use `structlog` and are emitted to stdout and, by default, to `logs/ketabmind.log` with rotation. Override via environment:

- `LOG_FILE`: absolute or relative path for the log file (`stdout` to disable file logging).
- `LOG_DIR`: directory for the default `ketabmind.log` file (ignored when `LOG_FILE` is set).
- `LOG_FILE_MAX_BYTES`: rotate after this many bytes (default `10000000`).
- `LOG_FILE_BACKUP_COUNT`: number of rotated files to keep (default `5`).
- `LOG_REDACT_FIELDS`: comma-separated list of extra keys to redact in logs.

## GPU setup (tested on RTX 3060)

1. Install the optional GPU dependencies:
   - `poetry install --with gpu`
   - or with pip: `pip install .[gpu]`
2. Verify CUDA visibility:
   - `python -c "import torch; print(torch.cuda.is_available())"`
3. Recommended starter models:
   - `microsoft/Phi-3-mini-4k-instruct` (fast on 12 GB VRAM)
   - `mistralai/Mistral-7B-Instruct` (use 4-bit loading on 12 GB VRAM)

Set `QDRANT_LOCATION` in `.env` for local vector storage:

```
QDRANT_LOCATION=./qdrant_local
# For unit tests set QDRANT_LOCATION=:memory:
```
