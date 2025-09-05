# Documentation

Set `EMBED_MODEL` in `.env` to choose the embedding backend:

- `small` (default) uses BGE small (384 dims)
- `base` uses BGE base (768 dims)
- `mock` uses a lightweight mock embedder for tests

The first run with a real model downloads roughly 400 MB of weights.

Set `LLM_BACKEND` in `.env` to choose the LLM backend (only `mock` is available).

Set `QDRANT_LOCATION` in `.env` for local vector storage:

```
QDRANT_LOCATION=./qdrant_local
# For unit tests set QDRANT_LOCATION=:memory:
```
