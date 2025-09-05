KetabMind — RAG system for books

Quickstart

- poetry install
- cp .env.example .env
- make test
- make run

Embedding models

- Set `EMBED_MODEL` to select the embedder:
  - `mock` (fast, offline; for tests)
  - `small` (BGE small, 384-dim)
  - `base` (BGE base, 768-dim)

LLM backend

- Set `LLM_BACKEND` to select the language model backend (only `mock` for now).

Querying

- POST `/query` with `{ "q": "your question", "top_k": 5 }`.
- Response contains `answer` and `contexts` (retrieved chunks).
