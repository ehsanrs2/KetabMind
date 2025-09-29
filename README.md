KetabMind — RAG system for books

Quickstart

- poetry install
- cp .env.example .env
- make test
- make run

Pipeline Overview
-----------------

1. **PDF ingestion:** `/index` accepts a local file path, normalizes it to a book ID, and converts PDF pages into structured text with metadata while skipping previously hashed files.
2. **Chunking:** The ingested pages are sliced into overlapping windows so long answers can reference surrounding context.
3. **Embedding:** Each chunk is embedded with the configured model to create vector representations.
4. **Qdrant storage:** Embeddings and payloads are written into the Qdrant collection (or skipped when already present).
5. **Retrieval:** Queries are embedded, searched against Qdrant, and reranked with lexical overlap scoring.
6. **Answer generation:** Retrieved passages are trimmed to the token budget, templated into a prompt, and sent to the LLM for a cited answer.

Example API calls
-----------------

```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"path": "/absolute/path/to/book.pdf", "collection": "books"}'
```

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"q": "Summarize chapter 3", "top_k": 5}'
```

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
