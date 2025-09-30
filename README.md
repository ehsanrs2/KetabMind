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
5. **Retrieval:** Queries are embedded, searched against Qdrant, and reranked with lexical overlap scoring. Enable the cross-encoder reranker when GPU budget allows to improve precision on longer answers.
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

Phase 1 – Ingestion & Normalization
-----------------------------------

### Environment variables

- `OCR_FA`: Enable Persian OCR fallback with Tesseract for scanned PDFs.
- `TESSDATA_PREFIX`: Path to the Tesseract language data files (required when `OCR_FA=true`).
- `NORMALIZE_FA`: Toggle Farsi-specific normalization pipeline prior to chunking.

### Example ingestion request

```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
        "path": "/absolute/path/to/book.pdf",
        "collection": "books",
        "metadata": {
          "author": "A. Author",
          "year": 2020,
          "subject": "mathematics"
        }
      }'
```

### CLI shortcut

```bash
make ingest-pdf INPUT=book.pdf AUTHOR="A. Author" YEAR=2020 SUBJECT=math
```

Phase 2 – Embeddings & Vector Index
-----------------------------------

### Supported models

- `bge-m3`
- `multilingual-e5-base`

### Environment variables

- `EMBED_MODEL_NAME`
- `EMBED_QUANT`
- `BATCH_SIZE`

### Example embedding build

```bash
EMBED_MODEL_NAME=bge-m3 EMBED_QUANT=8bit BATCH_SIZE=16 make up
```

Embedding models

- Set `EMBED_MODEL` to select the embedder:
  - `mock` (fast, offline; for tests)
  - `small` (BGE small, 384-dim)
  - `base` (BGE base, 768-dim)

LLM backend

- Set `LLM_BACKEND` to select the language model backend (only `mock` for now).

Phase 3 – Retrieval & Reranking
--------------------------------

### Reranker model

- Default cross-encoder: `bge-reranker-v2-m3` (balanced for bilingual corpora).
- Override with `RERANKER_MODEL_NAME` to experiment with other models supported by your backend.

### Environment variables

- `RERANKER_ENABLED`: Toggle the reranker stage (`true`/`false`).
- `RERANKER_MODEL_NAME`: Select the deployed reranker checkpoint.
- `HYBRID_WEIGHTS`: Comma-delimited weights applied to cosine, lexical, and reranker scores (e.g. `cosine=0.3,lexical=0.2,reranker=0.5`).
- `QUERY_DEBUG`: When `true`, expose additional ranking diagnostics on query responses.

### Example configuration

```bash
RERANKER_ENABLED=true \
RERANKER_MODEL_NAME=bge-reranker-v2-m3 \
HYBRID_WEIGHTS="cosine=0.3,lexical=0.2,reranker=0.5" \
QUERY_DEBUG=true make run
```

Hybrid weighting
----------------

- The final score uses normalized weights: `final = cosine*w_cos + lexical*w_lex + reranker*w_rerank`.
- Set weights to zero to disable a component without changing code.
- Persist weights in `.env` for repeatability and share across environments.

Querying

- POST `/query` with `{ "q": "your question", "top_k": 5 }`.
- Response contains `answer` and `citations`. Include `?debug=true` to also receive chunk scores and reranker diagnostics when `QUERY_DEBUG=true`.

Troubleshooting
---------------

- **GPU out-of-memory:** Decrease `BATCH_SIZE`, disable the reranker (`RERANKER_ENABLED=false`), or switch to CPU inference for the reranker component.
- **Request timeouts:** Reduce `top_k`, lower reranker weight in `HYBRID_WEIGHTS`, or increase the service timeout settings (for example, `REQUEST_TIMEOUT` in `.env`).

Running tests
-------------

All test dependencies are declared in `pyproject.toml`. To run the suite locally:

```bash
poetry install
pytest tests/test_api_query_fallback.py tests/e2e/test_index_query_phase1.py tests/e2e/test_query_debug.py
```

When running outside Poetry, ensure FastAPI and its multipart dependencies are installed:

```bash
pip install fastapi python-multipart pydantic-settings structlog qdrant-client
```
