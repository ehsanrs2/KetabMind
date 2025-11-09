KetabMind — RAG system for books

Quickstart

- poetry install
- cp .env.example .env
- make test
- make run

Usage walkthrough
-----------------

1. **Upload a book.** Open `http://localhost:3000/upload`, choose a PDF, fill in the metadata fields, and click **Upload**. The UI reports ingestion progress, shows a status badge for each document, and lets you trigger indexing as soon as the file is stored.

2. **Chat or search.** Navigate to `http://localhost:3000/chat`, start a new session, and ask a question. Use the built-in search box to pre-filter your sessions or jump directly to a specific book by ID without leaving the page.

3. **Export the answer.** Click the **Export** action beside any assistant message (or call the `/export` API) to generate a PDF or Word document with citations and confidence metadata, ready for download or sharing.

Backups
-------

Set the following environment variables before running the backup utilities:

- `QDRANT_STORAGE_DIR`: Directory where Qdrant stores its collections (e.g. `./qdrant/storage`).
- `UPLOAD_DIR`: Directory where uploaded documents are persisted.
- `QDRANT_BACKUP_DIR`: Directory where compressed backups should be written.

Create a compressed snapshot of the storage and upload directories:

```bash
QDRANT_STORAGE_DIR=/var/lib/qdrant \
UPLOAD_DIR=/data/uploads \
QDRANT_BACKUP_DIR=/backups \
./scripts/backup_qdrant.sh
```

Restore from a specific backup archive (stops services via `docker compose stop` by default, then restarts and waits for readiness):

```bash
QDRANT_STORAGE_DIR=/var/lib/qdrant \
UPLOAD_DIR=/data/uploads \
QDRANT_BACKUP_DIR=/backups \
./scripts/restore_qdrant.sh qdrant_backup_20240101010101.tar.gz
```

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

Book Management API
-------------------

- `GET /books` – Returns a paginated list of indexed books for the current user. Each entry
  includes the database identifier, vector-store status, version/hash metadata, and timestamps.
- `GET /books/{book_id}` – Fetches detailed metadata for a single book, combining relational
  data with the most recent index manifest entry.
- `PATCH /books/{book_id}/rename` – Updates the stored title (and optional description) while
  propagating the new title to the Qdrant payloads and manifest, ensuring search results reflect
  the change immediately.
- `DELETE /books/{book_id}` – Removes the book from the relational database, deletes the
  associated Qdrant points, clears manifest/JSONL artifacts, and prunes the uploaded files.

These endpoints keep the primary database and vector store synchronized so that administrative
actions are immediately reflected in retrieval and audit workflows.

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

Phase 4 – Answer Synthesis & Advanced Self-RAG
----------------------------------------------

### Environment variables

- `ANSWER_STYLE`: Controls formatting (e.g. `bullets`, `paragraph`).
- `CITATION_REQUIRED`: When `true`, the answer agent must cite every factual statement.
- `COVERAGE_THRESHOLD`: Minimum percentage of retrieved chunks that must map to cited answer spans before self-RAG retries.
- `CONFIDENCE_RULE`: Policy string describing when the agent is allowed to answer vs. refuse (e.g. `require_citation`).
- `BUDGET_MAX_CHUNKS`: Cap on the total chunk budget per query for initial and retry passes.

### Example query with debug output

```bash
curl "http://localhost:8000/query?debug=true" \
  -H "Content-Type: application/json" \
  -d '{"q": "Explain the main argument in chapter 2", "top_k": 5}'
```

Sample response (`debug=true`):

```json
{
  "answer": "- Chapter 2 argues that equitable access to textbooks improves retention rates by 20%. [book123:45-46]\n- The authors cite field trials showing higher exam scores when digital copies are distributed early. [book123:47]",
  "citations": [
    {"book_id": "book123", "page_start": 45, "page_end": 46},
    {"book_id": "book123", "page_start": 47, "page_end": 47}
  ],
  "coverage": {
    "retrieved_chunks": 6,
    "cited_chunks": 5,
    "percentage": 83.3
  },
  "debug": {
    "pass": 2,
    "reason": "coverage_below_threshold",
    "budget": {"max_chunks": 12, "used": 10}
  }
}
```

Phase 5 – System & DevOps
-------------------------

### Local Docker Compose workflow

- Copy the environment template and adjust any overrides:

  ```bash
  cp .env.example .env
  ```

- Launch the full stack (API, UI, Qdrant, Ollama, Prometheus, Grafana):

  ```bash
  docker compose up --build
  ```

- Qdrant state is stored in the named Docker volume `ketabmind_qdrant_data`. Use
  `docker volume inspect ketabmind_qdrant_data` or `docker run --rm -v
  ketabmind_qdrant_data:/data alpine ls /data` if you need to examine or back up
  the on-disk collections.

- Orchestrate via Makefile helpers:

  ```bash
  make up          # start services in the background
  make logs        # tail the compose logs
  make down        # stop and remove containers
  ```

- Grafana is available at `http://localhost:3001`, Prometheus at `http://localhost:9090`, the FastAPI backend at `http://localhost:8000`, and the Next.js UI at `http://localhost:3000`.

### Core environment variables

| Variable | Purpose | Default / Example |
| --- | --- | --- |
| `QDRANT_MODE` | Switches the API between embedded and remote Qdrant clients. | `remote` (docker compose) |
| `QDRANT_URL` | Endpoint for the vector store. | `http://qdrant:6333` |
| `QDRANT_COLLECTION` | Name of the active collection. | `books` |
| `EMBED_MODEL` | Selects embedding backend (`mock`, `small`, `base`). | `small` |
| `OLLAMA_HOST` | Base URL for the LLM runtime. | `http://ollama:11434` |
| `NEXT_PUBLIC_API_URL` | UI -> API bridge, consumed by the Next.js frontend. | `http://api:8000` |
| `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` | Grafana bootstrap credentials. | `admin` / `admin` |

Additional Prometheus and Grafana settings can be tuned via the mounted config in `./prometheus` and `./grafana/provisioning`.

### Common Make targets

- `make setup` – install Python deps and configure pre-commit hooks.
- `make lint` – run Ruff (lint + format check) and mypy.
- `make test` – execute the full pytest suite.
- `make run` – start the FastAPI server with autoreload for local dev without containers.
- `make up` / `make down` / `make logs` / `make restart` – docker-compose lifecycle helpers.
- `make seed` – load a sample dataset into the running docker-compose stack.
- `make index-sample` – run the ingestion sample script against local files without containers.
- `make qa` – launch the offline evaluation harness on `data/eval.jsonl`.

### Observability snapshots

- **Grafana dashboard:** Monitor ingestion throughput, query latency, and GPU health on the prebuilt panels under *Home → KetabMind*. Use the time-range selector and per-panel queries to drill into spikes or regressions.
- **Prometheus metrics:** Visit `http://localhost:9090/metrics` to confirm service-level counters (requests, token usage, GPU utilization). The `/graph` UI can run ad-hoc queries like `rate(http_requests_total[5m])` when debugging traffic patterns.

### Troubleshooting

- **Container will not start:** Run `docker compose ps` to confirm health checks. Inspect failing service logs with `docker compose logs <service>` and ensure dependent services (e.g., Qdrant) are healthy.
- **Permission denied on bind mounts:** The host user may not own `./data/*` directories. Fix with `sudo chown -R $USER:$USER data` (or adjust to your UID/GID) before starting Compose.
- **Port conflicts:** Override host ports with `docker compose up -d --force-recreate -p ketabmind` after editing the published ports (e.g., `8000:8000` → `8080:8000`) or set environment variable overrides via `.env`.
- **GPU not detected (future phases):** Ensure the NVIDIA Container Toolkit is installed, start Compose with `docker compose --profile gpu up`, and set `CUDA_VISIBLE_DEVICES`/`NVIDIA_VISIBLE_DEVICES` in `.env` to target available GPUs.
- **GPU tuning fails:** Confirm `poetry install --with gpu` succeeded, then run `poetry run python scripts/bench_gpu.py --device cuda:0` to verify CUDA visibility and gather memory stats. 【F:scripts/bench_gpu.py†L1-L144】
- **PDF export missing glyphs:** Install the desired `.ttf` file (e.g., Vazirmatn) and register it with ReportLab before exporting. See the example above for registering fonts programmatically. 【F:exporting/exporter.py†L88-L128】
- **FTS index corrupted or empty:** Delete the SQLite database at `FTS_SQLITE_PATH` (default `./data/fts.db`), run `poetry run python -c "from core.fts import reset_backend; reset_backend()"`, then re-run your ingestion job (e.g., `make ingest-pdf INPUT=book.pdf ...`) to rebuild lexical search. 【F:core/fts/__init__.py†L68-L118】【F:core/index/indexer.py†L160-L199】

Phase 6 – UI & Auth
-------------------

### Running the UI

- Start the entire stack (API, UI, vector store, observability) with Docker Compose:

  ```bash
  docker compose up --build
  ```

- Once services are healthy, open the Next.js frontend at `http://localhost:3000` and the FastAPI backend at `http://localhost:8000`.

### Logging in

- Navigate to `http://localhost:3000/login` and use one of the built-in development accounts:
  - `alice@example.com` / `wonderland`
  - `bob@example.com` / `builder`
- Successful login redirects you to `/chat` and stores the authenticated session in an HttpOnly cookie.

### Uploading and chatting

- After signing in, go to `/upload` to submit a PDF. Provide the title, author, year, subject metadata, pick a file, and click **Upload**. When the status badge reports a successful upload, select **Index now** to launch ingestion.
- Use the **Chat** page to create a session, send questions, and monitor the streaming answer. The UI test suite walks through the entire flow end-to-end (login → upload → index → chat) if you need an example interaction.

### Citation links and deep-link viewer

- Answers list citations such as `[book123:12-14]`. Each citation is rendered as a link (e.g., `/viewer?book=book123#page=12`) so you can jump directly to the referenced page. Clicking the link opens the built-in document viewer with the page anchor highlighted, enforcing access controls per user.

### Security notes

- Login responses set the JWT in an `HttpOnly` cookie and echo the token via an `x-csrf-token` header. The frontend must include that header on subsequent mutations to satisfy CSRF checks.
- Book viewer links are generated as signed URLs that embed the owner, book, filename, and page along with an expiry timestamp. Requests with invalid or expired signatures are rejected.
- The API enforces per-client rate limits. Query requests (`/query`) specifically apply an additional per-second limit so abusive clients receive a `429 Too Many Requests` response with a `Retry-After` hint.

Phase 7 – Product Features & Polish
-----------------------------------

### Answer exports

- Export any assistant response through the REST API:

  ```bash
  curl -X POST "http://localhost:8000/export" \
    -H "Content-Type: application/json" \
    -d '{"message_id": "<uuid>", "format": "pdf"}' > answer.pdf
  ```

- Supported formats are `pdf`, `word`, and `docx`. The exporter normalises bullet lists, citations, and coverage metrics before piping them through ReportLab (`PDF`) or `python-docx` (`Word`). 【F:apps/api/main.py†L1045-L1080】【F:exporting/exporter.py†L54-L190】
- Need branded fonts for Persian output? Drop the `.ttf` file in `resources/fonts/`, then register it before exporting:

  ```bash
  poetry run python - <<'PY'
  from reportlab.pdfbase import pdfmetrics
  from reportlab.pdfbase.ttfonts import TTFont

  pdfmetrics.registerFont(TTFont("Vazirmatn", "resources/fonts/Vazirmatn-Regular.ttf"))
  print("Registered Vazirmatn for PDF exports.")
  PY
  ```

  Update `exporting/exporter.py` to call `text_obj.setFont("Vazirmatn", 12)` if you prefer the new font globally.

### Full-text search

- Enable SQLite-backed FTS by setting the following in `.env`:

  ```env
  FTS_BACKEND=sqlite
  FTS_SQLITE_PATH=./data/fts.db
  ```

- Each ingestion run indexes pages into the FTS database so `/search` can return instant book/page matches alongside the vector pipeline. 【F:core/index/indexer.py†L160-L199】【F:apps/api/main.py†L655-L692】
- Query the index directly:

  ```bash
  curl "http://localhost:8000/search?query=تصاعدی&limit=5"
  ```

- Adjust `FTS_PAGE_LIMIT` and `FTS_VECTOR_MULTIPLIER` to balance lexical vs. semantic recall. 【F:core/config.py†L131-L134】【F:core/retrieve/retriever.py†L215-L337】

### GPU tuning & benchmarking

- `utils.gpu_opt.GPUOptimizer` dynamically trims prompts and right-sizes batch sizes using live CUDA memory stats. Call it from custom pipelines or run the helper benchmark:

  ```bash
  poetry run python scripts/bench_gpu.py --device cuda:0 --sample-text "GPU warmup"
  ```

- Combine with quantisation toggles to stay within VRAM budgets: set `EMBED_QUANT=4bit|8bit` and `LLM_LOAD_IN_4BIT=true` to switch the embedder/LLM into reduced-precision modes. 【F:utils/gpu_opt.py†L1-L140】
- When memory is tight, lower `BATCH_SIZE` (embeddings) or ask the optimiser for `adjust_batch_size(...)` before launching the workload. 【F:utils/gpu_opt.py†L77-L118】

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
