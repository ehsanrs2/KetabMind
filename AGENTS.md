# AGENTS.md — KetabMind

## Project Goal
KetabMind is a RAG-based system that indexes books (PDF/EPUB/text), retrieves relevant chunks, and answers questions with citations. The system must be testable, modular, and production-ready.

## Global Standards
- **Language & Runtime:** Python 3.11
- **Package Manager:** poetry (or uv)
- **Code Quality:** ruff + black, mypy (strict), pytest
- **Logging:** structlog (JSON-friendly)
- **Config:** pydantic-settings
- **CLI:** Typer
- **API:** FastAPI (REST + streaming)
- **Vector DB:** Qdrant (docker-compose service)
- **Embeddings:** pluggable (local first, mock allowed in tests)
- **Comments:** English only, concise and actionable
- **Outputs to user:** Prefer code diffs or full file dumps per file; minimal prose

## Repo Layout (expected)
- apps/api/ (FastAPI app)
- apps/ui/ (Next.js placeholder)
- core/ingest/ (PDF/EPUB → text)
- core/chunk/ (chunking)
- core/embed/ (embedding adapters)
- core/vector/ (Qdrant client)
- core/retrieve/ (retriever)
- core/answer/ (LLM orchestration with citations)
- core/self_rag/ (answer verification)
- core/eval/ (offline evaluation)
- scripts/ (bootstrap, indexing, utilities)
- configs/ (yaml / env-backed)
- tests/ (pytest)
- docs/ (architecture, requirements)

## Common Make Targets
- `make setup` → install tooling & pre-commit
- `make up` / `make down` → docker compose up/down (Qdrant, API, UI)
- `make lint`, `make test`, `make run`, `make index-sample`, `make qa`

## Agent Roles & Prompts

### 1) Architect Agent
**Responsibility:** Project skeleton, interfaces, contracts, CI, pre-commit.
**Acceptance:**
- pyproject.toml, .ruff.toml, .pre-commit-config.yaml, mypy.ini (if any)
- docker-compose.yml (qdrant + api + ui)
- Makefile with targets above
**Prompt Style:**
- Generate files with minimal placeholders and TODOs.
- Include one end-to-end PoC path (`scripts/index_sample.py`).

### 2) Ingestion Agent
**Responsibility:** PDF/EPUB → normalized UTF-8 text with page metadata.
**Acceptance:**
- `core/ingest/pdf_to_text.py` with Typer CLI:
  `python -m core.ingest.pdf_to_text --in IN.pdf --out OUT.jsonl`
- Unit tests: `tests/test_ingest_pdf.py`
**Constraints:**
- Heuristic header/footer removal (configurable)
- Return page_num, section if TOC available

### 3) Chunking Agent
**Responsibility:** Sliding-window chunking with overlap.
**Acceptance:**
- `core/chunk/chunker.py` + tests
- Output schema: `{text, book_id, page_start, page_end, chunk_id}`

### 4) Embedding & Vector Agent
**Responsibility:** Embedding adapter + Qdrant client.
**Acceptance:**
- `core/embed/adapter.py` (interface + local default/mocks)
- `core/vector/qdrant_client.py` (create_collection, upsert, query with filters)
- Tests using dockerized Qdrant (ephemeral collection)

### 5) Retrieval & Answer Orchestrator Agent
**Responsibility:** k-NN retrieval (+ simple rerank), prompt templating with citations.
**Acceptance:**
- `core/retrieve/retriever.py`, `core/answer/answerer.py`
- JSON response: `{answer, citations:[{book, chapter, page, snippet, score}], debug}`
- Tests with mocked LLM

### 6) Self-RAG Agent
**Responsibility:** Validate answers against contexts; detect hallucinations; requery if needed.
**Acceptance:**
- `core/self_rag/validator.py` + tests for detection heuristics

### 7) Eval Agent
**Responsibility:** Offline evaluation.
**Acceptance:**
- `core/eval/offline_eval.py` with EM/F1 + citation coverage
- CLI: `python -m core.eval.offline_eval --ds data/eval.jsonl`

### 8) API & UI Agent
**Responsibility:** Minimal FastAPI with `/index` & `/query`; Next.js placeholder.
**Acceptance:**
- `apps/api/main.py` (streaming responses + health check)
- `apps/ui/` simple chat page to call `/query`

## Response Format Requirement
- Return code files (diff or full content). Avoid long explanations.
- Each change must be buildable and testable: `make lint && make test`.

## Definition of Done (DoD)
- Lint + type-check pass
- Tests added/updated and passing
- Make targets runnable
- If touching API, add a minimal integration test
- Update docs if interfaces change

## Known Constraints
- Some books may include poor OCR; keep ingestion extensible.
- Large books: memory constraints → batch upserts to Qdrant.
- GPU optional for embeddings; provide CPU fallback/mocks for CI.

