# AGENTS.md

## Overview

This document defines the responsibilities of each agent in the KetabMind system.
Agents are modular components responsible for ingestion, normalization, embedding, retrieval, answer generation, UI handling, and evaluation.
The design emphasizes multilingual (especially Persian/English) support, reproducibility, and modularity.

---

## Agents

### 1. Ingestion & OCR Agent

**Responsibility:** Convert input books into structured, normalized text with metadata.

* **Digital PDFs:** Use PyMuPDF (or PDFMiner) to extract UTF-8 text. Apply configurable header/footer removal.
* **Scanned PDFs:** Use Tesseract OCR with Farsi language (`OCR_FA=true`). Configure the Tesseract language data path.
* **OCR fallback:** Attempt text extraction first; if text density < threshold or file is image-only, route through OCR with multilingual (fa+eng) configuration and log fallback usage.
* **EPUB:** Parse the spine in correct order, strip HTML tags, and preserve chapter order.
* **Output format:** JSONL with `{ book_id, version, page_num, section, text }`.
* **Deduplication:** Hash the entire file (e.g., SHA256) before processing. Skip ingestion when hash already exists for the collection and record the duplicate event.
* **Versioning:** Maintain monotonically increasing `version` per `book_id`. Store previous versions for auditability and allow rollback.
* **Metadata:** Extract author, year, subject, language, and additional user-supplied fields. Persist metadata alongside each page payload to support filters and analytics.
* **Metadata validation:** Normalize casing, coerce numeric fields, and reject ingestion when mandatory metadata (author, year, subject) is missing.

---

### 2. Normalization Agent

**Responsibility:** Apply Farsi-specific cleanup and text normalization before chunking/embedding.

* Unify Arabic vs. Persian variants of ی/ک using transliteration tables.
* Strip diacritics (harakat) and tatweel characters before downstream processing.
* Normalize spaces (replace with standard space, fix non-breaking, collapse multiple spaces).
* Apply ZWNJ where required (e.g., for prefixes like «می‌», «نمی‌», compound nouns, and suffixes such as «ها»).
* Standardize punctuation (convert Arabic punctuation to Persian equivalents, normalize quotes/dashes).
* Lowercasing: not used for Persian; rely on normalization and tokenization.
* Regex strategy: centralize normalization rules as ordered regex substitutions applied to each page/section; document each rule with comments and unit tests.
* Unicode helpers: use Python's `unicodedata` plus curated regex patterns; fallback to libraries like **Hazm** or **Parsivar** for tokenization/lemmatization where needed.

---

### 3. Embedding & Vector Agent

**Responsibility:** Generate multilingual embeddings and manage vector store.

* **Recommended models:** `bge-m3` or `intfloat/multilingual-e5-base`. Both work well for Persian.
* **Configuration:**

  * `EMBED_MODEL=multilingual`
  * `EMBED_MODEL_NAME=bge-m3` (or e5-base-multilingual)
* **Storage:** Ensure embeddings align with Qdrant schema.
* **Quantization:** Support configurable 4/8-bit quantization via `EMBED_QUANT`, with safe fallbacks to full precision when unsupported.
* **Batching:** Expose `BATCH_SIZE` to batch embedding requests, automatically tuning for throughput vs. memory.
* **Fallbacks:** Provide CPU execution and `mock` embedding backends for development/testing when GPUs are unavailable.
* **Collection management:** Recreate or clear Qdrant collections when switching models, quantization levels, or dimensionality.

---

### 4. Retrieval & Reranker Agent

**Responsibility:** Retrieve relevant chunks and re-rank for precision.

* Initial retrieval: cosine similarity over embeddings.
* Reranking: Cross-encoder (default `bge-reranker-v2-m3`, override via `RERANKER_MODEL_NAME`).
* Hybrid scoring: combine embedding cosine + lexical overlap (Farsi preprocessing) + reranker score. Configure weights with `HYBRID_WEIGHTS="cosine=0.3,lexical=0.2,reranker=0.5"` (values must sum to ≤1 after normalization).
* Environment toggles:
  * `RERANKER_ENABLED=true|false` to switch the reranker stage on/off per deployment.
  * `RERANKER_MODEL_NAME` to pin the cross-encoder checkpoint.
  * `HYBRID_WEIGHTS` to tune weighting across cosine, lexical, and reranker scores.
  * `QUERY_DEBUG=true` or HTTP `?debug=true` to surface hybrid score breakdowns for diagnostics.
* Return top-N passages with metadata: `{ book_id, page, snippet, score }`.
* Troubleshooting guidance: lower `HYBRID_WEIGHTS` for the reranker or disable it when GPU OOM occurs; shrink `top_k` or raise service timeouts for query timeouts.

---

### 5. Prompting & Answer Agent

**Responsibility:** Synthesize answers with citations in the query’s language.

* **Bilingual template:** Detect query language; respond in same language.
* **Citation format:** `[bookID:page_start-page_end]`.
* **Formatting:** Use bullet-points for clarity, include transparent citations.
* **Fallback:** If evidence insufficient, return: *“Not enough information to answer accurately.”*
* **Self-RAG:** If citation coverage is low, trigger a second retrieval+answer pass.

---

### 6. UI & RTL Agent

**Responsibility:** User interface rendering with multilingual/RTL support.

* **Pages:**

  * Book upload page.
  * Search/Chat page.
* **Citations:** Clickable, jumping to book pages.
* **RTL Support:** Persian font (e.g. Vazirmatn), `direction: rtl`, `unicode-bidi: plaintext`.
* **Auth:** Multi-user auth (JWT/OAuth). Each user has private space for their own books and sessions.

---

### 7. Evaluation Agent

**Responsibility:** Define and run evaluation metrics for system quality.

* Build a small Persian Q&A dataset.
* Compute **Exact Match (EM)**, **F1**, and **Citation Coverage** (percentage of answer sentences backed by retrieved contexts).
* Tokenize Persian answers word-level for F1.
* Run evaluations regularly and log results.

---

## System-Wide Considerations

* **Logging & Monitoring:**

  * Use `structlog` for structured logs.
  * Export Prometheus metrics (latency, queries, token usage).
  * Grafana dashboards for monitoring.
* **Testing & CI/CD:**

  * Unit, integration, and end-to-end tests.
  * Run via GitHub Actions.
* **Product Features:**

  * History & Sessions.
  * Bookmarks.
  * Export answers to PDF/Word.
  * Full-text search as pre-filter.
* **Commercialization:**

  * **SaaS version:** Centralized server with multi-user.
  * **Desktop version:** Local GPU/offline support via Ollama/Transformers.
  * **Pricing:** Subscription-based (storage size, number of books, query volume).

