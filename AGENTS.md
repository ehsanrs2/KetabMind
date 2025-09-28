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
* **EPUB:** Parse the spine in correct order, strip HTML tags, and preserve chapter order.
* **Output format:** JSONL with `{ book_id, version, page_num, section, text }`.
* **Deduplication:** Hash the entire file to prevent re-indexing duplicates.
* **Versioning:** Assign a unique ID/version for each upload to maintain history.
* **Metadata:** Extract author, year, subject, and store in payload for filtering (e.g., “only math books”).

---

### 2. Normalization Agent

**Responsibility:** Apply Farsi-specific cleanup and text normalization before chunking/embedding.

* Unify Arabic vs. Persian variants of ی/ک.
* Strip diacritics (harakat).
* Normalize spaces (replace with standard space, fix non-breaking).
* Apply ZWNJ where required.
* Lowercasing: not used for Persian; rely on normalization and tokenization.
* Lightweight regex + Unicode sufficient. Optionally use **Hazm** or **Parsivar** for tokenization/lemmatization.

---

### 3. Embedding & Vector Agent

**Responsibility:** Generate multilingual embeddings and manage vector store.

* **Recommended models:** `bge-m3` or `intfloat/multilingual-e5-base`. Both work well for Persian.
* **Configuration:**

  * `EMBED_MODEL=multilingual`
  * `EMBED_MODEL_NAME=bge-m3` (or e5-base-multilingual)
* **Storage:** Ensure embeddings align with Qdrant schema.
* **Model updates:** Clear old Qdrant collections when switching models.
* **Optimization:** Quantization (4/8-bit) for RTX 3060; batching support for multiple queries.

---

### 4. Retrieval & Reranker Agent

**Responsibility:** Retrieve relevant chunks and re-rank for precision.

* Initial retrieval: cosine similarity over embeddings.
* Reranking: Cross-encoder (e.g., `bge-reranker-v2-m3`).
* Hybrid scoring: combine embedding cosine + lexical overlap (Farsi preprocessing) + reranker score.
* Return top-N passages with metadata: `{ book_id, page, snippet, score }`.

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

