from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.ingest.epub import extract_pages as extract_epub_pages
from core.ingest.pdf_to_text import Page
from ingest.clean_rules import apply_rules, load_rules
from ingest.pdf_reader import read_pdf
from nlp.fa_normalize import normalize_fa

MetadataInput = Mapping[str, Any] | None


@dataclass(slots=True)
class IngestResult:
    """Materialized output of the ingestion pipeline."""

    book_id: str
    version: str
    file_hash: str
    metadata: dict[str, Any]
    pages: list[Page]
    records: list[dict[str, Any]]

    def chunk_source(self) -> list[tuple[str, int]]:
        """Return ready-to-chunk (text, page_num) tuples."""

        return [(page.text, page.page_num) for page in self.pages]


def _normalize_metadata(meta: MetadataInput) -> dict[str, Any]:
    if not meta:
        return {}
    return {
        str(key): value
        for key, value in meta.items()
        if value not in (None, "", [], {})
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_page(obj: Page | Mapping[str, Any]) -> Page:
    if isinstance(obj, Page):
        return Page(
            page_num=int(getattr(obj, "page_num", -1)),
            text=getattr(obj, "text", "") or "",
            section=getattr(obj, "section", None),
        )
    if isinstance(obj, Mapping):
        return Page(
            page_num=int(obj.get("page_num", -1)),
            text=str(obj.get("text", "")),
            section=obj.get("section"),
        )
    raise TypeError(f"Unsupported page object type: {type(obj)!r}")


def _load_pages(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".epub":
        return [_ensure_page(page_dict) for page_dict in extract_epub_pages(path)]
    raise ValueError(f"Unsupported file type for ingestion: {path.suffix}")


def _prepare_pages(
    pages: Sequence[Page | Mapping[str, Any]] | Iterable[Page | Mapping[str, Any]],
    *,
    book_id: str,
    version: str,
    file_hash: str,
    metadata: Mapping[str, Any],
    rules_cfg: Mapping[str, Any],
) -> tuple[list[Page], list[dict[str, Any]]]:
    processed_pages: list[Page] = []
    records: list[dict[str, Any]] = []
    for raw_page in pages:
        page = _ensure_page(raw_page)
        cleaned_text = apply_rules(page.text, rules_cfg)
        normalized_text = normalize_fa(cleaned_text)
        page.text = normalized_text
        record = {
            "book_id": book_id,
            "version": version,
            "file_hash": file_hash,
            "page_num": page.page_num,
            "section": page.section,
            "text": normalized_text,
            "meta": dict(metadata),
        }
        processed_pages.append(page)
        records.append(record)
    return processed_pages, records


def _new_book_id() -> str:
    return str(uuid.uuid4())


def _new_version() -> str:
    return datetime.now(timezone.utc).isoformat()


try:
    _DEFAULT_RULES = load_rules()
except TypeError:
    _DEFAULT_RULES = {}


def ingest_file(
    path: Path,
    *,
    metadata: MetadataInput = None,
    book_id: str | None = None,
    version: str | None = None,
    rules_cfg: Mapping[str, Any] | None = None,
) -> IngestResult:
    """Run the ingestion pipeline for a single document path."""

    normalized_meta = _normalize_metadata(metadata)
    book_identifier = book_id or _new_book_id()
    version_identifier = version or _new_version()
    file_hash = _hash_file(path)
    rules = _DEFAULT_RULES if rules_cfg is None else rules_cfg
    pages = _load_pages(path)
    processed_pages, records = _prepare_pages(
        pages,
        book_id=book_identifier,
        version=version_identifier,
        file_hash=file_hash,
        metadata=normalized_meta,
        rules_cfg=rules,
    )
    return IngestResult(
        book_id=book_identifier,
        version=version_identifier,
        file_hash=file_hash,
        metadata=normalized_meta,
        pages=processed_pages,
        records=records,
    )


def pages_to_records(
    pages: Iterable[Page | Mapping[str, Any]],
    *,
    book_id: str,
    version: str,
    file_hash: str,
    metadata: MetadataInput = None,
    rules_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert page objects to JSONL-ready dictionaries including metadata."""

    normalized_meta = _normalize_metadata(metadata)
    rules = _DEFAULT_RULES if rules_cfg is None else rules_cfg
    _, records = _prepare_pages(
        list(pages),
        book_id=book_id,
        version=version,
        file_hash=file_hash,
        metadata=normalized_meta,
        rules_cfg=rules,
    )
    return records


def write_records(records: Iterable[Mapping[str, Any]], out_path: Path) -> Path:
    """Persist JSONL records to disk, ensuring metadata is serializable."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            payload = dict(record)
            meta = payload.get("meta", {})
            if isinstance(meta, Mapping):
                payload["meta"] = dict(meta)
            else:
                payload["meta"] = {}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return out_path


__all__ = ["IngestResult", "ingest_file", "pages_to_records", "write_records"]
