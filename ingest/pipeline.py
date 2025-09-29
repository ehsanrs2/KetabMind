from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.ingest.pdf_to_text import Page
from ingest.clean_rules import apply_rules, load_rules

MetadataInput = Mapping[str, Any] | None


def _normalize_metadata(meta: MetadataInput) -> dict[str, Any]:
    if not meta:
        return {}
    return {
        str(key): value
        for key, value in meta.items()
        if value not in (None, "", [], {})
    }


try:
    _DEFAULT_RULES = load_rules()
except TypeError:
    _DEFAULT_RULES = {}


def pages_to_records(
    pages: Iterable[Page],
    *,
    book_id: str,
    version: str,
    file_hash: str,
    metadata: MetadataInput = None,
    rules_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert page objects to JSONL-ready dictionaries including metadata."""

    normalized_meta = _normalize_metadata(metadata)
    records: list[dict[str, Any]] = []
    rules = _DEFAULT_RULES if rules_cfg is None else rules_cfg
    for page in pages:
        raw_text = getattr(page, "text", "") or ""
        cleaned_text = apply_rules(raw_text, rules)
        record = {
            "book_id": book_id,
            "version": version,
            "file_hash": file_hash,
            "page_num": int(getattr(page, "page_num", -1)),
            "section": getattr(page, "section", None),
            "text": cleaned_text,
            "meta": dict(normalized_meta),
        }
        records.append(record)
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


__all__ = ["pages_to_records", "write_records"]
