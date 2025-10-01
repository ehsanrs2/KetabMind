from __future__ import annotations

import re
from typing import Any, Iterable

from answering.citations import build_citations
from answering.evidence_map import split_sentences

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")


def _page_num_from_context(context: dict[str, Any]) -> Any:
    metadata = context.get("metadata") or {}
    page_num = context.get("page")
    if page_num is None:
        page_num = metadata.get("page_num")
    return page_num


def _chunk_debug(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": context.get("id"),
        "book_id": context.get("book_id"),
        "page_num": _page_num_from_context(context),
        "cosine": context.get("cosine"),
        "lexical": context.get("lexical"),
        "reranker": context.get("reranker"),
        "hybrid": context.get("hybrid"),
    }


def _detect_language(answer: str) -> str:
    if _ARABIC_SCRIPT_RE.search(answer):
        return "fa"
    return "en"


def _support_from_context(context: dict[str, Any]) -> dict[str, Any] | None:
    support: dict[str, Any] = {}
    book_id = context.get("book_id")
    if book_id is not None:
        support["book_id"] = book_id
    page = _page_num_from_context(context)
    if page is not None:
        support["page"] = page
    hybrid = context.get("hybrid")
    if hybrid is not None:
        support["hybrid"] = hybrid
    return support or None


def _build_evidence_map(answer: str, contexts: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    sentences = split_sentences(answer, lang)
    supports = [
        entry for ctx in contexts if (entry := _support_from_context(ctx))
    ]
    evidence: list[dict[str, Any]] = []
    for sentence in sentences:
        evidence.append({"sentence": sentence, "supports": list(supports)})
    return evidence


def _build_used_contexts(contexts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    used: list[dict[str, Any]] = []
    for ctx in contexts:
        used.append(
            {
                "id": ctx.get("id"),
                "book_id": ctx.get("book_id"),
                "page": _page_num_from_context(ctx),
                "hybrid": ctx.get("hybrid"),
            }
        )
    return used


def _build_citations(contexts: list[dict[str, Any]]) -> list[str]:
    return build_citations(contexts, "[${book_id}:${page_range}]")


def _extract_stats(result: dict[str, Any]) -> dict[str, float]:
    debug = result.get("debug") or {}
    stats = debug.get("stats") or {}
    coverage = stats.get("coverage") or 0.0
    confidence = stats.get("confidence") or 0.0
    return {"coverage": float(coverage), "confidence": float(confidence)}


def build_query_response(result: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    """Serialize answer payload for the /query endpoint."""

    contexts = list(result.get("contexts", []))
    answer = result.get("answer", "")
    lang = _detect_language(answer)
    stats = _extract_stats(result)

    payload: dict[str, Any] = {
        "answer": answer,
        "citations": _build_citations(contexts),
        "meta": {
            "lang": lang,
            "coverage": stats["coverage"],
            "confidence": stats["confidence"],
        },
    }

    if debug:
        debug_info = dict(result.get("debug") or {})
        debug_info["chunks"] = [_chunk_debug(ctx) for ctx in contexts]
        payload["debug"] = debug_info
        payload["evidence_map"] = _build_evidence_map(answer, contexts, lang)
        payload["used_contexts"] = _build_used_contexts(contexts)

    return payload
