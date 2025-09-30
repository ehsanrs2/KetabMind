from __future__ import annotations

from typing import Any, Iterable


def _page_num_from_context(context: dict[str, Any]) -> Any:
    metadata = context.get("metadata") or {}
    page_num = context.get("page")
    if page_num is None:
        page_num = metadata.get("page_num")
    return page_num


def _citation_from_context(context: dict[str, Any]) -> dict[str, Any]:
    metadata = context.get("metadata") or {}
    citation: dict[str, Any] = {
        "book_id": context.get("book_id"),
        "page_num": _page_num_from_context(context),
        "version": metadata.get("version"),
    }
    chunk_id = context.get("id")
    if chunk_id is not None:
        citation["chunk_id"] = chunk_id
    return citation


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


def build_query_response(result: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    """Serialize answer payload for the /query endpoint."""

    contexts: Iterable[dict[str, Any]] = result.get("contexts", [])
    citations = [_citation_from_context(ctx) for ctx in contexts]

    payload: dict[str, Any] = {
        "answer": result.get("answer", ""),
        "citations": citations,
    }

    if debug:
        debug_info = dict(result.get("debug") or {})
        debug_info["chunks"] = [_chunk_debug(ctx) for ctx in contexts]
        payload["debug"] = debug_info

    return payload
