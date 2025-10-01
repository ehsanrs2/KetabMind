"""Helpers for building formatted citations from retrieved chunks."""

from __future__ import annotations

from collections import defaultdict
from string import Template


def merge_page_ranges(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping and consecutive page ranges.

    The input spans are inclusive ``(start, end)`` pairs that may be unordered.
    The result is sorted by the starting page.
    """

    if not spans:
        return []

    # Normalise and sort spans by their starting page.
    normalised = sorted((min(a, b), max(a, b)) for a, b in spans)

    merged: list[tuple[int, int]] = []
    for start, end in normalised:
        if not merged:
            merged.append((start, end))
            continue

        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def format_citation(book_id: str, ranges: list[tuple[int, int]], fmt: str) -> str:
    """Format ``book_id`` and ``ranges`` according to ``fmt``.

    The format string is interpreted as a :class:`string.Template`.  Each range
    provides the following placeholders:

    ``book_id``
        The identifier of the cited book.
    ``page_start`` / ``page_end``
        Inclusive start and end pages for the range.
    ``page``
        Alias for ``page_start``.
    ``page_range``
        Either ``"<page>"`` for single pages or ``"<start>-<end>"`` for spans.
    """

    template = Template(fmt)
    formatted: list[str] = []
    for page_start, page_end in ranges:
        if page_start > page_end:
            page_start, page_end = page_end, page_start
        page_range = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        substitutions = {
            "book_id": book_id,
            "page_start": page_start,
            "page_end": page_end,
            "page": page_start,
            "page_range": page_range,
        }
        formatted.append(template.safe_substitute(substitutions))
    return ", ".join(formatted)


def _extract_page_range(chunk: dict) -> tuple[int, int] | None:
    """Extract an inclusive page range from a chunk mapping."""

    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")

    if page_start is None:
        page_start = chunk.get("page")
    if page_start is None:
        page_start = chunk.get("page_num")

    if page_start is None and page_end is None:
        return None

    if page_start is None:
        page_start = page_end
    if page_end is None:
        page_end = page_start

    try:
        start = int(page_start)
        end = int(page_end)
    except (TypeError, ValueError):
        return None

    if start > end:
        start, end = end, start
    return start, end


def build_citations(chunks: list[dict], fmt: str) -> list[str]:
    """Aggregate ``chunks`` into sorted, de-duplicated citations."""

    spans_by_book: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for chunk in chunks:
        book_id = chunk.get("book_id")
        if not book_id:
            continue
        span = _extract_page_range(chunk)
        if span is None:
            continue
        spans_by_book[str(book_id)].append(span)

    entries: list[tuple[str, int, int]] = []
    for book_id, spans in spans_by_book.items():
        for start, end in merge_page_ranges(spans):
            entries.append((book_id, start, end))

    entries.sort(key=lambda item: (item[0], item[1], item[2]))

    seen: set[tuple[str, int, int]] = set()
    citations: list[str] = []
    for book_id, start, end in entries:
        key = (book_id, start, end)
        if key in seen:
            continue
        seen.add(key)
        citation = format_citation(book_id, [(start, end)], fmt)
        if citation:
            citations.append(citation)

    return citations


__all__ = [
    "build_citations",
    "merge_page_ranges",
    "format_citation",
]
