from __future__ import annotations

import pytest

from answering.citations import build_citations, format_citation, merge_page_ranges


def test_merge_page_ranges_overlapping_and_consecutive() -> None:
    spans = [(5, 3), (4, 4), (6, 8), (10, 10), (11, 12)]
    assert merge_page_ranges(spans) == [(3, 8), (10, 12)]


def test_format_citation_supports_page_range_placeholder() -> None:
    result = format_citation("book", [(1, 1), (2, 4)], "[${book_id}:${page_range}]")
    assert result == "[book:1], [book:2-4]"


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (
            [
                {"book_id": "b", "page_start": 1, "page_end": 2},
                {"book_id": "b", "page_start": 3, "page_end": 4},
                {"book_id": "b", "page": 5},
                {"book_id": "a", "page_start": 10, "page_end": 12},
                {"book_id": "a", "page_end": 13},
                {"book_id": "a", "page_start": 14, "page_end": 14},
                {"book_id": "a", "page_start": 14, "page_end": 14},
                {"book_id": "b", "page_start": 4, "page_end": 4},
            ],
            ["[a:10-14]", "[b:1-5]"],
        ),
        (
            [
                {"book_id": "b", "page": "2"},
                {"book_id": "b", "page": 3},
                {"book_id": "b", "page": "2"},
                {"book_id": "c", "page_start": 7, "page_end": 9},
                {"book_id": "c", "page_start": 11, "page_end": 11},
            ],
            ["[b:2-3]", "[c:7-9]", "[c:11]"],
        ),
    ],
)
def test_build_citations_groups_and_sorts(chunks: list[dict], expected: list[str]) -> None:
    fmt = "[${book_id}:${page_range}]"
    assert build_citations(chunks, fmt) == expected
