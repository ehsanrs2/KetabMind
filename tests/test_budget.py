from __future__ import annotations

from statistics import mean

import pytest

from answering.budget import select_contexts, trim_snippet

BUDGET_MAX_CHUNKS = 3


def _avg_pairwise_similarity(snippets: list[str]) -> float:
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in text.split()}

    if len(snippets) < 2:
        return 0.0

    pairs: list[float] = []
    for idx, left in enumerate(snippets):
        left_tokens = _tokenize(left)
        for right in snippets[idx + 1 :]:
            right_tokens = _tokenize(right)
            if not left_tokens or not right_tokens:
                similarity = 0.0
            else:
                similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            pairs.append(similarity)
    return mean(pairs) if pairs else 0.0


def test_select_contexts_respects_budget_and_diversifies() -> None:
    candidates = [
        {
            "id": "c1",
            "book_id": "book-a",
            "snippet": "Neural networks learn layered representations of data for tasks.",
            "hybrid": 0.95,
        },
        {
            "id": "c2",
            "book_id": "book-a",
            "snippet": "Deep neural networks rely on gradient descent and backpropagation.",
            "hybrid": 0.92,
        },
        {
            "id": "c3",
            "book_id": "book-b",
            "snippet": (
                "Statistical learning theory explores generalization bounds and VC dimension."
            ),
            "hybrid": 0.9,
        },
        {
            "id": "c4",
            "book_id": "book-c",
            "snippet": (
                "Reinforcement learning agents optimise rewards via value functions and policies."
            ),
            "hybrid": 0.88,
        },
    ]

    naive_top = candidates[:BUDGET_MAX_CHUNKS]
    naive_similarity = _avg_pairwise_similarity([c["snippet"] for c in naive_top])

    diversified = select_contexts(candidates, BUDGET_MAX_CHUNKS, diversity_alpha=0.5)
    diversified_similarity = _avg_pairwise_similarity([c["snippet"] for c in diversified])

    assert len(diversified) <= BUDGET_MAX_CHUNKS
    assert {c["id"] for c in diversified} <= {c["id"] for c in candidates}
    assert diversified_similarity < naive_similarity


def test_trim_snippet_focuses_on_question_terms() -> None:
    text = (
        "Neural networks are a series of algorithms that mimic the operations of a human brain "
        "to recognise relationships between vast amounts of data and improve learning over time."
    )
    question = "How do neural networks improve learning?"

    shorter = trim_snippet(text, question, 6)
    longer = trim_snippet(text, question, 8)

    assert "learning" in shorter.lower()
    assert len(shorter.split()) <= 8  # includes possible ellipsis tokens
    for term in {"neural", "networks", "improve", "learning"}:
        if term in longer.lower():
            break
    else:
        pytest.fail("trimmed snippet should include at least one query term")
    assert len(longer.split()) <= 10
    assert longer != shorter
