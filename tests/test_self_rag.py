from __future__ import annotations

from core.retrieve.retriever import ScoredChunk
from core.self_rag.validator import validate


def _mk_ctx(texts: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            id=f"c{idx}",
            book_id="b",
            page=1,
            snippet=text,
            cosine=0.0,
            lexical=0.0,
            reranker=0.0,
            hybrid=0.0,
        )
        for idx, text in enumerate(texts, 1)
    ]


def test_validator_flags_no_context_match() -> None:
    answer = "The sky is blue. Birds fly."
    ctx = _mk_ctx(["Dogs bark.", "Cats purr."])
    assert validate(answer, ctx, coverage_threshold=0.5) is False


def test_validator_accepts_with_overlap() -> None:
    answer = "The sky is blue."
    ctx = _mk_ctx(["A clear blue sky is visible."])
    assert validate(answer, ctx, coverage_threshold=0.2) is True
