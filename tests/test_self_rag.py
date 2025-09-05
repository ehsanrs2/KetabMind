from __future__ import annotations

from core.retrieve.retriever import ScoredChunk
from core.self_rag.validator import validate


def _mk_ctx(texts: list[str]) -> list[ScoredChunk]:
    return [ScoredChunk(text=t, book_id="b", page_start=1, page_end=1, score=0.0) for t in texts]


def test_validator_flags_no_context_match() -> None:
    answer = "The sky is blue. Birds fly."
    ctx = _mk_ctx(["Dogs bark.", "Cats purr."])
    assert validate(answer, ctx, coverage_threshold=0.5) is False


def test_validator_accepts_with_overlap() -> None:
    answer = "The sky is blue."
    ctx = _mk_ctx(["A clear blue sky is visible."])
    assert validate(answer, ctx, coverage_threshold=0.2) is True
