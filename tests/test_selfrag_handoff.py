from __future__ import annotations

from core.self_rag.runner import run_self_rag


class ScoredChunk:
    def __init__(
        self,
        *,
        id: str,
        book_id: str,
        page: int,
        snippet: str,
        cosine: float,
        lexical: float,
        reranker: float,
        hybrid: float,
    ) -> None:
        self.id = id
        self.book_id = book_id
        self.page = page
        self.snippet = snippet
        self.cosine = cosine
        self.lexical = lexical
        self.reranker = reranker
        self.hybrid = hybrid

    @property
    def text(self) -> str:
        return self.snippet


class FakeRetriever:
    def __init__(self) -> None:
        self.top_m = 10
        self.hybrid_weights = {"cosine": 0.6, "lexical": 0.4}
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, top_n: int | None = None) -> list[ScoredChunk]:
        self.calls.append(
            {
                "query": query,
                "top_m": self.top_m,
                "weights": dict(self.hybrid_weights),
                "top_n": top_n,
            }
        )
        if len(self.calls) == 1:
            return [
                ScoredChunk(
                    id="c1",
                    book_id="b",
                    page=1,
                    snippet="Symbolism and imagery are mentioned briefly in classical verses.",
                    cosine=0.1,
                    lexical=0.1,
                    reranker=0.0,
                    hybrid=0.2,
                )
            ]
        return [
            ScoredChunk(
                id="c2",
                book_id="b",
                page=2,
                snippet="Persian poetry uses metaphors and rich imagery to convey emotion.",
                cosine=0.8,
                lexical=0.6,
                reranker=0.7,
                hybrid=0.9,
            )
        ]


def synthesize_answer(question: str, contexts: list[ScoredChunk]) -> str:
    if any("metaphor" in ctx.text.lower() for ctx in contexts):
        return "Persian poetry uses metaphors and rich imagery to convey emotion."
    return "Cultural heritage influences its themes."


def test_second_pass_improves_coverage() -> None:
    retriever = FakeRetriever()

    result = run_self_rag(
        "Why is Persian poetry distinctive?",
        retriever=retriever,
        synthesizer=synthesize_answer,
        top_k=1,
        coverage_threshold=0.5,
        hybrid_threshold=0.3,
        keyword_limit=2,
        second_pass_top_m=30,
        hybrid_weight_delta=0.2,
    )

    assert result.passes == 2
    assert len(retriever.calls) == 2

    first_call, second_call = retriever.calls
    assert isinstance(first_call["query"], str)
    assert isinstance(second_call["query"], str)
    assert second_call["query"] != first_call["query"]
    assert second_call["query"].endswith("symbolism imagery")
    assert second_call["top_m"] > first_call["top_m"]
    assert second_call["weights"]["lexical"] > first_call["weights"]["lexical"]

    first_pass = result.debug["first_pass"]
    assert isinstance(first_pass, dict)
    assert result.coverage > first_pass["coverage"]
    assert result.avg_hybrid > first_pass["avg_hybrid"]

    assert result.reformulated_query == second_call["query"]
    assert "metaphors" in result.answer.lower()
