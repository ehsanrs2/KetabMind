from __future__ import annotations

from core.config import Settings
from core.retrieve.retriever import ScoredChunk

from self_rag.advanced import AnswerResult, reformulate_query, second_pass


class FakeRetriever:
    def __init__(self) -> None:
        self.top_m = 12
        self.hybrid_weights = {"cosine": 0.7, "lexical": 0.3}
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
        normalized = query.lower()
        if "metaphor" in normalized or "imagery" in normalized:
            return [
                ScoredChunk(
                    id="c2",
                    book_id="b",
                    page=2,
                    snippet="Metaphors and rich imagery define the poems.",
                    cosine=0.8,
                    lexical=0.7,
                    reranker=0.6,
                    hybrid=0.85,
                )
            ]
        return [
            ScoredChunk(
                id="c1",
                book_id="b",
                page=1,
                snippet="Symbolism and imagery appear in classical verses.",
                cosine=0.2,
                lexical=0.2,
                reranker=0.1,
                hybrid=0.25,
            )
        ]


class ReactiveLLM:
    def generate(self, prompt: str, stream: bool = False) -> str:
        if "metaphor" in prompt.lower() or "imagery" in prompt.lower():
            return "Metaphors and imagery make Persian poetry emotionally vivid."
        return "Persian poetry is rooted in cultural heritage."


def test_reformulate_query_appends_keywords() -> None:
    question = "Why is Persian poetry distinctive?"
    top_chunks = [
        {
            "text": "Symbolism and imagery appear in classical verses.",
            "metadata": {"entities": ["Classical Verses"]},
        },
        {
            "snippet": "Metaphors enrich emotional expression.",
            "metadata": {},
        },
    ]

    reformulated = reformulate_query(question, top_chunks)
    assert reformulated.startswith(question)
    assert "AND" in reformulated
    assert "symbolism" in reformulated.lower()
    assert "imagery" in reformulated.lower()


def test_second_pass_improves_coverage(monkeypatch) -> None:
    from core.answer import answerer

    fake_retriever = FakeRetriever()
    monkeypatch.setattr(answerer, "_retriever", fake_retriever)
    monkeypatch.setattr(answerer, "get_llm", lambda: ReactiveLLM())
    monkeypatch.setattr(answerer, "_refusal_rule", lambda: "never")

    settings = Settings()

    result = second_pass("Why is Persian poetry distinctive?", settings)
    assert isinstance(result, AnswerResult)
    assert result.passes == 2
    assert result.reformulated_query is not None
    lowered_query = result.reformulated_query.lower()
    assert "symbolism" in lowered_query
    assert "imagery" in lowered_query

    assert len(fake_retriever.calls) == 2
    first_call, second_call = fake_retriever.calls
    assert first_call["query"] == "Why is Persian poetry distinctive?"
    assert "AND" in str(second_call["query"])
    assert second_call["top_m"] > first_call["top_m"]
    assert second_call["weights"]["lexical"] > first_call["weights"]["lexical"]

    first_coverage = result.debug["first_pass"]["coverage"]
    assert result.coverage > first_coverage
    assert "metaphors" in result.answer.lower()
