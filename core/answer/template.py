from __future__ import annotations

from core.retrieve.retriever import ScoredChunk


def build_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    lines: list[str] = [
        "Use the provided contexts to answer the question.",
        "Cite sources as (book_id page_start-page_end).",
        f"Question: {question}",
        f"Contexts ({len(contexts)}):",
    ]
    for idx, ctx in enumerate(contexts, 1):
        lines.append(f"{idx}. ({ctx.book_id} {ctx.page_start}-{ctx.page_end}) {ctx.text}")
    lines.append("Answer:")
    return "\n".join(lines)
