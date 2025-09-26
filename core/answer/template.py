from __future__ import annotations

from core.retrieve.retriever import ScoredChunk


def build_prompt(question: str, contexts: list[ScoredChunk], sys_instructions: str) -> str:
    """Create an instruction-tuned prompt using ``question`` and ``contexts``."""

    system_section = sys_instructions.strip()
    if not system_section:
        system_section = "You are KetabMind, a helpful research assistant."
    lines: list[str] = [
        "System:",
        system_section,
        "",
        "Instructions:",
        (
            "Answer with citations like [book_id:page_start-page_end]. "
            "If missing evidence, say 'insufficient evidence'."
        ),
        "",
        f"Question: {question}",
        "",
        f"Contexts ({len(contexts)}):",
    ]
    for idx, ctx in enumerate(contexts, 1):
        lines.append(f"{idx}. ({ctx.book_id}:{ctx.page_start}-{ctx.page_end}) {ctx.text}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)
