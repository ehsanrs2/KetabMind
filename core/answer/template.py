from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
import re

from core.retrieve.retriever import ScoredChunk

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - transformers not installed
    AutoTokenizer = None  # type: ignore[assignment]


TokenCounter = Callable[[str], int]


def _character_count(text: str) -> int:
    return len(text)


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


@lru_cache(maxsize=8)
def _get_token_counter(tokenizer_name: str) -> TokenCounter:
    hint = tokenizer_name.lower()
    if "character" in hint:
        return _character_count
    if AutoTokenizer is None:
        return _word_count
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception:  # pragma: no cover - network/model errors
        return _word_count

    def count(text: str) -> int:
        if not text:
            return 0
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    return count


def estimate_token_count(texts: Iterable[str], tokenizer_name: str) -> int:
    counter = _get_token_counter(tokenizer_name)
    total = 0
    for text in texts:
        total += counter(text)
    return total


def select_contexts_within_budget(
    question: str,
    contexts: list[ScoredChunk],
    tokenizer_name: str,
    max_input_tokens: int,
) -> list[ScoredChunk]:
    if max_input_tokens <= 0 or not contexts:
        return []
    usable_budget = int(max_input_tokens * 0.75)
    if usable_budget <= 0:
        return []
    counter = _get_token_counter(tokenizer_name)
    selected: list[ScoredChunk] = []
    consumed = counter(question)
    if consumed >= usable_budget:
        return []
    for chunk in sorted(contexts, key=lambda c: c.score, reverse=True):
        chunk_tokens = counter(chunk.text)
        if consumed + chunk_tokens > usable_budget:
            break
        selected.append(chunk)
        consumed += chunk_tokens
    return selected


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
        lines.append(f"{idx}. ({ctx.book_id}:{ctx.page}) {ctx.text}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)
