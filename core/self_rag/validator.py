"""Simple Self-RAG validation utilities.

Detect hallucinations by checking citation coverage of the answer
against provided retrieved contexts.
"""

from __future__ import annotations

import re
from typing import Iterable

from core.retrieve.retriever import ScoredChunk


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _sentences(text: str) -> list[str]:
    """Very small sentence splitter.

    Splits on `.`, `?`, `!` and keeps non-empty trimmed segments.
    """
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    denom = len(sa | sb) or 1
    return inter / denom


def _coverage(answer: str, contexts_texts: list[str], match_threshold: float = 0.2) -> float:
    """Compute fraction of answer sentences supported by any context.

    A sentence is considered supported if token Jaccard overlap with at
    least one context is >= `match_threshold`.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    ctx_tokenized = [_tokens(t) for t in contexts_texts]
    supported = 0
    for s in sents:
        st = _tokens(s)
        best = 0.0
        for ct in ctx_tokenized:
            if not ct:
                continue
            # Jaccard similarity
            num = len(st & ct)
            den = len(st | ct) or 1
            score = num / den
            if score > best:
                best = score
        if best >= match_threshold:
            supported += 1
    return supported / max(1, len(sents))


def validate(answer: str, contexts: list[ScoredChunk], *, coverage_threshold: float = 0.4) -> bool:
    """Return True if answer is sufficiently grounded in contexts.

    If citation coverage is below the given threshold, return False.
    """
    ctx_texts = [c.text for c in contexts]
    cov = _coverage(answer, ctx_texts)
    return cov >= coverage_threshold

