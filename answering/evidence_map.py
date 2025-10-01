"""Utilities for mapping answer sentences to supporting contexts."""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Iterable

from embedding.adapter import EmbeddingAdapter


_STOPWORDS = {
    # English stopwords (small curated subset)
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "are",
    "was",
    "were",
    "will",
    "would",
    "could",
    "should",
    "has",
    "have",
    "had",
    "not",
    "but",
    "yet",
    "his",
    "her",
    "their",
    "its",
    "our",
    "you",
    "your",
    "yours",
    "they",
    "them",
    "she",
    "him",
    "who",
    "whom",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
    "into",
    "onto",
    "than",
    "then",
    "about",
    "over",
    "under",
    "after",
    "before",
    "because",
    "while",
    "if",
    "else",
    "of",
    "is",
    "in",
    "on",
    "at",
    "by",
    "to",
    "as",
    "it",
    "be",
    "an",
    "a",
    # Persian stopwords (selected subset)
    "است",
    "هست",
    "هستند",
    "بود",
    "بودند",
    "شد",
    "می",
    "شود",
    "برای",
    "این",
    "آن",
    "از",
    "به",
    "در",
    "که",
    "و",
}

def split_sentences(text: str, lang: str) -> list[str]:
    """Very small helper that splits ``text`` into sentences.

    The implementation intentionally uses simple, language-aware punctuation
    heuristics so that it remains lightweight for both English and Persian
    content.  Sentences are stripped of leading/trailing whitespace and empty
    segments are discarded.
    """

    if not text:
        return []

    separators = r"[\.\!\?]"
    if lang.lower().startswith("fa"):
        separators = r"[\.\!\?\u061f\u060c\u061b]"

    parts = re.split(separators + r"+", text)
    return [segment.strip() for segment in parts if segment and segment.strip()]


def compute_coverage(evidence_map: dict) -> float:
    """Return fraction of sentences that have at least one supporting context."""

    if not evidence_map:
        return 0.0

    supported = sum(1 for entry in evidence_map.values() if entry.get("contexts"))
    return supported / len(evidence_map)


def align_sentences_to_contexts(sentences: list[str], contexts: list[dict]) -> dict:
    """Align answer ``sentences`` with ``contexts`` using embeddings and lexical overlap."""

    evidence_map: dict[int, dict] = {}

    cleaned_sentences: dict[int, str] = {}
    for idx, sentence in enumerate(sentences):
        stripped = sentence.strip()
        evidence_map[idx] = {"sentence": sentence, "contexts": []}
        if stripped:
            cleaned_sentences[idx] = stripped

    if not cleaned_sentences or not contexts:
        return evidence_map

    adapter = _get_adapter()

    sentence_indices = list(cleaned_sentences.keys())
    sentence_texts = [cleaned_sentences[idx] for idx in sentence_indices]
    sentence_vectors = adapter.embed_texts(sentence_texts) if sentence_texts else []

    sentence_embeddings = {
        idx: _normalise_vector(vector)
        for idx, vector in zip(sentence_indices, sentence_vectors)
    }

    prepared_contexts: list[dict] = []
    to_embed_texts: list[str] = []
    to_embed_indices: list[int] = []

    for ctx in contexts:
        text = ctx.get("text") or ctx.get("content") or ""
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue

        tokens = _tokenise(stripped)
        vector = ctx.get("embedding") or ctx.get("vector")

        prepared = {
            "raw": ctx,
            "text": stripped,
            "tokens": tokens,
            "embedding": None,
        }

        if isinstance(vector, Iterable):
            float_vector = _normalise_vector([float(v) for v in vector])
            prepared["embedding"] = float_vector
        else:
            to_embed_indices.append(len(prepared_contexts))
            to_embed_texts.append(stripped)

        prepared_contexts.append(prepared)

    if not prepared_contexts:
        return evidence_map

    if to_embed_texts:
        embedded = adapter.embed_texts(to_embed_texts)
        for idx, vector in zip(to_embed_indices, embedded):
            prepared_contexts[idx]["embedding"] = _normalise_vector(vector)

    for sent_idx, sentence in cleaned_sentences.items():
        sentence_vector = sentence_embeddings.get(sent_idx)
        if not sentence_vector:
            continue

        sentence_tokens = _tokenise(sentence)
        scored_contexts: list[tuple[float, dict]] = []

        for ctx in prepared_contexts:
            context_vector = ctx.get("embedding")
            if not context_vector:
                continue

            lexical_score = _overlap_score(sentence_tokens, ctx["tokens"])
            if lexical_score < 0.4:
                continue

            cosine_score = _cosine_similarity(sentence_vector, context_vector)
            combined = 0.75 * cosine_score + 0.25 * lexical_score

            if combined <= 0:
                continue

            scored_contexts.append((combined, ctx))

        if not scored_contexts:
            continue

        scored_contexts.sort(key=lambda item: item[0], reverse=True)
        threshold = 0.6
        selected = [
            {"score": score, "context": ctx["raw"]}
            for score, ctx in scored_contexts
            if score >= threshold
        ]

        if not selected and scored_contexts:
            top_score, top_ctx = scored_contexts[0]
            selected = [{"score": top_score, "context": top_ctx["raw"]}]

        evidence_map[sent_idx]["contexts"] = selected

    return evidence_map


def _overlap_score(sentence_tokens: set[str], context_tokens: set[str]) -> float:
    if not sentence_tokens:
        return 0.0
    overlap = sentence_tokens & context_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(sentence_tokens)


def _tokenise(text: str) -> set[str]:
    pattern = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
    tokens = {match.group(0).lower() for match in pattern.finditer(text)}
    filtered: set[str] = set()
    for token in tokens:
        if len(token) <= 2:
            continue
        if token in _STOPWORDS:
            continue
        filtered.add(token)
    return filtered


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(min(dot, 1.0), -1.0)


def _normalise_vector(vector: Iterable[float]) -> list[float]:
    values = list(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


@lru_cache(maxsize=1)
def _get_adapter() -> EmbeddingAdapter:
    model_name = os.getenv("EMBED_MODEL_NAME")
    try:
        return EmbeddingAdapter(model_name=model_name)
    except ImportError:
        return EmbeddingAdapter(model_name="mock")


__all__ = [
    "align_sentences_to_contexts",
    "compute_coverage",
    "split_sentences",
]

