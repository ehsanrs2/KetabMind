"""Answer generation using retrieved contexts."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import structlog
from answering.citations import build_citations
from answering.guardrails import refusal_message, should_refuse
from core import config
from core.retrieve.retriever import Retriever, ScoredChunk
from core.self_rag.validator import citation_coverage

from .llm import LLM, ensure_stop_sequences, get_llm, truncate_to_token_budget
from .template import (
    build_prompt,
    estimate_token_count,
    select_contexts_within_budget,
)

_retriever = Retriever()
log = structlog.get_logger(__name__)

DEFAULT_SYSTEM_INSTRUCTIONS = "You are KetabMind, a helpful research assistant."
MODEL_TOKENIZER_OVERRIDES: dict[str, str] = {
    "mock": "mock-character-tokenizer",
}
STOP_SEQUENCES: list[str] = []
DEFAULT_REFUSAL_RULE = "balanced"
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass(slots=True)
class LLMConfig:
    backend: str
    model: str
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    tokenizer_name: str
    system_instructions: str
    stop_sequences: list[str]


def _get_env_int(name: str, default: int) -> int:
    raw = _get_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _get_env_float(name: str, default: float) -> float:
    raw = _get_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be a float") from exc


def _resolve_tokenizer_name(model: str) -> str:
    return MODEL_TOKENIZER_OVERRIDES.get(model, model)


def _load_config() -> LLMConfig:
    settings = config.settings
    backend = (_get_env("LLM_BACKEND", settings.llm_backend) or settings.llm_backend).lower()
    model = _get_env("LLM_MODEL", settings.llm_model) or settings.llm_model
    max_input_tokens = _get_env_int("LLM_MAX_INPUT_TOKENS", settings.llm_max_input_tokens)
    max_new_tokens = _get_env_int("LLM_MAX_NEW_TOKENS", settings.llm_max_new_tokens)
    temperature = _get_env_float("LLM_TEMPERATURE", settings.llm_temperature)
    top_p = _get_env_float("LLM_TOP_P", settings.llm_top_p)
    system_instructions = DEFAULT_SYSTEM_INSTRUCTIONS
    tokenizer_name = _resolve_tokenizer_name(model)
    return LLMConfig(
        backend=backend,
        model=model,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tokenizer_name=tokenizer_name,
        system_instructions=system_instructions,
        stop_sequences=list(STOP_SEQUENCES),
    )


def _get_env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is not None:
        return value
    return default


def _prepare_prompt(question: str, contexts: list[ScoredChunk], config: LLMConfig) -> str:
    prompt = build_prompt(question, contexts, config.system_instructions)
    return truncate_to_token_budget(prompt, config.tokenizer_name, config.max_input_tokens)


def _consume_response(result: str | Iterator[str]) -> str:
    if isinstance(result, str):
        return result
    return "".join(result)


def _serialize_context(chunk: ScoredChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "book_id": chunk.book_id,
        "page": chunk.page,
        "page_start": getattr(chunk, "page_start", chunk.page),
        "page_end": getattr(chunk, "page_end", chunk.page),
        "snippet": chunk.snippet,
        "text": chunk.text,
        "cosine": chunk.cosine,
        "lexical": chunk.lexical,
        "reranker": chunk.reranker,
        "hybrid": chunk.hybrid,
        "metadata": dict(chunk.metadata),
    }


def _average_hybrid(contexts: list[ScoredChunk]) -> float:
    if not contexts:
        return 0.0
    return sum(chunk.hybrid for chunk in contexts) / len(contexts)


def _detect_language(question: str) -> str:
    return "fa" if _ARABIC_SCRIPT_RE.search(question) else "en"


def _refusal_rule() -> str:
    return (_get_env("ANSWER_REFUSAL_RULE") or DEFAULT_REFUSAL_RULE).strip()


def answer(query: str, top_k: int = 8, *, book_id: str | None = None) -> dict[str, Any]:
    config = _load_config()
    contexts = _retriever.retrieve(query, top_k=top_k, book_id=book_id)
    log.info(
        "retrieval.success",
        book_id=book_id,
        total_contexts=len(contexts),
        top_snippets=[ctx.text[:100] for ctx in contexts[:3]],
    )
    if not any(c.text.strip() for c in contexts):
        follow_up = f"{query} explain further"
        contexts = _retriever.retrieve(follow_up, top_k=top_k, book_id=book_id)
        log.info(
            "retrieval.follow_up",
            book_id=book_id,
            original_query=query,
            follow_up=follow_up,
            total_contexts=len(contexts),
            top_snippets=[ctx.text[:100] for ctx in contexts[:3]],
        )
    selected_contexts = select_contexts_within_budget(
        query,
        contexts,
        config.tokenizer_name,
        config.max_input_tokens,
    )
    log.info(
        "contexts.selected",
        book_id=book_id,
        used_contexts=len(selected_contexts),
        used_snippets=[ctx.text[:100] for ctx in selected_contexts],
    )
    serialized_contexts = [_serialize_context(chunk) for chunk in selected_contexts]
    prompt = _prepare_prompt(query, selected_contexts, config)
    log.debug("prompt.built", book_id=book_id, prompt=prompt)
    llm: LLM = get_llm()
    raw_answer = llm.generate(prompt, stream=False)
    answer_text = ensure_stop_sequences(_consume_response(raw_answer), config.stop_sequences)
    language = _detect_language(query)
    coverage = citation_coverage(answer_text, selected_contexts)
    confidence = _average_hybrid(selected_contexts)
    rule = _refusal_rule()
    refused = should_refuse(coverage, confidence, rule)
    if refused:
        citations = build_citations(serialized_contexts, "[${book_id}:${page_start}-${page_end}]")
        answer_text = refusal_message(language, citations)
    log.info(
        "answer.generated",
        book_id=book_id,
        answer=answer_text,
        coverage=f"{coverage:.2f}",
        confidence=f"{confidence:.2f}",
        citations_in_answer=bool(re.search(r"\[.+?:\d", answer_text)),
    )
    est_input_tokens = estimate_token_count(
        [query, *(ctx.text for ctx in selected_contexts)],
        config.tokenizer_name,
    )
    return {
        "answer": answer_text,
        "contexts": serialized_contexts,
        "debug": {
            "prompt": prompt,
            "backend": config.backend,
            "model": config.model,
            "tokenizer": config.tokenizer_name,
            "max_input_tokens": config.max_input_tokens,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stats": {
                "total_contexts": len(contexts),
                "used_contexts": len(selected_contexts),
                "est_input_tokens": est_input_tokens,
                "coverage": coverage,
                "confidence": confidence,
            },
            "guardrails": {
                "rule": rule,
                "refused": refused,
            },
        },
    }


def stream_answer(
    query: str,
    top_k: int = 8,
    *,
    book_id: str | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield incremental answer tokens followed by final result."""

    config = _load_config()
    contexts = _retriever.retrieve(query, top_k=top_k, book_id=book_id)
    log.info(
        "retrieval.success",
        book_id=book_id,
        total_contexts=len(contexts),
        top_snippets=[ctx.text[:100] for ctx in contexts[:3]],
    )
    if not any(c.text.strip() for c in contexts):
        follow_up = f"{query} explain further"
        contexts = _retriever.retrieve(follow_up, top_k=top_k, book_id=book_id)
        log.info(
            "retrieval.follow_up",
            book_id=book_id,
            original_query=query,
            follow_up=follow_up,
            total_contexts=len(contexts),
            top_snippets=[ctx.text[:100] for ctx in contexts[:3]],
        )
    selected_contexts = select_contexts_within_budget(
        query,
        contexts,
        config.tokenizer_name,
        config.max_input_tokens,
    )
    log.info(
        "contexts.selected",
        book_id=book_id,
        used_contexts=len(selected_contexts),
        used_snippets=[ctx.text[:100] for ctx in selected_contexts],
    )
    serialized_contexts = [_serialize_context(chunk) for chunk in selected_contexts]
    prompt = _prepare_prompt(query, selected_contexts, config)
    log.debug("prompt.built", book_id=book_id, prompt=prompt)
    llm: LLM = get_llm()
    stream_result = llm.generate(prompt, stream=True)
    language = _detect_language(query)

    def _final_payload(answer_text: str) -> dict[str, Any]:
        final_answer = ensure_stop_sequences(answer_text, config.stop_sequences)
        coverage = citation_coverage(final_answer, selected_contexts)
        confidence = _average_hybrid(selected_contexts)
        est_input_tokens = estimate_token_count(
            [query, *(ctx.text for ctx in selected_contexts)],
            config.tokenizer_name,
        )
        debug_payload = {
            "prompt": prompt,
            "backend": config.backend,
            "model": config.model,
            "tokenizer": config.tokenizer_name,
            "max_input_tokens": config.max_input_tokens,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stats": {
                "total_contexts": len(contexts),
                "used_contexts": len(selected_contexts),
                "est_input_tokens": est_input_tokens,
                "coverage": coverage,
                "confidence": confidence,
            },
        }
        log.info(
            "answer.generated",
            book_id=book_id,
            answer=final_answer,
            coverage=f"{coverage:.2f}",
            confidence=f"{confidence:.2f}",
            citations_in_answer=bool(re.search(r"\[.+?:\d", final_answer)),
        )
        return {
            "answer": final_answer,
            "contexts": serialized_contexts,
            "meta": {
                "lang": language,
                "coverage": coverage,
                "confidence": confidence,
            },
            "debug": debug_payload,
        }

    if isinstance(stream_result, str):
        buffered = ensure_stop_sequences(stream_result, config.stop_sequences)
        if buffered:
            yield {"delta": buffered}
        yield _final_payload(buffered)
        return
    collected: list[str] = []
    for delta in stream_result:
        collected.append(delta)
        yield {"delta": delta}
    final_answer = "".join(collected)
    yield _final_payload(final_answer)
