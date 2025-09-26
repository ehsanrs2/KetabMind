"""Answer generation using retrieved contexts."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from core.retrieve.retriever import Retriever, ScoredChunk

from .llm import LLM, ensure_stop_sequences, get_llm, truncate_to_token_budget
from .template import build_prompt

_retriever = Retriever()

DEFAULT_SYSTEM_INSTRUCTIONS = "You are KetabMind, a helpful research assistant."
MODEL_TOKENIZER_OVERRIDES: dict[str, str] = {
    "mock": "mock-character-tokenizer",
}
STOP_SEQUENCES: list[str] = []


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
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be a float") from exc


def _resolve_tokenizer_name(model: str) -> str:
    return MODEL_TOKENIZER_OVERRIDES.get(model, model)


def _load_config() -> LLMConfig:
    backend = os.getenv("LLM_BACKEND", "mock").lower()
    model = os.getenv("LLM_MODEL", "mock")
    max_input_tokens = _get_env_int("LLM_MAX_INPUT_TOKENS", 4096)
    max_new_tokens = _get_env_int("LLM_MAX_NEW_TOKENS", 256)
    temperature = _get_env_float("LLM_TEMPERATURE", 0.2)
    top_p = _get_env_float("LLM_TOP_P", 0.95)
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


def _prepare_prompt(question: str, contexts: list[ScoredChunk], config: LLMConfig) -> str:
    prompt = build_prompt(question, contexts, config.system_instructions)
    return truncate_to_token_budget(prompt, config.tokenizer_name, config.max_input_tokens)


def _consume_response(result: str | Iterator[str]) -> str:
    if isinstance(result, str):
        return result
    return "".join(result)


def answer(query: str, top_k: int = 8) -> dict[str, Any]:
    config = _load_config()
    contexts = _retriever.retrieve(query, top_k)
    prompt = _prepare_prompt(query, contexts, config)
    llm: LLM = get_llm()
    raw_answer = llm.generate(prompt, stream=False)
    answer_text = ensure_stop_sequences(_consume_response(raw_answer), config.stop_sequences)
    return {
        "answer": answer_text,
        "contexts": [asdict(c) for c in contexts],
        "debug": {
            "prompt": prompt,
            "backend": config.backend,
            "model": config.model,
            "tokenizer": config.tokenizer_name,
            "max_input_tokens": config.max_input_tokens,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
    }


def stream_answer(query: str, top_k: int = 8) -> Iterable[dict[str, Any]]:
    """Yield incremental answer tokens followed by final result."""

    config = _load_config()
    contexts = _retriever.retrieve(query, top_k)
    prompt = _prepare_prompt(query, contexts, config)
    llm: LLM = get_llm()
    stream_result = llm.generate(prompt, stream=True)
    if isinstance(stream_result, str):
        buffered = ensure_stop_sequences(stream_result, config.stop_sequences)
        if buffered:
            yield {"delta": buffered}
        yield {"answer": buffered, "contexts": [asdict(c) for c in contexts]}
        return
    collected: list[str] = []
    for delta in stream_result:
        collected.append(delta)
        yield {"delta": delta}
    final_answer = ensure_stop_sequences("".join(collected), config.stop_sequences)
    yield {"answer": final_answer, "contexts": [asdict(c) for c in contexts]}
