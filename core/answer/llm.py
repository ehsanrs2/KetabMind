"""LLM backend loader and utilities."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLM(ABC):
    """Minimal language model interface."""

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        """Return a response for the given prompt."""


class MockLLM(LLM):
    """Mock LLM that synthesizes answers from prompt contexts."""

    _BULLET_PATTERN = re.compile(r"^\d+\. \(([^)]+)\)\s+(.*)$")

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        response = self._build_response(prompt)
        if not stream:
            return response
        return self._stream_response(response)

    def _build_response(self, prompt: str) -> str:
        base = "Mock response generated for testing."
        context_lines = self._extract_context_lines(prompt)
        echoes: list[str] = []
        for idx, line in enumerate(context_lines[:2], 1):
            match = self._BULLET_PATTERN.match(line)
            if match:
                echoes.append(f"Context {idx}: ({match.group(1)}) {match.group(2)}")
            else:
                echoes.append(f"Context {idx}: {line.strip()}")
        if not echoes:
            return base
        return "\n".join([base, *echoes])

    def _extract_context_lines(self, prompt: str) -> list[str]:
        lines = prompt.splitlines()
        context_lines: list[str] = []
        in_ctx = False
        for line in lines:
            if line.strip().startswith("Contexts"):
                in_ctx = True
                continue
            if in_ctx:
                if line.strip().startswith("Answer:"):
                    break
                if line.strip():
                    context_lines.append(line)
        return context_lines

    def _stream_response(self, response: str) -> Iterator[str]:
        for segment in response.split("\n"):
            yield segment + "\n"


def get_llm() -> LLM:
    """Instantiate the configured LLM backend."""

    backend = os.getenv("LLM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend in {"ollama", "transformers"}:
        raise NotImplementedError(f"LLM backend '{backend}' is not implemented yet.")
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


def truncate_to_token_budget(text: str, tokenizer_name: str, max_input_tokens: int) -> str:
    """Truncate ``text`` so that it does not exceed ``max_input_tokens``."""

    if max_input_tokens <= 0:
        return ""
    tokenizer_hint = tokenizer_name.lower()
    pattern = r"\S+\s*"
    if "character" in tokenizer_hint:
        approx = max_input_tokens
        return text[:approx].rstrip()
    end_index: int | None = None
    for token_count, match in enumerate(re.finditer(pattern, text), start=1):
        if token_count > max_input_tokens:
            end_index = match.start()
            break
    if end_index is None:
        return text
    return text[:end_index].rstrip()


def ensure_stop_sequences(text: str, stops: list[str]) -> str:
    """Ensure ``text`` stops at the first occurrence of any sequence in ``stops``."""

    if not stops:
        return text.rstrip()
    earliest = len(text)
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            earliest = min(earliest, idx)
    if earliest < len(text):
        return text[:earliest].rstrip()
    return text.rstrip()
