"""LLM backend loader."""

from __future__ import annotations

import os
import re
from typing import Protocol


class LLM(Protocol):
    """Minimal language model interface."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        """Return a response for the given prompt."""


class MockLLM:
    """Mock LLM that synthesizes answers from prompt contexts."""

    def generate(self, prompt: str) -> str:
        lines = prompt.splitlines()
        context_lines: list[str] = []
        in_ctx = False
        for line in lines:
            if line.startswith("Contexts"):
                in_ctx = True
                continue
            if in_ctx:
                if line.strip().startswith("Answer:"):
                    break
                context_lines.append(line)
        text = ""
        if context_lines:
            first = context_lines[0]
            match = re.match(r"\d+\. \([^)]*\) (.*)", first)
            text = match.group(1) if match else first
        if not text:
            text = prompt
        sentences = re.split(r"(?<=[.!?]) +", text.strip())
        return " ".join(sentences[:4])


def get_llm() -> LLM:
    """Instantiate the configured LLM backend."""

    backend = os.getenv("LLM_BACKEND", "mock")
    if backend == "mock":
        return MockLLM()
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")
