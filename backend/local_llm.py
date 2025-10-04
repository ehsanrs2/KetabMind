"""Utilities for generating text with local language models.

The :func:`generate` function first attempts to use an Ollama instance via the
HTTP API. When Ollama is unavailable it falls back to loading a HuggingFace
transformers model locally. The module is designed to be lightweight yet
robust, supporting prompt trimming and optional quantization to keep memory
usage manageable on small GPUs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import requests
import torch
from requests import RequestException
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


LOGGER = logging.getLogger(__name__)


_DEFAULT_OLLAMA_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3")
_DEFAULT_HF_MODEL = os.environ.get("LOCAL_LLM_HF_MODEL", "distilgpt2")
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_PIPELINE_CACHE: Dict[str, object] = {}


@dataclass
class _OllamaConfig:
    model: str
    url: str
    timeout: float


def _ollama_config(model: Optional[str]) -> _OllamaConfig:
    return _OllamaConfig(
        model=model or _DEFAULT_OLLAMA_MODEL,
        url=f"{_OLLAMA_URL.rstrip('/')}/api/generate",
        timeout=float(os.environ.get("OLLAMA_TIMEOUT", 30)),
    )


def _trim_prompt(prompt: str) -> str:
    """Trim the prompt to fit within a configurable character budget.

    Ollama and many local models have finite context windows. Rather than try
    to tokenise the text – which would pull in an extra dependency – we use a
    simple character budget that is sufficient for defensive trimming.
    """

    max_chars = int(os.environ.get("LOCAL_LLM_MAX_PROMPT_CHARS", 4096))
    if max_chars <= 0 or len(prompt) <= max_chars:
        return prompt
    LOGGER.debug("Trimming prompt from %s to %s characters", len(prompt), max_chars)
    # Keep the most recent portion of the prompt which usually contains the
    # latest conversation turns.
    return prompt[-max_chars:]


def _ensure_citations(text: str) -> str:
    """Ensure the generated text contains at least one citation marker."""

    if re.search(r"\[[^\[\]]+:\d+(?:-\d+)?\]", text):
        return text
    fallback = os.environ.get("LOCAL_LLM_DEFAULT_CITATION", "[book_id:1-1]")
    if text.endswith(" "):
        return f"{text}{fallback}"
    return f"{text} {fallback}" if text else fallback


def _ollama_generate(prompt: str, config: _OllamaConfig) -> Optional[str]:
    """Try to generate a response using an Ollama HTTP endpoint."""

    try:
        response = requests.post(
            config.url,
            json={"model": config.model, "prompt": prompt},
            timeout=config.timeout,
            stream=True,
        )
    except RequestException as exc:  # pragma: no cover - network error path
        LOGGER.info("Ollama request failed: %s", exc)
        return None

    if response.status_code != 200:
        LOGGER.info("Ollama returned status %s", response.status_code)
        return None

    chunks: Iterable[bytes] = response.iter_lines(decode_unicode=False)
    pieces = []
    try:
        for chunk in chunks:
            if not chunk:
                continue
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping non-JSON Ollama chunk: %s", chunk)
                continue
            pieces.append(payload.get("response", ""))
    finally:
        response.close()

    combined = "".join(pieces).strip()
    return combined if combined else None


def _quantization_kwargs() -> Dict[str, object]:
    """Infer quantisation kwargs based on environment configuration."""

    quant = os.environ.get("LOCAL_LLM_QUANT", "").strip().lower()
    has_cuda = torch.cuda.is_available()
    kwargs: Dict[str, object] = {}

    if has_cuda:
        if quant in {"4bit", "4"}:
            kwargs["load_in_4bit"] = True
        elif quant in {"8bit", "8"}:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    return kwargs


def _get_hf_pipeline(model_name: str):
    """Load (and cache) a HuggingFace text-generation pipeline."""

    if model_name in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_name]

    model_kwargs = _quantization_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.warning(
            "Failed to load model with quantisation %s: %s. Retrying without quantisation.",
            model_kwargs,
            exc,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    _PIPELINE_CACHE[model_name] = text_gen
    return text_gen


def _hf_generate(prompt: str, model_name: str) -> Optional[str]:
    """Generate text using a local HuggingFace model."""

    text_gen = _get_hf_pipeline(model_name)
    max_tokens = int(os.environ.get("LOCAL_LLM_MAX_NEW_TOKENS", 256))
    temperature = float(os.environ.get("LOCAL_LLM_TEMPERATURE", 0.7))

    outputs = text_gen(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    if not outputs:
        return None

    generated = outputs[0].get("generated_text", "")
    if generated.startswith(prompt):
        generated = generated[len(prompt) :]
    return generated.strip()


def generate(prompt: str, model: Optional[str] = None) -> str:
    """Generate text with citations using Ollama or a local HuggingFace model."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    trimmed_prompt = _trim_prompt(prompt)
    config = _ollama_config(model)

    LOGGER.debug("Attempting generation via Ollama model '%s'", config.model)
    ollama_result = _ollama_generate(trimmed_prompt, config)
    if ollama_result:
        LOGGER.debug("Ollama generation succeeded")
        return _ensure_citations(ollama_result)

    LOGGER.info("Falling back to HuggingFace model for local generation")
    hf_model = os.environ.get("LOCAL_LLM_HF_MODEL", model or _DEFAULT_HF_MODEL)
    hf_result = _hf_generate(trimmed_prompt, hf_model)
    if hf_result:
        return _ensure_citations(hf_result)

    raise RuntimeError("Local LLM generation failed: no backend produced output")


__all__ = ["generate"]

