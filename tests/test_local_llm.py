import json
from types import SimpleNamespace

import pytest

from backend import local_llm


class DummyResponse:
    def __init__(self, payloads, status_code=200):
        self.status_code = status_code
        self._payloads = payloads

    def iter_lines(self, decode_unicode=False):
        for payload in self._payloads:
            data = json.dumps(payload).encode()
            yield data

    def close(self):
        pass


def test_generate_prefers_ollama(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, stream=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        captured["stream"] = stream
        return DummyResponse([
            {"response": "Hello "},
            {"response": "world!"},
        ])

    monkeypatch.setattr(local_llm.requests, "post", fake_post)
    monkeypatch.setattr(local_llm, "_PIPELINE_CACHE", {})
    monkeypatch.setattr(local_llm, "_get_hf_pipeline", lambda *_, **__: (_ for _ in ()).throw(AssertionError("HF not expected")))

    result = local_llm.generate("test prompt", model="custom-model")

    assert result.startswith("Hello world!")
    assert "[book_id:" in result, "Citation marker should be appended"
    assert captured["json"]["model"] == "custom-model"
    assert captured["stream"] is True


def test_generate_falls_back_to_hf(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MAX_PROMPT_CHARS", "10")
    monkeypatch.setenv("LOCAL_LLM_QUANT", "4bit")
    monkeypatch.setenv("LOCAL_LLM_MAX_NEW_TOKENS", "5")

    def raise_post(*args, **kwargs):
        raise local_llm.RequestException("boom")

    monkeypatch.setattr(local_llm.requests, "post", raise_post)
    monkeypatch.setattr(local_llm, "_PIPELINE_CACHE", {})

    captured = {}

    class DummyTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            captured["tokenizer"] = (name, kwargs)
            return "tokenizer"

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            captured.setdefault("model_calls", []).append((name, kwargs))
            return SimpleNamespace()

    class DummyPipeline:
        def __call__(self, prompt, max_new_tokens=None, temperature=None):
            captured["prompt"] = prompt
            captured["max_new_tokens"] = max_new_tokens
            captured["temperature"] = temperature
            return [{"generated_text": prompt + " answer"}]

    def fake_pipeline(task, model=None, tokenizer=None, device=None):
        captured["pipeline"] = (task, model, tokenizer, device)
        return DummyPipeline()

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    monkeypatch.setattr(local_llm, "torch", fake_torch)
    monkeypatch.setattr(local_llm, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(local_llm, "AutoModelForCausalLM", DummyAutoModel)
    monkeypatch.setattr(local_llm, "pipeline", fake_pipeline)

    result = local_llm.generate("irrelevant and lengthy prompt", model="ignored")

    # Prompt is trimmed to 10 characters (the last 10)
    assert captured["prompt"] == "thy prompt", "Prompt should be trimmed to the configured character budget"
    assert captured["max_new_tokens"] == 5
    assert "[book_id:" in result

    # Ensure quantisation kwargs were used.
    _, kwargs = captured["model_calls"][0]
    assert "load_in_4bit" in kwargs


def test_generate_raises_on_empty_prompt():
    with pytest.raises(ValueError):
        local_llm.generate(" ")

