import importlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def reload_adapter(monkeypatch):
    module_name = "embedding.adapter"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_embed_texts_with_sentence_transformer(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_NAME", "bge-m3")
    module = reload_adapter(monkeypatch)

    class DummySentenceTransformer:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(
            self,
            texts,
            batch_size=None,
            convert_to_numpy=True,
            device=None,
            show_progress_bar=False,
        ):
            dim = 4
            return torch.ones((len(texts), dim)).numpy()

    monkeypatch.setattr(module, "SentenceTransformer", DummySentenceTransformer)

    adapter = module.EmbeddingAdapter()
    embeddings = adapter.embed_texts(["hello", "world"], batch_size=2)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 4


def test_embed_texts_with_automodel(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
    module = reload_adapter(monkeypatch)

    monkeypatch.setattr(module, "SentenceTransformer", None)

    class DummyAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
            batch_size = len(texts)
            seq_len = 5
            return {
                "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }

    class DummyAutoModel(torch.nn.Module):
        hidden_size = 6

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, input_ids=None, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            data = torch.arange(batch_size * seq_len * self.hidden_size, dtype=torch.float32)
            data = data.reshape(batch_size, seq_len, self.hidden_size)
            return SimpleNamespace(last_hidden_state=data)

    monkeypatch.setattr(module, "AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr(module, "AutoModel", DummyAutoModel)

    adapter = module.EmbeddingAdapter(device="cpu")
    embeddings = adapter.embed_texts(["foo", "bar"], batch_size=2)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == DummyAutoModel.hidden_size


def test_quantized_model_uses_bitsandbytes(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_NAME", "bge-m3")
    monkeypatch.setenv("EMBED_QUANT", "8bit")
    module = reload_adapter(monkeypatch)

    monkeypatch.setattr(module, "SentenceTransformer", None)
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(__version__="0.41"))

    captured_kwargs = {}

    class DummyAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

    class DummyAutoModel(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured_kwargs.update(kwargs)
            return cls()

        def eval(self):
            return self

    monkeypatch.setattr(module, "AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr(module, "AutoModel", DummyAutoModel)

    adapter = module.EmbeddingAdapter(device="cuda")
    assert adapter.quantization_mode == "8bit"
    assert captured_kwargs.get("device_map") == "auto"

    quant_config = captured_kwargs.get("quantization_config")
    assert quant_config is not None
    assert getattr(quant_config, "load_in_8bit", False)


def test_invalid_quantization_value(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_NAME", "bge-m3")
    monkeypatch.setenv("EMBED_QUANT", "2bit")
    module = reload_adapter(monkeypatch)

    with pytest.raises(ValueError):
        module.EmbeddingAdapter()
