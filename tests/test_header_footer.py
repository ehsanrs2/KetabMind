from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

if "typer" not in sys.modules:

    class _TyperStub:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple shim
            pass

        def command(self):  # pragma: no cover - simple shim
            def decorator(func):
                return func

            return decorator

    def _option_stub(*args, **kwargs):  # pragma: no cover - simple shim
        return kwargs.get("default")

    sys.modules["typer"] = SimpleNamespace(Typer=_TyperStub, Option=_option_stub)


if "pypdf" not in sys.modules:

    class _PdfReaderStub:  # pragma: no cover - simple shim
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PdfReader stub should not be instantiated in tests")

    sys.modules["pypdf"] = SimpleNamespace(PdfReader=_PdfReaderStub)


PY_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None


from ingest.clean_rules import apply_rules, load_rules
from ingest.pipeline import pages_to_records


@dataclass
class DummyPage:
    page_num: int
    text: str
    section: str | None = None


def test_apply_rules_removes_header_and_footer():
    rules = {
        "defaults": {"strip_surrounding_whitespace": True, "max_matches": 2},
        "header": {
            "regex": [{"pattern": r"^Sample Header"}],
            "zones": {"top": 2},
        },
        "footer": {
            "regex": [{"pattern": r"^Page \d+$"}],
            "zones": {"bottom": 1},
        },
    }
    text = "Sample Header\nChapter 1\nMain content\nPage 1\n"
    cleaned = apply_rules(text, rules)
    assert cleaned == "Chapter 1\nMain content"


def test_apply_rules_zone_only_removal():
    rules = {"header": {"zones": {"top": {"remove": 1}}}}
    text = "Header\nLine A\nLine B"
    cleaned = apply_rules(text, rules)
    assert cleaned == "Line A\nLine B"


def test_pages_to_records_uses_default_rules(monkeypatch):
    fake_rules = {"header": {"regex": [{"pattern": "^HEADER"}], "zones": {"top": 1}}}
    monkeypatch.setattr("ingest.pipeline._DEFAULT_RULES", fake_rules, raising=False)

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_apply(text: str, cfg):
        calls.append((text, cfg))
        return text.lower()

    monkeypatch.setattr("ingest.pipeline.apply_rules", fake_apply)

    page = DummyPage(page_num=1, text="HEADER\nBody")
    records = pages_to_records([page], book_id="b", version="v", file_hash="hash")

    assert records[0]["text"] == "header\nbody"
    assert calls and calls[0][0] == "HEADER\nBody"
    assert calls[0][1] is fake_rules


@pytest.mark.skipif(not PY_YAML_AVAILABLE, reason="PyYAML not installed")
def test_load_rules_reads_default_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "rules.yaml"
    cfg_path.write_text("header: {zones: {top: 1}}\n", encoding="utf-8")
    monkeypatch.setattr("ingest.clean_rules._DEFAULT_RULES_PATH", cfg_path)

    rules = load_rules()
    assert "header" in rules
    assert rules["header"]["zones"]["top"] == 1
