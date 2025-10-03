from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from click.testing import CliRunner

if "typer" not in sys.modules:

    class _TyperStub:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - shim
            pass

        def command(self):  # pragma: no cover - shim
            def decorator(func):
                return func

            return decorator

    def _option_stub(*args, **kwargs):  # pragma: no cover - shim
        return kwargs.get("default")

    sys.modules["typer"] = SimpleNamespace(Typer=_TyperStub, Option=_option_stub)

if "pypdf" not in sys.modules:

    class _PdfReaderStub:  # pragma: no cover - shim
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PdfReader stub should not be used in CLI tests")

    sys.modules["pypdf"] = SimpleNamespace(PdfReader=_PdfReaderStub)

if "structlog" not in sys.modules:

    class _LoggerStub:  # pragma: no cover - shim
        def info(self, *args, **kwargs) -> None:
            return None

        def debug(self, *args, **kwargs) -> None:
            return None

    sys.modules["structlog"] = SimpleNamespace(get_logger=lambda *args, **kwargs: _LoggerStub())

if "fitz" not in sys.modules:

    class _FitzStub(SimpleNamespace):  # pragma: no cover - shim
        def open(self, *args, **kwargs):
            raise RuntimeError("fitz stub should not be invoked in CLI tests")

    sys.modules["fitz"] = _FitzStub()

from cli.ingest_cli import run
from core.ingest.pdf_to_text import Page
from nlp.fa_normalize import normalize_fa


def test_ingest_cli_outputs_schema(tmp_path, monkeypatch):
    runner = CliRunner()
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"pdf")
    output_path = tmp_path / "out.jsonl"

    raw_text = "مي ك\u00a0"

    def fake_load_pages(_path):
        return [Page(page_num=1, text=raw_text, section="intro")]

    monkeypatch.setattr("ingest.pipeline._load_pages", fake_load_pages)

    meta = {"author": "Tester", "year": 2024}
    result = runner.invoke(
        run,
        [
            str(input_path),
            "--output",
            str(output_path),
            "--with-meta",
            json.dumps(meta),
        ],
    )

    assert result.exit_code == 0, result.stdout
    summary = json.loads(result.stdout.strip())

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines, "Expected at least one JSONL record"
    record = json.loads(lines[0])

    assert record["book_id"]
    assert record["version"]
    assert record["file_hash"] == summary["file_hash"]
    assert record["book_id"] == summary["book_id"]
    assert record["version"] == summary["version"]
    assert record["meta"] == meta
    assert record["text"] == normalize_fa(raw_text)
    assert "page_num" in record and record["page_num"] == 1
    assert "section" in record
    assert summary["records"] == len(lines)
