from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from core.eval.offline_eval import main as eval_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_offline_eval_prints_metrics() -> None:
    rows = [
        {
            "question": "What color is the sky?",
            "expected_answer": "blue",
            "predicted_answer": "sky is blue",
            "book_refs": ["The sky is blue on a clear day."],
        },
        {
            "question": "What do cats do?",
            "expected_answer": "purr",
            "predicted_answer": "cats purr",
            "contexts": [{"text": "Cats often purr when content."}],
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "ds.jsonl"
        _write_jsonl(ds, rows)
        # simulate CLI invocation
        sys.argv = ["offline_eval", "--ds", str(ds)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            eval_main()
        out = buf.getvalue()
        assert "EM:" in out and "F1:" in out and "Coverage:" in out
