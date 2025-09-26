from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.eval import offline_eval


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_offline_eval_metrics(tmp_path: Path) -> None:
    ds = tmp_path / "eval.jsonl"
    rows = [
        {
            "q": "Who wrote Pride and Prejudice?",
            "gold": "Jane Austen",
            "pred": "Jane Austen",
            "refs": [{"book_id": "b1", "page_start": 1, "page_end": 2}],
        }
    ]
    _write_jsonl(ds, rows)
    res = offline_eval.evaluate_path(ds)
    metrics = res.__dict__
    assert set(metrics) == {"em", "f1", "coverage"}
    for m in metrics.values():
        assert isinstance(m, float)
        assert 0.0 <= m <= 1.0
