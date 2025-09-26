"""Offline evaluation utilities.

CLI usage:

```bash
python -m core.eval.offline_eval --ds data/eval.jsonl
```

The dataset is JSON Lines with records containing:

```
{"q": str, "gold": str, "pred": str (optional),
 "refs": [{"book_id": str, "page_start": int, "page_end": int}]}
```

Metrics:
- EM: exact match between `pred` and `gold`
- F1: token-level F1 between `pred` and `gold`
- Coverage: fraction of sentences in `pred` with at least one citation
  (approximated by `len(refs)`)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import typer


def _normalize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization."""

    return re.findall(r"[a-z0-9]+", text.lower())


def _em(pred: str, gold: str) -> float:
    """Exact match score."""

    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _f1(pred: str, gold: str) -> float:
    """Token-level F1."""

    p = _normalize(pred)
    g = _normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    ps, gs = set(p), set(g)
    inter = len(ps & gs)
    if inter == 0:
        return 0.0
    prec = inter / len(ps)
    rec = inter / len(gs)
    return 2 * prec * rec / (prec + rec)


_SENT_SPLIT = re.compile(r"[.!?]+")


def _coverage(pred: str, refs: list[dict[str, Any]]) -> float:
    """Approximate citation coverage for `pred` using citation count."""

    sentences = [s.strip() for s in _SENT_SPLIT.split(pred) if s.strip()]
    if not sentences:
        return 0.0
    return min(len(refs), len(sentences)) / len(sentences)


@dataclass
class EvalResult:
    """Evaluation metrics."""

    em: float
    f1: float
    coverage: float


def evaluate_path(path: Path) -> EvalResult:
    """Evaluate dataset at `path`. Return averaged metrics."""

    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    cov_sum = 0.0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            gold = str(obj.get("gold", ""))
            pred = str(obj.get("pred", gold))
            refs = obj.get("refs", [])
            em_sum += _em(pred, gold)
            f1_sum += _f1(pred, gold)
            cov_sum += _coverage(pred, refs)
            total += 1
    if total == 0:
        return EvalResult(0.0, 0.0, 0.0)
    return EvalResult(em=em_sum / total, f1=f1_sum / total, coverage=cov_sum / total)


app = typer.Typer()

F = TypeVar("F", bound=Callable[..., Any])
command = cast(Callable[[F], F], app.command())


@command
def main(
    ds: Path = typer.Option(..., "--ds", help="Path to JSONL dataset"),  # noqa: B008
) -> None:
    """Evaluate dataset located at `ds` and print metrics."""

    res = evaluate_path(ds)
    typer.echo(f"EM: {res.em:.4f}")
    typer.echo(f"F1: {res.f1:.4f}")
    typer.echo(f"Coverage: {res.coverage:.4f}")


if __name__ == "__main__":  # pragma: no cover
    app()
