"""Offline evaluation for KetabMind.

Reads a JSONL dataset with fields:
- question: str
- expected_answer: str
- predicted_answer: str (optional if precomputed)
- book_refs | contexts: list[str] or list[{text: str}] (optional)

Computes EM, F1, and citation coverage over the dataset and prints metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalize(s: str) -> list[str]:
    import re

    s = s.lower().strip()
    return re.findall(r"[a-z0-9]+", s)


def _em(pred: str, gold: str) -> float:
    return 1.0 if " ".join(_normalize(pred)) == " ".join(_normalize(gold)) else 0.0


def _f1(pred: str, gold: str) -> float:
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
    prec = inter / max(1, len(ps))
    rec = inter / max(1, len(gs))
    return 2 * prec * rec / max(1e-9, (prec + rec))


def _sentence_cov(answer: str, contexts: list[str]) -> float:
    import re

    def sent_split(t: str) -> list[str]:
        return [x.strip() for x in re.split(r"[.!?]+", t) if x.strip()]

    def toks(t: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", t.lower()))

    sents = sent_split(answer)
    if not sents:
        return 0.0
    ctx_tok = [toks(c) for c in contexts]
    supported = 0
    for s in sents:
        st = toks(s)
        best = 0.0
        for ct in ctx_tok:
            if not ct:
                continue
            num = len(st & ct)
            den = len(st | ct) or 1
            best = max(best, num / den)
        if best >= 0.2:
            supported += 1
    return supported / max(1, len(sents))


@dataclass
class EvalResult:
    em: float
    f1: float
    coverage: float


def evaluate_path(path: Path) -> EvalResult:
    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    cov_sum = 0.0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gold = str(obj.get("expected_answer", ""))
            pred = str(obj.get("predicted_answer", ""))
            # contexts can be list[str] or list[dict[text]] in either key
            raw_ctx = obj.get("book_refs") or obj.get("contexts") or []
            ctx_texts: list[str] = []
            for c in raw_ctx:
                if isinstance(c, str):
                    ctx_texts.append(c)
                elif isinstance(c, dict) and "text" in c:
                    ctx_texts.append(str(c["text"]))
            em_sum += _em(pred, gold)
            f1_sum += _f1(pred, gold)
            cov_sum += _sentence_cov(pred, ctx_texts)
            total += 1
    if total == 0:
        return EvalResult(0.0, 0.0, 0.0)
    return EvalResult(em=em_sum / total, f1=f1_sum / total, coverage=cov_sum / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", required=True, help="Path to JSONL dataset")
    args = parser.parse_args()
    res = evaluate_path(Path(args.ds))
    print(f"EM: {res.em:.4f}")
    print(f"F1: {res.f1:.4f}")
    print(f"Coverage: {res.coverage:.4f}")


if __name__ == "__main__":
    main()

