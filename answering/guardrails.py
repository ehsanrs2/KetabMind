"""Guardrails for controlling answer refusals."""

from __future__ import annotations

import math
import re
from typing import Iterable

_THRESHOLDS: dict[str, tuple[float, float]] = {
    "strict": (0.75, 0.6),
    "balanced": (0.55, 0.4),
    "lenient": (0.35, 0.25),
}

_CUSTOM_RULE_RE = re.compile(r"(coverage|confidence)\s*<\s*([0-9]*\.?[0-9]+)")


def _clamp_metric(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _normalise_rule(rule: str | None) -> str:
    if not rule:
        return "balanced"
    return rule.strip().lower()


def _parse_custom_thresholds(rule: str) -> tuple[float, float] | None:
    coverage_threshold: float | None = None
    confidence_threshold: float | None = None

    for metric, value in _CUSTOM_RULE_RE.findall(rule):
        threshold = _clamp_metric(float(value))
        if metric == "coverage":
            coverage_threshold = threshold
        else:
            confidence_threshold = threshold

    if coverage_threshold is None and confidence_threshold is None:
        return None

    default_coverage, default_confidence = _THRESHOLDS["balanced"]
    return (
        coverage_threshold if coverage_threshold is not None else default_coverage,
        confidence_threshold if confidence_threshold is not None else default_confidence,
    )


def _resolve_thresholds(rule: str) -> tuple[float, float]:
    normalised = _normalise_rule(rule)
    if normalised in {"always", "refuse", "deny"}:
        return (1.0, 1.0)
    if normalised in {"never", "allow", "answer"}:
        return (-1.0, -1.0)

    key = normalised
    if key not in _THRESHOLDS:
        key = key.replace("-", "_")
    if key not in _THRESHOLDS:
        custom = _parse_custom_thresholds(normalised)
        if custom is not None:
            return custom
        key = "balanced"

    return _THRESHOLDS[key]


def should_refuse(coverage: float, confidence: float, rule: str) -> bool:
    """Return ``True`` when the answer should be refused under ``rule``.

    The ``rule`` parameter accepts the presets ``strict``, ``balanced`` and
    ``lenient`` as well as the control strings ``always`` and ``never``.
    Additionally, custom rules such as ``"coverage<0.6,confidence<0.3"`` can be
    supplied to fine-tune the guardrail thresholds.
    """

    thresholds = _resolve_thresholds(rule)
    if thresholds == (1.0, 1.0):
        return True
    if thresholds == (-1.0, -1.0):
        return False

    coverage_value = _clamp_metric(coverage)
    confidence_value = _clamp_metric(confidence)
    coverage_threshold, confidence_threshold = thresholds

    return coverage_value < coverage_threshold or confidence_value < confidence_threshold


def refusal_message(lang: str, citations: Iterable[str]) -> str:
    """Build a short refusal message in ``lang`` including ``citations``.

    ``lang`` is matched against English (``en``) and Persian (``fa``) prefixes,
    defaulting to English for unrecognised values.
    """

    lang_code = (lang or "en").strip().lower()
    if lang_code.startswith("fa"):
        base = "اطلاعات کافی برای پاسخ دقیق در دسترس نیست."
        intro = "منابع موجود: "
    else:
        base = "Not enough information to answer accurately."
        intro = "Available sources: "

    citation_list = [entry for entry in citations if entry]
    if not citation_list:
        return base

    return f"{base} {intro}{'; '.join(citation_list)}"


__all__ = ["should_refuse", "refusal_message"]
