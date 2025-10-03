"""Utilities for combining multiple retrieval scores with configurable weights."""

from __future__ import annotations

_COMPONENTS = ("cosine", "lexical", "reranker")


def parse_weights(spec: str) -> dict[str, float]:
    """Parse a weight specification string into a dictionary.

    The specification must contain comma-separated ``key=value`` pairs.
    Keys are normalized to lowercase and stripped of surrounding whitespace.

    Parameters
    ----------
    spec:
        A comma-separated list of weight assignments. Empty strings result
        in an empty dictionary.

    Returns
    -------
    dict[str, float]
        A mapping of component names to their weights.

    Raises
    ------
    ValueError
        If any entry in the specification is malformed or cannot be parsed
        as a floating point number.
    """

    if not spec:
        return {}

    weights: dict[str, float] = {}
    for entry in spec.split(","):
        item = entry.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight specification: '{item}'")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        try:
            weight = float(value.strip())
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid numeric weight for '{key}'") from exc
        weights[key] = weight
    return weights


def hybrid_score(
    cosine: float | None,
    lexical: float | None,
    reranker: float | None,
    weights: dict[str, float],
) -> float:
    """Combine individual scores into a single hybrid score.

    Missing components and weights are treated as zero. When the provided
    weights do not sum to one they are automatically normalized before the
    weighted sum is computed. If the total weight is zero the combined score
    is zero.
    """

    component_scores = {
        "cosine": 0.0 if cosine is None else cosine,
        "lexical": 0.0 if lexical is None else lexical,
        "reranker": 0.0 if reranker is None else reranker,
    }

    component_weights = {name: float(weights.get(name, 0.0)) for name in _COMPONENTS}

    total_weight = sum(component_weights.values())
    if total_weight == 0.0:
        return 0.0

    normalized_weights = {name: weight / total_weight for name, weight in component_weights.items()}

    return sum(component_scores[name] * normalized_weights[name] for name in _COMPONENTS)
