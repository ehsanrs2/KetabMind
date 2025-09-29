from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

_FLAG_MAP = {
    "ASCII": re.ASCII,
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "VERBOSE": re.VERBOSE,
}

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "resources" / "ingest" / "rules.yaml"


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load header/footer cleanup rules from YAML configuration."""

    target = Path(path) if path else _DEFAULT_RULES_PATH
    if not target.exists():
        return {}
    if yaml is None:
        return {}
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError("rules configuration must be a mapping")
    return dict(data)


def _compile_patterns(entries: Iterable[Any] | None) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    if not entries:
        return patterns
    for entry in entries:
        if isinstance(entry, str):
            pattern = entry
            flags_value = 0
        elif isinstance(entry, Mapping):
            pattern = str(entry.get("pattern", ""))
            flags_iter = entry.get("flags", [])
            flags_value = 0
            if isinstance(flags_iter, str):
                flags_iter = [flags_iter]
            for name in flags_iter or []:
                flags_value |= _FLAG_MAP.get(str(name).upper(), 0)
        else:
            continue
        if not pattern:
            continue
        patterns.append(re.compile(pattern, flags_value))
    return patterns


def _zone_directives(zone_cfg: Any, *, from_start: bool) -> tuple[int, int]:
    remove = 0
    inspect = 0
    if zone_cfg is None:
        return remove, inspect
    key = "top" if from_start else "bottom"
    if isinstance(zone_cfg, int):
        inspect = max(zone_cfg, 0)
    elif isinstance(zone_cfg, Mapping):
        value = zone_cfg.get(key)
        if value is None:
            value = zone_cfg.get("lines")
        if isinstance(value, Mapping):
            if "remove" in value:
                try:
                    remove = max(int(value.get("remove", 0)), 0)
                except (TypeError, ValueError):
                    remove = 0
            if "inspect" in value:
                try:
                    inspect = max(int(value.get("inspect", 0)), 0)
                except (TypeError, ValueError):
                    inspect = 0
            elif "look" in value or "max" in value:
                raw = value.get("look", value.get("max"))
                try:
                    inspect = max(int(raw or 0), 0)
                except (TypeError, ValueError):
                    inspect = 0
        else:
            try:
                inspect = max(int(value or 0), 0)
            except (TypeError, ValueError):
                inspect = 0
        if remove <= 0 and zone_cfg.get("remove") is not None:
            try:
                remove = max(int(zone_cfg.get("remove", 0)), 0)
            except (TypeError, ValueError):
                remove = 0
    else:
        try:
            inspect = max(int(zone_cfg), 0)
        except (TypeError, ValueError):
            inspect = 0
    if inspect <= 0 and remove > 0:
        inspect = remove
    return remove, inspect


def _trim_zone(lines: list[str], *, remove: int, from_start: bool) -> list[str]:
    if remove <= 0 or not lines:
        return lines
    remove = min(remove, len(lines))
    if from_start:
        return lines[remove:]
    return lines[:-remove] if remove < len(lines) else []


def _apply_regex(
    lines: list[str],
    patterns: list[re.Pattern[str]],
    *,
    from_start: bool,
    limit: int,
) -> list[str]:
    if not patterns or not lines:
        return lines
    max_removals = len(lines) if limit <= 0 else min(limit, len(lines))
    removed = 0
    while lines and removed < max_removals:
        idx = 0 if from_start else len(lines) - 1
        if any(p.search(lines[idx]) for p in patterns):
            lines.pop(idx)
            removed += 1
            continue
        break
    return lines


def apply_rules(text: str, rules_cfg: Mapping[str, Any] | None) -> str:
    """Remove recurring headers/footers from text according to configuration."""

    if not text or not rules_cfg:
        return text

    defaults = rules_cfg.get("defaults") if isinstance(rules_cfg, Mapping) else None
    strip_edges = True
    default_limit = 3
    if isinstance(defaults, Mapping):
        strip_edges = bool(defaults.get("strip_surrounding_whitespace", True))
        try:
            default_limit = int(defaults.get("max_matches", default_limit))
        except (TypeError, ValueError):
            default_limit = 3

    lines = text.splitlines()

    header_cfg = rules_cfg.get("header") if isinstance(rules_cfg, Mapping) else None
    footer_cfg = rules_cfg.get("footer") if isinstance(rules_cfg, Mapping) else None

    if isinstance(header_cfg, Mapping):
        patterns = _compile_patterns(header_cfg.get("regex"))
        remove, inspect = _zone_directives(header_cfg.get("zones"), from_start=True)
        lines = _trim_zone(lines, remove=remove, from_start=True)
        limit = header_cfg.get("max_matches", inspect or default_limit)
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            limit_int = default_limit
        lines = _apply_regex(lines, patterns, from_start=True, limit=limit_int)

    if isinstance(footer_cfg, Mapping):
        patterns = _compile_patterns(footer_cfg.get("regex"))
        remove, inspect = _zone_directives(footer_cfg.get("zones"), from_start=False)
        lines = _trim_zone(lines, remove=remove, from_start=False)
        limit = footer_cfg.get("max_matches", inspect or default_limit)
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            limit_int = default_limit
        lines = _apply_regex(lines, patterns, from_start=False, limit=limit_int)

    if strip_edges:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join(lines)


__all__ = ["apply_rules", "load_rules"]
