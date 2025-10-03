"""Persian text normalization utilities."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_fa"]

_ARABIC_YEH = "\u064a"
_ALEF_MAKSURA = "\u0649"
_PERSIAN_YEH = "\u06cc"
_ARABIC_KEHEH = "\u0643"
_PERSIAN_KEHEH = "\u06a9"

_ZWNJ = "\u200c"

_PERSIAN_LETTERS = "اآأبپتثجچحخدذرزژسشصضطظعغفقکگلمنهوهیی"  # includes double ی for ranges
_SUFFIXES = (
    "ها",
    "های",
    "هایی",
    "هایم",
    "هایت",
    "هایتان",
    "هایمان",
    "ها",
    "تر",
    "ترین",
    "گری",
    "گران",
    "گر",
)

_SPACE_REPLACEMENTS = {
    "\u00a0": " ",  # NBSP
    "\u200b": " ",  # ZWSP
    "\u202f": " ",  # narrow no-break space, common in Persian inputs
}

_SUFFIX_PATTERN = re.compile(rf"([{_PERSIAN_LETTERS}0-9۰-۹])(?P<suffix>{'|'.join(_SUFFIXES)})")
_SUFFIX_WITH_SPACE_PATTERN = re.compile(
    rf"([{_PERSIAN_LETTERS}0-9۰-۹])\s+(?P<suffix>{'|'.join(_SUFFIXES)})"
)
_PREFIX_PATTERN = re.compile(rf"\b(?P<prefix>ن?می)\s+(?=[{_PERSIAN_LETTERS}])")


def _remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_spaces(text: str) -> str:
    for bad, good in _SPACE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Collapse multiple consecutive spaces introduced by replacements
    text = re.sub(r"\s+", lambda m: " " if "\n" not in m.group(0) else m.group(0), text)
    return text


def _apply_suffix_zwnj(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        stem = match.group(1)
        suffix = match.group("suffix")
        return f"{stem}{_ZWNJ}{suffix}"

    text = _SUFFIX_WITH_SPACE_PATTERN.sub(_repl, text)
    return _SUFFIX_PATTERN.sub(_repl, text)


def _apply_prefix_zwnj(text: str) -> str:
    return _PREFIX_PATTERN.sub(lambda m: f"{m.group('prefix')}{_ZWNJ}", text)


def normalize_fa(text: str) -> str:
    """Normalize Persian text with lightweight heuristics."""
    if not text:
        return ""

    text = text.replace(_ARABIC_YEH, _PERSIAN_YEH).replace(_ALEF_MAKSURA, _PERSIAN_YEH)
    text = text.replace(_ARABIC_KEHEH, _PERSIAN_KEHEH)
    text = _remove_diacritics(text)
    text = _normalize_spaces(text)
    text = _apply_prefix_zwnj(text)
    text = _apply_suffix_zwnj(text)
    return text
