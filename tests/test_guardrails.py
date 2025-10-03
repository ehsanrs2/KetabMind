import pytest

from answering.guardrails import refusal_message, should_refuse


@pytest.mark.parametrize(
    "coverage,confidence,rule,expected",
    [
        (0.6, 0.8, "strict", True),
        (0.8, 0.7, "strict", False),
        (0.4, 0.6, "balanced", True),
        (0.7, 0.35, "balanced", True),
        (0.7, 0.5, "balanced", False),
        (0.3, 0.4, "lenient", True),
        (0.36, 0.3, "lenient", False),
        (0.9, 0.9, "always", True),
        (0.0, 0.0, "never", False),
        (0.49, 0.6, "coverage<0.5,confidence<0.2", True),
        (0.7, 0.25, "coverage<0.5,confidence<0.3", True),
        (0.7, 0.4, "coverage<0.5,confidence<0.3", False),
    ],
)
def test_should_refuse_thresholds(
    coverage: float, confidence: float, rule: str, expected: bool
) -> None:
    assert should_refuse(coverage, confidence, rule) is expected


@pytest.mark.parametrize(
    "lang,citations,expected",
    [
        ("en", [], "Not enough information to answer accurately."),
        (
            "en",
            ["[b:1-2]", "[c:3]"],
            "Not enough information to answer accurately. Available sources: [b:1-2]; [c:3]",
        ),
        (
            "fa",
            ["[کتاب:۱۲]"],
            "اطلاعات کافی برای پاسخ دقیق در دسترس نیست. منابع موجود: [کتاب:۱۲]",
        ),
        ("", ["[b:1]"], "Not enough information to answer accurately. Available sources: [b:1]"),
    ],
)
def test_refusal_message_variants(lang: str, citations: list[str], expected: str) -> None:
    assert refusal_message(lang, citations) == expected
