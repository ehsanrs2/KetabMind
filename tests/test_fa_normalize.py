import pytest

from nlp.fa_normalize import normalize_fa


def test_unify_arabic_persian_letters():
    text = "علي و كيان"
    expected = "علی و کیان"
    assert normalize_fa(text) == expected


def test_remove_diacritics():
    text = "عَلَيْهِمْ"
    expected = "علیهم"
    assert normalize_fa(text) == expected


def test_space_normalization_and_zwnj():
    text = "کتاب\u00a0ها و می \u200b رود"
    expected = "کتاب‌ها و می‌رود"
    assert normalize_fa(text) == expected


def test_preserve_latin_case():
    text = "این متن با MixedCase و نسخه v1.0Beta است"
    assert "MixedCase" in normalize_fa(text)
    assert "v1.0Beta" in normalize_fa(text)
