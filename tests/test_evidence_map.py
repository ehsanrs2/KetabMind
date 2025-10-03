import pytest

from answering.evidence_map import (
    align_sentences_to_contexts,
    compute_coverage,
    split_sentences,
)


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_MODEL_NAME", "mock")


def test_split_sentences_handles_multilingual() -> None:
    en = "Paris is beautiful. It attracts many visitors! Does it ever sleep?"
    fa = "تهران پایتخت ایران است. زبان رسمی فارسی است؟ بله!"

    en_sentences = split_sentences(en, "en")
    fa_sentences = split_sentences(fa, "fa")

    assert en_sentences == [
        "Paris is beautiful",
        "It attracts many visitors",
        "Does it ever sleep",
    ]
    assert fa_sentences == [
        "تهران پایتخت ایران است",
        "زبان رسمی فارسی است",
        "بله",
    ]


def test_align_and_compute_coverage_english_example() -> None:
    sentences = split_sentences(
        "Paris is the capital of France. Berlin is the capital of Germany.",
        lang="en",
    )

    contexts = [
        {"id": "c1", "text": "The capital of France is Paris."},
        {"id": "c2", "text": "Tehran is the capital of Iran."},
    ]

    evidence_map = align_sentences_to_contexts(sentences, contexts)
    coverage = compute_coverage(evidence_map)

    assert coverage == pytest.approx(0.5, rel=1e-2)
    assert evidence_map[0]["contexts"], "first sentence should be supported"
    assert not evidence_map[1]["contexts"], "second sentence should be unsupported"


def test_align_and_compute_coverage_persian_example() -> None:
    text = "تهران پایتخت ایران است. زبان رسمی کشور فارسی است."
    sentences = split_sentences(text, lang="fa")

    contexts = [
        {"id": "fa1", "text": "تهران به عنوان پایتخت ایران شناخته می‌شود."},
        {"id": "fa2", "text": "فارسی زبان رسمی جمهوری اسلامی ایران است."},
    ]

    evidence_map = align_sentences_to_contexts(sentences, contexts)
    coverage = compute_coverage(evidence_map)

    assert coverage == pytest.approx(1.0, rel=1e-2)
    assert all(item["contexts"] for item in evidence_map.values())
