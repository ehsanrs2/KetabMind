import re

from prompting.templates import ANSWER_MAX_TOKENS, bilingual_answer_template


def test_bilingual_template_english_bullets() -> None:
    question = "Who wrote the highlighted treatise?"
    contexts = [
        {
            "book_id": "book42",
            "page_start": 10,
            "page_end": 12,
            "text": "The treatise was authored by Scholar X in 1850.",
        },
        {
            "book_id": "book7",
            "page_start": 5,
            "page_end": 6,
            "snippet": "Scholar X expanded the work with new commentary.",
        },
    ]

    prompt = bilingual_answer_template(question, contexts, lang_auto=True, style="bullets")

    assert "Respond in English." in prompt
    assert f"under {ANSWER_MAX_TOKENS} tokens" in prompt
    assert "Format the answer as bullet points" in prompt
    assert "• " in prompt  # bullet instruction
    assert "Not enough information to answer accurately." in prompt
    assert "Question:" in prompt and question in prompt
    assert "Context passages (2):" in prompt
    assert "Answer:" in prompt
    assert "[book42:10-12]" in prompt
    assert "[book7:5-6]" in prompt


def test_bilingual_template_persian_paragraph() -> None:
    question = "نویسندهٔ شاهنامه چه کسی است؟"
    contexts = [
        {
            "book_id": "iranian-classics",
            "page_start": 1,
            "page_end": 1,
            "text": "شاهنامه توسط حکیم ابوالقاسم فردوسی نوشته شده است.",
        }
    ]

    prompt = bilingual_answer_template(question, contexts, style="paragraph")

    assert "پاسخ را به زبان فارسی رسمی بنویس." in prompt
    assert str(ANSWER_MAX_TOKENS) in prompt
    assert "پاسخ را در ۱ تا ۲ پاراگراف کوتاه بنویس" in prompt
    assert "متن‌های زمینه‌ای (1 مورد):" in prompt
    assert "پرسش:" in prompt and question in prompt
    assert "پاسخ:" in prompt
    assert "[iranian-classics:1-1]" in prompt
    # Ensure citation instructions remain in Persian section
    assert re.search(r"قالب \[\$\{book_id\}:\$\{page_start\}-\$\{page_end\}\]", prompt)
