"""Prompt templates for bilingual answers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from string import Template

ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "400"))
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")


def _detect_language(question: str, guidelines: dict | None, lang_auto: bool | str) -> str:
    if guidelines:
        lang_hint = guidelines.get("language") or guidelines.get("lang")
        if isinstance(lang_hint, str):
            lang = lang_hint.lower()
            if lang.startswith("fa"):
                return "fa"
            if lang.startswith("en"):
                return "en"
    if isinstance(lang_auto, str):
        lang = lang_auto.lower()
        if lang.startswith("fa"):
            return "fa"
        if lang.startswith("en"):
            return "en"
    if lang_auto and _ARABIC_SCRIPT_RE.search(question):
        return "fa"
    return "en"


def _select_style(default_style: str, guidelines: dict | None) -> str:
    style = default_style
    if guidelines:
        guideline_style = guidelines.get("style") or guidelines.get("format")
        if isinstance(guideline_style, str):
            style = guideline_style
    return "paragraph" if style == "paragraph" else "bullets"


def _gather_guidance(guidelines: dict | None, exclude: Iterable[str]) -> list[str]:
    if not guidelines:
        return []
    extra: list[str] = []
    for key, value in guidelines.items():
        if key in exclude:
            continue
        if isinstance(value, str) and value.strip():
            extra.append(value.strip())
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    extra.append(item.strip())
    return extra


def _format_contexts(contexts: list[dict], citation_fmt: str) -> list[str]:
    lines: list[str] = []
    for idx, ctx in enumerate(contexts, 1):
        snippet = (
            ctx.get("text") or ctx.get("snippet") or ctx.get("content") or ctx.get("passage") or ""
        )
        snippet = " ".join(str(snippet).split())
        metadata = {k: v for k, v in ctx.items() if isinstance(k, str)}
        book_id = metadata.get("book_id") or metadata.get("bookId") or "unknown"
        page_start = (
            metadata.get("page_start")
            if metadata.get("page_start") is not None
            else metadata.get("page")
        )
        page_end = metadata.get("page_end")
        if page_start is None:
            page_start = "?"
        if page_end is None:
            page_end = page_start
        metadata.setdefault("book_id", book_id)
        metadata.setdefault("page_start", page_start)
        metadata.setdefault("page_end", page_end)
        citation_example = Template(citation_fmt).safe_substitute(metadata)
        if snippet:
            lines.append(f"{idx}. {citation_example} {snippet}")
        else:
            lines.append(f"{idx}. {citation_example}")
    if not lines:
        lines.append("(no context provided)")
    return lines


def bilingual_answer_template(
    question: str,
    contexts: list[dict],
    lang_auto: bool | str = True,
    style: str = "bullets",
    citation_fmt: str = "[${book_id}:${page_start}-${page_end}]",
    guidelines: dict | None = None,
) -> str:
    """Build an instruction prompt that yields a bilingual answer with inline citations."""

    selected_language = _detect_language(question, guidelines, lang_auto)
    selected_style = _select_style(style, guidelines)
    extra_guidance = _gather_guidance(guidelines, {"language", "lang", "style", "format"})

    if selected_language == "fa":
        base_instructions = [
            "شما کتاب‌مایند هستید، یک دستیار پژوهشی دوزبانه.",
            "پاسخ را به زبان فارسی رسمی بنویس.",
            f"پاسخ را خلاصه و کمتر از {ANSWER_MAX_TOKENS} توکن نگه دار.",
            (
                "فقط از متون زمینه‌ای استفاده کن و برای هر جملهٔ حاوی اطلاعات "
                f"از قالب {citation_fmt} استفاده کن و مقادیر را جایگزین کن."
            ),
            (
                "استناد جعلی نساز. اگر شواهد کافی نبود، دقیقاً این جمله را بنویس: "
                '"Not enough information to answer accurately."'
            ),
        ]
        style_instruction = (
            "پاسخ را به صورت فهرست گلوله‌ای با نماد «•» در ابتدای هر مورد بنویس."
            if selected_style == "bullets"
            else "پاسخ را در ۱ تا ۲ پاراگراف کوتاه بنویس و هر جمله باید استناد داشته باشد."
        )
        context_header = f"متن‌های زمینه‌ای ({len(contexts)} مورد):"
        question_label = "پرسش:"
        answer_label = "پاسخ:"
    else:
        base_instructions = [
            "You are KetabMind, a bilingual research assistant.",
            "Respond in English.",
            f"Keep the answer concise (under {ANSWER_MAX_TOKENS} tokens).",
            (
                "Rely only on the context passages and add an inline citation using "
                f"{citation_fmt} to every factual sentence. Replace the placeholders with the correct metadata."
            ),
            (
                'Do not fabricate citations. If evidence is insufficient, reply exactly: "Not enough information to answer accurately."'
            ),
        ]
        style_instruction = (
            "Format the answer as bullet points starting with '• ' and keep each bullet to one sentence with a citation."
            if selected_style == "bullets"
            else "Write the answer in 1-2 short paragraphs and cite every sentence."
        )
        context_header = f"Context passages ({len(contexts)}):"
        question_label = "Question:"
        answer_label = "Answer:"

    base_instructions.append(style_instruction)
    base_instructions.append(
        "پاسخ فقط باید بر اساس اطلاعات همین متون باشد."
        if selected_language == "fa"
        else "Answer only with information grounded in these passages."
    )
    base_instructions.extend(extra_guidance)

    instruction_lines = "\n".join(f"- {line}" for line in base_instructions)
    context_lines = "\n".join(_format_contexts(contexts, citation_fmt))

    sections = [
        "System:",
        instruction_lines,
        "",
        question_label,
        question.strip(),
        "",
        context_header,
        context_lines,
        "",
        answer_label,
    ]
    return "\n".join(sections)
