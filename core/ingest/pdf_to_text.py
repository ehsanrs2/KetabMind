from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import typer
from PyPDF2 import PdfReader

app = typer.Typer(help="PDF to JSONL pages extractor")


@dataclass
class Page:
    page_num: int
    text: str
    section: str | None = None


def _clean_text(text: str, remove_headers: bool = True, remove_footers: bool = True) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    if remove_headers and len(lines) > 3:
        lines = lines[1:]
    if remove_footers and len(lines) > 3:
        lines = lines[:-1]
    cleaned = "\n".join(ln for ln in lines if ln.strip())
    return cleaned


def pdf_to_pages(
    path: Path, remove_headers: bool = True, remove_footers: bool = True
) -> list[Page]:
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for idx, p in enumerate(reader.pages, start=1):
        text = p.extract_text() or ""
        text = _clean_text(text, remove_headers=remove_headers, remove_footers=remove_footers)
        pages.append(Page(page_num=idx, text=text))
    return pages


def write_jsonl(pages: Iterable[Page], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")


@app.command()
def main(
    in_path: str = typer.Option(..., "--in", help="Input PDF path"),
    out_path: str = typer.Option(..., "--out", help="Output JSONL path"),
    no_remove_headers: bool = typer.Option(False, help="Do not remove headers"),
    no_remove_footers: bool = typer.Option(False, help="Do not remove footers"),
) -> None:
    pages = pdf_to_pages(
        Path(in_path), remove_headers=not no_remove_headers, remove_footers=not no_remove_footers
    )
    write_jsonl(pages, Path(out_path))


if __name__ == "__main__":
    app()
