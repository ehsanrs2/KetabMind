"""EPUB ingestion without external dependencies."""

from __future__ import annotations

import html
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TypedDict

from defusedxml import ElementTree as ET


class PageDict(TypedDict):
    page_num: int
    text: str
    section: str | None


def _html_to_text(raw_html: str) -> str:
    """Strip HTML tags and normalise whitespace."""

    text = re.sub("<[^>]+>", "", raw_html)
    return html.unescape(text).strip()


def _find_rootfile(zf: zipfile.ZipFile) -> str:
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = container.find(".//{*}rootfile")
    if rootfile is None:
        msg = "EPUB container missing rootfile reference"
        raise ValueError(msg)
    full_path = rootfile.attrib.get("full-path")
    if not full_path:
        msg = "EPUB rootfile missing full-path attribute"
        raise ValueError(msg)
    return full_path


def _iter_spine_items(zf: zipfile.ZipFile, opf_path: str) -> Iterable[str]:
    opf = ET.fromstring(zf.read(opf_path))
    manifest = {}
    for item in opf.findall(".//{*}manifest/{*}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if item_id and href:
            manifest[item_id] = href

    opf_dir = PurePosixPath(opf_path).parent

    for itemref in opf.findall(".//{*}spine/{*}itemref"):
        idref = itemref.attrib.get("idref")
        if not idref:
            continue
        href = manifest.get(idref)
        if not href:
            continue
        resource = (opf_dir / href).as_posix() if opf_dir != PurePosixPath("") else href
        yield resource


def extract_pages(path: Path, chars: int = 1024) -> Iterable[PageDict]:
    """Yield plain-text pages from an EPUB file using stdlib only."""

    with zipfile.ZipFile(path, "r") as zf:
        opf_path = _find_rootfile(zf)
        buffer = ""
        page_num = 1
        for resource in _iter_spine_items(zf, opf_path):
            try:
                raw = zf.read(resource).decode("utf-8", errors="ignore")
            except KeyError:  # pragma: no cover - broken spine entry
                continue
            buffer += _html_to_text(raw) + "\n"
            while len(buffer) >= chars:
                yield {
                    "page_num": page_num,
                    "text": buffer[:chars].strip(),
                    "section": None,
                }
                page_num += 1
                buffer = buffer[chars:]
        if buffer.strip():
            yield {"page_num": page_num, "text": buffer.strip(), "section": None}
