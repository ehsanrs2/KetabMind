"""Chunk text using a sliding window."""

from collections.abc import Iterable


def chunk_text(lines: Iterable[str], size: int = 100, overlap: int = 20) -> list[str]:
    """Return list of text chunks."""
    words: list[str] = []
    for line in lines:
        words.extend(line.split())
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks
