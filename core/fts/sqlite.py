"""SQLite FTS5 backend for per-book full-text search."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from . import FTSBackendProto, FTSMatch


class SqliteFTSBackend(FTSBackendProto):
    """SQLite FTS5 implementation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.available = True
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS pages
                USING fts5(book_id UNINDEXED, page_num UNINDEXED, text)
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self.available = False

    def index_book(self, book_id: str, pages: Iterable[tuple[int, str]]) -> None:
        cleaned_pages = [(str(book_id), int(page_num), text or "") for page_num, text in pages]
        if not cleaned_pages:
            return
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM pages WHERE book_id = ?", (book_id,))
            self._conn.executemany(
                "INSERT INTO pages (book_id, page_num, text) VALUES (?, ?, ?)",
                cleaned_pages,
            )

    def search(
        self,
        query: str,
        *,
        book_id: str | None = None,
        limit: int | None = None,
    ) -> Sequence[FTSMatch]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        limit_value = max(1, int(limit or 20))
        params: list[object] = [cleaned]
        sql = (
            "SELECT book_id, page_num, text, bm25(pages) AS score " "FROM pages WHERE pages MATCH ?"
        )
        if book_id:
            sql += " AND book_id = ?"
            params.append(book_id)
        sql += " ORDER BY score, page_num LIMIT ?"
        params.append(limit_value)

        with self._lock, self._conn:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        seen: set[tuple[str, int]] = set()
        results: list[FTSMatch] = []
        for row in rows:
            book = str(row["book_id"])
            try:
                page = int(row["page_num"])
            except (TypeError, ValueError):
                continue
            key = (book, page)
            if key in seen:
                continue
            seen.add(key)
            text = str(row["text"] or "")
            try:
                raw_score = float(row["score"])
            except (TypeError, ValueError):
                raw_score = 0.0
            score = 1.0 / (1.0 + max(raw_score, 0.0))
            results.append(FTSMatch(book_id=book, page_num=page, text=text, score=score))
        return results


__all__ = ["SqliteFTSBackend"]
