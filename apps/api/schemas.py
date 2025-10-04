from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from pydantic import BaseModel as BaseModel
    from pydantic import Field
else:  # pragma: no cover - runtime compatibility
    from utils.pydantic_compat import BaseModel, Field


class Metadata(BaseModel):  # type: ignore[misc]
    """Optional metadata attached to indexed books."""

    author: str | None = Field(default=None, description="Book author")
    year: int | str | None = Field(default=None, description="Publication year")
    subject: str | None = Field(default=None, description="Book subject or category")
    title: str | None = Field(default=None, description="Book title")

    def as_dict(self) -> dict[str, Any]:
        """Return a cleaned dictionary without empty values."""

        data = self.model_dump(exclude_none=True)
        return {k: v for k, v in data.items() if v not in ("", [], {})}


class IndexRequest(BaseModel):  # type: ignore[misc]
    path: str
    collection: str | None = None
    author: str | None = None
    year: int | str | None = None
    subject: str | None = None
    title: str | None = None

    def metadata(self) -> dict[str, Any]:
        """Return metadata payload cleaned of empty values."""

        meta = Metadata(
            author=self.author,
            year=self.year,
            subject=self.subject,
            title=self.title,
        )
        return meta.as_dict()


class MessageCreate(BaseModel):  # type: ignore[misc]
    role: str
    content: str
    citations: list[str] | None = None
    meta: dict[str, Any] | None = None


class ExportRequest(BaseModel):  # type: ignore[misc]
    message_id: int | str
    format: str | None = "pdf"


class BookmarkCreate(BaseModel):  # type: ignore[misc]
    message_id: int | str
    tag: str | None = None


class SessionCreate(BaseModel):  # type: ignore[misc]
    title: str | None = None
    book_id: int | str | None = None
