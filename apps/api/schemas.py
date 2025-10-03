from __future__ import annotations

from typing import Any

from utils.pydantic_compat import BaseModel, Field


class Metadata(BaseModel):
    """Optional metadata attached to indexed books."""

    author: str | None = Field(default=None, description="Book author")
    year: int | str | None = Field(default=None, description="Publication year")
    subject: str | None = Field(default=None, description="Book subject or category")
    title: str | None = Field(default=None, description="Book title")

    def as_dict(self) -> dict[str, Any]:
        """Return a cleaned dictionary without empty values."""

        data = self.model_dump(exclude_none=True)
        return {k: v for k, v in data.items() if v not in ("", [], {})}


class IndexRequest(BaseModel):
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
