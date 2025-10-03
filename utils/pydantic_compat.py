"""Lightweight compatibility helpers for optional Pydantic dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, get_type_hints


class _BaseModelFallback:
    """A minimal stand-in for :class:`pydantic.BaseModel`."""

    def __init__(self, **data: Any) -> None:
        annotations = get_type_hints(self.__class__, include_extras=True)
        for name in annotations:
            value = data.get(name, getattr(self.__class__, name, None))
            setattr(self, name, value)

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> _BaseModelFallback:
        return cls(**data)

    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> dict[str, Any]:
        annotations = get_type_hints(self.__class__, include_extras=True)
        payload = {name: getattr(self, name) for name in annotations}
        if exclude_none:
            return {k: v for k, v in payload.items() if v is not None}
        return payload

    def __iter__(self) -> Iterator[tuple[str, Any]]:  # pragma: no cover
        yield from self.model_dump().items()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}({self.model_dump()!r})"


def _field_fallback(default: Any = None, **_: Any) -> Any:  # pragma: no cover
    return default


if TYPE_CHECKING:  # pragma: no cover - provide a typed fallback for tooling
    BaseModelType = _BaseModelFallback
    FieldType = _field_fallback
else:  # pragma: no branch - runtime import with fallback
    try:  # pragma: no cover - import side effect
        from pydantic import BaseModel as BaseModelType
        from pydantic import Field as FieldType
    except ImportError:
        BaseModelType = _BaseModelFallback
        FieldType = _field_fallback


BaseModel = BaseModelType
Field = FieldType

__all__ = ["BaseModel", "Field"]
