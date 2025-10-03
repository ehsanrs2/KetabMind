from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, get_type_hints


class _FallbackBaseModel:
    """A minimal stand-in for :class:`pydantic.BaseModel`."""

    def __init__(self, **data: Any) -> None:
        annotations = get_type_hints(self.__class__, include_extras=True)
        for name in annotations:
            value = data.get(name, getattr(self.__class__, name, None))
            setattr(self, name, value)

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> _FallbackBaseModel:
        return cls(**data)

    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> dict[str, Any]:
        annotations = get_type_hints(self.__class__, include_extras=True)
        payload = {name: getattr(self, name) for name in annotations}
        if exclude_none:
            return {k: v for k, v in payload.items() if v is not None}
        return payload

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from self.model_dump().items()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_dump()!r})"


def _fallback_field(default: Any = None, **_: Any) -> Any:
    return default


# Prototypes that we will bind to either real pydantic or our fallback
BaseModelProto: type[Any]
FieldProto: Callable[..., Any]

try:
    from pydantic import BaseModel as _RealBaseModel
    from pydantic import Field as _real_field
except Exception:
    BaseModelProto = _FallbackBaseModel
    FieldProto = _fallback_field
else:
    BaseModelProto = _RealBaseModel
    FieldProto = _real_field  # typing: Callable[..., Any]


# Public re-exports with explicit annotations
BaseModel: type[Any] = BaseModelProto
Field: Callable[..., Any] = FieldProto

__all__ = ["BaseModel", "Field"]
