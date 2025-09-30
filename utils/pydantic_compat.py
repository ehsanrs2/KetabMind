from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING, get_type_hints

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from pydantic import BaseModel as BaseModelProto
    from pydantic import Field as FieldProto
else:
    _pydantic_spec = importlib.util.find_spec("pydantic")
    if _pydantic_spec is not None:  # pragma: no cover - passthrough when available
        _pydantic = importlib.import_module("pydantic")
        BaseModelProto = _pydantic.BaseModel
        FieldProto = _pydantic.Field
    else:

        class BaseModelProto:  # pragma: no cover - lightweight fallback implementation
            """A minimal stand-in for :class:`pydantic.BaseModel`."""

            def __init__(self, **data: Any) -> None:
                annotations = get_type_hints(self.__class__, include_extras=True)
                for name in annotations:
                    value = data.get(name, getattr(self.__class__, name, None))
                    setattr(self, name, value)

            @classmethod
            def model_validate(cls, data: dict[str, Any]) -> "BaseModelProto":
                return cls(**data)

            def model_dump(self, *, exclude_none: bool = False, **_: Any) -> dict[str, Any]:
                annotations = get_type_hints(self.__class__, include_extras=True)
                payload = {name: getattr(self, name) for name in annotations}
                if exclude_none:
                    return {k: v for k, v in payload.items() if v is not None}
                return payload

            def __iter__(self):  # pragma: no cover - compatibility helper
                yield from self.model_dump().items()

            def __repr__(self) -> str:  # pragma: no cover - debug helper
                return f"{self.__class__.__name__}({self.model_dump()!r})"

        def FieldProto(default: Any = None, **_: Any) -> Any:  # pragma: no cover - compat
            return default

BaseModel = BaseModelProto
Field = FieldProto

__all__ = ["BaseModel", "Field"]
