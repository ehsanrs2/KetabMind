from __future__ import annotations

from typing import Any, Callable


class CORSMiddleware:  # pragma: no cover - simple stub
    def __init__(self, app: Any, **options: Any) -> None:
        self.app = app
        self.allow_origin = options.get("allow_origin") or options.get("allow_origins", ["*"])

    async def __call__(
        self,
        first_arg: Any,
        second_arg: Callable[..., Any] | None = None,
        third_arg: Callable[..., Any] | None = None,
    ) -> Any:
        if isinstance(first_arg, dict):
            assert callable(second_arg) and callable(third_arg)
            await self.app(first_arg, second_arg, third_arg)  # type: ignore[arg-type]
            return None
        assert callable(second_arg)
        response = await second_arg(first_arg)
        headers = getattr(response, "headers", None)
        if isinstance(headers, dict):
            headers.setdefault("access-control-allow-origin", "*")
        return response
