from __future__ import annotations

from typing import Any, Callable


class CORSMiddleware:  # pragma: no cover - simple stub
    def __init__(self, app: Any, **_: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        await self.app(scope, receive, send)
