from __future__ import annotations

import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]
CallNext = Callable[[Any], Awaitable[Any]]


class RequestIDMiddleware:
    """Attach a request ID and structured logging context to each HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.header_name = os.getenv("REQUEST_ID_HEADER", "X-Request-ID")
        self._header_bytes = self.header_name.lower().encode("latin-1")
        self.log = structlog.get_logger("ketabmind.request")

    async def __call__(
        self,
        first_arg: Any,
        second_arg: ASGIReceive | CallNext | None = None,
        third_arg: ASGISend | None = None,
    ) -> Any:
        if isinstance(first_arg, dict):
            assert callable(second_arg) and callable(third_arg)
            return await self._asgi_call(first_arg, second_arg, third_arg)  # type: ignore[arg-type]
        assert callable(second_arg)
        return await self._stub_call(first_arg, second_arg)  # type: ignore[arg-type]

    async def _stub_call(self, request: Any, call_next: CallNext) -> Any:
        headers = getattr(request, "headers", {})
        request_id = headers.get(self.header_name.lower()) or str(uuid.uuid4())
        state = getattr(request, "state", None)
        if state is not None:
            state.request_id = request_id

        bind_contextvars(
            request_id=request_id,
            path=getattr(getattr(request, "url", None), "path", ""),
            method=getattr(request, "method", ""),
            status=None,
            latency_ms=None,
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
            latency = (time.perf_counter() - start) * 1000
            status_code = getattr(response, "status_code", 200)
            bind_contextvars(status=status_code, latency_ms=round(latency, 3))
            headers = getattr(response, "headers", None)
            if isinstance(headers, dict):
                headers.setdefault(self.header_name, request_id)
            self.log.info("http.request")
            return response
        except Exception:
            latency = (time.perf_counter() - start) * 1000
            bind_contextvars(status=500, latency_ms=round(latency, 3))
            self.log.exception("http.request.error")
            raise
        finally:
            clear_contextvars()

    async def _asgi_call(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        headers_list = scope.get("headers", [])
        request_id = self._find_header(headers_list) or str(uuid.uuid4())
        state = scope.get("state")
        if state is None:
            state = scope["state"] = {}
        if hasattr(state, "__dict__"):
            state.request_id = request_id
        else:
            try:
                state["request_id"] = request_id  # type: ignore[index]
            except TypeError:
                scope["state"] = {"request_id": request_id}

        bind_contextvars(
            request_id=request_id,
            path=scope.get("path", ""),
            method=scope.get("method", ""),
            status=None,
            latency_ms=None,
        )

        start_time = time.perf_counter()
        status_code: int | None = None
        finished = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code, finished

            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
                headers = message.setdefault("headers", [])
                if not self._header_present(headers):
                    headers.append(
                        (self.header_name.encode("latin-1"), request_id.encode("latin-1"))
                    )
                bind_contextvars(status=status_code)
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                if status_code is None:
                    status_code = 200
                    bind_contextvars(status=status_code)
                latency = (time.perf_counter() - start_time) * 1000
                bind_contextvars(latency_ms=round(latency, 3))
                finished = True
                self.log.info("http.request")
                clear_contextvars()

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            latency = (time.perf_counter() - start_time) * 1000
            if status_code is None:
                status_code = 500
            bind_contextvars(status=status_code, latency_ms=round(latency, 3))
            self.log.exception("http.request.error")
            clear_contextvars()
            raise
        finally:
            if not finished:
                clear_contextvars()

    def _find_header(self, headers: list[tuple[bytes, bytes]]) -> str | None:
        for key, value in headers:
            if key.lower() == self._header_bytes:
                return value.decode("latin-1")
        return None

    def _header_present(self, headers: list[tuple[bytes, bytes]]) -> bool:
        return any(key.lower() == self._header_bytes for key, _ in headers)
