from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, get_type_hints

__all__ = [
    "FastAPI",
    "Depends",
    "File",
    "Form",
    "HTTPException",
    "Query",
    "Request",
    "UploadFile",
    "status",
    "JSONResponse",
    "StreamingResponse",
]


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


status = SimpleNamespace(
    HTTP_200_OK=200,
    HTTP_401_UNAUTHORIZED=401,
    HTTP_403_FORBIDDEN=403,
    HTTP_400_BAD_REQUEST=400,
    HTTP_404_NOT_FOUND=404,
    HTTP_422_UNPROCESSABLE_ENTITY=422,
    HTTP_502_BAD_GATEWAY=502,
    HTTP_504_GATEWAY_TIMEOUT=504,
)


class Response:
    def __init__(self, content: Any, *, status_code: int = 200, headers: Optional[Mapping[str, str]] = None) -> None:
        self.status_code = status_code
        self._content = content
        self.headers: Dict[str, str] = dict(headers or {})

    def json(self) -> Any:
        return self._content

    def text(self) -> str:
        return str(self._content)


class JSONResponse(Response):
    def __init__(self, content: Any, *, status_code: int = 200, headers: Optional[Mapping[str, str]] = None) -> None:
        super().__init__(content, status_code=status_code, headers=headers)


class StreamingResponse(Response):
    def __init__(self, content: Iterable[str], *, status_code: int = 200, headers: Optional[Mapping[str, str]] = None, media_type: str | None = None) -> None:
        super().__init__(list(content), status_code=status_code, headers=headers)
        if media_type:
            self.headers.setdefault("content-type", media_type)

    def iter_text(self) -> Iterable[str]:
        yield from self._content

    def iter_lines(self) -> Iterable[str]:  # pragma: no cover - simple alias
        yield from self._content


@dataclass
class URL:
    path: str
    query: str = ""


class State:
    def __init__(self) -> None:
        pass


class Request:
    def __init__(self, method: str, url: URL, headers: Mapping[str, str] | None = None, query_params: Mapping[str, Any] | None = None, form: Mapping[str, Any] | None = None) -> None:
        self.method = method
        self.url = url
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = dict(query_params or {})
        self.form = dict(form or {})
        self.state = State()

    def header(self, name: str, default: Any = None) -> Any:
        return self.headers.get(name.lower(), default)


class UploadFile:
    def __init__(self, filename: str, data: bytes, content_type: str | None = None) -> None:
        self.filename = filename
        self._data = data
        self.content_type = content_type or "application/octet-stream"

    async def read(self) -> bytes:
        return self._data


class _Param:
    def __init__(self, default: Any = None) -> None:
        self.default = default


class Depends(_Param):
    def __init__(self, dependency: Callable[..., Any]) -> None:
        super().__init__(None)
        self.dependency = dependency


class Form(_Param):
    pass


class File(_Param):
    def __init__(self, default: Any = None, *, required: bool = True) -> None:
        super().__init__(default)
        self.required = required


class Query(_Param):
    pass


class FastAPI:
    def __init__(self, *, title: str | None = None) -> None:
        self.title = title or "FastAPI"
        self._routes: Dict[tuple[str, str], Callable[..., Any]] = {}
        self._middleware: list[Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]] = []

    def add_middleware(self, middleware_class: type[Any], **options: Any) -> None:
        middleware = middleware_class(self, **options)

        async def wrapper(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            result = middleware(request, call_next)
            if inspect.isawaitable(result):
                return await result  # type: ignore[return-value]
            return result  # type: ignore[return-value]

        self._middleware.append(wrapper)

    def middleware(self, typ: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if typ != "http":
            raise ValueError("Only 'http' middleware supported in lightweight FastAPI stub")

        def decorator(func: Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]) -> Callable[..., Any]:
            self._middleware.append(func)
            return func

        return decorator

    def _add_route(self, method: str, path: str, endpoint: Callable[..., Any]) -> Callable[..., Any]:
        self._routes[(method.upper(), path)] = endpoint
        return endpoint

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self._add_route("GET", path, func)

        return decorator

    def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self._add_route("POST", path, func)

        return decorator

    async def _call_route(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        files: Mapping[str, UploadFile] | None = None,
        headers: Mapping[str, str] | None = None,
        query_params: Mapping[str, Any] | None = None,
        form_data: Mapping[str, Any] | None = None,
    ) -> Response:
        key = (method.upper(), path)
        if key not in self._routes:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
        endpoint = self._routes[key]
        request = Request(method.upper(), URL(path), headers=headers, query_params=query_params, form=form_data)

        async def run_endpoint(req: Request) -> Response:
            try:
                result = await _invoke(endpoint, req, body, files, query_params or {}, form_data or {})
            except HTTPException:
                raise
            return _to_response(result)

        async def call_stack(index: int, req: Request) -> Response:
            if index >= len(self._middleware):
                return await run_endpoint(req)
            middleware = self._middleware[index]
            return await middleware(req, lambda r: call_stack(index + 1, r))

        try:
            return await call_stack(0, request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


async def _invoke(
    func: Callable[..., Any],
    request: Request,
    body: Mapping[str, Any] | None,
    files: Mapping[str, UploadFile] | None,
    query_params: Mapping[str, Any],
    form_data: Mapping[str, Any],
) -> Any:
    signature = inspect.signature(func)
    annotations = get_type_hints(func, include_extras=True)
    kwargs: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        default = param.default
        annotation = annotations.get(name, param.annotation)
        if isinstance(default, Depends):
            dependency = default.dependency
            dep_result = await _invoke(dependency, request, body, files, query_params, form_data)
            kwargs[name] = dep_result
        elif isinstance(default, File):
            if files and name in files:
                kwargs[name] = files[name]
            elif default.required:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Missing file parameter: {name}")
            else:
                kwargs[name] = default.default
        elif isinstance(default, Form):
            kwargs[name] = form_data.get(name, default.default)
        elif isinstance(default, Query):
            kwargs[name] = query_params.get(name, default.default)
        elif annotation is Request or annotation == Request:
            kwargs[name] = request
        elif body and name in body:
            value = body[name]
            if hasattr(annotation, "model_validate"):
                kwargs[name] = annotation.model_validate(value)
            else:
                kwargs[name] = value
        elif body and hasattr(annotation, "model_validate"):
            kwargs[name] = annotation.model_validate(body)
        else:
            if default is inspect._empty:
                kwargs[name] = None
            else:
                kwargs[name] = default
    result = func(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _to_response(result: Any) -> Response:
    if isinstance(result, Response):
        return result
    if result is None:
        return Response(None, status_code=status.HTTP_200_OK)
    if isinstance(result, (str, bytes)):
        return Response(result, status_code=status.HTTP_200_OK)
    if isinstance(result, Mapping):
        return JSONResponse(dict(result))
    if isinstance(result, Iterable):
        return JSONResponse(list(result))
    return JSONResponse(result)
