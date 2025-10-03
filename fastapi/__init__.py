# fastapi/__init__.py
from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

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
    def __init__(
        self,
        content: Any,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._content: Any = content
        self.headers: dict[str, str] = dict(headers or {})

    def json(self) -> Any:
        return self._content

    def text(self) -> str:
        return str(self._content)

    @property
    def content(self) -> Any:
        return self._content


class JSONResponse(Response):
    def __init__(
        self,
        content: Any,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(content, status_code=status_code, headers=headers)


class StreamingResponse(Response):
    def __init__(
        self,
        content: Iterable[str],
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
    ) -> None:
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
    limiter: Any | None

    def __init__(self) -> None:
        self.limiter = None


class Request:
    def __init__(
        self,
        method: str,
        url: URL,
        headers: Mapping[str, str] | None = None,
        query_params: Mapping[str, Any] | None = None,
        form: Mapping[str, Any] | None = None,
        path_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.url = url
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = dict(query_params or {})
        self.form = dict(form or {})
        self.state = State()
        self.path_params = dict(path_params or {})

    def header(self, name: str, default: Any = None) -> Any:
        return self.headers.get(name.lower(), default)


class UploadFile:
    """Lightweight UploadFile compatible with the test client."""

    def __init__(self, filename: str, data: bytes, content_type: str | None = None) -> None:
        self.filename = filename
        self._data = data
        self.content_type = content_type or "application/octet-stream"

    async def read(self) -> bytes:
        return self._data


# --- Parameter marker classes (Query/Form/File/Depends) ---------------------
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


# --- Helpers to parse Annotated[...] metadata --------------------------------
def _first_marker_from_annotation(annotation: Any) -> _Param | None:
    """Extract first supported marker (Query/Form/File/Depends) from an Annotated type."""
    if get_origin(annotation) is Annotated:
        # Annotated[T, meta1, meta2, ...]
        _, *meta = get_args(annotation)
        for m in meta:
            if isinstance(m, (Query, Form, File, Depends)):
                return m
    return None


def _is_request_type(annotation: Any) -> bool:
    return annotation is Request or annotation == Request


class FastAPI:
    def __init__(self, *, title: str | None = None) -> None:
        self.title = title or "FastAPI"
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
        self._route_patterns: list[tuple[str, re.Pattern[str], Callable[..., Any]]] = []
        self._middleware: list[
            Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]
        ] = []
        self.state = State()

    def add_middleware(self, middleware_class: type[Any], **options: Any) -> None:
        middleware = middleware_class(self, **options)

        async def wrapper(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            result = middleware(request, call_next)
            if inspect.isawaitable(result):
                return cast(Response, await result)
            return cast(Response, result)

        self._middleware.append(wrapper)

    def middleware(self, typ: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if typ != "http":
            raise ValueError("Only 'http' middleware supported in lightweight FastAPI stub")

        def decorator(
            func: Callable[
                [Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]
            ],
        ) -> Callable[..., Any]:
            self._middleware.append(func)
            return func

        return decorator

    def _add_route(
        self, method: str, path: str, endpoint: Callable[..., Any]
    ) -> Callable[..., Any]:
        if "{" in path and "}" in path:
            pattern = (
                "^"
                + re.sub(
                    r"{([a-zA-Z_][a-zA-Z0-9_]*)}",
                    r"(?P<\1>[^/]+)",
                    path,
                )
                + "$"
            )
            self._route_patterns.append((method.upper(), re.compile(pattern), endpoint))
        else:
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
        endpoint = self._routes.get(key)
        path_params: dict[str, Any] = {}
        if endpoint is None:
            for method_name, pattern, handler in self._route_patterns:
                if method_name != method.upper():
                    continue
                match = pattern.match(path)
                if match:
                    endpoint = handler
                    path_params = match.groupdict()
                    break
        if endpoint is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
        request = Request(
            method.upper(),
            URL(path),
            headers=headers,
            query_params=query_params,
            form=form_data,
            path_params=path_params,
        )

        async def run_endpoint(req: Request) -> Response:
            try:
                result = await _invoke(
                    endpoint,
                    req,
                    body,
                    files,
                    query_params or {},
                    form_data or {},
                )
            except HTTPException:
                raise
            return _to_response(result)

        async def call_stack(index: int, req: Request) -> Response:
            if index >= len(self._middleware):
                return await run_endpoint(req)
            middleware = self._middleware[index]
            return await middleware(
                req,
                lambda r: call_stack(index + 1, r),
            )

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
    """Call endpoint, filling parameters from Request / body / files based on markers."""
    signature = inspect.signature(func)
    annotations = get_type_hints(func, include_extras=True)
    kwargs: dict[str, Any] = {}

    for name, param in signature.parameters.items():
        default = param.default
        annotation = annotations.get(name, param.annotation)

        # 1) Prefer Annotated[...] markers if present
        marker = _first_marker_from_annotation(annotation)

        if isinstance(marker, Depends):
            dep = marker.dependency
            dep_result = await _invoke(dep, request, body, files, query_params, form_data)
            kwargs[name] = dep_result
            continue

        if isinstance(marker, File):
            if files and name in files:
                kwargs[name] = files[name]
            elif marker.required:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"Missing file parameter: {name}",
                )
            else:
                kwargs[name] = marker.default
            continue

        if isinstance(marker, Form):
            kwargs[name] = form_data.get(name, marker.default)
            continue

        if isinstance(marker, Query):
            kwargs[name] = query_params.get(name, marker.default)
            continue

        # 2) Legacy path: marker in default value (non-Annotated)
        if isinstance(default, Depends):
            dep_result = await _invoke(
                default.dependency, request, body, files, query_params, form_data
            )
            kwargs[name] = dep_result
            continue

        if isinstance(default, File):
            if files and name in files:
                kwargs[name] = files[name]
            elif default.required:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"Missing file parameter: {name}",
                )
            else:
                kwargs[name] = default.default
            continue

        if isinstance(default, Form):
            kwargs[name] = form_data.get(name, default.default)
            continue

        if isinstance(default, Query):
            kwargs[name] = query_params.get(name, default.default)
            continue

        # 3) Other common sources
        if _is_request_type(annotation):
            kwargs[name] = request
            continue

        if hasattr(request, "path_params") and name in request.path_params:
            kwargs[name] = request.path_params[name]
            continue

        if body and name in body:
            value = body[name]
            if hasattr(annotation, "model_validate"):
                kwargs[name] = annotation.model_validate(value)
            else:
                kwargs[name] = value
            continue

        if body and hasattr(annotation, "model_validate"):
            kwargs[name] = annotation.model_validate(body)
            continue

        # 4) Fallback to default or None
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
    if isinstance(result, str | bytes):
        return Response(result, status_code=status.HTTP_200_OK)
    if isinstance(result, Mapping):
        return JSONResponse(dict(result))
    if isinstance(result, Iterable):
        return JSONResponse(list(result))
    return JSONResponse(result)
