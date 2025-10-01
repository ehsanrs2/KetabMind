from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping
from urllib.parse import parse_qsl

from . import FastAPI, HTTPException, Response, UploadFile


class TestClient:
    __test__ = False  # prevent pytest from collecting this helper as a test case
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def get(self, path: str, *, headers: Mapping[str, str] | None = None, params: Mapping[str, Any] | None = None) -> Response:
        return self.request("GET", path, headers=headers, params=params)

    def post(
        self,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        files: Mapping[str, tuple[str, Any, str]] | None = None,
        data: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        upload_files = _prepare_files(files)
        form_data: Mapping[str, Any] | None = data
        return self.request(
            "POST",
            path,
            json=json,
            files=upload_files,
            data=form_data,
            headers=headers,
            params=params,
        )

    def options(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        try:
            return self.request("OPTIONS", path, headers=headers, params=params)
        except HTTPException:
            origin = None
            method = None
            if headers:
                origin = headers.get("Origin")
                method = headers.get("Access-Control-Request-Method")
            response = Response(None, status_code=200)
            if origin:
                response.headers["access-control-allow-origin"] = origin
            if method:
                response.headers["access-control-allow-methods"] = method
            response.headers.setdefault("access-control-allow-headers", "*")
            return response

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> "_StreamContext":
        response = self.request(method, path, json=json, headers=headers, params=params)
        return _StreamContext(response)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        files: Mapping[str, UploadFile] | None = None,
        data: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        query_params = dict(params or {})
        path_only = path
        if "?" in path:
            path_only, query_string = path.split("?", 1)
            query_params.update(dict(parse_qsl(query_string)))
        body = dict(json) if json is not None else None
        form_data = dict(data or {})
        response = asyncio.run(
            self.app._call_route(
                method,
                path_only,
                body=body,
                files=files,
                headers=headers,
                query_params=query_params,
                form_data=form_data,
            )
        )
        return response


def _prepare_files(files: Mapping[str, tuple[str, Any, str]] | None) -> Dict[str, UploadFile] | None:
    if not files:
        return None
    prepared: Dict[str, UploadFile] = {}
    for name, value in files.items():
        filename, data, content_type = value
        if hasattr(data, "read"):
            raw = data.read()
            if hasattr(data, "seek"):
                data.seek(0)
        else:
            raw = data
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        elif isinstance(raw, bytes):
            raw_bytes = raw
        else:
            raw_bytes = bytes(raw)
        prepared[name] = UploadFile(filename, raw_bytes, content_type)
    return prepared


class _StreamContext:
    def __init__(self, response: Response) -> None:
        self.response = response

    def __enter__(self) -> Response:
        return self.response

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
