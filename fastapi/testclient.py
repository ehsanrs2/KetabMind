from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping

from . import FastAPI, Response, UploadFile


class TestClient:
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
        body = dict(json) if json is not None else None
        form_data = dict(data or {})
        response = asyncio.run(
            self.app._call_route(
                method,
                path,
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
