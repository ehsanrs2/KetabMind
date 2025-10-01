"""JWT authentication utilities for the API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import timedelta
from typing import Any, Dict, Mapping, MutableMapping, TypedDict

from fastapi import HTTPException, Request, status

from core.config import settings


class _UserRecord(TypedDict):
    id: str
    email: str
    name: str
    hashed_password: str
    bookmarks: list[dict[str, Any]]
    sessions: list[dict[str, Any]]


def _hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_USERS: Dict[str, _UserRecord] = {
    "alice@example.com": {
        "id": "user-alice",
        "email": "alice@example.com",
        "name": "Alice",
        "hashed_password": _hash_password("wonderland"),
        "bookmarks": [
            {"id": "b1", "title": "Introduction to Retrieval"},
            {"id": "b2", "title": "Persian NLP Guide"},
        ],
        "sessions": [
            {"id": "s1", "topic": "Mock session"},
        ],
    },
    "bob@example.com": {
        "id": "user-bob",
        "email": "bob@example.com",
        "name": "Bob",
        "hashed_password": _hash_password("builder"),
        "bookmarks": [
            {"id": "b9", "title": "Advanced Prompting"},
        ],
        "sessions": [
            {"id": "s9", "topic": "Evaluation metrics"},
        ],
    },
}

_USERS_BY_ID: Dict[str, _UserRecord] = {record["id"]: record for record in _USERS.values()}
_DEFAULT_USER_ID = next(iter(_USERS_BY_ID))


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _json_loads(data: str) -> Mapping[str, Any]:
    loaded = json.loads(data)
    if not isinstance(loaded, Mapping):
        raise ValueError("JWT payload must be a mapping")
    return loaded


def _sign(message: str) -> str:
    secret = settings.jwt_secret.encode("utf-8")
    digest = hmac.new(secret, message.encode("utf-8"), hashlib.sha256).digest()
    return _b64encode(digest)


def _user_profile(record: _UserRecord) -> dict[str, Any]:
    return {"id": record["id"], "email": record["email"], "name": record["name"]}


def _get_user_record_by_id(user_id: str) -> _UserRecord:
    record = _USERS_BY_ID.get(user_id)
    if record is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
    return record


def authenticate_user(email: str, password: str) -> _UserRecord:
    record = _USERS.get(email.lower())
    if record is None or not hmac.compare_digest(record["hashed_password"], _hash_password(password)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return record


def create_access_token(data: Mapping[str, Any], expires_delta: timedelta | int | None = None) -> str:
    expires_seconds: int | None
    if isinstance(expires_delta, timedelta):
        expires_seconds = int(expires_delta.total_seconds())
    else:
        expires_seconds = int(expires_delta) if expires_delta is not None else settings.jwt_expiration_seconds
    now = int(time.time())
    payload: Dict[str, Any] = dict(data)
    payload.setdefault("iat", now)
    if expires_seconds is not None:
        payload["exp"] = now + expires_seconds
    header = {"alg": "HS256", "typ": "JWT"}
    header_segment = _b64encode(_json_dumps(header).encode("utf-8"))
    payload_segment = _b64encode(_json_dumps(payload).encode("utf-8"))
    signature_segment = _sign(f"{header_segment}.{payload_segment}")
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def verify_token(token: str) -> Mapping[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:  # pragma: no cover - defensive split error handling
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc
    expected_signature = _sign(f"{header_segment}.{payload_segment}")
    if not hmac.compare_digest(signature_segment, expected_signature):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token signature")
    try:
        payload = _json_loads(_b64decode(payload_segment).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token payload") from exc
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp < time.time():
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    return payload


def get_user_profile(user_id: str) -> dict[str, Any]:
    return _user_profile(_get_user_record_by_id(user_id))


def get_user_bookmarks(user_id: str) -> list[dict[str, Any]]:
    record = _get_user_record_by_id(user_id)
    return [dict(item) for item in record.get("bookmarks", [])]


def get_user_sessions(user_id: str) -> list[dict[str, Any]]:
    record = _get_user_record_by_id(user_id)
    return [dict(item) for item in record.get("sessions", [])]


def get_current_user(request: Request) -> dict[str, Any]:
    record = _USERS_BY_ID[_DEFAULT_USER_ID]
    profile = _user_profile(record)
    if not settings.auth_required:
        request.state.user = profile
        return profile
    cookie_header = request.headers.get("cookie", "")
    cookies = _parse_cookies(cookie_header)
    token = cookies.get("access_token")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    if request.method in {"POST", "PUT", "DELETE"}:
        csrf_header = request.headers.get("x-csrf-token")
        if csrf_header != token:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid CSRF token")
    payload = verify_token(token)
    user_id = payload.get("sub")
    if not isinstance(user_id, str):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token subject")
    record = _get_user_record_by_id(user_id)
    profile = _user_profile(record)
    request.state.user = profile
    return profile


def _parse_cookies(header: str) -> MutableMapping[str, str]:
    cookies: Dict[str, str] = {}
    if not header:
        return cookies
    parts = [part.strip() for part in header.split(";") if part.strip()]
    for part in parts:
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookies[name] = value
    return cookies
