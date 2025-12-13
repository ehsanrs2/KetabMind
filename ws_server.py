import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
from typing import Any, Dict

import structlog
from fastapi import FastAPI, WebSocket, status
from starlette.websockets import WebSocketDisconnect, WebSocketState


logger = structlog.get_logger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = "HS256"


def _get_ping_interval_seconds() -> float:
    raw = os.getenv("PING_INTERVAL_SECONDS", "20")
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "websocket.invalid_ping_interval",
            provided=raw,
            default="20",
        )
        return 20.0


PING_INTERVAL_SECONDS = _get_ping_interval_seconds()

app = FastAPI()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt(payload: Dict[str, Any], secret: str = JWT_SECRET) -> str:
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    header_segment = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_segment = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_segment}.{payload_segment}".encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    signature_segment = _b64url_encode(signature)
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def decode_jwt(token: str, secret: str = JWT_SECRET) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")

    header_segment, payload_segment, signature_segment = parts
    try:
        header = json.loads(_b64url_decode(header_segment))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid header") from exc

    if header.get("alg") != JWT_ALGORITHM:
        raise ValueError("Unsupported algorithm")

    signing_input = f"{header_segment}.{payload_segment}".encode()
    expected_signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    provided_signature = _b64url_decode(signature_segment)

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise ValueError("Signature verification failed")

    try:
        return json.loads(_b64url_decode(payload_segment))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid payload") from exc


def get_token(websocket: WebSocket) -> str | None:
    query_token = websocket.query_params.get("token")
    cookie_token = websocket.cookies.get("token")
    return query_token or cookie_token


async def ping_task(websocket: WebSocket) -> None:
    while websocket.application_state == WebSocketState.CONNECTED:
        await asyncio.sleep(PING_INTERVAL_SECONDS)
        try:
            await websocket.send_json({"type": "ping"})
            logger.info("websocket.ping_sent")
        except Exception:  # noqa: BLE001
            break


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    token = get_token(websocket)
    if not token:
        logger.warning("websocket.connection_rejected", reason="missing_token")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        claims = decode_jwt(token)
    except ValueError as exc:
        logger.warning("websocket.connection_rejected", reason=str(exc))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info("websocket.connected", client=str(websocket.client), claims=claims)

    ping = asyncio.create_task(ping_task(websocket))

    try:
        while True:
            message = await websocket.receive_json()
            logger.info("websocket.received", message=message)

            response = {**message, "echoed": True}
            await websocket.send_json(response)
            logger.info("websocket.sent", message=response)
    except WebSocketDisconnect:
        logger.info("websocket.disconnected", client=str(websocket.client))
    finally:
        ping.cancel()
        with contextlib.suppress(Exception):
            await ping
