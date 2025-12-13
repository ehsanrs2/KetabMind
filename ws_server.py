import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import uuid
from typing import Any, Dict

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, WebSocket, status
from starlette.websockets import WebSocketDisconnect, WebSocketState


JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = "HS256"
PING_INTERVAL_SECONDS = 20
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

logger = structlog.get_logger(__name__)

app = FastAPI()


async def create_redis_client() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=True)


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


async def close_redis(redis: aioredis.Redis | None) -> None:
    if not redis:
        return

    close_callable = getattr(redis, "aclose", None)
    if callable(close_callable):
        await close_callable()
    else:
        redis.close()


async def enqueue_job(redis: aioredis.Redis, payload: str, user_id: str | None) -> str:
    request_id = str(uuid.uuid4())
    data: dict[str, str] = {"request_id": request_id, "payload": payload}
    if user_id:
        data["user_id"] = user_id

    await redis.xadd("jobs", data)
    logger.info("redis.job_enqueued", request_id=request_id)
    return request_id


async def forward_results(websocket: WebSocket, request_id: str) -> None:
    stream_name = f"results:{request_id}"
    last_id = "0-0"

    while websocket.application_state == WebSocketState.CONNECTED:
        redis: aioredis.Redis | None = None
        try:
            redis = await create_redis_client()
            while websocket.application_state == WebSocketState.CONNECTED:
                messages = await redis.xread({stream_name: last_id}, block=1_000, count=10)
                if not messages:
                    continue

                for _, entries in messages:
                    for message_id, data in entries:
                        last_id = message_id
                        await websocket.send_json(data)
                        logger.info(
                            "websocket.forwarded_result",
                            request_id=request_id,
                            message_id=message_id,
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "results.listener_error",
                request_id=request_id,
                error=str(exc),
            )
            await asyncio.sleep(1)
        finally:
            await close_redis(redis)


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

    redis = await create_redis_client()
    ping = asyncio.create_task(ping_task(websocket))
    listener_tasks: list[asyncio.Task[None]] = []

    try:
        while True:
            message = await websocket.receive_json()
            logger.info("websocket.received", message=message)

            if message.get("type") == "start_job":
                payload = message.get("payload", "")
                user_id = claims.get("user_id") or claims.get("sub")
                request_id = await enqueue_job(redis, str(payload), user_id)
                listener = asyncio.create_task(forward_results(websocket, request_id))
                listener_tasks.append(listener)
                await websocket.send_json({"type": "job_started", "request_id": request_id})
                logger.info("websocket.job_started", request_id=request_id)
            else:
                response = {**message, "echoed": True}
                await websocket.send_json(response)
                logger.info("websocket.sent", message=response)
    except WebSocketDisconnect:
        logger.info("websocket.disconnected", client=str(websocket.client))
    finally:
        ping.cancel()
        with contextlib.suppress(Exception):
            await ping
        for task in listener_tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.gather(*listener_tasks, return_exceptions=True)
        await close_redis(redis)
