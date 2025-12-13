import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, Set

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

request_websocket_map: dict[str, WebSocket] = {}
request_task_map: dict[str, asyncio.Task[None]] = {}
websocket_requests_map: dict[WebSocket, Set[str]] = defaultdict(set)


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


def _remove_request(request_id: str, *, current_task: asyncio.Task[None] | None = None) -> list[asyncio.Task[None]]:
    task = request_task_map.pop(request_id, None)
    websocket = request_websocket_map.pop(request_id, None)

    if websocket:
        websocket_requests = websocket_requests_map.get(websocket)
        if websocket_requests is not None:
            websocket_requests.discard(request_id)
            if not websocket_requests:
                websocket_requests_map.pop(websocket, None)

    logger.info("request_mapping.removed", request_id=request_id)

    if task and task is not current_task and not task.done():
        task.cancel()
        return [task]
    return []


async def forward_results(websocket: WebSocket, request_id: str) -> None:
    stream_name = f"results:{request_id}"
    last_id = "0-0"
    redis: aioredis.Redis | None = None
    logger.info("results.listener_started", request_id=request_id)
    try:
        redis = await create_redis_client()
        loop = asyncio.get_running_loop()
        last_message_at = loop.time()

        while websocket.application_state == WebSocketState.CONNECTED:
            messages = await redis.xread({stream_name: last_id}, block=1_000, count=10)
            if not messages:
                if loop.time() - last_message_at >= 60:
                    logger.info("results.listener_timeout", request_id=request_id)
                    break
                continue

            last_message_at = loop.time()
            for _, entries in messages:
                for message_id, data in entries:
                    last_id = message_id
                    message_type = data.get("type")

                    if message_type == "chunk":
                        await websocket.send_json(data)
                        logger.info(
                            "websocket.forwarded_chunk",
                            request_id=request_id,
                            message_id=message_id,
                            seq=data.get("seq"),
                        )
                        continue

                    if message_type in {"done", "error"}:
                        await websocket.send_json(data)
                        logger.info(
                            "results.listener_completed",
                            request_id=request_id,
                            message_id=message_id,
                            message_type=message_type,
                        )
                        return

                    logger.info(
                        "websocket.forwarded_result",
                        request_id=request_id,
                        message_id=message_id,
                        message_type=message_type,
                    )
                    await websocket.send_json(data)
    except asyncio.CancelledError:
        logger.info("results.listener_cancelled", request_id=request_id)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "results.listener_error",
            request_id=request_id,
            error=str(exc),
        )
    finally:
        await close_redis(redis)
        cleanup_tasks = _remove_request(request_id, current_task=asyncio.current_task())
        if cleanup_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        logger.info("results.listener_stopped", request_id=request_id)


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

    try:
        while True:
            message = await websocket.receive_json()
            logger.info("websocket.received", message=message)

            if message.get("type") == "start_job":
                payload = message.get("payload", "")
                user_id = claims.get("user_id") or claims.get("sub")
                request_id = await enqueue_job(redis, str(payload), user_id)
                listener = asyncio.create_task(forward_results(websocket, request_id))
                request_websocket_map[request_id] = websocket
                request_task_map[request_id] = listener
                websocket_requests_map[websocket].add(request_id)
                logger.info("request_mapping.created", request_id=request_id)
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
        pending_cleanup: list[asyncio.Task[None]] = []
        for request_id in list(websocket_requests_map.get(websocket, set())):
            pending_cleanup.extend(_remove_request(request_id))
        if pending_cleanup:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.gather(*pending_cleanup, return_exceptions=True)
        await close_redis(redis)
