import asyncio
import os
import socket
import uuid
from typing import Any

import redis.asyncio as aioredis
import structlog

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GROUP_NAME = os.getenv("REDIS_GROUP", "workers")
STREAM_NAME = os.getenv("REDIS_STREAM", "jobs")
CONSUMER_NAME = os.getenv("REDIS_CONSUMER") or f"{socket.gethostname()}-{uuid.uuid4()}"

logger = structlog.get_logger(__name__)


def create_redis() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def ensure_group(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_create(STREAM_NAME, GROUP_NAME, id="$", mkstream=True)
        logger.info("redis.group_created", stream=STREAM_NAME, group=GROUP_NAME)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
        logger.info("redis.group_exists", stream=STREAM_NAME, group=GROUP_NAME)


async def ensure_consumer(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_createconsumer(STREAM_NAME, GROUP_NAME, CONSUMER_NAME)
        logger.info(
            "redis.consumer_created",
            stream=STREAM_NAME,
            group=GROUP_NAME,
            consumer=CONSUMER_NAME,
        )
    except aioredis.ResponseError as exc:
        if "NOGROUP" in str(exc):
            await ensure_group(redis)
            await ensure_consumer(redis)
            return
        logger.error(
            "redis.consumer_error",
            stream=STREAM_NAME,
            group=GROUP_NAME,
            consumer=CONSUMER_NAME,
            error=str(exc),
        )
        raise


async def handle_job(redis: aioredis.Redis, message_id: str, data: dict[str, Any]) -> None:
    request_id = data.get("request_id")
    logger.info("worker.processing", request_id=request_id, message_id=message_id)
    result_stream = f"results:{request_id}"
    chunks = [
        "Processing your request...",
        "Generating insights...",
        "Finalizing response...",
    ]

    for idx, chunk in enumerate(chunks, start=1):
        await redis.xadd(
            result_stream,
            {
                "type": "chunk",
                "request_id": request_id,
                "seq": idx,
                "data": chunk,
            },
        )
        await asyncio.sleep(0.2)

    await redis.xadd(
        result_stream,
        {
            "type": "done",
            "request_id": request_id,
            "result": "this is the result",
        },
    )
    await redis.xack(STREAM_NAME, GROUP_NAME, message_id)
    logger.info("worker.completed", request_id=request_id, stream=result_stream)


async def consume(redis: aioredis.Redis) -> None:
    while True:
        messages = await redis.xreadgroup(
            groupname=GROUP_NAME,
            consumername=CONSUMER_NAME,
            streams={STREAM_NAME: ">"},
            count=1,
            block=1_000,
        )
        if not messages:
            continue

        for _, entries in messages:
            for message_id, data in entries:
                try:
                    await handle_job(redis, message_id, data)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "worker.job_failed",
                        request_id=data.get("request_id"),
                        error=str(exc),
                    )
                    await asyncio.sleep(1)


async def main() -> None:
    while True:
        redis: aioredis.Redis | None = None
        try:
            redis = create_redis()
            await ensure_group(redis)
            await ensure_consumer(redis)
            await consume(redis)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("worker.error", error=str(exc))
            await asyncio.sleep(1)
        finally:
            if redis:
                close_callable = getattr(redis, "aclose", None)
                if callable(close_callable):
                    await close_callable()
                else:
                    redis.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("worker.stopped")
