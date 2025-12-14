import asyncio
import contextlib
import os

import redis.asyncio as aioredis
import structlog


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_NAME = os.getenv("REDIS_STREAM", "jobs")
CHECK_INTERVAL_SECONDS = float(os.getenv("AUTOSCALER_INTERVAL", "5"))
SCALE_THRESHOLD = int(os.getenv("AUTOSCALER_THRESHOLD", "20"))

logger = structlog.get_logger(__name__)


def create_redis_client() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def close_redis(redis: aioredis.Redis | None) -> None:
    if not redis:
        return

    close_callable = getattr(redis, "aclose", None)
    if callable(close_callable):
        await close_callable()
    else:
        redis.close()


async def monitor_queue(redis: aioredis.Redis) -> None:
    while True:
        length = await redis.xlen(STREAM_NAME)
        logger.info(
            "autoscaler.queue_size",
            stream=STREAM_NAME,
            length=length,
            threshold=SCALE_THRESHOLD,
        )

        if length > SCALE_THRESHOLD:
            logger.warning(
                "autoscaler.scale_up",
                stream=STREAM_NAME,
                length=length,
                threshold=SCALE_THRESHOLD,
                action="scale up",
            )
            print("scale up")

        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


async def main() -> None:
    redis: aioredis.Redis | None = None
    monitor_task: asyncio.Task[None] | None = None
    try:
        redis = create_redis_client()
        monitor_task = asyncio.create_task(monitor_queue(redis))
        await monitor_task
    except asyncio.CancelledError:
        if monitor_task:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await monitor_task
        raise
    finally:
        await close_redis(redis)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("autoscaler.stopped")
