"""GET /metrics, for Prometheus.

Counters and histograms are updated as things happen (metrics.py). The
three state gauges are refreshed here, per scrape, each with a short
timeout so a slow dependency cannot hang the scrape:

- review queue depth, from Postgres;
- whether Postgres and Redis answer (Redis only when it is the job queue);
- the latest eval pass rate per prompt version, from results files.

Unauthenticated, like /healthz: it holds no ids, names or tenants, and in
production it is reachable only inside the network (Module 8).
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.repositories import OperationsCounts
from frontdesk_agent.metrics import DEPENDENCY_UP, REVIEW_QUEUE, load_eval_scores

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

CHECK_TIMEOUT_S = 1.0


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    settings = get_settings()
    await asyncio.gather(
        _refresh_database(request), _refresh_redis(settings.job_queue, settings.redis_url)
    )
    load_eval_scores(Path(settings.eval_results_dir))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _refresh_database(request: Request) -> None:
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_S):
            async with request.app.state.sessionmaker() as session:
                await session.execute(text("SELECT 1"))
                depth = await OperationsCounts(session).pending_reviews_by_kind()
    except Exception:  # down, slow, or not migrated: all mean "not up"
        DEPENDENCY_UP.labels("postgres").set(0)
        return
    DEPENDENCY_UP.labels("postgres").set(1)
    for kind, n in depth.items():
        REVIEW_QUEUE.labels(kind).set(n)


async def _refresh_redis(job_queue: str, redis_url: str) -> None:
    if job_queue != "redis":
        return
    from redis.asyncio import Redis

    client = Redis.from_url(redis_url, socket_timeout=CHECK_TIMEOUT_S)
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_S):
            await client.ping()
        DEPENDENCY_UP.labels("redis").set(1)
    except Exception:
        DEPENDENCY_UP.labels("redis").set(0)
    finally:
        await client.aclose()
