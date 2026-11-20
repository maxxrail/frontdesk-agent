"""Building the running system from settings, in one place.

The API process and the worker both call ``build_services``; tests build a
``Services`` by hand with fakes. Jobs run through ``run_job`` whichever queue
delivered them, so the inline queue and the Redis worker behave the same.
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from frontdesk_agent.config import Settings
from frontdesk_agent.db.repositories import TenantDirectory
from frontdesk_agent.frontdesk.bandit import EpsilonGreedy
from frontdesk_agent.frontdesk.ports import Clock, Effects, JobQueue, SmsSender, SystemClock
from frontdesk_agent.frontdesk.service import FrontDesk, RateLimits
from frontdesk_agent.frontdesk.slotfill import SlotFiller
from frontdesk_agent.llm.base import LLMClient
from frontdesk_agent.logcontext import bind, bound, new_request_id
from frontdesk_agent.metrics import JOBS_FAILED, SMS_TURN_SECONDS, SMS_TURNS, sms_outcome
from frontdesk_agent.retrieval.embeddings import Embedder, default_embedder
from frontdesk_agent.tracing import job_span, trace_id_hex

logger = logging.getLogger(__name__)

JOBS = ("handle_inbound", "start_slot_fill", "expire_offer")


@dataclass
class Services:
    sessionmaker: async_sessionmaker[AsyncSession]
    frontdesk: FrontDesk
    sender: SmsSender
    queue: JobQueue
    clock: Clock = field(default_factory=SystemClock)


async def run_job(services: Services, job: str, **kwargs: Any) -> str:
    """Run one background job in its own transaction, then its side effects.

    Its log lines carry the job name, tenant and the request that queued it
    (``request_id`` in the job's arguments, when there was one)."""
    rid = kwargs.pop("request_id", None) or new_request_id()
    with job_span(job, kwargs.pop("trace", None)) as span:
        span.set_attribute("tenant.id", kwargs["tenant_id"])
        with bound(job=job, tenant_id=kwargs["tenant_id"], request_id=rid):
            bind(trace_id=trace_id_hex(span))
            start = time.perf_counter()
            try:
                outcome = await _run_job(services, job, **kwargs)
            except Exception:
                JOBS_FAILED.labels(job).inc()
                raise
            if job == "handle_inbound":
                SMS_TURNS.labels(sms_outcome(outcome)).inc()
                SMS_TURN_SECONDS.observe(time.perf_counter() - start)
            return outcome


async def _run_job(services: Services, job: str, **kwargs: Any) -> str:
    effects = Effects()
    tenant_id = uuid.UUID(kwargs["tenant_id"])
    async with services.sessionmaker() as session:
        fd = services.frontdesk
        if job == "handle_inbound":
            outcome = await fd.handle_inbound(
                session, tenant_id, uuid.UUID(kwargs["message_id"]), effects
            )
        elif job == "start_slot_fill":
            tenant = await TenantDirectory(session).get(tenant_id)
            exclude = kwargs.get("exclude_client_id")
            offer = await fd.slot_filler.start(
                session,
                tenant,
                effects,
                provider_id=uuid.UUID(kwargs["provider_id"]),
                starts_at=datetime.fromisoformat(kwargs["starts_at"]),
                ends_at=datetime.fromisoformat(kwargs["ends_at"]),
                now=services.clock.now(),
                exclude_client_id=uuid.UUID(exclude) if exclude else None,
            )
            outcome = "offered" if offer else "no_candidates"
        elif job == "expire_offer":
            tenant = await TenantDirectory(session).get(tenant_id)
            await fd.slot_filler.expire(
                session, tenant, effects, uuid.UUID(kwargs["offer_id"]), services.clock.now()
            )
            outcome = "checked"
        else:
            raise ValueError(f"unknown job: {job}")
        await session.commit()
        await effects.flush(session, services.sender, services.queue)
    return outcome


class InlineQueue:
    """Runs jobs in the current process. For local development without Redis:
    deferred jobs become sleeping tasks, so they are lost on restart."""

    def __init__(self) -> None:
        self.services: Services | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def enqueue(self, job: str, *, defer_seconds: float = 0, **kwargs: Any) -> None:
        async def run() -> None:
            if defer_seconds:
                await asyncio.sleep(defer_seconds)
            assert self.services is not None
            try:
                await run_job(self.services, job, **kwargs)
            except Exception:
                logger.exception("inline job failed", extra={"job": job})

        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class ArqQueue:
    """Enqueues onto Redis for the arq worker (frontdesk_agent.worker)."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: Any = None

    async def enqueue(self, job: str, *, defer_seconds: float = 0, **kwargs: Any) -> None:
        from arq import create_pool
        from arq.connections import RedisSettings

        if self._pool is None:
            self._pool = await create_pool(RedisSettings.from_dsn(self._redis_url))
        await self._pool.enqueue_job(job, _defer_by=defer_seconds or None, **kwargs)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()


def build_services(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    embedder: Embedder | None = None,
) -> Services:
    from frontdesk_agent.channels.twilio import ConsoleSender, TwilioSender

    sender: SmsSender = (
        TwilioSender(settings.twilio_account_sid, settings.twilio_auth_token)
        if settings.sms_sender == "twilio"
        else ConsoleSender()
    )
    queue: JobQueue = (
        ArqQueue(settings.redis_url) if settings.job_queue == "redis" else InlineQueue()
    )
    frontdesk = FrontDesk(
        llm=llm,
        embedder=embedder or default_embedder(),
        clock=SystemClock(),
        slot_filler=SlotFiller(EpsilonGreedy(epsilon=0.1, rng=random.Random())),
        prompt_version=settings.agent_prompt_version,
        limits=RateLimits(
            phone_messages=settings.rate_phone_messages,
            phone_window_s=settings.rate_phone_window_s,
            tenant_model_calls=settings.rate_tenant_model_calls,
            tenant_window_s=settings.rate_tenant_window_s,
        ),
    )
    services = Services(sessionmaker, frontdesk, sender, queue)
    if isinstance(queue, InlineQueue):
        queue.services = services
    return services
