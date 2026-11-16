"""Test harness for the front desk: a fixed clock, recording sender and queue,
a scripted model, and one small clinic.

    world = await make_world(session)          # clinic, provider, schedule, client
    desk = make_desk(db_url, fake)             # Services wired to fakes
    await text(desk, world, "hi")              # same path as the webhook
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Client, Provider, ProviderSchedule
from frontdesk_agent.db.repositories import (
    ClientRepository,
    ProviderRepository,
    ScheduleRepository,
    TenantDirectory,
)
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.frontdesk.bandit import EpsilonGreedy
from frontdesk_agent.frontdesk.service import FrontDesk, accept_inbound
from frontdesk_agent.frontdesk.slotfill import SlotFiller
from frontdesk_agent.llm.fake import FakeLLMClient
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.ingest import ingest_directory
from frontdesk_agent.retrieval.store import PolicyStore
from frontdesk_agent.services import Services, run_job

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
CLINIC_NUMBER = "+14165550100"
CLIENT_PHONE = "+14165550111"
# Monday 2030-01-07, 9:00 a.m. in Toronto (14:00 UTC).
NOW = datetime(2030, 1, 7, 14, 0, tzinfo=UTC)


class FixedClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current

    def advance(self, **kwargs: float) -> None:
        self.current += timedelta(**kwargs)


@dataclass
class RecordingSender:
    sent: list[tuple[str, str]] = field(default_factory=list)  # (to, body)
    fail: bool = False

    async def send(self, *, to: str, from_: str, body: str) -> str:
        if self.fail:
            raise RuntimeError("provider down")
        self.sent.append((to, body))
        return f"SM{uuid.uuid4().hex[:30]}"

    def bodies(self, to: str | None = None) -> list[str]:
        return [b for t, b in self.sent if to is None or t == to]


@dataclass
class RecordingQueue:
    jobs: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)

    async def enqueue(self, job: str, *, defer_seconds: float = 0, **kwargs: Any) -> None:
        self.jobs.append((job, defer_seconds, kwargs))

    def names(self) -> list[str]:
        return [j for j, _, _ in self.jobs]

    async def drain(self, services: Services, *, include_deferred: bool = False) -> list[str]:
        """Run queued jobs (and any they queue) until none are left."""
        outcomes: list[str] = []
        while True:
            ready = [j for j in self.jobs if include_deferred or not j[1]]
            if not ready:
                return outcomes
            job = ready[0]
            self.jobs.remove(job)
            outcomes.append(await run_job(services, job[0], **job[2]))


@dataclass(frozen=True)
class World:
    """Plain IDs, not ORM objects: tests roll back their own session to read
    what the desk committed, and that would expire loaded objects."""

    tenant_id: uuid.UUID
    provider_id: uuid.UUID
    client_id: uuid.UUID


async def make_world(session: AsyncSession, *, policies: bool = False) -> World:
    tenant = await TenantDirectory(session).create(
        name="Maple Dental",
        slug="maple-dental",
        business_type="dental office",
        sms_number=CLINIC_NUMBER,
        timezone="America/Toronto",
    )
    provider = await ProviderRepository(session, tenant.id).add(Provider(name="Dr. Osei"))
    for weekday in range(5):  # Monday to Friday, 9 to 5, 30-minute slots
        await ScheduleRepository(session, tenant.id).add(
            ProviderSchedule(
                provider_id=provider.id,
                weekday=weekday,
                start_time=time(9),
                end_time=time(17),
                slot_minutes=30,
            )
        )
    client = await ClientRepository(session, tenant.id).add(
        Client(first_name="Dana", last_name="Lee", phone=CLIENT_PHONE)
    )
    if policies:
        await ingest_directory(
            PolicyStore(session, tenant.id, default_embedder()), POLICIES / "maple-dental"
        )
    await session.commit()
    return World(tenant.id, provider.id, client.id)


@dataclass
class Desk:
    services: Services
    llm: FakeLLMClient
    sender: RecordingSender
    queue: RecordingQueue
    clock: FixedClock

    async def dispose(self) -> None:
        await self.services.sessionmaker.kw["bind"].dispose()


def make_desk(db_url: str, *, epsilon: float = 0.0, seed: int = 0) -> Desk:
    llm, sender, queue, clock = FakeLLMClient(), RecordingSender(), RecordingQueue(), FixedClock()
    frontdesk = FrontDesk(
        llm=llm,
        embedder=default_embedder(),
        clock=clock,
        slot_filler=SlotFiller(EpsilonGreedy(epsilon, rng=random.Random(seed))),
    )
    services = Services(make_sessionmaker(make_engine(db_url)), frontdesk, sender, queue, clock)
    return Desk(services, llm, sender, queue, clock)


async def text(
    desk: Desk, world: World, body: str, *, phone: str = CLIENT_PHONE, sid: str | None = None
) -> str:
    """A client texts the clinic: stored like the webhook does, then handled."""
    async with desk.services.sessionmaker() as session:
        tenant = await TenantDirectory(session).get(world.tenant_id)
        msg = await accept_inbound(
            session,
            tenant,
            from_=phone,
            to=CLINIC_NUMBER,
            body=body,
            sid=sid or f"SM{uuid.uuid4().hex[:30]}",
        )
        await session.commit()
    if msg is None:
        return "duplicate"
    return await run_job(
        desk.services, "handle_inbound", tenant_id=str(world.tenant_id), message_id=str(msg.id)
    )


def local(hhmm: str, day: int = 7) -> str:
    """A slot start in the clinic's timezone, as check_availability shows it."""
    return f"2030-01-{day:02d}T{hhmm}-05:00"
