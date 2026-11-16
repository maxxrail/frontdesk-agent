"""Waitlist slot filling: offers one at a time, first yes wins, expiry moves on."""

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import (
    Appointment,
    AppointmentStatus,
    Client,
    ClientResponseStats,
    OfferStatus,
    SlotOffer,
    WaitlistEntry,
    WaitlistStatus,
)
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    WaitlistRepository,
)
from frontdesk_agent.frontdesk.slotfill import OFFER_MINUTES, parse_reply
from frontdesk_agent.llm.fake import say
from frontdesk_agent.services import run_job
from tests.desk import NOW, Desk, World, make_desk, make_world, text

SLOT = NOW + timedelta(days=1)  # Tuesday 9:00 a.m. Toronto


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)  # epsilon 0: pure exploitation, deterministic
    yield d
    await d.dispose()


async def waitlist(desk: Desk, world: World, *names: str) -> list[Client]:
    out = []
    async with desk.services.sessionmaker() as s:
        for i, name in enumerate(names):
            c = await ClientRepository(s, world.tenant_id).add(
                Client(first_name=name, last_name="W", phone=f"+1416555{200 + i:04d}")
            )
            await WaitlistRepository(s, world.tenant_id).add(
                WaitlistEntry(client_id=c.id, earliest=NOW, latest=NOW + timedelta(days=7))
            )
            await s.commit()  # separate commits: distinct created_at, so FIFO ties are real
            out.append(c)
    return out


async def open_slot(desk: Desk, world: World) -> str:
    return await run_job(
        desk.services,
        "start_slot_fill",
        tenant_id=str(world.tenant_id),
        provider_id=str(world.provider_id),
        starts_at=SLOT.isoformat(),
        ends_at=(SLOT + timedelta(minutes=30)).isoformat(),
    )


async def rows(desk: Desk, model: Any) -> list[Any]:
    async with desk.services.sessionmaker() as s:
        return list((await s.execute(select(model).order_by(model.created_at))).scalars())


async def test_offer_decline_next_accept(desk: Desk, world: World) -> None:
    ana, ben = await waitlist(desk, world, "Ana", "Ben")
    assert await open_slot(desk, world) == "offered"
    assert desk.sender.bodies(ana.phone)[0].startswith("Hi Ana, a spot with Dr. Osei")
    assert desk.sender.bodies(ben.phone) == []  # one at a time
    assert desk.queue.jobs[0][0] == "expire_offer"
    assert desk.queue.jobs[0][1] == OFFER_MINUTES * 60

    desk.clock.advance(minutes=2)
    assert await text(desk, world, "no thanks", phone=ana.phone) == "offer_declined"
    assert desk.sender.bodies(ben.phone)[0].startswith("Hi Ben")

    desk.clock.advance(minutes=1)
    assert await text(desk, world, "YES", phone=ben.phone) == "offer_accepted"
    assert desk.sender.bodies(ben.phone)[-1].startswith("You're booked for Tuesday")
    [appt] = await rows(desk, Appointment)
    assert appt.client_id == ben.id and appt.starts_at == SLOT
    entries = {e.client_id: e.status for e in await rows(desk, WaitlistEntry)}
    assert entries == {ana.id: WaitlistStatus.WAITING, ben.id: WaitlistStatus.FILLED}
    assert [o.status for o in await rows(desk, SlotOffer)] == [
        OfferStatus.DECLINED,
        OfferStatus.ACCEPTED,
    ]


async def test_no_reply_expires_and_moves_on(desk: Desk, world: World) -> None:
    ana, ben = await waitlist(desk, world, "Ana", "Ben")
    await open_slot(desk, world)
    desk.clock.advance(minutes=OFFER_MINUTES - 1)
    await desk.queue.drain(desk.services, include_deferred=True)  # job fired early: no-op
    assert desk.sender.bodies(ben.phone) == []

    desk.clock.advance(minutes=2)
    # The early run consumed the queued expiry; run it again on time, as arq would.
    [offer] = await rows(desk, SlotOffer)
    await run_job(
        desk.services, "expire_offer", tenant_id=str(world.tenant_id), offer_id=str(offer.id)
    )
    assert desk.sender.bodies(ben.phone)[0].startswith("Hi Ben")
    statuses = [o.status for o in await rows(desk, SlotOffer)]
    assert statuses == [OfferStatus.EXPIRED, OfferStatus.OFFERED]


async def test_a_yes_after_the_slot_was_booked_elsewhere(desk: Desk, world: World) -> None:
    [ana] = await waitlist(desk, world, "Ana")
    await open_slot(desk, world)
    async with desk.services.sessionmaker() as s:  # someone books it by phone meanwhile
        walk_in = await ClientRepository(s, world.tenant_id).add(
            Client(first_name="Walk", last_name="In", phone="+14165550299")
        )
        await AppointmentRepository(s, world.tenant_id).book(
            client_id=walk_in.id,
            provider_id=world.provider_id,
            starts_at=SLOT,
            ends_at=SLOT + timedelta(minutes=30),
        )
        await s.commit()
    assert await text(desk, world, "yes", phone=ana.phone) == "offer_accepted"
    assert "just booked" in desk.sender.bodies(ana.phone)[-1]
    booked = [a for a in await rows(desk, Appointment) if a.status == AppointmentStatus.BOOKED]
    assert [a.client_id for a in booked] == [walk_in.id]  # exactly one booking, no double


async def test_fast_responders_are_offered_first_next_time(desk: Desk, world: World) -> None:
    ana, ben = await waitlist(desk, world, "Ana", "Ben")
    async with desk.services.sessionmaker() as s:  # history: Ben is quick, Ana never answers
        s.add_all(
            [
                ClientResponseStats(
                    tenant_id=world.tenant_id, client_id=ana.id, offers=4, reward_sum=0.0
                ),
                ClientResponseStats(
                    tenant_id=world.tenant_id, client_id=ben.id, offers=4, reward_sum=3.6
                ),
            ]
        )
        await s.commit()
    await open_slot(desk, world)
    assert desk.sender.bodies(ben.phone) and not desk.sender.bodies(ana.phone)


async def test_ordinary_replies_still_go_to_the_agent(desk: Desk, world: World) -> None:
    [ana] = await waitlist(desk, world, "Ana")
    await open_slot(desk, world)
    desk.llm.script(say("We're at 120 Main Street."))
    assert await text(desk, world, "where are you located?", phone=ana.phone) == "agent_reply"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("YES", True),
        ("yes please!", True),
        ("Ok.", True),
        ("no", False),
        ("nope", False),
        ("yes but can I do 10am instead", None),
        ("", None),
    ],
)
def test_parse_reply(body: str, expected: bool | None) -> None:
    assert parse_reply(body) is expected
