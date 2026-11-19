"""The audit trail: one append-only row per tool call and per review decision."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select, update
from sqlalchemy import text as sql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import AuditEvent, ReviewItem
from frontdesk_agent.db.repositories import AuditRepository, TenantDirectory
from frontdesk_agent.frontdesk.ports import Effects
from frontdesk_agent.llm.fake import call, say
from tests.desk import Desk, World, local, make_desk, make_world, text


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    yield d
    await d.dispose()


async def events(desk: Desk, world: World, kind: str) -> list[AuditEvent]:
    async with desk.services.sessionmaker() as s:
        rows = await AuditRepository(s, world.tenant_id).recent(kind=kind)
    return sorted(rows, key=lambda e: e.seq)


async def test_every_tool_call_is_recorded_with_its_outcome(desk: Desk, world: World) -> None:
    pid = str(world.provider_id)
    desk.llm.script(
        [call("drop_tables")],  # not a tool: rejected
        [call("check_availability", date="next tuesday")],  # invalid input: rejected
        [call("book_appointment", provider_id=pid, starts_at=local("10:15"))],  # not a slot: error
        [call("check_availability", date="2030-01-07")],
        [call("book_appointment", provider_id=pid, starts_at=local("10:00"))],
        say("Booked for Monday at 10:00 AM."),
    )
    assert await text(desk, world, "Monday at 10 please") == "agent_reply"

    rows = await events(desk, world, "tool_call")
    assert [(e.subject, e.outcome) for e in rows] == [
        ("drop_tables", "rejected"),
        ("check_availability", "rejected"),
        ("book_appointment", "error"),
        ("check_availability", "ok"),
        ("book_appointment", "ok"),
    ]
    assert {e.actor for e in rows} == {"agent"}
    assert len({e.conversation_id for e in rows}) == 1 and rows[0].conversation_id is not None
    assert rows[4].detail["input"]["starts_at"] == local("10:00")
    assert "not an open slot" in rows[2].detail["error"]


async def test_a_held_cancellation_is_recorded_as_held(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    from datetime import timedelta

    from frontdesk_agent.db.repositories import AppointmentRepository
    from tests.desk import NOW

    booking = await AppointmentRepository(session, world.tenant_id).book(
        client_id=world.client_id,
        provider_id=world.provider_id,
        starts_at=NOW + timedelta(hours=5),
        ends_at=NOW + timedelta(hours=5, minutes=30),
    )
    await session.commit()
    desk.llm.script(
        [call("cancel_appointment", appointment_id=str(booking.appointment.id))],
        say("I've passed that to the team."),
    )
    await text(desk, world, "cancel my appointment today")
    rows = await events(desk, world, "tool_call")
    assert [(e.subject, e.outcome) for e in rows] == [("cancel_appointment", "held")]


async def test_review_decisions_are_recorded_with_the_reviewer(desk: Desk, world: World) -> None:
    desk.llm.script(say("Swelling can be normal."))
    assert await text(desk, world, "my cheek is swollen after the extraction") == (
        "held_for_review:medical"
    )
    async with desk.services.sessionmaker() as s:
        item = await s.scalar(select(ReviewItem))
    assert item is not None

    desk.llm.script(say("ok"))  # in case the resumed conversation needs a turn
    effects = Effects()
    async with desk.services.sessionmaker() as s:
        await desk.services.frontdesk.decide(
            s,
            world.tenant_id,
            item.id,
            effects,
            approve=True,
            reply="Please call us so the dentist can take a look today.",
            reviewer="sam@maple.example",
        )
        await s.commit()

    [row] = await events(desk, world, "review_decision")
    assert (row.actor, row.subject, row.outcome, row.ref_id) == (
        "sam@maple.example",
        "medical",
        "approved",
        item.id,
    )
    assert row.detail == {"action": None, "reply_edited": True, "replied": True}


async def test_the_table_rejects_updates(desk: Desk, world: World) -> None:
    desk.llm.script([call("check_availability", date="2030-01-07")], say("Monday is open."))
    await text(desk, world, "anything monday?")
    async with desk.services.sessionmaker() as s:
        with pytest.raises(DBAPIError, match="append-only"):
            await s.execute(update(AuditEvent).values(outcome="ok", actor="someone else"))


async def test_one_tenant_cannot_read_anothers_audit_trail(desk: Desk, world: World) -> None:
    desk.llm.script([call("check_availability", date="2030-01-07")], say("Monday is open."))
    await text(desk, world, "anything monday?")
    async with desk.services.sessionmaker() as s:
        other = await TenantDirectory(s).create(
            name="Other", slug="other", business_type="clinic", sms_number="+12005550142"
        )
        await s.commit()
        assert await AuditRepository(s, other.id).recent() == []
        assert len(await AuditRepository(s, world.tenant_id).recent()) == 1
        await s.execute(sql("SELECT 1"))
