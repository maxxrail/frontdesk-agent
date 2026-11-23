"""Model routing (ADR 0019): a cheap model runs lookup-only steps; the main
model takes every step that replies or acts, and the rest of that turn.

The guarantee that matters is the second half: nothing the cheap model
decides is booked, cancelled, escalated or texted.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Appointment, AppointmentStatus, LLMCall
from frontdesk_agent.llm.fake import FakeLLMClient, call, say
from tests.desk import CLIENT_PHONE, Desk, World, local, make_desk, make_world, text


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session, policies=True)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[tuple[Desk, FakeLLMClient]]:
    d = make_desk(db_url)
    lookup = FakeLLMClient(model="lookup-model")
    d.services.frontdesk.lookup_llm = lookup
    yield d, lookup
    await d.dispose()


async def models_recorded(d: Desk) -> list[str]:
    async with d.services.sessionmaker() as s:
        rows = await s.scalars(select(LLMCall.model).order_by(LLMCall.created_at))
        return list(rows)


async def test_a_lookup_runs_on_the_cheap_model_and_the_reply_on_the_main_one(
    desk: tuple[Desk, FakeLLMClient], world: World
) -> None:
    d, lookup = desk
    lookup.script(
        [call("check_availability", date="2030-01-07")],
        say("cheap draft, never sent"),
    )
    d.llm.script(say("We have 11:00 or 2:00 today. Which suits you?"))

    assert await text(d, world, "anything today?") == "agent_reply"

    assert d.sender.bodies(CLIENT_PHONE)[-1].startswith("We have 11:00")
    # The main model saw the cheap model's lookup and its result.
    assert "check_availability" in str(d.llm.conversations[0]["messages"])
    # Every call is recorded, the discarded draft included.
    assert sorted(await models_recorded(d)) == ["fake-model", "lookup-model", "lookup-model"]


async def test_an_action_drafted_by_the_cheap_model_is_redone_by_the_main_one(
    desk: tuple[Desk, FakeLLMClient], world: World
) -> None:
    d, lookup = desk
    book = call("book_appointment", provider_id=str(world.provider_id), starts_at=local("11:00"))
    lookup.script([book])  # the cheap model's booking must not run
    d.llm.script(
        [call("book_appointment", provider_id=str(world.provider_id), starts_at=local("13:00"))],
        say("You're booked for 1:00 PM today."),
    )

    assert await text(d, world, "book me today at 1pm") == "agent_reply"

    async with d.services.sessionmaker() as s:
        booked = list(
            await s.scalars(
                select(Appointment.starts_at).where(Appointment.status == AppointmentStatus.BOOKED)
            )
        )
    assert len(booked) == 1  # only the main model's 13:00, never the draft's 11:00
    # After a handoff the main model keeps the turn: one draft, two main calls.
    assert len(lookup.conversations) == 1 and len(d.llm.conversations) == 2


async def test_a_lookup_mixed_with_an_action_is_handed_off(
    desk: tuple[Desk, FakeLLMClient], world: World
) -> None:
    d, lookup = desk
    lookup.script(
        [call("list_my_appointments"), call("escalate_to_human", reason="x", summary="y")]
    )
    d.llm.script(say("Happy to help. What would you like to do?"))

    assert await text(d, world, "hi") == "agent_reply"

    async with d.services.sessionmaker() as s:
        assert await s.scalar(select(func.count()).select_from(LLMCall)) == 2
    assert d.sender.bodies(CLIENT_PHONE)[-1].startswith("Happy to help")
