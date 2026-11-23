"""Rate limits per phone number and per tenant (Module 6, ADR 0018).

Windows are in database time, so these tests use small limits rather than
moving a clock: every message here arrives within the same few seconds.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import AuditEvent, MessageStatus, ReviewItem, SmsMessage
from frontdesk_agent.frontdesk.ports import Effects
from frontdesk_agent.frontdesk.service import TENANT_LIMIT_RETRY_S, RateLimits
from frontdesk_agent.llm.fake import call, say
from tests.desk import CLIENT_PHONE, Desk, World, make_desk, make_world, text

OTHER_PHONE = "+14165550122"


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    d.services.frontdesk.limits = RateLimits(phone_messages=3, tenant_model_calls=100)
    yield d
    await d.dispose()


async def test_a_flood_from_one_phone_pauses_for_a_person(desk: Desk, world: World) -> None:
    for i in range(3):
        desk.llm.script(say(f"reply {i}"))
        assert await text(desk, world, f"message {i}") == "agent_reply"
    calls = len(desk.llm.conversations)

    assert await text(desk, world, "message 3") == "held_for_review:rate_limit"
    assert await text(desk, world, "message 4") == "held_while_paused"
    assert len(desk.llm.conversations) == calls  # no model call for either
    assert len(desk.sender.bodies(CLIENT_PHONE)) == 3

    async with desk.services.sessionmaker() as s:
        item = await s.scalar(select(ReviewItem))
        assert item is not None and "over the limit of 3" in item.reason
        event = await s.scalar(select(AuditEvent).where(AuditEvent.kind == "guardrail"))
        assert event is not None and event.subject == "rate_limit"


async def test_other_phones_in_the_tenant_are_not_limited(desk: Desk, world: World) -> None:
    for i in range(4):
        desk.llm.script(say("ok"))
        await text(desk, world, f"message {i}")
    desk.llm.script(say("Hi! How can I help?"))
    assert await text(desk, world, "hello", phone=OTHER_PHONE) == "agent_reply"


async def test_a_staff_decision_starts_a_fresh_window(desk: Desk, world: World) -> None:
    for i in range(3):
        desk.llm.script(say("ok"))
        await text(desk, world, f"message {i}")
    await text(desk, world, "message 3")  # held: rate limit
    await text(desk, world, "message 4")  # held while paused

    async with desk.services.sessionmaker() as s:
        item = await s.scalar(select(ReviewItem))
        assert item is not None
        effects = Effects()
        await desk.services.frontdesk.decide(
            s, world.tenant_id, item.id, effects, approve=False, reply=None, reviewer="sam"
        )
        await effects.commit(s)
        await effects.flush(s, desk.sender, desk.queue)

    desk.llm.script(say("Thanks for waiting, how can I help?"))
    assert await desk.queue.drain(desk.services) == ["agent_reply"]  # message 4, answered


async def test_stop_works_whatever_the_rate(desk: Desk, world: World) -> None:
    for i in range(3):
        desk.llm.script(say("ok"))
        await text(desk, world, f"message {i}")
    assert await text(desk, world, "STOP") == "opted_out"


async def test_a_tenant_over_its_model_budget_defers_without_a_model_call(
    desk: Desk, world: World
) -> None:
    desk.services.frontdesk.limits = RateLimits(phone_messages=100, tenant_model_calls=2)
    desk.llm.script([call("list_my_appointments")], say("You have no appointments."))
    assert await text(desk, world, "what do I have booked?") == "agent_reply"  # 2 model calls

    assert await text(desk, world, "hello", phone=OTHER_PHONE) == "rate_limited:tenant"
    assert len(desk.llm.conversations) == 2
    [(job, defer, kwargs)] = desk.queue.jobs
    assert (job, defer) == ("handle_inbound", TENANT_LIMIT_RETRY_S)
    async with desk.services.sessionmaker() as s:
        msg = await s.get(SmsMessage, kwargs["message_id"])
        assert msg is not None and msg.status == MessageStatus.RECEIVED  # not dropped

    # Later, with room again, the deferred message is answered.
    desk.services.frontdesk.limits = RateLimits()
    desk.llm.script(say("Hi! How can I help?"))
    assert await desk.queue.drain(desk.services, include_deferred=True) == ["agent_reply"]
