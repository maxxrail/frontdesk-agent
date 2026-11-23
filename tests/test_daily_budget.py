"""The per-clinic daily model budget (ADR 0019).

Over budget, the agent is not called at all: the client gets the fixed
holding text, and the conversation goes to staff, paused, so a burst of
texts makes one review item rather than one each.
"""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import LLMCall, ReviewItem, ReviewKind, Tenant
from frontdesk_agent.frontdesk.service import BUDGET_HOLDING_MESSAGE
from frontdesk_agent.llm.fake import say
from tests.desk import CLIENT_PHONE, Desk, World, make_desk, make_world, text


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session, policies=True)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    fd = d.services.frontdesk
    fd.limits = replace(fd.limits, tenant_daily_budget_usd=Decimal("1.00"))
    yield d
    await d.dispose()


async def spend(desk: Desk, world: World, usd: str, *, days_ago: int = 0) -> None:
    async with desk.services.sessionmaker() as s:
        row = LLMCall(
            tenant_id=world.tenant_id,
            model="claude-haiku-4-5-20251001",
            prompt_name="agent",
            prompt_version="v3",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=Decimal(usd),
        )
        s.add(row)
        await s.flush()
        if days_ago:
            row.created_at = row.created_at - timedelta(days=days_ago)
        await s.commit()


async def test_under_budget_the_agent_answers(desk: Desk, world: World) -> None:
    await spend(desk, world, "0.99")
    desk.llm.script(say("Hi Dana, how can I help?"))
    assert await text(desk, world, "hi") == "agent_reply"


async def test_over_budget_sends_the_fixed_text_and_hands_off_once(
    desk: Desk, world: World
) -> None:
    await spend(desk, world, "1.00")
    # Nothing scripted: if the agent were called, the fake would raise.
    assert await text(desk, world, "can I book tomorrow?") == "held_for_review:budget"
    assert desk.sender.bodies(CLIENT_PHONE) == [BUDGET_HOLDING_MESSAGE]
    assert "911" in BUDGET_HOLDING_MESSAGE  # the unread message could be urgent
    assert desk.llm.call_count == 0 and not desk.llm.conversations

    assert await text(desk, world, "hello??") == "held_while_paused"
    async with desk.services.sessionmaker() as s:
        kinds = list(await s.scalars(select(ReviewItem.kind)))
    assert kinds == [ReviewKind.BUDGET]


async def test_a_tenant_budget_overrides_the_setting(desk: Desk, world: World) -> None:
    await spend(desk, world, "1.50")
    async with desk.services.sessionmaker() as s:
        tenant = await s.get(Tenant, world.tenant_id)
        assert tenant is not None
        tenant.daily_budget_usd = Decimal("2.00")
        await s.commit()
    desk.llm.script(say("Hi Dana, how can I help?"))
    assert await text(desk, world, "hi") == "agent_reply"


async def test_yesterdays_spend_does_not_count(desk: Desk, world: World) -> None:
    await spend(desk, world, "5.00", days_ago=2)
    desk.llm.script(say("Hi Dana, how can I help?"))
    assert await text(desk, world, "hi") == "agent_reply"
    async with desk.services.sessionmaker() as s:
        assert await s.scalar(select(func.count()).select_from(ReviewItem)) == 0
