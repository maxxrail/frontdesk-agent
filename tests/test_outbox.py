"""The SMS outbox (ADR 0020): a text is recorded in the transaction that
causes it, and a crash between commit and send no longer loses it."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Appointment, MessageDirection, MessageStatus, SmsMessage
from frontdesk_agent.frontdesk.outbox import deliver, sweep
from frontdesk_agent.frontdesk.ports import Effects
from frontdesk_agent.llm.fake import call, say
from tests.desk import CLIENT_PHONE, Desk, World, local, make_desk, make_world, text


class ProcessDied(BaseException):
    """Not an Exception: nothing in the app catches it, like a killed process."""


class DiesOnSend:
    async def send(self, *, to: str, from_: str, body: str) -> str:
        raise ProcessDied


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session, policies=True)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    yield d
    await d.dispose()


async def outbound(d: Desk) -> list[SmsMessage]:
    async with d.services.sessionmaker() as s:
        stmt = select(SmsMessage).where(SmsMessage.direction == MessageDirection.OUTBOUND)
        return list(await s.scalars(stmt))


async def test_a_crash_between_commit_and_send_leaves_the_text_for_the_sweeper(
    desk: Desk, world: World
) -> None:
    real_sender = desk.sender
    desk.services.sender = DiesOnSend()
    desk.llm.script(
        [call("book_appointment", provider_id=str(world.provider_id), starts_at=local("11:00"))],
        say("You're booked for 11:00 AM today."),
    )
    with pytest.raises(ProcessDied):
        await text(desk, world, "book me today at 11")

    # The booking and the text that reports it committed together.
    async with desk.services.sessionmaker() as s:
        assert await s.scalar(select(Appointment.id)) is not None
    [row] = await outbound(desk)
    assert row.status == MessageStatus.QUEUED and not real_sender.sent

    # A fresh row belongs to a send that may still be running: left alone.
    assert await sweep(desk.services.sessionmaker, real_sender) == 0
    assert await sweep(desk.services.sessionmaker, real_sender, older_than_s=0) == 1
    [row] = await outbound(desk)
    assert row.status == MessageStatus.SENT and row.provider_sid
    assert real_sender.bodies(CLIENT_PHONE) == ["You're booked for 11:00 AM today."]

    # Sent is final: another sweep sends nothing.
    assert await sweep(desk.services.sessionmaker, real_sender, older_than_s=0) == 0
    assert len(real_sender.sent) == 1


async def test_a_text_is_sent_once_even_if_delivered_twice(desk: Desk, world: World) -> None:
    desk.llm.script(say("We open at 8."))
    await text(desk, world, "when do you open")
    [row] = await outbound(desk)
    assert row.status == MessageStatus.SENT
    async with desk.services.sessionmaker() as s:
        assert await deliver(s, row.tenant_id, row.id, desk.sender) is False
    assert len(desk.sender.sent) == 1


async def test_flush_without_commit_refuses_to_drop_texts(desk: Desk, world: World) -> None:
    effects = Effects()
    effects.text(world.tenant_id, world.tenant_id, to=CLIENT_PHONE, from_="+15550000000", body="x")
    async with desk.services.sessionmaker() as s:
        with pytest.raises(RuntimeError, match="before Effects.commit"):
            await effects.flush(s, desk.sender, desk.queue)
