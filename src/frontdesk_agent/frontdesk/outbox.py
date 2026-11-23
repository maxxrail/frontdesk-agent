"""Outbound SMS through an outbox table (ADR 0020).

A text is written as a ``queued`` sms_messages row in the same transaction
as the booking or decision it reports, so the record of it can no longer be
lost. After commit it is sent and marked ``sent`` or ``failed``. A process
that dies between commit and send leaves the row queued; ``sweep`` sends it.

Delivery is at least once: a crash after the provider accepted a text but
before its row was marked sent means the sweeper sends it again. A rare
duplicate confirmation beats a booking the client never hears about.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from frontdesk_agent.db.models import MessageStatus
from frontdesk_agent.db.repositories import SmsMessageRepository, TenantDirectory

if TYPE_CHECKING:
    from frontdesk_agent.frontdesk.ports import SmsSender

logger = logging.getLogger(__name__)

# Rows younger than this belong to a post-commit send that may still be running.
SWEEP_AFTER_S = 60


async def deliver(
    session: AsyncSession, tenant_id: uuid.UUID, message_id: uuid.UUID, sender: "SmsSender"
) -> bool:
    """Send one queued text and record the result. False if it was not ours
    to send (already sent, or locked by another sender)."""
    msg = await SmsMessageRepository(session, tenant_id).lock_queued(message_id)
    if msg is None:
        await session.rollback()
        return False
    try:
        msg.provider_sid = await sender.send(to=msg.to_number, from_=msg.from_number, body=msg.body)
        msg.status = MessageStatus.SENT
    except Exception:  # a failed send must not lose the others or the record
        logger.warning("sms send failed", extra={"message_id": str(message_id)}, exc_info=True)
        msg.status = MessageStatus.FAILED
    status = msg.status
    try:
        await session.commit()
    except Exception:
        # The send is done, but its row would not save (in the Module 7 load
        # test: a provider id that broke a unique index). Never leave it
        # queued: the sweeper would send it again every minute, forever.
        logger.exception("sms result not saved", extra={"message_id": str(message_id)})
        await session.rollback()
        again = await SmsMessageRepository(session, tenant_id).lock_queued(message_id)
        if again is not None:
            again.status = status  # the provider id is lost; the text is not resent
            await session.commit()
    return True


async def sweep(
    sessionmaker: async_sessionmaker[AsyncSession],
    sender: "SmsSender",
    *,
    older_than_s: float = SWEEP_AFTER_S,
) -> int:
    """Send every text a crash left queued. Returns how many were sent or
    marked failed."""
    done = 0
    async with sessionmaker() as session:
        tenant_ids = await TenantDirectory(session).ids()
        for tenant_id in tenant_ids:
            stale = await SmsMessageRepository(session, tenant_id).queued_before(older_than_s)
            await session.rollback()
            for message_id in stale:
                done += await deliver(session, tenant_id, message_id, sender)
    if done:
        logger.warning("outbox sweep sent texts left queued", extra={"count": done})
    return done
