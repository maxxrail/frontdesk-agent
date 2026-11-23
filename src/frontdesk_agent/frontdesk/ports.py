"""What the front desk needs from the outside world, as small interfaces.

Production wires Twilio and Redis; tests and the local console wire fakes.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import MessageDirection, MessageStatus, SmsMessage


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class SmsSender(Protocol):
    async def send(self, *, to: str, from_: str, body: str) -> str:
        """Send one SMS and return the provider's message id."""
        ...


class JobQueue(Protocol):
    async def enqueue(self, job: str, *, defer_seconds: float = 0, **kwargs: Any) -> None: ...


@dataclass
class Effects:
    """Side effects collected during a transaction and run only after commit.

    Sending a text for a booking that then rolls back, or queueing a slot
    fill for a cancellation that never committed, is worse than a delay.
    Texts go through the outbox (frontdesk/outbox.py, ADR 0020): ``commit``
    writes them as queued rows inside the transaction, ``flush`` sends them.
    """

    sms: list[tuple[uuid.UUID, uuid.UUID, str, str, str]] = field(default_factory=list)
    jobs: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)
    _outbox: list[SmsMessage] | None = None

    def text(
        self, tenant_id: uuid.UUID, conversation_id: uuid.UUID, *, to: str, from_: str, body: str
    ) -> None:
        self.sms.append((tenant_id, conversation_id, to, from_, body))

    def job(self, name: str, *, defer_seconds: float = 0, **kwargs: Any) -> None:
        self.jobs.append((name, defer_seconds, kwargs))

    async def commit(self, session: AsyncSession) -> None:
        """Commit the transaction, with every text in it as a queued row."""
        self._outbox = [
            SmsMessage(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                direction=MessageDirection.OUTBOUND,
                status=MessageStatus.QUEUED,
                body=body[:1600],
                from_number=from_,
                to_number=to,
            )
            for tenant_id, conversation_id, to, from_, body in self.sms
        ]
        session.add_all(self._outbox)
        await session.commit()

    async def flush(self, session: AsyncSession, sender: SmsSender, queue: JobQueue) -> None:
        """Call after ``commit``: send the queued texts, then queue the jobs."""
        from frontdesk_agent.frontdesk.outbox import deliver

        if self._outbox is None:
            if self.sms:
                raise RuntimeError("Effects.flush before Effects.commit: texts would be lost")
            self._outbox = []
        for row in self._outbox:
            await deliver(session, row.tenant_id, row.id, sender)
        for name, defer, kwargs in self.jobs:
            await queue.enqueue(name, defer_seconds=defer, **kwargs)
        self.sms.clear()
        self.jobs.clear()
