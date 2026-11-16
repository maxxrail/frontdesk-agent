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
    """

    sms: list[tuple[uuid.UUID, uuid.UUID, str, str, str]] = field(default_factory=list)
    jobs: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)

    def text(
        self, tenant_id: uuid.UUID, conversation_id: uuid.UUID, *, to: str, from_: str, body: str
    ) -> None:
        self.sms.append((tenant_id, conversation_id, to, from_, body))

    def job(self, name: str, *, defer_seconds: float = 0, **kwargs: Any) -> None:
        self.jobs.append((name, defer_seconds, kwargs))

    async def flush(self, session: AsyncSession, sender: SmsSender, queue: JobQueue) -> None:
        """Call after commit. Each sent text is logged as its own row."""
        for tenant_id, conversation_id, to, from_, body in self.sms:
            try:
                sid: str | None = await sender.send(to=to, from_=from_, body=body)
                status = MessageStatus.SENT
            except Exception:  # a failed send must not lose the others or the log
                sid, status = None, MessageStatus.FAILED
            session.add(
                SmsMessage(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    direction=MessageDirection.OUTBOUND,
                    status=status,
                    body=body[:1600],
                    from_number=from_,
                    to_number=to,
                    provider_sid=sid,
                )
            )
        await session.commit()
        for name, defer, kwargs in self.jobs:
            await queue.enqueue(name, defer_seconds=defer, **kwargs)
        self.sms.clear()
        self.jobs.clear()
