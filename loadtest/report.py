"""Reply latency for a load test, from the database: what the SLO is about.

    uv run python loadtest/report.py --since-minutes 10

For every inbound text from a load-test number (+1999555...), the time until
the first outbound text in its conversation. Locust cannot see this: the
webhook answers Twilio at once and the reply is sent later by the worker.
A text with no reply is counted, not hidden.
"""

import argparse
from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from frontdesk_agent import aio
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import MessageDirection, MessageStatus, SmsMessage
from frontdesk_agent.db.session import make_engine, make_sessionmaker

LOAD_PREFIX = "+1999555"


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


async def main(since_minutes: float) -> None:
    since = datetime.now(UTC) - timedelta(minutes=since_minutes)
    engine = make_engine(get_settings().database_url)
    inbound, reply = SmsMessage, aliased(SmsMessage)
    # When the first reply after the text was sent: its updated_at, set by the
    # outbox as it records the send. Not created_at: Postgres's now() is the
    # time a transaction began, so an outbox row's created_at is when the job
    # started, which would measure only the queue wait.
    first_reply = (
        select(reply.updated_at)
        .where(
            reply.conversation_id == inbound.conversation_id,
            reply.direction == MessageDirection.OUTBOUND,
            reply.status == MessageStatus.SENT,
            reply.created_at >= inbound.created_at,
        )
        .order_by(reply.created_at)
        .limit(1)
        .scalar_subquery()
    )
    stmt = select(inbound.created_at, inbound.status, first_reply).where(
        inbound.direction == MessageDirection.INBOUND,
        inbound.from_number.startswith(LOAD_PREFIX),
        inbound.created_at >= since,
    )
    async with make_sessionmaker(engine)() as session:
        rows = (await session.execute(stmt)).all()
        queued = await session.scalar(
            select(func.count()).where(
                SmsMessage.status == MessageStatus.QUEUED,
                SmsMessage.to_number.startswith(LOAD_PREFIX),
            )
        )
    await engine.dispose()

    waits = [(r - sent).total_seconds() for sent, _, r in rows if r is not None]
    unanswered = [status.value for _, status, r in rows if r is None]
    print(f"Inbound texts from load-test numbers in the last {since_minutes:g} min: {len(rows)}")
    print(f"Answered: {len(waits)}   Unanswered: {len(unanswered)}   Outbox still queued: {queued}")
    if unanswered:
        print("Unanswered by status:", dict(Counter(unanswered)))
    if waits:
        print(
            "| reply time | p50 | p95 | max |\n|---|---|---|---|\n"
            f"| seconds | {percentile(waits, 0.5):.2f} | {percentile(waits, 0.95):.2f}"
            f" | {max(waits):.2f} |"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--since-minutes", type=float, default=15)
    aio.run(main(p.parse_args().since_minutes))
