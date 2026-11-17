"""Logs never carry a client's phone number or name (Module 6, done-when).

Unit tests pin what the redactor masks and what it must leave alone; the
end-to-end test runs real flows with SQL logging on and searches the output
for every seeded phone number, the way an operator would grep the logs.
"""

import io
import json
import logging
from collections.abc import Iterator

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.agent.risk import MEDICAL_HOLDING_MESSAGE
from frontdesk_agent.llm.fake import call, say
from frontdesk_agent.logsetup import (
    FIELD_MASK,
    PHONE_MASK,
    RedactingFormatter,
    configure_logging,
    redact,
)
from tests.desk import CLIENT_PHONE, CLINIC_NUMBER, Desk, World, local, make_desk, make_world, text


@pytest.mark.parametrize(
    "phone",
    [
        "+14165550111",
        "+442071838750",
        "4165550111",
        "416-555-0111",
        "416.555.0111",
        "416 555 0111",
        "(416) 555-0111",
        "1-416-555-0111",
        "+1 416 555 0111",
    ],
)
def test_phone_numbers_are_masked_in_any_common_format(phone: str) -> None:
    out = redact(f"texted {phone} back")
    assert out == f"texted {PHONE_MASK} back"


@pytest.mark.parametrize(
    "safe",
    [
        "2030-01-08T14:00-05:00",
        "2030-01-08 14:00:00",
        "id 33fbdc96-9436-47d9-baba-4bbd592f8885",
        "cost $185.00, 1709 input tokens, 998 ms",
        "SM3f2a9c41b7d84e0e9f6a1c2b3d4e5f60",
        "job 1700000000",
    ],
)
def test_ordinary_values_are_left_alone(safe: str) -> None:
    assert redact(safe) == safe


def record(msg: str, exc: BaseException | None = None, **extra: object) -> dict[str, object]:
    logger = logging.getLogger("t")
    rec = logger.makeRecord(
        "t", logging.ERROR, "f", 1, msg, (), (type(exc), exc, None) if exc else None, extra=extra
    )
    return dict(json.loads(RedactingFormatter().format(rec)))


def test_personal_fields_are_replaced_whole_and_phones_masked_everywhere() -> None:
    out = record(
        f"reply to {CLIENT_PHONE} failed",
        to=CLIENT_PHONE,
        first_name="Dana",
        body="hi it's Dana, call me on 416-555-0111",
        detail={"from": CLIENT_PHONE, "note": "callback +14165550111"},
        tenant="maple-dental",
    )
    assert out["message"] == f"reply to {PHONE_MASK} failed"
    assert out["to"] == out["first_name"] == out["body"] == FIELD_MASK
    assert out["detail"] == {"from": FIELD_MASK, "note": f"callback {PHONE_MASK}"}
    assert out["tenant"] == "maple-dental"  # not personal: kept


def test_tracebacks_are_redacted() -> None:
    try:
        raise ValueError(f"no client with phone {CLIENT_PHONE}")
    except ValueError as exc:
        out = record("lookup failed", exc)
    assert CLIENT_PHONE not in str(out["exception"]) and PHONE_MASK in str(out["exception"])


def test_configuring_twice_keeps_one_handler() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("INFO")
        configure_logging("INFO")
        ours = [h for h in root.handlers if isinstance(h.formatter, RedactingFormatter)]
        assert len(ours) == 1
    finally:
        root.handlers[:] = before


# -- end to end: grep the logs for seeded phone numbers --------------------------


@pytest.fixture
def captured() -> Iterator[io.StringIO]:
    """Every log line from every logger, at DEBUG, through the real formatter.
    SQL statement logging is on, to prove query parameters stay out."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(RedactingFormatter())
    root, sql = logging.getLogger(), logging.getLogger("sqlalchemy.engine")
    saved = (root.level, sql.level)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    sql.setLevel(logging.INFO)
    try:
        yield buf
    finally:
        root.removeHandler(handler)
        root.setLevel(saved[0])
        sql.setLevel(saved[1])


def formats(phone: str) -> list[str]:
    d = phone[-10:]
    return [
        phone,
        d,
        f"{d[:3]}-{d[3:6]}-{d[6:]}",
        f"({d[:3]}) {d[3:6]}-{d[6:]}",
        f"{d[:3]} {d[3:6]} {d[6:]}",
    ]


async def test_searching_the_logs_for_seeded_phone_numbers_finds_nothing(
    db_url: str, session: AsyncSession, captured: io.StringIO
) -> None:
    world: World = await make_world(session)  # the session fixture emptied the database
    desk: Desk = make_desk(db_url)

    # A booking, a medical hold, and a step-limit stop: SQL inserts of the
    # client, messages and appointments, and warnings from the agent loop.
    desk.llm.script(
        [call("check_availability", date="2030-01-07")],
        [call("book_appointment", provider_id=str(world.provider_id), starts_at=local("10:00"))],
        say("Booked for Monday at 10:00 AM."),
    )
    await text(desk, world, "can I come Monday at 10? or call me back on 416-555-0111")
    desk.llm.script(say("Swelling can be normal."))
    assert await text(desk, world, "my cheek is swollen after the extraction") == (
        "held_for_review:medical"
    )
    assert desk.sender.bodies()[-1] == MEDICAL_HOLDING_MESSAGE
    logging.getLogger("frontdesk_agent.test").error(
        "delivery failed", extra={"to": CLIENT_PHONE, "detail": f"carrier rejected {CLINIC_NUMBER}"}
    )
    # A database error whose message names the clashing phone number:
    # Postgres puts it in the error text itself, which hide_parameters cannot
    # hide, so only the formatter's traceback redaction stands between it and
    # the log.
    try:
        await make_world(session)
    except IntegrityError:
        logging.getLogger("frontdesk_agent.test").exception("seeding failed")
        await session.rollback()
    await desk.services.sessionmaker.kw["bind"].dispose()

    logs = captured.getvalue()
    assert "INSERT INTO" in logs  # SQL logging really was on
    for phone in (CLIENT_PHONE, CLINIC_NUMBER):
        for written in formats(phone):
            assert written not in logs, f"{written} found in the logs"
    assert "Dana" not in logs  # the seeded client's first name
