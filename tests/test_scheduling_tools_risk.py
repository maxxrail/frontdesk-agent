"""Availability, tool guards and the risk rules, without the agent loop."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, time, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.agent.loop import trim_history
from frontdesk_agent.agent.risk import classify
from frontdesk_agent.agent.scheduling import free_slots
from frontdesk_agent.agent.tools import TOOLS, ToolContext, execute
from frontdesk_agent.db.models import Client, ProviderSchedule, ReviewKind
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    ConversationRepository,
    ScheduleRepository,
    TenantDirectory,
)
from frontdesk_agent.frontdesk.service import _append
from frontdesk_agent.llm.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from frontdesk_agent.retrieval.embeddings import default_embedder
from tests.desk import NOW, World, local, make_world

ALL = set(TOOLS)


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session)


@pytest.fixture
async def ctx(session: AsyncSession, world: World) -> AsyncIterator[ToolContext]:
    tenant = await TenantDirectory(session).get(world.tenant_id)
    client = await ClientRepository(session, tenant.id).get(world.client_id)
    convo = await ConversationRepository(session, tenant.id).for_client(client.id)
    yield ToolContext(session, tenant, client, convo, default_embedder(), NOW, "SMtest")


async def run(ctx: ToolContext, name: str, **args: object) -> tuple[dict[str, object], bool]:
    out = await execute(ctx, name, dict(args), ALL)
    return json.loads(out.content), out.is_error


# -- availability --------------------------------------------------------------


async def test_slots_respect_schedule_lead_time_and_bookings(
    session: AsyncSession, ctx: ToolContext, world: World
) -> None:
    tenant = ctx.tenant
    slots = await free_slots(session, tenant, date(2030, 1, 7), now=NOW)  # Monday, 9 a.m.
    starts = [s.local(ctx.tz)[11:16] for s in slots]
    assert starts[0] == "10:00"  # 9:00 and 9:30 are inside the one-hour lead time
    assert starts[-1] == "16:30"  # the 30-minute slot that ends at close
    await AppointmentRepository(session, tenant.id).book(
        client_id=world.client_id,
        provider_id=world.provider_id,
        starts_at=NOW + timedelta(hours=2),
        ends_at=NOW + timedelta(hours=2, minutes=30),
    )
    again = [
        s.local(ctx.tz)[11:16] for s in await free_slots(session, tenant, date(2030, 1, 7), now=NOW)
    ]
    assert "11:00" not in again and len(again) == len(starts) - 1
    assert await free_slots(session, tenant, date(2030, 1, 12), now=NOW) == []  # Saturday


async def test_slots_follow_local_time_across_dst(
    session: AsyncSession, ctx: ToolContext, world: World
) -> None:
    await ScheduleRepository(session, ctx.tenant.id).add(
        ProviderSchedule(
            provider_id=world.provider_id, weekday=6, start_time=time(9), end_time=time(10)
        )
    )
    # Toronto springs forward on Sunday 2030-03-10: 9 a.m. local is 13:00 UTC, not 14:00.
    [first, _] = await free_slots(session, ctx.tenant, date(2030, 3, 10), now=NOW)
    assert first.starts_at == datetime(2030, 3, 10, 13, 0, tzinfo=UTC)


# -- tools ---------------------------------------------------------------------


async def test_bad_inputs_are_errors_the_model_can_fix(ctx: ToolContext) -> None:
    body, err = await run(ctx, "check_availability", date="next tuesday")
    assert err and "date" in str(body["error"])
    body, err = await run(ctx, "check_availability", date="2030-01-08", extra="x")
    assert err and "extra" in str(body["error"])
    body, err = await run(ctx, "check_availability", date="2029-12-01")
    assert err and "past" in str(body["error"])
    body, err = await run(ctx, "drop_tables")
    assert err and "Unknown tool" in str(body["error"])
    out = await execute(ctx, "book_appointment", {}, {"search_policy"})  # not allowed here
    assert out.is_error


async def test_booking_only_scheduled_open_slots(ctx: ToolContext, world: World) -> None:
    pid = str(world.provider_id)
    body, err = await run(ctx, "book_appointment", provider_id=pid, starts_at=local("10:15"))
    assert err and "not an open slot" in str(body["error"])
    body, err = await run(ctx, "book_appointment", provider_id=pid, starts_at="2030-01-07T10:00")
    assert err and "offset" in str(body["error"])
    body, err = await run(ctx, "book_appointment", provider_id=pid, starts_at=local("10:00"))
    assert not err and body["booked"] is True
    # Asking again for the same slot fails cleanly: it is no longer open.
    body2, err2 = await run(ctx, "book_appointment", provider_id=pid, starts_at=local("10:00"))
    assert err2 and "not an open slot" in str(body2["error"])


async def test_cannot_cancel_another_clients_appointment(
    session: AsyncSession, ctx: ToolContext, world: World
) -> None:
    other = await ClientRepository(session, world.tenant_id).add(
        Client(first_name="Eve", last_name="X", phone="+14165550222")
    )
    theirs = await AppointmentRepository(session, world.tenant_id).book(
        client_id=other.id,
        provider_id=world.provider_id,
        starts_at=NOW + timedelta(days=3),
        ends_at=NOW + timedelta(days=3, minutes=30),
    )
    body, err = await run(ctx, "cancel_appointment", appointment_id=str(theirs.appointment.id))
    assert err and body["error"] == "No such appointment for this client."
    assert ctx.cancelled == [] and ctx.held == []


async def test_late_cancel_is_held_not_executed(
    session: AsyncSession, ctx: ToolContext, world: World
) -> None:
    mine = await AppointmentRepository(session, world.tenant_id).book(
        client_id=world.client_id,
        provider_id=world.provider_id,
        starts_at=NOW + timedelta(hours=23),
        ends_at=NOW + timedelta(hours=23, minutes=30),
    )
    body, err = await run(ctx, "cancel_appointment", appointment_id=str(mine.appointment.id))
    assert not err and body["status"] == "needs_staff_approval"
    assert [h.kind for h in ctx.held] == [ReviewKind.LATE_CANCELLATION]
    assert mine.appointment.status == "booked"


def test_tool_schemas_are_strict() -> None:
    for tool in TOOLS.values():
        schema = tool.spec.input_schema
        assert schema["additionalProperties"] is False, tool.name
        assert "client_id" not in schema.get("properties", {}), tool.name


# -- risk ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("my gum is bleeding a lot", ReviewKind.MEDICAL),
        ("face is swollen and I can't breathe well", ReviewKind.MEDICAL),
        ("should I take ibuprofen before I come", ReviewKind.MEDICAL),
        ("the pain is getting worse since Tuesday", ReviewKind.MEDICAL),
        ("you charged me twice last month", ReviewKind.MONEY),
        ("can you waive the late fee", ReviewKind.MONEY),
        ("I'd like a refund", ReviewKind.MONEY),
        ("how much is a cleaning", None),
        ("is there a fee if I cancel late", None),
        ("do you take sun life insurance", None),
        ("can I book tuesday at 3", None),
    ],
)
def test_risk_rules(message: str, kind: ReviewKind | None) -> None:
    flag = classify(message)
    assert (flag.kind if flag else None) is kind


# -- transcript ------------------------------------------------------------------


def test_trim_never_starts_on_an_orphaned_tool_result() -> None:
    msgs: list[Message] = []
    for i in range(20):
        msgs += [
            Message.user_text(f"q{i}"),
            Message("assistant", [ToolUseBlock(f"t{i}", "search_policy", {"question": "x"})]),
            Message("user", [ToolResultBlock(f"t{i}", "r")]),
            Message("assistant", [TextBlock(f"a{i}")]),
        ]
    window = trim_history(msgs, keep=10)
    assert window[0].role == "user" and isinstance(window[0].content[0], TextBlock)
    assert len(window) <= 10


def test_append_keeps_roles_alternating() -> None:
    msgs = [Message("user", [ToolResultBlock("t1", "r")])]
    merged = _append(msgs, Message.user_text("new text"))
    assert len(merged) == 1 and len(merged[0].content) == 2
