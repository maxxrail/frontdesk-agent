"""OpenTelemetry traces: one SMS is one trace, with spans for each model
call, each tool call and retrieval, and no personal data in any span."""

from collections.abc import AsyncIterator, Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.config import Settings
from frontdesk_agent.llm.fake import FakeLLMClient, call, say
from frontdesk_agent.tracing import TracedLLMClient, init_tracing
from tests.desk import CLIENT_PHONE, Desk, World, make_desk, make_world, text

_EXPORTER = InMemorySpanExporter()
_registered = False
MODEL = "claude-haiku-4-5-20251001"  # a priced model, so spans carry a cost


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    global _registered
    if not _registered:  # one provider per process; add this exporter once
        init_tracing(Settings(), exporter=_EXPORTER)
        _registered = True
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


@pytest.fixture
async def traced(
    db_url: str, session: AsyncSession
) -> AsyncIterator[tuple[Desk, World, FakeLLMClient]]:
    world = await make_world(session, policies=True)
    desk = make_desk(db_url)
    fake = FakeLLMClient(model=MODEL)
    desk.services.frontdesk.llm = TracedLLMClient(fake)
    yield desk, world, fake
    await desk.dispose()


def by_name(spans: list[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def test_without_an_endpoint_nothing_is_exported() -> None:
    assert init_tracing(Settings(otel_endpoint="")) is False


async def test_one_sms_is_one_trace_with_every_step(
    traced: tuple[Desk, World, FakeLLMClient], spans: InMemorySpanExporter
) -> None:
    desk, world, fake = traced
    fake.script(
        [call("search_policy", question="cancellation policy")],
        say("We ask for 24 hours' notice to cancel."),
    )
    body = "what's your cancellation policy? my number is 416-555-0111"
    assert await text(desk, world, body) == "agent_reply"
    got = list(spans.get_finished_spans())

    [job] = by_name(got, "job handle_inbound")
    [turn] = by_name(got, "agent.turn")
    llm = by_name(got, "llm.call")
    [tool] = by_name(got, "tool search_policy")
    [search] = by_name(got, "retrieval.search")

    assert len({s.context.trace_id for s in got}) == 1  # one trace
    assert job.parent is None
    assert turn.parent is not None and turn.parent.span_id == job.context.span_id
    assert len(llm) == 2 and all(s.parent.span_id == turn.context.span_id for s in llm)  # type: ignore[union-attr]
    assert search.parent is not None and search.parent.span_id == tool.context.span_id

    # From a conversation id to the trace: both carry it.
    convo = job.attributes["conversation.id"]  # type: ignore[index]
    assert turn.attributes["conversation.id"] == convo  # type: ignore[index]

    # Every model call's tokens and cost, and the turn's total.
    for s in llm:
        assert s.attributes["gen_ai.request.model"] == MODEL  # type: ignore[index]
        assert s.attributes["gen_ai.usage.input_tokens"] > 0  # type: ignore[index, operator]
        assert s.attributes["llm.cost_usd"] > 0  # type: ignore[index, operator]
    total = sum(s.attributes["llm.cost_usd"] for s in llm)  # type: ignore[index, misc]
    assert turn.attributes["agent.cost_usd"] == pytest.approx(total)  # type: ignore[index]
    assert tool.attributes["tool.outcome"] == "ok"  # type: ignore[index]

    # No message text, phone number or name in any span.
    dump = " ".join(f"{s.name} {dict(s.attributes or {})}" for s in got)
    for secret in ("cancellation policy", "416-555-0111", "4165550111", CLIENT_PHONE, "Dana"):
        assert secret not in dump, secret


async def test_a_failed_tool_call_marks_its_span(
    traced: tuple[Desk, World, FakeLLMClient], spans: InMemorySpanExporter
) -> None:
    desk, world, fake = traced
    fake.script([call("book_appointment", provider_id="not-a-uuid")], say("Sorry, which time?"))
    await text(desk, world, "book me in")
    [tool] = by_name(list(spans.get_finished_spans()), "tool book_appointment")
    assert tool.attributes["tool.outcome"] == "rejected"  # type: ignore[index]
    assert not tool.status.is_ok


def test_the_real_client_is_always_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    from frontdesk_agent.config import get_settings
    from frontdesk_agent.dependencies import get_llm_client

    monkeypatch.setenv("FDA_ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    get_settings.cache_clear()
    try:
        assert isinstance(get_llm_client(), TracedLLMClient)
    finally:
        get_settings.cache_clear()
