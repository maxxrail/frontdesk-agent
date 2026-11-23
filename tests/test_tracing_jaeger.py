"""Done when: from one conversation ID you can open the full trace, with
every span's latency and cost. Against a real Jaeger, so opt-in:

    docker compose up -d jaeger
    FDA_TEST_JAEGER_URL=http://localhost:16686 uv run pytest tests/test_tracing_jaeger.py

The model is the scripted fake, so this costs nothing.
"""

import asyncio
import json
import os
import urllib.parse
import urllib.request
from typing import Any

import pytest
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.config import Settings
from frontdesk_agent.llm.fake import FakeLLMClient, call, say
from frontdesk_agent.tracing import TracedLLMClient, init_tracing
from tests.desk import make_desk, make_world, text

JAEGER = os.environ.get("FDA_TEST_JAEGER_URL", "")
pytestmark = pytest.mark.skipif(not JAEGER, reason="set FDA_TEST_JAEGER_URL to run")


def _get(path: str) -> Any:
    with urllib.request.urlopen(f"{JAEGER.rstrip('/')}{path}", timeout=5) as r:  # noqa: S310
        return json.load(r)


async def test_a_conversation_id_finds_its_full_trace_in_jaeger(
    db_url: str, session: AsyncSession
) -> None:
    otlp = JAEGER.replace(":16686", ":4318")
    init_tracing(Settings(otel_endpoint=otlp, service_name="frontdesk-agent-test"))
    world = await make_world(session, policies=True)
    desk = make_desk(db_url)
    fake = FakeLLMClient(model="claude-haiku-4-5-20251001")
    desk.services.frontdesk.llm = TracedLLMClient(fake)
    fake.script(
        [call("search_policy", question="cancellation policy")],
        say("We ask for 24 hours' notice."),
    )
    await text(desk, world, "what's your cancellation policy?")
    await desk.dispose()
    trace.get_tracer_provider().force_flush()  # type: ignore[attr-defined]

    async with desk.services.sessionmaker() as s:
        from sqlalchemy import select

        from frontdesk_agent.db.models import Conversation

        convo_id = str(await s.scalar(select(Conversation.id)))

    tags = urllib.parse.quote(json.dumps({"conversation.id": convo_id}))
    found: list[Any] = []
    for _ in range(20):  # Jaeger indexes asynchronously
        found = _get(f"/api/traces?service=frontdesk-agent-test&tags={tags}&limit=5")["data"]
        if found:
            break
        await asyncio.sleep(0.5)
    assert len(found) == 1, "no trace found for the conversation id"
    spans = found[0]["spans"]
    names = {sp["operationName"] for sp in spans}
    assert {"job handle_inbound", "agent.turn", "llm.call", "tool search_policy"} <= names
    # Every span has a latency. The fake model answers in microseconds, under
    # the Windows clock's tick, so only the whole job must be measurably > 0.
    assert all(isinstance(sp["duration"], int) for sp in spans)
    [job] = [sp for sp in spans if sp["operationName"] == "job handle_inbound"]
    assert job["duration"] > 0
    costs = [t["value"] for sp in spans for t in sp["tags"] if t["key"] == "llm.cost_usd"]
    assert len(costs) == 2 and all(c > 0 for c in costs)  # every model call has a cost
    print(f"\ntrace {found[0]['traceID']}: {len(spans)} spans, ${sum(costs):.6f}")
