"""Request, job, tenant and conversation ids on every log line (Module 7)."""

import asyncio
import io
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.agent.loop import MAX_STEPS
from frontdesk_agent.llm.fake import loop_turn
from frontdesk_agent.logcontext import bind, bound, current, request_id_from
from frontdesk_agent.logsetup import RedactingFormatter
from frontdesk_agent.main import create_app
from tests.desk import Desk, World, make_desk, make_world, text

log = logging.getLogger("frontdesk_agent.test")
Lines = Callable[[], list[dict[str, Any]]]


@pytest.fixture
def lines() -> Iterator[Lines]:
    """A function returning our JSON lines so far, via the real formatter.

    On the package logger, not the root: app startup (configure_logging)
    replaces the root's redacting handlers, which would drop this one.
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(RedactingFormatter())
    ours = logging.getLogger("frontdesk_agent")
    saved = ours.level
    ours.addHandler(handler)
    ours.setLevel(logging.INFO)
    try:
        yield lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]
    finally:
        ours.removeHandler(handler)
        ours.setLevel(saved)


def test_a_plain_caller_id_is_kept_and_anything_else_replaced() -> None:
    assert request_id_from("support-ticket-42") == "support-ticket-42"
    for bad in ["has space", "a\nb", "x" * 65, "", None, '"><script>']:
        rid = request_id_from(bad)
        assert len(rid) == 32 and rid != bad


def test_bound_restores_what_was_there_including_inner_binds() -> None:
    with bound(request_id="outer"):
        with bound(job="j"):
            bind(conversation_id="c")
            assert current() == {"request_id": "outer", "job": "j", "conversation_id": "c"}
        assert current() == {"request_id": "outer"}
    assert current() == {}


async def test_concurrent_tasks_do_not_share_fields() -> None:
    async def one(rid: str) -> dict[str, str]:
        with bound(request_id=rid):
            await asyncio.sleep(0)
            bind(conversation_id=f"c-{rid}")
            await asyncio.sleep(0)
            return dict(current())

    a, b = await asyncio.gather(one("a"), one("b"))
    assert a == {"request_id": "a", "conversation_id": "c-a"}
    assert b == {"request_id": "b", "conversation_id": "c-b"}


def test_every_line_of_a_request_carries_its_id(lines: Lines) -> None:
    app = create_app()

    async def noisy() -> dict[str, bool]:
        log.info("first")
        log.warning("second")
        return {"ok": True}

    app.add_api_route("/noisy", noisy)
    with TestClient(app) as http:
        r = http.get("/noisy", headers={"X-Request-ID": "trace-me-1"})
        r2 = http.get("/noisy")
    assert r.headers["X-Request-ID"] == "trace-me-1"
    other = r2.headers["X-Request-ID"]
    assert len(other) == 32
    ours = [ln for ln in lines() if ln["logger"] == "frontdesk_agent.test"]
    assert [ln["request_id"] for ln in ours] == ["trace-me-1", "trace-me-1", other, other]


@pytest.fixture
async def desk_world(db_url: str, session: AsyncSession) -> AsyncIterator[tuple[Desk, World]]:
    world = await make_world(session)
    d = make_desk(db_url)
    yield d, world
    await d.dispose()


async def test_a_job_logs_its_job_tenant_and_conversation(
    desk_world: tuple[Desk, World], lines: Lines
) -> None:
    desk, world = desk_world
    desk.llm.script(*[loop_turn() for _ in range(MAX_STEPS)])
    assert await text(desk, world, "hello?") == "held_for_review:step_limit"
    [warning] = [ln for ln in lines() if ln["message"] == "agent hit the step limit"]
    assert warning["job"] == "handle_inbound"
    assert warning["tenant_id"] == str(world.tenant_id)
    assert len(str(warning["conversation_id"])) == 36  # a UUID
    assert len(str(warning["request_id"])) == 32
    assert current() == {}  # nothing leaks out of the job
