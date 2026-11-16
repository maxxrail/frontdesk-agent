"""The real queue path: ArqQueue -> Redis -> an arq worker -> run_job.

Skipped locally when Redis is not running; CI provides Redis, and there a
missing Redis fails the build instead of skipping.
"""

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from arq.connections import RedisSettings, create_pool
from arq.worker import Worker
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent import worker as worker_module
from frontdesk_agent.db.repositories import TenantDirectory
from frontdesk_agent.frontdesk.service import accept_inbound
from frontdesk_agent.llm.fake import say
from frontdesk_agent.services import ArqQueue
from tests.desk import CLIENT_PHONE, CLINIC_NUMBER, Desk, World, make_desk, make_world

REDIS_URL = os.environ.get("FDA_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
async def redis_ok() -> AsyncIterator[None]:
    try:
        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        await pool.flushdb()
    except Exception as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"Redis not available: {exc!r}")
    yield
    await pool.flushdb()
    await pool.aclose()


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    yield d
    await d.dispose()


async def test_inbound_turn_runs_on_the_worker(redis_ok: None, desk: Desk, world: World) -> None:
    async with desk.services.sessionmaker() as s:
        tenant = await TenantDirectory(s).get(world.tenant_id)
        msg = await accept_inbound(
            s,
            tenant,
            from_=CLIENT_PHONE,
            to=CLINIC_NUMBER,
            body="when do you open?",
            sid=f"SM{uuid.uuid4().hex}",
        )
        await s.commit()
    assert msg is not None
    desk.llm.script(say("We open at 8 a.m. on weekdays."))

    queue = ArqQueue(REDIS_URL)
    await queue.enqueue("handle_inbound", tenant_id=str(world.tenant_id), message_id=str(msg.id))
    await queue.close()

    async def startup(ctx: dict[str, Any]) -> None:
        ctx["services"] = desk.services  # the fakes, instead of settings-built services

    w = Worker(
        functions=worker_module.WorkerSettings.functions,
        redis_settings=RedisSettings.from_dsn(REDIS_URL),
        on_startup=startup,
        burst=True,  # process what is queued, then stop
        poll_delay=0.05,
        handle_signals=False,
    )
    await w.main()
    await w.close()

    assert (w.jobs_complete, w.jobs_failed) == (1, 0)
    assert desk.sender.bodies(CLIENT_PHONE) == ["We open at 8 a.m. on weekdays."]
