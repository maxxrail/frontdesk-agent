"""The background worker: inbound SMS turns, slot filling, offer expiry.

    uv run arq frontdesk_agent.worker.WorkerSettings

Set FDA_JOB_QUEUE=redis on the API so it enqueues here instead of running
jobs inline. Jobs are retried by arq on failure; every job is idempotent
(handled messages and closed offers are skipped), so a retry is safe.
"""

from typing import Any, ClassVar

from arq.connections import RedisSettings
from prometheus_client import start_http_server

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.dependencies import get_llm_client
from frontdesk_agent.errortracking import init_sentry
from frontdesk_agent.logsetup import configure_logging
from frontdesk_agent.services import Services, build_services, run_job
from frontdesk_agent.tracing import init_tracing


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    # A failed job is logged at ERROR by arq; Sentry's logging integration
    # turns that into an event, redacted like every other.
    init_sentry(settings)
    init_tracing(settings)
    if settings.metrics_port:
        # The worker makes the model and tool calls; Prometheus scrapes it here.
        start_http_server(settings.metrics_port)
    engine = make_engine(settings.database_url)
    ctx["engine"] = engine
    ctx["services"] = build_services(settings, make_sessionmaker(engine), get_llm_client())


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["engine"].dispose()


def _job(name: str) -> Any:
    async def job(ctx: dict[str, Any], **kwargs: Any) -> str:
        services: Services = ctx["services"]
        return await run_job(services, name, **kwargs)

    # arq registers a function under its __qualname__, which for this closure
    # would be "_job.<locals>.job"; every enqueued job would then fail with
    # "function not found". tests/test_worker_redis.py guards this.
    job.__name__ = job.__qualname__ = name
    return job


handle_inbound = _job("handle_inbound")
start_slot_fill = _job("start_slot_fill")
expire_offer = _job("expire_offer")


class WorkerSettings:
    functions: ClassVar[list[Any]] = [handle_inbound, start_slot_fill, expire_offer]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_tries = 3
    # An agent turn is a few model calls; anything near this is stuck.
    job_timeout = 120
