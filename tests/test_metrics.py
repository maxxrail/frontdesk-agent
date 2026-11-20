"""Prometheus metrics (Module 7): counted as things happen, state read per
scrape, no ids or personal data in any label."""

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from prometheus_client import REGISTRY
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent import tracing
from frontdesk_agent.config import get_settings
from frontdesk_agent.llm.fake import FakeLLMClient, call, say
from frontdesk_agent.main import create_app
from frontdesk_agent.metrics import sms_outcome
from frontdesk_agent.tracing import TracedLLMClient
from tests.desk import CLIENT_PHONE, Desk, World, make_desk, make_world, text

MODEL = "claude-haiku-4-5-20251001"


def value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_outcome_labels_stay_bounded() -> None:
    assert sms_outcome("agent_reply") == "agent_reply"
    assert sms_outcome("held_for_review:guardrail_other_client") == (
        "held_for_review:guardrail_other_client"
    )
    for odd in ["Robert'); DROP", "x" * 100, "held:for:review", "+14165550111"]:
        assert sms_outcome(odd) == "other"


@pytest.fixture
async def traced(
    db_url: str, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Desk, World, FakeLLMClient]]:
    # Spans off: the metrics must not depend on traces being recorded.
    monkeypatch.setattr(tracing, "tracer", trace.NoOpTracer())
    world = await make_world(session, policies=True)
    desk = make_desk(db_url)
    fake = FakeLLMClient(model=MODEL)
    desk.services.frontdesk.llm = TracedLLMClient(fake)
    yield desk, world, fake
    await desk.dispose()


async def test_an_sms_turn_counts_its_calls_tokens_cost_and_tools(
    traced: tuple[Desk, World, FakeLLMClient],
) -> None:
    desk, world, fake = traced
    prompt = f"agent/{desk.services.frontdesk.prompt_version}"
    before = {
        "turns": value("frontdesk_sms_turns_total", outcome="agent_reply"),
        "seconds": value("frontdesk_sms_turn_seconds_count"),
        "calls": value("frontdesk_llm_calls_total", model=MODEL, prompt=prompt, outcome="ok"),
        "tokens": value("frontdesk_llm_tokens_total", model=MODEL, direction="input"),
        "cost": value("frontdesk_llm_cost_usd_total", model=MODEL, prompt=prompt),
        "tool": value("frontdesk_tool_calls_total", tool="search_policy", outcome="ok"),
        "turn_cost": value("frontdesk_agent_turn_cost_usd_count"),
    }
    fake.script([call("search_policy", question="hours")], say("We open at 9."))
    assert await text(desk, world, "when do you open?") == "agent_reply"

    assert value("frontdesk_sms_turns_total", outcome="agent_reply") == before["turns"] + 1
    assert value("frontdesk_sms_turn_seconds_count") == before["seconds"] + 1
    calls = value("frontdesk_llm_calls_total", model=MODEL, prompt=prompt, outcome="ok")
    assert calls == before["calls"] + 2
    assert value("frontdesk_llm_tokens_total", model=MODEL, direction="input") > before["tokens"]
    assert value("frontdesk_llm_cost_usd_total", model=MODEL, prompt=prompt) > before["cost"]
    assert value("frontdesk_tool_calls_total", tool="search_policy", outcome="ok") == (
        before["tool"] + 1
    )
    assert value("frontdesk_agent_turn_cost_usd_count") == before["turn_cost"] + 1


async def test_a_failed_model_call_is_counted_as_an_error(
    traced: tuple[Desk, World, FakeLLMClient],
) -> None:
    desk, world, fake = traced
    from frontdesk_agent.llm.base import LLMError

    before = value("frontdesk_llm_calls_total", model="unknown", prompt="x/v1", outcome="error")
    fake.script(LLMError("provider down"))
    with pytest.raises(LLMError):
        await desk.services.frontdesk.llm.converse(
            system="s", messages=[], tools=[], prompt_name="x", prompt_version="v1"
        )
    after = value("frontdesk_llm_calls_total", model="unknown", prompt="x/v1", outcome="error")
    assert after == before + 1


@pytest.fixture
def app_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("FDA_EVAL_RESULTS_DIR", str(results))
    monkeypatch.setenv("FDA_JOB_QUEUE", "redis")
    monkeypatch.setenv("FDA_REDIS_URL", "redis://127.0.0.1:1/0")  # nothing listens here
    get_settings.cache_clear()
    yield results
    get_settings.cache_clear()


async def test_metrics_report_state_and_never_ids(
    db_url: str, session: AsyncSession, app_env: Path
) -> None:
    world = await make_world(session)
    for i, (rate, started) in enumerate([(0.70, "2026-11-01"), (0.78, "2026-11-19")]):
        (app_env / f"run{i}.json").write_text(
            json.dumps(
                {
                    "run": {"model": MODEL, "prompt_version": "v2", "started_at": started},
                    "summary": {"pass_rate": rate},
                }
            ),
            encoding="utf-8",
        )
    (app_env / "notes.json").write_text("{}", encoding="utf-8")  # not a results file

    with TestClient(create_app(db_url)) as http:
        http.get("/healthz")
        body = http.get("/metrics").text

    assert 'frontdesk_dependency_up{dependency="postgres"} 1.0' in body
    assert 'frontdesk_dependency_up{dependency="redis"} 0.0' in body  # unreachable
    assert 'frontdesk_review_queue_depth{kind="medical"} 0.0' in body
    assert f'frontdesk_eval_pass_rate{{model="{MODEL}",prompt_version="v2"}} 0.78' in body
    assert 'frontdesk_http_requests_total{method="GET",route="/healthz",status="200"}' in body
    for secret in (str(world.tenant_id), str(world.client_id), CLIENT_PHONE, "Dana"):
        assert secret not in body


async def test_redis_up_is_reported(
    db_url: str, session: AsyncSession, app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    url = os.environ.get("FDA_TEST_REDIS_URL", "redis://localhost:6379/15")
    from redis.asyncio import Redis

    probe = Redis.from_url(url, socket_timeout=0.5)
    try:
        await probe.ping()
    except Exception:
        pytest.skip("no Redis at FDA_TEST_REDIS_URL")
    finally:
        await probe.aclose()
    monkeypatch.setenv("FDA_REDIS_URL", url)
    get_settings.cache_clear()
    with TestClient(create_app(db_url)) as http:
        body = http.get("/metrics").text
    assert 'frontdesk_dependency_up{dependency="redis"} 1.0' in body
