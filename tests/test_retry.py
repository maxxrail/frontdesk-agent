"""Retry policy tests.

Sleep is injected, so these assert on backoff behaviour without waiting.
"""

import pytest

from frontdesk_agent.llm.base import LLMError, LLMRateLimitError, LLMTimeoutError
from frontdesk_agent.llm.retry import RetryPolicy, with_retries


async def test_returns_immediately_on_success() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await with_retries(op) == "ok"
    assert calls == 1


async def test_retries_rate_limit_then_succeeds() -> None:
    attempts = 0
    slept: list[float] = []

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LLMRateLimitError("slow down")
        return "ok"

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    result = await with_retries(op, RetryPolicy(max_attempts=3), sleep=fake_sleep)

    assert result == "ok"
    assert attempts == 3
    assert len(slept) == 2


async def test_gives_up_after_max_attempts() -> None:
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        raise LLMTimeoutError("too slow")

    async def fake_sleep(seconds: float) -> None:
        return None

    with pytest.raises(LLMTimeoutError):
        await with_retries(op, RetryPolicy(max_attempts=4), sleep=fake_sleep)

    assert attempts == 4


async def test_does_not_retry_non_retryable_errors() -> None:
    attempts = 0

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        raise LLMError("bad request")

    with pytest.raises(LLMError):
        await with_retries(op, RetryPolicy(max_attempts=3))

    assert attempts == 1


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0, jitter=False)

    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(9) == 4.0


def test_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(base_delay_s=2.0, max_delay_s=8.0, jitter=True)

    for _ in range(50):
        assert 0.0 <= policy.delay_for(2) <= 4.0
