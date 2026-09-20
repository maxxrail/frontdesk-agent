"""Retry policy for provider calls.

Rate limits and 5xx responses are worth retrying; a malformed request is not.
Backoff is exponential with jitter, because synchronised retries from many
workers are how a struggling provider gets pushed over.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from frontdesk_agent.llm.base import (
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)

RETRYABLE: tuple[type[LLMError], ...] = (
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to retry and how long to wait between attempts."""

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Delay before the given 1-indexed attempt's retry."""
        raw = float(min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s))
        if not self.jitter:
            return raw
        # Full jitter: pick uniformly in [0, raw] to spread retries out.
        return float(random.uniform(0, raw))  # noqa: S311 - not security sensitive


async def with_retries[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run an operation, retrying only the failures that are worth retrying.

    `sleep` is injectable so tests can assert on backoff without waiting.
    """
    policy = policy or RetryPolicy()
    last_error: LLMError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except RETRYABLE as exc:
            last_error = exc
            if attempt == policy.max_attempts:
                break
            await sleep(policy.delay_for(attempt))

    assert last_error is not None
    raise last_error
