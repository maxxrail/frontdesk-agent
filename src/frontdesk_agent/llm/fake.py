"""A scripted LLM client for tests.

Tests must be deterministic, free and fast. Every test in this repo runs
against this class; nothing in the test suite or in CI ever calls a real model.
"""

import asyncio
from collections import deque
from collections.abc import Iterable

from frontdesk_agent.llm.base import LLMCompletion, LLMError


class FakeLLMClient:
    """Returns queued responses, or raises queued errors, in order.

    Pass strings to script successful replies and exceptions to script
    failures, which is how the retry tests drive timeouts and rate limits.
    """

    def __init__(
        self,
        responses: Iterable[str | Exception] | None = None,
        *,
        model: str = "fake-model",
        latency_ms: int = 1,
    ) -> None:
        self._queue: deque[str | Exception] = deque(responses or [])
        self._model = model
        self._latency_ms = latency_ms
        self.calls: list[dict[str, object]] = []

    def queue(self, *responses: str | Exception) -> None:
        """Add more scripted responses."""
        self._queue.extend(responses)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> LLMCompletion:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "max_tokens": max_tokens,
            }
        )
        await asyncio.sleep(0)

        if not self._queue:
            raise LLMError("FakeLLMClient ran out of scripted responses")

        nxt = self._queue.popleft()
        if isinstance(nxt, Exception):
            raise nxt

        # Token counts are rough stand-ins; only their presence is under test.
        return LLMCompletion(
            text=nxt,
            model=self._model,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(nxt.split()),
            latency_ms=self._latency_ms,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            stop_reason="end_turn",
        )
