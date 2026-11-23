"""A scripted LLM client for tests.

Tests must be deterministic, free and fast. Every test in this repo runs
against this class; nothing in the test suite or in CI ever calls a real model.
"""

import asyncio
from collections import deque
from collections.abc import Iterable

from frontdesk_agent.llm.base import LLMCompletion, LLMError
from frontdesk_agent.llm.messages import (
    AgentTurn,
    Block,
    Message,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)

ScriptedTurn = list[Block] | Exception


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
        self._turns: deque[ScriptedTurn] = deque()
        self.conversations: list[dict[str, object]] = []
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

    # -- tool use ----------------------------------------------------------

    def script(self, *turns: ScriptedTurn) -> None:
        """Queue agent turns: each is a list of blocks, or an exception."""
        self._turns.extend(turns)

    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
        context: str = "",
    ) -> AgentTurn:
        self.conversations.append(
            {
                "system": system,
                "context": context,
                "messages": [m.to_json() for m in messages],
                "tools": [t.name for t in tools],
            }
        )
        await asyncio.sleep(0)
        if not self._turns:
            raise LLMError("FakeLLMClient ran out of scripted agent turns")
        nxt = self._turns.popleft()
        if isinstance(nxt, Exception):
            raise nxt
        has_tools = any(isinstance(b, ToolUseBlock) for b in nxt)
        return AgentTurn(
            content=list(nxt),
            stop_reason="tool_use" if has_tools else "end_turn",
            model=self._model,
            input_tokens=sum(len(str(m.to_json()).split()) for m in messages),
            output_tokens=10,
            latency_ms=self._latency_ms,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )


_ids = iter(range(1, 1_000_000))


def call(name: str, **arguments: object) -> ToolUseBlock:
    """A scripted tool call with a unique id: ``fake.script([call("x", a=1)])``."""
    return ToolUseBlock(id=f"toolu_fake_{next(_ids)}", name=name, input=dict(arguments))


def say(text: str) -> list[Block]:
    """A scripted final reply."""
    return [TextBlock(text)]


def loop_turn() -> list[Block]:
    """A turn that calls a tool and never answers: for step-limit tests."""
    return [call("list_my_appointments")]
