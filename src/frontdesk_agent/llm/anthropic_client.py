"""The real provider client.

Imported lazily by the dependency wiring so the package works, and the tests
run, without the vendor SDK installed.
"""

import time
from typing import Any

from frontdesk_agent.llm.base import (
    LLMCompletion,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from frontdesk_agent.llm.messages import (
    AgentTurn,
    Block,
    Message,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)


class AnthropicLLMClient:
    """Adapts the Anthropic SDK to the LLMClient protocol.

    Errors are translated into this application's exception types so that the
    retry policy never has to know which vendor is behind the interface.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-haiku-4-5-20251001",
        timeout_s: float = 20.0,
        http_client: Any = None,  # tests inject an httpx client with a mock transport
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise LLMError(
                "The anthropic package is not installed. Add it with: uv add anthropic"
            ) from exc

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s, http_client=http_client)
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> LLMCompletion:

        started = time.perf_counter()
        message = await self._create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        text = "".join(block.text for block in message.content if block.type == "text")

        return LLMCompletion(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            stop_reason=message.stop_reason,
        )

    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> AgentTurn:
        started = time.perf_counter()
        response = await self._create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[m.to_json() for m in messages],
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
        )
        content: list[Block] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(block.id, block.name, dict(block.input)))
            # Other block types (thinking, server tools) are not used here.
        return AgentTurn(
            content=content,
            stop_reason=response.stop_reason,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )

    async def _create(self, **kwargs: Any) -> Any:
        """Call the SDK, translating its errors into this application's types."""
        import anthropic

        try:
            return await self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMServerError(str(exc)) from exc
            # 4xx other than rate limiting means the request itself is wrong.
            raise LLMError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMServerError(str(exc)) from exc
