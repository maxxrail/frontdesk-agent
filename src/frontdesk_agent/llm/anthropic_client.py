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
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise LLMError(
                "The anthropic package is not installed. Add it with: uv add anthropic"
            ) from exc

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s)
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
        import anthropic

        started = time.perf_counter()
        try:
            message: Any = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
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
