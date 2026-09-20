"""The provider-agnostic contract every LLM client implements."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class LLMError(Exception):
    """Base class for provider failures the application chooses to handle."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMRateLimitError(LLMError):
    """The provider rejected the call for rate limiting; retrying may succeed."""


class LLMServerError(LLMError):
    """The provider returned a 5xx; retrying may succeed."""


class LLMInvalidOutputError(LLMError):
    """The model answered, but the answer did not match the expected schema."""


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    """One model response, plus the metadata needed to cost and debug it.

    Module 2 persists these fields in the llm_calls table and Module 7 turns
    them into cost and latency dashboards, so they are captured from the start.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    prompt_name: str
    prompt_version: str
    stop_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """What the application needs from a language model.

    Deliberately small. A narrow interface is easy to fake, easy to swap and
    hard to leak provider details through.
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> LLMCompletion:
        """Send one request and return the completion."""
        ...
