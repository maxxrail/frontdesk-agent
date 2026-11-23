"""The provider-agnostic contract every LLM client implements."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from frontdesk_agent.llm.messages import AgentTurn, Message, ToolSpec


class LLMError(Exception):
    """Base class for provider failures the application chooses to handle."""


class LLMConfigurationError(LLMError):
    """The provider is not configured (no API key). Not retryable; a 503."""


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
    # Prompt-cache tokens, billed apart from input_tokens (which excludes them).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


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

    async def converse(
        self,
        *,
        system: str,
        messages: "list[Message]",
        tools: "list[ToolSpec]",
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
        context: str = "",
    ) -> "AgentTurn":
        """One step of a tool-using conversation (Module 4's agent loop).

        ``system`` is the same for every message to one clinic, so a provider
        may cache it (with the tools); ``context`` is the part that changes per
        message (who is texting, the time) and is sent after it, uncached.
        """
        ...
