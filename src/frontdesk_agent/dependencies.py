"""Dependency wiring.

One place decides which LLM client the application uses, so tests can swap in
the fake with a single dependency override.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from fastapi import Request

from frontdesk_agent.answering.service import PolicyAnswerer
from frontdesk_agent.config import get_settings
from frontdesk_agent.drafting.service import ReminderDrafter
from frontdesk_agent.llm.base import LLMClient, LLMConfigurationError
from frontdesk_agent.retrieval.embeddings import default_embedder


@lru_cache
def get_llm_client() -> LLMClient:
    """Build the real provider client from settings."""
    return _client(get_settings().llm_model)


@lru_cache
def get_lookup_llm_client() -> LLMClient | None:
    """The cheap model for the agent's lookup steps, or None when routing is off."""
    model = get_settings().llm_lookup_model
    return _client(model) if model else None


def _client(model: str) -> LLMClient:
    settings = get_settings()
    from frontdesk_agent.tracing import TracedLLMClient

    if settings.llm_provider == "fake":
        from frontdesk_agent.llm.fake import CannedLLMClient

        return TracedLLMClient(CannedLLMClient(latency_s=settings.llm_fake_latency_s))
    if not settings.anthropic_api_key:
        raise LLMConfigurationError(
            "FDA_ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    from frontdesk_agent.llm.anthropic_client import AnthropicLLMClient

    # Every model call made through the app is a span (tracing.py).
    return TracedLLMClient(
        AnthropicLLMClient(
            api_key=settings.anthropic_api_key,
            model=model,
            timeout_s=settings.llm_timeout_s,
        )
    )


def get_answerer() -> PolicyAnswerer:
    """The policy answering service, wired to the provider and measured defaults."""
    settings = get_settings()
    return PolicyAnswerer(
        get_llm_client(),
        default_embedder(),
        mode=settings.retrieval_mode,
        k=settings.retrieval_k,
        min_similarity=settings.retrieval_min_similarity,
    )


def get_drafter() -> ReminderDrafter:
    """The drafting service, wired to the configured provider."""
    return ReminderDrafter(get_llm_client())


class LazyLLMClient:
    """Resolves the real client on first use, so the API starts, and the SMS
    webhook stores messages, even before an API key is configured."""

    def __init__(self, resolve: Callable[[], LLMClient | None] = get_llm_client) -> None:
        self._resolve = resolve

    def _get(self) -> LLMClient:
        client = self._resolve()
        if client is None:  # pragma: no cover - only built when a model is set
            raise LLMConfigurationError("no model configured for this client")
        return client

    async def complete(self, **kwargs: Any) -> Any:
        return await self._get().complete(**kwargs)

    async def converse(self, **kwargs: Any) -> Any:
        return await self._get().converse(**kwargs)


def get_services(request: Request) -> "Services":
    """The front desk, sender and queue, built once in the app's lifespan."""
    services: Services = request.app.state.services
    return services


if TYPE_CHECKING:
    from frontdesk_agent.services import Services
