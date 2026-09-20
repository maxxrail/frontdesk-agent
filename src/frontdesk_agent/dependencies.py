"""Dependency wiring.

One place decides which LLM client the application uses, so tests can swap in
the fake with a single dependency override.
"""

from functools import lru_cache

from frontdesk_agent.config import get_settings
from frontdesk_agent.drafting.service import ReminderDrafter
from frontdesk_agent.llm.base import LLMClient, LLMError


@lru_cache
def get_llm_client() -> LLMClient:
    """Build the real provider client from settings."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise LLMError(
            "FDA_ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    from frontdesk_agent.llm.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
    )


def get_drafter() -> ReminderDrafter:
    """The drafting service, wired to the configured provider."""
    return ReminderDrafter(get_llm_client())
