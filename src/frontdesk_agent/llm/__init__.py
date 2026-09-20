"""LLM provider abstraction.

The rest of the application depends on the LLMClient protocol, never on a
vendor SDK. That keeps provider choice a one-file change and lets every test
run against a fake instead of a live model.
"""

from frontdesk_agent.llm.base import (
    LLMClient,
    LLMCompletion,
    LLMError,
    LLMInvalidOutputError,
    LLMTimeoutError,
)
from frontdesk_agent.llm.fake import FakeLLMClient

__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "LLMCompletion",
    "LLMError",
    "LLMInvalidOutputError",
    "LLMTimeoutError",
]
