"""The real Anthropic SDK, driven through a mocked HTTP transport.

No network and no key: httpx's MockTransport answers the SDK's request. This
checks what unit tests with the fake client cannot: that our adapter builds
a request the SDK sends correctly, and parses the SDK's real response types.
"""

import json
from typing import Any

import httpx2 as httpx  # the anthropic SDK 1.x is built on httpx2
import pytest

from frontdesk_agent.llm.anthropic_client import AnthropicLLMClient
from frontdesk_agent.llm.base import LLMError, LLMRateLimitError, LLMServerError
from frontdesk_agent.llm.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from frontdesk_agent.llm.thinking import thinking_off

TOOL_RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "check_availability",
            "input": {"date": "2030-01-07"},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 321, "output_tokens": 45},
}


def client_with(handler: Any) -> AnthropicLLMClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicLLMClient(api_key="test-key", http_client=http)


async def test_converse_round_trip_through_the_real_sdk() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=TOOL_RESPONSE)

    turn = await client_with(handler).converse(
        system="You are a front desk.",
        messages=[
            Message.user_text("any openings monday?"),
            Message("assistant", [ToolUseBlock("toolu_00", "search_policy", {"question": "x"})]),
            Message("user", [ToolResultBlock("toolu_00", "no results", is_error=True)]),
        ],
        tools=[ToolSpec("check_availability", "Find open slots.", {"type": "object"})],
        prompt_name="agent",
        prompt_version="v1",
    )

    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert body["system"] == "You are a front desk."
    assert body["tools"] == [
        {
            "name": "check_availability",
            "description": "Find open slots.",
            "input_schema": {"type": "object"},
        }
    ]
    assert body["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_00",
        "content": "no results",
        "is_error": True,
    }

    assert turn.stop_reason == "tool_use"
    assert turn.content == [
        TextBlock("Let me check."),
        ToolUseBlock("toolu_01", "check_availability", {"date": "2030-01-07"}),
    ]
    assert (turn.input_tokens, turn.output_tokens) == (321, 45)
    assert turn.model == "claude-haiku-4-5-20251001"


async def test_thinking_is_off_by_default_in_each_models_own_form() -> None:
    """Newer models think by default and it counts against max_tokens (llm/thinking.py)."""
    bodies: list[dict[str, Any]] = []
    text_reply = {
        **TOOL_RESPONSE,
        "content": [{"type": "text", "text": "{}"}],
        "stop_reason": "end_turn",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=text_reply)

    for model in ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-haiku-4-5-20251001"):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c = AnthropicLLMClient("k", model=model, http_client=http)
        await c.complete(system="s", user="u", prompt_name="p", prompt_version="v1")
    assert bodies[0]["thinking"] == {"type": "between_tools"}
    assert bodies[1]["thinking"] == {"type": "disabled"}
    assert "thinking" not in bodies[2]  # Haiku 4.5 does not think unless asked
    assert thinking_off("claude-opus-5-5") is None  # cannot be switched off; not used here


@pytest.mark.parametrize(
    ("status", "error"), [(429, LLMRateLimitError), (529, LLMServerError), (400, LLMError)]
)
async def test_provider_errors_are_translated(status: int, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"type": "error", "error": {"type": "x", "message": "nope"}}
        )

    client = client_with(handler)
    client._client = client._client.with_options(max_retries=0)  # our retry policy owns retries
    with pytest.raises(error):
        await client.converse(
            system="s",
            messages=[Message.user_text("hi")],
            tools=[],
            prompt_name="agent",
            prompt_version="v1",
        )


def test_messages_survive_json_storage() -> None:
    original = [
        Message.user_text("hi"),
        Message("assistant", [TextBlock("ok"), ToolUseBlock("t1", "x", {"a": 1})]),
        Message("user", [ToolResultBlock("t1", "done")]),
    ]
    stored = json.loads(json.dumps([m.to_json() for m in original]))
    assert [Message.from_json(m) for m in stored] == original
