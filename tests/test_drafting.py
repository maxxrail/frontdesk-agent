"""Drafting service tests: the happy path and every failure path."""

import json
from datetime import datetime

import pytest

from frontdesk_agent.drafting.schemas import ReminderRequest
from frontdesk_agent.drafting.service import ReminderDrafter
from frontdesk_agent.llm.base import LLMInvalidOutputError, LLMRateLimitError, LLMTimeoutError
from frontdesk_agent.llm.fake import FakeLLMClient
from frontdesk_agent.llm.retry import RetryPolicy

VALID_JSON = json.dumps(
    {
        "message": "Hi Dana, this is a reminder of your appointment on Tuesday, "
        "October 6 at 2:30 PM with Dr. Osei at 120 Main St. Reply YES to confirm "
        "or CALL to reschedule.",
        "confirm_keyword": "YES",
        "reschedule_keyword": "CALL",
    }
)

NO_RETRY = RetryPolicy(max_attempts=1)


@pytest.fixture
def request_payload() -> ReminderRequest:
    return ReminderRequest(
        business_name="Maple Dental",
        business_type="dental office",
        client_first_name="Dana",
        appointment_at=datetime(2026, 10, 6, 14, 30),
        provider_name="Dr. Osei",
        location="120 Main St",
    )


async def test_happy_path_returns_draft_and_metadata(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient([VALID_JSON])

    response = await ReminderDrafter(client, NO_RETRY).draft(request_payload)

    assert "Dana" in response.draft.message
    assert response.draft.confirm_keyword == "YES"
    assert response.prompt_name == "reminder_draft"
    assert response.prompt_version == "v1"
    assert response.input_tokens > 0
    assert response.repaired is False
    assert client.call_count == 1


async def test_prompt_receives_rendered_appointment(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient([VALID_JSON])

    await ReminderDrafter(client, NO_RETRY).draft(request_payload)

    sent_user = str(client.calls[0]["user"])
    assert "Tuesday, October 6 at 2:30 PM" in sent_user
    assert "Maple Dental" in str(client.calls[0]["system"])


async def test_strips_code_fences(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient([f"```json\n{VALID_JSON}\n```"])

    response = await ReminderDrafter(client, NO_RETRY).draft(request_payload)

    assert response.draft.confirm_keyword == "YES"


async def test_repairs_once_after_invalid_json(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient(["Sure! Here is your message.", VALID_JSON])

    response = await ReminderDrafter(client, NO_RETRY).draft(request_payload)

    assert response.repaired is True
    assert client.call_count == 2
    assert "rejected" in str(client.calls[1]["user"])


async def test_gives_up_after_two_invalid_replies(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient(["not json", "still not json"])

    with pytest.raises(LLMInvalidOutputError):
        await ReminderDrafter(client, NO_RETRY).draft(request_payload)

    assert client.call_count == 2


async def test_rejects_reply_missing_required_field(request_payload: ReminderRequest) -> None:
    partial = json.dumps({"message": "Hi Dana", "confirm_keyword": "YES"})
    client = FakeLLMClient([partial, partial])

    with pytest.raises(LLMInvalidOutputError):
        await ReminderDrafter(client, NO_RETRY).draft(request_payload)


async def test_rejects_message_over_sms_limit(request_payload: ReminderRequest) -> None:
    too_long = json.dumps(
        {
            "message": "x" * 301,
            "confirm_keyword": "YES",
            "reschedule_keyword": "CALL",
        }
    )
    client = FakeLLMClient([too_long, too_long])

    with pytest.raises(LLMInvalidOutputError):
        await ReminderDrafter(client, NO_RETRY).draft(request_payload)


async def test_rejects_unexpected_field(request_payload: ReminderRequest) -> None:
    extra = json.dumps(
        {
            "message": "Hi Dana, reply YES to confirm.",
            "confirm_keyword": "YES",
            "reschedule_keyword": "CALL",
            "price": "$180",
        }
    )
    client = FakeLLMClient([extra, extra])

    with pytest.raises(LLMInvalidOutputError):
        await ReminderDrafter(client, NO_RETRY).draft(request_payload)


async def test_retries_transient_provider_errors(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient([LLMRateLimitError("slow down"), VALID_JSON])
    policy = RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter=False)

    response = await ReminderDrafter(client, policy).draft(request_payload)

    assert response.draft.confirm_keyword == "YES"
    assert client.call_count == 2


async def test_propagates_timeout_after_retries(request_payload: ReminderRequest) -> None:
    client = FakeLLMClient([LLMTimeoutError("t"), LLMTimeoutError("t"), LLMTimeoutError("t")])
    policy = RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter=False)

    with pytest.raises(LLMTimeoutError):
        await ReminderDrafter(client, policy).draft(request_payload)

    assert client.call_count == 3


def test_request_rejects_full_name_in_first_name_field() -> None:
    with pytest.raises(ValueError, match="single name"):
        ReminderRequest(
            business_name="Maple Dental",
            business_type="dental office",
            client_first_name="Dana Whitfield",
            appointment_at=datetime(2026, 10, 6, 14, 30),
            provider_name="Dr. Osei",
            location="120 Main St",
        )
