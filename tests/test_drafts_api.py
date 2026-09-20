"""Endpoint tests, with the fake client injected via dependency override."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from frontdesk_agent.dependencies import get_drafter
from frontdesk_agent.drafting.service import ReminderDrafter
from frontdesk_agent.llm.base import LLMServerError
from frontdesk_agent.llm.fake import FakeLLMClient
from frontdesk_agent.llm.retry import RetryPolicy
from frontdesk_agent.main import create_app

VALID_JSON = json.dumps(
    {
        "message": "Hi Dana, reminder of your appointment on Tuesday, October 6 "
        "at 2:30 PM with Dr. Osei. Reply YES to confirm or CALL to reschedule.",
        "confirm_keyword": "YES",
        "reschedule_keyword": "CALL",
    }
)

PAYLOAD = {
    "business_name": "Maple Dental",
    "business_type": "dental office",
    "client_first_name": "Dana",
    "appointment_at": "2026-10-06T14:30:00",
    "provider_name": "Dr. Osei",
    "location": "120 Main St",
}

NO_RETRY = RetryPolicy(max_attempts=1)


def client_with(*responses: str | Exception) -> Iterator[TestClient]:
    app = create_app()
    fake = FakeLLMClient(list(responses))
    app.dependency_overrides[get_drafter] = lambda: ReminderDrafter(fake, NO_RETRY)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ok_client() -> Iterator[TestClient]:
    yield from client_with(VALID_JSON)


def test_returns_draft(ok_client: TestClient) -> None:
    response = ok_client.post("/v1/drafts/reminder", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert "Dana" in body["draft"]["message"]
    assert body["prompt_version"] == "v1"
    assert body["repaired"] is False


def test_rejects_missing_field(ok_client: TestClient) -> None:
    payload = {k: v for k, v in PAYLOAD.items() if k != "client_first_name"}

    assert ok_client.post("/v1/drafts/reminder", json=payload).status_code == 422


def test_rejects_bad_timestamp(ok_client: TestClient) -> None:
    payload = PAYLOAD | {"appointment_at": "next tuesday"}

    assert ok_client.post("/v1/drafts/reminder", json=payload).status_code == 422


def test_model_garbage_becomes_502() -> None:
    for test_client in client_with("not json", "still not json"):
        response = test_client.post("/v1/drafts/reminder", json=PAYLOAD)

        assert response.status_code == 502
        assert "usable draft" in response.json()["detail"]


def test_provider_outage_becomes_502() -> None:
    for test_client in client_with(LLMServerError("upstream down")):
        response = test_client.post("/v1/drafts/reminder", json=PAYLOAD)

        assert response.status_code == 502
        assert "unavailable" in response.json()["detail"]
