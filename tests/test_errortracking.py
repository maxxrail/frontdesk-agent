"""Sentry gets errors, never phone numbers, names, message text or keys.

A capturing transport stands in for the network: these tests read exactly
what would have been sent to Sentry and search it, like the log test does.
"""

import logging
from collections.abc import Iterator
from typing import Any

import pytest
import sentry_sdk
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from frontdesk_agent.config import Settings
from frontdesk_agent.errortracking import init_sentry, scrub_event
from frontdesk_agent.main import create_app
from tests.test_logsetup import formats

PHONE = "+14165550111"
NAME = "Dana Lee"
API_KEY = "fda_0123456789ab_not-a-real-key-but-shaped-like-one-xxxxxxxx"
MESSAGE = "my tooth has been throbbing since the extraction"


class CapturingTransport(Transport):
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.envelopes: list[Envelope] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)

    def sent(self) -> str:
        return "\n".join(e.serialize().decode("utf-8", "replace") for e in self.envelopes)


@pytest.fixture
def captured() -> Iterator[CapturingTransport]:
    transport = CapturingTransport()
    assert init_sentry(Settings(sentry_dsn=""), transport=transport)  # type: ignore[arg-type]
    try:
        yield transport
    finally:
        sentry_sdk.flush()
        sentry_sdk.init()  # no DSN: reporting off for the rest of the suite


def assert_clean(text: str) -> None:
    for written in formats(PHONE):
        assert written not in text, f"{written} reached Sentry"
    assert NAME not in text and "Dana" not in text
    assert API_KEY not in text
    assert MESSAGE not in text


def test_without_a_dsn_nothing_starts() -> None:
    assert init_sentry(Settings(sentry_dsn="")) is False  # type: ignore[arg-type]


def test_scrub_event_drops_request_details_and_masks_phones() -> None:
    event: Any = {
        "message": f"failed for {PHONE}",
        "request": {
            "url": "http://test/v1/channels/sms",
            "method": "POST",
            "headers": {"X-API-Key": API_KEY},
            "data": {"From": PHONE, "Body": MESSAGE},
        },
        "extra": {"to": PHONE, "first_name": "Dana", "detail": f"carrier rejected {PHONE}"},
    }
    out: Any = scrub_event(event)
    assert set(out["request"]) == {"url", "method"}
    assert out["extra"]["to"] == "[redacted]"
    assert out["extra"]["first_name"] == "[redacted]"
    assert_clean(str(out))


def test_a_failing_request_reaches_sentry_redacted(captured: CapturingTransport) -> None:
    app: FastAPI = create_app()

    async def boom(payload: dict[str, str]) -> None:
        who = NAME  # a local variable, under a name the scrub does not know
        logging.getLogger("frontdesk_agent.test").info(
            "handling", extra={"to": PHONE, "body": MESSAGE}
        )  # becomes a breadcrumb
        raise RuntimeError(f"delivery to {PHONE} failed ({len(who)} chars of name)")

    app.add_api_route("/boom", boom, methods=["POST"])
    with TestClient(app, raise_server_exceptions=False) as http:
        r = http.post(
            "/boom", json={"phone": PHONE, "body": MESSAGE}, headers={"X-API-Key": API_KEY}
        )
    assert r.status_code == 500
    sentry_sdk.flush()

    text = captured.sent()
    assert "RuntimeError" in text and "/boom" in text  # the error did arrive
    assert "[phone]" in text  # and was masked, not dropped
    assert_clean(text)


def test_an_error_log_with_personal_extras_is_redacted(captured: CapturingTransport) -> None:
    log = logging.getLogger("frontdesk_agent.test")
    try:
        raise ValueError(f"Key (tenant_id, phone)=(…, {PHONE}) already exists.")
    except ValueError:
        log.exception("seeding failed", extra={"to": PHONE, "client_name": NAME})
    sentry_sdk.flush()
    text = captured.sent()
    assert "seeding failed" in text and "ValueError" in text
    assert_clean(text)
