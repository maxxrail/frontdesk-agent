"""Twilio: verifying webhooks and sending SMS.

Signature checking uses Twilio's own RequestValidator rather than a hand-rolled
HMAC: the algorithm (URL plus sorted form fields, HMAC-SHA1, base64) has edge
cases around ports and query strings that the library already handles.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


def signature_is_valid(
    *, auth_token: str, url: str, params: dict[str, str], signature: str
) -> bool:
    from twilio.request_validator import RequestValidator

    if not auth_token or not signature:
        return False
    return bool(RequestValidator(auth_token).validate(url, params, signature))


class TwilioSender:
    def __init__(self, account_sid: str, auth_token: str) -> None:
        from twilio.rest import Client

        self._client = Client(account_sid, auth_token)

    async def send(self, *, to: str, from_: str, body: str) -> str:
        # The Twilio client is synchronous; keep it off the event loop.
        message = await asyncio.to_thread(
            self._client.messages.create, to=to, from_=from_, body=body
        )
        return str(message.sid)


class ConsoleSender:
    """Local development: print instead of sending."""

    async def send(self, *, to: str, from_: str, body: str) -> str:
        print(f"\n[SMS {from_} -> {to}]\n{body}\n", flush=True)
        logger.info("console sms", extra={"to": to})
        return "console"
