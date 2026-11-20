"""Exceptions to Sentry, with the same redaction as the logs applied first.

Sentry sees more than a log line: tracebacks with local variables, the
request that failed (headers, body), and breadcrumbs built from raw log
records before our formatter runs. So:

- Local variables, request bodies and default PII are never collected
  (``include_local_variables=False``, ``max_request_body_size="never"``,
  ``send_default_pii=False``). The webhook's body is the client's phone
  number and message; an API request's headers carry its key.
- Request headers, cookies and environment are dropped from every event;
  the method and path stay, which is what a debugger needs.
- Everything left, breadcrumbs included, goes through ``logsetup.scrub``:
  personal fields replaced whole, phone numbers masked in every string.

Without a DSN (tests, CI, a laptop without one) nothing is initialised and
nothing is sent. Tracing stays with OpenTelemetry; Sentry gets errors only.
"""

import logging

import sentry_sdk
from sentry_sdk.transport import Transport
from sentry_sdk.types import Event, Hint

from frontdesk_agent import __version__
from frontdesk_agent.config import Settings
from frontdesk_agent.logsetup import scrub

logger = logging.getLogger(__name__)

# Parts of an event's "request" that can hold credentials or personal data.
_REQUEST_DROP = ("headers", "cookies", "data", "env")


def scrub_event(event: Event, hint: Hint | None = None) -> Event:
    """``before_send``: strip request details, then redact everything left."""
    request = event.get("request")
    if isinstance(request, dict):
        for key in _REQUEST_DROP:
            request.pop(key, None)
    cleaned: Event = scrub("event", dict(event))
    return cleaned


def init_sentry(settings: Settings, *, transport: Transport | None = None) -> bool:
    """Start error reporting if a DSN is configured. Returns whether it did.

    ``transport`` lets tests capture what would be sent instead of sending it.
    """
    dsn = settings.sentry_dsn.get_secret_value()
    if not dsn and transport is None:
        return False
    sentry_sdk.init(
        dsn=dsn or "https://public@example.invalid/1",
        transport=transport,
        environment=settings.env,
        release=f"frontdesk-agent@{__version__}",
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        traces_sample_rate=None,
        # Breadcrumbs (raw log records) travel inside the event, so this one
        # hook redacts them too; a breadcrumb is never sent on its own.
        before_send=scrub_event,
    )
    logger.info("sentry error reporting on", extra={"environment": settings.env})
    return True
