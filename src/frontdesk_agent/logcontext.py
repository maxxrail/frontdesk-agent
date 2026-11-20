"""Who a log line is about: request, job, tenant, conversation.

Values live in one ContextVar, so every log line written while handling a
request or a job carries them without passing them around: the redacting
formatter adds them to each JSON line. asyncio copies the context into each
task, so concurrent requests and jobs never see each other's values.

    with bound(request_id=rid):      # the HTTP middleware
        ...
    with bound(job="handle_inbound", tenant_id=..., request_id=...):  # run_job
        bind(conversation_id=...)    # once the conversation is known

``bind`` inside a ``bound`` block needs no reset: leaving the block restores
what was there before it, inner binds included.
"""

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_fields: ContextVar[dict[str, str]] = ContextVar("log_fields", default={})  # noqa: B039

# A client may send its own X-Request-ID; it is echoed into logs, so only a
# short, plain token is accepted.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def current() -> dict[str, str]:
    return _fields.get()


def bind(**fields: str | None) -> None:
    """Add fields to the current context (None values are ignored)."""
    _fields.set({**_fields.get(), **{k: v for k, v in fields.items() if v is not None}})


@contextmanager
def bound(**fields: str | None) -> Iterator[None]:
    token = _fields.set({**_fields.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _fields.reset(token)


def new_request_id() -> str:
    return uuid.uuid4().hex


def request_id_from(header: str | None) -> str:
    """The caller's X-Request-ID if it is a plain token, else a fresh one."""
    if header and _SAFE_ID.fullmatch(header):
        return header
    return new_request_id()
