"""Logging that never writes a client's phone number or name.

Raw message text, phone numbers and names live only in Postgres. Everything
that leaves the process as a log line goes through ``RedactingFormatter``:

- extra fields with personal content (recipient, sender, name, message body)
  are replaced whole, because a pattern cannot recognise a name;
- phone numbers are masked wherever they appear: in the message, in any
  other extra field, and in exception tracebacks;
- SQLAlchemy is told to keep query parameters out of its errors
  (``hide_parameters`` in ``db/session.py``), since a failed insert would
  otherwise print the row's name and phone number in the traceback.

The output is one JSON object per line, so logs stay searchable after
redaction.
"""

import json
import logging
import re
import sys
from typing import Any

PHONE_MASK = "[phone]"
FIELD_MASK = "[redacted]"

# Extra fields whose whole value is personal. Matched case-insensitively.
SENSITIVE_FIELDS = frozenset(
    {
        "to",
        "from",
        "from_",
        "phone",
        "phone_number",
        "body",
        "text",
        "first_name",
        "last_name",
        "client_name",
        "full_name",
    }
)

# E.164 (+ and 10-15 digits), and North American numbers written with or
# without separators: 416 555 0111, (416) 555-0111, 416.555.0111, 4165550111,
# 1-416-555-0111. Area code and exchange start with 2-9 (the NANP rule),
# which keeps Unix timestamps out; word boundaries keep UUID fragments,
# dates, money and token counts out (tests/test_logsetup.py).
_PHONE = re.compile(
    r"\+\d{10,15}\b"
    r"|(?<![\w-])(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?![\w-])"
)

# Attributes every LogRecord has; anything else on a record came from `extra`.
_STANDARD = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}


def redact(text: str) -> str:
    """Mask every phone number in ``text``."""
    return _PHONE.sub(PHONE_MASK, text)


def _clean(key: str, value: Any) -> Any:
    if key.lower() in SENSITIVE_FIELDS:
        return FIELD_MASK
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _clean(str(k), v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(key, v) for v in value]
    return value


class RedactingFormatter(logging.Formatter):
    """One redacted JSON object per record."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                out[key] = _clean(key, value)
        if record.exc_info:
            out["exception"] = redact(self.formatException(record.exc_info))
        if record.stack_info:
            out["stack"] = redact(self.formatStack(record.stack_info))
        return json.dumps(out, default=lambda v: redact(str(v)))


def configure_logging(level: str = "INFO") -> None:
    """Route every logger (ours, uvicorn's, arq's, SQLAlchemy's) through the
    redacting formatter. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter())
    root = logging.getLogger()
    # Replace only our own handler (and Python's unformatted default), so a
    # second call does not double every line and pytest's capture survives.
    root.handlers[:] = [
        h
        for h in root.handlers
        if not isinstance(h.formatter, RedactingFormatter) and type(h) is not logging.StreamHandler
    ] + [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "arq"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
