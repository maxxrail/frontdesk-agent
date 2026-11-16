"""Human-readable dates for SMS text and prompts.

strftime's no-padding flags (``%-d``, ``%-I``) are a glibc extension: they work
on Linux (CI) and raise ``ValueError`` on Windows (the dev machine). Building
the day and hour from integers gives the same text on both.
"""

from datetime import datetime


def human_datetime(when: datetime, *, with_year: bool = False) -> str:
    """'Tuesday, October 6 at 2:30 PM', or with the year after the day."""
    hour = when.hour % 12 or 12
    day = f"{when:%A, %B} {when.day}"
    if with_year:
        day += f", {when.year}"
    return f"{day} at {hour}:{when:%M %p}"
