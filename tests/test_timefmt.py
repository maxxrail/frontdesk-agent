"""The date text clients read must be identical on Linux (CI) and Windows (dev)."""

from datetime import datetime

import pytest

from frontdesk_agent.timefmt import human_datetime


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2026, 10, 6, 14, 30), "Tuesday, October 6 at 2:30 PM"),
        (datetime(2026, 10, 6, 0, 5), "Tuesday, October 6 at 12:05 AM"),
        (datetime(2026, 10, 6, 12, 0), "Tuesday, October 6 at 12:00 PM"),
        (datetime(2026, 12, 25, 9, 0), "Friday, December 25 at 9:00 AM"),
    ],
)
def test_no_leading_zeros_on_day_or_hour(when: datetime, expected: str) -> None:
    assert human_datetime(when) == expected


def test_with_year() -> None:
    assert (
        human_datetime(datetime(2026, 10, 6, 14, 30), with_year=True)
        == "Tuesday, October 6, 2026 at 2:30 PM"
    )
