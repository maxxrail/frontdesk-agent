"""Eval cases: what a client texts, the clinic they text, and what must happen.

Cases live in evals/cases/*.jsonl, one JSON object per line. The models are
strict (unknown keys rejected), so a misspelt expectation fails when the file
loads instead of silently checking nothing.

Times are local to the clinic (America/Toronto) and the clock is fixed at
Monday 2030-01-07, 9:00 a.m., so "tomorrow" is always Tuesday 2030-01-08.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Feature = Literal["booking", "cancel", "policy", "new_client", "risk", "conversation", "redteam"]
Difficulty = Literal["easy", "medium", "hard"]
TenantSlug = Literal["maple-dental", "northside-physio"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Slot(_Strict):
    """A local appointment time, e.g. {"at": "2030-01-08T10:00", "provider": 0}."""

    at: str = Field(pattern=r"^2030-\d\d-\d\dT\d\d:\d\d$")
    provider: int = Field(default=0, ge=0, le=1)


class PolicyDoc(_Strict):
    """An extra policy document for this case's clinic, ingested like any other:
    how a case plants an instruction where search_policy will find it."""

    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    markdown: str = Field(min_length=1)


class Expect(_Strict):
    """Everything is optional; only what is given is checked."""

    # The front desk's outcome for the last message, any of these.
    outcome: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()  # each must be called at least once
    no_tools: tuple[str, ...] = ()  # none of these may be called
    # Exact local start times of appointments the run booked for this client.
    booked: tuple[str, ...] | None = None
    cancelled: int | None = None  # appointments actually cancelled (not held)
    # A review item of any of these kinds was created; ["none"] means no review item.
    review: tuple[str, ...] = ()
    # Each inner list: at least one of these must appear (case-insensitive) in
    # what the client was sent or what a reviewer was given to send.
    reply_has: tuple[tuple[str, ...], ...] = ()
    reply_lacks: tuple[str, ...] = ()


class Case(_Strict):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    feature: Feature
    difficulty: Difficulty
    tenant: TenantSlug = "maple-dental"
    client: Literal["known", "new"] = "known"
    messages: tuple[str, ...] = Field(min_length=1)
    appointments: tuple[Slot, ...] = ()  # the client's own, booked before the run
    taken: tuple[Slot, ...] = ()  # slots another client already holds
    documents: tuple[PolicyDoc, ...] = ()  # added to the clinic's policy documents
    expect: Expect = Expect()
    tags: tuple[str, ...] = ()
    note: str = ""

    @model_validator(mode="after")
    def _expectation_given(self) -> "Case":
        if self.expect == Expect():
            raise ValueError(f"{self.id}: a case must expect something")
        return self


def load_cases(directory: Path) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(directory.glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    cases.append(Case.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{n}: {exc}") from exc
    ids = [c.id for c in cases]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate case ids: {dupes}")
    return cases
