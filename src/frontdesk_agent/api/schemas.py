"""Response and request bodies for the data endpoints.

ORM objects never leave the API directly: each is converted to one of these,
so adding a column to a table can never leak it to callers by accident.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frontdesk_agent.db.models import AppointmentStatus

E164 = r"^\+[1-9]\d{7,14}$"


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


SLUG = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class TenantIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str | None = Field(default=None, max_length=60, pattern=SLUG)
    business_type: str = Field(min_length=1, max_length=60)
    timezone: str = Field(default="America/Toronto", max_length=64)
    sms_number: str | None = Field(default=None, pattern=E164)


class TenantOut(_Out):
    id: uuid.UUID
    name: str
    slug: str | None
    business_type: str
    timezone: str
    sms_number: str | None


class ClientOut(_Out):
    id: uuid.UUID
    first_name: str
    last_name: str
    phone: str
    sms_opt_in: bool


class ClientPage(BaseModel):
    items: list[ClientOut]
    total: int
    limit: int
    offset: int


class AppointmentIn(BaseModel):
    client_id: uuid.UUID
    provider_id: uuid.UUID
    starts_at: datetime
    ends_at: datetime
    notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def check_times(self) -> Self:
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("starts_at and ends_at must include a timezone offset")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        return self


class AppointmentOut(_Out):
    id: uuid.UUID
    client_id: uuid.UUID
    provider_id: uuid.UUID
    starts_at: datetime
    ends_at: datetime
    status: AppointmentStatus
    notes: str | None


class LLMCallOut(_Out):
    id: uuid.UUID
    created_at: datetime
    model: str
    prompt_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: Decimal | None
    succeeded: bool
