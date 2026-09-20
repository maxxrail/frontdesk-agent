"""Request and response models for the drafting endpoints.

Validation at the edge, so an invalid request never reaches the model and an
invalid model reply never reaches the caller.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

MAX_SMS_CHARS = 300


class ReminderRequest(BaseModel):
    """What the caller must supply to get a reminder drafted."""

    business_name: str = Field(min_length=1, max_length=120)
    business_type: str = Field(min_length=1, max_length=60, examples=["dental office"])
    client_first_name: str = Field(min_length=1, max_length=60)
    appointment_at: datetime
    provider_name: str = Field(min_length=1, max_length=120)
    location: str = Field(min_length=1, max_length=200)

    @field_validator("client_first_name")
    @classmethod
    def no_spaces_in_first_name(cls, v: str) -> str:
        """Guard against a full name arriving in the first-name field.

        The prompt forbids last names; this stops one slipping in anyway.
        """
        if " " in v.strip():
            raise ValueError("client_first_name must be a single name, not a full name")
        return v.strip()

    def appointment_human(self) -> str:
        """The appointment rendered the way it should read in the message."""
        return self.appointment_at.strftime("%A, %B %-d at %-I:%M %p")


class ReminderDraft(BaseModel):
    """The structured object the model must return.

    Extra keys are rejected: a model inventing fields is a signal worth failing on.
    """

    model_config = {"extra": "forbid"}

    message: str = Field(min_length=1, max_length=MAX_SMS_CHARS)
    confirm_keyword: str = Field(min_length=1, max_length=20)
    reschedule_keyword: str = Field(min_length=1, max_length=20)


class ReminderResponse(BaseModel):
    """What the endpoint returns: the draft plus how it was produced."""

    draft: ReminderDraft
    model: str
    prompt_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    repaired: bool = Field(
        default=False,
        description="True if the model's first reply failed validation and was retried.",
    )
