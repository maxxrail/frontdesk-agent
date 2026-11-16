"""The agent's tools.

Rules every tool follows, because the model is not trusted:

- Inputs are validated against a strict Pydantic model (unknown keys
  rejected). A bad call returns an error result the model can correct;
  it never raises into the loop.
- The tenant and the client come from the conversation, never from the
  model. ``book_appointment`` has no client parameter: the agent can only
  ever book, list or cancel for the person who is texting.
- Anything the risk policy holds (a cancellation inside 24 hours, an
  escalation) is recorded on the context and *not executed*. The
  conversation service turns it into a review item after the loop.
"""

import datetime as dt
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.agent.risk import LATE_CANCEL_WINDOW_HOURS
from frontdesk_agent.agent.scheduling import find_slot, free_slots
from frontdesk_agent.db.models import Appointment, Client, Conversation, ReviewKind, Tenant
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    NotFoundError,
    ProviderRepository,
    SlotTakenError,
)
from frontdesk_agent.llm.messages import ToolSpec
from frontdesk_agent.retrieval.embeddings import Embedder
from frontdesk_agent.retrieval.store import PolicyStore

MAX_DAYS_AHEAD = 60
MAX_SLOTS_SHOWN = 8


@dataclass(frozen=True, slots=True)
class HeldAction:
    """Something the risk policy stopped, waiting for a person."""

    kind: ReviewKind
    reason: str
    action: dict[str, Any] | None = None


@dataclass
class ToolContext:
    session: AsyncSession
    tenant: Tenant
    client: Client
    conversation: Conversation
    embedder: Embedder
    now: datetime
    message_key: str  # the inbound message's provider id, for idempotent booking
    held: list[HeldAction] = field(default_factory=list)
    cancelled: list[Appointment] = field(default_factory=list)  # for slot filling

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tenant.timezone)


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str
    is_error: bool = False


def ok(payload: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(json.dumps(payload, default=str))


def error(message: str) -> ToolOutcome:
    return ToolOutcome(json.dumps({"error": message}), is_error=True)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- inputs --------------------------------------------------------------------


class CheckAvailabilityInput(_Input):
    # A field named "date" cannot also use the bare type name "date".
    date: dt.date = Field(description="The day to check, YYYY-MM-DD, in the clinic's timezone.")
    provider_id: uuid.UUID | None = Field(
        default=None, description="Only this provider. Omit to check everyone."
    )


class BookAppointmentInput(_Input):
    provider_id: uuid.UUID = Field(description="From a check_availability result.")
    starts_at: datetime = Field(
        description="Exact start time from a check_availability result, with its UTC offset."
    )

    @field_validator("starts_at")
    @classmethod
    def aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("starts_at must include a UTC offset, as check_availability gives it")
        return v


class NoInput(_Input):
    pass


class CancelAppointmentInput(_Input):
    appointment_id: uuid.UUID = Field(description="From list_my_appointments.")


class SearchPolicyInput(_Input):
    question: str = Field(min_length=1, max_length=300)


class SaveClientNameInput(_Input):
    first_name: str = Field(min_length=1, max_length=60)
    last_name: str = Field(min_length=1, max_length=60)


class EscalateInput(_Input):
    reason: Literal["money", "medical", "complaint", "other"]
    summary: str = Field(min_length=1, max_length=300, description="One line for the staff.")


# -- implementations -----------------------------------------------------------


async def check_availability(ctx: ToolContext, args: CheckAvailabilityInput) -> ToolOutcome:
    today = ctx.now.astimezone(ctx.tz).date()
    if args.date < today:
        return error("That date is in the past.")
    if args.date > today + timedelta(days=MAX_DAYS_AHEAD):
        return error(f"Bookings open at most {MAX_DAYS_AHEAD} days ahead.")
    slots = await free_slots(
        ctx.session, ctx.tenant, args.date, now=ctx.now, provider_id=args.provider_id
    )
    return ok(
        {
            "date": args.date.isoformat(),
            "weekday": args.date.strftime("%A"),
            "open_slots": [
                {
                    "provider_id": str(s.provider_id),
                    "provider": s.provider_name,
                    "starts_at": s.local(ctx.tz),
                }
                for s in slots[:MAX_SLOTS_SHOWN]
            ],
            "more_available": max(0, len(slots) - MAX_SLOTS_SHOWN),
        }
    )


async def book_appointment(ctx: ToolContext, args: BookAppointmentInput) -> ToolOutcome:
    if not ctx.client.first_name:
        return error("Ask for the client's first and last name and call save_client_name first.")
    slot = await find_slot(ctx.session, ctx.tenant, args.provider_id, args.starts_at, now=ctx.now)
    if slot is None:
        return error("That time is not an open slot. Call check_availability again.")
    try:
        result = await AppointmentRepository(ctx.session, ctx.tenant.id).book(
            client_id=ctx.client.id,
            provider_id=slot.provider_id,
            starts_at=slot.starts_at,
            ends_at=slot.ends_at,
            # The same inbound text, retried by Twilio or the worker, books once.
            idempotency_key=f"sms:{ctx.message_key}:{slot.provider_id}:{slot.starts_at:%Y%m%dT%H%M}",
        )
    except SlotTakenError:
        return error("Someone just took that slot. Offer another time.")
    return ok(
        {
            "booked": True,
            "appointment_id": str(result.appointment.id),
            "provider": slot.provider_name,
            "starts_at": slot.local(ctx.tz),
        }
    )


async def list_my_appointments(ctx: ToolContext, args: NoInput) -> ToolOutcome:
    appts = await AppointmentRepository(ctx.session, ctx.tenant.id).upcoming_for_client(
        ctx.client.id, ctx.now
    )
    providers = {p.id: p.name for p in await ProviderRepository(ctx.session, ctx.tenant.id).list()}
    return ok(
        {
            "appointments": [
                {
                    "appointment_id": str(a.id),
                    "provider": providers.get(a.provider_id, "unknown"),
                    "starts_at": a.starts_at.astimezone(ctx.tz).isoformat(timespec="minutes"),
                }
                for a in appts
            ]
        }
    )


async def cancel_appointment(ctx: ToolContext, args: CancelAppointmentInput) -> ToolOutcome:
    repo = AppointmentRepository(ctx.session, ctx.tenant.id)
    try:
        appt = await repo.get(args.appointment_id)
    except NotFoundError:
        return error("No such appointment for this client.")
    # Same error as "not found": never confirm another client's booking exists.
    if appt.client_id != ctx.client.id or appt.status != "booked" or appt.starts_at <= ctx.now:
        return error("No such appointment for this client.")

    when = appt.starts_at.astimezone(ctx.tz).isoformat(timespec="minutes")
    if appt.starts_at - ctx.now < timedelta(hours=LATE_CANCEL_WINDOW_HOURS):
        ctx.held.append(
            HeldAction(
                ReviewKind.LATE_CANCELLATION,
                f"Cancellation of the {when} appointment, less than "
                f"{LATE_CANCEL_WINDOW_HOURS} hours ahead.",
                {"type": "cancel_appointment", "appointment_id": str(appt.id)},
            )
        )
        return ok(
            {
                "cancelled": False,
                "status": "needs_staff_approval",
                "note": "Late cancellations are confirmed by staff. Do not say it is cancelled. "
                "Write the reply you would send once staff approve it.",
            }
        )
    await repo.cancel(appt.id)
    ctx.cancelled.append(appt)
    return ok({"cancelled": True, "starts_at": when})


async def search_policy(ctx: ToolContext, args: SearchPolicyInput) -> ToolOutcome:
    hits = await PolicyStore(ctx.session, ctx.tenant.id, ctx.embedder).search(args.question, k=4)
    return ok(
        {
            "excerpts": [{"source": h.heading, "text": h.content} for h in hits],
            "note": "Use only these excerpts for policy facts. If they do not answer it, say so.",
        }
    )


async def save_client_name(ctx: ToolContext, args: SaveClientNameInput) -> ToolOutcome:
    if ctx.client.first_name:
        return error("This client already has a name on file.")
    ctx.client.first_name, ctx.client.last_name = args.first_name.strip(), args.last_name.strip()
    await ctx.session.flush()
    return ok({"saved": True})


async def escalate_to_human(ctx: ToolContext, args: EscalateInput) -> ToolOutcome:
    kind = {"money": ReviewKind.MONEY, "medical": ReviewKind.MEDICAL}.get(
        args.reason, ReviewKind.ESCALATION
    )
    ctx.held.append(HeldAction(kind, f"Agent escalated ({args.reason}): {args.summary}"))
    return ok({"escalated": True, "note": "Write the reply a staff member could send."})


# -- registry ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_model: type[_Input]
    run: Callable[[ToolContext, Any], Awaitable[ToolOutcome]]
    read_only: bool

    @property
    def spec(self) -> ToolSpec:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema["additionalProperties"] = False
        return ToolSpec(self.name, self.description, schema)


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "check_availability",
            "List open appointment slots on one day.",
            CheckAvailabilityInput,
            check_availability,
            read_only=True,
        ),
        Tool(
            "book_appointment",
            "Book an open slot for the client who is texting.",
            BookAppointmentInput,
            book_appointment,
            read_only=False,
        ),
        Tool(
            "list_my_appointments",
            "List the texting client's upcoming appointments.",
            NoInput,
            list_my_appointments,
            read_only=True,
        ),
        Tool(
            "cancel_appointment",
            "Cancel one of the texting client's appointments. "
            "Cancellations inside 24 hours are held for staff approval.",
            CancelAppointmentInput,
            cancel_appointment,
            read_only=False,
        ),
        Tool(
            "search_policy",
            "Search the clinic's policy documents.",
            SearchPolicyInput,
            search_policy,
            read_only=True,
        ),
        Tool(
            "save_client_name",
            "Save the name of a new client who has none on file.",
            SaveClientNameInput,
            save_client_name,
            read_only=False,
        ),
        Tool(
            "escalate_to_human",
            "Hand the conversation to staff: money disputes, "
            "medical questions, complaints, or anything you should not decide.",
            EscalateInput,
            escalate_to_human,
            read_only=False,
        ),
    ]
}


async def execute(
    ctx: ToolContext, name: str, raw: dict[str, Any], allowed: set[str]
) -> ToolOutcome:
    tool = TOOLS.get(name)
    if tool is None or name not in allowed:
        return error(f"Unknown tool: {name}")
    try:
        args = tool.input_model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        return error(f"Invalid input: {problems}")
    return await tool.run(ctx, args)
