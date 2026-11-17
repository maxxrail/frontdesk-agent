"""Tenant data endpoints: clients, appointments, model calls.

The tenant comes from the caller's credential (api/auth.py, ADR 0016), so a
request cannot name another tenant at all. Isolation still holds below the
API too, in the tenant-bound repositories (ADR 0005).

Tenants themselves are created with scripts/create_tenant.py, not over HTTP:
no tenant's credential should be able to create another tenant.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Response, status

from frontdesk_agent.api.auth import Admin, Reviewer, Session
from frontdesk_agent.api.schemas import (
    AppointmentIn,
    AppointmentOut,
    ClientOut,
    ClientPage,
    LLMCallOut,
)
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    IdempotencyConflictError,
    LLMCallRepository,
    NotFoundError,
)

router = APIRouter(prefix="/v1", tags=["tenant data"])


@router.get("/clients", response_model=ClientPage)
async def list_clients(
    who: Reviewer,
    session: Session,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ClientPage:
    repo = ClientRepository(session, who.tenant_id)
    items = await repo.list(limit=limit, offset=offset)
    return ClientPage(
        items=[ClientOut.model_validate(c) for c in items],
        total=await repo.count(),
        limit=limit,
        offset=offset,
    )


@router.post(
    "/appointments",
    response_model=AppointmentOut,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Replay: this Idempotency-Key already booked this appointment."},
        404: {"description": "Client or provider not found."},
        409: {"description": "Idempotency-Key reused with a different booking."},
    },
)
async def book_appointment(
    who: Admin,
    body: AppointmentIn,
    session: Session,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
) -> AppointmentOut:
    """Book an appointment. Send an ``Idempotency-Key`` header from anything
    that may retry (an SMS webhook will), and a retry cannot double-book."""
    repo = AppointmentRepository(session, who.tenant_id)
    try:
        result = await repo.book(**body.model_dump(), idempotency_key=idempotency_key)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Client or provider not found.") from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await session.commit()
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return AppointmentOut.model_validate(result.appointment)


@router.get("/appointments", response_model=list[AppointmentOut])
async def list_appointments(
    who: Reviewer, session: Session, start: datetime, end: datetime
) -> list[AppointmentOut]:
    appts = await AppointmentRepository(session, who.tenant_id).between(start, end)
    return [AppointmentOut.model_validate(a) for a in appts]


@router.post("/appointments/{appointment_id}/cancel", response_model=AppointmentOut)
async def cancel_appointment(
    who: Admin, appointment_id: uuid.UUID, session: Session
) -> AppointmentOut:
    try:
        appt = await AppointmentRepository(session, who.tenant_id).cancel(appointment_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found.") from exc
    await session.commit()
    return AppointmentOut.model_validate(appt)


@router.get("/llm-calls", response_model=list[LLMCallOut])
async def list_llm_calls(
    who: Admin,
    session: Session,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[LLMCallOut]:
    calls = await LLMCallRepository(session, who.tenant_id).recent(limit=limit)
    return [LLMCallOut.model_validate(c) for c in calls]
