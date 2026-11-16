"""Tenant-scoped data endpoints.

The tenant comes from the URL for now. Module 6 replaces that with the tenant
bound to the caller's API key, so a caller can no longer name someone else's
tenant at all. Until then, isolation is enforced below the API, in the
repositories, which is where it has to hold regardless of how auth evolves.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.api.schemas import (
    AppointmentIn,
    AppointmentOut,
    ClientOut,
    ClientPage,
    LLMCallOut,
    TenantIn,
    TenantOut,
)
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    IdempotencyConflictError,
    LLMCallRepository,
    NotFoundError,
    TenantDirectory,
)
from frontdesk_agent.db.session import get_session

router = APIRouter(prefix="/v1/tenants", tags=["tenants"])

Session = Annotated[AsyncSession, Depends(get_session)]


async def tenant_scope(tenant_id: uuid.UUID, session: Session) -> uuid.UUID:
    """Resolve the tenant once per request; 404 if it does not exist."""
    try:
        await TenantDirectory(session).get(tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tenant not found.") from exc
    return tenant_id


TenantId = Annotated[uuid.UUID, Depends(tenant_scope)]


@router.post("", response_model=TenantOut, status_code=status.HTTP_201_CREATED)
async def create_tenant(body: TenantIn, session: Session) -> TenantOut:
    tenant = await TenantDirectory(session).create(**body.model_dump())
    await session.commit()
    return TenantOut.model_validate(tenant)


@router.get("/{tenant_id}/clients", response_model=ClientPage)
async def list_clients(
    tenant_id: TenantId,
    session: Session,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ClientPage:
    repo = ClientRepository(session, tenant_id)
    items = await repo.list(limit=limit, offset=offset)
    return ClientPage(
        items=[ClientOut.model_validate(c) for c in items],
        total=await repo.count(),
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{tenant_id}/appointments",
    response_model=AppointmentOut,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Replay: this Idempotency-Key already booked this appointment."},
        404: {"description": "Tenant, client or provider not found."},
        409: {"description": "Idempotency-Key reused with a different booking."},
    },
)
async def book_appointment(
    tenant_id: TenantId,
    body: AppointmentIn,
    session: Session,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
) -> AppointmentOut:
    """Book an appointment. Send an ``Idempotency-Key`` header from anything
    that may retry (an SMS webhook will), and a retry cannot double-book."""
    repo = AppointmentRepository(session, tenant_id)
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


@router.get("/{tenant_id}/appointments", response_model=list[AppointmentOut])
async def list_appointments(
    tenant_id: TenantId, session: Session, start: datetime, end: datetime
) -> list[AppointmentOut]:
    appts = await AppointmentRepository(session, tenant_id).between(start, end)
    return [AppointmentOut.model_validate(a) for a in appts]


@router.post("/{tenant_id}/appointments/{appointment_id}/cancel", response_model=AppointmentOut)
async def cancel_appointment(
    tenant_id: TenantId, appointment_id: uuid.UUID, session: Session
) -> AppointmentOut:
    try:
        appt = await AppointmentRepository(session, tenant_id).cancel(appointment_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found.") from exc
    await session.commit()
    return AppointmentOut.model_validate(appt)


@router.get("/{tenant_id}/llm-calls", response_model=list[LLMCallOut])
async def list_llm_calls(
    tenant_id: TenantId,
    session: Session,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[LLMCallOut]:
    calls = await LLMCallRepository(session, tenant_id).recent(limit=limit)
    return [LLMCallOut.model_validate(c) for c in calls]
