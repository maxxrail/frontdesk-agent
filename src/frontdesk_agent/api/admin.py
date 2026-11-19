"""Admin endpoints: API keys, users, and the audit trail. Admins only.

A new API key is returned once, in the creation response; only its digest
is stored, so it cannot be shown again (ADR 0016).
"""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from frontdesk_agent.api.auth import Admin, Session
from frontdesk_agent.db.models import Role
from frontdesk_agent.db.repositories import (
    ApiKeyRepository,
    AuditRepository,
    EmailTakenError,
    NotFoundError,
    UserRepository,
)
from frontdesk_agent.security import MIN_PASSWORD_LENGTH, hash_password, new_api_key

router = APIRouter(prefix="/v1", tags=["admin"])


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    role: Role = Role.REVIEWER


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    prefix: str
    role: Role
    created_at: datetime
    revoked_at: datetime | None


class NewApiKeyOut(ApiKeyOut):
    key: str = Field(description="Shown once. Store it now; it cannot be retrieved later.")


class UserIn(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=200)
    role: Role = Role.REVIEWER


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    role: Role
    active: bool
    created_at: datetime
    last_login_at: datetime | None


class AuditEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    kind: str
    actor: str
    subject: str
    outcome: str
    conversation_id: uuid.UUID | None
    ref_id: uuid.UUID | None
    detail: dict[str, Any]
    created_at: datetime


@router.post("/api-keys", response_model=NewApiKeyOut, status_code=status.HTTP_201_CREATED)
async def create_api_key(who: Admin, body: ApiKeyIn, session: Session) -> NewApiKeyOut:
    new = new_api_key()
    row = await ApiKeyRepository(session, who.tenant_id).create(
        name=body.name, prefix=new.prefix, key_hash=new.digest, role=body.role
    )
    await session.commit()
    return NewApiKeyOut(**ApiKeyOut.model_validate(row).model_dump(), key=new.key)


@router.get("/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(who: Admin, session: Session) -> list[ApiKeyOut]:
    rows = await ApiKeyRepository(session, who.tenant_id).list(limit=200)
    return [ApiKeyOut.model_validate(r) for r in rows]


@router.post("/api-keys/{key_id}/revoke", response_model=ApiKeyOut)
async def revoke_api_key(who: Admin, key_id: uuid.UUID, session: Session) -> ApiKeyOut:
    try:
        row = await ApiKeyRepository(session, who.tenant_id).revoke(key_id, datetime.now(UTC))
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found.") from exc
    await session.commit()
    return ApiKeyOut.model_validate(row)


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(who: Admin, body: UserIn, session: Session) -> UserOut:
    try:
        user = await UserRepository(session, who.tenant_id).create(
            email=body.email, password_hash=hash_password(body.password), role=body.role
        )
    except EmailTakenError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await session.commit()
    return UserOut.model_validate(user)


@router.get("/users", response_model=list[UserOut])
async def list_users(who: Admin, session: Session) -> list[UserOut]:
    rows = await UserRepository(session, who.tenant_id).list(limit=200)
    return [UserOut.model_validate(r) for r in rows]


@router.get("/audit-events", response_model=list[AuditEventOut])
async def list_audit_events(
    who: Admin,
    session: Session,
    kind: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[AuditEventOut]:
    rows = await AuditRepository(session, who.tenant_id).recent(kind=kind, limit=limit)
    return [AuditEventOut.model_validate(r) for r in rows]
