"""Who is calling, and for which tenant (ADR 0016).

Every tenant endpoint depends on ``Reviewer`` or ``Admin``. The tenant comes
from the credential: an ``X-API-Key`` for machine clients, or a bearer token
from ``POST /v1/auth/login`` for dashboard users. No request can name a
tenant, so no request can name someone else's.
"""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import Role, User
from frontdesk_agent.db.repositories import CredentialDirectory
from frontdesk_agent.db.session import get_session
from frontdesk_agent.security import (
    dummy_password_hash,
    issue_token,
    key_matches,
    key_prefix,
    read_token,
    verify_password,
)

Session = Annotated[AsyncSession, Depends(get_session)]


@dataclass(frozen=True)
class Principal:
    tenant_id: uuid.UUID
    role: Role
    actor: str  # "user:<email>" or "key:<name>": what the audit trail records


def _unauthorized() -> HTTPException:
    # One message for every failure: which part was wrong is not the
    # caller's business.
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Invalid or missing credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def jwt_secret() -> str:
    secret = get_settings().jwt_secret.get_secret_value()
    if not secret:  # pragma: no cover - create_app refuses to start without it
        raise RuntimeError("FDA_JWT_SECRET is not set")
    return secret


async def current_principal(
    session: Session,
    x_api_key: Annotated[str | None, Header(max_length=200)] = None,
    authorization: Annotated[str | None, Header(max_length=4096)] = None,
) -> Principal:
    if x_api_key:
        prefix = key_prefix(x_api_key)
        key = await CredentialDirectory(session).api_key_by_prefix(prefix) if prefix else None
        if key is None or key.revoked_at is not None or not key_matches(x_api_key, key.key_hash):
            raise _unauthorized()
        return Principal(key.tenant_id, key.role, f"key:{key.name}")

    if authorization and authorization.startswith("Bearer "):
        try:
            claims = read_token(authorization.removeprefix("Bearer ").strip(), jwt_secret())
            user = await session.get(User, uuid.UUID(claims["sub"]))
        except (jwt.InvalidTokenError, ValueError) as exc:
            raise _unauthorized() from exc
        if user is None or not user.active or str(user.tenant_id) != claims["tid"]:
            raise _unauthorized()
        return Principal(user.tenant_id, user.role, f"user:{user.email}")

    raise _unauthorized()


def require(role: Role) -> Callable[[Principal], Awaitable[Principal]]:
    """A dependency that admits ``role`` (and admins, who can do anything)."""

    async def check(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if role is Role.ADMIN and principal.role is not Role.ADMIN:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "This needs an admin.")
        return principal

    return check


# Admin can do everything a reviewer can.
Reviewer = Annotated[Principal, Depends(require(Role.REVIEWER))]
Admin = Annotated[Principal, Depends(require(Role.ADMIN))]


# -- login ----------------------------------------------------------------------

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    role: Role


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, session: Session) -> TokenOut:
    """Exchange an email and password for a short-lived bearer token."""
    user = await CredentialDirectory(session).user_by_email(body.email)
    # Hash even when the email is unknown, so timing does not reveal which
    # emails have accounts.
    ok = verify_password(body.password, user.password_hash if user else dummy_password_hash())
    if user is None or not ok or not user.active:
        raise _unauthorized()
    now = datetime.now(UTC)
    ttl = timedelta(minutes=get_settings().jwt_ttl_minutes)
    user.last_login_at = now
    await session.commit()
    token = issue_token(
        user_id=user.id, tenant_id=user.tenant_id, secret=jwt_secret(), ttl=ttl, now=now
    )
    return TokenOut(access_token=token, expires_in=int(ttl.total_seconds()), role=user.role)
