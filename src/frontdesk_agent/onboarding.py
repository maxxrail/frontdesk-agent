"""Giving a new tenant its first credentials.

Tenants are created from the command line (scripts/create_tenant.py,
scripts/seed.py), never over HTTP: no tenant's credential should be able to
create another tenant (ADR 0016). This sets up the first admin, who can then
add users and keys through the API.
"""

import secrets
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Role
from frontdesk_agent.db.repositories import ApiKeyRepository, UserRepository
from frontdesk_agent.security import hash_password, new_api_key


@dataclass(frozen=True)
class FirstAccess:
    email: str
    password: str  # generated here, shown once
    api_key: str  # shown once


async def first_admin(session: AsyncSession, tenant_id: uuid.UUID, email: str) -> FirstAccess:
    """An admin user with a generated password, and an admin API key.

    Neither secret is stored in a recoverable form; the caller shows them
    once. The admin should change the password after first login.
    """
    password = secrets.token_urlsafe(18)
    await UserRepository(session, tenant_id).create(
        email=email, password_hash=hash_password(password), role=Role.ADMIN
    )
    key = new_api_key()
    await ApiKeyRepository(session, tenant_id).create(
        name="first admin key", prefix=key.prefix, key_hash=key.digest, role=Role.ADMIN
    )
    return FirstAccess(email.strip().lower(), password, key.key)
