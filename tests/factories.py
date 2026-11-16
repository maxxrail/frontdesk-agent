"""Small builders for test data. Synthetic only."""

import itertools
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Client, Provider, Tenant
from frontdesk_agent.db.repositories import ClientRepository, ProviderRepository, TenantDirectory

_n = itertools.count()

SLOT = datetime(2030, 1, 7, 15, 0, tzinfo=UTC)
HALF_HOUR = timedelta(minutes=30)


async def make_tenant(session: AsyncSession, name: str = "Clinic") -> Tenant:
    return await TenantDirectory(session).create(name=name, business_type="dental office")


async def make_client(session: AsyncSession, tenant: Tenant, phone: str | None = None) -> Client:
    return await ClientRepository(session, tenant.id).add(
        Client(
            first_name="Test",
            last_name=f"Client{next(_n)}",
            phone=phone or f"+1416555{next(_n) % 10000:04d}",
        )
    )


async def make_provider(session: AsyncSession, tenant: Tenant) -> Provider:
    return await ProviderRepository(session, tenant.id).add(Provider(name="Dr. Test"))
