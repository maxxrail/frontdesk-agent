"""A retried booking must never create a second appointment."""

import asyncio
import uuid

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Appointment, Client, Provider, Tenant
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    IdempotencyConflictError,
    SlotTakenError,
)
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from tests.factories import HALF_HOUR, SLOT, make_client, make_provider, make_tenant


@pytest.fixture
async def setup(session: AsyncSession) -> tuple[Tenant, Client, Provider]:
    tenant = await make_tenant(session)
    client = await make_client(session, tenant)
    provider = await make_provider(session, tenant)
    await session.commit()
    return tenant, client, provider


async def _count(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(Appointment)) or 0)


async def test_same_key_twice_books_once(
    session: AsyncSession, setup: tuple[Tenant, Client, Provider]
) -> None:
    tenant, client, provider = setup
    repo = AppointmentRepository(session, tenant.id)
    args = dict(
        client_id=client.id,
        provider_id=provider.id,
        starts_at=SLOT,
        ends_at=SLOT + HALF_HOUR,
        idempotency_key="sms-msg-123",
    )

    first = await repo.book(**args)  # type: ignore[arg-type]
    second = await repo.book(**args)  # type: ignore[arg-type]

    assert first.created and not second.created
    assert first.appointment.id == second.appointment.id
    assert await _count(session) == 1


async def test_same_key_different_booking_is_a_conflict(
    session: AsyncSession, setup: tuple[Tenant, Client, Provider]
) -> None:
    tenant, client, provider = setup
    repo = AppointmentRepository(session, tenant.id)
    await repo.book(
        client_id=client.id,
        provider_id=provider.id,
        starts_at=SLOT,
        ends_at=SLOT + HALF_HOUR,
        idempotency_key="k",
    )
    with pytest.raises(IdempotencyConflictError):
        await repo.book(
            client_id=client.id,
            provider_id=provider.id,
            starts_at=SLOT + HALF_HOUR,
            ends_at=SLOT + 2 * HALF_HOUR,
            idempotency_key="k",
        )


async def test_no_key_means_no_deduplication(
    session: AsyncSession, setup: tuple[Tenant, Client, Provider]
) -> None:
    tenant, client, provider = setup
    repo = AppointmentRepository(session, tenant.id)
    for i in range(2):
        start = SLOT + i * HALF_HOUR
        await repo.book(
            client_id=client.id, provider_id=provider.id, starts_at=start, ends_at=start + HALF_HOUR
        )
    assert await _count(session) == 2


async def test_one_booking_per_provider_slot(
    session: AsyncSession, setup: tuple[Tenant, Client, Provider]
) -> None:
    """Enforced by a partial unique index, not by code that checks first."""
    tenant, client, provider = setup
    repo = AppointmentRepository(session, tenant.id)
    first = await repo.book(
        client_id=client.id, provider_id=provider.id, starts_at=SLOT, ends_at=SLOT + HALF_HOUR
    )
    with pytest.raises(SlotTakenError):
        await repo.book(
            client_id=client.id, provider_id=provider.id, starts_at=SLOT, ends_at=SLOT + HALF_HOUR
        )
    # The savepoint kept the session usable, and a cancelled slot is free again.
    await repo.cancel(first.appointment.id)
    again = await repo.book(
        client_id=client.id, provider_id=provider.id, starts_at=SLOT, ends_at=SLOT + HALF_HOUR
    )
    assert again.created


async def test_keys_are_per_tenant(session: AsyncSession) -> None:
    for name in ("A", "B"):
        t = await make_tenant(session, name)
        c, p = await make_client(session, t), await make_provider(session, t)
        result = await AppointmentRepository(session, t.id).book(
            client_id=c.id,
            provider_id=p.id,
            starts_at=SLOT,
            ends_at=SLOT + HALF_HOUR,
            idempotency_key="shared-key",
        )
        assert result.created
    assert await _count(session) == 2


async def test_concurrent_replays_race_safely(
    db_url: str, setup: tuple[Tenant, Client, Provider]
) -> None:
    """Ten simultaneous requests, separate connections, one key: one booking.

    A check-then-insert in application code fails this test; the unique
    partial index plus ON CONFLICT is what makes it pass.
    """
    tenant, client, provider = setup
    engine = make_engine(db_url)
    sessionmaker = make_sessionmaker(engine)
    key = f"race-{uuid.uuid4()}"

    async def attempt() -> uuid.UUID:
        async with sessionmaker() as s:
            r = await AppointmentRepository(s, tenant.id).book(
                client_id=client.id,
                provider_id=provider.id,
                starts_at=SLOT,
                ends_at=SLOT + HALF_HOUR,
                idempotency_key=key,
            )
            await s.commit()
            return r.appointment.id

    try:
        ids = await asyncio.gather(*(attempt() for _ in range(10)))
        async with sessionmaker() as s:
            assert await _count(s) == 1
    finally:
        await engine.dispose()
    assert len(set(ids)) == 1


async def test_api_replay_returns_200_with_the_same_booking(
    api: httpx.AsyncClient, setup: tuple[Tenant, Client, Provider]
) -> None:
    tenant, client, provider = setup
    body = {
        "client_id": str(client.id),
        "provider_id": str(provider.id),
        "starts_at": SLOT.isoformat(),
        "ends_at": (SLOT + HALF_HOUR).isoformat(),
    }
    url = f"/v1/tenants/{tenant.id}/appointments"
    headers = {"Idempotency-Key": "twilio-SM123"}

    first = await api.post(url, json=body, headers=headers)
    replay = await api.post(url, json=body, headers=headers)
    changed = await api.post(
        url, json=body | {"ends_at": (SLOT + 2 * HALF_HOUR).isoformat()}, headers=headers
    )

    assert (first.status_code, replay.status_code, changed.status_code) == (201, 200, 409)
    assert first.json()["id"] == replay.json()["id"]
