"""Tenant isolation: tenant B can never see or touch tenant A's rows.

Every test here uses A's *real* IDs from B's side. Guessing random UUIDs would
prove nothing; the threat is a bug that passes through a valid foreign ID.
"""

from dataclasses import dataclass
from datetime import timedelta

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Appointment, Client, Role, Tenant, WaitlistEntry
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    LLMCallRepository,
    NotFoundError,
    WaitlistRepository,
)
from frontdesk_agent.llm.base import LLMCompletion
from tests.conftest import act_as
from tests.factories import HALF_HOUR, SLOT, make_client, make_provider, make_tenant


@dataclass
class World:
    a: Tenant
    b: Tenant
    a_client: Client
    a_appt: Appointment


@pytest.fixture
async def world(session: AsyncSession) -> World:
    a = await make_tenant(session, "Clinic A")
    b = await make_tenant(session, "Clinic B")
    a_client = await make_client(session, a, phone="+14165550111")
    a_provider = await make_provider(session, a)
    a_appt = (
        await AppointmentRepository(session, a.id).book(
            client_id=a_client.id,
            provider_id=a_provider.id,
            starts_at=SLOT,
            ends_at=SLOT + HALF_HOUR,
        )
    ).appointment
    await WaitlistRepository(session, a.id).add(
        WaitlistEntry(client_id=a_client.id, earliest=SLOT, latest=SLOT + timedelta(days=7))
    )
    await LLMCallRepository(session, a.id).record(
        LLMCompletion(
            text="{}",
            model="m",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            prompt_name="p",
            prompt_version="v1",
        )
    )
    # B has its own client, so "empty" results below are not just an empty DB.
    await make_client(session, b)
    await session.commit()
    return World(a, b, a_client, a_appt)


async def test_get_client_by_foreign_id_is_not_found(session: AsyncSession, world: World) -> None:
    with pytest.raises(NotFoundError):
        await ClientRepository(session, world.b.id).get(world.a_client.id)


async def test_lookup_by_phone_does_not_cross_tenants(session: AsyncSession, world: World) -> None:
    assert await ClientRepository(session, world.b.id).by_phone("+14165550111") is None
    assert await ClientRepository(session, world.a.id).by_phone("+14165550111") is not None


async def test_lists_and_counts_are_scoped(session: AsyncSession, world: World) -> None:
    b_clients = ClientRepository(session, world.b.id)
    assert [c.id for c in await b_clients.list()] != [world.a_client.id]
    assert world.a_client.id not in {c.id for c in await b_clients.list()}
    assert await b_clients.count() == 1


async def test_appointments_are_scoped(session: AsyncSession, world: World) -> None:
    b_appts = AppointmentRepository(session, world.b.id)
    with pytest.raises(NotFoundError):
        await b_appts.get(world.a_appt.id)
    assert await b_appts.between(SLOT - HALF_HOUR, SLOT + HALF_HOUR * 4) == []
    with pytest.raises(NotFoundError):
        await b_appts.cancel(world.a_appt.id)


async def test_cannot_book_another_tenants_client_or_provider(
    session: AsyncSession, world: World
) -> None:
    """The foreign key alone would allow this: the client row exists."""
    b_provider = await make_provider(session, world.b)
    with pytest.raises(NotFoundError):
        await AppointmentRepository(session, world.b.id).book(
            client_id=world.a_client.id,
            provider_id=b_provider.id,
            starts_at=SLOT,
            ends_at=SLOT + HALF_HOUR,
        )


async def test_waitlist_and_llm_calls_are_scoped(session: AsyncSession, world: World) -> None:
    window = (SLOT, SLOT + HALF_HOUR)
    assert await WaitlistRepository(session, world.b.id).waiting_for(*window) == []
    assert len(await WaitlistRepository(session, world.a.id).waiting_for(*window)) == 1
    assert await LLMCallRepository(session, world.b.id).recent() == []
    assert len(await LLMCallRepository(session, world.a.id).recent()) == 1


async def test_a_forged_tenant_id_is_overwritten(session: AsyncSession, world: World) -> None:
    forged = Client(first_name="Eve", last_name="X", phone="+14165550199", tenant_id=world.a.id)
    saved = await ClientRepository(session, world.b.id).add(forged)
    assert saved.tenant_id == world.b.id
    assert await ClientRepository(session, world.a.id).by_phone("+14165550199") is None


async def test_api_lists_only_the_callers_tenant(
    api: httpx.AsyncClient, session: AsyncSession, world: World
) -> None:
    await act_as(api, session, world.b.id, Role.REVIEWER)
    body = (await api.get("/v1/clients")).json()
    assert body["total"] == 1
    assert world.a_client.id not in {item["id"] for item in body["items"]}


async def test_a_request_cannot_name_another_tenant(
    api: httpx.AsyncClient, session: AsyncSession, world: World
) -> None:
    """The tenant comes only from the credential (ADR 0016): the old
    tenant-in-URL routes are gone, and a tenant_id parameter is ignored."""
    await act_as(api, session, world.b.id, Role.REVIEWER)
    assert (await api.get(f"/v1/tenants/{world.a.id}/clients")).status_code == 404
    body = (await api.get("/v1/clients", params={"tenant_id": str(world.a.id)})).json()
    assert world.a_client.id not in {item["id"] for item in body["items"]}


async def test_api_cannot_cancel_through_another_tenant(
    api: httpx.AsyncClient, session: AsyncSession, world: World
) -> None:
    await act_as(api, session, world.b.id)
    r = await api.post(f"/v1/appointments/{world.a_appt.id}/cancel")
    assert r.status_code == 404
    await act_as(api, session, world.a.id)
    r = await api.get(
        "/v1/appointments",
        params={"start": SLOT.isoformat(), "end": (SLOT + HALF_HOUR).isoformat()},
    )
    assert r.json()[0]["status"] == "booked"


async def test_llm_calls_endpoint_is_scoped(
    api: httpx.AsyncClient, session: AsyncSession, world: World
) -> None:
    await act_as(api, session, world.b.id)
    assert (await api.get("/v1/llm-calls")).json() == []
    await act_as(api, session, world.a.id)
    calls = (await api.get("/v1/llm-calls")).json()
    assert len(calls) == 1 and calls[0]["prompt_name"] == "p"
