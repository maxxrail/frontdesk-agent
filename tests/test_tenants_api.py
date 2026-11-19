"""The tenant data API: status codes, validation, pagination.

The tenant comes from the API key (tests/conftest.py act_as), never the URL.
"""

import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Role
from tests.conftest import act_as
from tests.factories import HALF_HOUR, SLOT, make_client, make_provider, make_tenant


async def test_tenant_data_needs_a_credential(api: httpx.AsyncClient) -> None:
    assert (await api.get("/v1/clients")).status_code == 401


async def test_clients_paginate(api: httpx.AsyncClient, session: AsyncSession) -> None:
    tenant = await make_tenant(session)
    for _ in range(5):
        await make_client(session, tenant)
    await session.commit()
    await act_as(api, session, tenant.id, Role.REVIEWER)

    page1 = (await api.get("/v1/clients?limit=2")).json()
    page3 = (await api.get("/v1/clients?limit=2&offset=4")).json()
    assert page1["total"] == 5 and len(page1["items"]) == 2 and len(page3["items"]) == 1
    assert (await api.get("/v1/clients?limit=500")).status_code == 422


async def test_booking_validation_and_cancel(api: httpx.AsyncClient, session: AsyncSession) -> None:
    tenant = await make_tenant(session)
    client, provider = await make_client(session, tenant), await make_provider(session, tenant)
    await session.commit()
    await act_as(api, session, tenant.id)
    url = "/v1/appointments"
    good = {
        "client_id": str(client.id),
        "provider_id": str(provider.id),
        "starts_at": SLOT.isoformat(),
        "ends_at": (SLOT + HALF_HOUR).isoformat(),
    }

    naive = good | {"starts_at": "2030-01-07T15:00:00", "ends_at": "2030-01-07T15:30:00"}
    backwards = good | {"ends_at": (SLOT - HALF_HOUR).isoformat()}
    unknown_client = good | {"client_id": str(uuid.uuid4())}
    assert (await api.post(url, json=naive)).status_code == 422
    assert (await api.post(url, json=backwards)).status_code == 422
    assert (await api.post(url, json=unknown_client)).status_code == 404

    booked = await api.post(url, json=good)
    assert booked.status_code == 201
    cancelled = await api.post(f"{url}/{booked.json()['id']}/cancel")
    assert cancelled.json()["status"] == "cancelled"


async def test_readyz_is_ok_when_postgres_answers(api: httpx.AsyncClient) -> None:
    r = await api.get("/readyz")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"


def test_readyz_is_503_when_postgres_is_down(client: TestClient) -> None:
    """``client`` points at an unreachable database. /healthz stays 200."""
    assert client.get("/healthz").status_code == 200
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["database"] == "unreachable"


async def test_missing_api_key_is_503_not_500(
    api: httpx.AsyncClient, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both model-backed endpoints, with no key configured."""
    from frontdesk_agent.config import get_settings
    from frontdesk_agent.dependencies import get_llm_client

    monkeypatch.setenv("FDA_ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()
    get_llm_client.cache_clear()
    try:
        tenant = await make_tenant(session)
        await session.commit()
        await act_as(api, session, tenant.id)
        r = await api.post("/v1/policy/answer", json={"question": "fee?"})
        assert r.status_code == 503 and "not configured" in r.json()["detail"]
        draft = {
            "business_name": "Maple Dental",
            "business_type": "dental office",
            "client_first_name": "Dana",
            "appointment_at": "2030-01-07T15:00:00",
            "provider_name": "Dr. Osei",
            "location": "120 Main St",
        }
        assert (await api.post("/v1/drafts/reminder", json=draft)).status_code == 503
    finally:
        get_settings.cache_clear()
        get_llm_client.cache_clear()
