"""Auth (ADR 0016): API keys for machines, login tokens for people, roles,
and the tenant always taken from the credential."""

import uuid
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import ApiKey, Role, Tenant, User
from frontdesk_agent.db.repositories import UserRepository
from frontdesk_agent.main import create_app
from frontdesk_agent.security import (
    hash_password,
    issue_token,
    key_matches,
    key_prefix,
    new_api_key,
    read_token,
    verify_password,
)
from tests.conftest import act_as
from tests.factories import make_tenant

SECRET = "unit-test-secret-" + "y" * 32
NOW = datetime(2030, 1, 7, 12, tzinfo=UTC)


# -- passwords, keys, tokens ---------------------------------------------------


def test_passwords_hash_with_a_salt_and_verify() -> None:
    a, b = hash_password("correct horse battery"), hash_password("correct horse battery")
    assert a != b and a.startswith("scrypt$")  # salted: same password, different hash
    assert verify_password("correct horse battery", a)
    assert not verify_password("correct horse batterY", a)
    assert not verify_password("anything", "not-a-hash")
    assert not verify_password("anything", "md5$1$1$1$x$y")


def test_api_keys_store_only_a_digest() -> None:
    new = new_api_key()
    assert new.key.startswith(f"fda_{new.prefix}_") and new.digest not in new.key
    assert key_prefix(new.key) == new.prefix and key_matches(new.key, new.digest)
    assert not key_matches(new.key + "x", new.digest)
    for junk in ("", "fda", "fda__x", "abc_123456789012_x", f"fda_{new.prefix}_"):
        assert key_prefix(junk) is None


def token(**kw: object) -> str:
    args: dict[str, object] = dict(
        user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), secret=SECRET, ttl=timedelta(hours=1), now=NOW
    )
    return issue_token(**(args | kw))  # type: ignore[arg-type]


def test_tokens_expire_and_are_tied_to_the_secret_and_algorithm() -> None:
    good = token(now=datetime.now(UTC))
    assert read_token(good, SECRET)["tid"]
    with pytest.raises(jwt.ExpiredSignatureError):
        read_token(token(now=datetime.now(UTC) - timedelta(hours=2)), SECRET)
    with pytest.raises(jwt.InvalidSignatureError):
        read_token(good, "another-secret-" + "z" * 32)
    claims = jwt.decode(good, options={"verify_signature": False})
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    with pytest.raises(jwt.InvalidTokenError):  # "alg: none" is not trusted
        read_token(unsigned, SECRET)


# -- over HTTP -----------------------------------------------------------------


async def make_user(
    session: AsyncSession, tenant: Tenant, role: Role, email: str = "sam@maple.example"
) -> User:
    user = await UserRepository(session, tenant.id).create(
        email=email, password_hash=hash_password("a long password!"), role=role
    )
    await session.commit()
    return user


async def login(api: httpx.AsyncClient, email: str, password: str) -> httpx.Response:
    return await api.post("/v1/auth/login", json={"email": email, "password": password})


async def test_login_gives_a_token_that_scopes_to_the_users_tenant(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    tenant = await make_tenant(session)
    await make_user(session, tenant, Role.REVIEWER, email="Sam@Maple.Example")
    r = await login(api, "sam@maple.example", "a long password!")  # email is case-insensitive
    assert r.status_code == 200 and r.json()["role"] == "reviewer"
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert (await api.get("/v1/reviews", headers=headers)).status_code == 200
    assert read_token(r.json()["access_token"], get_settings().jwt_secret.get_secret_value())[
        "tid"
    ] == str(tenant.id)


async def test_failed_logins_all_look_the_same(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    tenant = await make_tenant(session)
    await make_user(session, tenant, Role.ADMIN)
    wrong = await login(api, "sam@maple.example", "not the password")
    unknown = await login(api, "nobody@maple.example", "a long password!")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


async def test_deactivating_or_demoting_a_user_applies_at_once(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    """The token names the user; every request re-reads them."""
    tenant = await make_tenant(session)
    user = await make_user(session, tenant, Role.ADMIN)
    tok = (await login(api, "sam@maple.example", "a long password!")).json()["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}
    assert (await api.get("/v1/users", headers=headers)).status_code == 200

    user.role = Role.REVIEWER
    await session.commit()
    assert (await api.get("/v1/users", headers=headers)).status_code == 403

    user.active = False
    await session.commit()
    assert (await api.get("/v1/reviews", headers=headers)).status_code == 401
    assert (await login(api, "sam@maple.example", "a long password!")).status_code == 401


async def test_a_token_for_one_tenant_cannot_claim_another(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    a, b = await make_tenant(session, "A"), await make_tenant(session, "B")
    user = await make_user(session, a, Role.ADMIN)
    forged = issue_token(
        user_id=user.id,
        tenant_id=b.id,
        secret=get_settings().jwt_secret.get_secret_value(),
        ttl=timedelta(hours=1),
        now=datetime.now(UTC),
    )
    r = await api.get("/v1/clients", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


async def test_bad_and_revoked_api_keys_are_refused(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    tenant = await make_tenant(session)
    await session.commit()
    key = await act_as(api, session, tenant.id)
    assert (await api.get("/v1/clients")).status_code == 200
    for bad in (key[:-1] + ("A" if key[-1] != "A" else "B"), "fda_000000000000_x", "garbage"):
        assert (await api.get("/v1/clients", headers={"X-API-Key": bad})).status_code == 401

    created = (await api.post("/v1/api-keys", json={"name": "ci", "role": "reviewer"})).json()
    assert created["key"].startswith("fda_")
    listed = (await api.get("/v1/api-keys")).json()
    assert all("key" not in k for k in listed)  # shown once, never again
    assert (await api.get("/v1/reviews", headers={"X-API-Key": created["key"]})).status_code == 200

    assert (await api.post(f"/v1/api-keys/{created['id']}/revoke")).status_code == 200
    assert (await api.get("/v1/reviews", headers={"X-API-Key": created["key"]})).status_code == 401
    rows = await session.get(ApiKey, uuid.UUID(created["id"]))
    assert rows is not None and created["key"] not in rows.key_hash


async def test_reviewers_cannot_use_admin_endpoints(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    tenant = await make_tenant(session)
    await session.commit()
    await act_as(api, session, tenant.id, Role.REVIEWER)
    assert (await api.get("/v1/reviews")).status_code == 200
    for method, path in [
        ("GET", "/v1/users"),
        ("POST", "/v1/api-keys"),
        ("GET", "/v1/audit-events"),
        ("GET", "/v1/llm-calls"),
        ("POST", f"/v1/appointments/{uuid.uuid4()}/cancel"),
        ("PUT", "/v1/policy/documents"),
    ]:
        r = await api.request(method, path, json={})
        assert r.status_code == 403, (method, path, r.status_code)


async def test_admins_manage_users(api: httpx.AsyncClient, session: AsyncSession) -> None:
    tenant = await make_tenant(session)
    await session.commit()
    await act_as(api, session, tenant.id)
    new = {"email": "kim@maple.example", "password": "another long one", "role": "reviewer"}
    assert (await api.post("/v1/users", json=new)).status_code == 201
    assert (await api.post("/v1/users", json=new)).status_code == 409
    assert (await api.post("/v1/users", json=new | {"password": "short"})).status_code == 422
    assert (await login(api, "kim@maple.example", "another long one")).status_code == 200


async def test_the_server_refuses_to_start_without_a_signing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FDA_JWT_SECRET", "")
    get_settings.cache_clear()
    try:
        app = create_app("postgresql+psycopg://nobody:nothing@127.0.0.1:1/none")
        with pytest.raises(RuntimeError, match="FDA_JWT_SECRET"):
            async with app.router.lifespan_context(app):
                pass
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


async def test_a_new_tenants_first_admin_can_log_in_and_use_the_key(
    api: httpx.AsyncClient, session: AsyncSession
) -> None:
    from frontdesk_agent.onboarding import first_admin

    tenant = await make_tenant(session)
    access = await first_admin(session, tenant.id, "Owner@Maple.Example")
    await session.commit()
    r = await login(api, "owner@maple.example", access.password)
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert (await api.get("/v1/users", headers={"X-API-Key": access.api_key})).status_code == 200
