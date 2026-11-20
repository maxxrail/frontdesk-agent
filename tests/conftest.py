"""Shared test fixtures.

Tests that need Postgres use the ``db_url``, ``session`` or ``api`` fixtures
and are skipped (not failed) when no database is available, so the unit tests
still run anywhere. CI always provides Postgres, so nothing is skipped there.

The test database is separate from the dev database (``frontdesk_test`` by
default) because every test truncates it.

    FDA_TEST_DATABASE_URL   override the test database URL
    FDA_SKIP_DB_TESTS=1     skip every database test
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from frontdesk_agent import aio
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import Role
from frontdesk_agent.db.repositories import ApiKeyRepository
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.main import create_app
from frontdesk_agent.security import new_api_key

# A throwaway signing key, so the app starts in tests (it refuses to without
# one). Settings are cached at import, hence the cache_clear.
os.environ.setdefault("FDA_JWT_SECRET", "test-only-not-a-secret-" + "x" * 32)
# Never report to a real Sentry project from tests, whatever .env says.
# Environment variables win over .env in pydantic-settings.
os.environ["FDA_SENTRY_DSN"] = ""
get_settings.cache_clear()

aio.use_selector_loop_policy()

UNREACHABLE_DB = "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client backed by a freshly built app; no database required."""
    with TestClient(create_app(UNREACHABLE_DB)) as test_client:
        yield test_client


def _test_db_url() -> str:
    explicit = os.environ.get("FDA_TEST_DATABASE_URL")
    if explicit:
        return explicit
    return (
        make_url(get_settings().database_url)
        .set(database="frontdesk_test")
        .render_as_string(hide_password=False)
    )


async def _ensure_database(url: str) -> None:
    """Create the test database if it is missing (needs CREATEDB locally)."""
    target = make_url(url)
    admin = create_async_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target.database}
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        await admin.dispose()


def alembic_config(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture(scope="session")
def db_url() -> str:
    """The migrated test database, or a skip if Postgres is not available."""
    if os.environ.get("FDA_SKIP_DB_TESTS") == "1":
        pytest.skip("FDA_SKIP_DB_TESTS=1")
    url = _test_db_url()
    try:
        aio.run(_ensure_database(url))
    except Exception as exc:  # unreachable, auth failure, no CREATEDB...
        if os.environ.get("CI"):
            # In CI a missing database is a broken pipeline, never a skip.
            raise
        pytest.skip(f"Postgres not available for tests: {exc!r}")
    command.upgrade(alembic_config(url), "head")
    return url


@pytest.fixture
async def session(db_url: str) -> AsyncIterator[AsyncSession]:
    """A session on an empty, migrated database."""
    engine = make_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE tenants CASCADE"))
    async with make_sessionmaker(engine)() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def api(db_url: str, session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    """An async HTTP client against an app wired to the test database.

    Depends on ``session`` so the database is truncated first, and so tests can
    arrange data with repositories and then call the API.
    """
    app = create_app(db_url)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def act_as(
    api: httpx.AsyncClient, session: AsyncSession, tenant_id: uuid.UUID, role: Role = Role.ADMIN
) -> str:
    """Make ``api`` call as a real API key of ``tenant_id`` with ``role``;
    returns the key."""
    new = new_api_key()
    await ApiKeyRepository(session, tenant_id).create(
        name=f"test-{role.value}", prefix=new.prefix, key_hash=new.digest, role=role
    )
    await session.commit()
    api.headers["X-API-Key"] = new.key
    return new.key
