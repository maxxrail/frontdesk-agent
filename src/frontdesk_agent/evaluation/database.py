"""The eval database: separate from development data, migrated, wiped per run.

Every run starts with TRUNCATE, so the URL is checked first: the database
name must end in ``_eval`` or ``_test``. Pointing the suite at real data
by mistake then fails before anything is deleted.
"""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[3]


class UnsafeDatabaseError(ValueError):
    pass


def eval_url(database_url: str) -> str:
    """The eval database on the same server as ``database_url``."""
    return make_url(database_url).set(database="frontdesk_eval").render_as_string(False)


def check_safe(url: str) -> None:
    name = make_url(url).database or ""
    if not name.endswith(("_eval", "_test")):
        raise UnsafeDatabaseError(
            f"refusing to wipe database {name!r}: eval databases must end in _eval or _test"
        )


async def prepare(url: str) -> None:
    """Create the database if needed, migrate it, and empty it."""
    check_safe(url)
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

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["database_url"] = url
    # Alembic's env.py runs its own event loop; keep it off this one.
    await asyncio.to_thread(command.upgrade, cfg, "head")

    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE tenants CASCADE"))
    finally:
        await engine.dispose()
