"""Alembic environment: runs migrations over the app's async engine."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from frontdesk_agent import aio
from frontdesk_agent.config import get_settings
from frontdesk_agent.db import models  # noqa: F401  (registers tables on the metadata)
from frontdesk_agent.db.base import Base
from frontdesk_agent.db.session import make_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    # Tests pass an explicit URL through the Alembic config; otherwise use settings.
    return config.attributes.get("database_url") or get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (alembic upgrade --sql)."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = make_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    aio.run(run_migrations_online())
