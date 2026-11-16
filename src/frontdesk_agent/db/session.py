"""Engine and session lifecycle.

The engine is created in the app's lifespan and stored on ``app.state``, not at
import time. Importing the package therefore never opens a connection, and each
test can build an app pointed at its own database.
"""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def make_engine(database_url: str) -> AsyncEngine:
    # pool_pre_ping drops dead connections after a database restart instead of
    # failing the first request that draws one.
    return create_async_engine(database_url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False lets handlers read attributes after commit
    # without an implicit (and async-illegal) lazy reload.
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One session per request; rolled back if the handler raises."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session
