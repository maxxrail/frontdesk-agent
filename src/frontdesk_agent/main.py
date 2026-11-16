"""Application entrypoint.

Run locally with:  uvicorn frontdesk_agent.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from frontdesk_agent import __version__
from frontdesk_agent.api.channels import router as channels_router
from frontdesk_agent.api.drafts import router as drafts_router
from frontdesk_agent.api.health import router as health_router
from frontdesk_agent.api.policy import router as policy_router
from frontdesk_agent.api.reviews import router as reviews_router
from frontdesk_agent.api.tenants import router as tenants_router
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.dependencies import LazyLLMClient
from frontdesk_agent.llm.base import LLMConfigurationError
from frontdesk_agent.services import ArqQueue, build_services


def create_app(database_url: str | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton, so tests can construct a
    fresh app with different settings instead of sharing global state.
    """
    url = database_url or get_settings().database_url

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Creating an engine does not connect, so the app starts even when
        # Postgres is down; /readyz reports it instead of the process crashing.
        engine = make_engine(url)
        app.state.engine = engine
        app.state.sessionmaker = make_sessionmaker(engine)
        app.state.services = build_services(get_settings(), app.state.sessionmaker, LazyLLMClient())
        try:
            yield
        finally:
            queue = app.state.services.queue
            if isinstance(queue, ArqQueue):
                await queue.close()
            await engine.dispose()

    app = FastAPI(
        title="frontdesk-agent",
        version=__version__,
        summary="Multi-tenant LLM agent service for appointment front offices.",
        lifespan=lifespan,
    )

    @app.exception_handler(LLMConfigurationError)
    async def llm_not_configured(request: Request, exc: LLMConfigurationError) -> JSONResponse:
        # Raised while building the dependency, before any endpoint's own
        # error handling runs; without this it was an unhandled 500.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "The language model is not configured on this server."},
        )

    app.include_router(health_router)
    app.include_router(drafts_router)
    app.include_router(tenants_router)
    app.include_router(policy_router)
    app.include_router(reviews_router)
    app.include_router(channels_router)
    return app


app = create_app()
