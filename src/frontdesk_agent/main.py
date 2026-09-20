"""Application entrypoint.

Run locally with:  uvicorn frontdesk_agent.main:app --reload
"""

from fastapi import FastAPI

from frontdesk_agent import __version__
from frontdesk_agent.api.drafts import router as drafts_router
from frontdesk_agent.api.health import router as health_router


def create_app() -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton, so tests can construct a
    fresh app with different settings instead of sharing global state.
    """
    app = FastAPI(
        title="frontdesk-agent",
        version=__version__,
        summary="Multi-tenant LLM agent service for appointment front offices.",
    )
    app.include_router(health_router)
    app.include_router(drafts_router)
    return app


app = create_app()
