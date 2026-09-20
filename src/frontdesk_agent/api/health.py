"""Health and readiness endpoints.

Kept separate from business routes because the deploy pipeline and the
load balancer call these, not users. /healthz answers "is the process up";
/readyz will answer "can it serve traffic" once Module 2 adds a database.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from frontdesk_agent import __version__
from frontdesk_agent.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    env: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness: the process is running and can serve a request."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=__version__,
        env=settings.env,
    )


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    """Readiness: dependencies are reachable.

    Module 2 adds a real database check here. Until then it mirrors /healthz
    so the deploy pipeline has a stable endpoint to call.
    """
    return await healthz()
