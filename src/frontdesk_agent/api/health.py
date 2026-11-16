"""Health and readiness endpoints.

Kept separate from business routes because the deploy pipeline and the
load balancer call these, not users. /healthz answers "is the process up";
/readyz answers "can it serve traffic", which now means "can it reach Postgres".
"""

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from frontdesk_agent import __version__
from frontdesk_agent.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    env: str


class ReadyResponse(HealthResponse):
    database: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness: the process is running and can serve a request.

    Never touches the database: a database outage must not make the
    orchestrator restart healthy app containers in a loop.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=__version__,
        env=settings.env,
    )


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(request: Request, response: Response) -> ReadyResponse:
    """Readiness: 200 if Postgres answers, 503 if it does not."""
    base = await healthz()
    engine: AsyncEngine = request.app.state.engine
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        database = "ok"
    except Exception:  # any failure here means "not ready", whatever the cause
        database = "unreachable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        **base.model_dump() | {"status": "ok" if database == "ok" else "degraded"},
        database=database,
    )
