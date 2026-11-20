"""Application entrypoint.

Run locally with:  uvicorn frontdesk_agent.main:app --reload
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from frontdesk_agent import __version__
from frontdesk_agent.api.admin import router as admin_router
from frontdesk_agent.api.auth import router as auth_router
from frontdesk_agent.api.channels import router as channels_router
from frontdesk_agent.api.drafts import router as drafts_router
from frontdesk_agent.api.health import router as health_router
from frontdesk_agent.api.policy import router as policy_router
from frontdesk_agent.api.reviews import router as reviews_router
from frontdesk_agent.api.tenants import router as tenants_router
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.dependencies import LazyLLMClient
from frontdesk_agent.errortracking import init_sentry
from frontdesk_agent.llm.base import LLMConfigurationError
from frontdesk_agent.logcontext import bound, request_id_from
from frontdesk_agent.logsetup import configure_logging
from frontdesk_agent.services import ArqQueue, build_services
from frontdesk_agent.tracing import SpanKind, init_tracing, trace_id_hex, tracer


def create_app(database_url: str | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton, so tests can construct a
    fresh app with different settings instead of sharing global state.
    """
    url = database_url or get_settings().database_url
    configure_logging(get_settings().log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not get_settings().jwt_secret.get_secret_value():
            # Refuse to serve rather than sign login tokens with a guessable
            # key. Checked at startup, not import, so tools can still import
            # the app. Generate one with:
            #   python -c "import secrets; print(secrets.token_urlsafe(48))"
            raise RuntimeError("FDA_JWT_SECRET is not set; refusing to start (ADR 0016).")
        # Here, not at import: importing the app (tests, tools) must never
        # start sending errors to Sentry.
        init_sentry(get_settings())
        init_tracing(get_settings())
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

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Every log line for this request carries one id, echoed back to the
        caller so a support ticket can quote it."""
        rid = request_id_from(request.headers.get("X-Request-ID"))
        request.state.request_id = rid
        method = request.method
        with tracer.start_as_current_span(f"{method}", kind=SpanKind.SERVER) as span:
            with bound(request_id=rid, trace_id=trace_id_hex(span)):
                response = await call_next(request)
            # The route template, not the path: no ids or numbers in the name.
            route = getattr(request.scope.get("route"), "path", None) or "unmatched"
            span.update_name(f"{method} {route}")
            span.set_attributes(
                {
                    "http.request.method": method,
                    "http.route": route,
                    "http.response.status_code": response.status_code,
                    "request.id": rid,
                }
            )
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(LLMConfigurationError)
    async def llm_not_configured(request: Request, exc: LLMConfigurationError) -> JSONResponse:
        # Raised while building the dependency, before any endpoint's own
        # error handling runs; without this it was an unhandled 500.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "The language model is not configured on this server."},
        )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(drafts_router)
    app.include_router(tenants_router)
    app.include_router(policy_router)
    app.include_router(reviews_router)
    app.include_router(channels_router)
    return app


app = create_app()
