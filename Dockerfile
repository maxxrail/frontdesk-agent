# Multi-stage build: the final image carries no build tools and no root user.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# ---

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FDA_PROMPTS_DIR=/app/prompts

# Run as a non-root user: a container that is compromised should not be root.
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Migrations ship in the image so the same artifact that serves traffic can
# migrate the database it serves (compose's migrate service, Module 8's deploy).
COPY --chown=appuser alembic.ini ./
COPY --chown=appuser migrations ./migrations
# Prompts are read at runtime; policies and scripts let the same image ingest
# documents: docker compose run --rm api python scripts/ingest_policies.py
COPY --chown=appuser prompts ./prompts
COPY --chown=appuser data ./data
COPY --chown=appuser scripts ./scripts
USER appuser

EXPOSE 8000

# The health check is what the orchestrator polls in Module 8.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "frontdesk_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
