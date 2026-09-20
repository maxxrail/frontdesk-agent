# frontdesk-agent

A multi-tenant LLM agent service that runs the front office of an
appointment-based business over SMS: it answers policy questions from the
business's own documents, books and cancels appointments, fills cancelled slots
from a waitlist, and routes anything risky to a human for approval.

Every model call is logged with tokens, cost and latency. Every prompt change is
gated by an eval suite in CI.

**Status:** Module 0 of 9 complete. The service is a skeleton with health
endpoints; the agent arrives in Module 4.

All data in this repo is synthetic. No real patient or customer data is used.

## Quick start

```bash
git clone https://github.com/<you>/frontdesk-agent.git
cd frontdesk-agent
cp .env.example .env

# With Docker (also starts Postgres with pgvector)
docker compose up --build

# Or locally with uv
uv sync --extra dev
uv run uvicorn frontdesk_agent.main:app --reload
```

Then:

```bash
curl http://localhost:8000/healthz
# {"status":"ok","service":"frontdesk-agent","version":"0.0.0","env":"local"}
```

Interactive API docs: <http://localhost:8000/docs>

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type check
uv run pytest            # tests
uv run pre-commit install  # run lint and format on every commit
```

CI runs all of the above plus a Docker build and a container health check on
every pull request.

## Layout

```
src/frontdesk_agent/
  main.py          app factory and entrypoint
  config.py        settings from environment variables
  api/health.py    /healthz and /readyz
tests/             pytest suite
docs/adr/          architecture decision records
.github/workflows/ CI
```

## Roadmap

| Module | Adds | Tag |
| --- | --- | --- |
| 0 | Repo, tooling, CI, Docker | v0.0 |
| 1 | LLM endpoint with structured output | v0.1 |
| 2 | Postgres, migrations, tenant isolation | v0.2 |
| 3 | Policy retrieval with pgvector, measured | v0.3 |
| 4 | Agent with tools, SMS channel, review queue | v0.4 |
| 5 | Eval suite gating pull requests | v0.5 |
| 6 | Guardrails, auth, redaction, threat model | v0.6 |
| 7 | Tracing, metrics, cost and latency dashboards | v0.7 |
| 8 | AWS deploy with Terraform and continuous delivery | v1.0 |
| 9 | TypeScript review dashboard and write-up | v1.1 |

## Decisions

Architecture decisions are recorded in [`docs/adr/`](docs/adr/).

## License

MIT
