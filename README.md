# frontdesk-agent

A multi-tenant LLM agent service that runs the front office of an
appointment-based business over SMS: it answers policy questions from the
business's own documents, books and cancels appointments, fills cancelled slots
from a waitlist, and routes anything risky to a human for approval.

Every model call is logged with tokens, cost and latency. Every prompt change is
gated by an eval suite in CI.

**Status:** Module 4 of 9 complete. Clients text the clinic; an agent answers
policy questions from the clinic's own documents, finds open slots, books and
cancels, and hands anything risky to a person: late cancellations, money,
and medical questions wait in a review queue and send nothing until approved.
When a slot frees up, waitlisted clients are offered it one at a time, ranked
by a bandit that learns who answers fastest.

| Measured | |
| --- | --- |
| Retrieval recall@5 / MRR (60 golden questions) | 0.95 / 0.876 |
| Agent step limit | 6 model calls, tested with a model that never stops |
| Forged webhooks rejected | 4 of 4 forgery kinds, 403 |
| Waitlist bandit vs first-come-first-served (simulated) | 0.654 vs 0.192 mean reward |
| Tests | 161 |

How these were chosen, including why hybrid search lost: [ADR 0008](docs/adr/0008-retrieval-chosen-from-measurements.md).

All data in this repo is synthetic. No real patient or customer data is used.

## Quick start

```bash
git clone https://github.com/<you>/frontdesk-agent.git
cd frontdesk-agent
cp .env.example .env

# With Docker (also starts Postgres with pgvector)
docker compose up --build

# Or locally with uv (Postgres from compose: docker compose up -d db)
uv sync --extra dev
uv run alembic upgrade head
uv run python scripts/seed.py   # two synthetic clinics with policies; prints tenant IDs
uv run uvicorn frontdesk_agent.main:app --reload
```

Then:

```bash
curl http://localhost:8000/healthz
# {"status":"ok","service":"frontdesk-agent","version":"0.0.0","env":"local"}
```

Draft a reminder (needs `FDA_ANTHROPIC_API_KEY` in `.env`):

```bash
curl -X POST http://localhost:8000/v1/drafts/reminder \
  -H 'Content-Type: application/json' \
  -d '{
    "business_name": "Maple Dental",
    "business_type": "dental office",
    "client_first_name": "Dana",
    "appointment_at": "2026-10-06T14:30:00",
    "provider_name": "Dr. Osei",
    "location": "120 Main St"
  }'
```

List a seeded clinic's clients, then book idempotently (send the same request
twice: 201, then 200 with the same appointment):

```bash
curl "http://localhost:8000/v1/tenants/<tenant_id>/clients?limit=5"

curl -X POST http://localhost:8000/v1/tenants/<tenant_id>/appointments \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-1' \
  -d '{"client_id": "<client_id>", "provider_id": "<provider_id>",
       "starts_at": "2030-03-04T10:00:00-05:00",
       "ends_at": "2030-03-04T10:30:00-05:00"}'
```

Ask a policy question (needs `FDA_ANTHROPIC_API_KEY`). The answer cites its
sources; a question the documents do not cover is refused with
`"needs_human": true`:

```bash
curl -X POST http://localhost:8000/v1/tenants/<tenant_id>/policy/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "is there a fee if I cancel the morning of?"}'
```

See what retrieval finds, with no model call: `POST .../policy/search` with the
same body.

Interactive API docs: <http://localhost:8000/docs>

## Text the front desk

**From your terminal, no Twilio needed.** Same code path as a real SMS; replies
print instead of sending. Needs `FDA_ANTHROPIC_API_KEY`:

```bash
uv run python scripts/sms_console.py --clinic maple-dental
you> what's your cancellation policy?
you> can I come in tomorrow at 10?
you> my gum is bleeding a lot          # held for review: try /reviews, /approve 1
```

**From your phone, with Twilio.**

1. Get a Twilio number (a trial account can text your own verified phone).
2. Start the stack with a worker: `docker compose up --build`, and expose it:
   `ngrok http 8000`.
3. In `.env`: `FDA_TWILIO_ACCOUNT_SID`, `FDA_TWILIO_AUTH_TOKEN`,
   `FDA_PUBLIC_BASE_URL=https://<your-ngrok-host>`, `FDA_SMS_SENDER=twilio`.
4. Point Maple Dental at your number:
   `docker compose run --rm api python scripts/seed.py --reset --sms-number +1XXXXXXXXXX`
5. In the Twilio console, set the number's "A message comes in" webhook to
   `https://<your-ngrok-host>/v1/channels/sms` (HTTP POST).

Then text it. Pending reviews: `GET /v1/tenants/<tenant_id>/reviews`; decide
with `POST .../reviews/<id>/approve` or `/reject` and a body like
`{"reviewer": "you", "reply": "optional edited text"}`.

A text that is exactly `CANCEL` is an opt-out keyword to Twilio and is never
treated as a cancellation; "cancel my appointment" works. To change Twilio's
keywords, use Advanced Opt-Out on a Messaging Service.

## Measuring

```bash
uv run python evals/retrieval.py               # chunkers x search modes -> evals/reports/
uv run python evals/policy_answers.py --dry-run   # free; estimates a real run's cost
uv run python evals/policy_answers.py             # real model, ~60 calls, a few cents
```

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type check
uv run pytest            # tests (database tests skip if Postgres is absent)
uv run alembic revision --autogenerate -m "..."  # after changing a model
uv run pre-commit install  # run lint and format on every commit
```

Database tests use a separate `frontdesk_test` database, created on first run
and truncated before every test, so they never touch your seeded dev data. Set
`FDA_SKIP_DB_TESTS=1` to skip them. Migrations create the `vector` extension,
so the database role needs permission to create extensions: the compose and
CI users have it, and on RDS it is `rds_superuser`.

CI runs all of the above against a real Postgres, plus `alembic check`, the
seed script, a Docker build and a container health check on every pull request.

## Layout

```
src/frontdesk_agent/
  main.py            app factory and entrypoint
  config.py          settings from environment variables
  dependencies.py    wiring: which LLM client the app uses
  prompts.py         loads and renders versioned prompt files
  api/health.py      /healthz and /readyz
  api/drafts.py      /v1/drafts/reminder
  api/tenants.py     /v1/tenants/... clients, appointments, llm-calls
  db/models.py       tables
  db/repositories.py tenant-bound data access (read this first)
  retrieval/         chunking, embeddings, ingestion, search, metrics
  answering/         grounded policy answers with citations and refusal
  api/policy.py      /v1/tenants/.../policy/answer, search, documents
  api/channels.py    Twilio webhook: signature check, dedupe, enqueue
  api/reviews.py     review queue: list, approve, edit and approve, reject
  agent/             tools, risk policy, scheduling, the agent loop
  frontdesk/         inbound handling, review decisions, slot filling, bandit
  services.py        wiring, job runner, inline and Redis queues
  worker.py          arq worker: uv run arq frontdesk_agent.worker.WorkerSettings
  llm/pricing.py     cost per call, recorded in llm_calls
  llm/base.py        LLMClient protocol, errors, LLMCompletion
  llm/fake.py        scripted client; every test runs against this
  llm/retry.py       backoff with jitter for transient failures
  llm/anthropic_client.py  the real provider adapter
  drafting/          request and response schemas, drafting service
data/policies/       synthetic policy documents, one folder per tenant slug
evals/               golden set, evaluation scripts, reports
migrations/          Alembic migrations
scripts/seed.py      synthetic demo data, schedules and policies
scripts/sms_console.py  text the agent from a terminal
prompts/             versioned prompt files
tests/               pytest suite
docs/adr/            architecture decision records
.github/workflows/   CI
```

## Roadmap

| Module | Adds | Tag |
| --- | --- | --- |
| 0 | Repo, tooling, CI, Docker ✅ | v0.0 |
| 1 | LLM endpoint with structured output ✅ | v0.1 |
| 2 | Postgres, migrations, tenant isolation ✅ | v0.2 |
| 3 | Policy retrieval with pgvector, measured ✅ | v0.3 |
| 4 | Agent with tools, SMS channel, review queue ✅ | v0.4 |
| 5 | Eval suite gating pull requests | v0.5 |
| 6 | Guardrails, auth, redaction, threat model | v0.6 |
| 7 | Tracing, metrics, cost and latency dashboards | v0.7 |
| 8 | AWS deploy with Terraform and continuous delivery | v1.0 |
| 9 | TypeScript review dashboard and write-up | v1.1 |

## Decisions

Architecture decisions are recorded in [`docs/adr/`](docs/adr/).

## License

MIT
