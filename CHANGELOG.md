# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.0] - 2026-11-16

Module 5: evals that gate pull requests.

### Added

- `evals/run.py`, the eval suite, sending each case through the production
  inbound path against a fresh clinic in its own database: deterministic
  checks, an LLM judge for tone and faithfulness (`prompts/judge/v1`), 30
  simulated multi-turn conversations (`prompts/client_sim/v1`), and the
  Module 3 policy-answer eval. One budget caps every model call in a run
  ($3 default); a full run costs about $1.40-1.70.
- 132 cases tagged by feature and difficulty, 40 of them red team
  (injection, other clients' data, another clinic, diagnosis pressure), plus
  30 scenarios; with the 60 policy questions, 192 cases.
- A CI job (`.github/workflows/evals.yml`) that runs the suite when the
  prompts, agent, model setting or evals change, posts a score table on the
  PR, and fails below thresholds set from measured run-to-run noise
  (`evals/thresholds.json`, `evals/README.md`). Tested on PR #12: a
  deliberately worse prompt was blocked at 66% with 26 regressions listed.
- Risk rules measured on 130 tuning and 60 held-out labelled messages; on
  the held-out set, missed medical messages fell from 75% to 20% and missed
  money messages from 93% to 47% (false positives 4% to 8%).
- Judge agreement with 40 hand labels: faithfulness 95% (kappa 0.90), tone
  82.5% (kappa 0.38); tone is reported as a trend, not gated.
- Agent prompt `v2`, now the default (`FDA_AGENT_PROMPT_VERSION`): books the
  first free provider when the client has no preference, works out dates
  itself, offers at most three times. Booking cases went from 36% to 46-64%.
- `check_availability` takes an optional `from_time` and a provider by name
  (ADR 0014): the 8 slots it shows covered only the first two hours of a
  day, and 14 of 18 expected bookings were later than that.
- `llm/thinking.py`: extended thinking is off for every model call, in each
  model's own form, since thinking counts against `max_tokens`.
- ADR 0015: the agent stays on Claude Haiku 4.5 for now; Haiku
  5.5 ties on pass rate at 1/8 of the cost but writes replies over the SMS
  limit and leaked its reasoning into one draft.

### Fixed

- Two idempotent booking replays racing each other could get
  `SlotTakenError` instead of the original booking.
- A held message's draft is told the team already has it; the prompt said
  to call `escalate_to_human`, which read-only mode does not offer.
- The eval judge and client simulator spent their token limits on thinking
  (52 of 129 verdicts cut off; simulated conversations ended early).
- The leak check no longer fails a reply for repeating a name the client
  typed; the medical-advice check no longer matches "whether you can take
  Advil".
- Migrations, scripts and DB tests run on Windows (psycopg needs a selector
  event loop there).

## [0.4.0] - 2026-11-14

### Added

- An SMS agent loop written by hand (ADR 0010): call the model, run tools,
  feed results back, stop at a reply or after 6 model calls.
- `converse()` on the LLM interface, with provider-neutral, JSON-storable
  message types; the Anthropic adapter is tested against the real SDK through
  a mocked HTTP transport.
- Seven tools with strict schemas, scoped server-side to the texting client:
  check_availability, book_appointment, list_my_appointments,
  cancel_appointment, search_policy, save_client_name, escalate_to_human.
- Availability from per-provider weekly schedules, in the clinic's timezone
  (DST-safe), with a one-hour lead time.
- Risk policy in code (ADR 0011): rule-based money and medical detection
  before the model runs; late cancellations held by the tool; the model can
  escalate. Holds create a review item and pause the conversation.
- Review queue: `GET /v1/tenants/{id}/reviews`, `.../approve` (optionally
  with an edited reply), `.../reject`. The agent resumes with a staff note
  and processes messages held meanwhile.
- Twilio webhook `POST /v1/channels/sms` with signature verification (403 on
  forgery, 503 when unconfigured) and MessageSid de-duplication (ADR 0012).
- arq worker on Redis for agent turns, slot filling and offer expiry, plus an
  inline queue for local development; texts and jobs only run after commit.
- Waitlist slot filling (ADR 0013): one offer at a time, YES books, NO or 15
  minutes of silence moves on, ranked by an epsilon-greedy bandit.
- STOP and START handling; new texters become client records.
- `scripts/sms_console.py`: text the agent from a terminal through the same
  code path as Twilio.
- Seed data: provider schedules matching each clinic's hours document, and
  `--sms-number` to use a real Twilio number.
- Migration 0003: provider_schedules, conversations, sms_messages,
  review_items, slot_offers, client_response_stats.
- 64 new tests (161 total), including a real arq worker on Redis.
- ADRs 0010 to 0013.

### Changed

- One booked appointment per provider per start time, enforced by a partial
  unique index; booking runs in a savepoint and raises `SlotTakenError`.
- docker-compose runs a worker and Redis; CI has a Redis service and checks
  the image can load the agent prompt and the worker's jobs.

### Fixed

- Worker jobs were registered under their closure name, so arq would have
  failed every job in production with "function not found". Caught by the
  new Redis integration test before release.

## [0.3.0] - 2026-11-14

### Added

- Policy retrieval: `policy_documents` and `policy_chunks` with 256-dimension
  pgvector embeddings, an HNSW index, and a Postgres-generated full-text column
  with a GIN index (migration 0002, reversible).
- Heading-aware chunking with overlap, plus a fixed-window baseline, so the
  choice between them is measured (ADR 0008).
- WordLlama embeddings: weights ship in the pip package, no API key, no
  network, CPU only. `Embedder` protocol for swapping in a hosted model.
- Vector, keyword and hybrid (reciprocal rank fusion) search, all tenant-bound.
  Deterministic tie-breaking, so metrics reproduce exactly.
- Idempotent ingestion: unchanged documents are skipped; a changed document,
  chunker or embedding model re-ingests; deleted files are pruned.
- `POST /v1/tenants/{id}/policy/answer`: answers only from retrieved chunks,
  cites them, and refuses (`needs_human: true`) when the documents do not
  cover the question or the model cannot cite a source (ADR 0009).
- `POST /v1/tenants/{id}/policy/search` and `PUT /v1/tenants/{id}/policy/documents`.
- Every model call recorded in `llm_calls` with cost, from a pricing table
  verified against the provider's page.
- 35 synthetic policy documents for two clinics with deliberately different
  rules, and a 60-question golden set (10 unanswerable, 2 cross-clinic traps).
- `evals/retrieval.py`: recall@5, recall at a fixed word budget, MRR and hit@1
  across 6 chunkers x 3 modes, written to `evals/reports/`.
- `evals/policy_answers.py`: real-model check of answering, grounding and
  refusal on the golden set, with a free `--dry-run` cost estimate.
- `scripts/ingest_policies.py`; the seed script now ingests policies too.
- 38 new tests (97 total), including a retrieval quality floor in CI.
- ADRs 0008 and 0009.

### Changed

- Default search mode is vector, not hybrid: hybrid never beat it on the
  golden set (ADR 0008). `FDA_RETRIEVAL_MODE` switches it.
- Tenants have an optional unique `slug`, used for policy folders.
- CI checks re-ingestion is a no-op, runs the retrieval report, uploads it,
  and checks the Docker image can load its prompts and embedder.

### Fixed

- The Docker image could not find its prompt files, so
  `/v1/drafts/reminder` failed inside the container (present since 0.1.0;
  CI only called `/healthz`). Prompts now ship in the image and are located
  through `FDA_PROMPTS_DIR`.
- A whitespace-only question embedded to a zero vector, whose cosine distance
  is NaN and slipped past every threshold. Questions are stripped and
  validated, and zero vectors return no results.
- A missing `FDA_ANTHROPIC_API_KEY` made both model-backed endpoints return an
  unhandled 500 with a traceback. It is now a 503 with a clear message.

## [0.2.0] - 2026-11-12

### Added

- Postgres data layer: tenants, clients, providers, appointments,
  waitlist_entries and llm_calls, with UUID keys and database timestamps.
- Alembic migration 0001, reversible, including the enum types autogenerate
  forgets to drop.
- Tenant-bound repositories: no method takes a tenant argument, so code cannot
  express a cross-tenant query (ADR 0005). Bookings check that the client and
  provider belong to the same tenant, which a foreign key alone does not.
- Idempotent booking: unique partial index plus `ON CONFLICT`, so concurrent
  retries with one key create one appointment; reusing a key for a different
  booking is a 409 (ADR 0006).
- Endpoints: `POST /v1/tenants`, `GET /v1/tenants/{id}/clients`,
  `POST|GET /v1/tenants/{id}/appointments`,
  `POST /v1/tenants/{id}/appointments/{appt_id}/cancel`,
  `GET /v1/tenants/{id}/llm-calls`.
- `/readyz` checks Postgres and returns 503 when it is unreachable;
  `/healthz` never touches the database.
- `scripts/seed.py`: two synthetic clinics, 40 clients each, fixed seed,
  phone numbers in the 555-01xx range reserved for fiction.
- 27 new tests (59 total): tenant isolation with real foreign IDs, a
  10-way concurrent booking race, migration round trip, model/migration drift
  check, and schema rules every future table must meet.
- ADRs 0005, 0006 and 0007.

### Changed

- CI runs a `pgvector/pgvector:pg16` service, applies migrations, runs
  `alembic check` and the seed script, then the tests. In CI a missing
  database fails the build instead of skipping tests.
- `docker compose up` runs a one-shot `migrate` service before the API starts.
- Added sqlalchemy 2.1, alembic, psycopg 3 and (dev) faker; `uv.lock`
  regenerated.
- The package ships `py.typed`, and mypy now also checks `scripts/` and
  `migrations/`.

## [0.1.0] - 2026-09-18

### Added

- `POST /v1/drafts/reminder`: drafts an appointment reminder SMS as validated
  JSON, never free text.
- `LLMClient` protocol with an Anthropic implementation and a scripted
  `FakeLLMClient`; no test or CI run calls a real model.
- Retry policy with exponential backoff and full jitter, retrying only
  timeouts, rate limits and 5xx responses.
- Versioned prompt files under `prompts/`, rendered with fail-loud placeholder
  substitution; prompt name and version recorded on every call.
- One repair attempt when model output fails schema validation, then 502.
- Token counts, latency and model recorded on every completion.
- ADRs 0003 and 0004.

### Changed

- `anthropic` added as a dependency; `uv.lock` regenerated.
- The LLM interface does not expose `temperature`, which the current SDK's
  `messages.create` no longer accepts.

## [0.0.0] - 2026-09-18

### Added

- Repository scaffold: `src/` layout, pyproject with ruff, mypy and pytest.
- FastAPI app with `/healthz` and `/readyz`, built through a factory.
- Environment-based configuration with Pydantic settings, no secrets in git.
- Multi-stage Dockerfile running as a non-root user, plus docker-compose with
  Postgres and pgvector for local development.
- GitHub Actions CI: lint, format check, types, tests, Docker build, and a
  container health check.
- Issue and pull request templates, pre-commit hooks, ADRs 0001 and 0002.
