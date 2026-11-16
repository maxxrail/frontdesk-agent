# frontdesk-agent

A multi-tenant LLM agent that runs the front office of appointment businesses
over SMS: answers policy questions from each clinic's own documents, books and
cancels, fills cancelled slots from a waitlist, and routes risky actions to a
human review queue. It is Maxx's portfolio project for AI engineering roles,
built as a 10-module course. Modules 0 to 4 are done. The plan for 5 to 9 is
in `docs/roadmap.md`; read it before starting module work.

## Commands

```
uv sync --extra dev                  # install
uv run ruff check . && uv run ruff format --check .
uv run mypy                          # strict, covers src, tests, scripts, migrations
uv run pytest                        # needs Postgres (pgvector) and Redis; see below
uv run alembic upgrade head          # after pulling or changing models
uv run alembic revision --autogenerate -m "..."   # after changing a model
uv run alembic check                 # models and migrations agree
uv run python scripts/seed.py --reset            # two synthetic clinics
uv run python evals/retrieval.py                 # retrieval report -> evals/reports/
uv run python scripts/sms_console.py             # text the agent (real model, costs cents)
docker compose up -d db redis                    # local Postgres + Redis
```

Every change must pass ruff, ruff format, mypy and pytest before a commit.
DB tests skip when Postgres is absent locally; in CI they must run.

## Workflow for every module

1. `git switch main && git pull`, then `git switch -c module-N-<topic>`.
2. Read the module in `docs/roadmap.md`. List what needs a human (accounts,
   money, secrets) and say so at the start, not at the end.
3. Build in small commits. Tests come with the code, not after.
4. For every guarantee the module claims, break the code on purpose once and
   confirm a test fails (mutation check), then restore. Mention it in the PR.
5. One ADR per real decision in `docs/adr/NNNN-title.md`, with the numbers
   that decided it. Update `CHANGELOG.md` (new version section), the README
   status table and the roadmap row, and bump the version in
   `pyproject.toml` and `src/frontdesk_agent/__init__.py`.
6. Push the branch and open a PR with `gh pr create`. The PR body says what
   was built, what was verified and how, and what was NOT verified.
7. Stop and report. Maxx reviews and merges; then tag `vX.Y` on main.

## Rules that must never break

- Never push to main, never force-push, never rewrite pushed history.
- Never commit `.env` or any key or token. Never print secrets in output.
- Synthetic data only. No real patient or client data, ever.
- Tenant isolation: repositories are bound to one tenant at construction and
  no method takes a tenant argument (ADR 0005). Every new table has a
  non-null `tenant_id` FK; `tests/test_schema.py` enforces it.
- Agent tools never take a client or tenant argument; both come from the
  conversation (ADR 0010).
- Risk holds (late cancellation, money, medical, escalation) send nothing
  until a human decides, except the fixed medical safety message (ADR 0011).
- Texts and queued jobs run only after the transaction commits (`Effects`).
- Every model call is recorded in `llm_calls` with tokens, latency and cost.
  Prices live in `src/frontdesk_agent/llm/pricing.py`; verify them against
  https://platform.claude.com/docs/en/about-claude/pricing before changing.
- Tests never call a real model; use `FakeLLMClient` (`script`, `call`, `say`).
- Spending money: ask Maxx before any run expected to cost more than $2 (real
  eval runs, load tests against the real model, cloud resources). Print the
  cost of every paid run.
- Do not run `terraform apply` or `terraform destroy`; prepare the plan and
  ask Maxx to run them.

## Map

```
src/frontdesk_agent/
  api/            HTTP: health, drafts, tenants, policy, reviews, channels (Twilio webhook)
  agent/          tools, risk policy, scheduling, the agent loop (max 6 steps)
  frontdesk/      inbound handling, review decisions, slot filling, bandit, Effects
  retrieval/      chunking, WordLlama embeddings, pgvector store, metrics
  answering/      grounded policy answers with citations
  llm/            LLMClient protocol, Anthropic adapter, fake, pricing, messages
  db/             models, tenant-bound repositories, sessions
  services.py     wiring, run_job, inline and arq queues
  worker.py       arq worker
prompts/<name>/vN.md   versioned prompts; new behaviour = new version file
evals/            golden sets, eval scripts, reports
docs/adr/         decisions 0001-0013; read the relevant ones before changing an area
```

## Style

Python 3.12, FastAPI, SQLAlchemy 2.1 async, psycopg 3, Pydantic v2, mypy
strict, ruff (line length 100). Docstrings explain why, not what. Prefer the
existing patterns over new frameworks; a new dependency needs a reason in the
PR. Windows is the dev machine (PowerShell, Git for Windows); CI is Linux.
