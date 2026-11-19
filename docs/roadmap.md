# Roadmap: modules 5 to 9

Modules 0 to 5 are complete (v0.5.0, 379 tests). Each module below ends with a
merged PR and a tag. "Needs Maxx" lists what an agent cannot do alone.

## Module 5: Evals that gate pull requests (v0.5) ✅

Done in v0.5.0 (PR #4). Results: `evals/README.md`; open follow-ups: issues #5-#11.

No prompt or model change merges without a measured effect.

Build:
- `evals/` with three layers, each runnable alone:
  1. Deterministic checks: valid JSON, right tool chosen, correct slot
     booked, reply under 320 characters, no medical-advice phrases.
  2. Model-graded checks: an LLM judge scores tone and faithfulness to the
     cited policy against a written rubric. Validate the judge first:
     prepare 40 outputs for Maxx to hand-label, then report agreement.
  3. Simulated conversations: a scripted client model runs 30 multi-turn
     scenarios end to end against a test database.
- At least 150 cases in version control, tagged by feature and difficulty.
  Every bug found becomes a case. Reuse `evals/golden/retrieval.jsonl` and
  `evals/policy_answers.py` from Module 3.
- Measure the risk rules (`agent/risk.py`): false-positive and false-negative
  rates on labelled messages.
- A red-team set of 40 cases: prompt injection inside an SMS, requests for
  another client's data, attempts to book for another tenant, pressure for a
  diagnosis.
- A GitHub Actions job that runs the suite when `prompts/`, `src/frontdesk_agent/agent/`
  or the model setting changes, posts a score table on the PR, and fails
  below thresholds. A results file per run, chartable across prompt versions.
- Compare `claude-haiku-4-5` with `claude-haiku-5-5` (a tenth of the price)
  and record the decision in an ADR.

Done when:
- A deliberately worsened prompt is blocked by CI, and the PR shows which
  cases regressed.
- Judge agreement with the hand labels is a number in `evals/README.md`.
- All 40 red-team cases pass, or each failure has an open issue.
- A full run costs under the budget Maxx sets, printed at the end.

Needs Maxx: the 40 hand labels; `ANTHROPIC_API_KEY` as a GitHub Actions
secret (Settings > Secrets and variables > Actions); the eval budget.

## Module 6: Guardrails, security and privacy (v0.6) ✅

Done in v0.6.0 (PR #14). Threat model: `docs/threat-model.md`; decisions: ADR 0016-0018.

The system fails safe and leaves an audit trail.

Build:
- API key auth per tenant for machine clients and JWT login for dashboard
  users, with reviewer and admin roles. The tenant comes from the credential,
  replacing today's tenant-in-URL.
- Input guardrails: length limits; SMS text and retrieved documents are
  untrusted data that can never change tool permissions.
- Output guardrails: a final check that blocks replies containing another
  client's details, payment data or clinical advice.
- Redaction of phone numbers and names before anything reaches logs or
  traces; raw text lives only in Postgres.
- Rate limiting per tenant and per phone number; an audit event for every
  tool call and every review decision.
- Dependabot, `pip-audit` and secret scanning, all blocking CI.
- `docs/threat-model.md` walking the OWASP Top 10 for LLM Applications.

Done when:
- Searching application logs for any seeded phone number returns nothing.
- An injected instruction inside a retrieved policy document does not
  trigger a tool call, shown by an eval case.
- The threat model names the residual risks accepted, and why.

Needs Maxx: enabling Dependabot and secret scanning in the GitHub repo settings.

## Module 7: Observability, cost and latency (v0.7)

Answer what a conversation cost, how slow the slowest were, and why one failed.

Build:
- Structured JSON logs with request ID and conversation ID on every line.
- OpenTelemetry traces: one SMS is one trace, with spans for retrieval, each
  model call and each tool call.
- Prometheus metrics and a Grafana dashboard: requests, error rate, p50/p95
  latency, tokens and dollars per conversation, tool failure rate, review
  queue depth, eval score by prompt version.
- Sentry for exceptions, redaction applied first.
- Cost controls: prompt caching for the long system prompt, a small model for
  classification and a stronger one only for the final reply, and a
  per-tenant daily budget that degrades to a canned response.
- An outbox table for outbound SMS (ADR 0012 notes the gap).
- A Locust load test: 50 concurrent conversations with the fake client, then
  a short run against the real model.
- Three written SLOs (e.g. p95 reply under 8 s), one alert each.

Done when:
- From one conversation ID you can open the full trace with every span's
  latency and cost.
- A README table shows cost per conversation before and after routing and
  caching.
- Killing Redis during the load test raises an alert and the API degrades
  instead of crashing.

Needs Maxx: a Sentry account (free tier) and its DSN; approval for the
real-model load test.

## Module 8: Deploy to AWS with Terraform and continuous delivery (v1.0)

Live at a public URL; every merge to main reaches it without a console.

Build:
- Terraform for the stack: ECS Fargate or App Runner, RDS Postgres with
  pgvector, ElastiCache or a small Redis container, Secrets Manager,
  CloudWatch logs. Staging and production from the same modules.
- GitHub Actions: build and push the image, run migrations, deploy to
  staging, smoke test plus a 20-case eval subset, then production behind a
  manual approval.
- Zero-downtime rolling deploys and a one-command rollback.
- Prompt versions behind a feature flag, with a shadow mode that runs a new
  prompt on live traffic and logs its output without sending it.
- `docs/runbook.md`: provider outage, runaway cost alert, bad prompt release,
  database restore (practise the restore once and time it).
- One postmortem for a failure caused on purpose in staging.

Done when:
- A merged PR is live in production within 15 minutes with no manual step
  except the approval click.
- Rollback is rehearsed and takes under 5 minutes.
- `terraform destroy` then `terraform apply` rebuilds a working environment.
- `v1.0` is tagged.

Needs Maxx: an AWS account with a billing alarm set first; running
`terraform apply`/`destroy`; AWS credentials as GitHub secrets (prefer OIDC);
a Twilio number pointed at the deployed webhook.

## Module 9: TypeScript dashboard and portfolio polish (v1.1)

A reviewer gives the repo about two minutes; make them count.

Build:
- A React + TypeScript (Vite) review dashboard: the review queue, a
  conversation viewer with the trace link, and eval scores by prompt version.
  Generate the API client from FastAPI's OpenAPI schema.
- A README that leads with the live URL, a 90-second demo video, the
  architecture diagram, and one table of real numbers: eval pass rate,
  recall@5, p95 latency, cost per conversation. Every number links to the
  script or dashboard that produced it.
- A short write-up of the three hardest problems and how each was measured
  out of. Publish as a blog post or LinkedIn article.

Done when:
- A stranger can text the demo number and watch the conversation appear in
  the dashboard.
- `v1.1` is tagged and the repo is pinned on Maxx's GitHub profile.

Needs Maxx: recording the demo video; publishing the write-up; pinning the repo.
