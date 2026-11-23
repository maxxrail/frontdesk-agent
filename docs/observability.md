# Observability

Module 7 set out to answer three questions about any conversation: what it
cost, how slow the slowest replies were, and why one failed. This page says
where each answer is. Locally, `docker compose up -d` starts everything below.

| Tool | Local URL | What it holds |
|---|---|---|
| Jaeger | http://localhost:16686 | One trace per inbound SMS |
| Prometheus | http://localhost:9090 | Metrics and the alert rules (`/alerts`) |
| Grafana | http://localhost:3000 | The `frontdesk-agent` dashboard |
| Sentry | sentry.io, when `FDA_SENTRY_DSN` is set | Exceptions, redacted before sending |
| Logs | `docker compose logs api worker` | JSON lines, redacted |

## What did this conversation cost, and where did the time go?

Open Jaeger, pick the service `frontdesk-agent`, and search by tag:

```
conversation.id=<the conversation's uuid>
```

Each inbound text is one trace. It starts at the webhook (`POST
/v1/channels/sms`) and continues through the queued job (`job
handle_inbound`) into `agent.turn`, which carries the turn's total
`agent.cost_usd`. Under it, every `llm.call` span has its model, tokens
(cache reads and writes included), latency and `llm.cost_usd`, and every
`tool <name>` span has its own latency. Verified on 2026-11-23: a load-test
text's trace had 10 spans, from the 57 ms webhook to the three 0.9 s model
calls.

The conversation id is in every log line of that turn too
(`conversation_id`), next to `request_id` and `trace_id`, so a log line
leads to its trace.

For a clinic rather than one conversation, `GET /v1/llm-calls` (admin) lists
recent model calls with tokens, latency and cost.

## How slow are the slowest replies?

Grafana, row "SMS turns": p50 and p95 handling time. `SmsReplyLatencySLO`
fires when p95 stays over 8 s for 5 minutes (`docs/slo.md`). Under load,
`loadtest/report.py` measures the time to the reply's send from the
database (`loadtest/README.md`).

## Why did this one fail?

- A job that raised: the stack trace is in Sentry, with phone numbers, names
  and message text scrubbed before sending (`errortracking.py`). The error's
  log line carries the job, tenant, conversation, request and trace ids, so
  it leads to the trace.
- A reply that was held, not failed: the review queue, and the `guardrail`
  rows in `audit_events` naming the rule.
- A text that never went out: `sms_messages` with status `failed`, or
  `queued` for over a minute, which means the outbox sweeper has not yet
  caught up (ADR 0020).

## Cost controls on the dashboard

Row "Model cost and latency": dollars per SMS turn, spend per hour, and
three panels for ADR 0019 and 0020. They show how often the cheap model's
draft is kept, the share of input tokens read from the prompt cache, and
jobs the API ran itself because Redis was down.
