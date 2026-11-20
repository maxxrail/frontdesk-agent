# Service level objectives

Three promises about what a clinic and its clients experience, each with one
alert. The rules are in `ops/prometheus/alerts.yml`, tested by
`ops/prometheus/alerts_test.yml` (`promtool test rules`, run in CI), and
drawn on the Grafana dashboard (`ops/grafana`). Locally,
`docker compose up -d` shows firing alerts at http://localhost:9090/alerts.
Sending them to a person (email, Slack, a pager) comes with the deploy in
Module 8.

| # | Objective | Measured by | Alert | Fires after |
|---|---|---|---|---|
| 1 | 95% of inbound SMS are handled within **8 seconds** | p95 of `frontdesk_sms_turn_seconds` over 5 min | `SmsReplyLatencySLO` | 5 min over |
| 2 | **99.5%** of Twilio webhook requests succeed | 5xx share of `frontdesk_http_requests_total{route="/v1/channels/sms"}` over 5 min | `WebhookAvailabilitySLO` | 5 min over |
| 3 | **98%** of model calls succeed | `outcome="error"` share of `frontdesk_llm_calls_total` over 10 min | `ModelCallErrorSLO` | 10 min over |

Two more alerts say the system itself is unwell rather than an objective
being missed: `DependencyDown` (Postgres or Redis stopped answering the API's
probe for 30 s) and `TargetDown` (Prometheus cannot scrape the API or the
worker for 1 min).

## Why these numbers

**8 seconds.** A text conversation tolerates a pause; a client who has not
heard back within ten seconds often texts again. Twilio waits 15 seconds for
the webhook, but the webhook returns at once and the turn runs in the
worker, so the 8 seconds is the worker's: up to six model calls at Claude
Haiku 4.5's measured p95 of about 1.5 s per call (PR #14 eval runs) leaves
little room, which is why the cost-routing work in this module also
watches latency. The time covers the agent turn up to the reply or the hold;
SMS delivery by the carrier is outside it.

**99.5% for the webhook.** The webhook only verifies, stores and queues, so
it should almost never fail; a 5xx makes Twilio retry, and a stored message
is never lost. At a clinic's volume (hundreds of texts a day) 0.5% is one or
two failures a day before the alert fires.

**98% for model calls.** Counted after the client's own retries (`llm/retry.py`),
so these are calls that really failed. A failed call fails its job, which
the worker retries; the client's message stays stored, so this is a ticket,
not a page.

## Error budgets

Over 30 days: SLO 1 allows 5% of turns over 8 s; SLO 2 allows 0.5% of
webhook requests to fail; SLO 3 allows 2% of model calls to fail. The
alerts above are simple threshold alerts over short windows. Burn-rate
alerts over the 30-day budget are worth adding once there is real traffic to
calibrate them against.
