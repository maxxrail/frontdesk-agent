# 20. An outbox for outbound SMS, and running jobs in process when Redis is down

- Status: accepted
- Date: 2026-11-23

## Context

ADR 0012 left a gap: texts were sent after the transaction committed and
logged after they were sent, so a process that died in between sent a text
with no record of it. The roadmap's done-when for this module also says
that killing Redis during a load test must raise an alert while the API
degrades instead of crashing. Looking at that path showed a worse gap. The
webhook stores the message, then enqueues its job. With Redis down the
enqueue raised and Twilio got a 500. Twilio's retry is then a duplicate,
which by design queues nothing (ADR 0012), so the client was never answered.

## Decision

**Transactional outbox.** `Effects.commit` writes each text as a `queued`
`sms_messages` row in the same transaction as the booking or decision it
reports. After commit, `deliver` locks each row (`FOR UPDATE SKIP LOCKED`),
sends it and marks it `sent` or `failed`. A worker cron job (every minute,
and at startup) sends rows that have been queued for over 60 s. Calling
`flush` without `commit` raises rather than dropping texts.

Delivery is **at least once**. If the process dies after the provider
accepts a text but before its row is marked, the sweeper sends it again.
For a front desk, a rare duplicate confirmation is better than a booking
the client never hears about. If a sent text's row will not save, it is
marked with its send result and the provider id dropped, never left queued,
so no failure can make the sweeper resend in a loop.

**Run jobs in process when Redis is unavailable.** `ArqQueue` falls back to
the in-process queue on a Redis or socket error, counts it in
`frontdesk_queue_fallback_total` and logs a warning. Each Redis attempt is
capped at 2 s, and after a failure Redis is skipped for 10 s, so an outage
costs one slow request per 10 s, not every request. Deferred jobs (offer
expiry, the tenant-limit retry) run that way are lost if the API restarts
during the outage. `DependencyDown{dependency="redis"}` pages for it.

## Numbers that decided it

From the load test (`loadtest/README.md`), 50 clients, Redis stopped for 60 s:

| | First run | After the fixes |
|---|---|---|
| Webhook requests failed | 0 of 414 | 0 of 398 |
| Webhook p99 / max | 19 s / 42 s | 1.9 s / 3.8 s |
| Texts left unanswered | 0 | 0 |
| Outbox rows left queued | 413 | 0 |
| Reply p95 | not measured correctly [^1] | 3.24 s |

The first run's 42 s requests are past Twilio's 15 s timeout. They came from
name lookups and a Redis that was still loading, which the connect timeout
alone does not cover. The 413 queued rows were a bug in the local console
sender, which gave every text the same provider id and broke a unique index.
With the outbox that would have looped. Both are fixed and tested.

[^1]: The report first measured from the outbound row's `created_at`, which
    in Postgres is when the job's transaction began. It now uses the time the
    send is recorded.

## Consequences

- An outbound row can be `queued` for up to a minute after a crash; the
  dashboard's message counts include them.
- In-process fallback puts agent turns on the API's event loop and database
  pool while Redis is down. At 50 clients that was fine (reply p95 3.24 s).
  Module 8's deploy should alert on `frontdesk_queue_fallback_total` rising,
  not only on Redis being down.
- Jobs already in Redis when it died ran once the worker came back, up to
  34 s late, but none were lost.
