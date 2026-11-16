# 12. SMS in through a verified webhook, work done by a queue worker

- Status: accepted
- Date: 2026-11-14

## Context

Twilio posts each inbound SMS to a webhook, waits up to 15 seconds, and
retries on failure. An agent turn can take several model calls. Anyone who
finds the URL can post to it.

## Decision

The webhook does the minimum and returns at once:

1. **Signature check** with Twilio's `RequestValidator` against the
   configured public URL (`FDA_PUBLIC_BASE_URL`), because behind a tunnel or
   load balancer the URL this process sees is not the one Twilio signed.
   Missing, wrong, tampered or wrong-URL signatures get a 403. With no auth
   token configured, every request gets a 503, never a silent accept.
2. Route by the number texted, find or create the client, store the message.
   A unique index on `(tenant_id, provider_sid)` makes a Twilio retry a no-op.
3. Commit, enqueue `handle_inbound`, reply with empty TwiML.

Replies go out later from the worker through Twilio's REST API.

The queue is **arq on Redis** (`frontdesk_agent.worker`), with three jobs:
`handle_inbound`, `start_slot_fill` and `expire_offer` (delayed 15 minutes).
For local development `FDA_JOB_QUEUE=inline` runs the same jobs in the API
process. Both go through one `run_job` function, so they cannot drift.

Side effects are collected during a job and run **only after commit**
(`frontdesk/ports.py: Effects`): texts are sent and follow-up jobs queued
once the booking or cancellation they describe is durable. Each outbound
text is logged, including failed sends.

Each conversation row is locked for the length of a turn, so two quick texts
from one client cannot interleave and corrupt the transcript.

## Consequences

- `tests/test_worker_redis.py` runs a real arq worker against Redis. It
  caught a bug no other test could: arq registers jobs by `__qualname__`, and
  every job would have failed in production with "function not found".
- If the process dies after sending a text but before logging it, the log
  misses one row. An outbox table would close that; it is noted for Module 7.
- A message that is exactly `CANCEL` is an opt-out keyword to Twilio, so it
  never becomes a cancellation request. Clients write "cancel my
  appointment"; the README says how to change Twilio's keywords.
- arq is in maintenance mode. It is small, async and enough here; the queue
  sits behind a two-method interface if it needs replacing.
