# 18. Rate limits per phone and per tenant, counted in Postgres

- Status: accepted
- Date: 2026-11-19

## Context

Every inbound text can cost up to six model calls (ADR 0010). Nothing
stopped one phone, or a script sending from many numbers, from running up
a clinic's model bill or burying staff. The roadmap asks for rate limits
per tenant and per phone number.

## Decision

**Count rows that already exist, in Postgres.** Inbound texts are in
`sms_messages` (indexed on conversation and time), model calls in
`llm_calls` (indexed on tenant and time). Both checks run in `handle_inbound`
under the conversation lock, before any model call, in database time
(`now()`). Redis was the other choice. It is not there in inline mode (local
development, the SMS console, tests), and a counter in Redis can disagree
with what Postgres says happened. At front-desk volumes two indexed counts
per message cost nothing that matters.

**Per phone: 12 texts in 10 minutes, then a person.** A phone is one client
and one conversation within a tenant. The longest scripted eval case sends 3
texts and simulated conversations stop at 10 turns, so no legitimate eval
conversation comes near 12 in 10 minutes. The text that crosses the limit
becomes a review item with no model call, and the conversation pauses, so
the rest of a flood is held with no model calls either. The window counts
only texts after the latest review decision in the conversation: once a
person has looked, the client starts fresh. Without that, approving would
re-hold the queued texts at once. STOP and START are checked first, so
opting out is never limited.

**Per tenant: 1000 model calls an hour, then defer.** That is about 170
agent turns at the six-step worst case, or several hundred typical ones: far
above a single clinic's front desk, low enough to cap a runaway at a few
dollars an hour on Haiku 4.5. Past it, a new message is re-queued after 60
seconds and stays `received`. It is not dropped, because the client did
nothing wrong. It is not held either, because one review item per message
would bury the queue. The count is of model calls, not messages, so
deferred messages do not count against the tenant, and the backlog drains
as the window moves.

All four numbers are settings (`FDA_RATE_*`).

## Consequences

- Four mutation checks: removing either limit, the reset after a decision,
  or the per-conversation scope each fails a test in
  `tests/test_rate_limits.py`.
- One phone texting several clinics is limited by each clinic separately.
  A cross-tenant count would need a query that is not bound to one tenant
  (ADR 0005); the per-tenant cap bounds the damage instead.
- A deferred message waits at least a minute; under a sustained flood the
  wait grows. Module 7's per-tenant daily budget with a canned reply covers
  cost over days; this covers bursts.
- Not limited here: the HTTP API, including login attempts. Logins use
  scrypt (about 55 ms a guess, ADR 0016); throttling them is named in the
  threat model as a residual risk.
