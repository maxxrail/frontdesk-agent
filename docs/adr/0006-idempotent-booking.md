# 6. Idempotent booking enforced by the database

- Status: accepted
- Date: 2026-11-12

## Context

SMS providers retry webhooks when they do not get a fast response, and the
agent itself may retry a tool call. Without protection, one text message
becomes two appointments.

The obvious fix, look up the key and insert if it is absent, has a race: two
requests arriving together both see no row and both insert.

## Decision

- `appointments.idempotency_key` is optional, and a **unique partial index**
  on `(tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL`
  guarantees at most one booking per key per tenant.
- `book()` inserts with `ON CONFLICT DO NOTHING`. If nothing was inserted,
  it returns the existing booking with `created=False`.
- Reusing a key for a *different* booking (another time, client or
  provider) is an error, not a silent replay: the API returns 409.
- Over HTTP the key is the `Idempotency-Key` header. A first booking is 201,
  a replay 200 with the same body.

## Consequences

- `test_concurrent_replays_race_safely` fires ten simultaneous bookings on
  separate connections with one key and asserts one row. Swapping in
  check-then-insert makes it fail, so the test guards the real property.
- Keys are scoped per tenant, so two businesses cannot collide.
- Bookings without a key, from a dashboard for example, are never
  deduplicated. That is intended.
- Keys are never expired. At this volume the table is small; Module 7 can
  add retention if it matters.
