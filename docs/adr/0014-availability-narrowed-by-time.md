# 14. check_availability takes an earliest start time

- Status: accepted
- Date: 2026-11-16

## Context

`check_availability` shows at most `MAX_SLOTS_SHOWN = 8` open slots, sorted by
time, plus a count of the rest. The cap keeps tool results (and so input
tokens) small. The demo clinics have two providers on 30-minute slots, so a
full day has far more slots than that:

| Clinic           | Open slots on a full day |
|------------------|--------------------------|
| Maple Dental     | 44 (Mon-Thu), 32 (Fri)   |
| Northside Physio | 40-48 (weekdays), 20 (Sat) |

The 8 shown are the first two hours after opening. The agent prompt says to
offer only times the tool returned, so a client who asks for "tomorrow at 2"
could not be booked without the model guessing a slot it was never shown.
In the eval set, **14 of the 18 expected bookings** (78%) fall outside the
first 8 slots. The eval runner's own end-to-end tests failed on this once
they first ran against Postgres.

## Options

1. **An optional `from_time` filter** (HH:MM, clinic time). The result keeps
   its size; the model asks for the part of the day the client wants.
2. **Raise the cap** to cover a whole day: 48 slots is about 6x the tool
   result's tokens on every availability call, for every client, to serve
   the minority who name a time.
3. **One row per time** instead of per provider and time: halves the rows
   but still reaches only 4 hours, and hides who is free.

## Decision

Option 1. `from_time` is optional, so existing calls behave as before. Its
description tells the model when to set it ("at 2" -> 14:00, "afternoon" ->
12:00) and why (later times are cut off). The prompt stays at `v1`: the
change is in the tool schema the model already receives, and keeping the
prompt fixed means a prompt comparison in the evals still compares only
prompts.

## Consequences

- `tests/test_scheduling_tools_risk.py::test_from_time_reaches_slots_past_the_first_page`
  pins the behaviour; removing the filter fails it and three eval runner tests.
- Whether the real model actually sets `from_time` is measured by the paid
  booking cases, not assumed. If it does not, the next step is a prompt rule
  (a `v2`), compared against `v1` in the evals.
- No `to_time`: the result is already sorted, and an earliest start covers
  "after 4", "at 2" and "afternoon". "Before 10" is served by the default.

## Addendum: providers by name (2026-11-16)

The first paid run (agent/v1, Haiku 4.5, $1.67) confirmed the model sets
`from_time`: in every failed booking case reviewed where the client named a
time, the call set it. It also showed a
second gap in the same tool. In **6 of 132 cases** the model passed a
provider's name ("Lena Park") as `provider_id`, which failed validation and
cost a retry. Before its first availability call the model has no ids, only
the name the client used.

`provider_id` became `provider`, taking a name or an id. Every word given,
minus titles ("Dr", "with"), must appear in exactly one active provider's
name in this clinic; otherwise the error lists the clinic's providers so the
model can ask. Lookup goes through the tenant-bound `ProviderRepository`, so
another clinic's provider is unknown by name and by id (tested).
