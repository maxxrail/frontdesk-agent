# 13. Filling cancelled slots: a workflow, ranked by a bandit

- Status: accepted
- Date: 2026-11-14

## Context

When an appointment is cancelled the slot should go to someone on the
waitlist. Asking a model to run this would add cost and unpredictability to
a process whose every step is known.

## Decision

Slot filling is a **workflow**, not an agent
(Anthropic's "Building effective agents" draws the same line):

1. A committed cancellation queues `start_slot_fill`.
2. Waiting clients whose window covers the slot are ranked; the first gets a
   text: reply YES within 15 minutes, or NO.
3. YES books it. The partial unique index on booked
   `(provider_id, starts_at)` makes "first yes wins" a database guarantee:
   a YES after someone else booked the slot gets a polite "just booked"
   instead of a double booking. NO, or no reply by the `expire_offer` job,
   moves to the next client.

Offers go out one at a time, so nobody is told a slot is theirs and then
told it is gone.

**Ranking is an epsilon-greedy bandit** (`frontdesk/bandit.py`). Each client
is an arm, and the reward is how fast they took the offer: 1.0 for an instant
yes, falling to 0 at the window's end, 0 for no or silence. With probability
0.9 it offers first to the best estimated responder; with 0.1 it explores. A
prior of 0.5 (weight 1) stops one lucky answer from dominating, and ties go
to whoever joined the waitlist first.

In a simulation of six clients with hidden answer rates (`tests/test_bandit.py`,
400 slots, 20 seeds) the first offer earns a mean reward of **0.654** with
the bandit, against 0.192 first-come-first-served and 0.218 random. The best
possible is 0.72.

## Consequences

- Only YES/NO-style replies go to the workflow. Anything else ("yes but can
  I do 10 instead") goes to the agent as usual.
- The bandit optimises for the clinic, not for fairness. Exploration and the
  FIFO tie-break soften that, but a client who never answers will be offered
  slots less often. That trade-off belongs in the clinic's settings in a
  real product.
- Stats are per client per tenant, in `client_response_stats`.
