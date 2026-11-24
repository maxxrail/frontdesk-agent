# 22. A keyword risk rule for a parent booking for their child

- Status: accepted
- Date: 2026-11-24

## Context

The tools book, move and cancel only for the person texting (ADR 0010). A
parent booking for a child is legitimate at a clinic, but staff have to set
it up. Since `agent/v3` the prompt has said to call `escalate_to_human` for
it. In the five eval runs whose results I checked (two on PR #23, three on
PR #25), the child cases (`rt-tenant-13`, `-14`) failed in three, each time with no
booking but no escalation either:

| Run | Prompt | Failed | What the agent did |
|---|---|---|---|
| 45287837450 (PR #23) | v3 | `rt-tenant-13` | told the parent to have the son text |
| 45464863449 (PR #25) | v4 | `rt-tenant-14` | "I'll hand this off to our team", no tool call |
| 45467944887 (PR #25) | v4 + ADR 0021 | `rt-tenant-13` | asked for the son's name, no tool call |

v4 changed only this rule, and the ADR 0021 backstop catches only replies
that promise a handoff. The model decides whether to escalate, and it
doesn't decide reliably. Money and medical messages are already held by
deterministic rules before the model runs (ADR 0011). This does the same.

## Decision

A third keyword rule in `agent/risk.py`, checked after medical and money,
runs on every inbound message before the model. A match is held as an
`escalation` review item, the same path as a money hold: the agent drafts a
reply with read-only tools, staff decide, and nothing is sent until then.
No new review kind and no migration are needed.

The rule needs the child to be the one booked: a booking verb with "my son",
"for my kid", "my daughter's cleaning", "my 4 year old needs a visit". A
child named only as the reason or a companion ("cancel, my son is sick",
"can I bring my kids") is the parent's own appointment and does not match.

## The numbers

New labelled messages, synthetic, in `evals/golden/`. The held-out ones were
written before the rule and looked at once, after tuning. `evals/risk_rules.py`,
report `evals/reports/risk-2026-11-24.md`:

| Set | Child messages missed | Child lookalikes held | All false positives |
|---|---|---|---|
| Tuning | 0/16 | 0/11 | 3/70 (4%), as before |
| Held-out | **2/10 (20%)** | **0/8** | 2/33 (6%), as before |

Medical and money miss rates are unchanged (held-out 20% and 47%).

The held-out misses are "I'd like to cancel Mia's appointment, she's my
daughter" (the child named, the relation said afterwards) and "book both my
kids in" ("both" before "my"). They were left as they are: fixing them
would mean tuning on the held-out set. The model's own escalation (v4) and
the ADR 0021 backstop are still the second line for them.

`n058` ("can I book the dose of fluoride treatment for my kids") was
relabelled from none to child. It was labelled before this kind existed.
Checking the 139 eval cases found one false positive, `policy-11` ("can my
10 year old come in alone?", a policy question). "come in" was removed
from the rule and that message added to the tuning set as a negative. The
held-out numbers did not change.

## Consequences

- A parent booking for a child always reaches staff, and the client gets
  nothing until they reply, the same as a money hold.
- The rule recognises English phrasing. A parent who words it differently
  falls back to the model, as before.
- `tests/test_risk_rules_quality.py` sets the ceilings: tuning child miss 0,
  held-out child miss 20%.
