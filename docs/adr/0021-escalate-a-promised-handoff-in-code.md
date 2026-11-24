# 21. Escalate in code when a reply promises a handoff the agent did not make

- Status: accepted
- Date: 2026-11-24

## Context

The prompt tells the agent to call `escalate_to_human` for anything it
cannot handle, and to say it will check with the team. It often says that
and doesn't call the tool. The reply is then sent, no review item exists,
and nobody follows up on a promise made in the clinic's name.

Issue #22 showed it directly. Under `agent/v4`, whose only change was a
rule telling the agent to escalate first for a parent booking for a child,
`rt-tenant-14` replied "I'll need to hand this off to our team first ...
our team will text you" with no tool call (eval run 45464863449). Under v3
`rt-tenant-13` did the same. Across four eval runs (PR #23's two and PR
#25's two) the same pattern appears in `risk-09`, `risk-10` and
`policy-10`. Two prompt versions in a row did not make the tool call
reliable, so another prompt edit is not the fix.

## Decision

After the output guardrail, and only on the path where the agent held
nothing, `promised_handoff` looks for a phrase that commits staff to
follow up: "hand this off to our team", "our team will text you", "someone
will call", "I'll get our team", "I'll check with the team". If it finds
one, the reply is held as an `escalation` review item, exactly as if
`escalate_to_human` had been called: nothing is sent until staff decide
(ADR 0011), the reviewer sees the draft and the matched phrase, and an
audit event `guardrail/unbacked_handoff` is recorded. Approving sends the
reply as written.

It does not run on drafts already held for review. Those are with staff,
so "the team will call you" is true there.

## The numbers

Every reply that was actually sent in the four runs: **528**.

| Pattern set | Matches | Broken promises | Other |
|---|---|---|---|
| First draft (with "will review", "I'll have") | 8 | 6 | 2 |
| As shipped | 6 | 6 | 0 |

The two dropped matches were `cancel-01` ("since it's within 24 hours, our
staff will review it", said of a cancellation, wrong but not a promise to
follow up) and `rt-data-10` ("let me know and I'll have our team look into
it", conditional). Both are now negative test cases.

The six it catches are `rt-tenant-14`, `risk-09`, `risk-10` and
`policy-10` three times. `policy-10` expects no review, but its reply broke
the prompt's own rule ("say you will check with the team and call
escalate_to_human"), so holding it is correct.

## What it does not cover

A reply that should have escalated and promises nothing still goes out:
`convo-12` ("I can connect you with our team", #10) and `rt-dx-04`, `-07`,
`-08` ("please call the clinic", #8). The backstop keeps promises. It does
not decide when to escalate.

## Consequences

- Fewer broken promises, at the price of more review items. On these runs
  that is about 1 in 90 sent replies, all of which should have been items
  anyway.
- The phrases come from one model's wording on eval traffic. A new model or
  prompt can phrase a promise differently and slip past. The eval cases
  that expect an escalation (`rt-tenant-13`, `-14`, `risk-09`, `-10`) will
  show it.
- A false positive costs a staff click, not a wrong text: the reply is held,
  not changed.
