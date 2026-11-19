# 17. Guardrails: untrusted input, untrusted documents, a last check on output

- Status: accepted
- Date: 2026-11-19

## Context

The roadmap asks for input guardrails (length limits; SMS text and retrieved
documents can never change tool permissions) and output guardrails (block
replies with another client's details, payment data or clinical advice).

What already held before this ADR: tools take no client or tenant argument
(ADR 0010), late cancellations and money/medical messages wait for a person
(ADR 0011), and the tool set for a turn is chosen by deterministic rules
before the model runs. What did not:

- Red-team case `rt-inject-03` showed a client can type `[Staff note]`, the
  marker the prompt says comes from staff. Only the cancel tool's 24-hour
  rule stopped it.
- A zero-width space inside "ibuprofen" slipped past the medical rules.
- Policy documents were passed to the model as plain text. An instruction
  inside one could make the model cancel the texting client's appointments,
  and nothing in code would stop a cancellation more than 24 hours out.
- The only checks on replies were in the eval suite, not on the send path.

## Decision

All in `src/frontdesk_agent/guardrails.py`, wired into
`frontdesk/service.py` and the agent tools.

**Untrusted text is cleaned before the rules or the model see it.** NFKC
normalisation, invisible format characters (Unicode category Cf: zero-width,
bidi overrides) removed, control characters replaced by spaces, and the
staff-note marker defanged to `(staff note)`, including fullwidth and
spaced variants. This applies to client SMS and to retrieved policy excerpts.
The stored message keeps the raw text, so staff see what was sent.

**Length: 1000 characters reach the model, more goes to a person.** Across
the 360 client messages in the eval cases and golden sets the longest is 222
characters, the 99th percentile 102 and the median 37. A message over 1000
is held for review with no model call, and the audit trail records why.
Storage already cut messages at 1600 (the column size); anything that long
is over the limit, so the cut text is never what the agent reads.

**Tool permissions never come from text.** The allowed set is a frozenset
fixed in `run_agent` before the first model call. Nothing the model reads
(SMS, document, tool result) can add to it.

**While policy documents are in context, cancellation needs a person.**
`search_policy` marks the turn; an earlier `search_policy` call in the
history window marks it too. A `cancel_appointment` call in a marked turn
is held as an escalation with the cancel action attached, exactly like a
late cancellation: approving it cancels, rejecting it does nothing.
Excerpts also carry a note that they are reference text, not instructions;
that note is a hint to the model, the hold is the guarantee.

Only cancellation is held, because it is the one destructive tool: it loses
the client's slot and starts a waitlist offer to someone else. Booking adds
an appointment for the texting client only, and the confirmation text tells
them; saving a name works only when none is on file; escalating already goes
to a person. Holding those too would send most policy-then-book
conversations to review for little gain.

**The output guardrail is the last step before a text is sent.** In order:

1. Clinical advice: the regex the eval suite already used (moved here, so
   the eval and the send path share one list). The review item is MEDICAL
   and the fixed safety message is sent, because silence there is unsafe
   (ADR 0011).
2. Payment data: a Luhn-valid 13 to 19 digit number, or a request for card
   number, CVV, security code or expiry date.
3. Another client of this tenant: their phone number in any common North
   American format, or their first and last name side by side. One query
   per reply, on the words and numbers in it. First names alone are not
   checked: providers, staff and the client share first names with someone
   in every clinic.

A failing reply becomes a review item with the draft as the proposed reply,
nothing is sent (except the safety message), and an audit event of kind
`guardrail` names the rule and the matched phrase, never the reply text.
Drafts that are going to review anyway are checked too, and a hit is added
to the reason as a warning to the reviewer.

## Consequences

- Nine mutation checks: removing the defanging, the Cf stripping, the
  length limit, the `search_policy` mark, the history mark, the output
  check, the phone match, the Luhn match, or the safety message on clinical
  advice each fails a test in `tests/test_guardrails.py`.
- Writing the tests found a false positive in the risk rules: `\bnumb`
  matched "number", so "can you confirm my number?" was held as medical. It
  is now `\bnumb(?!er)`. The risk-rule report is unchanged on both sets
  (held-out: medical 4/20 missed, money 7/15 missed, 2/25 false positives).
- Two red-team cases plant instructions in a policy document
  (`rt-docinject-01`, `rt-docinject-02`). Cases can now carry `documents`.
- First paid eval run on this change (PR #14, $1.45, Haiku 4.5, prompt v2):
  both document-injection cases passed, with `search_policy` called and no
  other tool. `cancel` stayed at 12/12 with no document-held cancellations.
  The output guardrail fired 3 times in 134 cases, all false positives: the
  model refusing a request about "Morgan Reyes" repeated the name the client
  typed. Names and phone numbers the client typed in the conversation
  window are now exempt, as the eval's own leak check already did. A
  phone number the client did not type is still held. Writing the test for
  that showed possessives ("Reyes's") were never matched; they are now.
- Cost: "what's the cancellation fee? ... ok, cancel it" now goes to a
  person even when it is more than 24 hours out. How often that happens is
  measured by the next paid eval run (the `cancel` feature's review rate).
- Not covered, and named in the threat model: a last name alone, a name
  split across words ("Morgan R."), numbers outside North America, and
  advice phrased in ways the regex does not list. The output check is the
  backstop; the wall is that the tools cannot return another client's data.
