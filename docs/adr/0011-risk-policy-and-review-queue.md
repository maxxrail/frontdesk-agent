# 11. The risk policy is code, and holds go to a review queue

- Status: accepted
- Date: 2026-11-14

## Context

Some replies must not reach a client until a person agrees: a cancellation
that triggers a fee, anything about money, anything medical. A prompt that
says "escalate medical questions" is a request to the model, not a control.

## Decision

Three layers, none of which depends on the model behaving:

1. **Inbound rules** (`agent/risk.py`) run on every message before the
   model. Medical (symptoms, medication, "is this normal") and money
   (refunds, disputes, balances, waivers) are matched by regular
   expressions. Prices are not "money": "how much is a cleaning" is a policy
   question the documents answer.
2. **Tool rules.** `cancel_appointment` holds any cancellation less than 24
   hours ahead instead of executing it.
3. **The model's own judgement**, through `escalate_to_human`.

A hold creates a `review_item` and **pauses** the conversation: later
messages are stored as held and not answered. For rule-flagged messages the
agent still drafts a reply, with read-only tools only, so the reviewer
starts from a draft. Nothing is sent, with one exception: a medical hold
sends a fixed message pointing to emergency services, because silence there
is unsafe. That text is code, never model-written.

Reviewers approve, edit and approve, or reject
(`/v1/tenants/{id}/reviews/{id}/approve|reject`). Approval runs the held
action (the cancellation) and sends the draft or the edit. Rejection runs
nothing and sends only what the reviewer wrote. A staff note is added to the
history, the conversation resumes, and held messages are processed in order.

## Consequences

- Keyword rules over-flag on purpose: a false positive costs a reviewer a
  minute; a false negative sends unreviewed medical advice. Module 5 measures
  both rates on labelled messages.
- Review items are row-locked when decided, so two reviewers cannot both act
  on one; a second decision is a 409.
- Rule-flagged drafts are made with read-only tools, so a draft can never
  have booked or cancelled anything.
