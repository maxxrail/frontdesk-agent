# 10. A hand-written agent loop with server-scoped tools

- Status: accepted
- Date: 2026-11-14

## Context

The agent books, lists and cancels appointments and answers policy questions
over SMS. Agent frameworks would provide the loop, but hide the parts that
matter most here: when it stops, what the model is allowed to touch, and
what is recorded.

## Decision

`agent/loop.py` is about 100 lines: call the model, run any tool calls,
append the results, repeat. It stops at a reply with no tool calls, or after
**6 model calls**. Hitting the limit sends nothing; it creates a review item.

The model interface gained one method, `converse`, over provider-neutral
message types (`llm/messages.py`) that serialise to JSON, so a conversation
is stored in Postgres and resumes days later. The Anthropic adapter is
tested against the real SDK through a mocked HTTP transport, which caught
that SDK 1.x is built on `httpx2`, not `httpx`.

Seven tools, each with a strict Pydantic input model (unknown keys rejected)
whose JSON Schema is what the model sees:

| Tool | Writes | Notes |
| --- | --- | --- |
| `check_availability` | no | schedule minus bookings, 1-hour lead time, 60 days ahead |
| `book_appointment` | yes | only an exact open slot; idempotent per inbound message |
| `list_my_appointments` | no | |
| `cancel_appointment` | yes | inside 24 hours: held for review, not executed |
| `search_policy` | no | Module 3 retrieval |
| `save_client_name` | yes | only for a texter with no name on file |
| `escalate_to_human` | yes | money, medical, complaint, other |

The spec named four tools; the last three exist because a real thread needs
them (which appointment to cancel, who a new texter is, a way out).

**No tool takes a client or tenant argument.** Both come from the
conversation. The model cannot book for someone else, and asking about
another client's appointment ID gets the same error as a missing one.

Tool errors (bad input, a taken slot, an unknown tool) return an error
result the model can read and fix; they never raise into the loop.

## Consequences

- `test_step_limit_stops_a_model_that_loops_forever` scripts a model that
  calls a tool forever and asserts exactly 6 calls and nothing sent.
- Every model call is a row in `llm_calls` with tokens, latency and cost.
- History is trimmed to the last 30 messages, always starting at client text,
  so a cut never orphans a tool result (which the provider would reject).
- No framework to upgrade, and also no framework features: parallel tool
  execution, streaming and memory summarisation would be written here when
  needed.
