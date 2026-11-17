# 15. The agent stays on Claude Haiku 4.5 for now

- Status: accepted (Maxx, 2026-11-16)
- Date: 2026-11-16

## Context

The roadmap asks for a measured comparison of `claude-haiku-4-5` (the agent
model since Module 1) with `claude-haiku-5-5`, which costs a tenth as much
per token ($0.10 / $0.50 per million input / output, against $1 / $5).

Both ran the full eval suite (132 cases, prompt `agent/v2`, same code,
commit df48dab and its PR merge), after two harness fixes the first Haiku 5.5
run exposed (below). Thinking is off for both (`llm/thinking.py`): Haiku 4.5
does not think unless asked; Haiku 5.5 gets `{"type": "disabled"}`.

| | Haiku 4.5 | Haiku 5.5 |
|---|---|---|
| Pass rate | 78.0% (other runs: 79.5-82.6%) | **78.8%** |
| booking / cancel / policy | 55% / 92% / 78% | 64% / 92% / 83% |
| red team | 78% | 82% |
| risk | **89%** | 67% |
| Replies over 320 characters | 7 | **17** |
| Agent cost per case | $0.0048 | **$0.0006** (8x less) |
| Model call p50 | 1.4 s (rate-limited run; others 0.87-0.99 s) | 0.81 s |
| Simulations reaching their goal | 24/30 | 23/30 |

Run-to-run noise on one model is about 3 points overall and 8-13 cases
(`evals/README.md`), so the pass rates are a tie.

## What decided it

1. **Length.** Haiku 5.5 writes about 1.5x the output tokens per call (96
   against 65). Every one of its 6 risk failures is a reply over the SMS
   limit, mostly safety messages with an extra paragraph.
2. **Reasoning in the reply.** One draft for a reviewer began "This is a
   money dispute, so I should not handle it directly. Per the rules, I
   escalate, but escalate_to_human is not available here... Reply to send:".
   Anthropic's guidance warns that newer models with thinking disabled can
   write their reasoning into the visible answer. A reviewer could approve
   that draft as is.

Cost does not outweigh these yet: the agent's model spend is about $0.005 a
message on Haiku 4.5.

## Decision

Keep `claude-haiku-4-5` as `FDA_LLM_MODEL` for v0.5.

Next experiment (not in this release): Haiku 5.5 with adaptive thinking on
at `low` effort, a larger `max_tokens` so thinking cannot cut replies off,
and an explicit "reply with the message only" instruction, compared on the
same suite. If it holds the pass rate and the SMS limit, switch: the saving
is about 87% of agent model cost.

## Found on the way

The first Haiku 5.5 run (74.2%) was unfair to it, and fixing that also
helps Haiku 4.5:

- A held message's draft runs with read-only tools, so `escalate_to_human`
  is not offered, but the prompt says to call it. Haiku 5.5 called the
  missing tool in 14 cases (12 failed only for that) and drafted "I wasn't
  able to pass your message to the team", which is false. Read-only runs now
  get a note that the team already has the message (`agent/loop.py`).
- The medical-advice check matched "whether you can take Advil", a reply
  that defers the question.
