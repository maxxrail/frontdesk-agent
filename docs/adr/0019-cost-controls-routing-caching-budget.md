# 19. Cost controls: lookups on a cheap model, prompt caching, a daily budget

- Status: accepted (Maxx, 2026-11-23)
- Date: 2026-11-23

## Context

The roadmap asks for three cost controls: prompt caching for the long
system prompt, a small model for the easy work and a stronger one for the
reply, and a per-tenant daily budget that degrades to a canned response.

Two facts from the last eval run on main (run 44632195740, prompt
`agent/v2`, Haiku 4.5, 260 agent calls) shaped all three:

- **Input is 87% of agent cost**, and an agent call averages **2,166 input
  tokens** (largest per-case average 4,082).
- **Haiku 4.5 caches nothing under 4,096 tokens**, the documented minimum
  for that model. The cacheable prefix (tools plus system prompt) is about
  1,800 tokens. So prompt caching on the agent's model saves exactly $0.
  Haiku 5.5 caches from 512 tokens and costs a tenth as much, but ADR 0015
  kept Haiku 4.5 because it scored 89% on risk cases against 67%.

## Decision

**Route by step.** With `FDA_LLM_LOOKUP_MODEL` (default `claude-haiku-5-5`)
the cheap model drafts every agent step. A draft that only calls read-only
tools (check availability, list appointments, search policy) runs as is.
Anything else, a reply, a booking, a cancellation or an escalation, is
thrown away. The main model (Haiku 4.5) then takes that step and the rest
of the turn. Every decision and every text a client or a reviewer reads
still comes from the model ADR 0015 chose. Discarded drafts are recorded
and costed like any other call, so the saving is measured net of them.

Routing on the *kind* of step, not on a classifier's guess about the
message, keeps the guarantee simple to state and test: the cheap model can
look, never act. Two tests script a cheap model that tries to book, or to
look up and escalate in the same step, and check that nothing it decided
ran.

**Cache what is stable.** The prompt is split into `# System` (rules, the
clinic's name) and a new `# Context` (who is texting, the current time), and
the adapter puts one cache breakpoint after the system prompt. Tools render
before it, so the breakpoint covers both. The time used to sit at the top of
the prompt, which would have made every call a cache miss. On Haiku 4.5 the
breakpoint is a no-op at no charge. On the lookup model it caches.
`llm_calls` now records cache reads and writes, priced at 0.1x and 1.25x the
input rate (verified on the pricing page on 2026-11-23).

**A daily budget per clinic.** Before the agent runs, the clinic's model
spend since its local midnight is compared with `tenants.daily_budget_usd`,
or `FDA_TENANT_DAILY_BUDGET_USD` ($5). Over budget, no model is called. The
client gets the fixed holding text (the medical one, with the 911 line,
because the unread message could be urgent), and the conversation goes to
staff as a `budget` review item and pauses, so a burst of texts makes one
item. It is its own review kind because, unlike a risk hold, it sends a text
at once. Spend is counted in database time, like the rate limits (ADR 0018).

## Numbers that decided it

Two full eval runs on PR #23, same code and prompt (`agent/v3`), Haiku 4.5
for every reply:

| | Routing off (run 45277481734) | Routing on (run 45276177182) |
|---|---|---|
| Agent cost per case | $0.0053 | **$0.0044** (-17%) |
| Pass rate, 137 cases | 76.6% | **86.9%** |
| Risk cases, 18 | **17** (94.4%) | 15 (83.3%) |
| Model calls | 272 | 454 |
| Model time per message, p50 / p95 | 1.46 s / 2.55 s | 2.14 s / 3.35 s |
| Cache reads (all models) | 0 | 496K tokens |

The saving is smaller than the 25-30% estimated beforehand. The steps that
move to the cheap model are the cheap ones: lookups come early in a turn,
with a short history. The expensive calls are the replies, which carry the
tool results and stay on Haiku 4.5, where caching cannot help.

Maxx had asked to fall back if risk cases dropped. They dropped by two.
Both extra failures were drafts written for staff on medical holds, one
over 320 characters and one that tripped the advice check while asking
whether a painkiller was okay. Every client got the fixed safety text.
Haiku 4.5 wrote those drafts in both setups, and run-to-run noise on this
suite is 8-13 cases. Maxx chose to keep routing on, and the over-long staff
drafts are issue #24.

## Consequences

- About 0.7 s more model time per message, well inside the 8 s latency SLO
  (`docs/slo.md`).
- `frontdesk_agent_routes_total{route="lookup"|"handoff"}` shows how often
  the cheap model's draft is kept.
- If the main model later moves to one with a low cache minimum, the
  breakpoint starts saving with no code change. If the risk score on the
  routed setup does not recover (#24), set `FDA_LLM_LOOKUP_MODEL=` to turn
  routing off. The eval workflow takes `lookup_model: off` to compare.
- The budget can only be checked between messages, so a clinic can go over
  by at most one turn (six steps).
