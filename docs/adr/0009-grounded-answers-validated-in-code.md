# 9. Grounded answers, validated in code

- Status: accepted
- Date: 2026-11-14

## Context

The worst failure for a front desk is a confident, invented policy: a fee
that does not exist, or the other clinic's cancellation window. Prompting the
model to "only use the context" reduces this but does not guarantee it, and
ADR 0008 showed retrieval similarity cannot detect unanswerable questions.

## Decision

`POST /v1/tenants/{id}/policy/answer` works like this:

1. Retrieve the tenant's top 5 chunks. If there are none, or the best
   similarity is below 0.12, refuse without calling the model.
2. Label the chunks C1 to C5 in the prompt. Models copy short labels
   reliably and UUIDs unreliably.
3. The model returns strict JSON: `answerable`, `answer`, `citations`.
   Invalid JSON gets one repair attempt, then a 502, as in ADR 0004.
4. Code decides the outcome, not the model:
   - `answerable: false` is a refusal, `not_in_documents`.
   - An answer with no citations, only citations to labels it was never
     given, or empty text is a refusal, `ungrounded_answer`.
   - Otherwise the answer is returned with each citation resolved to its
     chunk ID, document and heading.
5. Every model call, including failed ones, is written to `llm_calls` with
   tokens, latency and cost from `llm/pricing.py`.

A refusal is a 200 with `needs_human: true` and a fixed message. Refusing is
the correct result, not an error, and Module 4 routes it to the review queue.

## Consequences

- An answer can never reach a client without pointing at text this tenant
  wrote. It can still misread that text; that is what the Module 5 eval
  suite and the Module 4 review queue are for.
- The client's question sits in `<question>` tags and the prompt says to
  treat it as data. That is a first layer, not a defence; Module 6 adds the
  injection tests and threat model.
- Unit tests script every model behaviour with the fake client.
  `evals/policy_answers.py` measures the real model's behaviour on all 60
  golden questions; a dry run estimates a full Haiku 4.5 run at about $0.05.
