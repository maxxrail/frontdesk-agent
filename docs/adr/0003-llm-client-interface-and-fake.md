# 3. A narrow LLM interface with a fake for tests

- Status: accepted
- Date: 2026-09-18

## Context

Calling a provider SDK directly from application code creates three problems.
Tests either cost money and flake, or they mock the SDK's internals and break
on every SDK upgrade. Swapping providers touches every call site. And provider
specific exceptions leak into business logic.

## Decision

The application depends on an `LLMClient` protocol with a single `complete`
method returning an `LLMCompletion`. Two implementations exist:
`AnthropicLLMClient` for production and `FakeLLMClient` for tests.

No test in this repository calls a real model. The fake is scripted with a
queue of replies and exceptions, which is how the retry and repair paths are
exercised deterministically.

Provider errors are translated at the boundary into `LLMTimeoutError`,
`LLMRateLimitError`, `LLMServerError` and `LLMError`, so the retry policy can
decide what is worth retrying without knowing the vendor.

`LLMCompletion` carries model, token counts, latency and prompt version from
the start, because Module 2 persists them and Module 7 charts them.

## Consequences

- The test suite runs in under a second and costs nothing, in CI included.
- Changing provider is one new class plus one line of wiring.
- The interface deliberately omits streaming and tool use. Module 4 will widen
  it for tools, and that will be a new ADR.
- The installed SDK no longer exposes `temperature` on `messages.create`, so
  the interface does not carry it. Determinism comes from a constrained prompt
  and strict schema validation instead.
