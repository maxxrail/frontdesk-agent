# 4. Versioned prompt files and strict structured output

- Status: accepted
- Date: 2026-09-18

## Context

A prompt is a behavioural dependency. Changing one changes what the service
does, often subtly, and a prompt buried in a Python string is invisible in code
review and impossible to attribute after the fact.

Free-text model output has a related problem: the caller cannot tell a good
reply from a plausible-looking bad one.

## Decision

Prompts live in `prompts/<name>/<version>.md`, loaded and rendered by
`frontdesk_agent.prompts`. A published version is never edited in place; a
change means a new version file and a caller switch. Every model call records
the prompt name and version.

Rendering fails if any placeholder is unfilled, because a prompt containing a
literal `{{client_first_name}}` is worse than an error.

The model must return JSON matching `ReminderDraft`, which forbids extra keys
and caps the message at 300 characters. Invalid output gets exactly one repair
attempt, with the rejection reason appended, then the request fails with 502.

## Consequences

- Any logged output can be traced to the exact prompt text that produced it.
- Prompt changes appear as reviewable diffs, and from Module 5 they trigger evals.
- One repair attempt, not a loop: two consecutive failures mean the prompt or
  the model is wrong, and further retries only spend money.
- Forbidding extra keys means a model inventing a `price` field fails loudly.
  That is deliberate given what this service is not allowed to say.
