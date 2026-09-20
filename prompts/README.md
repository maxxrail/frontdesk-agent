# Prompts

Prompts are versioned files, not string literals in the code. A prompt is a
behavioural dependency: changing one changes what the service does, so it gets
the same treatment as code.

- One directory per prompt, one file per version: `reminder_draft/v1.md`.
- Never edit a published version in place. Add `v2.md` and switch the caller.
- Every model call records the prompt name and version, so any output in the
  logs can be traced to the exact text that produced it.
- From Module 5 on, changing anything here triggers the eval suite in CI.

Placeholders use `{{name}}` and are substituted by `frontdesk_agent.prompts`.
Rendering fails loudly if a placeholder has no value, because a prompt with a
literal `{{client_first_name}}` in it is worse than an error.
