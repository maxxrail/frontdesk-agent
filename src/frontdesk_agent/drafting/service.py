"""Turning appointment details into a validated SMS draft.

The flow: render a versioned prompt, call the model with retries, parse the
reply as strict JSON, and if that fails, give the model exactly one chance to
repair its own output before giving up.
"""

import json
import logging
import re

from pydantic import ValidationError

from frontdesk_agent.drafting.schemas import ReminderDraft, ReminderRequest, ReminderResponse
from frontdesk_agent.llm.base import LLMClient, LLMCompletion, LLMInvalidOutputError
from frontdesk_agent.llm.retry import RetryPolicy, with_retries
from frontdesk_agent.prompts import load_prompt

logger = logging.getLogger(__name__)

PROMPT_NAME = "reminder_draft"
PROMPT_VERSION = "v1"

# Models sometimes wrap JSON in ``` fences despite instructions not to.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_draft(text: str) -> ReminderDraft:
    """Parse a model reply into a ReminderDraft, or raise LLMInvalidOutputError."""
    cleaned = _FENCE.sub("", text).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMInvalidOutputError(f"reply was not valid JSON: {exc}") from exc

    try:
        return ReminderDraft.model_validate(payload)
    except ValidationError as exc:
        raise LLMInvalidOutputError(f"reply did not match the schema: {exc}") from exc


class ReminderDrafter:
    """Drafts reminder messages using whatever LLMClient it is given."""

    def __init__(self, client: LLMClient, retry_policy: RetryPolicy | None = None) -> None:
        self._client = client
        self._retry_policy = retry_policy or RetryPolicy()

    async def draft(self, request: ReminderRequest) -> ReminderResponse:
        prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
        system, user = prompt.render(
            business_name=request.business_name,
            business_type=request.business_type,
            client_first_name=request.client_first_name,
            appointment_human=request.appointment_human(),
            provider_name=request.provider_name,
            location=request.location,
        )

        completion = await self._call(system, user)
        repaired = False

        try:
            draft = _parse_draft(completion.text)
        except LLMInvalidOutputError as first_failure:
            # One repair attempt. Two failures in a row means something is
            # wrong with the prompt or the model, and retrying will not fix it.
            logger.warning(
                "invalid model output, attempting repair",
                extra={"prompt": PROMPT_NAME, "version": PROMPT_VERSION},
            )
            repair_user = (
                f"{user}\n\n"
                "Your previous reply was rejected because it was not valid JSON "
                "matching the required shape. Reply with the JSON object only."
            )
            completion = await self._call(system, repair_user)
            try:
                draft = _parse_draft(completion.text)
            except LLMInvalidOutputError as second_failure:
                raise second_failure from first_failure
            repaired = True

        return ReminderResponse(
            draft=draft,
            model=completion.model,
            prompt_name=completion.prompt_name,
            prompt_version=completion.prompt_version,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
            repaired=repaired,
        )

    async def _call(self, system: str, user: str) -> LLMCompletion:
        async def operation() -> LLMCompletion:
            return await self._client.complete(
                system=system,
                user=user,
                prompt_name=PROMPT_NAME,
                prompt_version=PROMPT_VERSION,
                max_tokens=512,
            )

        return await with_retries(operation, self._retry_policy)
