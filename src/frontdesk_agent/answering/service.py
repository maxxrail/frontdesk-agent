"""Answer a question from retrieved policy chunks, or refuse.

The flow, and where each guarantee comes from:

1. Retrieve the tenant's top chunks. Nothing relevant at all: refuse without
   calling the model (saves money; ``no_relevant_documents``).
2. Ask the model to answer from those chunks only, citing them by label.
3. Validate in code, not by trusting the model: invalid JSON gets one repair
   attempt, then a 502. An answer whose citations are missing or point at
   chunks it was never given is refused (``ungrounded_answer``). A model that
   says the excerpts do not cover it is refused (``not_in_documents``).
4. Record the call in ``llm_calls`` with tokens, latency and cost.

A refusal is a normal 200 with ``needs_human: true``. Module 4 routes those to
the review queue instead of texting the client.
"""

import json
import logging
import re
import uuid
from typing import TypedDict

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.answering.schemas import Citation, ModelAnswer, PolicyAnswer, RefusalReason
from frontdesk_agent.db.repositories import LLMCallRepository, TenantDirectory
from frontdesk_agent.llm.base import LLMClient, LLMCompletion, LLMInvalidOutputError
from frontdesk_agent.llm.pricing import cost_usd
from frontdesk_agent.llm.retry import RetryPolicy, with_retries
from frontdesk_agent.prompts import load_prompt
from frontdesk_agent.retrieval.embeddings import Embedder
from frontdesk_agent.retrieval.store import PolicyStore, SearchHit, SearchMode

logger = logging.getLogger(__name__)

PROMPT_NAME = "policy_answer"
PROMPT_VERSION = "v1"

REFUSAL_TEXT = (
    "I don't have that information in our policies. A team member will follow up with you."
)


class _Usage(TypedDict, total=False):
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse(text: str) -> ModelAnswer:
    try:
        return ModelAnswer.model_validate(json.loads(_FENCE.sub("", text).strip()))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMInvalidOutputError(f"policy answer did not match the schema: {exc}") from exc


def format_excerpts(hits: list[SearchHit]) -> tuple[str, dict[str, SearchHit]]:
    """Label hits C1..Cn. Short labels, because models copy UUIDs unreliably."""
    labels = {f"C{i}": hit for i, hit in enumerate(hits, start=1)}
    text = "\n\n".join(f"[{label}] {hit.heading}\n{hit.content}" for label, hit in labels.items())
    return text, labels


class PolicyAnswerer:
    def __init__(
        self,
        client: LLMClient,
        embedder: Embedder,
        *,
        mode: SearchMode = "vector",
        k: int = 5,
        min_similarity: float = 0.12,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._mode = mode
        self._k = k
        self._min_similarity = min_similarity
        self._retry_policy = retry_policy or RetryPolicy()

    async def answer(
        self, session: AsyncSession, tenant_id: uuid.UUID, question: str
    ) -> PolicyAnswer:
        tenant = await TenantDirectory(session).get(tenant_id)
        hits = await PolicyStore(session, tenant_id, self._embedder).search(
            question, k=self._k, mode=self._mode
        )
        sims = [h.vector_similarity for h in hits if h.vector_similarity is not None]
        top = max(sims, default=None)

        if not hits or (top is not None and top < self._min_similarity):
            return self._refusal("no_relevant_documents", top)

        excerpts, labels = format_excerpts(hits)
        system, user = load_prompt(PROMPT_NAME, PROMPT_VERSION).render(
            business_name=tenant.name,
            business_type=tenant.business_type,
            excerpts=excerpts,
            question=question.strip(),
        )

        completion = await self._call(system, user)
        try:
            parsed = _parse(completion.text)
        except LLMInvalidOutputError as first_failure:
            logger.warning("invalid policy answer, attempting repair")
            await self._record(session, tenant_id, completion, succeeded=False)
            repair = (
                f"{user}\n\nYour previous reply was rejected because it was not valid JSON "
                "matching the required shape. Reply with the JSON object only."
            )
            completion = await self._call(system, repair)
            try:
                parsed = _parse(completion.text)
            except LLMInvalidOutputError as second_failure:
                await self._record(session, tenant_id, completion, succeeded=False)
                await session.commit()
                raise second_failure from first_failure

        await self._record(session, tenant_id, completion, succeeded=True)
        await session.commit()

        if not parsed.answerable:
            return self._refusal("not_in_documents", top, completion)

        cited = [labels[c] for c in dict.fromkeys(parsed.citations) if c in labels]
        invented = [c for c in parsed.citations if c not in labels]
        if invented:
            logger.warning("model cited excerpts it was not given", extra={"labels": invented})
        if not cited or not parsed.answer.strip():
            # Claims to answer but cannot point at a source: treat as invented.
            return self._refusal("ungrounded_answer", top, completion)

        return PolicyAnswer(
            answerable=True,
            answer=parsed.answer.strip(),
            needs_human=False,
            citations=[
                Citation(
                    chunk_id=h.chunk_id,
                    document_slug=h.document_slug,
                    heading=h.heading,
                    excerpt=h.content[:240],
                )
                for h in cited
            ],
            retrieval_mode=self._mode,
            top_similarity=top,
            **self._usage(completion),
        )

    def _refusal(
        self,
        reason: RefusalReason,
        top: float | None,
        completion: LLMCompletion | None = None,
    ) -> PolicyAnswer:
        return PolicyAnswer(
            answerable=False,
            answer=REFUSAL_TEXT,
            needs_human=True,
            refusal_reason=reason,
            citations=[],
            retrieval_mode=self._mode,
            top_similarity=top,
            **self._usage(completion),
        )

    @staticmethod
    def _usage(c: LLMCompletion | None) -> _Usage:
        if c is None:
            return {}
        return {
            "model": c.model,
            "prompt_version": c.prompt_version,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "latency_ms": c.latency_ms,
        }

    async def _record(
        self, session: AsyncSession, tenant_id: uuid.UUID, c: LLMCompletion, *, succeeded: bool
    ) -> None:
        await LLMCallRepository(session, tenant_id).record(
            c,
            cost_usd=cost_usd(c.model, c.input_tokens, c.output_tokens),
            succeeded=succeeded,
        )

    async def _call(self, system: str, user: str) -> LLMCompletion:
        async def operation() -> LLMCompletion:
            return await self._client.complete(
                system=system,
                user=user,
                prompt_name=PROMPT_NAME,
                prompt_version=PROMPT_VERSION,
                max_tokens=400,
            )

        return await with_retries(operation, self._retry_policy)
