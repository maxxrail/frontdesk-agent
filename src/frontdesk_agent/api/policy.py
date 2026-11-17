"""Policy endpoints: answer a question, inspect retrieval, add a document."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from frontdesk_agent.answering.schemas import (
    IngestOut,
    PolicyAnswer,
    PolicyDocumentIn,
    PolicyQuestion,
    SearchResult,
)
from frontdesk_agent.answering.service import PolicyAnswerer
from frontdesk_agent.api.auth import Admin, Reviewer, Session
from frontdesk_agent.config import get_settings
from frontdesk_agent.dependencies import get_answerer
from frontdesk_agent.llm.base import LLMError, LLMInvalidOutputError
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.store import PolicyStore

router = APIRouter(prefix="/v1/policy", tags=["policy"])


@router.post(
    "/answer",
    response_model=PolicyAnswer,
    responses={502: {"description": "The model failed or returned unusable output."}},
)
async def answer_question(
    who: Reviewer,
    body: PolicyQuestion,
    session: Session,
    answerer: Annotated[PolicyAnswerer, Depends(get_answerer)],
) -> PolicyAnswer:
    """Answer from this tenant's documents only, with citations.

    When the documents do not cover the question the reply is still a 200,
    with ``answerable: false`` and ``needs_human: true``: refusing is the
    correct outcome, not an error.
    """
    try:
        return await answerer.answer(session, who.tenant_id, body.question)
    except LLMInvalidOutputError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The model did not return a usable answer."
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The model provider is unavailable."
        ) from exc


@router.post("/search", response_model=list[SearchResult])
async def search_policies(
    who: Reviewer, body: PolicyQuestion, session: Session
) -> list[SearchResult]:
    """Retrieval only, no model call: what the answer endpoint would see."""
    settings = get_settings()
    hits = await PolicyStore(session, who.tenant_id, default_embedder()).search(
        body.question, k=settings.retrieval_k, mode=settings.retrieval_mode
    )
    return [
        SearchResult(
            chunk_id=h.chunk_id,
            document_slug=h.document_slug,
            heading=h.heading,
            content=h.content,
            score=h.score,
            vector_similarity=h.vector_similarity,
        )
        for h in hits
    ]


@router.put("/documents", response_model=IngestOut)
async def put_document(who: Admin, body: PolicyDocumentIn, session: Session) -> IngestOut:
    """Create or replace one document. Re-sending unchanged text is a no-op."""
    result = await PolicyStore(session, who.tenant_id, default_embedder()).ingest(
        slug=body.slug, markdown=body.markdown
    )
    await session.commit()
    return IngestOut(slug=result.slug, chunks=result.chunks, skipped=result.skipped)
