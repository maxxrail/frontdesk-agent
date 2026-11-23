"""The human review queue, and the conversation log a reviewer needs.

Approve, edit-and-approve (approve with your own ``reply``), or reject. The
tenant and the reviewer come from the caller's credential (ADR 0016): the
audit trail records who decided, not a name typed into the request.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from frontdesk_agent.api.auth import Principal, Reviewer, Session
from frontdesk_agent.db.models import (
    MessageDirection,
    MessageStatus,
    ReviewKind,
    ReviewStatus,
)
from frontdesk_agent.db.repositories import (
    ConversationRepository,
    NotFoundError,
    ReviewRepository,
    SmsMessageRepository,
)
from frontdesk_agent.dependencies import get_services
from frontdesk_agent.frontdesk.ports import Effects
from frontdesk_agent.frontdesk.service import ReviewAlreadyDecidedError, ReviewNeedsReplyError
from frontdesk_agent.services import Services

router = APIRouter(prefix="/v1", tags=["review"])


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    kind: ReviewKind
    status: ReviewStatus
    reason: str
    client_message: str
    proposed_reply: str
    action: dict[str, Any] | None
    final_reply: str | None
    decided_by: str | None
    decided_at: datetime | None
    created_at: datetime


class Decision(BaseModel):
    reply: str | None = Field(
        default=None,
        max_length=1600,
        description="Approve: replaces the drafted reply. Reject: optional message to send.",
    )


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    direction: MessageDirection
    status: MessageStatus
    body: str
    created_at: datetime


@router.get("/reviews", response_model=list[ReviewOut])
async def list_reviews(
    who: Reviewer,
    session: Session,
    status_filter: Annotated[ReviewStatus | None, Query(alias="status")] = ReviewStatus.PENDING,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ReviewOut]:
    items = await ReviewRepository(session, who.tenant_id).by_status(status_filter, limit=limit)
    return [ReviewOut.model_validate(i) for i in items]


@router.get("/reviews/{review_id}", response_model=ReviewOut)
async def get_review(who: Reviewer, review_id: uuid.UUID, session: Session) -> ReviewOut:
    try:
        return ReviewOut.model_validate(
            await ReviewRepository(session, who.tenant_id).get(review_id)
        )
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review item not found.") from exc


async def _decide(
    services: Services,
    session: Session,
    who: Principal,
    review_id: uuid.UUID,
    body: Decision,
    *,
    approve: bool,
) -> ReviewOut:
    effects = Effects()
    try:
        item = await services.frontdesk.decide(
            session,
            who.tenant_id,
            review_id,
            effects,
            approve=approve,
            reply=body.reply,
            reviewer=who.actor,
        )
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review item not found.") from exc
    except ReviewAlreadyDecidedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ReviewNeedsReplyError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    await effects.commit(session)
    out = ReviewOut.model_validate(item)
    await effects.flush(session, services.sender, services.queue)
    return out


@router.post("/reviews/{review_id}/approve", response_model=ReviewOut)
async def approve(
    who: Reviewer,
    review_id: uuid.UUID,
    body: Decision,
    session: Session,
    services: Annotated[Services, Depends(get_services)],
) -> ReviewOut:
    """Run the held action and send the reply (``reply`` overrides the draft)."""
    return await _decide(services, session, who, review_id, body, approve=True)


@router.post("/reviews/{review_id}/reject", response_model=ReviewOut)
async def reject(
    who: Reviewer,
    review_id: uuid.UUID,
    body: Decision,
    session: Session,
    services: Annotated[Services, Depends(get_services)],
) -> ReviewOut:
    """Do not run the held action. Sends ``reply`` only if one is given."""
    return await _decide(services, session, who, review_id, body, approve=False)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def conversation_messages(
    who: Reviewer, conversation_id: uuid.UUID, session: Session
) -> list[MessageOut]:
    try:
        await ConversationRepository(session, who.tenant_id).get(conversation_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.") from exc
    msgs = await SmsMessageRepository(session, who.tenant_id).in_conversation(conversation_id)
    return [MessageOut.model_validate(m) for m in msgs]
