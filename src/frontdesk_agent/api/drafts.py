"""Drafting endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from frontdesk_agent.dependencies import get_drafter
from frontdesk_agent.drafting.schemas import ReminderRequest, ReminderResponse
from frontdesk_agent.drafting.service import ReminderDrafter
from frontdesk_agent.llm.base import LLMError, LLMInvalidOutputError

router = APIRouter(prefix="/v1/drafts", tags=["drafts"])


@router.post(
    "/reminder",
    response_model=ReminderResponse,
    summary="Draft an appointment reminder SMS",
    responses={
        502: {"description": "The model failed or returned unusable output."},
    },
)
async def draft_reminder(
    request: ReminderRequest,
    drafter: Annotated[ReminderDrafter, Depends(get_drafter)],
) -> ReminderResponse:
    """Draft one reminder message.

    A model failure is a 502, not a 500: the fault is upstream, and the
    distinction matters when reading dashboards in Module 7.
    """
    try:
        return await drafter.draft(request)
    except LLMInvalidOutputError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model did not return a usable draft.",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model provider is unavailable.",
        ) from exc
