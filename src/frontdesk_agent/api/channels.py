"""The Twilio inbound SMS webhook.

    Twilio -> POST /v1/channels/sms (form-encoded, signed)

The request handler does the minimum and returns fast, because Twilio waits
at most 15 seconds and retries on failure: verify the signature, route to the
tenant by the number texted, store the message (deduplicated by MessageSid),
commit, enqueue the agent turn, and answer with empty TwiML. The reply goes
out later through the REST API, from the worker.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from frontdesk_agent.api.tenants import Session
from frontdesk_agent.channels.twilio import signature_is_valid
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.repositories import TenantDirectory
from frontdesk_agent.dependencies import get_services
from frontdesk_agent.frontdesk.service import accept_inbound
from frontdesk_agent.services import Services

router = APIRouter(prefix="/v1/channels", tags=["channels"])

EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


@router.post(
    "/sms",
    responses={
        403: {"description": "Missing or invalid X-Twilio-Signature."},
        404: {"description": "The number texted belongs to no tenant."},
        503: {"description": "No Twilio auth token is configured."},
    },
)
async def inbound_sms(
    request: Request,
    session: Session,
    services: Annotated[Services, Depends(get_services)],
) -> Response:
    settings = get_settings()
    if not settings.twilio_auth_token:
        # Refuse rather than accept unsigned messages from anyone.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SMS channel not configured.")

    form = {k: str(v) for k, v in (await request.form()).items()}
    url = settings.public_base_url.rstrip("/") + request.url.path
    if request.url.query:
        url += f"?{request.url.query}"
    if not signature_is_valid(
        auth_token=settings.twilio_auth_token,
        url=url,
        params=form,
        signature=request.headers.get("X-Twilio-Signature", ""),
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid signature.")

    to, from_, body, sid = (form.get(k, "") for k in ("To", "From", "Body", "MessageSid"))
    if not (to and from_ and sid):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Missing To, From or MessageSid.")
    tenant = await TenantDirectory(session).by_sms_number(to)
    if tenant is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown destination number.")

    msg = await accept_inbound(session, tenant, from_=from_, to=to, body=body, sid=sid)
    await session.commit()
    if msg is not None:  # None: Twilio retried a message we already have
        await services.queue.enqueue(
            "handle_inbound", tenant_id=str(tenant.id), message_id=str(msg.id)
        )
    return Response(EMPTY_TWIML, media_type="application/xml")
