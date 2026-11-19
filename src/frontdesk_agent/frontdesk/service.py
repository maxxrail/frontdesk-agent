"""Handling one inbound SMS, and resuming after a human decision.

Order of checks for every inbound message (ADR 0011):

1. Already handled? Do nothing (worker retries are safe).
2. STOP / START keywords: update consent, no reply.
3. An open waitlist offer and a YES or NO: the slot-fill workflow answers.
4. Conversation paused for review: hold the message until a decision.
   A message over the length limit goes to review without a model call.
5. Risk policy flags money or medical: the agent drafts a reply with
   read-only tools, a review item holds it, nothing is sent (medical gets
   a fixed safety message).
6. Otherwise the agent runs with all tools. A held action (late
   cancellation, escalation) or the step limit turns into a review item
   instead of a sent reply.
7. The output guardrail checks the reply last: another client's details,
   card data or clinical advice turn it into a review item, not a text.

The risk rules and the model see the message after ``clean_untrusted``
(guardrails.py); the stored message keeps the raw text for staff.

Texts and jobs are collected in Effects and only happen after commit.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.agent.loop import PROMPT_VERSION, AgentRun, run_agent, trim_history
from frontdesk_agent.agent.risk import MEDICAL_HOLDING_MESSAGE, classify
from frontdesk_agent.agent.tools import HeldAction, ToolContext
from frontdesk_agent.db.models import (
    Appointment,
    Client,
    Conversation,
    ConversationStatus,
    MessageStatus,
    ReviewItem,
    ReviewKind,
    ReviewStatus,
    SmsMessage,
    Tenant,
)
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    AuditRepository,
    ClientRepository,
    ConversationRepository,
    NotFoundError,
    ReviewRepository,
    SlotOfferRepository,
    SmsMessageRepository,
    TenantDirectory,
)
from frontdesk_agent.frontdesk.ports import Clock, Effects
from frontdesk_agent.frontdesk.slotfill import SlotFiller, parse_reply
from frontdesk_agent.guardrails import (
    MAX_INBOUND_CHARS,
    Finding,
    check_reply,
    clean_untrusted,
    inbound_too_long,
)
from frontdesk_agent.llm.base import LLMClient
from frontdesk_agent.llm.messages import Message, TextBlock
from frontdesk_agent.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

# Mirrors the carrier-level keywords Twilio acts on for long codes. Note that
# a message that is exactly "CANCEL" is an opt-out to Twilio, not a request to
# cancel an appointment; the agent never sees it. See the README.
STOP_WORDS = {"stop", "stopall", "unsubscribe", "cancel", "end", "quit", "optout"}
START_WORDS = {"start", "unstop", "yes start"}

Outcome = str  # short label for logs and tests: "agent_reply", "held_for_review", ...


@dataclass
class FrontDesk:
    llm: LLMClient
    embedder: Embedder
    clock: Clock
    slot_filler: SlotFiller
    # Which prompts/agent/<version>.md to run: a setting, so evals can compare versions.
    prompt_version: str = PROMPT_VERSION

    # -- inbound -----------------------------------------------------------

    async def handle_inbound(
        self, session: AsyncSession, tenant_id: uuid.UUID, message_id: uuid.UUID, effects: Effects
    ) -> Outcome:
        now = self.clock.now()
        tenant = await TenantDirectory(session).get(tenant_id)
        msg = await SmsMessageRepository(session, tenant.id).get(message_id)
        convo = await ConversationRepository(session, tenant.id).for_client(
            (await self._conversation_client(session, tenant, msg)).id
        )
        await session.refresh(msg)  # re-read under the conversation lock
        if msg.status != MessageStatus.RECEIVED:
            return "already_handled"
        client = await ClientRepository(session, tenant.id).get(convo.client_id)
        word = msg.body.strip().lower().rstrip(".!")

        if word in STOP_WORDS:
            client.sms_opt_in = False
            msg.status = MessageStatus.HANDLED
            return "opted_out"
        if word in START_WORDS:
            client.sms_opt_in = True
            msg.status = MessageStatus.HANDLED
            return "opted_in"
        if not client.sms_opt_in:
            msg.status = MessageStatus.HANDLED
            return "ignored_opted_out"

        offer = await SlotOfferRepository(session, tenant.id).open_for_client(client.id, now)
        answer = parse_reply(msg.body) if offer else None
        if offer is not None and answer is not None:
            await self.slot_filler.respond(session, tenant, effects, offer, client, answer, now)
            msg.status = MessageStatus.HANDLED
            return "offer_accepted" if answer else "offer_declined"

        if convo.status == ConversationStatus.PAUSED:
            msg.status = MessageStatus.HELD
            return "held_while_paused"

        if inbound_too_long(msg.body):
            reason = (
                f"Message is {len(msg.body)} characters, over the {MAX_INBOUND_CHARS} limit;"
                " the agent did not read it."
            )
            self._hold(session, convo, msg, HeldAction(ReviewKind.ESCALATION, reason), None)
            await _audit_guardrail(session, convo, msg, "too_long", reason)
            return "held_for_review:too_long"

        body = clean_untrusted(msg.body)
        history = [Message.from_json(m) for m in convo.transcript]
        # Merge, not append: after a step-limit stop the history ends with tool
        # results (a user message), and roles must alternate.
        history = _append(history, Message.user_text(body))
        ctx = ToolContext(
            session, tenant, client, convo, self.embedder, now, msg.provider_sid or str(msg.id)
        )

        flag = classify(body)
        if flag is not None:
            run = await run_agent(
                self.llm, ctx, history, read_only=True, prompt_version=self.prompt_version
            )
            self._save(convo, history, run)
            held = HeldAction(flag.kind, flag.reason)
            held = await _flag_draft(session, convo, msg, client, held, run.reply)
            self._hold(session, convo, msg, held, run.reply)
            if flag.kind is ReviewKind.MEDICAL:
                self._text(effects, tenant, convo, client, MEDICAL_HOLDING_MESSAGE)
            return f"held_for_review:{flag.kind.value}"

        run = await run_agent(self.llm, ctx, history, prompt_version=self.prompt_version)
        self._save(convo, history, run)
        for appt in ctx.cancelled:
            self._queue_slot_fill(effects, tenant, appt, client)

        if ctx.held:
            held = await _flag_draft(session, convo, msg, client, _combine(ctx.held), run.reply)
            self._hold(session, convo, msg, held, run.reply)
            return f"held_for_review:{ctx.held[0].kind.value}"
        if run.stopped == "step_limit" or not run.reply:
            self._hold(
                session,
                convo,
                msg,
                HeldAction(
                    ReviewKind.ESCALATION, f"The agent did not finish within {run.steps} steps."
                ),
                None,
            )
            return "held_for_review:step_limit"

        finding = await check_reply(session, tenant.id, client.id, run.reply)
        if finding is not None:
            reason = f"Reply held by the output guardrail: it contained {finding.reason}."
            self._hold(session, convo, msg, HeldAction(finding.kind, reason), run.reply)
            await _audit_guardrail(session, convo, msg, finding.rule, reason)
            if finding.kind is ReviewKind.MEDICAL:
                self._text(effects, tenant, convo, client, MEDICAL_HOLDING_MESSAGE)
            return f"held_for_review:guardrail_{finding.rule}"

        self._text(effects, tenant, convo, client, run.reply)
        msg.status = MessageStatus.HANDLED
        return "agent_reply"

    # -- review decisions --------------------------------------------------

    async def decide(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        review_id: uuid.UUID,
        effects: Effects,
        *,
        approve: bool,
        reply: str | None,
        reviewer: str,
    ) -> ReviewItem:
        """Approve (optionally with an edited reply) or reject a held item.

        Approving runs the held action and sends the reply. Rejecting runs
        nothing and sends only what the reviewer wrote, if anything. The
        conversation resumes once no items are pending for it, and messages
        that arrived meanwhile are processed in order.
        """
        now = self.clock.now()
        tenant = await TenantDirectory(session).get(tenant_id)
        reviews = ReviewRepository(session, tenant.id)
        item = await reviews.lock(review_id)
        if item.status != ReviewStatus.PENDING:
            raise ReviewAlreadyDecidedError(f"review item is already {item.status.value}")
        convo = await ConversationRepository(session, tenant.id).get(item.conversation_id)
        client = await ClientRepository(session, tenant.id).get(convo.client_id)

        text = (reply if reply is not None else item.proposed_reply if approve else "").strip()
        if approve and not text:
            raise ReviewNeedsReplyError("approving needs a reply to send; none was drafted")

        note = f"[Staff note] {'Approved' if approve else 'Rejected'}: {item.reason}"
        if approve and item.action and item.action.get("type") == "cancel_appointment":
            appt = await self._approve_cancel(session, tenant, item, now)
            if appt is not None:
                self._queue_slot_fill(effects, tenant, appt, client)
            else:
                note += " The appointment was no longer active, so nothing was cancelled."

        item.status = ReviewStatus.APPROVED if approve else ReviewStatus.REJECTED
        item.final_reply = text or None
        item.decided_by = reviewer
        item.decided_at = now
        await AuditRepository(session, tenant.id).record(
            kind="review_decision",
            actor=reviewer[:120],
            subject=item.kind.value,
            outcome=item.status.value,
            conversation_id=convo.id,
            ref_id=item.id,
            detail={
                "action": (item.action or {}).get("type"),
                "reply_edited": reply is not None and reply.strip() != (item.proposed_reply or ""),
                "replied": bool(text),
            },
        )

        transcript = [Message.from_json(m) for m in convo.transcript]
        transcript = _append(transcript, Message.user_text(note))
        transcript = _append(
            transcript, Message("assistant", [TextBlock(text or "(No reply was sent.)")])
        )
        convo.transcript = [m.to_json() for m in trim_history(transcript)]
        if text:
            self._text(effects, tenant, convo, client, text)

        await session.flush()
        if not await reviews.pending_for(convo.id):
            convo.status = ConversationStatus.ACTIVE
            held = await SmsMessageRepository(session, tenant.id).in_conversation(
                convo.id, status=MessageStatus.HELD
            )
            for m in held:
                m.status = MessageStatus.RECEIVED
                effects.job("handle_inbound", tenant_id=str(tenant.id), message_id=str(m.id))
        return item

    # -- helpers -----------------------------------------------------------

    async def _conversation_client(
        self, session: AsyncSession, tenant: Tenant, msg: SmsMessage
    ) -> Client:
        convo = await ConversationRepository(session, tenant.id).get(msg.conversation_id)
        return await ClientRepository(session, tenant.id).get(convo.client_id)

    async def _approve_cancel(
        self, session: AsyncSession, tenant: Tenant, item: ReviewItem, now: datetime
    ) -> Appointment | None:
        assert item.action is not None
        repo = AppointmentRepository(session, tenant.id)
        try:
            appt = await repo.get(uuid.UUID(item.action["appointment_id"]))
        except NotFoundError:
            return None
        if appt.status != "booked" or appt.starts_at <= now:
            return None
        return await repo.cancel(appt.id)

    def _hold(
        self,
        session: AsyncSession,
        convo: Conversation,
        msg: SmsMessage,
        held: HeldAction,
        draft: str | None,
    ) -> None:
        session.add(
            ReviewItem(
                tenant_id=convo.tenant_id,
                conversation_id=convo.id,
                kind=held.kind,
                reason=held.reason[:500],
                client_message=msg.body,
                proposed_reply=(draft or "")[:1600],
                action=held.action,
            )
        )
        convo.status = ConversationStatus.PAUSED
        msg.status = MessageStatus.HANDLED

    @staticmethod
    def _save(convo: Conversation, history: list[Message], run: AgentRun) -> None:
        convo.transcript = [m.to_json() for m in trim_history(history + run.new_messages)]

    @staticmethod
    def _text(
        effects: Effects, tenant: Tenant, convo: Conversation, client: Client, body: str
    ) -> None:
        if tenant.sms_number:
            effects.text(tenant.id, convo.id, to=client.phone, from_=tenant.sms_number, body=body)

    @staticmethod
    def _queue_slot_fill(
        effects: Effects, tenant: Tenant, appt: Appointment, client: Client
    ) -> None:
        effects.job(
            "start_slot_fill",
            tenant_id=str(tenant.id),
            provider_id=str(appt.provider_id),
            starts_at=appt.starts_at.isoformat(),
            ends_at=appt.ends_at.isoformat(),
            exclude_client_id=str(client.id),
        )


async def accept_inbound(
    session: AsyncSession, tenant: Tenant, *, from_: str, to: str, body: str, sid: str
) -> SmsMessage | None:
    """Store an inbound SMS for ``tenant``: find or create the client and their
    conversation, and record the message. None if ``sid`` was seen before.
    Shared by the Twilio webhook and the local SMS console."""
    client, _ = await ClientRepository(session, tenant.id).get_or_create_by_phone(from_)
    convo = await ConversationRepository(session, tenant.id).for_client(client.id)
    # The column holds 1600, Twilio's limit. Other channels could send more;
    # the cut text is still over MAX_INBOUND_CHARS, so it goes to a person.
    return await SmsMessageRepository(session, tenant.id).record_inbound(
        conversation_id=convo.id,
        body=body[:MAX_STORED_CHARS],
        from_number=from_,
        to_number=to,
        provider_sid=sid,
    )


MAX_STORED_CHARS = 1600


class ReviewAlreadyDecidedError(ValueError):
    pass


class ReviewNeedsReplyError(ValueError):
    pass


async def _flag_draft(
    session: AsyncSession,
    convo: Conversation,
    msg: SmsMessage,
    client: Client,
    held: HeldAction,
    draft: str | None,
) -> HeldAction:
    """A draft going to review anyway is checked too, so the reviewer is warned
    before approving it as written."""
    finding: Finding | None = (
        await check_reply(session, convo.tenant_id, client.id, draft) if draft else None
    )
    if finding is None:
        return held
    note = f"Warning: the draft contains {finding.reason}."
    await _audit_guardrail(session, convo, msg, finding.rule, note)
    return HeldAction(held.kind, f"{held.reason} {note}", held.action)


async def _audit_guardrail(
    session: AsyncSession, convo: Conversation, msg: SmsMessage, rule: str, reason: str
) -> None:
    """The reason names the rule and the matched phrase, never the reply text."""
    await AuditRepository(session, convo.tenant_id).record(
        kind="guardrail",
        actor="system",
        subject=rule,
        outcome="held",
        conversation_id=convo.id,
        ref_id=msg.id,
        detail={"reason": reason[:300]},
    )


def _combine(held: list[HeldAction]) -> HeldAction:
    """Several holds in one turn become one review item; the first decides its
    kind, and a late cancellation's action is kept."""
    action = next((h.action for h in held if h.action), None)
    return HeldAction(held[0].kind, " ".join(h.reason for h in held), action)


def _append(messages: list[Message], message: Message) -> list[Message]:
    """Append, merging into the last message if it has the same role, so the
    transcript always alternates user and assistant as providers require."""
    if messages and messages[-1].role == message.role:
        merged = Message(message.role, [*messages[-1].content, *message.content])
        return [*messages[:-1], merged]
    return [*messages, message]
