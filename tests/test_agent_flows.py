"""End-to-end front-desk flows with a scripted model.

Each test is a story: a client texts, the scripted model calls tools, and we
check what happened in Postgres and what was (or was not) texted back. These
cover the Module 4 "done when" list.
"""

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.agent.loop import MAX_STEPS, READ_ONLY_NOTE
from frontdesk_agent.agent.risk import MEDICAL_HOLDING_MESSAGE
from frontdesk_agent.db.models import (
    Appointment,
    AppointmentStatus,
    Conversation,
    ConversationStatus,
    LLMCall,
    MessageStatus,
    ReviewItem,
    ReviewKind,
    ReviewStatus,
    SmsMessage,
)
from frontdesk_agent.db.repositories import AppointmentRepository
from frontdesk_agent.frontdesk.ports import Effects
from frontdesk_agent.llm.fake import call, loop_turn, say
from frontdesk_agent.prompts import PromptError
from tests.desk import CLIENT_PHONE, NOW, Desk, World, local, make_desk, make_world, text


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session, policies=True)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    yield d
    await d.dispose()


async def scalar(desk: Desk, stmt: object) -> object:
    """Read what the desk committed, in a fresh session (results stay usable)."""
    async with desk.services.sessionmaker() as s:
        return await s.scalar(stmt)  # type: ignore[call-overload]


async def test_ask_policy_then_book_a_slot(desk: Desk, world: World, session: AsyncSession) -> None:
    """Done when: text in, ask about cancellation, book a slot, see it in Postgres."""
    desk.llm.script(
        [call("search_policy", question="cancellation policy")],
        say("We ask for 24 hours' notice. Cancelling later than that is a $50 fee."),
    )
    assert await text(desk, world, "what's your cancellation policy?") == "agent_reply"
    first_prompt = desk.llm.conversations[1]["messages"]
    assert "24 hours" in str(first_prompt)  # the tool result reached the model

    desk.llm.script(
        [call("check_availability", date="2030-01-07")],
        [call("book_appointment", provider_id=str(world.provider_id), starts_at=local("11:00"))],
        say("You're booked with Dr. Osei on Monday, January 7 at 11:00 AM."),
    )
    assert await text(desk, world, "can I come in today at 11?") == "agent_reply"

    appt = await scalar(desk, select(Appointment).where(Appointment.client_id == world.client_id))
    assert isinstance(appt, Appointment) and appt.status == AppointmentStatus.BOOKED
    assert appt.starts_at == NOW + timedelta(hours=2)  # 11:00 Toronto = 16:00 UTC
    assert desk.sender.bodies(CLIENT_PHONE)[-1].startswith("You're booked")
    calls = await scalar(desk, select(func.count()).select_from(LLMCall))
    assert calls == 5  # every model call is costed and logged


async def test_step_limit_stops_a_model_that_loops_forever(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    """Done when: a model that loops forever is stopped by the step limit."""
    desk.llm.script(*[loop_turn() for _ in range(MAX_STEPS + 5)])
    assert await text(desk, world, "hello?") == "held_for_review:step_limit"

    assert len(desk.llm.conversations) == MAX_STEPS  # not one call more
    assert desk.sender.sent == []  # nothing half-finished went to the client
    item = await scalar(desk, select(ReviewItem))
    assert isinstance(item, ReviewItem) and item.kind == ReviewKind.ESCALATION
    assert "6 steps" in item.reason


async def test_late_cancellation_waits_for_approval(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    """Done when: a late cancellation waits in review and sends nothing until approved."""
    async with desk.services.sessionmaker() as s:
        booked = await AppointmentRepository(s, world.tenant_id).book(
            client_id=world.client_id,
            provider_id=world.provider_id,
            starts_at=NOW + timedelta(hours=5),
            ends_at=NOW + timedelta(hours=5, minutes=30),
        )
        await s.commit()
    appt_id = str(booked.appointment.id)

    desk.llm.script(
        [call("list_my_appointments")],
        [call("cancel_appointment", appointment_id=appt_id)],
        say("Your 2:00 PM appointment today is cancelled. Note the $50 late fee."),
    )
    assert await text(desk, world, "I need to cancel today's appointment") == (
        "held_for_review:late_cancellation"
    )
    assert desk.sender.sent == []
    assert desk.queue.jobs == []  # no slot fill for a cancellation that has not happened
    item = await scalar(desk, select(ReviewItem))
    assert isinstance(item, ReviewItem) and item.status == ReviewStatus.PENDING
    assert item.action == {"type": "cancel_appointment", "appointment_id": appt_id}
    still = await scalar(desk, select(Appointment.status).where(Appointment.id == appt_id))
    assert still == AppointmentStatus.BOOKED

    # Messages that arrive while paused are held, not answered.
    assert await text(desk, world, "hello??") == "held_while_paused"
    assert desk.sender.sent == []

    # A person approves; the held cancel runs, the reply goes out, the slot refills.
    desk.llm.script(say("Thanks for your patience. Anything else?"))
    effects = Effects()
    async with desk.services.sessionmaker() as s:
        await desk.services.frontdesk.decide(
            s, world.tenant_id, item.id, effects, approve=True, reply=None, reviewer="sam"
        )
        await s.commit()
        await effects.flush(s, desk.sender, desk.queue)
    assert desk.sender.bodies() == [
        "Your 2:00 PM appointment today is cancelled. Note the $50 late fee."
    ]
    assert sorted(desk.queue.names()) == ["handle_inbound", "start_slot_fill"]
    gone = await scalar(desk, select(Appointment.status).where(Appointment.id == appt_id))
    assert gone == AppointmentStatus.CANCELLED

    # The held "hello??" is answered now, with the staff note in the history.
    await desk.queue.drain(desk.services)
    assert desk.sender.bodies()[-1] == "Thanks for your patience. Anything else?"
    assert "[Staff note] Approved" in str(desk.llm.conversations[-1]["messages"])


async def test_cancellation_outside_24_hours_is_automatic(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    async with desk.services.sessionmaker() as s:
        booked = await AppointmentRepository(s, world.tenant_id).book(
            client_id=world.client_id,
            provider_id=world.provider_id,
            starts_at=NOW + timedelta(days=3),
            ends_at=NOW + timedelta(days=3, minutes=30),
        )
        await s.commit()
    desk.llm.script(
        [call("cancel_appointment", appointment_id=str(booked.appointment.id))],
        say("Done, your Thursday appointment is cancelled."),
    )
    assert await text(desk, world, "cancel thursday please") == "agent_reply"
    assert desk.queue.names() == ["start_slot_fill"]
    assert await scalar(desk, select(func.count()).select_from(ReviewItem)) == 0


async def test_medical_question_is_held_with_a_safety_message(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    desk.llm.script(say("Swelling after an extraction can be normal; ice it."))  # a bad draft
    assert await text(desk, world, "my cheek is really swollen after the extraction") == (
        "held_for_review:medical"
    )
    # The model's draft is held for a person; only the fixed safety text goes out.
    assert desk.sender.bodies() == [MEDICAL_HOLDING_MESSAGE]
    item = await scalar(desk, select(ReviewItem))
    assert isinstance(item, ReviewItem) and item.proposed_reply.startswith("Swelling")
    # The draft ran with read-only tools: it could not have booked or cancelled.
    assert "book_appointment" not in desk.llm.conversations[0]["tools"]  # type: ignore[operator]
    assert "search_policy" in desk.llm.conversations[0]["tools"]  # type: ignore[operator]
    # It is told the team already has it, since escalate_to_human is not offered.
    assert "escalate_to_human" not in desk.llm.conversations[0]["tools"]  # type: ignore[operator]
    assert READ_ONLY_NOTE in str(desk.llm.conversations[0]["system"])


async def test_money_question_is_held_silently(desk: Desk, world: World) -> None:
    desk.llm.script(say("I'll ask the team about a refund."))
    assert await text(desk, world, "you charged me twice, I want a refund") == (
        "held_for_review:money"
    )
    assert desk.sender.sent == []


async def test_agent_can_escalate_on_its_own(desk: Desk, world: World) -> None:
    desk.llm.script(
        [call("escalate_to_human", reason="complaint", summary="Unhappy with last visit")],
        say("I'm sorry to hear that. A manager will follow up today."),
    )
    assert await text(desk, world, "honestly the last visit was awful") == (
        "held_for_review:escalation"
    )
    assert desk.sender.sent == []


async def test_reject_runs_nothing_and_sends_only_the_reviewers_text(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    desk.llm.script(say("Sure, refund on its way!"))
    await text(desk, world, "please refund my deposit")
    item = await scalar(desk, select(ReviewItem))
    assert isinstance(item, ReviewItem)
    effects = Effects()
    async with desk.services.sessionmaker() as s:
        await desk.services.frontdesk.decide(
            s,
            world.tenant_id,
            item.id,
            effects,
            approve=False,
            reply="Please call us at the front desk about deposits.",
            reviewer="sam",
        )
        await s.commit()
        await effects.flush(s, desk.sender, desk.queue)
    assert desk.sender.bodies() == ["Please call us at the front desk about deposits."]
    convo = await scalar(desk, select(Conversation))
    assert isinstance(convo, Conversation) and convo.status == ConversationStatus.ACTIVE


async def test_stop_and_start(desk: Desk, world: World) -> None:
    assert await text(desk, world, "STOP") == "opted_out"
    assert await text(desk, world, "are you open friday") == "ignored_opted_out"
    assert desk.llm.conversations == [] and desk.sender.sent == []
    assert await text(desk, world, "START") == "opted_in"


async def test_new_texter_must_give_a_name_before_booking(desk: Desk, world: World) -> None:
    stranger = "+14165550188"
    desk.llm.script(
        [call("book_appointment", provider_id=str(world.provider_id), starts_at=local("13:00"))],
        [call("save_client_name", first_name="Ari", last_name="Cohen")],
        [call("book_appointment", provider_id=str(world.provider_id), starts_at=local("13:00"))],
        say("Thanks Ari, you're booked for 1:00 PM today."),
    )
    assert await text(desk, world, "book me at 1pm, I'm Ari Cohen", phone=stranger) == "agent_reply"
    results = str(desk.llm.conversations[1]["messages"])
    assert "save_client_name first" in results  # the first attempt was refused


async def test_twilio_retry_of_the_same_message_is_a_no_op(desk: Desk, world: World) -> None:
    desk.llm.script(say("We open at 8."))
    assert await text(desk, world, "when do you open", sid="SMsame") == "agent_reply"
    assert await text(desk, world, "when do you open", sid="SMsame") == "duplicate"
    assert len(desk.sender.sent) == 1


async def test_a_failed_send_is_logged_not_lost(
    desk: Desk, world: World, session: AsyncSession
) -> None:
    desk.sender.fail = True
    desk.llm.script(say("We open at 8."))
    await text(desk, world, "when do you open")
    failed = await scalar(
        desk, select(func.count()).where(SmsMessage.status == MessageStatus.FAILED)
    )
    assert failed == 1


async def test_prompt_version_setting_reaches_the_model_and_the_record(
    desk: Desk, world: World
) -> None:
    """Evals compare prompt versions; the version must be the one that ran."""
    desk.services.frontdesk.prompt_version = "v1"
    desk.llm.script(say("Hi Dana, how can I help?"))
    assert await text(desk, world, "hi") == "agent_reply"
    recorded = await scalar(desk, select(LLMCall.prompt_version))
    assert recorded == "v1"

    desk.services.frontdesk.prompt_version = "v999"
    with pytest.raises(PromptError):  # a missing version fails loudly, never falls back
        await text(desk, world, "hello again")
