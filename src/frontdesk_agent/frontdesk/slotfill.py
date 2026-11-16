"""Filling a cancelled slot from the waitlist.

A workflow, not an agent: every step is fixed, so no model is involved.

    slot frees up -> rank waiting clients (bandit) -> text #1 an offer
      -> YES: book it (first yes wins, enforced by the slot's unique index)
      -> NO or no reply in OFFER_MINUTES: text the next client
      -> nobody left: the slot stays open for normal booking

Offers go out one at a time so nobody hears "yes, it's yours" and then
"sorry, someone else was faster". Expiry is a delayed job on the queue.
"""

import random
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import (
    Client,
    OfferStatus,
    SlotOffer,
    Tenant,
    WaitlistEntry,
    WaitlistStatus,
)
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    ConversationRepository,
    ProviderRepository,
    ResponseStatsRepository,
    SlotOfferRepository,
    SlotTakenError,
    WaitlistRepository,
)
from frontdesk_agent.frontdesk.bandit import Arm, EpsilonGreedy, response_reward
from frontdesk_agent.frontdesk.ports import Effects

OFFER_MINUTES = 15

YES = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "yes please", "i'll take it"}
NO = {"no", "n", "nope", "no thanks", "no thank you", "pass"}


def parse_reply(body: str) -> bool | None:
    """True for yes, False for no, None for anything else (goes to the agent)."""
    text = body.strip().lower().rstrip(".!")
    if text in YES:
        return True
    if text in NO:
        return False
    return None


class SlotFiller:
    def __init__(self, bandit: EpsilonGreedy | None = None) -> None:
        self._bandit = bandit or EpsilonGreedy(rng=random.Random())

    async def start(
        self,
        session: AsyncSession,
        tenant: Tenant,
        effects: Effects,
        *,
        provider_id: uuid.UUID,
        starts_at: datetime,
        ends_at: datetime,
        now: datetime,
        exclude_client_id: uuid.UUID | None = None,
    ) -> SlotOffer | None:
        """Called after a cancellation commits. Returns the first offer, if any."""
        return await self._offer_next(
            session, tenant, effects, provider_id, starts_at, ends_at, now, exclude_client_id
        )

    async def respond(
        self,
        session: AsyncSession,
        tenant: Tenant,
        effects: Effects,
        offer: SlotOffer,
        client: Client,
        accepted: bool,
        now: datetime,
    ) -> None:
        offer.responded_at = now
        window = timedelta(minutes=OFFER_MINUTES).total_seconds()
        stats = ResponseStatsRepository(session, tenant.id)
        convo = await ConversationRepository(session, tenant.id).for_client(client.id)

        if not accepted:
            offer.status = OfferStatus.DECLINED
            await stats.record(client.id, 0.0)
            self._text(
                effects, tenant, convo.id, client, "No problem, you're still on the waitlist."
            )
            await self._offer_next(
                session, tenant, effects, offer.provider_id, offer.starts_at, offer.ends_at, now
            )
            return

        try:
            await AppointmentRepository(session, tenant.id).book(
                client_id=client.id,
                provider_id=offer.provider_id,
                starts_at=offer.starts_at,
                ends_at=offer.ends_at,
                idempotency_key=f"offer:{offer.id}",
            )
        except SlotTakenError:
            offer.status = OfferStatus.TAKEN
            self._text(
                effects,
                tenant,
                convo.id,
                client,
                "Sorry, that time was just booked. You're still on the waitlist.",
            )
            return

        offer.status = OfferStatus.ACCEPTED
        entry = await WaitlistRepository(session, tenant.id).get(offer.waitlist_entry_id)
        entry.status = WaitlistStatus.FILLED
        elapsed = (now - offer.created_at).total_seconds() if offer.created_at else window
        await stats.record(client.id, response_reward(elapsed, window))
        when = _human(offer.starts_at, tenant)
        self._text(effects, tenant, convo.id, client, f"You're booked for {when}. See you then.")

    async def expire(
        self,
        session: AsyncSession,
        tenant: Tenant,
        effects: Effects,
        offer_id: uuid.UUID,
        now: datetime,
    ) -> None:
        """Delayed job: if the offer is still open, close it and move on."""
        offer = await SlotOfferRepository(session, tenant.id).get(offer_id)
        if offer.status != OfferStatus.OFFERED or offer.expires_at > now:
            return  # answered already, or the job ran early
        offer.status = OfferStatus.EXPIRED
        await ResponseStatsRepository(session, tenant.id).record(offer.client_id, 0.0)
        await self._offer_next(
            session, tenant, effects, offer.provider_id, offer.starts_at, offer.ends_at, now
        )

    async def _offer_next(
        self,
        session: AsyncSession,
        tenant: Tenant,
        effects: Effects,
        provider_id: uuid.UUID,
        starts_at: datetime,
        ends_at: datetime,
        now: datetime,
        exclude_client_id: uuid.UUID | None = None,
    ) -> SlotOffer | None:
        if starts_at <= now:
            return None
        booked = await AppointmentRepository(session, tenant.id).booked_between(starts_at, ends_at)
        if any(a.provider_id == provider_id for a in booked):
            return None  # booked by other means since the cancellation

        offers = SlotOfferRepository(session, tenant.id)
        already = {o.client_id for o in await offers.for_slot(provider_id, starts_at)}
        clients = ClientRepository(session, tenant.id)
        entries: list[tuple[WaitlistEntry, Client]] = []
        for e in await WaitlistRepository(session, tenant.id).waiting_for(starts_at, ends_at):
            if e.provider_id not in (None, provider_id):
                continue
            if e.client_id in already or e.client_id == exclude_client_id:
                continue
            c = await clients.get(e.client_id)
            if c.sms_opt_in:
                entries.append((e, c))
        if not entries:
            return None

        stats = await ResponseStatsRepository(session, tenant.id).for_clients(
            [c.id for _, c in entries]
        )
        arms = [
            Arm(
                c.id,
                stats[c.id].offers if c.id in stats else 0,
                stats[c.id].reward_sum if c.id in stats else 0.0,
                e.created_at.timestamp(),
            )
            for e, c in entries
        ]
        chosen = self._bandit.choose(arms)
        entry, client = next((e, c) for e, c in entries if c.id == chosen.client_id)

        offer = SlotOffer(
            client_id=client.id,
            waitlist_entry_id=entry.id,
            provider_id=provider_id,
            starts_at=starts_at,
            ends_at=ends_at,
            rank=len(already) + 1,
            expires_at=now + timedelta(minutes=OFFER_MINUTES),
        )
        await offers.add(offer)
        provider = await ProviderRepository(session, tenant.id).get(provider_id)
        convo = await ConversationRepository(session, tenant.id).for_client(client.id)
        name = f"Hi {client.first_name}, a" if client.first_name else "A"
        self._text(
            effects,
            tenant,
            convo.id,
            client,
            f"{name} spot with {provider.name} just opened: {_human(starts_at, tenant)}. "
            f"Reply YES in the next {OFFER_MINUTES} minutes to take it, or NO to pass.",
        )
        effects.job(
            "expire_offer",
            defer_seconds=OFFER_MINUTES * 60,
            tenant_id=str(tenant.id),
            offer_id=str(offer.id),
        )
        return offer

    @staticmethod
    def _text(
        effects: Effects, tenant: Tenant, conversation_id: uuid.UUID, client: Client, body: str
    ) -> None:
        if tenant.sms_number:
            effects.text(
                tenant.id, conversation_id, to=client.phone, from_=tenant.sms_number, body=body
            )


def _human(when: datetime, tenant: Tenant) -> str:
    return when.astimezone(ZoneInfo(tenant.timezone)).strftime("%A, %B %-d at %-I:%M %p")
