"""Tenant-scoped data access.

The rule this module exists to enforce: **no repository method takes a tenant
argument.** The tenant is bound once, at construction, and every query and
every insert goes through ``_scoped`` / ``_claim``. Application code therefore
cannot express a cross-tenant query by mistake; it would have to build a second
repository for the other tenant, which is visible in review. ADR 0005.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.base import TenantScoped
from frontdesk_agent.db.models import (
    ApiKey,
    Appointment,
    AppointmentStatus,
    AuditEvent,
    Client,
    ClientResponseStats,
    Conversation,
    LLMCall,
    MessageDirection,
    MessageStatus,
    OfferStatus,
    Provider,
    ProviderSchedule,
    ReviewItem,
    ReviewKind,
    ReviewStatus,
    Role,
    SlotOffer,
    SmsMessage,
    Tenant,
    User,
    WaitlistEntry,
    WaitlistStatus,
)
from frontdesk_agent.llm.base import LLMCompletion


class NotFoundError(LookupError):
    """The row does not exist *for this tenant*. Deliberately the same error
    whether it is missing or belongs to someone else: no existence oracle."""


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused with a different request body."""


class SlotTakenError(ValueError):
    """The provider already has a booked appointment at that start time."""


class TenantDirectory:
    """The only unscoped access: creating and looking up tenants themselves.

    Module 6 puts this behind admin auth. Nothing else in the app should use it.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        name: str,
        business_type: str,
        timezone: str = "America/Toronto",
        sms_number: str | None = None,
        slug: str | None = None,
    ) -> Tenant:
        tenant = Tenant(
            slug=slug,
            name=name,
            business_type=business_type,
            timezone=timezone,
            sms_number=sms_number,
        )
        self._session.add(tenant)
        await self._session.flush()
        return tenant

    async def get(self, tenant_id: uuid.UUID) -> Tenant:
        tenant = await self._session.get(Tenant, tenant_id)
        if tenant is None:
            raise NotFoundError("tenant not found")
        return tenant

    async def by_slug(self, slug: str) -> Tenant | None:
        result = await self._session.execute(select(Tenant).where(Tenant.slug == slug))
        return result.scalar_one_or_none()

    async def by_sms_number(self, sms_number: str) -> Tenant | None:
        """Route an inbound SMS to its tenant (Module 4)."""
        result = await self._session.execute(select(Tenant).where(Tenant.sms_number == sms_number))
        return result.scalar_one_or_none()


class TenantRepository[M: TenantScoped]:
    model: type[M]

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID) -> None:
        self._session = session
        self._tenant_id = tenant_id

    @property
    def tenant_id(self) -> uuid.UUID:
        return self._tenant_id

    def _scoped(self) -> Select[M]:
        return select(self.model).where(self.model.tenant_id == self._tenant_id)

    def _claim(self, obj: M) -> M:
        # Overwrite, never trust: whatever tenant_id the caller set is replaced.
        obj.tenant_id = self._tenant_id
        return obj

    async def get(self, obj_id: uuid.UUID) -> M:
        result = await self._session.execute(self._scoped().where(self.model.id == obj_id))
        obj = result.scalar_one_or_none()
        if obj is None:
            raise NotFoundError(f"{self.model.__name__} not found")
        return obj

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[M]:
        stmt = (
            self._scoped()
            .order_by(self.model.created_at, self.model.id)
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(stmt)).scalars())

    async def count(self) -> int:
        stmt = (
            select(func.count())
            .select_from(self.model)
            .where(self.model.tenant_id == self._tenant_id)
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def add(self, obj: M) -> M:
        self._session.add(self._claim(obj))
        await self._session.flush()
        return obj


class ClientRepository(TenantRepository[Client]):
    model = Client

    async def by_phone(self, phone: str) -> Client | None:
        result = await self._session.execute(self._scoped().where(Client.phone == phone))
        return result.scalar_one_or_none()

    async def get_or_create_by_phone(self, phone: str) -> tuple[Client, bool]:
        """A texter we have never seen becomes a client record, flagged by
        empty names, so the conversation has someone to belong to."""
        existing = await self.by_phone(phone)
        if existing is not None:
            return existing, False
        return await self.add(Client(first_name="", last_name="", phone=phone)), True


class ProviderRepository(TenantRepository[Provider]):
    model = Provider


@dataclass(frozen=True, slots=True)
class BookingResult:
    appointment: Appointment
    created: bool  # False when an idempotency key replayed an earlier booking


class AppointmentRepository(TenantRepository[Appointment]):
    model = Appointment

    async def by_idempotency_key(self, key: str) -> Appointment | None:
        result = await self._session.execute(
            self._scoped().where(Appointment.idempotency_key == key)
        )
        return result.scalar_one_or_none()

    async def between(self, start: datetime, end: datetime) -> list[Appointment]:
        stmt = (
            self._scoped()
            .where(Appointment.starts_at >= start, Appointment.starts_at < end)
            .order_by(Appointment.starts_at)
        )
        return list((await self._session.execute(stmt)).scalars())

    async def book(
        self,
        *,
        client_id: uuid.UUID,
        provider_id: uuid.UUID,
        starts_at: datetime,
        ends_at: datetime,
        notes: str | None = None,
        idempotency_key: str | None = None,
    ) -> BookingResult:
        """Book once. Replaying the same key returns the original booking.

        The client and provider are checked against *this* tenant first. A
        foreign key alone would happily let tenant A book tenant B's client,
        because the referenced row exists, just in the wrong tenant.
        """
        await ClientRepository(self._session, self._tenant_id).get(client_id)
        await ProviderRepository(self._session, self._tenant_id).get(provider_id)

        values: dict[str, Any] = {
            "id": uuid.uuid4(),
            "tenant_id": self._tenant_id,
            "client_id": client_id,
            "provider_id": provider_id,
            "starts_at": starts_at,
            "ends_at": ends_at,
            "notes": notes,
            "idempotency_key": idempotency_key,
            "status": AppointmentStatus.BOOKED,
        }
        stmt = insert(Appointment).values(**values)
        if idempotency_key is not None:
            # Atomic against concurrent replays: the unique partial index
            # decides the winner; the loser inserts nothing.
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["tenant_id", "idempotency_key"],
                index_where=text("idempotency_key IS NOT NULL"),
            )
        try:
            # A savepoint, so a slot conflict rolls back this insert only and
            # the caller's transaction (an agent turn, say) stays usable.
            async with self._session.begin_nested():
                inserted = (await self._session.execute(stmt.returning(Appointment.id))).scalar()
        except IntegrityError as exc:
            if "uq_appointments_provider_slot_booked" not in str(exc.orig):
                raise
            # ON CONFLICT covers only the key index. When two replays race,
            # Postgres may check the slot index first and raise here even
            # though the winner is our own replay, so look for it by key.
            existing = (
                await self.by_idempotency_key(idempotency_key)
                if idempotency_key is not None
                else None
            )
            if existing is None:
                raise SlotTakenError("that slot was just booked by someone else") from exc
        else:
            if inserted is not None:
                return BookingResult(await self.get(inserted), created=True)
            assert idempotency_key is not None
            existing = await self.by_idempotency_key(idempotency_key)
        if existing is None:  # pragma: no cover - the conflict implies it exists
            raise RuntimeError("idempotency conflict without an existing row")
        same = (
            existing.client_id == client_id
            and existing.provider_id == provider_id
            and existing.starts_at == starts_at
            and existing.ends_at == ends_at
        )
        if not same:
            raise IdempotencyConflictError(
                "idempotency key was already used for a different booking"
            )
        return BookingResult(existing, created=False)

    async def cancel(self, appointment_id: uuid.UUID) -> Appointment:
        appt = await self.get(appointment_id)
        appt.status = AppointmentStatus.CANCELLED
        await self._session.flush()
        return appt

    async def booked_between(self, start: datetime, end: datetime) -> list[Appointment]:
        """Booked appointments overlapping [start, end)."""
        stmt = self._scoped().where(
            Appointment.status == AppointmentStatus.BOOKED,
            Appointment.starts_at < end,
            Appointment.ends_at > start,
        )
        return list((await self._session.execute(stmt)).scalars())

    async def upcoming_for_client(self, client_id: uuid.UUID, now: datetime) -> list[Appointment]:
        stmt = (
            self._scoped()
            .where(
                Appointment.client_id == client_id,
                Appointment.status == AppointmentStatus.BOOKED,
                Appointment.starts_at > now,
            )
            .order_by(Appointment.starts_at)
        )
        return list((await self._session.execute(stmt)).scalars())


class ScheduleRepository(TenantRepository[ProviderSchedule]):
    model = ProviderSchedule

    async def for_weekday(self, weekday: int) -> list[ProviderSchedule]:
        stmt = self._scoped().where(ProviderSchedule.weekday == weekday)
        return list((await self._session.execute(stmt)).scalars())


class WaitlistRepository(TenantRepository[WaitlistEntry]):
    model = WaitlistEntry

    async def waiting_for(self, start: datetime, end: datetime) -> list[WaitlistEntry]:
        """Entries whose window covers a freed slot, oldest first (fairness)."""
        stmt = (
            self._scoped()
            .where(
                WaitlistEntry.status == WaitlistStatus.WAITING,
                WaitlistEntry.earliest <= start,
                WaitlistEntry.latest >= end,
            )
            .order_by(WaitlistEntry.created_at)
        )
        return list((await self._session.execute(stmt)).scalars())


class LLMCallRepository(TenantRepository[LLMCall]):
    model = LLMCall

    async def record(
        self,
        completion: LLMCompletion,
        *,
        cost_usd: Decimal | None = None,
        succeeded: bool = True,
    ) -> LLMCall:
        return await self.add(
            LLMCall(
                model=completion.model,
                prompt_name=completion.prompt_name,
                prompt_version=completion.prompt_version,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cache_read_tokens=completion.cache_read_tokens,
                cache_write_tokens=completion.cache_write_tokens,
                latency_ms=completion.latency_ms,
                stop_reason=completion.stop_reason,
                cost_usd=cost_usd,
                succeeded=succeeded,
            )
        )

    async def recent(self, *, limit: int = 50) -> list[LLMCall]:
        stmt = self._scoped().order_by(LLMCall.created_at.desc(), LLMCall.id).limit(limit)
        return list((await self._session.execute(stmt)).scalars())

    async def count_since(self, seconds: float) -> int:
        """Model calls in the last ``seconds`` of database time, for rate limits."""
        since = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, seconds)
        stmt = select(func.count()).where(
            LLMCall.tenant_id == self._tenant_id, LLMCall.created_at > since
        )
        return int((await self._session.execute(stmt)).scalar_one())


class ConversationRepository(TenantRepository[Conversation]):
    model = Conversation

    async def for_client(self, client_id: uuid.UUID) -> Conversation:
        """The client's single thread, created on first contact.

        Locked FOR UPDATE: two messages from one client handled at once must
        not interleave their agent turns and corrupt the transcript.
        """
        stmt = self._scoped().where(Conversation.client_id == client_id).with_for_update()
        convo = (await self._session.execute(stmt)).scalar_one_or_none()
        if convo is not None:
            return convo
        stmt_ins = (
            insert(Conversation)
            .values(id=uuid.uuid4(), tenant_id=self._tenant_id, client_id=client_id, transcript=[])
            .on_conflict_do_nothing(index_elements=["tenant_id", "client_id"])
        )
        await self._session.execute(stmt_ins)
        return (await self._session.execute(stmt)).scalar_one()


class SmsMessageRepository(TenantRepository[SmsMessage]):
    model = SmsMessage

    async def record_inbound(
        self,
        *,
        conversation_id: uuid.UUID,
        body: str,
        from_number: str,
        to_number: str,
        provider_sid: str | None,
    ) -> SmsMessage | None:
        """Store an inbound SMS. Returns None if this provider_sid was seen
        before: Twilio retried the webhook, and the retry must do nothing."""
        stmt = (
            insert(SmsMessage)
            .values(
                id=uuid.uuid4(),
                tenant_id=self._tenant_id,
                conversation_id=conversation_id,
                direction=MessageDirection.INBOUND,
                status=MessageStatus.RECEIVED,
                body=body[:1600],
                from_number=from_number,
                to_number=to_number,
                provider_sid=provider_sid,
            )
            .on_conflict_do_nothing(
                index_elements=["tenant_id", "provider_sid"],
                index_where=text("provider_sid IS NOT NULL"),
            )
            .returning(SmsMessage.id)
        )
        new_id = (await self._session.execute(stmt)).scalar()
        return None if new_id is None else await self.get(new_id)

    async def inbound_count_since(self, conversation_id: uuid.UUID, seconds: float) -> int:
        """Inbound texts in the last ``seconds`` of database time, counting only
        those after the latest review decision in the conversation: once a person
        has looked, the client starts a fresh window."""
        since = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, seconds)
        decided = (
            select(func.max(ReviewItem.updated_at))
            .where(
                ReviewItem.tenant_id == self._tenant_id,
                ReviewItem.conversation_id == conversation_id,
                ReviewItem.status != ReviewStatus.PENDING,
            )
            .scalar_subquery()
        )
        stmt = select(func.count()).where(
            SmsMessage.tenant_id == self._tenant_id,
            SmsMessage.conversation_id == conversation_id,
            SmsMessage.direction == MessageDirection.INBOUND,
            SmsMessage.created_at > since,
            SmsMessage.created_at > func.coalesce(decided, since),
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def in_conversation(
        self, conversation_id: uuid.UUID, *, status: MessageStatus | None = None
    ) -> list[SmsMessage]:
        stmt = self._scoped().where(SmsMessage.conversation_id == conversation_id)
        if status is not None:
            stmt = stmt.where(SmsMessage.status == status)
        stmt = stmt.order_by(SmsMessage.created_at, SmsMessage.id)
        return list((await self._session.execute(stmt)).scalars())


class ReviewRepository(TenantRepository[ReviewItem]):
    model = ReviewItem

    async def by_status(self, status: ReviewStatus | None, *, limit: int = 50) -> list[ReviewItem]:
        stmt = self._scoped()
        if status is not None:
            stmt = stmt.where(ReviewItem.status == status)
        stmt = stmt.order_by(ReviewItem.created_at, ReviewItem.id).limit(limit)
        return list((await self._session.execute(stmt)).scalars())

    async def lock(self, review_id: uuid.UUID) -> ReviewItem:
        """Row-locked fetch, so two reviewers cannot both decide one item."""
        stmt = self._scoped().where(ReviewItem.id == review_id).with_for_update()
        item = (await self._session.execute(stmt)).scalar_one_or_none()
        if item is None:
            raise NotFoundError("review item not found")
        return item

    async def pending_for(self, conversation_id: uuid.UUID) -> list[ReviewItem]:
        stmt = self._scoped().where(
            ReviewItem.conversation_id == conversation_id,
            ReviewItem.status == ReviewStatus.PENDING,
        )
        return list((await self._session.execute(stmt)).scalars())


class SlotOfferRepository(TenantRepository[SlotOffer]):
    model = SlotOffer

    async def open_for_client(self, client_id: uuid.UUID, now: datetime) -> SlotOffer | None:
        stmt = (
            self._scoped()
            .where(
                SlotOffer.client_id == client_id,
                SlotOffer.status == OfferStatus.OFFERED,
                SlotOffer.expires_at > now,
            )
            .order_by(SlotOffer.created_at.desc())
            .limit(1)
            .with_for_update()
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def for_slot(self, provider_id: uuid.UUID, starts_at: datetime) -> list[SlotOffer]:
        stmt = (
            self._scoped()
            .where(SlotOffer.provider_id == provider_id, SlotOffer.starts_at == starts_at)
            .order_by(SlotOffer.rank)
        )
        return list((await self._session.execute(stmt)).scalars())


class ResponseStatsRepository(TenantRepository[ClientResponseStats]):
    model = ClientResponseStats

    async def for_clients(
        self, client_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, ClientResponseStats]:
        stmt = self._scoped().where(ClientResponseStats.client_id.in_(client_ids))
        return {s.client_id: s for s in (await self._session.execute(stmt)).scalars()}

    async def record(self, client_id: uuid.UUID, reward: float) -> None:
        stmt = (
            insert(ClientResponseStats)
            .values(
                id=uuid.uuid4(),
                tenant_id=self._tenant_id,
                client_id=client_id,
                offers=1,
                reward_sum=reward,
            )
            .on_conflict_do_update(
                index_elements=["tenant_id", "client_id"],
                set_={
                    "offers": ClientResponseStats.offers + 1,
                    "reward_sum": ClientResponseStats.reward_sum + reward,
                    "updated_at": func.now(),
                },
            )
        )
        await self._session.execute(stmt)


class AuditRepository(TenantRepository[AuditEvent]):
    """Append-only. There is deliberately no update or delete here, and the
    table rejects UPDATE at the database level (migration 0004)."""

    model = AuditEvent

    async def record(
        self,
        *,
        kind: str,
        actor: str,
        subject: str,
        outcome: str,
        conversation_id: uuid.UUID | None = None,
        ref_id: uuid.UUID | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AuditEvent:
        return await self.add(
            AuditEvent(
                kind=kind,
                actor=actor,
                subject=subject,
                outcome=outcome,
                conversation_id=conversation_id,
                ref_id=ref_id,
                detail=detail or {},
            )
        )

    async def recent(self, *, kind: str | None = None, limit: int = 100) -> list[AuditEvent]:
        stmt = self._scoped()
        if kind is not None:
            stmt = stmt.where(AuditEvent.kind == kind)
        stmt = stmt.order_by(AuditEvent.seq.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars())


class CredentialDirectory:
    """The two lookups that happen before the tenant is known: a user by
    email at login, and an API key by its public prefix.

    Like TenantDirectory, deliberately not tenant-bound (ADR 0005, 0016):
    the tenant is what these lookups establish. Nothing else may use them.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def user_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email.strip().lower())
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def api_key_by_prefix(self, prefix: str) -> ApiKey | None:
        stmt = select(ApiKey).where(ApiKey.prefix == prefix)
        return (await self._session.execute(stmt)).scalar_one_or_none()


class EmailTakenError(Exception):
    pass


class UserRepository(TenantRepository[User]):
    model = User

    async def create(self, *, email: str, password_hash: str, role: Role) -> User:
        email = email.strip().lower()
        if await CredentialDirectory(self._session).user_by_email(email) is not None:
            raise EmailTakenError("a user with that email already exists")
        return await self.add(User(email=email, password_hash=password_hash, role=role))


class ApiKeyRepository(TenantRepository[ApiKey]):
    model = ApiKey

    async def create(self, *, name: str, prefix: str, key_hash: str, role: Role) -> ApiKey:
        return await self.add(ApiKey(name=name, prefix=prefix, key_hash=key_hash, role=role))

    async def revoke(self, key_id: uuid.UUID, now: datetime) -> ApiKey:
        key = await self.get(key_id)
        if key.revoked_at is None:
            key.revoked_at = now
            await self._session.flush()
        return key


class OperationsCounts:
    """Counts across every tenant, for /metrics only.

    Not tenant-bound, like the two directories above, because an operator's
    dashboard is about the whole system. It returns numbers, never rows, so
    nothing a tenant owns can leave through it.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def pending_reviews_by_kind(self) -> dict[str, int]:
        stmt = (
            select(ReviewItem.kind, func.count())
            .where(ReviewItem.status == ReviewStatus.PENDING)
            .group_by(ReviewItem.kind)
        )
        counts = {k.value: 0 for k in ReviewKind}
        for kind, n in (await self._session.execute(stmt)).all():
            counts[kind.value] = int(n)
        return counts
