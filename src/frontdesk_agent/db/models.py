"""Tables.

Fourteen tables: tenants, plus thirteen tenant-scoped ones. Kept deliberately small;
later modules add columns through migrations, not by guessing now.
"""

import enum
import uuid
from datetime import datetime, time
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from frontdesk_agent.db.base import Base, IdAndTimestamps, TenantScoped
from frontdesk_agent.retrieval.embeddings import EMBEDDING_DIM


class AppointmentStatus(enum.StrEnum):
    BOOKED = "booked"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class WaitlistStatus(enum.StrEnum):
    WAITING = "waiting"
    OFFERED = "offered"
    FILLED = "filled"
    EXPIRED = "expired"


def _enum(e: type[enum.StrEnum], name: str) -> Enum:
    # Store the lowercase value, not the Python member name.
    return Enum(e, name=name, values_callable=lambda members: [m.value for m in members])


class Tenant(Base, IdAndTimestamps):
    """One business. The root of isolation: everything else hangs off this."""

    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(120))
    # Stable handle for scripts and document folders: data/policies/<slug>/.
    slug: Mapped[str | None] = mapped_column(String(60), unique=True)
    business_type: Mapped[str] = mapped_column(String(60))
    timezone: Mapped[str] = mapped_column(String(64), default="America/Toronto")
    # The SMS number clients text. Unique across tenants: an inbound message
    # is routed to a tenant by the number it was sent to (Module 4).
    sms_number: Mapped[str | None] = mapped_column(String(20), unique=True)


class Client(Base, TenantScoped):
    """A person who books with the business. Synthetic data only."""

    __tablename__ = "clients"
    __table_args__ = (UniqueConstraint("tenant_id", "phone"),)

    first_name: Mapped[str] = mapped_column(String(60))
    last_name: Mapped[str] = mapped_column(String(60))
    phone: Mapped[str] = mapped_column(String(20))  # E.164
    email: Mapped[str | None] = mapped_column(String(254))
    sms_opt_in: Mapped[bool] = mapped_column(default=True)


class Provider(Base, TenantScoped):
    """Someone clients book time with: a dentist, a stylist, a physio."""

    __tablename__ = "providers"

    name: Mapped[str] = mapped_column(String(120))
    title: Mapped[str | None] = mapped_column(String(60))
    active: Mapped[bool] = mapped_column(default=True)


class Appointment(Base, TenantScoped):
    __tablename__ = "appointments"
    __table_args__ = (
        CheckConstraint("ends_at > starts_at", name="ends_after_starts"),
        # Closes the double-booking race at the database, not just in code:
        # two concurrent requests with the same key cannot both insert.
        Index(
            "uq_appointments_tenant_idempotency_key",
            "tenant_id",
            "idempotency_key",
            unique=True,
            postgresql_where="idempotency_key IS NOT NULL",
        ),
        Index("ix_appointments_tenant_starts_at", "tenant_id", "starts_at"),
        # One booked appointment per provider per start time, enforced by the
        # database: two agents (or an agent and the waitlist worker) racing for
        # the same slot cannot both win. Cancelled rows do not count.
        Index(
            "uq_appointments_provider_slot_booked",
            "provider_id",
            "starts_at",
            unique=True,
            postgresql_where="status = 'booked'",
        ),
    )

    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    provider_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("providers.id", ondelete="RESTRICT"))
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[AppointmentStatus] = mapped_column(
        _enum(AppointmentStatus, "appointment_status"), default=AppointmentStatus.BOOKED
    )
    notes: Mapped[str | None] = mapped_column(String(500))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))


class WaitlistEntry(Base, TenantScoped):
    """A client who wants an earlier slot. Module 4 fills cancellations from here."""

    __tablename__ = "waitlist_entries"

    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    provider_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("providers.id", ondelete="SET NULL")
    )
    earliest: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latest: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[WaitlistStatus] = mapped_column(
        _enum(WaitlistStatus, "waitlist_status"), default=WaitlistStatus.WAITING
    )


class LLMCall(Base, TenantScoped):
    """One model call: what it cost and how long it took.

    Mirrors ``LLMCompletion`` from Module 1. Module 7 builds cost and latency
    dashboards on this table, which is why it exists before the agent does.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_tenant_created_at", "tenant_id", "created_at"),)

    model: Mapped[str] = mapped_column(String(80))
    prompt_name: Mapped[str] = mapped_column(String(80))
    prompt_version: Mapped[str] = mapped_column(String(20))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    stop_reason: Mapped[str | None] = mapped_column(String(40))
    succeeded: Mapped[bool] = mapped_column(default=True)


class PolicyDocument(Base, TenantScoped):
    """One source document: cancellation policy, insurance FAQ, and so on."""

    __tablename__ = "policy_documents"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)

    slug: Mapped[str] = mapped_column(String(120))
    title: Mapped[str] = mapped_column(String(200))
    # Re-ingesting an unchanged document is a no-op, decided by this hash
    # together with the chunker and embedding model that produced the chunks.
    content_sha256: Mapped[str] = mapped_column(String(64))
    chunker: Mapped[str] = mapped_column(String(40))
    embedding_model: Mapped[str] = mapped_column(String(80))


class PolicyChunk(Base, TenantScoped):
    """A retrievable piece of a document, embedded and full-text indexed."""

    __tablename__ = "policy_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal"),
        Index(
            "ix_policy_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_policy_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("policy_documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Generated by Postgres, so keyword search can never drift from content.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', heading || ' ' || content)", persisted=True),
    )


class ProviderSchedule(Base, TenantScoped):
    """When a provider takes appointments: one row per weekday block."""

    __tablename__ = "provider_schedules"
    __table_args__ = (
        CheckConstraint("weekday BETWEEN 0 AND 6", name="weekday_range"),
        CheckConstraint("end_time > start_time", name="end_after_start"),
        CheckConstraint("slot_minutes BETWEEN 5 AND 240", name="slot_minutes_range"),
    )

    provider_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"), index=True
    )
    weekday: Mapped[int] = mapped_column(SmallInteger)  # 0 = Monday, as Python's weekday()
    start_time: Mapped[time] = mapped_column(Time)  # local time in the tenant's timezone
    end_time: Mapped[time] = mapped_column(Time)
    slot_minutes: Mapped[int] = mapped_column(SmallInteger, default=30)


class ConversationStatus(enum.StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"  # waiting on a human decision; the agent does not reply


class Conversation(Base, TenantScoped):
    """One ongoing SMS thread per client, surviving restarts and spanning days."""

    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("tenant_id", "client_id"),)

    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    status: Mapped[ConversationStatus] = mapped_column(
        _enum(ConversationStatus, "conversation_status"), default=ConversationStatus.ACTIVE
    )
    # The model-facing history (Message.to_json() entries), trimmed to a window.
    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)


class MessageDirection(enum.StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class MessageStatus(enum.StrEnum):
    RECEIVED = "received"  # stored, not yet handled
    HELD = "held"  # arrived while the conversation was paused
    HANDLED = "handled"
    SENT = "sent"
    FAILED = "failed"


class SmsMessage(Base, TenantScoped):
    """Every SMS in or out, as the provider saw it."""

    __tablename__ = "sms_messages"
    __table_args__ = (
        # Twilio retries webhooks; its MessageSid makes a retry a no-op.
        Index(
            "uq_sms_messages_tenant_provider_sid",
            "tenant_id",
            "provider_sid",
            unique=True,
            postgresql_where="provider_sid IS NOT NULL",
        ),
        Index("ix_sms_messages_conversation_created", "conversation_id", "created_at"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    direction: Mapped[MessageDirection] = mapped_column(
        _enum(MessageDirection, "message_direction")
    )
    status: Mapped[MessageStatus] = mapped_column(_enum(MessageStatus, "message_status"))
    body: Mapped[str] = mapped_column(String(1600))
    from_number: Mapped[str] = mapped_column(String(20))
    to_number: Mapped[str] = mapped_column(String(20))
    provider_sid: Mapped[str | None] = mapped_column(String(64))


class ReviewKind(enum.StrEnum):
    LATE_CANCELLATION = "late_cancellation"
    MONEY = "money"
    MEDICAL = "medical"
    ESCALATION = "escalation"  # the agent itself asked for a human


class ReviewStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewItem(Base, TenantScoped):
    """Something a person must decide before the client hears back."""

    __tablename__ = "review_items"
    __table_args__ = (Index("ix_review_items_tenant_status", "tenant_id", "status"),)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    kind: Mapped[ReviewKind] = mapped_column(_enum(ReviewKind, "review_kind"))
    status: Mapped[ReviewStatus] = mapped_column(
        _enum(ReviewStatus, "review_status"), default=ReviewStatus.PENDING
    )
    reason: Mapped[str] = mapped_column(String(500))
    client_message: Mapped[str] = mapped_column(String(1600))
    proposed_reply: Mapped[str] = mapped_column(String(1600))
    # The held action, run only on approval: {"type": "cancel_appointment", ...}.
    action: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    final_reply: Mapped[str | None] = mapped_column(String(1600))
    decided_by: Mapped[str | None] = mapped_column(String(120))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OfferStatus(enum.StrEnum):
    OFFERED = "offered"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    TAKEN = "taken"  # someone else accepted first


class SlotOffer(Base, TenantScoped):
    """A freed slot offered to one waitlisted client."""

    __tablename__ = "slot_offers"
    __table_args__ = (
        Index("ix_slot_offers_client_status", "client_id", "status"),
        Index("ix_slot_offers_tenant_slot", "tenant_id", "provider_id", "starts_at"),
    )

    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    waitlist_entry_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("waitlist_entries.id", ondelete="CASCADE")
    )
    provider_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("providers.id", ondelete="CASCADE"))
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rank: Mapped[int] = mapped_column(SmallInteger)
    status: Mapped[OfferStatus] = mapped_column(
        _enum(OfferStatus, "offer_status"), default=OfferStatus.OFFERED
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    responded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ClientResponseStats(Base, TenantScoped):
    """What the bandit has learned about how quickly a client takes offers."""

    __tablename__ = "client_response_stats"
    __table_args__ = (UniqueConstraint("tenant_id", "client_id"),)

    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    offers: Mapped[int] = mapped_column(Integer, default=0)
    reward_sum: Mapped[float] = mapped_column(default=0.0)


class AuditEvent(Base, TenantScoped):
    """Who did what, append-only: every agent tool call and every review
    decision (Module 6).

    Rows are never updated: the repository has no update method and a
    database trigger rejects UPDATE (migration 0004). Raw values are allowed
    here because this is Postgres, not a log (logsetup.py).
    """

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_tenant_seq", "tenant_id", "seq"),)

    # Insertion order. created_at is the transaction's start time (now()), so
    # every event in one agent turn shares it; seq keeps what happened first.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), unique=True)
    kind: Mapped[str] = mapped_column(String(40))  # tool_call | review_decision
    actor: Mapped[str] = mapped_column(String(120))  # "agent", or who decided
    subject: Mapped[str] = mapped_column(String(80))  # tool name, or review kind
    outcome: Mapped[str] = mapped_column(String(40))  # ok | error | rejected | held | ...
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )
    ref_id: Mapped[uuid.UUID | None] = mapped_column()  # e.g. the review item
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
