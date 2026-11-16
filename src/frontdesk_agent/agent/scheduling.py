"""Open slots, computed from provider schedules minus booked appointments.

Schedules are local wall-clock times in the tenant's timezone; everything
stored and compared is timezone-aware UTC. DST is handled by zoneinfo.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import Provider, Tenant
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ProviderRepository,
    ScheduleRepository,
)

# No bookings closer than this to "now": the clinic needs time to prepare.
MIN_LEAD = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class Slot:
    provider_id: uuid.UUID
    provider_name: str
    starts_at: datetime  # aware, UTC
    ends_at: datetime

    def local(self, tz: ZoneInfo) -> str:
        return self.starts_at.astimezone(tz).isoformat(timespec="minutes")


async def free_slots(
    session: AsyncSession,
    tenant: Tenant,
    day: date,
    *,
    now: datetime,
    provider_id: uuid.UUID | None = None,
) -> list[Slot]:
    tz = ZoneInfo(tenant.timezone)
    providers = {
        p.id: p
        for p in await ProviderRepository(session, tenant.id).list(limit=200)
        if p.active and (provider_id is None or p.id == provider_id)
    }
    blocks = [
        b
        for b in await ScheduleRepository(session, tenant.id).for_weekday(day.weekday())
        if b.provider_id in providers
    ]
    if not blocks:
        return []

    day_start = datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(UTC)
    booked = await AppointmentRepository(session, tenant.id).booked_between(
        day_start, day_start + timedelta(days=1, hours=1)
    )
    busy: dict[uuid.UUID, list[tuple[datetime, datetime]]] = {}
    for a in booked:
        busy.setdefault(a.provider_id, []).append((a.starts_at, a.ends_at))

    earliest = now + MIN_LEAD
    slots: list[Slot] = []
    for b in blocks:
        start = datetime.combine(day, b.start_time, tzinfo=tz)
        end = datetime.combine(day, b.end_time, tzinfo=tz)
        step = timedelta(minutes=b.slot_minutes)
        while start + step <= end:
            s_utc, e_utc = start.astimezone(UTC), (start + step).astimezone(UTC)
            clash = any(s_utc < be and e_utc > bs for bs, be in busy.get(b.provider_id, []))
            if s_utc >= earliest and not clash:
                slots.append(Slot(b.provider_id, providers[b.provider_id].name, s_utc, e_utc))
            start += step
    return sorted(slots, key=lambda s: (s.starts_at, s.provider_name))


async def find_slot(
    session: AsyncSession,
    tenant: Tenant,
    provider_id: uuid.UUID,
    starts_at: datetime,
    *,
    now: datetime,
) -> Slot | None:
    """The open slot starting exactly then, or None. The model cannot book a
    time that is not on the schedule, already taken, or in the past."""
    tz = ZoneInfo(tenant.timezone)
    local_day = starts_at.astimezone(tz).date()
    for slot in await free_slots(session, tenant, local_day, now=now, provider_id=provider_id):
        if slot.starts_at == starts_at.astimezone(UTC):
            return slot
    return None


def provider_label(p: Provider) -> str:
    return f"{p.name} ({p.title})" if p.title else p.name
