"""Seed two synthetic clinics for local development and demos.

    uv run python scripts/seed.py            # 40 clients per clinic
    uv run python scripts/seed.py --reset    # wipe all tenant data first

Everything is fake (Faker, fixed seed), so the output is identical on every
machine. Phone numbers use the 555-01xx range reserved for fiction.
"""

import argparse
import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from faker import Faker
from sqlalchemy import text

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import Client, Provider, ProviderSchedule, WaitlistEntry
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    ProviderRepository,
    ScheduleRepository,
    TenantDirectory,
    WaitlistRepository,
)
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.ingest import ingest_directory
from frontdesk_agent.retrieval.store import PolicyStore

POLICY_ROOT = Path(__file__).resolve().parent.parent / "data" / "policies"


@dataclass(frozen=True)
class Clinic:
    name: str
    slug: str
    business_type: str
    sms_number: str
    area_code: str
    providers: tuple[tuple[str, str], ...]
    # weekday (0 = Monday) -> (open, close), matching the clinic's hours document
    hours: dict[int, tuple[str, str]]


CLINICS = (
    Clinic(
        "Maple Dental",
        "maple-dental",
        "dental office",
        "+14165550100",
        "416",
        (("Dr. Amara Osei", "Dentist"), ("Lena Park", "Hygienist")),
        {
            0: ("08:00", "19:00"),
            1: ("08:00", "19:00"),
            2: ("08:00", "19:00"),
            3: ("08:00", "19:00"),
            4: ("08:00", "16:00"),
        },
    ),
    Clinic(
        "Northside Physio",
        "northside-physio",
        "physiotherapy clinic",
        "+16475550100",
        "647",
        (("Sam Ritter", "Physiotherapist"), ("Priya Nair", "Physiotherapist")),
        {
            0: ("07:00", "19:00"),
            1: ("10:00", "20:00"),
            2: ("07:00", "19:00"),
            3: ("10:00", "20:00"),
            4: ("07:00", "19:00"),
            5: ("08:00", "13:00"),
        },
    ),
)
MAX_CLIENTS = 89  # 555-0110..0198 stays inside the range reserved for fiction


def fake_phone(area_code: str, i: int) -> str:
    """NANP 555-0100..0199 numbers are reserved for fictional use."""
    return f"+1{area_code}555{110 + i:04d}"


async def seed(clients_per_tenant: int, reset: bool, sms_number: str | None = None) -> None:
    fake = Faker("en_CA")
    Faker.seed(579)
    rng = random.Random(579)

    engine = make_engine(get_settings().database_url)
    sessionmaker = make_sessionmaker(engine)
    embedder = default_embedder()
    async with sessionmaker() as session:
        if reset:
            await session.execute(text("TRUNCATE tenants CASCADE"))

        tomorrow = datetime.now(UTC).replace(hour=14, minute=0, second=0, microsecond=0)
        tomorrow += timedelta(days=1)

        for clinic in CLINICS:
            tenant = await TenantDirectory(session).create(
                name=clinic.name,
                slug=clinic.slug,
                business_type=clinic.business_type,
                # --sms-number points the first clinic at your real Twilio number.
                sms_number=sms_number if sms_number and clinic is CLINICS[0] else clinic.sms_number,
            )
            providers = ProviderRepository(session, tenant.id)
            clients = ClientRepository(session, tenant.id)
            appts = AppointmentRepository(session, tenant.id)
            waitlist = WaitlistRepository(session, tenant.id)

            staff = [
                await providers.add(Provider(name=name, title=title))
                for name, title in clinic.providers
            ]
            schedules = ScheduleRepository(session, tenant.id)
            for provider in staff:
                for weekday, (opens, closes) in clinic.hours.items():
                    await schedules.add(
                        ProviderSchedule(
                            provider_id=provider.id,
                            weekday=weekday,
                            start_time=time.fromisoformat(opens),
                            end_time=time.fromisoformat(closes),
                            slot_minutes=30,
                        )
                    )
            people = [
                await clients.add(
                    Client(
                        first_name=fake.first_name(),
                        last_name=fake.last_name(),
                        phone=fake_phone(clinic.area_code, i),
                        email=fake.safe_email(),
                    )
                )
                for i in range(clients_per_tenant)
            ]

            # A week of half-hour appointments, roughly a third of clients booked.
            for i, person in enumerate(rng.sample(people, k=len(people) // 3)):
                start = tomorrow + timedelta(days=i % 5, minutes=30 * (i // 5))
                await appts.book(
                    client_id=person.id,
                    provider_id=rng.choice(staff).id,
                    starts_at=start,
                    ends_at=start + timedelta(minutes=30),
                )
            for person in rng.sample(people, k=3):
                await waitlist.add(
                    WaitlistEntry(
                        client_id=person.id,
                        earliest=tomorrow,
                        latest=tomorrow + timedelta(days=7),
                    )
                )
            policies = await ingest_directory(
                PolicyStore(session, tenant.id, embedder), POLICY_ROOT / clinic.slug
            )
            print(
                f"seeded {clinic.name}: tenant_id={tenant.id}, "
                f"{len(policies)} policy documents, {sum(p.chunks for p in policies)} chunks"
            )

        await session.commit()
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", type=int, default=40, help="clients per clinic")
    parser.add_argument("--reset", action="store_true", help="delete all tenant data first")
    parser.add_argument(
        "--sms-number", help="your Twilio number (E.164) for Maple Dental, to text it for real"
    )
    args = parser.parse_args()
    if not 1 <= args.clients <= MAX_CLIENTS:
        parser.error(f"--clients must be between 1 and {MAX_CLIENTS}")
    asyncio.run(seed(args.clients, args.reset, args.sms_number))


if __name__ == "__main__":
    main()
