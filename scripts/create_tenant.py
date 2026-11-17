"""Create a tenant and its first admin (ADR 0016).

    uv run python scripts/create_tenant.py --name "Maple Dental" \\
        --business-type "dental office" --sms-number +14165550100 \\
        --admin-email owner@maple.example

Prints the tenant id, the admin's generated password and an admin API key.
Both secrets are shown once and cannot be retrieved later. Tenants are not
created over HTTP: no tenant's credential should be able to create another.
"""

import argparse

from pydantic import ValidationError

from frontdesk_agent import aio
from frontdesk_agent.api.schemas import TenantIn
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.repositories import TenantDirectory
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.onboarding import first_admin


async def create(tenant: TenantIn, admin_email: str) -> None:
    engine = make_engine(get_settings().database_url)
    async with make_sessionmaker(engine)() as session:
        row = await TenantDirectory(session).create(**tenant.model_dump())
        access = await first_admin(session, row.id, admin_email)
        await session.commit()
    await engine.dispose()
    print(f"tenant {row.name}: id={row.id}")
    print(f"admin login: {access.email}  password: {access.password}  (shown once)")
    print(f"admin API key: {access.api_key}  (shown once)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--business-type", required=True)
    p.add_argument("--sms-number", help="the tenant's texting number, E.164")
    p.add_argument("--slug")
    p.add_argument("--timezone", default="America/Toronto")
    p.add_argument("--admin-email", required=True)
    args = p.parse_args()
    try:
        tenant = TenantIn(
            name=args.name,
            business_type=args.business_type,
            sms_number=args.sms_number,
            slug=args.slug,
            timezone=args.timezone,
        )
    except ValidationError as exc:
        p.error(str(exc))
    aio.run(create(tenant, args.admin_email))


if __name__ == "__main__":
    main()
