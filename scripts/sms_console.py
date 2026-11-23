"""Text the front desk from your terminal, no Twilio needed.

    uv run python scripts/sms_console.py --clinic maple-dental
    uv run python scripts/sms_console.py --clinic northside-physio --phone +16475550120

Your lines go through the same path as the Twilio webhook (stored, then the
same job the worker runs), and replies print instead of sending. Needs a
seeded database and FDA_ANTHROPIC_API_KEY. Each line is a real model call:
a few cents for a whole session on Haiku.

Commands:
    /reviews            list pending review items
    /approve N [text]   approve item N, optionally with your own reply
    /reject N [text]    reject item N, optionally sending text
    /quit
"""

import argparse
import asyncio
import uuid

from frontdesk_agent import aio
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import ReviewStatus
from frontdesk_agent.db.repositories import ReviewRepository, TenantDirectory
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.dependencies import get_llm_client, get_lookup_llm_client
from frontdesk_agent.frontdesk.ports import Effects
from frontdesk_agent.frontdesk.service import accept_inbound
from frontdesk_agent.services import InlineQueue, Services, build_services, run_job


async def main(clinic: str, phone: str) -> None:
    settings = get_settings().model_copy(update={"sms_sender": "console", "job_queue": "inline"})
    engine = make_engine(settings.database_url)
    sessionmaker = make_sessionmaker(engine)
    services: Services = build_services(
        settings, sessionmaker, get_llm_client(), lookup_llm=get_lookup_llm_client()
    )
    async with sessionmaker() as session:
        tenant = await TenantDirectory(session).by_slug(clinic)
        if tenant is None or not tenant.sms_number:
            raise SystemExit(f"no tenant '{clinic}' with an SMS number; run scripts/seed.py")
        tenant_id, number = tenant.id, tenant.sms_number
    print(f"Texting {clinic} ({number}) as {phone}. /quit to stop.\n")

    pending: list[uuid.UUID] = []
    while True:
        try:
            line = (await asyncio.to_thread(input, "you> ")).strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reviews":
            async with sessionmaker() as session:
                items = await ReviewRepository(session, tenant_id).by_status(ReviewStatus.PENDING)
            pending = [i.id for i in items]
            for n, i in enumerate(items, 1):
                print(f"  {n}. [{i.kind.value}] {i.reason}\n     draft: {i.proposed_reply!r}")
            if not items:
                print("  (none pending)")
            continue
        if line.startswith(("/approve", "/reject")):
            cmd, _, rest = line.partition(" ")
            num, _, text = rest.partition(" ")
            if not num.isdigit() or not 0 < int(num) <= len(pending):
                print("  run /reviews first, then /approve N or /reject N")
                continue
            effects = Effects()
            async with sessionmaker() as session:
                await services.frontdesk.decide(
                    session,
                    tenant_id,
                    pending[int(num) - 1],
                    effects,
                    approve=cmd == "/approve",
                    reply=text or None,
                    reviewer="console",
                )
                await session.commit()
                await effects.flush(session, services.sender, services.queue)
            await asyncio.sleep(0.1)
            continue

        async with sessionmaker() as session:
            tenant = await TenantDirectory(session).get(tenant_id)
            msg = await accept_inbound(
                session, tenant, from_=phone, to=number, body=line, sid=f"console-{uuid.uuid4()}"
            )
            await session.commit()
        assert msg is not None
        outcome = await run_job(
            services, "handle_inbound", tenant_id=str(tenant_id), message_id=str(msg.id)
        )
        if outcome != "agent_reply":
            print(f"  ({outcome})")
        await asyncio.sleep(0.1)  # let inline follow-up jobs (slot fills) print
    assert isinstance(services.queue, InlineQueue)
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clinic", default="maple-dental")
    parser.add_argument("--phone", default="+14165550110", help="defaults to a seeded client")
    args = parser.parse_args()
    aio.run(main(args.clinic, args.phone))
