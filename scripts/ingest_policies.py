"""Ingest policy documents for every tenant that has a folder.

    uv run python scripts/ingest_policies.py
    uv run python scripts/ingest_policies.py --slug maple-dental

Each tenant's documents live in data/policies/<tenant slug>/*.md. Unchanged
documents are skipped, so this is safe to run on every deploy.
"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import select

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import Tenant
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.ingest import ingest_directory
from frontdesk_agent.retrieval.store import PolicyStore

POLICY_ROOT = Path(__file__).resolve().parent.parent / "data" / "policies"


async def run(only_slug: str | None) -> None:
    engine = make_engine(get_settings().database_url)
    embedder = default_embedder()
    async with make_sessionmaker(engine)() as session:
        stmt = select(Tenant).where(Tenant.slug.is_not(None))
        if only_slug:
            stmt = stmt.where(Tenant.slug == only_slug)
        tenants = list((await session.execute(stmt)).scalars())
        if not tenants:
            raise SystemExit("no matching tenants with a slug; run scripts/seed.py first")
        for tenant in tenants:
            folder = POLICY_ROOT / str(tenant.slug)
            if not folder.is_dir():
                print(f"{tenant.slug}: no folder at {folder}, skipped")
                continue
            results = await ingest_directory(PolicyStore(session, tenant.id, embedder), folder)
            changed = [r for r in results if not r.skipped]
            print(
                f"{tenant.slug}: {len(results)} documents, {len(changed)} re-ingested, "
                f"{sum(r.chunks for r in results)} chunks"
            )
        await session.commit()
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slug", help="only this tenant")
    asyncio.run(run(parser.parse_args().slug))


if __name__ == "__main__":
    main()
