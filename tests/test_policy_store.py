"""Ingestion and search against real Postgres + pgvector."""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import PolicyChunk, Tenant
from frontdesk_agent.retrieval.chunking import ChunkingConfig
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.ingest import ingest_directory
from frontdesk_agent.retrieval.store import PolicyStore, SearchMode
from tests.factories import make_tenant

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
FEE = "# Cancellations\n\n## Late cancellation fee\n\nA $50 fee applies with under 24 hours notice."


def store(session: AsyncSession, tenant: Tenant) -> PolicyStore:
    return PolicyStore(session, tenant.id, default_embedder())


async def _chunk_count(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(PolicyChunk)) or 0)


async def test_ingest_is_idempotent(session: AsyncSession) -> None:
    s = store(session, await make_tenant(session))
    first = await s.ingest(slug="cancellation", markdown=FEE)
    again = await s.ingest(slug="cancellation", markdown=FEE)
    assert not first.skipped and again.skipped
    assert await _chunk_count(session) == first.chunks == 1


async def test_changed_text_or_chunker_replaces_chunks(session: AsyncSession) -> None:
    s = store(session, await make_tenant(session))
    await s.ingest(slug="c", markdown=FEE)
    changed = await s.ingest(slug="c", markdown=FEE.replace("$50", "$60"))
    assert not changed.skipped
    hits = await s.search("late fee", k=5)
    assert ["$60" in h.content for h in hits] == [True]
    rechunked = await s.ingest(
        slug="c", markdown=FEE.replace("$50", "$60"), config=ChunkingConfig("fixed", 40, 5)
    )
    assert not rechunked.skipped and await _chunk_count(session) == 1


async def test_directory_ingest_prunes_deleted_documents(
    session: AsyncSession, tmp_path: Path
) -> None:
    s = store(session, await make_tenant(session))
    (tmp_path / "a.md").write_text(FEE)
    (tmp_path / "b.md").write_text("# B\n\n## Hours\n\nOpen Monday.")
    await ingest_directory(s, tmp_path)
    (tmp_path / "b.md").unlink()
    await ingest_directory(s, tmp_path)
    assert {h.document_slug for h in await s.search("Monday hours fee", k=10)} == {"a"}


async def test_keyword_column_is_generated_by_postgres(session: AsyncSession) -> None:
    s = store(session, await make_tenant(session))
    await s.ingest(slug="c", markdown=FEE)
    [hit] = await s.search("cancellation", mode="keyword")
    assert hit.document_slug == "c" and hit.score > 0


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
async def test_search_never_crosses_tenants(session: AsyncSession, mode: SearchMode) -> None:
    """Both clinics have a cancellation policy with different numbers."""
    dental, physio = await make_tenant(session, "Dental"), await make_tenant(session, "Physio")
    await ingest_directory(store(session, dental), POLICIES / "maple-dental")
    await ingest_directory(store(session, physio), POLICIES / "northside-physio")

    hits = await store(session, physio).search("how much notice to cancel", k=20, mode=mode)
    assert hits and all("$50" not in h.content and "24 hours" not in h.content for h in hits)
    whitening = await store(session, physio).search("teeth whitening price", k=20, mode=mode)
    assert all("whitening" not in h.document_slug for h in whitening)


async def test_same_slug_in_two_tenants_is_allowed(session: AsyncSession) -> None:
    for name in ("A", "B"):
        await store(session, await make_tenant(session, name)).ingest(slug="faq", markdown=FEE)
    assert await _chunk_count(session) == 2


async def test_blank_query_returns_nothing_instead_of_nan(session: AsyncSession) -> None:
    s = store(session, await make_tenant(session))
    await s.ingest(slug="c", markdown=FEE)
    assert await s.search("", mode="vector") == []


async def test_rankings_are_deterministic(session: AsyncSession) -> None:
    s = store(session, await make_tenant(session))
    await ingest_directory(s, POLICIES / "maple-dental")
    runs = [[h.chunk_id for h in await s.search("fee", k=10, mode="hybrid")] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
