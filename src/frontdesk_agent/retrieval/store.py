"""Ingesting documents and searching them, always inside one tenant.

Built on the same tenant-bound repository base as Module 2 (ADR 0005): no
method takes a tenant argument, so one clinic's search can never surface
another clinic's policy.
"""

import hashlib
import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import String, cast, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.models import PolicyChunk, PolicyDocument
from frontdesk_agent.db.repositories import TenantRepository
from frontdesk_agent.retrieval.chunking import ChunkingConfig, chunk_document, parse_markdown
from frontdesk_agent.retrieval.embeddings import Embedder
from frontdesk_agent.tracing import tracer

SearchMode = Literal["vector", "keyword", "hybrid"]

# Reciprocal rank fusion constant. 60 is the value from the original RRF paper
# and is insensitive in practice; it is not tuned on the golden set.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: uuid.UUID
    document_slug: str
    ordinal: int
    heading: str
    content: str
    score: float  # mode-specific: cosine, ts_rank, or fused RRF score
    vector_similarity: float | None = None  # cosine similarity, if computed


@dataclass(frozen=True, slots=True)
class IngestResult:
    slug: str
    chunks: int
    skipped: bool  # unchanged content, chunker and model: nothing re-embedded


class PolicyStore(TenantRepository[PolicyChunk]):
    model = PolicyChunk

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID, embedder: Embedder) -> None:
        super().__init__(session, tenant_id)
        self._embedder = embedder

    # -- ingestion ---------------------------------------------------------

    async def ingest(
        self, *, slug: str, markdown: str, config: ChunkingConfig | None = None
    ) -> IngestResult:
        """Create or replace one document and its chunks. Idempotent."""
        config = config or ChunkingConfig()
        sha = hashlib.sha256(markdown.encode()).hexdigest()

        existing = (
            await self._session.execute(
                select(PolicyDocument).where(
                    PolicyDocument.tenant_id == self._tenant_id, PolicyDocument.slug == slug
                )
            )
        ).scalar_one_or_none()
        if (
            existing is not None
            and existing.content_sha256 == sha
            and existing.chunker == config.label
            and existing.embedding_model == self._embedder.name
        ):
            count = await self._session.scalar(
                select(func.count()).where(PolicyChunk.document_id == existing.id)
            )
            return IngestResult(slug, int(count or 0), skipped=True)

        parsed = parse_markdown(markdown, fallback_title=slug.replace("-", " ").title())
        chunks = chunk_document(parsed, config)
        vectors = self._embedder.embed([c.embedding_text() for c in chunks])

        if existing is None:
            existing = PolicyDocument(tenant_id=self._tenant_id, slug=slug)
            self._session.add(existing)
        else:
            await self._session.execute(
                delete(PolicyChunk).where(PolicyChunk.document_id == existing.id)
            )
        existing.title = parsed.title
        existing.content_sha256 = sha
        existing.chunker = config.label
        existing.embedding_model = self._embedder.name
        await self._session.flush()

        for chunk, vector in zip(chunks, vectors, strict=True):
            self._session.add(
                self._claim(
                    PolicyChunk(
                        document_id=existing.id,
                        ordinal=chunk.ordinal,
                        heading=chunk.heading[:300],
                        content=chunk.content,
                        embedding=vector,
                    )
                )
            )
        await self._session.flush()
        return IngestResult(slug, len(chunks), skipped=False)

    async def remove_missing(self, keep_slugs: set[str]) -> int:
        """Delete documents not in ``keep_slugs`` (a policy was retired)."""
        result = await self._session.execute(
            delete(PolicyDocument)
            .where(PolicyDocument.tenant_id == self._tenant_id)
            .where(PolicyDocument.slug.not_in(keep_slugs))
            .returning(PolicyDocument.id)
        )
        return len(result.all())

    # -- search ------------------------------------------------------------

    async def search(
        self, query: str, *, k: int = 5, mode: SearchMode = "hybrid"
    ) -> list[SearchHit]:
        # The query is client text, so it is not an attribute; counts are.
        with tracer.start_as_current_span("retrieval.search") as span:
            if mode == "vector":
                hits = (await self._vector(query, k))[:k]
            elif mode == "keyword":
                hits = (await self._keyword(query, k))[:k]
            else:
                hits = await self._hybrid(query, k)
            span.set_attributes(
                {"retrieval.mode": mode, "retrieval.k": k, "retrieval.hits": len(hits)}
            )
            return hits

    async def _vector(self, query: str, limit: int) -> list[SearchHit]:
        [qvec] = self._embedder.embed([query])
        if not any(qvec):
            # Cosine distance to a zero vector is NaN in pgvector, and NaN
            # compares false against every threshold. No signal, no results.
            return []
        # A larger candidate list than the HNSW default, because the tenant
        # filter is applied after the index scan; see ADR 0008.
        await self._session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        distance = PolicyChunk.embedding.cosine_distance(qvec)
        stmt = (
            select(PolicyChunk, PolicyDocument.slug, distance.label("distance"))
            .join(PolicyDocument, PolicyDocument.id == PolicyChunk.document_id)
            .where(PolicyChunk.tenant_id == self._tenant_id)
            # Ties broken by position, never by a random UUID: identical
            # inputs must give identical rankings, or metrics wobble run to run.
            .order_by(distance, PolicyDocument.slug, PolicyChunk.ordinal)
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            SearchHit(c.id, slug, c.ordinal, c.heading, c.content, 1 - d, vector_similarity=1 - d)
            for c, slug, d in rows
        ]

    async def _keyword(self, query: str, limit: int) -> list[SearchHit]:
        # plainto_tsquery ANDs every word, which almost never matches a
        # conversational question. Turning the ANDs into ORs and letting
        # ts_rank_cd reward matches that cluster together works far better.
        tsquery = func.to_tsquery(
            "english", func.replace(cast(func.plainto_tsquery("english", query), String), "&", "|")
        )
        rank = func.ts_rank_cd(PolicyChunk.tsv, tsquery)
        stmt = (
            select(PolicyChunk, PolicyDocument.slug, rank.label("rank"))
            .join(PolicyDocument, PolicyDocument.id == PolicyChunk.document_id)
            .where(PolicyChunk.tenant_id == self._tenant_id)
            .where(PolicyChunk.tsv.op("@@")(tsquery))
            .order_by(rank.desc(), PolicyDocument.slug, PolicyChunk.ordinal)
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            SearchHit(c.id, slug, c.ordinal, c.heading, c.content, float(r)) for c, slug, r in rows
        ]

    async def _hybrid(self, query: str, k: int, candidates: int = 20) -> list[SearchHit]:
        vector_hits = await self._vector(query, candidates)
        keyword_hits = await self._keyword(query, candidates)
        fused: dict[uuid.UUID, float] = {}
        by_id: dict[uuid.UUID, SearchHit] = {}
        for hits in (vector_hits, keyword_hits):
            for rank, hit in enumerate(hits, start=1):
                fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                by_id.setdefault(hit.chunk_id, hit)
        similarity = {h.chunk_id: h.vector_similarity for h in vector_hits}
        ranked = sorted(
            fused,
            key=lambda cid: (-fused[cid], by_id[cid].document_slug, by_id[cid].ordinal),
        )[:k]
        return [
            SearchHit(
                cid,
                by_id[cid].document_slug,
                by_id[cid].ordinal,
                by_id[cid].heading,
                by_id[cid].content,
                fused[cid],
                vector_similarity=similarity.get(cid),
            )
            for cid in ranked
        ]
