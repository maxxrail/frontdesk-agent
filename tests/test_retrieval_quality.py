"""Retrieval quality as a regression test.

The golden set is small and the embedder is deterministic, so the measured
numbers are exact. The floors sit a little below today's results
(recall@5 0.95, MRR 0.876 for section:160/30, vector) so a real regression
fails CI while a one-question wobble from a document edit does not.
Run evals/retrieval.py for the full comparison.
"""

import re
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.config import get_settings
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.evaluation import evaluate, load_golden
from frontdesk_agent.retrieval.ingest import ingest_directory
from frontdesk_agent.retrieval.store import PolicyStore
from tests.factories import make_tenant

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = load_golden(ROOT / "evals" / "golden" / "retrieval.jsonl")


def test_golden_set_shape() -> None:
    assert len(GOLDEN) == 60
    assert sum(not q.answerable for q in GOLDEN) == 10
    assert len({q.id for q in GOLDEN}) == 60


def test_every_evidence_phrase_exists_exactly_once_in_its_tenant() -> None:
    """Catches a policy edit that silently invalidates a label."""

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text)

    corpus = {
        t: norm(" ".join(p.read_text() for p in (ROOT / "data" / "policies" / t).glob("*.md")))
        for t in {q.tenant for q in GOLDEN}
    }
    for q in GOLDEN:
        for phrase in q.evidence:
            assert corpus[q.tenant].count(norm(phrase)) == 1, (q.id, phrase)


async def test_retrieval_meets_its_floor(session: AsyncSession) -> None:
    stores = {}
    for slug in sorted({q.tenant for q in GOLDEN}):
        s = PolicyStore(session, (await make_tenant(session, slug)).id, default_embedder())
        await ingest_directory(s, ROOT / "data" / "policies" / slug)
        stores[slug] = s
    summary = await evaluate(stores, GOLDEN, mode=get_settings().retrieval_mode, k=5)
    assert summary.recall_at_k >= 0.90, summary.recall_at_k
    assert summary.mrr >= 0.82, summary.mrr
