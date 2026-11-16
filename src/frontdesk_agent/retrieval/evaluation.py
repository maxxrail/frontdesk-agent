"""Retrieval metrics against a labeled golden set.

Labels are evidence *phrases*, not chunk IDs: a chunk is relevant if it
contains the phrase (after whitespace normalisation). That lets one golden
set score every chunking strategy fairly, which chunk IDs could not.
"""

import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

from frontdesk_agent.retrieval.store import PolicyStore, SearchHit, SearchMode

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: str
    tenant: str
    question: str
    evidence: tuple[str, ...]
    answerable: bool


def load_golden(path: Path) -> list[GoldenQuestion]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(
                GoldenQuestion(
                    d["id"], d["tenant"], d["question"], tuple(d["evidence"]), d["answerable"]
                )
            )
    return out


def _relevant(hit: SearchHit, evidence: str) -> bool:
    return normalise(evidence) in normalise(hit.content)


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: GoldenQuestion
    recall_at_k: float  # share of the question's evidence phrases found in the top k
    recall_at_budget: float  # same, but within a fixed number of retrieved words
    reciprocal_rank: float  # 1 / rank of the first relevant chunk (0 if none in top k)
    words_at_k: int  # how much text the top k hands to the model
    top_similarity: float  # best cosine similarity among the hits, for abstention


def _recall(question: GoldenQuestion, hits: list[SearchHit]) -> float:
    if not question.evidence:
        return 0.0
    found = sum(any(_relevant(h, e) for h in hits) for e in question.evidence)
    return found / len(question.evidence)


def within_budget(hits: list[SearchHit], words: int) -> list[SearchHit]:
    """Hits in rank order until ``words`` words of context are used.

    Comparing chunkers at a fixed k flatters big chunks: five 160-word chunks
    hand the model four times the text of five 40-word chunks. A word budget
    compares them at equal prompt cost.
    """
    out: list[SearchHit] = []
    used = 0
    for h in hits:
        if used >= words:
            break
        out.append(h)
        used += len(h.content.split())
    return out


def score(
    question: GoldenQuestion, hits: list[SearchHit], k: int, budget_words: int = 300
) -> QuestionResult:
    top = hits[:k]
    rr = 0.0
    for rank, h in enumerate(top, start=1):
        if any(_relevant(h, e) for e in question.evidence):
            rr = 1.0 / rank
            break
    sims = [h.vector_similarity for h in hits if h.vector_similarity is not None]
    return QuestionResult(
        question,
        recall_at_k=_recall(question, top),
        recall_at_budget=_recall(question, within_budget(hits, budget_words)),
        reciprocal_rank=rr,
        words_at_k=sum(len(h.content.split()) for h in top),
        top_similarity=max(sims, default=0.0),
    )


@dataclass(frozen=True, slots=True)
class RunSummary:
    recall_at_k: float
    recall_at_budget: float
    words_at_k: float
    mrr: float
    hit_at_1: float
    mean_latency_ms: float
    results: list[QuestionResult]


async def evaluate(
    stores: dict[str, PolicyStore],
    golden: list[GoldenQuestion],
    *,
    mode: SearchMode,
    k: int = 5,
    budget_words: int = 300,
) -> RunSummary:
    """Score every answerable question; unanswerable ones are kept for the
    abstention analysis but excluded from recall and MRR."""
    results: list[QuestionResult] = []
    elapsed = 0.0
    for q in golden:
        start = time.perf_counter()
        # Always compute vector similarity, even in keyword mode, so the
        # abstention threshold can be studied on the same footing.
        # Fetch more than k so the word-budget metric has hits to draw on.
        hits = await stores[q.tenant].search(q.question, k=max(k, 20), mode=mode)
        elapsed += time.perf_counter() - start
        if mode == "keyword":
            # Keyword hits carry no similarity; fetch it separately (untimed)
            # so the abstention analysis is comparable across modes.
            vec = await stores[q.tenant].search(q.question, k=1, mode="vector")
            sim = vec[0].vector_similarity if vec else 0.0
            results.append(replace(score(q, hits, k, budget_words), top_similarity=sim or 0.0))
        else:
            results.append(score(q, hits, k, budget_words))
    answerable = [r for r in results if r.question.answerable]
    n = len(answerable) or 1
    return RunSummary(
        recall_at_k=sum(r.recall_at_k for r in answerable) / n,
        recall_at_budget=sum(r.recall_at_budget for r in answerable) / n,
        words_at_k=sum(r.words_at_k for r in answerable) / n,
        mrr=sum(r.reciprocal_rank for r in answerable) / n,
        hit_at_1=sum(r.reciprocal_rank == 1.0 for r in answerable) / n,
        mean_latency_ms=1000 * elapsed / max(len(results), 1),
        results=results,
    )


@dataclass(frozen=True, slots=True)
class Threshold:
    value: float
    answerable_kept: float  # share of answerable questions that pass the gate
    unanswerable_refused: float  # share of unanswerable questions stopped by it

    @property
    def balanced_accuracy(self) -> float:
        return (self.answerable_kept + self.unanswerable_refused) / 2


def sweep_thresholds(results: list[QuestionResult]) -> list[Threshold]:
    """Every cut-off between observed similarities, for the abstention gate."""
    pos = [r.top_similarity for r in results if r.question.answerable]
    neg = [r.top_similarity for r in results if not r.question.answerable]
    if not pos or not neg:
        return []
    out = []
    for value in sorted({round(s, 3) for s in pos + neg}):
        out.append(
            Threshold(
                value,
                sum(s >= value for s in pos) / len(pos),
                sum(s < value for s in neg) / len(neg),
            )
        )
    return out
