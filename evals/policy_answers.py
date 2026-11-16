"""End-to-end check of /policy/answer behaviour with the real model.

    uv run python evals/policy_answers.py --dry-run   # free: checks the pipeline, estimates cost
    uv run python evals/policy_answers.py             # real calls, about 60 of them

Runs every golden question through the same PolicyAnswerer the API uses, on
throwaway tenants inside a rolled-back transaction, and reports:

- answered:   answerable questions that got an answer
- grounded:   of those, answers citing a chunk that contains the evidence
- refused:    unanswerable questions correctly refused (the two cross-clinic
              traps included)
- cost, p50 and p95 latency

Needs FDA_ANTHROPIC_API_KEY for a real run. With Haiku 4.5 a full run is
roughly 60 calls x ~900 input tokens: well under $0.10.
"""

import argparse
import asyncio
import json
import statistics
from datetime import date
from decimal import Decimal
from pathlib import Path

from frontdesk_agent.answering.service import PolicyAnswerer
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import LLMCall
from frontdesk_agent.db.repositories import LLMCallRepository, TenantDirectory
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.dependencies import get_llm_client
from frontdesk_agent.llm.base import LLMClient, LLMError
from frontdesk_agent.llm.fake import FakeLLMClient
from frontdesk_agent.llm.pricing import cost_usd
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.evaluation import load_golden, normalise
from frontdesk_agent.retrieval.ingest import ingest_documents, read_policy_folder
from frontdesk_agent.retrieval.store import PolicyStore

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "golden" / "retrieval.jsonl"
REPORTS = ROOT / "evals" / "reports"

BUSINESS_TYPES = {"maple-dental": "dental office", "northside-physio": "physiotherapy clinic"}
DRY_REPLY = json.dumps({"answerable": False, "answer": "", "citations": []})


async def run(dry_run: bool) -> dict[str, object]:
    settings = get_settings()
    golden = load_golden(GOLDEN)
    client: LLMClient = (
        FakeLLMClient([DRY_REPLY] * len(golden), model=settings.llm_model)
        if dry_run
        else get_llm_client()
    )
    embedder = default_embedder()
    answerer = PolicyAnswerer(
        client,
        embedder,
        mode=settings.retrieval_mode,
        k=settings.retrieval_k,
        min_similarity=settings.retrieval_min_similarity,
    )
    engine = make_engine(settings.database_url)
    rows = []
    try:
        async with engine.connect() as conn:
            outer = await conn.begin()
            # Savepoint mode: the answerer's commit() releases a savepoint,
            # and rolling back the outer transaction discards everything.
            session_factory = make_sessionmaker(engine)
            async with session_factory(
                bind=conn, join_transaction_mode="create_savepoint"
            ) as session:
                tenants = {}
                for slug in sorted({q.tenant for q in golden}):
                    t = await TenantDirectory(session).create(
                        name=slug.replace("-", " ").title(),
                        business_type="dental office"
                        if "dental" in slug
                        else "physiotherapy clinic",
                    )
                    await ingest_documents(
                        PolicyStore(session, t.id, embedder),
                        read_policy_folder(ROOT / "data" / "policies" / slug),
                    )
                    tenants[slug] = t.id
                for q in golden:
                    try:
                        result = await answerer.answer(session, tenants[q.tenant], q.question)
                    except LLMError as exc:
                        rows.append({"id": q.id, "error": type(exc).__name__})
                        continue
                    cited = [
                        (await PolicyStore(session, tenants[q.tenant], embedder).get(c.chunk_id))
                        for c in result.citations
                    ]
                    grounded = result.answerable and any(
                        normalise(e) in normalise(chunk.content)
                        for e in q.evidence
                        for chunk in cited
                    )
                    rows.append(
                        {
                            "id": q.id,
                            "tenant": q.tenant,
                            "question": q.question,
                            "answerable_label": q.answerable,
                            "answered": result.answerable,
                            "grounded": grounded,
                            "refusal_reason": result.refusal_reason,
                            "answer": result.answer,
                            "latency_ms": result.latency_ms,
                        }
                    )
                calls: list[LLMCall] = []
                for tid in tenants.values():
                    calls += await LLMCallRepository(session, tid).recent(limit=200)
            await outer.rollback()
    finally:
        await engine.dispose()
    return _summarise(rows, calls, dry_run)


def _summarise(
    rows: list[dict[str, object]], calls: list[LLMCall], dry_run: bool
) -> dict[str, object]:
    ok = [r for r in rows if "error" not in r]
    pos = [r for r in ok if r["answerable_label"]]
    neg = [r for r in ok if not r["answerable_label"]]
    answered = [r for r in pos if r["answered"]]
    lat = sorted(int(c.latency_ms) for c in calls) or [0]
    cost = sum((c.cost_usd or Decimal(0)) for c in calls)
    return {
        "dry_run": dry_run,
        "model": get_settings().llm_model,
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "answered": f"{len(answered)}/{len(pos)}",
        "grounded": f"{sum(bool(r['grounded']) for r in answered)}/{len(answered)}",
        "refused": f"{sum(not r['answered'] for r in neg)}/{len(neg)}",
        "model_calls": len(calls),
        "cost_usd": "n/a (dry run)" if dry_run else str(cost),
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        **({"estimated_real_run_cost_usd": _estimate(calls)} if dry_run else {}),
        "rows": rows,
    }


def _estimate(calls: list[LLMCall]) -> str:
    """The fake counts words, not tokens; English runs ~1.35 tokens per word.
    Assumes ~80 output tokens per answer."""
    tokens_in = int(sum(c.input_tokens for c in calls) * 1.35)
    est = cost_usd(get_settings().llm_model, tokens_in, 80 * len(calls))
    return f"{est} for ~{tokens_in} input tokens" if est is not None else "unknown model price"


def write(summary: dict[str, object]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"policy-answers-{date.today():%Y%m%d}{'-dry' if summary['dry_run'] else ''}"
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    head = {k: v for k, v in summary.items() if k != "rows"}
    lines = [f"# Policy answers, {date.today():%Y-%m-%d}", ""]
    lines += [f"- **{k}**: {v}" for k, v in head.items()]
    misses = [
        r
        for r in summary["rows"]  # type: ignore[attr-defined]
        if "error" in r
        or (r["answerable_label"] and not (r["answered"] and r["grounded"]))
        or (not r["answerable_label"] and r["answered"])
    ]
    lines += ["", "## Questions to look at", ""]
    lines += [
        f"- {r['id']} ({r.get('tenant')}): {r.get('question')} -> "
        f"{r.get('refusal_reason') or r.get('answer') or r.get('error')}"
        for r in misses
    ] or ["- none"]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return path.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="scripted model, no API calls")
    summary = asyncio.run(run(parser.parse_args().dry_run))
    for k, v in summary.items():
        if k != "rows":
            print(f"{k:>22}: {v}")
    print(f"\nreport: {write(summary).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
