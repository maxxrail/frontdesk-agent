"""Measure retrieval quality on the golden set and write a report.

    uv run python evals/retrieval.py                  # full comparison
    uv run python evals/retrieval.py --quick          # chosen config only

Every chunking strategy is ingested into throwaway tenants inside a
transaction that is rolled back, so this never changes stored data and needs
only a migrated database (FDA_DATABASE_URL). Results are written to
evals/reports/retrieval-<date>.md and .json.
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.repositories import TenantDirectory
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.retrieval.chunking import ChunkingConfig
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.evaluation import RunSummary, evaluate, load_golden, sweep_thresholds
from frontdesk_agent.retrieval.ingest import ingest_documents, read_policy_folder
from frontdesk_agent.retrieval.store import PolicyStore, SearchMode

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "golden" / "retrieval.jsonl"
REPORTS = ROOT / "evals" / "reports"
POLICIES = ROOT / "data" / "policies"

CONFIGS = [
    ChunkingConfig("section", 40, 10),
    ChunkingConfig("section", 80, 15),
    ChunkingConfig("section", 160, 30),
    ChunkingConfig("fixed", 40, 10),
    ChunkingConfig("fixed", 80, 15),
    ChunkingConfig("fixed", 160, 30),
]
MODES: list[SearchMode] = ["vector", "keyword", "hybrid"]
K = 5
BUDGET = 300  # words of retrieved context, roughly 400 tokens


async def run(configs: list[ChunkingConfig], modes: list[SearchMode]) -> list[dict[str, object]]:
    golden = load_golden(GOLDEN)
    tenants = sorted({q.tenant for q in golden})
    folders = {t: read_policy_folder(POLICIES / t) for t in tenants}
    embedder = default_embedder()
    engine = make_engine(get_settings().database_url)
    rows: list[dict[str, object]] = []
    try:
        for config in configs:
            async with make_sessionmaker(engine)() as session:
                stores: dict[str, PolicyStore] = {}
                for t in tenants:
                    tenant = await TenantDirectory(session).create(
                        name=f"eval {t}", business_type="eval"
                    )
                    stores[t] = PolicyStore(session, tenant.id, embedder)
                    await ingest_documents(stores[t], folders[t], config)
                chunks = sum([await s.count() for s in stores.values()])
                for mode in modes:
                    summary = await evaluate(stores, golden, mode=mode, k=K, budget_words=BUDGET)
                    rows.append(_row(config, mode, chunks, summary))
                    print(_line(rows[-1]), flush=True)
                await session.rollback()  # leave no trace
    finally:
        await engine.dispose()
    return rows


def _row(config: ChunkingConfig, mode: SearchMode, chunks: int, s: RunSummary) -> dict[str, object]:
    sweep = sweep_thresholds(s.results)
    best = max(sweep, key=lambda t: (t.balanced_accuracy, t.answerable_kept), default=None)
    return {
        "chunker": config.label,
        "mode": mode,
        "chunks": chunks,
        f"recall@{K}": round(s.recall_at_k, 3),
        f"recall@{BUDGET}w": round(s.recall_at_budget, 3),
        f"words@{K}": round(s.words_at_k),
        "mrr": round(s.mrr, 3),
        "hit@1": round(s.hit_at_1, 3),
        "latency_ms": round(s.mean_latency_ms, 1),
        "abstain_threshold": best and best.value,
        "abstain_kept": best and round(best.answerable_kept, 3),
        "abstain_refused": best and round(best.unanswerable_refused, 3),
        "lowest_answerable_similarity": round(
            min(r.top_similarity for r in s.results if r.question.answerable), 3
        ),
        "misses": [
            r.question.id for r in s.results if r.question.answerable and r.recall_at_k < 1.0
        ],
    }


def _line(r: dict[str, object]) -> str:
    return (
        f"{r['chunker']:<14} {r['mode']:<8} chunks={r['chunks']:<4} "
        f"recall@{K}={r[f'recall@{K}']:.3f} recall@{BUDGET}w={r[f'recall@{BUDGET}w']:.3f} "
        f"words@{K}={r[f'words@{K}']:<4} mrr={r['mrr']:.3f} hit@1={r['hit@1']:.3f} "
        f"{r['latency_ms']}ms"
    )


def write_report(rows: list[dict[str, object]]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"retrieval-{date.today():%Y%m%d}"
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    golden = load_golden(GOLDEN)
    answerable = sum(q.answerable for q in golden)
    lines = [
        f"# Retrieval report, {date.today():%Y-%m-%d}",
        "",
        f"Golden set: {len(golden)} questions ({answerable} answerable, "
        f"{len(golden) - answerable} unanswerable), embedder `{default_embedder().name}`, "
        f"k = {K}. Recall and MRR are over answerable questions only.",
        "",
        f"Recall@{BUDGET}w gives every configuration the same {BUDGET} words of retrieved "
        f"context, the fair comparison between chunk sizes. Words@{K} is how much text "
        f"the top {K} hands the model.",
        "",
        f"| Chunker | Mode | Chunks | Recall@{K} | Recall@{BUDGET}w | Words@{K} | MRR | Hit@1 "
        "| ms/query |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['chunker']} | {r['mode']} | {r['chunks']} | {r[f'recall@{K}']:.3f} | "
            f"{r[f'recall@{BUDGET}w']:.3f} | {r[f'words@{K}']} | "
            f"{r['mrr']:.3f} | {r['hit@1']:.3f} | {r['latency_ms']} |"
        )
    lines += [
        "",
        "## Can similarity alone refuse unanswerable questions?",
        "",
        "The best balanced-accuracy cut-off on top vector similarity, per chunker "
        "(similarity does not depend on the search mode). If the answerable share "
        "kept is well below 1, a similarity gate cannot be the refusal mechanism.",
        "",
        "| Chunker | Threshold | Answerable kept | Unanswerable refused | Lowest answerable |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["mode"] != "vector":
            continue
        lines.append(
            f"| {r['chunker']} | {r['abstain_threshold']} | {r['abstain_kept']} | "
            f"{r['abstain_refused']} | {r['lowest_answerable_similarity']} |"
        )
    lines += ["", "## Misses (answerable questions with recall below 1)", ""]
    for r in rows:
        lines.append(f"- {r['chunker']} {r['mode']}: {', '.join(r['misses']) or 'none'}")  # type: ignore[arg-type]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return stem.with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="only the default chunker, hybrid")
    args = parser.parse_args()
    configs = [ChunkingConfig()] if args.quick else CONFIGS
    modes: list[SearchMode] = ["hybrid"] if args.quick else MODES
    rows = asyncio.run(run(configs, modes))
    path = write_report(rows)
    print(f"\nreport: {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
