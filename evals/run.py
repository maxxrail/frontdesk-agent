"""Run the eval suite and write a results file.

    uv run python evals/run.py --dry-run          # free: canned model, checks the pipeline
    uv run python evals/run.py                    # real model, capped at --budget (default $3)
    uv run python evals/run.py --model claude-haiku-5-5 --prompt-version v2
    uv run python evals/run.py --only redteam,risk --layers agent

Layers (--layers, comma-separated; default all):
  agent      the cases in evals/cases/*.jsonl, through the real front desk,
             scored by deterministic checks (layer 1)
  policy     the Module 3 policy-answer golden set (evals/golden/retrieval.jsonl)
  judge      an LLM judge scores tone and faithfulness of agent replies (layer 2)
  sim        scripted multi-turn conversations with a simulated client (layer 3)

Needs Postgres. The suite uses its own database (frontdesk_eval on the server
in FDA_DATABASE_URL, or FDA_EVAL_DATABASE_URL) and wipes it at the start.
Writes evals/results/<timestamp>-<model>-<prompt>.json; with --markdown, also
the PR score table. Exits 1 if --gate is set and the run fails its thresholds.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from frontdesk_agent import aio
from frontdesk_agent.config import get_settings
from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.evaluation import database
from frontdesk_agent.evaluation.budget import Budget, BudgetedClient
from frontdesk_agent.evaluation.cases import Case, load_cases
from frontdesk_agent.evaluation.checks import check_case
from frontdesk_agent.evaluation.results import (
    CaseResult,
    gate,
    load_json,
    render_markdown,
    summarise,
)
from frontdesk_agent.evaluation.runner import CaseRun, EvalContext, run_cases
from frontdesk_agent.llm.base import LLMClient, LLMCompletion
from frontdesk_agent.llm.messages import AgentTurn, Message, TextBlock, ToolSpec
from frontdesk_agent.retrieval.embeddings import default_embedder

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases"
RESULTS = ROOT / "evals" / "results"
LAYERS = ("agent", "policy", "judge", "sim")


class CannedClient:
    """For --dry-run: answers every agent turn with one fixed reply, every
    structured call with a refusal. Exercises the pipeline for free."""

    model = "fake-model"

    async def complete(
        self,
        *,
        system: str,
        user: str,
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> LLMCompletion:
        text = json.dumps({"answerable": False, "answer": "", "citations": []})
        return LLMCompletion(text, self.model, len(user) // 4, 20, 1, prompt_name, prompt_version)

    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> AgentTurn:
        return AgentTurn(
            [TextBlock("Thanks for your message. A team member will follow up.")],
            "end_turn",
            self.model,
            500,
            15,
            1,
            prompt_name,
            prompt_version,
        )


def real_client(model: str) -> LLMClient:
    from frontdesk_agent.llm.anthropic_client import AnthropicLLMClient

    settings = get_settings()
    if not settings.anthropic_api_key:
        sys.exit("FDA_ANTHROPIC_API_KEY is not set (in .env or the environment).")
    return AnthropicLLMClient(
        settings.anthropic_api_key, model=model, timeout_s=settings.llm_timeout_s
    )


def git_sha() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def select(cases: list[Case], only: str | None) -> list[Case]:
    if not only:
        return cases
    wanted = {w.strip() for w in only.split(",")}
    return [c for c in cases if c.id in wanted or c.feature in wanted or wanted & set(c.tags)]


async def run_agent_layer(
    ctx: EvalContext, cases: list[Case], concurrency: int
) -> tuple[list[CaseResult], str | None]:
    done = 0

    def progress(case: Case, run: CaseRun) -> None:
        nonlocal done
        done += 1
        ok = all(c.passed for c in check_case(case, run))
        print(f"  [{done:>3}/{len(cases)}] {'pass' if ok else 'FAIL'}  {case.id}", flush=True)

    runs, stopped = await run_cases(ctx, cases, concurrency=concurrency, on_done=progress)
    return [CaseResult(c, r, check_case(c, r)) for c, r in zip(cases, runs, strict=True)], stopped


async def run_policy_layer(llm: LLMClient, url: str, dry_run: bool) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "evals"))
    from policy_answers import run as run_policy

    s = await run_policy(dry_run, client=llm, database_url=url)
    return {
        "headline": f"answered {s['answered']}, grounded {s['grounded']}, refused {s['refused']}",
        "answered": s["answered"],
        "grounded": s["grounded"],
        "refused": s["refused"],
        "errors": s["errors"],
        "rows": s["rows"],
    }


async def main_async(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[CaseResult], Budget]:
    settings = get_settings()
    model = args.model or settings.llm_model
    prompt_version = args.prompt_version or settings.agent_prompt_version
    layers = [x.strip() for x in args.layers.split(",")]
    if unknown := set(layers) - set(LAYERS):
        sys.exit(f"unknown layers: {sorted(unknown)}")
    cases = select(load_cases(CASES), args.only)

    budget = Budget(Decimal(args.budget))

    def client(for_model: str) -> BudgetedClient:
        inner: LLMClient = CannedClient() if args.dry_run else real_client(for_model)
        return BudgetedClient(inner, budget, allow_unpriced=args.dry_run)

    llm = client(model)
    url = os.environ.get("FDA_EVAL_DATABASE_URL") or database.eval_url(settings.database_url)
    await database.prepare(url)
    engine = make_engine(url)
    ctx = EvalContext(make_sessionmaker(engine), llm, default_embedder(), prompt_version)

    started = time.perf_counter()
    report: dict[str, Any] = {
        "run": {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": "fake-model (dry run)" if args.dry_run else model,
            "prompt_version": prompt_version,
            "git_sha": git_sha(),
            "layers": layers,
            "budget_usd": str(args.budget),
        },
        "layers": {},
    }
    results: list[CaseResult] = []
    stopped: str | None = None
    try:
        if "agent" in layers:
            print(f"agent cases: {len(cases)} (model {model}, prompt agent/{prompt_version})")
            results, stopped = await run_agent_layer(ctx, cases, args.concurrency)
        if "policy" in layers and not stopped:
            print("policy answers: 60 golden questions")
            report["layers"]["policy_answers"] = await run_policy_layer(llm, url, args.dry_run)
        if "judge" in layers and not stopped:
            from frontdesk_agent.evaluation.judge import judge_results

            report["layers"]["judge"] = await judge_results(
                client(args.judge_model), results, dry_run=args.dry_run
            )
        if "sim" in layers and not stopped:
            from frontdesk_agent.evaluation.simulation import run_simulations

            print("simulated conversations")
            report["layers"]["simulation"] = await run_simulations(
                ctx, args.concurrency, sim_llm=client(args.sim_model)
            )
    finally:
        await engine.dispose()

    report["run"].update(
        {
            "spent_usd": str(budget.spent_usd.quantize(Decimal("0.0001"))),
            "spent_by_prompt": {k: str(v) for k, v in budget.by_prompt.items()},
            "stopped": stopped,
            "seconds": round(time.perf_counter() - started),
        }
    )
    return report, results, budget


def finish(
    args: argparse.Namespace, report: dict[str, Any], results: list[CaseResult], budget: Budget
) -> int:
    """Score, gate, write the results file and the PR table, print the cost."""
    run = report["run"]

    summary = summarise(results)
    report["summary"] = summary
    report["cases"] = [r.as_dict() for r in results]

    baseline = load_json(Path(args.baseline)) if args.baseline else None
    thresholds = load_json(Path(args.thresholds)) or {}
    g = gate(summary, results, thresholds, baseline)
    report["gate"] = {
        "ok": g.ok,
        "failures": g.failures,
        "regressions": g.regressions,
        "fixed": g.fixed,
    }

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    tag = "dry" if args.dry_run else str(run["model"]).removeprefix("claude-")
    out = Path(args.out) if args.out else RESULTS / f"{stamp}-{tag}-{run['prompt_version']}.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    md = render_markdown(report, g, baseline)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")

    print()
    print(md)
    print(f"results: {out}")
    # CLAUDE.md: print the cost of every paid run.
    print(f"COST OF THIS RUN: ${run['spent_usd']} ({budget.calls} model calls, cap ${args.budget})")
    return 0 if (g.ok or not args.gate) else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="canned model, no API calls, free")
    p.add_argument("--model", help="override FDA_LLM_MODEL for the agent")
    p.add_argument("--judge-model", default="claude-sonnet-5-5")
    p.add_argument("--sim-model", default="claude-haiku-5-5", help="plays the client")
    p.add_argument("--prompt-version", help="override FDA_AGENT_PROMPT_VERSION")
    p.add_argument("--budget", default="3.00", help="hard cap in USD for this run")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--only", help="comma-separated case ids, features or tags")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--baseline", default=str(ROOT / "evals" / "baseline.json"))
    p.add_argument("--thresholds", default=str(ROOT / "evals" / "thresholds.json"))
    p.add_argument("--markdown", help="also write the PR score table here")
    p.add_argument("--out", help="results file path")
    p.add_argument("--gate", action="store_true", help="exit 1 if the run fails its thresholds")
    args = p.parse_args()
    report, results, budget = aio.run(main_async(args))
    sys.exit(finish(args, report, results, budget))


if __name__ == "__main__":
    main()
