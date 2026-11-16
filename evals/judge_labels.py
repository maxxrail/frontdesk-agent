"""Validate the LLM judge against hand labels.

    uv run python evals/judge_labels.py export evals/results/<run>.json
        Picks 40 agent replies from a real run and writes evals/judge/labels.csv
        with empty "tone" and "faithfulness" columns for a person to fill in.

    uv run python evals/judge_labels.py agree
        Runs the judge on the labelled rows (about $0.20, capped at --budget)
        and reports agreement and Cohen's kappa per dimension to
        evals/judge/agreement.json, for evals/README.md.

Labelling rules are the judge's rubric, prompts/judge/v1.md, word for word:
tone is pass/fail; faithfulness is pass, fail, or n/a when the reply states no
policy facts. Judge verdicts are not in the CSV, so they cannot anchor the
labels.
"""

import argparse
import asyncio
import csv
import json
import random
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from frontdesk_agent.config import get_settings
from frontdesk_agent.evaluation.budget import Budget, BudgetedClient
from frontdesk_agent.evaluation.judge import JudgeItem, Verdict, judge_items

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "evals" / "judge" / "labels.csv"
AGREEMENT = ROOT / "evals" / "judge" / "agreement.json"
COLUMNS = [
    "id",
    "business_type",
    "client_messages",
    "policy_excerpts",
    "reply",
    "tone",
    "faithfulness",
    "notes",
]
SEP = "\n---\n"


def pick(items: list[dict[str, Any]], n: int, seed: int = 5) -> list[dict[str, Any]]:
    """Up to a third from replies the judge failed, so both labels occur; half
    of the rest with policy excerpts, so faithfulness is tested; then random."""
    rng = random.Random(seed)
    rng.shuffle(items)
    failed = [
        i for i in items if "fail" in (i["verdict"].get("tone"), i["verdict"].get("faithfulness"))
    ]
    chosen = failed[: n // 3]
    rest = [i for i in items if i not in chosen]
    with_policy = [i for i in rest if i["excerpts"]][: (n - len(chosen)) // 2]
    chosen += with_policy
    chosen += [i for i in rest if i not in chosen][: n - len(chosen)]
    return sorted(chosen, key=lambda i: i["id"])


def export(results_path: Path, n: int) -> None:
    report = json.loads(results_path.read_text(encoding="utf-8"))
    items = report.get("layers", {}).get("judge", {}).get("items")
    if not items:
        sys.exit(
            "that results file has no judge layer; run evals/run.py with --layers including judge"
        )
    if LABELS.exists():
        sys.exit(f"{LABELS} exists; move it away first so no labels are overwritten")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for i in pick(items, n):
            w.writerow(
                {
                    "id": i["id"],
                    "business_type": i["business_type"],
                    "client_messages": "\n".join(i["client_messages"]),
                    "policy_excerpts": SEP.join(i["excerpts"]) or "(none)",
                    "reply": i["reply"],
                    "tone": "",
                    "faithfulness": "",
                    "notes": "",
                }
            )
    print(f"wrote {n} rows to {LABELS}. Fill in tone (pass/fail) and faithfulness (pass/fail/n/a).")


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa: agreement corrected for what chance alone would give."""
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(
        (sum(a == k for a, _ in pairs) / n) * (sum(b == k for _, b in pairs) / n) for k in labels
    )
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 3)


def read_labels() -> list[dict[str, str]]:
    with LABELS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [
        r["id"]
        for r in rows
        if r["tone"].strip().lower() not in ("pass", "fail")
        or r["faithfulness"].strip().lower() not in ("pass", "fail", "n/a")
    ]
    if bad:
        sys.exit(f"rows without a valid label: {', '.join(bad)}")
    return rows


async def agree(budget_usd: str, model: str) -> dict[str, Any]:
    from frontdesk_agent.llm.anthropic_client import AnthropicLLMClient

    rows = read_labels()
    settings = get_settings()
    if not settings.anthropic_api_key:
        sys.exit("FDA_ANTHROPIC_API_KEY is not set")
    budget = Budget(Decimal(budget_usd))
    llm = BudgetedClient(AnthropicLLMClient(settings.anthropic_api_key, model=model), budget)
    items = [
        JudgeItem(
            r["id"],
            r["business_type"],
            r["client_messages"].splitlines(),
            [] if r["policy_excerpts"] == "(none)" else r["policy_excerpts"].split(SEP),
            r["reply"],
        )
        for r in rows
    ]
    verdicts = await judge_items(llm, items)
    out: dict[str, Any] = {
        "judged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge_model": model,
        "rows": len(rows),
        "cost_usd": str(budget.spent_usd),
    }
    disagreements = []
    for field in ("tone", "faithfulness"):
        pairs = []
        for r, v in zip(rows, verdicts, strict=True):
            if isinstance(v, Verdict):
                human, judge = r[field].strip().lower(), getattr(v, field)
                pairs.append((human, judge))
                if human != judge:
                    disagreements.append(
                        {
                            "id": r["id"],
                            "field": field,
                            "human": human,
                            "judge": judge,
                            "judge_reason": getattr(v, f"{field}_reason"),
                        }
                    )
        out[field] = {
            "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
            "kappa": kappa(pairs),
            "n": len(pairs),
        }
    out["unjudged"] = sum(not isinstance(v, Verdict) for v in verdicts)
    out["disagreements"] = disagreements
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("results", type=Path)
    e.add_argument("-n", type=int, default=40)
    a = sub.add_parser("agree")
    a.add_argument("--budget", default="0.50")
    a.add_argument("--judge-model", default="claude-sonnet-5-5")
    args = p.parse_args()
    if args.cmd == "export":
        export(args.results, args.n)
        return
    out = asyncio.run(agree(args.budget, args.judge_model))
    AGREEMENT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for field in ("tone", "faithfulness"):
        d = out[field]
        print(f"{field:>13}: agreement {d['agreement']}, kappa {d['kappa']} (n={d['n']})")
    print(f"written to {AGREEMENT}")
    print(f"COST OF THIS RUN: ${out['cost_usd']}")


if __name__ == "__main__":
    main()
