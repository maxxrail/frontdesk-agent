"""Eval scores across runs, for charting by prompt version and model.

    uv run python evals/history.py            # prints a table, writes history.csv

Reads every results file in evals/results/ (CI artifacts saved there too) and
writes evals/results/history.csv: one row per run, one column per score.
"""

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
FEATURES = ["booking", "cancel", "policy", "new_client", "risk", "conversation", "redteam"]


def row(path: Path) -> dict[str, Any]:
    r = json.loads(path.read_text(encoding="utf-8"))
    run, s, layers = r["run"], r.get("summary", {}), r.get("layers", {})
    judge, sim = layers.get("judge", {}), layers.get("simulation", {})
    out: dict[str, Any] = {
        "file": path.name,
        "started_at": run.get("started_at"),
        "model": run.get("model"),
        "prompt_version": run.get("prompt_version"),
        "git_sha": str(run.get("git_sha", ""))[:7],
        "pass_rate": s.get("pass_rate"),
    }
    for f in FEATURES:
        out[f] = s.get("by_feature", {}).get(f, {}).get("rate")
    out["policy_answers"] = layers.get("policy_answers", {}).get("headline")
    out["judge_tone"] = judge.get("tone_pass_rate")
    out["judge_faithful"] = judge.get("faithfulness_pass_rate")
    out["simulation"] = sim.get("rate")
    out["cost_usd"] = run.get("spent_usd")
    out["p95_call_ms"] = s.get("p95_call_ms")
    return out


def main() -> None:
    # Dry runs use a canned model; their scores mean nothing.
    rows = sorted(
        (r for r in map(row, RESULTS.glob("*.json")) if "fake" not in str(r["model"])),
        key=lambda r: str(r["started_at"]),
    )
    if not rows:
        print("no results yet")
        return
    with (RESULTS / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = [
        "started_at",
        "model",
        "prompt_version",
        "pass_rate",
        "redteam",
        "judge_tone",
        "judge_faithful",
        "simulation",
        "cost_usd",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
