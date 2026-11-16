"""Measure the keyword risk rules (agent/risk.py) on labelled messages.

    uv run python evals/risk_rules.py

Free and deterministic: no model, no database. Reports, for both the tuning
set (risk.jsonl) and the held-out set (risk_holdout.jsonl), the share of
medical and money messages the rules miss and the share of ordinary messages
they hold by mistake. Rules are tuned on risk.jsonl only, so the held-out
numbers are the honest ones. Writes evals/reports/risk-<date>.md and .json.
"""

import json
from datetime import date
from pathlib import Path

from frontdesk_agent.evaluation.risk_metrics import RiskReport, load_risk_cases, measure

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "tuning": ROOT / "evals" / "golden" / "risk.jsonl",
    "held-out": ROOT / "evals" / "golden" / "risk_holdout.jsonl",
}
REPORTS = ROOT / "evals" / "reports"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def render(reports: dict[str, RiskReport]) -> str:
    lines = [
        f"# Risk rules: {date.today().isoformat()}",
        "",
        "Missed = a risky message the rules did not hold. False positive = an ordinary",
        "message held for review. Held-out numbers are the honest ones (rules are tuned",
        "on the tuning set only).",
        "",
        "| Set | Medical missed | Money missed | False positives |",
        "|---|---|---|---|",
    ]
    for name, r in reports.items():
        med, mon = r.missed.get("medical", []), r.missed.get("money", [])
        lines.append(
            f"| {name} | {len(med)}/{r.counts.get('medical', 0)} "
            f"({_pct(r.miss_rate('medical'))}) | {len(mon)}/{r.counts.get('money', 0)} "
            f"({_pct(r.miss_rate('money'))}) | {len(r.false_positives)}/"
            f"{r.counts.get('none', 0)} ({_pct(r.false_positive_rate)}) |"
        )
    for name, r in reports.items():
        lines += ["", f"## {name}: what the rules got wrong", ""]
        for label, cases in r.missed.items():
            lines += [f'- missed {label}: {c.id} "{c.text}"' for c in cases]
        lines += [f'- held, wrong kind ({got}): {c.id} "{c.text}"' for c, got in r.wrong_kind]
        lines += [f"- false positive on '{m}': {c.id} \"{c.text}\"" for c, m in r.false_positives]
    return "\n".join(lines) + "\n"


def main() -> None:
    reports = {name: measure(load_risk_cases(path)) for name, path in SETS.items()}
    REPORTS.mkdir(exist_ok=True)
    stem = REPORTS / f"risk-{date.today().isoformat()}"
    text = render(reports)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    stem.with_suffix(".json").write_text(
        json.dumps({k: r.as_dict() for k, r in reports.items()}, indent=2), encoding="utf-8"
    )
    print(text.split("\n## ")[0])
    print(f"Written to {stem}.md and .json")


if __name__ == "__main__":
    main()
