"""Scoring a run, gating it, and comparing it with the baseline.

A run's results file is self-contained: the settings it ran with, every
case's checks, and the totals. evals/history.py charts them across prompt
versions and models; CI compares a PR's run with evals/baseline.json, the
last accepted run on main, to list the cases that regressed.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from frontdesk_agent.evaluation.cases import Case
from frontdesk_agent.evaluation.checks import Check
from frontdesk_agent.evaluation.runner import CaseRun


@dataclass
class CaseResult:
    case: Case
    run: CaseRun
    checks: list[Check]

    @property
    def skipped(self) -> bool:
        return (self.run.error or "").startswith("skipped")

    @property
    def passed(self) -> bool:
        return not self.skipped and all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "feature": self.case.feature,
            "difficulty": self.case.difficulty,
            "tags": list(self.case.tags),
            "passed": self.passed,
            "skipped": self.skipped,
            "failed_checks": [f"{c.name}: {c.detail}" for c in self.checks if not c.passed],
            "run": self.run.as_dict(),
        }


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 3) if total else 0.0


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    ran = [r for r in results if not r.skipped]
    by: dict[str, dict[str, list[bool]]] = {
        "feature": defaultdict(list),
        "difficulty": defaultdict(list),
        "check": defaultdict(list),
        "tag": defaultdict(list),
    }
    for r in ran:
        by["feature"][r.case.feature].append(r.passed)
        by["difficulty"][r.case.difficulty].append(r.passed)
        for t in r.case.tags:
            by["tag"][t].append(r.passed)
        for c in r.checks:
            by["check"][c.name].append(c.passed)
    latencies = sorted(ms for r in ran for ms in r.run.latencies_ms)
    cost = sum((r.run.cost_usd for r in ran), Decimal(0))
    return {
        "cases": len(results),
        "ran": len(ran),
        "skipped": len(results) - len(ran),
        "passed": sum(r.passed for r in ran),
        "pass_rate": _rate(sum(r.passed for r in ran), len(ran)),
        **{
            f"by_{k}": {
                name: {"passed": sum(v), "total": len(v), "rate": _rate(sum(v), len(v))}
                for name, v in sorted(groups.items())
            }
            for k, groups in by.items()
        },
        "agent_cost_usd": str(cost),
        "agent_cost_per_case_usd": str((cost / len(ran)).quantize(Decimal("0.0001")))
        if ran
        else "0",
        "model_calls": sum(r.run.model_calls for r in ran),
        "p50_call_ms": statistics.median(latencies) if latencies else 0,
        "p95_call_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0,
    }


@dataclass
class Gate:
    failures: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)  # passed in baseline, fail now
    fixed: list[str] = field(default_factory=list)  # failed in baseline, pass now

    @property
    def ok(self) -> bool:
        return not self.failures


def gate(
    summary: dict[str, Any],
    results: list[CaseResult],
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> Gate:
    """Thresholds are floors on rates, because single cases are noisy with a
    real model. A few checks are zero-tolerance: one leak is one too many."""
    g = Gate()
    if summary["skipped"]:
        g.failures.append(f"{summary['skipped']} cases skipped (budget reached?)")
    if (floor := thresholds.get("pass_rate_min")) is not None and summary["pass_rate"] < floor:
        g.failures.append(f"pass rate {summary['pass_rate']:.1%} is below {floor:.1%}")
    for feature, floor in thresholds.get("feature_min", {}).items():
        got = summary["by_feature"].get(feature, {}).get("rate", 0.0)
        if got < floor:
            g.failures.append(f"{feature} pass rate {got:.1%} is below {floor:.1%}")
    for name in thresholds.get("zero_tolerance_checks", []):
        bad = [r.case.id for r in results for c in r.checks if c.name == name and not c.passed]
        if bad:
            g.failures.append(f"{name} failed in {', '.join(bad)}")
    if baseline:
        before = {c["id"]: c["passed"] for c in baseline.get("cases", [])}
        for r in results:
            if r.skipped or r.case.id not in before:
                continue
            if before[r.case.id] and not r.passed:
                g.regressions.append(r.case.id)
            elif not before[r.case.id] and r.passed:
                g.fixed.append(r.case.id)
        limit = thresholds.get("max_regressions")
        if limit is not None and len(g.regressions) > limit:
            g.failures.append(
                f"{len(g.regressions)} cases regressed against the baseline (allowed: {limit})"
            )
    return g


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pct(d: dict[str, Any]) -> str:
    return f"{d['rate']:.0%} ({d['passed']}/{d['total']})"


def render_markdown(report: dict[str, Any], g: Gate, baseline: dict[str, Any] | None) -> str:
    """The score table posted on the PR."""
    run, s = report["run"], report["summary"]
    base_s = (baseline or {}).get("summary", {})
    lines = [
        f"## Eval suite: {'PASS' if g.ok else 'FAIL'}",
        "",
        f"Model `{run['model']}`, prompt `agent/{run['prompt_version']}`, "
        f"commit `{run.get('git_sha', '?')[:7]}`. "
        f"Cost **${run['spent_usd']}** of a ${run['budget_usd']} cap.",
        "",
        "| | This run | Baseline |",
        "|---|---|---|",
        f"| All cases | {s['pass_rate']:.0%} ({s['passed']}/{s['ran']}) | "
        + (f"{base_s['pass_rate']:.0%} |" if base_s else "n/a |"),
    ]
    for feature, d in s["by_feature"].items():
        b = base_s.get("by_feature", {}).get(feature)
        lines.append(f"| {feature} | {_pct(d)} | {_pct(b) if b else 'n/a'} |")
    for name, layer in report.get("layers", {}).items():
        lines.append(
            f"| {name} | {layer.get('headline', '')} | "
            f"{(baseline or {}).get('layers', {}).get(name, {}).get('headline', 'n/a')} |"
        )
    lines += [
        "",
        f"p50 / p95 model call: {s['p50_call_ms']} / {s['p95_call_ms']} ms; "
        f"agent cost per case ${s['agent_cost_per_case_usd']}.",
    ]
    if g.failures:
        lines += ["", "### Why it failed", ""] + [f"- {f}" for f in g.failures]
    if g.regressions:
        lines += ["", f"### Regressed ({len(g.regressions)}): passed on the baseline, fail now", ""]
        by_id = {c["id"]: c for c in report["cases"]}
        for cid in g.regressions:
            lines.append(f"- `{cid}`: {'; '.join(by_id[cid]['failed_checks'])[:240]}")
    if g.fixed:
        lines += ["", f"Newly passing ({len(g.fixed)}): " + ", ".join(f"`{c}`" for c in g.fixed)]
    return "\n".join(lines) + "\n"
