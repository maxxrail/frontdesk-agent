"""False-positive and false-negative rates of the keyword risk rules.

The two errors are not equal (see agent/risk.py). A false negative on a
medical message lets the model reply to it unreviewed; a false positive costs
a reviewer a minute. So the numbers are reported per kind, and the one that
gates CI is the medical miss rate.

"Missed" means the rules did not flag a risky message at all. A medical
message flagged as money is still held for a human, so it counts as caught
but is reported separately as "wrong kind".
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from frontdesk_agent.agent.risk import RiskFlag, classify
from frontdesk_agent.db.models import ReviewKind

Label = Literal["medical", "money", "none"]

_KIND: dict[ReviewKind, Label] = {ReviewKind.MEDICAL: "medical", ReviewKind.MONEY: "money"}


@dataclass(frozen=True, slots=True)
class RiskCase:
    id: str
    text: str
    label: Label
    note: str = ""


def load_risk_cases(path: Path) -> list[RiskCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(RiskCase(d["id"], d["text"], d["label"], d.get("note", "")))
    return cases


@dataclass
class RiskReport:
    counts: dict[Label, int] = field(default_factory=dict)
    missed: dict[Label, list[RiskCase]] = field(default_factory=dict)  # risky, not flagged
    wrong_kind: list[tuple[RiskCase, Label]] = field(default_factory=list)  # held, mislabelled
    false_positives: list[tuple[RiskCase, str]] = field(default_factory=list)  # (case, matched)

    def miss_rate(self, label: Label) -> float:
        n = self.counts.get(label, 0)
        return len(self.missed.get(label, [])) / n if n else 0.0

    @property
    def false_positive_rate(self) -> float:
        n = self.counts.get("none", 0)
        return len(self.false_positives) / n if n else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "counts": self.counts,
            "medical_miss_rate": round(self.miss_rate("medical"), 3),
            "money_miss_rate": round(self.miss_rate("money"), 3),
            "false_positive_rate": round(self.false_positive_rate, 3),
            "missed": {k: [c.id for c in v] for k, v in self.missed.items()},
            "wrong_kind": [c.id for c, _ in self.wrong_kind],
            "false_positives": [c.id for c, _ in self.false_positives],
        }


def measure(cases: list[RiskCase], rule: Callable[[str], RiskFlag | None] = classify) -> RiskReport:
    report = RiskReport()
    for case in cases:
        report.counts[case.label] = report.counts.get(case.label, 0) + 1
        flag = rule(case.text)
        got: Label = _KIND.get(flag.kind, "none") if flag else "none"
        if case.label == "none":
            if flag is not None:
                report.false_positives.append((case, flag.matched))
        elif flag is None:
            report.missed.setdefault(case.label, []).append(case)
        elif got != case.label:
            report.wrong_kind.append((case, got))
    return report
