"""The keyword risk rules, measured against labelled messages.

The ceilings sit at today's held-out numbers, so a rule change that misses
more risky messages, or holds more ordinary ones, fails CI. Lower them when
the rules improve; never raise them without an ADR. evals/risk_rules.py
writes the full report.
"""

from pathlib import Path

import pytest

from frontdesk_agent.evaluation.risk_metrics import RiskCase, load_risk_cases, measure

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden"
TUNING = load_risk_cases(GOLDEN / "risk.jsonl")
HELD_OUT = load_risk_cases(GOLDEN / "risk_holdout.jsonl")


@pytest.mark.parametrize("cases", [TUNING, HELD_OUT], ids=["tuning", "held-out"])
def test_set_shape(cases: list[RiskCase]) -> None:
    assert len({c.id for c in cases}) == len(cases)
    assert {c.label for c in cases} == {"medical", "money", "none"}


def test_sets_do_not_overlap() -> None:
    """A held-out message copied into the tuning set would flatter the rules."""
    assert not {c.text.lower() for c in TUNING} & {c.text.lower() for c in HELD_OUT}


def test_held_out_rates_do_not_regress() -> None:
    r = measure(HELD_OUT)
    assert r.miss_rate("medical") <= 0.75, r.as_dict()
    assert r.miss_rate("money") <= 0.94, r.as_dict()
    assert r.false_positive_rate <= 0.04, r.as_dict()


def test_tuning_rates_do_not_regress() -> None:
    r = measure(TUNING)
    assert r.miss_rate("medical") <= 0.50, r.as_dict()
    assert r.miss_rate("money") <= 0.40, r.as_dict()
    assert r.false_positive_rate <= 0.07, r.as_dict()
