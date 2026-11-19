"""The eval case files themselves: well formed, big enough, and consistent.

A case with a typo in a tool name would expect something that can never
happen and fail forever; one whose tags disagree with the risk rules would
make the report lie about which defence caught what.
"""

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from frontdesk_agent.agent.risk import classify
from frontdesk_agent.agent.tools import TOOLS
from frontdesk_agent.db.models import ReviewKind
from frontdesk_agent.evaluation.cases import Case, load_cases
from frontdesk_agent.evaluation.simulation import load_scenarios
from frontdesk_agent.retrieval.evaluation import load_golden

ROOT = Path(__file__).resolve().parent.parent
CASES = load_cases(ROOT / "evals" / "cases")
SCENARIOS = load_scenarios()
POLICY_GOLDEN = load_golden(ROOT / "evals" / "golden" / "retrieval.jsonl")
OUTCOMES = {
    "agent_reply",
    "opted_out",
    "opted_in",
    "ignored_opted_out",
    "held_while_paused",
    *(f"held_for_review:{k.value}" for k in ReviewKind),
    "held_for_review:step_limit",
    "held_for_review:too_long",
    *(f"held_for_review:guardrail_{r}" for r in ("other_client", "payment_data")),
    "held_for_review:guardrail_clinical_advice",
}


def test_suite_size_meets_the_module_target() -> None:
    """Roadmap: at least 150 cases in version control, 40 of them red team."""
    assert len(CASES) + len(POLICY_GOLDEN) >= 150
    assert Counter(c.feature for c in CASES)["redteam"] >= 40
    assert len(SCENARIOS) == 30


def test_every_feature_has_every_difficulty_or_close() -> None:
    by_feature = Counter((c.feature, c.difficulty) for c in CASES)
    for feature in {c.feature for c in CASES}:
        assert sum(by_feature[(feature, d)] > 0 for d in ("easy", "medium", "hard")) >= 2, feature


@pytest.mark.parametrize("case", CASES + [s.as_case() for s in SCENARIOS], ids=lambda c: c.id)
def test_case_refers_only_to_real_things(case: Case) -> None:
    e = case.expect
    for tool in (*e.tools, *e.no_tools):
        assert tool in TOOLS, tool
    for outcome in e.outcome:
        assert outcome in OUTCOMES, outcome
    for kind in e.review:
        assert kind == "none" or kind in {k.value for k in ReviewKind}, kind
    if "none" in e.review:
        assert e.review == ("none",), "'none' cannot be combined with a kind"
    now = datetime(2030, 1, 7)
    for at in [*(s.at for s in (*case.appointments, *case.taken)), *(e.booked or ())]:
        when = datetime.fromisoformat(at)
        assert now <= when <= now + timedelta(days=60), at


def test_risk_case_tags_match_the_rules() -> None:
    """'model-only' must mean the keyword rules really miss the message."""
    for c in CASES:
        if c.feature != "risk":
            continue
        caught = any(classify(m) for m in c.messages)
        assert ("caught-by-rules" in c.tags) == caught, c.id
        assert ("model-only" in c.tags) == (not caught), c.id


def test_red_team_covers_the_four_attack_kinds() -> None:
    tags = Counter(t for c in CASES if c.feature == "redteam" for t in c.tags)
    for kind in ("injection", "other-client", "other-tenant", "diagnosis"):
        assert tags[kind] >= 10, (kind, tags[kind])
