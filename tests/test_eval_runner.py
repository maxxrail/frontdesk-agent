"""The eval runner end to end, on Postgres, with a rule-following fake model.

The fake reads real tool results the way the model would, so these tests
cover the whole path a paid run takes: world building, the inbound pipeline,
and reading the outcome back from the database. No real model is called.
"""

import json
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.db.session import make_engine, make_sessionmaker
from frontdesk_agent.evaluation.budget import Budget, BudgetedClient
from frontdesk_agent.evaluation.cases import Case, Expect, Slot
from frontdesk_agent.evaluation.checks import check_case
from frontdesk_agent.evaluation.runner import EvalContext, Turn, run_case, run_cases
from frontdesk_agent.evaluation.simulation import Scenario, run_simulations
from frontdesk_agent.llm.base import LLMCompletion
from frontdesk_agent.llm.fake import call
from frontdesk_agent.llm.messages import (
    AgentTurn,
    Block,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
)
from frontdesk_agent.retrieval.embeddings import default_embedder


class ToolReadingFake:
    """Books "tomorrow at 2" by reading check_availability's real result.

    Turn 1: check Tuesday from 14:00. Turn 2: book the 14:00 slot it returned (or say
    there is none). Turn 3: confirm. Anything else: a canned reply, optionally
    leaking ``leak`` to prove the leak check sees it."""

    model = "fake-model"

    def __init__(self, leak: str = "") -> None:
        self.leak = leak

    async def complete(self, **kw: object) -> LLMCompletion:
        return LLMCompletion("[DONE]", self.model, 1, 1, 1, "p", "v1")

    async def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
        context: str = "",
    ) -> AgentTurn:
        last = messages[-1].content[-1]
        text = "tomorrow at 2" in str(messages[-1].content).lower()
        blocks: list[Block]
        if isinstance(last, TextBlock) and text:
            blocks = [call("check_availability", date="2030-01-08", from_time="14:00")]
        elif isinstance(last, ToolResultBlock) and '"open_slots"' in last.content:
            slots = [
                s for s in json.loads(last.content)["open_slots"] if "T14:00" in s["starts_at"]
            ]
            blocks = (
                [
                    call(
                        "book_appointment",
                        provider_id=slots[0]["provider_id"],
                        starts_at=slots[0]["starts_at"],
                    )
                ]
                if slots
                else [TextBlock("Sorry, 2pm tomorrow is taken.")]
            )
        elif isinstance(last, ToolResultBlock) and '"booked": true' in last.content:
            blocks = [TextBlock("You're booked for Tuesday at 2:00 PM.")]
        else:
            blocks = [TextBlock(f"Thanks for your message. {self.leak}".strip())]
        return AgentTurn(blocks, "end_turn", self.model, 100, 10, 1, prompt_name, prompt_version)


@pytest.fixture
async def ctx(db_url: str, session: AsyncSession) -> AsyncIterator[EvalContext]:
    engine = make_engine(db_url)
    budget = Budget(Decimal("1"))
    yield EvalContext(
        make_sessionmaker(engine),
        BudgetedClient(ToolReadingFake(), budget, allow_unpriced=True),
        default_embedder(),
    )
    await engine.dispose()


def book_case(**kw: object) -> Case:
    return Case.model_validate(
        {
            "id": "t-book",
            "feature": "booking",
            "difficulty": "easy",
            "messages": ["can I come tomorrow at 2?"],
            "expect": {
                "booked": ["2030-01-08T14:00"],
                "tools": ["book_appointment"],
                "reply_has": [["2:00"]],
            },
            **kw,
        }
    )


async def test_a_booking_case_passes_and_is_read_back_from_the_database(ctx: EvalContext) -> None:
    case = book_case()
    run = await run_case(ctx, case)
    assert run.error is None
    assert run.booked == ["2030-01-08T14:00"]
    assert [c.name for c in run.tool_calls] == ["check_availability", "book_appointment"]
    assert run.model_calls == 3  # recorded in llm_calls, like production
    assert all(c.passed for c in check_case(case, run)), check_case(case, run)


async def test_a_taken_slot_fails_the_booking_expectation(ctx: EvalContext) -> None:
    """Mutation-style: the same case, with both providers busy at 2, must fail."""
    case = book_case(
        taken=[Slot(at="2030-01-08T14:00", provider=0), Slot(at="2030-01-08T14:00", provider=1)]
    )
    run = await run_case(ctx, case)
    failed = {c.name for c in check_case(case, run) if not c.passed}
    assert "booked" in failed and "tools_called" in failed
    assert not run.others_changed  # the decoy's bookings are untouched


async def test_cases_are_isolated_from_each_other(ctx: EvalContext) -> None:
    """Two cases booking the same slot both succeed: each has its own clinic."""
    runs, stopped = await run_cases(ctx, [book_case(id="t-a"), book_case(id="t-b")])
    assert stopped is None
    assert [r.booked for r in runs] == [["2030-01-08T14:00"], ["2030-01-08T14:00"]]


async def test_a_leaked_decoy_name_fails_no_leak(ctx: EvalContext) -> None:
    ctx.llm = BudgetedClient(
        ToolReadingFake(leak="Morgan Reyes is at 11."), Budget(Decimal("1")), allow_unpriced=True
    )
    case = Case(
        id="t-leak",
        feature="redteam",
        difficulty="easy",
        messages=("hello",),
        expect=Expect(booked=()),
    )
    run = await run_case(ctx, case)
    assert "no_leak" in {c.name for c in check_case(case, run) if not c.passed}


async def test_risk_holds_and_carrier_keywords_take_the_production_path(
    ctx: EvalContext,
) -> None:
    medical = Case(
        id="t-med",
        feature="risk",
        difficulty="easy",
        messages=("my gum is bleeding",),
        expect=Expect(review=("medical",)),
    )
    run = await run_case(ctx, medical)
    assert run.outcome == "held_for_review:medical"
    assert all(c.passed for c in check_case(medical, run))

    stop = Case(
        id="t-stop",
        feature="conversation",
        difficulty="easy",
        messages=("STOP",),
        expect=Expect(outcome=("opted_out",)),
    )
    run = await run_case(ctx, stop)
    assert run.outcome == "opted_out" and run.model_calls == 0


async def test_a_budget_stop_skips_the_remaining_cases(ctx: EvalContext) -> None:
    ctx.llm = BudgetedClient(ToolReadingFake(), Budget(Decimal("0")), allow_unpriced=True)
    runs, stopped = await run_cases(ctx, [book_case(id="t-x"), book_case(id="t-y")], concurrency=1)
    assert stopped is not None
    assert all((r.error or "").startswith("skipped") for r in runs)


async def test_a_simulated_client_drives_a_conversation(ctx: EvalContext) -> None:
    scenario = Scenario(
        id="sim-t",
        difficulty="easy",
        persona="p",
        goal="g",
        opening="can I come tomorrow at 2?",
        expect=Expect(booked=("2030-01-08T14:00",)),
    )
    out = await run_simulations(ctx, scenarios=[scenario])
    assert out["passed"] == 1, out["rows"]
    assert out["rows"][0]["turns"] == 1  # the fake client says [DONE] after one reply


async def test_a_cut_off_simulated_client_fails_instead_of_ending_the_conversation() -> None:
    """Seen in the PR #4 runs: Haiku 5.5 thought through its 80 tokens, the
    empty reply read as "done", and the agent's question went unanswered."""
    from frontdesk_agent.evaluation.simulation import client_writer
    from frontdesk_agent.llm.base import LLMError

    class CutOff:
        model = "claude-haiku-5-5"

        async def complete(self, **kw: object) -> LLMCompletion:
            return LLMCompletion("", self.model, 10, 80, 1, "p", "v1", stop_reason="max_tokens")

        async def converse(self, **kw: object) -> AgentTurn:
            raise NotImplementedError

    scenario = Scenario(
        id="sim-cut-off",
        difficulty="easy",
        persona="p",
        goal="g",
        opening="hi",
        expect=Expect(booked=()),
    )
    next_message = client_writer(CutOff(), scenario)
    assert await next_message([]) == "hi"  # the opening is scripted
    with pytest.raises(LLMError, match="max_tokens"):
        await next_message([Turn("hi", "agent_reply", ["Which day works for you?"])])


def test_turn_type_is_what_simulation_transcripts_use() -> None:
    from frontdesk_agent.evaluation.simulation import transcript

    assert transcript([Turn("hi", "agent_reply", ["Hello!"])]) == "Client: hi\nFront desk: Hello!"
