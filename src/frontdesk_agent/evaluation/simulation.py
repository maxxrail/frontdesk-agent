"""Layer 3: simulated multi-turn conversations.

A second model plays a client with a persona and a goal, texting the real
front desk until it is done (or a turn limit). Success is judged on what the
database shows afterwards, with the same expectations as the fixed cases, so
the simulated client's opinion of how it went counts for nothing.

The simulated client runs on a small, fixed model (claude-haiku-5-5 by
default) so a change of the agent's model does not also change the tester.
"""

import asyncio
from pathlib import Path
from typing import Any

from pydantic import Field

from frontdesk_agent.demo_clinics import CLINICS
from frontdesk_agent.evaluation.cases import Case, Difficulty, Expect, Slot, TenantSlug, _Strict
from frontdesk_agent.evaluation.checks import check_case
from frontdesk_agent.evaluation.runner import EvalContext, NextMessage, Turn, run_case
from frontdesk_agent.llm.base import LLMClient, LLMError
from frontdesk_agent.prompts import load_prompt

PROMPT_NAME = "client_sim"
PROMPT_VERSION = "v1"
DONE = "[DONE]"
SCENARIOS = Path(__file__).resolve().parents[3] / "evals" / "scenarios.jsonl"


class Scenario(_Strict):
    id: str = Field(pattern=r"^sim-[a-z0-9-]+$")
    difficulty: Difficulty
    tenant: TenantSlug = "maple-dental"
    client: str = Field(default="known", pattern="^(known|new)$")
    persona: str
    goal: str
    opening: str
    max_turns: int = Field(default=6, ge=1, le=10)
    appointments: tuple[Slot, ...] = ()
    taken: tuple[Slot, ...] = ()
    expect: Expect

    def as_case(self) -> Case:
        return Case(
            id=self.id,
            feature="conversation",
            difficulty=self.difficulty,
            tenant=self.tenant,
            client="new" if self.client == "new" else "known",
            messages=(self.opening,),
            appointments=self.appointments,
            taken=self.taken,
            expect=self.expect,
            tags=("simulation",),
        )


def load_scenarios(path: Path = SCENARIOS) -> list[Scenario]:
    return [
        Scenario.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def transcript(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        lines.append(f"Client: {t.message}")
        lines += [f"Front desk: {s}" for s in t.sent] or ["Front desk: (no reply)"]
    return "\n".join(lines)


def client_writer(sim_llm: LLMClient, scenario: Scenario) -> NextMessage:
    clinic = next(c for c in CLINICS if c.slug == scenario.tenant)

    async def next_message(turns: list[Turn]) -> str | None:
        if not turns:
            return scenario.opening
        system, user = load_prompt(PROMPT_NAME, PROMPT_VERSION).render(
            clinic_name=clinic.name,
            business_type=clinic.business_type,
            persona=scenario.persona,
            goal=scenario.goal,
            transcript=transcript(turns),
        )
        out = await sim_llm.complete(
            system=system,
            user=user,
            prompt_name=PROMPT_NAME,
            prompt_version=PROMPT_VERSION,
            max_tokens=80,
        )
        if out.stop_reason == "max_tokens" and not out.text.strip():
            # An empty reply would read as "the client is done" and end the
            # conversation early; a cut-off reply is a failure to report.
            raise LLMError(f"simulated client cut off at max_tokens ({out.output_tokens} tokens)")
        text = out.text.strip().strip('"')
        return None if not text or DONE in text else text

    return next_message


async def run_simulations(
    ctx: EvalContext,
    concurrency: int = 4,
    *,
    sim_llm: LLMClient | None = None,
    scenarios: list[Scenario] | None = None,
) -> dict[str, Any]:
    """Run every scenario. ``sim_llm`` plays the client; by default the same
    budgeted client as the agent (in a dry run, the canned model)."""
    scenarios = scenarios if scenarios is not None else load_scenarios()
    writer_llm = sim_llm or ctx.llm
    gate = asyncio.Semaphore(concurrency)

    async def one(s: Scenario) -> dict[str, Any]:
        async with gate:
            case = s.as_case()
            run = await run_case(ctx, case, client_writer(writer_llm, s), max_turns=s.max_turns)
            checks = check_case(case, run)
            return {
                "id": s.id,
                "difficulty": s.difficulty,
                "passed": all(c.passed for c in checks),
                "turns": len(run.turns),
                "failed_checks": [f"{c.name}: {c.detail}" for c in checks if not c.passed],
                "transcript": transcript(run.turns),
                "cost_usd": str(run.cost_usd),
            }

    rows = list(await asyncio.gather(*(one(s) for s in scenarios)))
    passed = sum(r["passed"] for r in rows)
    return {
        "headline": f"{passed}/{len(rows)} conversations reached their goal",
        "passed": passed,
        "total": len(rows),
        "rate": round(passed / len(rows), 3) if rows else 0.0,
        "avg_turns": round(sum(r["turns"] for r in rows) / len(rows), 1) if rows else 0,
        "rows": rows,
    }
