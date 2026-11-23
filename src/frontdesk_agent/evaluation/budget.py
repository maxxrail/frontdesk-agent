"""A spending cap around any LLMClient.

Every paid eval run goes through this, so a bug that loops, or a case set
that grew, stops at the cap instead of at the end of the month. One Budget is
shared by every client in a run (the agent, the judge, the simulated client),
so the cap is on the run, not per model. The check runs before each call, so
a run can overshoot by at most the calls already in flight.
"""

from collections import defaultdict
from decimal import Decimal

from frontdesk_agent.llm.base import LLMClient, LLMCompletion
from frontdesk_agent.llm.messages import AgentTurn, Message, ToolSpec
from frontdesk_agent.llm.pricing import Usage, cost_of


class BudgetExceededError(RuntimeError):
    """Deliberately not an LLMError: nothing should retry or swallow it."""


class Budget:
    def __init__(self, cap_usd: Decimal) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = Decimal(0)
        self.calls = 0
        self.by_prompt: dict[str, Decimal] = defaultdict(Decimal)

    def check(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise BudgetExceededError(
                f"eval budget of ${self.cap_usd} reached (spent ${self.spent_usd})"
            )

    def add(self, cost: Decimal, prompt_name: str) -> None:
        self.calls += 1
        self.spent_usd += cost
        self.by_prompt[prompt_name] += cost


class BudgetedClient:
    def __init__(self, inner: LLMClient, budget: Budget, *, allow_unpriced: bool = False):
        self.inner = inner
        self.budget = budget
        self.allow_unpriced = allow_unpriced  # only the fake model has no price

    def _add(self, call: Usage, prompt_name: str) -> None:
        cost = cost_of(call)
        if cost is None:
            if not self.allow_unpriced:
                # Without a price the cap means nothing; stop rather than guess.
                self.budget.spent_usd = self.budget.cap_usd
                raise BudgetExceededError(f"no price for model {call.model!r} in llm/pricing.py")
            cost = Decimal(0)
        self.budget.add(cost, prompt_name)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        prompt_name: str,
        prompt_version: str,
        max_tokens: int = 1024,
    ) -> LLMCompletion:
        self.budget.check()
        out = await self.inner.complete(
            system=system,
            user=user,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
        )
        self._add(out, prompt_name)
        return out

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
        self.budget.check()
        turn = await self.inner.converse(
            system=system,
            messages=messages,
            tools=tools,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
        )
        self._add(turn, prompt_name)
        return turn
