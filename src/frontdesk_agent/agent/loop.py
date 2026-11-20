"""The agent loop, written by hand rather than with a framework.

    call the model -> run any tool calls -> feed results back -> repeat

It stops at a reply with no tool calls, or after ``max_steps`` model calls.
The step limit is what stops a model that keeps calling tools forever; it is
tested with a scripted fake that never stops (ADR 0010).

Everything provider-specific lives behind LLMClient; everything with side
effects lives in tools; this file only sequences them and keeps the record.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from frontdesk_agent.agent.tools import TOOLS, ToolContext, execute
from frontdesk_agent.db.repositories import LLMCallRepository
from frontdesk_agent.llm.base import LLMClient, LLMCompletion
from frontdesk_agent.llm.messages import (
    AgentTurn,
    Block,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from frontdesk_agent.llm.pricing import cost_usd
from frontdesk_agent.llm.retry import RetryPolicy, with_retries
from frontdesk_agent.metrics import TURN_COST
from frontdesk_agent.prompts import load_prompt
from frontdesk_agent.timefmt import human_datetime
from frontdesk_agent.tracing import tracer

logger = logging.getLogger(__name__)

PROMPT_NAME = "agent"
PROMPT_VERSION = "v1"  # the default; Settings.agent_prompt_version overrides it
MAX_STEPS = 6

StopReason = Literal["reply", "step_limit"]


@dataclass
class AgentRun:
    reply: str | None  # the final text; None if the loop hit the step limit
    stopped: StopReason
    steps: int
    new_messages: list[Message] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    cost_usd: Decimal = Decimal(0)  # every model call in this run, priced


def system_prompt(ctx: ToolContext, prompt_version: str = PROMPT_VERSION) -> str:
    c = ctx.client
    label = f"{c.first_name} {c.last_name}".strip() or "a new client with no name on file"
    system, _ = load_prompt(PROMPT_NAME, prompt_version).render(
        business_name=ctx.tenant.name,
        business_type=ctx.tenant.business_type,
        client_label=label,
        now_local=human_datetime(ctx.now.astimezone(ctx.tz), with_year=True),
        timezone=ctx.tenant.timezone,
    )
    return system


# Appended to the system prompt in read-only mode. The prompt tells the agent
# to call escalate_to_human for clinical and money questions, but a held
# message is already with staff and that tool is not offered. Without this
# note Haiku 5.5 called the missing tool in 14 of 132 eval cases and drafted
# "I wasn't able to pass your message to the team" (ADR 0015).
READ_ONLY_NOTE = (
    "\n\nThis message has already been passed to the clinic team, and nothing you"
    " write is sent until a staff member approves it. escalate_to_human is not"
    " available and not needed. You may look things up. Write the short reply a"
    " staff member could send; do not say you could not reach the team."
)


async def run_agent(
    llm: LLMClient,
    ctx: ToolContext,
    history: list[Message],
    *,
    read_only: bool = False,
    max_steps: int = MAX_STEPS,
    retry_policy: RetryPolicy | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> AgentRun:
    """Run the loop on ``history`` (which must end with the client's message).

    With ``read_only`` the agent can look things up but not book, cancel or
    save: used to draft a reply for a human to review.
    """
    with tracer.start_as_current_span("agent.turn") as span:
        span.set_attributes(
            {
                "conversation.id": str(ctx.conversation.id),
                "agent.read_only": read_only,
                "llm.prompt": f"{PROMPT_NAME}/{prompt_version}",
            }
        )
        run = await _run(llm, ctx, history, read_only, max_steps, retry_policy, prompt_version)
        span.set_attributes(
            {
                "agent.steps": run.steps,
                "agent.stopped": run.stopped,
                "agent.tool_calls": len(run.tool_calls),
                "agent.cost_usd": float(run.cost_usd),
            }
        )
        TURN_COST.observe(float(run.cost_usd))
        return run


async def _run(
    llm: LLMClient,
    ctx: ToolContext,
    history: list[Message],
    read_only: bool,
    max_steps: int,
    retry_policy: RetryPolicy | None,
    prompt_version: str,
) -> AgentRun:
    # Fixed here, before the model reads anything: no text it reads later
    # (the client's, a document's, a tool result's) can widen it.
    allowed = frozenset(n for n, t in TOOLS.items() if t.read_only or not read_only)
    specs = [TOOLS[n].spec for n in TOOLS if n in allowed]
    ctx.documents_in_context = ctx.documents_in_context or _read_documents(history)
    system = system_prompt(ctx, prompt_version) + (READ_ONLY_NOTE if read_only else "")
    messages = list(history)
    run = AgentRun(reply=None, stopped="step_limit", steps=0)
    policy = retry_policy or RetryPolicy()

    for step in range(1, max_steps + 1):
        run.steps = step

        async def call() -> AgentTurn:
            return await llm.converse(
                system=system,
                messages=messages,
                tools=specs,
                prompt_name=PROMPT_NAME,
                prompt_version=prompt_version,
                max_tokens=600,
            )

        turn = await with_retries(call, policy)
        cost = cost_usd(turn.model, turn.input_tokens, turn.output_tokens)
        run.cost_usd += cost or Decimal(0)
        await LLMCallRepository(ctx.session, ctx.tenant.id).record(
            _as_completion(turn), cost_usd=cost
        )
        messages.append(turn.as_message())
        run.new_messages.append(turn.as_message())

        if not turn.tool_calls:
            run.reply = turn.text or None
            run.stopped = "reply"
            return run

        results: list[Block] = []
        for tool_call in turn.tool_calls:
            run.tool_calls.append(tool_call.name)
            outcome = await execute(ctx, tool_call.name, tool_call.input, allowed)
            results.append(ToolResultBlock(tool_call.id, outcome.content, outcome.is_error))
        messages.append(Message("user", results))
        run.new_messages.append(Message("user", results))

    logger.warning("agent hit the step limit", extra={"steps": max_steps})
    return run


def _read_documents(history: list[Message]) -> bool:
    """Policy text from an earlier turn is still in the window the model sees."""
    blocks = (b for m in history for b in m.content)
    return any(isinstance(b, ToolUseBlock) and b.name == "search_policy" for b in blocks)


def _as_completion(turn: AgentTurn) -> LLMCompletion:
    return LLMCompletion(
        text=turn.text,
        model=turn.model,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        latency_ms=turn.latency_ms,
        prompt_name=turn.prompt_name,
        prompt_version=turn.prompt_version,
        stop_reason=turn.stop_reason,
    )


def trim_history(messages: list[Message], keep: int = 30) -> list[Message]:
    """Keep the most recent messages, starting at a client text message.

    A cut that started at a tool result, with its tool call dropped, is a
    request the provider rejects; so the window always opens on user text.
    """
    if len(messages) <= keep:
        return messages
    window = messages[-keep:]
    for i, m in enumerate(window):
        if m.role == "user" and any(isinstance(b, TextBlock) for b in m.content):
            return window[i:]
    return window[-1:]
