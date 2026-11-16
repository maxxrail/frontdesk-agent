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
)
from frontdesk_agent.llm.pricing import cost_usd
from frontdesk_agent.llm.retry import RetryPolicy, with_retries
from frontdesk_agent.prompts import load_prompt

logger = logging.getLogger(__name__)

PROMPT_NAME = "agent"
PROMPT_VERSION = "v1"
MAX_STEPS = 6

StopReason = Literal["reply", "step_limit"]


@dataclass
class AgentRun:
    reply: str | None  # the final text; None if the loop hit the step limit
    stopped: StopReason
    steps: int
    new_messages: list[Message] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


def system_prompt(ctx: ToolContext) -> str:
    c = ctx.client
    label = f"{c.first_name} {c.last_name}".strip() or "a new client with no name on file"
    system, _ = load_prompt(PROMPT_NAME, PROMPT_VERSION).render(
        business_name=ctx.tenant.name,
        business_type=ctx.tenant.business_type,
        client_label=label,
        now_local=ctx.now.astimezone(ctx.tz).strftime("%A, %B %-d, %Y at %-I:%M %p"),
        timezone=ctx.tenant.timezone,
    )
    return system


async def run_agent(
    llm: LLMClient,
    ctx: ToolContext,
    history: list[Message],
    *,
    read_only: bool = False,
    max_steps: int = MAX_STEPS,
    retry_policy: RetryPolicy | None = None,
) -> AgentRun:
    """Run the loop on ``history`` (which must end with the client's message).

    With ``read_only`` the agent can look things up but not book, cancel or
    save: used to draft a reply for a human to review.
    """
    allowed = {n for n, t in TOOLS.items() if t.read_only or not read_only}
    specs = [TOOLS[n].spec for n in TOOLS if n in allowed]
    system = system_prompt(ctx)
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
                prompt_version=PROMPT_VERSION,
                max_tokens=600,
            )

        turn = await with_retries(call, policy)
        await LLMCallRepository(ctx.session, ctx.tenant.id).record(
            _as_completion(turn),
            cost_usd=cost_usd(turn.model, turn.input_tokens, turn.output_tokens),
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
