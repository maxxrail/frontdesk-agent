"""Run eval cases through the real front desk, against a real database.

Each message takes the production path: stored as an inbound SMS, handled by
the ``handle_inbound`` job, its texts and jobs released by Effects after
commit. Only the edges are swapped: a fixed clock, a sender that records
instead of texting, and whatever LLMClient the caller passes (the real model
for a paid run, the scripted fake in tests).

What happened is then read back from Postgres, not from the agent's own
account of it: a booking counts if the appointment row exists.
"""

import asyncio
import json
import logging
import random
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from frontdesk_agent.agent.loop import PROMPT_VERSION
from frontdesk_agent.db.models import AppointmentStatus
from frontdesk_agent.db.repositories import (
    AppointmentRepository,
    ClientRepository,
    ConversationRepository,
    LLMCallRepository,
    ReviewRepository,
    TenantDirectory,
)
from frontdesk_agent.evaluation.budget import BudgetExceededError
from frontdesk_agent.evaluation.cases import Case
from frontdesk_agent.evaluation.world import NOW, EvalWorld, build_world, local, local_text
from frontdesk_agent.frontdesk.bandit import EpsilonGreedy
from frontdesk_agent.frontdesk.service import FrontDesk, accept_inbound
from frontdesk_agent.frontdesk.slotfill import SlotFiller
from frontdesk_agent.llm.base import LLMClient
from frontdesk_agent.llm.messages import Message, ToolResultBlock, ToolUseBlock
from frontdesk_agent.retrieval.embeddings import Embedder
from frontdesk_agent.services import Services, run_job

logger = logging.getLogger(__name__)


class _Clock:
    """Fixed for the whole case. Advancing it between messages would make a
    slot offered in one reply fall inside the booking lead time by the next."""

    def __init__(self) -> None:
        self.current = NOW

    def now(self) -> datetime:
        return self.current


@dataclass
class _Sender:
    sent: list[tuple[str, str]] = field(default_factory=list)

    async def send(self, *, to: str, from_: str, body: str) -> str:
        self.sent.append((to, body))
        return f"SM{uuid.uuid4().hex[:30]}"


@dataclass
class _Queue:
    jobs: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)

    async def enqueue(self, job: str, *, defer_seconds: float = 0, **kwargs: Any) -> None:
        self.jobs.append((job, defer_seconds, kwargs))


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class Review:
    kind: str
    reason: str
    proposed_reply: str


@dataclass
class Turn:
    message: str
    outcome: str
    sent: list[str]  # texts to the client after this message


@dataclass
class CaseRun:
    case_id: str
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    booked: list[str] = field(default_factory=list)  # new bookings, local start times
    cancelled: list[str] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    latencies_ms: list[int] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    # Other clients' appointments (either clinic) that are no longer booked.
    others_changed: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def outcome(self) -> str:
        return self.turns[-1].outcome if self.turns else "not_run"

    @property
    def sent(self) -> list[str]:
        return [t for turn in self.turns for t in turn.sent]

    @property
    def visible(self) -> list[str]:
        """Everything a client got or a reviewer was offered to send."""
        return self.sent + [r.proposed_reply for r in self.reviews if r.proposed_reply]

    @property
    def excerpts(self) -> list[str]:
        """Policy text the agent retrieved: what its policy facts must come from."""
        out = []
        for c in self.tool_calls:
            if c.name == "search_policy" and not c.is_error:
                out += [e["text"] for e in json.loads(c.result).get("excerpts", [])]
        return out

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_usd"] = str(self.cost_usd)
        d["outcome"] = self.outcome
        return d


@dataclass
class EvalContext:
    sessionmaker: async_sessionmaker[AsyncSession]
    llm: LLMClient
    embedder: Embedder
    prompt_version: str = PROMPT_VERSION


# Given the turns so far, the client's next text, or None to stop.
NextMessage = Callable[[list["Turn"]], Awaitable[str | None]]


async def run_case(
    ctx: EvalContext, case: Case, next_message: NextMessage | None = None, max_turns: int = 10
) -> CaseRun:
    """Run one case. By default the client sends ``case.messages`` in order;
    a simulation passes ``next_message`` to write each text in reply."""
    run = CaseRun(case.id)

    async def scripted(turns: list[Turn]) -> str | None:
        return case.messages[len(turns)] if len(turns) < len(case.messages) else None

    next_message = next_message or scripted
    async with ctx.sessionmaker() as session:
        world = await build_world(session, case, ctx.embedder)
        await session.commit()
    run.secrets = list(world.secrets)
    setup = {local_text(local(s)) for s in case.appointments}

    clock, sender, queue = _Clock(), _Sender(), _Queue()
    frontdesk = FrontDesk(
        llm=ctx.llm,
        embedder=ctx.embedder,
        clock=clock,
        slot_filler=SlotFiller(EpsilonGreedy(0.0, rng=random.Random(0))),
        prompt_version=ctx.prompt_version,
    )
    services = Services(ctx.sessionmaker, frontdesk, sender, queue, clock)
    try:
        while len(run.turns) < max_turns and (body := await next_message(run.turns)):
            before = len(sender.sent)
            outcome = await _text(services, world, body)
            await _drain(services, queue)
            run.turns.append(
                Turn(
                    body, outcome, [b for to, b in sender.sent[before:] if to == world.client_phone]
                )
            )
    except BudgetExceededError:
        raise
    except Exception as exc:  # one broken case must not end the run
        logger.exception("eval case failed", extra={"case": case.id})
        run.error = f"{type(exc).__name__}: {exc}"
    await _collect(ctx, world, run, setup)
    return run


async def _text(services: Services, world: EvalWorld, body: str) -> str:
    async with services.sessionmaker() as session:
        tenant = await TenantDirectory(session).get(world.tenant_id)
        msg = await accept_inbound(
            session,
            tenant,
            from_=world.client_phone,
            to=world.clinic_number,
            body=body,
            sid=f"SM{uuid.uuid4().hex[:30]}",
        )
        await session.commit()
    assert msg is not None
    return await run_job(
        services, "handle_inbound", tenant_id=str(world.tenant_id), message_id=str(msg.id)
    )


async def _drain(services: Services, queue: _Queue) -> None:
    """Run immediate follow-up jobs (a slot fill after a cancellation)."""
    queue_jobs = queue.jobs
    while ready := [j for j in queue_jobs if not j[1]]:
        job = ready[0]
        queue_jobs.remove(job)
        await run_job(services, job[0], **job[2])


async def _collect(ctx: EvalContext, world: EvalWorld, run: CaseRun, setup: set[str]) -> None:
    async with ctx.sessionmaker() as session:
        tid = world.tenant_id
        me = await ClientRepository(session, tid).by_phone(world.client_phone)
        for other_tid in (tid, world.other_tenant_id):
            for a in await AppointmentRepository(session, other_tid).list(limit=500):
                mine = me is not None and a.client_id == me.id
                if not mine and a.status != AppointmentStatus.BOOKED:
                    run.others_changed.append(f"{local_text(a.starts_at)} {a.status}")
        if me is not None:
            for a in await AppointmentRepository(session, tid).list(limit=500):
                if a.client_id != me.id:
                    continue
                at = local_text(a.starts_at)
                if a.status == AppointmentStatus.BOOKED and at not in setup:
                    run.booked.append(at)
                elif a.status == AppointmentStatus.CANCELLED:
                    run.cancelled.append(at)
            convo = await ConversationRepository(session, tid).for_client(me.id)
            run.tool_calls = _tool_calls([Message.from_json(m) for m in convo.transcript])
        run.reviews = [
            Review(r.kind.value, r.reason, r.proposed_reply)
            for r in await ReviewRepository(session, tid).by_status(None, limit=50)
        ]
        calls = await LLMCallRepository(session, tid).recent(limit=500)
        run.model_calls = len(calls)
        run.input_tokens = sum(c.input_tokens for c in calls)
        run.output_tokens = sum(c.output_tokens for c in calls)
        run.cost_usd = sum((c.cost_usd or Decimal(0) for c in calls), Decimal(0))
        run.latencies_ms = sorted(c.latency_ms for c in calls)
        await session.rollback()  # for_client may have created a row; keep the record clean


def _tool_calls(transcript: list[Message]) -> list[ToolCall]:
    results = {
        b.tool_use_id: b for m in transcript for b in m.content if isinstance(b, ToolResultBlock)
    }
    out = []
    for m in transcript:
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                r = results.get(b.id)
                out.append(
                    ToolCall(b.name, b.input, r.content if r else "", bool(r and r.is_error))
                )
    return out


async def run_cases(
    ctx: EvalContext,
    cases: Sequence[Case],
    *,
    concurrency: int = 4,
    on_done: Callable[[Case, CaseRun], None] | None = None,
) -> tuple[list[CaseRun], str | None]:
    """Run every case; returns the runs and, if the budget ran out, why.

    Cases not started when the budget runs out are reported as skipped, never
    as passes or failures.
    """
    gate = asyncio.Semaphore(concurrency)
    stopped: list[str] = []

    async def one(case: Case) -> CaseRun:
        async with gate:
            if stopped:
                return CaseRun(case.id, error="skipped: budget reached")
            try:
                run = await run_case(ctx, case)
            except BudgetExceededError as exc:
                stopped.append(str(exc))
                return CaseRun(case.id, error=f"skipped: {exc}")
            if on_done:
                on_done(case, run)
            return run

    runs = await asyncio.gather(*(one(c) for c in cases))
    return list(runs), (stopped[0] if stopped else None)
