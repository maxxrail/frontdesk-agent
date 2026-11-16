"""Layer 2: an LLM judge grades tone and faithfulness of agent replies.

A judge is a model too, so it is not trusted until it agrees with a person:
evals/judge_agreement.py compares it with hand labels (evals/judge/labels.csv)
and the agreement is reported in evals/README.md. Verdicts are pass/fail
rather than a 1-5 score, because agreement on a binary label is easy to
measure and hard to argue with.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from frontdesk_agent.agent.risk import MEDICAL_HOLDING_MESSAGE
from frontdesk_agent.demo_clinics import CLINICS
from frontdesk_agent.evaluation.budget import BudgetExceededError
from frontdesk_agent.evaluation.results import CaseResult
from frontdesk_agent.llm.base import LLMClient, LLMError
from frontdesk_agent.prompts import load_prompt

PROMPT_NAME = "judge"
PROMPT_VERSION = "v1"
# Room for the JSON verdict. Thinking is switched off for the judge (see
# judge_thinking), so this is the answer alone.
MAX_TOKENS = 1024


def judge_thinking(model: str) -> dict[str, str] | None:
    """Thinking off for the judge, in the form its model accepts.

    Sonnet 5.5 thinks by default and its thinking counts against max_tokens:
    at 300 tokens, 52 of 129 verdicts in a CI run came back cut off or empty.
    It rejects {"type": "disabled"}; "between_tools" is its off switch. A
    short pass/fail grade does not need thinking, and it would multiply the
    judge's cost per run.
    """
    return {"type": "between_tools"} if model.startswith("claude-sonnet-5-5") else None


MAX_EXCERPT_CHARS = 700


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tone: Literal["pass", "fail"]
    tone_reason: str
    faithfulness: Literal["pass", "fail", "n/a"]
    faithfulness_reason: str


@dataclass(frozen=True)
class JudgeItem:
    id: str
    business_type: str
    client_messages: list[str]
    excerpts: list[str]
    reply: str


def items_from_results(results: list[CaseResult]) -> list[JudgeItem]:
    """One item per case that produced a reply worth grading: the last text the
    client got, or the draft a reviewer was given. The fixed medical safety
    message is not model-written, so it is never graded."""
    items = []
    for r in results:
        replies = [t for t in r.run.visible if t != MEDICAL_HOLDING_MESSAGE]
        if r.skipped or not replies:
            continue
        clinic = next(c for c in CLINICS if c.slug == r.case.tenant)
        items.append(
            JudgeItem(
                r.case.id,
                clinic.business_type,
                list(r.case.messages),
                r.run.excerpts,
                replies[-1],
            )
        )
    return items


def render(item: JudgeItem) -> tuple[str, str]:
    excerpts = "\n\n".join(f"[{i}] {e[:MAX_EXCERPT_CHARS]}" for i, e in enumerate(item.excerpts, 1))
    return load_prompt(PROMPT_NAME, PROMPT_VERSION).render(
        business_type=item.business_type,
        client_messages="\n".join(f"- {m}" for m in item.client_messages),
        excerpts=excerpts or "(none)",
        reply=item.reply,
    )


def parse(text: str) -> Verdict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the judge's answer")
    return Verdict.model_validate(json.loads(text[start : end + 1]))


async def judge_item(llm: LLMClient, item: JudgeItem) -> Verdict | str:
    """A verdict, or the reason there is none. One retry on unparseable output."""
    system, user = render(item)
    last = ""
    for _ in range(2):
        try:
            out = await llm.complete(
                system=system,
                user=user,
                prompt_name=PROMPT_NAME,
                prompt_version=PROMPT_VERSION,
                max_tokens=MAX_TOKENS,
            )
            if out.stop_reason == "max_tokens":
                last = f"cut off at max_tokens ({out.output_tokens} tokens)"
                continue
            return parse(out.text)
        except (ValueError, ValidationError) as exc:
            last = f"unparseable: {exc}"[:200]
        except LLMError as exc:
            last = f"{type(exc).__name__}: {exc}"[:200]
    return last


async def judge_items(
    llm: LLMClient, items: list[JudgeItem], concurrency: int = 4
) -> list[Verdict | str]:
    gate = asyncio.Semaphore(concurrency)

    async def one(item: JudgeItem) -> Verdict | str:
        async with gate:
            return await judge_item(llm, item)

    return list(await asyncio.gather(*(one(i) for i in items)))


def rate(verdicts: list[Verdict], field: Literal["tone", "faithfulness"]) -> tuple[int, int]:
    graded = [getattr(v, field) for v in verdicts if getattr(v, field) != "n/a"]
    return sum(g == "pass" for g in graded), len(graded)


async def judge_results(
    llm: LLMClient, results: list[CaseResult], *, dry_run: bool
) -> dict[str, Any]:
    items = items_from_results(results)
    if dry_run:
        return {"headline": f"dry run: {len(items)} replies would be judged", "items": len(items)}
    try:
        verdicts = await judge_items(llm, items)
    except BudgetExceededError as exc:
        return {"headline": f"not run: {exc}"}
    ok = [v for v in verdicts if isinstance(v, Verdict)]
    tone, tone_n = rate(ok, "tone")
    faith, faith_n = rate(ok, "faithfulness")
    return {
        "headline": f"tone {tone}/{tone_n}, faithful {faith}/{faith_n}",
        "prompt": f"{PROMPT_NAME}/{PROMPT_VERSION}",
        "tone_pass_rate": round(tone / tone_n, 3) if tone_n else None,
        "faithfulness_pass_rate": round(faith / faith_n, 3) if faith_n else None,
        "unjudged": len(verdicts) - len(ok),
        "items": [
            {
                **asdict(i),
                "verdict": v.model_dump() if isinstance(v, Verdict) else {"error": v},
            }
            for i, v in zip(items, verdicts, strict=True)
        ],
    }
