"""Answering policy questions: grounding, refusal, repair and cost records.

Every test scripts the model with FakeLLMClient, so these check what *our*
code does with each kind of model reply. Whether a real model refuses well is
measured by evals/policy_answers.py against the golden set (costs cents).
"""

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from frontdesk_agent.answering.service import REFUSAL_TEXT, PolicyAnswerer
from frontdesk_agent.db.models import Tenant
from frontdesk_agent.db.repositories import LLMCallRepository
from frontdesk_agent.dependencies import get_answerer
from frontdesk_agent.llm.base import LLMInvalidOutputError
from frontdesk_agent.llm.fake import FakeLLMClient
from frontdesk_agent.llm.retry import RetryPolicy
from frontdesk_agent.main import create_app
from frontdesk_agent.retrieval.embeddings import default_embedder
from frontdesk_agent.retrieval.ingest import ingest_directory
from frontdesk_agent.retrieval.store import PolicyStore
from tests.factories import make_tenant

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
HAIKU = "claude-haiku-4-5-20251001"


def reply(answerable: bool = True, answer: str = "", citations: list[str] | None = None) -> str:
    return json.dumps({"answerable": answerable, "answer": answer, "citations": citations or []})


def answerer(fake: FakeLLMClient) -> PolicyAnswerer:
    return PolicyAnswerer(fake, default_embedder(), retry_policy=RetryPolicy(max_attempts=1))


@pytest.fixture
async def dental(session: AsyncSession) -> Tenant:
    tenant = await make_tenant(session, "Maple Dental")
    await ingest_directory(
        PolicyStore(session, tenant.id, default_embedder()), POLICIES / "maple-dental"
    )
    await session.commit()
    return tenant


async def test_grounded_answer_maps_citations_to_real_chunks(
    session: AsyncSession, dental: Tenant
) -> None:
    fake = FakeLLMClient([reply(True, "Under 24 hours' notice is a $50 fee.", ["C1"])], model=HAIKU)
    result = await answerer(fake).answer(session, dental.id, "is there a fee to cancel late")

    assert result.answerable and not result.needs_human
    assert result.answer == "Under 24 hours' notice is a $50 fee."
    assert [c.document_slug for c in result.citations] == ["cancellation-policy"]
    assert "Late cancellation fee" in result.citations[0].heading
    # The prompt carried labelled excerpts and the question, nothing else.
    assert "[C1]" in str(fake.calls[0]["user"]) and "<question>" in str(fake.calls[0]["user"])


async def test_every_call_is_recorded_with_cost(session: AsyncSession, dental: Tenant) -> None:
    fake = FakeLLMClient([reply(True, "Yes.", ["C1"])], model=HAIKU)
    await answerer(fake).answer(session, dental.id, "do you bill sun life")
    [call] = await LLMCallRepository(session, dental.id).recent()
    assert call.prompt_name == "policy_answer" and call.succeeded
    assert call.cost_usd is not None and call.cost_usd > 0


async def test_model_says_not_covered_becomes_a_refusal(
    session: AsyncSession, dental: Tenant
) -> None:
    fake = FakeLLMClient([reply(False)])
    result = await answerer(fake).answer(session, dental.id, "how much are implants")
    assert not result.answerable and result.needs_human
    assert result.refusal_reason == "not_in_documents"
    assert result.answer == REFUSAL_TEXT and result.citations == []


@pytest.mark.parametrize(
    "model_reply",
    [
        reply(True, "Implants are $3,000.", []),  # no citation at all
        reply(True, "Implants are $3,000.", ["C9"]),  # cites an excerpt it never saw
        reply(True, "  ", ["C1"]),  # cites, but says nothing
    ],
    ids=["uncited", "invented-citation", "empty-answer"],
)
async def test_ungrounded_answers_are_refused(
    session: AsyncSession, dental: Tenant, model_reply: str
) -> None:
    result = await answerer(FakeLLMClient([model_reply])).answer(
        session, dental.id, "how much are implants"
    )
    assert result.refusal_reason == "ungrounded_answer" and result.needs_human
    assert "$3,000" not in result.answer


async def test_one_repair_attempt_then_success(session: AsyncSession, dental: Tenant) -> None:
    fake = FakeLLMClient(["Sure! The fee is $50.", reply(True, "The fee is $50.", ["C1"])])
    result = await answerer(fake).answer(session, dental.id, "late cancellation fee?")
    assert result.answerable and fake.call_count == 2
    calls = await LLMCallRepository(session, dental.id).recent()
    assert sorted(c.succeeded for c in calls) == [False, True]


async def test_two_invalid_replies_raise_and_are_still_recorded(
    session: AsyncSession, dental: Tenant
) -> None:
    fake = FakeLLMClient(["nope", '{"answerable": true}'])
    with pytest.raises(LLMInvalidOutputError):
        await answerer(fake).answer(session, dental.id, "late cancellation fee?")
    calls = await LLMCallRepository(session, dental.id).recent()
    assert [c.succeeded for c in calls] == [False, False]


async def test_no_documents_means_no_model_call(session: AsyncSession) -> None:
    empty = await make_tenant(session, "Empty")
    fake = FakeLLMClient([])
    result = await answerer(fake).answer(session, empty.id, "what are your hours")
    assert result.refusal_reason == "no_relevant_documents" and fake.call_count == 0


async def test_prompt_only_contains_this_tenants_policies(session: AsyncSession) -> None:
    dental, physio = await make_tenant(session, "Dental"), await make_tenant(session, "Physio")
    for t, slug in ((dental, "maple-dental"), (physio, "northside-physio")):
        await ingest_directory(PolicyStore(session, t.id, default_embedder()), POLICIES / slug)
    fake = FakeLLMClient([reply(False)])
    await answerer(fake).answer(session, physio.id, "how much is teeth whitening")
    prompt = str(fake.calls[0]["user"])
    assert "whitening" not in prompt.split("<question>")[0].lower()
    assert "Teeth Whitening" not in prompt and "$450" not in prompt
    assert "Physio" in str(fake.calls[0]["system"])  # the right business is named


# -- HTTP ----------------------------------------------------------------------


@pytest.fixture
async def policy_api(
    db_url: str, dental: Tenant
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeLLMClient]]:
    """An HTTP client whose app uses a scripted model; tests queue its replies."""
    fake = FakeLLMClient(model=HAIKU)
    app = create_app(db_url)
    app.dependency_overrides[get_answerer] = lambda: answerer(fake)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            yield client, fake


async def test_http_answer_search_and_document_upload(
    policy_api: tuple[httpx.AsyncClient, FakeLLMClient], dental: Tenant
) -> None:
    client, fake = policy_api
    fake.queue(reply(True, "We open at 8 a.m.", ["C1"]))
    base = f"/v1/tenants/{dental.id}/policy"

    r = await client.post(f"{base}/answer", json={"question": "when do you open"})
    assert r.status_code == 200 and r.json()["answerable"] is True
    assert (await client.post(f"{base}/answer", json={"question": "   "})).status_code == 422
    unknown = f"/v1/tenants/{uuid.uuid4()}/policy/answer"
    assert (await client.post(unknown, json={"question": "hours?"})).status_code == 404

    hits = (await client.post(f"{base}/search", json={"question": "parking"})).json()
    assert hits[0]["document_slug"] == "location-and-parking"

    doc = {"slug": "implants", "markdown": "# Implants\n\n## Price\n\nImplants are $3,200."}
    first = (await client.put(f"{base}/documents", json=doc)).json()
    again = (await client.put(f"{base}/documents", json=doc)).json()
    assert first["chunks"] == 1 and not first["skipped"] and again["skipped"]
    hits = (await client.post(f"{base}/search", json={"question": "implant cost"})).json()
    assert hits[0]["document_slug"] == "implants"


async def test_http_model_garbage_is_502(
    policy_api: tuple[httpx.AsyncClient, FakeLLMClient], dental: Tenant
) -> None:
    client, fake = policy_api
    fake.queue("garbage", "more garbage")
    r = await client.post(f"/v1/tenants/{dental.id}/policy/answer", json={"question": "fee?"})
    assert r.status_code == 502
