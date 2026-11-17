"""The Twilio webhook and the review API, over HTTP.

Signatures are made with Twilio's own RequestValidator, the way Twilio makes
them, against the public URL the service is configured with.
"""

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from twilio.request_validator import RequestValidator

from frontdesk_agent.config import get_settings
from frontdesk_agent.db.models import ReviewItem, Role, SmsMessage
from frontdesk_agent.dependencies import get_services
from frontdesk_agent.llm.fake import loop_turn, say
from frontdesk_agent.main import create_app
from tests.conftest import act_as
from tests.desk import CLIENT_PHONE, CLINIC_NUMBER, Desk, World, make_desk, make_world, text

TOKEN = "test-auth-token-0123456789abcdef"
PUBLIC = "https://frontdesk.example.com"
URL = f"{PUBLIC}/v1/channels/sms"


def form(body: str = "hi", sid: str = "SM0001", to: str = CLINIC_NUMBER) -> dict[str, str]:
    return {"MessageSid": sid, "From": CLIENT_PHONE, "To": to, "Body": body, "NumMedia": "0"}


def signed(params: dict[str, str], url: str = URL) -> dict[str, str]:
    return {"X-Twilio-Signature": RequestValidator(TOKEN).compute_signature(url, params)}


@pytest.fixture
def twilio_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FDA_TWILIO_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("FDA_PUBLIC_BASE_URL", PUBLIC)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def world(session: AsyncSession) -> World:
    return await make_world(session)


@pytest.fixture
async def desk(db_url: str, world: World) -> AsyncIterator[Desk]:
    d = make_desk(db_url)
    yield d
    await d.dispose()


@pytest.fixture
async def http(db_url: str, desk: Desk, twilio_env: None) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(db_url)
    app.dependency_overrides[get_services] = lambda: desk.services
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://internal") as c:
            yield c


async def count(desk: Desk, model: type) -> int:
    async with desk.services.sessionmaker() as s:
        return int(await s.scalar(select(func.count()).select_from(model)) or 0)


async def test_valid_signature_stores_the_message_and_queues_a_turn(
    http: httpx.AsyncClient, desk: Desk
) -> None:
    params = form("what time do you open?")
    r = await http.post("/v1/channels/sms", data=params, headers=signed(params))
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
    assert "<Response>" in r.text
    assert desk.queue.names() == ["handle_inbound"]
    assert await count(desk, SmsMessage) == 1


async def test_forged_webhook_is_403(http: httpx.AsyncClient, desk: Desk) -> None:
    """Done when: a forged webhook without a valid signature gets a 403."""
    params = form()
    no_sig = await http.post("/v1/channels/sms", data=params)
    wrong = await http.post("/v1/channels/sms", data=params, headers={"X-Twilio-Signature": "x"})
    tampered = await http.post(
        "/v1/channels/sms", data={**params, "Body": "cancel everything"}, headers=signed(params)
    )
    other_url = await http.post(
        "/v1/channels/sms", data=params, headers=signed(params, "https://evil.example/x")
    )
    assert [r.status_code for r in (no_sig, wrong, tampered, other_url)] == [403] * 4
    assert desk.queue.jobs == [] and await count(desk, SmsMessage) == 0


async def test_twilio_retry_is_stored_once(http: httpx.AsyncClient, desk: Desk) -> None:
    params = form(sid="SMretry")
    for _ in range(3):
        r = await http.post("/v1/channels/sms", data=params, headers=signed(params))
        assert r.status_code == 200
    assert await count(desk, SmsMessage) == 1 and desk.queue.names() == ["handle_inbound"]


async def test_unknown_number_is_404(http: httpx.AsyncClient) -> None:
    params = form(to="+19995550100")
    r = await http.post("/v1/channels/sms", data=params, headers=signed(params))
    assert r.status_code == 404


async def test_no_auth_token_refuses_everything(
    db_url: str, desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FDA_TWILIO_AUTH_TOKEN", "")
    get_settings.cache_clear()
    app = create_app(db_url)
    app.dependency_overrides[get_services] = lambda: desk.services
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://i") as c:
                r = await c.post("/v1/channels/sms", data=form())
        assert r.status_code == 503
    finally:
        get_settings.cache_clear()


async def test_review_api_list_edit_approve_and_conflicts(
    http: httpx.AsyncClient, desk: Desk, world: World, session: AsyncSession
) -> None:
    desk.llm.script(say("Sure, I'll refund you right away!"))
    await text(desk, world, "can I get a refund for last week")
    await act_as(http, session, world.tenant_id, Role.REVIEWER)
    base = "/v1/reviews"

    [item] = (await http.get(base)).json()
    assert item["kind"] == "money" and item["proposed_reply"].startswith("Sure")
    assert (await http.get(f"{base}?status=approved")).json() == []

    edited = "Refunds are handled by our office manager; she will call you today."
    r = await http.post(f"{base}/{item['id']}/approve", json={"reply": edited})
    assert r.status_code == 200 and r.json()["final_reply"] == edited
    assert r.json()["decided_by"] == "key:test-reviewer"  # from the credential, not the body
    assert desk.sender.bodies() == [edited]  # the edit was sent, not the draft

    again = await http.post(f"{base}/{item['id']}/approve", json={})
    assert again.status_code == 409

    convo = r.json()["conversation_id"]
    log = (await http.get(f"/v1/conversations/{convo}/messages")).json()
    assert [m["direction"] for m in log] == ["inbound", "outbound"]


async def test_approving_an_item_with_no_draft_needs_a_reply(
    http: httpx.AsyncClient, desk: Desk, world: World, session: AsyncSession
) -> None:
    desk.llm.script(*[loop_turn() for _ in range(10)])
    await text(desk, world, "hello")  # step limit: no draft
    await act_as(http, session, world.tenant_id, Role.REVIEWER)
    base = "/v1/reviews"
    [item] = (await http.get(base)).json()
    r = await http.post(f"{base}/{item['id']}/approve", json={})
    assert r.status_code == 422
    ok = await http.post(f"{base}/{item['id']}/approve", json={"reply": "Hi! How can we help?"})
    assert ok.status_code == 200
    assert await count(desk, ReviewItem) == 1
