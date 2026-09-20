"""Health endpoint tests.

These are deliberately small. Their job is to prove the CI pipeline runs
real tests against a real app from day one.
"""

from fastapi.testclient import TestClient

from frontdesk_agent import __version__


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "frontdesk-agent"
    assert body["version"] == __version__


def test_readyz_returns_ok(client: TestClient) -> None:
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404
