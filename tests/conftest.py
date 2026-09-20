"""Shared test fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from frontdesk_agent.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client backed by a freshly built app."""
    with TestClient(create_app()) as test_client:
        yield test_client
