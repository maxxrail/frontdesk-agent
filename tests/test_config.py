"""Configuration tests."""

import pytest

from frontdesk_agent.config import Settings


def test_defaults_are_local() -> None:
    settings = Settings()

    assert settings.env == "local"
    assert settings.service_name == "frontdesk-agent"


def test_env_is_validated() -> None:
    with pytest.raises(ValueError):
        Settings(env="prod")  # type: ignore[arg-type]


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDA_ENV", "staging")

    assert Settings().env == "staging"
