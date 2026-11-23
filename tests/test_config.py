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


def test_the_fake_model_is_refused_in_production() -> None:
    """A load-test setting left on in production would text clients canned nonsense."""
    with pytest.raises(ValueError, match="never production"):
        Settings(env="production", llm_provider="fake")
    assert Settings(env="staging", llm_provider="fake").llm_provider == "fake"
