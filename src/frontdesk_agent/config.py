"""Configuration, loaded from environment variables only (twelve-factor).

Never read a .env file in production; it exists for local development only.
Secrets belong in the environment or a secret manager, never in git.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Every field must have a safe default or be required explicitly, so a
    missing variable fails loudly at startup rather than quietly at request time.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FDA_", extra="ignore")

    env: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    service_name: str = "frontdesk-agent"

    # Module 2 fills this in; the default keeps local startup working.
    database_url: str = "postgresql+psycopg://frontdesk:frontdesk@localhost:5432/frontdesk"

    # LLM provider. The key has no default on purpose: a missing key must fail
    # loudly rather than silently fall back to some other credential.
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_timeout_s: float = 20.0


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so config is parsed once per process."""
    return Settings()
