"""Configuration, loaded from environment variables only (twelve-factor).

Never read a .env file in production; it exists for local development only.
Secrets belong in the environment or a secret manager, never in git.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
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
    # The agent prompt version (prompts/agent/<version>.md). Changing it, like
    # changing llm_model, runs the eval suite in CI (Module 5).
    agent_prompt_version: str = "v2"

    # Retrieval, chosen from measurements (ADR 0008). Vector search beat hybrid
    # on the golden set with the current embedder; hybrid stays one setting away.
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    retrieval_k: int = 5
    # Below every answerable question in the golden set (lowest: 0.14). It only
    # stops gibberish from costing a model call; the model decides what the
    # documents do not cover.
    retrieval_min_similarity: float = 0.12

    # SMS channel (Module 4). The auth token has no default: without it the
    # webhook refuses every request rather than accepting unsigned ones.
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    # The public URL Twilio calls, e.g. https://abc.ngrok.app. Twilio signs the
    # exact URL it requested, which behind a tunnel or load balancer is not
    # the URL this process sees, so it must be configured, not inferred.
    public_base_url: str = "http://localhost:8000"
    # "twilio" sends real texts; "console" prints them (local development).
    sms_sender: Literal["twilio", "console"] = "console"

    # Auth (Module 6, ADR 0016). Signs dashboard login tokens. No default: the
    # app refuses to start without it rather than sign with a guessable key.
    # SecretStr keeps it out of reprs and logs.
    jwt_secret: SecretStr = SecretStr("")
    jwt_ttl_minutes: int = 60

    # Background jobs: "redis" for the arq worker, "inline" to run them inside
    # the API process (local development without Redis).
    job_queue: Literal["redis", "inline"] = "inline"
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so config is parsed once per process."""
    return Settings()
