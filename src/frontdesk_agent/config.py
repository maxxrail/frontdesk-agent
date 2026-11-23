"""Configuration, loaded from environment variables only (twelve-factor).

Never read a .env file in production; it exists for local development only.
Secrets belong in the environment or a secret manager, never in git.
"""

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
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
    # The cheap model that runs the agent's lookup-only steps; every reply and
    # action stays on llm_model (ADR 0019). Empty: llm_model does everything.
    llm_lookup_model: str = "claude-haiku-5-5"
    llm_timeout_s: float = 20.0
    # "fake" answers every agent turn with one lookup and a canned reply after
    # a realistic delay, at no cost: for load tests (loadtest/). Refused in
    # production, where it would answer clients with nonsense.
    llm_provider: Literal["anthropic", "fake"] = "anthropic"
    llm_fake_latency_s: float = 0.9
    # The agent prompt version (prompts/agent/<version>.md). Changing it, like
    # changing llm_model, runs the eval suite in CI (Module 5).
    agent_prompt_version: str = "v3"

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

    # Rate limits (Module 6, ADR 0018), checked before any model call. Per
    # phone: more texts than this in the window pauses the conversation for a
    # person. Per tenant: more model calls than this in the window defers
    # new messages instead of dropping them.
    rate_phone_messages: int = 12
    rate_phone_window_s: int = 600
    rate_tenant_model_calls: int = 1000
    rate_tenant_window_s: int = 3600
    # Daily model spend per clinic (USD, local day) before replies degrade to
    # a fixed text and staff take over (ADR 0019). A tenant row can override it.
    tenant_daily_budget_usd: Decimal = Decimal("5.00")

    # Error reporting (Module 7). Empty: nothing is sent. Events are redacted
    # before sending (errortracking.py). SecretStr keeps it out of reprs.
    sentry_dsn: SecretStr = SecretStr("")
    # OTLP/HTTP collector for traces, e.g. http://localhost:4318 (Jaeger in
    # docker compose). Empty: spans are made but not exported.
    otel_endpoint: str = ""
    # The worker serves /metrics on this port (0: off). The API serves its own.
    metrics_port: int = 0
    # Where eval results files are read for the eval-score metric.
    eval_results_dir: str = "evals/results"

    # Background jobs: "redis" for the arq worker, "inline" to run them inside
    # the API process (local development without Redis).
    job_queue: Literal["redis", "inline"] = "inline"
    redis_url: str = "redis://localhost:6379/0"

    @model_validator(mode="after")
    def _no_fake_model_in_production(self) -> "Settings":
        if self.env == "production" and self.llm_provider == "fake":
            raise ValueError("FDA_LLM_PROVIDER=fake is for load tests, never production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so config is parsed once per process."""
    return Settings()
