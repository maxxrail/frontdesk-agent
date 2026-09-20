# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-09-18

### Added

- `POST /v1/drafts/reminder`: drafts an appointment reminder SMS as validated
  JSON, never free text.
- `LLMClient` protocol with an Anthropic implementation and a scripted
  `FakeLLMClient`; no test or CI run calls a real model.
- Retry policy with exponential backoff and full jitter, retrying only
  timeouts, rate limits and 5xx responses.
- Versioned prompt files under `prompts/`, rendered with fail-loud placeholder
  substitution; prompt name and version recorded on every call.
- One repair attempt when model output fails schema validation, then 502.
- Token counts, latency and model recorded on every completion.
- ADRs 0003 and 0004.

### Changed

- `anthropic` added as a dependency; `uv.lock` regenerated.
- The LLM interface does not expose `temperature`, which the current SDK's
  `messages.create` no longer accepts.

## [0.0.0] - 2026-09-18

### Added

- Repository scaffold: `src/` layout, pyproject with ruff, mypy and pytest.
- FastAPI app with `/healthz` and `/readyz`, built through a factory.
- Environment-based configuration with Pydantic settings, no secrets in git.
- Multi-stage Dockerfile running as a non-root user, plus docker-compose with
  Postgres and pgvector for local development.
- GitHub Actions CI: lint, format check, types, tests, Docker build, and a
  container health check.
- Issue and pull request templates, pre-commit hooks, ADRs 0001 and 0002.
