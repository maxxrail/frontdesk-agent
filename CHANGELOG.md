# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
