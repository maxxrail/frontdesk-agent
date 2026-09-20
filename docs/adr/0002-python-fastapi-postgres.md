# 2. Python, FastAPI and Postgres for the service

- Status: accepted
- Date: 2026-09-18

## Context

The service needs an HTTP API, background work, a relational store and vector
search, and it has to be readable by engineers at the companies I am targeting.
Several of their job postings name Python with FastAPI, Postgres and AWS
explicitly.

## Decision

Python 3.12 with FastAPI for the API, Postgres with the pgvector extension for
both relational data and embeddings, and Docker for local and deployed
environments.

Postgres holds the vectors rather than a dedicated vector database. At this
scale pgvector is sufficient, and one datastore means one backup story, one
migration tool and one set of credentials.

## Consequences

- Tooling matches what the target employers use, so the repo reads as relevant.
- Type hints plus Pydantic give validation at the edges and a typed core, which
  mypy can check in CI.
- If retrieval outgrows pgvector, moving to a dedicated vector store is a
  contained change behind the retrieval interface. A new ADR would record it.
