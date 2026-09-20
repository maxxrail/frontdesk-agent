# 1. Record architecture decisions

- Status: accepted
- Date: 2026-09-18

## Context

This project is a portfolio piece as much as a working service. A reviewer
reading the repo cannot see the alternatives I considered or why I rejected
them, and in six months neither will I.

## Decision

Every decision that would be expensive to reverse gets a short ADR in
`docs/adr/`, numbered in order, following Michael Nygard's format: context,
decision, consequences. Short is the point. A decision that takes a page to
explain probably needs a smaller decision first.

An ADR is never edited after it is accepted. If the decision changes, write a
new ADR that supersedes it and link both ways.

## Consequences

- Reviewers can trace the reasoning without reading the diff history.
- Each module of the course produces at least one ADR, so the repo accumulates
  a written record of judgement, not just code.
- ADRs that turned out wrong stay in the repo, superseded. That history is the
  useful part.
