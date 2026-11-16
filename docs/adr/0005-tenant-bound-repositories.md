# 5. Tenant isolation through tenant-bound repositories

- Status: accepted
- Date: 2026-11-12

## Context

Every business in this service shares one database. A single query that
forgets `WHERE tenant_id = ...` leaks one clinic's clients to another, and
for a front office that means names and phone numbers. The usual fix,
remembering the filter at every call site, fails the first time someone is
in a hurry.

Three options were considered:

1. **Filter at each call site.** Simple, and wrong eventually. Nothing stops
   a missing filter except review.
2. **Postgres row-level security.** Strong, enforced by the database itself.
   But it needs a per-request `SET` of the tenant on the connection, which is
   easy to get wrong with connection pooling, and it hides the rule from
   anyone reading the Python.
3. **Repositories bound to one tenant at construction.** No method accepts
   a tenant argument; every query goes through one `_scoped()` helper and
   every insert through `_claim()`, which overwrites `tenant_id`.

## Decision

Option 3. `ClientRepository(session, tenant_id)` can only ever see that
tenant. Code that wants another tenant has to build a second repository,
which is obvious in a diff.

Two gaps that option 3 does not close by itself are closed explicitly:

- **Cross-tenant foreign keys.** A foreign key only checks that a row
  exists, not whose it is. `AppointmentRepository.book` therefore loads the
  client and provider through *this* tenant's repositories before inserting.
- **Future tables.** `test_schema.py` walks the metadata and fails if any
  table other than `tenants` lacks a non-null `tenant_id` foreign key, or
  lacks an index leading with it.

A row in another tenant raises the same `NotFoundError` as a missing row, so
the API cannot be used to probe which IDs exist.

## Consequences

- Isolation is tested with the other tenant's *real* IDs, the case that
  matters, in `test_tenant_isolation.py`. Deliberately removing the tenant
  filter fails eight of those tests.
- The tenant still comes from the URL in this module. Module 6 binds it to
  the caller's API key; the repositories do not change.
- Row-level security remains a possible second layer once traffic is real.
