# 16. Credentials carry the tenant: API keys, login tokens, two roles

- Status: accepted
- Date: 2026-11-17

## Context

Until Module 6 the tenant came from the URL (`/v1/tenants/{tenant_id}/...`)
and there was no authentication. Isolation held in the repositories
(ADR 0005), but any caller could name any tenant, decide any review, and
spend model money on `/v1/drafts/reminder`. The roadmap asks for API keys
for machine clients, logins for dashboard users with reviewer and admin
roles, and the tenant taken from the credential.

## Decision

**The tenant comes only from the credential.** Routes lose the tenant
segment (`/v1/reviews`, `/v1/clients`, ...). Every tenant endpoint depends
on `Reviewer` or `Admin` (`api/auth.py`), which yields a `Principal`
(tenant, role, actor). A request has no way to name a tenant; a
`tenant_id` query parameter is ignored (tested).

**Machine clients: API keys.** `fda_<12-hex prefix>_<43-char secret>`, 256
random bits. Stored: the prefix (to find the row) and a SHA-256 digest,
never the key, which is shown once at creation. A fast hash is right here:
with 256 bits there is nothing to brute-force, and a check costs 0.8 µs.
Keys can be revoked; a revoked key fails at once.

**People: password login, short-lived tokens.** Passwords are hashed with
scrypt (N=2^14, r=8, p=1, 16-byte salt) from the standard library, about
55 ms per hash on the dev machine, so no new dependency for hashing.
`POST /v1/auth/login` returns an HS256 JWT (PyJWT, the one new dependency)
valid for 60 minutes. The token only names the user; every request re-reads
the user, so deactivating someone or changing their role applies on the
next request, not when the token expires (tested). The algorithm is pinned,
so `alg: none` is rejected. An unknown email is checked against a dummy
hash, so a failed login takes as long either way and returns the same body.

**Two roles.** Reviewers read clients, appointments, reviews and
conversations, decide reviews, and ask policy questions. Admins can also
book and cancel through the API, change policy documents (which steer what
the agent tells clients), read model costs, and manage keys, users and the
audit trail. The reviewer recorded on a decision is the credential's
actor (`user:<email>` or `key:<name>`), no longer a name typed in the body.

**Tenants are created from the command line**, not over HTTP
(`scripts/create_tenant.py`, `scripts/seed.py`), with a first admin user
(generated password) and an admin API key, each shown once. No tenant's
credential can create another tenant.

**Two lookups cross tenants**, in `CredentialDirectory`: a user by email at
login, and a key by prefix. They establish the tenant, so they cannot be
tenant-bound, like `TenantDirectory` in ADR 0005. Emails are unique across
tenants for the same reason.

**The server refuses to start without `FDA_JWT_SECRET`**, rather than sign
tokens with a default. CI and tests set a throwaway value.

## Options not taken

- **OAuth / an identity provider.** The right answer for a real product,
  but it needs an account and a hosted service; nothing here depends on
  the token format, so it can replace the login endpoint later.
- **Tokens that carry the role.** Smaller per-request cost (no user read),
  but a demoted admin would keep admin rights for up to 60 minutes.
- **bcrypt or argon2.** Equally good, but a new native dependency; scrypt is
  in the standard library.

## Consequences

- Breaking API change: every tenant URL moved, and review decisions no
  longer take a `reviewer` field.
- The Twilio webhook is unchanged: Twilio's signature authenticates it, and
  the tenant comes from the number texted (ADR 0012).
- Password change and reset, and login rate limiting, are not built here;
  rate limiting arrives later in this module.
