"""Passwords, API keys and login tokens (ADR 0016). No web framework here.

- Passwords: scrypt from the standard library, a memory-hard KDF, with a
  random salt per password, compared in constant time.
- API keys: ``fda_<prefix>_<secret>`` with a 256-bit secret. Only the SHA-256
  digest is stored. A fast hash is right for a long random secret (nothing
  to brute-force); the prefix is public and used to find the row.
- Login tokens: HS256 JWTs that expire. The token only names the user; every
  request re-reads the user, so deactivating someone or changing their role
  takes effect at once, not when the token runs out.
"""

import base64
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from typing import Any

import jwt

# scrypt cost: 2^14 x 8 is the commonly recommended interactive setting
# (about 16 MiB and tens of milliseconds per hash).
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32
MIN_PASSWORD_LENGTH = 12

KEY_PREFIX = "fda"
_ISSUER = "frontdesk-agent"
_AUDIENCE = "dashboard"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), dklen=_DKLEN
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, _unb64(expected))


@cache
def dummy_password_hash() -> str:
    """Checked against when an email is unknown, so a failed login takes as
    long whether or not the account exists."""
    return hash_password(secrets.token_urlsafe(16))


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class NewApiKey:
    key: str  # shown once, never stored
    prefix: str
    digest: str


def new_api_key() -> NewApiKey:
    prefix = secrets.token_hex(6)
    key = f"{KEY_PREFIX}_{prefix}_{secrets.token_urlsafe(32)}"
    return NewApiKey(key, prefix, key_digest(key))


def key_prefix(key: str) -> str | None:
    """The lookup prefix of a well-formed key, or None."""
    parts = key.split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_PREFIX or len(parts[1]) != 12 or not parts[2]:
        return None
    return parts[1]


def key_matches(key: str, digest: str) -> bool:
    return hmac.compare_digest(key_digest(key), digest)


def issue_token(
    *, user_id: uuid.UUID, tenant_id: uuid.UUID, secret: str, ttl: timedelta, now: datetime
) -> str:
    claims = {
        "sub": str(user_id),
        "tid": str(tenant_id),
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def read_token(token: str, secret: str) -> dict[str, Any]:
    """The claims of a valid, unexpired token; raises jwt.InvalidTokenError.

    The algorithm is pinned, so a token signed with "none" or another
    algorithm is rejected rather than trusted.
    """
    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        audience=_AUDIENCE,
        issuer=_ISSUER,
        options={"require": ["sub", "tid", "exp", "iat"]},
    )
