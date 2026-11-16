"""Structural rules about the schema. No database needed: these read metadata.

They exist so that a future table cannot quietly break tenant isolation.
"""

from sqlalchemy import ForeignKey

from frontdesk_agent.db import models  # noqa: F401
from frontdesk_agent.db.base import Base

UNSCOPED = {"tenants"}


def test_every_table_except_tenants_has_a_non_null_tenant_fk() -> None:
    offenders = []
    for name, table in Base.metadata.tables.items():
        if name in UNSCOPED:
            continue
        col = table.columns.get("tenant_id")
        fks: set[ForeignKey] = set(col.foreign_keys) if col is not None else set()
        if col is None or col.nullable or not any(fk.target_fullname == "tenants.id" for fk in fks):
            offenders.append(name)
    assert offenders == [], f"tables missing a non-null tenant_id FK: {offenders}"


def test_every_tenant_scoped_table_indexes_tenant_id() -> None:
    """Every scoped query filters on tenant_id, so it must lead an index."""
    missing = []
    for name, table in Base.metadata.tables.items():
        if name in UNSCOPED:
            continue
        leading = {idx.columns.values()[0].name for idx in table.indexes}
        if "tenant_id" not in leading:
            missing.append(name)
    assert missing == []


def test_schema_has_the_expected_tables() -> None:
    assert set(Base.metadata.tables) == {
        "tenants",
        "clients",
        "providers",
        "appointments",
        "waitlist_entries",
        "llm_calls",
        "policy_documents",
        "policy_chunks",
        "provider_schedules",
        "conversations",
        "sms_messages",
        "review_items",
        "slot_offers",
        "client_response_stats",
    }
