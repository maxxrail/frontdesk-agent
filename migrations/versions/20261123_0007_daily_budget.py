"""daily model budget per tenant

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_KINDS = "'late_cancellation', 'money', 'medical', 'escalation'"


def upgrade() -> None:
    op.add_column("tenants", sa.Column("daily_budget_usd", sa.Numeric(10, 2), nullable=True))
    op.execute("ALTER TYPE review_kind ADD VALUE IF NOT EXISTS 'budget'")


def downgrade() -> None:
    # Postgres cannot drop an enum value, so rebuild the type. Budget holds
    # become escalations rather than being deleted.
    op.execute("UPDATE review_items SET kind = 'escalation' WHERE kind = 'budget'")
    op.execute("ALTER TYPE review_kind RENAME TO review_kind_old")
    op.execute(f"CREATE TYPE review_kind AS ENUM ({OLD_KINDS})")
    op.execute(
        "ALTER TABLE review_items ALTER COLUMN kind TYPE review_kind USING kind::text::review_kind"
    )
    op.execute("DROP TYPE review_kind_old")
    op.drop_column("tenants", "daily_budget_usd")
