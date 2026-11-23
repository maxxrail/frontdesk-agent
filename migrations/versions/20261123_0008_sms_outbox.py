"""queued status for outbound SMS (the outbox)

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_STATUSES = "'received', 'held', 'handled', 'sent', 'failed'"


def upgrade() -> None:
    # A new enum value cannot be used in the transaction that adds it, and the
    # index below uses it; so add it in its own committed step.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE message_status ADD VALUE IF NOT EXISTS 'queued'")
    # The sweeper looks for old queued rows; keep that scan small.
    op.execute(
        "CREATE INDEX ix_sms_messages_queued ON sms_messages (created_at) WHERE status = 'queued'"
    )


def downgrade() -> None:
    # Postgres cannot drop an enum value, so rebuild the type. A queued text
    # was never sent; record it as failed rather than lose the row.
    op.execute("DROP INDEX ix_sms_messages_queued")
    op.execute("UPDATE sms_messages SET status = 'failed' WHERE status = 'queued'")
    op.execute("ALTER TYPE message_status RENAME TO message_status_old")
    op.execute(f"CREATE TYPE message_status AS ENUM ({OLD_STATUSES})")
    op.execute(
        "ALTER TABLE sms_messages ALTER COLUMN status TYPE message_status"
        " USING status::text::message_status"
    )
    op.execute("DROP TYPE message_status_old")
