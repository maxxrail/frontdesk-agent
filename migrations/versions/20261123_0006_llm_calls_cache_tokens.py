"""llm_calls cache tokens

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-23 16:37:04.965027
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_calls",
        sa.Column("cache_read_tokens", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "llm_calls",
        sa.Column("cache_write_tokens", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("llm_calls", "cache_write_tokens")
    op.drop_column("llm_calls", "cache_read_tokens")
