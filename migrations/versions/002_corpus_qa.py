"""Add corpus QA fields.

Revision ID: 002
Revises: 001
Create Date: 2026-05-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column(
            "qa_status",
            sa.String(40),
            nullable=False,
            server_default="auto_generated",
        ),
    )
    op.add_column(
        "dataset_cards",
        sa.Column(
            "qa_status",
            sa.String(40),
            nullable=False,
            server_default="auto_generated",
        ),
    )
    for column_name in [
        "task_labels_verified",
        "modality_labels_verified",
        "behavior_labels_verified",
        "brain_regions_verified",
        "linked_papers_verified",
        "notebook_tested",
    ]:
        op.add_column(
            "dataset_cards",
            sa.Column(column_name, sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    op.add_column(
        "dataset_cards",
        sa.Column("reviewer_notes", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("dataset_cards", "reviewer_notes")
    for column_name in [
        "notebook_tested",
        "linked_papers_verified",
        "brain_regions_verified",
        "behavior_labels_verified",
        "modality_labels_verified",
        "task_labels_verified",
    ]:
        op.drop_column("dataset_cards", column_name)
    op.drop_column("dataset_cards", "qa_status")
    op.drop_column("datasets", "qa_status")
