"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2024-01-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Datasets table
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(80), nullable=False),
        sa.Column("source_id", sa.String(255), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.String(1000), nullable=True),
        sa.Column("license", sa.String(255), nullable=True),
        sa.Column("species", sa.JSON(), nullable=True),
        sa.Column("modalities", sa.JSON(), nullable=True),
        sa.Column("brain_regions", sa.JSON(), nullable=True),
        sa.Column("tasks", sa.JSON(), nullable=True),
        sa.Column("behaviors", sa.JSON(), nullable=True),
        sa.Column("data_standards", sa.JSON(), nullable=True),
        sa.Column("has_behavior", sa.Boolean(), nullable=True),
        sa.Column("has_trials", sa.Boolean(), nullable=True),
        sa.Column("has_raw_data", sa.Boolean(), nullable=True),
        sa.Column("has_processed_data", sa.Boolean(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_source", "datasets", ["source"])
    op.create_index("ix_datasets_source_id", "datasets", ["source_id"])

    # Dataset assets table
    op.create_table(
        "dataset_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("path", sa.String(1000), nullable=False),
        sa.Column("asset_type", sa.String(120), nullable=True),
        sa.Column("file_format", sa.String(80), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("subject_id", sa.String(255), nullable=True),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("modality", sa.String(120), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
    )
    op.create_index("ix_dataset_assets_dataset_id", "dataset_assets", ["dataset_id"])

    # Papers table
    op.create_table(
        "papers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("openalex_id", sa.String(255), nullable=True),
        sa.Column("doi", sa.String(255), nullable=True),
        sa.Column("title", sa.String(1000), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("authors_json", sa.JSON(), nullable=True),
        sa.Column("url", sa.String(1000), nullable=True),
        sa.Column("concepts", sa.JSON(), nullable=True),
        sa.Column("linked_dataset_ids", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_papers_openalex_id", "papers", ["openalex_id"])
    op.create_index("ix_papers_doi", "papers", ["doi"])

    # Ontology terms table
    op.create_table(
        "ontology_terms",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("term_id", sa.String(255), nullable=False),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("category", sa.String(255), nullable=True),
        sa.Column("parent_id", sa.String(255), nullable=True),
        sa.Column("synonyms", sa.JSON(), nullable=True),
        sa.Column("definition", sa.Text(), nullable=True),
        sa.Column("examples", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("term_id"),
    )

    # Dataset cards table
    op.create_table(
        "dataset_cards",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("why_relevant", sa.JSON(), nullable=True),
        sa.Column("analysis_readiness_score", sa.Integer(), nullable=False),
        sa.Column("strengths", sa.JSON(), nullable=True),
        sa.Column("limitations", sa.JSON(), nullable=True),
        sa.Column("missing_fields", sa.JSON(), nullable=True),
        sa.Column("suggested_analyses", sa.JSON(), nullable=True),
        sa.Column("provenance_json", sa.JSON(), nullable=True),
        sa.Column("card_markdown", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
    )
    op.create_index("ix_dataset_cards_dataset_id", "dataset_cards", ["dataset_id"])

    # Embeddings table
    op.create_table(
        "embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("entity_type", sa.String(80), nullable=False),
        sa.Column("entity_id", sa.Uuid(), nullable=False),
        sa.Column("text_for_embedding", sa.Text(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),  # Use JSON for SQLite, Vector for Postgres
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_embeddings_entity_type", "embeddings", ["entity_type"])
    op.create_index("ix_embeddings_entity_id", "embeddings", ["entity_id"])

    # Search logs table
    op.create_table(
        "search_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("parsed_intent_json", sa.JSON(), nullable=True),
        sa.Column("result_ids", sa.JSON(), nullable=True),
        sa.Column("result_payload_json", sa.JSON(), nullable=True),
        sa.Column("user_feedback_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("search_logs")
    op.drop_table("embeddings")
    op.drop_table("dataset_cards")
    op.drop_table("ontology_terms")
    op.drop_table("papers")
    op.drop_table("dataset_assets")
    op.drop_table("datasets")
