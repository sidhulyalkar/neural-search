"""SQLAlchemy models for the Neural Search backend core."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from neural_search.db import Base, JSONType, UUIDType, vector_type


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1000))
    license: Mapped[str | None] = mapped_column(String(255))
    species: Mapped[list[str]] = mapped_column(JSONType, default=list)
    modalities: Mapped[list[str]] = mapped_column(JSONType, default=list)
    brain_regions: Mapped[list[str]] = mapped_column(JSONType, default=list)
    tasks: Mapped[list[str]] = mapped_column(JSONType, default=list)
    behaviors: Mapped[list[str]] = mapped_column(JSONType, default=list)
    data_standards: Mapped[list[str]] = mapped_column(JSONType, default=list)
    has_behavior: Mapped[bool] = mapped_column(Boolean, default=False)
    has_trials: Mapped[bool] = mapped_column(Boolean, default=False)
    has_raw_data: Mapped[bool] = mapped_column(Boolean, default=False)
    has_processed_data: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    assets: Mapped[list["DatasetAsset"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    cards: Mapped[list["DatasetCard"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetAsset(Base):
    __tablename__ = "dataset_assets"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        UUIDType, ForeignKey("datasets.id"), nullable=False, index=True
    )
    path: Mapped[str] = mapped_column(String(1000), nullable=False)
    asset_type: Mapped[str | None] = mapped_column(String(120))
    file_format: Mapped[str | None] = mapped_column(String(80))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    subject_id: Mapped[str | None] = mapped_column(String(255))
    session_id: Mapped[str | None] = mapped_column(String(255))
    modality: Mapped[str | None] = mapped_column(String(120))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    dataset: Mapped[Dataset] = relationship(back_populates="assets")


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    openalex_id: Mapped[str | None] = mapped_column(String(255), index=True)
    doi: Mapped[str | None] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)
    publication_year: Mapped[int | None] = mapped_column(Integer)
    authors_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    url: Mapped[str | None] = mapped_column(String(1000))
    concepts: Mapped[list[str]] = mapped_column(JSONType, default=list)
    linked_dataset_ids: Mapped[list[str]] = mapped_column(JSONType, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class OntologyTerm(Base):
    __tablename__ = "ontology_terms"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    term_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(255))
    parent_id: Mapped[str | None] = mapped_column(String(255))
    synonyms: Mapped[list[str]] = mapped_column(JSONType, default=list)
    definition: Mapped[str | None] = mapped_column(Text)
    examples: Mapped[list[str]] = mapped_column(JSONType, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class DatasetCard(TimestampMixin, Base):
    __tablename__ = "dataset_cards"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        UUIDType, ForeignKey("datasets.id"), nullable=False, index=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    why_relevant: Mapped[list[str]] = mapped_column(JSONType, default=list)
    analysis_readiness_score: Mapped[int] = mapped_column(Integer, nullable=False)
    strengths: Mapped[list[str]] = mapped_column(JSONType, default=list)
    limitations: Mapped[list[str]] = mapped_column(JSONType, default=list)
    missing_fields: Mapped[list[str]] = mapped_column(JSONType, default=list)
    suggested_analyses: Mapped[list[str]] = mapped_column(JSONType, default=list)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    card_markdown: Mapped[str] = mapped_column(Text, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="cards")


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    entity_id: Mapped[UUID] = mapped_column(UUIDType, nullable=False, index=True)
    text_for_embedding: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(vector_type(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[UUID] = mapped_column(UUIDType, primary_key=True, default=uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_intent_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    result_ids: Mapped[list[str]] = mapped_column(JSONType, default=list)
    result_payload_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    user_feedback_json: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

