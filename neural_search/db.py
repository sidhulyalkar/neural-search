"""Database foundation and portable column types."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


JSONType = JSON().with_variant(JSONB, "postgresql")
UUIDType = Uuid(as_uuid=True)


def vector_type(dimensions: int | None = None) -> Any:
    """Return a pgvector type when installed, otherwise a JSON fallback.

    The fallback keeps local tests and SQLite demos dependency-free while preserving
    the model surface needed for Postgres + pgvector deployments.
    """

    try:
        from pgvector.sqlalchemy import Vector
    except Exception:
        return JSONType
    return Vector(dimensions)


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""

