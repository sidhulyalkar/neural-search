"""Database foundation and portable column types."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

JSONType = JSON().with_variant(JSONB, "postgresql")
UUIDType = Uuid(as_uuid=True)


def vector_type(dimensions: int | None = None) -> Any:
    """Return JSON locally and pgvector for PostgreSQL when installed.

    The fallback keeps local tests and SQLite demos dependency-free while preserving
    the model surface needed for Postgres + pgvector deployments.
    """

    try:
        from pgvector.sqlalchemy import Vector
    except Exception:
        return JSONType
    # Start from a fresh JSON() so we don't hit "dialect already present"
    # on the module-level JSONType that already carries JSONB for postgresql.
    return JSON().with_variant(Vector(dimensions), "postgresql")


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""
