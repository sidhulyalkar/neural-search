"""Evidence-backed file inspection claim schemas."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ClaimType = Literal[
    "metadata_presence",
    "usability",
    "label",
    "analysis_affordance",
    "warning",
]


class FileInspectionClaim(BaseModel):
    """A conservative claim derived from inspecting dataset files or manifests."""

    claim_id: str
    dataset_id: str
    claim_type: ClaimType
    field: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    source_path: str
    extractor: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator(
        "claim_id",
        "dataset_id",
        "field",
        "value",
        "evidence",
        "source_path",
        "extractor",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


def make_claim_id(
    dataset_id: str,
    field: str,
    value: Any,
    source_path: str | Path,
) -> str:
    """Create stable claim IDs for deterministic fixture builds."""

    key = "|".join([dataset_id, field, str(value), str(source_path)])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"claim:{dataset_id}:{field}:{digest}"
