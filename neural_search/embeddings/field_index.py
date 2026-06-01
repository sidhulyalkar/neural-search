"""Field-specific embedding cache records for normalized corpus data."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from neural_search.embeddings.base import EmbeddingProvider
from neural_search.normalized import NormalizedRecord, load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord

DATASET_EMBEDDING_FIELDS = [
    "title",
    "description",
    "tasks",
    "behavioral_events",
    "modalities",
    "brain_regions",
    "analysis_goals",
    "data_standards",
    "combined_scientific_summary",
]

PAPER_EMBEDDING_FIELDS = [
    "title",
    "abstract",
    "extracted_labels",
    "combined_scientific_summary",
]


class FieldEmbeddingRecord(BaseModel):
    """One vector for one normalized source record field."""

    record_id: str
    record_type: Literal["dataset", "paper"]
    field_name: str
    text: str
    embedding: list[float]
    provider_name: str
    model_name: str
    dimension: int = Field(gt=0)
    normalize: bool
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def embedding_matches_dimension(self) -> FieldEmbeddingRecord:
        if len(self.embedding) != self.dimension:
            raise ValueError("embedding length must match dimension")
        return self


def _record_id(record: NormalizedRecord) -> str:
    if isinstance(record, NormalizedDatasetRecord):
        return record.dataset_id
    return record.paper_id


def _label_text(labels: Sequence[object]) -> str:
    pieces: list[str] = []
    for label in labels:
        for attr in ("label", "id", "evidence_text"):
            value = getattr(label, attr, None)
            if value:
                pieces.append(str(value))
    return " ".join(dict.fromkeys(pieces))


def _affordance_text(record: NormalizedDatasetRecord) -> str:
    pieces: list[str] = []
    for affordance in record.analysis_affordances:
        pieces.extend(
            [
                affordance.analysis_id,
                affordance.support_level,
                *affordance.required_fields_present,
                *affordance.helpful_fields_present,
                *affordance.evidence,
            ]
        )
    return " ".join(dict.fromkeys(piece for piece in pieces if piece))


def field_texts_for_record(record: NormalizedRecord) -> dict[str, str]:
    """Return non-empty v0.4 embedding fields for a normalized record."""

    if isinstance(record, NormalizedDatasetRecord):
        fields = {
            "title": record.title,
            "description": record.description or "",
            "tasks": _label_text(record.tasks),
            "behavioral_events": _label_text(record.behavioral_events),
            "modalities": _label_text(record.modalities),
            "brain_regions": _label_text(record.brain_regions),
            "analysis_goals": " ".join(
                [
                    _label_text(record.analysis_goals),
                    _affordance_text(record),
                ]
            ),
            "data_standards": _label_text(record.data_standards),
        }
        fields["combined_scientific_summary"] = " ".join(
            value for value in fields.values() if value
        )
        return {
            field: fields.get(field, "").strip()
            for field in DATASET_EMBEDDING_FIELDS
            if fields.get(field, "").strip()
        }

    fields = {
        "title": record.title,
        "abstract": record.abstract or "",
        "extracted_labels": _label_text(record.extracted_labels),
    }
    fields["combined_scientific_summary"] = " ".join(
        value for value in fields.values() if value
    )
    return {
        field: fields.get(field, "").strip()
        for field in PAPER_EMBEDDING_FIELDS
        if fields.get(field, "").strip()
    }


def build_field_embedding_records(
    records: Iterable[NormalizedRecord],
    provider: EmbeddingProvider,
    *,
    created_at: str | None = None,
) -> list[FieldEmbeddingRecord]:
    """Generate field-specific embeddings with provider metadata."""

    texts: list[str] = []
    metadata: list[tuple[str, Literal["dataset", "paper"], str]] = []
    for record in records:
        record_type: Literal["dataset", "paper"] = (
            "dataset" if isinstance(record, NormalizedDatasetRecord) else "paper"
        )
        for field_name, text in field_texts_for_record(record).items():
            texts.append(text)
            metadata.append((_record_id(record), record_type, field_name))

    if not texts:
        return []

    vectors = provider.embed_batch(texts)
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    return [
        FieldEmbeddingRecord(
            record_id=record_id,
            record_type=record_type,
            field_name=field_name,
            text=text,
            embedding=vector,
            provider_name=provider.provider_name,
            model_name=provider.model_name,
            dimension=provider.dimension,
            normalize=provider.normalize,
            created_at=timestamp,
        )
        for (record_id, record_type, field_name), text, vector in zip(
            metadata,
            texts,
            vectors,
            strict=True,
        )
    ]


def validate_field_embedding_cache(
    records: Sequence[FieldEmbeddingRecord],
    *,
    expected_provider_name: str | None = None,
    expected_model_name: str | None = None,
    expected_dimension: int | None = None,
    expected_normalize: bool | None = None,
) -> None:
    """Fail if cache metadata mixes incompatible providers or dimensions."""

    if not records:
        return

    first = records[0]
    expected = {
        "provider_name": expected_provider_name or first.provider_name,
        "model_name": expected_model_name or first.model_name,
        "dimension": expected_dimension or first.dimension,
        "normalize": first.normalize if expected_normalize is None else expected_normalize,
    }
    for record in records:
        actual: Mapping[str, object] = {
            "provider_name": record.provider_name,
            "model_name": record.model_name,
            "dimension": record.dimension,
            "normalize": record.normalize,
        }
        mismatches = [
            key
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        ]
        if mismatches:
            fields = ", ".join(mismatches)
            raise ValueError(f"embedding cache metadata mismatch: {fields}")


def write_field_embedding_cache(
    records: Iterable[FieldEmbeddingRecord],
    path: str | Path,
) -> Path:
    """Write field embeddings as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")
    return output


def read_field_embedding_cache(
    path: str | Path,
    *,
    expected_provider_name: str | None = None,
    expected_model_name: str | None = None,
    expected_dimension: int | None = None,
    expected_normalize: bool | None = None,
) -> list[FieldEmbeddingRecord]:
    """Read and validate field embeddings from JSONL."""

    records: list[FieldEmbeddingRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(FieldEmbeddingRecord.model_validate_json(line))
    validate_field_embedding_cache(
        records,
        expected_provider_name=expected_provider_name,
        expected_model_name=expected_model_name,
        expected_dimension=expected_dimension,
        expected_normalize=expected_normalize,
    )
    return records


def build_cache_from_normalized_path(
    input_path: str | Path,
    output_path: str | Path,
    provider: EmbeddingProvider,
) -> list[FieldEmbeddingRecord]:
    """Load normalized records, embed selected fields, and write a cache."""

    records = load_normalized_records(input_path)
    embeddings = build_field_embedding_records(records, provider)
    write_field_embedding_cache(embeddings, output_path)
    return embeddings
