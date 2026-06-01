"""Helpers for provenance-aware normalized corpus records."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import BaseModel

from neural_search.schemas import (
    EvidenceLabel,
    LabelEvidence,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)

NormalizedRecord: TypeAlias = NormalizedDatasetRecord | NormalizedPaperRecord
EXTRACTOR_VERSION = "v0.3.0"


def _safe_token(value: str, *, lower: bool = False) -> str:
    cleaned = value.strip()
    if lower:
        cleaned = cleaned.lower()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("stable ID components must not be empty")
    return cleaned


def make_dataset_id(source: str, source_id: str) -> str:
    """Make a stable normalized dataset ID such as `dataset:dandi:000026`."""

    return f"dataset:{_safe_token(source, lower=True)}:{_safe_token(source_id)}"


def make_paper_id(source: str, source_id: str) -> str:
    """Make a stable normalized paper ID such as `paper:openalex:W123456789`."""

    return f"paper:{_safe_token(source, lower=True)}:{_safe_token(source_id)}"


def make_evidence_label_id(
    label_type: str,
    label: str,
    source: str | None = None,
) -> str:
    """Make a stable evidence label ID such as `label:modality:neuropixels`."""

    parts = [
        "label",
        _safe_token(label_type.replace("-", "_"), lower=True),
        _safe_token(label.replace(" ", "_"), lower=True),
    ]
    if source:
        parts.append(_safe_token(source, lower=True))
    return ":".join(parts)


def stable_normalized_id(prefix: str, source: str, source_id: str) -> str:
    """Backward-compatible stable record ID helper."""

    if prefix == "dataset":
        return make_dataset_id(source, source_id)
    if prefix == "paper":
        return make_paper_id(source, source_id)
    return f"{_safe_token(prefix, lower=True)}:{_safe_token(source, lower=True)}:{_safe_token(source_id)}"


def evidence_label_from_extraction(
    label: LabelEvidence,
    label_type: str,
    *,
    source_field: str | None = None,
    source_value: str | None = None,
    extractor_name: str = "neural_search.rule_extractor",
) -> EvidenceLabel:
    """Convert existing extraction labels into the v0.3 provenance schema."""

    return EvidenceLabel(
        id=label.id,
        label=label.label,
        label_type=label_type,
        confidence=label.confidence,
        evidence_text=label.evidence,
        source_field=source_field,
        source_value=source_value,
        extractor_name=extractor_name,
        extractor_version=EXTRACTOR_VERSION,
    )


def record_to_dict(record: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized record or plain mapping to a JSON-serializable dict."""

    if isinstance(record, BaseModel):
        return record.model_dump(mode="json", exclude_none=True)
    return dict(record)


def record_from_dict(payload: dict[str, Any]) -> NormalizedRecord:
    """Validate a dict as either a normalized dataset or paper record."""

    if "dataset_id" in payload:
        return NormalizedDatasetRecord.model_validate(payload)
    if "paper_id" in payload:
        return NormalizedPaperRecord.model_validate(payload)
    raise ValueError("record must include dataset_id or paper_id")


def write_json(record: BaseModel | dict[str, Any] | list[Any], path: str | Path) -> Path:
    """Write one normalized record or a list of records as JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, list):
        payload: Any = [record_to_dict(item) for item in record]
    else:
        payload = record_to_dict(record)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_json(path: str | Path) -> NormalizedRecord | list[NormalizedRecord]:
    """Read one normalized record or a list of records from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record_from_dict(item) for item in payload]
    return record_from_dict(payload)


def write_jsonl(records: Iterable[BaseModel | dict[str, Any]], path: str | Path) -> Path:
    """Write normalized records as deterministic JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), sort_keys=True))
            handle.write("\n")
    return output


def read_jsonl(path: str | Path) -> list[NormalizedRecord]:
    """Load normalized dataset and paper records from JSONL."""

    records: list[NormalizedRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(record_from_dict(json.loads(line)))
    return records


def dump_records_jsonl(records: Iterable[BaseModel | dict[str, Any]], path: str | Path) -> Path:
    """Backward-compatible alias for `write_jsonl`."""

    return write_jsonl(records, path)


def load_records_jsonl(path: str | Path) -> list[NormalizedRecord]:
    """Backward-compatible alias for `read_jsonl`."""

    return read_jsonl(path)


def load_normalized_records(input_path: str | Path) -> list[NormalizedRecord]:
    """Load normalized records from a JSON file, JSONL file, or directory."""

    path = Path(input_path)
    if not path.exists():
        return []
    if path.is_dir():
        records: list[NormalizedRecord] = []
        for child in sorted([*path.glob("*.jsonl"), *path.glob("*.json")]):
            records.extend(load_normalized_records(child))
        return records
    if path.suffix == ".jsonl":
        return read_jsonl(path)

    records = read_json(path)
    return records if isinstance(records, list) else [records]
