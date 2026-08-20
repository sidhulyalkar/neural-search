"""Application service for resolving the active searchable corpus."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.runtime import artifact_status


@lru_cache(maxsize=4)
def _load_jsonl(path_str: str, mtime_ns: int) -> tuple[dict[str, Any], ...]:
    del mtime_ns  # cache key only
    records: list[dict[str, Any]] = []
    path = Path(path_str)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return tuple(records)


def _dataset(record: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = record.get("dataset")
    return nested if isinstance(nested, Mapping) else record


def dataset_identity(record: Mapping[str, Any]) -> str:
    dataset = _dataset(record)
    source = str(dataset.get("source") or record.get("source") or "unknown")
    source_id = str(
        dataset.get("source_id")
        or dataset.get("id")
        or record.get("source_id")
        or record.get("dataset_id")
        or "unknown"
    )
    if source_id.startswith(f"{source}:"):
        return source_id
    return f"{source}:{source_id}"


def dataset_lookup_keys(record: Mapping[str, Any]) -> set[str]:
    dataset = _dataset(record)
    values = {
        dataset_identity(record),
        str(dataset.get("id") or ""),
        str(dataset.get("source_id") or ""),
        str(record.get("dataset_id") or ""),
        str(record.get("source_id") or ""),
    }
    return {value.casefold() for value in values if value}


class CorpusAccessService:
    """Resolve verified real-corpus assets with a deterministic demo fallback."""

    def load(self) -> tuple[list[dict[str, Any]], str]:
        status = artifact_status("full_corpus_v09")
        if status["usable"]:
            path = Path(status["absolute_path"])
            return list(_load_jsonl(str(path), path.stat().st_mtime_ns)), "full_corpus_v09"
        return build_demo_seed(), "demo_fallback"

    def find(self, dataset_id: str) -> tuple[dict[str, Any], str]:
        records, source = self.load()
        wanted = dataset_id.casefold()
        for record in records:
            if wanted in dataset_lookup_keys(record):
                return record, source
        raise ValueError(f"Dataset not found in active corpus: {dataset_id}")
