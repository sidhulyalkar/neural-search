"""Application service for resolving the active searchable corpus."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.runtime import PROFILES, artifact_status


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
    """Resolve the real corpus with an explicit, profile-aware demo policy."""

    def __init__(
        self,
        *,
        profile: str | None = None,
        allow_demo_fallback: bool | None = None,
    ) -> None:
        configured = (profile or os.getenv("NEURAL_SEARCH_PROFILE", "demo")).strip().lower()
        self.profile = configured if configured in PROFILES else "demo"
        self.allow_demo_fallback = allow_demo_fallback

    def _fallback_allowed(self, override: bool | None) -> bool:
        if override is not None:
            return override
        if self.allow_demo_fallback is not None:
            return self.allow_demo_fallback
        return self.profile == "demo"

    def load(
        self,
        *,
        allow_demo_fallback: bool | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        status = artifact_status("full_corpus_v09")
        if status["usable"]:
            path = Path(status["absolute_path"])
            return (
                list(_load_jsonl(str(path), path.stat().st_mtime_ns)),
                "full_corpus_v09",
            )

        if self._fallback_allowed(allow_demo_fallback):
            return build_demo_seed(), "demo_fallback"

        raise ValueError(
            "The active execution profile requires the real dataset corpus, but "
            "`full_corpus_v09` is unavailable. Run `neural-search profile check "
            f"{self.profile}` for exact readiness, install a published researcher "
            "artifact bundle when available, or build the corpus locally. Neural Search "
            "will not silently substitute demo data for a research-profile query."
        )

    def find(
        self,
        dataset_id: str,
        *,
        allow_demo_fallback: bool | None = None,
    ) -> tuple[dict[str, Any], str]:
        records, source = self.load(allow_demo_fallback=allow_demo_fallback)
        wanted = dataset_id.casefold()
        for record in records:
            if wanted in dataset_lookup_keys(record):
                return record, source
        raise ValueError(f"Dataset not found in active corpus: {dataset_id}")
